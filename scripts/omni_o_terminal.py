import argparse, os, sys, json, io, time, math, torch, threading, queue, logging, contextlib, warnings
import numpy as np

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.vam import VAM, VAMConfig
from serve.realtime import SileroVAD

SAMPLE_RATE = 16000
AUDIO_SR = 24000
SAMPLES_PER_FRAME = 1920


def asr_run(model, samples):
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    r = model.generate(input=samples, cache={}, language='auto', use_itn=True)
    return rich_transcription_postprocess(r[0]['text']).strip() if r else ''


def init_model(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print('Loading ASR...')
    with contextlib.redirect_stdout(io.StringIO()):
        from funasr import AutoModel
        asr = AutoModel(model=os.path.join(root, args.sensevoice_dir),
                        trust_remote_code=True, device=args.device,
                        disable_update=True, batch_size=1)

    print('Loading model...')
    config = VAMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.hidden_size // 96,
        num_key_value_heads=args.hidden_size // 192,
        use_moe=args.use_moe,
    )
    ckpt_dir = os.path.join(root, args.load_from)
    weight = args.weight
    if not weight.endswith('.pth'):
        if args.use_moe and not weight.endswith('_moe'):
            weight = f'{weight}_moe.pth'
        else:
            weight = f'{weight}.pth'
    ckpt_path = os.path.join(ckpt_dir, weight)

    model = VAM(config,
                audio_encoder_path=os.path.join(root, args.sensevoice_dir),
                vision_model_path=os.path.join(root, args.siglip_dir))
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys (encoders): {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)}')
    if model.audio_encoder is not None:
        model.audio_encoder.to(args.device)
    model = model.half().eval().to(args.device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, args.tokenizer_dir))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  {args.weight}: {params:.2f}M')

    print('Loading Mimi...')
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained(os.path.join(root, args.mimi_dir)).eval().to(args.device)
    if args.device != 'cpu':
        mimi = mimi.half()

    print('Loading VAD...')
    vad = SileroVAD()
    return model, tokenizer, asr, mimi, vad


def record_audio(args, vad):
    import sounddevice as sd
    vad.reset()
    buffer = []
    ring = []
    speaking = False
    speech_samples = 0
    silence_samples = 0
    tail_silence = 0
    min_speech = int(SAMPLE_RATE * args.min_speech_ms / 1000)
    min_silence = int(SAMPLE_RATE * args.min_silence_ms / 1000)

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=512, device=args.mic)
    with stream:
        while True:
            chunk, _ = stream.read(512)
            chunk = chunk.flatten()

            for i in range(0, len(chunk), 512):
                w = chunk[i:i + 512]
                if len(w) < 512:
                    w = np.pad(w, (0, 512 - len(w)))
                prob = vad(w, SAMPLE_RATE)

                if prob > args.vad_threshold:
                    silence_samples = tail_silence = 0
                    speech_samples += len(w)
                    buffer.append(w)
                    if speech_samples >= min_speech and not speaking:
                        speaking = True
                        buffer = ring + buffer
                        ring = []
                elif speaking:
                    silence_samples += len(w)
                    tail_silence += 1
                    buffer.append(w)
                    if silence_samples >= min_silence:
                        if tail_silence > 1:
                            del buffer[-(tail_silence - 1):]
                        audio = np.concatenate(buffer)
                        return audio
                else:
                    if speech_samples > 0:
                        buffer.clear()
                    speech_samples = 0
                    ring = [w]

                if args.wait_key and not speaking:
                    import select
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.read(1)
                        return None


def play_audio(pcm, rate=AUDIO_SR):
    import sounddevice as sd
    sd.play(pcm, rate)
    sd.wait()


def mimi_decode(model, codes_2d, device):
    codes = codes_2d.T.unsqueeze(0).to(device)
    codes = torch.where(codes >= 2049, torch.zeros_like(codes), codes)
    with torch.no_grad():
        audio = model.decode(codes).audio_values.squeeze().float().cpu().numpy()
    return audio


def build_prompt(tokenizer, history, text):
    msgs = history + [{"role": "user", "content": text}]
    t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return torch.tensor(tokenizer(t)['input_ids'], dtype=torch.long, device='cpu')[None, ...]


def generate_response(model, tokenizer, mimi, x, device, max_new_tokens=512):
    audio_frames = []
    text_out = ''
    with torch.no_grad():
        for y, af in model.generate(
            x, tokenizer.eos_token_id, stream=True, return_audio_codes=True,
            max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.85,
        ):
            if y is not None:
                ans = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                new_text = ans[len(text_out):]
                if new_text:
                    print(new_text, end='', flush=True)
                    text_out = ans
            if af:
                audio_frames.append(af)
    print()
    if audio_frames:
        codes = [f for f in audio_frames if f and len(f) == 8]
        if codes:
            codes_t = torch.tensor(codes, dtype=torch.long)
            pcm = mimi_decode(mimi, codes_t, device)
            return text_out, pcm
    return text_out, None


def warmup(model, mimi, device):
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]], device=device)
        au = torch.full((1, 8, 3), 2049, dtype=torch.long, device=device)
        model.forward(torch.cat((au, ids.unsqueeze(1)), dim=1))
        if model.audio_encoder is not None:
            try:
                model.audio_encoder.model(
                    torch.zeros(1, 100, 560, device=device),
                    torch.tensor([100], device=device))
            except Exception:
                pass
        if mimi is not None:
            mimi.decode(torch.zeros(1, 8, 1, dtype=torch.long, device=device))


def main():
    parser = argparse.ArgumentParser(description='Omni-O Terminal Voice Chat')
    parser.add_argument('--load_from', default='checkpoint/omni-o')
    parser.add_argument('--weight', default='omni-o')
    parser.add_argument('--tokenizer_dir', default='checkpoint/omni/native_hf')
    parser.add_argument('--sensevoice_dir', default='checkpoint/sensevoice')
    parser.add_argument('--siglip_dir', default='checkpoint/siglip')
    parser.add_argument('--mimi_dir', default='checkpoint/mimi')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_new_tokens', default=256, type=int)
    parser.add_argument('--vad_threshold', default=0.5, type=float)
    parser.add_argument('--min_speech_ms', default=128, type=int)
    parser.add_argument('--min_silence_ms', default=800, type=int)
    parser.add_argument('--mic', default=None, type=int, help='Microphone device index')
    parser.add_argument('--wait_key', default=0, type=int,
                        help='Press Enter to start recording (0=auto VAD)')
    args = parser.parse_args()

    model, tokenizer, asr, mimi, vad = init_model(args)
    device = args.device
    x = build_prompt(tokenizer, [], 'Please introduce yourself.')
    print('Warmup...')
    warmup(model, mimi, device)
    print('Warmup done!\n')

    import sounddevice as sd
    history = []

    print('=== Omni-O Terminal Voice Chat ===')
    print(f'Mic: {sd.query_devices(args.mic, "input")["name"] if args.mic is not None else "default"}')
    print(f'Say something (VAD: silence>{args.min_silence_ms}ms = end of speech)')
    print()

    while True:
        if args.wait_key:
            input('Press Enter to record...')
            print('Recording... (speak now)')
            audio = record_audio(args, vad)
            if audio is None:
                continue
        else:
            audio = record_audio(args, vad)
            if audio is None:
                continue

        print(f'\r  Captured {len(audio) / SAMPLE_RATE:.1f}s audio')

        print('  ASR...', end=' ', flush=True)
        st = time.time()
        text = asr_run(asr, audio)
        print(f'"{text}" ({time.time() - st:.1f}s)')
        if not text:
            print('  (no speech detected)')
            continue

        history.append({"role": "user", "content": text})

        print('  Generating...', end=' ', flush=True)
        st = time.time()
        x = build_prompt(tokenizer, history[:-1], text)
        x = x.to(device)
        text_resp, pcm = generate_response(model, tokenizer, mimi, x, device,
                                            max_new_tokens=args.max_new_tokens)
        print(f'  ({time.time() - st:.1f}s)')

        if text_resp:
            history.append({"role": "assistant", "content": text_resp})

        if pcm is not None and len(pcm) > 0:
            print('  Playing...', end=' ', flush=True)
            play_audio(pcm)
            print('done')

        print()


if __name__ == '__main__':
    main()

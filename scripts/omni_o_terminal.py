import argparse, os, sys, io, time, torch, threading, queue, logging, contextlib, warnings
import numpy as np

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.vam import VAM, VAMConfig
from serve.realtime import SileroVAD

SAMPLE_RATE = 16000
AUDIO_SR = 24000


class RealtimeRecorder:
    def __init__(self, vad, threshold=0.5, min_speech_ms=128, min_silence_ms=800, mic=None):
        self.vad = vad
        self.threshold = threshold
        self.min_speech = int(SAMPLE_RATE * min_speech_ms / 1000)
        self.min_silence = int(SAMPLE_RATE * min_silence_ms / 1000)
        self.mic = mic
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.state = 'idle'
        self.buffer = []
        self.ring = []
        self.speaking = False
        self.speech_samples = 0
        self.silence_samples = 0
        self.tail_silence = 0
        self.interrupt = False

    def _feed(self, w):
        prob = self.vad(w, SAMPLE_RATE)
        with self.lock:
            if prob > self.threshold:
                self.silence_samples = self.tail_silence = 0
                self.speech_samples += len(w)
                self.buffer.append(w)
                if self.speech_samples >= self.min_speech and not self.speaking:
                    self.speaking = True
                    self.buffer = self.ring + self.buffer
                    self.ring = []
                if self.speaking and self.state in ('processing', 'playing'):
                    self.interrupt = True
            elif self.speaking:
                self.silence_samples += len(w)
                self.tail_silence += 1
                self.buffer.append(w)
                if self.silence_samples >= self.min_silence:
                    if self.tail_silence > 1:
                        del self.buffer[-(self.tail_silence - 1):]
                    audio = np.concatenate(self.buffer)
                    self.buffer.clear()
                    self.speaking = False
                    self.speech_samples = self.silence_samples = self.tail_silence = 0
                    self.q.put(audio)
            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self.ring = [w]

    def start(self):
        import sounddevice as sd
        def _run():
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                blocksize=512, device=self.mic) as stream:
                while getattr(self, '_running', True):
                    chunk, _ = stream.read(512)
                    self._feed(chunk.flatten())
        self._running = True
        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False


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
    model.load_state_dict(state, strict=False)
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


def mimi_decode(model, codes_2d, device):
    codes = codes_2d.T.unsqueeze(0).to(device)
    codes = torch.where(codes >= 2049, torch.zeros_like(codes), codes)
    with torch.no_grad():
        audio = model.decode(codes).audio_values.squeeze().float().cpu().numpy()
    return audio


def generate_response(recorder, model, tokenizer, mimi, x, device, max_new_tokens=512):
    audio_frames = []
    text_out = ''
    interrupted = False
    with torch.no_grad():
        for y, af in model.generate(
            x, tokenizer.eos_token_id, stream=True, return_audio_codes=True,
            max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.85,
        ):
            with recorder.lock:
                if recorder.interrupt:
                    interrupted = True
                    break
            if y is not None:
                ans = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                new_text = ans[len(text_out):]
                if new_text:
                    print(new_text, end='', flush=True)
                    text_out = ans
            if af:
                audio_frames.append(af)
    print()
    if interrupted:
        print('  [interrupted]')
    if audio_frames and not interrupted:
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


def build_prompt(tokenizer, history, text):
    msgs = history + [{"role": "user", "content": text}]
    t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return torch.tensor(tokenizer(t)['input_ids'], dtype=torch.long, device='cpu')[None, ...]


def main():
    parser = argparse.ArgumentParser(description='Omni-O Terminal Voice Chat (with interrupt)')
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
    parser.add_argument('--min_silence_ms', default=600, type=int)
    parser.add_argument('--mic', default=None, type=int, help='Microphone device index')
    args = parser.parse_args()

    model, tokenizer, asr, mimi, vad = init_model(args)
    device = args.device

    print('Warmup...')
    warmup(model, mimi, device)
    print('Warmup done!\n')

    import sounddevice as sd
    recorder = RealtimeRecorder(vad, args.vad_threshold, args.min_speech_ms,
                                args.min_silence_ms, args.mic)
    recorder.start()

    history = []
    mic_name = sd.query_devices(args.mic, 'input')['name'] if args.mic is not None else 'default'
    print('=== Omni-O Terminal Voice Chat (interruptible) ===')
    print(f'Mic: {mic_name}')
    print('Speak to start — silence >=600ms = end of turn')
    print('Speak during playback to interrupt')
    print()

    try:
        while True:
            audio = recorder.q.get()
            if len(audio) < SAMPLE_RATE * 0.1:
                continue

            seconds = len(audio) / SAMPLE_RATE
            print(f'\r  {seconds:.1f}s audio  ASR...', end=' ', flush=True)
            st = time.time()
            text = asr_run(asr, audio)
            print(f'"{text}" ({time.time() - st:.1f}s)')
            if not text.strip():
                continue

            history.append({"role": "user", "content": text})

            with recorder.lock:
                recorder.state = 'processing'

            x = build_prompt(tokenizer, history[:-1], text).to(device)
            print('  ', end='', flush=True)
            st = time.time()
            text_resp, pcm = generate_response(recorder, model, tokenizer, mimi,
                                                x, device, args.max_new_tokens)

            with recorder.lock:
                interrupted = recorder.interrupt
                recorder.interrupt = False
                recorder.state = 'playing' if not interrupted else 'idle'

            if text_resp:
                if not interrupted:
                    history.append({"role": "assistant", "content": text_resp})

            if pcm is not None and len(pcm) > 0:
                print(f'  Playing... ({time.time() - st:.1f}s gen)', end=' ', flush=True)
                sd.play(pcm, AUDIO_SR)
                # Poll playback with interrupt check
                while sd.get_stream().active:
                    with recorder.lock:
                        if recorder.interrupt:
                            sd.stop()
                            print('[interrupted]', end=' ')
                            with recorder.lock:
                                recorder.interrupt = False
                                recorder.state = 'idle'
                            break
                    time.sleep(0.05)
                print('done')

            with recorder.lock:
                recorder.state = 'idle'

    except KeyboardInterrupt:
        print('\nBye!')
    finally:
        recorder.stop()


if __name__ == '__main__':
    main()

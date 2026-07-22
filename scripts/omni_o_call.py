import argparse, os, sys, json, time, math, torch, threading, queue, base64, io, logging, contextlib, warnings
import numpy as np
from PIL import Image
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.vam import VAM, VAMConfig
from serve.realtime import RealtimeSession

M = {}
V = {}  # voice_name -> {ref_codes, spk_emb}
MODEL_LOCK = threading.Lock()
VOICES_BUILTIN, VOICES_UNSEEN, VOICES_MANUAL = [], [], []
SAMPLES_PER_FRAME = 1920
REF_FRAMES = 300

def sse(d): return f"data: {json.dumps(d)}\n\n"

def asr_run(samples):
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    r = M['asr'].generate(input=samples, cache={}, language='auto', use_itn=True)
    return rich_transcription_postprocess(r[0]['text']).strip() if r else ''

def prep_audio(samples):
    m, dev = M['model'], M['device']
    proc = m.audio_processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    mel = proc.input_features.squeeze(0).unsqueeze(0).to(dev)
    vlen = proc.attention_mask.sum().item()
    return mel, torch.tensor([vlen], device=dev), m.config.audio_special_token * (vlen or 1)

def prep_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    return {k: v.to(M['device']) for k, v in M['model'].vision_processor(images=img, return_tensors="pt").items()}

def build_ids(prompt, history):
    tok, dev = M['tokenizer'], M['device']
    cfg = M['cfg']
    hist = history[-cfg.max_history_turns:] if cfg.max_history_turns > 0 else []
    msgs = hist + [{"role": "user", "content": prompt}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return torch.tensor(tok(t)['input_ids'], dtype=torch.long, device=dev)[None, ...]

def _mimi_decode(frames):
    codes = [f for f in frames if f and len(f) == 8]
    if not codes or not M['mimi']: return None
    mc = torch.tensor(codes, dtype=torch.long, device=M['device']).T.unsqueeze(0)
    mc = torch.where(mc >= 2049, torch.zeros_like(mc), mc)
    with torch.no_grad():
        au = M['mimi'].decode(mc).audio_values.squeeze().cpu().numpy()
    return au, mc.shape[-1]

def pcm_bytes(frames, ov):
    r = _mimi_decode(frames)
    if r is None: return None
    au, T = r
    if ov > 0: au = au[int(ov * len(au) / T):]
    return (au * 32767).astype('int16').tobytes()

def stream_pcm(frames, flush=False):
    if not M['mimi']: return
    cf, ov_max, n = M['cfg'].audio_chunk_frames, M['cfg'].audio_overlap, len(frames)
    if not flush and n >= cf and n % cf == 0:
        ov = min(ov_max, n - cf)
        p = pcm_bytes(frames[-(cf + ov):], ov)
        if p: yield p
    elif flush:
        rem = n % cf
        if rem:
            ov = min(ov_max, n - rem)
            p = pcm_bytes(frames[-(rem + ov):], ov)
            if p: yield p

def register_voice(name, value, group='manual'):
    V[name] = value
    groups = {'builtin': VOICES_BUILTIN, 'unseen': VOICES_UNSEEN, 'manual': VOICES_MANUAL}
    dst = groups[group]
    if name not in dst: dst.append(name)
    for k, lst in groups.items():
        if k != group and name in lst: lst.remove(name)

def voice_args(name):
    if name and name != 'default' and name in V:
        v = V[name]
        dev = M['device']
        rc = v['ref_codes'].unsqueeze(0).to(dev)
        se = v['spk_emb'].half().unsqueeze(0).to(dev) if 'spk_emb' in v else None
        return {'ref_codes': rc, 'spk_emb': se}
    return {}

def run_generate(x, audio_inputs, audio_lens, pixel_values, **kw):
    with MODEL_LOCK, torch.no_grad():
        yield from M['model'].generate(
            x, M['tokenizer'].eos_token_id, stream=True, return_audio_codes=True,
            audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, **kw)

def prepare_turn(text, samples, image_b64, do_asr_for_image):
    audio_inputs = audio_lens = pixel_values = None
    prompt = text or ''
    user_text = text or ''
    asr_thread, asr_result = None, [None]
    if samples is not None:
        if image_b64 and do_asr_for_image:
            user_text = asr_run(samples)
            prompt = user_text
        else:
            audio_inputs, audio_lens, prompt = prep_audio(samples)
            if M['cfg'].max_history_turns > 0:
                sa = samples.copy()
                def _a(): asr_result[0] = asr_run(sa)
                asr_thread = threading.Thread(target=_a); asr_thread.start()
    if image_b64:
        pixel_values = prep_image(image_b64)
        m = M['model']
        prompt = (prompt + "\n\n" if prompt else "") + "请描述这张图片\n\n" + m.config.image_special_token * m.config.image_token_len
    return audio_inputs, audio_lens, pixel_values, prompt, user_text, asr_thread, asr_result


def init_web_app():
    from flask import Flask, request, Response, send_from_directory
    from flask_cors import CORS
    from flask_sock import Sock

    app = Flask(__name__, static_folder='.')
    CORS(app)
    sock = Sock(app)

    @app.route('/')
    def index(): return send_from_directory('.', 'omni_o_web.html')
    @app.route('/call')
    def call_page(): return send_from_directory('.', 'omni_o_web.html')

    @app.route('/voices')
    def get_voices():
        return json.dumps({'builtin': sorted(VOICES_BUILTIN), 'unseen': sorted(VOICES_UNSEEN), 'manual': sorted(VOICES_MANUAL)})

    @app.route('/models')
    def get_models():
        return json.dumps({'models': [M.get('model_name', 'omni-o')], 'current': M.get('model_name', 'omni-o')})

    @app.route('/chat', methods=['POST'])
    def chat():
        d = request.json
        history = d.get('history', [])
        samples = None
        if d.get('audio'):
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(base64.b64decode(d['audio']))).set_frame_rate(16000).set_channels(1).set_sample_width(2)
            samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        va = voice_args(d.get('voice', 'default'))

        def gen():
            audio_inputs, audio_lens, pixel_values, prompt, user_text, asr_th, asr_res = prepare_turn(
                d.get('text', ''), samples, d.get('image'), do_asr_for_image=True)
            x = build_ids(prompt, history)
            asr_sent = False
            if user_text and samples is not None and d.get('image'):
                yield sse({'type': 'user_prompt', 'content': user_text}); asr_sent = True
            frames, text_ttft, audio_ttft = [], None, None
            t0 = time.time(); hi = 0
            for y, af in run_generate(x, audio_inputs, audio_lens, pixel_values,
                                       max_new_tokens=d.get('max_tokens', 512),
                                       temperature=d.get('temperature', 1), top_p=0.85, **va):
                if not asr_sent and asr_th and not asr_th.is_alive():
                    asr_th.join()
                    if asr_res[0]: yield sse({'type': 'user_prompt', 'content': asr_res[0]})
                    asr_sent = True
                if y is not None:
                    if text_ttft is None:
                        text_ttft = (time.time() - t0) * 1000
                        yield sse({'type': 'ttft', 'text_ttft': round(text_ttft, 1)})
                    ans = M['tokenizer'].decode(y[0].tolist(), skip_special_tokens=True)
                    if ans and ans[-1] != '\ufffd' and len(ans) > hi:
                        yield sse({'type': 'text', 'content': ans[hi:]}); hi = len(ans)
                if af:
                    if audio_ttft is None:
                        audio_ttft = (time.time() - t0) * 1000
                        yield sse({'type': 'ttft', 'audio_ttft': round(audio_ttft, 1)})
                    frames.append(af)
                    for pcm in stream_pcm(frames):
                        b64 = base64.b64encode(pcm).decode()
                        for i in range(0, len(b64), 2000):
                            yield sse({'type': 'pcm', 'c': b64[i:i+2000], 'd': i+2000 >= len(b64)})
            for pcm in stream_pcm(frames, flush=True):
                b64 = base64.b64encode(pcm).decode()
                for i in range(0, len(b64), 2000):
                    yield sse({'type': 'pcm', 'c': b64[i:i+2000], 'd': i+2000 >= len(b64)})
            if not asr_sent:
                if asr_th:
                    asr_th.join()
                    if asr_res[0]: yield sse({'type': 'user_prompt', 'content': asr_res[0]})
                else:
                    yield sse({'type': 'user_prompt', 'content': prompt})
            yield sse({'type': 'done'})
        return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @sock.route('/ws/realtime')
    def realtime(ws):
        session = RealtimeSession(M['vad_path'])
        q = queue.Queue(); alive = [True]; state = {'history': [], 'image': None}
        n_hist = M['cfg'].max_history_turns

        def push_audio(data):
            return session.push_chunk(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

        def set_ctx(msg):
            h = msg.get('history') or []
            state['history'] = h[-n_hist:] if n_hist > 0 else []
            if 'image' in msg: state['image'] = msg.get('image')
            if 'voice' in msg: state['voice'] = msg.get('voice', 'default')

        def poll_interrupt():
            while True:
                try: data = q.get_nowait()
                except queue.Empty: return False
                if isinstance(data, bytes):
                    if push_audio(data) == 'interrupt': return True
                    ws.send(json.dumps({'type': 'vad', 'speaking': session.speaking}))
                else:
                    m = json.loads(data)
                    if m.get('type') == 'context': set_ctx(m)
                    elif m.get('type') in ('stop', 'end'):
                        if m['type'] == 'end': alive[0] = False
                        session.interrupt = True; return True

        def recv_loop():
            while alive[0]:
                try:
                    data = ws.receive(timeout=1)
                    if data is None: alive[0] = False; break
                    q.put(data)
                except: alive[0] = False; break

        threading.Thread(target=recv_loop, daemon=True).start()
        try:
            while alive[0]:
                try: data = q.get(timeout=0.05)
                except queue.Empty: continue
                if isinstance(data, str):
                    m = json.loads(data)
                    if m.get('type') == 'context': set_ctx(m)
                    elif m.get('type') == 'stop': session.interrupt = True
                    elif m.get('type') == 'end': break
                    continue
                if session.generating:
                    push_audio(data); ws.send(json.dumps({'type': 'vad', 'speaking': session.speaking})); continue
                status = push_audio(data)
                ws.send(json.dumps({'type': 'vad', 'speaking': session.speaking}))
                if status != 'speech_end': continue

                session.generating = True
                audio = session.get_audio()
                ws.send(json.dumps({'type': 'generating'}))
                audio_inputs, audio_lens, pixel_values, prompt, user_text, asr_th, asr_res = prepare_turn(
                    '', audio, state['image'], do_asr_for_image=True)
                if state['image']: state['image'] = None
                x = build_ids(prompt, state['history'])
                va_rt = voice_args(state.get('voice', 'default'))

                frames, full_text, interrupted = [], '', False
                for y, af in run_generate(x, audio_inputs, audio_lens, pixel_values,
                                           max_new_tokens=512, temperature=0.7, **va_rt):
                    if poll_interrupt() or session.interrupt: interrupted = True; break
                    if y is not None:
                        ans = M['tokenizer'].decode(y[0].tolist(), skip_special_tokens=True)
                        if ans and ans[-1] != '\ufffd' and len(ans) > len(full_text):
                            ws.send(json.dumps({'type': 'text', 'content': ans[len(full_text):]})); full_text = ans
                    if af:
                        frames.append(af)
                        for pcm in stream_pcm(frames):
                            ws.send(json.dumps({'type': 'pcm', 'data': base64.b64encode(pcm).decode()}))
                if not interrupted:
                    for pcm in stream_pcm(frames, flush=True):
                        ws.send(json.dumps({'type': 'pcm', 'data': base64.b64encode(pcm).decode()}))
                if asr_th:
                    asr_th.join(); user_text = asr_res[0] or user_text
                if n_hist > 0:
                    if user_text: state['history'].append({'role': 'user', 'content': user_text})
                    if full_text: state['history'].append({'role': 'assistant', 'content': full_text})
                    state['history'] = state['history'][-n_hist:]
                ws.send(json.dumps({'type': 'done', 'interrupted': interrupted or session.interrupt}))
                session.generating = False; session.interrupt = False
        finally:
            alive[0] = False
    return app


def init_model(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    M['cfg'] = args
    M['device'] = args.device

    with contextlib.redirect_stdout(io.StringIO()):
        from funasr import AutoModel
        M['asr'] = AutoModel(model=os.path.join(root, args.sensevoice_dir), trust_remote_code=True, device=args.device, disable_update=True)

    ckpt_dir = os.path.join(root, args.load_from)
    is_hf = os.path.exists(os.path.join(ckpt_dir, 'config.json')) and \
            (os.path.exists(os.path.join(ckpt_dir, 'model.safetensors')) or
             os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')))

    if is_hf:
        model = VAM.from_pretrained(
            ckpt_dir,
            audio_encoder_path=os.path.join(root, args.sensevoice_dir),
            vision_model_path=os.path.join(root, args.siglip_dir),
        )
    else:
        config = VAMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.hidden_size // 96,
            num_key_value_heads=args.hidden_size // 192,
            use_moe=args.use_moe,
        )
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
            print(f'  Missing keys (expected for encoders): {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')
        if model.audio_encoder is not None:
            model.audio_encoder.to(args.device)
        if model.vision_encoder is not None:
            model.vision_encoder.to(args.device)

    M['model'] = model.half().eval().to(args.device)

    tok_dir = os.path.join(root, args.tokenizer_dir)
    from transformers import AutoTokenizer
    M['tokenizer'] = AutoTokenizer.from_pretrained(tok_dir)

    model_name = args.weight or os.path.basename(args.load_from.rstrip('/'))
    M['model_name'] = model_name
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Loaded {model_name}: {params:.2f}M')

    try:
        from transformers import MimiModel
        mimi_path = os.path.join(root, args.mimi_dir)
        M['mimi'] = MimiModel.from_pretrained(mimi_path).eval().to(args.device)
        if args.device != 'cpu':
            M['mimi'] = M['mimi'].half()
        print('Mimi loaded')
    except Exception as e:
        M['mimi'] = None
        print(f'Mimi load failed: {e}')

    try:
        from modelscope.models.audio.sv.DTDNN import CAMPPlus
        import torchaudio
        M['campplus'] = CAMPPlus(feat_dim=80, embedding_size=192, growth_rate=32, bn_size=4,
                                 init_channels=128, config_str='batchnorm-relu', memory_efficient=True)
        camp_path = os.path.join(root, 'checkpoint/campplus/campplus_cn_common.pt')
        sd = torch.load(camp_path, map_location='cpu')
        M['campplus'].load_state_dict({k: v.float() for k, v in sd.items()})
        M['campplus'] = M['campplus'].eval().to(args.device)
        M['mel_fn'] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            n_mels=80, f_min=20, f_max=7600, norm='slaney', mel_scale='slaney',
        ).to(args.device)
        print('CAM++ loaded')
    except Exception as e:
        M['campplus'] = M['mel_fn'] = None
        print(f'CAM++ load failed (voice clone will be unavailable): {e}')

    M['vad_path'] = os.path.join(root, args.vad_dir, 'silero_vad.onnx')

    spk_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'speaker')
    for fn, group in [('voices.pt', 'builtin'), ('voices_unseen.pt', 'unseen')]:
        fp = os.path.join(spk_dir, fn)
        if os.path.exists(fp):
            for speaker, v in torch.load(fp, map_location='cpu').items():
                if speaker not in V:
                    register_voice(speaker, v, group=group)
    if V: print(f'Loaded {len(V)} voices')

    print('Warmup...')
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]], device=args.device)
        au = torch.full((1, 8, 3), 2049, dtype=torch.long, device=args.device)
        M['model'].forward(torch.cat((au, ids.unsqueeze(1)), dim=1))
        if M['model'].audio_encoder is not None:
            try:
                M['model'].audio_encoder.model(torch.zeros(1, 100, 560, device=args.device), torch.tensor([100], device=args.device))
            except Exception:
                pass
        if M['mimi']:
            M['mimi'].decode(torch.zeros(1, 8, 1, dtype=torch.long, device=args.device))
    print('Warmup done! Ready.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Omni-O Real-time Voice Call')
    p.add_argument('--load_from', default='checkpoint/omni-o-hf', help='HF 模型权重目录（自动检测 .pth 目录兼容）')
    p.add_argument('--weight', default='', help='权重文件名（仅 .pth 模式，不含后缀）')
    p.add_argument('--tokenizer_dir', default='checkpoint/omni/native_hf', help='tokenizer目录')
    p.add_argument('--sensevoice_dir', default='checkpoint/sensevoice', help='SenseVoice ASR目录')
    p.add_argument('--siglip_dir', default='checkpoint/siglip', help='SigLIP视觉编码器目录')
    p.add_argument('--mimi_dir', default='checkpoint/mimi', help='Mimi解码器目录')
    p.add_argument('--vad_dir', default='checkpoint/vad', help='VAD模型目录')
    p.add_argument('--hidden_size', default=768, type=int)
    p.add_argument('--num_hidden_layers', default=8, type=int)
    p.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--port', default=7860, type=int)
    p.add_argument('--audio_chunk_frames', default=4, type=int)
    p.add_argument('--audio_overlap', default=2, type=int)
    p.add_argument('--max_history_turns', default=0, type=int)
    args = p.parse_args()

    init_model(args)
    app = init_web_app()
    print(f'Omni-O Call server started at http://0.0.0.0:{args.port}/')
    app.run(host='0.0.0.0', port=args.port, threaded=True)

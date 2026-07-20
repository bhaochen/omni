import numpy as np


class SileroVAD:
    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.h, self.c = np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        self.h[:], self.c[:] = 0, 0

    def __call__(self, chunk, sr=16000):
        out, self.h, self.c = self.session.run(None, {"input": chunk.reshape(1, -1).astype(np.float32), "h": self.h, "c": self.c, "sr": np.array(sr, dtype="int64")})
        return float(out[0][0])


class RealtimeSession:
    def __init__(self, vad_path, sr=16000, threshold=0.8, min_speech_ms=128, min_silence_ms=800):
        self.vad, self.sr, self.threshold = SileroVAD(vad_path), sr, threshold
        self.min_speech, self.min_silence = int(sr * min_speech_ms / 1000), int(sr * min_silence_ms / 1000)
        self.reset()

    def reset(self):
        self.vad.reset()
        self.buffer, self.ring, self.speaking, self.generating, self.interrupt = [], [], False, False, False
        self.speech_samples = self.silence_samples = self.tail_silence = 0

    def push_chunk(self, chunk, W=1024):
        for i in range(0, max(len(chunk), 1), W):
            w = chunk[i:i + W]
            if len(w) < W:
                w = np.pad(w, (0, W - len(w)))
            prob = self.vad(w, self.sr)
            if prob > self.threshold:
                self.silence_samples = self.tail_silence = 0
                self.speech_samples += len(w)
                self.buffer.append(w)
                if self.speech_samples >= self.min_speech and not self.speaking:
                    self.speaking = True
                    self.buffer = self.ring + self.buffer
                    self.ring = []
                if self.generating and self.speaking:
                    self.interrupt = True
                    return 'interrupt'
            elif self.speaking:
                self.silence_samples += len(w)
                self.tail_silence += 1
                self.buffer.append(w)
                if self.silence_samples >= self.min_silence:
                    if self.tail_silence > 1:
                        del self.buffer[-(self.tail_silence - 1):]
                    self.speaking, self.speech_samples, self.silence_samples, self.tail_silence = False, 0, 0, 0
                    return 'speech_end'
            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self.ring = [w]
        return 'listening'

    def get_audio(self):
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        return audio

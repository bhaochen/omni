import os
import logging
import contextlib
import io
import warnings
import torch
import numpy as np
from torch import nn
from types import SimpleNamespace

from transformers import logging as hf_logging


class SenseVoiceAudioProcessor:
    def __init__(self, frontend):
        self.frontend = frontend

    def __call__(self, wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, **kwargs):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        return SimpleNamespace(
            input_features=fbank,
            attention_mask=(torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0),
        )


class SenseVoiceAudioEncoder(nn.Module):
    """Frozen SenseVoice audio encoder (funasr AutoModel). Returns frame embeddings."""

    def __init__(self, model_path: str = None):
        super().__init__()
        self.model = None
        self.processor = None
        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)

    @torch.no_grad()
    def load(self, model_path: str):
        if not os.path.exists(model_path):
            warnings.warn(f"[SenseVoiceAudioEncoder] path not found: {model_path}")
            return
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel
            m = AutoModel(model=model_path, trust_remote_code=True, disable_update=True, device="cpu")
        encoder, frontend = m.model.encoder, m.kwargs["frontend"]
        for p in encoder.parameters():
            p.requires_grad = False
        self.model = encoder.eval().float()
        self.processor = SenseVoiceAudioProcessor(frontend.eval())

    @property
    def hidden_size(self) -> int:
        if self.model is None:
            return 0
        return self.model.config.d_model if hasattr(self.model.config, "d_model") else self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, fbank, audio_lens=None):
        if self.model is None:
            return None
        if audio_lens is None:
            audio_lens = torch.tensor([fbank.size(1)] * fbank.size(0), device=fbank.device)
        emb, _ = self.model(fbank, audio_lens)
        return emb

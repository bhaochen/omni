import os
import torch
import torch.nn.functional as F
import warnings
from torch import nn
from typing import Optional

from transformers import SiglipVisionModel, SiglipImageProcessor
from transformers import logging as hf_logging


class SiglipVisionEncoder(nn.Module):
    """Frozen SigLIP vision encoder. Returns last-hidden-state embeddings (B, T, D)."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.model = None
        self.processor = None
        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)

    @torch.no_grad()
    def load(self, model_path: str):
        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            warnings.warn(f"[SiglipVisionEncoder] failed to load from {model_path}")
            return
        processor = SiglipImageProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.eval()
        self.processor = processor

    @property
    def hidden_size(self) -> int:
        if self.model is None:
            return 0
        return self.model.config.hidden_size

    def preprocess(self, image):
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        return self.processor(images=image, return_tensors='pt')

    @torch.no_grad()
    def encode(self, pixel_values):
        if self.model is None:
            return None
        if hasattr(pixel_values, 'keys'):
            pixel_values = {k: (v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v) for k, v in pixel_values.items()}
        outputs = self.model(**pixel_values)
        return outputs.last_hidden_state

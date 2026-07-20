import os
import math
import torch
import warnings
import logging
import contextlib
import io
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import SiglipVisionModel, SiglipImageProcessor, logging as hf_logging

from core import RMSNorm, precompute_freqs_cis, Block, MOEFeedForward
from models.lm.config import LMConfig
from models.lm.model import LMForCausalLM
from models.vam.config import VAMConfig
from encoders.audio import SenseVoiceAudioEncoder, SenseVoiceAudioProcessor
from encoders.vision import SiglipVisionEncoder
from projectors import MMVisionProjector, MMAudioProjector


class TalkerHead(nn.Module):
    def __init__(self, in_features, out_features, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, rank, bias=False), nn.GELU(), nn.Linear(rank, out_features, bias=False))
            for _ in range(num_layers)
        ])

    def forward(self, x):
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class TalkerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Embedding(num_embeddings, rank), nn.GELU(), nn.Linear(rank, embedding_dim, bias=False))
            for _ in range(num_layers)
        ])

    def forward(self, x):
        base_out = self.base(x)
        return sum(base_out[:, i, :] + self.adapters[i](x[:, i, :]) for i in range(len(self.adapters))) / self.num_layers


class TalkerModule(nn.Module):
    def __init__(self, config: VAMConfig):
        super().__init__()
        self.talker_config = LMConfig(hidden_size=config.talker_hidden_size, use_moe=config.use_moe)
        self.layers = nn.ModuleList([Block(l, self.talker_config) for l in range(config.num_talker_hidden_layers)])
        self.norm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = TalkerHead(config.talker_hidden_size, config.audio_vocab_size)
        self.embed_tokens = TalkerEmbedding(config.audio_vocab_size, config.talker_hidden_size)
        self.codec_proj = nn.Sequential(
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            nn.GELU(),
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )
        self.text_scale, self.audio_scale = nn.Parameter(torch.tensor(3.0)), nn.Parameter(torch.tensor(1.0))
        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.talker_config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


class VAM(LMForCausalLM):
    config_class = VAMConfig

    def __init__(self, config: VAMConfig = None, audio_encoder_path: str = None, vision_model_path: str = None):
        config = config or VAMConfig()
        super().__init__(config)
        object.__setattr__(self, 'thinker', self.model)
        object.__setattr__(self.model, 'lm_head', self.lm_head)
        self.talker = TalkerModule(config)
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size, target_tokens=config.image_token_len)
        self.audio_pad_token, self.audio_stop_token, self.audio_spk_token = config.audio_pad_token, config.audio_stop_token, config.audio_spk_token
        audio_encoder = SenseVoiceAudioEncoder(audio_encoder_path) if audio_encoder_path else SenseVoiceAudioEncoder()
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_encoder.processor)
        vision_encoder = SiglipVisionEncoder(vision_model_path) if vision_model_path else SiglipVisionEncoder()
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_encoder.processor)

    @staticmethod
    def load_sensevoice(path):
        if not os.path.exists(path):
            warnings.warn(f"[VAM] SenseVoice path not found: {path}")
            return None, None
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel
            m = AutoModel(model=path, trust_remote_code=True, disable_update=True, device="cpu")
        encoder, frontend = m.model.encoder, m.kwargs["frontend"]
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder.eval().float(), SenseVoiceAudioProcessor(frontend.eval())

    @staticmethod
    def load_vision(path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[VAM] Vision model path not found: {path}. vision_encoder will be None!")
            return None, None
        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval(), processor

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()):
            return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor([valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device)
        with torch.no_grad():
            emb, _ = self.audio_encoder.model(valid_fbank, valid_lens)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = [self.audio_proj(emb[i, :max(1, min(valid_lens[i].item(), emb.size(1)))].unsqueeze(0).to(proj_dtype)).squeeze(0) for i in range(emb.size(0))]
        if batch_mask.all():
            return emb_list
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def inject_audio_features(self, tokens, h, audio_feats, seqlen):
        if audio_feats is None or not self.config.audio_ids:
            return h
        marker = self.config.audio_ids[0]
        out = []
        for b in range(h.size(0)):
            hb, seq, i = h[b], tokens[b].tolist(), 0
            af = audio_feats[b] if audio_feats[b] is not None else None
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = min(af.size(0), i - start)
                        hb = torch.cat((hb[:start], af[:inject_len], hb[start + inject_len:]), dim=0)
                        af = None
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    @torch.compiler.disable
    def get_image_embeddings(self, image_inputs):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: (v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v) for k, v in image_inputs.items()}
            pixel_attention_mask = image_inputs.get('pixel_attention_mask')
            if pixel_attention_mask is not None and not pixel_attention_mask.any():
                pv = image_inputs['pixel_values']
                return pv.new_zeros(pv.size(0), pv.size(1), self.config.image_hidden_size)
        with torch.no_grad():
            outputs = self.vision_encoder.model(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def encode_image_inputs(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None:
            return None
        mask = pixel_values.flatten(1).any(1)
        if not mask.any():
            return pixel_values.new_zeros(pixel_values.size(0), self.config.image_token_len, self.config.hidden_size)
        with torch.no_grad():
            emb = self.vision_encoder.model(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        emb = self.vision_proj(emb)
        if mask.all():
            return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0,
                audio_inputs=None, audio_lens=None, pixel_values=None, **args):
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
            text_ids = input_ids
            audio_ids = torch.full((batch_size, 8, seq_length), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
        else:
            batch_size, _, seq_length = input_ids.shape
            text_ids, audio_ids = input_ids[:, 8, :], input_ids[:, :8, :]
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        n_thinker, n_talker = len(self.thinker.layers), len(self.talker.layers)
        past_key_values = past_key_values or ([None] * (n_thinker + n_talker))
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        if self.thinker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.thinker.freqs_cos, self.thinker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        if self.talker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.talker.talker_config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.talker.freqs_cos, self.talker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        presents = []

        hidden_states = self.thinker.dropout(self.thinker.embed_tokens(text_ids))
        position_embeddings = (self.thinker.freqs_cos[start_pos:start_pos + seq_length], self.thinker.freqs_sin[start_pos:start_pos + seq_length])
        if audio_inputs is not None and start_pos == 0:
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            hidden_states = self.inject_audio_features(text_ids, hidden_states, audio_features, seq_length)
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                img_emb = self.get_image_embeddings(pixel_values).to(hidden_states.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num, c, im_h, im_w = pixel_values.shape
                stack_dim = 1 if bs > 1 else 0
                vision_tensors = torch.stack([self.encode_image_inputs(pixel_values[:, i, :, :, :]) for i in range(num)], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=text_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=seq_length)
        bridge_states = hidden_states
        for i, (layer, past_key_value) in enumerate(zip(self.thinker.layers, past_key_values[:n_thinker])):
            hidden_states, present = layer(hidden_states, position_embeddings, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
            if i == self.config.bridge_layer:
                bridge_states = hidden_states
        h_thinker = self.thinker.norm(hidden_states)

        talker_emb = self.talker.embed_tokens(audio_ids)
        spk_emb = args.get('spk_emb', None)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(spk_mask, self.talker.spk_proj(spk_emb).unsqueeze(1), talker_emb)
        hidden_states = self.talker.embed_proj(bridge_states) * self.talker.text_scale + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        talker_pos_emb = (self.talker.freqs_cos[start_pos:start_pos + seq_length], self.talker.freqs_sin[start_pos:start_pos + seq_length])
        for layer, past_key_value in zip(self.talker.layers, past_key_values[n_thinker:]):
            hidden_states, present = layer(hidden_states, talker_pos_emb, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        h_talker = self.talker.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        aux_loss = sum(l.mlp.aux_loss for l in list(self.thinker.layers) + list(self.talker.layers) if isinstance(l.mlp, MOEFeedForward))
        aux_loss += sum(p.sum() for p in self.audio_proj.parameters()) * 0 + sum(p.sum() for p in self.vision_proj.parameters()) * 0 + sum(p.sum() for p in self.talker.lm_head.adapters.parameters()) * 0 + sum(p.sum() for p in self.talker.spk_proj.parameters()) * 0
        text_logits = self.thinker.lm_head(h_thinker[:, slice_indices, :])
        audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])

        out = MoeCausalLMOutputWithPast(aux_loss=aux_loss, logits=text_logits, past_key_values=presents)
        out.audio_logits = audio_logits
        return out

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, return_audio_codes=False, **args):
        if stream:
            return self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args)
        tokens = list(self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args))
        return tokens[-1] if tokens else input_ids

    def stream_generate(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes=False, **args):
        start_pos, past_kvs, text_finished, first_finished = input_ids.shape[1], None, False, True
        audio_codes = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8
        audio_buffer = torch.full((1, 8, start_pos), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
        spk_emb = args.get('spk_emb', None)
        ref_codes = args.get('ref_codes', None)
        ref_len = ref_codes.shape[2] if ref_codes is not None else 0
        spk_reserve = 1 if spk_emb is not None else 0
        fill_end = start_pos
        fill_start = max(spk_reserve, start_pos - ref_len)
        if ref_codes is not None and fill_start < fill_end:
            audio_buffer[:, :, fill_start:fill_end] = ref_codes[:, :, -(fill_end - fill_start):]
        if spk_emb is not None and fill_start > 0:
            audio_buffer[:, :, fill_start - 1] = self.audio_spk_token
        think_end_step, generated_tokens = None, ([] if args.get('open_thinking', False) else None)
        while input_ids.shape[1] < start_pos + max_new_tokens:
            if past_kvs is None or not use_cache:
                out = self.forward(torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
            else:
                out = self.forward(torch.cat((audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
            past_kvs = out.past_key_values

            logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
            if rp != 1.0:
                seen = list(set(input_ids[0].tolist()))
                score = logits[seen]
                logits[seen] = torch.where(score > 0, score / rp, score * rp)
            if top_p and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                logits[sorted_i[mask]] = -float('Inf')
            text_token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

            if text_finished:
                text_token = args.get('enter_token_id', 201) if first_finished else args.get('pad_token_id', 0)
                first_finished = False

            step = input_ids.shape[1] - start_pos
            audio_step = step - 1
            if generated_tokens is not None:
                generated_tokens.append(text_token)
                if not think_end_step and generated_tokens[-len(self.config.think_end_ids):] == list(self.config.think_end_ids):
                    think_end_step = step + 2
                audio_step = (step - think_end_step) if think_end_step else -1
            for i, al in enumerate(out.audio_logits):
                if audio_step < i:
                    audio_codes[i].append(self.audio_pad_token)
                else:
                    logits_i = al[0, -1, :].clone() / 0.2
                    for prev_code in audio_codes[i][-3:]:
                        score = logits_i[prev_code]
                        logits_i[prev_code] = torch.where(score > 0, score / 1.05, score * 1.05)
                    top_val, top_idx = logits_i.topk(50)
                    code = top_idx[torch.multinomial(F.softmax(top_val, dim=-1), 1)].item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

            if text_finished and all(audio_stop_pos[i] is not None for i in range(8)):
                break

            input_ids = torch.cat((input_ids, torch.tensor([[text_token]], device=input_ids.device)), dim=1)
            audio_buffer = torch.cat((audio_buffer, torch.full((1, 8, 1), self.audio_pad_token, dtype=torch.long, device=input_ids.device)), dim=2)
            for i in range(min(audio_step + 1, 8)):
                audio_buffer[0, i, -1] = audio_codes[i][-1]

            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active_layers = sum(1 for i in range(8) if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i])
                if active_layers >= 8:
                    audio_frame = frame
            if not text_finished:
                yield input_ids[:, start_pos:], audio_frame
                if text_token == eos_token_id:
                    text_finished = True
            else:
                yield None, audio_frame

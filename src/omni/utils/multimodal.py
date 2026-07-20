import os
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from omni.utils.training import Logger, is_main_process
from omni.models import MiniMindOmni


def get_vlm_model_params(model, config, ignore_patterns=('vision_encoder',)):
    def should_count(n):
        return not any(p in n for p in ignore_patterns)

    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model', vision_model_path='../model/siglip2-base-p32-256-ve', save_dir='../out', device='cuda', freeze_llm=0):
    from transformers import AutoTokenizer
    from omni.models.vlm import MiniMindVLM
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)

    if from_weight != 'none':
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 1、全部冻结，只打开 vision_proj 梯度
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    # 2、解冻 LLM（可选，用于全参微调）
    if freeze_llm == 0:
        for name, param in model.named_parameters():
            if 'vision_proj' in name or 'talker' in name or 'audio_proj' in name:
                param.requires_grad = True

    model = model.to(device)
    Logger(f'LLM params: {get_vlm_model_params(model, vlm_config):.2f}M (vision encoder frozen)')
    return model, tokenizer


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


def log_model_params(model, ignore_patterns=('audio_encoder', 'vision_encoder')):
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    cfg = model.config
    n_routed = getattr(cfg, 'n_routed_experts', getattr(cfg, 'num_experts', 0))
    n_active = getattr(cfg, 'num_experts_per_tok', 0)
    n_shared = getattr(cfg, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def init_omni_model(omni_config, from_weight='full_sft', tokenizer_path='../model', audio_encoder_path='../model/SenseVoiceSmall', vision_model_path='../model/siglip2-base-p32-256-ve', save_dir='../out', device='cuda', freeze_backbone='none', from_resume=0):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindOmni(omni_config, audio_encoder_path=audio_encoder_path, vision_model_path=vision_model_path)

    if from_weight != 'none':
        moe_suffix = '_moe' if omni_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            param_shapes = {k: v.shape for k, v in model.named_parameters()}
            incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
            if incompatible:
                Logger(f'跳过shape不匹配的权重: {incompatible}')
                weights = {k: v for k, v in weights.items() if k not in incompatible}
            model.load_state_dict(weights, strict=False)
            Logger(f'已加载权重: {weight_path}')
            if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                n_talker = omni_config.num_talker_hidden_layers
                n_thinker = len(model.thinker.layers)
                has_talker = any(k.startswith('talker.layers.') for k in weights)
                if not has_talker and n_talker > 0:
                    for i in range(n_talker):
                        src = n_thinker - n_talker + i
                        model.talker.layers[i].load_state_dict(model.thinker.layers[src].state_dict())
                    Logger(f'Talker层初始化: 复制thinker layers[{n_thinker-n_talker}:{n_thinker}] → talker layers[0:{n_talker}]')

    if freeze_backbone == 'all':
        for param in model.model.parameters():
            param.requires_grad = False
    elif freeze_backbone == 'last1':
        for param in model.model.parameters():
            param.requires_grad = False
        if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            for param in model.model.layers[-1].parameters():
                param.requires_grad = True
    return model.to(device), tokenizer


def omni_checkpoint(omni_config, weight='pretrain_omni', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if omni_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.') and not k.startswith('vision_encoder.')}
        state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

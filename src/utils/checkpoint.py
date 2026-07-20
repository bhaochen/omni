import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def save_checkpoint(path, state_dict, tmp_suffix=".tmp"):
    tmp = path + tmp_suffix
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def iter_module_state_dict(model):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    raw = getattr(raw, '_orig_mod', raw)
    return raw.state_dict()


def unwrap(model):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw, '_orig_mod', raw)

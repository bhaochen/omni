import os
import torch
import torch.distributed as dist


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_distributed():
    return dist.is_initialized()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


def destroy():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

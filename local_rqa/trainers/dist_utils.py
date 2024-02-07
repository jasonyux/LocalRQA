import torch
import torch.distributed as dist


def barrier():
    if dist.is_initialized():
        torch.distributed.barrier()
    return


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main():
    return get_rank() == 0
def is_main_process():
    try:
        import torch.distributed as dist
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    except:
        return True

import math


def cosine_lr_scheduler(
    it: int, lr_min: float, lr_max: float, warmup_iters: int, cosine_cycle_iters: int
):
    if it < warmup_iters:
        lr = it / warmup_iters * lr_max
    elif it <= cosine_cycle_iters:
        num = it - warmup_iters
        denom = cosine_cycle_iters - warmup_iters
        cos_out = math.cos(num / denom * math.pi)
        lr = lr_min + 1 / 2 * (1 + cos_out) * (lr_max - lr_min)
    else:
        lr = lr_min

    return lr

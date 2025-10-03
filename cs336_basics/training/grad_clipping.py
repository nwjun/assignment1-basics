import torch
from typing import Iterable


def grad_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)

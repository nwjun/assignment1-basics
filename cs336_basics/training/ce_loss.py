import torch
from torch import Tensor


def log_softmax(x: Tensor, dim: int):
    x_max = torch.max(x, dim=dim, keepdim=True).values  # for numerical stability
    x_stable = x - x_max
    logsumexp = torch.log(torch.sum(torch.exp(x_stable), dim=dim, keepdim=True))
    return x_stable - logsumexp


def cross_entropy(logits: Tensor, targets: Tensor):
    log_probs = log_softmax(logits, dim=-1)
    idx = [torch.arange(len(targets)), targets]
    return -log_probs[idx].mean()

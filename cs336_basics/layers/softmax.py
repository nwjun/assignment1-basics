import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # keepdim=True so that broadcasting works
    # return (values, indices)
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_vals  # for numerical stability
    x_exp = torch.exp(x)
    sum_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / sum_exp

from collections.abc import Callable
from typing import Iterable, Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Iterable[float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                exp_avg = state.get("exp_avg", torch.zeros_like(p))
                exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(p))

                grad = p.grad
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad 
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq = beta2 * exp_avg_sq + grad**2 * (1 - beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                lr_t = lr * math.sqrt(bias_c2) / bias_c1

                denom = exp_avg_sq.sqrt().add_(eps)
                # p = p - lr_t * exp_avg / denom
                p.addcdiv_(exp_avg, denom, value=-lr_t)
                # p.data -= lr * weight_decay * p.data
                if weight_decay != 0:
                    p.add_(p, alpha=-lr*weight_decay)

                state["t"] = t + 1
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss

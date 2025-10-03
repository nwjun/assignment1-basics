from collections.abc import Callable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] += 1
        return loss


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

    for lr in [1e1, 1e2, 1e3]:
        losses = []
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            loss_out = loss.cpu().item()
            losses.append(loss_out)
            loss.backward()
            opt.step()

        plt.plot(losses, label=f"lr={lr}")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.show()

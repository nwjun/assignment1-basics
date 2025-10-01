import torch
from torch import nn, Tensor
from typing import Callable

from . import Linear


class GLU(nn.Module):
    def __init__(
        self, func: Callable, d_model: int, d_ff: int, device=None, dtype=None, **kwargs
    ):
        """
        Gated Linear Unit (GLU) layer.
        Args:
            func: Activation function to apply (e.g., SiLU, ReLU).
            d_model: Input feature dimension.
            d_ff: Hidden feature dimension.
            device: Device for the tensors.
            dtype: Data type for the tensors.
        """
        super().__init__(**kwargs)

        self.func = func
        self.linear_1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear_2 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x_func = self.func(self.linear_1(x))
        return x_func * self.linear_2(x)


class SiLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, **kwargs):
        """
        SwiGLU layer.
        Args:
            d_model: Input feature dimension.
            d_ff: Hidden feature dimension.
            device: Device for the tensors.
            dtype: Data type for the tensors.
        """
        super().__init__(**kwargs)
        self.glu = GLU(
            func=SiLU(), d_model=d_model, d_ff=d_ff, device=device, dtype=dtype
        )
        self.output_proj = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(self.glu(x))

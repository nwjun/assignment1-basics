import math
import torch
from torch import nn, Tensor


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None, **kwargs
    ):
        """
        Linear layer with truncated normal initialization.
        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            device: Device for the tensors.
            dtype: Data type for the tensors.
        """
        super().__init__(**kwargs)

        std = math.sqrt(2 / (in_features + out_features))
        a = -3 * std
        b = 3 * std

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0.0, std, a, b)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T

import torch
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device=None, dtype=None, **kwargs
    ):
        """
        Root Mean Square Layer Normalization (RMSNorm) layer.
        Args:
            d_model: Feature dimension.
            eps: Small value to avoid division by zero.
            device: Device for the tensors.
            dtype: Data type for the tensors.
        """
        super().__init__(**kwargs)
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def rms(self, x):
        out = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return out

    def forward(self, x: Tensor) -> Tensor:
        # upcast input to torch.float32 before normalization and downcast to original dtype
        in_dtype = x.dtype

        x = x.to(dtype=torch.float32)
        x = x * self.gain / self.rms(x)

        return x.to(dtype=in_dtype)

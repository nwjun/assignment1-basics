from einops import rearrange
import torch
from torch import nn, Tensor
from jaxtyping import Bool

from . import Linear, RoPE, softmax


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Tensor:
    """
    Scaled Dot-Product Attention.
    Args:
        q: Query tensor of shape (..., num_queries, head_dim).
        k: Key tensor of shape (..., num_keys, head_dim).
        v: Value tensor of shape (..., num_keys, head_dim).
        mask: Optional boolean mask tensor of shape (..., num_queries, num_keys),
              where True indicates positions to attend to and False indicates positions to ignore.
    Returns:
        Tensor of shape (..., num_queries, head_dim) after applying attention.
    """
    head_dim = k.shape[-1]
    scale = torch.sqrt(torch.tensor(head_dim, dtype=q.dtype, device=q.device))
    scores = q @ k.transpose(-2, -1) / scale

    if mask is not None:
        # mask True = attend
        # mask False = block → fill with -inf
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)

    return attn @ v


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        """
        Multi-head Self-Attention layer with optional RoPE.
        Args:
            d_model: Dimension of the input features.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length for RoPE (if using RoPE).
            theta: RoPE parameter (if using RoPE).
            device: Device for the tensors.
            dtype: Data type for the tensors.
        """
        super().__init__(**kwargs)

        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        self.linear_q = Linear(d_model, d_model, device, dtype)
        self.linear_k = Linear(d_model, d_model, device, dtype)
        self.linear_v = Linear(d_model, d_model, device, dtype)
        self.linear_o = Linear(d_model, d_model, device, dtype)

        if max_seq_len and theta is not None:
            self.rope = RoPE(self.head_dim, theta, max_seq_len)

    def forward(self, x: Tensor, token_pos=None):
        # x: (batch, seq, d_model)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q = rearrange(q, "b t (h d) -> (b h) t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> (b h) t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> (b h) t d", h=self.num_heads)

        if hasattr(self, "rope") and token_pos is not None:
            token_pos = (
                torch.arange(q.shape[-2], device=x.device)
                .unsqueeze(0)
                .expand(q.shape[0], -1)
            )
            q = self.rope(q, token_pos)
            k = self.rope(k, token_pos)

        seq_len = x.shape[1]
        mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device) 
        )
        mask = mask.expand(q.shape[0], -1, -1)  # match batch*num_heads

        x = scaled_dot_product_attention(q, k, v, mask)
        x = rearrange(x, "(b h) t d -> b t (h d)", h=self.num_heads)

        x = self.linear_o(x)

        return x

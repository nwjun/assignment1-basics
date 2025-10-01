import torch
from torch import nn, Tensor


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        """
        Args:
            d_k (int): Embedding dimension size for the query or key tensor.
            theta (float): RoPE parameter.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Precompute sin and cos buffers
        position = torch.arange(max_seq_len, device=device).float()
        dim = torch.arange(d_k // 2, device=device).float()
        freqs = 1.0 / (theta ** (2 * dim / d_k))

        angles = position[:, None] * freqs[None, :]  # (seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        if token_positions.dtype != torch.long:
            token_positions = token_positions.long()

        # Index into precomputed tables
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)

        # Split even/odd channels
        x_e = x[..., ::2]  # (..., seq_len, d_k/2)  even indices
        x_o = x[..., 1::2]  # (..., seq_len, d_k/2)  odd indices

        # Rotate and re-interleave
        x_rot_e = x_e * cos - x_o * sin
        x_rot_o = x_e * sin + x_o * cos
        x_rot = torch.stack((x_rot_e, x_rot_o), dim=-1).flatten(
            -2
        )  # interleave back to (..., seq_len, d_k)

        return x_rot

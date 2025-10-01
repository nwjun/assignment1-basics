import torch
from torch import nn, Tensor


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, device=None, dtype=None, **kwargs
    ):
        """
        Embedding layer with truncated normal initialization.
        Args:
            num_embeddings: Size of the vocabulary.
            embedding_dim: Dimension of each embedding vector.
            device: Device for the tensors.
            dtype: Data type for the tensors.
        """
        super().__init__(**kwargs)

        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Initialize weights with truncated normal distribution
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding[token_ids]

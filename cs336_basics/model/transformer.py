from torch import nn, Tensor

from ..layers import MultiheadSelfAttention, RMSNorm, SwiGLU, Embedding, Linear


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attn = MultiheadSelfAttention(
            d_model, num_heads, theta=theta, max_seq_len=max_seq_len
        )
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor, token_pos=None):
        x = x + self.attn(self.ln1(x), token_pos=token_pos)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleDict({})
        for i in range(num_layers):
            self.layers[str(i)] = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
            )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: Tensor):
        x = self.token_embeddings(x)

        for i in range(len(self.layers)):
            x = self.layers[str(i)](x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

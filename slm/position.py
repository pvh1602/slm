"""Positional encoding implementations for the tiny transformer."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    f"""Fixed sinusoidal positional encoding.

    Objective:
        Add deterministic sine/cosine position vectors to token embeddings.
        For position `p` and channel index `i` (the i-th element in the d_model-dimensional embedding vector), define:
        PE[p, 2i]   = sin(p / 10000^(2i / d_model))
        PE[p, 2i+1] = cos(p / 10000^(2i / d_model))

        we have 1 / 10^(1/x) = exp(-log(10)/x)

    Expected input:
        x: Tensor [batch, seq_len, d_model]

    Expected return:
        Tensor [batch, seq_len, d_model]
    """

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        # TODO: create buffer pe with shape [1, max_seq_len, d_model].
        # raise NotImplementedError

        # self.register_buffer("pe", torch.zeros(1, max_seq_len, d_model))
        pe = torch.zeros(max_seq_len, d_model)
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0)) # pe.shape = (1, max_seq_len, d_model)



    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Add sinusoidal position vectors.

        Args:
            x: Token embeddings [batch, seq_len, d_model].
            start_pos: Absolute start position for incremental decoding.

        Returns:
            Position-augmented embeddings [batch, seq_len, d_model].
        """

        seq_len = x.shape[1]
        return x + self.pe[:,start_pos:start_pos + seq_len]


class LearnedAbsolutePositionalEncoding(nn.Module):
    """Learned absolute position embeddings."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        # TODO: create nn.Embedding(max_seq_len, d_model).
        raise NotImplementedError

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Add learned absolute position embeddings.

        Args:
            x: Token embeddings [batch, seq_len, d_model].
            start_pos: Absolute position of x[:, 0].

        Returns:
            Position-augmented embeddings [batch, seq_len, d_model].
        """

        raise NotImplementedError


def build_alibi_bias(
    n_heads: int,
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ALiBi attention bias.

    Objective:
        Return a per-head linear distance penalty to add to attention logits.

    Args:
        n_heads: Number of attention heads.
        q_len: Query sequence length.
        k_len: Key sequence length, including cached keys if any.
        device: Output device.
        dtype: Output dtype.

    Returns:
        Bias tensor broadcastable to attention logits:
            [1, n_heads, q_len, k_len]

    TODO:
        Implement ALiBi slopes and relative-distance bias.
    """

    raise NotImplementedError


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables.

    Args:
        head_dim: Per-head dimension. Must be even.
        max_seq_len: Maximum positions to precompute.
        theta: RoPE base.
        device: Optional device.

    Returns:
        Tuple `(cos, sin)`, each [max_seq_len, head_dim // 2].

    TODO:
        Implement inverse frequencies and outer product with positions.
    """

    raise NotImplementedError


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Apply RoPE to query/key tensors.

    Args:
        x: Tensor [batch, n_heads, seq_len, head_dim].
        cos: Precomputed cos table [max_seq_len, head_dim // 2].
        sin: Precomputed sin table [max_seq_len, head_dim // 2].
        start_pos: Absolute position of x[:, :, 0].

    Returns:
        RoPE-rotated tensor with same shape as x.

    TODO:
        Split x into even/odd or front/back real-imag pairs, rotate, and return.
        Pick one layout and document it. The rest of the project should use the
        same layout consistently.
    """

    raise NotImplementedError



if __name__ == "__main__":

    d_model = 512
    max_seq_len = 100
    bs = 3
    sin_pe = SinusoidalPositionalEncoding(d_model, max_seq_len)
    print(sin_pe.pe.shape)
    x = torch.zeros((bs, max_seq_len, d_model))
    print(x.shape)
    out = sin_pe(x)
    # print(out)
    # import matplotlib.pyplot as plt

    # cax = plt.matshow(sin_pe.pe.squeeze(0))
    # plt.gcf().colorbar(cax)
    # plt.savefig("../assets/sinosoid_pe.png")
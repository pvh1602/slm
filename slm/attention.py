"""Causal self-attention with optional positional schemes and KV cache."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PosEncodingName
from .kv_cache import LayerKVCache, append_to_cache
from .position import apply_rope, build_alibi_bias, precompute_rope_frequencies


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Objective:
        Implement attention for both training full sequences and incremental
        decoding with KV cache.

    Input shape:
        x: [batch, q_len, d_model]

    Output shape:
        y: [batch, q_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        pos_encoding: PosEncodingName,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pos_encoding = pos_encoding

        # TODO: define q_proj, k_proj, v_proj, out_proj, dropout.
        # TODO: if pos_encoding == "rope", precompute cos/sin buffers.
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: LayerKVCache | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Run causal self-attention.

        Args:
            x: Hidden states [batch, q_len, d_model].
            layer_cache: Optional cache for this layer during generation.
            start_pos: Absolute position of x[:, 0].
            use_cache: If True, append new K/V to cache and attend over all cached keys.

        Returns:
            Attention output [batch, q_len, d_model].

        TODO:
            1. Project q/k/v.
            2. Reshape to [batch, n_heads, q_len, head_dim].
            3. Apply RoPE to q/k if enabled.
            4. Append k/v to cache if use_cache.
            5. Compute scaled dot-product attention.
            6. Add causal mask.
            7. Add ALiBi bias if enabled.
            8. Softmax, weighted sum over values, output projection.
        """

        raise NotImplementedError


def build_causal_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    past_len: int = 0,
) -> torch.Tensor:
    """Build additive causal mask for attention logits.

    Args:
        q_len: Number of current query tokens.
        k_len: Number of total key tokens, including past cache.
        device: Mask device.
        dtype: Mask dtype.
        past_len: Number of keys that came from previous cache.

    Returns:
        Additive mask [1, 1, q_len, k_len], with 0 for allowed positions and
        large negative values for disallowed future positions.

    TODO:
        Implement carefully for both training (`past_len=0`) and decoding.
    """

    raise NotImplementedError

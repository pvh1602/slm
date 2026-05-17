"""Causal self-attention with optional positional schemes and KV cache."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops 

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
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        
        # TODO: if pos_encoding == "rope", precompute cos/sin buffers.
        # raise NotImplementedError


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

        q_len = x.shape[1]
        if use_cache and layer_cache is not None:
            past_len = layer_cache.key.shape[2]
        else:
            past_len = 0

        # Project q/k/v
        q = self.q_proj(x)
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)

        # Reshape to [batch, n_heads, q_len, head_dim]
        q = einops.rearrange(q, 'b l (n h) -> b n l h', n=self.n_heads)
        k_new = einops.rearrange(k_new, 'b l (n h) -> b n l h', n=self.n_heads)
        v_new = einops.rearrange(v_new, 'b l (n h) -> b n l h', n=self.n_heads)

        # TODO Apply RoPE if enabled

        # TODO Append k/v to cache if use_cache.
        if use_cache:
            k_all = torch.cat([layer_cache.key, k_new], dim=2) # if cache exist
            v_all = torch.cat([layer_cache.value, v_new], dim=2)
            k_len = k_all.shape[2]
        else:
            k_all = k_new
            v_all = v_new
            k_len = q_len

        # Compute scaled dot-product attention.
        attn_score = einops.einsum(q, k_all, 'b n q_l h, b n k_l h -> b n q_l k_l') / (self.head_dim ** 0.5)

        # Add casual mask
        attn_mask = build_causal_mask(q_len, k_len, x.device, x.dtype, past_len)
        attn_score = attn_score + attn_mask

        # TODO Add ALiBi bias if enabled.


        # Softmax 
        attn_score = torch.softmax(attn_score, dim=-1)
        if self.dropout is not None:
            attn_score = self.dropout(attn_score)
        
        attn_weight = einops.einsum(attn_score, v_all, 'b n q k, b n k h -> b n q h')

        # Merge heads
        attn_weight = einops.rearrange(attn_weight, 'b n q h -> b q (n h)')

        attn_weight = self.out_proj(attn_weight)

        return attn_weight



        # raise NotImplementedError


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

    # # Training mode, full sequence
    # if past_len == 0: 
    #     mask = torch.tril(torch.ones(q_len, k_len), dtype=dtype, device=device).view(1, 1, q_len, k_len)
    #     mask = mask.masked_fill(mask==0,float('-inf'))
    #     mask = mask.masked_fill(mask==1, 0)

    # # Decoding mode
    # if past_len != 0:
    #     assert past_len == k_len - q_len, print("The k_len should be equal past_len + q_len")
    #     mask = torch.zeros((q_len, k_len), device=device, dtype=dtype)
    #     allowed = 
    #     mask = mask.masked_fill()

    q_pos = torch.arange(q_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]

    # If using cache, current query row i corresponds to absolute cache index past_len + i
    allowed = k_pos <= (past_len + q_pos)
    mask = torch.zeros((q_len, k_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, value=float('-inf'))
    
    return mask[None, None, :, :]

    # raise NotImplementedError

"""Tiny decoder-only transformer language model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .kv_cache import KVCache
from .position import LearnedAbsolutePositionalEncoding, SinusoidalPositionalEncoding
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Transformer MLP block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: Linear d_model -> d_ff, GELU, Linear d_ff -> d_model, Dropout.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Tensor [batch, seq_len, d_model].

        Returns:
            Tensor [batch, seq_len, d_model].
        """

        raise NotImplementedError


class TransformerBlock(nn.Module):
    """Pre-LN decoder transformer block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # TODO: LayerNorm, CausalSelfAttention, LayerNorm, FeedForward.
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        layer_cache=None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Run one transformer block.

        Args:
            x: Hidden states [batch, seq_len, d_model].
            layer_cache: Optional LayerKVCache for this block.
            start_pos: Absolute position of first token in x.
            use_cache: Whether to append/use KV cache.

        Returns:
            Updated hidden states [batch, seq_len, d_model].
        """

        raise NotImplementedError


class TinyTransformerLM(nn.Module):
    """Small decoder-only Transformer language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.vocab_size is None:
            raise ValueError("cfg.vocab_size must be set before constructing model")
        self.cfg = cfg

        # TODO:
        # - token embedding
        # - optional sinusoidal or learned absolute pos embedding
        # - transformer blocks
        # - final layer norm
        # - lm_head
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the language model.

        Args:
            input_ids: Token ids [batch, seq_len].
            targets: Optional next-token targets [batch, seq_len].
            loss_mask: Optional float mask [batch, seq_len], 1 for valid loss positions.
            kv_cache: Optional KVCache used during generation.
            start_pos: Absolute position of input_ids[:, 0].
            use_cache: If True, blocks append/use KV cache.

        Returns:
            Dict containing:
                logits: [batch, seq_len, vocab_size]
                loss: scalar tensor if targets are provided, else absent or None.

        TODO:
            Implement embeddings, blocks, logits, and masked CE loss.
        """

        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate token ids.

        Args:
            input_ids: Prompt ids [batch, prompt_len].
            max_new_tokens: Number of new tokens to sample.
            temperature: Sampling temperature. Use 0 or greedy path if you prefer.
            use_cache: Whether to use incremental KV cache.

        Returns:
            Full generated ids [batch, prompt_len + max_new_tokens].

        TODO:
            Implement two paths:
            - no cache: feed full prefix every step,
            - with cache: prefill once, then feed only last token.
        """

        raise NotImplementedError

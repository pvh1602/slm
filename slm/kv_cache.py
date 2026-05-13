"""KV-cache data structures and simple pruning policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    """KV cache for one transformer layer.

    Attributes:
        key: Tensor [batch, n_heads, cache_len, head_dim] or None.
        value: Tensor [batch, n_heads, cache_len, head_dim] or None.
        positions: LongTensor [cache_len] storing absolute token positions.
    """

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    positions: torch.Tensor | None = None


class KVCache:
    """KV cache for all transformer layers."""

    def __init__(self, n_layers: int):
        """Create empty caches.

        Args:
            n_layers: Number of transformer layers.
        """

        self.layers = [LayerKVCache() for _ in range(n_layers)]

    def clear(self) -> None:
        """Reset all layer caches to empty."""

        for layer in self.layers:
            layer.key = None
            layer.value = None
            layer.positions = None


def append_to_cache(
    layer_cache: LayerKVCache,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    new_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append new K/V states to one layer cache.

    Args:
        layer_cache: Mutable cache for one layer.
        new_key: New keys [batch, n_heads, q_len, head_dim].
        new_value: New values [batch, n_heads, q_len, head_dim].
        new_positions: Absolute positions [q_len].

    Returns:
        `(all_key, all_value)` after appending, each
        [batch, n_heads, cache_len + q_len, head_dim].

    TODO:
        Concatenate old and new K/V on dim=2 and update layer_cache.
    """

    raise NotImplementedError


def prune_cache_keep_last(layer_cache: LayerKVCache, keep_last: int) -> None:
    """Prune one layer cache to keep only the most recent tokens.

    Args:
        layer_cache: Cache to modify in place.
        keep_last: Number of most recent cache entries to keep.

    Returns:
        None. The cache object is modified in place.

    TODO:
        Slice key/value/positions along cache dimension.
    """

    raise NotImplementedError


def prune_cache_bos_plus_last(layer_cache: LayerKVCache, keep_last: int) -> None:
    """Keep first token plus most recent tokens.

    Args:
        layer_cache: Cache to modify in place.
        keep_last: Number of recent entries to keep in addition to slot 0.

    Returns:
        None.

    TODO:
        Build keep indices like [0] + last keep_last positions, remove duplicates,
        and gather key/value/positions.
    """

    raise NotImplementedError


def prune_cache_strided_plus_last(
    layer_cache: LayerKVCache,
    stride: int,
    keep_last: int,
) -> None:
    """Keep a strided subset of old tokens plus recent tokens.

    Args:
        layer_cache: Cache to modify in place.
        stride: Keep every `stride`-th old token.
        keep_last: Always keep this many most recent tokens.

    Returns:
        None.

    TODO:
        This is a toy baseline for sparse long-context memory.
    """

    raise NotImplementedError

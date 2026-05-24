# =============================================================================
# FILE: speculative_decoding/core/kv_cache_manager.py
# PURPOSE: Manage HuggingFace-style past_key_values with rollback semantics.
# =============================================================================
"""Lightweight wrapper around ``past_key_values`` adding rollback support."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..exceptions import KVCacheRollbackError

# A HuggingFace past_key_values tuple. Each element is (key, value).
# Key/value tensors have shape (batch, num_heads, seq_len, head_dim).
PastKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

# Axis index of the sequence-length dimension inside an HF KV-cache tensor.
SEQ_LEN_AXIS: int = 2


def rollback_past_kv(past_kv: PastKV, n: int) -> PastKV:
    """Functionally truncate ``n`` positions from the end of every KV tensor.

    This is the canonical rollback for an HF-style past_key_values. Used both
    internally by :class:`KVCacheManager` and (occasionally) directly by the
    decoder when it is convenient to manipulate the raw tuple.

    Args:
        past_kv: The cache to truncate.
        n: Number of positions to drop from the end. ``n == 0`` is a no-op.

    Returns:
        A new ``PastKV`` whose sequence length is reduced by ``n`` positions.

    Raises:
        KVCacheRollbackError: If ``n`` is negative.
    """
    if n < 0:
        raise KVCacheRollbackError(f"rollback amount must be >= 0; got {n}")
    if n == 0:
        return past_kv
    rolled: list[tuple[torch.Tensor, torch.Tensor]] = []
    for (k, v) in past_kv:
        # ``[:, :, :-n, :]`` is a view; ``.contiguous()`` materialises it so
        # downstream attention kernels never silently re-copy.
        new_k = k[:, :, :-n, :].contiguous()  # shape: [B, H, L-n, D]
        new_v = v[:, :, :-n, :].contiguous()  # shape: [B, H, L-n, D]
        rolled.append((new_k, new_v))
    return tuple(rolled)


def get_past_kv_length(past_kv: Optional[PastKV]) -> int:
    """Return the sequence length of an HF past_key_values tuple.

    Args:
        past_kv: The cache to inspect. ``None`` is treated as empty.

    Returns:
        Number of positions cached (0 for an empty/``None`` cache).
    """
    if past_kv is None or len(past_kv) == 0:
        return 0
    # Every layer must agree on sequence length; we read the first.
    first_key = past_kv[0][0]
    return int(first_key.shape[SEQ_LEN_AXIS])


class KVCacheManager:
    """Owns the current ``past_key_values`` for one model and supports rollback.

    A :class:`KVCacheManager` is a thin wrapper that exists so the decoder
    can manipulate caches without each call site re-implementing rollback or
    length-tracking logic. There is one manager per model (draft, target).

    Attributes:
        past_kv: The current cache tuple, or ``None`` if no forward passes
            have run yet.
    """

    def __init__(self, past_kv: Optional[PastKV] = None) -> None:
        """Construct a cache manager, optionally seeded with an existing cache.

        Args:
            past_kv: Initial cache, typically ``None`` for a fresh model.
        """
        self.past_kv: Optional[PastKV] = past_kv

    # ---- Public API -----------------------------------------------------

    def update(self, past_kv: PastKV) -> None:
        """Replace the current cache with the one returned by a forward pass.

        Args:
            past_kv: The new cache (typically from a model's forward call).
        """
        self.past_kv = past_kv

    def rollback(self, n: int) -> PastKV:
        """Truncate the last ``n`` positions from every layer's K and V.

        Args:
            n: Non-negative number of positions to drop. ``n == 0`` is a no-op
                and returns the cache unchanged.

        Returns:
            The newly-truncated cache (also stored as ``self.past_kv``).

        Raises:
            KVCacheRollbackError: If ``n < 0`` or ``n`` exceeds the current
                cached length.
        """
        if n < 0:
            raise KVCacheRollbackError(f"rollback amount must be >= 0; got {n}")
        if n == 0:
            # Trivial no-op; keep the existing cache (possibly None).
            return self.past_kv if self.past_kv is not None else tuple()
        current_len = self.get_sequence_length()
        if n > current_len:
            raise KVCacheRollbackError(
                f"cannot rollback {n} positions; cache only has {current_len}"
            )
        assert self.past_kv is not None  # implied by current_len > 0
        self.past_kv = rollback_past_kv(self.past_kv, n)
        return self.past_kv

    def get_sequence_length(self) -> int:
        """Return the number of positions currently cached.

        Returns:
            An integer >= 0.
        """
        return get_past_kv_length(self.past_kv)

    def current(self) -> Optional[PastKV]:
        """Return the current cache tuple (or ``None`` if empty)."""
        return self.past_kv

    def reset(self) -> None:
        """Drop the cache entirely (next forward pass will start fresh)."""
        self.past_kv = None

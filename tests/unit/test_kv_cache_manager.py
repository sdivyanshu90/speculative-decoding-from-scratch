# =============================================================================
# FILE: tests/unit/test_kv_cache_manager.py
# PURPOSE: Unit tests for KVCacheManager rollback semantics.
# =============================================================================
"""Unit tests for :class:`KVCacheManager`.

The shared helper :func:`tests.conftest.make_past_kv` constructs a synthetic
cache whose K and V values encode the position index, so tests can verify
that rollback drops the **correct** positions, not just the correct count.
"""

from __future__ import annotations

import pytest
import torch

from speculative_decoding import KVCacheManager, KVCacheRollbackError
from tests.conftest import (
    DEFAULT_HEAD_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    make_past_kv,
)


def test_rollback_by_n() -> None:
    """Extending the cache to length 10 then rolling back 3 yields length 7."""
    pkv = make_past_kv(seq_len=10)
    mgr = KVCacheManager(pkv)
    assert mgr.get_sequence_length() == 10
    mgr.rollback(3)
    assert mgr.get_sequence_length() == 7
    # The retained positions must be 0..6 (not 3..9).
    for (k, v) in mgr.current():  # type: ignore[union-attr]
        # k[b, h, i, d] == i in our synthetic fixture.
        retained_indices = k[0, 0, :, 0].tolist()
        assert retained_indices == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # V is the same but offset by 1000.0.
        retained_v = v[0, 0, :, 0].tolist()
        assert retained_v == [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0]


def test_rollback_zero_is_noop() -> None:
    """Rolling back 0 returns the cache unchanged."""
    pkv = make_past_kv(seq_len=5)
    mgr = KVCacheManager(pkv)
    before_k = mgr.current()[0][0].clone()  # type: ignore[index]
    mgr.rollback(0)
    after_k = mgr.current()[0][0]  # type: ignore[index]
    assert mgr.get_sequence_length() == 5
    assert torch.equal(before_k, after_k)


def test_rollback_exceeds_length_raises() -> None:
    """Rolling back more positions than cached raises KVCacheRollbackError."""
    pkv = make_past_kv(seq_len=3)
    mgr = KVCacheManager(pkv)
    with pytest.raises(KVCacheRollbackError) as exc_info:
        mgr.rollback(4)
    assert "cannot rollback" in str(exc_info.value).lower()


def test_rollback_negative_raises() -> None:
    """Negative rollback amount raises KVCacheRollbackError."""
    mgr = KVCacheManager(make_past_kv(seq_len=3))
    with pytest.raises(KVCacheRollbackError):
        mgr.rollback(-1)


def test_rollback_preserves_batch_dimension() -> None:
    """Rollback only touches axis 2 (seq); batch dim must be unaffected."""
    pkv = make_past_kv(seq_len=8, batch=3)
    mgr = KVCacheManager(pkv)
    mgr.rollback(3)
    for (k, v) in mgr.current():  # type: ignore[union-attr]
        assert k.shape[0] == 3  # batch
        assert v.shape[0] == 3
        assert k.shape[2] == 5  # seq_len after rollback
        assert v.shape[2] == 5


def test_rollback_preserves_head_dimension() -> None:
    """Head and head_dim dimensions must be unaffected by rollback."""
    pkv = make_past_kv(seq_len=8)
    mgr = KVCacheManager(pkv)
    mgr.rollback(2)
    for (k, v) in mgr.current():  # type: ignore[union-attr]
        assert k.shape[1] == DEFAULT_NUM_HEADS
        assert k.shape[3] == DEFAULT_HEAD_DIM
        assert v.shape[1] == DEFAULT_NUM_HEADS
        assert v.shape[3] == DEFAULT_HEAD_DIM


def test_sequence_length_tracking() -> None:
    """get_sequence_length() must reflect the most recent update / rollback."""
    mgr = KVCacheManager()
    assert mgr.get_sequence_length() == 0
    mgr.update(make_past_kv(seq_len=5))
    assert mgr.get_sequence_length() == 5
    mgr.rollback(2)
    assert mgr.get_sequence_length() == 3
    mgr.reset()
    assert mgr.get_sequence_length() == 0


def test_multi_layer_rollback_consistency() -> None:
    """All layers must end up with the same sequence length after rollback."""
    pkv = make_past_kv(seq_len=10, num_layers=DEFAULT_NUM_LAYERS)
    mgr = KVCacheManager(pkv)
    mgr.rollback(4)
    seq_lens = {k.shape[2] for (k, v) in mgr.current()}  # type: ignore[union-attr]
    seq_lens.update({v.shape[2] for (k, v) in mgr.current()})  # type: ignore[union-attr]
    assert seq_lens == {6}


def test_rollback_on_empty_cache_with_zero_is_safe() -> None:
    """Rollback(0) on a never-updated manager must not raise."""
    mgr = KVCacheManager()
    mgr.rollback(0)
    assert mgr.get_sequence_length() == 0


def test_rollback_on_empty_cache_with_nonzero_raises() -> None:
    """Rollback(n>0) on an empty cache raises."""
    mgr = KVCacheManager()
    with pytest.raises(KVCacheRollbackError):
        mgr.rollback(1)


def test_current_returns_none_when_never_updated() -> None:
    """A freshly-constructed manager has no cache."""
    mgr = KVCacheManager()
    assert mgr.current() is None

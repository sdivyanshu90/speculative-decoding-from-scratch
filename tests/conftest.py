# =============================================================================
# FILE: tests/conftest.py
# PURPOSE: Shared pytest fixtures: mock language models, helpers, sample config.
# =============================================================================
"""Shared test fixtures.

All tests rely on :class:`MockHFLikeModel` — a small ``nn.Module`` that quacks
like a HuggingFace causal LM (accepts ``input_ids`` / ``past_key_values`` /
``use_cache=True`` and returns a namespace with ``.logits`` and
``.past_key_values``). The cache it returns is a shape-correct
``tuple[tuple[Tensor, Tensor], ...]`` so the real :class:`KVCacheManager`
rollback math is exercised end-to-end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Optional

import pytest
import torch
from torch import nn

from speculative_decoding import (
    DraftModel,
    SpeculativeDecodingConfig,
    TargetModel,
)
from speculative_decoding.core.kv_cache_manager import (
    PastKV,
    get_past_kv_length,
)

# ---- Mock-model defaults ---------------------------------------------------
DEFAULT_VOCAB_SIZE: int = 16
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_NUM_HEADS: int = 2
DEFAULT_HEAD_DIM: int = 4

# Type alias for logits-computing functions used by the mock.
LogitsFn = Callable[[torch.Tensor, int], torch.Tensor]


# ============================================================================
# MOCK HF-LIKE MODEL
# ============================================================================


class MockHFLikeModel(nn.Module):
    """A minimal stand-in for an HF ``PreTrainedModel`` used by all tests.

    Args:
        vocab_size: Vocabulary size.
        logits_fn: Function ``(input_ids[B, S], past_kv_len) -> logits[B, S, V]``.
            Defaults to a uniform logits-of-zero (i.e. a flat distribution).
        num_layers / num_heads / head_dim: Cache tensor shape parameters.
        max_position_embeddings: Optional context-window limit. ``None``
            disables the check.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        logits_fn: Optional[LogitsFn] = None,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_heads: int = DEFAULT_NUM_HEADS,
        head_dim: int = DEFAULT_HEAD_DIM,
        max_position_embeddings: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.logits_fn: LogitsFn = logits_fn or self._uniform_logits
        cfg_kwargs: dict[str, int] = {"vocab_size": vocab_size}
        if max_position_embeddings is not None:
            cfg_kwargs["max_position_embeddings"] = max_position_embeddings
        self.config = SimpleNamespace(**cfg_kwargs)
        # Diagnostic — every forward call appends a (past_len, input_ids) tuple.
        self.forward_call_log: list[tuple[int, torch.Tensor]] = []

    def _uniform_logits(self, input_ids: torch.Tensor, past_len: int) -> torch.Tensor:
        """Default logits_fn: zeros everywhere -> uniform distribution."""
        return torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            dtype=torch.float32,
        )

    def _extend_cache(
        self,
        past_kv: Optional[PastKV],
        past_len: int,
        n_new: int,
        batch: int,
        device: torch.device,
    ) -> PastKV:
        """Append ``n_new`` zero-initialized positions to every layer's K, V."""
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            new_part = torch.zeros(
                batch,
                self.num_heads,
                n_new,
                self.head_dim,
                dtype=torch.float32,
                device=device,
            )
            if past_kv is not None and layer_idx < len(past_kv):
                old_k, old_v = past_kv[layer_idx]
                # Sanity check: old cache's seq_len must equal the past_len
                # the model was told about.
                assert old_k.shape[2] == past_len, (
                    f"layer {layer_idx} key seq_len {old_k.shape[2]} != past_len {past_len}"
                )
                k = torch.cat([old_k, new_part], dim=2)
                v = torch.cat([old_v, new_part], dim=2)
            else:
                k = new_part.clone()
                v = new_part.clone()
            layers.append((k, v))
        return tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[PastKV] = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        """HF-style forward returning a namespace with logits + past_kv."""
        past_len = get_past_kv_length(past_key_values)
        logits = self.logits_fn(input_ids, past_len)
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"logits_fn returned vocab dim {logits.shape[-1]}; expected {self.vocab_size}"
            )
        new_kv = self._extend_cache(
            past_key_values,
            past_len=past_len,
            n_new=input_ids.shape[1],
            batch=input_ids.shape[0],
            device=input_ids.device,
        )
        self.forward_call_log.append((past_len, input_ids.detach().clone()))
        return SimpleNamespace(logits=logits, past_key_values=new_kv)

    # ``DraftModel`` calls ``hf_model.eval()`` at construction time; nn.Module
    # gives us this for free, but we override to keep the chain explicit.
    def eval(self) -> "MockHFLikeModel":  # type: ignore[override]
        super().eval()
        return self


# ============================================================================
# LOGITS-FN HELPERS
# ============================================================================


def fixed_logits(logits_per_token: torch.Tensor) -> LogitsFn:
    """Return a logits_fn that yields the same ``[V]`` logits at every position.

    Args:
        logits_per_token: Tensor of shape ``[vocab]``. Returned (broadcast)
            for every position the mock is asked about.
    """
    vec = logits_per_token.detach().clone()

    def fn(input_ids: torch.Tensor, past_len: int) -> torch.Tensor:
        b, s = input_ids.shape
        return vec.view(1, 1, -1).expand(b, s, vec.shape[0]).contiguous()

    return fn


def argmax_favoring(vocab_size: int, favored_token: int, strength: float = 100.0) -> LogitsFn:
    """Return a logits_fn that strongly favors a single token at every step."""
    base = torch.zeros(vocab_size)
    base[favored_token] = strength
    return fixed_logits(base)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture()
def vocab_size() -> int:
    return DEFAULT_VOCAB_SIZE


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def dtype() -> torch.dtype:
    return torch.float32


@pytest.fixture()
def small_config(vocab_size: int) -> SpeculativeDecodingConfig:
    """A baseline config used by integration / loop tests."""
    return SpeculativeDecodingConfig(
        speculation_length=3,
        max_new_tokens=16,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        target_model_name="mock-target",
        draft_model_name="mock-draft",
        device="cpu",
        dtype=torch.float32,
        seed=12345,
        eos_token_id=None,
    )


@pytest.fixture()
def uniform_draft(vocab_size: int, device: torch.device, dtype: torch.dtype) -> DraftModel:
    """Draft model that emits a uniform distribution at every position."""
    hf = MockHFLikeModel(vocab_size=vocab_size)
    return DraftModel(hf, device=device, dtype=dtype)


@pytest.fixture()
def uniform_target(vocab_size: int, device: torch.device, dtype: torch.dtype) -> TargetModel:
    """Target model that emits a uniform distribution at every position."""
    hf = MockHFLikeModel(vocab_size=vocab_size)
    return TargetModel(hf, device=device, dtype=dtype)


@pytest.fixture()
def matched_models(
    vocab_size: int, device: torch.device, dtype: torch.dtype
) -> tuple[DraftModel, TargetModel]:
    """Draft and target with identical logits — should yield 100% acceptance."""
    # Use a non-uniform but deterministic logits_fn so the argmax token is well-
    # defined and matches across draft and target.
    favored = argmax_favoring(vocab_size, favored_token=7, strength=10.0)
    draft_hf = MockHFLikeModel(vocab_size=vocab_size, logits_fn=favored)
    target_hf = MockHFLikeModel(vocab_size=vocab_size, logits_fn=favored)
    return (
        DraftModel(draft_hf, device=device, dtype=dtype),
        TargetModel(target_hf, device=device, dtype=dtype),
    )


@pytest.fixture()
def mismatched_models(
    vocab_size: int, device: torch.device, dtype: torch.dtype
) -> tuple[DraftModel, TargetModel]:
    """Draft favors token 1, target favors token 2 — high rejection rate."""
    draft_hf = MockHFLikeModel(
        vocab_size=vocab_size, logits_fn=argmax_favoring(vocab_size, 1, strength=20.0)
    )
    target_hf = MockHFLikeModel(
        vocab_size=vocab_size, logits_fn=argmax_favoring(vocab_size, 2, strength=20.0)
    )
    return (
        DraftModel(draft_hf, device=device, dtype=dtype),
        TargetModel(target_hf, device=device, dtype=dtype),
    )


# ============================================================================
# HELPERS FOR CACHE-CONSTRUCTION TESTS
# ============================================================================


def make_past_kv(
    seq_len: int,
    *,
    batch: int = 1,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    fill_with_position_index: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> PastKV:
    """Build a synthetic PastKV with each position's value = its global index.

    With ``fill_with_position_index=True`` (the default), every ``[b, h, i, d]``
    entry equals ``i``, so tests can verify that rollback truly drops the
    correct positions (the surviving tensor must contain values ``0..L-n-1``).
    """
    device = device or torch.device("cpu")
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    if fill_with_position_index:
        idx = torch.arange(seq_len, dtype=dtype, device=device)  # shape: [L]
        per_pos = idx.view(1, 1, seq_len, 1).expand(
            batch, num_heads, seq_len, head_dim
        ).contiguous()
    else:
        per_pos = torch.zeros(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    for _ in range(num_layers):
        layers.append((per_pos.clone(), per_pos.clone() + 1000.0))
    return tuple(layers)

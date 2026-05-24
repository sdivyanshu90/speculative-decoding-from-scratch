# =============================================================================
# FILE: tests/integration/test_edge_cases.py
# PURPOSE: Edge-case integration tests for SpeculativeDecoder.
# =============================================================================
"""Tests covering all the gnarly corners called out in §1.5 of the docs."""

from __future__ import annotations

import pytest
import torch

from speculative_decoding import (
    ContextLengthError,
    DraftModel,
    InvalidPromptError,
    ModelCompatibilityError,
    SpeculativeDecoder,
    SpeculativeDecodingConfig,
    TargetModel,
)
from tests.conftest import (
    DEFAULT_VOCAB_SIZE,
    MockHFLikeModel,
    argmax_favoring,
)


def _build_matched_decoder(
    *,
    gamma: int = 3,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    eos: int | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_position_embeddings: int | None = None,
    favored_token: int = 4,
) -> tuple[SpeculativeDecoder, DraftModel, TargetModel]:
    """Helper returning a decoder where draft == target (full acceptance)."""
    favored = argmax_favoring(vocab_size, favored_token=favored_token, strength=10.0)
    draft_hf = MockHFLikeModel(
        vocab_size=vocab_size,
        logits_fn=favored,
        max_position_embeddings=max_position_embeddings,
    )
    target_hf = MockHFLikeModel(
        vocab_size=vocab_size,
        logits_fn=favored,
        max_position_embeddings=max_position_embeddings,
    )
    draft = DraftModel(draft_hf, device=torch.device("cpu"), dtype=torch.float32)
    target = TargetModel(target_hf, device=torch.device("cpu"), dtype=torch.float32)
    cfg = SpeculativeDecodingConfig(
        speculation_length=gamma,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        target_model_name="mock-target",
        draft_model_name="mock-draft",
        device="cpu",
        dtype=torch.float32,
        seed=0,
        eos_token_id=eos,
    )
    return SpeculativeDecoder(draft, target, cfg), draft, target


# ----------------------------------------------------------------------------
# 1. Context length overflow
# ----------------------------------------------------------------------------


def test_prompt_longer_than_max_context() -> None:
    """Prompt + requested new tokens exceeding context window raises clearly."""
    decoder, _, _ = _build_matched_decoder(
        gamma=2,
        max_new_tokens=10,
        max_position_embeddings=8,  # very tight context window
    )
    long_prompt = torch.arange(5, dtype=torch.long)  # 5 + 10 = 15 > 8
    with pytest.raises(ContextLengthError) as exc_info:
        decoder.generate(long_prompt)
    assert "context window" in str(exc_info.value).lower()


# ----------------------------------------------------------------------------
# 2. Vocab-size mismatch between draft and target
# ----------------------------------------------------------------------------


def test_mismatched_vocab_sizes_raises() -> None:
    """Different vocab sizes between draft and target raises at construction."""
    draft_hf = MockHFLikeModel(vocab_size=16)
    target_hf = MockHFLikeModel(vocab_size=32)
    draft = DraftModel(draft_hf, device=torch.device("cpu"), dtype=torch.float32)
    target = TargetModel(target_hf, device=torch.device("cpu"), dtype=torch.float32)
    cfg = SpeculativeDecodingConfig(
        speculation_length=2,
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        target_model_name="mock-target",
        draft_model_name="mock-draft",
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ModelCompatibilityError) as exc_info:
        SpeculativeDecoder(draft, target, cfg)
    assert "vocab_size" in str(exc_info.value)


# ----------------------------------------------------------------------------
# 3. γ = 1 — degenerate but must still work
# ----------------------------------------------------------------------------


def test_speculation_length_one() -> None:
    """γ = 1 degenerates to (essentially) standard decoding; still correct."""
    decoder, _, _ = _build_matched_decoder(gamma=1, max_new_tokens=6, temperature=0.0)
    prompt = torch.tensor([0, 1], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    assert metrics.tokens_generated == 6
    assert metrics.acceptance_rate == 1.0
    # Full-acceptance + γ=1 -> 2 tokens per iter -> n_iterations = 3.
    assert metrics.n_iterations == 3


# ----------------------------------------------------------------------------
# 4. γ ≥ max_new_tokens — generation completes in one iteration
# ----------------------------------------------------------------------------


def test_speculation_length_equals_max_tokens() -> None:
    """γ ≥ max_new_tokens: one iteration suffices."""
    decoder, _, _ = _build_matched_decoder(gamma=8, max_new_tokens=5, temperature=0.0)
    prompt = torch.tensor([0], dtype=torch.long)
    _out_ids, metrics = decoder.generate(prompt)
    assert metrics.tokens_generated == 5
    assert metrics.n_iterations == 1


# ----------------------------------------------------------------------------
# 5. Empty prompt
# ----------------------------------------------------------------------------


def test_empty_prompt_raises() -> None:
    """A zero-length prompt is rejected by InvalidPromptError."""
    decoder, _, _ = _build_matched_decoder(gamma=2, max_new_tokens=4)
    empty = torch.tensor([], dtype=torch.long)
    with pytest.raises(InvalidPromptError) as exc_info:
        decoder.generate(empty)
    assert "at least one token" in str(exc_info.value)


# ----------------------------------------------------------------------------
# 6. EOS inside a draft sequence
# ----------------------------------------------------------------------------


def test_eos_mid_draft_truncates_correctly() -> None:
    """EOS encountered at any position in the accepted prefix truncates output."""
    vocab = DEFAULT_VOCAB_SIZE
    eos = 5
    # Both models pick EOS greedily — generation halts on the first token.
    decoder, _, _ = _build_matched_decoder(
        gamma=4,
        max_new_tokens=20,
        temperature=0.0,
        eos=eos,
        favored_token=eos,
    )
    prompt = torch.tensor([0, 1], dtype=torch.long)
    out_ids, _ = decoder.generate(prompt)
    new_tokens = out_ids[0, prompt.shape[0]:].tolist()
    assert eos in new_tokens
    # Exactly one EOS, at the end.
    assert new_tokens[-1] == eos
    assert new_tokens.count(eos) == 1


# ----------------------------------------------------------------------------
# 7. Prompt as 2-D tensor — also accepted
# ----------------------------------------------------------------------------


def test_prompt_accepts_2d_tensor() -> None:
    """[1, L] prompts work identically to [L] prompts."""
    decoder, _, _ = _build_matched_decoder(gamma=2, max_new_tokens=4)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    assert out_ids.shape == (1, 7)
    assert metrics.tokens_generated == 4


# ----------------------------------------------------------------------------
# 8. Stochastic decoding still terminates correctly
# ----------------------------------------------------------------------------


def test_stochastic_decoding_completes() -> None:
    """High-temperature decoding still produces exactly max_new_tokens."""
    decoder, _, _ = _build_matched_decoder(gamma=3, max_new_tokens=10, temperature=2.0)
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    assert metrics.tokens_generated == 10
    assert out_ids.shape == (1, 13)

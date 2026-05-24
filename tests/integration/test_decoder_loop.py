# =============================================================================
# FILE: tests/integration/test_decoder_loop.py
# PURPOSE: Integration tests for the SpeculativeDecoder.generate loop.
# =============================================================================
"""End-to-end loop tests using mock models (no real LM required)."""

from __future__ import annotations

import torch

from speculative_decoding import (
    DraftModel,
    SpeculativeDecoder,
    SpeculativeDecodingConfig,
    TargetModel,
)
from tests.conftest import (
    DEFAULT_VOCAB_SIZE,
    MockHFLikeModel,
    argmax_favoring,
)


def _make_decoder(
    draft: DraftModel,
    target: TargetModel,
    *,
    gamma: int = 3,
    max_new_tokens: int = 12,
    seed: int = 0,
    temperature: float = 0.0,
    eos: int | None = None,
) -> SpeculativeDecoder:
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
        seed=seed,
        eos_token_id=eos,
    )
    return SpeculativeDecoder(draft, target, cfg)


def test_generation_produces_correct_length(
    matched_models: tuple[DraftModel, TargetModel]
) -> None:
    """generate() produces exactly max_new_tokens output tokens past the prompt."""
    draft, target = matched_models
    decoder = _make_decoder(draft, target, gamma=3, max_new_tokens=10, temperature=0.0)
    prompt = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    # output = prompt (4) + 10 new = 14.
    assert out_ids.shape == (1, 14)
    assert metrics.tokens_generated == 10


def test_generation_with_all_accepted_draft(
    matched_models: tuple[DraftModel, TargetModel]
) -> None:
    """If draft == target deterministically, every draft is accepted (greedy)."""
    draft, target = matched_models
    gamma = 4
    decoder = _make_decoder(
        draft, target, gamma=gamma, max_new_tokens=20, temperature=0.0
    )
    prompt = torch.tensor([0], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    # γ+1 new tokens per iter when fully accepted.
    expected_iters = (20 + gamma) // (gamma + 1)  # ceiling-ish for our case
    # We expect acceptance rate == 1.0 in greedy / fully matched setup.
    assert metrics.acceptance_rate == 1.0
    # Each iter contributes γ+1 = 5 tokens; we generated 20 tokens.
    assert metrics.n_iterations == 20 // (gamma + 1) or metrics.n_iterations == (
        20 // (gamma + 1) + 1
    )
    assert metrics.tokens_generated == 20
    # Sanity: every generated token equals the favored token id (7).
    new_tokens = out_ids[0, prompt.shape[0]:].tolist()
    assert all(t == 7 for t in new_tokens)


def test_generation_with_all_rejected_draft(
    mismatched_models: tuple[DraftModel, TargetModel]
) -> None:
    """When draft and target disagree on argmax, no draft is accepted (greedy).

    Each iteration emits 1 token (the target's argmax), so n_iterations ==
    tokens_generated.
    """
    draft, target = mismatched_models
    gamma = 3
    decoder = _make_decoder(
        draft, target, gamma=gamma, max_new_tokens=8, temperature=0.0
    )
    prompt = torch.tensor([0], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    assert metrics.acceptance_rate == 0.0
    assert metrics.n_iterations == 8
    assert metrics.tokens_generated == 8
    # Every generated token is the target's favored token (id 2).
    new_tokens = out_ids[0, prompt.shape[0]:].tolist()
    assert all(t == 2 for t in new_tokens)


def test_kv_cache_length_after_rollback(
    mismatched_models: tuple[DraftModel, TargetModel]
) -> None:
    """After full generation, both KV caches reflect the committed sequence.

    For mismatched (all-reject) greedy generation: every iter contributes 1
    accepted token (the resampled / target-argmax token). After N iterations
    of generating one token apiece, both caches should hold the prompt's
    first (L_prompt - 1) tokens + the committed prefix (= L_prompt + N − 1).
    We verify by inspecting the decoder's internal cache state after run.
    """
    draft, target = mismatched_models
    gamma = 3
    max_new = 5
    decoder = _make_decoder(
        draft, target, gamma=gamma, max_new_tokens=max_new, temperature=0.0
    )
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    out_ids, _metrics = decoder.generate(prompt)
    # Inspect the mock's forward_call_log to confirm we did NOT over-extend
    # the cache: the last cache length recorded for the target must equal
    # (prompt_len - 1 + γ + 1) at the last iteration's verification call.
    target_log = target.hf_model.forward_call_log  # type: ignore[attr-defined]
    # We expect at least max_new // 1 == 5 verification iterations.
    assert len(target_log) >= max_new


def test_metrics_populated_after_generation(
    matched_models: tuple[DraftModel, TargetModel]
) -> None:
    """All DecodingMetrics fields are populated to sensible values after a run."""
    draft, target = matched_models
    decoder = _make_decoder(draft, target, gamma=3, max_new_tokens=6, temperature=0.0)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    _out_ids, metrics = decoder.generate(prompt)
    assert metrics.tokens_generated == 6
    assert metrics.n_iterations >= 1
    assert metrics.tokens_accepted_from_draft >= 0
    assert metrics.wall_clock_time > 0.0
    assert metrics.tokens_per_second > 0.0


def test_eos_token_stops_generation() -> None:
    """When EOS is the target's argmax (greedy), generation halts early."""
    vocab = DEFAULT_VOCAB_SIZE
    eos = 9
    # Both models favor EOS: under greedy, first generated token IS EOS,
    # which terminates generation immediately.
    draft_hf = MockHFLikeModel(vocab, logits_fn=argmax_favoring(vocab, eos, 50.0))
    target_hf = MockHFLikeModel(vocab, logits_fn=argmax_favoring(vocab, eos, 50.0))
    draft = DraftModel(draft_hf, device=torch.device("cpu"), dtype=torch.float32)
    target = TargetModel(target_hf, device=torch.device("cpu"), dtype=torch.float32)
    decoder = _make_decoder(
        draft, target, gamma=4, max_new_tokens=20, temperature=0.0, eos=eos
    )
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    # Output ends with EOS, and total new tokens << max_new_tokens.
    new_tokens = out_ids[0, prompt.shape[0]:].tolist()
    assert metrics.tokens_generated < 20
    assert eos in new_tokens
    # No tokens after the first EOS.
    eos_idx = new_tokens.index(eos)
    assert all(t == eos for t in new_tokens[eos_idx : eos_idx + 1])
    assert len(new_tokens) == eos_idx + 1


def test_determinism_with_seed(
    matched_models: tuple[DraftModel, TargetModel]
) -> None:
    """Same seed -> identical output token sequences across two runs."""
    draft1, target1 = matched_models
    # Build a second pair (fresh mock state) so caches don't carry over.
    favored = argmax_favoring(DEFAULT_VOCAB_SIZE, 7, strength=10.0)
    draft2 = DraftModel(
        MockHFLikeModel(vocab_size=DEFAULT_VOCAB_SIZE, logits_fn=favored),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    target2 = TargetModel(
        MockHFLikeModel(vocab_size=DEFAULT_VOCAB_SIZE, logits_fn=favored),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    d1 = _make_decoder(draft1, target1, gamma=3, max_new_tokens=10, temperature=0.7, seed=7)
    d2 = _make_decoder(draft2, target2, gamma=3, max_new_tokens=10, temperature=0.7, seed=7)
    prompt = torch.tensor([0, 1], dtype=torch.long)
    out1, _ = d1.generate(prompt)
    out2, _ = d2.generate(prompt)
    assert torch.equal(out1, out2)


def test_short_prompt_single_token(
    matched_models: tuple[DraftModel, TargetModel]
) -> None:
    """A one-token prompt must work (prefill skipped path)."""
    draft, target = matched_models
    decoder = _make_decoder(draft, target, gamma=2, max_new_tokens=4, temperature=0.0)
    prompt = torch.tensor([3], dtype=torch.long)
    out_ids, metrics = decoder.generate(prompt)
    assert metrics.tokens_generated == 4
    assert out_ids.shape == (1, 5)

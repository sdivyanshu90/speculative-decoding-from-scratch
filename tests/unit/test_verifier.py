# =============================================================================
# FILE: tests/unit/test_verifier.py
# PURPOSE: Unit tests for the TokenVerifier (modified rejection sampling).
# =============================================================================
"""Unit tests for :class:`TokenVerifier`."""

from __future__ import annotations

import math

import pytest
import torch

from speculative_decoding import Sampler, TokenVerifier, VerifierShapeError


def _log_softmax_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Helper: convert probability vector(s) to log-probabilities."""
    return torch.log(probs.clamp_min(1e-20))


def _make_sampler(seed: int = 0, temperature: float = 1.0) -> Sampler:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return Sampler(temperature=temperature, top_p=1.0, top_k=0, generator=gen)


def test_accept_all_tokens() -> None:
    """When q >= p for every drafted token, all drafts must be accepted.

    With q >> p, log(q/p) > 0 so the acceptance probability is min(1, q/p) = 1
    and the verifier should never reject. The bonus token must then be drawn
    from ``target_log_probs[γ]``.
    """
    gamma = 3
    vocab = 4
    drafts = torch.tensor([0, 1, 2])  # shape: [γ]
    # Draft says each drafted token has prob 0.1 — and target says 0.9.
    draft_lp = _log_softmax_from_probs(
        torch.tensor(
            [
                [0.1, 0.3, 0.3, 0.3],
                [0.3, 0.1, 0.3, 0.3],
                [0.3, 0.3, 0.1, 0.3],
            ]
        )
    )
    target_lp = _log_softmax_from_probs(
        torch.tensor(
            [
                [0.9, 0.04, 0.03, 0.03],
                [0.04, 0.9, 0.03, 0.03],
                [0.04, 0.03, 0.9, 0.03],
                # Bonus: concentrate at token 3
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
    )
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=0)
    result = verifier.verify(drafts, draft_lp, target_lp, sampler)
    assert result.n_drafts_accepted == gamma
    assert result.bonus_used is True
    # First γ tokens are the drafts, last is bonus drawn from row γ.
    assert result.accepted_tokens[:gamma] == [0, 1, 2]
    assert len(result.accepted_tokens) == gamma + 1


def test_reject_first_token() -> None:
    """When q << p for the first draft, that draft is rejected and the rest discarded.

    With q[0, draft_token] ≈ 0 and p[0, draft_token] ≈ 1, acceptance prob ≈ 0
    so any uniform draw should reject.
    """
    gamma = 3
    drafts = torch.tensor([0, 1, 2])  # shape: [γ]
    # Draft: heavily commits to token 0 at position 0.
    draft_probs = torch.tensor(
        [
            [0.97, 0.01, 0.01, 0.01],  # pos 0
            [0.25, 0.25, 0.25, 0.25],  # pos 1
            [0.25, 0.25, 0.25, 0.25],  # pos 2
        ]
    )
    draft_lp = _log_softmax_from_probs(draft_probs)
    # Target: almost-zero weight on token 0 at position 0 -> rejection.
    target_probs = torch.tensor(
        [
            [0.001, 0.333, 0.333, 0.333],  # pos 0
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],  # bonus position
        ]
    )
    target_lp = _log_softmax_from_probs(target_probs)
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=0)
    result = verifier.verify(drafts, draft_lp, target_lp, sampler)
    assert result.n_drafts_accepted == 0
    assert result.bonus_used is False
    # Output is exactly one resampled token at the first-rejection position.
    assert len(result.accepted_tokens) == 1
    assert result.accepted_tokens[0] != 0  # cannot be the rejected draft token


def test_adjusted_distribution_sums_to_one() -> None:
    """The residual ``(q − p)_+`` normalised is a valid probability distribution."""
    p = torch.tensor([0.6, 0.3, 0.1])  # shape: [V]
    q = torch.tensor([0.2, 0.5, 0.3])  # shape: [V]
    residual = torch.clamp(q - p, min=0.0)  # shape: [V]
    total = residual.sum()
    normalised = residual / total
    assert torch.isclose(normalised.sum(), torch.tensor(1.0), atol=1e-6)
    # Sanity: the rejected token (max of p) must NOT have positive residual mass
    # if q[that index] < p[that index] (which is the rejection case).
    assert normalised[0].item() == pytest.approx(0.0, abs=1e-6)


def test_acceptance_probability_clamped_to_one() -> None:
    """When q > p, acceptance probability is clamped to 1 (never above)."""
    # Manually verify with a tiny draft (γ=1): if q is much larger than p,
    # log(q/p) > 0 so we clamp to 0 (in log space) which is exp(0) == 1.
    vocab = 4
    draft_token_id = 2
    # Construct q with very high mass on token 2 (almost 1.0), p with low.
    q = torch.tensor([0.01, 0.01, 0.97, 0.01])
    p = torch.tensor([0.97, 0.01, 0.01, 0.01])  # draft thinks token 0 most likely
    # But the drafted token is 2 — for which q(2) = 0.97 >> p(2) = 0.01.
    drafts = torch.tensor([draft_token_id])
    draft_lp = _log_softmax_from_probs(p).unsqueeze(0)  # shape: [1, V]
    target_lp = _log_softmax_from_probs(q).unsqueeze(0).repeat(2, 1)  # γ+1 = 2
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=42)
    # Repeat many times: should ALWAYS accept (acceptance prob is 1).
    for trial in range(100):
        sampler.generator = torch.Generator()
        sampler.generator.manual_seed(trial)
        result = verifier.verify(drafts, draft_lp, target_lp, sampler)
        assert result.n_drafts_accepted == 1, f"trial {trial} unexpectedly rejected"


def test_bonus_token_sampled_on_full_acceptance() -> None:
    """When all γ drafts accepted, the final emitted token comes from target_log_probs[γ]."""
    gamma = 2
    drafts = torch.tensor([1, 2])  # shape: [γ]
    # Draft and target agree perfectly on these tokens; both have high mass.
    draft_lp = _log_softmax_from_probs(
        torch.tensor(
            [
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
            ]
        )
    )
    target_lp = _log_softmax_from_probs(
        torch.tensor(
            [
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                # Bonus is a delta on token 3.
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=0)
    result = verifier.verify(drafts, draft_lp, target_lp, sampler)
    assert result.n_drafts_accepted == gamma
    assert result.bonus_used is True
    # Bonus is sampled from the delta at index 3.
    assert result.accepted_tokens[-1] == 3


def test_greedy_acceptance_deterministic() -> None:
    """At temperature=0, acceptance is deterministic: argmax match ⟺ accept."""
    gamma = 3
    drafts = torch.tensor([0, 1, 5])  # last token mismatches target's argmax
    vocab = 8
    # Build target logits whose argmax is [0, 1, 2, 7] (bonus = 7).
    target_logits = torch.full((gamma + 1, vocab), -10.0)
    for pos, tok in enumerate([0, 1, 2, 7]):
        target_logits[pos, tok] = 10.0
    # Draft logits: any (greedy path ignores draft log probs entirely).
    draft_logits = torch.full((gamma, vocab), -10.0)
    for pos, tok in enumerate(drafts.tolist()):
        draft_logits[pos, tok] = 10.0

    greedy_sampler = Sampler(temperature=0.0, top_p=1.0, top_k=0)
    draft_lp = greedy_sampler.log_probs(draft_logits)
    target_lp = greedy_sampler.log_probs(target_logits)
    verifier = TokenVerifier()

    # Drafts 0,1 match (target argmax also 0,1) and draft 5 mismatches (target argmax 2).
    # Expect: accept 0, accept 1, reject at position 2 -> emit 2 (target argmax).
    result = verifier.verify(drafts, draft_lp, target_lp, greedy_sampler)
    assert result.n_drafts_accepted == 2
    assert result.bonus_used is False
    assert result.accepted_tokens == [0, 1, 2]


def test_log_probability_numerical_stability() -> None:
    """Verifier must not produce NaN/Inf with extreme log-probabilities."""
    gamma = 2
    drafts = torch.tensor([0, 1])
    # Use logits that yield log-probs very close to log(0) for several tokens.
    draft_lp = torch.tensor(
        [
            [-1e-6, -50.0, -50.0, -50.0],
            [-50.0, -1e-6, -50.0, -50.0],
        ]
    )
    target_lp = torch.tensor(
        [
            [-1e-6, -50.0, -50.0, -50.0],
            [-50.0, -1e-6, -50.0, -50.0],
            [-1.0, -1.0, -1.0, -math.log(4.0) * 2.0],
        ]
    )
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=1)
    result = verifier.verify(drafts, draft_lp, target_lp, sampler)
    # No NaN/Inf in outputs (token ids should always be valid).
    for tok in result.accepted_tokens:
        assert 0 <= tok < 4


def test_single_token_speculation() -> None:
    """γ = 1: smallest non-trivial speculation length must work."""
    gamma = 1
    drafts = torch.tensor([2])  # shape: [γ]
    draft_lp = _log_softmax_from_probs(torch.tensor([[0.1, 0.1, 0.7, 0.1]]))
    target_lp = _log_softmax_from_probs(
        torch.tensor(
            [
                [0.1, 0.1, 0.7, 0.1],  # verify pos
                [0.7, 0.1, 0.1, 0.1],  # bonus pos
            ]
        )
    )
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=0)
    result = verifier.verify(drafts, draft_lp, target_lp, sampler)
    assert result.n_drafts_accepted in (0, 1)
    # Output is 1 token regardless of accept/reject outcome
    # (drafted-and-accepted-plus-bonus = 2 tokens; rejected-and-resampled = 1 token).
    if result.bonus_used:
        assert len(result.accepted_tokens) == 2
    else:
        assert len(result.accepted_tokens) == 1


def test_shape_validation_raises_on_bad_input() -> None:
    """Mismatched γ between draft tokens and log-probs raises VerifierShapeError."""
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=0)
    drafts = torch.tensor([0, 1])  # γ = 2
    # draft_lp says γ = 3 (mismatch)
    draft_lp = torch.zeros(3, 4)
    target_lp = torch.zeros(3, 4)
    with pytest.raises(VerifierShapeError):
        verifier.verify(drafts, draft_lp, target_lp, sampler)


def test_shape_validation_vocab_mismatch() -> None:
    """Mismatched vocab dim between draft and target log-probs raises."""
    verifier = TokenVerifier()
    sampler = _make_sampler(seed=0)
    drafts = torch.tensor([0])  # γ = 1
    draft_lp = torch.zeros(1, 4)
    target_lp = torch.zeros(2, 5)  # different vocab
    with pytest.raises(VerifierShapeError):
        verifier.verify(drafts, draft_lp, target_lp, sampler)

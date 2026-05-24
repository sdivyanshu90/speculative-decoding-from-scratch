# =============================================================================
# FILE: tests/unit/test_sampling.py
# PURPOSE: Unit tests for the temperature / top-k / top-p / Sampler utilities.
# =============================================================================
"""Unit tests for sampling primitives."""

from __future__ import annotations

import math

import pytest
import torch

from speculative_decoding import Sampler, SamplingConfigError
from speculative_decoding.core.sampling import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
)


def test_temperature_zero_is_greedy() -> None:
    """Temperature=0 must yield a delta distribution at the argmax."""
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    out = apply_temperature(logits, temperature=0.0)
    # Argmax is index 1; that position should be 0 (== log 1), others -inf.
    assert out[1].item() == 0.0
    for i in (0, 2, 3):
        assert math.isinf(out[i].item()) and out[i].item() < 0
    # softmax should be a delta at index 1.
    probs = torch.softmax(out, dim=-1)
    assert torch.isclose(probs[1], torch.tensor(1.0))
    for i in (0, 2, 3):
        assert torch.isclose(probs[i], torch.tensor(0.0))


def test_top_k_filters_vocabulary() -> None:
    """After top-k, exactly k tokens have finite logits; the rest are -inf."""
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 0.0])
    out = apply_top_k(logits, top_k=2)
    finite_count = (out > float("-inf")).sum().item()
    assert finite_count == 2
    # The kept tokens must be the top two: indices 1 (value 5.0) and 2 (value 3.0).
    finite_indices = [i for i, v in enumerate(out.tolist()) if not math.isinf(v)]
    assert sorted(finite_indices) == [1, 2]


def test_top_k_zero_is_noop() -> None:
    """top_k=0 disables the filter; all tokens retain their original logits."""
    logits = torch.tensor([1.0, 5.0, 3.0])
    out = apply_top_k(logits, top_k=0)
    assert torch.equal(out, logits)


def test_top_p_filters_vocabulary() -> None:
    """Top-p keeps the smallest-prefix mass >= p; sum of kept probs >= p."""
    # Probabilities sorted descending: 0.5, 0.3, 0.1, 0.1.
    # Cumulative: 0.5, 0.8, 0.9, 1.0.
    # top_p = 0.6 -> keep until cumulative crosses 0.6: keep 0.5 + 0.3 = 0.8.
    probs = torch.tensor([0.5, 0.3, 0.1, 0.1])
    logits = torch.log(probs)  # produces probs exactly after softmax
    out = apply_top_p(logits, top_p=0.6)
    surviving_probs = torch.softmax(out, dim=-1)
    # Kept indices must include 0 and 1; mass kept >= top_p.
    assert surviving_probs[0].item() > 0
    assert surviving_probs[1].item() > 0
    assert surviving_probs.sum().item() >= 0.6 - 1e-6


def test_top_p_always_keeps_at_least_one_token() -> None:
    """Even with tiny top_p, at least one token must remain (no division by 0)."""
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    out = apply_top_p(logits, top_p=0.01)
    kept_indices = [i for i, v in enumerate(out.tolist()) if not math.isinf(v)]
    assert len(kept_indices) >= 1
    # The single kept token should be the argmax.
    assert 0 in kept_indices


def test_top_p_one_is_noop() -> None:
    """top_p=1.0 disables the filter; all tokens retain their logits."""
    logits = torch.tensor([1.0, 2.0, 0.5])
    out = apply_top_p(logits, top_p=1.0)
    assert torch.equal(out, logits)


def test_sampling_without_replacement() -> None:
    """Two samplers with the same seed must produce identical outputs."""
    vocab = 8
    logits = torch.randn(vocab)
    sampler1 = Sampler(temperature=1.0, generator=torch.Generator().manual_seed(99))
    sampler2 = Sampler(temperature=1.0, generator=torch.Generator().manual_seed(99))
    samples1 = [sampler1.sample(logits).item() for _ in range(20)]
    samples2 = [sampler2.sample(logits).item() for _ in range(20)]
    assert samples1 == samples2


def test_invalid_temperature_raises() -> None:
    """Negative temperature in constructor or apply_temperature raises."""
    with pytest.raises(SamplingConfigError):
        Sampler(temperature=-0.5)
    with pytest.raises(SamplingConfigError):
        apply_temperature(torch.tensor([1.0, 2.0]), temperature=-1.0)


def test_invalid_top_p_raises() -> None:
    """top_p must be in (0, 1]."""
    with pytest.raises(SamplingConfigError):
        Sampler(top_p=0.0)
    with pytest.raises(SamplingConfigError):
        Sampler(top_p=1.5)
    with pytest.raises(SamplingConfigError):
        apply_top_p(torch.tensor([1.0, 2.0]), top_p=-0.1)


def test_invalid_top_k_raises() -> None:
    """top_k must be >= 0."""
    with pytest.raises(SamplingConfigError):
        Sampler(top_k=-1)
    with pytest.raises(SamplingConfigError):
        apply_top_k(torch.tensor([1.0, 2.0]), top_k=-1)


def test_sampler_log_probs_sum_to_one() -> None:
    """exp of log_probs returned by Sampler.log_probs sum to 1 along last dim."""
    sampler = Sampler(temperature=1.0)
    logits = torch.randn(3, 8)
    log_probs = sampler.log_probs(logits)
    sums = torch.exp(log_probs).sum(dim=-1)
    for s in sums.tolist():
        assert math.isclose(s, 1.0, abs_tol=1e-5)


def test_sampler_greedy_returns_argmax() -> None:
    """Greedy sampler always returns argmax, never anything else."""
    sampler = Sampler(temperature=0.0)
    logits = torch.tensor([1.0, 5.0, 3.0])
    for _ in range(20):
        assert sampler.sample(logits).item() == 1


def test_sample_from_probs_respects_distribution() -> None:
    """Many samples from a fixed prob vector should follow the distribution."""
    probs = torch.tensor([0.7, 0.2, 0.1])
    gen = torch.Generator().manual_seed(0)
    sampler = Sampler(generator=gen)
    samples = [sampler.sample_from_probs(probs).item() for _ in range(5000)]
    # Empirical frequencies should be within ~2% of the true probs.
    counts = [samples.count(i) / len(samples) for i in range(3)]
    assert abs(counts[0] - 0.7) < 0.03
    assert abs(counts[1] - 0.2) < 0.03
    assert abs(counts[2] - 0.1) < 0.03


def test_sampler_is_greedy_flag() -> None:
    """is_greedy must be True iff temperature is effectively zero."""
    assert Sampler(temperature=0.0).is_greedy is True
    assert Sampler(temperature=0.001).is_greedy is False
    assert Sampler(temperature=1.0).is_greedy is False

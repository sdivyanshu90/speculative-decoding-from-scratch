# =============================================================================
# FILE: tests/unit/test_metrics.py
# PURPOSE: Unit tests for DecodingMetrics accumulation and derived quantities.
# =============================================================================
"""Unit tests for :class:`DecodingMetrics`."""

from __future__ import annotations

import time

import pytest

from speculative_decoding import DecodingMetrics


def test_fresh_metrics_are_zero() -> None:
    """All counters and derived metrics on a fresh instance are zero / safe."""
    m = DecodingMetrics(speculation_length=4)
    assert m.tokens_generated == 0
    assert m.tokens_accepted_from_draft == 0
    assert m.n_iterations == 0
    assert m.mean_tokens_per_iteration == 0.0
    assert m.acceptance_rate == 0.0
    assert m.tokens_per_second == 0.0
    # theoretical_speedup at α=0: speedup = 1 / (γc + 1) = 1 / 1 = 1.
    assert m.theoretical_speedup == pytest.approx(1.0)


def test_record_iteration_updates_counters() -> None:
    """record_iteration increments iter, draft-accepted and total counters."""
    m = DecodingMetrics(speculation_length=3)
    m.record_iteration(n_drafts_accepted=2, n_tokens_emitted=3)
    assert m.n_iterations == 1
    assert m.tokens_accepted_from_draft == 2
    assert m.tokens_generated == 3
    m.record_iteration(n_drafts_accepted=3, n_tokens_emitted=4)
    assert m.n_iterations == 2
    assert m.tokens_accepted_from_draft == 5
    assert m.tokens_generated == 7


def test_acceptance_rate_calculation() -> None:
    """α = tokens_accepted_from_draft / (n_iterations * γ)."""
    m = DecodingMetrics(speculation_length=5)
    # 4 iters, 12 drafts accepted out of 4*5 = 20 total drafted -> α = 0.6
    m.record_iteration(3, 4)
    m.record_iteration(3, 4)
    m.record_iteration(3, 4)
    m.record_iteration(3, 4)
    assert m.acceptance_rate == pytest.approx(0.6)


def test_mean_tokens_per_iteration() -> None:
    """Mean tokens per iteration = tokens_generated / n_iterations."""
    m = DecodingMetrics(speculation_length=2)
    m.record_iteration(1, 2)
    m.record_iteration(2, 3)
    assert m.mean_tokens_per_iteration == pytest.approx(2.5)


def test_theoretical_speedup_formula() -> None:
    """Theoretical speedup matches the formula at known values of (α, γ, c)."""
    # α = 0.5, γ = 4, c = 0: speedup = (1 − 0.5^5) / 0.5 = (1 − 0.03125)/0.5 = 1.9375.
    m = DecodingMetrics(speculation_length=4, cost_ratio=0.0)
    # Achieve α=0.5: 2 drafts accepted out of 4 per iter; we run 1 iter.
    m.record_iteration(n_drafts_accepted=2, n_tokens_emitted=3)
    assert m.acceptance_rate == pytest.approx(0.5)
    assert m.theoretical_speedup == pytest.approx(1.9375)


def test_theoretical_speedup_alpha_one_limit() -> None:
    """At α=1, speedup degenerates to (γ+1) / (γc + 1)."""
    m = DecodingMetrics(speculation_length=4, cost_ratio=0.25)
    # All drafts accepted: γ = 4 drafts, γ+1 = 5 tokens per iter.
    m.record_iteration(n_drafts_accepted=4, n_tokens_emitted=5)
    assert m.acceptance_rate == pytest.approx(1.0)
    # Expected: (4+1) / (4*0.25 + 1) = 5 / 2 = 2.5
    assert m.theoretical_speedup == pytest.approx(2.5)


def test_start_stop_measures_wall_clock() -> None:
    """start() / stop() populate wall_clock_time with a positive number."""
    m = DecodingMetrics(speculation_length=1)
    m.start()
    time.sleep(0.02)
    m.stop()
    assert m.wall_clock_time > 0.0


def test_tokens_per_second_zero_when_not_stopped() -> None:
    """tokens_per_second is 0 until wall_clock_time is populated."""
    m = DecodingMetrics(speculation_length=1)
    m.record_iteration(0, 1)
    assert m.tokens_per_second == 0.0


def test_to_dict_serializes_all_fields() -> None:
    """to_dict emits every public field as primitive types."""
    m = DecodingMetrics(speculation_length=2, cost_ratio=0.1)
    m.record_iteration(1, 2)
    m.start()
    m.stop()
    payload = m.to_dict()
    required = {
        "speculation_length",
        "cost_ratio",
        "tokens_generated",
        "tokens_accepted_from_draft",
        "n_iterations",
        "mean_tokens_per_iteration",
        "acceptance_rate",
        "theoretical_speedup",
        "wall_clock_time",
        "tokens_per_second",
    }
    assert required.issubset(payload.keys())
    # All values must be JSON-serialisable primitives (no torch / numpy).
    for v in payload.values():
        assert isinstance(v, (int, float))


def test_generate_report_contains_key_lines() -> None:
    """generate_report includes the most important metrics."""
    m = DecodingMetrics(speculation_length=3)
    m.record_iteration(2, 3)
    report = m.generate_report()
    assert "Speculation length" in report
    assert "Acceptance rate" in report
    assert "Tokens generated" in report
    assert "Theoretical speedup" in report


def test_stop_is_idempotent() -> None:
    """Calling stop() twice doesn't overwrite a finalised wall_clock_time."""
    m = DecodingMetrics(speculation_length=1)
    m.start()
    time.sleep(0.005)
    m.stop()
    first = m.wall_clock_time
    time.sleep(0.005)
    m.stop()
    assert m.wall_clock_time == first

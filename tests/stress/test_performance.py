# =============================================================================
# FILE: tests/stress/test_performance.py
# PURPOSE: Throughput, latency, memory and γ-tradeoff benchmarks (mock models).
# =============================================================================
"""Stress / performance tests.

These run with mocks so they're CPU-fast and CI-safe; they print a tabular
summary to stdout for visual inspection.

Skip them in fast CI with: ``pytest -m "not stress"``.
"""

from __future__ import annotations

import gc
import time
import tracemalloc

import pytest
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

# Per-iteration latency budget for mock-model decoding (very generous; mocks
# are CPU-only and don't reflect real model timings).
DEFAULT_PER_ITER_LATENCY_BUDGET_S: float = 0.5


def _make_decoder(
    gamma: int,
    max_new_tokens: int,
    *,
    matched: bool = True,
    favored: int = 3,
) -> SpeculativeDecoder:
    vocab = DEFAULT_VOCAB_SIZE
    draft_hf = MockHFLikeModel(vocab, logits_fn=argmax_favoring(vocab, favored, 8.0))
    target_hf = MockHFLikeModel(
        vocab,
        logits_fn=argmax_favoring(
            vocab, favored if matched else (favored + 1) % vocab, 8.0
        ),
    )
    draft = DraftModel(draft_hf, device=torch.device("cpu"), dtype=torch.float32)
    target = TargetModel(target_hf, device=torch.device("cpu"), dtype=torch.float32)
    cfg = SpeculativeDecodingConfig(
        speculation_length=gamma,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        target_model_name="mock-target",
        draft_model_name="mock-draft",
        device="cpu",
        dtype=torch.float32,
        seed=0,
    )
    return SpeculativeDecoder(draft, target, cfg)


# ----------------------------------------------------------------------------
# 1. Throughput across γ
# ----------------------------------------------------------------------------


@pytest.mark.stress
def test_throughput_benchmark() -> None:
    """Report tokens/sec for γ in {1, 3, 5, 7, 10} on matched mock models."""
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    rows: list[tuple[int, float, float]] = []
    print("\n--- Throughput benchmark (γ, tokens/s, α) ---")
    print(f"{'γ':>3} {'tokens/s':>12} {'α':>8}")
    for gamma in [1, 3, 5, 7, 10]:
        decoder = _make_decoder(gamma=gamma, max_new_tokens=60, matched=True)
        _, metrics = decoder.generate(prompt)
        rows.append((gamma, metrics.tokens_per_second, metrics.acceptance_rate))
        print(f"{gamma:>3} {metrics.tokens_per_second:>12.2f} {metrics.acceptance_rate:>8.3f}")
    # Sanity: tokens/sec should be a positive finite number.
    for gamma, tps, alpha in rows:
        assert tps > 0.0
        assert 0.0 <= alpha <= 1.0


# ----------------------------------------------------------------------------
# 2. Per-iteration latency budget
# ----------------------------------------------------------------------------


@pytest.mark.stress
def test_latency_per_iteration() -> None:
    """Each speculative iteration must finish under the budget (default 500 ms)."""
    decoder = _make_decoder(gamma=5, max_new_tokens=20, matched=True)
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    _, metrics = decoder.generate(prompt)
    mean_iter_s = (
        metrics.wall_clock_time / metrics.n_iterations
        if metrics.n_iterations > 0
        else 0.0
    )
    print(f"\nmean iter wall-clock: {mean_iter_s*1000:.2f} ms over {metrics.n_iterations} iters")
    assert mean_iter_s <= DEFAULT_PER_ITER_LATENCY_BUDGET_S


# ----------------------------------------------------------------------------
# 3. Memory stability over long generation
# ----------------------------------------------------------------------------


@pytest.mark.stress
def test_memory_stability_over_long_generation() -> None:
    """Generating 1000 tokens does not blow up memory unboundedly.

    With mock models, every generated position appends a fixed-size slice to
    every layer's K and V tensors. Total cache memory should be approximately
    linear in sequence length, with no quadratic / leak-like growth.
    """
    decoder = _make_decoder(gamma=4, max_new_tokens=1000, matched=True)
    prompt = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    gc.collect()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    t0 = time.perf_counter()
    out_ids, metrics = decoder.generate(prompt)
    t1 = time.perf_counter()

    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff = snap2.compare_to(snap1, "filename")
    total_bytes = sum(stat.size_diff for stat in diff)
    print(
        f"\nLong-generation: {metrics.tokens_generated} tok in {t1-t0:.2f}s, "
        f"memory delta ~{total_bytes/1e6:.2f} MB"
    )
    assert metrics.tokens_generated == 1000
    # Loose upper bound — cache grows linearly with sequence; for our small
    # mock (2 layers, 2 heads, head_dim 4, fp32) total cache is ~16 KB per
    # position. 1004 positions -> ~16 MB worst case. Allow 64 MB headroom.
    assert total_bytes < 64 * 1024 * 1024


# ----------------------------------------------------------------------------
# 4. γ vs. acceptance vs. throughput tradeoff
# ----------------------------------------------------------------------------


@pytest.mark.stress
def test_acceptance_rate_vs_gamma_tradeoff() -> None:
    """Tabulate (γ, α, tokens/s) and verify monotonic consistency.

    With *matched* models (deterministic accept), α == 1.0 for every γ; so
    we use this test to verify that throughput trends positively with γ
    (more parallel work per iter).
    """
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    print("\n--- γ tradeoff (matched models, α = 1) ---")
    print(f"{'γ':>3} {'α':>8} {'tokens/s':>12} {'iters':>6}")
    rows = []
    for gamma in [1, 2, 4, 6, 8, 10]:
        decoder = _make_decoder(gamma=gamma, max_new_tokens=60, matched=True)
        _, metrics = decoder.generate(prompt)
        rows.append((gamma, metrics.acceptance_rate, metrics.tokens_per_second, metrics.n_iterations))
        print(f"{gamma:>3} {metrics.acceptance_rate:>8.3f} {metrics.tokens_per_second:>12.2f} {metrics.n_iterations:>6}")
    # Number of iterations must strictly DECREASE with γ for fixed total tokens.
    iters = [r[3] for r in rows]
    for prev, curr in zip(iters, iters[1:]):
        assert curr <= prev, f"iters should decrease monotonically; got {iters}"
    # All α should equal 1.0 under matched models.
    assert all(abs(r[1] - 1.0) < 1e-9 for r in rows)

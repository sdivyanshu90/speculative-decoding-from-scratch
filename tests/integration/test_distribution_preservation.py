# =============================================================================
# FILE: tests/integration/test_distribution_preservation.py
# PURPOSE: Statistical correctness — speculative decoding preserves target dist.
# =============================================================================
"""Empirical correctness test for the modified rejection sampling math.

THEORY (see ``docs/TECHNICAL_DOCUMENTATION.md`` §1.4):
    For any token position, the marginal distribution of the speculative-
    decoding emitted token equals the target model's distribution at that
    position — irrespective of how dissimilar the draft and target are.

METHOD:
    * Build mocks where the target's distribution is a known ``q`` (vocab=4)
      and the draft's distribution is a deliberately different ``p``.
    * Configure ``γ = 1``, ``max_new_tokens = 1``: each run emits ONE token,
      which (by the proof) must be drawn from ``q``.
    * Run N independent trials with different seeds, collect emitted tokens.
    * Chi-squared goodness-of-fit test against the expected counts ``N·q``.
    * Assert ``p_value > 0.05``: we cannot reject the null that observed
      counts came from ``q``.

If this test FAILS the implementation is mathematically incorrect — the bug
is most likely in :class:`TokenVerifier`'s acceptance ratio or in the
residual-distribution sampling.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from speculative_decoding import (
    DraftModel,
    SpeculativeDecoder,
    SpeculativeDecodingConfig,
    TargetModel,
)
from tests.conftest import MockHFLikeModel, fixed_logits

# Test parameters.
N_TRIALS: int = 8000
VOCAB: int = 4
TARGET_PROBS: List[float] = [0.10, 0.40, 0.30, 0.20]
DRAFT_PROBS: List[float] = [0.40, 0.30, 0.20, 0.10]
SIGNIFICANCE_LEVEL: float = 0.05

# Try to import scipy; skip the test gracefully if not installed (e.g. CI
# without the dev extras).
scipy = pytest.importorskip("scipy")
from scipy.stats import chisquare  # noqa: E402


def _build_decoder(seed: int) -> SpeculativeDecoder:
    """Build a fresh decoder so each trial uses an independent RNG state."""
    target_logits = torch.log(torch.tensor(TARGET_PROBS))  # shape: [V]
    draft_logits = torch.log(torch.tensor(DRAFT_PROBS))  # shape: [V]
    target_hf = MockHFLikeModel(vocab_size=VOCAB, logits_fn=fixed_logits(target_logits))
    draft_hf = MockHFLikeModel(vocab_size=VOCAB, logits_fn=fixed_logits(draft_logits))
    draft = DraftModel(draft_hf, device=torch.device("cpu"), dtype=torch.float32)
    target = TargetModel(target_hf, device=torch.device("cpu"), dtype=torch.float32)
    cfg = SpeculativeDecodingConfig(
        speculation_length=1,
        max_new_tokens=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        target_model_name="mock-target",
        draft_model_name="mock-draft",
        device="cpu",
        dtype=torch.float32,
        seed=seed,
        eos_token_id=None,
    )
    return SpeculativeDecoder(draft, target, cfg)


@pytest.mark.integration
def test_output_distribution_matches_target_distribution() -> None:
    """CRITICAL CORRECTNESS TEST — see module docstring."""
    counts = [0] * VOCAB
    prompt = torch.tensor([0], dtype=torch.long)
    for trial in range(N_TRIALS):
        decoder = _build_decoder(seed=trial + 1)
        out_ids, _metrics = decoder.generate(prompt)
        # output = prompt + 1 new token; we collect the new token only.
        new_token = int(out_ids[0, -1].item())
        assert 0 <= new_token < VOCAB
        counts[new_token] += 1

    observed = counts
    expected = [N_TRIALS * p for p in TARGET_PROBS]
    chi2, p_value = chisquare(f_obs=observed, f_exp=expected)
    # Helpful diagnostics on failure.
    print(f"\nDistribution preservation: observed={observed}, expected={expected}")
    print(f"chi2={chi2:.3f}, p_value={p_value:.4f}")
    assert p_value > SIGNIFICANCE_LEVEL, (
        f"Output distribution differs from target distribution (chi2={chi2:.3f}, "
        f"p={p_value:.4f}). observed={observed}, expected={expected}. "
        f"Likely bug in TokenVerifier.verify or residual sampling."
    )


@pytest.mark.integration
def test_output_distribution_with_skewed_draft() -> None:
    """Repeat the chi-squared test with the draft strongly biased away from q.

    This exercises the rejection branch heavily: the draft's argmax (token 0)
    contradicts the target's mode (token 1), so most iterations require
    resampling from the residual. The output must still be q-distributed.
    """
    global DRAFT_PROBS  # mutate for this test only
    saved = list(DRAFT_PROBS)
    try:
        # Set the draft to a near-delta on token 0; target favors token 1.
        DRAFT_PROBS[:] = [0.85, 0.05, 0.05, 0.05]
        counts = [0] * VOCAB
        prompt = torch.tensor([0], dtype=torch.long)
        for trial in range(N_TRIALS):
            decoder = _build_decoder(seed=trial + 10001)
            out_ids, _ = decoder.generate(prompt)
            new_token = int(out_ids[0, -1].item())
            counts[new_token] += 1
        expected = [N_TRIALS * p for p in TARGET_PROBS]
        chi2, p_value = chisquare(f_obs=counts, f_exp=expected)
        print(
            f"\nSkewed-draft preservation: observed={counts}, expected={expected}, "
            f"chi2={chi2:.3f}, p={p_value:.4f}"
        )
        assert p_value > SIGNIFICANCE_LEVEL
    finally:
        DRAFT_PROBS[:] = saved

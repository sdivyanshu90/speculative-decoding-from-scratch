# =============================================================================
# FILE: speculative_decoding/metrics.py
# PURPOSE: DecodingMetrics — accumulate and report speculative-decoding stats.
# =============================================================================
"""Metrics for one speculative-decoding generation run.

A :class:`DecodingMetrics` instance is created per call to
:meth:`SpeculativeDecoder.generate`, mutated in-place during the loop, then
returned to the caller. The class is **not** thread-safe; one per call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecodingMetrics:
    """Per-generation speculative-decoding statistics.

    Mutable. The decoder calls :meth:`record_iteration` after every
    speculative iteration and :meth:`start` / :meth:`stop` to bracket the
    wall-clock measurement.

    Attributes:
        speculation_length: The configured ``γ`` (used by speedup formula).
        cost_ratio: ``t_draft / t_target`` used in the theoretical speedup
            formula. ``0.0`` represents the "draft is free" upper bound.
        tokens_generated: Total tokens emitted (excluding the prompt).
        tokens_accepted_from_draft: Count of tokens accepted from the draft.
        n_iterations: Number of speculative iterations executed.
        wall_clock_time: Seconds elapsed between :meth:`start` and
            :meth:`stop`. ``0.0`` if not yet stopped.
    """

    speculation_length: int = 1
    cost_ratio: float = 0.0
    tokens_generated: int = 0
    tokens_accepted_from_draft: int = 0
    n_iterations: int = 0
    wall_clock_time: float = 0.0

    # Internal timer state.
    _t_start: float = field(default=0.0, repr=False)
    _running: bool = field(default=False, repr=False)

    # ---- Lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Begin the wall-clock measurement."""
        self._t_start = time.perf_counter()
        self._running = True

    def stop(self) -> None:
        """End the wall-clock measurement and finalize ``wall_clock_time``."""
        if self._running:
            self.wall_clock_time = time.perf_counter() - self._t_start
            self._running = False

    # ---- Recording ------------------------------------------------------

    def record_iteration(self, n_drafts_accepted: int, n_tokens_emitted: int) -> None:
        """Update counters for one speculative iteration.

        Args:
            n_drafts_accepted: Number of γ draft tokens accepted this iter
                (0..γ).
            n_tokens_emitted: Total tokens committed this iter, equal to
                ``n_drafts_accepted + 1`` (the +1 is the resample or bonus).
        """
        self.n_iterations += 1
        self.tokens_accepted_from_draft += n_drafts_accepted
        self.tokens_generated += n_tokens_emitted

    # ---- Derived metrics ------------------------------------------------

    @property
    def mean_tokens_per_iteration(self) -> float:
        """Average tokens emitted per iteration (== 1 + α·γ in expectation)."""
        if self.n_iterations == 0:
            return 0.0
        return self.tokens_generated / self.n_iterations

    @property
    def acceptance_rate(self) -> float:
        """Empirical per-draft-token acceptance rate ``α``.

        ``tokens_accepted_from_draft / (n_iterations * γ)``.
        """
        if self.n_iterations == 0 or self.speculation_length == 0:
            return 0.0
        return self.tokens_accepted_from_draft / (
            self.n_iterations * self.speculation_length
        )

    @property
    def theoretical_speedup(self) -> float:
        """Theoretical speedup from §1.5 of the technical docs.

        ``speedup = (1 - α^{γ+1}) / ((1 - α) * (γc + 1))``.

        Special-cased for ``α == 1`` (limit yields ``(γ+1) / (γc + 1)``).
        """
        alpha = self.acceptance_rate
        gamma = self.speculation_length
        c = self.cost_ratio
        if alpha >= 1.0 - 1e-12:
            return (gamma + 1) / (gamma * c + 1.0)
        if alpha <= 0.0:
            return 1.0 / (gamma * c + 1.0)
        numerator = 1.0 - alpha ** (gamma + 1)
        denominator = (1.0 - alpha) * (gamma * c + 1.0)
        return numerator / denominator

    @property
    def tokens_per_second(self) -> float:
        """Wall-clock tokens/sec across the whole generation."""
        if self.wall_clock_time <= 0.0:
            return 0.0
        return self.tokens_generated / self.wall_clock_time

    # ---- Reporting ------------------------------------------------------

    def generate_report(self) -> str:
        """Human-readable multi-line summary.

        Returns:
            A formatted string ready to print or log.
        """
        lines = [
            "=" * 60,
            "Speculative Decoding Metrics",
            "=" * 60,
            f"  Speculation length (γ)        : {self.speculation_length}",
            f"  Draft/target cost ratio (c)    : {self.cost_ratio:.4f}",
            f"  Iterations                     : {self.n_iterations}",
            f"  Tokens generated               : {self.tokens_generated}",
            f"  Tokens accepted from draft     : {self.tokens_accepted_from_draft}",
            f"  Mean tokens / iteration        : {self.mean_tokens_per_iteration:.3f}",
            f"  Acceptance rate (α)            : {self.acceptance_rate:.4f}",
            f"  Theoretical speedup            : {self.theoretical_speedup:.3f}x",
            f"  Wall-clock time (s)            : {self.wall_clock_time:.4f}",
            f"  Tokens / second                : {self.tokens_per_second:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "speculation_length": self.speculation_length,
            "cost_ratio": self.cost_ratio,
            "tokens_generated": self.tokens_generated,
            "tokens_accepted_from_draft": self.tokens_accepted_from_draft,
            "n_iterations": self.n_iterations,
            "mean_tokens_per_iteration": self.mean_tokens_per_iteration,
            "acceptance_rate": self.acceptance_rate,
            "theoretical_speedup": self.theoretical_speedup,
            "wall_clock_time": self.wall_clock_time,
            "tokens_per_second": self.tokens_per_second,
        }

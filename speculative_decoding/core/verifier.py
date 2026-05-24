# =============================================================================
# FILE: speculative_decoding/core/verifier.py
# PURPOSE: Modified Rejection Sampling verifier (Leviathan et al., 2023).
# =============================================================================
"""The token-level verifier — implements the modified rejection-sampling rule
that guarantees the output sequence is identically distributed to a direct
sample from the target model.

The math: at each draft position ``t`` with proposed token ``d_t``,
    accept with probability  min(1, q(d_t) / p(d_t))
where ``p`` is the draft model's distribution and ``q`` is the target's. On
rejection, the resampled token comes from the residual ``(q - p)_+`` distribution.
See ``docs/TECHNICAL_DOCUMENTATION.md`` §1.4 for the full preservation proof.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from ..exceptions import VerifierShapeError
from .sampling import Sampler

logger = logging.getLogger(__name__)

# When computing the residual distribution ``max(0, q - p)``, this is the
# floor below which we declare the residual numerically zero. Reached only
# when ``p`` and ``q`` are essentially identical at the rejected position,
# which should not happen because we got there via a rejection.
RESIDUAL_NORMALISATION_FLOOR: float = 1e-12


@dataclass
class VerificationResult:
    """The outcome of verifying one batch of ``γ`` draft tokens.

    Attributes:
        accepted_tokens: Final tokens to commit to the output, in order.
            Length is ``n_drafts_accepted + 1`` (always at least one — either
            a resampled token at the first rejection or a bonus token if all
            drafts were accepted).
        n_drafts_accepted: Number of draft tokens (from the input ``γ``) that
            were accepted, in ``{0, 1, …, γ}``.
        bonus_used: True iff all ``γ`` drafts were accepted and the final
            committed token is a bonus sample from the target's distribution
            at position ``γ`` (i.e. ``target_log_probs[γ]``).
    """

    accepted_tokens: list[int]
    n_drafts_accepted: int
    bonus_used: bool


class TokenVerifier:
    """Performs Modified Rejection Sampling on a batch of draft tokens.

    The verifier is **stateless**: every call to :meth:`verify` is a pure
    function of its inputs (plus the sampler's RNG state).
    """

    def __init__(self, generator: Optional[torch.Generator] = None) -> None:
        """Construct the verifier.

        Args:
            generator: Optional :class:`torch.Generator` for deterministic
                acceptance / resampling decisions. ``None`` uses the global
                RNG.
        """
        self.generator = generator

    # ---- Core algorithm --------------------------------------------------

    def verify(
        self,
        draft_tokens: torch.Tensor,
        draft_log_probs: torch.Tensor,
        target_log_probs: torch.Tensor,
        sampler: Sampler,
    ) -> VerificationResult:
        """Run modified rejection sampling.

        Args:
            draft_tokens: LongTensor of shape ``[γ]`` — the draft model's
                proposed tokens.
            draft_log_probs: FloatTensor of shape ``[γ, vocab]`` — log
                probabilities **under the sampler-transformed** draft
                distribution at each draft position.
            target_log_probs: FloatTensor of shape ``[γ + 1, vocab]`` — log
                probabilities under the sampler-transformed target
                distribution at each of the γ verification positions plus
                one bonus position.
            sampler: Sampler used for residual / bonus token sampling. Must be
                the same instance (i.e. same params) used to produce the
                draft and target log probabilities, otherwise the
                distribution-preservation guarantee no longer holds.

        Returns:
            A :class:`VerificationResult`.

        Raises:
            VerifierShapeError: If any input tensor has the wrong shape or
                inconsistent vocab dimension.
        """
        self._validate_shapes(draft_tokens, draft_log_probs, target_log_probs)
        gamma = int(draft_tokens.shape[0])
        device = draft_tokens.device

        # ---- Greedy fast path ---------------------------------------------
        if sampler.is_greedy:
            return self._verify_greedy(draft_tokens, target_log_probs)

        accepted: list[int] = []

        for t in range(gamma):
            token_id = int(draft_tokens[t].item())
            # Compute log acceptance ratio: log(q/p) = log_q - log_p. Cap at 0
            # since the acceptance probability is min(1, q/p), i.e.
            # exp(min(0, log q - log p)).
            log_q = target_log_probs[t, token_id]
            log_p = draft_log_probs[t, token_id]
            log_accept = torch.minimum(log_q - log_p, torch.zeros((), device=device))

            # Sample u ~ U(0, 1); accept iff log(u) < log_accept.
            u = torch.rand((), device=device, generator=self.generator)
            # Guard against log(0); ``torch.rand`` excludes 1.0 but may yield 0.
            # We use ``log1p(-1+u)`` ≡ ``log(u)`` after clamping.
            log_u = torch.log(u.clamp_min(torch.finfo(u.dtype).tiny))

            if bool((log_u < log_accept).item()):
                # Accept: commit the draft token and move to the next position.
                accepted.append(token_id)
                continue

            # ---- Rejection path: sample from the residual --------------
            resampled = self._sample_residual(
                draft_log_probs[t], target_log_probs[t], sampler
            )
            accepted.append(int(resampled))
            logger.debug(
                "verify: rejected draft token %d at position %d/%d; resampled %d",
                token_id,
                t,
                gamma,
                resampled,
            )
            return VerificationResult(
                accepted_tokens=accepted,
                n_drafts_accepted=t,
                bonus_used=False,
            )

        # All γ drafts accepted: sample the bonus token from target_log_probs[γ].
        bonus_probs = torch.exp(target_log_probs[gamma])  # shape: [vocab]
        bonus_probs = self._safe_normalise(bonus_probs)
        bonus = int(sampler.sample_from_probs(bonus_probs).item())
        accepted.append(bonus)
        logger.debug(
            "verify: all %d drafts accepted; bonus token %d", gamma, bonus
        )
        return VerificationResult(
            accepted_tokens=accepted,
            n_drafts_accepted=gamma,
            bonus_used=True,
        )

    # ---- Helpers ---------------------------------------------------------

    def _verify_greedy(
        self,
        draft_tokens: torch.Tensor,
        target_log_probs: torch.Tensor,
    ) -> VerificationResult:
        """Greedy-mode verification: accept iff argmax(target) == draft_token.

        On the first mismatch we emit the target's argmax token (which acts
        both as the "rejected → resampled" token AND as the natural greedy
        choice). If all γ drafts match, the bonus token is the argmax at
        position γ.
        """
        gamma = int(draft_tokens.shape[0])
        target_argmax = target_log_probs.argmax(dim=-1)  # shape: [γ + 1]
        accepted: list[int] = []
        for t in range(gamma):
            tgt = int(target_argmax[t].item())
            drf = int(draft_tokens[t].item())
            if drf == tgt:
                accepted.append(drf)
                continue
            # Mismatch: commit the target's argmax in place of the draft.
            accepted.append(tgt)
            return VerificationResult(
                accepted_tokens=accepted,
                n_drafts_accepted=t,
                bonus_used=False,
            )
        # All matched: emit bonus from target's argmax at position γ.
        accepted.append(int(target_argmax[gamma].item()))
        return VerificationResult(
            accepted_tokens=accepted,
            n_drafts_accepted=gamma,
            bonus_used=True,
        )

    def _sample_residual(
        self,
        draft_log_probs_t: torch.Tensor,
        target_log_probs_t: torch.Tensor,
        sampler: Sampler,
    ) -> int:
        """Sample one token from the adjusted distribution ``(q − p)_+``.

        We compute in probability space (after softmax) because the residual
        operation ``max(0, q − p)`` is not affine in log-space.

        Args:
            draft_log_probs_t: Vector of shape ``[vocab]``.
            target_log_probs_t: Vector of shape ``[vocab]``.
            sampler: Used only as a source of randomness here (the sampler's
                transformation has already been applied upstream).

        Returns:
            A token id.
        """
        p = torch.exp(draft_log_probs_t)  # shape: [vocab]
        q = torch.exp(target_log_probs_t)  # shape: [vocab]
        residual = torch.clamp(q - p, min=0.0)  # shape: [vocab]
        residual = self._safe_normalise(residual)
        return int(sampler.sample_from_probs(residual).item())

    @staticmethod
    def _safe_normalise(weights: torch.Tensor) -> torch.Tensor:
        """Normalize a non-negative vector to sum to 1; fall back to uniform.

        Fallback is reached only when the input is identically zero (which
        cannot legitimately occur in the residual-sampling branch but we
        guard against it for robustness).

        Args:
            weights: Non-negative tensor of shape ``[vocab]``.

        Returns:
            Probability vector of shape ``[vocab]``.
        """
        total = weights.sum()
        if total < RESIDUAL_NORMALISATION_FLOOR:
            logger.warning(
                "residual distribution had zero mass; falling back to uniform"
            )
            return torch.full_like(weights, 1.0 / weights.numel())
        return weights / total

    @staticmethod
    def _validate_shapes(
        draft_tokens: torch.Tensor,
        draft_log_probs: torch.Tensor,
        target_log_probs: torch.Tensor,
    ) -> None:
        """Confirm tensor ranks and that ``γ`` and vocab dims match.

        Raises:
            VerifierShapeError: On any shape mismatch.
        """
        if draft_tokens.dim() != 1:
            raise VerifierShapeError(
                f"draft_tokens must be 1-D [γ]; got shape {tuple(draft_tokens.shape)}"
            )
        if draft_log_probs.dim() != 2:
            raise VerifierShapeError(
                f"draft_log_probs must be 2-D [γ, vocab]; got shape "
                f"{tuple(draft_log_probs.shape)}"
            )
        if target_log_probs.dim() != 2:
            raise VerifierShapeError(
                f"target_log_probs must be 2-D [γ+1, vocab]; got shape "
                f"{tuple(target_log_probs.shape)}"
            )
        gamma = draft_tokens.shape[0]
        if draft_log_probs.shape[0] != gamma:
            raise VerifierShapeError(
                f"draft_log_probs first dim must equal γ ({gamma}); got "
                f"{draft_log_probs.shape[0]}"
            )
        if target_log_probs.shape[0] != gamma + 1:
            raise VerifierShapeError(
                f"target_log_probs first dim must equal γ+1 ({gamma + 1}); got "
                f"{target_log_probs.shape[0]}"
            )
        if draft_log_probs.shape[1] != target_log_probs.shape[1]:
            raise VerifierShapeError(
                f"vocab size mismatch: draft has {draft_log_probs.shape[1]}, "
                f"target has {target_log_probs.shape[1]}"
            )

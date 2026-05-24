# =============================================================================
# FILE: speculative_decoding/decoder.py
# PURPOSE: SpeculativeDecoder — top-level orchestration of the draft/verify loop.
# =============================================================================
"""The :class:`SpeculativeDecoder` orchestrator.

This module wires the draft model, target model, sampler, verifier, KV-cache
managers and metrics together into the canonical Leviathan et al. (2023)
speculative-decoding loop. See ``docs/TECHNICAL_DOCUMENTATION.md`` §2 for the
end-to-end algorithm.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from .config import SpeculativeDecodingConfig
from .core.kv_cache_manager import KVCacheManager
from .core.sampling import Sampler
from .core.verifier import TokenVerifier
from .exceptions import (
    ContextLengthError,
    InvalidPromptError,
    ModelCompatibilityError,
)
from .metrics import DecodingMetrics
from .models.draft_model import DraftModel
from .models.target_model import TargetModel

logger = logging.getLogger(__name__)

# When the underlying HF model exposes ``config.max_position_embeddings`` we
# refuse to generate beyond it. ``None`` means "no limit known".


class SpeculativeDecoder:
    """End-to-end speculative decoder.

    Args:
        draft_model: A :class:`DraftModel`.
        target_model: A :class:`TargetModel`.
        config: A validated :class:`SpeculativeDecodingConfig`.

    Raises:
        ModelCompatibilityError: If draft and target vocab sizes differ.
    """

    def __init__(
        self,
        draft_model: DraftModel,
        target_model: TargetModel,
        config: SpeculativeDecodingConfig,
    ) -> None:
        if draft_model.vocab_size != target_model.vocab_size:
            raise ModelCompatibilityError(
                f"draft vocab_size ({draft_model.vocab_size}) != "
                f"target vocab_size ({target_model.vocab_size}); models "
                f"must share an identical tokenizer."
            )

        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config

        # Sampler/verifier are deterministic given the (optional) generator.
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)
            torch.manual_seed(config.seed)
        self._generator = generator

        self.sampler = Sampler(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            generator=generator,
        )
        self.verifier = TokenVerifier(generator=generator)

    # ---- Public API -----------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """Generate up to ``config.max_new_tokens`` new tokens given a prompt.

        Args:
            prompt_ids: LongTensor of shape ``[L]`` or ``[1, L]``.

        Returns:
            ``(output_ids, metrics)`` where ``output_ids`` has shape
            ``[1, L + n_generated]`` and ``metrics`` is a populated
            :class:`DecodingMetrics`.

        Raises:
            InvalidPromptError: If the prompt is empty.
            ContextLengthError: If the prompt plus requested tokens would
                exceed the underlying model's positional encoding.
        """
        prompt_ids = self._normalise_prompt(prompt_ids)
        self._check_context_length(prompt_ids.shape[1])

        gamma = self.config.speculation_length
        max_new = self.config.max_new_tokens
        eos = self.config.eos_token_id

        metrics = DecodingMetrics(speculation_length=gamma)
        metrics.start()

        device = self.draft_model.device
        committed: list[int] = prompt_ids[0].tolist()

        # ---- Prefill both models with prompt[:-1]; last token is in flight.
        last_token, draft_kv_mgr, target_kv_mgr = self._prefill(prompt_ids)

        # ---- Main loop ---------------------------------------------------
        generated_count = 0
        eos_reached = False
        while generated_count < max_new and not eos_reached:
            # 1. Draft phase: generate γ candidate tokens.
            last_token_2d = torch.tensor(
                [[last_token]], dtype=torch.long, device=device
            )  # shape: [1, 1]
            draft_out = self.draft_model.speculate(
                last_token=last_token_2d,
                past_key_values=draft_kv_mgr.current(),
                gamma=gamma,
                sampler=self.sampler,
            )
            draft_kv_mgr.update(draft_out.past_key_values)

            # 2. Verification: ONE target pass over [last_token, d_1..d_γ].
            target_input = torch.cat(
                [
                    torch.tensor([[last_token]], dtype=torch.long, device=device),
                    draft_out.tokens.unsqueeze(0),  # shape: [1, γ]
                ],
                dim=1,
            )  # shape: [1, γ + 1]
            target_out = self.target_model.parallel_score(
                input_ids=target_input,
                past_key_values=target_kv_mgr.current(),
                sampler=self.sampler,
            )
            target_kv_mgr.update(target_out.past_key_values)

            # 3. Acceptance: modified rejection sampling.
            result = self.verifier.verify(
                draft_tokens=draft_out.tokens,
                draft_log_probs=draft_out.log_probs,
                target_log_probs=target_out.log_probs,
                sampler=self.sampler,
            )

            # 4. Apply EOS / max-tokens truncation BEFORE cache bookkeeping.
            new_tokens = list(result.accepted_tokens)
            n_drafts = result.n_drafts_accepted

            if eos is not None:
                new_tokens, n_drafts, eos_reached = self._truncate_at_eos(
                    new_tokens, n_drafts, eos
                )

            remaining = max_new - generated_count
            if len(new_tokens) > remaining:
                # Hit the per-call token budget mid-iteration.
                new_tokens = new_tokens[:remaining]
                n_drafts = min(n_drafts, len(new_tokens))

            n_emitted = len(new_tokens)
            committed.extend(new_tokens)
            generated_count += n_emitted

            # Metrics: use the VERIFIER's accept count (not the post-truncation
            # value), so α reflects acceptance success, not boundary effects.
            metrics.record_iteration(
                n_drafts_accepted=result.n_drafts_accepted,
                n_tokens_emitted=n_emitted,
            )
            logger.debug(
                "iter %d: emitted=%d, drafts_accepted=%d/%d, total_generated=%d/%d",
                metrics.n_iterations,
                n_emitted,
                n_drafts,
                gamma,
                generated_count,
                max_new,
            )

            # 5. Cache bookkeeping — only if we are continuing.
            if eos_reached or generated_count >= max_new:
                break

            new_last_token = new_tokens[-1]
            self._sync_caches_after_iteration(
                draft_kv_mgr=draft_kv_mgr,
                target_kv_mgr=target_kv_mgr,
                draft_tokens=draft_out.tokens,
                result_n_drafts=result.n_drafts_accepted,
                effective_n_drafts=n_drafts,
                gamma=gamma,
            )
            last_token = new_last_token

        metrics.stop()
        out_ids = torch.tensor(committed, dtype=torch.long, device=device).unsqueeze(0)
        logger.info(
            "generate: %d tokens in %d iter (α=%.3f, %.2f tok/s)",
            metrics.tokens_generated,
            metrics.n_iterations,
            metrics.acceptance_rate,
            metrics.tokens_per_second,
        )
        return out_ids, metrics

    # ---- Helpers --------------------------------------------------------

    def _normalise_prompt(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Coerce a 1-D or 2-D prompt into a ``[1, L]`` tensor."""
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # shape: [1, L]
        if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
            raise InvalidPromptError(
                f"prompt_ids must be 1-D [L] or 2-D [1, L]; got shape "
                f"{tuple(prompt_ids.shape)}"
            )
        if prompt_ids.shape[1] == 0:
            raise InvalidPromptError("prompt_ids must contain at least one token")
        return prompt_ids.to(self.draft_model.device)

    def _check_context_length(self, prompt_len: int) -> None:
        """Raise :class:`ContextLengthError` if generation would overflow."""
        total = prompt_len + self.config.max_new_tokens
        for label, model in (("draft", self.draft_model), ("target", self.target_model)):
            cfg = getattr(model.hf_model, "config", None)
            if cfg is None:
                continue
            limit = getattr(cfg, "max_position_embeddings", None)
            if limit is not None and total > limit:
                raise ContextLengthError(
                    f"{label} model context window is {limit}, but prompt "
                    f"({prompt_len}) + max_new_tokens ({self.config.max_new_tokens}) "
                    f"= {total}."
                )

    def _prefill(
        self,
        prompt_ids: torch.Tensor,
    ) -> Tuple[int, KVCacheManager, KVCacheManager]:
        """Prefill both models with prompt[:-1]; return last token + managers."""
        prompt_len = prompt_ids.shape[1]
        last_token = int(prompt_ids[0, -1].item())

        draft_mgr = KVCacheManager()
        target_mgr = KVCacheManager()

        if prompt_len == 1:
            # Single-token prompt: nothing to prefill; both caches stay empty.
            return last_token, draft_mgr, target_mgr

        # Prefill prompt[:-1] through both models. ``last_token`` is fed by
        # the main loop on the first draft step.
        prefix = prompt_ids[:, :-1]  # shape: [1, L-1]
        draft_pref = self.draft_model.forward(prefix, past_key_values=None)
        draft_mgr.update(draft_pref.past_key_values)
        target_pref = self.target_model.forward(prefix, past_key_values=None)
        target_mgr.update(target_pref.past_key_values)

        return last_token, draft_mgr, target_mgr

    @staticmethod
    def _truncate_at_eos(
        new_tokens: list[int],
        n_drafts: int,
        eos: int,
    ) -> Tuple[list[int], int, bool]:
        """If EOS is present, truncate at and including the first occurrence.

        Returns:
            ``(truncated_tokens, possibly_reduced_n_drafts, eos_reached)``.
        """
        for idx, tok in enumerate(new_tokens):
            if tok == eos:
                kept = new_tokens[: idx + 1]
                # Drafts beyond EOS are also discarded.
                new_n_drafts = min(n_drafts, idx)
                return kept, new_n_drafts, True
        return new_tokens, n_drafts, False

    def _sync_caches_after_iteration(
        self,
        draft_kv_mgr: KVCacheManager,
        target_kv_mgr: KVCacheManager,
        draft_tokens: torch.Tensor,
        result_n_drafts: int,
        effective_n_drafts: int,
        gamma: int,
    ) -> None:
        """Roll back (or catch up) both caches to match the committed prefix.

        See ``docs/TECHNICAL_DOCUMENTATION.md`` §3.2 for the bookkeeping math.

        Args:
            draft_kv_mgr: Draft cache manager.
            target_kv_mgr: Target cache manager.
            draft_tokens: The γ draft tokens from this iteration.
            result_n_drafts: Drafts accepted **by the verifier** (before any
                EOS / max-tokens truncation).
            effective_n_drafts: Drafts the loop actually committed (post
                truncation; ``≤ result_n_drafts``).
            gamma: Speculation length.
        """
        # If truncation chopped some accepted drafts, treat them as rejected
        # for cache purposes.
        n = effective_n_drafts

        if n == gamma:
            # Case A: all γ drafts (and the bonus) committed. Draft cache is
            # one short — feed the last draft token through to catch up.
            # Target cache is already up-to-date.
            _, new_draft_kv = self.draft_model.commit_token(
                token=int(draft_tokens[-1].item()),
                past_key_values=draft_kv_mgr.current(),
            )
            draft_kv_mgr.update(new_draft_kv)
            return

        # Case B: rejection or truncation at position n (0 ≤ n < γ).
        # Draft cache: rollback by (γ − 1) − n.
        # Target cache: rollback by γ − n.
        draft_rollback = (gamma - 1) - n
        target_rollback = gamma - n
        if draft_rollback > 0:
            draft_kv_mgr.rollback(draft_rollback)
        if target_rollback > 0:
            target_kv_mgr.rollback(target_rollback)

# =============================================================================
# FILE: speculative_decoding/models/draft_model.py
# PURPOSE: DraftModel — wraps a small HF causal LM for fast γ-step speculation.
# =============================================================================
"""Draft-model wrapper.

The draft model's job in a speculative decoding iteration is to extend the
context by ``γ`` tokens, returning both the sampled tokens and the **full
sampler-transformed log-probability vector** at each drafted position (so the
verifier can compute exact acceptance ratios).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ..core.kv_cache_manager import PastKV
from ..core.sampling import Sampler
from .base_model import AbstractLanguageModel, ModelForwardOutput

logger = logging.getLogger(__name__)


@dataclass
class DraftOutput:
    """Output of a single draft phase.

    Attributes:
        tokens: LongTensor of shape ``[γ]`` — the γ proposed tokens.
        log_probs: FloatTensor of shape ``[γ, vocab]`` — sampler-transformed
            log-probabilities at each draft position. Tokens are sampled from
            ``log_probs.exp()`` so the verifier's ``log p(d_t)`` lookup is
            consistent.
        past_key_values: The draft model's updated cache.
    """

    tokens: torch.Tensor
    log_probs: torch.Tensor
    past_key_values: PastKV


class DraftModel(AbstractLanguageModel):
    """Wraps a HuggingFace causal LM as the draft component.

    Args:
        hf_model: A loaded ``transformers.PreTrainedModel`` (any causal LM).
        device: Torch device the model is on.
        dtype: Torch dtype of the model weights.

    The wrapper supports any model whose ``forward(input_ids, past_key_values,
    use_cache=True)`` returns an object with ``logits`` and
    ``past_key_values`` attributes — which covers every standard HF causal LM.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.hf_model = hf_model
        self.device = device
        self.dtype = dtype
        # ``vocab_size`` may live in ``config.vocab_size`` (HF) or be set
        # directly by a subclass / mock.
        cfg = getattr(hf_model, "config", None)
        self.vocab_size = int(getattr(cfg, "vocab_size", 0)) if cfg is not None else 0
        if self.vocab_size == 0:
            # Mocks/tests may not set ``config.vocab_size``; allow override
            # via an explicit attribute.
            self.vocab_size = int(getattr(hf_model, "vocab_size", 0))
        self.hf_model.eval()

    # ---- AbstractLanguageModel hooks ------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[PastKV] = None,
    ) -> ModelForwardOutput:
        """Run a single forward pass through the wrapped model.

        Args:
            input_ids: LongTensor of shape ``[batch, seq_len]``.
            past_key_values: Optional prior cache.

        Returns:
            A :class:`ModelForwardOutput`.
        """
        with torch.no_grad():
            out = self.hf_model(
                input_ids=input_ids.to(self.device),
                past_key_values=past_key_values,
                use_cache=True,
            )
        # Old HF returns a tuple; new HF returns a ModelOutput. Handle both.
        logits = getattr(out, "logits", None)
        new_kv = getattr(out, "past_key_values", None)
        if logits is None:
            logits, new_kv = out[0], out[1]
        return ModelForwardOutput(logits=logits, past_key_values=new_kv)

    # ---- Speculation ----------------------------------------------------

    def speculate(
        self,
        last_token: torch.Tensor,
        past_key_values: Optional[PastKV],
        gamma: int,
        sampler: Sampler,
    ) -> DraftOutput:
        """Generate ``γ`` candidate tokens autoregressively.

        Args:
            last_token: LongTensor of shape ``[1, 1]`` — the most recent
                committed token (entering the draft loop as the first input).
            past_key_values: Cache prior to drafting (contains everything
                up to but not including ``last_token``).
            gamma: Number of draft tokens to produce (``> 0``).
            sampler: Sampler used to produce token ids and the log-prob
                vectors handed to the verifier.

        Returns:
            A :class:`DraftOutput`.
        """
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1; got {gamma}")

        draft_tokens: list[int] = []
        log_prob_rows: list[torch.Tensor] = []
        cur_input = last_token.to(self.device)  # shape: [1, 1]
        cur_kv = past_key_values

        for step in range(gamma):
            out = self.forward(cur_input, past_key_values=cur_kv)
            cur_kv = out.past_key_values
            # out.logits shape: [1, 1, vocab]; we want shape [vocab].
            next_logits = out.logits[0, -1, :]  # shape: [vocab]
            log_probs = sampler.log_probs(next_logits)  # shape: [vocab]
            token = sampler.sample(next_logits)  # shape: scalar
            token_id = int(token.item())
            draft_tokens.append(token_id)
            log_prob_rows.append(log_probs)
            cur_input = token.view(1, 1)
            logger.debug("draft step %d/%d -> token %d", step + 1, gamma, token_id)

        tokens_tensor = torch.tensor(
            draft_tokens, dtype=torch.long, device=self.device
        )  # shape: [γ]
        log_probs_tensor = torch.stack(log_prob_rows, dim=0)  # shape: [γ, vocab]
        return DraftOutput(
            tokens=tokens_tensor,
            log_probs=log_probs_tensor,
            past_key_values=cur_kv,
        )

    # ---- Catch-up for fully-accepted iterations -------------------------

    def commit_token(
        self,
        token: int,
        past_key_values: Optional[PastKV],
    ) -> Tuple[torch.Tensor, PastKV]:
        """Process one already-committed token through the draft to update its KV.

        This is used when **all γ draft tokens were accepted**: in that case
        the draft cache holds only ``γ`` of the new tokens (one short), so we
        feed the bonus token through draft to keep the cache in sync. See
        ``docs/TECHNICAL_DOCUMENTATION.md`` §3.2 for the bookkeeping.

        Args:
            token: A previously committed token id.
            past_key_values: The draft cache prior to processing ``token``.

        Returns:
            ``(next_logits_unused, new_past_key_values)``. The logits are
            returned for completeness; the decoder typically discards them.
        """
        input_ids = torch.tensor(
            [[token]], dtype=torch.long, device=self.device
        )  # shape: [1, 1]
        out = self.forward(input_ids, past_key_values=past_key_values)
        return out.logits[0, -1, :], out.past_key_values  # shape: [vocab], PastKV

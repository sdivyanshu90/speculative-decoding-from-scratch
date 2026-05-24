# =============================================================================
# FILE: speculative_decoding/models/target_model.py
# PURPOSE: TargetModel — wraps a large HF causal LM for one-pass verification.
# =============================================================================
"""Target-model wrapper.

In one speculative iteration, the target model runs a **single** forward pass
over ``[last_committed_token, d_1, …, d_γ]`` (length ``γ + 1``) and yields
the γ+1 next-token distributions needed by the verifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..core.kv_cache_manager import PastKV
from ..core.sampling import Sampler
from .base_model import AbstractLanguageModel, ModelForwardOutput

logger = logging.getLogger(__name__)


@dataclass
class TargetScoreOutput:
    """Output of one target-model verification pass.

    Attributes:
        log_probs: FloatTensor of shape ``[γ + 1, vocab]`` — sampler-
            transformed log-probabilities at each of the γ verification
            positions plus the bonus position.
        past_key_values: The target model's updated cache.
    """

    log_probs: torch.Tensor
    past_key_values: PastKV


class TargetModel(AbstractLanguageModel):
    """Wraps a HuggingFace causal LM as the verification component.

    Args:
        hf_model: A loaded ``transformers.PreTrainedModel``.
        device: Torch device.
        dtype: Torch dtype.
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
        cfg = getattr(hf_model, "config", None)
        self.vocab_size = int(getattr(cfg, "vocab_size", 0)) if cfg is not None else 0
        if self.vocab_size == 0:
            self.vocab_size = int(getattr(hf_model, "vocab_size", 0))
        self.hf_model.eval()

    # ---- AbstractLanguageModel hooks ------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[PastKV] = None,
    ) -> ModelForwardOutput:
        """Run a single forward pass.

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
        logits = getattr(out, "logits", None)
        new_kv = getattr(out, "past_key_values", None)
        if logits is None:
            logits, new_kv = out[0], out[1]
        return ModelForwardOutput(logits=logits, past_key_values=new_kv)

    # ---- Verification single-pass scoring -------------------------------

    def parallel_score(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[PastKV],
        sampler: Sampler,
    ) -> TargetScoreOutput:
        """Score ``γ+1`` candidate positions in one forward pass.

        Args:
            input_ids: LongTensor of shape ``[1, γ + 1]`` containing
                ``[last_committed_token, d_1, …, d_γ]``.
            past_key_values: Cache prior to verification (contains everything
                up to but not including ``last_committed_token``).
            sampler: Sampler used to transform raw logits before extracting
                log-probabilities.

        Returns:
            A :class:`TargetScoreOutput`.
        """
        out = self.forward(input_ids, past_key_values=past_key_values)
        # out.logits shape: [1, γ+1, vocab]; squeeze batch.
        raw_logits = out.logits[0]  # shape: [γ+1, vocab]
        log_probs = sampler.log_probs(raw_logits)  # shape: [γ+1, vocab]
        logger.debug("target parallel_score over %d positions", input_ids.shape[1])
        return TargetScoreOutput(log_probs=log_probs, past_key_values=out.past_key_values)

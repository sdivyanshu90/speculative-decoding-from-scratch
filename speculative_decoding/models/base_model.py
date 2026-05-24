# =============================================================================
# FILE: speculative_decoding/models/base_model.py
# PURPOSE: AbstractLanguageModel ABC — the contract DraftModel/TargetModel implement.
# =============================================================================
"""Common interface for any language model usable inside speculative decoding.

The interface is intentionally minimal: a single ``forward`` that takes
``input_ids`` plus an optional ``past_key_values`` and returns the next-token
logits plus the updated cache. Both draft and target models implement this;
mock models in the test suite implement it too.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from ..core.kv_cache_manager import PastKV


@dataclass
class ModelForwardOutput:
    """The output of a single model forward pass.

    Attributes:
        logits: FloatTensor of shape ``[batch, seq_len, vocab]``. ``seq_len``
            equals the length of ``input_ids`` passed in.
        past_key_values: The new HuggingFace-style cache after this pass.
    """

    logits: torch.Tensor
    past_key_values: PastKV


class AbstractLanguageModel(ABC):
    """ABC for any language model that can be plugged into the decoder.

    Concrete implementations must expose:
      * ``vocab_size``
      * ``device``
      * ``dtype``
      * a ``forward`` returning a :class:`ModelForwardOutput`.

    Subclasses MAY wrap a HuggingFace ``PreTrainedModel`` (see
    :class:`speculative_decoding.models.draft_model.DraftModel`) or be a pure
    test stub.
    """

    # ---- Required attributes -------------------------------------------

    vocab_size: int
    device: torch.device
    dtype: torch.dtype

    # ---- Required methods ----------------------------------------------

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[PastKV] = None,
    ) -> ModelForwardOutput:
        """Run a forward pass.

        Args:
            input_ids: LongTensor of shape ``[batch, seq_len]``.
            past_key_values: Optional prior cache to extend.

        Returns:
            A :class:`ModelForwardOutput`.
        """
        raise NotImplementedError

    # ---- Convenience ---------------------------------------------------

    def prefill(self, prompt_ids: torch.Tensor) -> ModelForwardOutput:
        """Run a fresh forward pass over a prompt (no prior cache).

        Args:
            prompt_ids: LongTensor of shape ``[batch, prompt_len]``.

        Returns:
            A :class:`ModelForwardOutput` with the full prompt cache.
        """
        return self.forward(prompt_ids, past_key_values=None)

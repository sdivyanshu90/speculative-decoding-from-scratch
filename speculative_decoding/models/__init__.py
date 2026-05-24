# =============================================================================
# FILE: speculative_decoding/models/__init__.py
# PURPOSE: Public exports for the model layer.
# =============================================================================
"""Model abstractions for speculative decoding."""

from .base_model import AbstractLanguageModel, ModelForwardOutput
from .draft_model import DraftModel, DraftOutput
from .target_model import TargetModel, TargetScoreOutput

__all__ = [
    "AbstractLanguageModel",
    "ModelForwardOutput",
    "DraftModel",
    "DraftOutput",
    "TargetModel",
    "TargetScoreOutput",
]

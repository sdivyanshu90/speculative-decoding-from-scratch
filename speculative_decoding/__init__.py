# =============================================================================
# FILE: speculative_decoding/__init__.py
# PURPOSE: Public API of the speculative_decoding package.
# =============================================================================
"""Speculative Decoding — a from-scratch implementation of Leviathan et al. (2023).

Public symbols:

    SpeculativeDecoder         — top-level orchestrator
    SpeculativeDecodingConfig  — dataclass holding all runtime config
    DecodingMetrics            — per-generation statistics container
    DraftModel, TargetModel    — HuggingFace causal-LM wrappers
    Sampler                    — temperature / top-k / top-p sampling utility
    TokenVerifier              — modified rejection sampling verifier
    KVCacheManager             — past_key_values rollback helper
    Various exceptions in :mod:`speculative_decoding.exceptions`
"""

from __future__ import annotations

from .config import SpeculativeDecodingConfig
from .core.kv_cache_manager import KVCacheManager, PastKV
from .core.sampling import Sampler
from .core.verifier import TokenVerifier, VerificationResult
from .decoder import SpeculativeDecoder
from .exceptions import (
    ContextLengthError,
    InvalidPromptError,
    KVCacheRollbackError,
    ModelCompatibilityError,
    SamplingConfigError,
    SpeculativeDecodingConfigError,
    SpeculativeDecodingError,
    VerifierShapeError,
)
from .metrics import DecodingMetrics
from .models.base_model import AbstractLanguageModel, ModelForwardOutput
from .models.draft_model import DraftModel, DraftOutput
from .models.target_model import TargetModel, TargetScoreOutput

__all__ = [
    # Config / orchestrator
    "SpeculativeDecoder",
    "SpeculativeDecodingConfig",
    "DecodingMetrics",
    # Models
    "AbstractLanguageModel",
    "ModelForwardOutput",
    "DraftModel",
    "DraftOutput",
    "TargetModel",
    "TargetScoreOutput",
    # Core
    "Sampler",
    "TokenVerifier",
    "VerificationResult",
    "KVCacheManager",
    "PastKV",
    # Exceptions
    "SpeculativeDecodingError",
    "SpeculativeDecodingConfigError",
    "SamplingConfigError",
    "KVCacheRollbackError",
    "VerifierShapeError",
    "ModelCompatibilityError",
    "ContextLengthError",
    "InvalidPromptError",
]

__version__ = "0.1.0"

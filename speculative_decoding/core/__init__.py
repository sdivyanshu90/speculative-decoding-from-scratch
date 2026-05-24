# =============================================================================
# FILE: speculative_decoding/core/__init__.py
# PURPOSE: Public exports for the algorithmic core (sampler, verifier, cache).
# =============================================================================
"""Algorithmic primitives: sampling, verification, KV-cache management."""

from .kv_cache_manager import KVCacheManager, PastKV
from .sampling import Sampler
from .verifier import TokenVerifier, VerificationResult

__all__ = [
    "Sampler",
    "TokenVerifier",
    "VerificationResult",
    "KVCacheManager",
    "PastKV",
]

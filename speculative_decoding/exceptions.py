# =============================================================================
# FILE: speculative_decoding/exceptions.py
# PURPOSE: Custom exception hierarchy for the speculative-decoding package.
# =============================================================================
"""Exception types raised by the speculative-decoding library.

All exceptions inherit from :class:`SpeculativeDecodingError` so that callers
can catch every library-raised error with a single ``except`` clause if they
wish.
"""

from __future__ import annotations


class SpeculativeDecodingError(Exception):
    """Base class for every exception raised by this package."""


class SpeculativeDecodingConfigError(SpeculativeDecodingError, ValueError):
    """A :class:`SpeculativeDecodingConfig` failed validation."""


class SamplingConfigError(SpeculativeDecodingError, ValueError):
    """A :class:`Sampler` was constructed with invalid parameters."""


class KVCacheRollbackError(SpeculativeDecodingError, RuntimeError):
    """A KV-cache rollback was requested for more positions than are cached."""


class VerifierShapeError(SpeculativeDecodingError, ValueError):
    """The token verifier received tensors with mismatched or invalid shapes."""


class ModelCompatibilityError(SpeculativeDecodingError, ValueError):
    """Draft and target models are not compatible (e.g., different vocab sizes)."""


class ContextLengthError(SpeculativeDecodingError, ValueError):
    """A prompt or in-flight generation would exceed the model's context window."""


class InvalidPromptError(SpeculativeDecodingError, ValueError):
    """The supplied prompt is empty or otherwise malformed."""

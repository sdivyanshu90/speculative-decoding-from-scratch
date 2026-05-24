# =============================================================================
# FILE: speculative_decoding/core/sampling.py
# PURPOSE: Temperature / top-k / top-p sampling utilities and the Sampler class.
# =============================================================================
"""Sampling utilities used by both the draft generator and the verifier.

All public functions accept and return tensors in **logit space** (not
probabilities). Sampling is the only operation that materialises a discrete
token; every probability computation upstream is kept in log-space for
numerical stability.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..exceptions import SamplingConfigError

# Sentinel value used to mask out filtered logits. Using ``-inf`` is safe
# because ``softmax(-inf) == 0`` and ``log_softmax(-inf) == -inf``.
NEG_INF: float = float("-inf")

# Numerical floor for greedy-mode logit comparisons; we never get this close
# in practice because logits live on a much smaller scale than ``-inf``.
GREEDY_TEMPERATURE_THRESHOLD: float = 1e-8


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature, treating ``temperature == 0`` as greedy.

    Args:
        logits: Tensor of shape ``[..., vocab]``.
        temperature: Non-negative scalar.

    Returns:
        A tensor of the same shape as ``logits``. For ``temperature == 0``,
        a one-hot mask at the argmax is returned (logits = 0 for argmax,
        -inf elsewhere), which when fed to softmax yields a delta distribution.

    Raises:
        SamplingConfigError: If ``temperature`` is negative.
    """
    if temperature < 0:
        raise SamplingConfigError(f"temperature must be >= 0; got {temperature}")
    if temperature < GREEDY_TEMPERATURE_THRESHOLD:
        # Greedy mode: build a delta distribution at the argmax.
        argmax = logits.argmax(dim=-1, keepdim=True)  # shape: [..., 1]
        out = torch.full_like(logits, NEG_INF)  # shape: [..., vocab]
        out.scatter_(dim=-1, index=argmax, value=0.0)
        return out
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask out all but the top-k logits along the last axis.

    Args:
        logits: Tensor of shape ``[..., vocab]``.
        top_k: Number of tokens to keep. ``0`` disables filtering.

    Returns:
        A tensor of the same shape; non-top-k positions are set to ``-inf``.

    Raises:
        SamplingConfigError: If ``top_k`` is negative.
    """
    if top_k < 0:
        raise SamplingConfigError(f"top_k must be >= 0; got {top_k}")
    if top_k == 0:
        return logits
    vocab_size = logits.shape[-1]
    k = min(top_k, vocab_size)
    # ``topk`` returns the k largest values per row; everything below the
    # smallest kept value gets masked.
    kth_value = torch.topk(logits, k=k, dim=-1).values[..., -1:]  # shape: [..., 1]
    mask = logits < kth_value  # shape: [..., vocab]
    return logits.masked_fill(mask, NEG_INF)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Mask out the tail of the distribution beyond cumulative probability p.

    Args:
        logits: Tensor of shape ``[..., vocab]``.
        top_p: Probability mass in ``(0, 1]``. ``1.0`` disables filtering.

    Returns:
        A tensor of the same shape; tail positions set to ``-inf``.

    Raises:
        SamplingConfigError: If ``top_p`` is not in ``(0, 1]``.
    """
    if not (0.0 < top_p <= 1.0):
        raise SamplingConfigError(f"top_p must be in (0, 1]; got {top_p}")
    if top_p >= 1.0:
        return logits

    # Sort logits descending; compute cumulative softmax along the sorted axis.
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)  # shape: [..., vocab]

    # Identify the smallest prefix whose cumulative mass >= top_p. Everything
    # strictly past that prefix is filtered. Shift right by one so the boundary
    # token itself is retained (guarantees at least one token kept).
    sorted_mask = cumulative > top_p  # shape: [..., vocab]
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Scatter the sorted mask back to original token positions.
    mask = torch.zeros_like(sorted_mask)  # shape: [..., vocab]
    mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
    return logits.masked_fill(mask, NEG_INF)


class Sampler:
    """Stateless sampler combining temperature, top-k and top-p filtering.

    All transformations are applied in this order: temperature → top-k → top-p.
    The class is intentionally lightweight; it carries no torch parameters and
    is safe to share across threads / devices.

    Attributes:
        temperature: Sampling temperature. ``0.0`` is greedy.
        top_p: Nucleus sampling threshold in ``(0, 1]``.
        top_k: Top-k cutoff. ``0`` disables top-k.
        generator: Optional :class:`torch.Generator` for reproducible sampling.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Construct and validate the sampler.

        Args:
            temperature: See class docstring.
            top_p: See class docstring.
            top_k: See class docstring.
            generator: Optional torch generator for deterministic sampling.

        Raises:
            SamplingConfigError: If any parameter is out of range.
        """
        if temperature < 0:
            raise SamplingConfigError(f"temperature must be >= 0; got {temperature}")
        if not (0.0 < top_p <= 1.0):
            raise SamplingConfigError(f"top_p must be in (0, 1]; got {top_p}")
        if top_k < 0:
            raise SamplingConfigError(f"top_k must be >= 0; got {top_k}")

        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.generator = generator

    @property
    def is_greedy(self) -> bool:
        """Return True if temperature is effectively zero (greedy decoding)."""
        return self.temperature < GREEDY_TEMPERATURE_THRESHOLD

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature, top-k and top-p transformations in order.

        Args:
            logits: Tensor of shape ``[..., vocab]``.

        Returns:
            Transformed logits, same shape; filtered tokens are ``-inf``.
        """
        x = apply_temperature(logits, self.temperature)
        x = apply_top_k(x, self.top_k)
        x = apply_top_p(x, self.top_p)
        return x

    def log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the log-softmax of the transformed logits.

        Args:
            logits: Tensor of shape ``[..., vocab]``.

        Returns:
            Log-probabilities of the same shape. Filtered positions have
            value ``-inf``.
        """
        transformed = self.transform_logits(logits)  # shape: [..., vocab]
        return torch.log_softmax(transformed, dim=-1)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the softmax probabilities of the transformed logits."""
        return torch.softmax(self.transform_logits(logits), dim=-1)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token id per row from the transformed logits.

        Args:
            logits: Tensor of shape ``[..., vocab]``. The function flattens
                the leading dimensions, samples per row, then reshapes.

        Returns:
            LongTensor of shape ``[...]`` containing sampled token ids.
        """
        # Greedy fast path: avoid building a probability vector.
        if self.is_greedy:
            return logits.argmax(dim=-1)

        transformed = self.transform_logits(logits)  # shape: [..., vocab]
        probs = torch.softmax(transformed, dim=-1)  # shape: [..., vocab]

        # ``torch.multinomial`` requires 1-D or 2-D input; flatten the batch.
        original_shape = probs.shape[:-1]
        flat_probs = probs.reshape(-1, probs.shape[-1])  # shape: [N, vocab]
        sampled = torch.multinomial(flat_probs, num_samples=1, generator=self.generator)
        return sampled.reshape(original_shape)

    def sample_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample a token id from an already-normalised probability vector.

        Used by the rejection-sampling resample step where the residual
        distribution has already been computed.

        Args:
            probs: Tensor of shape ``[..., vocab]``; rows must sum to 1.

        Returns:
            LongTensor of shape ``[...]``.
        """
        original_shape = probs.shape[:-1]
        flat = probs.reshape(-1, probs.shape[-1])  # shape: [N, vocab]
        sampled = torch.multinomial(flat, num_samples=1, generator=self.generator)
        return sampled.reshape(original_shape)

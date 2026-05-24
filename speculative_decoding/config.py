# =============================================================================
# FILE: speculative_decoding/config.py
# PURPOSE: Configuration dataclass for the speculative decoding system.
# =============================================================================
"""Configuration objects for speculative decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

from .exceptions import SpeculativeDecodingConfigError

# Default speculation length. Conservative starting point in the literature.
DEFAULT_SPECULATION_LENGTH: int = 5

# Default sampling parameters.
DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_TOP_P: float = 1.0
DEFAULT_TOP_K: int = 0  # 0 means top-k disabled.

# Permitted device strings.
ALLOWED_DEVICES: tuple[str, ...] = ("cpu", "cuda", "mps")

# Map of torch dtype string -> torch.dtype, used by :meth:`from_dict`.
_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
}


@dataclass
class SpeculativeDecodingConfig:
    """Runtime configuration for a :class:`SpeculativeDecoder` instance.

    Attributes:
        speculation_length: ``γ`` — the number of draft tokens generated per
            iteration. Must be a positive integer. Typical range 3–10.
        max_new_tokens: Total number of tokens to generate (excluding the
            prompt). Must be positive.
        temperature: Sampling temperature. ``0.0`` selects greedy decoding,
            which collapses the acceptance criterion to argmax equality.
        top_p: Nucleus-sampling cumulative-probability threshold in ``(0, 1]``.
            Use ``1.0`` to disable.
        top_k: Top-k sampling cutoff. ``0`` disables top-k.
        target_model_name: HuggingFace model identifier for the target model
            (e.g., ``"facebook/opt-1.3b"``).
        draft_model_name: HuggingFace model identifier for the draft model
            (e.g., ``"facebook/opt-125m"``).
        device: Torch device string. Must be one of :data:`ALLOWED_DEVICES`.
        dtype: Torch dtype for model weights and activations.
        seed: Optional random seed. ``None`` uses the global RNG state.
        eos_token_id: Token id at which generation halts early. ``None`` means
            generation runs for the full ``max_new_tokens``.
    """

    speculation_length: int = DEFAULT_SPECULATION_LENGTH
    max_new_tokens: int = 64
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    target_model_name: str = ""
    draft_model_name: str = ""
    device: str = "cpu"
    dtype: torch.dtype = field(default=torch.float32)
    seed: Optional[int] = None
    eos_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate the configuration immediately after construction."""
        self.validate()

    def validate(self) -> None:
        """Validate field values and raise on any constraint violation.

        Raises:
            SpeculativeDecodingConfigError: If any field is invalid.
        """
        if not isinstance(self.speculation_length, int) or self.speculation_length < 1:
            raise SpeculativeDecodingConfigError(
                f"speculation_length must be a positive int; got {self.speculation_length!r}"
            )
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise SpeculativeDecodingConfigError(
                f"max_new_tokens must be a positive int; got {self.max_new_tokens!r}"
            )
        if self.temperature < 0:
            raise SpeculativeDecodingConfigError(
                f"temperature must be >= 0; got {self.temperature!r}"
            )
        if not (0.0 < self.top_p <= 1.0):
            raise SpeculativeDecodingConfigError(
                f"top_p must be in (0, 1]; got {self.top_p!r}"
            )
        if self.top_k < 0:
            raise SpeculativeDecodingConfigError(
                f"top_k must be >= 0; got {self.top_k!r}"
            )
        if self.device not in ALLOWED_DEVICES:
            raise SpeculativeDecodingConfigError(
                f"device must be one of {ALLOWED_DEVICES}; got {self.device!r}"
            )
        if not isinstance(self.dtype, torch.dtype):
            raise SpeculativeDecodingConfigError(
                f"dtype must be a torch.dtype; got {type(self.dtype).__name__}"
            )
        if self.seed is not None and not isinstance(self.seed, int):
            raise SpeculativeDecodingConfigError(
                f"seed must be int or None; got {type(self.seed).__name__}"
            )
        if self.eos_token_id is not None and (
            not isinstance(self.eos_token_id, int) or self.eos_token_id < 0
        ):
            raise SpeculativeDecodingConfigError(
                f"eos_token_id must be a non-negative int or None; got {self.eos_token_id!r}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeculativeDecodingConfig":
        """Build a config from a plain dict (e.g., parsed JSON/YAML).

        Args:
            data: A mapping of field names to values. The ``dtype`` value may
                be a string such as ``"float16"`` or a :class:`torch.dtype`.

        Returns:
            A validated :class:`SpeculativeDecodingConfig` instance.

        Raises:
            SpeculativeDecodingConfigError: If validation fails.
        """
        kwargs = dict(data)
        dtype_value = kwargs.get("dtype")
        if isinstance(dtype_value, str):
            try:
                kwargs["dtype"] = _DTYPE_BY_NAME[dtype_value.lower()]
            except KeyError as exc:
                raise SpeculativeDecodingConfigError(
                    f"Unknown dtype string {dtype_value!r}; "
                    f"allowed: {sorted(_DTYPE_BY_NAME)}"
                ) from exc
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-friendly dict.

        ``dtype`` is emitted as its string name (e.g., ``"float16"``).

        Returns:
            A dict suitable for ``json.dumps``.
        """
        payload = asdict(self)
        # ``torch.dtype`` does not survive JSON serialization; emit a string.
        payload["dtype"] = str(self.dtype).replace("torch.", "")
        return payload

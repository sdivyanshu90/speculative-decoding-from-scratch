# =============================================================================
# FILE: tests/unit/test_config.py
# PURPOSE: Unit tests for SpeculativeDecodingConfig.
# =============================================================================
"""Unit tests for :class:`SpeculativeDecodingConfig`."""

from __future__ import annotations

import pytest
import torch

from speculative_decoding import (
    SpeculativeDecodingConfig,
    SpeculativeDecodingConfigError,
)


def _base_kwargs() -> dict:
    return {
        "speculation_length": 5,
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "target_model_name": "foo",
        "draft_model_name": "bar",
        "device": "cpu",
        "dtype": torch.float32,
        "seed": 1,
    }


def test_construct_valid_config() -> None:
    """A valid config constructs without error and stores fields."""
    cfg = SpeculativeDecodingConfig(**_base_kwargs())
    assert cfg.speculation_length == 5
    assert cfg.max_new_tokens == 32
    assert cfg.dtype is torch.float32
    assert cfg.seed == 1


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_invalid_speculation_length_raises(bad_value: int) -> None:
    """Zero or negative speculation_length is rejected."""
    kwargs = _base_kwargs()
    kwargs["speculation_length"] = bad_value
    with pytest.raises(SpeculativeDecodingConfigError, match="speculation_length"):
        SpeculativeDecodingConfig(**kwargs)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_invalid_max_new_tokens_raises(bad_value: int) -> None:
    """Zero or negative max_new_tokens is rejected."""
    kwargs = _base_kwargs()
    kwargs["max_new_tokens"] = bad_value
    with pytest.raises(SpeculativeDecodingConfigError, match="max_new_tokens"):
        SpeculativeDecodingConfig(**kwargs)


def test_negative_temperature_raises() -> None:
    """Negative temperature is rejected."""
    kwargs = _base_kwargs()
    kwargs["temperature"] = -0.1
    with pytest.raises(SpeculativeDecodingConfigError, match="temperature"):
        SpeculativeDecodingConfig(**kwargs)


@pytest.mark.parametrize("bad_value", [0.0, -0.5, 1.01, 2.0])
def test_invalid_top_p_raises(bad_value: float) -> None:
    """top_p outside (0, 1] is rejected."""
    kwargs = _base_kwargs()
    kwargs["top_p"] = bad_value
    with pytest.raises(SpeculativeDecodingConfigError, match="top_p"):
        SpeculativeDecodingConfig(**kwargs)


def test_invalid_top_k_raises() -> None:
    """Negative top_k is rejected."""
    kwargs = _base_kwargs()
    kwargs["top_k"] = -1
    with pytest.raises(SpeculativeDecodingConfigError, match="top_k"):
        SpeculativeDecodingConfig(**kwargs)


def test_invalid_device_raises() -> None:
    """Unknown device strings are rejected."""
    kwargs = _base_kwargs()
    kwargs["device"] = "tpu"
    with pytest.raises(SpeculativeDecodingConfigError, match="device"):
        SpeculativeDecodingConfig(**kwargs)


def test_invalid_dtype_raises() -> None:
    """Non-torch.dtype dtype is rejected."""
    kwargs = _base_kwargs()
    kwargs["dtype"] = "float16"  # string, not torch.dtype
    with pytest.raises(SpeculativeDecodingConfigError, match="dtype"):
        SpeculativeDecodingConfig(**kwargs)


def test_from_dict_parses_dtype_string() -> None:
    """from_dict converts a 'float16' string into torch.float16."""
    data = _base_kwargs()
    data["dtype"] = "float16"
    cfg = SpeculativeDecodingConfig.from_dict(data)
    assert cfg.dtype is torch.float16


def test_from_dict_unknown_dtype_string_raises() -> None:
    """Unknown dtype string raises."""
    data = _base_kwargs()
    data["dtype"] = "complex42"
    with pytest.raises(SpeculativeDecodingConfigError, match="dtype"):
        SpeculativeDecodingConfig.from_dict(data)


def test_to_dict_roundtrip() -> None:
    """to_dict followed by from_dict (with stringified dtype) round-trips."""
    cfg = SpeculativeDecodingConfig(**_base_kwargs())
    payload = cfg.to_dict()
    assert isinstance(payload["dtype"], str)
    cfg2 = SpeculativeDecodingConfig.from_dict(payload)
    assert cfg2.speculation_length == cfg.speculation_length
    assert cfg2.dtype is cfg.dtype
    assert cfg2.seed == cfg.seed


def test_invalid_seed_type_raises() -> None:
    """Non-int seed is rejected."""
    kwargs = _base_kwargs()
    kwargs["seed"] = "1234"  # type: ignore[assignment]
    with pytest.raises(SpeculativeDecodingConfigError, match="seed"):
        SpeculativeDecodingConfig(**kwargs)


def test_invalid_eos_token_id_raises() -> None:
    """Negative or non-int eos_token_id is rejected."""
    kwargs = _base_kwargs()
    kwargs["eos_token_id"] = -1
    with pytest.raises(SpeculativeDecodingConfigError, match="eos_token_id"):
        SpeculativeDecodingConfig(**kwargs)

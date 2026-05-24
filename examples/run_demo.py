# =============================================================================
# FILE: examples/run_demo.py
# PURPOSE: End-to-end demo: speculative decoding vs. standard target decoding.
# =============================================================================
"""Loads real HuggingFace models, runs speculative decoding, and compares to
standard autoregressive decoding from the target model alone.

Run with: ``python examples/run_demo.py``

Default models: ``facebook/opt-125m`` (draft) and ``facebook/opt-1.3b`` (target).
Override with environment variables ``SPECDEC_DRAFT_MODEL`` and
``SPECDEC_TARGET_MODEL``.
"""

from __future__ import annotations

import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decoding import (
    DraftModel,
    SpeculativeDecoder,
    SpeculativeDecodingConfig,
    TargetModel,
)

# Default model identifiers. Both share the OPT tokenizer.
DEFAULT_DRAFT_MODEL: str = "facebook/opt-125m"
DEFAULT_TARGET_MODEL: str = "facebook/opt-1.3b"
DEFAULT_PROMPT: str = "The unique advantage of speculative decoding is that"
DEFAULT_MAX_NEW_TOKENS: int = 64
DEFAULT_SPECULATION_LENGTH: int = 5
DEFAULT_TEMPERATURE: float = 0.0  # greedy: highest acceptance, deterministic.


def pick_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """Pick a reasonable dtype for the chosen device."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def baseline_generate(
    hf_model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Standard autoregressive decoding for wall-clock comparison."""
    hf_model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        out = hf_model.generate(
            prompt_ids.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    return out, elapsed


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=os.environ.get("SPECDEC_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    draft_name = os.environ.get("SPECDEC_DRAFT_MODEL", DEFAULT_DRAFT_MODEL)
    target_name = os.environ.get("SPECDEC_TARGET_MODEL", DEFAULT_TARGET_MODEL)
    prompt_text = os.environ.get("SPECDEC_PROMPT", DEFAULT_PROMPT)

    device = pick_device()
    dtype = pick_dtype(device)

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Draft model:  {draft_name}")
    print(f"Target model: {target_name}")
    print(f"Prompt: {prompt_text!r}\n")

    # ---- Tokenizer (shared) ----
    tokenizer = AutoTokenizer.from_pretrained(target_name)

    # ---- Load models ----
    print("Loading draft model ...")
    draft_hf = AutoModelForCausalLM.from_pretrained(draft_name, torch_dtype=dtype).to(
        device
    )
    print("Loading target model ...")
    target_hf = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=dtype).to(
        device
    )

    draft = DraftModel(draft_hf, device=device, dtype=dtype)
    target = TargetModel(target_hf, device=device, dtype=dtype)

    config = SpeculativeDecodingConfig(
        speculation_length=DEFAULT_SPECULATION_LENGTH,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=1.0,
        top_k=0,
        target_model_name=target_name,
        draft_model_name=draft_name,
        device=device.type,
        dtype=dtype,
        seed=0,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoder = SpeculativeDecoder(draft, target, config)

    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # ---- Speculative decoding -----
    print("\n--- Speculative Decoding ---")
    spec_start = time.perf_counter()
    spec_out, metrics = decoder.generate(prompt_ids)
    spec_elapsed = time.perf_counter() - spec_start
    spec_text = tokenizer.decode(spec_out[0], skip_special_tokens=True)
    print(spec_text)
    print(metrics.generate_report())
    print(f"Outer wall-clock: {spec_elapsed:.3f}s")

    # ---- Baseline -----
    print("\n--- Baseline (target alone, HF .generate) ---")
    base_out, base_elapsed = baseline_generate(
        target_hf,
        tokenizer,
        prompt_ids,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TEMPERATURE,
        device,
    )
    base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
    print(base_text)
    print(f"Baseline wall-clock: {base_elapsed:.3f}s")

    # ---- Comparison -----
    speedup = base_elapsed / spec_elapsed if spec_elapsed > 0 else float("nan")
    print("\n--- Comparison ---")
    print(f"  Acceptance rate (α): {metrics.acceptance_rate:.3f}")
    print(f"  Tokens generated (spec / base): "
          f"{metrics.tokens_generated} / {base_out.shape[1] - prompt_ids.shape[1]}")
    print(f"  Wall-clock speedup:  {speedup:.2f}x")


if __name__ == "__main__":
    main()

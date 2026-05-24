# Speculative Decoding — From Scratch

A complete, production-ready implementation of **Speculative Decoding**
(Leviathan, Kalman, Matias — Google Research, 2023) with full unit, integration
and stress test coverage. Built on top of PyTorch + HuggingFace Transformers.

## What is speculative decoding?

A small **draft** model proposes the next γ tokens; a large **target** model
verifies them in a single parallel forward pass. Accepted tokens are committed
verbatim; the first rejected token is resampled from a residual distribution
that **mathematically preserves** the target distribution. End-to-end you get
1.5×–3× wall-clock speedup over standard autoregressive decoding, with zero
quality loss.

See **`docs/TECHNICAL_DOCUMENTATION.md`** for the full theory, architecture,
proofs, hyperparameter guide and known pitfalls.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, PyTorch ≥ 2.1, Transformers ≥ 4.40.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative_decoding import (
    DraftModel, TargetModel, SpeculativeDecoder, SpeculativeDecodingConfig,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
draft_hf  = AutoModelForCausalLM.from_pretrained("facebook/opt-125m",  torch_dtype=dtype).to(device)
target_hf = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b",  torch_dtype=dtype).to(device)

decoder = SpeculativeDecoder(
    DraftModel(draft_hf, device=device, dtype=dtype),
    TargetModel(target_hf, device=device, dtype=dtype),
    SpeculativeDecodingConfig(
        speculation_length=5,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        device=device.type,
        dtype=dtype,
        seed=42,
        eos_token_id=tokenizer.eos_token_id,
        draft_model_name="facebook/opt-125m",
        target_model_name="facebook/opt-1.3b",
    ),
)

prompt = tokenizer.encode("The unique advantage of speculative decoding is that", return_tensors="pt")
output_ids, metrics = decoder.generate(prompt)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
print(metrics.generate_report())
```

## Run the demo

```bash
python examples/run_demo.py
```

Compares speculative vs. standard decoding side-by-side and prints
acceptance rate, theoretical speedup, and measured tokens/second.

## Architecture

```
SpeculativeDecoder
  ├── DraftModel        ── speculate γ tokens autoregressively
  ├── TargetModel       ── parallel score γ+1 positions in 1 forward pass
  ├── TokenVerifier     ── modified rejection sampling
  ├── KVCacheManager    ── O(1) rollback on rejection
  ├── Sampler           ── temperature / top-k / top-p
  └── DecodingMetrics   ── acceptance rate, speedup, throughput
```

Documentation: see `docs/TECHNICAL_DOCUMENTATION.md` for the full theoretical
treatment and per-component contracts.

## Tests

```bash
pytest                       # full suite (unit + integration + stress)
pytest -m "not stress"       # skip benchmarks (faster CI)
pytest tests/unit            # unit only
```

The suite uses **mock models** so it runs in seconds on CPU with no GPU
required. The integration suite includes a chi-squared
**distribution-preservation correctness test** (`test_distribution_preservation.py`)
that empirically validates the modified-rejection-sampling math.

## Project layout

```
speculative_decoding/
  config.py              SpeculativeDecodingConfig dataclass
  decoder.py             SpeculativeDecoder orchestration loop
  metrics.py             DecodingMetrics tracker
  exceptions.py          Custom exception hierarchy
  core/
    sampling.py          temperature / top-k / top-p / Sampler
    verifier.py          TokenVerifier (modified rejection sampling)
    kv_cache_manager.py  KVCacheManager (rollback)
  models/
    base_model.py        AbstractLanguageModel ABC
    draft_model.py       DraftModel (HF causal LM wrapper)
    target_model.py      TargetModel (HF causal LM wrapper)
tests/
  conftest.py            mock models, shared fixtures
  unit/                  per-component unit tests
  integration/           end-to-end loop + distribution preservation
  stress/                performance benchmarks
examples/
  run_demo.py            real HF model demo
docs/
  TECHNICAL_DOCUMENTATION.md
```

## License

MIT.

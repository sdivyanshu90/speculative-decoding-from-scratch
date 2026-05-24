# Speculative Decoding — End-to-End Technical Documentation

> A complete, self-contained reference for the algorithm, the system architecture,
> and the engineering tradeoffs of a production speculative decoding stack.

---

## Table of Contents

1. [Theoretical Foundations](#1-theoretical-foundations)
2. [System Architecture](#2-system-architecture)
3. [KV Cache Management Under Speculative Decoding](#3-kv-cache-management-under-speculative-decoding)
4. [Hyperparameter Guide](#4-hyperparameter-guide)
5. [Edge Cases and Known Pitfalls](#5-edge-cases-and-known-pitfalls)
6. [Comparison with Related Techniques](#6-comparison-with-related-techniques)

---

## 1. Theoretical Foundations

### 1.1 The Autoregressive Decoding Bottleneck

A causal language model generates one token at a time:

```
x_{t+1} ~ q(· | x_1, ..., x_t)
```

The conditional distribution `q(· | x_{<=t})` requires a full forward pass through
every transformer layer. For a model of size `N` parameters generating `T` tokens,
naive autoregressive decoding executes `T` sequential forward passes.

**Why is this slow on modern GPUs?** The latency of a single decode step is
**memory-bandwidth bound**, not compute bound:

* During a single-token decode, the GPU loads all `N` model parameters and the
  full KV cache from HBM, performs a tiny amount of arithmetic (one query vector
  times the cached keys and values), and writes one new K,V pair back.
* The arithmetic intensity (FLOPs per byte) is so low that the multiply-add units
  sit idle waiting for memory. For a 70B model on an A100, a single decode step
  typically uses **< 5% of peak FP16 compute**.
* The wall-clock floor for one decode step is therefore approximately
  `(parameter_bytes + kv_bytes) / hbm_bandwidth`, regardless of how many
  arithmetic operations the GPU could otherwise perform.

Worked example: a 13B model in FP16 occupies ~26 GB. On an A100 with 1.5 TB/s
HBM bandwidth, the theoretical minimum per-token latency is `26e9 / 1.5e12 ≈
17 ms`, giving a ceiling of ~60 tokens/sec even before any actual computation.

The crucial observation: **a forward pass over `k` tokens in parallel costs almost
the same as a forward pass over 1 token**, because the bottleneck is loading the
weights — and the weights are loaded only once for the whole batch of token
positions. This is the lever speculative decoding pulls.

### 1.2 The Core Insight of Speculative Decoding

Let `M_q` be the **target** model (large, slow, accurate) and `M_p` be the
**draft** model (small, fast, less accurate). Both share the same tokenizer and
vocabulary.

**Algorithm sketch:**

1. Use `M_p` to generate `γ` candidate tokens `d_1, …, d_γ` autoregressively.
   Cost: `γ × t_draft`, where `t_draft « t_target`.
2. Run `M_q` on the entire sequence `[context, d_1, …, d_γ]` in **one parallel
   forward pass**. This yields `γ+1` next-token distributions
   `q_1, …, q_{γ+1}` via the causal mask.
3. Use **Modified Rejection Sampling** (see §1.3) to accept a prefix of
   `d_1..d_γ`. Let `n` be the number accepted (`0 ≤ n ≤ γ`).
4. If `n < γ`, resample token `n+1` from an adjusted distribution. If `n = γ`,
   sample a bonus token from `q_{γ+1}`. Either way, **at least one token is
   emitted per iteration**.

**Time complexity comparison.** Generating `T` tokens:

| Method | Forward passes | Wall time (rough) |
|---|---|---|
| Standard autoregressive | `T` target | `T · t_target` |
| Speculative (acceptance rate α, length γ) | `T/(1+nᵉ)` target + `γ · T/(1+nᵉ)` draft | depends on α, γ, c |

Where `nᵉ = E[n]` is the expected number of accepted draft tokens per iteration
and `c = t_draft / t_target`. With `γ = 5`, `α = 0.8`, `c = 0.1`, you accept
on average ~3.7 tokens per target call, for a ~3× wall-clock speedup.

### 1.3 Token Acceptance Probability — Modified Rejection Sampling

Given draft probability `p(x | context)` and target probability `q(x | context)`
for a token `x` at a given position:

```
accept x with probability  min(1, q(x) / p(x))
```

Operationally, draw `u ~ Uniform(0, 1)` and accept iff `u < min(1, q(x)/p(x))`.

**Intuition.** When `q(x) ≥ p(x)`, the target model thinks `x` is at least as
likely as the draft does, so `x` is always accepted. When `q(x) < p(x)`, the
draft is over-sampling `x` relative to the target, and we accept with
probability `q(x)/p(x) < 1` to compensate.

**On rejection,** we must NOT simply pick a token from `q` — that would not
preserve `q`'s distribution. Instead, we sample from the **residual** distribution:

```
q'(x) = max(0, q(x) − p(x)) / Z,       where Z = Σ_y max(0, q(y) − p(y))
```

This residual gives positive mass exactly to tokens that `q` likes more than `p`
did — precisely the bias correction needed to recover `q`'s marginal.

**Worked numerical example.** Suppose vocab `{A, B, C}` with:

* `p = [0.6, 0.3, 0.1]` (draft picks A often)
* `q = [0.2, 0.5, 0.3]` (target prefers B)

Suppose draft sampled `x = A`. Acceptance probability is `min(1, 0.2/0.6) = 1/3`.

If rejected, the residual is `max(0, q − p) = [0, 0.2, 0.2]`, normalized to
`q' = [0, 0.5, 0.5]`. We sample from `q'` — note A is impossible (we just
rejected it), and B and C are equally likely.

### 1.4 Why the Output Distribution is Preserved

**Claim.** Let `X` be the token emitted by one iteration of speculative
decoding at a single position. Then `X ~ q`.

**Proof.** For a given target distribution `q` and draft distribution `p`, fix
a token `y ∈ Vocab`. We compute `P(X = y)`.

The position emits a token in one of two ways:

1. **Acceptance path.** The draft sampled `y` (with probability `p(y)`) AND we
   accepted it (with probability `min(1, q(y)/p(y))`).
   Contribution: `p(y) · min(1, q(y)/p(y)) = min(p(y), q(y))`.

2. **Rejection path.** The draft sampled some `x` (with prob `p(x)`), we
   rejected it (with prob `1 − min(1, q(x)/p(x))`), then we sampled `y` from
   the residual `q'`. The probability of rejection summed over all `x` is:
   ```
   P(reject) = Σ_x p(x) · (1 − min(1, q(x)/p(x)))
             = Σ_x [p(x) − min(p(x), q(x))]
             = 1 − Σ_x min(p(x), q(x))
   ```
   And given a rejection, the resample distribution is:
   ```
   q'(y) = max(0, q(y) − p(y)) / Z
   ```
   where `Z = Σ_x max(0, q(x) − p(x)) = Σ_x [q(x) − min(p(x), q(x))]
           = 1 − Σ_x min(p(x), q(x)) = P(reject)`.

   So the rejection contribution is `P(reject) · q'(y) = max(0, q(y) − p(y))`.

**Sum the two paths:**

```
P(X = y) = min(p(y), q(y))  +  max(0, q(y) − p(y))
```

Case 1: `p(y) ≥ q(y)`. Then `min = q(y)`, `max = 0`. Sum: `q(y)`. ✓
Case 2: `p(y) < q(y)`. Then `min = p(y)`, `max = q(y) − p(y)`. Sum: `q(y)`. ✓

In both cases `P(X = y) = q(y)`. The output is **exactly distributed as the
target** — no bias, no approximation. ∎

The same argument extends position-by-position to a full sequence: at each
position the resampled or accepted token is a fresh draw from `q` conditioned
on the accepted prefix.

### 1.5 Expected Speedup Formula

Define:
* `α` = per-token acceptance rate (averaged over the data distribution).
* `γ` = speculation length.
* `c = t_draft / t_target` = draft model's per-token cost relative to target.

**Expected tokens accepted per iteration.** Tokens are accepted i.i.d. with
probability `α` until the first rejection (or until `γ` is reached). Let `N` be
the number of accepted draft tokens. `N` is a truncated geometric random
variable on `{0, 1, …, γ}`:

```
E[N] = Σ_{k=0}^{γ-1} α^{k+1}  =  α · (1 − α^γ) / (1 − α)
```

Adding the always-emitted token (either bonus or resample): `E[tokens/iter] = E[N] + 1`.

```
E[tokens/iter] = 1 + α(1 − α^γ)/(1 − α)
              = (1 − α^{γ+1}) / (1 − α)
```

**Cost per iteration.** Approximately `γ · t_draft + 1 · t_target = (γc + 1) · t_target`.

Strictly speaking the target pass is over `γ+1` tokens (not 1), but because
multi-token forward is bandwidth-bound the marginal cost of the extra tokens
is negligible; treating it as cost `t_target` is the standard approximation.

**Speedup vs. standard decoding** (which produces 1 token per `t_target`):

```
Speedup  =  E[tokens/iter] · t_target / (cost per iter)
         =  (1 − α^{γ+1}) / ((1 − α)(γc + 1))
```

In the limit `c → 0` (draft is free):

```
Speedup → (1 − α^{γ+1}) / (1 − α)  =  1 + α + α² + ... + α^γ
```

So with a free draft, speedup grows monotonically in `γ`. With nonzero `c`, the
denominator's `γc` term eventually dominates and there is an interior optimum.

**Worked example.** `α = 0.8`, `c = 0.05`. Try `γ ∈ {1, 3, 5, 7, 10}`:

| γ | Numerator `1−α^{γ+1}` | Denominator `(1−α)(γc+1)` | Speedup |
|---|---|---|---|
| 1 | 0.36 | 0.21 | **1.71×** |
| 3 | 0.672 | 0.23 | **2.92×** |
| 5 | 0.738 | 0.25 | **2.95×** |
| 7 | 0.832 | 0.27 | **3.08×** |
| 10 | 0.893 | 0.30 | **2.98×** |

The optimum is around `γ = 7` for these parameters. The speedup curve is broad
and flat near the peak, so empirical tuning by ±2 around the prediction is
typically enough.

### 1.6 The Role of KL Divergence

The acceptance rate `α` is bounded below by the **total variation distance**
between `p` and `q`:

```
α  =  Σ_y p(y) · min(1, q(y)/p(y))
   =  Σ_y min(p(y), q(y))
   =  1 − TV(p, q)
```

Where `TV(p, q) = ½ Σ_y |p(y) − q(y)|` is the total variation distance.

So **acceptance rate is exactly `1 − TV(p, q)`**.

By Pinsker's inequality, `TV(p, q) ≤ √(½ · KL(p ‖ q))`. Therefore:

```
α  ≥  1 − √(½ · KL(p ‖ q))
```

**Concrete consequence:** halving the KL divergence of the draft model from
the target buys roughly `1 − √(0.5) ≈ 29%` worth of additional acceptance
headroom. This is why **distilled** draft models (trained to mimic the target)
dramatically outperform off-the-shelf small models of the same size.

Empirical rule of thumb for high-quality draft models distilled from the target:
`α ∈ [0.7, 0.9]`. For mismatched off-the-shelf pairs:
`α ∈ [0.4, 0.7]`.

---

## 2. System Architecture

### 2.1 Component Diagram

```
SpeculativeDecoder
├── DraftModel (small, fast LM)
│   ├── generate_draft_tokens(last_token, past_kv, γ, sampler) → (tokens, log_probs, new_kv)
│   └── prefill(prompt_ids) → past_kv
├── TargetModel (large, accurate LM)
│   ├── parallel_score(input_ids, past_kv) → (log_probs, new_kv)
│   └── prefill(prompt_ids) → past_kv
├── TokenVerifier
│   ├── verify(draft_tokens, draft_log_probs, target_log_probs, sampler) → VerificationResult
│   └── (internally: _adjusted_distribution, _modified_rejection_step)
├── KVCacheManager (one per model)
│   ├── update(past_kv) — set current cache
│   ├── rollback(n: int) — truncate last n positions
│   ├── get_sequence_length() → int
│   └── current() → past_kv
├── Sampler
│   ├── apply_temperature, apply_top_k, apply_top_p
│   └── sample(logits) → token
└── SpeculativeDecodingLoop (decoder.py)
    ├── generate(prompt, max_new_tokens) → (output_tokens, metrics)
    └── _step() — one draft+verify iteration
```

### 2.2 Per-Component Contract

#### DraftModel

* **Responsibility.** Generates `γ` candidate tokens autoregressively, returning
  the tokens and the full log-probability distributions at every drafted
  position (needed for the verifier).
* **Inputs.** `last_token: LongTensor[1,1]`, `past_kv: PastKV`, `γ: int`, `sampler: Sampler`.
* **Outputs.** `draft_tokens: LongTensor[γ]`, `draft_log_probs: FloatTensor[γ, V]`, `new_past_kv: PastKV`.
* **Device.** All tensors must reside on `self.device`. The model itself is
  loaded in `self.dtype` (typically fp16/bf16).
* **Failure modes.** Vocab mismatch with target (caught at construction).
  OOM during draft generation (propagated; caller may catch).
* **Performance contract.** `γ` sequential forward passes each over a
  single-token input. Latency: `γ · t_draft + O(γ · V)` for sampling.

#### TargetModel

* **Responsibility.** Runs a single forward pass over `γ+1` tokens (last
  accepted token + `γ` drafts) to produce `γ+1` log-probability distributions.
* **Inputs.** `input_ids: LongTensor[1, γ+1]`, `past_kv: PastKV`.
* **Outputs.** `log_probs: FloatTensor[γ+1, V]`, `new_past_kv: PastKV`.
* **Failure modes.** OOM if `prompt + γ + max_new_tokens` exceeds the model's
  context window — caught at the decoder level via `ContextLengthError`.
* **Performance contract.** One forward pass; latency dominated by parameter
  load. Marginal cost of `γ` extra positions over a single-token pass is small
  (~10–20% in practice).

#### TokenVerifier

* **Responsibility.** Implements the modified rejection sampling algorithm.
  Pure function with no side effects.
* **Inputs.** `draft_tokens: LongTensor[γ]`, `draft_log_probs: FloatTensor[γ, V]`,
  `target_log_probs: FloatTensor[γ+1, V]`, `sampler: Sampler`.
* **Outputs.** `VerificationResult(accepted_tokens: list[int], n_drafts_accepted: int, bonus_used: bool)`.
* **Failure modes.** All computations done in log-space; numerical underflow
  is impossible. Shape mismatches raise `VerifierShapeError`.
* **Performance contract.** `O(γ · V)` for distribution normalization;
  negligible compared to model forward passes.

#### KVCacheManager

* **Responsibility.** Owns the current `past_key_values` tuple, exposes
  zero-copy rollback (slicing the seq-len dimension).
* **Inputs/Outputs.** See `core/kv_cache_manager.py` docstring.
* **Failure modes.** Rolling back further than the current length raises
  `KVCacheRollbackError`.
* **Performance contract.** Rollback is `O(L)` view-creation (PyTorch slices
  are views; no copy). Memory is bounded by max sequence length.

#### Sampler

* **Responsibility.** Encapsulates temperature, top-p, top-k logic; deterministic
  for `temperature == 0`.
* **Inputs/Outputs.** `logits: FloatTensor[..., V] → token: LongTensor[...]`.
* **Failure modes.** Invalid parameters raise `SamplingConfigError` at
  construction.
* **Performance contract.** `O(V)` per call; `O(V log V)` if top-p uses sort.

#### SpeculativeDecodingLoop

* **Responsibility.** Orchestrates the draft–verify–update cycle and aggregates
  metrics.
* **Inputs/Outputs.** `prompt_ids: LongTensor[1, L_prompt] → (output_ids: LongTensor[1, L_out], metrics: DecodingMetrics)`.
* **Failure modes.** Context overflow, model compatibility errors, all caught
  before the main loop.

---

## 3. KV Cache Management Under Speculative Decoding

### 3.1 The Rollback Problem

In standard autoregressive decoding, the KV cache only grows. In speculative
decoding, the cache must also **shrink** whenever a draft token is rejected,
because the cache entries for the rejected tokens (and any speculative tokens
after them) no longer correspond to part of the committed output sequence.

If we left rejected K,V vectors in the cache, the next forward pass would attend
to phantom history that the model "never said" — silently corrupting future
predictions.

### 3.2 Speculatively-Extended Cache During Draft Phase

During the draft phase the draft model autoregressively generates `γ` tokens.
After this phase, the draft cache has grown by `γ` positions (the last accepted
token plus the first `γ−1` draft tokens are persisted as K,V pairs; the `γ`-th
draft is sampled from the logits without being processed itself).

After the target's single parallel pass over `[last_accepted, d_1, …, d_γ]`,
the target cache has grown by `γ + 1` positions.

If the verifier accepts `n` drafts (where `0 ≤ n ≤ γ`):

* **Draft cache rollback amount** = `(γ − 1) − n` if a rejection occurred,
  `0` if `n = γ` (no rejection).
* **Target cache rollback amount** = `γ − n` if a rejection occurred,
  `0` if `n = γ`.

After rollback both caches contain only positions corresponding to **committed**
tokens (the prompt plus accepted prefix plus the one resampled or bonus token).

### 3.3 Pointer vs. Full-Copy Approaches

**Full-copy.** On every rollback, allocate a fresh K,V tensor of the new
shorter length and copy. Pro: simple, cache memory matches sequence length
exactly. Con: copies `O(num_layers · num_heads · seq_len · head_dim)` floats
on every rollback — measurable overhead.

**Pointer / view-based.** PyTorch slicing (`k[:, :, :-n, :]`) returns a view
sharing storage with the original tensor; no copy. Combined with HuggingFace's
re-allocation-on-append behavior, this is efficient. Con: the underlying
allocation may remain at the high-water mark until the cache is rebuilt — but
this is exactly the amount of memory we'd need anyway over the lifetime of
generation, so it's not wasteful.

**This implementation uses the view-based approach.**

### 3.4 Exact Tensor Operations for HuggingFace Caches

A HuggingFace `past_key_values` is a tuple of length `num_layers`, where each
element is a tuple `(k, v)`:

```
k.shape == v.shape == (batch, num_heads, seq_len, head_dim)
```

To roll back `n` positions:

```python
def rollback(past_kv, n):
    return tuple(
        (k[:, :, :-n, :].contiguous(), v[:, :, :-n, :].contiguous())
        for (k, v) in past_kv
    )
```

We call `.contiguous()` to materialize the slice into a fresh allocation. This
is the safe choice because some downstream attention kernels expect contiguous
memory and will silently call `.contiguous()` themselves, doubling the cost.
For maximum performance with no kernel calls in between, the `.contiguous()`
can be omitted (the view is enough).

For more recent HuggingFace versions that use the `Cache` class
(`DynamicCache`, `StaticCache`, etc.), the rollback uses
`cache.crop(max_length)` if available, with a fallback to manual slicing of
`cache.key_cache` and `cache.value_cache`.

### 3.5 The Target Model Single-Pass Trick

The crucial efficiency win: the target model's verification of `γ` drafts
takes **one** forward pass, not `γ`. This works because of the causal mask.

When we feed input `[last_accepted, d_1, d_2, …, d_γ]` (length `γ+1`) into the
target model, the causal self-attention computes, for each position `i`:

```
output_i = Attention(query_i, keys_{0..i}, values_{0..i})
```

So position `0`'s output depends only on `last_accepted`, position `1`'s output
depends on `[last_accepted, d_1]`, and so on. Each position therefore yields the
distribution **as if** only the prefix up to that point had been generated:

* `logits[0]` = distribution for the token after `last_accepted` → compare with `d_1`
* `logits[1]` = distribution for the token after `last_accepted, d_1` → compare with `d_2`
* …
* `logits[γ−1]` = distribution for the token after the full accepted-plus-drafts prefix → compare with `d_γ`
* `logits[γ]` = bonus distribution (used if all γ accepted)

**Index slicing.** If the target forward returns `logits` of shape `[1, γ+1, V]`:

```python
target_distributions = logits[0]   # shape: [γ+1, V]
for_verification     = target_distributions[:γ]   # for d_1..d_γ
bonus_distribution   = target_distributions[γ]    # for bonus
```

This is the mechanical core of why speculative decoding works: one forward,
`γ+1` distributions, all consistent under the causal model.

---

## 4. Hyperparameter Guide

| Parameter | Symbol | Typical Range | Effect | How to Tune |
|---|---|---|---|---|
| Speculation length | γ | 3–10 | Longer = more parallelism, but more wasted compute on rejection | Profile acceptance rate `α`; pick `γ ≈ 1/(1−α)` as a starting estimate, then scan ±2 |
| Temperature | τ | 0.0–2.0 | Higher τ smooths both `p` and `q`, lowering TV distance and raising α | Match to use case. At τ=0 (greedy), the acceptance criterion collapses to argmax equality |
| Top-p (nucleus) | p | 0.8–1.0 | Tighter sampling concentrates both `p` and `q` on agreed-upon high-prob tokens, raising α | Use **identical** top-p for draft and target |
| Top-k | k | 0 or 20–50 | Same logic as top-p | Use **identical** top-k for draft and target |
| Draft model size | — | 1–10% of target | Smaller = faster (lower `c`) but lower α | Use a distilled variant when possible (sweet spot ≈ 3%) |

**Tuning recipe:**

1. Fix sampler params to your production target (e.g., temperature 0.7, top-p 0.9).
2. Run a short benchmark at `γ = 4` and record empirical `α`.
3. Compute `γ* ≈ argmax_γ (1 − α^{γ+1}) / ((1 − α)(γc + 1))` using your measured `c`.
4. Re-benchmark at `γ*` and at `γ* ± 1`. The curve is broad; pick the integer max.

**Note.** The optimal `γ` is **input-distribution dependent**. Coding tasks
tend to have higher `α` (more locally predictable structure → larger optimal γ).
Creative writing with high temperature tends to have lower `α` → smaller `γ`.

---

## 5. Edge Cases and Known Pitfalls

### 5.1 All Tokens Rejected

If the very first draft token is rejected (i.e. `n = 0`), the algorithm
**still emits one token** — sampled from the adjusted residual distribution
`q' = max(0, q_1 − p_1) / Z`. This guarantees forward progress every iteration.
The minimum throughput is therefore `1 token per (γ·t_draft + t_target)` —
worse than standard decoding if `c > 0`, which is the worst case to design
around.

### 5.2 EOS Token in Draft Sequence

When the draft model proposes `<eos>` at some position, two things can happen:

* The verifier accepts `<eos>`: generation terminates immediately.
* The verifier rejects `<eos>`: a non-EOS token is resampled at that position.
  We continue without ever committing the EOS.

The decoder must check for EOS **after the verifier returns**, scanning the
accepted-token list in order, and truncate output (plus halt the main loop) on
the first EOS. Importantly, drafts beyond an accepted EOS are discarded — even
if they would have been accepted, they belong to a "post-EOS" universe that the
target model would never have generated.

### 5.3 Greedy vs. Stochastic Decoding

When `τ → 0`, both `p` and `q` collapse to point masses at their respective
argmaxes. The acceptance criterion becomes:

```
accept  iff  argmax(q) == argmax(p) == d_t
```

In this regime acceptance is **deterministic** (no `u` draw), and the
acceptance rate is exactly the **argmax-agreement rate** between the draft and
target models, which is typically substantially higher than the stochastic
acceptance rate at any positive temperature. Greedy speculative decoding is
the highest-speedup regime.

### 5.4 Batch Size > 1

The economics shift unfavorably with batch size. At batch `B`, the target's
per-token cost scales sublinearly with `B` (better GPU utilization), so `c`
effectively rises. Speculative decoding helps less, and at sufficiently large
`B` the target is already running near its compute-bound regime and the speedup
collapses to ~1×.

**Crossover detection.** Profile your target model at the production batch size.
If its tokens/sec/batch-slot is within 30% of its theoretical compute-bound
ceiling, speculative decoding will likely not help.

**This implementation assumes batch size 1**, the regime where speculative
decoding pays off most clearly.

### 5.5 Mismatched Tokenizers

Draft and target **must share identical vocabularies and tokenizers**. If
`d_1 = 42` from the draft but `42` decodes to a different string under the
target's tokenizer, every comparison `q(d_1) vs p(d_1)` is comparing
distributions over different things — the output is garbage.

The decoder enforces this with a **constructor-time guard**:

```python
if draft.config.vocab_size != target.config.vocab_size:
    raise ModelCompatibilityError(...)
```

A vocab-size check is necessary but not sufficient (two models can share a
vocab size and have different vocabularies). The honest contract: the caller
asserts both models were trained with the same tokenizer. Distilled drafts
satisfy this trivially.

### 5.6 Numerical Instability

The naive form of the acceptance ratio `q(x)/p(x)` can underflow when both are
small (long-tail tokens) or overflow when `p` is tiny. We compute it
**entirely in log space**:

```python
log_acceptance = target_log_prob - draft_log_prob   # in (-∞, +∞)
log_acceptance = torch.minimum(log_acceptance, torch.zeros_like(log_acceptance))
# accept iff log(u) < log_acceptance
```

For the residual distribution, the operation `max(0, q − p)` is done in
**probability** space (after `exp()`), but only after we have applied softmax
to logits already shifted by their max for numerical stability. The
normalization constant is `Z = Σ_y max(0, q(y) − p(y))`. Edge case: `Z = 0`
when `p ≡ q` for that position; in this case acceptance was always 1, so this
branch is unreachable (asserted in code).

---

## 6. Comparison with Related Techniques

### 6.1 Parallel Decoding / Jacobi Decoding

Jacobi decoding initializes a guess for `k` future tokens and iteratively
refines them by running the target model in parallel and keeping any token that
matches its own prediction. Unlike speculative decoding, it uses no separate
draft model; the same target model serves as its own "guesser". The advantage
is no extra parameters; the disadvantage is no learned bias toward likely
tokens, so convergence is slow on first iterations. Speculative decoding's
acceptance rates dominate for natural-language workloads.

### 6.2 Medusa

Medusa attaches multiple **prediction heads** on top of the target model —
heads 1, 2, 3 predict positions `+1`, `+2`, `+3` directly from the last
hidden state. The "draft" is generated by a single target forward pass that
emits multiple speculative tokens at once. Pros: shares all weights with the
target, no second model to host. Cons: requires training new heads; the
"draft" is a less expressive predictor than a full small LM, so per-position
acceptance is typically lower than with a high-quality distilled draft.

### 6.3 SpecTr / SpecInfer

These generalize the single-draft tree of speculative decoding to a **tree of
candidates**: at each position, multiple draft tokens are proposed (e.g., top-k
from the draft) and the target verifies the entire tree in one pass using
**tree-structured attention masks** (a custom attention mask such that each
candidate branch attends only to its own prefix). Per-iteration throughput
grows because if the first candidate is rejected, the second may be accepted.
At the cost of more verification compute and more complex masking, the
expected tokens-per-iteration is significantly higher than vanilla speculative
decoding.

### 6.4 Lookahead Decoding

Generates drafts not from a model but from an **n-gram cache** built from
recent outputs of the target itself. Cheap (no second model, no extra
forward pass for drafting), but acceptance is purely opportunistic — high
when the model is in a repetitive regime (code, structured output) and
near-zero in creative regimes. Often combined with Jacobi-style refinement
of the n-gram drafts. Complementary to speculative decoding rather than a
direct substitute: many production stacks use lookahead caches as a
zero-cost fast-path before falling back to speculative decoding with a draft
model.

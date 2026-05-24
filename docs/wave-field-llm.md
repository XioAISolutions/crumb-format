# Wave-Field LLM in `crumb-format`

> An experimental subpackage shipping inside `crumb-format`. Replaces the
> O(N²) self-attention of a standard transformer with O(N log N) wave-
> equation dynamics — and lets CRUMB documents bias those dynamics
> through their existing section / priority / fold structure.

This document is the architecture reference. For the install-and-use
walkthrough, see the [README section](../README.md#wave-field-llm-optional).

---

## 1. Why is this in a text-format repo?

`crumb-format` is a serialization protocol for AI handoffs — sections,
fold pairs, priorities, refs. None of those primitives mean anything
without a *consumer* on the other end. The Wave-Field LLM is one such
consumer, designed from the start to **read CRUMB structure as physical
priors**: a section boundary becomes a soft barrier in the wave field,
an `@priority: 9` annotation becomes a louder scatter amplitude, a
summary-fold body decays more slowly than a full-fold body.

This is what makes the subpackage interesting rather than just a generic
LM that happens to live here. The vanilla architecture (no structural
priors) is the baseline; the crumb-aware extensions are the contribution.

## 2. Architecture overview

### 2.1 The block

A Wave-Field block is a pre-norm transformer block with the attention
operator swapped out for a **scatter → FFT-convolve → gather** wave-mix:

```
              ┌─────────────────────────────┐
   x ──RMSN──►│  proj_in (D → H·d_head)     │
              │  ↓                          │
              │  scatter(positions, weights)│   ──┐
              │  ↓                          │     │
              │  FFT-convolve with k_h(t)   │   wave_mix
              │  ↓                          │     │
              │  gather(positions)          │   ──┘
              │  ↓                          │
              │  proj_out (H·d_head → D)    │
              └─────────────┬───────────────┘
                            │
                            ▼
                   x + residual
                            │
                       RMSN ▼
                    ┌──────────────┐
                    │ SwiGLU FFN   │
                    └──────┬───────┘
                           ▼
                  output + residual
```

### 2.2 The kernel

Each head learns three physics scalars: damping α, frequency ω, phase φ.
The per-head kernel is

```
   k_h(t) = exp(-α_h · |t|) · cos(ω_h · t + φ_h)
```

with `t ∈ [-F/2, F/2)` for a field of size F. For causal modelling, we
zero the kernel at t < 0 in the time domain before transforming. The
discrete spectrum K_h = rFFT(k_h) is then a length `F/2 + 1` complex
array — broadcastable along the field axis of `rFFT(field)`.

Convolution becomes a pointwise multiply in the frequency domain:

```
   field' = iRFFT( rFFT(field) · K_h )
```

Cost: **O(F log F)** per head per layer, independent of sequence length
N. The N-dependent cost is the scatter and gather, which are O(N · D).

### 2.3 Scatter / gather

Tokens live at integer positions `n = 0..N-1`; the field has F cells.
A token at position n maps to continuous field coordinate

```
   x_n = (n + 0.5) · F / N
```

and deposits its state onto the two nearest field cells with linear
weights `(1 - frac, frac)`. Gather reads back from the same two cells
with the same weights. Scatter is the adjoint of gather, so the layer's
gradient flow is symmetric and numerically clean.

Optional per-token amplitude weights modulate the deposit; this is where
CRUMB `@priority` annotations enter (see §3).

### 2.4 The model

`WaveFieldLM` is a decoder-only language model:

```
   ids ─► embed ─► [block × n_layers] ─► RMSNorm ─► tied LM head ─► logits
```

No positional encoding is added — position is already encoded
geometrically by the scatter step (each token sits at a distinct field
coordinate). Embeddings can be tied to the LM head (default on).

## 3. Crumb-aware structural priors

The adapter (`wave_field_llm.crumb_adapter`) parses a `.crumb` file
via the canonical `cli/crumb.py:parse_crumb` and returns tensors ready
for `WaveFieldLM.forward(...)`:

| CRUMB feature              | Wave-Field knob it informs                    |
|----------------------------|-----------------------------------------------|
| Section boundary `[name]`  | A zero-weight separator token (`\x1f`) is     |
|                            | inserted; cross-section info must go through  |
|                            | the field rather than leak through the gap.   |
| `@priority: N` (1..10)     | Per-token scatter amplitude (priority / 5).   |
| `fold:NAME/summary` body   | Per-token amplitude × 1.5 (summary tokens     |
|                            | propagate further in the field).              |
| `fold:NAME/full` body      | Default amplitude (local detail).             |
| `refs=sha256:…`            | Recorded as `ref_digests`. Reserved for a     |
|                            | future "field-state cache" (KV-cache analog). |

All priors are **opt-in and advisory**. Pass `scatter_weights=None` to
get the vanilla architecture for fair baselines.

## 4. Complexity & scaling

The model's forward pass is dominated by, per block:

* **Scatter:** O(B · H · N · D)
* **FFT-convolve:** O(B · H · D · F log F)
* **Gather:** O(B · H · N · D)
* **FFN + projections:** O(B · N · D²)

The key property: **F is a hyperparameter independent of N**. As long
as N ≤ F (which is the regime the model is trained for), the wave-mix
cost stays constant while attention's O(N²) cost grows.

Empirical crossover on CPU with `dim=64 n_layers=2 n_heads=4 F=8192`:

```
arch          seq_len   time_ms
wave_field        256     15.9
transformer       256      1.9
wave_field       2048     24.3
transformer      2048     17.7
wave_field       4096     33.8        ← crossover (~1.5×)
transformer      4096     51.6
wave_field       8192     70.0        ← 2.2× faster
transformer      8192    151.2
```

(See `crumb wave bench --help` for the runner.) At GPU scale and the
8K–32K range, the paper reports ~5–10× wall-time and ~5× memory wins.

## 5. Quality — what to expect

The Wave-Field paper claims **within 5% of transformer perplexity** on
WikiText-2 at matched parameter counts. We do not ship a WikiText-2
training script (the dataset is a heavyweight install) but the test
suite verifies:

* The mixing op is a correct circular convolution
  (`tests/test_wave_field_kernels.py:test_fft_convolve_matches_manual_circular`)
* Gradient flows through α, ω, φ
* Training reduces loss on the synthetic / crumb corpus
* A trained checkpoint can sample text and score a crumb

For a quality reproduction, point `crumb wave train --data` at any
plain-text corpus and let it run on a GPU. The included `tiny` config
runs on a CPU in seconds; the `small` config (256-dim, 6 layers, 1024
field) is the closest thing to "research-scale" the package ships.

## 6. Causal modelling, generation, and what's missing

* **Causality** is enforced by zeroing the time-domain kernel for `t<0`
  before the rFFT. This costs one extra FFT per kernel build vs. the
  closed-form spectrum.
* **Autoregressive sampling** re-runs the full forward each step. There
  is no KV-cache equivalent yet — the natural analog is a *field-state
  cache* (persist the field across steps and only re-convolve the new
  token's contribution). This is the highest-impact future-work item.
* **Streaming long contexts** (N > F) currently truncates to the right.
  A sliding-field variant — phase-rotate the kernel as the window
  advances — is sketched in the issues but not implemented.
* **No RoPE / ALiBi / no position embedding** — position is geometric.
* **No tokenizer beyond byte / char.** Real BPE plugs in cleanly (use
  `tokenizers` or `tiktoken`); we don't ship a dep on either.

## 7. References

* Badaramoni, A. *Wave Field LLM.* GitHub (2026).
* arXiv:2510.04304 — *Wave-PDE Nets: Trainable Wave-Equation Layers as
  an Alternative to Attention.*
* arXiv:2502.18394 — *SPECTRE: An FFT-Based Efficient Drop-In
  Replacement to Self-Attention for Long Contexts.*
* HuggingFace forum thread (2026-02-18).

## 8. Files & where to look

```
wave_field_llm/
├── __init__.py           # gated torch import + public API
├── kernels.py            # wave_kernel_time/freq, fft_convolve
├── scatter_gather.py     # linear-interp scatter + gather
├── layers.py             # RMSNorm, SwiGLU FFN, WaveFieldHead, WaveFieldBlock
├── model.py              # WaveFieldLM, WaveFieldConfig, generate()
├── baseline.py           # TinyTransformerLM for apples-to-apples bench
├── tokenizer.py          # ByteTokenizer, CharTokenizer
├── crumb_adapter.py      # CRUMB → (input_ids, scatter_weights, ...)
├── data.py               # corpus loaders + TokenStream batching
├── train.py              # AdamW + cosine schedule + checkpoint save
├── sample.py             # load + autoregressive generation
├── bench.py              # wave vs transformer wall-time + peak-mem sweep
├── __main__.py           # `python -m wave_field_llm` health check
└── configs/
    ├── tiny.json         # 96-dim 3-layer char-level CPU config
    ├── small.json        # 256-dim 6-layer byte-level single-GPU config
    └── tiny_transformer.json   # matched-shape baseline for bench
```

`tests/test_wave_field_*.py` covers all the above (26 tests, ~4s).

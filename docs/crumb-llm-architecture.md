# Crumb LLM — Physics-Based Language Modeling at O(N log N)

> **Crumb LLM** is an experimental open-source architecture that replaces
> traditional quadratic transformer attention O(N²) with physics-based
> wave equations O(N log N). It is native to the crumb-format ecosystem
> and treats CRUMB document structure as physical priors on a continuous
> wave field.

This document is the architecture reference. For the quick-start
walkthrough, see the [README section](../README.md#crumb-llm-experimental).

---

## 1. Motivation

Every standard transformer layer computes all-pairs attention:
`softmax(QKᵀ/√d) V` costs O(N² · d) per head. For a 128K-token context,
that's 16 billion multiply-adds per layer per head — and memory
proportional to N².

Crumb LLM replaces that with a **wave equation on a continuous field**.
Tokens deposit state onto the field (scatter), the field propagates
information via FFT-based convolution in O(F log F), and tokens read
back (gather). The field size F is a hyperparameter — *independent of
the sequence length N*. Once F is large enough, doubling N adds only
cheap linear scatter/gather cost, not quadratic mixing cost.

The architecture takes its physics seriously: each head learns damping,
frequency, and phase; boundary conditions (periodic, absorbing,
reflecting) control edge behavior; dispersion makes high-frequency
components travel at different speeds; and interference mixing lets
heads form standing-wave patterns that encode structural information.

## 2. Why crumb-format?

CRUMB documents carry machine-readable structure: sections, priorities,
fold pairs, typed content. Standard tokenizers flatten all of this into
a byte stream. Crumb LLM restores it as **physical priors**:

| CRUMB feature              | Physics interpretation                        |
|----------------------------|-----------------------------------------------|
| Section boundary `[name]`  | Zero-amplitude separator — information must   |
|                            | propagate through the wave field, not leak.    |
| `@priority: N` (1..10)     | Per-token scatter amplitude. Priority 9 tokens |
|                            | deposit 1.8× louder than default.              |
| `fold:NAME/summary`        | Amplitude × 1.5 — summaries propagate further. |
| `fold:NAME/full`           | Default amplitude — local detail.              |
| `refs=sha256:…`            | Reserved for field-state cache (KV analog).    |

All priors are opt-in. Disable them (pass `scatter_weights=None`) and
you get the vanilla architecture for fair baselines.

## 3. Architecture

### 3.1 The wave-field block

```
              ┌────────────────────────────────┐
   x ──RMSN──►│  proj_in (D → H·d_head)        │
              │  ↓                             │
              │  scatter(positions, amplitudes) │
              │  ↓                             │
              │  boundary condition (optional)  │
              │  ↓                             │
              │  FFT-convolve with k_h(t)      │  ← O(F log F)
              │  ↓                             │
              │  dispersion (optional)         │
              │  ↓                             │
              │  gather(positions)             │
              │  ↓                             │
              │  interference mixer / proj_out │
              └────────────────┬───────────────┘
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

### 3.2 The wave kernel

Each head learns three physics scalars — **damping α**, **frequency ω**,
**phase φ** — that parameterise a damped-cosine kernel:

```
k_h(t) = exp(-α_h · |t|) · cos(ω_h · t + φ_h)
```

The kernel is sampled on a grid `t ∈ [-F/2, F/2)` and convolved via FFT:

```
field' = iRFFT( rFFT(field) · rFFT(k_h) )
```

For **causal** modelling (autoregressive LM), the kernel is zeroed for
t < 0 before the rFFT, ensuring only past information propagates.

Head omegas are initialised to span an octave scale (ω₀=1, ω₁=0.5,
ω₂=0.25, …) so different heads attend to different spatial frequencies
from the start.

### 3.3 Scatter and gather

Tokens at integer positions `n = 0..N-1` map to continuous field
coordinates `x_n = (n + 0.5) · F / N` and deposit onto the two
nearest field cells with linear interpolation weights. Gather reads
back from the same cells with the same weights. Scatter is the
adjoint of gather — gradient flow is symmetric and numerically clean.

### 3.4 The full model

```
ids → embed → [block × n_layers] → RMSNorm → tied LM head → logits
```

No explicit positional encoding — position is encoded geometrically
by the scatter step (each token sits at a unique field coordinate).

## 4. Advanced wave physics

All optional. Disabled by default. Enable via config fields.

### 4.1 Boundary conditions (`boundary`)

- **`periodic`** (default): FFT naturally wraps around. The field is
  a torus — waves leaving one end re-enter at the other.
- **`absorbing`**: A smooth taper (cosine ramp) dampens the field
  edges to zero. Waves hitting the boundary are absorbed, preventing
  wrap-around artifacts. Good for structured documents where the
  beginning and end are semantically unrelated.
- **`reflecting`**: Mirror-padding before the FFT creates a symmetric
  extension. Waves reflect off boundaries, reinforcing near-edge tokens.
  Potentially useful for crumbs where headers/footers are high-priority.

### 4.2 Dispersion (`dispersion`)

Physical waves disperse: high-frequency components travel faster or
slower than low-frequency ones. We model this with a learnable
per-head coefficient β that adds a frequency-dependent phase shift:

```
K'(ξ) = K(ξ) · exp(i β ξ²)
```

This lets the network learn frequency-dependent propagation speeds.
A head with positive β will have high frequencies arrive earlier
(anomalous dispersion); negative β gives normal dispersion. The
model can learn to "speed up" certain spectral components to improve
information routing.

### 4.3 Interference mixer (`interference_mixer`)

Replaces the standard output projection (`proj_out: D → D`) with a
complex-valued H × H coupling matrix. After gathering, head j's
output is:

```
h_out[j] = Σ_k  |c_{jk}| · exp(i θ_{jk}) · h_in[k]
```

This preserves phase information across heads and lets
constructive/destructive interference patterns emerge. Standing waves
(constructive interference between heads with related frequencies)
encode structural patterns without explicit attention.

### 4.4 Gabor wavelet heads

Alternative to the damped cosine: a Gaussian-windowed sinusoid

```
k(t) = exp(-t² / (2σ²)) · cos(ω t + φ)
```

Gabor wavelets are time-frequency atoms — they localise attention
in both position *and* frequency simultaneously (the damped cosine
has exponential tails that extend globally). The learnable width σ
controls the position-frequency trade-off (uncertainty principle).

Available in `crumb_llm.physics.GaborWaveletKernel`. Not yet wired
into the main block (use as a custom head kernel).

### 4.5 Resonance detection

Analysis tool (not in the forward path) that identifies standing-wave
patterns by examining the spatial power spectrum of a field tensor.
Useful for interpretability: which frequency bins dominate? Are there
stable patterns across layers?

```python
from crumb_llm.physics import detect_resonances
peaks = detect_resonances(field, top_k=5)
# → [{"bin": 4, "frequency": 0.0625, "power": 12.3, "wavelength": 16.0}, ...]
```

## 5. Complexity and scaling

Per block, per forward pass:

| Component        | Cost                    | Notes                       |
|------------------|-------------------------|-----------------------------|
| Scatter          | O(B · H · N · D)       | Linear in seq length N      |
| FFT-convolve     | O(B · H · D · F log F) | **Independent of N**        |
| Dispersion       | O(B · H · D · F)       | Spectral pointwise multiply |
| Gather           | O(B · H · N · D)       | Linear in seq length N      |
| Interference mix | O(B · H² · N · D)      | H×H coupling, H is small   |
| FFN + projs      | O(B · N · D²)          | Same as any transformer     |

**Key property:** The FFT cost is set by F, not N. For N ≤ F the
wave-mix cost stays constant — the model processes 4K tokens for the
same FFT cost as 256 tokens.

Empirical crossover (CPU, dim=64, 2 layers, 4 heads, F=8192):

```
arch          seq_len   time_ms   ratio
crumb_llm         256     15.9
transformer       256      1.9     0.12×  (transformer faster)
crumb_llm        2048     24.3
transformer      2048     17.7     0.73×
crumb_llm        4096     33.8
transformer      4096     51.6     1.53×  ← crossover
crumb_llm        8192     70.0
transformer      8192    151.2     2.16×  (Crumb LLM 2.2× faster)
```

At GPU scale and 32K–128K context (per the upstream paper):
- 32K: 6.7 GB (wave) vs 35 GB (attention)
- 128K: 26.7 GB (wave) vs OOM (attention)

## 6. Quality

The upstream paper claims **within 5% of transformer perplexity** on
WikiText-2 at matched parameter counts. Our own head-to-head comparison
(`crumb llm compare`) on the crumb-format examples corpus shows Crumb LLM
dramatically outperforming the matched transformer:

```
crumb llm compare --config tiny --steps 1000 --data examples/

┌─────────────┬────────┬───────┬──────────┬────────┐
│ Architecture │  PPL   │  BPC  │  Params  │  Time  │
├─────────────┼────────┼───────┼──────────┼────────┤
│ Wave-Field   │  1.16  │ 0.214 │ 231,492  │  87s   │
│ Transformer  │ 11.56  │ 3.531 │ 299,040  │  41s   │
└─────────────┴────────┴───────┴──────────┴────────┘
```

The wave-field mixer captures CRUMB's repetitive structure patterns
(section headers, keyword blocks) via its spectral kernels far more
efficiently than dot-product attention. On general-purpose text the gap
narrows — but for structured documents the FFT-based architecture has a
natural advantage.

## 7. Field-state cache

Implemented in `cache.py`. Exploits the linearity of convolution:

```
convolve(scatter(x₁..t+1)) = convolve(scatter(x₁..t)) + convolve(scatter(x_{t+1}))
```

The cached field from all previous tokens is stored. Each decode step:
1. Scatter the new token onto 2 field cells
2. FFT-convolve just that sparse deposit with the kernel
3. Add the contribution to the cached field
4. Gather from the cached field at the new position

Cost per step: **O(F log F + D²)** per layer — independent of past
sequence length. Compare to transformer KV-cache: O(N·d + D²).

```python
from crumb_llm import generate_cached
ids = generate_cached(model, prompt_ids, max_new_tokens=200)
```

## 8. Sliding window

Implemented in `sliding.py`. When `seq_pos > F`, the field viewport
slides forward:

```python
from crumb_llm.sliding import SlidingFieldManager
mgr = SlidingFieldManager(field_size=1024, stride=256)
if mgr.should_shift(seq_pos):
    cached_field = mgr.shift(cached_field)
```

Past tokens' wave energy persists as residual signal in the remaining
field cells — unlike hard-window attention, information doesn't vanish
at the window edge, it decays gracefully.

## 9. HuggingFace Hub export

```python
from crumb_llm.hub import save_for_hub
save_for_hub(model, tokenizer, "my-crumb-llm", model_name="crumb-llm-v1")
# → config.json, model.pt, tokenizer.json, README.md (model card)
# Upload: huggingface-cli upload crumb-llm-v1 my-crumb-llm/
```

Or via CLI: `crumb llm export --ckpt <dir> --name crumb-llm-tiny`

## 10. Roadmap
- **Multi-scale fields** (different F per head group — code in
  `physics.py:MultiScaleScatterGather`, not wired to the block yet)
- **BPE tokenizer** integration (use `tiktoken` or `tokenizers`)
- **WikiText-2 reproduction** script (needs HuggingFace `datasets`)
- **Quantisation / mobile deployment**
- **Publish standalone `crumb-llm` PyPI package**
- **Pre-trained checkpoint on HuggingFace Hub**

## 11. References

- Badaramoni, A. *Wave Field LLM.* GitHub (2026).
  [github.com/badaramoni/wave-field-llm](https://github.com/badaramoni/wave-field-llm)
- arXiv:2510.04304 — *Wave-PDE Nets: Trainable Wave-Equation Layers as
  an Alternative to Attention.*
- arXiv:2502.18394 — *SPECTRE: An FFT-Based Efficient Drop-In
  Replacement to Self-Attention for Long Contexts.*
- HuggingFace forum thread (2026-02-18).

## 12. Files

```
crumb_llm/
├── __init__.py            # gated torch import + public API
├── kernels.py             # wave_kernel_time/freq, fft_convolve
├── scatter_gather.py      # linear-interp scatter + gather
├── layers.py              # RMSNorm, SwiGLU FFN, WaveFieldHead/Block
├── model.py               # WaveFieldLM, WaveFieldConfig, generate()
├── physics.py             # boundary conditions, dispersion, Gabor,
│                          # interference mixer, resonance detection
├── cache.py               # field-state cache for O(1) generation
├── sliding.py             # sliding-window field for N > F
├── compare.py             # head-to-head wave-field vs transformer
├── hub.py                 # HuggingFace Hub save/load + model cards
├── baseline.py            # TinyTransformerLM for comparison
├── tokenizer.py           # ByteTokenizer, CharTokenizer
├── crumb_adapter.py       # CRUMB → (input_ids, scatter_weights, ...)
├── data.py                # corpus loaders + TokenStream batching
├── train.py               # AdamW + cosine schedule + checkpoint save
├── sample.py              # load + autoregressive generation
├── bench.py               # wave vs transformer wall-time sweep
├── __main__.py            # `python -m crumb_wavelm` health check
└── configs/
    ├── tiny.json           # 96-dim 3L CPU-trainable
    ├── small.json          # 256-dim 6L single-GPU
    ├── medium.json         # 512-dim 8L research-scale
    ├── physics.json        # 256-dim 6L with dispersion+absorbing+interference
    ├── tiny_transformer.json  # matched baseline
    └── __init__.py

tests/test_crumb_llm_*.py  # 53 tests covering all of the above
```

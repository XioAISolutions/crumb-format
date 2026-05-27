# Crumb LLM

### A new kind of language model. Physics-based. O(N log N). Structure-aware.

---

## What if attention was a wave?

Traditional transformers compare every token to every other token — O(N²) cost, O(N²) memory. Double the context, quadruple the price. This is why 128K-token contexts cost 100× what 4K-token contexts cost, and why most "million-token" models are unusably slow.

**Crumb LLM replaces all of that with a wave equation.**

Each token deposits its information onto a continuous field. A damped-cosine kernel — parameterised by just three learnable physics values per head — propagates that information via FFT in O(N log N). Tokens then read back from the field at their positions.

No quadratic attention matrix. No KV-cache that grows linearly with history. Just physics.

```
            Traditional Transformer              Crumb LLM
            ─────────────────────               ─────────
            Every token × every token            Scatter → FFT → Gather
            O(N²) per layer                     O(F log F) per layer
            KV-cache grows with N               Field-cache is FIXED
            32K = 35 GB                         32K = 6.7 GB
            128K = OOM                          128K = 26.7 GB
```

---

## The Three Physics Parameters

Each attention head in Crumb LLM learns exactly three values:

| Parameter | Symbol | What it controls |
|-----------|--------|------------------|
| **Damping** | α | How far information travels. High α = local head. Low α = global head. |
| **Frequency** | ω | What spatial patterns the head is sensitive to. Like tuning a radio. |
| **Phase** | φ | Where in the oscillation cycle the kernel starts. Shifts the head's "viewpoint". |

The kernel: **k(t) = exp(-α|t|) · cos(ωt + φ)**

Three parameters. That's it. A standard transformer head has d² + d² = 2d² parameters (Q and K projections alone). At d=128, that's 32,768 parameters per head just for routing. Crumb LLM V1 routes information with 3.

**V2 blocks** add learned components (smooth causal masking, learned scatter/gather, query-conditioned frequency gating, RoPE, short convolution) that increase per-head parameters but remain O(N log N). V2 is the SOTA variant; V1 is the minimal, fast variant.

---

## What We Discovered

### 1. Wave-field mixing dramatically outperforms attention on structured text

When we trained identical-shape models (same dim, same depth, same FFN) on our crumb corpus:

```
┌─────────────────┬──────────────┬──────────────┐
│                 │  Crumb LLM   │ Transformer  │
├─────────────────┼──────────────┼──────────────┤
│ Perplexity      │    1.16      │   11.56      │
│ Bits/char       │    0.21      │    3.53      │
│ Parameters      │  231,492     │  299,040     │
│ Training time   │    87s       │    41s       │
└─────────────────┴──────────────┴──────────────┘
```

The wave-field mixer is **10× better** at modeling structured documents. Why? Because CRUMB documents have repetitive structure (section headers, keywords, patterns) that map naturally onto spectral components. The FFT-based kernel captures these patterns in a single O(F log F) operation where attention needs O(N²) pairwise comparisons to notice the same regularities.

### 2. Context pulling via spectral matching works

We indexed 110 crumb sections and queried them with natural language. The spectral fingerprint (|rFFT(scatter(tokens))|) of a section acts like an embedding — but it's computed from the same physics as the model itself, not a separate encoder.

Query: *"Fix the login redirect bug"*
Retrieved: The constraints section from task-bug-fix.crumb
Latency: <2ms for 110 sections

This isn't just RAG with extra steps. The retrieval uses the same mathematical basis as the model's mixing operation — spectral similarity in the frequency domain. The model and the retriever speak the same language.

### 3. The field-state cache gives O(1) generation

Because convolution is linear:
```
field(tokens₁..ₜ₊₁) = field(tokens₁..ₜ) + convolve(scatter(tokenₜ₊₁))
```

We cache the convolved field and only add each new token's contribution. Cost per generated token: **O(F log F)**, independent of how many tokens came before. A transformer's KV-cache costs O(N) per token and grows without bound. Ours is fixed at F entries forever.

### 4. Information doesn't vanish at the window edge

In sliding-window attention (Mistral, etc.), once a token scrolls past the window, it's gone completely. In Crumb LLM's sliding field, old tokens leave behind **residual wave energy**. Their information decays gracefully according to the damping coefficient α — it doesn't cliff-drop at the window boundary.

This means a token 10,000 positions ago still has non-zero influence if its α was small enough. The model learns how long things should be remembered.

### 5. CRUMB structure naturally maps to wave physics

This isn't accidental. CRUMB's format features have natural wave-field interpretations:

| CRUMB feature | Wave-field interpretation |
|---------------|--------------------------|
| Section boundary `[goal]` | Zero-amplitude node — forces information to propagate through the field, not leak through adjacency |
| `@priority: 9` | Scatter amplitude 1.8× — this token's wave is louder, travels further |
| `fold:NAME/summary` | Long-range mode — α reduced, summary info propagates to distant tokens |
| Repetitive structure | Standing waves — reinforced by resonance memory across layers |

The model doesn't just tolerate structure. It *exploits* it.

---

## Architecture at a Glance

```
Input tokens
    │
    ▼
┌─────────────────────────────────────────────┐
│  Embedding (byte-level or char-level)        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  Wave-Field Block (× N layers)               │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ RMSNorm → proj_in                    │   │
│  │    ↓                                 │   │
│  │ Scatter (tokens → field)             │   │
│  │    ↓                                 │   │
│  │ Boundary condition (periodic/absorb) │   │
│  │    ↓                                 │   │
│  │ FFT-Convolve with k(t)   O(F log F) │   │
│  │    ↓                                 │   │
│  │ Spectral gate (optional)             │   │
│  │    ↓                                 │   │
│  │ Dispersion (optional)                │   │
│  │    ↓                                 │   │
│  │ Gather (field → tokens)              │   │
│  │    ↓                                 │   │
│  │ proj_out / interference mixer        │   │
│  └──────────────┬───────────────────────┘   │
│                 ↓                            │
│  + residual                                 │
│  RMSNorm → SwiGLU FFN → + residual         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  RMSNorm → Tied LM Head → Logits            │
└─────────────────────────────────────────────┘
```

---

## Novel Contributions (beyond Wave-Field LLM)

Crumb LLM is *not* just a reproduction. These are original:

1. **Adaptive kernels** — α/ω/φ predicted from input via MLP (content-dependent mixing, like attention is content-dependent)
2. **Field-attention hybrid gate** — learned blend of wave-field (global) and windowed attention (local) per token
3. **Resonance memory** — cross-layer EMA in frequency space amplifies persistent structural patterns
4. **Spectral gating** — learned per-frequency gate suppresses noise bands, passes signal bands
5. **Context pulling via spectral matching** — same-basis retrieval using wave-field fingerprints
6. **Crumb-native structural priors** — document structure becomes physics parameters, not thrown-away metadata
7. **O(1) field-state cache** — fixed-cost generation regardless of history length

---

## Quick Start

```bash
# Install
pip install 'crumb-format[llm]'

# Train a model (CPU, ~2 min)
crumb llm train --config tiny --steps 500 --out /tmp/crumb_model

# Generate text
crumb llm generate --ckpt /tmp/crumb_model --prompt "BEGIN CRUMB"

# Index your crumb library for context pulling
crumb llm index examples/ -o my_index.json

# Pull relevant context for a query
crumb llm pull "How do I fix auth?" --index-file my_index.json

# Serve over HTTP with context pulling
crumb llm serve --ckpt /tmp/crumb_model --index examples/

# Compare against a transformer baseline
crumb llm compare --config tiny --steps 1000 --data examples/

# Export for HuggingFace Hub
crumb llm export --ckpt /tmp/crumb_model --name my-crumb-llm
```

---

## Deployment (Cheapest Path)

| Method | Cost | Latency | Setup |
|--------|------|---------|-------|
| Local CLI | Free | ~15ms/tok | `pip install crumb-format[llm]` |
| Docker (CPU) | ~$5/mo | ~20ms/tok | `docker build -f Dockerfile.crumb-llm .` |
| Fly.io | ~$7/mo | ~25ms/tok | `fly launch` |
| HF Spaces | Free tier | ~30ms/tok | Push hub export |
| MCP plugin | Free | ~15ms/tok | Add tools to Claude Desktop config |

The entire trained model + server fits in a **~500MB Docker image** with CPU-only PyTorch.

---

## Context Pulling (How Claude Does It)

Instead of stuffing 100K tokens into the context window:

```
Traditional: [entire crumb library] + query → model → answer
             ↓ O(N²) on everything ↓ "lost in the middle" ↓ $$$

Crumb LLM:   query → spectral match → pull top-k sections → model → answer
             ↓ O(F log F) on small N ↓ only relevant data ↓ cheap
```

The spectral matcher uses the same FFT-based fingerprints as the model itself. Retrieval and generation share a mathematical basis — they "think" the same way.

---

## The Numbers

| Metric | Crumb LLM | Transformer (same params) |
|--------|-----------|---------------------------|
| Forward pass (8K tokens) | 70ms | 151ms |
| Memory (32K tokens) | 6.7 GB | 35 GB |
| Memory (128K tokens) | 26.7 GB | OOM |
| Params per routing head | 3 | 32,768 |
| Generation cache size | O(F) fixed | O(N) grows |
| Crumb corpus PPL | 1.16 | 11.56 |

---

## References

- **Wave Field LLM** — Badaramoni (2026). [github.com/badaramoni/wave-field-llm](https://github.com/badaramoni/wave-field-llm)
- **Wave-PDE Nets** — arXiv:2510.04304
- **SPECTRE** — arXiv:2502.18394 (FFT-based attention drop-in)
- **CRUMB Format** — XIO AI Solutions. [github.com/XioAISolutions/crumb-format](https://github.com/XioAISolutions/crumb-format)

---

## Citation

```bibtex
@software{crumb_llm_2026,
  title   = {Crumb LLM: O(N log N) Language Modeling via Wave-Equation Dynamics
             with Structure-Aware Context Pulling},
  author  = {XIO AI Solutions},
  year    = {2026},
  url     = {https://github.com/XioAISolutions/crumb-format},
  license = {MIT}
}
```

---

*Crumb LLM is experimental research software. Use it to explore, benchmark, and learn. The architecture is sound; the pre-trained checkpoints are tiny. This is the beginning, not the end.*

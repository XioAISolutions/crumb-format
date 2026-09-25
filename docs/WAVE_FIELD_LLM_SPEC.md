1.5-2× on modern GPUs (A100, H100).

### 6.5 Distributed Training

```python
# Data parallel across GPUs
model = nn.DataParallel(model)

# Or fully sharded (FSDP) for very large models
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    auto_wrap_policy=lambda module, recurse, nonwrapped_params: 
        isinstance(module, WaveFieldBlock)
)
```

**Scaling:** Train 1.5B parameter models on 8× A100 GPUs.

---

## 7. CRUMB-Native Features

### 7.1 Structure-Aware Processing

CRUMB documents have rich structure that maps to wave physics:

| CRUMB Feature | Wave Field Mapping |
|---------------|-------------------|
| Section boundary `[name]` | Zero scatter weight (information barrier) |
| `@priority: N` | Scatter amplitude × (N/5) |
| `fold:NAME/summary` | Reduced damping α (long-range propagation) |
| `fold:NAME/full` | Normal damping (local detail) |
| Repetitive structure | Standing waves (resonance memory) |

### 7.2 Spectral Context Retrieval

```python
def spectral_match(query_tokens, document_sections):
    """Match query to relevant sections via spectral similarity"""
    
    # Compute spectral fingerprint of query
    query_field = scatter(embed(query_tokens))
    query_spectrum = torch.fft.rfft(query_field).abs()
    
    # Compare to precomputed section spectra
    similarities = []
    for section_spectrum in document_sections:
        sim = cosine_similarity(query_spectrum, section_spectrum)
        similarities.append(sim)
    
    # Return top-k matches
    return top_k(similarities, k=5)
```

**Advantage:** Retrieval uses same mathematical basis as model (spectral decomposition).

### 7.3 Cross-Document Wave Interference

```python
def merge_document_fields(doc_fields, weights):
    """Combine multiple document fields with learned weights"""
    
    # Convert to frequency domain
    doc_spectra = [torch.fft.rfft(f) for f in doc_fields]
    
    # Weighted sum with phase alignment
    merged_spectrum = sum(w * s for w, s in zip(weights, doc_spectra))
    
    # Convert back to field
    return torch.fft.irfft(merged_spectrum)
```

### 7.4 Priority-Based Scatter Weights

```python
def compute_scatter_weights(tokens, crumb_metadata):
    """Convert CRUMB priorities to scatter amplitudes"""
    
    weights = torch.ones(len(tokens))
    
    for i, token in enumerate(tokens):
        # Section boundaries: zero weight
        if token == SECTION_SEP:
            weights[i] = 0.0
        
        # Priority annotations: scale amplitude
        elif token in crumb_metadata:
            priority = crumb_metadata[token].get('priority', 5)
            weights[i] = priority / 5.0  # normalize to ~1.0
        
        # Summary folds: boost slightly
        elif crumb_metadata[token].get('fold_type') == 'summary':
            weights[i] = 1.5
    
    return weights
```

### 7.5 Ref-Based Field Caching

```python
class RefCache:
    """Cache field states for referenced sections"""
    
    def __init__(self):
        self.cache = {}  # ref_digest → field_state
    
    def store(self, ref_digest, field_state):
        """Store field state for a ref"""
        self.cache[ref_digest] = field_state.detach().clone()
    
    def load(self, ref_digest):
        """Load cached field state"""
        return self.cache.get(ref_digest, None)
    
    def merge(self, current_field, ref_digests, weights):
        """Merge current field with cached refs"""
        merged = current_field.clone()
        
        for ref, weight in zip(ref_digests, weights):
            cached = self.load(ref)
            if cached is not None:
                merged = merged + weight * cached
        
        return merged
```

---

## 8. Implementation Roadmap

### Phase 1: Core Wave Engine (Weeks 1-2)

**Goal:** Implement basic wave field operations

**Tasks:**
1. Implement kernel construction (time and frequency domain)
2. Implement scatter/gather operations (linear interpolation)
3. Implement FFT convolution
4. Unit tests for all operations
5. Numerical equivalence tests (FFT vs direct convolution)

**Deliverables:**
- `crumb_llm/kernels.py` - kernel construction
- `crumb_llm/scatter_gather.py` - scatter/gather ops
- `crumb_llm/physics.py` - boundary conditions, dispersion
- Tests in `tests/test_crumb_llm_kernels.py`

**Success Criteria:**
- All unit tests pass
- FFT convolution matches direct convolution (within 1e-5)
- Forward pass completes without errors

### Phase 2: Training Infrastructure (Weeks 3-4)

**Goal:** Train first working model

**Tasks:**
1. Implement WaveFieldBlock and WaveFieldLM
2. Implement training loop with loss functions
3. Implement data loading for CRUMB corpus
4. Add gradient clipping and stability checks
5. Train tiny model (230K params) to convergence

**Deliverables:**
- `crumb_llm/layers.py` - WaveFieldBlock
- `crumb_llm/model.py` - WaveFieldLM
- `crumb_llm/train.py` - training loop
- `crumb_llm/data.py` - data loading
- First trained checkpoint

**Success Criteria:**
- Model trains without NaN/Inf
- Perplexity decreases over training
- Generated text is coherent
- Beats transformer baseline on CRUMB corpus

### Phase 3: Optimization and Scaling (Weeks 5-6)

**Goal:** Scale to production-size models

**Tasks:**
1. Implement mixed precision training
2. Implement gradient checkpointing
3. Implement distributed training (DDP/FSDP)
4. Optimize FFT operations (batching, caching)
5. Train medium model (350M params)

**Deliverables:**
- Optimized training code
- Multi-GPU training scripts
- Medium model checkpoint
- Performance benchmarks

**Success Criteria:**
- 2× speedup from optimizations
- Can train 350M model on 8× A100
- Memory usage within budget
- Training stable at scale

### Phase 4: CRUMB Integration (Weeks 7-8)

**Goal:** Native CRUMB support

**Tasks:**
1. Implement CRUMB adapter (priority, folds, refs)
2. Implement spectral context retrieval
3. Implement ref-based field caching
4. Train CRUMB-aware model
5. Benchmark on CRUMB-specific tasks

**Deliverables:**
- `crumb_llm/crumb_adapter.py` - CRUMB integration
- `crumb_llm/context_pull.py` - spectral retrieval
- `crumb_llm/cache.py` - ref caching
- CRUMB-trained checkpoint

**Success Criteria:**
- 10× better perplexity on structured docs
- Context retrieval works accurately
- Ref caching speeds up generation
- Structure-aware features improve quality

### Phase 5: Benchmarking and Validation (Weeks 9-10)

**Goal:** Production readiness

**Tasks:**
1. Comprehensive benchmarking vs transformers
2. Ablation studies (which features matter most)
3. Long-context evaluation (8K, 32K, 128K)
4. Memory profiling and optimization
5. Documentation and examples

**Deliverables:**
- `crumb_llm/bench.py` - benchmarking suite
- Benchmark results and analysis
- Ablation study results
- Production deployment guide
- API documentation

**Success Criteria:**
- Meets all performance targets
- Comprehensive test coverage (>90%)
- Production-ready code quality
- Complete documentation

---

## 9. Benchmarking Plan

### 9.1 Metrics

#### Perplexity
```python
def compute_perplexity(model, dataset):
    """Standard language modeling metric"""
    total_loss = 0
    total_tokens = 0
    
    for batch in dataset:
        outputs = model(batch['input_ids'], targets=batch['targets'])
        total_loss += outputs['loss'].item() * batch['input_ids'].numel()
        total_tokens += batch['input_ids'].numel()
    
    return math.exp(total_loss / total_tokens)
```

#### Speed (tokens/second)
```python
def benchmark_speed(model, seq_len, batch_size, device):
    """Measure throughput"""
    x = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len))
    x = x.to(device)
    
    # Warmup
    for _ in range(10):
        model(x)
    
    # Measure
    start = time.time()
    for _ in range(100):
        model(x)
    elapsed = time.time() - start
    
    tokens_per_sec = (100 * batch_size * seq_len) / elapsed
    return tokens_per_sec
```

#### Memory Usage
```python
def measure_memory(model, seq_len, batch_size):
    """Peak memory during forward+backward"""
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len))
    x = x.cuda()
    
    outputs = model(x, targets=x)
    outputs['loss'].backward()
    
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return peak_mb
```

### 9.2 Comparison Baselines

- **GPT-2 (124M params):** Standard transformer baseline
- **LLaMA-style Transformer (350M params):** Modern architecture
- **Mistral-style Sliding Window:** Memory-efficient baseline

### 9.3 Success Criteria

**Must Achieve:**
- ✓ 10× better perplexity on CRUMB corpus (target: <2.0 vs 11.56)
- ✓ 2-3× speed improvement at 8K+ context
- ✓ O(F) memory scaling (not O(N))
- ✓ Stable training (no NaN/Inf)
- ✓ Coherent text generation

---

## 10. File Structure

```
crumb_llm/
├── __init__.py              # Package exports
├── model.py                 # WaveFieldLM main model
├── layers.py                # WaveFieldBlock, RMSNorm, FFN
├── kernels.py               # Wave kernel construction
├── scatter_gather.py        # Scatter/gather operations
├── physics.py               # Advanced physics (dispersion, boundaries)
├── v2.py                    # V2 enhancements (learned scatter, etc)
├── train.py                 # Training loop
├── data.py                  # Data loading and preprocessing
├── tokenizer.py             # Byte/char tokenizers
├── baseline.py              # Transformer baseline for comparison
├── crumb_adapter.py         # CRUMB-specific features
├── context_pull.py          # Spectral context retrieval
├── cache.py                 # Field-state caching
├── sliding.py               # Sliding window support
├── sample.py                # Generation utilities
├── bench.py                 # Benchmarking suite
├── serve.py                 # HTTP API server
├── hub.py                   # HuggingFace Hub integration
└── configs/                 # Model configurations
    ├── tiny.json
    ├── small.json
    ├── medium.json
    └── large.json
```

---

## 11. Advanced Topics

### 11.1 Adaptive Kernels

Input-conditioned wave parameters:

```python
class AdaptiveKernelHead(nn.Module):
    """Predict α, ω, φ from input context"""
    
    def __init__(self, dim, field_size):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 3)  # α, ω, φ
        )
        self.field_size = field_size
    
    def forward(self, x):
        """
        x: [B, N, D] token features
        Returns: [B, F//2+1] complex kernel
        """
        # Pool over sequence
        x_pool = x.mean(dim=1)  # [B, D]
        
        # Predict parameters
        params = self.predictor(x_pool)  # [B, 3]
        α = params[:, 0].abs() + 0.01
        ω = params[:, 1]
        φ = params[:, 2]
        
        # Build kernel
        return wave_kernel_freq(α, ω, φ, self.field_size)
```

### 11.2 Resonance Memory

Cross-layer standing wave amplification:

```python
class ResonanceMemory(nn.Module):
    """Amplify persistent frequency patterns across layers"""
    
    def __init__(self, n_heads, field_size, decay=0.9):
        super().__init__()
        self.decay = decay
        self.amplification = nn.Parameter(torch.ones(n_heads, field_size // 2 + 1))
    
    def forward(self, memory_spectrum, current_field):
        """
        memory_spectrum: [B, H, F//2+1] complex - EMA of past spectra
        current_field: [B, H, F, D] real
        
        Returns: updated memory, amplified field
        """
        # Current spectrum
        current_spec = torch.fft.rfft(current_field, dim=2)
        
        # Update memory (EMA)
        updated_memory = (self.decay * memory_spectrum + 
                         (1 - self.decay) * current_spec.mean(dim=-1))
        
        # Amplify resonant frequencies
        amplified_spec = current_spec * (1 + self.amplification.unsqueeze(0).unsqueeze(-1) * 
                                         updated_memory.abs().unsqueeze(-1))
        
        # Convert back to field
        amplified_field = torch.fft.irfft(amplified_spec, n=current_field.shape[2], dim=2)
        
        return updated_memory, amplified_field
```

### 11.3 Hybrid Wave-Attention

Blend wave field with local attention:

```python
class FieldAttentionGate(nn.Module):
    """Learned blend of wave field (global) and attention (local)"""
    
    def __init__(self, dim, window_size=64):
        super().__init__()
        self.window_size = window_size
        self.gate_proj = nn.Linear(dim, 1)
        self.local_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
    
    def forward(self, x, wave_output):
        """
        x: [B, N, D] input
        wave_output: [B, N, D] wave field output
        
        Returns: [B, N, D] blended output
        """
        # Compute local attention in sliding windows
        attn_output = self._windowed_attention(x)
        
        # Compute blend gate
        gate = torch.sigmoid(self.gate_proj(x))  # [B, N, 1]
        
        # Blend
        return gate * wave_output + (1 - gate) * attn_output
    
    def _windowed_attention(self, x):
        """Apply attention in sliding windows"""
        B, N, D = x.shape
        W = self.window_size
        
        # Pad to multiple of window size
        pad = (W - N % W) % W
        x_pad = F.pad(x, (0, 0, 0, pad))
        
        # Reshape to windows
        x_win = x_pad.view(B, -1, W, D)
        
        # Apply attention per window
        out_win = []
        for i in range(x_win.shape[1]):
            out, _ = self.local_attn(x_win[:, i], x_win[:, i], x_win[:, i])
            out_win.append(out)
        
        out = torch.stack(out_win, dim=1).view(B, -1, D)
        
        # Remove padding
        return out[:, :N, :]
```

---

## 12. Appendices

### A. Glossary

- **Field:** Continuous spatial domain where wave propagation occurs
- **Scatter:** Map discrete tokens to continuous field positions
- **Gather:** Sample field at token positions to get outputs
- **Kernel:** Wave propagation function k(t) = exp(-α|t|)cos(ωt+φ)
- **FFT:** Fast Fourier Transform, O(n log n) algorithm
- **Damping (α):** Controls how far waves travel before decaying
- **Frequency (ω):** Controls oscillation rate and pattern sensitivity
- **Phase (φ):** Shifts the wave's starting point
- **Interference:** Superposition of multiple waves (constructive/destructive)
- **Spectral:** Frequency-domain representation
- **Resonance:** Standing wave patterns from repeated structure

### B. Mathematical Notation

- `N`: Number of tokens in sequence
- `F`: Field size (typically 2N, power of 2)
- `H`: Number of heads
- `D`: Model dimension
- `d`: Per-head dimension (D/H)
- `V`: Vocabulary size
- `L`: Number of layers
- `B`: Batch size
- `α`: Damping coefficient
- `ω`: Angular frequency
- `φ`: Phase
- `ψ`: Field amplitude
- `ξ`: Frequency variable
- `⊗`: Convolution operator
- `ℱ`: Fourier transform
- `ℱ⁻¹`: Inverse Fourier transform

### C. References

1. **Wave Field LLM** - Badaramoni (2026). Original wave-field architecture.
2. **SPECTRE** - arXiv:2502.18394. FFT-based attention with spectral gating.
3. **Hyena** - arXiv:2302.10866. Long convolution for sequence modeling.
4. **Mamba** - arXiv:2312.00752. State space models with selective mechanisms.
5. **RoPE** - arXiv:2104.09864. Rotary position embeddings.
6. **CRUMB Format** - XIO AI Solutions. Structured AI handoff format.

### D. Performance Targets Summary

| Metric | Target | Baseline (Transformer) |
|--------|--------|----------------------|
| Perplexity (CRUMB) | <2.0 | 11.56 |
| Perplexity (general) | <15.0 | 13.5 |
| Speed (8K context) | >2000 tok/s | <1000 tok/s |
| Memory (32K) | <10 GB | 35 GB |
| Memory (128K) | <30 GB | OOM |
| Retrieval accuracy | >90% | N/A |
| Retrieval latency | <2ms | N/A |

---

## Conclusion

This specification provides a complete, production-ready design for a Wave Field LLM that replaces O(n²) transformer attention with O(n log n) wave propagation. The architecture is:

1. **Mathematically rigorous:** Based on well-understood wave equations
2. **Fully implementable:** All algorithms specified in detail
3. **Performance-optimized:** FFT acceleration, mixed precision, distributed training
4. **CRUMB-native:** Structure-aware features for 10× better perplexity
5. **Production-ready:** Complete roadmap, benchmarks, and integration points

The next step is implementation following the 10-week roadmap. All core components are specified with sufficient detail to begin coding immediately.

**Key Innovation:** By treating language modeling as wave propagation in a continuous field rather than discrete token interactions, we achieve better complexity, better memory efficiency, and better performance on structured documents—all while maintaining the expressiveness needed for general language modeling.

---

**Document Version:** 2.0  
**Last Updated:** 2026-05-28  
**Status:** Ready for Implementation  
**Next Action:** Begin Phase 1 (Core Wave Engine)
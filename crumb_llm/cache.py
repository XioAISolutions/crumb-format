"""Field-state cache for O(1)-per-token autoregressive generation.

In a standard transformer, the KV-cache stores past keys and values so
each generation step only computes attention for the new token. The
field-state cache is the wave-field analog:

    The convolved field from all previous tokens is cached.
    For each new token we only need to:
      1. Project the new token (one linear)
      2. Scatter it onto 2 field cells
      3. Convolve JUST that sparse deposit with the kernel (O(F) or O(F log F))
      4. Add the result to the cached field
      5. Gather from the cached field at the new position (O(1))
      6. Project out + FFN on the single new token

    Cost per step: O(F log F + D²) per layer — independent of past length.
    Compare to transformer KV-cache: O(N·d + D²) per step — grows with N.

The key insight is that convolution is LINEAR:

    convolve(scatter(x₁..t+1)) = convolve(scatter(x₁..t)) + convolve(scatter(x_{t+1}))

so we can decompose the field into "past" (cached) + "new" (computed).

Usage:
    cache = FieldStateCache.init(model, batch_size=1)
    for token_id in prompt_ids:
        logits, cache = model.forward_cached(token_id, cache)
    # Now generate:
    for _ in range(max_new_tokens):
        next_id = sample(logits)
        logits, cache = model.forward_cached(next_id, cache)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LayerFieldState:
    """Cached convolved field for one layer."""
    field: Tensor          # [B, H, F, d_head] — accumulated convolved field
    n_tokens_seen: int     # how many tokens have been scattered so far


@dataclass
class FieldStateCache:
    """Per-layer field states + generation metadata."""
    layers: list[LayerFieldState]
    seq_pos: int           # current sequence position (next token goes here)

    @classmethod
    def init(cls, model, batch_size: int = 1, device=None) -> "FieldStateCache":
        """Create an empty cache from a WaveFieldLM."""
        from .model import WaveFieldLM
        assert isinstance(model, WaveFieldLM)
        device = device or next(model.parameters()).device
        cfg = model.cfg
        d_head = cfg.dim // cfg.n_heads
        layers = []
        for _ in range(cfg.n_layers):
            f = torch.zeros(batch_size, cfg.n_heads, cfg.field_size, d_head, device=device)
            layers.append(LayerFieldState(field=f, n_tokens_seen=0))
        return cls(layers=layers, seq_pos=0)


def _scatter_single_token(
    h: Tensor,           # [B, H, 1, d_head]
    token_pos: int,      # integer position of the new token
    total_tokens: int,   # N — current sequence length (for coordinate mapping)
    field_size: int,
) -> Tensor:
    """Scatter one token onto the field. Returns [B, H, F, d_head]."""
    B, H, _, D = h.shape
    F_ = field_size
    device, dtype = h.device, h.dtype

    # Map token position to continuous field coordinate.
    x = min((token_pos + 0.5) * F_ / max(total_tokens, 1), F_ - 1 - 1e-6)
    left = int(x)
    frac = x - left
    right = min(left + 1, F_ - 1)

    sparse_field = h.new_zeros((B, H, F_, D))
    sparse_field[:, :, left, :] = h[:, :, 0, :] * (1.0 - frac)
    sparse_field[:, :, right, :] = h[:, :, 0, :] * frac
    return sparse_field


def _convolve_sparse_field(
    sparse_field: Tensor,    # [B, H, F, d_head]
    kernel_freq: Tensor,     # [H, F//2+1]
    field_size: int,
) -> Tensor:
    """Convolve a sparse field (only 2 nonzero cells) with the kernel."""
    # Even though the field is sparse, we still do the FFT — the kernel
    # is dense in frequency domain so there's no shortcut. This is still
    # O(F log F) per step, but F is fixed and small.
    spec = torch.fft.rfft(sparse_field, n=field_size, dim=-2)
    kf = kernel_freq.unsqueeze(0).unsqueeze(-1)  # [1, H, F//2+1, 1]
    out = torch.fft.irfft(spec * kf, n=field_size, dim=-2)
    return out


def _gather_single_token(
    field: Tensor,       # [B, H, F, d_head]
    token_pos: int,
    total_tokens: int,
    field_size: int,
) -> Tensor:
    """Gather one token from the field. Returns [B, H, 1, d_head]."""
    F_ = field_size
    x = min((token_pos + 0.5) * F_ / max(total_tokens, 1), F_ - 1 - 1e-6)
    left = int(x)
    frac = x - left
    right = min(left + 1, F_ - 1)
    val = field[:, :, left, :] * (1.0 - frac) + field[:, :, right, :] * frac
    return val.unsqueeze(2)  # [B, H, 1, d_head]


def forward_cached_block(
    block,
    x_new: Tensor,               # [B, 1, D] — the new token's state
    layer_state: LayerFieldState,
    token_pos: int,
    total_tokens: int,
) -> tuple[Tensor, LayerFieldState]:
    """Process one new token through a WaveFieldBlock with caching.

    Returns (output_token [B, 1, D], updated LayerFieldState).
    """
    cfg = block.cfg
    H, d_head = cfg.n_heads, block.d_head
    F_ = cfg.field_size
    B = x_new.shape[0]

    # Pre-norm + project.
    h_normed = block.norm_mix(x_new)
    h = block.proj_in(h_normed).view(B, 1, H, d_head).transpose(1, 2)  # [B,H,1,d]

    # Build kernel (cached per step — recomputed, but cheap: just 3 params).
    kfs = [head.kernel_freq() for head in block.heads]
    kernel = torch.stack(kfs, dim=0)  # [H, F//2+1]

    # Scatter new token, convolve, add to cached field.
    sparse = _scatter_single_token(h, token_pos, total_tokens, F_)
    contribution = _convolve_sparse_field(sparse, kernel, F_)
    new_field = layer_state.field + contribution

    # Gather from the accumulated field at the new token's position.
    out = _gather_single_token(new_field, token_pos, total_tokens, F_)

    # Interference mixer or standard projection.
    if block.mixer is not None:
        out = block.mixer(out)
    out = out.transpose(1, 2).reshape(B, 1, cfg.dim)
    mixed = block.proj_out(out)

    # Residual + FFN.
    h_out = x_new + mixed
    h_out = h_out + block.ffn(block.norm_ffn(h_out))

    updated = LayerFieldState(field=new_field, n_tokens_seen=layer_state.n_tokens_seen + 1)
    return h_out, updated


@torch.no_grad()
def generate_cached(
    model,
    input_ids: Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Tensor:
    """Autoregressive generation with field-state caching.

    Prefill: runs the full forward once on the prompt to populate caches.
    Decode: processes one token at a time using the cached fields.

    Args:
        model: WaveFieldLM instance.
        input_ids: [B, T] prompt tokens.
        max_new_tokens: how many tokens to generate.
        temperature: sampling temperature.
        top_k: top-k filtering.
        eos_token_id: stop on this token.

    Returns:
        [B, T + generated] token ids.
    """
    B, T = input_ids.shape
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Phase 1: Prefill — run full forward to get initial field states.
    cache = FieldStateCache.init(model, batch_size=B, device=device)
    x = model.embed(input_ids)
    for layer_idx, block in enumerate(model.blocks):
        # Full forward for the prompt.
        x_out = block(x)
        # Populate the cache by scattering all prompt tokens.
        h = block.proj_in(block.norm_mix(x)).view(B, T, block.cfg.n_heads, block.d_head).transpose(1, 2)
        from .scatter_gather import scatter_linear
        full_field = scatter_linear(h, block.cfg.field_size)
        kfs = [head.kernel_freq() for head in block.heads]
        kernel = torch.stack(kfs, dim=0)
        full_field = _convolve_sparse_field(full_field, kernel, block.cfg.field_size)
        cache.layers[layer_idx] = LayerFieldState(field=full_field, n_tokens_seen=T)
        x = x_out
    x = model.norm_out(x)
    if model.lm_head is None:
        logits = x @ model.embed.weight.t()
    else:
        logits = model.lm_head(x)

    # Get the last logit for sampling.
    last_logits = logits[:, -1, :]
    cache.seq_pos = T

    ids = input_ids
    for _ in range(max_new_tokens):
        # Sample next token.
        scaled = last_logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(scaled, k=min(top_k, scaled.size(-1)))
            scaled = torch.where(
                scaled < v[..., [-1]], torch.full_like(scaled, float("-inf")), scaled
            )
        probs = torch.softmax(scaled, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if eos_token_id is not None and (next_id == eos_token_id).all():
            break

        # Phase 2: Decode — one token through all layers with cache.
        x_tok = model.embed(next_id)  # [B, 1, D]
        total_tokens = cache.seq_pos + 1
        for layer_idx, block in enumerate(model.blocks):
            x_tok, cache.layers[layer_idx] = forward_cached_block(
                block, x_tok, cache.layers[layer_idx],
                token_pos=cache.seq_pos, total_tokens=total_tokens,
            )
        x_tok = model.norm_out(x_tok)
        if model.lm_head is None:
            last_logits = (x_tok @ model.embed.weight.t()).squeeze(1)
        else:
            last_logits = model.lm_head(x_tok).squeeze(1)
        cache.seq_pos += 1

    return ids

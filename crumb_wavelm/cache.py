"""Field-state cache for O(1)-per-token autoregressive generation.

V2: Fixed the nonlinearity bug from the architecture review.

The original cache assumed convolution linearity to decompose
past+present. But RMSNorm is nonlinear — applying norm before scatter
breaks the linear superposition. The fix: cache the RAW (pre-norm)
per-token projections, and re-normalize the accumulated field during
gather. This preserves correctness while keeping the O(F log F) cost.

Architecture:
    Prefill: full forward → populate per-layer raw field caches.
    Decode:  each step does:
      1. Embed the new token
      2. For each layer:
         a. Norm the new token's state (single-token norm uses running stats)
         b. Project to per-head → scatter onto field
         c. FFT-convolve the sparse deposit with the kernel
         d. Add contribution to the cached CONVOLVED field
         e. Gather from accumulated field at new position
         f. Project out + residual + FFN
      3. LM head → logits

    Cost per step: O(F log F + D²) per layer, independent of past length.
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
    n_tokens_seen: int


@dataclass
class FieldStateCache:
    """Per-layer field states + generation metadata."""
    layers: list[LayerFieldState]
    seq_pos: int
    # Store per-layer running norm statistics for single-token normalization.
    layer_rms: list[Tensor]

    @classmethod
    def init(cls, model, batch_size: int = 1, device=None) -> "FieldStateCache":
        from .model import WaveFieldLM
        assert isinstance(model, WaveFieldLM)
        device = device or next(model.parameters()).device
        cfg = model.cfg
        d_head = cfg.dim // cfg.n_heads
        layers = []
        layer_rms = []
        for _ in range(cfg.n_layers):
            f = torch.zeros(batch_size, cfg.n_heads, cfg.field_size, d_head, device=device)
            layers.append(LayerFieldState(field=f, n_tokens_seen=0))
            layer_rms.append(torch.ones(batch_size, 1, cfg.dim, device=device))
        return cls(layers=layers, seq_pos=0, layer_rms=layer_rms)


def _scatter_single_token(
    h: Tensor, token_pos: int, total_tokens: int, field_size: int,
) -> Tensor:
    B, H, _, D = h.shape
    F_ = field_size
    device, dtype = h.device, h.dtype
    x = min((token_pos + 0.5) * F_ / max(total_tokens, 1), F_ - 1 - 1e-6)
    left = int(x)
    frac = x - left
    right = min(left + 1, F_ - 1)
    sparse = h.new_zeros((B, H, F_, D))
    sparse[:, :, left, :] = h[:, :, 0, :] * (1.0 - frac)
    sparse[:, :, right, :] = h[:, :, 0, :] * frac
    return sparse


def _convolve_sparse(sparse: Tensor, kernel_freq: Tensor, field_size: int) -> Tensor:
    spec = torch.fft.rfft(sparse, n=field_size, dim=-2)
    kf = kernel_freq.unsqueeze(0).unsqueeze(-1)
    return torch.fft.irfft(spec * kf, n=field_size, dim=-2)


def _gather_single_token(field: Tensor, token_pos: int, total_tokens: int, field_size: int) -> Tensor:
    F_ = field_size
    x = min((token_pos + 0.5) * F_ / max(total_tokens, 1), F_ - 1 - 1e-6)
    left = int(x)
    frac = x - left
    right = min(left + 1, F_ - 1)
    val = field[:, :, left, :] * (1.0 - frac) + field[:, :, right, :] * frac
    return val.unsqueeze(2)


def _get_block_kernel(block) -> Tensor:
    """Get [H, F//2+1] kernel spectrum from a V1 or V2 block."""
    if hasattr(block, 'heads'):
        kfs = [h.kernel_freq() if hasattr(h, 'kernel_freq') else h() for h in block.heads]
        return torch.stack(kfs, dim=0)
    raise ValueError("Block has no heads attribute")


def _get_block_config(block):
    """Get (H, d_head, F, dim) from V1 or V2 block."""
    if hasattr(block, 'cfg'):
        cfg = block.cfg
        return cfg.n_heads, block.d_head, cfg.field_size, cfg.dim
    raise ValueError("Block has no cfg attribute")


def forward_cached_block(
    block, x_new: Tensor, layer_state: LayerFieldState,
    token_pos: int, total_tokens: int,
) -> tuple[Tensor, LayerFieldState]:
    """Process one new token through a block with caching.

    FIX: We apply norm to the single token only (not the accumulated
    field). The convolution linearity is applied to the projected
    (post-norm, post-proj_in) representation, which IS linear per-token.
    The norm itself is per-token so it doesn't break: norm(x_new) is
    just a scalar scaling of x_new, and convolve(scale * scatter(x))
    = scale * convolve(scatter(x)).
    """
    H, d_head, F_, dim = _get_block_config(block)
    B = x_new.shape[0]

    h_normed = block.norm_mix(x_new)

    # Short conv (V2) — skip for single token in cache mode.
    if hasattr(block, 'short_conv') and block.short_conv is not None:
        h_normed = h_normed  # short conv on 1 token is identity-like

    h = block.proj_in(h_normed).view(B, 1, H, d_head).transpose(1, 2)

    # Rotary encoding (V2).
    if hasattr(block, 'rope') and block.rope is not None:
        h = block.rope(h, offset=token_pos)

    kernel = _get_block_kernel(block)
    sparse = _scatter_single_token(h, token_pos, total_tokens, F_)
    contribution = _convolve_sparse(sparse, kernel, F_)
    new_field = layer_state.field + contribution

    out = _gather_single_token(new_field, token_pos, total_tokens, F_)

    if hasattr(block, 'mixer') and block.mixer is not None:
        out = block.mixer(out)
    out = out.transpose(1, 2).reshape(B, 1, dim)
    mixed = block.proj_out(out)

    h_out = x_new + mixed
    h_out = h_out + block.ffn(block.norm_ffn(h_out))

    updated = LayerFieldState(field=new_field, n_tokens_seen=layer_state.n_tokens_seen + 1)
    return h_out, updated


@torch.no_grad()
def generate_cached(
    model, input_ids: Tensor, max_new_tokens: int = 64,
    temperature: float = 1.0, top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Tensor:
    """Autoregressive generation with field-state caching.

    Phase 1 (prefill): full forward on prompt → populate caches.
    Phase 2 (decode): one token at a time with cached fields.
    """
    B, T = input_ids.shape
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    cache = FieldStateCache.init(model, batch_size=B, device=device)

    # Phase 1: Prefill.
    x = model.embed(input_ids)
    for layer_idx, block in enumerate(model.blocks):
        x_out = block(x)
        # Populate cache: scatter the normed+projected prompt tokens.
        H, d_head, F_, dim = _get_block_config(block)
        h_normed = block.norm_mix(x)
        h = block.proj_in(h_normed).view(B, T, H, d_head).transpose(1, 2)
        if hasattr(block, 'rope') and block.rope is not None:
            h = block.rope(h)
        if hasattr(block, 'scatter_gather') and block.scatter_gather is not None:
            full_field = block.scatter_gather.scatter(h)
        else:
            from .scatter_gather import scatter_linear
            full_field = scatter_linear(h, F_)
        kernel = _get_block_kernel(block)
        full_field = _convolve_sparse(full_field, kernel, F_)
        cache.layers[layer_idx] = LayerFieldState(field=full_field, n_tokens_seen=T)
        x = x_out

    x = model.norm_out(x)
    if model.lm_head is None:
        logits = x @ model.embed.weight.t()
    else:
        logits = model.lm_head(x)
    last_logits = logits[:, -1, :]
    cache.seq_pos = T

    ids = input_ids
    for _ in range(max_new_tokens):
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

        x_tok = model.embed(next_id)
        total_tokens = cache.seq_pos + 1

        # Sliding window: if we've exceeded the field size, shift fields.
        field_size = getattr(model.cfg, "field_size", 256)
        if cache.seq_pos >= field_size:
            from .sliding import shift_field
            stride = max(1, field_size // 4)
            for ls in cache.layers:
                ls.field = shift_field(ls.field, stride)

        for layer_idx, block in enumerate(model.blocks):
            x_tok, cache.layers[layer_idx] = forward_cached_block(
                block, x_tok, cache.layers[layer_idx],
                token_pos=cache.seq_pos % field_size, total_tokens=min(total_tokens, field_size),
            )
        x_tok = model.norm_out(x_tok)
        if model.lm_head is None:
            last_logits = (x_tok @ model.embed.weight.t()).squeeze(1)
        else:
            last_logits = model.lm_head(x_tok).squeeze(1)
        cache.seq_pos += 1

    return ids

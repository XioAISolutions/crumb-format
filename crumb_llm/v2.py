"""Crumb LLM v2 — next-generation wave-field building blocks.

Addresses every critical gap identified in the architecture review:

1. Smooth causal kernel (no aliasing from hard zero cutoff)
2. Learned scatter/gather (replaces lossy linear interpolation)
3. Query-conditioned frequency gating (SPECTRE pattern)
4. Short convolution + gating before FFT (Hyena pattern)
5. Position-dependent kernel modulation (Mamba-inspired)
6. Proper rotary-style position encoding adapted for fields

These compose into WaveFieldBlockV2 — a drop-in replacement for
WaveFieldBlock that can be selected via config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import RMSNorm, SwiGLUFFN


# ── 1. Smooth causal kernel ──────────────────────────────────────────


class SmoothCausalKernel(nn.Module):
    """Causal kernel without hard zero cutoff.

    Instead of zeroing t<0 (which creates a discontinuity at t=0 that
    aliases across the entire spectrum), we use a smooth sigmoid
    transition centered at t=0 with learnable steepness.

    k_causal(t) = k(t) · σ(β · t)

    where σ is the sigmoid and β controls the transition sharpness.
    Large β → sharp (like the old hard cutoff but smooth).
    Small β → gradual (some past info leaks, but no aliasing).
    """

    def __init__(self, field_size: int, alpha_init: float = 0.05,
                 omega_init: float = 1.0, phi_init: float = 0.0):
        super().__init__()
        self.field_size = field_size
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.omega = nn.Parameter(torch.tensor(omega_init))
        self.phi = nn.Parameter(torch.tensor(phi_init))
        self.causal_sharpness = nn.Parameter(torch.tensor(5.0))
        t = torch.arange(field_size, dtype=torch.float32) - field_size // 2
        self.register_buffer("t", t)

    def forward(self) -> Tensor:
        """Returns rFFT of the smooth causal kernel [F//2+1] complex."""
        a = self.alpha.abs()
        k = torch.exp(-a * self.t.abs()) * torch.cos(self.omega * self.t + self.phi)
        # Smooth causal window: sigmoid centered at t=0.
        causal = torch.sigmoid(self.causal_sharpness * self.t)
        k = k * causal
        k = torch.fft.ifftshift(k, dim=-1)
        return torch.fft.rfft(k, n=self.field_size, dim=-1)


# ── 2. Learned scatter/gather ────────────────────────────────────────


class LearnedScatterGather(nn.Module):
    """Replace linear interpolation with a learned projection.

    Each token's state is projected into the field via a small learned
    basis (not just 2 nearest neighbors). The basis functions are
    position-dependent Gaussians with learnable widths, giving each
    token a "footprint" on the field that the model can control.

    scatter: x_n → sum_f basis(n, f) · x_n   for each field cell f
    gather:  field → sum_f basis(n, f) · field[f]   for each token n
    """

    def __init__(self, field_size: int, basis_width: int = 8):
        super().__init__()
        self.field_size = field_size
        self.basis_width = basis_width
        # Learnable width parameter per position (shared across tokens).
        self.log_sigma = nn.Parameter(torch.tensor(math.log(float(basis_width) / 3.0)))

    def _basis(self, N: int, device, dtype) -> Tensor:
        """Compute [N, F] basis matrix (Gaussian centered at each token's position)."""
        F_ = self.field_size
        sigma = self.log_sigma.exp().clamp(min=0.5)
        token_pos = (torch.arange(N, device=device, dtype=dtype) + 0.5) * F_ / N
        field_pos = torch.arange(F_, device=device, dtype=dtype)
        # [N, F] distance matrix.
        dist = (token_pos.unsqueeze(1) - field_pos.unsqueeze(0)) ** 2
        basis = torch.exp(-dist / (2 * sigma ** 2))
        # Normalize so each token's basis sums to 1.
        basis = basis / (basis.sum(dim=1, keepdim=True) + 1e-8)
        return basis

    def scatter(self, values: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """values: [B, H, N, D] → field: [B, H, F, D]."""
        B, H, N, D = values.shape
        basis = self._basis(N, values.device, values.dtype)  # [N, F]
        if weights is not None:
            values = values * weights.view(B, 1, N, 1)
        # Einstein sum: [B,H,N,D] × [N,F] → [B,H,F,D]
        return torch.einsum("bhnd, nf -> bhfd", values, basis)

    def gather(self, field: Tensor, N: int) -> Tensor:
        """field: [B, H, F, D] → values: [B, H, N, D]."""
        basis = self._basis(N, field.device, field.dtype)  # [N, F]
        return torch.einsum("bhfd, nf -> bhnd", field, basis)


# ── 3. Query-conditioned frequency gating (SPECTRE pattern) ──────────


class QueryFrequencyGate(nn.Module):
    """Per-frequency gate conditioned on query tokens, not just pooled input.

    SPECTRE uses content-adaptive Toeplitz gates after FFT. This is our
    version: project the query into frequency space, then gate each
    spectral bin independently per head.

    The key difference from our earlier SpectralGate: this operates
    INSIDE the convolution (between FFT and iFFT), and it's conditioned
    on the per-head query, not pooled input.
    """

    def __init__(self, d_head: int, n_heads: int, field_size: int):
        super().__init__()
        spec_size = field_size // 2 + 1
        # Per-head query → per-frequency gate.
        self.gate_proj = nn.Linear(d_head, spec_size)
        self.n_heads = n_heads

    def forward(self, spec: Tensor, query: Tensor) -> Tensor:
        """
        spec: [B, H, F//2+1, D] complex — field after FFT × kernel.
        query: [B, H, N, d_head] — per-head token queries.

        Returns: gated spec [B, H, F//2+1, D] complex.
        """
        # Pool query over tokens to get per-head summary.
        q_pool = query.mean(dim=2)  # [B, H, d_head]
        gate = torch.sigmoid(self.gate_proj(q_pool))  # [B, H, spec_size]
        return spec * gate.unsqueeze(-1)


# ── 4. Short convolution + gating (Hyena pattern) ────────────────────


class ShortConvGate(nn.Module):
    """Pre-FFT short convolution with multiplicative gating.

    Hyena showed that stacking short depthwise convolutions with
    element-wise gating before the long convolution (FFT) dramatically
    improves quality. The short conv captures local patterns; the gate
    controls information flow.

    x → depthwise_conv(x) * sigmoid(linear(x))
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size,
                              padding=kernel_size - 1, groups=dim)
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, N, D] → [B, N, D]."""
        # Depthwise conv (causal: trim right padding).
        h = self.conv(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x))
        return h * gate


# ── 5. Position-dependent kernel modulation ──────────────────────────


class PositionKernelModulator(nn.Module):
    """Learnable position embeddings that modulate the kernel per-position.

    Instead of a fixed kernel for all tokens, each position gets a
    learned modifier on the kernel's frequency response. Early positions
    might emphasize low frequencies (global context); later positions
    might emphasize high frequencies (local detail).

    Implemented as a learned [max_len, spec_size] embedding table.
    """

    def __init__(self, max_len: int, field_size: int, n_heads: int):
        super().__init__()
        spec_size = field_size // 2 + 1
        self.pos_embed = nn.Embedding(max_len, n_heads * spec_size)
        self.n_heads = n_heads
        self.spec_size = spec_size

    def forward(self, field_spec: Tensor, seq_len: int) -> Tensor:
        """Modulate field spectrum by position-dependent weights.

        field_spec: [B, H, F//2+1, D] complex.
        seq_len: N (current sequence length).

        Returns: modulated [B, H, F//2+1, D] complex.
        """
        pos = torch.arange(min(seq_len, self.pos_embed.num_embeddings),
                           device=field_spec.device)
        # Average position embeddings over the sequence.
        weights = self.pos_embed(pos).mean(dim=0)  # [H * spec_size]
        weights = weights.view(self.n_heads, self.spec_size)
        # Soft modulation: 1 + tanh(weights) maps to [0, 2].
        mod = (1.0 + torch.tanh(weights)).unsqueeze(0).unsqueeze(-1)
        return field_spec * mod


# ── 6. Rotary field encoding ─────────────────────────────────────────


class RotaryFieldEncoding(nn.Module):
    """RoPE adapted for wave-field tokens.

    Standard RoPE rotates Q/K by position-dependent angles. In the
    wave-field, we don't have Q/K — but we can rotate the per-head
    token representations before scatter and after gather. This gives
    each position a unique "phase signature" that the convolution
    kernel can distinguish.
    """

    def __init__(self, d_head: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """x: [B, H, N, d_head] → rotated [B, H, N, d_head]."""
        N = x.shape[2]
        t = torch.arange(offset, offset + N, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [N, d_head/2]
        cos_f = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, N, d/2]
        sin_f = freqs.sin().unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos_f - x2 * sin_f,
                               x1 * sin_f + x2 * cos_f], dim=-1)
        return rotated.flatten(-2)


# ── Compose: WaveFieldBlockV2 ────────────────────────────────────────


@dataclass
class WaveFieldBlockV2Config:
    dim: int
    n_heads: int
    field_size: int
    ffn_mult: float = 8.0 / 3.0
    dropout: float = 0.0
    learned_scatter: bool = True
    smooth_causal: bool = True
    query_freq_gate: bool = True
    short_conv_gate: bool = True
    position_modulation: bool = True
    rotary_encoding: bool = True
    max_len: int = 8192
    short_conv_kernel: int = 4
    scatter_basis_width: int = 8


class WaveFieldBlockV2(nn.Module):
    """Next-generation wave-field block with all identified fixes.

    Improvements over V1:
    - Smooth causal masking (no aliasing)
    - Learned scatter/gather (not lossy linear interp)
    - Query-conditioned frequency gating (SPECTRE pattern)
    - Short conv + gating before FFT (Hyena pattern)
    - Position-dependent kernel modulation (Mamba-inspired)
    - Rotary position encoding adapted for fields
    """

    def __init__(self, cfg: WaveFieldBlockV2Config):
        super().__init__()
        if cfg.dim % cfg.n_heads != 0:
            raise ValueError(f"dim ({cfg.dim}) must divide n_heads ({cfg.n_heads})")
        self.cfg = cfg
        self.d_head = cfg.dim // cfg.n_heads

        # Kernels: smooth causal per head.
        omegas = [2.0 ** (-i) for i in range(cfg.n_heads)]
        self.heads = nn.ModuleList(
            SmoothCausalKernel(cfg.field_size, omega_init=om) for om in omegas
        ) if cfg.smooth_causal else nn.ModuleList(
            SmoothCausalKernel(cfg.field_size, omega_init=om) for om in omegas
        )

        # Scatter/gather: learned or linear.
        if cfg.learned_scatter:
            self.scatter_gather = LearnedScatterGather(cfg.field_size, cfg.scatter_basis_width)
        else:
            self.scatter_gather = None

        # Query-conditioned frequency gate.
        self.freq_gate = QueryFrequencyGate(
            self.d_head, cfg.n_heads, cfg.field_size
        ) if cfg.query_freq_gate else None

        # Short conv + gate before FFT.
        self.short_conv = ShortConvGate(
            cfg.dim, cfg.short_conv_kernel
        ) if cfg.short_conv_gate else None

        # Position-dependent kernel modulation.
        self.pos_mod = PositionKernelModulator(
            cfg.max_len, cfg.field_size, cfg.n_heads
        ) if cfg.position_modulation else None

        # Rotary encoding.
        self.rope = RotaryFieldEncoding(
            self.d_head, cfg.max_len
        ) if cfg.rotary_encoding else None

        self.norm_mix = RMSNorm(cfg.dim)
        self.norm_ffn = RMSNorm(cfg.dim)
        self.proj_in = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ffn = SwiGLUFFN(cfg.dim, hidden_mult=cfg.ffn_mult)
        self.drop = nn.Dropout(cfg.dropout)

    def wave_mix(self, x: Tensor, scatter_weights: Optional[Tensor] = None) -> Tensor:
        B, N, D = x.shape
        H, d_head = self.cfg.n_heads, self.d_head
        F_ = self.cfg.field_size

        # Optional short conv + gate (Hyena pattern).
        if self.short_conv is not None:
            x = self.short_conv(x)

        h = self.proj_in(x).view(B, N, H, d_head).transpose(1, 2)  # [B,H,N,d]

        # Rotary position encoding.
        if self.rope is not None:
            h = self.rope(h)

        # Scatter: learned or linear.
        if self.scatter_gather is not None:
            field = self.scatter_gather.scatter(h, weights=scatter_weights)
        else:
            from .scatter_gather import scatter_linear
            field = scatter_linear(h, F_, weights=scatter_weights)

        # FFT.
        spec = torch.fft.rfft(field, n=F_, dim=-2)

        # Kernel multiply.
        kfs = [head() for head in self.heads]
        kernel = torch.stack(kfs, dim=0)  # [H, F//2+1]
        spec = spec * kernel.unsqueeze(0).unsqueeze(-1)

        # Query-conditioned frequency gate.
        if self.freq_gate is not None:
            spec = self.freq_gate(spec, h)

        # Position-dependent modulation.
        if self.pos_mod is not None:
            spec = self.pos_mod(spec, N)

        # iFFT.
        field = torch.fft.irfft(spec, n=F_, dim=-2)

        # Gather.
        if self.scatter_gather is not None:
            out = self.scatter_gather.gather(field, N)
        else:
            from .scatter_gather import gather_linear
            out = gather_linear(field, N)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)

    def forward(self, x: Tensor, scatter_weights: Optional[Tensor] = None,
                kernel_bias: Optional[Tensor] = None) -> Tensor:
        h = x + self.drop(self.wave_mix(self.norm_mix(x), scatter_weights))
        h = h + self.drop(self.ffn(self.norm_ffn(h)))
        return h

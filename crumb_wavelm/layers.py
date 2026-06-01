"""Wave-Field building blocks: norm, FFN, single head, full block.

The block follows a standard pre-norm transformer-style sandwich, but
the mixing op is the wave-field convolution rather than dot-product
attention:

    h ← x + WaveMix(Norm(x))
    y ← h + FFN(Norm(h))

WaveMix:
    1. Project x → per-head feature slabs (linear, [B,N,D] → [B,H,N,d])
    2. Scatter onto field of size F
    3. FFT-convolve with per-head damped-cosine kernel
    4. Gather back to per-token states
    5. Concatenate heads and project out (linear, [B,N,D] → [B,N,D])

A causal mask is needed for autoregressive language modelling. Because
the convolution is global (the kernel touches both past and future
field cells), we enforce causality by zero-ing the *future* half of
the kernel in the time domain before transforming it. See
``CausalWaveFieldHead``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .kernels import wave_kernel_time, wave_kernel_freq, fft_convolve
from .scatter_gather import scatter_linear, gather_linear


# ── Norm + FFN ────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root-mean-square layernorm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 for stability, cast back to input dtype.
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward, matching Llama-style transformers.

    h = silu(W_g x) * W_u x;  out = W_d h
    """

    def __init__(self, dim: int, hidden_mult: float = 8.0 / 3.0, bias: bool = False):
        super().__init__()
        hidden = int(round(dim * hidden_mult / 64.0)) * 64
        hidden = max(hidden, 64)
        self.gate = nn.Linear(dim, hidden, bias=bias)
        self.up = nn.Linear(dim, hidden, bias=bias)
        self.down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Wave-Field heads ─────────────────────────────────────────────────


class WaveFieldHead(nn.Module):
    """One head's wave parameters and kernel construction.

    Holds three learnable scalars:
        alpha (damping ≥ 0, enforced via abs())
        omega (frequency)
        phi   (phase)

    ``forward`` returns the frequency-domain kernel ``[F//2+1]``, cached
    per-step. The actual convolution lives in ``WaveFieldBlock`` so all
    heads' kernels can be stacked into a single FFT call.
    """

    def __init__(
        self,
        field_size: int,
        alpha_init: float = 0.05,
        omega_init: float = 1.0,
        phi_init: float = 0.0,
        kernel_mode: str = "freq",
        causal: bool = True,
    ):
        super().__init__()
        if kernel_mode not in ("freq", "time"):
            raise ValueError(f"kernel_mode must be 'freq' or 'time', got {kernel_mode!r}")
        self.field_size = field_size
        self.kernel_mode = kernel_mode
        self.causal = causal
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.omega = nn.Parameter(torch.tensor(float(omega_init)))
        self.phi = nn.Parameter(torch.tensor(float(phi_init)))

    def kernel_freq(self) -> Tensor:
        if self.kernel_mode == "freq" and not self.causal:
            # Closed-form spectrum, no FFT needed.
            return wave_kernel_freq(self.alpha, self.omega, self.phi, self.field_size)
        # Causal heads must zero the negative-t half of the kernel, which
        # is only possible in the time domain — then we rFFT it.
        k_t = wave_kernel_time(self.alpha, self.omega, self.phi, self.field_size)
        if self.causal:
            # Time grid is centred at F//2 — that's t=0. Zero out t<0.
            mask = torch.zeros_like(k_t)
            mask[..., self.field_size // 2 :] = 1.0
            k_t = k_t * mask
        # ifftshift so t=0 sits at index 0 (rFFT convention).
        k_t = torch.fft.ifftshift(k_t, dim=-1)
        return torch.fft.rfft(k_t, n=self.field_size, dim=-1)


# ── Full Wave-Field block ────────────────────────────────────────────


@dataclass
class WaveFieldBlockConfig:
    dim: int
    n_heads: int
    field_size: int
    ffn_mult: float = 8.0 / 3.0
    causal: bool = True
    kernel_mode: str = "freq"
    dropout: float = 0.0
    boundary: str = "periodic"       # periodic | absorbing | reflecting
    dispersion: bool = False         # add learnable frequency-dependent phase shift
    interference_mixer: bool = False # replace proj_out with complex interference
    adaptive_kernels: bool = False   # input-conditioned α/ω/φ
    spectral_gate: bool = False      # learned per-frequency gate
    resonance_memory: bool = False   # cross-layer standing-wave amplification
    hybrid_gate: bool = False        # blend wave-field + local attention
    hybrid_window: int = 64          # window size for hybrid local attention


class WaveFieldBlock(nn.Module):
    """Pre-norm wave-field block: x → x + WaveMix(N(x)) → + FFN(N(·)).

    Per-head feature dim is ``dim // n_heads``. The block's mixing cost
    is O(B · H · D_head · F log F) plus two N×D projections — i.e. the
    sequence length N only appears in the (cheap) scatter/gather.
    """

    def __init__(self, cfg: WaveFieldBlockConfig):
        super().__init__()
        if cfg.dim % cfg.n_heads != 0:
            raise ValueError(
                f"dim ({cfg.dim}) must be divisible by n_heads ({cfg.n_heads})"
            )
        self.cfg = cfg
        self.d_head = cfg.dim // cfg.n_heads

        # Per-head wave params; we stack their kernels into [H, F//2+1].
        # Initialise omegas to span a range of spatial scales (one octave
        # per head, capped); damping starts small so kernels are wide.
        omegas_init = [2.0 ** (-i) for i in range(cfg.n_heads)]
        self.heads = nn.ModuleList(
            WaveFieldHead(
                field_size=cfg.field_size,
                alpha_init=0.05,
                omega_init=om,
                phi_init=0.0,
                kernel_mode=cfg.kernel_mode,
                causal=cfg.causal,
            )
            for om in omegas_init
        )

        self.norm_mix = RMSNorm(cfg.dim)
        self.norm_ffn = RMSNorm(cfg.dim)
        self.proj_in = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ffn = SwiGLUFFN(cfg.dim, hidden_mult=cfg.ffn_mult)
        self.drop = nn.Dropout(cfg.dropout)

        # Advanced physics (all optional, off by default).
        from .physics import BoundaryCondition, DispersionLayer, InterferenceMixer
        self.bc = BoundaryCondition(cfg.boundary)
        self.dispersion = DispersionLayer(cfg.n_heads) if cfg.dispersion else None
        if cfg.interference_mixer:
            self.mixer = InterferenceMixer(cfg.n_heads)
            self.proj_out = nn.Identity()
        else:
            self.mixer = None
            self.proj_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

        # Novel innovations (Crumb LLM originals).
        from .v2 import AdaptiveKernelHead, SpectralGate, ResonanceMemory, FieldAttentionGate
        self.adaptive_heads = nn.ModuleList(
            AdaptiveKernelHead(cfg.dim, cfg.field_size, causal=cfg.causal)
            for _ in range(cfg.n_heads)
        ) if cfg.adaptive_kernels else None
        self.spectral_gate = SpectralGate(cfg.n_heads, cfg.field_size, cfg.dim) if cfg.spectral_gate else None
        self.resonance = ResonanceMemory(cfg.n_heads, cfg.field_size) if cfg.resonance_memory else None
        self.hybrid_gate = FieldAttentionGate(cfg.dim, window_size=cfg.hybrid_window) if cfg.hybrid_gate else None

    # ── Wave mixing op ───────────────────────────────────────────────

    def wave_mix(
        self,
        x: Tensor,
        scatter_weights: Optional[Tensor] = None,
        kernel_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """x: [B, N, D] → mixed [B, N, D]."""
        B, N, D = x.shape
        H, d_head = self.cfg.n_heads, self.d_head
        F_ = self.cfg.field_size

        # Project, then reshape to per-head slabs.
        h = self.proj_in(x).view(B, N, H, d_head).transpose(1, 2)  # [B,H,N,d]

        # Build kernels — static or adaptive.
        if self.adaptive_heads is not None:
            # Adaptive: kernels are input-conditioned [B, F//2+1] per head.
            kfs = [ah(x) for ah in self.adaptive_heads]   # list of [B, F//2+1]
            kernel = torch.stack(kfs, dim=1)               # [B, H, F//2+1]
        else:
            kfs = [head.kernel_freq() for head in self.heads]
            kernel = torch.stack(kfs, dim=0)               # [H, F//2+1]

        if kernel_bias is not None:
            kernel = kernel + kernel_bias

        field = scatter_linear(h, F_, weights=scatter_weights)      # [B,H,F,d]

        from .physics import apply_boundary, unapply_reflecting, BoundaryCondition
        taper_w = max(1, F_ // 16)
        field = apply_boundary(field, self.bc, taper_width=taper_w)
        actual_F = field.shape[-2]

        # FFT convolve.
        spec = torch.fft.rfft(field, n=actual_F, dim=-2)
        if kernel.dim() == 2:
            # Static kernel: [H, F//2+1] → broadcast over batch.
            kf = kernel.unsqueeze(0).unsqueeze(-1)
        else:
            # Adaptive kernel: [B, H, F//2+1] → [B, H, F//2+1, 1].
            kf = kernel.unsqueeze(-1)
        # Pad kernel if reflecting BC expanded the field.
        if kf.shape[-2] != spec.shape[-2]:
            kf_full = torch.zeros(*kf.shape[:-2], spec.shape[-2], kf.shape[-1],
                                  dtype=kf.dtype, device=kf.device)
            n_copy = min(kf.shape[-2], spec.shape[-2])
            kf_full[..., :n_copy, :] = kf[..., :n_copy, :]
            kf = kf_full
        spec = spec * kf
        if self.dispersion is not None:
            spec = self.dispersion(spec, actual_F)
        field = torch.fft.irfft(spec, n=actual_F, dim=-2)

        if self.bc == BoundaryCondition.REFLECTING:
            field = unapply_reflecting(field, taper_width=taper_w)

        # Spectral gate (input-conditioned frequency filtering).
        if self.spectral_gate is not None:
            x_pooled = x.mean(dim=1)  # [B, D]
            field = self.spectral_gate(field, x_pooled)

        out = gather_linear(field, N)                               # [B,H,N,d]

        # Interference mixing (if enabled) or standard projection.
        if self.mixer is not None:
            out = self.mixer(out)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)

    # ── Full block forward ───────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        scatter_weights: Optional[Tensor] = None,
        kernel_bias: Optional[Tensor] = None,
        resonance_state: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        wave_out = self.wave_mix(self.norm_mix(x), scatter_weights, kernel_bias)

        # Hybrid gate: blend wave-field with local attention.
        if self.hybrid_gate is not None:
            wave_out = self.hybrid_gate(self.norm_mix(x), wave_out)

        h = x + self.drop(wave_out)
        h = h + self.drop(self.ffn(self.norm_ffn(h)))

        # Resonance memory: update and return new state if enabled.
        if self.resonance is not None and resonance_state is not None:
            # Reconstruct the field for resonance (re-scatter the output).
            H, d_head = self.cfg.n_heads, self.d_head
            h_proj = self.proj_in(self.norm_mix(h)).view(
                h.shape[0], h.shape[1], H, d_head
            ).transpose(1, 2)
            field_for_res = scatter_linear(h_proj, self.cfg.field_size)
            new_state, _ = self.resonance(resonance_state, field_for_res)
            return h, new_state

        return h

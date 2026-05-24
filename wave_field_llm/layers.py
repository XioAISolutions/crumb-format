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
        self.proj_out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ffn = SwiGLUFFN(cfg.dim, hidden_mult=cfg.ffn_mult)
        self.drop = nn.Dropout(cfg.dropout)

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

        # Stack each head's [F//2+1] kernel → [H, F//2+1].
        kfs = [head.kernel_freq() for head in self.heads]
        kernel = torch.stack(kfs, dim=0)                            # [H, F//2+1]

        # Optional per-head additive bias on the kernel (used by the
        # crumb-aware adapter to bias damping per request).
        if kernel_bias is not None:
            kernel = kernel + kernel_bias

        field = scatter_linear(h, F_, weights=scatter_weights)      # [B,H,F,d]
        field = fft_convolve(field, kernel, F_)                     # [B,H,F,d]
        out = gather_linear(field, N)                               # [B,H,N,d]

        # Re-merge heads, project out.
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_out(out)

    # ── Full block forward ───────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        scatter_weights: Optional[Tensor] = None,
        kernel_bias: Optional[Tensor] = None,
    ) -> Tensor:
        h = x + self.drop(
            self.wave_mix(self.norm_mix(x), scatter_weights, kernel_bias)
        )
        h = h + self.drop(self.ffn(self.norm_ffn(h)))
        return h

"""Damped-cosine wave kernels and FFT convolution.

The Wave-Field layer convolves the scattered field with a per-head kernel

    k(t) = exp(-α |t|) · cos(ω t + φ),   t ∈ [-F/2, F/2)

parameterised by three learnable scalars (α, ω, φ). Convolution is done
in the frequency domain via real-input FFT — the only reason the whole
architecture is O(F log F) rather than O(F^2).

Two construction paths are provided:

* ``wave_kernel_time``  — build k(t) on a fixed grid, then rFFT it.
  Useful for inspection, debugging, and the numerical equivalence test
  (FFT-convolve vs. direct convolution on small inputs).

* ``wave_kernel_freq``  — construct K(ω) directly in the frequency domain
  as a closed-form Lorentzian-pair (the analytic Fourier transform of
  the damped cosine on the infinite line). Skips one FFT per forward.

Both return ``[F//2 + 1]`` complex tensors (the rFFT of a length-F real
signal), broadcastable against the rFFT of the scattered field along the
spatial axis.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ── Time-domain construction ─────────────────────────────────────────


def wave_kernel_time(
    alpha: Tensor,
    omega: Tensor,
    phi: Tensor,
    field_size: int,
    dt: float = 1.0,
) -> Tensor:
    """Construct k(t) = exp(-α|t|) cos(ω t + φ) sampled on a centred grid.

    Args:
        alpha:  positive damping coefficient, shape ``[...]``
        omega:  angular frequency, shape ``[...]``
        phi:    phase, shape ``[...]``
        field_size: number of samples F (typically a power of two).
        dt:     sample spacing.

    Returns:
        Tensor of shape ``[..., F]`` with the kernel sampled at
        t = (-F/2, ..., F/2 - 1) · dt. The grid is centred so that the
        kernel's peak sits at t=0 (for φ=0), which is the convention
        ``torch.fft.rfft`` expects for *causal-like, zero-centred* kernels
        when paired with the FFT-shift trick in ``fft_convolve``.
    """
    F = field_size
    device = alpha.device
    dtype = alpha.dtype
    # Centre the grid so the peak is at index F//2; we ifftshift before rfft.
    t = (torch.arange(F, device=device, dtype=dtype) - F // 2) * dt
    # Broadcast: t is [F], scalars are [...]; result is [..., F].
    a = alpha.abs().unsqueeze(-1)
    w = omega.unsqueeze(-1)
    p = phi.unsqueeze(-1)
    return torch.exp(-a * t.abs()) * torch.cos(w * t + p)


# ── Frequency-domain construction ────────────────────────────────────


def wave_kernel_freq(
    alpha: Tensor,
    omega: Tensor,
    phi: Tensor,
    field_size: int,
    dt: float = 1.0,
) -> Tensor:
    """Analytic rFFT of exp(-α|t|) cos(ω t + φ) on the discrete grid.

    The continuous Fourier transform of exp(-α|t|) cos(ω t + φ) is a sum
    of two shifted Lorentzians:

        K(ξ) = ½ e^{ iφ} · 2α / (α² + (ξ - ω)²)
             + ½ e^{-iφ} · 2α / (α² + (ξ + ω)²)

    We sample ξ at the rFFT bins of a length-F real signal with sample
    spacing ``dt`` and return the resulting complex spectrum. This avoids
    one FFT per forward (we only inverse-FFT after multiplication).

    Args:
        alpha, omega, phi: shape ``[...]``.
        field_size:        F.
        dt:                sample spacing.

    Returns:
        Tensor of shape ``[..., F//2 + 1]``, complex dtype.
    """
    F = field_size
    device = alpha.device
    real_dtype = alpha.dtype
    # rFFT bin frequencies in radians per unit t.
    # torch.fft.rfftfreq returns cycles per sample; multiply by 2π / dt.
    xi = torch.fft.rfftfreq(F, d=dt, device=device, dtype=real_dtype) * (2 * math.pi)
    a = alpha.abs().unsqueeze(-1)
    w = omega.unsqueeze(-1)
    p = phi.unsqueeze(-1)
    # Two Lorentzian lobes around ±ω, phase-rotated by ±φ.
    j = torch.complex(torch.zeros_like(p), p)
    pos = torch.exp(j) * (a / (a * a + (xi - w) ** 2))
    neg = torch.exp(-j) * (a / (a * a + (xi + w) ** 2))
    return pos + neg


# ── FFT convolution ──────────────────────────────────────────────────


def fft_convolve(
    field: Tensor,
    kernel_freq: Tensor,
    field_size: int,
) -> Tensor:
    """Circular convolve along the field axis (penultimate axis).

    Args:
        field:        ``[B, H, F, D]`` real tensor (B batch, H heads,
                      F field positions, D per-head feature dim).
        kernel_freq:  ``[H, F//2 + 1]`` complex spectrum — broadcastable
                      against rFFT(field) along the F axis.
        field_size:   F, passed explicitly so irfft restores the exact
                      length (rfft → irfft is otherwise length-ambiguous
                      for odd F).

    Returns:
        Real tensor of shape ``[B, H, F, D]``.

    Notes:
        We rFFT the field along ``dim=-2``. The spectrum is multiplied by
        ``kernel_freq[None, :, :, None]`` (broadcasting across batch and
        feature axes). The result is then iRFFT'd back. The whole op is
        O(B · H · D · F log F).
    """
    if field.shape[-2] != field_size:
        raise ValueError(
            f"field axis -2 has size {field.shape[-2]}, expected {field_size}"
        )
    # rFFT along the field axis.
    spec = torch.fft.rfft(field, n=field_size, dim=-2)
    # Broadcast kernel: [H, F//2+1] → [1, H, F//2+1, 1].
    kf = kernel_freq.unsqueeze(0).unsqueeze(-1)
    out = torch.fft.irfft(spec * kf, n=field_size, dim=-2)
    return out

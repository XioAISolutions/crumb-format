"""Advanced wave-equation physics for Crumb LLM.

Extends the base damped-cosine kernel with richer physics:

* **Dispersion** — frequency-dependent wave speed.  Physical waves
  disperse (high-frequency components travel faster or slower than low).
  We add a learnable dispersion coefficient β that modulates the kernel
  in the frequency domain:  K'(ξ) = K(ξ) · exp(i β ξ²).

* **Boundary conditions** — periodic (default, implicit in FFT),
  absorbing (taper the field edges to zero), or reflecting (mirror-pad
  the field before convolving, then un-mirror).

* **Gabor wavelet heads** — replace the infinite damped cosine with a
  localised Gabor wavelet (Gaussian-windowed sinusoid).  This gives
  heads spatial *and* frequency selectivity simultaneously, analogous
  to time-frequency atoms in signal processing.

* **Interference mixer** — after the per-head wave-field gather, mix
  heads via learned complex coupling coefficients so standing-wave and
  interference patterns emerge.  This is a lightweight replacement for
  the output projection that preserves phase information.

* **Resonance detector** — an analysis tool (not in the forward path)
  that identifies standing-wave patterns in the field by looking for
  peaks in the spatial spectrum.  Useful for interpretability.
"""

from __future__ import annotations

import math
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor


# ── Boundary conditions ──────────────────────────────────────────────


class BoundaryCondition(Enum):
    PERIODIC = "periodic"       # default: FFT wraps around
    ABSORBING = "absorbing"     # taper edges to zero (no reflections)
    REFLECTING = "reflecting"   # mirror-pad before FFT


def apply_boundary(field: Tensor, bc: BoundaryCondition, taper_width: int = 16) -> Tensor:
    """Apply boundary condition to a field tensor ``[B, H, F, D]``."""
    if bc == BoundaryCondition.PERIODIC:
        return field
    if bc == BoundaryCondition.ABSORBING:
        F = field.shape[-2]
        tw = min(taper_width, F // 4)
        taper = torch.ones(F, device=field.device, dtype=field.dtype)
        ramp = torch.linspace(0.0, 1.0, tw, device=field.device, dtype=field.dtype)
        taper[:tw] = ramp
        taper[-tw:] = ramp.flip(0)
        return field * taper.view(1, 1, F, 1)
    if bc == BoundaryCondition.REFLECTING:
        # Mirror-pad: reflect the edges so the FFT "sees" a symmetric signal.
        F = field.shape[-2]
        tw = min(taper_width, F // 4)
        left = field[:, :, :tw, :].flip(-2)
        right = field[:, :, -tw:, :].flip(-2)
        padded = torch.cat([left, field, right], dim=-2)
        return padded  # caller must un-pad after convolving
    return field


def unapply_reflecting(field: Tensor, taper_width: int = 16) -> Tensor:
    """Remove mirror-padding added by absorbing BC."""
    tw = taper_width
    return field[:, :, tw:-tw, :]


# ── Dispersion ───────────────────────────────────────────────────────


class DispersionLayer(nn.Module):
    """Frequency-dependent phase shift applied in the spectral domain.

    Models wave dispersion: different frequency components travel at
    different speeds. The dispersion relation is ξ → β·ξ² (quadratic,
    the simplest non-trivial dispersion).

    Applied *after* the kernel multiply and *before* iRFFT:
        spec' = spec · exp(i β ξ²)
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(n_heads))

    def forward(self, spec: Tensor, field_size: int) -> Tensor:
        """spec: [B, H, F//2+1, D] complex."""
        xi = torch.fft.rfftfreq(field_size, device=spec.device, dtype=self.beta.dtype)
        xi2 = xi * xi * (2 * math.pi) ** 2
        # Phase shift per head: [H, F//2+1] → [1, H, F//2+1, 1]
        phase = self.beta.unsqueeze(-1) * xi2.unsqueeze(0)
        shift = torch.polar(torch.ones_like(phase), phase)
        return spec * shift.unsqueeze(0).unsqueeze(-1)


# ── Gabor wavelet head ───────────────────────────────────────────────


class GaborWaveletKernel(nn.Module):
    """Gabor wavelet: Gaussian-windowed sinusoid.

    k(t) = exp(-t² / (2σ²)) · cos(ω t + φ)

    Unlike the damped cosine (exponential decay), the Gaussian envelope
    gives finite support in *both* time and frequency — a true
    time-frequency atom. The learnable width σ controls the trade-off
    between spatial precision and frequency precision (uncertainty
    principle).

    Parameters:
        sigma:  Gaussian width (spatial spread)
        omega:  centre frequency
        phi:    phase
    """

    def __init__(self, field_size: int, sigma_init: float = 8.0,
                 omega_init: float = 1.0, phi_init: float = 0.0,
                 causal: bool = True):
        super().__init__()
        self.field_size = field_size
        self.causal = causal
        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        self.omega = nn.Parameter(torch.tensor(float(omega_init)))
        self.phi = nn.Parameter(torch.tensor(float(phi_init)))
        # Pre-compute centred time grid.
        t = torch.arange(field_size, dtype=torch.float32) - field_size // 2
        self.register_buffer("t", t)

    def forward(self) -> Tensor:
        """Return rFFT of the Gabor kernel, ready for spectral multiply."""
        s = self.sigma.abs().clamp(min=0.5)
        gauss = torch.exp(-self.t ** 2 / (2 * s * s))
        wave = torch.cos(self.omega * self.t + self.phi)
        k = gauss * wave
        if self.causal:
            mask = torch.zeros_like(k)
            mask[self.field_size // 2:] = 1.0
            k = k * mask
        k = torch.fft.ifftshift(k, dim=-1)
        return torch.fft.rfft(k, n=self.field_size, dim=-1)


# ── Interference mixer ───────────────────────────────────────────────


class InterferenceMixer(nn.Module):
    """Mix heads via learned complex coupling — preserves phase info.

    Instead of a simple linear out-projection that discards the wave
    structure, this mixer lets heads interfere: the output of head j at
    position p is a complex-weighted sum of all heads' outputs at p.

    h_out[j] = Σ_k  c_{jk} · h_in[k]

    where c_{jk} = |c_{jk}| · exp(i θ_{jk}) are learned complex weights.
    The result is projected back to real via taking the real part (or
    modulus, configurable).

    This is a lightweight H × H mixing (not D × D), so cost is negligible.
    """

    def __init__(self, n_heads: int, output_mode: str = "real"):
        super().__init__()
        if output_mode not in ("real", "modulus"):
            raise ValueError(f"output_mode must be 'real' or 'modulus', got {output_mode!r}")
        self.output_mode = output_mode
        self.magnitude = nn.Parameter(torch.eye(n_heads))
        self.phase = nn.Parameter(torch.zeros(n_heads, n_heads))

    def forward(self, heads: Tensor) -> Tensor:
        """heads: [B, H, N, d_head] real → [B, H, N, d_head] real."""
        # Build complex coupling matrix [H, H].
        C = torch.polar(self.magnitude.abs(), self.phase)
        # heads to complex: treat as real part, imag = 0.
        h_c = torch.complex(heads, torch.zeros_like(heads))
        # Mix: [H_out, H_in] × [B, H_in, N, d] → [B, H_out, N, d]
        mixed = torch.einsum("jk, bknD -> bjnD", C, h_c)
        if self.output_mode == "real":
            return mixed.real
        return mixed.abs()


# ── Multi-scale field (different F per head group) ────────────────────


class MultiScaleScatterGather(nn.Module):
    """Groups heads into scales with different field sizes.

    Example: 8 heads → [4 heads at F=256, 2 at F=512, 2 at F=1024].
    Each group independently scatters, convolves, and gathers at its
    own resolution. Short-range heads use small F (cheap); long-range
    heads use large F (more expensive but reach further). After gathering
    back to per-token states, all groups concatenate along the head dim.
    """

    def __init__(self, scales: list[tuple[int, int]]):
        """scales: [(n_heads_1, field_size_1), (n_heads_2, field_size_2), ...]."""
        super().__init__()
        self.scales = scales

    @property
    def total_heads(self) -> int:
        return sum(h for h, _ in self.scales)

    @property
    def field_sizes(self) -> list[int]:
        return [f for _, f in self.scales]


# ── Resonance analysis (interpretability) ─────────────────────────────


@torch.no_grad()
def detect_resonances(
    field: Tensor,
    top_k: int = 5,
) -> list[dict]:
    """Identify standing-wave patterns in a field tensor.

    Standing waves appear as sharp peaks in the spatial power spectrum.
    We rFFT along the field axis, take the power (|X(ξ)|²), and report
    the top-k frequency bins.

    Args:
        field: [F, D] or [B, H, F, D].
        top_k: number of peaks to return.

    Returns:
        list of dicts with keys (bin, frequency, power, wavelength).
    """
    if field.dim() == 2:
        field = field.unsqueeze(0).unsqueeze(0)
    elif field.dim() == 3:
        field = field.unsqueeze(0)
    # Average across B, H, D to get a spatial power spectrum.
    spec = torch.fft.rfft(field, dim=-2)
    power = (spec.abs() ** 2).mean(dim=(0, 1, 3))  # [F//2+1]
    F = field.shape[-2]
    vals, idxs = torch.topk(power, min(top_k, power.numel()))
    results = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        freq = i / F
        wlen = F / max(i, 1)
        results.append({
            "bin": i,
            "frequency": round(freq, 4),
            "power": round(v, 4),
            "wavelength": round(wlen, 2),
        })
    return results

"""crumb_coherence.metrics — the numbers M0/M1 are judged on (§6).

* lowband_trajectory_variance : how much the low band wanders over time (drift).
* delta_e_vs_ref              : perceptual color drift (CIEDE2000 on frame means).
* highfreq_ssim               : did the correction harm detail? (must stay high).
* temporal_flicker            : short-time frame-diff energy.

All take frames shaped [T,3,H,W] in [0,1]. Returns are plain floats or [T]
tensors so a harness can take means / percentage drops directly.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .core import low_band_box

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Drift magnitude
# --------------------------------------------------------------------------- #
def lowband_trajectory_variance(frames: torch.Tensor, cutoff_frac: float) -> float:
    """Mean over low-band cells of their temporal variance (magnitude spectrum).

    A steady scene has a near-constant low-band magnitude; drift makes it wander,
    inflating this number. Anchoring should shrink it.
    """
    mags = []
    for t in range(frames.shape[0]):
        X = torch.fft.rfft2(frames[t])            # [3,H,Wf]
        mags.append(low_band_box(X, cutoff_frac).abs())   # [3,kh,kw]
    traj = torch.stack(mags, 0)                   # [T,3,kh,kw]
    var = traj.var(dim=0, unbiased=False)         # per-cell temporal variance
    return float(var.mean().item())


# --------------------------------------------------------------------------- #
# Color drift — CIEDE2000 on the per-frame mean color vs a reference frame.
# --------------------------------------------------------------------------- #
def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb [...,3] in [0,1] -> Lab [...,3] (D65)."""
    lin = _srgb_to_linear(rgb.clamp(0, 1))
    r, g, b = lin.unbind(-1)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # D65 reference white
    x = x / 0.95047
    z = z / 1.08883

    def f(t):
        d = 6.0 / 29.0
        return torch.where(t > d ** 3, t.clamp(min=EPS) ** (1.0 / 3.0),
                           t / (3 * d * d) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return torch.stack((L, a, bb), -1)


def _ciede2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """Batched CIEDE2000 between two [...,3] Lab tensors -> [...] deltaE."""
    L1, a1, b1 = lab1.unbind(-1)
    L2, a2, b2 = lab2.unbind(-1)
    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - torch.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7 + EPS)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1)
    C2p = torch.sqrt(a2p * a2p + b2 * b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = torch.where(dhp > 180, dhp - 360, dhp)
    dhp = torch.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dhp) / 2)

    Lbar = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbar = torch.where((h1p - h2p).abs() > 180, (hsum + 360) / 2, hsum / 2)
    T = (1 - 0.17 * torch.cos(torch.deg2rad(hbar - 30))
         + 0.24 * torch.cos(torch.deg2rad(2 * hbar))
         + 0.32 * torch.cos(torch.deg2rad(3 * hbar + 6))
         - 0.20 * torch.cos(torch.deg2rad(4 * hbar - 63)))
    dtheta = 30 * torch.exp(-(((hbar - 275) / 25) ** 2))
    Rc = 2 * torch.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7 + EPS))
    Sl = 1 + (0.015 * (Lbar - 50) ** 2) / torch.sqrt(20 + (Lbar - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -torch.sin(torch.deg2rad(2 * dtheta)) * Rc
    return torch.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                      + Rt * (dCp / Sc) * (dHp / Sh))


def delta_e_vs_ref(frames: torch.Tensor, ref_idx: int = 0) -> torch.Tensor:
    """Per-frame CIEDE2000 of mean color vs frame `ref_idx`. Returns [T]."""
    means = frames.mean(dim=(-1, -2))             # [T,3]
    lab = _rgb_to_lab(means)                       # [T,3]
    ref = lab[ref_idx:ref_idx + 1]                 # [1,3]
    return _ciede2000(lab, ref.expand_as(lab))     # [T]


# --------------------------------------------------------------------------- #
# Detail preservation — SSIM on the high-pass residual.
# --------------------------------------------------------------------------- #
def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x.unbind(-3)
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(-3)   # [...,1,H,W]


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    rad = max(1, int(round(3 * sigma)))
    xs = torch.arange(-rad, rad + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    return k / k.sum()


def _blur(img: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    # img [N,1,H,W] separable Gaussian blur with reflect padding.
    k = _gaussian_kernel(sigma, img.device)
    r = k.numel() // 2
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    img = F.pad(img, (r, r, 0, 0), mode="reflect")
    img = F.conv2d(img, kx)
    img = F.pad(img, (0, 0, r, r), mode="reflect")
    img = F.conv2d(img, ky)
    return img


def _ssim(a: torch.Tensor, b: torch.Tensor, win: int = 11) -> torch.Tensor:
    """Windowed SSIM between [N,1,H,W] maps -> [N] mean SSIM."""
    pad = win // 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = F.avg_pool2d(a, win, 1, pad)
    mu_b = F.avg_pool2d(b, win, 1, pad)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = F.avg_pool2d(a * a, win, 1, pad) - mu_a2
    vb = F.avg_pool2d(b * b, win, 1, pad) - mu_b2
    vab = F.avg_pool2d(a * b, win, 1, pad) - mu_ab
    s = ((2 * mu_ab + C1) * (2 * vab + C2)) / (
        (mu_a2 + mu_b2 + C1) * (va + vb + C2))
    return s.mean(dim=(-1, -2, -3))


def highfreq_ssim(frames_in: torch.Tensor, frames_out: torch.Tensor) -> float:
    """Mean SSIM of the high-pass (detail) residual, in vs out. ~1 = detail kept."""
    li = _rgb_to_luma(frames_in)                   # [T,1,H,W]
    lo = _rgb_to_luma(frames_out)
    hi = li - _blur(li)
    ho = lo - _blur(lo)
    return float(_ssim(hi, ho).mean().item())


# --------------------------------------------------------------------------- #
# Temporal flicker
# --------------------------------------------------------------------------- #
def temporal_flicker(frames: torch.Tensor) -> float:
    """Mean squared consecutive-frame difference energy."""
    if frames.shape[0] < 2:
        return 0.0
    d = frames[1:] - frames[:-1]
    return float((d * d).mean().item())

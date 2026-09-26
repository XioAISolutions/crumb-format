"""crumb_coherence.core — training-free low-band spectral coherence for long video.

Implements PLUGIN_SPEC_crumb_coherence.md §1, §2, §6. The engine holds a tiny
rolling anchor over the low-frequency FFT box of the stream and softly blends
each incoming frame toward it, killing global/low-band drift (exposure, color
cast, large-scale layout) while leaving motion (low-band *phase*) free.

The recurrence `anchor <- rho*anchor + (1-rho)*A(Xlo)` is literally the
`z_t = lambda*z_{t-1} + B*x_t` state of wfvideo.py, degenerated to one mode per
low-k cell (rho == lambda). That is the honest link to the research model; the
plugin does not import it.

Conventions
-----------
* Frames are float tensors in [0,1], shape [C,H,W] (C=3) or [T,C,H,W].
* rfft2 gives X of shape [C,H,Wf] with Wf = W//2+1. Width holds only the
  non-negative frequencies (0..W//2), so its low band is a plain [:kw] slice.
  Height holds the full spectrum (0,1,..,-1), so its low band wraps around: we
  `fftshift` the height axis, take a centered block, and `ifftshift` back. After
  the shift, DC sits at box row `kh//2`, box col `0`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

EPS = 1e-8
CUT_COOLDOWN_N = 5      # frames of fast (rho=0.5) adaptation after a soft cut
CUT_RHO = 0.5


# --------------------------------------------------------------------------- #
# Color space (§3.1) — decorrelate exposure (Y) from color (Cb,Cr).
# BT.601 full-range; chroma carries a +0.5 offset so everything stays in [0,1].
# Operates on [...,3,H,W]; the round-trip is exact up to float32 rounding.
# --------------------------------------------------------------------------- #
def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x.unbind(dim=-3)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack((y, cb, cr), dim=-3)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y, cb, cr = x.unbind(dim=-3)
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), dim=-3)


# --------------------------------------------------------------------------- #
# Low-band box crop / write-back (§2 steps 2 & 5)
# --------------------------------------------------------------------------- #
def box_dims(cutoff_frac: float, H: int, W: int) -> tuple[int, int]:
    """Box half-extents in cells: kh ~= cutoff_frac*H, kw ~= cutoff_frac*W."""
    kh = max(1, int(round(cutoff_frac * H)))
    kw = max(1, int(round(cutoff_frac * W)))
    return kh, kw


def _crop_box(X: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    """Centered [C,kh,kw] low-band block of an rfft2 tensor X:[C,H,Wf]."""
    H = X.shape[-2]
    Xs = torch.fft.fftshift(X, dim=-2)
    r0 = H // 2 - kh // 2
    return Xs[..., r0:r0 + kh, :kw]


def low_band_box(X: torch.Tensor, cutoff_frac: float) -> torch.Tensor:
    """Crop the rfft2 tensor to its lowest-frequency [C,kh,kw] box.

    Assumes an even width (W = 2*(Wf-1)); the moving-ball clips here are even.
    """
    H, Wf = X.shape[-2], X.shape[-1]
    W = 2 * (Wf - 1)
    kh, kw = box_dims(cutoff_frac, H, W)
    return _crop_box(X, kh, kw)


def put_low_band(X: torch.Tensor, Xlo: torch.Tensor) -> torch.Tensor:
    """Write the [C,kh,kw] low-band box back into a copy of X, out-of-place.

    High-frequency coefficients of X are untouched, so irfft2 reconstructs them
    to float precision — the plugin only ever moves the low band.
    """
    H = X.shape[-2]
    kh, kw = Xlo.shape[-2], Xlo.shape[-1]
    Xs = torch.fft.fftshift(X, dim=-2).clone()
    r0 = H // 2 - kh // 2
    Xs[..., r0:r0 + kh, :kw] = Xlo
    return torch.fft.ifftshift(Xs, dim=-2)


# --------------------------------------------------------------------------- #
# Gaussian band weight (§3.3) — 1 at DC, tapering to ~0 by the band edge.
# DC lives at box (kh//2, 0). sigma is expressed as a fraction of Nyquist:
# with kh = cutoff_frac*H the Nyquist is kh/(2*cutoff_frac) cells, so
# sigma_cells = sigma_k_frac * kh / (2*cutoff_frac). cutoff_frac is threaded in
# (a keyword beyond the four spec params) to keep that mapping physical.
# --------------------------------------------------------------------------- #
def gaussian_band_weight(kh: int, kw: int, sigma_k_frac: float,
                         device="cpu", cutoff_frac: float = 0.10) -> torch.Tensor:
    dc_row = kh // 2
    sig_h = max(EPS, sigma_k_frac * kh / (2.0 * cutoff_frac))
    sig_w = max(EPS, sigma_k_frac * kw / (2.0 * cutoff_frac))
    i = torch.arange(kh, device=device, dtype=torch.float32).view(kh, 1)
    j = torch.arange(kw, device=device, dtype=torch.float32).view(1, kw)
    dy = (i - dc_row) / sig_h
    dx = j / sig_w                      # DC column is 0
    return torch.exp(-0.5 * (dy * dy + dx * dx))   # [kh,kw], w(DC)=1


# --------------------------------------------------------------------------- #
# Phase-anchoring band weight (§3.2, M0.2). The Gaussian above is DC-centric —
# right for exposure/color (magnitude) drift, wrong for *positional* drift. A
# wandering low-freq structure shifts by p(t): by the shift theorem its low-band
# coefficients are Â_k·exp(-i k·p(t)) — magnitude ~constant, position living
# entirely in the phase φ_k = -k·p(t), which GROWS with k. So position lives in
# the non-DC cells (and most strongly at higher k), exactly where the Gaussian
# taper is weakest. This companion weight is FLAT across the low band and ZERO at
# DC (DC carries no position, and re-anchoring it is just more magnitude locking),
# so phase_anchor>0 pulls the position-carrying cells toward the (near-static)
# anchor without touching the exposure lock.
# --------------------------------------------------------------------------- #
def phase_band_weight(kh: int, kw: int, device="cpu") -> torch.Tensor:
    wp = torch.ones(kh, kw, device=device, dtype=torch.float32)
    wp[kh // 2, 0] = 0.0                            # DC cell: no positional info
    return wp


# --------------------------------------------------------------------------- #
# EMA anchor update (§2 step 3). First-value initialization (anchor := Xlo on
# frame 1) removes the warmup bias that a from-zero EMA would need correcting,
# so this is a plain leaky integrator; n_seen is kept for interface parity.
# --------------------------------------------------------------------------- #
def ema_update(anchor: torch.Tensor, x: torch.Tensor, rho: float,
               n_seen: Optional[int] = None) -> torch.Tensor:
    return rho * anchor + (1.0 - rho) * x


# --------------------------------------------------------------------------- #
# target_lowband — THE load-bearing §3.2 choice: which low-band content is
# drift (anchor it) vs motion (leave it free).
#   complex   : anchor the full complex band  -> strongest, but freezes pans.
#   magnitude : anchor |low band|, keep the frame's own phase -> fixes
#               exposure/color-energy drift while motion (phase) stays free.
#   dc_only   : anchor only the DC cell (global mean per channel) -> a rolling
#               exposure/white-balance lock; cheapest, most conservative.
# --------------------------------------------------------------------------- #
def target_lowband(Xlo: torch.Tensor, anchor: torch.Tensor,
                   anchor_mode: str) -> torch.Tensor:
    if anchor_mode == "complex":
        return anchor
    if anchor_mode == "magnitude":
        # Re-inject the anchored magnitude onto the frame's current phase.
        phasor = Xlo / (Xlo.abs() + EPS)          # unit-modulus, frame's phase
        return anchor.abs() * phasor
    if anchor_mode == "dc_only":
        out = Xlo.clone()
        dc_row = Xlo.shape[-2] // 2
        out[..., dc_row, 0] = anchor[..., dc_row, 0]
        return out
    raise ValueError(f"unknown anchor_mode {anchor_mode!r}")


# --------------------------------------------------------------------------- #
# Cut detection (§2 step 6)
# --------------------------------------------------------------------------- #
def detect_cut(Xlo: torch.Tensor, prev_lowband: torch.Tensor,
               thresh: float) -> bool:
    denom = prev_lowband.abs().pow(2).sum().sqrt()
    if denom <= EPS:
        return False
    rel = (Xlo - prev_lowband).abs().pow(2).sum().sqrt() / denom
    return bool(rel.item() > thresh)


# --------------------------------------------------------------------------- #
# State (§1) — the only persistent memory, and it is tiny.
# --------------------------------------------------------------------------- #
@dataclass
class CoherenceState:
    anchor: torch.Tensor            # complex [C,kh,kw] — EMA'd low-band box
    prev_lowband: torch.Tensor      # complex [C,kh,kw] — for cut detection
    n_seen: int = 0                 # frames folded into the anchor
    warm: bool = False
    cut_cooldown: int = 0           # frames of fast adaptation left after a cut
    H: int = 0
    W: int = 0


# --------------------------------------------------------------------------- #
# Spectral engine (§1, orchestrates §2)
# --------------------------------------------------------------------------- #
class SpectralCoherenceEngine:
    def __init__(self,
                 cutoff_frac: float = 0.10,
                 sigma_k_frac: float = 0.06,
                 alpha: float = 0.5,
                 rho: float = 0.995,
                 color_space: str = "ycbcr",
                 anchor_mode: str = "magnitude",
                 reset_on_cut: bool = True,
                 cut_thresh: float = 0.35,
                 phase_anchor: float = 0.0):
        if anchor_mode not in ("magnitude", "dc_only", "complex"):
            raise ValueError(f"unknown anchor_mode {anchor_mode!r}")
        if color_space not in ("ycbcr", "rgb"):
            raise ValueError(f"unknown color_space {color_space!r}")
        if phase_anchor < 0.0:
            raise ValueError(f"phase_anchor must be >= 0, got {phase_anchor}")
        self.cutoff_frac = cutoff_frac
        self.sigma_k_frac = sigma_k_frac
        self.alpha = alpha
        self.rho = rho
        self.color_space = color_space
        self.anchor_mode = anchor_mode
        self.reset_on_cut = reset_on_cut
        self.cut_thresh = cut_thresh
        # phase_anchor (M0.2): extra flat, DC-excluded anchoring weight added to
        # the Gaussian in `complex` mode only. 0.0 => shipped behaviour exactly
        # (byte-identical). It targets *positional* (low-band phase) drift — the
        # wandering-hotspot case — which the DC-centric Gaussian under-corrects.
        self.phase_anchor = phase_anchor
        self._wcache: dict = {}     # (kh,kw) -> gaussian weight
        self._pcache: dict = {}     # (kh,kw) -> flat phase weight

    # -- box geometry ------------------------------------------------------ #
    def _dims(self, H: int, W: int) -> tuple[int, int]:
        # dc_only keeps just the DC cell -> a 24-byte anchor (§1).
        if self.anchor_mode == "dc_only":
            return 1, 1
        return box_dims(self.cutoff_frac, H, W)

    def _weight(self, kh: int, kw: int, device) -> torch.Tensor:
        key = (kh, kw, str(device))
        w = self._wcache.get(key)
        if w is None:
            w = gaussian_band_weight(kh, kw, self.sigma_k_frac, device=device,
                                     cutoff_frac=self.cutoff_frac)
            self._wcache[key] = w
        return w

    def _blend_weight(self, kh: int, kw: int, device) -> torch.Tensor:
        """Effective per-cell anchor weight w in [0,1]-ish. In `complex` mode
        with phase_anchor>0, superpose a flat, DC-excluded phase weight onto the
        Gaussian so the position-carrying cells get anchored too. Capped at 1 so
        alpha*w stays a valid convex blend factor after the alpha scale."""
        w = self._weight(kh, kw, device)
        if self.anchor_mode != "complex" or self.phase_anchor <= 0.0:
            return w
        key = (kh, kw, str(device))
        wp = self._pcache.get(key)
        if wp is None:
            wp = phase_band_weight(kh, kw, device=device)
            self._pcache[key] = wp
        return torch.clamp(w + self.phase_anchor * wp, max=1.0)

    # -- public interface -------------------------------------------------- #
    def init_state(self, H: int, W: int, device="cpu") -> CoherenceState:
        kh, kw = self._dims(H, W)
        z = torch.zeros(3, kh, kw, dtype=torch.complex64, device=device)
        return CoherenceState(anchor=z.clone(), prev_lowband=z.clone(),
                              n_seen=0, warm=False, cut_cooldown=0, H=H, W=W)

    def state_bytes(self, H: int, W: int) -> int:
        kh, kw = self._dims(H, W)
        return 3 * kh * kw * 8          # complex64 = 8 bytes/cell, C=3

    def reset(self, state: CoherenceState) -> CoherenceState:
        state.anchor = torch.zeros_like(state.anchor)
        state.prev_lowband = torch.zeros_like(state.prev_lowband)
        state.n_seen = 0
        state.warm = False
        state.cut_cooldown = 0
        return state

    def process_frame(self, frame: torch.Tensor, state: CoherenceState):
        # alpha=0 is an exact no-op toggle: return the input untouched so an A/B
        # switch is byte-identical (§6 invariant i). Skips the FFT round-trip.
        if self.alpha == 0.0:
            return frame, state

        kh, kw = self._dims(state.H, state.W)
        device = frame.device
        xc = rgb_to_ycbcr(frame) if self.color_space == "ycbcr" else frame
        X = torch.fft.rfft2(xc)                     # [C,H,Wf] complex64
        Xlo = _crop_box(X, kh, kw)                  # [C,kh,kw]

        is_cut = state.warm and detect_cut(Xlo, state.prev_lowband,
                                           self.cut_thresh)

        if is_cut and self.reset_on_cut:
            # Hard reset: drop scene-A anchor, seed with scene B, skip the blend.
            state.anchor = Xlo.clone()
            state.prev_lowband = Xlo.clone()
            state.n_seen = 1
            state.warm = True
            state.cut_cooldown = 0
            return frame, state

        # rho: adapt fast for a few frames after a soft cut, else steady drift.
        if is_cut and not self.reset_on_cut:
            state.cut_cooldown = CUT_COOLDOWN_N
        rho_eff = CUT_RHO if state.cut_cooldown > 0 else self.rho

        # Update anchor (first-value init -> plain EMA thereafter).
        if not state.warm:
            state.anchor = Xlo.clone()
            state.n_seen = 1
            state.warm = True
        else:
            state.n_seen += 1
            state.anchor = ema_update(state.anchor, Xlo, rho_eff, state.n_seen)

        # On a soft cut we still refreshed the anchor but skip blending this
        # frame so the new scene is not pulled toward the stale target.
        if is_cut and not self.reset_on_cut:
            state.prev_lowband = Xlo.clone()
            state.cut_cooldown = max(0, state.cut_cooldown - 1)
            return frame, state

        # Weighted blend toward the driftful anchor (§2 step 4). The weight is
        # the DC-centric Gaussian, plus (complex + phase_anchor>0) a flat
        # DC-excluded term so positional low-band phase drift is anchored too.
        target = target_lowband(Xlo, state.anchor, self.anchor_mode)
        w = self._blend_weight(kh, kw, device)      # [kh,kw] real, <=1
        aw = (self.alpha * w).clamp(max=1.0)
        Xlo_new = (1.0 - aw) * Xlo + aw * target

        X_new = put_low_band(X, Xlo_new)
        xc_out = torch.fft.irfft2(X_new, s=(state.H, state.W))
        out = ycbcr_to_rgb(xc_out) if self.color_space == "ycbcr" else xc_out
        out = out.clamp(0.0, 1.0)

        state.prev_lowband = Xlo.clone()
        if state.cut_cooldown > 0:
            state.cut_cooldown -= 1
        return out, state

    def process_segment(self, frames: torch.Tensor, state: CoherenceState):
        """Batched-segment path = streaming path, frame by frame.

        Implemented as the process_frame loop so the two paths agree exactly
        (§6 invariant ii): a generator emitting clips gets the same result as
        one feeding frames one at a time.
        """
        out = []
        for t in range(frames.shape[0]):
            y, state = self.process_frame(frames[t], state)
            out.append(y)
        return torch.stack(out, 0), state


# --------------------------------------------------------------------------- #
# Stats-EMA baseline (§5) — the floor the FFT engine must beat. Rolling EMA of
# per-channel mean/std matched into each frame (AdaIN-in-time). No FFT, ~24 B.
# Same interface as the spectral engine so wrap_generator / the A/B harness
# never know which is running.
# --------------------------------------------------------------------------- #
@dataclass
class StatsState:
    mu: torch.Tensor                # [C] running mean
    sd: torch.Tensor                # [C] running std
    n_seen: int = 0
    warm: bool = False
    H: int = 0
    W: int = 0


class StatsEMAEngine:
    def __init__(self,
                 alpha: float = 0.5,
                 rho: float = 0.995,
                 color_space: str = "ycbcr"):
        self.alpha = alpha
        self.rho = rho
        self.color_space = color_space

    def init_state(self, H: int, W: int, device="cpu") -> StatsState:
        z = torch.zeros(3, device=device)
        return StatsState(mu=z.clone(), sd=z.clone(), n_seen=0, warm=False,
                          H=H, W=W)

    def state_bytes(self, H: int, W: int) -> int:
        return 2 * 3 * 4               # mean+std, C=3, float32 = 24 bytes

    def reset(self, state: StatsState) -> StatsState:
        state.mu = torch.zeros_like(state.mu)
        state.sd = torch.zeros_like(state.sd)
        state.n_seen = 0
        state.warm = False
        return state

    def process_frame(self, frame: torch.Tensor, state: StatsState):
        if self.alpha == 0.0:
            return frame, state
        xc = rgb_to_ycbcr(frame) if self.color_space == "ycbcr" else frame
        mu = xc.mean(dim=(-1, -2))                 # [C]
        sd = xc.std(dim=(-1, -2))                  # [C]
        if not state.warm:
            state.mu, state.sd = mu.clone(), sd.clone()
            state.n_seen, state.warm = 1, True
        else:
            state.mu = ema_update(state.mu, mu, self.rho)
            state.sd = ema_update(state.sd, sd, self.rho)
            state.n_seen += 1
        # Pull frame stats a fraction alpha toward the anchored stats.
        tgt_mu = (1 - self.alpha) * mu + self.alpha * state.mu
        tgt_sd = (1 - self.alpha) * sd + self.alpha * state.sd
        norm = (xc - mu[..., None, None]) / (sd[..., None, None] + EPS)
        xc_out = norm * tgt_sd[..., None, None] + tgt_mu[..., None, None]
        out = ycbcr_to_rgb(xc_out) if self.color_space == "ycbcr" else xc_out
        return out.clamp(0.0, 1.0), state

    def process_segment(self, frames: torch.Tensor, state: StatsState):
        out = []
        for t in range(frames.shape[0]):
            y, state = self.process_frame(frames[t], state)
            out.append(y)
        return torch.stack(out, 0), state

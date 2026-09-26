"""M0.5 — ghosting test + scale-selective phase correction (2026-09-26).

Tests the owner hypothesis that complex_mc's HF damage (hf_ref ~0.93) is BAND
INCONSISTENCY: the correction shifts the LOW band (to pin the wandering bump)
while the HIGH band stays put, so every sharp edge gets a double-image / ghost.
A *consistent* correction (shift ALL bands together) should not ghost.

Three experiments, all CPU, reusing the exact hotspot pipeline / metrics that
produced the M0.3/M0.4 frontier (data.make_clip_batch + run_m0 helpers), so the
numbers are directly comparable to the OWNER-VERIFIED ADDENDUM in
IMPL_NOTES_M0_3.md / IMPL_NOTES_M0_4.md:

  E1  metric-floor calibration (P2). hf_ssim(img, exact_subpixel_shift(img, d))
      for d in {0.5, 1.5, 3.0} px on the hotspot INPUT (and, for a direct hf_ref
      bound, on the CLEAN control). Even a PERFECTLY consistent shift decorrelates
      the high-pass residual — this calibrates what the S gate can even mean for a
      translation-type correction.

  E2  mc variants on the hotspot at mc in {0.5, 0.85}:
        (a) BROADBAND consistent shift — apply the estimated correction to ALL
            spectral cells (== translating the whole frame), then the complex
            low-band blend. High band moves WITH the low band -> no ghost.
        (b) LOWBAND-ONLY rigid shift — the current complex_mc (mc_taper=False),
            the variant whose hf_ref sits at ~0.93.
      Reports drift%(pos) / hf_ref / hf_chg / motion% for both.

  E3  scale-selective. Per-band (radial FFT) energy attribution of the injected
      bump vs the balls, then the wander correction applied ONLY on the bump's
      dominant (innermost) radial band, low-band-only (high band untouched, balls
      protected). Sweep strength; report the frontier (drift vs hf_ref vs motion).

Usage:  python crumb_coherence/scripts/run_m05.py [--frames 48] [--grid 64]
                                                  [--seed 1234] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # wavefield_video/
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)                                 # for `import run_m0`

from data import make_clip_batch                           # noqa: E402
from crumb_coherence import (                              # noqa: E402
    SpectralCoherenceEngine, highfreq_ssim, estimate_lowband_shift,
    apply_lowband_shift, rgb_to_ycbcr, ycbcr_to_rgb, gaussian_band_weight,
    put_low_band,
)
from crumb_coherence.core import (                         # noqa: E402
    _crop_box, _box_freqs, box_dims, ema_update,
)
import run_m0                                              # noqa: E402

inject_hotspot = run_m0.inject_hotspot
track_positions = run_m0.track_positions
motion_pct = run_m0.motion_pct
lowband_energy_positions = run_m0.lowband_energy_positions
position_variance = run_m0.position_variance
pct_drop = run_m0.pct_drop

EPS = 1e-8
DEADBAND = 1e-2      # same sub-pixel no-op threshold the engine uses (M0.4)
SIGMA_K_FRAC = 0.06  # engine default Gaussian band sigma (matches frontier runs)


# --------------------------------------------------------------------------- #
# Exact consistent (broadband) sub-pixel translation via the shift theorem.
# No interpolation, applied to EVERY spectral cell -> a whole-frame translation.
# --------------------------------------------------------------------------- #
def _fullframe_ramp(H: int, W: int, Wf: int, dy: float, dx: float, device, dtype):
    """exp(-i 2π (fy·dy + fx·dx)) over the NATIVE (unshifted) rfft2 layout:
    height freqs are fftfreq(H) (0,1/H,..,-1/H), width is the rfft slice c/W."""
    fy = torch.fft.fftfreq(H, device=device).view(1, H, 1)          # cycles/px
    fx = (torch.arange(Wf, device=device).float() / W).view(1, 1, Wf)
    return torch.exp(-2j * math.pi * (fy * dy + fx * dx)).to(dtype)  # [1,H,Wf]


def fullframe_shift_frames(frames: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Translate every frame by (dy, dx) px, all bands consistently. [T,C,H,W]."""
    T, C, H, W = frames.shape
    X = torch.fft.rfft2(frames)                                     # [T,C,H,Wf]
    Wf = X.shape[-1]
    ramp = _fullframe_ramp(H, W, Wf, dy, dx, frames.device, X.dtype)
    return torch.fft.irfft2(X * ramp, s=(H, W))


# --------------------------------------------------------------------------- #
# Engine variants for E2/E3, built from the same primitives core uses so they
# are faithful to complex / complex_mc and directly comparable to the frontier.
# --------------------------------------------------------------------------- #
def _run_common_setup(H, W, cutoff, device):
    kh, kw = box_dims(cutoff, H, W)
    w = gaussian_band_weight(kh, kw, SIGMA_K_FRAC, device=device, cutoff_frac=cutoff)
    return kh, kw, w


def broadband_mc_segment(frames, mc_strength, alpha, rho, cutoff):
    """E2(a): complex blend + BROADBAND (whole-frame) motion compensation.

    Identical to `complex_mc` except the estimated correction is applied to the
    FULL spectrum (all bands) rather than only the low-band box, so the high band
    is translated consistently with the low band -> no double-image / ghost. The
    cost: the sharp balls (all high band) are dragged by the same offset."""
    T, C, H, W = frames.shape
    device = frames.device
    kh, kw, w = _run_common_setup(H, W, cutoff, device)
    aw = (alpha * w).clamp(max=1.0)
    anchor = torch.zeros(C, kh, kw, dtype=torch.complex64, device=device)
    warm, n_seen = False, 0
    out = []
    for t in range(T):
        xc = rgb_to_ycbcr(frames[t])
        X = torch.fft.rfft2(xc)
        Xlo = _crop_box(X, kh, kw)
        if not warm:
            anchor = Xlo.clone(); n_seen = 1; warm = True
        else:
            n_seen += 1
            anchor = ema_update(anchor, Xlo, rho)
        Xsrc = X
        if mc_strength > 0.0 and n_seen > 1:
            dy, dx = estimate_lowband_shift(Xlo, anchor, H, W)
            sdy, sdx = -mc_strength * dy, -mc_strength * dx
            if max(abs(sdy), abs(sdx)) >= DEADBAND:
                Wf = X.shape[-1]
                ramp = _fullframe_ramp(H, W, Wf, sdy, sdx, device, X.dtype)
                Xsrc = X * ramp                       # whole-frame translation
        Xlo_src = _crop_box(Xsrc, kh, kw)
        Xlo_new = (1.0 - aw) * Xlo_src + aw * anchor  # complex target
        X_new = put_low_band(Xsrc, Xlo_new)           # low band edited; HF = shifted HF
        xc_out = torch.fft.irfft2(X_new, s=(H, W))
        out.append(ycbcr_to_rgb(xc_out).clamp(0.0, 1.0))
    return torch.stack(out, 0)


def _radial_taper(kh, kw, H, W, r_inner, device):
    """1.0 on box cells with radial freq < r_inner (cyc/px), else 0.0. A hard
    inner-band mask on the shift's PHASE (still unit-modulus, magnitude-safe)."""
    ky, kx = _box_freqs(kh, kw, H, W, device=device)    # [kh,1],[1,kw] cyc/px
    r = torch.sqrt(ky * ky + kx * kx)                    # [kh,kw]
    return (r < r_inner).to(torch.float32)


def scaleselective_mc_segment(frames, mc_strength, r_inner, alpha, rho, cutoff):
    """E3: complex_mc, but the re-centering shift is applied ONLY on the bump's
    dominant innermost radial band (hard mask), low-band-only (HF untouched, so
    balls / motion are protected). r_inner in cycles/pixel."""
    T, C, H, W = frames.shape
    device = frames.device
    kh, kw, w = _run_common_setup(H, W, cutoff, device)
    aw = (alpha * w).clamp(max=1.0)
    taper = _radial_taper(kh, kw, H, W, r_inner, device)
    anchor = torch.zeros(C, kh, kw, dtype=torch.complex64, device=device)
    warm, n_seen = False, 0
    out = []
    for t in range(T):
        xc = rgb_to_ycbcr(frames[t])
        X = torch.fft.rfft2(xc)
        Xlo = _crop_box(X, kh, kw)
        if not warm:
            anchor = Xlo.clone(); n_seen = 1; warm = True
        else:
            n_seen += 1
            anchor = ema_update(anchor, Xlo, rho)
        src = Xlo
        if mc_strength > 0.0 and n_seen > 1:
            dy, dx = estimate_lowband_shift(Xlo, anchor, H, W)
            sdy, sdx = -mc_strength * dy, -mc_strength * dx
            if max(abs(sdy), abs(sdx)) >= DEADBAND:
                src = apply_lowband_shift(Xlo, sdy, sdx, H, W, taper=taper)
        Xlo_new = (1.0 - aw) * src + aw * anchor
        X_new = put_low_band(X, Xlo_new)              # low band only; HF exact
        xc_out = torch.fft.irfft2(X_new, s=(H, W))
        out.append(ycbcr_to_rgb(xc_out).clamp(0.0, 1.0))
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- #
# Radial (per-band) energy attribution for E3.
# --------------------------------------------------------------------------- #
def radial_energy(signal: torch.Tensor, edges: list) -> list:
    """Fraction of |FFT|^2 energy of `signal` [T,C,H,W] in each radial band
    [edges[i], edges[i+1]) in cycles/pixel. Returns a list of fractions."""
    T, C, H, W = signal.shape
    X = torch.fft.fft2(signal)                          # full 2-D FFT
    power = (X.abs() ** 2).sum(dim=(0, 1))              # [H,W] over T,C
    fy = torch.fft.fftfreq(H).view(H, 1)
    fx = torch.fft.fftfreq(W).view(1, W)
    r = torch.sqrt(fy * fy + fx * fx)                   # [H,W] cyc/px
    total = float(power.sum().item()) + EPS
    fracs = []
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1])
        fracs.append(float(power[m].sum().item()) / total)
    return fracs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--alpha", type=float, default=0.95)     # hotspot default
    ap.add_argument("--rho", type=float, default=0.995)
    ap.add_argument("--cutoff", type=float, default=0.14)    # hotspot default
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    T, G = args.frames, args.grid
    A, RHO, CUT = args.alpha, args.rho, args.cutoff

    clip, meta = make_clip_batch(1, T, G, G, seed=args.seed, collisions=True,
                                 return_meta=True)
    control = clip[0, :T].clone()
    gt_pos = meta["pos"][0, :T].clone()
    gt_col = meta["col"][0].clone()
    drifted = inject_hotspot(control)
    bump = drifted - control                                 # injected field only

    control_traj = track_positions(control, gt_pos, gt_col)
    posvar_drift = position_variance(lowband_energy_positions(drifted, CUT))

    def evaluate(corrected):
        posvar_c = position_variance(lowband_energy_positions(corrected, CUT))
        d_pos = pct_drop(posvar_drift, posvar_c)
        hf_ref = highfreq_ssim(control, corrected)          # vs CLEAN truth (gate)
        hf_chg = highfreq_ssim(drifted, corrected)          # vs input (diagnostic)
        corr_traj = track_positions(corrected, gt_pos, gt_col)
        pct, n_seg, _, _ = motion_pct(control_traj, corr_traj)
        return dict(drift=d_pos, hf_ref=hf_ref, hf_chg=hf_chg,
                    motion=pct, segs=n_seg)

    def gates(r):
        return (r["drift"] >= 60.0 and r["hf_ref"] >= 0.98
                and not math.isnan(r["motion"]) and abs(r["motion"] - 100.0) <= 5.0)

    record = dict(config=dict(frames=T, grid=G, seed=args.seed, alpha=A,
                              rho=RHO, cutoff=CUT, posvar_drift=posvar_drift))

    print("=" * 84)
    print(f"M0.5  ghosting test + scale-selective  grid={G}x{G}  frames={T}  "
          f"seed={args.seed}")
    print(f"  alpha={A}  rho={RHO}  cutoff_frac={CUT}  "
          f"injected pos_var={posvar_drift:.3f}px^2")
    print(f"  gates: D=drift>=60  S=hf_ref>=0.98  M=|motion-100|<=5")
    print("=" * 84)

    # ---------------- E1: metric-floor calibration (P2) ---------------- #
    print("\n[E1] metric floor  hf_ssim(img, exact_subpixel_shift(img, d)) — the")
    print("     best hf_ssim ANY translation-type correction of magnitude d can score.")
    print(f"{'d(px)':>7}{'hf_ssim(drifted)':>20}{'hf_ssim(control)':>20}")
    e1 = []
    for d in (0.5, 1.5, 3.0):
        sh_dr = fullframe_shift_frames(drifted, 0.0, d)     # shift in x by d
        sh_ct = fullframe_shift_frames(control, 0.0, d)
        hd = highfreq_ssim(drifted, sh_dr)
        hc = highfreq_ssim(control, sh_ct)
        e1.append(dict(d=d, hf_drifted=hd, hf_control=hc))
        print(f"{d:>7.1f}{hd:>20.4f}{hc:>20.4f}")
    record["E1_metric_floor"] = e1
    print("     Reading: hf_ssim(control) directly UPPER-BOUNDS hf_ref for any method")
    print("     that effectively translates content by ~d px, ghost-free or not.")

    # ---------------- E2: broadband vs lowband-only ---------------- #
    print("\n[E2] mc variants on the hotspot  (drift%pos / hf_ref / hf_chg / motion%)")
    print(f"{'variant':<26}{'mc':>5}{'drift%':>9}{'hf_ref':>9}{'hf_chg':>9}"
          f"{'motion%':>9}{'segs':>6}  gates")
    e2 = []
    for mc in (0.5, 0.85):
        # (a) broadband consistent shift
        r_bb = evaluate(broadband_mc_segment(drifted, mc, A, RHO, CUT))
        # (b) current lowband-only rigid shift (== complex_mc mc_taper=False)
        eng = SpectralCoherenceEngine(alpha=A, rho=RHO, cutoff_frac=CUT,
                                      anchor_mode="complex_mc", mc_strength=mc,
                                      mc_taper=False)
        st = eng.init_state(G, G)
        corr, _ = eng.process_segment(drifted, st)
        r_lb = evaluate(corr)
        for tag, r in (("(a) broadband", r_bb), ("(b) lowband-only rigid", r_lb)):
            g = "".join(["D" if r["drift"] >= 60 else ".",
                         "S" if r["hf_ref"] >= 0.98 else ".",
                         "M" if (not math.isnan(r["motion"])
                                 and abs(r["motion"] - 100) <= 5) else "."])
            mo = "  n/a  " if math.isnan(r["motion"]) else f"{r['motion']:8.1f}%"
            print(f"{tag:<26}{mc:>5.2f}{r['drift']:>8.1f}%{r['hf_ref']:>9.4f}"
                  f"{r['hf_chg']:>9.4f}{mo}{r['segs']:>6}  [{g}]  "
                  f"{'PASS' if gates(r) else 'fail'}")
            e2.append(dict(variant=tag, mc=mc, **r, gates=g))
    record["E2_variants"] = e2

    # ---------------- E3: scale-selective ---------------- #
    print("\n[E3] per-band (radial FFT) energy attribution — where the bump lives"
          " vs the balls")
    edges = [0.0, 0.03, 0.06, 0.14, 0.5]
    band_names = ["inner[0,.03)", "mid1[.03,.06)", "mid2[.06,.14)", "outer[.14,.5)"]
    bump_e = radial_energy(bump, edges)
    balls_e = radial_energy(control[1:] - control[:-1], edges)  # frame-diff = balls
    print(f"{'band (cyc/px)':<18}{'bump %E':>10}{'balls %E':>10}")
    for nm, be, ba in zip(band_names, bump_e, balls_e):
        print(f"{nm:<18}{100*be:>9.2f}%{100*ba:>9.2f}%")
    record["E3_attribution"] = dict(edges=edges, bands=band_names,
                                    bump=bump_e, balls=balls_e)
    dominant = band_names[int(torch.tensor(bump_e).argmax().item())]
    print(f"     bump dominant band: {dominant}  ("
          f"balls dominant: {band_names[int(torch.tensor(balls_e).argmax().item())]})")

    print("\n[E3] scale-selective correction (inner radial band only, HF untouched)")
    print(f"{'r_inner':>9}{'mc':>6}{'drift%':>9}{'hf_ref':>9}{'hf_chg':>9}"
          f"{'motion%':>9}  gates")
    e3 = []
    for r_inner in (0.03, 0.06, 0.10):
        for mc in (0.5, 0.85, 1.0):
            r = evaluate(scaleselective_mc_segment(drifted, mc, r_inner, A, RHO, CUT))
            g = "".join(["D" if r["drift"] >= 60 else ".",
                         "S" if r["hf_ref"] >= 0.98 else ".",
                         "M" if (not math.isnan(r["motion"])
                                 and abs(r["motion"] - 100) <= 5) else "."])
            mo = "  n/a  " if math.isnan(r["motion"]) else f"{r['motion']:8.1f}%"
            print(f"{r_inner:>9.3f}{mc:>6.2f}{r['drift']:>8.1f}%{r['hf_ref']:>9.4f}"
                  f"{r['hf_chg']:>9.4f}{mo}  [{g}]  "
                  f"{'PASS' if gates(r) else 'fail'}")
            e3.append(dict(r_inner=r_inner, mc=mc, **r, gates=g))
    record["E3_frontier"] = e3

    # ---------------- reference rows (honest baselines) ---------------- #
    print("\n[ref] honest baselines (from the same pipeline, for context)")
    eng = SpectralCoherenceEngine(alpha=A, rho=RHO, cutoff_frac=CUT,
                                  anchor_mode="complex")
    st = eng.init_state(G, G)
    corr, _ = eng.process_segment(drifted, st)
    r = evaluate(corr)
    print(f"{'complex (champion)':<26}{'-':>5}{r['drift']:>8.1f}%{r['hf_ref']:>9.4f}"
          f"{r['hf_chg']:>9.4f}{r['motion']:>8.1f}%  [{'PASS' if gates(r) else 'fail'}]")
    record["ref_complex"] = r
    print("=" * 84)

    if args.json:
        d = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(d, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()

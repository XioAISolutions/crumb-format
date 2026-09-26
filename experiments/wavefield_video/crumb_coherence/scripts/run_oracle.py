"""M0.7 — the oracle four-arm test (2026-09-26).

Locates the remaining hotspot headroom (M0.6 winner = 64.0% drift removal, full
[DSME] pass) by separating THREE error sources before we touch the estimator
again, exactly per reviews/DEEP_DIVE_M05_astra.md ("the single decisive
experiment"):

  Arm 1  ORACLE COMPONENT (ceiling). Replace the drifted frame's WHOLE low-band
         box with the CLEAN control's low band. Perfect nuisance removal; the
         upper bound of any component-space fix. If even this cannot clear
         hf_ref>=0.98 RAW but clears it REGISTERED, the S gate itself is the
         problem (Astra Outcome 2 / H3).

  Arm 2  ORACLE BAND-PHASE. The engine's complex_mc pipeline REPRODUCED exactly
         (same alpha/rho/cutoff/band/edge/strength/EMA-blend), but the per-frame
         displacement estimate is replaced by the ORACLE displacement computed
         from the injector's known bump centers. Differs from Arm 3 ONLY in the
         displacement source, so Arm2 vs Arm3 isolates ESTIMATOR error; Arm2 vs
         Arm1 isolates DECOMPOSITION error (bump energy outside the r<band mask).

  Arm 3  ENGINE BAND-PHASE. The shipped M0.6 winner config, real correlation
         estimator. Context row.

  Arm 4  X - B_hat + B_hat_stable. Estimate the coherent moving low-band
         component from the engine's own displacement estimate and re-place it at
         the stable (anchor) position. The "even better variant", made realistic.

CW-SSIM is intentionally SKIPPED (it needs a new wavelet dependency; the task
forbids new deps). Noted, not faked.

This script does NOT modify crumb_coherence/ — it reuses the shipped helpers
read-only and reproduces the engine loop locally where an oracle substitution is
needed (Arm 2), so every number stays directly comparable to the M0.3-M0.6
frontier (same data.make_clip_batch + run_m0 metrics + hotspot params).

Usage:  python crumb_coherence/scripts/run_oracle.py [--frames 48] [--grid 64]
                                                     [--seed 1234] [--json out]
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
    put_low_band, low_band_box, radial_band_mask,
)
from crumb_coherence.core import (                         # noqa: E402
    _crop_box, box_dims, ema_update,
)
import run_m0                                              # noqa: E402

inject_hotspot = run_m0.inject_hotspot
track_positions = run_m0.track_positions
motion_pct = run_m0.motion_pct
lowband_energy_positions = run_m0.lowband_energy_positions
position_variance = run_m0.position_variance
pct_drop = run_m0.pct_drop

EPS = 1e-8
DEADBAND = 1e-2       # same sub-pixel no-op threshold the engine uses (M0.4)
SIGMA_K_FRAC = 0.06   # engine default Gaussian band sigma (matches frontier runs)

# M0.6 winner config (the shipped hotspot default), shared by arms 2/3/4.
MC_STRENGTH = 0.5
MC_BAND = 0.03
MC_EDGE = "hard"


# --------------------------------------------------------------------------- #
# True bump centres, straight from run_m0.inject_hotspot (the oracle knows the
# injector). cy/cx are the per-frame Gaussian-bump centre in pixels.
# --------------------------------------------------------------------------- #
def true_bump_centres(T: int, H: int, W: int):
    """(cy[T], cx[T]) — the exact per-frame bump centre inject_hotspot used."""
    t = torch.linspace(0, 1, T)
    cy = (H / 2) + (H / 4) * torch.sin(2 * math.pi * 0.7 * t)
    cx = (W / 2) + (W / 4) * torch.sin(2 * math.pi * 0.4 * t + 1.1)
    return cy, cx


# --------------------------------------------------------------------------- #
# Exact whole-frame sub-pixel translation via the shift theorem (for oracle
# registration only). Applied to EVERY spectral cell -> no interpolation.
# --------------------------------------------------------------------------- #
def _fullframe_shift_frame(frame: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Translate a single frame [C,H,W] by (dy, dx) px, all bands consistently."""
    C, H, W = frame.shape
    X = torch.fft.rfft2(frame)
    Wf = X.shape[-1]
    fy = torch.fft.fftfreq(H, device=frame.device).view(1, H, 1)
    fx = (torch.arange(Wf, device=frame.device).float() / W).view(1, 1, Wf)
    ramp = torch.exp(-2j * math.pi * (fy * dy + fx * dx)).to(X.dtype)
    return torch.fft.irfft2(X * ramp, s=(H, W))


def oracle_register(corrected: torch.Tensor, control: torch.Tensor,
                    cutoff: float) -> torch.Tensor:
    """Align each corrected frame to the clean control by undoing the residual
    global translation between them, measured with the SHIPPED phase-correlation
    helper (estimate_lowband_shift). This is the task's "oracle registration":
    if hf_ref jumps after this alignment, the raw S gate was penalising a benign
    sub-pixel translation (Astra Outcome 2 / H3), not real detail damage."""
    T, C, H, W = corrected.shape
    out = []
    for t in range(T):
        c_lo = low_band_box(torch.fft.rfft2(rgb_to_ycbcr(corrected[t])), cutoff)
        a_lo = low_band_box(torch.fft.rfft2(rgb_to_ycbcr(control[t])), cutoff)
        dy, dx = estimate_lowband_shift(c_lo, a_lo, H, W)   # corrected vs clean
        aligned = _fullframe_shift_frame(corrected[t], -dy, -dx)
        out.append(aligned.clamp(0.0, 1.0))
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- #
# Arm 1 — ORACLE COMPONENT: swap the whole low-band box with the clean control's.
# --------------------------------------------------------------------------- #
def arm_oracle_component(drifted: torch.Tensor, control: torch.Tensor,
                         cutoff: float) -> torch.Tensor:
    T, C, H, W = drifted.shape
    kh, kw = box_dims(cutoff, H, W)
    out = []
    for t in range(T):
        Xd = torch.fft.rfft2(rgb_to_ycbcr(drifted[t]))
        Xc = torch.fft.rfft2(rgb_to_ycbcr(control[t]))
        Xlo_clean = _crop_box(Xc, kh, kw)
        X_new = put_low_band(Xd, Xlo_clean)                 # clean low band, drift HF
        out.append(ycbcr_to_rgb(torch.fft.irfft2(X_new, s=(H, W))).clamp(0.0, 1.0))
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- #
# Arm 2 — ORACLE BAND-PHASE: the engine's complex_mc loop, byte-for-byte, but the
# displacement is the ORACLE offset of the current bump centre from an EMA(rho) of
# the true centres — the exact quantity estimate_lowband_shift TRIES to measure
# against the (complex-EMA) anchor, with zero leakage/parabola noise. (Mirroring
# the anchor's phase-implied position by an EMA of centres is an approximation of
# the complex-EMA anchor; documented as such.)
# --------------------------------------------------------------------------- #
def arm_oracle_bandphase(drifted, cy, cx, alpha, rho, cutoff,
                         mc=MC_STRENGTH, r_band=MC_BAND, edge=MC_EDGE):
    T, C, H, W = drifted.shape
    device = drifted.device
    kh, kw = box_dims(cutoff, H, W)
    w = gaussian_band_weight(kh, kw, SIGMA_K_FRAC, device=device, cutoff_frac=cutoff)
    aw = (alpha * w).clamp(max=1.0)
    band = radial_band_mask(kh, kw, H, W, r_band, device=device, edge=edge)
    anchor = torch.zeros(C, kh, kw, dtype=torch.complex64, device=device)
    ema_cy = ema_cx = 0.0
    warm, n_seen = False, 0
    out = []
    for t in range(T):
        X = torch.fft.rfft2(rgb_to_ycbcr(drifted[t]))
        Xlo = _crop_box(X, kh, kw)
        if not warm:
            anchor = Xlo.clone(); n_seen = 1; warm = True
            ema_cy, ema_cx = float(cy[t]), float(cx[t])
        else:
            n_seen += 1
            anchor = ema_update(anchor, Xlo, rho)
            ema_cy = rho * ema_cy + (1.0 - rho) * float(cy[t])
            ema_cx = rho * ema_cx + (1.0 - rho) * float(cx[t])
        src = Xlo
        if mc > 0.0 and n_seen > 1:
            dy = float(cy[t]) - ema_cy                       # oracle offset (y)
            dx = float(cx[t]) - ema_cx                       # oracle offset (x)
            sdy, sdx = -mc * dy, -mc * dx
            if max(abs(sdy), abs(sdx)) >= DEADBAND:
                src = apply_lowband_shift(Xlo, sdy, sdx, H, W, taper=band)
        Xlo_new = (1.0 - aw) * src + aw * anchor
        X_new = put_low_band(X, Xlo_new)
        out.append(ycbcr_to_rgb(torch.fft.irfft2(X_new, s=(H, W))).clamp(0.0, 1.0))
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- #
# Arm 4 — X - B_hat + B_hat_stable.
#
# Exact definition used (simplest faithful causal version):
#   * B_hat_0     := the EMA(rho) low-band anchor A  (the coherent low-band
#                    template at its stable position; p* = 0 by construction).
#   * p_t         := estimate_lowband_shift(Xlo_t, A) — the engine's own
#                    displacement estimate (offset of the current low band from A).
#   * B_hat_t     := apply_lowband_shift(A, p_t)  — the coherent template placed
#                    at the CURRENT (drifting) position.
#   * B_hat_stable:= A.
#   Output low band = Xlo_t - B_hat_t + B_hat_stable = Xlo_t + (A - shift(A, p_t)).
# Applied over the WHOLE low-band box (Astra's W(k) == 1 on the box, 0 outside);
# the high band is untouched. CAVEAT (documented): A also holds the static
# background/ball low-band, so a non-zero p_t moves that too — the price of the
# simplest B_hat. A W(k) restricted to r<band would curb this (future work).
# --------------------------------------------------------------------------- #
def arm_component_subtract(drifted, rho, cutoff):
    T, C, H, W = drifted.shape
    device = drifted.device
    kh, kw = box_dims(cutoff, H, W)
    anchor = torch.zeros(C, kh, kw, dtype=torch.complex64, device=device)
    warm, n_seen = False, 0
    out = []
    for t in range(T):
        X = torch.fft.rfft2(rgb_to_ycbcr(drifted[t]))
        Xlo = _crop_box(X, kh, kw)
        if not warm:
            anchor = Xlo.clone(); n_seen = 1; warm = True
        else:
            n_seen += 1
            anchor = ema_update(anchor, Xlo, rho)
        Xlo_new = Xlo
        if n_seen > 1:
            dy, dx = estimate_lowband_shift(Xlo, anchor, H, W)
            if max(abs(dy), abs(dx)) >= DEADBAND:
                b_hat_t = apply_lowband_shift(anchor, dy, dx, H, W)   # template @ now
                Xlo_new = Xlo - b_hat_t + anchor                      # X - B + B_stable
        # No coherence blend: the component swap IS the whole correction (the
        # "simplest faithful" X - B_hat + B_hat_stable). HF band untouched.
        X_new = put_low_band(X, Xlo_new)
        out.append(ycbcr_to_rgb(torch.fft.irfft2(X_new, s=(H, W))).clamp(0.0, 1.0))
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--alpha", type=float, default=0.95)      # hotspot default
    ap.add_argument("--rho", type=float, default=0.995)
    ap.add_argument("--cutoff", type=float, default=0.14)     # hotspot default
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
    cy, cx = true_bump_centres(T, G, G)

    control_traj = track_positions(control, gt_pos, gt_col)
    posvar_drift = position_variance(lowband_energy_positions(drifted, CUT))

    def evaluate(corrected):
        posvar_c = position_variance(lowband_energy_positions(corrected, CUT))
        d_pos = pct_drop(posvar_drift, posvar_c)
        hf_ref = highfreq_ssim(control, corrected)              # vs CLEAN (gate)
        hf_reg = highfreq_ssim(control, oracle_register(corrected, control, CUT))
        hf_chg = highfreq_ssim(drifted, corrected)              # vs input (diag)
        corr_traj = track_positions(corrected, gt_pos, gt_col)
        pct, n_seg, _, _ = motion_pct(control_traj, corr_traj)
        return dict(drift=d_pos, hf_ref=hf_ref, hf_reg=hf_reg, hf_chg=hf_chg,
                    motion=pct, segs=n_seg)

    def gates(r):
        return (r["drift"] >= 60.0 and r["hf_ref"] >= 0.98
                and not math.isnan(r["motion"]) and abs(r["motion"] - 100.0) <= 5.0)

    def flags(r):
        return "".join(["D" if r["drift"] >= 60 else ".",
                        "S" if r["hf_ref"] >= 0.98 else ".",
                        "M" if (not math.isnan(r["motion"])
                                and abs(r["motion"] - 100) <= 5) else "."])

    record = dict(config=dict(frames=T, grid=G, seed=args.seed, alpha=A, rho=RHO,
                              cutoff=CUT, mc=MC_STRENGTH, band=MC_BAND, edge=MC_EDGE,
                              posvar_drift=posvar_drift))

    print("=" * 92)
    print(f"M0.7  oracle four-arm test   grid={G}x{G}  frames={T}  seed={args.seed}")
    print(f"  alpha={A}  rho={RHO}  cutoff_frac={CUT}  mc={MC_STRENGTH}  "
          f"band={MC_BAND}  edge={MC_EDGE}   injected pos_var={posvar_drift:.3f}px^2")
    print(f"  gates: D=drift>=60  S=hf_ref(raw)>=0.98  M=|motion-100|<=5     "
          f"(CW-SSIM skipped: needs a new dep)")
    print("=" * 92)

    # Arm 3 — the shipped M0.6 winner engine (real correlation estimator).
    engine = SpectralCoherenceEngine(
        alpha=A, rho=RHO, cutoff_frac=CUT, anchor_mode="complex_mc",
        mc_strength=MC_STRENGTH, mc_band=MC_BAND, mc_est="corr", mc_edge=MC_EDGE)
    arm3_corrected, _ = engine.process_segment(drifted, engine.init_state(G, G))

    # Build all arms.
    arms = [
        ("1 oracle component", arm_oracle_component(drifted, control, CUT)),
        ("2 oracle band-phase", arm_oracle_bandphase(drifted, cy, cx, A, RHO, CUT)),
        ("3 engine band-phase", arm3_corrected),
        ("4 X-Bhat+Bhat_stable", arm_component_subtract(drifted, RHO, CUT)),
    ]

    print(f"{'arm':<22}{'drift%':>9}{'hf_ref':>9}{'hf_reg':>9}{'hf_chg':>9}"
          f"{'motion%':>9}{'segs':>6}  gates")
    print("-" * 92)
    results = []
    for tag, corrected in arms:
        r = evaluate(corrected)
        mo = "  n/a  " if math.isnan(r["motion"]) else f"{r['motion']:8.1f}%"
        print(f"{tag:<22}{r['drift']:>8.1f}%{r['hf_ref']:>9.4f}{r['hf_reg']:>9.4f}"
              f"{r['hf_chg']:>9.4f}{mo}{r['segs']:>6}  [{flags(r)}]  "
              f"{'PASS' if gates(r) else 'fail'}")
        results.append(dict(arm=tag, **r, pass_all=gates(r)))
    record["arms"] = results

    # Reference rows (context, from the same pipeline).
    print("-" * 92)
    r_dr = evaluate(drifted)
    print(f"{'ref drifted (input)':<22}{r_dr['drift']:>8.1f}%{r_dr['hf_ref']:>9.4f}"
          f"{r_dr['hf_reg']:>9.4f}{r_dr['hf_chg']:>9.4f}"
          f"{r_dr['motion']:>8.1f}%{r_dr['segs']:>6}  [{flags(r_dr)}]")
    record["ref_drifted"] = r_dr
    print("=" * 92)
    print("Reading:")
    print("  Arm1 = component-space CEILING (perfect low-band removal).")
    print("  Arm2 - Arm3  isolates ESTIMATOR error (oracle vs corr, same pipeline).")
    print("  Arm1 - Arm2  isolates DECOMPOSITION error (bump energy outside r<band).")
    print("  hf_reg >> hf_ref would mean the raw S gate penalises benign translation.")

    if args.json:
        d = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(d, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()

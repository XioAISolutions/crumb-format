"""M0 — synthetic-drift test for crumb_coherence (PLUGIN_SPEC §7, milestone M0).

Take a real, coherent moving-ball clip (data.make_clip_batch), inject KNOWN
low-band drift, then run the stats-EMA baseline and the spectral engine
(dc_only / magnitude / complex) and report, on TWO ORTHOGONAL AXES:

  * drift reduction %   — the thing the engine is supposed to kill
  * motion preservation % — the thing it must NOT touch (real ball motion)
  * highfreq_ssim(in,out) — did we harm detail?
  * state bytes

Two scenarios, two different drift axes (this is the §3.2 magnitude-vs-phase
split made measurable):

  gain_field : exposure ramp + wandering low-freq COLOR field. This is
               magnitude/DC drift -> drift axis = lowband_trajectory_variance
               (temporal variance of the low-band *magnitude* spectrum) and
               delta_e_vs_ref (mean color drift). `magnitude` mode should clear
               the >=60% bar here.
  hotspot    : a mean-preserving wandering low-freq Gaussian bump. Its position
               wanders => this is low-band *phase* drift. drift axis =
               lowband_energy_position_variance (how much the low-band energy
               centroid wanders over time). `magnitude` keeps phase free so it
               barely touches this; `complex` anchors phase so it removes it but
               freezes ball motion. That contrast is the M0 lesson, not a bug.

M0.1 refinement (2026-09-26):
  * Motion% now comes from a MULTI-BLOB TRACKER (semantic_metrics: per-ball
    color+position matching against GT), not a single brightness centroid or a
    peak-argmax path. A single global centroid is dragged by the drift field and
    fuses the three balls; the tracker follows each ball's own local centroid, so
    drift wobble and ball fusion no longer contaminate the number.
  * hotspot ΔE is reported in ABSOLUTE terms (a bounded-artifact check), not as a
    %-drop off a near-zero mean-preserved baseline — the old %-drop produced a
    meaningless "negative" number.

Everything runs on CPU. A tiny side-by-side mp4 is written if imageio is present.

Usage:  python crumb_coherence/scripts/run_m0.py [--frames 48] [--grid 64]
                                                 [--scenario gain_field|hotspot]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # wavefield_video/
sys.path.insert(0, _ROOT)

from data import make_clip_batch, PALETTE_ANCHORS, RADIUS   # noqa: E402
from semantic_metrics import (                              # noqa: E402
    detect_blobs, _assignment, MATCH_RADIUS_MULT,
)
from crumb_coherence import (                               # noqa: E402
    SpectralCoherenceEngine, StatsEMAEngine,
    lowband_trajectory_variance, delta_e_vs_ref, highfreq_ssim, temporal_flicker,
    low_band_box, put_low_band,
)


# --------------------------------------------------------------------------- #
# Known low-band drift injectors.
# --------------------------------------------------------------------------- #
def inject_drift(frames: torch.Tensor) -> torch.Tensor:
    """gain_field: exposure ramp + wandering low-freq color field. Both are
    global / low-spatial-frequency (they land in the anchored band) and they
    leave ball positions (phase) untouched. This is MAGNITUDE drift."""
    T, C, H, W = frames.shape
    t = torch.linspace(0, 1, T)
    gain = (1.0 + 0.25 * t).view(T, 1, 1, 1)              # exposure ramp 1.0 -> 1.25
    bias = (0.08 * t).view(T, 1, 1, 1)                    # brightness lift 0 -> 0.08
    ys = torch.linspace(0, 1, H).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W).view(1, 1, 1, W)
    amp = torch.tensor([0.10, 0.06, 0.12]).view(1, C, 1, 1)   # per-channel = color cast
    phase = (2 * math.pi * 0.5 * t).view(T, 1, 1, 1)     # slow spatial wander
    field = (amp
             * (0.5 + 0.5 * torch.sin(2 * math.pi * xs + phase))
             * (0.5 + 0.5 * torch.cos(2 * math.pi * ys + 0.7 * phase)))
    return (frames * gain + bias + field).clamp(0, 1)


def inject_hotspot(frames: torch.Tensor) -> torch.Tensor:
    """Mean-preserving wandering hotspot: a low-frequency Gaussian bump whose
    position wanders slowly; its spatial mean is subtracted so per-frame scalar
    statistics stay ~constant (the stats-EMA baseline is blind to it, the
    spectral engine tracks it). This is POSITIONAL (low-band phase) drift."""
    T, C, H, W = frames.shape
    t = torch.linspace(0, 1, T)
    ys = torch.arange(H).float().view(1, H, 1)
    xs = torch.arange(W).float().view(1, 1, W)
    cy = (H / 2) + (H / 4) * torch.sin(2 * math.pi * 0.7 * t)      # [T]
    cx = (W / 2) + (W / 4) * torch.sin(2 * math.pi * 0.4 * t + 1.1)
    sig = max(H, W) / 8.0
    bump = torch.exp(-((ys - cy.view(T, 1, 1)) ** 2) / (2 * sig * sig)) * \
           torch.exp(-((xs - cx.view(T, 1, 1)) ** 2) / (2 * sig * sig))
    bump = bump.unsqueeze(1)                                     # [T,1,H,W]
    amp = torch.tensor([0.22, 0.16, 0.10]).view(1, C, 1, 1)
    field = amp * bump
    field = field - field.mean(dim=(2, 3), keepdim=True)          # mean-preserving
    return (frames + field).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Motion axis — MULTI-BLOB TRACKER (reuses semantic_metrics matching logic).
# --------------------------------------------------------------------------- #
def _gt_color_ids(gt_col: torch.Tensor) -> list:
    """Map each ball's RGB [nb,3] to its palette-anchor index (same rule the
    detector uses internally to label a blob's color_index)."""
    anchors = F.normalize(PALETTE_ANCHORS.float(), dim=-1)
    normalized = F.normalize(gt_col.float(), dim=-1)
    return (normalized @ anchors.T).argmax(-1).tolist()


def track_positions(frames: torch.Tensor, gt_pos: torch.Tensor,
                    gt_col: torch.Tensor, radius: float = RADIUS) -> torch.Tensor:
    """Per-ball detected trajectory. frames [T,3,H,W], gt_pos [T,nb,2] (x,y),
    gt_col [nb,3]. Returns [T,nb,2] of matched detection centroids, NaN where a
    ball was not detected/matched in that frame.

    Matching per frame is the SAME maximum-cardinality minimum-distance,
    same-color, within-tolerance assignment used in semantic_metrics — we just
    keep the matched *position* instead of only the error. Because it follows
    each ball's own local (2-sigma) centroid, a smooth low-band drift field
    cannot drag it (no sharp maxima) and the three balls never fuse into one
    reading the way a single global centroid does.
    """
    T = frames.shape[0]
    nb = gt_pos.shape[1]
    cids = _gt_color_ids(gt_col)
    detected = detect_blobs(frames, radius=radius)
    tol = MATCH_RADIUS_MULT * radius
    out = torch.full((T, nb, 2), float("nan"))
    for t in range(T):
        blobs = detected[t]
        if not blobs:
            continue
        positions = gt_pos[t].tolist()
        penalty = (nb + 1) * (tol + 1)
        costs, matched_pos = [], []
        for pos, cid in zip(positions, cids):
            row, mrow = [], []
            for det in blobs:
                d = math.dist(pos, det["position"])
                valid = (cid == det["color_index"]) and d <= tol
                row.append(d if valid else 2 * penalty)
                mrow.append(det["position"] if valid else None)
            costs.append(row + [penalty] * nb)   # each GT ball may go unmatched
            matched_pos.append(mrow)
        assign = _assignment(costs)
        for i, col in enumerate(assign):
            if 0 <= col < len(blobs) and matched_pos[i][col] is not None:
                out[t, i, 0] = matched_pos[i][col][0]
                out[t, i, 1] = matched_pos[i][col][1]
    return out


def motion_pct(control_traj: torch.Tensor, corrected_traj: torch.Tensor):
    """Path length preserved, measured over frame-pairs where a ball is tracked
    in BOTH the clean control and the corrected clip (apples-to-apples segments).

    Returns (pct, n_segments, L_control, L_corrected). pct = 100*L_corr/L_ctrl:
    ~100 means motion untouched; <<100 means the engine froze large-scale motion.
    """
    T, nb, _ = control_traj.shape
    L_ctrl = L_corr = 0.0
    n_seg = 0
    for i in range(nb):
        for t in range(T - 1):
            seg = torch.stack([control_traj[t, i], control_traj[t + 1, i],
                               corrected_traj[t, i], corrected_traj[t + 1, i]])
            if torch.isnan(seg).any():
                continue
            L_ctrl += float((control_traj[t + 1, i] - control_traj[t, i]).norm())
            L_corr += float((corrected_traj[t + 1, i] - corrected_traj[t, i]).norm())
            n_seg += 1
    if L_ctrl <= 1e-8:
        return float("nan"), n_seg, L_ctrl, L_corr
    return 100.0 * L_corr / L_ctrl, n_seg, L_ctrl, L_corr


# --------------------------------------------------------------------------- #
# Hotspot drift axis — LOW-BAND ENERGY POSITION VARIANCE.
# The hotspot is a wandering low-freq structure; its drift is *where* the
# low-band energy sits, not how bright it is. We reconstruct the low band alone,
# strip the static (temporal-mean) component, and track the centroid of the
# wandering positive lobe. Variance over time = drift magnitude.
# --------------------------------------------------------------------------- #
def lowband_energy_positions(frames: torch.Tensor, cutoff_frac: float) -> torch.Tensor:
    """[T,2] centroid (x,y) of the wandering low-band energy per frame."""
    T, C, H, W = frames.shape
    luma = frames.mean(dim=1)                              # [T,H,W]
    lows = []
    for t in range(T):
        X = torch.fft.rfft2(luma[t])
        Xlo = low_band_box(X, cutoff_frac)
        Xz = put_low_band(torch.zeros_like(X), Xlo)        # keep only the low band
        lows.append(torch.fft.irfft2(Xz, s=(H, W)))
    lows = torch.stack(lows, 0)                            # [T,H,W]
    resid = lows - lows.mean(dim=0, keepdim=True)          # strip static component
    w = resid.clamp(min=0.0)                               # positive lobe = bump
    ys = torch.arange(H).view(1, H, 1).float()
    xs = torch.arange(W).view(1, 1, W).float()
    wsum = w.sum(dim=(1, 2)) + 1e-8
    cx = (w * xs).sum(dim=(1, 2)) / wsum
    cy = (w * ys).sum(dim=(1, 2)) / wsum
    return torch.stack((cx, cy), dim=1)                   # [T,2]


def position_variance(pos: torch.Tensor) -> float:
    """Sum of x/y temporal variance (px^2). Zero for a still hotspot."""
    return float(pos.var(dim=0, unbiased=False).sum().item())


def pct_drop(before: float, after: float) -> float:
    if before <= 1e-12:
        return 0.0
    return 100.0 * (1.0 - after / before)


# --------------------------------------------------------------------------- #
# Engine table (shared by both scenarios).
# --------------------------------------------------------------------------- #
def _engines(alpha, rho, cutoff, hotspot_alpha, hotspot_cutoff, scenario,
             hotspot_phase_anchor=0.0, hotspot_mc_strength=0.0):
    """Engine list. For the hotspot (positional/phase drift) the spectral modes
    may use a slightly wider band / stronger blend so the phase-anchoring
    `complex` mode can actually clear the bar — the `magnitude` DEFAULT is left
    at its shipped value on purpose (it is *supposed* to leave phase free).

    M0.2: the hotspot `complex` engine also gets phase_anchor>0 — a flat,
    DC-excluded companion weight that anchors the *position-carrying* low-band
    cells the DC-centric Gaussian under-corrects. FALSIFIED (kept at 0).

    M0.3: adds `complex_mc` — motion-compensated anchoring. It measures the
    current low band's sub-pixel offset from the anchor (low-pass phase
    correlation) and undoes mc_strength of it before the blend, re-centering the
    wandering bump instead of pulling it toward a lagging target. In gain_field
    (magnitude drift, no positional wander) the measured shift is ~0, so
    complex_mc is run at full strength there as a do-no-harm check."""
    a = hotspot_alpha if scenario == "hotspot" else alpha
    c = hotspot_cutoff if scenario == "hotspot" else cutoff
    pa = hotspot_phase_anchor if scenario == "hotspot" else 0.0
    mcs = hotspot_mc_strength if scenario == "hotspot" else 1.0
    return [
        ("stats-EMA baseline", StatsEMAEngine(alpha=a, rho=rho)),
        ("spectral dc_only", SpectralCoherenceEngine(
            alpha=a, rho=rho, cutoff_frac=c, anchor_mode="dc_only")),
        ("spectral magnitude", SpectralCoherenceEngine(
            alpha=a, rho=rho, cutoff_frac=c, anchor_mode="magnitude")),
        ("spectral complex", SpectralCoherenceEngine(
            alpha=a, rho=rho, cutoff_frac=c, anchor_mode="complex",
            phase_anchor=pa)),
        ("spectral complex_mc", SpectralCoherenceEngine(
            alpha=a, rho=rho, cutoff_frac=c, anchor_mode="complex_mc",
            mc_strength=mcs)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--rho", type=float, default=0.995)
    ap.add_argument("--cutoff", type=float, default=0.10)
    # Hotspot is positional (phase) drift: a wider band + stronger blend let the
    # complex mode reach the bar. Magnitude default stays shipped (phase free).
    ap.add_argument("--hotspot-alpha", type=float, default=0.95)
    ap.add_argument("--hotspot-cutoff", type=float, default=0.14)
    # M0.2 phase-anchor strength for the hotspot complex engine (0 = shipped).
    ap.add_argument("--hotspot-phase-anchor", type=float, default=0.0)
    # M0.3 motion-compensation strength for the hotspot complex_mc engine
    # (0 = byte-identical to complex; 1 = fully re-center the wandering bump).
    ap.add_argument("--hotspot-mc-strength", type=float, default=0.5)
    # Comma-separated sweep over mc_strength for the hotspot complex_mc engine,
    # printed as an extra block (shape of the new knob). Empty = no sweep.
    ap.add_argument("--hotspot-mc-sweep", type=str, default="0,0.5,0.85,1.0")
    ap.add_argument("--scenario", choices=["gain_field", "hotspot"], default="gain_field")
    ap.add_argument("--outdir", type=str, default=os.path.join(_ROOT, "crumb_coherence", "out"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    T, G = args.frames, args.grid

    # Coherent control clip + GT metadata (per-frame ball positions & colors for
    # the tracker). collisions couple the balls so motion is a genuine test.
    clip, meta = make_clip_batch(1, T, G, G, seed=args.seed, collisions=True,
                                 return_meta=True)          # [1,T+1,3,H,W], meta
    control = clip[0, :T].clone()                           # [T,3,H,W]
    gt_pos = meta["pos"][0, :T].clone()                    # [T,nb,2] (x,y)
    gt_col = meta["col"][0].clone()                        # [nb,3]
    drifted = (inject_hotspot(control) if args.scenario == "hotspot"
               else inject_drift(control))

    cutoff = args.cutoff
    control_traj = track_positions(control, gt_pos, gt_col)
    n_balls = gt_pos.shape[1]

    print("=" * 82)
    print(f"M0 drift test  scenario={args.scenario}  grid={G}x{G}  frames={T}  "
          f"seed={args.seed}  balls={n_balls}")
    if args.scenario == "hotspot":
        print(f"  alpha={args.hotspot_alpha}  rho={args.rho}  "
              f"cutoff_frac={args.hotspot_cutoff}  "
              f"phase_anchor={args.hotspot_phase_anchor}  "
              f"mc_strength={args.hotspot_mc_strength}  (hotspot overrides)")
    else:
        print(f"  alpha={args.alpha}  rho={args.rho}  cutoff_frac={cutoff}")
    print("=" * 82)

    if args.scenario == "gain_field":
        _run_gain_field(control, drifted, control_traj, gt_pos, gt_col,
                        cutoff, args)
    else:
        _run_hotspot(control, drifted, control_traj, gt_pos, gt_col,
                     cutoff, args)


def _fmt_motion(pct, n_seg):
    return "  n/a  " if math.isnan(pct) else f"{pct:6.1f}%"


def _run_gain_field(control, drifted, control_traj, gt_pos, gt_col, cutoff, args):
    var_drift = lowband_trajectory_variance(drifted, cutoff)
    de_drift = float(delta_e_vs_ref(drifted).mean().item())
    var_ctrl = lowband_trajectory_variance(control, cutoff)
    flick_drift = temporal_flicker(drifted)

    print(f"injected drift  : lowband_mag_var={var_drift:.4e}  meanΔE={de_drift:.3f}  "
          f"flicker={flick_drift:.2e}")
    print(f"clean control   : lowband_mag_var={var_ctrl:.4e}  "
          f"(noise floor; drift adds on top)")
    print("-" * 82)
    print(f"{'engine':<22}{'drift%(var)':>12}{'drift%(ΔE)':>12}"
          f"{'hf_ssim':>9}{'motion%':>9}{'segs':>6}{'state':>9}  verdict")
    print("-" * 82)

    results = {}
    for name, eng in _engines(args.alpha, args.rho, cutoff, args.hotspot_alpha,
                              args.hotspot_cutoff, "gain_field",
                              hotspot_mc_strength=args.hotspot_mc_strength):
        st = eng.init_state(args.grid, args.grid)
        corrected, _ = eng.process_segment(drifted, st)
        var_c = lowband_trajectory_variance(corrected, cutoff)
        de_c = float(delta_e_vs_ref(corrected).mean().item())
        ssim = highfreq_ssim(drifted, corrected)
        corr_traj = track_positions(corrected, gt_pos, gt_col)
        pct, n_seg, _, _ = motion_pct(control_traj, corr_traj)
        d_var = pct_drop(var_drift, var_c)
        d_de = pct_drop(de_drift, de_c)
        sb = eng.state_bytes(args.grid, args.grid)
        ok_drift = d_var >= 60.0
        ok_ssim = ssim >= 0.98
        ok_motion = (not math.isnan(pct)) and abs(pct - 100.0) <= 5.0
        verdict = "PASS" if (ok_drift and ok_ssim and ok_motion) else "fail"
        flags = "".join(["D" if ok_drift else ".",
                         "S" if ok_ssim else ".",
                         "M" if ok_motion else "."])
        results[name] = corrected
        print(f"{name:<22}{d_var:>11.1f}%{d_de:>11.1f}%{ssim:>9.4f}"
              f"{_fmt_motion(pct, n_seg)}{n_seg:>6}{sb:>8}B  {verdict} [{flags}]")

    print("-" * 82)
    print("flags: D=drift(var)>=60%  S=ssim>=0.98  M=tracker motion within 5%")
    _print_state_headline(args)
    print("Reading: 'magnitude' should clear D+S+M (magnitude drift is exactly "
          "what it\n         anchors). 'complex' may also remove drift but can "
          "drop motion% (it\n         freezes low-band phase). dc_only/stats miss "
          "the spatial color field.")

    if not args.no_video:
        _save_video(control, drifted, results.get("spectral magnitude"), args.outdir,
                    "m0_gain_field.mp4")


def _run_hotspot(control, drifted, control_traj, gt_pos, gt_col, cutoff, args):
    hc = args.hotspot_cutoff
    posvar_drift = position_variance(lowband_energy_positions(drifted, hc))
    posvar_ctrl = position_variance(lowband_energy_positions(control, hc))
    de_drift = float(delta_e_vs_ref(drifted).mean().item())
    flick_drift = temporal_flicker(drifted)

    print(f"injected drift  : lowband_pos_var={posvar_drift:.3f}px^2  "
          f"(bump wanders)  |ΔE|drifted={de_drift:.3f}  flicker={flick_drift:.2e}")
    print(f"clean control   : lowband_pos_var={posvar_ctrl:.3f}px^2  "
          f"(noise floor: ball motion leaks a little low-band energy)")
    print(f"  NOTE: hotspot is mean-preserving, so ΔE is ~0 by construction — it "
          f"is a\n        BOUNDED-ARTIFACT check (corrected |ΔE| must stay small), "
          f"NOT the drift axis.")
    print("-" * 82)
    print(f"{'engine':<22}{'drift%(pos)':>12}{'|ΔE|corr':>10}"
          f"{'hf_ssim':>9}{'motion%':>9}{'segs':>6}{'state':>9}  verdict")
    print("-" * 82)

    results = {}
    for name, eng in _engines(args.alpha, args.rho, cutoff, args.hotspot_alpha,
                              hc, "hotspot",
                              hotspot_phase_anchor=args.hotspot_phase_anchor,
                              hotspot_mc_strength=args.hotspot_mc_strength):
        st = eng.init_state(args.grid, args.grid)
        corrected, _ = eng.process_segment(drifted, st)
        posvar_c = position_variance(lowband_energy_positions(corrected, hc))
        de_c = float(delta_e_vs_ref(corrected).mean().item())
        ssim = highfreq_ssim(drifted, corrected)
        corr_traj = track_positions(corrected, gt_pos, gt_col)
        pct, n_seg, _, _ = motion_pct(control_traj, corr_traj)
        d_pos = pct_drop(posvar_drift, posvar_c)
        sb = eng.state_bytes(args.grid, args.grid)
        ok_drift = d_pos >= 60.0
        ok_ssim = ssim >= 0.98
        ok_motion = (not math.isnan(pct)) and abs(pct - 100.0) <= 5.0
        ok_de = de_c <= max(1.0, 2.0 * de_drift)   # bounded artifact, not a drop
        verdict = "PASS" if (ok_drift and ok_ssim and ok_motion) else "fail"
        flags = "".join(["D" if ok_drift else ".",
                         "S" if ok_ssim else ".",
                         "M" if ok_motion else ".",
                         "E" if ok_de else "!"])
        results[name] = corrected
        print(f"{name:<22}{d_pos:>11.1f}%{de_c:>10.3f}{ssim:>9.4f}"
              f"{_fmt_motion(pct, n_seg)}{n_seg:>6}{sb:>8}B  {verdict} [{flags}]")

    print("-" * 82)
    print("flags: D=drift(pos)>=60%  S=ssim>=0.98  M=motion within 5%  "
          "E=|ΔE| bounded")

    # M0.3 sweep: shape of the mc_strength knob on the complex_mc engine.
    sweep = [s.strip() for s in args.hotspot_mc_sweep.split(",") if s.strip()]
    if sweep:
        print("-" * 82)
        print(f"complex_mc mc_strength sweep (drift bar >=60%, motion >=97, "
              f"ssim >=0.98):")
        print(f"{'mc_strength':<14}{'drift%(pos)':>12}{'|ΔE|corr':>10}"
              f"{'hf_ssim':>9}{'motion%':>9}{'segs':>6}  verdict")
        for s in sweep:
            mcs = float(s)
            eng = SpectralCoherenceEngine(
                alpha=args.hotspot_alpha, rho=args.rho, cutoff_frac=hc,
                anchor_mode="complex_mc", mc_strength=mcs)
            st = eng.init_state(args.grid, args.grid)
            corrected, _ = eng.process_segment(drifted, st)
            posvar_c = position_variance(lowband_energy_positions(corrected, hc))
            de_c = float(delta_e_vs_ref(corrected).mean().item())
            ssim = highfreq_ssim(drifted, corrected)
            corr_traj = track_positions(corrected, gt_pos, gt_col)
            pct, n_seg, _, _ = motion_pct(control_traj, corr_traj)
            d_pos = pct_drop(posvar_drift, posvar_c)
            ok = (d_pos >= 60.0 and ssim >= 0.98
                  and (not math.isnan(pct)) and pct >= 97.0)
            print(f"{mcs:<14.3f}{d_pos:>11.1f}%{de_c:>10.3f}{ssim:>9.4f}"
                  f"{_fmt_motion(pct, n_seg)}{n_seg:>6}  "
                  f"{'PASS' if ok else 'fail'}")

    _print_state_headline(args)
    print("Reading: the hotspot WANDERS -> its drift is low-band *phase*. "
          "'magnitude'\n         keeps phase free (by design) so it barely moves "
          "drift%(pos) but keeps\n         motion. 'complex' anchors phase -> "
          "removes the wander; watch motion% for\n         whether it also freezes "
          "ball motion (the §3.2 cost of phase-anchoring).\n         stats-EMA is "
          "blind (mean-preserving bump). The negative %-ΔE artifact\n         is "
          "gone: |ΔE|corr is reported absolute and bounded.")

    if not args.no_video:
        _save_video(control, drifted, results.get("spectral complex_mc"),
                    args.outdir, "m0_hotspot.mp4")


def _print_state_headline(args):
    print("state bytes @512x512 (spec headline):")
    for name, eng in _engines(args.alpha, args.rho, args.cutoff, args.hotspot_alpha,
                              args.hotspot_cutoff, args.scenario):
        print(f"    {name:<22}{eng.state_bytes(512, 512):>8} B")
    print("=" * 82)


def _save_video(control, drifted, corrected, outdir, fname):
    try:
        import imageio.v2 as imageio
    except Exception as exc:                              # noqa: BLE001
        print(f"[video] imageio unavailable ({exc}); skipping mp4.")
        return
    if corrected is None:
        return
    os.makedirs(outdir, exist_ok=True)
    T = control.shape[0]
    gap = torch.ones(3, control.shape[-2], 2)             # white separators
    frames = []
    for t in range(T):
        row = torch.cat([control[t], gap, drifted[t], gap, corrected[t]], dim=-1)
        img = (row.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).numpy()
        frames.append(img)
    path = os.path.join(outdir, fname)
    try:
        imageio.mimsave(path, frames, fps=12, macro_block_size=1)
        print(f"[video] wrote {path}  (control | drifted | corrected)")
    except Exception as exc:                              # noqa: BLE001
        gif = path.replace(".mp4", ".gif")
        imageio.mimsave(gif, frames, fps=12)
        print(f"[video] mp4 failed ({exc}); wrote {gif} instead.")


if __name__ == "__main__":
    main()

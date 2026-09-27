"""M2 — legitimate-motion TRAP test (Astra's top-priority falsification, 2026-09-26).

Every M0/M1 success contained something we WANTED reduced, so a skeptic can still
say the plugin is just "a low-frequency variance suppressor" rather than "a
coherence engine that distinguishes drift from intended evolution." This test
kills (or fails to kill) the dangerous narrow hypothesis:

  H_suppress: crumb_coherence suppresses low-frequency temporal change whether it
              is pathological (drift) OR semantically correct (intended motion).

Design (reviews/DEEP_DIVE_M1_assess.md section 2), 64x64, NO injected wander —
the scene IS the clean ground truth (raw == clean here):
  * one LARGE soft Gaussian object translating at 0.15 px/frame in x (its spectral
    energy sits at r ~ 0.016 cyc/px, INSIDE the r<0.03 band complex_mc corrects —
    this is the trap: legitimate low-frequency MOTION living where the plugin acts);
  * slow sinusoidal global illumination (intended low-band temporal change);
  * 3 small balls on independent trajectories (legit HF motion; also make HF-SSIM
    meaningful — if the plugin damaged detail it would show here).

Frozen SHIPPING config (no tuning): complex_mc, band=0.03, strength=0.5, corr,
hard edge, alpha=0.95, rho=0.995, cutoff=0.14.

The plugin should do essentially NOTHING. Kill criteria (PASS = all hold):
  (1) intentional centroid-motion retention  > 95%
  (2) low-frequency trajectory distortion     < 5%
  (3) HF-SSIM vs clean                         > 0.98
  (4) clean low-band variance ratio plugin/raw in [0.90, 1.10]

If it FAILS, H_suppress is CONFIRMED (the bad news): the shipping complex_mc
cannot tell a translating object from wander and damages legitimate low-frequency
motion; the product claim weakens from "distinguishes drift from intent" to
"frequency-selective stabilizer", and complex_mc must be gated (velocity-aware /
drift-detected) before it is safe on general content.

Naming (per the assessment): the M0.7 arms are the "geometric-displacement
control" and "known-component replacement" (NOT "oracle"). This test uses neither.

Usage:  python crumb_coherence/scripts/run_trap.py [--frames 96] [--grid 64]
                                                   [--seed 0] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from crumb_coherence import (                                # noqa: E402
    SpectralCoherenceEngine, highfreq_ssim, lowband_trajectory_variance,
    low_band_box, put_low_band,
)

EPS = 1e-8
# Frozen shipping config — do NOT tune (the whole point of the trap).
SHIP_CFG = dict(alpha=0.95, rho=0.995, cutoff_frac=0.14, anchor_mode="complex_mc",
                mc_strength=0.5, mc_band=0.03, mc_est="corr", mc_edge="hard")
OBJ_VELOCITY = 0.15          # px/frame, x-translation of the large object
OBJ_SIGMA = 10.0             # large + soft -> low-frequency (inside the r<0.03 band)


# --------------------------------------------------------------------------- #
# Trap scene: large translating object + slow illumination + 3 independent balls.
# There is NO nuisance drift; this scene is the clean ground truth.
# --------------------------------------------------------------------------- #
def make_trap_scene(T: int, H: int, W: int):
    ys = torch.arange(H).float().view(1, H, 1)
    xs = torch.arange(W).float().view(1, 1, W)
    t = torch.arange(T).float()

    # LARGE soft object, translating at 0.15 px/frame in x (y fixed).
    ox = 0.30 * W + OBJ_VELOCITY * t
    oy = torch.full((T,), 0.50 * H)
    obj = torch.exp(-(((xs - ox.view(T, 1, 1)) ** 2)
                      + ((ys - oy.view(T, 1, 1)) ** 2)) / (2 * OBJ_SIGMA ** 2))  # [T,H,W]

    scene = torch.full((T, 3, H, W), 0.08)                   # dim background
    scene = scene + obj.unsqueeze(1) * torch.tensor([0.55, 0.42, 0.30]).view(1, 3, 1, 1)

    # 3 small balls, independent linear trajectories, distinct colors (HF detail).
    balls = [
        (0.16 * W, 0.16 * H, 0.30, 0.20, (0.90, 0.12, 0.12), 2.0),
        (0.78 * W, 0.24 * H, -0.20, 0.35, (0.12, 0.88, 0.20), 2.0),
        (0.62 * W, 0.86 * H, 0.15, -0.30, (0.20, 0.35, 0.95), 2.4),
    ]
    for bx0, by0, bvx, bvy, col, bsig in balls:
        bx = bx0 + bvx * t
        by = by0 + bvy * t
        b = torch.exp(-(((xs - bx.view(T, 1, 1)) ** 2)
                        + ((ys - by.view(T, 1, 1)) ** 2)) / (2 * bsig ** 2))
        scene = scene + b.unsqueeze(1) * torch.tensor(col).view(1, 3, 1, 1)

    # Slow intentional sinusoidal illumination (~1 cycle over the clip).
    illum = (1.0 + 0.12 * torch.sin(2 * math.pi * t / T)).view(T, 1, 1, 1)
    scene = (scene * illum).clamp(0.0, 1.0)
    return scene, torch.stack((ox, oy), dim=1)               # scene [T,3,H,W], analytic obj [T,2]


# --------------------------------------------------------------------------- #
# Low-band energy centroid path (self-referential; same construction family as
# run_m0.lowband_energy_positions, but stripping the per-frame SPATIAL mean (DC)
# so the centroid is the object's ABSOLUTE position, not a wander-vs-mean lobe).
# --------------------------------------------------------------------------- #
def lowband_centroid_path(frames: torch.Tensor, cutoff: float) -> torch.Tensor:
    T, C, H, W = frames.shape
    ys = torch.arange(H).float().view(1, H, 1)
    xs = torch.arange(W).float().view(1, 1, W)
    out = []
    for tt in range(T):
        X = torch.fft.rfft2(frames[tt].mean(0))              # luma [H,Wf]
        Xz = put_low_band(torch.zeros_like(X), low_band_box(X, cutoff))
        low = torch.fft.irfft2(Xz, s=(H, W))                 # [H,W] low band
        w = (low - low.mean()).clamp(min=0.0)                # strip DC, positive lobe
        wsum = float(w.sum()) + EPS
        cx = float((w * xs).sum()) / wsum
        cy = float((w * ys).sum()) / wsum
        out.append([cx, cy])
    return torch.tensor(out)                                 # [T,2]


def path_len(p: torch.Tensor) -> float:
    return float((p[1:] - p[:-1]).norm(dim=1).sum())


def bbox_diag(p: torch.Tensor) -> float:
    return float((p.max(0).values - p.min(0).values).norm())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    T, G = args.frames, args.grid
    CUT = SHIP_CFG["cutoff_frac"]

    scene, obj_true = make_trap_scene(T, G, G)               # raw == clean (no wander)

    eng = SpectralCoherenceEngine(**SHIP_CFG)
    plugin, _ = eng.process_segment(scene, eng.init_state(G, G))

    # ---- centroid trajectories (same estimator both sides) ---------------- #
    clean_c = lowband_centroid_path(scene, CUT)
    plug_c = lowband_centroid_path(plugin, CUT)

    # (1) intentional centroid-motion retention (path length ratio)
    len_clean, len_plug = path_len(clean_c), path_len(plug_c)
    retention = 100.0 * len_plug / (len_clean + EPS)

    # (2) low-frequency trajectory distortion: max per-frame deviation / clean range
    dev = (plug_c - clean_c).norm(dim=1)                     # [T]
    clean_range = bbox_diag(clean_c) + EPS
    distortion = 100.0 * float(dev.max()) / clean_range
    distortion_mean = 100.0 * float(dev.mean()) / clean_range

    # (3) HF-SSIM vs clean
    hf = highfreq_ssim(scene, plugin)

    # (4) clean low-band variance ratio plugin/raw (raw == scene == clean)
    lbv_raw = lowband_trajectory_variance(scene, CUT)
    lbv_plug = lowband_trajectory_variance(plugin, CUT)
    var_ratio = lbv_plug / (lbv_raw + EPS)

    # validity check: does the estimator track the KNOWN object on the clean scene?
    est_x_range = float(clean_c[:, 0].max() - clean_c[:, 0].min())
    true_x_range = float(obj_true[:, 0].max() - obj_true[:, 0].min())

    passes = dict(retention=retention > 95.0, distortion=distortion < 5.0,
                  hf=hf > 0.98, var_ratio=0.90 <= var_ratio <= 1.10)
    all_pass = all(passes.values())

    print("=" * 84)
    print(f"M2  legitimate-motion TRAP   grid={G}x{G}  frames={T}  seed={args.seed}")
    print(f"  scene: large obj @ {OBJ_VELOCITY}px/frame (sigma={OBJ_SIGMA}) + "
          f"illum sinusoid + 3 balls;  NO injected wander")
    print(f"  frozen config: complex_mc band={SHIP_CFG['mc_band']} "
          f"strength={SHIP_CFG['mc_strength']} corr hard  alpha={SHIP_CFG['alpha']} "
          f"rho={SHIP_CFG['rho']} cutoff={CUT}")
    print("=" * 84)
    print(f"  estimator validity: clean centroid x-range {est_x_range:.2f}px vs "
          f"true object x-range {true_x_range:.2f}px  (should be close)")
    print("-" * 84)
    print(f"  (1) centroid-motion retention : {retention:6.1f}%   "
          f"(clean path {len_clean:.2f}px, plugin {len_plug:.2f}px)   "
          f"[{'PASS' if passes['retention'] else 'FAIL'}  >95%]")
    print(f"  (2) trajectory distortion     : {distortion:6.2f}%   "
          f"(mean {distortion_mean:.2f}%; max dev {float(dev.max()):.3f}px / "
          f"range {clean_range:.2f}px)   [{'PASS' if passes['distortion'] else 'FAIL'}  <5%]")
    print(f"  (3) HF-SSIM vs clean          : {hf:6.4f}   "
          f"[{'PASS' if passes['hf'] else 'FAIL'}  >0.98]")
    print(f"  (4) low-band var ratio p/raw  : {var_ratio:6.3f}   "
          f"(raw {lbv_raw:.4e} -> plugin {lbv_plug:.4e})   "
          f"[{'PASS' if passes['var_ratio'] else 'FAIL'}  in [0.90,1.10]]")
    print("-" * 84)
    print(f"  VERDICT: {'PASS — plugin leaves legitimate motion alone' if all_pass else 'FAIL — H_suppress CONFIRMED'}")
    if not all_pass:
        print("  -> The frozen complex_mc cannot distinguish a translating object from")
        print("     wander: it damages legitimate low-frequency motion. complex_mc must")
        print("     be GATED (velocity-aware / drift-detected) before it is safe on")
        print("     general content. (magnitude mode, which leaves phase free, is the")
        print("     safe default for content with legitimate low-frequency motion.)")
    print("=" * 84)

    if args.json:
        rec = dict(config=dict(frames=T, grid=G, seed=args.seed, ship=SHIP_CFG,
                               obj_velocity=OBJ_VELOCITY, obj_sigma=OBJ_SIGMA),
                   retention=retention, distortion=distortion,
                   distortion_mean=distortion_mean, hf_ssim=hf,
                   lowband_var_raw=lbv_raw, lowband_var_plugin=lbv_plug,
                   var_ratio=var_ratio, est_x_range=est_x_range,
                   true_x_range=true_x_range, passes=passes, all_pass=all_pass)
        d = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(d, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()

"""M2/M3 — legitimate-motion TRAP test + the M3 velocity-coherence GATE.

Every M0/M1 success contained something we WANTED reduced, so a skeptic can still
say the plugin is just "a low-frequency variance suppressor" rather than "a
coherence engine that distinguishes drift from intended evolution." This test
kills (or fails to kill) the dangerous narrow hypothesis:

  H_suppress: crumb_coherence suppresses low-frequency temporal change whether it
              is pathological (drift) OR semantically correct (intended motion).

M2 (frozen complex_mc, --legacy) CONFIRMED H_suppress: the engine damaged
legitimate low-frequency motion (retention 54.7%/70.8%, distortion 64.1%/58.9%).
M3 adds the GATE (mc_gate): the correction is scaled by the DIRECTIONAL
PERSISTENCE of the low-band motion, so coherent motion (a pan / translation /
accelerating subject) passes untouched (gate g->0, a provable identity) while
incoherent wander (what complex_mc exists to remove) is still corrected (g->1).

Design (reviews/DEEP_DIVE_M1_assess.md section 2), 64x64, NO injected wander —
the scene IS the clean ground truth (raw == clean here):
  * ONE large soft Gaussian object with a LEGITIMATE low-band trajectory, its
    spectral energy at r ~ 0.016 cyc/px, INSIDE the r<0.03 band complex_mc
    corrects (the trap: legitimate low-frequency MOTION where the plugin acts).
    --scene translate : constant-velocity pan (the M2 scene).
    --scene accelerate: monotone +x, speed ramps up (2nd legit class; acceptance 2).
    --scene curve     : a gentle ~50 deg turning arc (stress test; reported, not gated).
  * slow sinusoidal global illumination (intended low-band temporal change);
  * 3 small balls on independent trajectories (legit HF motion; also make HF-SSIM
    meaningful — if the plugin damaged detail it would show here).

Frozen SHIPPING config (M3): complex_mc, band=0.03, strength=0.5, corr, hard
edge, alpha=0.95, rho=0.995, cutoff=0.14, mc_gate ON. --legacy drops the gate
(the pre-M3 frozen behaviour, kept reachable for the record: acceptance 4).

The gated plugin should do essentially NOTHING on this legit scene. Kill criteria
(PASS = all hold):
  (1) intentional centroid-motion retention  > 95%
  (2) low-frequency trajectory distortion     < 5%
  (3) HF-SSIM vs clean                         > 0.98
  (4) clean low-band variance ratio plugin/raw in [0.90, 1.10]

Usage:
  python crumb_coherence/scripts/run_trap.py [--frames 96] [--scene translate]
         [--legacy] [--diag] [--json out.json]
  # acceptance sweep (both horizons x {translate, accelerate}):
  for s in translate accelerate; do for f in 96 192; do
      python crumb_coherence/scripts/run_trap.py --scene $s --frames $f; done; done
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
# Frozen shipping config — M3 adds mc_gate (+ its lo/hi/decay). --legacy strips
# the gate to reproduce the pre-M3 frozen behaviour (acceptance 4).
SHIP_CFG = dict(alpha=0.95, rho=0.995, cutoff_frac=0.14, anchor_mode="complex_mc",
                mc_strength=0.5, mc_band=0.03, mc_est="corr", mc_edge="hard",
                mc_gate=True, mc_gate_lo=0.35, mc_gate_hi=0.75, mc_gate_decay=0.9,
                mc_gate_pos_decay=0.5)
OBJ_SIGMA = 10.0             # large + soft -> low-frequency (inside the r<0.03 band)
OBJ_VELOCITY = 0.15          # px/frame, x-translation for --scene translate (M2)
# accelerate / curve span 0.25W -> 0.65W in normalized progress so they stay
# on-grid at every horizon; the point is only whether the plugin leaves them be.
OBJ_X0, OBJ_XSPAN = 0.25, 0.40
CURVE_DEG = 50.0             # arc angle for --scene curve (velocity rotates this far)


# --------------------------------------------------------------------------- #
# Object trajectory per scene. Every path is a LEGITIMATE motion (raw == clean);
# the point is whether the plugin leaves it be. `translate` is the VERBATIM M2
# scene (frame-indexed 0.15 px/frame) so --legacy reproduces the documented FAIL;
# the others use normalized progress s in [0,1] to stay on-grid at any horizon.
# --------------------------------------------------------------------------- #
def object_path(scene: str, T: int, H: int, W: int):
    """Return (ox, oy) each [T] — the object's centre per frame for `scene`."""
    s = torch.linspace(0.0, 1.0, T)
    t = torch.arange(T).float()
    if scene == "translate":                       # constant velocity (M2 scene)
        ox = 0.30 * W + OBJ_VELOCITY * t
        oy = torch.full((T,), 0.50 * H)
    elif scene == "accelerate":                    # monotone +x, speed ramps up
        ox = (OBJ_X0 + OBJ_XSPAN * s * s) * W      # ease-in: v grows, same sign
        oy = torch.full((T,), 0.50 * H)
    elif scene == "curve":                         # gentle turning arc (stress)
        th = math.radians(CURVE_DEG)
        phi = th * s
        R = (OBJ_XSPAN * W) / max(math.sin(th), EPS)   # match the x-extent
        ox = (OBJ_X0 * W) + R * torch.sin(phi)
        oy = (0.40 * H) + R * (1.0 - torch.cos(phi))   # drifts down as it turns
    elif scene == "wander":                        # DIAGNOSTIC ONLY (not legit):
        # the M0 hotspot's oscillating Lissajous path — incoherent wander that
        # RETURNS. Used only to read the gate's coherence C on a case it SHOULD
        # keep correcting; never an acceptance scene (raw != a motion to protect).
        ox = (0.50 * W) + (0.25 * W) * torch.sin(2 * math.pi * 0.4 * s + 1.1)
        oy = (0.50 * H) + (0.25 * H) * torch.sin(2 * math.pi * 0.7 * s)
    else:
        raise ValueError(f"unknown scene {scene!r}")
    return ox, oy


def make_trap_scene(scene: str, T: int, H: int, W: int):
    ys = torch.arange(H).float().view(1, H, 1)
    xs = torch.arange(W).float().view(1, 1, W)
    t = torch.arange(T).float()

    ox, oy = object_path(scene, T, H, W)
    obj = torch.exp(-(((xs - ox.view(T, 1, 1)) ** 2)
                      + ((ys - oy.view(T, 1, 1)) ** 2)) / (2 * OBJ_SIGMA ** 2))  # [T,H,W]

    base = torch.full((T, 3, H, W), 0.08)                    # dim background
    base = base + obj.unsqueeze(1) * torch.tensor([0.55, 0.42, 0.30]).view(1, 3, 1, 1)

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
        base = base + b.unsqueeze(1) * torch.tensor(col).view(1, 3, 1, 1)

    # Slow intentional sinusoidal illumination (~1 cycle over the clip).
    illum = (1.0 + 0.12 * torch.sin(2 * math.pi * t / T)).view(T, 1, 1, 1)
    scene_px = (base * illum).clamp(0.0, 1.0)
    return scene_px, torch.stack((ox, oy), dim=1)            # [T,3,H,W], obj [T,2]


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
# Per-frame gate diagnostic: run the engine frame-by-frame, capturing the gate's
# coherence C(t) and applied gate g(t). Prints a compact summary so the gate's
# behaviour on this scene is visible (and honest) rather than a black box.
# --------------------------------------------------------------------------- #
def gate_trace(eng: SpectralCoherenceEngine, scene_px: torch.Tensor,
               H: int, W: int):
    st = eng.init_state(H, W)
    Cs, gs = [], []
    for tt in range(scene_px.shape[0]):
        _, st = eng.process_frame(scene_px[tt], st)
        Cs.append(st.gate_C)
        gs.append(st.gate_g)
    return torch.tensor(Cs), torch.tensor(gs)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", choices=["translate", "accelerate", "curve",
                                        "wander"], default="translate")
    ap.add_argument("--legacy", action="store_true",
                    help="drop the M3 gate (reproduce the pre-M3 frozen behaviour)")
    ap.add_argument("--diag", action="store_true",
                    help="print the per-frame gate coherence C(t) / gate g(t)")
    # gate knobs (override SHIP_CFG for tuning / sweeps)
    ap.add_argument("--gate-lo", type=float, default=None)
    ap.add_argument("--gate-hi", type=float, default=None)
    ap.add_argument("--gate-decay", type=float, default=None)
    ap.add_argument("--gate-pos-decay", type=float, default=None)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    T, G = args.frames, args.grid
    CUT = SHIP_CFG["cutoff_frac"]

    cfg = dict(SHIP_CFG)
    if args.legacy:
        cfg["mc_gate"] = False
    if args.gate_lo is not None:
        cfg["mc_gate_lo"] = args.gate_lo
    if args.gate_hi is not None:
        cfg["mc_gate_hi"] = args.gate_hi
    if args.gate_decay is not None:
        cfg["mc_gate_decay"] = args.gate_decay
    if args.gate_pos_decay is not None:
        cfg["mc_gate_pos_decay"] = args.gate_pos_decay

    scene, obj_true = make_trap_scene(args.scene, T, G, G)   # raw == clean

    eng = SpectralCoherenceEngine(**cfg)
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

    mode = "LEGACY (no gate)" if args.legacy else "M3 gated"
    print("=" * 84)
    print(f"TRAP  scene={args.scene}  {mode}   grid={G}x{G}  frames={T}  seed={args.seed}")
    print(f"  object: legit low-band path (sigma={OBJ_SIGMA}) + illum sinusoid + 3 balls;"
          f"  NO injected wander")
    print(f"  config: complex_mc band={cfg['mc_band']} strength={cfg['mc_strength']} "
          f"corr hard  alpha={cfg['alpha']} rho={cfg['rho']} cutoff={CUT}")
    if not args.legacy:
        print(f"          gate ON  lo={cfg['mc_gate_lo']} hi={cfg['mc_gate_hi']} "
              f"decay={cfg['mc_gate_decay']}")
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
    print(f"  VERDICT: {'PASS — plugin leaves legitimate motion alone' if all_pass else 'FAIL'}")
    print("=" * 84)

    C_arr = g_arr = None
    if args.diag and not args.legacy:
        C_arr, g_arr = gate_trace(SpectralCoherenceEngine(**cfg), scene, G, G)
        # coherence rises after the first couple of frames; summarize the settled part
        tail = slice(min(4, T - 1), T)
        print("  gate diagnostic (per-frame coherence C, applied gate g):")
        print(f"    C(t): mean {float(C_arr[tail].mean()):.3f}  "
              f"min {float(C_arr[tail].min()):.3f}  max {float(C_arr[tail].max()):.3f}")
        print(f"    g(t): mean {float(g_arr[tail].mean()):.3f}  "
              f"min {float(g_arr[tail].min()):.3f}  max {float(g_arr[tail].max()):.3f}   "
              f"(g->0 = correction OFF, motion preserved; g->1 = full correction)")
        step = max(1, T // 12)
        idxs = list(range(0, T, step))
        print("    t   : " + " ".join(f"{i:4d}" for i in idxs))
        print("    C   : " + " ".join(f"{float(C_arr[i]):4.2f}" for i in idxs))
        print("    g   : " + " ".join(f"{float(g_arr[i]):4.2f}" for i in idxs))
        print("=" * 84)

    if args.json:
        rec = dict(config=dict(frames=T, grid=G, seed=args.seed, scene=args.scene,
                               legacy=args.legacy, cfg=cfg),
                   retention=retention, distortion=distortion,
                   distortion_mean=distortion_mean, hf_ssim=hf,
                   lowband_var_raw=lbv_raw, lowband_var_plugin=lbv_plug,
                   var_ratio=var_ratio, est_x_range=est_x_range,
                   true_x_range=true_x_range, passes=passes, all_pass=all_pass)
        if C_arr is not None:
            rec["gate_C"] = C_arr.tolist()
            rec["gate_g"] = g_arr.tolist()
        d = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(d, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()

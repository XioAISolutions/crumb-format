"""M0 — synthetic-drift test for crumb_coherence (PLUGIN_SPEC §7, milestone M0).

Take a real, coherent moving-ball clip (data.make_clip_batch), inject KNOWN
low-band drift (slow exposure ramp + a wandering low-frequency brightness/color
field), then run the stats-EMA baseline and the spectral engine (dc_only /
magnitude / complex) and report:

  * drift reduction %   (lowband_trajectory_variance and delta_e_vs_ref)
  * highfreq_ssim(in,out)   — did we harm detail?
  * motion preservation     — corrected centroid path length vs the clean control
  * state bytes

M0 criteria: >=60% drift removed, highfreq_ssim >= 0.98, motion within 5%.

Everything runs on CPU. A tiny side-by-side mp4 is written if imageio is present.

Usage:  python crumb_coherence/scripts/run_m0.py [--frames 48] [--grid 64]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # wavefield_video/
sys.path.insert(0, _ROOT)

from data import make_clip_batch                          # noqa: E402
from crumb_coherence import (                             # noqa: E402
    SpectralCoherenceEngine, StatsEMAEngine,
    lowband_trajectory_variance, delta_e_vs_ref, highfreq_ssim, temporal_flicker,
)
from crumb_coherence.metrics import _blur                 # noqa: E402


# --------------------------------------------------------------------------- #
# Known low-band drift: exposure ramp + wandering low-freq color field.
# Both are global / low-spatial-frequency, so they land squarely in the band
# the plugin anchors, and they leave ball positions (phase) untouched.
# --------------------------------------------------------------------------- #
def inject_drift(frames: torch.Tensor) -> torch.Tensor:
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
    spectral engine tracks it). Drift metric: lowband_trajectory_variance."""
    T, C, H, W = frames.shape
    t = torch.linspace(0, 1, T)
    ys = torch.arange(H).float().view(1, H, 1)
    xs = torch.arange(W).float().view(1, 1, W)
    cy = (H / 2) + (H / 4) * torch.sin(2 * math.pi * 0.7 * t)      # [T]
    cx = (W / 2) + (W / 4) * torch.sin(2 * math.pi * 0.4 * t + 1.1)
    sig = max(H, W) / 8.0
    # broadcast: [T,H,1] & [T,1,W]
    bump = torch.exp(-((ys - cy.view(T, 1, 1)) ** 2 + (xs - cx.view(T, 1, 1)) ** 2 * 0)
                        / (2 * sig * sig))
    bump = torch.exp(-((ys - cy.view(T, 1, 1)) ** 2) / (2 * sig * sig)) * \
           torch.exp(-((xs - cx.view(T, 1, 1)) ** 2) / (2 * sig * sig))
    bump = bump.unsqueeze(1)                                     # [T,1,H,W]
    amp = torch.tensor([0.22, 0.16, 0.10]).view(1, C, 1, 1)
    field = amp * bump
    field = field - field.mean(dim=(2, 3), keepdim=True)          # mean-preserving
    return (frames + field).clamp(0, 1)


def peak_path(frames: torch.Tensor) -> float:
    """Path length of the brightest high-passed point per frame (robust motion
    measure: low-frequency drift cannot create sharp maxima)."""
    T, C, H, W = frames.shape
    luma = frames.max(dim=1).values.unsqueeze(1)
    hp = (luma - _blur(luma, sigma=2.0)).clamp(min=0.0).squeeze(1)   # [T,H,W]
    flat = hp.flatten(1)
    idx = flat.argmax(dim=1)
    yy = (idx // W).float(); xx = (idx % W).float()
    pts = torch.stack((xx, yy), dim=1)
    return float((pts[1:] - pts[:-1]).norm(dim=1).sum().item())


def centroids(frames: torch.Tensor) -> torch.Tensor:
    """Brightness-weighted centroid of the moving balls (high-pass isolates
    them from the smooth drift field). frames [T,3,H,W] -> [T,2] (x,y)."""
    T, C, H, W = frames.shape
    luma = frames.max(dim=1).values.unsqueeze(1)         # [T,1,H,W]
    hp = (luma - _blur(luma, sigma=2.5)).clamp(min=0.0)   # ball energy
    ys = torch.arange(H).view(1, H, 1).float()
    xs = torch.arange(W).view(1, 1, W).float()
    w = hp.squeeze(1)                                     # [T,H,W]
    wsum = w.sum(dim=(1, 2)) + 1e-8
    cx = (w * xs).sum(dim=(1, 2)) / wsum
    cy = (w * ys).sum(dim=(1, 2)) / wsum
    return torch.stack((cx, cy), dim=1)                  # [T,2]


def path_length(c: torch.Tensor) -> float:
    return float((c[1:] - c[:-1]).norm(dim=1).sum().item())


def pct_drop(before: float, after: float) -> float:
    if before <= 1e-12:
        return 0.0
    return 100.0 * (1.0 - after / before)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--rho", type=float, default=0.995)
    ap.add_argument("--cutoff", type=float, default=0.10)
    ap.add_argument("--scenario", choices=["gain_field", "hotspot"], default="gain_field")
    ap.add_argument("--outdir", type=str, default=os.path.join(_ROOT, "crumb_coherence", "out"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    T, G = args.frames, args.grid

    # Coherent control clip (CPU, small grid). collisions couple the balls so
    # the motion is a genuine test, not a straight line.
    clip = make_clip_batch(1, T, G, G, seed=args.seed, collisions=True)  # [1,T+1,3,H,W]
    control = clip[0, :T].clone()                        # [T,3,H,W]
    drifted = (inject_hotspot(control) if args.scenario == "hotspot"
               else inject_drift(control))

    cutoff = args.cutoff
    var_drift = lowband_trajectory_variance(drifted, cutoff)
    de_drift = float(delta_e_vs_ref(drifted).mean().item())
    var_ctrl = lowband_trajectory_variance(control, cutoff)
    pl_control = peak_path(control)
    flick_drift = temporal_flicker(drifted)

    print("=" * 78)
    print(f"M0 drift test  scenario={args.scenario}  grid={G}x{G}  frames={T}  seed={args.seed}")
    print(f"  alpha={args.alpha}  rho={args.rho}  cutoff_frac={cutoff}")
    print("=" * 78)
    print(f"injected drift  : lowband_var={var_drift:.4e}  meanΔE={de_drift:.3f}  "
          f"flicker={flick_drift:.2e}")
    print(f"clean control   : lowband_var={var_ctrl:.4e}  peak_path={pl_control:.2f}px")
    print(f"  (control lowband_var is the noise floor; drift adds on top)")
    print("-" * 78)
    header = (f"{'engine':<22}{'drift%(var)':>11}{'drift%(ΔE)':>11}"
              f"{'hf_ssim':>9}{'motion%':>9}{'state':>10}  verdict")
    print(header)
    print("-" * 78)

    engines = [
        ("stats-EMA baseline", StatsEMAEngine(alpha=args.alpha, rho=args.rho)),
        ("spectral dc_only", SpectralCoherenceEngine(
            alpha=args.alpha, rho=args.rho, cutoff_frac=cutoff, anchor_mode="dc_only")),
        ("spectral magnitude", SpectralCoherenceEngine(
            alpha=args.alpha, rho=args.rho, cutoff_frac=cutoff, anchor_mode="magnitude")),
        ("spectral complex", SpectralCoherenceEngine(
            alpha=args.alpha, rho=args.rho, cutoff_frac=cutoff, anchor_mode="complex")),
    ]

    results = {}
    for name, eng in engines:
        st = eng.init_state(G, G)
        corrected, _ = eng.process_segment(drifted, st)
        var_c = lowband_trajectory_variance(corrected, cutoff)
        de_c = float(delta_e_vs_ref(corrected).mean().item())
        ssim = highfreq_ssim(drifted, corrected)
        pl_c = peak_path(corrected)
        motion_pct = 100.0 * pl_c / (pl_control + 1e-8)
        d_var = pct_drop(var_drift, var_c)
        d_de = pct_drop(de_drift, de_c)
        sb = eng.state_bytes(G, G)
        ok_drift = d_var >= 60.0
        ok_ssim = ssim >= 0.98
        ok_motion = abs(motion_pct - 100.0) <= 5.0
        verdict = "PASS" if (ok_drift and ok_ssim and ok_motion) else "fail"
        flags = "".join(["D" if ok_drift else ".",
                         "S" if ok_ssim else ".",
                         "M" if ok_motion else "."])
        results[name] = corrected
        print(f"{name:<22}{d_var:>10.1f}%{d_de:>10.1f}%{ssim:>9.4f}"
              f"{motion_pct:>8.1f}%{sb:>9}B  {verdict} [{flags}]")

    print("-" * 78)
    print("verdict flags: D=drift>=60%  S=ssim>=0.98  M=motion within 5%")
    print("state bytes @512x512 (spec headline):")
    for name, eng in engines:
        print(f"    {name:<22}{eng.state_bytes(512, 512):>8} B")
    print("=" * 78)
    print("Reading: 'complex' should remove the most drift but FREEZE motion "
          "(motion% << 100)\n         — that is the §3.2 failure the 'magnitude' "
          "default is designed to avoid.")

    if not args.no_video:
        _save_video(control, drifted, results.get("spectral magnitude"),
                    args.outdir)


def _save_video(control, drifted, corrected, outdir):
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
    path = os.path.join(outdir, "m0_sidebyside.mp4")
    try:
        imageio.mimsave(path, frames, fps=12, macro_block_size=1)
        print(f"[video] wrote {path}  (control | drifted | corrected-magnitude)")
    except Exception as exc:                              # noqa: BLE001
        gif = os.path.join(outdir, "m0_sidebyside.gif")
        imageio.mimsave(gif, frames, fps=12)
        print(f"[video] mp4 failed ({exc}); wrote {gif} instead.")


if __name__ == "__main__":
    main()

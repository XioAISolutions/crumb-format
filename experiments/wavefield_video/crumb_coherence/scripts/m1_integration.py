"""M1 — integration rung: the crumb_coherence plugin on a REAL wave-model
rollout, with a watchable before/after (2026-09-26).

M0 tuned + froze the plugin on SYNTHETIC clips. M1 is the first artifact of the
plugin applied to ACTUAL model-generated video: a trained wave-kernel
VideoPredictor rolled out autoregressively (content = moving balls, "wave" = the
model's dispersion kernel). There is NO clean reference for model output, so
every number here is SELF-REFERENTIAL (raw rollout vs plugin output, or the
clip's own temporal statistics) — stated honestly, never dressed up as accuracy.

Frame source (default, robust, zero re-render): reuse an existing rendered
rollout under demo_out/*/prediction/. render_rollout.py saves those PNGs
NEAREST-upscaled by an integer `scale` from the model's native grid, so resizing
each PNG back down to the native grid recovers the true model output losslessly
(pixel-replicated blocks collapse exactly). Alternatively `--ckpt` shells out to
render_rollout.py to render a fresh rollout first (e.g. the owner's S2 at 64x64,
if present) and then integrates that.

Plugin configs run (shipped kwargs, package FROZEN — nothing under
crumb_coherence/ is modified; this script only *calls* it):
  * luminance / energy axis  -> anchor_mode="magnitude" (the gain_field winner);
    grid-agnostic, holds low-band energy/contrast steady, leaves phase (motion)
    free. This is the config featured in the before/after MP4.
  * recommended hotspot cfg  -> anchor_mode="complex_mc", mc_strength=0.5,
    mc_band=0.03, mc_est="corr", mc_edge="hard". NOTE: mc_band is in cycles/pixel;
    at a small native grid (16/32) the r<0.03 inner band sits BELOW the
    fundamental (1/16=0.0625), so the band mask collapses to DC and complex_mc
    degenerates to plain `complex` low-band anchoring. Reported honestly; the
    positional correction that shone at grid 64 needs grid-64 model output
    (S2@64, not available locally).

Self-referential metrics (reuse crumb_coherence/metrics.py where possible):
  * low-band drift energy   : lowband_trajectory_variance (magnitude wander) +
    a local low-band energy-centroid variance (scene-centroid drift proxy).
  * cycle / flicker         : temporal_flicker overall, plus a band split
    (flicker below vs above the plugin's cutoff) — low should drop, HIGH must
    NOT (that would mean the plugin froze motion).
  * detail preservation     : highfreq_ssim(raw, plugin) must stay ~1 (plugin
    only moves the low band); plus a stillness guard (plugin flicker not ~0).

Usage:
  python crumb_coherence/scripts/m1_integration.py                     # reuse default rollout
  python crumb_coherence/scripts/m1_integration.py --frames 128
  python crumb_coherence/scripts/m1_integration.py --ckpt runs_dd3/model_wave_S2.pt --grid 64
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # wavefield_video/
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from crumb_coherence import (                              # noqa: E402
    SpectralCoherenceEngine, lowband_trajectory_variance, temporal_flicker,
    highfreq_ssim, low_band_box, put_low_band,
)
from crumb_coherence.core import box_dims, _box_freqs      # noqa: E402

DEFAULT_ROLLOUT = os.path.join(_ROOT, "demo_out", "seed_170001_wave", "prediction")
RENDERS_DIR = os.path.join(_ROOT, "renders")


# --------------------------------------------------------------------------- #
# Frame loading — recover native model frames from the upscaled prediction PNGs.
# --------------------------------------------------------------------------- #
def _read_grid_from_metrics(rollout_dir: str, fallback: int | None) -> int:
    """Native grid from the sibling metrics.json config, else the CLI fallback."""
    mj = os.path.join(os.path.dirname(rollout_dir.rstrip("/")), "metrics.json")
    if os.path.exists(mj):
        with open(mj) as f:
            cfg = json.load(f).get("config", {})
        if "grid" in cfg:
            return int(cfg["grid"])
    if fallback is None:
        raise SystemExit(f"no metrics.json next to {rollout_dir}; pass --grid")
    return fallback


def load_rollout_frames(rollout_dir: str, n_frames: int,
                        grid: int | None) -> tuple[torch.Tensor, int, int]:
    """Load up to n_frames prediction PNGs and recover the native grid.

    render_rollout.py wrote each PNG as an integer-NEAREST upscale of the native
    grid, so resizing back down to (grid, grid) with NEAREST returns the exact
    native model output (a pixel-replicated block collapses to its own value).
    Returns (frames [T,3,H,W] float in [0,1], native_grid, view_png_size).
    """
    from PIL import Image
    files = sorted(f for f in os.listdir(rollout_dir) if f.endswith(".png"))
    if not files:
        raise SystemExit(f"no PNG frames in {rollout_dir}")
    files = files[:n_frames]
    g = _read_grid_from_metrics(rollout_dir, grid)
    frames = []
    png_size = None
    for name in files:
        im = Image.open(os.path.join(rollout_dir, name)).convert("RGB")
        png_size = im.size[0]
        native = im.resize((g, g), Image.NEAREST)         # exact inverse of the upscale
        arr = torch.from_numpy(np.asarray(native, dtype=np.float32) / 255.0)
        frames.append(arr.permute(2, 0, 1).contiguous())   # [3,g,g] in [0,1]
    return torch.stack(frames, 0), g, png_size


def render_fresh(ckpt: str, grid: int, frames: int, seed: int,
                 out_dir: str) -> str:
    """Shell out to render_rollout.py (its exact CLI) to produce a fresh rollout,
    then return its prediction/ dir. Used only when --ckpt is given (e.g. S2@64).
    render_rollout.py owns the model/rollout loop; we do not reimplement it."""
    cfg = ckpt.replace(".pt", ".json").replace("model_", "result_")
    cmd = [sys.executable, os.path.join(_ROOT, "render_rollout.py"),
           "--ckpt", ckpt, "--kind", "wave", "--kernel-version", "dispersion",
           "--grid", str(grid), "--frames", str(frames), "--seed", str(seed),
           "--device", "cpu", "--out", out_dir]
    if os.path.exists(cfg):
        cmd += ["--config", cfg]
    print(f"[render] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=_ROOT)
    return os.path.join(out_dir, "prediction")


# --------------------------------------------------------------------------- #
# Plugin runner.
# --------------------------------------------------------------------------- #
def run_engine(frames: torch.Tensor, **cfg) -> torch.Tensor:
    eng = SpectralCoherenceEngine(**cfg)
    H, W = frames.shape[-2], frames.shape[-1]
    out, _ = eng.process_segment(frames, eng.init_state(H, W))
    return out


# --------------------------------------------------------------------------- #
# Self-referential metrics (no clean reference exists for model output).
# --------------------------------------------------------------------------- #
def _reconstruct_low(frames: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Low-band-only reconstruction [T,3,H,W] (HF zeroed) — for band splitting."""
    T, C, H, W = frames.shape
    out = []
    for t in range(T):
        X = torch.fft.rfft2(frames[t])
        Xz = put_low_band(torch.zeros_like(X), low_band_box(X, cutoff))
        out.append(torch.fft.irfft2(Xz, s=(H, W)))
    return torch.stack(out, 0)


def lowband_centroid_var(frames: torch.Tensor, cutoff: float) -> float:
    """Temporal variance (px^2) of the wandering low-band energy centroid — the
    scene-centroid drift proxy. Reconstruct the low band, strip its temporal mean
    (the static component), and track the centroid of the positive lobe over time
    (same construction run_m0 uses for the synthetic hotspot, here on the clip's
    OWN low band — self-referential, no reference needed)."""
    T, C, H, W = frames.shape
    lows = _reconstruct_low(frames, cutoff).mean(dim=1)    # [T,H,W] luma low band
    resid = lows - lows.mean(dim=0, keepdim=True)          # strip static component
    w = resid.clamp(min=0.0)                               # wandering positive lobe
    ys = torch.arange(H).view(1, H, 1).float()
    xs = torch.arange(W).view(1, 1, W).float()
    wsum = w.sum(dim=(1, 2)) + 1e-8
    cx = (w * xs).sum(dim=(1, 2)) / wsum
    cy = (w * ys).sum(dim=(1, 2)) / wsum
    pos = torch.stack((cx, cy), dim=1)                     # [T,2]
    return float(pos.var(dim=0, unbiased=False).sum().item())


def band_split_flicker(frames: torch.Tensor, cutoff: float) -> tuple[float, float]:
    """(low-band flicker, high-band flicker) — temporal_flicker on the low-only
    and high-only reconstructions. Anchoring should shrink the LOW number
    (drift/flicker) while leaving the HIGH number (motion detail) intact."""
    low = _reconstruct_low(frames, cutoff)
    high = frames - low
    return temporal_flicker(low), temporal_flicker(high)


def evaluate(raw: torch.Tensor, corrected: torch.Tensor, cutoff: float) -> dict:
    lo_c_raw = lowband_centroid_var(raw, cutoff)
    lo_c_cor = lowband_centroid_var(corrected, cutoff)
    lbv_raw = lowband_trajectory_variance(raw, cutoff)
    lbv_cor = lowband_trajectory_variance(corrected, cutoff)
    lf_raw, hf_raw = band_split_flicker(raw, cutoff)
    lf_cor, hf_cor = band_split_flicker(corrected, cutoff)

    def drop(a, b):                                        # % reduction, +ve = good
        return 100.0 * (1.0 - b / a) if a > 1e-12 else 0.0

    return dict(
        lowband_mag_var_raw=lbv_raw, lowband_mag_var_cor=lbv_cor,
        lowband_mag_var_drop=drop(lbv_raw, lbv_cor),
        centroid_var_raw=lo_c_raw, centroid_var_cor=lo_c_cor,
        centroid_var_drop=drop(lo_c_raw, lo_c_cor),
        low_flicker_raw=lf_raw, low_flicker_cor=lf_cor,
        low_flicker_drop=drop(lf_raw, lf_cor),
        high_flicker_raw=hf_raw, high_flicker_cor=hf_cor,
        high_flicker_ratio=(hf_cor / hf_raw) if hf_raw > 1e-12 else float("nan"),
        detail_hfssim=highfreq_ssim(raw, corrected),       # vs raw model output
        plugin_flicker=temporal_flicker(corrected),        # stillness guard (>0)
        raw_flicker=temporal_flicker(raw),
    )


def band_note(grid: int, cutoff: float, mc_band: float) -> str:
    """Diagnose complex_mc band degeneracy at this grid (honesty aid)."""
    kh, kw = box_dims(cutoff, grid, grid)
    ky, kx = _box_freqs(kh, kw, grid, grid)
    r = torch.sqrt(ky * ky + kx * kx)
    n_in = int((r < mc_band).sum().item())
    fund = 1.0 / grid
    return (f"low-band box {kh}x{kw}; cells with r<{mc_band} = {n_in} "
            f"(fundamental 1/{grid}={fund:.4f} cyc/px). "
            + ("DEGENERATE (DC-only) -> complex_mc == complex at this grid."
               if n_in <= 1 else "non-degenerate."))


# --------------------------------------------------------------------------- #
# Before/after artifact.
# --------------------------------------------------------------------------- #
def _to_uint8(frame: torch.Tensor) -> "object":
    """[3,H,W] float in [0,1] -> PIL RGB image."""
    from PIL import Image
    arr = (frame.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8)
    return Image.fromarray(arr.contiguous().numpy(), mode="RGB")


def _panel(frame: torch.Tensor, view: int, label: str) -> "object":
    """One upscaled, labelled panel (PIL). Labels are drawn with PIL.ImageDraw —
    local ffmpeg has no drawtext, so text is baked into the pixels here instead."""
    from PIL import Image, ImageDraw
    im = _to_uint8(frame).resize((view, view), Image.NEAREST)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, view - 1, 12], fill=(5, 7, 11))     # label strip
    d.text((3, 2), label, fill=(220, 230, 240))            # default bitmap font
    return im


def write_before_after(raw, corrected, out_mp4: str, grid: int, label: str,
                        fps: int = 12) -> str:
    """Side-by-side (left=raw, right=plugin) MP4 via ffmpeg rawvideo stdin (no
    image-lib MP4 dep, no drawtext needed). Returns the ffmpeg command string."""
    from PIL import Image
    view = max(64, (256 // grid) * grid)                   # ~256px panels, integer x
    sep = 4
    W = view * 2 + sep
    H = view + (view % 2)                                  # even height for yuv420p
    if W % 2:
        W += 1
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for t in range(raw.shape[0]):
        canvas = Image.new("RGB", (W, H), (5, 7, 11))
        canvas.paste(_panel(raw[t], view, "RAW rollout"), (0, 0))
        canvas.paste(_panel(corrected[t], view, f"PLUGIN {label}"), (view + sep, 0))
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"ffmpeg failed (exit {ret}); is ffmpeg on PATH?")
    return " ".join(cmd)


def save_pairs(raw, corrected, out_dir: str, grid: int, label: str,
               k: int = 6) -> list:
    """Save k evenly-spaced before/after pair PNGs (raw|plugin), labelled."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    view = max(64, (256 // grid) * grid)
    sep = 4
    T = raw.shape[0]
    idxs = sorted(set(int(round(i)) for i in torch.linspace(0, T - 1, k).tolist()))
    saved = []
    for t in idxs:
        canvas = Image.new("RGB", (view * 2 + sep, view), (5, 7, 11))
        canvas.paste(_panel(raw[t], view, "RAW"), (0, 0))
        canvas.paste(_panel(corrected[t], view, f"PLUGIN {label}"), (view + sep, 0))
        path = os.path.join(out_dir, f"pair_{t:04d}.png")
        canvas.save(path)
        saved.append(path)
    return saved


# --------------------------------------------------------------------------- #
def _fmt(r: dict) -> str:
    return (f"    low-band mag var : {r['lowband_mag_var_raw']:.4e} -> "
            f"{r['lowband_mag_var_cor']:.4e}  ({r['lowband_mag_var_drop']:+.1f}% )\n"
            f"    centroid var(px2): {r['centroid_var_raw']:.4f} -> "
            f"{r['centroid_var_cor']:.4f}  ({r['centroid_var_drop']:+.1f}% )\n"
            f"    low-band flicker : {r['low_flicker_raw']:.4e} -> "
            f"{r['low_flicker_cor']:.4e}  ({r['low_flicker_drop']:+.1f}% )\n"
            f"    high-band flicker: {r['high_flicker_raw']:.4e} -> "
            f"{r['high_flicker_cor']:.4e}  (ratio {r['high_flicker_ratio']:.3f}; "
            f"~1 = motion kept)\n"
            f"    detail hf_ssim(raw,plugin): {r['detail_hfssim']:.4f}  "
            f"(~1 = detail preserved)\n"
            f"    stillness guard  : plugin flicker {r['plugin_flicker']:.4e} vs "
            f"raw {r['raw_flicker']:.4e}  (must be >> 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", default=DEFAULT_ROLLOUT,
                    help="prediction/ dir of PNG frames (default: demo_out wave rollout)")
    ap.add_argument("--ckpt", default="", help="render fresh from this checkpoint first")
    ap.add_argument("--frames", type=int, default=64, help="max frames to use")
    ap.add_argument("--grid", type=int, default=None, help="native grid (else from metrics.json)")
    ap.add_argument("--seed", type=int, default=170001)
    ap.add_argument("--lum-cutoff", type=float, default=0.10)   # gain_field default
    ap.add_argument("--lum-alpha", type=float, default=0.9)
    ap.add_argument("--rho", type=float, default=0.995)
    ap.add_argument("--hot-cutoff", type=float, default=0.14)   # hotspot default
    ap.add_argument("--hot-alpha", type=float, default=0.95)
    ap.add_argument("--json", default=os.path.join(RENDERS_DIR, "m1_metrics.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    rollout_dir = args.rollout_dir
    if args.ckpt:
        g = args.grid or 64
        rollout_dir = render_fresh(args.ckpt, g, args.frames, args.seed,
                                   os.path.join(RENDERS_DIR, "m1_fresh_rollout"))

    frames, grid, png_size = load_rollout_frames(rollout_dir, args.frames, args.grid)
    T = frames.shape[0]

    print("=" * 84)
    print("M1  integration — crumb_coherence on a REAL wave-model rollout")
    print(f"  source: {os.path.relpath(rollout_dir, _ROOT)}  frames={T}  "
          f"native grid={grid}x{grid}  (display PNG {png_size}px)")
    print("  NO clean reference exists for model output -> all numbers are")
    print("  SELF-REFERENTIAL (raw rollout vs plugin, or the clip's own stats).")
    print("=" * 84)

    # --- Config A: luminance / energy axis (featured) --------------------- #
    lum = run_engine(frames, alpha=args.lum_alpha, rho=args.rho,
                     cutoff_frac=args.lum_cutoff, anchor_mode="magnitude")
    r_lum = evaluate(frames, lum, args.lum_cutoff)
    print(f"\n[A] luminance/energy axis  anchor_mode=magnitude  "
          f"cutoff={args.lum_cutoff}  alpha={args.lum_alpha}  rho={args.rho}")
    print(_fmt(r_lum))

    # --- Config B: recommended hotspot config ----------------------------- #
    hot = run_engine(frames, alpha=args.hot_alpha, rho=args.rho,
                     cutoff_frac=args.hot_cutoff, anchor_mode="complex_mc",
                     mc_strength=0.5, mc_band=0.03, mc_est="corr", mc_edge="hard")
    r_hot = evaluate(frames, hot, args.hot_cutoff)
    print(f"\n[B] recommended hotspot cfg  anchor_mode=complex_mc  mc_band=0.03  "
          f"strength=0.5  corr  hard  cutoff={args.hot_cutoff}  alpha={args.hot_alpha}")
    print(f"    band diagnosis: {band_note(grid, args.hot_cutoff, 0.03)}")
    print(_fmt(r_hot))

    # --- Artifact: featured = luminance axis ------------------------------ #
    out_mp4 = os.path.join(RENDERS_DIR, "m1_before_after.mp4")
    ff = write_before_after(frames, lum, out_mp4, grid, "magnitude")
    pairs = save_pairs(frames, lum, os.path.join(RENDERS_DIR, "m1_pairs"), grid,
                       "magnitude")
    print(f"\n[artifact] wrote {os.path.relpath(out_mp4, _ROOT)}  (left=raw, "
          f"right=plugin[magnitude]; labels baked via PIL, ffmpeg drawtext N/A)")
    print(f"[artifact] wrote {len(pairs)} pairs under "
          f"{os.path.relpath(os.path.join(RENDERS_DIR, 'm1_pairs'), _ROOT)}/")
    print(f"[artifact] ffmpeg: {ff}")

    print("\nFalsifies 'the plugin helps' if, on this real rollout:")
    print("  - low-band drift energy / centroid var / low-band flicker do NOT drop")
    print("    (plugin does not stabilize the low band), OR")
    print("  - detail hf_ssim(raw,plugin) collapses (< ~0.95: detail damaged), OR")
    print("  - high-band flicker ratio -> 0 or plugin flicker -> 0 (plugin froze")
    print("    the scene into stillness — the anti-collapse guard).")

    record = dict(config=dict(source=rollout_dir, frames=T, grid=grid,
                              lum_cutoff=args.lum_cutoff, hot_cutoff=args.hot_cutoff,
                              rho=args.rho),
                  luminance=r_lum, hotspot=r_hot,
                  band_note=band_note(grid, args.hot_cutoff, 0.03))
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[json] wrote {os.path.relpath(args.json, _ROOT)}")
    print("=" * 84)


if __name__ == "__main__":
    main()

"""M1b — demo v2: make the plugin's effect LEGIBLE to a human eye (2026-09-26).

Owner feedback on the M1 before/after (16x16 upscaled, magnitude axis): "not sure
if that video is good or not." Correct — at grid 16 the effect is subtle. This
builds three self-evident artifacts with the SAME frozen engine (nothing under
crumb_coherence/ changes; this only calls it):

  (a) THREE-PANEL SYNTHETIC (the clean proof): the M0 hotspot at 64x64, ~96
      frames, panels upscaled x4. [DRIFTED input | ENGINE output | CLEAN truth].
      Here the drift is KNOWN and CONTROLLED (~44 px^2 low-band position variance)
      and the engine config is the M0.6 winner (complex_mc band=0.03, non-
      degenerate at grid 64), so the corrected panel visibly sits like the clean
      one while the drifted panel wanders. -> renders/m1b_three_panel.mp4

  (b) BLINK COMPARATOR (real rollout): the grid-16 wave-model rollout, upscaled
      x8, playing forward while the SOURCE toggles raw<->plugin every ~0.4s so the
      eye catches the relative wobble; frame index + source burned in. ->
      renders/m1b_blink_real.mp4

  (c) CENTROID-TRAJECTORY (real rollout): per-frame low-band energy centroid of
      raw vs plugin over the full rollout, as a static figure (matplotlib if
      present, else a PIL line-drawer) AND a burn-in video (video beside the graph
      with moving markers). -> renders/m1b_trajectory.png + renders/m1b_trajectory.mp4

Real-rollout config: complex_mc band=0.03 (the recommended config). At grid 16
mc_band=0.03 is below the fundamental (1/16=0.0625) so it acts as `complex`
low-band anchoring — the position-stabilizing behaviour, which is what a blink /
trajectory reveals. HONEST LIMIT: on real content that anchoring cannot tell
drift from real motion, so any centroid tightening mixes drift-removal with
motion-damping. The SYNTHETIC three-panel is the controlled proof; the real-
rollout artifacts show the plugin acting on true model output, subtly, at g16.

Watch order: three-panel -> blink -> trajectory.

Usage:
  python crumb_coherence/scripts/m1_demo_v2.py                # all three
  python crumb_coherence/scripts/m1_demo_v2.py --only three_panel
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from data import make_clip_batch                             # noqa: E402
import run_m0                                                # noqa: E402
from m1_integration import (                                 # noqa: E402
    load_rollout_frames, run_engine, _panel, _reconstruct_low,
    DEFAULT_ROLLOUT, RENDERS_DIR,
)

inject_hotspot = run_m0.inject_hotspot

# M0.6 winner (non-degenerate at grid 64) — used for BOTH the synthetic panel and
# the real rollout (where at g16 it degrades to complex anchoring; see header).
HOTSPOT_CFG = dict(alpha=0.95, rho=0.995, cutoff_frac=0.14, anchor_mode="complex_mc",
                   mc_strength=0.5, mc_band=0.03, mc_est="corr", mc_edge="hard")
BG = (5, 7, 11)


# --------------------------------------------------------------------------- #
# ffmpeg rawvideo-stdin encoder (same dependency-free pattern as m1_integration).
# --------------------------------------------------------------------------- #
def encode(canvas_iter, W: int, H: int, out_mp4: str, fps: int) -> str:
    assert W % 2 == 0 and H % 2 == 0, f"even dims required, got {W}x{H}"
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = 0
    for canvas in canvas_iter:
        proc.stdin.write(canvas.tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed encoding {out_mp4}; is ffmpeg on PATH?")
    return f"{n} frames -> {os.path.relpath(out_mp4, _ROOT)}"


def _even(x: int) -> int:
    return x + (x % 2)


# --------------------------------------------------------------------------- #
# (a) THREE-PANEL SYNTHETIC — the clean, controlled proof.
# --------------------------------------------------------------------------- #
def three_panel(out_mp4: str, T: int = 96, grid: int = 64, view: int = 256,
                fps: int = 12) -> str:
    torch.manual_seed(1234)
    clip, meta = make_clip_batch(1, T, grid, grid, seed=1234, collisions=True,
                                 return_meta=True)
    control = clip[0, :T].clone()                            # CLEAN truth
    drifted = inject_hotspot(control)                        # DRIFTED input
    corrected = run_engine(drifted, **HOTSPOT_CFG)           # ENGINE output (g64)
    sep = 6
    W = _even(view * 3 + sep * 2)
    H = _even(view)

    def frames():
        for t in range(T):
            c = Image.new("RGB", (W, H), BG)
            c.paste(_panel(drifted[t], view, "DRIFTED input"), (0, 0))
            c.paste(_panel(corrected[t], view, "ENGINE output"), (view + sep, 0))
            c.paste(_panel(control[t], view, "CLEAN truth"), (2 * (view + sep), 0))
            yield c

    return encode(frames(), W, H, out_mp4, fps)


# --------------------------------------------------------------------------- #
# (b) BLINK COMPARATOR — real rollout, source toggles raw<->plugin every ~0.4s.
# --------------------------------------------------------------------------- #
def blink_real(frames: torch.Tensor, plugin: torch.Tensor, out_mp4: str,
               grid: int, scale: int = 8, fps: int = 12,
               block_sec: float = 0.4) -> str:
    view = _even(grid * scale)
    block = max(1, round(block_sec * fps))
    T = frames.shape[0]

    def gen():
        for t in range(T):
            use_plugin = (t // block) % 2 == 1
            src = plugin if use_plugin else frames
            tag = f"{'PLUGIN' if use_plugin else 'RAW '}  f{t:03d}"
            yield _panel(src[t], view, tag)

    return encode(gen(), view, view, out_mp4, fps)


# --------------------------------------------------------------------------- #
# (c) CENTROID TRAJECTORY — low-band energy centroid path, raw vs plugin.
# --------------------------------------------------------------------------- #
def lowband_centroid_path(frames: torch.Tensor, cutoff: float) -> torch.Tensor:
    """[T,2] (x,y) per-frame centroid of the low-band positive lobe, with the
    per-frame spatial DC removed so the wandering structure (not the static mean)
    drives the centroid. Self-referential; no reference needed."""
    low = _reconstruct_low(frames, cutoff).mean(dim=1)       # [T,H,W] luma low band
    T, H, W = low.shape
    low = low - low.mean(dim=(1, 2), keepdim=True)           # strip per-frame DC
    w = low.clamp(min=0.0)
    ys = torch.arange(H).view(1, H, 1).float()
    xs = torch.arange(W).view(1, 1, W).float()
    wsum = w.sum(dim=(1, 2)) + 1e-8
    cx = (w * xs).sum(dim=(1, 2)) / wsum
    cy = (w * ys).sum(dim=(1, 2)) / wsum
    return torch.stack((cx, cy), dim=1)                      # [T,2]


def _bounds(*paths):
    allp = torch.cat(paths, 0)
    lo = allp.min(0).values - 0.5
    hi = allp.max(0).values + 0.5
    span = torch.clamp(hi - lo, min=1e-3)
    return lo, hi, span


def _map_pt(pt, lo, span, box):
    """data (x,y) -> pixel (px,py) inside a [x0,y0,x1,y1] box (y flipped)."""
    x0, y0, x1, y1 = box
    fx = float((pt[0] - lo[0]) / span[0])
    fy = float((pt[1] - lo[1]) / span[1])
    px = x0 + fx * (x1 - x0)
    py = y1 - fy * (y1 - y0)                                  # flip y for image coords
    return px, py


def draw_graph_panel(size: int, raw_path, plugin_path, t: int | None,
                     lo, span) -> "Image.Image":
    """PIL line-drawer: both centroid paths + optional current-frame markers."""
    im = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(im)
    m = 34
    box = (m, m, size - m, size - m)
    d.rectangle(box, outline=(70, 80, 95))
    d.text((m, 6), "low-band centroid path (x,y)", fill=(210, 220, 230))
    d.text((m, size - 22), "raw", fill=(235, 90, 90))
    d.text((m + 40, size - 22), "plugin", fill=(90, 210, 235))

    def polyline(path, color):
        pts = [_map_pt(path[i], lo, span, box) for i in range(path.shape[0])]
        d.line(pts, fill=color, width=1)

    polyline(raw_path, (150, 70, 70))
    polyline(plugin_path, (70, 150, 165))
    if t is not None:
        for path, col in ((raw_path, (235, 90, 90)), (plugin_path, (90, 210, 235))):
            px, py = _map_pt(path[t], lo, span, box)
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col)
    return im


def save_trajectory_png(raw_path, plugin_path, out_png: str) -> str:
    """Static figure: matplotlib if available, else the PIL line-drawer."""
    rp, pp = raw_path.numpy(), plugin_path.numpy()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(rp[:, 0], rp[:, 1], "-", color="tab:red", lw=1, label="raw")
        ax[0].plot(pp[:, 0], pp[:, 1], "-", color="tab:cyan", lw=1, label="plugin")
        ax[0].set_title("low-band centroid path (x,y)")
        ax[0].set_xlabel("x (px)"); ax[0].set_ylabel("y (px)")
        ax[0].legend(); ax[0].invert_yaxis(); ax[0].set_aspect("equal", "box")
        tt = range(rp.shape[0])
        ax[1].plot(tt, rp[:, 0], color="tab:red", lw=1, label="raw x")
        ax[1].plot(tt, pp[:, 0], color="tab:cyan", lw=1, label="plugin x")
        ax[1].plot(tt, rp[:, 1], "--", color="tab:red", lw=1, label="raw y")
        ax[1].plot(tt, pp[:, 1], "--", color="tab:cyan", lw=1, label="plugin y")
        ax[1].set_title("centroid vs frame"); ax[1].set_xlabel("frame")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        return "matplotlib"
    except Exception as exc:                                 # PIL fallback
        lo, hi, span = _bounds(raw_path, plugin_path)
        im = draw_graph_panel(512, raw_path, plugin_path, None, lo, span)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        im.save(out_png)
        return f"PIL fallback ({type(exc).__name__})"


def trajectory(frames: torch.Tensor, plugin: torch.Tensor, out_png: str,
               out_mp4: str, grid: int, cutoff: float, fps: int = 12) -> tuple:
    raw_path = lowband_centroid_path(frames, cutoff)
    plugin_path = lowband_centroid_path(plugin, cutoff)
    backend = save_trajectory_png(raw_path, plugin_path, out_png)

    lo, hi, span = _bounds(raw_path, plugin_path)
    view = _even(grid * 16)                                  # video panel side
    graph = view
    sep = 6
    W = _even(view + sep + graph)
    H = _even(view)
    T = frames.shape[0]

    def gen():
        for t in range(T):
            c = Image.new("RGB", (W, H), BG)
            c.paste(_panel(frames[t], view, f"RAW rollout  f{t:03d}"), (0, 0))
            c.paste(draw_graph_panel(graph, raw_path, plugin_path, t, lo, span),
                    (view + sep, 0))
            yield c

    enc = encode(gen(), W, H, out_mp4, fps)
    rawvar = float(raw_path.var(0).sum())
    pluvar = float(plugin_path.var(0).sum())
    return backend, enc, rawvar, pluvar


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["three_panel", "blink", "trajectory"],
                    default="", help="run just one artifact (default: all three)")
    ap.add_argument("--rollout-dir", default=DEFAULT_ROLLOUT)
    ap.add_argument("--syn-frames", type=int, default=96)
    ap.add_argument("--real-frames", type=int, default=256)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(RENDERS_DIR, exist_ok=True)
    want = (lambda k: args.only in ("", k))

    print("=" * 84)
    print("M1b  legibility demos   (frozen engine; watch order: three-panel -> "
          "blink -> trajectory)")
    print("=" * 84)

    # (a) synthetic three-panel — the clean proof.
    if want("three_panel"):
        out = os.path.join(RENDERS_DIR, "m1b_three_panel.mp4")
        msg = three_panel(out, T=args.syn_frames, fps=args.fps)
        print(f"[a] three-panel (64x64 hotspot, x4): {msg}")
        print("    [DRIFTED input | ENGINE output | CLEAN truth] — corrected should "
              "sit like clean.")

    # Real rollout (shared by blink + trajectory).
    need_real = want("blink") or want("trajectory")
    if need_real:
        frames, grid, png = load_rollout_frames(args.rollout_dir, args.real_frames,
                                                None)
        plugin = run_engine(frames, **HOTSPOT_CFG)
        deg = "complex-anchoring (mc_band<fundamental)" if grid < 64 else "band-limited"
        print(f"[real] {os.path.relpath(args.rollout_dir, _ROOT)}  frames="
              f"{frames.shape[0]}  grid={grid}  (plugin acts as {deg} at this grid)")

    # (b) blink comparator.
    if want("blink"):
        out = os.path.join(RENDERS_DIR, "m1b_blink_real.mp4")
        msg = blink_real(frames, plugin, out, grid, scale=8, fps=args.fps)
        print(f"[b] blink (x8, toggle every ~0.4s): {msg}")

    # (c) centroid trajectory.
    if want("trajectory"):
        png = os.path.join(RENDERS_DIR, "m1b_trajectory.png")
        mp4 = os.path.join(RENDERS_DIR, "m1b_trajectory.mp4")
        backend, enc, rawvar, pluvar = trajectory(frames, plugin, png, mp4, grid,
                                                  HOTSPOT_CFG["cutoff_frac"],
                                                  fps=args.fps)
        print(f"[c] trajectory png ({backend}) -> {os.path.relpath(png, _ROOT)}")
        print(f"    burn-in: {enc}")
        print(f"    centroid path variance (px^2): raw={rawvar:.3f}  "
              f"plugin={pluvar:.3f}  ({100*(1-pluvar/rawvar):+.1f}% vs raw)"
              if rawvar > 1e-9 else "    centroid variance ~0 (no wander)")

    print("=" * 84)
    print("Watch order: renders/m1b_three_panel.mp4 -> m1b_blink_real.mp4 -> "
          "m1b_trajectory.mp4 (+ .png)")


if __name__ == "__main__":
    main()

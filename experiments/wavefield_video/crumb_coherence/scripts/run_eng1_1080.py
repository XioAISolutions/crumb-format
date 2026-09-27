"""ENG-1 — the 1080p real-footage gate (customer-safety engineering test, 2026-09-26).

The science gates pass (M0/M1). Before we can honestly sell a "coherence pass",
one gate is missing: REAL FOOTAGE AT FULL RESOLUTION. This runs the FROZEN default
pipeline (NO hand tuning) over real 1080x1920 30fps clips and asks two things:
  * does the pass harm legitimate motion at real resolution?
  * on clean, non-drifted content the correct behavior is NON-INTERFERENCE — it
    should change almost nothing measurable.

Frozen config (do NOT retune): complex_mc, mc_band=0.03, mc_strength=0.5,
phase_anchor=0.0, alpha=0.95, rho=0.995, cutoff_frac=0.14. (mc_est=corr,
mc_edge=hard are the shipped defaults; at 1080p the r<0.03 band is NON-degenerate
— fundamental 1/1920~5.2e-4 cyc/px — so this actually exercises the spatial-phase
mechanism M1 could not at grid 16.)

STREAMING: 1080x1920x3 float32 = ~24 MB/frame; ~680 frames would be ~16 GB, so we
never hold the clip in memory. ffmpeg decodes to a rawvideo pipe; each frame is
processed by the STATEFUL engine (state carried frame-to-frame — the shipped
streaming API) and released; metrics accumulate online; only an ~8 s side-by-side
excerpt is buffered/encoded.

Pass criteria (ALL must hold, evaluated on the native-1080 run of each clip):
  1. detail retention median            >= 0.97
  2. motion magnitude ratio             in [0.90, 1.10]
  3. no introduced jitter: corrected max frame-to-frame MAE <= 1.5x its own p95
  4. runtime                            <= 30 min / 22 s clip (CPU) — reported either way
  5. sanity: on the synthetic DRIFTED fixture the frozen pass STILL removes drift
     (>= 60% pos-drift reduction) — do not regress the science while chasing safety.

Honesty: if a criterion fails, it is reported plainly with numbers. Parameters are
NEVER tuned to pass. A measured "this is why we cannot sell yet" is a valid result.

Usage:
  python crumb_coherence/scripts/run_eng1_1080.py
  python crumb_coherence/scripts/run_eng1_1080.py --excerpt-sec 8 --resolutions 1080,720
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from crumb_coherence import (                                # noqa: E402
    SpectralCoherenceEngine, low_band_box, put_low_band,
)
from crumb_coherence.metrics import _rgb_to_luma, _blur       # noqa: E402
from data import make_clip_batch                              # noqa: E402
import run_m0                                                 # noqa: E402

EPS = 1e-8
SHIP_CFG = dict(alpha=0.95, rho=0.995, cutoff_frac=0.14, anchor_mode="complex_mc",
                mc_strength=0.5, mc_band=0.03, phase_anchor=0.0)
DATA_REAL = os.path.join(_ROOT, "data_real")
REPORTS = os.path.join(_ROOT, "reports")
RENDERS = os.path.join(_ROOT, "renders")
SRC_DIR = os.path.expanduser("~/math-reels/dist")
CLIPS = ["e_curve", "fib_spiral"]


# --------------------------------------------------------------------------- #
# ffmpeg helpers.
# --------------------------------------------------------------------------- #
def ffprobe_dims(path: str) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0:s=,", path],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h, rate = out.split(",")[:3]
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else float(num)
    return int(w), int(h), fps


def ensure_local(name: str) -> str:
    os.makedirs(DATA_REAL, exist_ok=True)
    dst = os.path.join(DATA_REAL, f"{name}.mp4")
    if os.path.exists(dst):
        return dst
    src = os.path.join(SRC_DIR, f"{name}.mp4")
    if os.path.exists(src):
        shutil.copy(src, dst)
        return dst
    raise SystemExit(f"missing {dst} and source {src}; copy the clip into data_real/")


def _read_exact(pipe, n: int):
    """Read exactly n bytes from a pipe (stdout may return short reads)."""
    chunks, got = [], 0
    while got < n:
        c = pipe.read(n - got)
        if not c:
            break
        chunks.append(c); got += len(c)
    return b"".join(chunks) if got == n else None


def decode_pipe(path: str, W: int, H: int):
    """ffmpeg decoder -> raw rgb24 frames at exactly WxH (scaled if needed)."""
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def encoder_pipe(W: int, H: int, fps: float, out_mp4: str):
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", f"{fps:.4f}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt",
         "yuv420p", "-crf", "20", out_mp4],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------- #
# Per-frame metric primitives.
# --------------------------------------------------------------------------- #
def hf_luma_vec(frame3: torch.Tensor) -> torch.Tensor:
    """Flattened high-pass luma residual (luma - gaussian blur) for a [3,H,W]."""
    lum = _rgb_to_luma(frame3.unsqueeze(0))                  # [1,1,H,W]
    return (lum - _blur(lum)).flatten()


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = float(a.norm() * b.norm()) + EPS
    return float((a * b).sum() / denom)


def centroid_of(field: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor):
    """(x,y) centroid of a [H,W] field's positive lobe after spatial-DC strip."""
    w = (field - field.mean()).clamp(min=0.0)
    wsum = float(w.sum()) + EPS
    return float((w * xs).sum()) / wsum, float((w * ys).sum()) / wsum


# --------------------------------------------------------------------------- #
# Stream one clip through the frozen engine, accumulating metrics online.
# --------------------------------------------------------------------------- #
def process_clip(path: str, W: int, H: int, fps: float, cutoff: float,
                 excerpt_frames: int, sxs_out: str | None):
    eng = SpectralCoherenceEngine(**SHIP_CFG)
    state = eng.init_state(H, W)
    ys = torch.arange(H).float().view(H, 1)
    xs = torch.arange(W).float().view(1, W)

    # online accumulators (float64 where summed over many frames)
    lb_sum_r = lb_sq_r = lb_sum_c = lb_sq_c = None      # low-band |box| stats [3,kh,kw]
    cen_r, cen_c = [], []                                # centroid paths
    detail = []                                          # per-frame HF pearson
    raw_diff, cor_diff = [], []                          # frame-to-frame MAE series
    motion_raw = motion_cor = 0.0
    prev_r = prev_c = None
    n = 0
    eng_s = 0.0                                          # engine-only time (product cost)

    dec = decode_pipe(path, W, H)
    enc = None
    view_w = 540
    view_h = int(round(H * view_w / W))
    if view_h % 2:
        view_h += 1
    sep = 6
    if sxs_out:
        enc = encoder_pipe(2 * view_w + sep, view_h, fps, sxs_out)

    fsize = W * H * 3
    t0 = time.perf_counter()
    while True:
        buf = _read_exact(dec.stdout, fsize)
        if buf is None:
            break
        raw = (torch.from_numpy(
            np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()).float() / 255.0
            ).permute(2, 0, 1).contiguous()                 # [3,H,W]
        _t = time.perf_counter()
        cor, state = eng.process_frame(raw, state)          # THE product (timed alone)
        eng_s += time.perf_counter() - _t

        # low-band magnitude stats (matches lowband_trajectory_variance defn)
        Xr = torch.fft.rfft2(raw); Xc = torch.fft.rfft2(cor)
        lbr = low_band_box(Xr, cutoff).abs().double()        # [3,kh,kw]
        lbc = low_band_box(Xc, cutoff).abs().double()
        if lb_sum_r is None:
            lb_sum_r = torch.zeros_like(lbr); lb_sq_r = torch.zeros_like(lbr)
            lb_sum_c = torch.zeros_like(lbc); lb_sq_c = torch.zeros_like(lbc)
        lb_sum_r += lbr; lb_sq_r += lbr * lbr
        lb_sum_c += lbc; lb_sq_c += lbc * lbc

        # centroid of the low-band reconstruction (luma), same estimator both sides
        rec_r = torch.fft.irfft2(put_low_band(torch.zeros_like(Xr),
                                              low_band_box(Xr, cutoff)), s=(H, W)).mean(0)
        rec_c = torch.fft.irfft2(put_low_band(torch.zeros_like(Xc),
                                              low_band_box(Xc, cutoff)), s=(H, W)).mean(0)
        cen_r.append(centroid_of(rec_r, ys, xs))
        cen_c.append(centroid_of(rec_c, ys, xs))

        # detail retention (HF pearson, raw vs corrected)
        detail.append(pearson(hf_luma_vec(raw), hf_luma_vec(cor)))

        # motion + frame-to-frame MAE series
        if prev_r is not None:
            dr = float((raw - prev_r).abs().mean())
            dc = float((cor - prev_c).abs().mean())
            motion_raw += dr; motion_cor += dc
            raw_diff.append(dr); cor_diff.append(dc)
        prev_r, prev_c = raw, cor

        # side-by-side excerpt (native pass only)
        if enc is not None and n < excerpt_frames:
            enc.stdin.write(_compose_sxs(raw, cor, view_w, view_h, sep, n))
        n += 1

    dec.wait()
    if enc is not None:
        enc.stdin.close(); enc.wait()
    runtime = time.perf_counter() - t0

    def mean_var(sm, sq):
        mean = sm / n
        var = (sq / n) - mean * mean
        return float(var.clamp(min=0).mean())

    lbv_raw = mean_var(lb_sum_r, lb_sq_r)
    lbv_cor = mean_var(lb_sum_c, lb_sq_c)
    cen_r = torch.tensor(cen_r); cen_c = torch.tensor(cen_c)
    var_r = float(cen_r.var(0, unbiased=False).sum())
    var_c = float(cen_c.var(0, unbiased=False).sum())
    detail = torch.tensor(detail)
    rd = torch.tensor(raw_diff); cd = torch.tensor(cor_diff)
    cor_ratio = float(cd.max() / (cd.quantile(0.95) + EPS)) if len(cd) else float("nan")
    raw_ratio = float(rd.max() / (rd.quantile(0.95) + EPS)) if len(rd) else float("nan")

    return dict(
        frames=n, W=W, H=H,
        low_band_var_ratio=lbv_cor / (lbv_raw + EPS),
        low_band_var_raw=lbv_raw, low_band_var_cor=lbv_cor,
        centroid_stability_ratio=var_c / (var_r + EPS),
        centroid_var_raw=var_r, centroid_var_cor=var_c,
        detail_retention_median=float(detail.median()) if n else float("nan"),
        detail_retention_min=float(detail.min()) if n else float("nan"),
        motion_ratio=motion_cor / (motion_raw + EPS),
        jitter_cor_max_over_p95=cor_ratio, jitter_raw_max_over_p95=raw_ratio,
        runtime_s=runtime, per_frame_ms=1000.0 * runtime / max(1, n),
    )


def _compose_sxs(raw, cor, vw, vh, sep, idx):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (2 * vw + sep, vh), (5, 7, 11))
    for frame, x0, label in ((raw, 0, "RAW"), (cor, vw + sep, "CORRECTED")):
        arr = (frame.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8)
        im = Image.fromarray(arr.contiguous().numpy(), "RGB").resize((vw, vh), Image.BILINEAR)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, vw - 1, 14], fill=(5, 7, 11))
        d.text((4, 3), f"{label}  f{idx:04d}", fill=(225, 233, 240))
        canvas.paste(im, (x0, 0))
    return canvas.tobytes()


# --------------------------------------------------------------------------- #
# Sanity: the frozen pass must still remove drift on the synthetic fixture.
# --------------------------------------------------------------------------- #
def sanity_drift(seed: int = 1234, T: int = 48, G: int = 64) -> dict:
    torch.manual_seed(seed)
    clip, _ = make_clip_batch(1, T, G, G, seed=seed, collisions=True, return_meta=True)
    control = clip[0, :T].clone()
    drifted = run_m0.inject_hotspot(control)
    cut = SHIP_CFG["cutoff_frac"]
    eng = SpectralCoherenceEngine(**SHIP_CFG)
    corrected, _ = eng.process_segment(drifted, eng.init_state(G, G))
    pv_d = run_m0.position_variance(run_m0.lowband_energy_positions(drifted, cut))
    pv_c = run_m0.position_variance(run_m0.lowband_energy_positions(corrected, cut))
    drift = run_m0.pct_drop(pv_d, pv_c)
    return dict(drift_pct=drift, passes=drift >= 60.0)


# --------------------------------------------------------------------------- #
def evaluate(m: dict, runtime_budget_s: float) -> dict:
    p = dict(
        detail=m["detail_retention_median"] >= 0.97,
        motion=0.90 <= m["motion_ratio"] <= 1.10,
        jitter=(not np.isnan(m["jitter_cor_max_over_p95"]))
        and m["jitter_cor_max_over_p95"] <= 1.5,
        runtime=m["runtime_s"] <= runtime_budget_s,
    )
    p["all"] = all(p.values())
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--resolutions", default="1080,720",
                    help="comma list of target WIDTHS (portrait); 1080=native, 720=downscale")
    ap.add_argument("--excerpt-sec", type=float, default=8.0)
    ap.add_argument("--runtime-budget-min", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(RENDERS, exist_ok=True)
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    widths = [int(w) for w in args.resolutions.split(",") if w.strip()]
    budget = args.runtime_budget_min * 60.0

    print("=" * 88)
    print("ENG-1  1080p real-footage gate   frozen: complex_mc band=0.03 str=0.5 "
          "alpha=0.95 rho=0.995 cut=0.14")
    print("=" * 88)

    sanity = sanity_drift()
    print(f"[sanity] synthetic drifted fixture: drift removal {sanity['drift_pct']:.1f}%  "
          f"[{'PASS' if sanity['passes'] else 'FAIL'}  >=60%]")

    record = dict(config=SHIP_CFG, sanity=sanity, clips={})
    verdict_fail = [] if sanity["passes"] else ["sanity(drift regressed)"]

    for name in clips:
        path = ensure_local(name)
        w0, h0, fps = ffprobe_dims(path)
        record["clips"][name] = {}
        print(f"\n[{name}] source {w0}x{h0} @ {fps:.2f}fps  -> {path}")
        for width in widths:
            # portrait: keep aspect, target width `width`
            W = width - (width % 2)
            H = int(round(h0 * W / w0)); H -= H % 2
            native = (width == max(widths))
            excerpt = int(round(args.excerpt_sec * fps))
            sxs = os.path.join(RENDERS, f"eng1_{name}_sidebyside.mp4") if native else None
            m = process_clip(path, W, H, fps, SHIP_CFG["cutoff_frac"], excerpt, sxs)
            p = evaluate(m, budget)
            record["clips"][name][str(width)] = dict(metrics=m, passes=p)
            tag = "native" if native else "downscale"
            print(f"  [{width}p {tag}] {W}x{H} {m['frames']}f  "
                  f"detail_med={m['detail_retention_median']:.4f}  "
                  f"motion_ratio={m['motion_ratio']:.3f}  "
                  f"lowband_var_ratio={m['low_band_var_ratio']:.3f}  "
                  f"centroid_stab={m['centroid_stability_ratio']:.3f}")
            print(f"           jitter cor_max/p95={m['jitter_cor_max_over_p95']:.2f} "
                  f"(raw {m['jitter_raw_max_over_p95']:.2f})  "
                  f"runtime={m['runtime_s']:.1f}s ({m['per_frame_ms']:.1f} ms/f)")
            if native:
                fails = [k for k, v in p.items() if k != "all" and not v]
                print(f"           NATIVE GATE: {'PASS' if p['all'] else 'FAIL -> ' + ','.join(fails)}")
                if not p["all"]:
                    verdict_fail.append(f"{name}:{','.join(fails)}")

    overall = "PASS" if not verdict_fail else "FAIL"
    record["verdict"] = dict(status=overall, failures=verdict_fail)

    # cost/quality table (native vs downscale)
    print("\n" + "-" * 88)
    print("cost/quality (per clip x resolution)")
    print(f"{'clip':<14}{'res':>6}{'ms/frame':>10}{'runtime_s':>11}"
          f"{'detail_med':>12}{'motion':>9}{'lbvar':>8}")
    for name in clips:
        for width in widths:
            m = record["clips"][name][str(width)]["metrics"]
            print(f"{name:<14}{width:>5}p{m['per_frame_ms']:>10.1f}{m['runtime_s']:>11.1f}"
                  f"{m['detail_retention_median']:>12.4f}{m['motion_ratio']:>9.3f}"
                  f"{m['low_band_var_ratio']:>8.3f}")
    print("-" * 88)
    print(f"VERDICT: {overall}" + ("" if overall == "PASS" else f"  ({'; '.join(verdict_fail)})"))
    print("=" * 88)

    with open(os.path.join(REPORTS, "ENG1_1080.json"), "w") as f:
        json.dump(record, f, indent=2)
    _write_md(record, clips, widths)
    print(f"[out] reports/ENG1_1080.json  reports/ENG1_1080.md  "
          f"renders/eng1_*_sidebyside.mp4")


def _write_md(record, clips, widths):
    L = ["# ENG-1 — 1080p real-footage gate\n",
         f"Verdict: **{record['verdict']['status']}**"
         + ("" if record['verdict']['status'] == 'PASS'
            else "  (failures: " + "; ".join(record['verdict']['failures']) + ")"),
         "",
         f"Frozen config: `{record['config']}`",
         f"Sanity (synthetic drift removal): {record['sanity']['drift_pct']:.1f}% "
         f"({'PASS' if record['sanity']['passes'] else 'FAIL'}, gate >=60%)",
         "",
         "## Native-1080 gate (per clip)", "",
         "| clip | detail med (>=0.97) | motion ratio [0.90,1.10] | jitter max/p95 (<=1.5) | runtime s (<=1800) | PASS |",
         "| --- | --- | --- | --- | --- | --- |"]
    native_w = str(max(widths))
    for name in clips:
        c = record["clips"][name].get(native_w)
        if not c:
            continue
        m, p = c["metrics"], c["passes"]
        L.append(f"| {name} | {m['detail_retention_median']:.4f} | "
                 f"{m['motion_ratio']:.3f} | {m['jitter_cor_max_over_p95']:.2f} "
                 f"(raw {m['jitter_raw_max_over_p95']:.2f}) | {m['runtime_s']:.1f} | "
                 f"{'PASS' if p['all'] else 'FAIL'} |")
    L += ["", "## Cost / quality (resolution sweep)", "",
          "| clip | res | ms/frame | runtime s | detail med | motion | lowband var ratio | centroid stab |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in clips:
        for width in widths:
            m = record["clips"][name][str(width)]["metrics"]
            L.append(f"| {name} | {width}p | {m['per_frame_ms']:.1f} | {m['runtime_s']:.1f} | "
                     f"{m['detail_retention_median']:.4f} | {m['motion_ratio']:.3f} | "
                     f"{m['low_band_var_ratio']:.3f} | {m['centroid_stability_ratio']:.3f} |")
    L += ["", "## Reading",
          "- detail retention = per-frame HF (luma-highpass) Pearson corr, corrected vs raw; median.",
          "- motion ratio = mean|frame diff| corrected / raw (1.0 = motion untouched; <0.90 = suppressed).",
          "- low-band var ratio = temporal variance of |low-band| corrected/raw (1.0 = non-interference;",
          "  <1 = the pass flattened legitimate low-frequency change).",
          "- centroid stability ratio = low-band centroid path variance corrected/raw.",
          "- jitter = corrected frame-to-frame MAE max / its own p95 (raw shown for context; if raw is",
          "  also high the spike is in the content, not introduced by the pass)."]
    with open(os.path.join(REPORTS, "ENG1_1080.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

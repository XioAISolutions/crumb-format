"""Minute-scale consistency of a generated video (LONGLIVE_4090.md).

Reads an mp4 as a stream (constant RAM at any length), samples ``--per-sec``
frames per second, and measures four ways minutes-long generation is known to
fail:

  drift     semantic similarity of each window to the first window falls off
            (identity / scene drift). Frozen image features: DINOv2-small via
            transformers when available (``--encoder dinov2``), else ``pixel``
            (normalized 32x18 thumbnails, only for tests and smoke runs).
  fade      mean luma leaves the first window's band (fade to black / white)
  flatten   spatial contrast collapses (the "grey soup" end state)
  freeze    motion (mean |frame - previous sampled frame|) drops toward zero
  colour    mean saturation drifts (the slow over-saturation of long rollouts)

Every metric is reported per ``--window`` seconds, relative to the first window,
so a 30 s baseline clip and a 5-minute clip are judged on the same scale:

    python long_eval.py video.mp4 --out report.json [--encoder dinov2]

Pre-registered read (fixed in LONGLIVE_4090.md before any run):
    drift_ratio(w) = sim(window w, window 0) / sim(window 1, window 0)
    PASS  every window: drift_ratio >= 0.9, luma and contrast within +-25 % of
          window 0, motion >= 25 % of window 0, saturation within +-25 %.
    The first failing window's start time is the clip's coherent horizon.
"""
import argparse
import json
import math

import numpy as np

THRESH = dict(drift=0.9, luma=0.25, contrast=0.25, motion=0.25, sat=0.25)


# ---------------------------------------------------------------- encoders
class PixelEncoder:
    name = "pixel"

    def __call__(self, frames):                      # uint8 [n, H, W, 3]
        import torch
        import torch.nn.functional as F
        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        x = F.adaptive_avg_pool2d(x, (18, 32)).flatten(1)
        x = x - x.mean(1, keepdim=True)
        return (x / (x.norm(dim=1, keepdim=True) + 1e-8)).numpy()


class Dinov2Encoder:
    name = "dinov2"

    def __init__(self, model_id="facebook/dinov2-small", device=None):
        import torch
        from transformers import AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, frames):
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
            out = self.model(pixel_values=(x - self.mean) / self.std)
            f = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0]
            f = F.normalize(f.float(), dim=1)
        return f.cpu().numpy()


def make_encoder(name):
    if name == "pixel":
        return PixelEncoder()
    if name == "dinov2":
        return Dinov2Encoder()
    if name == "auto":
        try:
            return Dinov2Encoder()
        except Exception as e:                           # no transformers / no weights
            print(f"[long_eval] dinov2 unavailable ({type(e).__name__}: {e}); using pixel features")
            return PixelEncoder()
    raise ValueError(name)


# ---------------------------------------------------------------- per-frame stats
def frame_stats(frames):
    """luma mean, spatial luma std, mean HSV-style saturation for uint8 [n,H,W,3]."""
    f = frames.astype(np.float32) / 255.0
    luma = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)          # [n, H, W]
    mx, mn = f.max(-1), f.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return luma.mean((1, 2)), luma.std((1, 2)), sat.mean((1, 2))


def sample_video(path, per_sec=2.0, max_side=448):
    """Yield (t_seconds, uint8 frame) sampled ``per_sec`` times a second, frames
    downscaled so the long side is <= max_side (stats and encoders are scale-robust)."""
    import imageio.v2 as imageio
    r = imageio.get_reader(str(path))
    try:
        fps = float(r.get_meta_data().get("fps", 24.0))
        step = max(1, int(round(fps / per_sec)))
        for i, fr in enumerate(r):
            if i % step:
                continue
            h, w = fr.shape[:2]
            s = max(1, int(math.ceil(max(h, w) / max_side)))
            yield i / fps, np.asarray(fr[::s, ::s, :3])
    finally:
        r.close()


# ---------------------------------------------------------------- evaluation
def evaluate(samples, encoder, window=30.0, batch=32):
    """samples: iterable of (t, frame). Returns the report dict."""
    feats, ts, luma, con, sat, motion = [], [], [], [], [], []
    buf, buf_t = [], []
    prev = None

    def flush():
        nonlocal prev
        if not buf:
            return
        arr = np.stack(buf)
        feats.append(encoder(arr))
        l, c, s = frame_stats(arr)
        luma.extend(l.tolist()); con.extend(c.tolist()); sat.extend(s.tolist())
        for fr in arr:
            f = fr.astype(np.float32) / 255.0
            motion.append(float("nan") if prev is None or prev.shape != f.shape
                          else float(np.abs(f - prev).mean()))
            prev = f
        ts.extend(buf_t)
        buf.clear(); buf_t.clear()

    for t, fr in samples:
        buf.append(fr); buf_t.append(t)
        if len(buf) >= batch:
            flush()
    flush()
    if not ts:
        raise ValueError("no frames")
    F = np.concatenate(feats)
    ts = np.asarray(ts)
    win = np.floor(ts / window).astype(int)
    n_w = int(win.max()) + 1
    cent = []
    for w in range(n_w):
        m = F[win == w].mean(0)
        cent.append(m / (np.linalg.norm(m) + 1e-8))
    ref = F[win == 0]

    def sim_to_ref(w):                                # mean cosine of window w frames to window 0 frames
        return float((F[win == w] @ ref.T).mean())

    def wmean(x, w):
        v = np.asarray(x)[win == w]
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    base = dict(luma=wmean(luma, 0), contrast=wmean(con, 0), motion=wmean(motion, 0),
                sat=wmean(sat, 0), self_sim=sim_to_ref(0))
    s1 = sim_to_ref(1) if n_w > 1 else base["self_sim"]
    rows, horizon = [], None
    for w in range(n_w):
        r = dict(start_s=w * window, sim_to_first=sim_to_ref(w),
                 drift_ratio=sim_to_ref(w) / s1 if s1 > 0 else float("nan"),
                 sim_to_prev=float(cent[w] @ cent[w - 1]) if w else 1.0,
                 luma=wmean(luma, w), contrast=wmean(con, w), motion=wmean(motion, w),
                 sat=wmean(sat, w))
        fails = []
        if w >= 1 and r["drift_ratio"] < THRESH["drift"]:
            fails.append("drift")
        for k, key in (("luma", "luma"), ("contrast", "contrast"), ("sat", "sat")):
            if base[k] > 1e-6 and abs(r[k] / base[k] - 1) > THRESH[key]:
                fails.append({"luma": "fade", "contrast": "flatten", "sat": "colour"}[k])
        if base["motion"] > 1e-6 and not math.isnan(r["motion"]) and r["motion"] < THRESH["motion"] * base["motion"]:
            fails.append("freeze")
        r["fails"] = fails
        if fails and horizon is None:
            horizon = w * window
        rows.append(r)
    dur = float(ts.max())
    return dict(encoder=encoder.name, window_s=window, duration_s=dur, n_samples=int(len(ts)),
                baseline=base, thresholds=THRESH, windows=rows,
                coherent_horizon_s=dur if horizon is None else horizon,
                verdict="PASS" if horizon is None else f"FAIL at {horizon:.0f}s ({','.join(rows[int(horizon // window)]['fails'])})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--encoder", choices=["auto", "dinov2", "pixel"], default="auto")
    ap.add_argument("--per-sec", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=30.0)
    a = ap.parse_args(argv)
    rep = evaluate(sample_video(a.video, a.per_sec), make_encoder(a.encoder), a.window)
    rep["video"] = str(a.video)
    for r in rep["windows"]:
        print(f"{r['start_s']:6.0f}s  sim {r['sim_to_first']:.3f}  drift {r['drift_ratio']:.3f}  "
              f"luma {r['luma']:.3f}  contrast {r['contrast']:.3f}  motion {r['motion']:.4f}  "
              f"sat {r['sat']:.3f}  {' '.join(r['fails'])}")
    print(f"[long_eval] {rep['verdict']}; coherent horizon {rep['coherent_horizon_s']:.0f}s "
          f"of {rep['duration_s']:.0f}s ({rep['encoder']})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=2)
    return rep


if __name__ == "__main__":
    main()

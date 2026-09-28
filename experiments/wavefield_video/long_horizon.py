"""Long-horizon tooling for the 2-5 minute target (LONG_HORIZON.md).

Four things the 1,024-frame stream test did not answer:

  budget   -- what 2-5 minutes of video actually costs in latent steps, tokens,
              recurrent-state bytes vs a full-context attention KV cache, for
              real video VAEs (not the 16x16 toy grid).
  memory   -- how long each wave pole *remembers*: half-life in frames/seconds
              and retention at 2 and 5 minutes. "state = fade x state + frame"
              only carries the past as long as fade^t is not ~0.
  stream   -- a resumable StreamSession: generate any number of frames in chunks
              at constant memory, save/load the whole recurrent state (so a
              5-minute render can be checkpointed, resumed or branched), and a
              HealthMonitor that flags fade-to-black / flatten / freeze / blow-up
              -- the collapse modes this campaign has already hit.

Nothing here claims quality. It measures cost, memory reach and collapse, so
the next GPU runs are judged on the axes that decide whether minutes work.

    python long_horizon.py budget --seconds 120 300
    python long_horizon.py memory --pole-param softplus halflife
    python long_horizon.py stream --frames 7200 --chunk 600 --pole-param halflife
"""
import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import torch

from wfvideo import VideoPredictor, WaveMix3D

# ----------------------------------------------------------------------------- budget
# Public VAE compression factors (spatial stride, temporal stride, latent channels,
# DiT patch). "none" is the pixel-grid toy the campaign trains on today.
VAES = {
    "none": dict(s=1, t=1, c=3, patch=1),
    "wan21": dict(s=8, t=4, c=16, patch=2),     # Wan2.1 VAE 4x8x8, DiT patch 1x2x2
    "ltx": dict(s=32, t=8, c=128, patch=1),     # LTX-Video VAE 8x32x32, patch 1
}


def frame_budget(seconds, fps=24, height=480, width=848, vae="ltx", dim=512,
                 layers=12, n_modes=3, linear_pad=True, kv_bytes=2):
    """Cost of one ``seconds``-long clip. Returns a dict of plain numbers.

    wave_state_bytes: the persistent recurrent state -- per layer
        n_modes x dim x Hp x Wp complex64 -- independent of ``seconds``.
    kv_cache_bytes: what a causal transformer must keep to attend to the
        whole clip (2 x layers x tokens x dim x kv_bytes). Sliding-window
        attention caps this at the window, which is how shipping long-video
        systems actually bound memory -- so this is the *full-context* bound.
    attn_pairs / wave_ops_rel: N^2 vs N log2 N mixing work for the whole clip
        (relative, not wall-clock: FlashAttention changes constants, not order).
    """
    v = VAES[vae]
    frames = int(round(seconds * fps))
    lat_t = 1 + (frames - 1) // v["t"] if v["t"] > 1 else frames
    lat_h = math.ceil(height / v["s"] / v["patch"])
    lat_w = math.ceil(width / v["s"] / v["patch"])
    per_step = lat_h * lat_w
    tokens = lat_t * per_step
    hp, wp = (2 * lat_h, 2 * lat_w) if linear_pad else (lat_h, lat_w)
    state = layers * n_modes * dim * hp * wp * 8
    kv = 2 * layers * tokens * dim * kv_bytes
    return {
        "seconds": seconds, "fps": fps, "frames": frames, "vae": vae,
        "latent_steps": lat_t, "latent_hw": [lat_h, lat_w], "tokens_per_step": per_step,
        "tokens": tokens, "wave_state_bytes": state, "kv_cache_bytes": kv,
        "attn_pairs": tokens * tokens,
        "wave_ops_rel": round(tokens * math.log2(max(tokens, 2))),
        "attn_over_wave": round(tokens / math.log2(max(tokens, 2)), 1),
    }


# ----------------------------------------------------------------------------- memory
def retention(half_life, frames):
    """Fraction of a ripple's amplitude left after ``frames``: 0.5 ** (t / hl)."""
    return 0.5 ** (frames / half_life)


def memory_report(model, fps=24, horizons_s=(1, 10, 120, 300), floor=0.01):
    """Per-layer DC half-lives of every wave pole in ``model`` and how many modes
    still hold >= ``floor`` of their amplitude at each horizon.

    A mode that keeps < 1% after 2 minutes cannot, on its own, make frame 2880
    depend on frame 0 -- so ``reach_s`` (longest half-life x log2(1/floor))
    is the recurrent state's memory reach. Returns a JSON-able dict."""
    layers = []
    for i, mod in enumerate(m for m in model.modules() if isinstance(m, WaveMix3D)):
        hl = mod.half_lives()
        if hl is None:
            continue
        hl = hl.detach().flatten().double()
        row = {"layer": i, "pole_param": mod.pole_param, "n_poles": hl.numel(),
               "half_life_frames": {"min": round(hl.min().item(), 3),
                                    "median": round(hl.median().item(), 3),
                                    "max": round(hl.max().item(), 3)},
               "reach_s": round(hl.max().item() * math.log2(1 / floor) / fps, 3),
               "modes_alive": {}}
        for h in horizons_s:
            alive = (retention(hl, h * fps) >= floor).sum().item()
            row["modes_alive"][f"{h}s"] = f"{alive}/{hl.numel()}"
        layers.append(row)
    return {"fps": fps, "floor": floor, "layers": layers}


# ----------------------------------------------------------------------------- health
@dataclass
class HealthMonitor:
    """Flags the collapse modes long rollouts hit, each only after it has held
    for ``patience`` consecutive frames (thresholds fixed here, before any run):

      fade      mean intensity < fade_ratio x context mean   (latent 'fade to black')
      flatten   spatial std   < flat_ratio x context std     (contrast collapse)
      freeze    mean |f_t - f_(t-1)| < freeze_ratio x context motion (copy-last collapse)
      nonfinite any NaN/Inf                                   (immediate)

    Ratios are measured against each sample's OWN context clip, and streaks are
    kept per batch sample, so one collapsed rollout cannot hide behind a healthy
    one in the batch average. ``first[k]`` is the earliest frame any sample
    collapsed; ``first_sample[k]`` says which. ``freeze`` is skipped for a sample
    whose context is itself static."""
    fade_ratio: float = 0.5
    flat_ratio: float = 0.5
    freeze_ratio: float = 0.1
    patience: int = 24
    ref: dict = field(default_factory=dict)      # per-sample [B] tensors
    runs: dict = field(default_factory=dict)     # per-sample [B] streak counters
    first: dict = field(default_factory=dict)
    first_sample: dict = field(default_factory=dict)
    frame_index: int = 0
    _prev: torch.Tensor = None

    @staticmethod
    def _stats(f):                               # f: [B,C,H,W] -> per-sample [B]
        return f.flatten(1).mean(1), f.flatten(2).std(-1).mean(1)

    def calibrate(self, context):
        """context: [B,T,C,H,W] frames the rollout starts from."""
        c = context.float()
        B = c.shape[0]
        m, sd = self._stats(c.flatten(0, 1))
        motion = ((c[:, 1:] - c[:, :-1]).abs().flatten(1).mean(1) if c.shape[1] > 1
                  else torch.zeros(B))
        self.ref = {"mean": m.view(B, -1).mean(1), "std": sd.view(B, -1).mean(1),
                    "motion": motion}
        self._prev = c[:, -1]
        return self

    def update(self, frame):
        """frame: [B,C,H,W]. Returns batch-mean stats for logging; flags are per sample."""
        f = frame.float()
        mean, std = self._stats(f)
        motion = ((f - self._prev).abs().flatten(1).mean(1) if self._prev is not None
                  else torch.zeros(f.shape[0]))
        self._prev = f
        conds = {
            "fade": mean < self.fade_ratio * self.ref["mean"],
            "flatten": std < self.flat_ratio * self.ref["std"],
            "freeze": (self.ref["motion"] > 1e-6) & (motion < self.freeze_ratio * self.ref["motion"]),
        }
        for k, bad in conds.items():
            run = self.runs.get(k, torch.zeros_like(bad, dtype=torch.long))
            run = torch.where(bad, run + 1, torch.zeros_like(run))
            self.runs[k] = run
            hit = (run >= self.patience).nonzero().flatten()
            if hit.numel() and k not in self.first:
                self.first[k] = self.frame_index - self.patience + 1
                self.first_sample[k] = int(hit[0])
        bad = ~torch.isfinite(f.flatten(1)).all(1)
        if bad.any() and "nonfinite" not in self.first:
            self.first["nonfinite"] = self.frame_index
            self.first_sample["nonfinite"] = int(bad.nonzero()[0])
        self.frame_index += 1
        return {"mean": mean.mean().item(), "std": std.mean().item(), "motion": motion.mean().item()}

    @property
    def healthy(self):
        return not self.first

    def state_dict(self):
        """Everything needed to continue monitoring after a StreamSession resume:
        a streak that spans the checkpoint keeps counting and earlier flags stay."""
        return {"ref": dict(self.ref), "runs": dict(self.runs), "first": dict(self.first),
                "first_sample": dict(self.first_sample), "frame_index": self.frame_index,
                "prev": None if self._prev is None else self._prev.detach().clone()}

    def load_state_dict(self, d):
        self.ref, self.runs, self.first = dict(d["ref"]), dict(d["runs"]), dict(d["first"])
        self.first_sample = dict(d.get("first_sample", {}))
        self.frame_index, self._prev = d["frame_index"], d["prev"]
        return self


# ----------------------------------------------------------------------------- stream
def _clone_state(s):
    if isinstance(s, dict):
        return {k: _clone_state(v) for k, v in s.items()}
    if isinstance(s, (list, tuple)):
        return type(s)(_clone_state(v) for v in s)
    return s.detach().clone() if torch.is_tensor(s) else s


def state_nbytes(s):
    if isinstance(s, dict):
        return sum(state_nbytes(v) for v in s.values())
    if isinstance(s, (list, tuple)):
        return sum(state_nbytes(v) for v in s)
    return s.element_size() * s.nelement() if torch.is_tensor(s) else 0


def model_fingerprint(model):
    """Short hash of parameter names+shapes+values: a saved state only resumes
    on the exact weights that produced it."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


class StreamSession:
    """Chunked, resumable O(1)-memory rollout of a streamable VideoPredictor.

    session = StreamSession(model).warm(context)     # context [B,T,3,H,W]
    for chunk in session.generate(7200, chunk=600):  # [B,chunk,3,H,W] on CPU
        ...
    session.save("t=3600.pt"); StreamSession(model).load("t=3600.pt")  # resume/branch

    The saved payload is the recurrent state + last frame + absolute frame index;
    its size does not grow with how far the stream has run (for pure wave/ssm
    arms; local+global hybrids add their fixed T-frame window buffer)."""

    def __init__(self, model, clamp=(0.0, 1.0)):
        assert model._streamable(), "StreamSession needs a recurrent arm (wave dispersion/ssm/fused)"
        self.model = model.eval()
        self.clamp = clamp
        self.states = None
        self.frame = None
        self.t = 0
        self.extra = None

    @torch.no_grad()
    def warm(self, context):
        B, T = context.shape[:2]
        dev = next(self.model.parameters()).device
        self.states = self.model.stream_init(B, dev)
        self.t = 0
        for t in range(T):
            self.frame, self.states = self.model.stream_step(context[:, t].to(dev), self.states, self.t)
            self.frame = self._post(self.frame)
            self.t += 1
        return self

    def _post(self, f):
        f = f.float()
        return f.clamp(*self.clamp) if self.clamp is not None else f

    @torch.no_grad()
    def generate(self, n_frames, chunk=256):
        """Yield [B, <=chunk, C, H, W] CPU tensors until ``n_frames`` are produced.
        Each yielded frame is the model's prediction fed back as the next input."""
        assert self.states is not None, "call warm() or load() first"
        done = 0
        while done < n_frames:
            k = min(chunk, n_frames - done)
            out = []
            for _ in range(k):
                out.append(self.frame.cpu())
                self.frame, self.states = self.model.stream_step(self.frame, self.states, self.t)
                self.frame = self._post(self.frame)
                self.t += 1
            done += k
            yield torch.stack(out, 1)

    def state_bytes(self):
        return state_nbytes(self.states) + state_nbytes(self.frame)

    def save(self, path, extra=None):
        """``extra``: optional dict saved alongside (e.g. a HealthMonitor state_dict)."""
        torch.save({"states": _clone_state(self.states), "frame": self.frame.detach().clone(),
                    "t": self.t, "fingerprint": model_fingerprint(self.model),
                    "extra": extra}, path)

    def load(self, path):
        """Restore a saved session; returns the ``extra`` dict saved with it (or None)."""
        dev = next(self.model.parameters()).device
        ck = torch.load(path, map_location=dev, weights_only=True)
        if ck["fingerprint"] != model_fingerprint(self.model):
            raise ValueError("saved stream state was produced by different weights")
        self.states, self.frame, self.t = ck["states"], ck["frame"], ck["t"]
        self.extra = ck.get("extra")
        return self


# ----------------------------------------------------------------------------- CLI
def _fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def build_model(a, pole_param):
    return VideoPredictor(a.dim, a.layers, a.heads, a.frames, a.grid, a.grid, "wave",
                          causal=True, ffn_mult=a.ffn_mult, kernel_version="dispersion",
                          linear_pad=True, fuse=getattr(a, "fuse", "none"),
                          pole_param=pole_param, hl_min=a.hl_min, hl_max=a.hl_max)


def cmd_budget(a):
    rows = [frame_budget(s, a.fps, a.height, a.width, v, a.dim, a.layers, a.n_modes)
            for s in a.seconds for v in a.vae]
    print(f"{'clip':>6} {'vae':>6} {'lat steps':>9} {'tok/step':>8} {'tokens':>10} "
          f"{'wave state':>11} {'full KV':>10} {'N/log2N':>9}")
    for r in rows:
        print(f"{r['seconds']:>5}s {r['vae']:>6} {r['latent_steps']:>9} {r['tokens_per_step']:>8} "
              f"{r['tokens']:>10} {_fmt_bytes(r['wave_state_bytes']):>11} "
              f"{_fmt_bytes(r['kv_cache_bytes']):>10} {r['attn_over_wave']:>8}x")
    return rows


def cmd_memory(a):
    out = {}
    for pp in a.pole_param:
        torch.manual_seed(a.seed)
        rep = memory_report(build_model(a, pp), fps=a.fps)
        out[pp] = rep
        L0 = rep["layers"][0]
        print(f"{pp:>9}: half-life frames min/med/max = {L0['half_life_frames']['min']}/"
              f"{L0['half_life_frames']['median']}/{L0['half_life_frames']['max']}  "
              f"reach={L0['reach_s']}s  alive(>=1%) {L0['modes_alive']}")
    return out


def cmd_stream(a):
    from data import make_clip_batch
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    m = build_model(a, a.pole_param[0]).to(dev)
    if a.ckpt:
        m.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True)["state"])
    # Context stays on CPU for the monitor; warm() moves it to the model's device.
    ctx = make_clip_batch(a.batch, a.frames, a.grid, a.grid, seed=70000)[:, :a.frames]
    mon = HealthMonitor(patience=a.patience).calibrate(ctx)
    sess = StreamSession(m)
    if a.resume:
        sess.load(a.resume)
        if sess.extra and "health" in sess.extra:
            mon.load_state_dict(sess.extra["health"])
        else:                       # older state file: flags start fresh, indices stay absolute
            mon.frame_index = sess.t
    else:
        sess.warm(ctx)
        mon.frame_index = sess.t    # flag frames use the same absolute index as the log
    log, t0, t_start = [], time.time(), sess.t
    sizes = set()
    for chunk in sess.generate(a.stream_frames, chunk=a.chunk):
        for i in range(chunk.shape[1]):
            st = mon.update(chunk[:, i])
        sizes.add(sess.state_bytes())
        row = {"frame": sess.t,       # fps: frames generated by THIS invocation only
               "fps": round((sess.t - t_start) / max(time.time() - t0, 1e-9), 1),
               "state_bytes": sess.state_bytes(), "last": {k: round(v, 5) for k, v in st.items()},
               "flags": dict(mon.first)}
        log.append(row)
        print(f"STREAM t={row['frame']:6d} state={_fmt_bytes(row['state_bytes'])} "
              f"fps={row['fps']} mean={st['mean']:.4f} std={st['std']:.4f} "
              f"motion={st['motion']:.5f} flags={row['flags']}", flush=True)
    if a.save_state:
        sess.save(a.save_state, extra={"health": mon.state_dict()})
    res = {"mode": "long_horizon_stream", "pole_param": a.pole_param[0], "trained": bool(a.ckpt),
           "frames": a.stream_frames, "grid": a.grid,
           "context_ref": {k: v.tolist() for k, v in mon.ref.items()},
           "collapse_sample": dict(mon.first_sample),
           "state_bytes_constant": len(sizes) == 1, "collapse": dict(mon.first), "log": log}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--seconds", type=float, nargs="+", default=[120, 300])
    b.add_argument("--fps", type=int, default=24)
    b.add_argument("--height", type=int, default=480)
    b.add_argument("--width", type=int, default=848)
    b.add_argument("--vae", nargs="+", choices=sorted(VAES), default=["wan21", "ltx"])
    for p in (sub.add_parser("memory"), sub.add_parser("stream")):
        p.add_argument("--pole-param", nargs="+", choices=["softplus", "halflife"],
                       default=["softplus", "halflife"])
        p.add_argument("--hl-min", type=float, default=2.0)
        p.add_argument("--hl-max", type=float, default=4096.0)
        p.add_argument("--fps", type=int, default=24)
        p.add_argument("--grid", type=int, default=16)
        p.add_argument("--frames", type=int, default=16, help="context window T")
        p.add_argument("--heads", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--ffn-mult", type=float, default=4.0,
                       help="match the checkpoint (train_compare's result JSON 'ffn_mult')")
    for p in sub.choices.values():
        p.add_argument("--dim", type=int, default=128)
        p.add_argument("--layers", type=int, default=4)
    # Only the budget honors --n-modes: VideoPredictor builds WaveMix3D with its
    # default 3 modes, so memory/stream must not pretend to take it.
    b.add_argument("--n-modes", type=int, default=3)
    s = sub.choices["stream"]
    s.add_argument("--stream-frames", type=int, default=7200, help="7200 = 5 min at 24 fps")
    s.add_argument("--chunk", type=int, default=600)
    s.add_argument("--batch", type=int, default=2)
    s.add_argument("--patience", type=int, default=24)
    s.add_argument("--ckpt", default="", help="train_compare.py model_*.pt (else untrained)")
    s.add_argument("--save-state", default="")
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="where the stream runs (default: cuda when available)")
    s.add_argument("--fuse", choices=["none", "local_wave"], default="none",
                   help="local_wave for checkpoints of the local+wave hybrid (E_* arms)")
    s.add_argument("--resume", default="")
    s.add_argument("--out", default="")
    a = ap.parse_args(argv)
    return {"budget": cmd_budget, "memory": cmd_memory, "stream": cmd_stream}[a.cmd](a)


if __name__ == "__main__":
    main()

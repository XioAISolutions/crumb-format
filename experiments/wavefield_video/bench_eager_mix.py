#!/usr/bin/env python3
"""Eager vs torch.compile micro-bench for the wave path (eager-mix question)."
forward + streaming rollout, eager vs compiled. Writes bench_mix_<dev>.json."""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from wfvideo import VideoPredictor

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def timeit(fn, iters=6, warm=2):
    for _ in range(warm): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync()
    return round((time.perf_counter() - t0) / iters * 1e3, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    m = VideoPredictor(a.dim, a.layers, a.heads, a.frames, a.grid, a.grid, "wave",
                       causal=True, kernel_version="dispersion", linear_pad=True,
                       gate=True, local_fuse=True).to(dev).eval()
    x = torch.randn(a.batch, a.frames, 3, a.grid, a.grid, device=dev)
    out = {"device": dev, "params": sum(p.numel() for p in m.parameters())}
    with torch.no_grad():
        out["fwd_eager_ms"] = timeit(lambda: m(x), a.iters)
        mc = torch.compile(m)
        try:
            out["fwd_compiled_ms"] = timeit(lambda: mc(x), a.iters)
        except Exception as e:
            out["fwd_compiled_ms"] = "FAIL: " + str(e)[:180]
        def roll(mod):
            s = mod.stream_init(a.batch, dev)
            f = x[:, -1]
            for i in range(a.steps):
                f, s = mod.stream_step(f, s, a.frames + i)
            return f
        out["stream_eager_ms_per_step"] = round(timeit(lambda: roll(m), 3) / a.steps, 3)
        try:
            out["stream_compiled_ms_per_step"] = round(timeit(lambda: roll(mc), 3) / a.steps, 3)
        except Exception as e:
            out["stream_compiled_ms_per_step"] = "FAIL: " + str(e)[:180]
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bench_mix_{dev}.json")
    open(p, "w").write(json.dumps(out, indent=1))
    print(json.dumps(out))

if __name__ == "__main__":
    main()

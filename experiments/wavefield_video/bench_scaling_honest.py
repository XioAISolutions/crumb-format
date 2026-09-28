"""Re-measure the wave-vs-attention mixing cost (the "417x" in OCEAN.md) fairly.

kernels2d_smoke.py timed ONE un-warmed call of QK^T then (QK^T)V with no softmax
against one FFT convolution. This keeps its shapes ([1, 64ch, T, S], one head,
fp32) but adds warm-up, median-of-reps timing, and three attention variants:

  orig     kernels2d_smoke's exact op (no softmax), for reproduction
  naive    softmax(QK^T/sqrt(d)) V, materialized N x N
  sdpa     torch.nn.functional.scaled_dot_product_attention (fused / memory-
           efficient kernel -- what a real model would call)

It is still ONE mixing op of one 64-channel head, not a model step: a video
model does this per head per layer per denoising step, and on a GPU the
constants differ a lot (see LONG_HORIZON.md).

    python bench_scaling_honest.py --out runs_long/bench_scaling_cpu.json
"""
import argparse
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from kernels2d_smoke import fft_conv2d, wave_kernel_2d


def _time(fn, reps, warmup, sync):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000.0


def bench(shapes, C=64, reps=5, warmup=2, device="cpu", attn_max_n=16384):
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    rows = []
    for T, S in shapes:
        N = T * S
        x = torch.randn(1, C, T, S, device=device)
        k = wave_kernel_2d(0.02, 0.02, 0.5, 0.5, 0.0, 0.0, T, S).to(device)
        q = x.reshape(1, C, N).transpose(1, 2)                         # [1,N,C]
        row = {"T": T, "S": S, "N": N,
               "wave_ms": round(_time(lambda: fft_conv2d(x, k), reps, warmup, sync), 3)}
        if N <= attn_max_n:
            row["orig_ms"] = round(_time(
                lambda: torch.matmul(torch.matmul(q, q.transpose(-1, -2)) / C ** 0.5, q),
                reps, warmup, sync), 3)
            row["naive_ms"] = round(_time(
                lambda: torch.softmax(torch.matmul(q, q.transpose(-1, -2)) / C ** 0.5, -1) @ q,
                reps, warmup, sync), 3)
        q4 = q[:, None]                                                # [1,1,N,C]
        if device == "cpu" and N > attn_max_n:   # CPU SDPA falls back to an N x N matrix
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        row["sdpa_ms"] = round(_time(lambda: F.scaled_dot_product_attention(q4, q4, q4),
                                     reps, warmup, sync), 3)
        for kname in ("orig", "naive", "sdpa"):
            if f"{kname}_ms" in row:
                row[f"{kname}_over_wave"] = round(row[f"{kname}_ms"] / row["wave_ms"], 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--attn-max-n", type=int, default=16384,
                    help="skip the materialized N x N variants above this N (RAM)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    shapes = [(16, 64), (32, 128), (64, 256), (128, 512)]              # N = 1k, 4k, 16k, 64k
    rows = bench(shapes, reps=a.reps, warmup=a.warmup, device=a.device, attn_max_n=a.attn_max_n)
    res = {"device": a.device, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "machine": platform.processor() or platform.machine(), "channels": 64,
           "reps": a.reps, "warmup": a.warmup, "rows": rows}
    if a.device == "cuda":
        res["gpu"] = torch.cuda.get_device_name()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    main()

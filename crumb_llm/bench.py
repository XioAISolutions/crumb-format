"""Wall-time + memory benchmarks: wave-field vs. transformer baseline.

Sweeps sequence length and reports forward-pass time and peak-memory
delta for both architectures, mirroring the headline numbers from the
Badaramoni paper:

    32K tokens:  wave 6.7 GB  vs  attention 35 GB
    128K tokens: wave 26 GB   vs  attention OOM

Our defaults are much smaller (CPU-friendly), but the same trend should
be visible: attention scales O(N²) in time, wave-field scales O(N log N)
once you saturate the field-size FFT cost.

Usage:
    python -m crumb_llm.bench --lens 256,1024,4096
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch

from .baseline import TinyTransformerLM, TransformerConfig
from .model import WaveFieldLM, WaveFieldConfig


@dataclass
class BenchRow:
    arch: str
    seq_len: int
    time_ms: float
    peak_mem_mb: float
    ok: bool
    note: str = ""


def _peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    # CPU fallback: not meaningful, return 0.
    return 0.0


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _bench_one(model, x, warmup: int = 1, runs: int = 3) -> tuple[float, float]:
    device = next(model.parameters()).device
    x = x.to(device)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _reset_peak()
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / runs
    return dt * 1000.0, _peak_mem_mb()


def benchmark(
    lens: list[int],
    dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    field_size: int = 4096,
    batch_size: int = 1,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wave-field has a fixed field size; its forward is O(F log F) per layer
    # plus N×D projections. Build once.
    wave = WaveFieldLM(WaveFieldConfig(
        vocab_size=256, dim=dim, n_layers=n_layers, n_heads=n_heads,
        field_size=field_size, causal=True,
    )).to(device)

    for N in lens:
        x = torch.randint(0, 256, (batch_size, N), device="cpu")
        # Wave field.
        try:
            t_ms, mem = _bench_one(wave, x)
            rows.append(BenchRow("wave_field", N, t_ms, mem, ok=True))
        except (RuntimeError, MemoryError) as e:
            rows.append(BenchRow("wave_field", N, float("nan"), 0.0, ok=False, note=str(e)[:120]))

        # Transformer baseline — block_size = N so the position embedding fits.
        try:
            tx = TinyTransformerLM(TransformerConfig(
                vocab_size=256, dim=dim, n_layers=n_layers, n_heads=n_heads,
                block_size=N,
            )).to(device)
            t_ms, mem = _bench_one(tx, x)
            rows.append(BenchRow("transformer", N, t_ms, mem, ok=True))
            del tx
        except (RuntimeError, MemoryError) as e:
            rows.append(BenchRow("transformer", N, float("nan"), 0.0, ok=False, note=str(e)[:120]))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def format_rows(rows: list[BenchRow]) -> str:
    out = [f"{'arch':<13} {'seq_len':>8} {'time_ms':>10} {'peak_MB':>10}  ok  note"]
    for r in rows:
        out.append(
            f"{r.arch:<13} {r.seq_len:>8d} {r.time_ms:>10.2f} {r.peak_mem_mb:>10.1f}  "
            f"{'Y' if r.ok else 'N'}  {r.note}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark wave-field vs transformer forward pass.")
    ap.add_argument("--lens", default="256,1024,4096", help="Comma-separated sequence lengths.")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--field-size", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args(argv)
    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    rows = benchmark(
        lens=lens, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
        field_size=args.field_size, batch_size=args.batch,
    )
    print(format_rows(rows))


if __name__ == "__main__":
    main()

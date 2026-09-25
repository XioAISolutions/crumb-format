# R6 — Triton fused-kernel prototypes

**Scope:** new files only — `triton_fused.py`, `bench_triton.py`, this note. No
existing module touched, no commits. Author host is CPU-only; **CUDA runs are
done by the operator** (commands at the bottom).

## Why

`torch.compile` cannot codegen the dispersion mixer: the hot path runs through
`torch.fft.*` on `cfloat` intermediates and Inductor bails (see
`bench_eager_mix.py` → `fwd_compiled_ms: "FAIL..."`). The two elementwise ops
around the FFTs are, however, trivially fusible. Triton has **no complex
dtype**, so both kernels take real/imag as _separate packed float32 tensors_
and do the complex algebra by hand.

## Kernels & signatures (`triton_fused.py`)

### kernelA — `fused_complex_mul(xr, xi, kr, ki, gate=None, block=1024)`

Elementwise `out = gate * (X * K)`, one pass:

```
out_re = gate * (xr*kr - xi*ki)
out_im = gate * (xr*ki + xi*kr)
```

- Maps to the `X * K` (spectrum × transfer) step in
  `WaveMix3D._wave_dispersion` / `_wave_separable`; `gate` folds a real
  content-gate scale into the same pass (eager needs mul + a separate scale
  launch + a temporary).
- `xr,xi,kr,ki`: float32 **CUDA** tensors, broadcastable. `gate`: optional
  float32 real scale (broadcastable) or `None` (compile-time `HAS_GATE`
  constexpr — no gate load emitted).
- Returns `(out_re, out_im)` at the broadcast shape.

### kernelB — `fused_step_inner(zr, zi, lr, li, b, xr, xi, block=1024)`

One frame of the streaming recurrence `state = lam*state + b*x`, one pass:

```
z_re = lr*zr - li*zi + b*xr
z_im = lr*zi + li*zr + b*xi
```

- Maps to the per-cell complex FMA in `WaveMix3D.step()`
  (`state = lam_b*state + Bg*x_hat`); `lam` complex, `b` (= `Bg`) **real**.
- All args float32 **CUDA**, broadcastable (so `lam [M,H,W]` and `b [M,1,1]`
  broadcast over batch/channel). Returns new `(state_re, state_im)`.

### Eager references (device-agnostic, ground truth)

`eager_complex_mul(xr, xi, kr, ki, gate=None)` and
`eager_step_inner(zr, zi, lr, li, b, xr, xi)` — same math in plain torch, used
by the self-check and the bench.

### Self-check — `self_check(seed=0)`

Triton-vs-eager, `max|err| < 1e-4`, covers kernelA (gate on/off, broadcast K)
and kernelB (broadcast lam/b). **CUDA-only; on CPU it prints a SKIP line and
returns True** (clean exit 0). `python triton_fused.py` runs it as `__main__`.

## Limits / caveats

- **float32 only**, **CUDA only** — the wrappers raise on non-CUDA / non-f32
  inputs; the `__main__` / bench paths skip cleanly instead.
- Triton needs **planar** re/im; a torch `cfloat` tensor's `.real`/`.imag` are
  interleaved (stride-2) views. Callers must `.contiguous()` (deinterleave)
  before/around the kernel — the bench does this **outside** the timed region so
  the numbers are kernel-vs-kernel, not deinterleave overhead.
- Wrappers `torch.broadcast_tensors(...).contiguous()` — convenient but
  **materializes** broadcast operands. For the fair micro-bench the bench
  pre-expands K/gate/lam/b to full shape first (so the in-loop broadcast is a
  no-op). The 1b full-chain bench leaves K broadcast (realistic; FFTs dominate).
- `block=1024` is an untuned default; a real deployment should autotune
  `BLOCK` / `num_warps` per shape.
- Prototype does **not** wire into `wfvideo.py` — it validates the kernels in
  isolation. Integration would replace the `X*K`/gate and `step` state-update
  lines with calls to these wrappers on `.real`/`.imag` views.

## bench_triton.py

CUDA micro-bench over **grid 16 & 32, heads 8** (dispersion hot shapes,
`dim=64 → dh=8`, `T=8 → L=16`, `Hp=Wp=2·grid`, `n_modes=3`). Per config:

- **cplxmul**: `cplxmul_eager_ms` vs `cplxmul_fused_ms` (+ `_speedup`, `_max_err`)
- **chain**: `rfftn → mul → irfftn`, eager vs fused middle mul
  (`chain_eager_ms` / `chain_fused_ms` / `_speedup` / `_max_err`)
- **step**: `state = lam*state + b*x`, `step_eager_ms` vs `step_fused_ms`
  Writes `bench_triton.json`. On CPU it writes a `{"status":"skipped",...}` stub
  and exits 0.

## Exact operator run commands (CUDA box)

```bash
cd <repo>/experiments/wavefield_video

# 0. deps (if needed) — torch w/ CUDA + triton
python -c "import triton, torch; print(triton.__version__, torch.version.cuda)"
# pip install triton          # only if the import above fails

# 1. correctness: Triton vs eager (must print 'self-check PASSED', exit 0)
python triton_fused.py

# 2. benchmark -> writes bench_triton.json, prints per-config JSON lines
python bench_triton.py
cat bench_triton.json
```

Expected: self-check `max|err| ~1e-6..1e-4`; `bench_triton.json` has a
`configs` array (grid 16 & 32) with the eager/fused ms, per-op `speedup`, and
`max_err` fields. If run on CPU by mistake, both scripts skip cleanly (the
JSON will be the `status: skipped` stub) — re-run on the GPU host.

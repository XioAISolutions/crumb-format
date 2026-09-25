#!/usr/bin/env python3
"""CUDA micro-bench: eager complex-FFT ops vs fused Triton kernels (R6).

Two comparisons, over grid 16/32 with 8 heads (the dispersion mixer's hot
shapes):

  1. complex multiply  -- the ``X * K`` (+ optional gate) step:
       * cplxmul: eager complex-tensor mul vs fused_complex_mul (with gate)
       * chain:   rfftn -> mul -> irfftn, eager mul vs fused mul in the middle
  2. streaming step    -- ``state = lam*state + b*x`` one-frame recurrence:
       eager complex-tensor FMA vs fused_step_inner

Writes bench_triton.json. CUDA only; on CPU it writes a skip stub and exits 0
(the operator owns the CUDA box). Run: python bench_triton.py
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from triton_fused import (
    HAS_TRITON, eager_complex_mul, eager_step_inner,
    fused_complex_mul, fused_step_inner,
)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return round((time.perf_counter() - t0) / iters * 1e3, 4)


def bench_config(g, nh=8, B=2, T=8, dim=64, n_modes=3, dev="cuda"):
    dh = dim // nh
    L = 2 * T                       # temporal zero-pad (linear conv)
    Hp = Wp = 2 * g                 # spatial zero-pad
    res = {"grid": g, "heads": nh, "dh": dh, "L": L, "Hp": Hp, "Wp": Wp}

    # ---- 1a. complex multiply (with gate), the raw fused op ---------------
    # Pre-expand K and gate to X's full shape OUTSIDE the timed region so this
    # is a true kernel-vs-kernel comparison: eager broadcasts for free while
    # the fused wrapper would otherwise materialize the broadcast in-loop.
    Wr = Wp // 2 + 1                                  # rfft last-dim size
    X = torch.randn(B, nh, L, Hp, Wr, dh, dtype=torch.cfloat, device=dev)
    K = torch.randn(nh, L, Hp, Wr, dtype=torch.cfloat, device=dev)[None, ..., None]
    K = K.expand_as(X).contiguous()
    gate = torch.rand(B, nh, 1, 1, 1, dh, device=dev).expand_as(X.real).contiguous()
    # deinterleave complex -> planar re/im once (contiguous), so the wrapper's
    # broadcast_tensors/.contiguous() is a no-op inside the timed loop.
    xr, xi = X.real.contiguous(), X.imag.contiguous()
    kr, ki = K.real.contiguous(), K.imag.contiguous()

    res["cplxmul_eager_ms"] = timeit(
        lambda: eager_complex_mul(xr, xi, kr, ki, gate))
    res["cplxmul_fused_ms"] = timeit(
        lambda: fused_complex_mul(xr, xi, kr, ki, gate))
    # correctness (eager ground truth vs fused)
    er = eager_complex_mul(xr, xi, kr, ki, gate)
    fr = fused_complex_mul(xr, xi, kr, ki, gate)
    res["cplxmul_max_err"] = float(max((er[0] - fr[0]).abs().max(),
                                       (er[1] - fr[1]).abs().max()))
    res["cplxmul_speedup"] = round(res["cplxmul_eager_ms"] / res["cplxmul_fused_ms"], 3)

    # ---- 1b. full rfftn -> mul -> irfftn chain ---------------------------
    h = torch.randn(B, nh, T, g, g, dh, device=dev)
    Kc = torch.randn(nh, T, g, g // 2 + 1, dtype=torch.cfloat, device=dev)[None, ..., None]

    def chain_eager():
        Xf = torch.fft.rfftn(h, dim=(2, 3, 4))
        Yf = Xf * Kc
        return torch.fft.irfftn(Yf, s=(T, g, g), dim=(2, 3, 4))

    def chain_fused():
        Xf = torch.fft.rfftn(h, dim=(2, 3, 4))
        yr, yi = fused_complex_mul(Xf.real, Xf.imag, Kc.real, Kc.imag)
        Yf = torch.complex(yr, yi)
        return torch.fft.irfftn(Yf, s=(T, g, g), dim=(2, 3, 4))

    res["chain_eager_ms"] = timeit(chain_eager)
    res["chain_fused_ms"] = timeit(chain_fused)
    res["chain_max_err"] = float((chain_eager() - chain_fused()).abs().max())
    res["chain_speedup"] = round(res["chain_eager_ms"] / res["chain_fused_ms"], 3)

    # ---- 2. streaming step: state = lam*state + b*x ----------------------
    M = n_modes * nh
    state = torch.randn(B, M, Hp, Wp, dtype=torch.cfloat, device=dev)
    xin = torch.randn(B, M, Hp, Wp, dtype=torch.cfloat, device=dev)
    lam = (torch.rand(M, Hp, Wp, device=dev) * 0.9 * torch.exp(
        1j * torch.rand(M, Hp, Wp, device=dev) * 6.28).to(torch.cfloat))[None]
    lam = lam.expand(B, M, Hp, Wp).contiguous()        # full shape, timed fairly
    bg = torch.rand(M, 1, 1, device=dev)[None].expand(B, M, Hp, Wp).contiguous()
    zr, zi = state.real.contiguous(), state.imag.contiguous()
    lr, li = lam.real.contiguous(), lam.imag.contiguous()
    xr2, xi2 = xin.real.contiguous(), xin.imag.contiguous()

    def step_eager():
        return eager_step_inner(zr, zi, lr, li, bg, xr2, xi2)

    res["step_eager_ms"] = timeit(step_eager)
    res["step_fused_ms"] = timeit(
        lambda: fused_step_inner(zr, zi, lr, li, bg, xr2, xi2))
    es = step_eager()
    fs = fused_step_inner(zr, zi, lr, li, bg, xr2, xi2)
    res["step_max_err"] = float(max((es[0] - fs[0]).abs().max(),
                                    (es[1] - fs[1]).abs().max()))
    res["step_speedup"] = round(res["step_eager_ms"] / res["step_fused_ms"], 3)
    return res


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bench_triton.json")
    if not (HAS_TRITON and torch.cuda.is_available()):
        stub = {"status": "skipped",
                "reason": "no CUDA/Triton on this host; run on the CUDA box",
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "triton_available": HAS_TRITON}
        open(out_path, "w").write(json.dumps(stub, indent=1))
        print(json.dumps(stub))
        return

    out = {"device": "cuda",
           "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "configs": []}
    for g in (16, 32):
        cfg = bench_config(g)
        out["configs"].append(cfg)
        print(json.dumps(cfg))
    open(out_path, "w").write(json.dumps(out, indent=1))
    print(f"[bench_triton] wrote {out_path}")


if __name__ == "__main__":
    main()

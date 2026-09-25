#!/usr/bin/env python3
"""Fused Triton prototypes for the wave-field FFT path (R6).

torch.compile cannot codegen our complex-FFT mixer (it bails on the cfloat
intermediates), so the two hot elementwise ops are hand-written as Triton
kernels. Triton has NO complex dtype, so real/imag parts are packed as two
separate float32 tensors and the complex algebra is done by hand.

Two kernels, each a SINGLE pass over the flattened spatial (all-dims) index:

  kernelA  fused_complex_mul
      Elementwise complex multiply of an rfft spectrum X = (xr + i*xi) by a
      kernel spectrum K = (kr + i*ki), with an OPTIONAL real gate scale g:
          out = g * (X * K)
          out_re = g * (xr*kr - xi*ki)
          out_im = g * (xr*ki + xi*kr)
      This is the ``X * K`` step inside WaveMix3D._wave_dispersion /
      _wave_separable. Eager needs a complex-mul launch + a separate gate-scale
      launch (and a temporary); the fused kernel does mul->gate->store in one
      pass with no intermediate DRAM traffic.

  kernelB  fused_step_inner
      One frame of the streaming recurrence ``state = lam*state + b*x`` over a
      flattened [B, M, H, W] (M = n_modes*n_heads) tensor packed as 2 reals,
      with lam complex, x complex, b a REAL input gain:
          z_re = lr*zr - li*zi + b*xr
          z_im = lr*zi + li*zr + b*xi
      This is the per-cell complex FMA in WaveMix3D.step(). Eager materializes
      lam*state and b*x as two temporaries then adds; the fused kernel keeps
      everything in registers.

CUDA only. On CPU (no Triton GPU) the eager references still run and the
self-check exits cleanly with a skip message -- the operator has the CUDA box.
"""
import sys

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                   # triton not installed
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Eager reference implementations (device-agnostic, used by the self-check
# and by bench_triton.py as the ground truth to compare the kernels against).
# ---------------------------------------------------------------------------
def eager_complex_mul(xr, xi, kr, ki, gate=None):
    """Reference for kernelA: out = gate * (X * K), returned as (re, im)."""
    out_re = xr * kr - xi * ki
    out_im = xr * ki + xi * kr
    if gate is not None:
        out_re = out_re * gate
        out_im = out_im * gate
    return out_re, out_im


def eager_step_inner(zr, zi, lr, li, b, xr, xi):
    """Reference for kernelB: state = lam*state + b*x, returned as (re, im)."""
    new_re = lr * zr - li * zi + b * xr
    new_im = lr * zi + li * zr + b * xi
    return new_re, new_im


if HAS_TRITON:

    # -- kernelA -----------------------------------------------------------
    @triton.jit
    def _complex_mul_kernel(
        xr_ptr, xi_ptr, kr_ptr, ki_ptr, g_ptr, or_ptr, oi_ptr,
        n_elem,
        HAS_GATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        xr = tl.load(xr_ptr + offs, mask=mask, other=0.0)
        xi = tl.load(xi_ptr + offs, mask=mask, other=0.0)
        kr = tl.load(kr_ptr + offs, mask=mask, other=0.0)
        ki = tl.load(ki_ptr + offs, mask=mask, other=0.0)
        o_re = xr * kr - xi * ki
        o_im = xr * ki + xi * kr
        if HAS_GATE:
            g = tl.load(g_ptr + offs, mask=mask, other=0.0)
            o_re = o_re * g
            o_im = o_im * g
        tl.store(or_ptr + offs, o_re, mask=mask)
        tl.store(oi_ptr + offs, o_im, mask=mask)

    # -- kernelB -----------------------------------------------------------
    @triton.jit
    def _step_inner_kernel(
        zr_ptr, zi_ptr, lr_ptr, li_ptr, b_ptr, xr_ptr, xi_ptr,
        or_ptr, oi_ptr,
        n_elem,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        zr = tl.load(zr_ptr + offs, mask=mask, other=0.0)
        zi = tl.load(zi_ptr + offs, mask=mask, other=0.0)
        lr = tl.load(lr_ptr + offs, mask=mask, other=0.0)
        li = tl.load(li_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        xr = tl.load(xr_ptr + offs, mask=mask, other=0.0)
        xi = tl.load(xi_ptr + offs, mask=mask, other=0.0)
        n_re = lr * zr - li * zi + b * xr
        n_im = lr * zi + li * zr + b * xi
        tl.store(or_ptr + offs, n_re, mask=mask)
        tl.store(oi_ptr + offs, n_im, mask=mask)


def _check_cuda_f32(*tensors):
    for t in tensors:
        if t is None:
            continue
        if not t.is_cuda:
            raise ValueError("fused kernels require CUDA tensors")
        if t.dtype != torch.float32:
            raise ValueError("fused kernels are float32-only")


def fused_complex_mul(xr, xi, kr, ki, gate=None, block=1024):
    """kernelA wrapper: out = gate * (X*K). Inputs are broadcast to a common
    shape, then flattened; returns (out_re, out_im) with that broadcast shape.

    xr,xi,kr,ki: float32 CUDA tensors (broadcastable). gate: optional float32
    real scale (broadcastable) or None."""
    if not HAS_TRITON:
        raise RuntimeError("triton unavailable")
    if gate is None:
        xr, xi, kr, ki = torch.broadcast_tensors(xr, xi, kr, ki)
    else:
        xr, xi, kr, ki, gate = torch.broadcast_tensors(xr, xi, kr, ki, gate)
    _check_cuda_f32(xr, xi, kr, ki, gate)
    xr, xi, kr, ki = (t.contiguous() for t in (xr, xi, kr, ki))
    shape = xr.shape
    xr, xi, kr, ki = (t.reshape(-1) for t in (xr, xi, kr, ki))
    n = xr.numel()
    o_re = torch.empty_like(xr)
    o_im = torch.empty_like(xr)
    g_flat = gate.contiguous().reshape(-1) if gate is not None else xr  # unused ptr if no gate
    grid = (triton.cdiv(n, block),)
    _complex_mul_kernel[grid](
        xr, xi, kr, ki, g_flat, o_re, o_im,
        n, HAS_GATE=(gate is not None), BLOCK=block,
    )
    return o_re.reshape(shape), o_im.reshape(shape)


def fused_step_inner(zr, zi, lr, li, b, xr, xi, block=1024):
    """kernelB wrapper: state = lam*state + b*x for one frame. Inputs are
    broadcast to a common shape (so lam [M,H,W] and b [M,1,1] can broadcast
    over the batch), flattened, and updated in one pass. Returns the new
    (state_re, state_im)."""
    if not HAS_TRITON:
        raise RuntimeError("triton unavailable")
    zr, zi, lr, li, b, xr, xi = torch.broadcast_tensors(zr, zi, lr, li, b, xr, xi)
    _check_cuda_f32(zr, zi, lr, li, b, xr, xi)
    zr, zi, lr, li, b, xr, xi = (t.contiguous() for t in (zr, zi, lr, li, b, xr, xi))
    shape = zr.shape
    zr, zi, lr, li, b, xr, xi = (t.reshape(-1) for t in (zr, zi, lr, li, b, xr, xi))
    n = zr.numel()
    o_re = torch.empty_like(zr)
    o_im = torch.empty_like(zr)
    grid = (triton.cdiv(n, block),)
    _step_inner_kernel[grid](
        zr, zi, lr, li, b, xr, xi, o_re, o_im, n, BLOCK=block,
    )
    return o_re.reshape(shape), o_im.reshape(shape)


# ---------------------------------------------------------------------------
# Self-check: Triton kernel vs eager reference (CUDA only; clean skip on CPU).
# ---------------------------------------------------------------------------
def _max_err(a, b):
    return (a - b).abs().max().item()


def self_check(seed=0):
    if not (HAS_TRITON and torch.cuda.is_available()):
        print("[triton_fused] self-check SKIPPED (no CUDA/Triton on this host); "
              "eager references are importable. Run on the CUDA box.")
        return True
    torch.manual_seed(seed)
    dev = "cuda"
    ok = True

    # kernelA: complex mul, with and without gate. Shapes exercise broadcasting
    # (kernel K broadcast over batch/channel, mimicking X[B,nh,L,Hp,Wp,dh] * G).
    B, nh, L, Hp, Wp, dh = 2, 4, 8, 12, 12, 8
    X = torch.randn(B, nh, L, Hp, Wp, dh, dtype=torch.cfloat, device=dev)
    K = torch.randn(nh, L, Hp, Wp, dtype=torch.cfloat, device=dev)[None, ..., None]
    g = torch.rand(B, nh, 1, 1, 1, dh, device=dev)
    for use_gate in (False, True):
        gate = g if use_gate else None
        gr = eager_complex_mul(X.real, X.imag, K.real, K.imag, gate)
        tr = fused_complex_mul(X.real, X.imag, K.real, K.imag, gate)
        er = max(_max_err(gr[0], tr[0]), _max_err(gr[1], tr[1]))
        tag = "with-gate" if use_gate else "no-gate  "
        good = er < 1e-4
        ok = ok and good
        print(f"[kernelA complex_mul {tag}] max|err|={er:.2e}  {'OK' if good else 'FAIL'}")

    # kernelB: streaming recurrence step. lam [M,Hp,Wp] and b [M,1,1] broadcast
    # over batch B and channel dh, matching WaveMix3D.step's state update.
    Bm, M, Hp, Wp = 2, 6, 16, 16
    state = torch.randn(Bm, M, Hp, Wp, dtype=torch.cfloat, device=dev)
    x = torch.randn(Bm, M, Hp, Wp, dtype=torch.cfloat, device=dev)
    lam = (torch.rand(M, Hp, Wp, device=dev) * 0.9) * torch.exp(
        1j * torch.rand(M, Hp, Wp, device=dev) * 6.28).to(torch.cfloat)
    lam = lam[None]                                   # [1,M,Hp,Wp]
    b = torch.rand(M, 1, 1, device=dev)[None]         # [1,M,1,1] real gain
    gr = eager_step_inner(state.real, state.imag, lam.real, lam.imag,
                          b, x.real, x.imag)
    tr = fused_step_inner(state.real, state.imag, lam.real, lam.imag,
                          b, x.real, x.imag)
    er = max(_max_err(gr[0], tr[0]), _max_err(gr[1], tr[1]))
    good = er < 1e-4
    ok = ok and good
    print(f"[kernelB step_inner] max|err|={er:.2e}  {'OK' if good else 'FAIL'}")

    print("[triton_fused] self-check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if self_check() else 1)

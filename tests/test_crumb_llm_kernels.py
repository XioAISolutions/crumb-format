"""Wave-Field kernel + FFT-convolve correctness tests."""

import math

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.kernels import (  # noqa: E402
    wave_kernel_time,
    wave_kernel_freq,
    fft_convolve,
)


def test_kernel_time_shape_and_peak():
    alpha = torch.tensor(0.1)
    omega = torch.tensor(0.0)
    phi = torch.tensor(0.0)
    F = 16
    k = wave_kernel_time(alpha, omega, phi, F)
    assert k.shape == (F,)
    # For ω=φ=0, the kernel reduces to exp(-α|t|) — peak at the centre.
    assert torch.argmax(k).item() == F // 2


def test_kernel_time_is_real_and_finite():
    alpha = torch.tensor(0.05)
    omega = torch.tensor(2.0)
    phi = torch.tensor(0.5)
    k = wave_kernel_time(alpha, omega, phi, 32)
    assert k.dtype.is_floating_point
    assert torch.isfinite(k).all()


def test_fft_convolve_matches_manual_circular():
    """FFT convolve must equal an explicit Σ over circular shifts."""
    torch.manual_seed(0)
    F, H, D, B = 16, 2, 3, 1
    field = torch.randn(B, H, F, D)

    # Hand-build a circular kernel in time, t=0 at index 0.
    k_t = torch.zeros(H, F)
    k_t[0, 0] = 1.0
    k_t[0, 1] = 0.5
    k_t[1, 0] = 0.7
    k_t[1, 3] = 0.3
    K = torch.fft.rfft(k_t, n=F, dim=-1)

    out = fft_convolve(field, K, F)

    manual = torch.zeros_like(field)
    for h in range(H):
        for i in range(F):
            for j in range(F):
                manual[0, h, i, :] += k_t[h, j] * field[0, h, (i - j) % F, :]

    torch.testing.assert_close(out, manual, atol=1e-5, rtol=1e-5)


def test_fft_convolve_rejects_size_mismatch():
    field = torch.randn(1, 1, 8, 1)
    K = torch.fft.rfft(torch.zeros(1, 8), n=8, dim=-1)
    with pytest.raises(ValueError, match="field axis -2"):
        fft_convolve(field, K, field_size=16)


def test_kernel_freq_is_complex():
    a = torch.tensor(0.1)
    w = torch.tensor(1.0)
    p = torch.tensor(0.0)
    K = wave_kernel_freq(a, w, p, 32)
    assert K.is_complex()
    assert K.shape == (32 // 2 + 1,)


def test_kernel_freq_gradient_flow():
    """All three physics scalars must receive non-zero gradients."""
    a = torch.tensor(0.1, requires_grad=True)
    w = torch.tensor(1.0, requires_grad=True)
    p = torch.tensor(0.3, requires_grad=True)
    K = wave_kernel_freq(a, w, p, 32)
    # Sum of magnitudes; backward should populate .grad on all three.
    K.abs().sum().backward()
    assert a.grad is not None and a.grad.abs() > 0
    assert w.grad is not None and w.grad.abs() > 0
    assert p.grad is not None and p.grad.abs() > 0

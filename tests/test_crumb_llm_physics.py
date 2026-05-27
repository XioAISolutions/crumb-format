"""Tests for advanced wave physics: boundary conditions, dispersion,
Gabor kernels, interference mixer, resonance detection."""

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.physics import (  # noqa: E402
    BoundaryCondition,
    DispersionLayer,
    GaborWaveletKernel,
    InterferenceMixer,
    apply_boundary,
    detect_resonances,
    unapply_reflecting,
)
from crumb_llm.layers import WaveFieldBlock, WaveFieldBlockConfig  # noqa: E402
from crumb_llm.model import WaveFieldLM, WaveFieldConfig  # noqa: E402


# ── Boundary conditions ──────────────────────────────────────────────


def test_periodic_boundary_is_identity():
    field = torch.randn(1, 2, 32, 4)
    out = apply_boundary(field, BoundaryCondition.PERIODIC)
    torch.testing.assert_close(out, field)


def test_absorbing_boundary_tapers_edges():
    field = torch.ones(1, 1, 64, 1)
    out = apply_boundary(field, BoundaryCondition.ABSORBING, taper_width=8)
    assert out.shape == field.shape
    # Edges should be < 1, interior should be exactly 1.
    assert out[0, 0, 0, 0].item() < 0.01
    assert out[0, 0, 32, 0].item() == 1.0
    assert out[0, 0, -1, 0].item() < 0.01


def test_reflecting_boundary_expands_field():
    field = torch.randn(1, 1, 32, 4)
    out = apply_boundary(field, BoundaryCondition.REFLECTING, taper_width=4)
    # Mirror-padding adds 2 * taper_width.
    assert out.shape[-2] == 32 + 2 * 4
    # Un-padding recovers original size.
    recovered = unapply_reflecting(out, taper_width=4)
    assert recovered.shape == field.shape
    torch.testing.assert_close(recovered, field)


# ── Dispersion ───────────────────────────────────────────────────────


def test_dispersion_preserves_shape():
    disp = DispersionLayer(n_heads=4)
    F = 32
    spec = torch.randn(2, 4, F // 2 + 1, 8, dtype=torch.cfloat)
    out = disp(spec, field_size=F)
    assert out.shape == spec.shape
    assert out.is_complex()


def test_dispersion_gradient_flows():
    disp = DispersionLayer(n_heads=2)
    spec = torch.randn(1, 2, 9, 4, dtype=torch.cfloat)
    out = disp(spec, field_size=16)
    out.abs().sum().backward()
    assert disp.beta.grad is not None
    assert disp.beta.grad.abs().sum() > 0


# ── Gabor wavelet ────────────────────────────────────────────────────


def test_gabor_kernel_output_shape():
    gk = GaborWaveletKernel(field_size=32)
    K = gk()
    assert K.is_complex()
    assert K.shape == (32 // 2 + 1,)


def test_gabor_causal_zeros_past():
    gk = GaborWaveletKernel(field_size=32, causal=True)
    # The time-domain kernel before rFFT has t<0 zeroed.
    s = gk.sigma.abs().clamp(min=0.5)
    gauss = torch.exp(-gk.t ** 2 / (2 * s * s))
    wave = torch.cos(gk.omega * gk.t + gk.phi)
    k = gauss * wave
    mask = torch.zeros_like(k)
    mask[32 // 2:] = 1.0
    k_causal = k * mask
    assert k_causal[:32 // 2].abs().max() == 0


def test_gabor_gradient_flows():
    gk = GaborWaveletKernel(field_size=32)
    K = gk()
    K.abs().sum().backward()
    assert gk.sigma.grad is not None
    assert gk.omega.grad is not None
    assert gk.phi.grad is not None


# ── Interference mixer ───────────────────────────────────────────────


def test_interference_mixer_shape():
    mixer = InterferenceMixer(n_heads=4, output_mode="real")
    x = torch.randn(2, 4, 16, 8)
    out = mixer(x)
    assert out.shape == x.shape


def test_interference_mixer_modulus_mode():
    mixer = InterferenceMixer(n_heads=3, output_mode="modulus")
    x = torch.randn(1, 3, 8, 4)
    out = mixer(x)
    assert out.shape == x.shape
    assert (out >= 0).all()


def test_interference_mixer_gradient_flows():
    mixer = InterferenceMixer(n_heads=2)
    x = torch.randn(1, 2, 4, 3, requires_grad=True)
    out = mixer(x)
    out.sum().backward()
    assert x.grad is not None
    assert mixer.magnitude.grad is not None
    assert mixer.phase.grad is not None


# ── Resonance detection ──────────────────────────────────────────────


def test_detect_resonances_returns_peaks():
    # Build a field with a known spatial frequency.
    F = 64
    t = torch.arange(F, dtype=torch.float32)
    field = torch.sin(2 * 3.14159 * 4 * t / F).unsqueeze(-1)  # 4 cycles per F
    peaks = detect_resonances(field, top_k=3)
    assert len(peaks) == 3
    assert peaks[0]["bin"] == 4  # dominant frequency bin
    assert peaks[0]["power"] > peaks[1]["power"]


# ── Integration: block with advanced physics ─────────────────────────


def test_block_with_dispersion_runs():
    cfg = WaveFieldBlockConfig(
        dim=32, n_heads=4, field_size=64,
        dispersion=True, boundary="periodic",
    )
    block = WaveFieldBlock(cfg)
    x = torch.randn(2, 16, 32)
    out = block(x)
    assert out.shape == x.shape


def test_block_with_absorbing_boundary():
    cfg = WaveFieldBlockConfig(
        dim=32, n_heads=4, field_size=64,
        boundary="absorbing",
    )
    block = WaveFieldBlock(cfg)
    x = torch.randn(1, 8, 32)
    out = block(x)
    assert out.shape == x.shape


def test_block_with_interference_mixer():
    cfg = WaveFieldBlockConfig(
        dim=32, n_heads=4, field_size=64,
        interference_mixer=True,
    )
    block = WaveFieldBlock(cfg)
    x = torch.randn(1, 8, 32)
    out = block(x)
    assert out.shape == x.shape


def test_full_model_with_all_physics():
    """End-to-end: model with dispersion + absorbing BC + interference."""
    cfg = WaveFieldConfig(
        vocab_size=32, dim=16, n_layers=2, n_heads=2, field_size=32,
        boundary="absorbing", dispersion=True, interference_mixer=True,
    )
    model = WaveFieldLM(cfg)
    x = torch.randint(0, 32, (1, 8))
    y = torch.randint(0, 32, (1, 8))
    out = model(x, targets=y)
    assert out["logits"].shape == (1, 8, 32)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

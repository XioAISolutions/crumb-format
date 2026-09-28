"""Tests for Crumb LLM v2 building blocks."""

import math

import pytest

torch = pytest.importorskip("torch")

from crumb_wavelm.v2 import (  # noqa: E402
    SmoothCausalKernel,
    LearnedScatterGather,
    QueryFrequencyGate,
    ShortConvGate,
    PositionKernelModulator,
    RotaryFieldEncoding,
    WaveFieldBlockV2,
    WaveFieldBlockV2Config,
)


# ── Smooth causal kernel ─────────────────────────────────────────────


def test_smooth_kernel_no_hard_discontinuity():
    """The smooth kernel should NOT have a sharp jump at t=0."""
    k = SmoothCausalKernel(field_size=64)
    K = k()
    assert K.is_complex()
    # Reconstruct time domain to check smoothness.
    k_t = torch.fft.irfft(K, n=64)
    # Check that values near the center don't jump to exactly zero.
    center = 64 // 2
    diffs = (k_t[center - 1] - k_t[center]).abs()
    # With smooth sigmoid, the jump should be small (not like hard cutoff).
    assert diffs < 0.5, f"discontinuity at center: {diffs}"


def test_smooth_kernel_gradient_flows():
    k = SmoothCausalKernel(field_size=32)
    K = k()
    K.abs().sum().backward()
    assert k.alpha.grad is not None
    assert k.omega.grad is not None
    assert k.causal_sharpness.grad is not None


# ── Learned scatter/gather ───────────────────────────────────────────


def test_learned_scatter_shape():
    sg = LearnedScatterGather(field_size=32, basis_width=4)
    v = torch.randn(2, 4, 8, 16)
    field = sg.scatter(v)
    assert field.shape == (2, 4, 32, 16)


def test_learned_gather_shape():
    sg = LearnedScatterGather(field_size=32)
    field = torch.randn(2, 4, 32, 16)
    out = sg.gather(field, N=8)
    assert out.shape == (2, 4, 8, 16)


def test_learned_scatter_gradient():
    sg = LearnedScatterGather(field_size=16)
    v = torch.randn(1, 2, 4, 8, requires_grad=True)
    field = sg.scatter(v)
    field.sum().backward()
    assert v.grad is not None
    assert sg.log_sigma.grad is not None


def test_learned_scatter_conserves_mass():
    sg = LearnedScatterGather(field_size=32)
    v = torch.ones(1, 1, 4, 1)
    field = sg.scatter(v)
    # Basis is normalized, so total mass ≈ N.
    assert torch.allclose(field.sum(), torch.tensor(4.0), atol=0.1)


# ── Query frequency gate ─────────────────────────────────────────────


def test_query_gate_shape():
    gate = QueryFrequencyGate(d_head=8, n_heads=4, field_size=32)
    spec = torch.randn(2, 4, 17, 8, dtype=torch.cfloat)
    query = torch.randn(2, 4, 16, 8)
    out = gate(spec, query)
    assert out.shape == spec.shape
    assert out.is_complex()


def test_query_gate_varies_with_input():
    gate = QueryFrequencyGate(d_head=8, n_heads=2, field_size=16)
    spec = torch.randn(1, 2, 9, 4, dtype=torch.cfloat)
    q1 = torch.randn(1, 2, 4, 8)
    q2 = torch.randn(1, 2, 4, 8) * 5
    out1 = gate(spec, q1)
    out2 = gate(spec, q2)
    assert not torch.allclose(out1, out2)


# ── Short conv gate ──────────────────────────────────────────────────


def test_short_conv_shape():
    sc = ShortConvGate(dim=16, kernel_size=4)
    x = torch.randn(2, 8, 16)
    out = sc(x)
    assert out.shape == x.shape


def test_short_conv_is_causal():
    """Output at position t should not depend on input at t+1."""
    sc = ShortConvGate(dim=4, kernel_size=3)
    sc.eval()
    x = torch.randn(1, 6, 4)
    out1 = sc(x)
    # Zero out future tokens.
    x_masked = x.clone()
    x_masked[:, 4:, :] = 0
    out2 = sc(x_masked)
    # Positions 0..3 should be identical.
    torch.testing.assert_close(out1[:, :4, :], out2[:, :4, :])


# ── Position kernel modulator ────────────────────────────────────────


def test_pos_mod_shape():
    pm = PositionKernelModulator(max_len=64, field_size=32, n_heads=4)
    spec = torch.randn(2, 4, 17, 8, dtype=torch.cfloat)
    out = pm(spec, seq_len=16)
    assert out.shape == spec.shape


# ── Rotary field encoding ────────────────────────────────────────────


def test_rope_shape():
    rope = RotaryFieldEncoding(d_head=8)
    x = torch.randn(2, 4, 16, 8)
    out = rope(x)
    assert out.shape == x.shape


def test_rope_different_positions_differ():
    rope = RotaryFieldEncoding(d_head=8)
    x = torch.ones(1, 1, 4, 8)
    out = rope(x)
    # Different positions should produce different rotations.
    assert not torch.allclose(out[:, :, 0, :], out[:, :, 1, :])


# ── Full V2 block ────────────────────────────────────────────────────


def test_v2_block_forward():
    cfg = WaveFieldBlockV2Config(dim=32, n_heads=4, field_size=64)
    block = WaveFieldBlockV2(cfg)
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.shape == x.shape


def test_v2_block_backward():
    cfg = WaveFieldBlockV2Config(dim=32, n_heads=4, field_size=64)
    block = WaveFieldBlockV2(cfg)
    x = torch.randn(1, 8, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_v2_block_all_features_off():
    cfg = WaveFieldBlockV2Config(
        dim=32, n_heads=4, field_size=64,
        learned_scatter=False, smooth_causal=True,
        query_freq_gate=False, short_conv_gate=False,
        position_modulation=False, rotary_encoding=False,
    )
    block = WaveFieldBlockV2(cfg)
    x = torch.randn(1, 8, 32)
    out = block(x)
    assert out.shape == x.shape


def test_v2_param_count_reasonable():
    cfg = WaveFieldBlockV2Config(dim=64, n_heads=4, field_size=128, max_len=256)
    block = WaveFieldBlockV2(cfg)
    n = sum(p.numel() for p in block.parameters())
    assert n < 500_000, f"too many params: {n}"

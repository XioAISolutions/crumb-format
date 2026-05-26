"""Tests for novel Crumb LLM architectural innovations."""

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.v2 import (  # noqa: E402
    AdaptiveKernelHead,
    FieldAttentionGate,
    ResonanceMemory,
    SpectralGate,
)


# ── Adaptive kernels ─────────────────────────────────────────────────


def test_adaptive_kernel_output_shape():
    head = AdaptiveKernelHead(dim=32, field_size=64)
    x = torch.randn(2, 8, 32)
    K = head(x)
    assert K.is_complex()
    assert K.shape == (2, 64 // 2 + 1)


def test_adaptive_kernel_varies_with_input():
    head = AdaptiveKernelHead(dim=16, field_size=32)
    x1 = torch.randn(1, 4, 16)
    x2 = torch.randn(1, 4, 16) * 10
    K1 = head(x1)
    K2 = head(x2)
    assert not torch.allclose(K1, K2)


def test_adaptive_kernel_gradient_flows():
    head = AdaptiveKernelHead(dim=16, field_size=32)
    x = torch.randn(1, 4, 16, requires_grad=True)
    K = head(x)
    K.abs().sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ── Field-attention hybrid gate ──────────────────────────────────────


def test_gate_blends_two_signals():
    gate = FieldAttentionGate(dim=16, window_size=4)
    x = torch.randn(1, 8, 16)
    wave_out = torch.randn(1, 8, 16)
    blended = gate(x, wave_out)
    assert blended.shape == (1, 8, 16)


def test_gate_gradient_flows():
    gate = FieldAttentionGate(dim=16, window_size=4)
    x = torch.randn(1, 6, 16, requires_grad=True)
    wave_out = torch.randn(1, 6, 16)
    blended = gate(x, wave_out)
    blended.sum().backward()
    assert x.grad is not None


def test_gate_extreme_values():
    """When gate=1.0, output should equal wave_output."""
    gate = FieldAttentionGate(dim=16, window_size=4)
    # Force gate to 1.0 by setting bias very high.
    with torch.no_grad():
        gate.gate_proj.bias.fill_(100.0)
    x = torch.randn(1, 4, 16)
    wave_out = torch.randn(1, 4, 16)
    blended = gate(x, wave_out)
    torch.testing.assert_close(blended, wave_out, atol=1e-3, rtol=1e-3)


# ── Resonance memory ────────────────────────────────────────────────


def test_resonance_memory_shapes():
    mem = ResonanceMemory(n_heads=4, field_size=32)
    memory_spec = torch.zeros(2, 4, 17)  # B, H, F//2+1
    field = torch.randn(2, 4, 32, 8)
    updated, amplified = mem(memory_spec, field)
    assert updated.shape == (2, 4, 17)
    assert amplified.shape == field.shape


def test_resonance_memory_accumulates():
    mem = ResonanceMemory(n_heads=2, field_size=16)
    memory = torch.zeros(1, 2, 9)
    field = torch.randn(1, 2, 16, 4)
    updated1, _ = mem(memory, field)
    updated2, _ = mem(updated1, field)
    # Memory should grow (inject adds to it).
    assert updated2.abs().sum() > updated1.abs().sum()


def test_resonance_amplifies_persistent_frequencies():
    mem = ResonanceMemory(n_heads=1, field_size=16)
    # Feed the same field multiple times — standing wave.
    t = torch.arange(16, dtype=torch.float32)
    field = torch.sin(2 * 3.14159 * 2 * t / 16).view(1, 1, 16, 1)
    memory = torch.zeros(1, 1, 9)
    for _ in range(5):
        memory, amplified = mem(memory, field)
    # The dominant frequency should be amplified above the original.
    assert amplified.abs().max() > field.abs().max()


# ── Spectral gate ───────────────────────────────────────────────────


def test_spectral_gate_shape():
    sg = SpectralGate(n_heads=4, field_size=32, dim=16)
    field = torch.randn(2, 4, 32, 8)
    x_pooled = torch.randn(2, 16)
    gated = sg(field, x_pooled)
    assert gated.shape == field.shape


def test_spectral_gate_is_input_conditioned():
    sg = SpectralGate(n_heads=2, field_size=16, dim=8)
    field = torch.randn(1, 2, 16, 4)
    out1 = sg(field, torch.randn(1, 8))
    out2 = sg(field, torch.randn(1, 8) * 5)
    assert not torch.allclose(out1, out2)


def test_spectral_gate_gradient():
    sg = SpectralGate(n_heads=2, field_size=16, dim=8)
    field = torch.randn(1, 2, 16, 4, requires_grad=True)
    x = torch.randn(1, 8)
    gated = sg(field, x)
    gated.sum().backward()
    assert field.grad is not None

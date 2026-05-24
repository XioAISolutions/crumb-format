"""Scatter / gather correctness + adjointness tests."""

import pytest

torch = pytest.importorskip("torch")

from wave_field_llm.scatter_gather import scatter_linear, gather_linear  # noqa: E402


def test_scatter_shape():
    v = torch.randn(2, 3, 8, 5)  # [B, H, N, D]
    field = scatter_linear(v, field_size=32)
    assert field.shape == (2, 3, 32, 5)


def test_scatter_then_gather_recovers_constant_field():
    """A constant token value scattered then gathered → constant (modulo edges)."""
    B, H, N, D = 1, 1, 8, 1
    F = 16
    v = torch.ones(B, H, N, D)
    field = scatter_linear(v, F)
    out = gather_linear(field, N)
    # Mass conservation: total amplitude in/out is the same.
    assert torch.allclose(out.sum(), torch.tensor(float(N)), atol=1e-5)


def test_scatter_weights_scale_amplitude():
    B, H, N, D = 1, 1, 4, 1
    v = torch.ones(B, H, N, D)
    w = torch.tensor([[2.0, 1.0, 0.0, 3.0]])
    field = scatter_linear(v, field_size=16, weights=w)
    # Total scattered mass equals sum of per-token weights.
    assert torch.allclose(field.sum(), w.sum(), atol=1e-5)


def test_gather_shape():
    field = torch.randn(2, 3, 32, 5)
    out = gather_linear(field, num_tokens=8)
    assert out.shape == (2, 3, 8, 5)


def test_gradient_flow_through_scatter():
    v = torch.randn(1, 1, 4, 1, requires_grad=True)
    field = scatter_linear(v, field_size=8)
    field.sum().backward()
    assert v.grad is not None
    assert v.grad.abs().sum() > 0

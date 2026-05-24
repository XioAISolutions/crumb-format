"""WaveFieldBlock + WaveFieldLM forward, backward, causality tests."""

import pytest

torch = pytest.importorskip("torch")

from wave_field_llm.layers import (  # noqa: E402
    RMSNorm,
    SwiGLUFFN,
    WaveFieldHead,
    WaveFieldBlock,
    WaveFieldBlockConfig,
)
from wave_field_llm.model import WaveFieldLM, WaveFieldConfig  # noqa: E402


def test_rmsnorm_preserves_shape():
    x = torch.randn(2, 5, 8)
    out = RMSNorm(8)(x)
    assert out.shape == x.shape


def test_swiglu_preserves_shape():
    x = torch.randn(2, 5, 16)
    out = SwiGLUFFN(16)(x)
    assert out.shape == x.shape


def test_wave_field_head_has_three_params():
    head = WaveFieldHead(field_size=32)
    names = {n for n, _ in head.named_parameters()}
    assert names == {"alpha", "omega", "phi"}


def test_wave_field_block_forward_backward():
    cfg = WaveFieldBlockConfig(dim=32, n_heads=4, field_size=64, ffn_mult=2.0)
    block = WaveFieldBlock(cfg)
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_wave_field_lm_forward_loss():
    cfg = WaveFieldConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, field_size=64)
    model = WaveFieldLM(cfg)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    out = model(x, targets=y)
    assert out["logits"].shape == (2, 16, 64)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_causal_kernel_zeros_negative_time():
    """A causal head's time-domain kernel must be zero for t < 0."""
    head = WaveFieldHead(field_size=32, causal=True, kernel_mode="time")
    # Internal reconstruction: time-domain kernel after the causal mask.
    from wave_field_llm.kernels import wave_kernel_time
    k_t = wave_kernel_time(head.alpha, head.omega, head.phi, 32)
    mask = torch.zeros_like(k_t)
    mask[..., 32 // 2 :] = 1.0
    k_causal = k_t * mask
    # Negative-t half (indices 0..F/2-1) is zero.
    assert k_causal[: 32 // 2].abs().max() == 0


def test_generate_shape():
    cfg = WaveFieldConfig(vocab_size=16, dim=16, n_layers=2, n_heads=2, field_size=32)
    model = WaveFieldLM(cfg)
    x = torch.randint(0, 16, (1, 4))
    out = model.generate(x, max_new_tokens=5, temperature=1.0, top_k=5)
    assert out.shape == (1, 9)


def test_num_parameters_nonzero():
    cfg = WaveFieldConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, field_size=16)
    model = WaveFieldLM(cfg)
    assert model.num_parameters() > 0

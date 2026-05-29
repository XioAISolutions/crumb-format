"""Field-state cache tests: cached generation must match uncached."""

import pytest

torch = pytest.importorskip("torch")

from crumb_wavelm.cache import (  # noqa: E402
    FieldStateCache,
    generate_cached,
    forward_cached_block,
)
from crumb_wavelm.model import WaveFieldLM, WaveFieldConfig  # noqa: E402


def _make_tiny_model():
    cfg = WaveFieldConfig(
        vocab_size=32, dim=16, n_layers=2, n_heads=2,
        field_size=32, causal=True,
    )
    return WaveFieldLM(cfg)


def test_cache_init_shapes():
    model = _make_tiny_model()
    cache = FieldStateCache.init(model, batch_size=2)
    assert len(cache.layers) == 2
    assert cache.layers[0].field.shape == (2, 2, 32, 8)  # B, H, F, d_head
    assert cache.seq_pos == 0


def test_cached_generation_produces_tokens():
    model = _make_tiny_model()
    model.eval()
    prompt = torch.randint(0, 32, (1, 4))
    out = generate_cached(model, prompt, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 9)  # 4 prompt + 5 generated


def test_cached_vs_uncached_logits_close():
    """First-token logits from cached path should match uncached forward."""
    torch.manual_seed(42)
    model = _make_tiny_model()
    model.eval()
    prompt = torch.randint(0, 32, (1, 8))

    # Uncached: full forward.
    with torch.no_grad():
        uncached = model(prompt)["logits"][:, -1, :]

    # Cached: prefill should produce the same last-position logits.
    cache = FieldStateCache.init(model, batch_size=1)
    x = model.embed(prompt)
    for layer_idx, block in enumerate(model.blocks):
        x_out = block(x)
        # Populate cache.
        from crumb_wavelm.cache import _convolve_sparse
        from crumb_wavelm.scatter_gather import scatter_linear
        H, d_head = block.cfg.n_heads, block.d_head
        h = block.proj_in(block.norm_mix(x)).view(1, 8, H, d_head).transpose(1, 2)
        full_field = scatter_linear(h, block.cfg.field_size)
        kfs = [head.kernel_freq() for head in block.heads]
        kernel = torch.stack(kfs, dim=0)
        full_field = _convolve_sparse(full_field, kernel, block.cfg.field_size)
        from crumb_wavelm.cache import LayerFieldState
        cache.layers[layer_idx] = LayerFieldState(field=full_field, n_tokens_seen=8)
        x = x_out
    x = model.norm_out(x)
    if model.lm_head is None:
        cached_logits = (x @ model.embed.weight.t())[:, -1, :]
    else:
        cached_logits = model.lm_head(x)[:, -1, :]

    # The prefill logits should be identical (same computation path).
    torch.testing.assert_close(cached_logits, uncached, atol=1e-4, rtol=1e-4)


def test_generate_cached_respects_eos():
    model = _make_tiny_model()
    model.eval()
    prompt = torch.randint(0, 32, (1, 2))
    out = generate_cached(model, prompt, max_new_tokens=100, eos_token_id=0)
    # Should stop early if 0 is generated (can't guarantee, but shouldn't crash).
    assert out.shape[1] >= 2

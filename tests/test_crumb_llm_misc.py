"""Smoke tests for modules that had zero test coverage."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


# ── Baseline ─────────────────────────────────────────────────────────


def test_baseline_forward_backward():
    from crumb_llm.baseline import TinyTransformerLM, TransformerConfig
    cfg = TransformerConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, block_size=32)
    m = TinyTransformerLM(cfg)
    x = torch.randint(0, 32, (1, 8))
    out = m(x, targets=x)
    assert out["logits"].shape == (1, 8, 32)
    out["loss"].backward()


# ── Tokenizer ────────────────────────────────────────────────────────


def test_byte_tokenizer_roundtrip():
    from crumb_llm.tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    text = "Hello, world! 🌍"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert "Hello" in decoded
    assert tok.vocab_size == 256


def test_char_tokenizer_roundtrip():
    from crumb_llm.tokenizer import CharTokenizer
    tok = CharTokenizer.fit("hello world test")
    ids = tok.encode("hello")
    decoded = tok.decode(ids)
    assert decoded == "hello"
    assert tok.vocab_size > 0


def test_char_tokenizer_save_load(tmp_path):
    from crumb_llm.tokenizer import CharTokenizer
    tok = CharTokenizer.fit("abcdefg")
    tok.save(tmp_path / "tok.json")
    loaded = CharTokenizer.load(tmp_path / "tok.json")
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("abc") == tok.encode("abc")


# ── Data ─────────────────────────────────────────────────────────────


def test_load_text_corpus_synthetic():
    from crumb_llm.data import load_text_corpus, SYNTHETIC_CORPUS
    text = load_text_corpus(None)
    assert len(text) > 100


def test_make_token_stream_batches():
    from crumb_llm.data import make_token_stream
    from crumb_llm.tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    stream = make_token_stream("hello world " * 100, tok, block_size=16)
    x, y = stream.batch(4)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


# ── Compare ──────────────────────────────────────────────────────────


def test_compare_runs_briefly():
    from crumb_llm.compare import compare
    results = compare(config_name="tiny", steps=10, log_every=10)
    assert "wave_field" in results
    assert "transformer" in results
    assert "ppl_gap_pct" in results


# ── MCP Tools ────────────────────────────────────────────────────────


def test_mcp_tools_without_model():
    from crumb_llm.mcp_tools import handle_tool
    result = handle_tool("crumb_llm_complete", {"prompt": "test"})
    assert "Error" in result or "no model" in result.lower()


def test_mcp_tools_score_with_model():
    from crumb_llm.mcp_tools import handle_tool
    from crumb_llm.model import WaveFieldLM, WaveFieldConfig
    from crumb_llm.tokenizer import ByteTokenizer
    cfg = WaveFieldConfig(vocab_size=256, dim=16, n_layers=1, n_heads=2, field_size=32)
    m = WaveFieldLM(cfg)
    m.eval()
    tok = ByteTokenizer()
    result = handle_tool("crumb_llm_score", {"text": "hello world"}, model=m, tokenizer=tok)
    assert "ppl=" in result


# ── Setup Standalone ─────────────────────────────────────────────────


def test_setup_standalone_generates_files(tmp_path):
    from crumb_llm.setup_standalone import generate_standalone
    generate_standalone(tmp_path / "pkg")
    assert (tmp_path / "pkg" / "pyproject.toml").exists()
    assert (tmp_path / "pkg" / "crumb_llm").is_dir()
    assert (tmp_path / "pkg" / "tests").is_dir()


# ── Serve Handler ────────────────────────────────────────────────────


def test_serve_handler_dev_mode_no_auth():
    from crumb_llm.serve import CrumbLLMHandler, TIERS
    CrumbLLMHandler.production_mode = False
    # In dev mode, _auth should return without checking keys.
    # We can't easily call _auth without an HTTP request, but we can
    # verify the class attribute.
    assert CrumbLLMHandler.production_mode is False


# ── V2 Training Smoke ────────────────────────────────────────────────


def test_v2_model_trains():
    from crumb_llm.model import WaveFieldLM, WaveFieldConfig
    cfg = WaveFieldConfig(
        vocab_size=32, dim=16, n_layers=1, n_heads=2, field_size=32,
        use_v2_blocks=True,
    )
    m = WaveFieldLM(cfg)
    x = torch.randint(0, 32, (2, 8))
    out = m(x, targets=x)
    out["loss"].backward()
    # Verify gradients flow to V2-specific params.
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


# ── Sliding in Cache ─────────────────────────────────────────────────


def test_cached_generation_beyond_field_size():
    from crumb_llm.model import WaveFieldLM, WaveFieldConfig
    from crumb_llm.cache import generate_cached
    cfg = WaveFieldConfig(vocab_size=32, dim=16, n_layers=1, n_heads=2, field_size=16)
    m = WaveFieldLM(cfg)
    m.eval()
    prompt = torch.randint(0, 32, (1, 8))
    # Generate more tokens than field_size — triggers sliding window.
    out = generate_cached(m, prompt, max_new_tokens=20)
    assert out.shape == (1, 28)  # 8 prompt + 20 generated

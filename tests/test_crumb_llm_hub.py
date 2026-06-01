"""Hub save/load roundtrip test."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crumb_wavelm.hub import save_for_hub, load_hub_model  # noqa: E402
from crumb_wavelm.model import WaveFieldLM, WaveFieldConfig  # noqa: E402
from crumb_wavelm.tokenizer import ByteTokenizer  # noqa: E402


def test_save_and_load_roundtrip(tmp_path):
    cfg = WaveFieldConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, field_size=32)
    model = WaveFieldLM(cfg)
    tok = ByteTokenizer()
    save_for_hub(model, tok, tmp_path / "hub_test", model_name="test-model")

    # Check files exist.
    assert (tmp_path / "hub_test" / "config.json").exists()
    assert (tmp_path / "hub_test" / "model.pt").exists()
    assert (tmp_path / "hub_test" / "tokenizer.json").exists()
    assert (tmp_path / "hub_test" / "README.md").exists()

    # Load back.
    loaded_model, loaded_tok = load_hub_model(tmp_path / "hub_test")
    assert loaded_model.cfg.dim == 16
    assert loaded_model.cfg.n_layers == 2
    assert loaded_tok.vocab_size == 256

    # Forward pass matches.
    x = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        orig = model(x)["logits"]
        loaded = loaded_model(x)["logits"]
    torch.testing.assert_close(orig, loaded)


def test_model_card_contains_metadata(tmp_path):
    cfg = WaveFieldConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, field_size=32)
    model = WaveFieldLM(cfg)
    save_for_hub(model, ByteTokenizer(), tmp_path / "card_test")
    card = (tmp_path / "card_test" / "README.md").read_text()
    assert "crumb-llm" in card
    assert "wave-field" in card
    assert "O(N log N)" in card.replace(" ", "").replace("-", "") or "O(N log N)" in card

"""End-to-end smoke: train tiny config for a handful of steps, then sample."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.sample import generate, load_checkpoint  # noqa: E402
from crumb_llm.train import load_config, train  # noqa: E402


def test_train_tiny_converges_a_bit(tmp_path):
    cfg = load_config("tiny")
    # Shrink for speed.
    cfg["dim"] = 32
    cfg["n_layers"] = 2
    cfg["n_heads"] = 2
    cfg["field_size"] = 64
    cfg["block_size"] = 64
    cfg["batch_size"] = 4

    metrics = train(
        config=cfg,
        data_path=None,           # uses examples/ or synthetic fallback
        steps=40,
        out_dir=tmp_path / "run",
        log_every=20,
        eval_every=0,
    )
    assert metrics["params"] > 0
    assert metrics["final_loss"] < 5.5  # below random for 256-class problem


def test_checkpoint_roundtrip_and_sample(tmp_path):
    cfg = load_config("tiny")
    cfg["dim"] = 32
    cfg["n_layers"] = 2
    cfg["n_heads"] = 2
    cfg["field_size"] = 64
    cfg["block_size"] = 64
    cfg["batch_size"] = 4
    train(
        config=cfg,
        data_path=None,
        steps=10,
        out_dir=tmp_path / "run",
        log_every=20,
        eval_every=0,
    )
    model, tok = load_checkpoint(tmp_path / "run")
    assert tok.vocab_size > 0
    text = generate(
        ckpt_dir=tmp_path / "run",
        prompt="The",
        max_new_tokens=5,
        seed=0,
    )
    assert isinstance(text, str)
    assert len(text) > 0

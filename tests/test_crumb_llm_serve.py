"""Serve module unit tests (handler logic, not HTTP)."""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.serve import CrumbLLMHandler  # noqa: E402
from crumb_llm.model import WaveFieldLM, WaveFieldConfig  # noqa: E402
from crumb_llm.tokenizer import ByteTokenizer  # noqa: E402


def test_handler_class_attributes_settable():
    cfg = WaveFieldConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, field_size=32)
    model = WaveFieldLM(cfg)
    model.eval()
    tok = ByteTokenizer()
    CrumbLLMHandler.model = model
    CrumbLLMHandler.tokenizer = tok
    CrumbLLMHandler.model_info = {"arch": "crumb_llm", "params": 100}
    assert CrumbLLMHandler.model is not None
    assert CrumbLLMHandler.tokenizer is not None

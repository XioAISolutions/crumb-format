"""Registry tests for standalone CrumbLLM checkpoints."""

import json
from pathlib import Path

from crumb_llm.registry import (
    download_model,
    find_model,
    list_models,
    register_local_model,
)


def test_register_and_download_local_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "ckpt.pt").write_bytes(b"checkpoint")
    (source / "tokenizer.json").write_text('{"type": "byte"}')

    registry = tmp_path / "registry.json"
    register_local_model(
        "local-tiny",
        source,
        aliases=["tiny-local"],
        registry_path=registry,
    )

    found = find_model("tiny-local", [registry])
    assert found["id"] == "local-tiny"
    assert found["source_dir"] == str(source.resolve())

    out = download_model("local-tiny", out_dir=tmp_path / "downloaded", registry=[registry])
    assert (out / "ckpt.pt").read_bytes() == b"checkpoint"
    assert (out / "tokenizer.json").exists()
    assert json.loads((out / "crumb_model.json").read_text())["id"] == "local-tiny"


def test_download_file_registry(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "version": 1,
        "models": [{
            "id": "file-model",
            "files": [{"path": "weights.bin", "local_path": "payload.bin"}],
        }],
    }))

    out = download_model("file-model", out_dir=tmp_path / "out", registry=[registry])
    assert (out / "weights.bin").read_bytes() == b"abc"


def test_list_models_marks_planned_builtin():
    models = {m["id"]: m for m in list_models()}
    assert "crumb-llm-tiny" in models
    assert models["crumb-llm-tiny"]["downloadable"] is False

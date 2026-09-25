#!/usr/bin/env python3
"""Prepare or publish a CrumbLLM checkpoint as a registry artifact.

Default flow:

    python scripts/publish_crumb_llm_model.py

This exports ``pretrained/crumb-llm-tiny`` to a HuggingFace-compatible
folder under ``dist/``, writes ``manifest.json`` with checksums, and writes a
registry JSON snippet. Add ``--upload`` only when ``huggingface_hub`` is
installed and you have a valid HuggingFace token configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    return {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_metrics(ckpt_dir: Path) -> dict:
    path = ckpt_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def export_hub(ckpt_dir: Path, out_dir: Path, model_id: str) -> None:
    from crumb_llm.hub import save_for_hub
    from crumb_llm.sample import load_checkpoint

    metrics = load_metrics(ckpt_dir)
    training_info = (
        "Tiny local checkpoint trained on the bundled crumb-format examples "
        f"corpus. Steps: {metrics.get('steps', 'unknown')}; "
        f"final loss: {metrics.get('final_loss', 'unknown')}; "
        f"parameters: {metrics.get('params', 'unknown')}."
    )
    model, tok = load_checkpoint(ckpt_dir)
    save_for_hub(model, tok, out_dir, model_name=model_id, training_info=training_info)


def _artifact_files(out_dir: Path, exclude: set[Path]) -> list[Path]:
    resolved_exclude = {p.resolve() for p in exclude}
    return [
        path
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.resolve() not in resolved_exclude
    ]


def write_manifest(
    ckpt_dir: Path,
    out_dir: Path,
    model_id: str,
    hf_repo: str,
    registry_out: Path,
) -> dict:
    files = [
        file_record(path, out_dir)
        for path in _artifact_files(out_dir, {out_dir / "manifest.json", registry_out})
    ]
    manifest = {
        "schema": "crumb-llm-model-manifest/v1",
        "model_id": model_id,
        "hf_repo": hf_repo,
        "source_checkpoint": str(ckpt_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": load_metrics(ckpt_dir),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def write_registry(out_dir: Path, registry_out: Path, model_id: str, hf_repo: str, revision: str) -> None:
    files = [
        {
            "path": record["path"],
            "sha256": record["sha256"],
        }
        for record in [
            file_record(path, out_dir)
            for path in _artifact_files(out_dir, {registry_out})
        ]
    ]
    registry = {
        "version": 1,
        "models": [
            {
                "id": model_id,
                "aliases": ["tiny"],
                "name": "Crumb LLM Tiny",
                "description": (
                    "Tiny char-level Wave-Field LM checkpoint trained on the "
                    "crumb-format examples corpus. Experimental demo model, "
                    "not a production-quality general LLM."
                ),
                "status": "downloadable",
                "format": "hub",
                "license": "MIT",
                "tags": ["wave-field", "tiny", "experimental"],
                "hf_repo": hf_repo,
                "revision": revision,
                "files": files,
            }
        ],
    }
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    registry_out.write_text(json.dumps(registry, indent=2, sort_keys=True))


def upload_folder(out_dir: Path, hf_repo: str, private: bool, commit_message: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install with:\n"
            "    python -m pip install huggingface_hub\n"
            "Then run this script again with --upload."
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo,
        repo_type="model",
        folder_path=str(out_dir),
        commit_message=commit_message,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare/publish a CrumbLLM model artifact.")
    ap.add_argument("--ckpt", default="pretrained/crumb-llm-tiny", help="Training checkpoint directory.")
    ap.add_argument("--out", default="dist/crumb-llm-tiny-hub", help="Hub-format output directory.")
    ap.add_argument("--model-id", default="crumb-llm-tiny")
    ap.add_argument("--hf-repo", default="XioAISolutions/crumb-llm-tiny")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--registry-out", default=None, help="Registry JSON output path.")
    ap.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub.")
    ap.add_argument("--private", action="store_true", help="Create/upload private HF repo.")
    ap.add_argument("--commit-message", default="publish crumb-llm tiny checkpoint")
    args = ap.parse_args(argv)

    ckpt_dir = Path(args.ckpt).expanduser()
    out_dir = Path(args.out).expanduser()
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    registry_out = Path(args.registry_out).expanduser() if args.registry_out else out_dir / "registry.json"
    if not registry_out.is_absolute():
        registry_out = ROOT / registry_out

    export_hub(ckpt_dir, out_dir, args.model_id)
    write_manifest(ckpt_dir, out_dir, args.model_id, args.hf_repo, registry_out)
    write_registry(out_dir, registry_out, args.model_id, args.hf_repo, args.revision)

    print(f"Hub artifact: {out_dir}")
    print(f"Registry JSON: {registry_out}")
    print("Files:")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rec = file_record(path, out_dir)
            print(f"  {rec['path']}  {rec['bytes']} bytes  sha256:{rec['sha256']}")

    if args.upload:
        upload_folder(out_dir, args.hf_repo, private=args.private, commit_message=args.commit_message)
        print(f"Uploaded to https://huggingface.co/{args.hf_repo}")
    else:
        print()
        print("Upload later with:")
        print("  python -m pip install huggingface_hub")
        print(f"  python {Path(__file__).as_posix()} --upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

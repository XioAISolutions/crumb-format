"""Model registry and checkpoint downloader for ``crumb-wavelm``.

The registry intentionally stays small and stdlib-only:

* built-in package registry: ``crumb_wavelm/models.json``
* user registry: ``~/.cache/crumb-wavelm/registry.json``
* optional env registries: ``CRUMB_LLM_REGISTRY`` (``os.pathsep`` separated)
* optional command registries: ``--registry PATH_OR_URL``

Entries can point at a local source directory, explicit files, or a
HuggingFace repo with file names. Downloading copies/fetches into a cache
directory and returns a checkpoint directory that ``crumb-wavelm generate`` can
consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PACKAGE_REGISTRY = Path(__file__).with_name("models.json")
DEFAULT_CACHE_ROOT = Path(os.environ.get("CRUMB_WAVELM_CACHE", Path.home() / ".cache" / "crumb-wavelm"))
DEFAULT_USER_REGISTRY = DEFAULT_CACHE_ROOT / "registry.json"


class RegistryError(RuntimeError):
    """Raised when a model registry entry cannot be resolved or downloaded."""


def _safe_model_dir(model_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id.strip())
    return safe.strip(".-") or "model"


def _read_text_source(source: str | Path) -> tuple[str, Path | None]:
    src = str(source)
    parsed = urllib.parse.urlparse(src)
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(src, timeout=30) as response:
            return response.read().decode("utf-8"), None
    if parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
    else:
        path = Path(src).expanduser()
    return path.read_text(encoding="utf-8"), path.parent


def _load_registry_source(source: str | Path) -> dict[str, dict[str, Any]]:
    text, base = _read_text_source(source)
    raw = json.loads(text)
    models = raw.get("models", [])
    if isinstance(models, dict):
        models = [{**spec, "id": model_id} for model_id, spec in models.items()]
    out: dict[str, dict[str, Any]] = {}
    for spec in models:
        model_id = spec.get("id")
        if not model_id:
            continue
        item = dict(spec)
        item["_registry_source"] = str(source)
        item["_registry_base"] = str(base) if base else None
        out[model_id] = item
    return out


def registry_sources(extra: list[str | Path] | None = None) -> list[str | Path]:
    sources: list[str | Path] = []
    if PACKAGE_REGISTRY.exists():
        sources.append(PACKAGE_REGISTRY)
    if DEFAULT_USER_REGISTRY.exists():
        sources.append(DEFAULT_USER_REGISTRY)
    env_sources = os.environ.get("CRUMB_LLM_REGISTRY", "")
    for part in env_sources.split(os.pathsep):
        if part.strip():
            sources.append(part.strip())
    if extra:
        sources.extend(extra)
    return sources


def load_registry(extra: list[str | Path] | None = None) -> dict[str, dict[str, Any]]:
    """Load and merge registries. Later sources override earlier ones."""
    merged: dict[str, dict[str, Any]] = {}
    for source in registry_sources(extra):
        merged.update(_load_registry_source(source))
    return merged


def find_model(model_id: str, extra: list[str | Path] | None = None) -> dict[str, Any]:
    registry = load_registry(extra)
    if model_id in registry:
        return registry[model_id]
    for spec in reversed(list(registry.values())):
        if model_id in spec.get("aliases", []):
            return spec
    raise RegistryError(f"unknown CrumbLLM model: {model_id!r}")


def list_models(extra: list[str | Path] | None = None, include_planned: bool = True) -> list[dict[str, Any]]:
    models = []
    for spec in load_registry(extra).values():
        if not include_planned and spec.get("status") == "planned":
            continue
        public = {k: v for k, v in spec.items() if not k.startswith("_")}
        public["downloadable"] = is_downloadable(spec)
        models.append(public)
    return sorted(models, key=lambda m: m.get("id", ""))


def is_downloadable(spec: dict[str, Any]) -> bool:
    if spec.get("source_dir"):
        return True
    if spec.get("files"):
        return True
    return False


def _resolve_local_path(value: str, base: str | None) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute() or not base:
        return p
    return Path(base) / p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_file(path: Path, expected_sha: str | None) -> None:
    if expected_sha and _sha256(path) != expected_sha:
        raise RegistryError(f"checksum mismatch for {path}")


def _download_url(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)


def _hf_url(repo: str, filename: str, revision: str = "main") -> str:
    quoted_file = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    return f"https://huggingface.co/{repo}/resolve/{revision}/{quoted_file}"


def _copy_source_dir(spec: dict[str, Any], target: Path, force: bool) -> None:
    base = spec.get("_registry_base")
    source_dir = _resolve_local_path(str(spec["source_dir"]), base)
    if not source_dir.exists():
        raise RegistryError(f"registered source_dir does not exist: {source_dir}")
    target.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        dest = target / item.name
        if dest.exists() and force:
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def _materialize_files(spec: dict[str, Any], target: Path, force: bool) -> None:
    base = spec.get("_registry_base")
    repo = spec.get("hf_repo")
    revision = spec.get("revision", "main")
    target.mkdir(parents=True, exist_ok=True)

    for file_spec in spec.get("files", []):
        rel = file_spec["path"]
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected_sha = file_spec.get("sha256")
        if dest.exists() and not force:
            _check_file(dest, expected_sha)
            continue
        url = file_spec.get("url")
        local_path = file_spec.get("local_path") or file_spec.get("source")
        optional = bool(file_spec.get("optional"))
        try:
            if local_path:
                src = _resolve_local_path(str(local_path), base)
                if not src.exists():
                    raise FileNotFoundError(src)
                shutil.copy2(src, dest)
            else:
                if not url:
                    if not repo:
                        raise RegistryError(f"file {rel!r} has no url/local_path/hf_repo")
                    url = _hf_url(str(repo), rel, revision=revision)
                _download_url(str(url), dest)
            _check_file(dest, expected_sha)
        except Exception:
            if optional:
                if dest.exists():
                    dest.unlink()
                continue
            raise


def download_model(
    model_id: str,
    out_dir: str | Path | None = None,
    *,
    registry: list[str | Path] | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download/copy a registered model and return the local checkpoint dir."""
    spec = find_model(model_id, registry)
    if not is_downloadable(spec):
        raise RegistryError(
            f"model {model_id!r} is listed but has no downloadable files yet. "
            "Register a local checkpoint with `crumb-wavelm register ...` or add URLs."
        )

    root = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_ROOT / "models"
    target = Path(out_dir).expanduser() if out_dir else root / _safe_model_dir(spec["id"])
    if spec.get("source_dir"):
        _copy_source_dir(spec, target, force=force)
    if spec.get("files"):
        _materialize_files(spec, target, force=force)

    meta = {k: v for k, v in spec.items() if not k.startswith("_")}
    meta["local_path"] = str(target)
    (target / "crumb_model.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return target


def register_local_model(
    model_id: str,
    checkpoint_dir: str | Path,
    *,
    name: str | None = None,
    description: str | None = None,
    registry_path: str | Path | None = None,
    aliases: list[str] | None = None,
) -> Path:
    """Add/update a local checkpoint entry in the user registry."""
    ckpt = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt.exists():
        raise RegistryError(f"checkpoint directory does not exist: {ckpt}")
    if not (ckpt / "ckpt.pt").exists() and not (ckpt / "config.json").exists():
        raise RegistryError(
            f"{ckpt} is not a CrumbLLM checkpoint or Hub-format directory "
            "(expected ckpt.pt or config.json)"
        )

    registry_file = Path(registry_path).expanduser() if registry_path else DEFAULT_USER_REGISTRY
    if registry_file.exists():
        raw = json.loads(registry_file.read_text(encoding="utf-8"))
    else:
        raw = {"version": 1, "models": []}
    models = raw.get("models", [])
    if isinstance(models, dict):
        models = [{**spec, "id": mid} for mid, spec in models.items()]
    models = [m for m in models if m.get("id") != model_id]
    models.append({
        "id": model_id,
        "aliases": aliases or [],
        "name": name or model_id,
        "description": description or "Local CrumbLLM checkpoint.",
        "status": "local",
        "format": "checkpoint",
        "source_dir": str(ckpt),
        "license": "local",
    })
    raw["models"] = sorted(models, key=lambda m: m.get("id", ""))
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(raw, indent=2, sort_keys=True))
    return registry_file

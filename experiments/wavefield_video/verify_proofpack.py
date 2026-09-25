#!/usr/bin/env python3
"""Verify a proofpack's complete file inventory using only the standard library.

This checks integrity against manifest.json, not authenticity of the manifest.
It never executes reproduction commands or deserializes checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

FORMAT = "wavefield-proofpack"
SCHEMA_VERSION = 1
ROLES = {"result", "checkpoint", "config", "artifact", "verifier", "instructions"}


class ProofpackError(ValueError):
    """The bundle or its manifest is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProofpackError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ProofpackError(f"non-finite JSON value: {value}")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_object_pairs,
                          parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProofpackError(f"cannot read JSON {path}: {exc}") from exc


def _safe_relative(value) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ProofpackError(f"invalid manifest path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or ":" in value:
        raise ProofpackError(f"unsafe or non-canonical manifest path: {value!r}")
    if value == "." or value == "manifest.json":
        raise ProofpackError(f"reserved manifest path: {value!r}")
    return value


def inventory(root: Path) -> dict[str, Path]:
    """Inventory ordinary files; symlinks and special files are never followed."""
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ProofpackError(f"symlink is not allowed: {relative}")
        if path.is_file():
            files[relative] = path
        elif not path.is_dir():
            raise ProofpackError(f"non-regular file is not allowed: {relative}")
    return files


def validate_manifest(manifest) -> list[dict]:
    if not isinstance(manifest, dict):
        raise ProofpackError("manifest must be a JSON object")
    if manifest.get("format") != FORMAT:
        raise ProofpackError("unrecognized manifest format")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ProofpackError("unsupported manifest schema_version")
    reproduction = manifest.get("reproduction")
    if not isinstance(reproduction, dict):
        raise ProofpackError("manifest reproduction must be an object")
    commands = reproduction.get("commands")
    if not isinstance(commands, list) or not commands or any(
        not isinstance(command, str) or not command.strip() or "\x00" in command
        for command in commands
    ):
        raise ProofpackError("manifest reproduction.commands must contain nonempty strings")
    cwd = reproduction.get("cwd")
    if not isinstance(cwd, str) or not cwd.strip() or "\x00" in cwd:
        raise ProofpackError("manifest reproduction.cwd must be a nonempty string")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ProofpackError("manifest files must be a nonempty list")
    paths = set()
    roles = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes", "role"}:
            raise ProofpackError("file entries require path, sha256, size_bytes and role")
        relative = _safe_relative(entry["path"])
        if relative in paths:
            raise ProofpackError(f"duplicate manifest path: {relative}")
        paths.add(relative)
        digest = entry["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ProofpackError(f"invalid SHA-256 for {relative}")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise ProofpackError(f"invalid size_bytes for {relative}")
        role = entry["role"]
        if not isinstance(role, str) or role not in ROLES:
            raise ProofpackError(f"invalid role for {relative}")
        roles.add(role)
    required_roles = {"result", "checkpoint", "config", "verifier", "instructions"}
    if not required_roles <= roles:
        raise ProofpackError(f"missing required artifact roles: {sorted(required_roles - roles)}")
    for relative, role in (("VERIFY.md", "instructions"), ("verify_proofpack.py", "verifier")):
        if not any(entry["path"] == relative and entry["role"] == role for entry in entries):
            raise ProofpackError(f"missing required {relative} entry")
    return entries


def verify_proofpack(bundle: str | Path) -> dict:
    """Rehash every listed file and reject missing, extra, or malformed contents."""
    root = Path(bundle).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ProofpackError(f"bundle must be a real directory: {root}")
    files = inventory(root)
    if "manifest.json" not in files:
        raise ProofpackError("missing manifest.json")
    manifest = read_json(files.pop("manifest.json"))
    entries = validate_manifest(manifest)
    expected = {entry["path"] for entry in entries}
    actual = set(files)
    if expected != actual:
        raise ProofpackError(f"inventory mismatch: missing={sorted(expected - actual)}, "
                             f"extra={sorted(actual - expected)}")
    failures = []
    for entry in entries:
        path = files[entry["path"]]
        if path.stat().st_size != entry["size_bytes"]:
            failures.append(f"size mismatch: {entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {entry['path']}")
    if failures:
        raise ProofpackError("; ".join(failures))
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", default=".", help="proofpack directory (default: .)")
    args = parser.parse_args(argv)
    try:
        manifest = verify_proofpack(args.bundle)
    except (ProofpackError, OSError) as exc:
        print(f"FAIL proofpack: {exc}", file=sys.stderr)
        return 1
    print(f"PASS proofpack: {len(manifest['files'])} files rehashed with SHA-256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

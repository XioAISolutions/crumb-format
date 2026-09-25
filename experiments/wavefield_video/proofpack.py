#!/usr/bin/env python3
"""Bundle a completed run, explicit configuration, and exact reproduction commands.

Example (run from the training source directory):
    python3 proofpack.py runs/example proofs/example --config config.json \
        --repro-command 'python3 train_compare.py --kind wave --out runs/example'

Existing output directories are never overwritten. The entire run directory is
copied, with a SHA-256 manifest, VERIFY.md, and a standard-library-only verifier.
Checkpoints are copied as opaque bytes; commands are recorded, never executed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import sys
import tempfile

from verify_proofpack import (FORMAT, SCHEMA_VERSION, ProofpackError, inventory,
                             read_json, sha256_file, verify_proofpack)

CONFIG_NAMES = {"config.json", "config.yaml", "config.yml", "config.toml", "config.ini"}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def _is_result(path: Path) -> bool:
    return path.suffix.lower() == ".json" and (
        path.stem in {"result", "results"} or path.stem.startswith("result_")
    )


def _instructions(commands: list[str], cwd: str, config_paths: list[str]) -> str:
    # Longer fences preserve commands containing their own Markdown backticks.
    fence = "`" * max(3, max((len(part) + 1 for command in commands
                              for part in re.findall(r"`+", command)), default=3))
    return (
        "# Verify and reproduce this run\n\n"
        "From this extracted bundle directory, verify every file with Python 3.10+:\n\n"
        "```sh\npython3 verify_proofpack.py .\n```\n\n"
        "A successful check prints `PASS proofpack` and exits 0. A changed, missing, "
        "extra, or invalid file fails with exit 1. No training commands or checkpoint "
        "contents are executed by verification.\n\n"
        "## Exact recorded reproduction commands\n\n"
        "The producer supplied the commands below. Run them in the original training "
        "source checkout with its training dependencies installed. The recorded "
        "working directory is included; on another machine, relocate that checkout "
        "and adjust only its directory prefix. Commands may write training outputs.\n\n"
        f"{fence}sh\ncd -- {shlex.quote(cwd)}\n" + "\n".join(commands) + f"\n{fence}\n\n"
        "Bundled configuration: " + ", ".join(f"`{path}`" for path in config_paths) + ".\n\n"
        "The entire input run is under `run/`. Results, checkpoints, configurations, "
        "these instructions, and this standalone verifier are hashed in `manifest.json`. "
        "The manifest cannot hash itself: keep its SHA-256 from the producer separately "
        "if you need an external integrity anchor. Hashes establish byte integrity "
        "against this manifest, not authorship or numerical reproducibility. Source "
        "code and training dependencies are not implicitly bundled.\n"
    )


def create_proofpack(
    run_dir: str | Path,
    output: str | Path,
    *,
    config: str | Path | None = None,
    repro_commands: list[str] | None = None,
    repro_cwd: str | Path | None = None,
) -> Path:
    """Create a new proofpack directory and return its absolute path.

    A run must contain result*.json and a .pt/.pth/.ckpt/.safetensors checkpoint.
    Config defaults to unambiguously named config files anywhere inside the run.
    Commands must be explicit, or stored as repro_commands/repro_command in one
    JSON config. They are not reconstructed from incomplete result metrics.
    """
    source_arg = Path(run_dir).expanduser()
    if source_arg.is_symlink() or not source_arg.is_dir():
        raise ProofpackError(f"run_dir must be a real directory: {source_arg}")
    source = source_arg.resolve()
    target = Path(output).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise ProofpackError(f"output already exists: {target}")
    if target.resolve().is_relative_to(source):
        raise ProofpackError("output must be outside the input run directory")
    run_files = inventory(source)
    result_files = [path for path in run_files.values() if _is_result(path)]
    if not result_files:
        raise ProofpackError("run needs a result.json, results.json, or result_*.json file")
    for path in result_files:
        if not isinstance(read_json(path), dict):
            raise ProofpackError(f"result JSON must be an object: {path}")
    if not any(path.suffix.lower() in CHECKPOINT_SUFFIXES for path in run_files.values()):
        raise ProofpackError("run needs a .pt, .pth, .ckpt, or .safetensors checkpoint")
    if config is None:
        configs = [path for path in run_files.values() if path.name.lower() in CONFIG_NAMES]
        if not configs:
            raise ProofpackError("run has no config file; supply --config with the exact run configuration")
    else:
        config_arg = Path(config).expanduser()
        if config_arg.is_symlink() or not config_arg.is_file():
            raise ProofpackError(f"config must be a regular file: {config_arg}")
        configs = [config_arg.resolve()]
    config_data = []
    for path in configs:
        if path.suffix.lower() == ".json":
            data = read_json(path)
            if not isinstance(data, dict):
                raise ProofpackError(f"config JSON must be an object: {path}")
            config_data.append(data)
    commands = repro_commands
    if commands is None:
        recorded = [data for data in config_data
                    if "repro_commands" in data or "repro_command" in data]
        if len(recorded) == 1:
            value = recorded[0].get("repro_commands", recorded[0].get("repro_command"))
            commands = [value] if isinstance(value, str) else value
    if not isinstance(commands, list) or not commands or any(
        not isinstance(command, str) or not command.strip() or "\x00" in command
        for command in commands
    ):
        raise ProofpackError("provide exact --repro-command(s), or config JSON repro_commands")
    cwd = str(Path.cwd() if repro_cwd is None else repro_cwd)
    if not cwd.strip() or "\x00" in cwd:
        raise ProofpackError("repro_cwd must be a nonempty path")
    verifier = Path(__file__).resolve().with_name("verify_proofpack.py")
    if not verifier.is_file():
        raise ProofpackError(f"standalone verifier is missing: {verifier}")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        roles = {}
        for relative, path in run_files.items():
            destination = stage / "run" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            role = "result" if _is_result(path) else "artifact"
            if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                role = "checkpoint"
            roles[f"run/{relative}"] = role
        config_paths = []
        for path in configs:
            if path.is_relative_to(source):
                relative = "run/" + path.relative_to(source).as_posix()
            else:
                relative = "config/" + path.name
                (stage / "config").mkdir(exist_ok=True)
                shutil.copyfile(path, stage / relative)
            roles[relative] = "config"
            config_paths.append(relative)
        shutil.copyfile(verifier, stage / "verify_proofpack.py")
        roles["verify_proofpack.py"] = "verifier"
        (stage / "VERIFY.md").write_text(_instructions(commands, cwd, config_paths), encoding="utf-8")
        roles["VERIFY.md"] = "instructions"
        entries = [
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
             "role": roles[relative]}
            for relative, path in inventory(stage).items()
        ]
        manifest = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(source),
            "producer_environment": {"python": platform.python_version(), "platform": platform.platform()},
            "reproduction": {"cwd": cwd, "commands": commands},
            "files": entries,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")
        verify_proofpack(stage)
        if target.exists() or target.is_symlink():
            raise ProofpackError(f"output appeared while building: {target}")
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="completed run directory")
    parser.add_argument("output", help="new output directory, outside run_dir")
    parser.add_argument("--config", help="exact run configuration; otherwise discover config.* in run_dir")
    parser.add_argument("--repro-command", action="append", help="exact shell command; repeat for multiple commands")
    parser.add_argument("--repro-cwd", help="reproduction working directory (default: current directory)")
    args = parser.parse_args(argv)
    try:
        target = create_proofpack(args.run_dir, args.output, config=args.config,
                                 repro_commands=args.repro_command, repro_cwd=args.repro_cwd)
    except (ProofpackError, OSError) as exc:
        print(f"FAIL proofpack: {exc}", file=sys.stderr)
        return 1
    print(f"PASS proofpack: {target}")
    print(f"manifest.json SHA-256: {sha256_file(target / 'manifest.json')}")
    print(f"Verify: python3 {shlex.quote(str(target / 'verify_proofpack.py'))} {shlex.quote(str(target))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

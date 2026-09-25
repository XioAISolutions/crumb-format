#!/usr/bin/env bash
# Audit the Python sources and the original v3/v4 CPU smokes without touching
# their existing outputs. Usage: PY=/path/to/python bash audit_wavefield.sh [DIR]
# DIR (or AUDIT_OUT) must be new; by default a retained temporary dir is used.
set -euo pipefail
if [[ $# -gt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: PY=python3 bash audit_wavefield.sh [NEW_AUDIT_DIR]"
    echo "Also accepts AUDIT_OUT; writes manifest.json, logs/, source/, and pycache/."
    [[ $# -le 1 ]] && exit 0 || exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
exec "$PY" - "$ROOT" "${1:-${AUDIT_OUT:-}}" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time

root = Path(sys.argv[1]).resolve()
try:
    if sys.argv[2]:
        output = Path(sys.argv[2]).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="wavefield-audit-r5-")).resolve()
except OSError as exc:
    print(f"FAIL audit_setup: {exc}", file=sys.stderr)
    sys.exit(2)

source = output / "source"
logs = output / "logs"
source.mkdir()
logs.mkdir()
manifest = {
    "schema_version": 1,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_root": str(root),
    "audit_dir": str(output),
    "python": sys.executable,
    "python_version": sys.version,
    "device": "cpu",
    "checks": [],
    "sources": [],
}


def record(name, code, log, **extra):
    item = {"name": name, "status": "PASS" if code == 0 else "FAIL",
            "exit_code": code, "log": str(log.relative_to(output)), **extra}
    manifest["checks"].append(item)
    print(f"{item['status']:4s} {name} (exit={code}) -> {log}", flush=True)


print(f"Audit directory: {output}", flush=True)
# Snapshot before compiling/running: both smokes cd to their own directory, and
# v4's inline assertions use literal smoke_v4 paths regardless of OUT.
setup_log = logs / "snapshot.log"
try:
    ignored = {".git", "__pycache__", ".venv", "venv"}
    python_files = sorted(
        p for p in root.rglob("*.py")
        if p.is_file() and not (ignored & set(p.relative_to(root).parts))
        and output not in p.parents
    )
    if not python_files:
        raise RuntimeError("no Python sources found")
    files = python_files + [root / "run_smoke_v3.sh", root / "run_smoke_v4.sh"]
    for path in files:
        relative = path.relative_to(root)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest["sources"].append({
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    setup_log.write_text(f"Copied {len(python_files)} Python sources and both smoke scripts.\n")
    record("snapshot", 0, setup_log)
except Exception as exc:
    setup_log.write_text(f"{type(exc).__name__}: {exc}\n")
    record("snapshot", 1, setup_log)
    python_files = []

compile_log = logs / "py_compile.log"
compile_failures = 0
with compile_log.open("w") as log:
    if not python_files:
        log.write("FAIL: source snapshot unavailable\n")
        compile_failures += 1
    for original in python_files:
        relative = original.relative_to(root)
        try:
            cache = output / "pycache" / relative.with_suffix(".pyc")
            cache.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(str(source / relative), cfile=str(cache), doraise=True)
            log.write(f"PASS {relative.as_posix()}\n")
        except Exception as exc:
            compile_failures += 1
            log.write(f"FAIL {relative.as_posix()}: {exc}\n")
record("py_compile", int(compile_failures > 0), compile_log,
       files=len(python_files), failures=compile_failures)

environment = os.environ.copy()
environment.update({"PY": sys.executable, "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
environment.setdefault("OMP_NUM_THREADS", "2")
environment.setdefault("MKL_NUM_THREADS", "2")
manifest["environment"] = {key: environment[key] for key in (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
for version in (3, 4):
    name = f"smoke_v{version}"
    log_path = logs / f"{name}.log"
    command = ["bash", "-o", "pipefail", f"run_{name}.sh"]
    environment["OUT"] = name
    started = time.monotonic()
    print(f"RUN  {name} -> {log_path}", flush=True)
    with log_path.open("w") as log:
        try:
            completed = subprocess.run(command, cwd=source, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
            code = completed.returncode
        except OSError as exc:
            log.write(f"{type(exc).__name__}: {exc}\n")
            code = 1
    # sanity_check.py reports failed assertions in text but exits successfully.
    # Propagate those failures in addition to shell/pipeline exit statuses.
    reported_failure = bool(re.search(r"SANITY FAILURES-PRESENT|^.*\sFAIL\s*$",
                                     log_path.read_text(errors="replace"), re.MULTILINE))
    if reported_failure and code == 0:
        code = 1
    record(name, code, log_path, command=command, cwd="source", out=name,
           seconds=round(time.monotonic() - started, 3),
           reported_failure=reported_failure)

manifest["status"] = "PASS" if all(
    check["status"] == "PASS" for check in manifest["checks"]
) else "FAIL"
manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest_path = output / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print("\nPASS/FAIL manifest", flush=True)
for check in manifest["checks"]:
    print(f"{check['status']:4s} {check['name']}", flush=True)
print(f"{manifest['status']} overall -> {manifest_path}", flush=True)
sys.exit(0 if manifest["status"] == "PASS" else 1)
PY

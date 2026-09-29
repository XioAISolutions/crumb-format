#!/usr/bin/env python3
"""One fail-closed top-up of an existing gpuq2 queue (stdlib, POSIX only).

Usage::

    python gpuq_keeper.py --queue /path/to/gpu_queue \
        --manifest /path/to/keeper.json --max-queue 8

The JSON manifest is an explicit allowlist of basenames in queue/templates::

    {
      "tpl_longlive.sh": "/path/to/runs_longlive_w16/status.txt",
      "tpl_once.sh": {"enabled": true},
      "tpl_disabled.sh": {"enabled": false}
    }

A string means enabled with that status path; object entries require a boolean
``enabled`` and optionally ``status_path`` (string or null). Relative status
paths are relative to the manifest, never inferred from shell text. Unlisted
templates are disabled. All four queue directories (templates/pending/done/
failed) must already exist; missing archives are errors, not empty history.

Status uses the first whitespace-delimited token. DONE and RUNNING block
enqueue, including edited templates. Missing status permits a first
attempt; empty/unknown/unreadable status is an error. Only an explicit SLICED
status permits repeating a successful digest. An identical failed digest is
always blocked, even with SLICED. FAILED status grants no resume permission,
but edited content can retry: failure identity comes from the archived bytes.

SHA-256 covers exact script bytes, including legacy jobs in all archives.
The conductor keeps running jobs in pending, so no second GPU scheduler or
wrapper is needed. Scripts are syntax-checked with bash -n, never executed by
the keeper; queued bytes and therefore their exit behavior are unchanged.
The bash selected from PATH must diagnose unterminated heredocs; validators
that silently accept them (such as Bash 3.2) fail closed.

An exclusive flock covers validation, inspection and publication. A complete,
fsynced temporary file is hard-linked into pending without overwriting a path.
Only our own temporary file is removed; existing jobs are never changed. Read
errors abort the tick. Publication errors can leave earlier complete jobs from
this tick; they are deliberately not rolled back and the next tick dedups them.

The cap counts every pending *.sh, including running and hidden jobs. It bounds
this keeper and other instances using its lock; unrelated producers must use
the same lock to share that guarantee. The existing conductor is untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid


class KeeperError(Exception):
    """Unsafe or unavailable input; do not add further jobs this tick."""


def _read_regular(path: Path) -> bytes:
    # O_NONBLOCK prevents a FIFO in an input slot from hanging the keeper.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise KeeperError(f"not a regular file: {path}")
        body = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise KeeperError(f"file changed while reading: {path}")
    return body


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise KeeperError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _manifest(path: Path) -> dict[str, tuple[bool, Path | None]]:
    document = json.loads(_read_regular(path), object_pairs_hook=_unique_object)
    if not isinstance(document, dict):
        raise KeeperError("manifest must be a template-to-status object")
    result = {}
    for name, entry in document.items():
        if not re.fullmatch(r"tpl_[A-Za-z0-9_.-]+\.sh", name):
            raise KeeperError(f"invalid template basename: {name!r}")
        if isinstance(entry, str):
            entry = {"enabled": True, "status_path": entry}
        if (not isinstance(entry, dict)
                or set(entry) - {"enabled", "status_path"}
                or type(entry.get("enabled")) is not bool):
            raise KeeperError(f"invalid manifest entry: {name}")
        status_path = entry.get("status_path")
        if status_path is not None:
            if not isinstance(status_path, str) or not status_path.strip():
                raise KeeperError(f"invalid status_path: {name}")
            status_path = Path(status_path)
            if not status_path.is_absolute():
                status_path = path.parent / status_path
        result[name] = (entry["enabled"], status_path)
    return result


def _status(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        words = _read_regular(path).decode("utf-8").split()
    except FileNotFoundError:
        return None
    if not words or words[0] not in {"DONE", "SLICED", "RUNNING", "FAILED"}:
        raise KeeperError(f"empty or unknown status: {path}")
    return words[0]


def _validate_script(name: str, body: bytes) -> None:
    text = body.decode("utf-8")
    if "\0" in text or not any(
        line.strip() and not line.lstrip().startswith("#") for line in text.splitlines()
    ):
        raise KeeperError(f"empty or invalid script: {name}")
    # Do not inherit BASH_ENV, exported functions, or shell option variables.
    bash = shutil.which("bash")  # Match the conductor's interpreter selection.
    if bash is None:
        raise KeeperError("bash is required for script validation")
    # Bash 3.2 silently accepts incomplete heredocs even with -n. Check this
    # capability instead of accepting a script with an inconclusive validator.
    probe = subprocess.run(
        [bash, "--noprofile", "--norc", "-n"],
        input=b"cat <<GPUQ_KEEPER_UNCLOSED\n", capture_output=True, timeout=10,
        env={"PATH": os.defpath, "LC_ALL": "C"},
    )
    if not probe.returncode and not probe.stderr:
        raise KeeperError("bash validator cannot detect incomplete heredocs")
    check = subprocess.run(
        [bash, "--noprofile", "--norc", "-n"], input=body,
        capture_output=True, timeout=10, env={"PATH": os.defpath, "LC_ALL": "C"},
    )
    # bash reports an unterminated heredoc as a warning with rc=0.
    if check.returncode or check.stderr:
        raise KeeperError(f"invalid bash script: {name}: "
                          f"{check.stderr.decode('utf-8', errors='replace').strip()}")


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _snapshot(queue: Path) -> tuple[int, set[str], set[str], set[str]]:
    hashes = {}
    count = 0
    # The conductor only moves pending -> failed/done. Inspect pending FIRST:
    # a move either leaves its digest here, is seen in an archive, or causes
    # a read error that aborts. Never ignore a disappearing pending entry.
    for folder in ("pending", "failed", "done"):
        paths = sorted(p for p in (queue / folder).iterdir() if p.name.endswith(".sh"))
        hashes[folder] = {_digest(_read_regular(p)) for p in paths}
        if folder == "pending":
            count = len(paths)
    return count, hashes["pending"], hashes["failed"], hashes["done"]


def _publish(queue: Path, body: bytes, digest: str) -> str:
    # Stage in pending for same-filesystem link atomicity, but NEVER use .sh:
    # even a hidden *.sh would be visible to the conductor's Path.glob.
    fd, temporary = tempfile.mkstemp(prefix=".keeper-", suffix=".tmp", dir=queue / "pending")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        name = f"gpuq_job_9k_{digest}_{uuid.uuid4().hex}.sh"
        os.link(temporary, queue / "pending" / name)
        directory_fd = os.open(queue / "pending", os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return name
    finally:
        os.unlink(temporary)


def run_keeper(queue: Path, manifest: Path, max_queue: int = 8) -> dict:
    """Top up once; return enqueued basenames and skip reasons, or raise KeeperError."""
    if type(max_queue) is not int or max_queue <= 0:
        raise KeeperError("max_queue must be a positive integer")
    queue, manifest = Path(queue), Path(manifest)
    try:
        for directory in (queue, *(queue / n for n in ("templates", "pending", "done", "failed"))):
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise KeeperError(f"not a real queue directory: {directory}")
        lock_fd = os.open(queue / ".keeper.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
                          | os.O_NONBLOCK, 0o600)
        with os.fdopen(lock_fd, "r+") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise KeeperError("keeper lock is not a regular file")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise KeeperError("keeper lock is held by another process") from exc
            entries = _manifest(manifest)
            skipped, candidates = {}, []
            # Validate all enabled inputs before exposing any new job.
            for name, (enabled, status_path) in sorted(entries.items()):
                if not enabled:
                    skipped[name] = "disabled"
                    continue
                status = _status(status_path)
                if status in {"DONE", "RUNNING"}:
                    skipped[name] = f"status-{status.lower()}"
                    continue
                body = _read_regular(queue / "templates" / name)
                _validate_script(name, body)
                candidates.append((name, body, _digest(body), status_path))
            count, pending, failed, done = _snapshot(queue)
            enqueued = []
            # Remember everything observed during this tick, even if a fast job
            # moves out of pending between publications.
            observed_pending = set(pending)
            for name, body, digest, status_path in candidates:
                count, pending, new_failed, new_done = _snapshot(queue)
                status = _status(status_path)
                observed_pending.update(pending)
                failed.update(new_failed)
                done.update(new_done)
                if status in {"DONE", "RUNNING"}:
                    skipped[name] = f"status-{status.lower()}"
                elif digest in failed:
                    skipped[name] = "identical-failed"
                elif digest in observed_pending:
                    skipped[name] = "identical-pending"
                elif digest in done and status != "SLICED":
                    skipped[name] = "identical-done-without-sliced"
                elif count >= max_queue:
                    skipped[name] = "queue-full"
                else:
                    enqueued.append(_publish(queue, body, digest))
                    observed_pending.add(digest)
                    count += 1
            return {"enqueued": enqueued, "skipped": skipped, "pending": count}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise KeeperError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-queue", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        result = run_keeper(args.queue, args.manifest, args.max_queue)
    except KeeperError as exc:
        print(f"gpuq-keeper: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

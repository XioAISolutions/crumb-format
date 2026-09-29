"""Local keeper contract tests; no GPU, SSH, or conductor process required."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).with_name("gpuq_keeper.py")
BODY = b"#!/bin/bash\nprintf 'work\\n'\n"


@pytest.fixture
def keeper():
    spec = importlib.util.spec_from_file_location("gpuq_keeper", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def queue(tmp_path):
    root = tmp_path / "queue"
    for name in ("pending", "done", "failed", "templates"):
        (root / name).mkdir(parents=True)
    return root


def configure(queue, body=BODY, *, status=None, enabled=True, name="tpl_work.sh"):
    (queue / "templates" / name).write_bytes(body)
    entry = {"enabled": enabled}
    if status is not None:
        entry["status_path"] = status
    manifest = queue / "manifest.json"
    manifest.write_text(json.dumps({name: entry}))
    return manifest


def jobs(queue, folder="pending"):
    return sorted((queue / folder).glob("*.sh"))


def invoke(queue, manifest, cap=8):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--queue", str(queue),
         "--manifest", str(manifest), "--max-queue", str(cap)],
        capture_output=True, text=True, timeout=15,
    )


def test_enqueues_exact_bytes_without_executing_and_then_deduplicates(keeper, queue):
    marker = queue / "should-not-exist"
    body = f"#!/bin/bash\ntouch '{marker}'\nexit 23\n".encode()
    manifest = configure(queue, body)
    first = keeper.run_keeper(queue, manifest)
    assert len(first["enqueued"]) == 1
    assert jobs(queue)[0].read_bytes() == body
    assert not marker.exists()
    second = keeper.run_keeper(queue, manifest)
    assert second["enqueued"] == []
    assert len(jobs(queue)) == 1
    assert subprocess.run(["bash", str(jobs(queue)[0])]).returncode == 23


@pytest.mark.parametrize("folder", ["pending", "failed"])
def test_content_dedup_uses_legacy_names_and_preserves_jobs(keeper, queue, folder):
    manifest = configure(queue)
    existing = queue / folder / "legacy-name.sh"
    existing.write_bytes(BODY)
    before = (existing.read_bytes(), existing.stat().st_ino)
    assert keeper.run_keeper(queue, manifest)["enqueued"] == []
    assert (existing.read_bytes(), existing.stat().st_ino) == before
    assert jobs(queue) == ([existing] if folder == "pending" else [])


def test_running_job_still_pending_blocks_duplicate(keeper, queue):
    manifest = configure(queue)
    (queue / "pending" / "running.sh").write_bytes(BODY)
    (queue / "state").mkdir()
    summary = queue / "state" / "summary.json"
    summary.write_text(json.dumps({"current": {"job": "running.sh"}}))
    before = summary.read_bytes()
    assert keeper.run_keeper(queue, manifest)["enqueued"] == []
    assert summary.read_bytes() == before


def test_changed_failed_template_can_retry_but_original_cannot(keeper, queue):
    manifest = configure(queue)
    (queue / "failed" / "failed-attempt.sh").write_bytes(BODY)
    assert not keeper.run_keeper(queue, manifest)["enqueued"]
    changed = BODY + b"# repaired configuration\n"
    (queue / "templates" / "tpl_work.sh").write_bytes(changed)
    assert len(keeper.run_keeper(queue, manifest)["enqueued"]) == 1
    assert jobs(queue)[0].read_bytes() == changed
    jobs(queue)[0].rename(queue / "failed" / "second-attempt.sh")
    assert not keeper.run_keeper(queue, manifest)["enqueued"]


def test_duplicate_template_bodies_only_enqueue_once(keeper, queue):
    manifest = configure(queue)
    (queue / "templates" / "tpl_alias.sh").write_bytes(BODY)
    manifest.write_text(json.dumps({
        "tpl_work.sh": {"enabled": True}, "tpl_alias.sh": {"enabled": True},
    }))
    assert len(keeper.run_keeper(queue, manifest)["enqueued"]) == 1


@pytest.mark.parametrize("status", ["DONE", "DONE 8/8", "RUNNING"])
def test_status_blocks_even_with_changed_content(keeper, queue, status):
    manifest = configure(queue, status="status.txt")
    (queue / "status.txt").write_text(status + "\n")
    (queue / "done" / "old.sh").write_bytes(BODY + b"# old\n")
    assert not keeper.run_keeper(queue, manifest)["enqueued"]


def test_changed_failure_can_retry_with_persisted_failed_status(keeper, queue):
    manifest = configure(queue, status="status.txt")
    (queue / "status.txt").write_text("FAILED generation\n")
    (queue / "failed" / "old-failure.sh").write_bytes(BODY)
    assert not keeper.run_keeper(queue, manifest)["enqueued"]
    (queue / "templates" / "tpl_work.sh").write_bytes(BODY + b"# repair\n")
    assert len(keeper.run_keeper(queue, manifest)["enqueued"]) == 1


def test_success_needs_explicit_sliced_manifest_to_resume(keeper, queue):
    manifest = configure(queue)
    (queue / "done" / "previous-slice.sh").write_bytes(BODY)
    (queue / "status.txt").write_text("SLICED 1/8\n")
    assert not keeper.run_keeper(queue, manifest)["enqueued"]
    manifest.write_text(json.dumps({"tpl_work.sh": "status.txt"}))
    assert len(keeper.run_keeper(queue, manifest)["enqueued"]) == 1
    assert not keeper.run_keeper(queue, manifest)["enqueued"]
    jobs(queue)[0].rename(queue / "done" / "second-slice.sh")
    (queue / "status.txt").write_text("DONE 8/8\n")
    assert not keeper.run_keeper(queue, manifest)["enqueued"]


def test_failed_digest_vetoes_stale_sliced_status(keeper, queue):
    manifest = configure(queue, status="status.txt")
    (queue / "status.txt").write_text("SLICED\n")
    (queue / "failed" / "failed-slice.sh").write_bytes(BODY)
    assert not keeper.run_keeper(queue, manifest)["enqueued"]


def test_missing_status_only_allows_first_attempt(keeper, queue):
    manifest = configure(queue, status="not-created-yet.txt")
    assert len(keeper.run_keeper(queue, manifest)["enqueued"]) == 1
    jobs(queue)[0].rename(queue / "done" / "complete.sh")
    assert not keeper.run_keeper(queue, manifest)["enqueued"]


@pytest.mark.parametrize("status", ["", "  ", "UNDONE", "SLICEDISH", "unknown"])
def test_invalid_status_fails_closed(keeper, queue, status):
    manifest = configure(queue, status="status.txt")
    (queue / "status.txt").write_text(status)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert jobs(queue) == []


def test_disabled_and_unlisted_templates_skip(keeper, queue):
    manifest = configure(queue, b"invalid (", enabled=False, status="not-readable")
    (queue / "templates" / "tpl_unlisted.sh").write_bytes(BODY)
    assert not keeper.run_keeper(queue, manifest)["enqueued"]


@pytest.mark.parametrize("body", [
    b"", b" \n\t", b"#!/bin/bash\n# only comments\n", b"if then\n",
    b"echo ok\x00\n", b"\xff\n", b"cat <<MISSING\nunfinished\n",
])
def test_invalid_script_rejected_before_any_publication(keeper, queue, body):
    manifest = configure(queue, body, name="tpl_zbad.sh")
    (queue / "templates" / "tpl_agood.sh").write_bytes(BODY)
    manifest.write_text(json.dumps({
        "tpl_agood.sh": {"enabled": True}, "tpl_zbad.sh": {"enabled": True},
    }))
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert jobs(queue) == []


@pytest.mark.parametrize("document", [
    "not json", "[]", '{"tpl_work.sh": {}}',
    '{"tpl_work.sh": {"enabled": "false"}}',
    '{"tpl_work.sh": {"enabled": true, "typo": 1}}',
    '{"../outside.sh": {"enabled": true}}',
    '{"tpl_work.sh": {"enabled": true, "status_path": 1}}',
    '{"tpl_work.sh": {"enabled": true}, "tpl_work.sh": {"enabled": false}}',
])
def test_malformed_manifest_fails_closed(keeper, queue, document):
    manifest = configure(queue)
    manifest.write_text(document)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert not jobs(queue)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_capacity_fails_closed(keeper, queue, cap):
    manifest = configure(queue)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest, max_queue=cap)
    assert not jobs(queue)


def test_capacity_counts_unrelated_and_hidden_pending_jobs(keeper, queue):
    manifest = configure(queue)
    for name in ("unrelated.sh", ".hidden.sh"):
        (queue / "pending" / name).write_bytes(b"echo existing\n")
    entries = {}
    for i in range(5):
        name = f"tpl_{i}.sh"
        (queue / "templates" / name).write_text(f"echo {i}\n")
        entries[name] = {"enabled": True}
    manifest.write_text(json.dumps(entries))
    assert len(keeper.run_keeper(queue, manifest, max_queue=3)["enqueued"]) == 1
    assert len(jobs(queue)) == 3
    assert not keeper.run_keeper(queue, manifest, max_queue=3)["enqueued"]


def test_flock_excludes_another_process(queue):
    manifest = configure(queue)
    with (queue / ".keeper.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = invoke(queue, manifest)
    assert result.returncode != 0
    assert "lock" in result.stderr.lower()
    assert not jobs(queue)
    assert invoke(queue, manifest).returncode == 0


def test_concurrent_processes_obey_capacity_and_dedup(queue):
    manifest = configure(queue)
    command = [sys.executable, str(SCRIPT), "--queue", str(queue),
               "--manifest", str(manifest), "--max-queue", "1"]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) for _ in range(6)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0 or "lock" in stderr.lower(), (stdout, stderr)
    assert len(jobs(queue)) == 1
    assert jobs(queue)[0].read_bytes() == BODY


def test_cli_reports_error_and_never_enqueues_invalid_template(queue):
    manifest = configure(queue, b"if then\n")
    result = invoke(queue, manifest)
    assert result.returncode != 0
    assert "invalid bash script" in result.stderr
    assert not jobs(queue)


def test_validation_does_not_source_bash_env(keeper, queue, monkeypatch):
    marker = queue / "startup-ran"
    startup = queue / "startup.sh"
    startup.write_text(f"touch '{marker}'\n")
    monkeypatch.setenv("BASH_ENV", str(startup))
    manifest = configure(queue)
    assert len(keeper.run_keeper(queue, manifest)["enqueued"]) == 1
    assert not marker.exists()


def test_atomic_publication_exposes_only_complete_script(keeper, queue, monkeypatch):
    manifest = configure(queue)
    real_link = os.link
    observations = []

    def observe(source, destination, **kwargs):
        assert not jobs(queue)
        assert Path(source).suffix != ".sh"
        assert Path(source).read_bytes() == BODY
        real_link(source, destination, **kwargs)
        observations.append(Path(destination).read_bytes())

    monkeypatch.setattr(keeper.os, "link", observe)
    keeper.run_keeper(queue, manifest)
    assert observations == [BODY]


def test_publication_error_keeps_existing_jobs_and_cleans_only_own_temp(keeper, queue, monkeypatch):
    manifest = configure(queue)
    existing = queue / "pending" / "keep.sh"
    existing.write_bytes(b"echo existing\n")
    before = set(queue.rglob("*"))

    def fail(*args, **kwargs):
        raise OSError("injected publish failure")

    monkeypatch.setattr(keeper.os, "link", fail)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert existing.read_bytes() == b"echo existing\n"
    assert set(queue.rglob("*")) - before == {queue / ".keeper.lock"}


def test_no_clobber_on_publication_collision(keeper, queue, monkeypatch):
    manifest = configure(queue)
    original_link = os.link
    collided = []

    def collide(source, destination, **kwargs):
        Path(destination).write_bytes(b"echo another producer\n")
        collided.append(Path(destination))
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(keeper.os, "link", collide)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert len(collided) == 1
    assert collided[0].read_bytes() == b"echo another producer\n"


@pytest.mark.parametrize("folder", ["templates", "pending", "done", "failed"])
def test_nonregular_script_paths_fail_closed(keeper, queue, folder):
    manifest = configure(queue)
    path = queue / folder / ("tpl_work.sh" if folder == "templates" else "legacy.sh")
    if path.exists():
        path.unlink()
    path.symlink_to(queue / "absent")
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert path.is_symlink()


def test_missing_failure_archive_is_not_treated_as_empty(keeper, queue):
    manifest = configure(queue)
    (queue / "failed").rmdir()
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert not jobs(queue)


def test_move_after_pending_listing_aborts_tick(keeper, queue, monkeypatch):
    manifest = configure(queue)
    running = queue / "pending" / "legacy.sh"
    running.write_bytes(BODY)
    read = keeper._read_regular

    def move_then_read(path):
        if path == running:
            running.rename(queue / "failed" / running.name)
        return read(path)

    monkeypatch.setattr(keeper, "_read_regular", move_then_read)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert not jobs(queue)
    assert jobs(queue, "failed")[0].read_bytes() == BODY


@pytest.mark.parametrize("archive", ["done", "failed"])
def test_move_after_pending_read_still_blocks_duplicate(keeper, queue, monkeypatch, archive):
    manifest = configure(queue, status="status.txt")
    (queue / "status.txt").write_text("SLICED 1/8\n")
    running = queue / "pending" / "legacy.sh"
    running.write_bytes(BODY)
    read = keeper._read_regular

    def read_then_move(path):
        body = read(path)
        if path == running:
            running.rename(queue / archive / running.name)
        return body

    monkeypatch.setattr(keeper, "_read_regular", read_then_move)
    assert not keeper.run_keeper(queue, manifest)["enqueued"]
    assert not jobs(queue)
    assert jobs(queue, archive)[0].read_bytes() == BODY


def test_status_turning_done_before_publication_skips(keeper, queue, monkeypatch):
    manifest = configure(queue, status="status.txt")
    status = queue / "status.txt"
    status.write_text("SLICED 1/8\n")
    snapshot = keeper._snapshot
    inspections = 0

    def finish_during_inspection(root):
        nonlocal inspections
        inspections += 1
        if inspections == 2:
            status.write_text("DONE 8/8\n")
        return snapshot(root)

    monkeypatch.setattr(keeper, "_snapshot", finish_during_inspection)
    assert not keeper.run_keeper(queue, manifest)["enqueued"]


@pytest.mark.parametrize("input_kind", ["template", "status", "pending"])
def test_unreadable_input_aborts_before_enqueue(keeper, queue, monkeypatch, input_kind):
    manifest = configure(queue, status="status.txt")
    status = queue / "status.txt"
    status.write_text("SLICED\n")
    pending = queue / "pending" / "existing.sh"
    pending.write_bytes(b"echo existing\n")
    target = {"template": queue / "templates" / "tpl_work.sh",
              "status": status, "pending": pending}[input_kind]
    read = keeper._read_regular

    def unavailable(path):
        if path == target:
            raise PermissionError("injected unreadable input")
        return read(path)

    monkeypatch.setattr(keeper, "_read_regular", unavailable)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert jobs(queue) == [pending]


def test_inconclusive_bash_validator_fails_closed(keeper, queue, monkeypatch):
    manifest = configure(queue)
    monkeypatch.setattr(keeper.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 0, b"", b""))
    with pytest.raises(keeper.KeeperError, match="validator"):
        keeper.run_keeper(queue, manifest)
    assert not jobs(queue)


def test_validator_timeout_fails_closed(keeper, queue, monkeypatch):
    manifest = configure(queue)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)

    monkeypatch.setattr(keeper.subprocess, "run", timeout)
    with pytest.raises(keeper.KeeperError):
        keeper.run_keeper(queue, manifest)
    assert not jobs(queue)

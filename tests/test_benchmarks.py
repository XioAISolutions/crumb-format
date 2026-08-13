"""The corpus benchmark, run as a test so a regression fails the build.

`crumb measure` was a thing you could run. Nothing ran it. This suite makes the
numbers load-bearing:

* **Format independence** — the five `short-*` files are one conversation
  exported five ways. If OpenAI and ChatGPT-export inputs produce materially
  different handoffs, that is a reader bug, and only a shared corpus surfaces
  it.
* **No silent regression** — every result is compared against a committed
  baseline, so an extractor change that quietly drops facts fails here rather
  than shipping.

The guards are also tested for their ability to *fail*. A green check that
cannot go red is decoration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import run_corpus  # noqa: E402

RESULTS = run_corpus.run()
SHORT = [r for r in RESULTS if r["file"].startswith("short-")]


def test_corpus_covers_every_supported_format():
    """A format with no corpus entry has no regression guard."""
    from cli import transcripts

    covered = {r["format"] for r in RESULTS}
    missing = sorted(set(transcripts.FORMATS) - covered)
    assert not missing, f"formats with no corpus fixture: {missing}"


def test_corpus_includes_a_long_session():
    """Every extraction fault found so far only appeared at length."""
    assert any(r["messages"] >= 100 for r in RESULTS), (
        "corpus has no long transcript; budget scaling and recency selection "
        "are untested without one"
    )


def test_the_same_conversation_measures_the_same_in_every_format():
    """Compression must not depend on which provider exported the transcript."""
    assert len(SHORT) == 5, f"expected 5 short fixtures, found {len(SHORT)}"
    retentions = {r["retention"] for r in SHORT}
    assert max(retentions) - min(retentions) <= 0.10, {r["format"]: r["retention"] for r in SHORT}

    sizes = [r["crumb_tokens"] for r in SHORT]
    assert max(sizes) - min(sizes) <= 25, (
        f"same conversation produced differently sized handoffs: "
        f"{ {r['format']: r['crumb_tokens'] for r in SHORT} }"
    )


def test_no_result_regressed_against_the_baseline():
    failures = run_corpus.check_against_baseline(RESULTS)
    assert not failures, "\n".join(failures)


def test_short_transcripts_retain_almost_everything():
    for row in SHORT:
        assert row["retention"] >= 0.90, f"{row['file']} retains only {row['retention']}"


def test_long_transcripts_still_compress_hard():
    """Retention falls with length by design; compression must not."""
    long_rows = [r for r in RESULTS if r["messages"] >= 100]
    for row in long_rows:
        assert row["saved_pct"] >= 80, f"{row['file']} saved only {row['saved_pct']}%"


# ── the guards must be able to fail ──────────────────────────────────

def test_baseline_check_detects_a_retention_drop():
    degraded = [dict(r) for r in RESULTS]
    degraded[0]["retention"] = max(0.0, degraded[0]["retention"] - 0.5)
    failures = run_corpus.check_against_baseline(degraded)
    assert any("retention fell" in f for f in failures), failures


def test_baseline_check_detects_a_savings_drop():
    degraded = [dict(r) for r in RESULTS]
    degraded[0]["saved_pct"] = degraded[0]["saved_pct"] - 20
    failures = run_corpus.check_against_baseline(degraded)
    assert any("savings fell" in f for f in failures), failures


def test_baseline_check_detects_a_format_misdetection():
    degraded = [dict(r) for r in RESULTS]
    degraded[0]["format"] = "openai-messages" if degraded[0]["format"] != "openai-messages" else "claude-export"
    failures = run_corpus.check_against_baseline(degraded)
    assert any("baseline says" in f for f in failures), failures


def test_format_independence_check_detects_divergence():
    diverged = [dict(r) for r in SHORT]
    diverged[0]["retention"] = 0.2
    assert run_corpus.check_format_independence(diverged)


# ── the corpus is committed, not generated at test time ──────────────

def test_corpus_files_are_committed():
    corpus = REPO_ROOT / "benchmarks" / "corpus"
    files = sorted(corpus.glob("*.json")) + sorted(corpus.glob("*.jsonl"))
    assert len(files) >= 6, f"only {len(files)} corpus files present"

    tracked = subprocess.run(
        ["git", "ls-files", str(corpus)], capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout.split()
    assert len(tracked) >= 6, (
        "corpus files are not tracked by git; the baseline would describe inputs "
        "that reviewers cannot see"
    )


def test_baseline_is_committed_and_parseable():
    baseline = json.loads(run_corpus.BASELINE.read_text(encoding="utf-8"))
    assert baseline["results"]
    assert "tolerance" in baseline


def test_runner_exits_zero_and_emits_json():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "run_corpus.py"), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)
    assert {"file", "format", "saved_pct", "retention", "tokenizer"} <= set(rows[0])


def test_build_corpus_is_deterministic(tmp_path: Path):
    """Regenerating must not churn the diff, or the baseline means nothing."""
    import importlib

    build = importlib.import_module("build_corpus")
    before = {p.name: p.read_bytes() for p in run_corpus.CORPUS.iterdir() if p.is_file()}
    build.main()
    after = {p.name: p.read_bytes() for p in run_corpus.CORPUS.iterdir() if p.is_file()}
    assert before == after, "build_corpus.py is not deterministic"

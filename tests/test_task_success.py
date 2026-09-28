"""Tests for the task-success harness.

The harness exists because `crumb measure` answers a *recoverability* question
— did the token survive — and the literature is explicit that this is not the
same as whether the compressed context still works. So the properties that
make this benchmark trustworthy are asserted here rather than assumed, and the
first version of the file failed most of them.

The one that matters most: **probes must not be sampled from a region that a
particular strategy is structurally guaranteed to keep.** The original design
drew probes only from early turns, which handed `head-truncate-10` a free win
and scored every recency strategy at zero. That is not a benchmark, it is a
seating chart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import task_success  # noqa: E402
from cli import transcripts  # noqa: E402

CORPUS = REPO_ROOT / "benchmarks" / "corpus"
LONG = CORPUS / "long-claude-code.jsonl"


def load(path: Path):
    return transcripts.load_messages(path.read_text(encoding="utf-8"))


ROWS = task_success.run(provider="oracle")


# ── probes ───────────────────────────────────────────────────────────

def test_probe_generation_is_deterministic():
    """A benchmark whose questions change between runs cannot be cited."""
    _, messages = load(LONG)
    first = task_success.build_probes(messages)
    second = task_success.build_probes(messages)
    assert [(p.expected, p.question) for p in first] == [(p.expected, p.question) for p in second]


def test_probes_are_not_all_from_one_position():
    """The bug this file was rewritten for.

    Sampling only early turns hands `head-truncate-10` every probe it is
    guaranteed to answer and scores recency strategies at zero — measuring
    where a probe came from rather than what a strategy preserved.
    """
    _, messages = load(LONG)
    positions = {p.position for p in task_success.build_probes(messages)}
    assert len(positions) > 1, f"all probes came from one position: {positions}"


def test_no_single_fact_family_dominates():
    """A real session repeats module_00, module_01, ... — without a cap the
    benchmark measures one identifier shape and calls it task success."""
    _, messages = load(LONG)
    probes = task_success.build_probes(messages)
    families = [task_success._family(p.expected) for p in probes]
    for family in set(families):
        assert families.count(family) <= task_success.MAX_PER_FAMILY, (
            f"family {family!r} supplied {families.count(family)} of {len(probes)} probes"
        )


def test_a_probe_never_leaks_its_own_answer():
    """The blank must actually be blank."""
    _, messages = load(LONG)
    for probe in task_success.build_probes(messages):
        assert "____" in probe.question
        assert probe.expected.lower() not in probe.question.lower()


def test_probes_carry_enough_context_to_be_answerable():
    """A question that is only a blank is unanswerable by anything."""
    _, messages = load(LONG)
    for probe in task_success.build_probes(messages):
        stem = probe.question.split("____")[0]
        assert len(stem.strip()) > 20, f"probe has no locating context: {probe.question!r}"


# ── grading ──────────────────────────────────────────────────────────

def test_grading_accepts_a_models_wrapping_but_not_a_wrong_answer():
    assert task_success.grades("502", "502")
    assert task_success.grades("  `502`.  ", "502")
    assert task_success.grades("The 502", "502")
    assert not task_success.grades("404", "502")
    assert not task_success.grades("UNKNOWN", "502")
    assert not task_success.grades("", "502")


def test_unknown_is_never_scored_as_success():
    """The escape hatch must not be worth taking."""
    for reply in ("UNKNOWN", "unknown", "  Unknown  "):
        assert not task_success.grades(reply, "anything")


# ── the run ──────────────────────────────────────────────────────────

def test_full_transcript_is_the_ceiling():
    """The oracle answers by substring, so uncompressed context must score 1.0.

    Anything less means a probe is unanswerable from its own source, which
    would make every compressed score meaningless.
    """
    for row in [r for r in ROWS if r["condition"] == "full-transcript"]:
        assert row["success"] == 1.0, f"{row['file']} full transcript scored {row['success']}"


def test_every_condition_runs_on_every_file():
    files = {r["file"] for r in ROWS}
    for condition in task_success.CONDITIONS:
        assert {r["file"] for r in ROWS if r["condition"] == condition} == files


def test_success_is_reported_with_cost():
    """Completion alone is not a comparison — a strategy that keeps everything
    trivially wins on success and loses on tokens."""
    for row in ROWS:
        assert row["tokens"] > 0
        assert "tokenizer" in row


def test_the_harness_can_show_crumb_losing():
    """Credibility depends on this being possible — and it currently happens.

    On the oracle ceiling a recency window ties or beats CRUMB on several
    corpus files at a fraction of the tokens. If no condition ever beat CRUMB,
    the harness would be flattering its own format and should be distrusted.
    """
    beaten = False
    for path in {r["file"] for r in ROWS}:
        rows = [r for r in ROWS if r["file"] == path]
        crumb = next(r for r in rows if r["condition"] == "crumb")
        for other in rows:
            if other["condition"] in {"crumb", "full-transcript"}:
                continue
            if other["success"] > crumb["success"]:
                beaten = True
            if other["success"] >= crumb["success"] and other["tokens"] < crumb["tokens"]:
                beaten = True
    assert beaten, (
        "no condition matches or beats CRUMB anywhere — verify the harness is "
        "measuring honestly rather than flattering its own format"
    )


def test_oracle_output_refuses_to_be_read_as_a_model_result():
    report = task_success.format_report(ROWS)
    assert "NOT a model result" in report
    assert "ceiling" in report


def test_runner_emits_json_and_exits_zero():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "task_success.py"),
         "--provider", "oracle", "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)
    assert {"file", "condition", "success", "tokens", "by_position"} <= set(rows[0])


def test_a_real_provider_requires_a_model_name():
    """Silently defaulting a model name would make a published number
    unattributable."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "task_success.py"),
         "--provider", "ollama"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "--model is required" in result.stderr

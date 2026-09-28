"""Tests for the strategy comparison.

The comparison exists to be checkable, so the properties that make it credible
are asserted rather than assumed: every strategy is deterministic, the baseline
is a real baseline, and the table is allowed to show CRUMB losing.

That last one matters. A comparison built so its author always wins is
marketing. These tests fail if the harness stops being able to report a loss.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import compare  # noqa: E402

ROWS = compare.run()


def _rows_for(strategy: str) -> list[dict]:
    return [r for r in ROWS if r["strategy"] == strategy]


def test_every_strategy_runs_on_every_corpus_file():
    files = {r["file"] for r in ROWS}
    for strategy in compare.STRATEGIES:
        assert {r["file"] for r in _rows_for(strategy)} == files


def test_strategies_are_deterministic():
    """A comparison that changes between runs cannot be cited."""
    assert compare.run() == ROWS


def test_full_transcript_is_the_zero_baseline():
    for row in _rows_for("full-transcript"):
        assert row["saved_pct"] == 0.0
        assert row["retention"] == 1.0


def test_every_row_reports_what_it_dropped():
    """The column the hosted alternatives cannot produce."""
    for row in ROWS:
        assert "lost_examples" in row
        if row["retention"] < 1.0:
            assert row["lost_examples"], f"{row['strategy']} lost facts but named none"


def test_the_corpus_contains_tool_traffic():
    """Real agent sessions are mostly tool calls; without one, clearing
    strategies measure nothing and appear free."""
    tool_rows = _rows_for("anthropic-style-clear-tools")
    assert any(r["saved_pct"] > 5 for r in tool_rows), (
        "no corpus file has enough tool traffic for tool-result clearing to do "
        "anything — the strategy is being measured against nothing"
    )


def test_crumb_beats_truncation_on_long_sessions():
    """The one comparative claim worth making, held to the data."""
    long_rows = {r["strategy"]: r for r in ROWS if r["file"].startswith("long-")}
    crumb = long_rows["crumb"]
    for rival in ("recency-window-4", "recency-window-10", "head-truncate-10"):
        assert crumb["retention"] > long_rows[rival]["retention"], (
            f"crumb no longer retains more than {rival} on a long session"
        )


def test_the_comparison_can_show_crumb_losing():
    """Credibility depends on this being possible — and it currently happens.

    On tool-heavy sessions, tool-result clearing retains more than CRUMB at far
    lower compression. If no row anywhere shows CRUMB behind on some axis, the
    harness has become self-serving and should be distrusted.
    """
    by_file: dict[str, list[dict]] = {}
    for row in ROWS:
        by_file.setdefault(row["file"], []).append(row)

    crumb_is_beaten_somewhere = False
    for rows in by_file.values():
        crumb = next(r for r in rows if r["strategy"] == "crumb")
        others = [r for r in rows if r["strategy"] not in {"crumb", "full-transcript"}]
        if any(o["retention"] > crumb["retention"] for o in others):
            crumb_is_beaten_somewhere = True
        if any(o["saved_pct"] > crumb["saved_pct"] for o in others):
            crumb_is_beaten_somewhere = True
    assert crumb_is_beaten_somewhere, (
        "no strategy beats CRUMB on any axis in any file — verify the harness is "
        "measuring honestly rather than flattering its own format"
    )


def test_runner_emits_json_and_exits_zero():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "compare.py"), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)
    assert {"file", "strategy", "saved_pct", "retention", "lost_examples"} <= set(rows[0])


def test_table_discloses_that_vendor_rows_are_mechanisms_not_implementations():
    table = compare.format_table(ROWS)
    assert "not a vendor's implementation" in table or "No row is a vendor's score" in table


# ── external artifacts ───────────────────────────────────────────────
# Every strategy above is one this repository implements, which is a real
# limit: replicating a competitor's mechanism and scoring it is not measuring
# the competitor. benchmarks/external/ is how a real tool gets measured, and
# these tests hold the properties that make such a comparison trustworthy.

def _run_with_external(tmp_path, artifacts: dict[str, str]) -> list[dict]:
    """Run the comparison with a temporary external tool directory."""
    original = compare.EXTERNAL
    compare.EXTERNAL = tmp_path
    try:
        for relative, text in artifacts.items():
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return compare.run()
    finally:
        compare.EXTERNAL = original


def test_external_artifacts_are_measured_by_the_same_code(tmp_path):
    """A supplied artifact is scored on the same axes as every built-in row."""
    corpus = sorted(compare.CORPUS.glob("*.jsonl"))[0]
    rows = _run_with_external(
        tmp_path, {f"rival/{corpus.stem}.txt": "the 502 came from /api/ledger after 1200ms"}
    )
    row = next(r for r in rows if r["strategy"] == "rival" and r["file"] == corpus.name)
    assert row["status"] == "measured"
    assert row["tokens"] > 0
    assert 0.0 <= row["retention"] <= 1.0


def test_unsupplied_cases_are_reported_missing_not_skipped(tmp_path):
    """An unmeasured case must stay visible.

    Dropping it would make the table read as though the tool had been measured
    on every transcript, which is the precise dishonesty this harness exists to
    prevent. The row is emitted with null metrics and rendered as `—`.
    """
    corpus = sorted(compare.CORPUS.glob("*.jsonl"))[0]
    rows = _run_with_external(tmp_path, {f"rival/{corpus.stem}.txt": "only one case supplied"})

    rival = [r for r in rows if r["strategy"] == "rival"]
    corpus_files = {r["file"] for r in rows if r["strategy"] == "crumb"}
    assert {r["file"] for r in rival} == corpus_files, "rival was skipped on some transcripts"

    missing = [r for r in rival if r["status"] == "missing"]
    assert missing, "expected unsupplied cases to be reported"
    for row in missing:
        assert row["tokens"] is None and row["retention"] is None

    table = compare.format_table(rows)
    assert "no artifact supplied for this case" in table


def test_external_rows_carry_no_style_suffix(tmp_path):
    """`-style` means "we replicated a mechanism". Real output must not claim it."""
    corpus = sorted(compare.CORPUS.glob("*.jsonl"))[0]
    rows = _run_with_external(tmp_path, {f"ai-memory/{corpus.stem}.txt": "real output"})
    external = [r for r in rows if r["strategy"] == "ai-memory"]
    assert external and not any(r["strategy"].endswith("-style") for r in external)


def test_footer_states_whether_anything_external_has_been_measured():
    """The reader must be able to tell an empty comparison from a completed one."""
    table = compare.format_table(ROWS)
    if compare.external_tools():
        assert "real artifacts supplied" in table
    else:
        assert "No external tool has supplied artifacts yet" in table


def test_the_protocol_is_documented():
    """A corpus nobody knows how to run against is not an invitation."""
    readme = REPO_ROOT / "benchmarks" / "external" / "README.md"
    assert readme.is_file(), "benchmarks/external/README.md is the protocol; it must exist"
    text = readme.read_text(encoding="utf-8")
    assert "SOURCE.md" in text, "provenance requirement is missing"
    assert "missing" in text, "the unsupplied-case behaviour must be documented"

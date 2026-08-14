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

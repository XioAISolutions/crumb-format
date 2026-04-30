"""Tests for cli/deadlines.py and the `crumb lint --check-deadlines` flag.

One test per corner case Codex caught while reviewing the v1.4 deadlines
draft (PR #23). The doc was stripped to a stub when iteration costs
exceeded the doc's value; this test file is where each defect actually
lives now.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

# cli/ goes on the path so plain `import deadlines` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from cli import crumb, linting  # noqa: E402
from cli.deadlines import (  # noqa: E402
    DeadlineParseError,
    ParsedDeadline,
    check_deadline_lines,
    is_overdue,
    parse_deadline,
)


# ── parse_deadline: each Codex finding becomes one test ────────────────


class TestParseDeadlineHappyPath:
    def test_valid_date_only(self):
        # Codex finding: date-only deadlines must NOT be subjected to a tz check.
        result = parse_deadline("2026-04-30")
        assert result.kind == "date"
        assert result.value == date(2026, 4, 30)

    def test_valid_datetime_with_z(self):
        # Codex finding: Python 3.10's fromisoformat rejects 'Z'; we normalize.
        result = parse_deadline("2026-04-30T17:00:00Z")
        assert result.kind == "datetime"
        assert result.value.tzinfo == timezone.utc
        assert result.value.year == 2026 and result.value.hour == 17

    def test_valid_datetime_with_positive_offset(self):
        # Codex finding: offset datetimes must round-trip without local-converted getter bugs.
        result = parse_deadline("2026-04-30T15:00:00+02:00")
        assert result.kind == "datetime"
        assert result.value.tzinfo is not None
        assert result.value.utcoffset() == timedelta(hours=2)

    def test_valid_datetime_with_negative_offset(self):
        result = parse_deadline("2026-04-30T15:00:00-07:00")
        assert result.value.utcoffset() == timedelta(hours=-7)


class TestParseDeadlineRejects:
    def test_tzless_datetime(self):
        # Codex finding: datetime.fromisoformat accepts tz-less, grammar doesn't.
        with pytest.raises(DeadlineParseError, match="Z or"):
            parse_deadline("2026-04-30T15:00:00")

    def test_no_seconds(self):
        # Codex finding: fromisoformat too lenient — accepts HH:MM, grammar requires HH:MM:SS.
        with pytest.raises(DeadlineParseError):
            parse_deadline("2026-04-30T15:00+00:00")

    def test_fractional_seconds(self):
        # Codex finding: same — fractional seconds are outside the v1.4 grammar.
        with pytest.raises(DeadlineParseError):
            parse_deadline("2026-04-30T15:00:00.123Z")

    def test_second_precision_offset(self):
        # Codex finding: ±HH:MM:SS offsets are outside the grammar.
        with pytest.raises(DeadlineParseError):
            parse_deadline("2026-04-30T15:00:00+02:00:30")

    def test_out_of_range_month(self):
        # Codex finding: stdlib silently normalizes month 13 in some constructors;
        # date.fromisoformat actually raises here, but we test the contract.
        with pytest.raises(DeadlineParseError, match="real calendar date"):
            parse_deadline("2026-13-01")

    def test_out_of_range_day_feb_30(self):
        # Codex finding: same, for Feb 30.
        with pytest.raises(DeadlineParseError):
            parse_deadline("2026-02-30")

    def test_out_of_range_day_in_datetime(self):
        with pytest.raises(DeadlineParseError):
            parse_deadline("2026-02-30T00:00:00Z")

    def test_human_friendly_garbage(self):
        with pytest.raises(DeadlineParseError):
            parse_deadline("Friday")

    def test_empty(self):
        with pytest.raises(DeadlineParseError):
            parse_deadline("")

    def test_loose_yyyymmdd_form(self):
        # date.fromisoformat in 3.11+ accepts "20260430"; the v1.4 grammar does not.
        with pytest.raises(DeadlineParseError):
            parse_deadline("20260430")


# ── is_overdue ─────────────────────────────────────────────────────────


class TestIsOverdue:
    def test_past_date_is_overdue(self):
        parsed = parse_deadline("2020-01-01")
        assert is_overdue(parsed) is True

    def test_future_date_is_not_overdue(self):
        parsed = parse_deadline("2099-12-31")
        assert is_overdue(parsed) is False

    def test_past_datetime_is_overdue(self):
        parsed = parse_deadline("2020-01-01T00:00:00Z")
        assert is_overdue(parsed) is True

    def test_future_datetime_is_not_overdue(self):
        parsed = parse_deadline("2099-12-31T23:59:59Z")
        assert is_overdue(parsed) is False

    def test_offset_datetime_compared_correctly(self):
        # 2020-01-01T00:00:00+02:00 is 2019-12-31T22:00:00Z, both in the past.
        parsed = parse_deadline("2020-01-01T00:00:00+02:00")
        assert is_overdue(parsed, now=datetime(2020, 1, 2, tzinfo=timezone.utc)) is True


# ── check_deadline_lines (lint surface) ────────────────────────────────


class TestCheckDeadlineLines:
    def test_silent_when_no_deadline(self):
        lines = ["- to=any  do=ship", "- to=human  do=approve"]
        assert list(check_deadline_lines(lines)) == []

    def test_overdue_emits_finding(self):
        lines = ["- to=any  do=ship  deadline=2020-01-01"]
        findings = list(check_deadline_lines(lines))
        assert len(findings) == 1
        assert findings[0].code == "overdue_deadline"
        assert "2020-01-01" in findings[0].message

    def test_malformed_emits_finding(self):
        lines = ["- to=any  do=ship  deadline=Friday"]
        findings = list(check_deadline_lines(lines))
        assert len(findings) == 1
        assert findings[0].code == "malformed_deadline"

    def test_future_silent(self):
        lines = ["- to=any  do=ship  deadline=2099-12-31"]
        assert list(check_deadline_lines(lines)) == []

    def test_mix_of_valid_and_malformed(self):
        lines = [
            "- to=any  do=a  deadline=2099-12-31",
            "- to=any  do=b  deadline=Friday",
            "- to=any  do=c  deadline=2020-01-01",
        ]
        findings = list(check_deadline_lines(lines))
        # one malformed, one overdue; the future one is silent
        codes = sorted(f.code for f in findings)
        assert codes == ["malformed_deadline", "overdue_deadline"]


# ── crumb lint --check-deadlines integration ───────────────────────────


CRUMB_WITH_OVERDUE = """\
BEGIN CRUMB
v=1.3
kind=task
title=test
source=test
---
[goal]
ship

[context]
- nothing

[constraints]
- nothing

[handoff]
- to=any  do=ship  deadline=2020-01-01
- to=human  do=approve  deadline=Friday
END CRUMB
"""


class TestLintCheckDeadlinesFlag:
    def _run(self, tmp_path, *extra_args):
        path = tmp_path / "x.crumb"
        path.write_text(CRUMB_WITH_OVERDUE, encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            crumb.main(["lint", str(path), *extra_args])
        return exc.value.code

    def test_without_flag_no_warnings(self, tmp_path, capsys):
        # Without --check-deadlines, the overdue + malformed deadlines are not surfaced.
        rc = self._run(tmp_path)
        assert rc == 0
        out = capsys.readouterr()
        assert "deadline" not in (out.out + out.err).lower()

    def test_with_flag_emits_warnings(self, tmp_path, capsys):
        rc = self._run(tmp_path, "--check-deadlines")
        # Lint exits 0 on warnings without --strict.
        assert rc == 0
        captured = capsys.readouterr()
        all_output = captured.out + captured.err
        assert "overdue_deadline" in all_output
        assert "malformed_deadline" in all_output

    def test_strict_promotes_to_exit_1(self, tmp_path):
        # With --strict, warnings become exit 1 (matching cli/linting.py:235
        # convention; exit 2 stays reserved for parse failures).
        rc = self._run(tmp_path, "--check-deadlines", "--strict")
        assert rc == 1

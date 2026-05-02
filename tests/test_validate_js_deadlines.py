"""Smoke harness for the JS deadline parser in validators/validate.js.

Mirrors the most important cases from tests/test_deadlines.py to confirm
cross-language parity. We invoke node as a subprocess and assert exit
code + stdout shape rather than embed Node-specific fixtures, so this
only runs when node is on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

NODE = shutil.which("node")
VALIDATOR_JS = Path(__file__).resolve().parent.parent / "validators" / "validate.js"


@pytest.fixture(scope="module")
def js_runner():
    """Yield a function that runs a Node snippet against validate.js."""
    if NODE is None:
        pytest.skip("node not on PATH; JS validator deadline tests skipped")

    def run(snippet: str) -> tuple[str, str, int]:
        program = dedent(f"""
            const v = require({str(VALIDATOR_JS)!r});
            try {{
                {snippet}
            }} catch (e) {{
                process.stdout.write('CAUGHT:' + e.name + ':' + e.message);
                process.exit(0);
            }}
        """).strip()
        proc = subprocess.run(
            [NODE, "-e", program],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout, proc.stderr, proc.returncode

    return run


# ── parseDeadline happy path ───────────────────────────────────────────


class TestJsParseDeadlineHappyPath:
    def test_date_only(self, js_runner):
        out, _, rc = js_runner(
            "const r = v.parseDeadline('2026-04-30'); "
            "process.stdout.write(r.kind);"
        )
        assert rc == 0
        assert out == "date"

    def test_datetime_z(self, js_runner):
        out, _, rc = js_runner(
            "const r = v.parseDeadline('2026-04-30T17:00:00Z'); "
            "process.stdout.write(r.kind);"
        )
        assert rc == 0
        assert out == "datetime"

    def test_datetime_offset(self, js_runner):
        out, _, rc = js_runner(
            "const r = v.parseDeadline('2026-04-30T15:00:00+02:00'); "
            "process.stdout.write(r.kind);"
        )
        assert rc == 0
        assert out == "datetime"


# ── parseDeadline rejects ──────────────────────────────────────────────


class TestJsParseDeadlineRejects:
    @pytest.mark.parametrize("bad", [
        "2026-04-30T15:00:00",          # no tz
        "2026-04-30T15:00+00:00",       # no seconds
        "2026-04-30T15:00:00.123Z",     # fractional seconds
        "2026-13-01",                   # month 13
        "2026-02-30",                   # Feb 30
        "2026-02-30T00:00:00Z",         # Feb 30 in datetime
        "Friday",                       # garbage
        "",                             # empty
        "20260430",                     # loose YYYYMMDD form
        "2026-04-30T12:00:00+24:00",    # offset hour 24 (max is 23)
        "2026-04-30T12:00:00+99:99",    # offset hour 99 + minute 99
        "2026-04-30T12:00:00-24:00",    # negative offset out of range
        "2026-04-30T12:00:00+12:60",    # offset minute 60 (max is 59)
    ])
    def test_each_rejected(self, js_runner, bad):
        out, _, rc = js_runner(
            f"v.parseDeadline({bad!r}); process.stdout.write('NO-THROW');"
        )
        assert rc == 0
        assert out.startswith("CAUGHT:DeadlineParseError"), (
            f"expected DeadlineParseError on {bad!r}, got {out!r}"
        )


# ── isOverdueDeadline ──────────────────────────────────────────────────


class TestJsIsOverdueDeadline:
    def test_past_datetime_overdue(self, js_runner):
        out, _, rc = js_runner(
            "const p = v.parseDeadline('2020-01-01T00:00:00Z'); "
            "process.stdout.write(String(v.isOverdueDeadline(p)));"
        )
        assert rc == 0
        assert out == "true"

    def test_future_datetime_not_overdue(self, js_runner):
        out, _, rc = js_runner(
            "const p = v.parseDeadline('2099-12-31T00:00:00Z'); "
            "process.stdout.write(String(v.isOverdueDeadline(p)));"
        )
        assert rc == 0
        assert out == "false"

    def test_past_date_overdue(self, js_runner):
        out, _, rc = js_runner(
            "const p = v.parseDeadline('2020-01-01'); "
            "process.stdout.write(String(v.isOverdueDeadline(p)));"
        )
        assert rc == 0
        assert out == "true"


# ── still validates the existing parser surface ────────────────────────


class TestJsValidatorStillWorks:
    def test_parse_crumb_still_exposed(self, js_runner):
        # Round-trip: the existing parseCrumb export should still work
        # alongside the new deadline functions.
        out, _, rc = js_runner(
            'const text = "BEGIN CRUMB\\nv=1.3\\nkind=mem\\nsource=test\\n---\\n[consolidated]\\n- ok\\nEND CRUMB"; '
            'const r = v.parseCrumb(text); '
            'process.stdout.write(r.headers.kind);'
        )
        assert rc == 0
        assert out == "mem"

"""The compression receipt — a crumb that records what it cost.

`crumb capture` always knew the numbers; it printed them to stderr, where they
died with the shell that ran the command. A crumb pasted into another tool
arrived with no record of what had been dropped to make it. The `measured=`
header carries that with the artifact.

Two properties make it worth trusting, and both are asserted here:

1. **It describes itself.** The receipt is part of the file it measures, so
   writing it changes the token count it reports. `embed_receipt` iterates to a
   fixed point, and `crumb measure` on the finished file reproduces the header
   exactly.
2. **It cannot inflate its own score.** Retention is a substring test against
   digit-heavy patterns, so a receipt reading `->3502` would make a source
   `502` count as retained. The receipt is excluded from the fact haystack.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cli import measure as measure_mod  # noqa: E402
from cli import transcripts  # noqa: E402

CLI = REPO_ROOT / "crumb_cli.py"
CORPUS = REPO_ROOT / "benchmarks" / "corpus" / "long-claude-code.jsonl"


def capture(tmp_path: Path, *args: str) -> Path:
    source = tmp_path / "src.jsonl"
    source.write_text(CORPUS.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), "capture", "--transcript", "src.jsonl",
         "--dir", ".crumb", *args],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    return next((tmp_path / ".crumb").glob("*.crumb"))


# ── the receipt is written and readable ──────────────────────────────

def test_capture_records_what_the_compression_cost(tmp_path):
    receipt = measure_mod.parse_receipt(capture(tmp_path).read_text(encoding="utf-8"))
    assert receipt is not None, "capture wrote no measured= header"
    assert receipt["source_tokens"] > receipt["crumb_tokens"] > 0
    assert 0.0 <= receipt["retention_pct"] <= 100.0
    assert receipt["tokenizer"], "a number with no named tokenizer is not a measurement"


def test_the_receipt_can_be_declined(tmp_path):
    assert measure_mod.parse_receipt(
        capture(tmp_path, "--no-receipt").read_text(encoding="utf-8")
    ) is None


def test_a_crumb_without_a_receipt_parses_as_none():
    assert measure_mod.parse_receipt("BEGIN CRUMB\nv=1.4\n---\n[goal]\nx\nEND CRUMB") is None


# ── it describes the file it is part of ──────────────────────────────

def test_crumb_measure_reproduces_the_header_exactly(tmp_path):
    """The property that makes the receipt citable rather than decorative."""
    crumb_path = capture(tmp_path)
    receipt = measure_mod.parse_receipt(crumb_path.read_text(encoding="utf-8"))

    result = subprocess.run(
        [sys.executable, str(CLI), "measure", str(crumb_path),
         "--source", "src.jsonl", "--json"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout)

    assert receipt["source_tokens"] == measured["tokens"]["source"]
    assert receipt["crumb_tokens"] == measured["tokens"]["crumb"]
    assert receipt["tokenizer"] == measured["tokens"]["tokenizer"]
    assert abs(receipt["saved_pct"] - measured["tokens"]["saved_pct"]) < 0.05
    assert abs(receipt["retention_pct"] - measured["retention"]["overall"] * 100) < 0.05


def test_the_reported_token_count_includes_the_receipt_itself(tmp_path):
    """Whoever pastes the crumb pays for the receipt's bytes too.

    Measuring the body and labelling it as the whole file would understate the
    artifact by the size of its own header.
    """
    text = capture(tmp_path).read_text(encoding="utf-8")
    receipt = measure_mod.parse_receipt(text)
    whole, _ = measure_mod.count_tokens(text)
    body, _ = measure_mod.count_tokens(measure_mod.strip_receipt(text))

    assert receipt["crumb_tokens"] == whole
    assert whole > body, "the receipt should cost something; it is real text"


def test_embedding_is_idempotent(tmp_path):
    """Re-embedding replaces the receipt rather than stacking a second one."""
    source = CORPUS.read_text(encoding="utf-8")
    fmt, messages = transcripts.load_messages(source)
    source_text = transcripts.transcript_text(messages)
    crumb = transcripts.to_crumb(messages, fmt=fmt)

    once = measure_mod.embed_receipt(source_text, crumb)
    twice = measure_mod.embed_receipt(source_text, once)
    assert once == twice
    assert once.count("measured=") == 1


# ── it cannot inflate its own score ──────────────────────────────────

def test_the_receipt_is_excluded_from_fact_matching():
    """A digit in the receipt must not satisfy a fact from the source.

    Retention matches substrings, and error codes are `[45]\\d{2}`. A crumb
    whose receipt happens to contain `3502` would otherwise report the source's
    `502` as retained while the handoff itself never mentions it.
    """
    source = "The endpoint returned 502 after a 1200ms timeout in src/api/ledger.ts"
    body = (
        "BEGIN CRUMB\nv=1.4\nkind=task\nsource=test\n---\n"
        "[goal]\nSomething unrelated\n[context]\nNo facts here\n"
        "[constraints]\n- none\nEND CRUMB"
    )
    # A receipt whose digits contain the source's error code and number.
    gamed = body.replace("---", "measured=91200->3502 tokens, 96.2% saved, 99.9% retention, x\n---")

    honest = measure_mod.measure(source, body)
    with_receipt = measure_mod.measure(source, gamed)

    assert with_receipt.facts_retained == honest.facts_retained, (
        "the receipt's own digits were counted as retained facts"
    )


def test_stripping_a_receipt_leaves_the_body_untouched():
    body = "BEGIN CRUMB\nv=1.4\nkind=task\nsource=test\n---\n[goal]\nx\nEND CRUMB"
    with_receipt = body.replace("---", "measured=10->5 tokens, 50.0% saved, 1.0% retention, x\n---")
    assert measure_mod.strip_receipt(with_receipt) == body


# ── it does not break the format ─────────────────────────────────────

def test_a_crumb_carrying_a_receipt_still_validates(tmp_path):
    """`measured=` is an unknown header to the spec, which pins that unknown
    headers are ignored — so this needs no spec change."""
    crumb_path = capture(tmp_path)
    result = subprocess.run(
        [sys.executable, str(CLI), "validate", str(crumb_path)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_receipt_survives_a_parse_render_round_trip(tmp_path):
    """The parser must carry the header through rather than dropping it."""
    import crumb_core

    text = capture(tmp_path).read_text(encoding="utf-8")
    parsed = crumb_core.parse_crumb(text)
    assert measure_mod.RECEIPT_KEY in parsed["headers"]

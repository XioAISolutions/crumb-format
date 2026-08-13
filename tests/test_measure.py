"""Tests for `crumb measure` — token savings and fact retention.

The point of these tests is that the numbers are honest: the tokenizer
names itself, retention actually falls when facts are dropped, and the
threshold gate exits non-zero so it can guard a pipeline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cli import measure  # noqa: E402
from cli import transcripts  # noqa: E402


SOURCE = """\
user: The cart drops on refresh at /checkout/payment, returning a 503 from the
edge. Latency is 450ms on the hydrate call. Do not change the CartService API.
assistant: The bug is in src/checkout/CartService.ts — sessionStorage is cleared
by the redirect. Move hydration into middleware.ts and add a test in
tests/checkout_refresh.spec.ts. See https://example.com/docs/cart for the spec.
"""

FAITHFUL_CRUMB = """\
BEGIN CRUMB
v=1.4
kind=task
title=Fix cart drop
source=test
---
[goal]
Move cart hydration out of sessionStorage.

[context]
- Files: src/checkout/CartService.ts, middleware.ts, tests/checkout_refresh.spec.ts
- Edge returns 503; hydrate call takes 450ms
- Spec: https://example.com/docs/cart

[constraints]
- Do not change the CartService API
END CRUMB
"""

LOSSY_CRUMB = """\
BEGIN CRUMB
v=1.4
kind=task
title=Fix cart drop
source=test
---
[goal]
Fix the cart.

[context]
- Something is wrong on refresh.

[constraints]
- Be careful.
END CRUMB
"""


# ── tokenisation ─────────────────────────────────────────────────────

def test_tokenizer_names_itself():
    count, method = measure.count_tokens("hello world, this is some text")
    assert count > 0
    assert method.startswith(("tiktoken:", "heuristic:"))


def test_heuristic_fallback_is_labelled_inexact(monkeypatch):
    """A count from len(text)//4 must never be presented as exact."""
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    monkeypatch.delitem(sys.modules, "tiktoken", raising=False)
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    count, method = measure.count_tokens("x" * 400)
    assert count == 100
    assert method == "heuristic:chars/4"
    assert not measure.tokenizer_is_exact(method)


def test_exact_tokenizer_path_is_used_when_available(monkeypatch):
    """Stub tiktoken so the exact branch is exercised without a network fetch."""
    import types

    class _Enc:
        def encode(self, text):
            return text.split()

    stub = types.SimpleNamespace(get_encoding=lambda name: _Enc())
    monkeypatch.setitem(sys.modules, "tiktoken", stub)
    count, method = measure.count_tokens("one two three four")
    assert count == 4
    assert method == "tiktoken:o200k_base"
    assert measure.tokenizer_is_exact(method)


def test_unknown_encoding_falls_back_rather_than_raising(monkeypatch):
    import types

    def boom(name):
        raise ValueError("unknown encoding")

    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=boom))
    _, method = measure.count_tokens("some text here", encoding="not-a-real-encoding")
    assert method == "heuristic:chars/4"


# ── fact extraction ──────────────────────────────────────────────────

def test_facts_are_bucketed_without_double_counting():
    buckets = measure.extract_facts(SOURCE)
    assert "src/checkout/cartservice.ts" in buckets["paths"]
    assert "https://example.com/docs/cart" in buckets["urls"]
    assert "503" in buckets["errors"]
    assert "450ms" in buckets["numbers"]
    # A token claimed by one category is not counted again by a later one.
    all_facts = [f for bucket in buckets.values() for f in bucket]
    assert len(all_facts) == len(set(all_facts))


def test_bare_small_integers_are_not_counted_as_facts():
    buckets = measure.extract_facts("We looked at step 3 and the 2 remaining cases.")
    assert not buckets["numbers"]


# ── the measurement ──────────────────────────────────────────────────

def test_faithful_crumb_scores_higher_retention_than_lossy_one():
    faithful = measure.measure(SOURCE, FAITHFUL_CRUMB)
    lossy = measure.measure(SOURCE, LOSSY_CRUMB)
    assert faithful.retention > lossy.retention
    assert lossy.retention < 0.3


def test_lost_facts_are_reported_by_name():
    result = measure.measure(SOURCE, LOSSY_CRUMB)
    lost = {f for cat in result.categories.values() for f in cat.lost}
    assert "src/checkout/cartservice.ts" in lost


def test_savings_can_be_negative_when_the_crumb_is_larger():
    """Structure has a cost. A tiny source must not report fake savings."""
    result = measure.measure("short", FAITHFUL_CRUMB)
    assert result.saved_pct < 0
    assert result.ratio < 1


def test_json_output_is_parseable_and_carries_the_caveat():
    result = measure.measure(SOURCE, FAITHFUL_CRUMB)
    payload = json.loads(measure.format_json(result, source_label="s", crumb_label="c"))
    assert payload["tokens"]["tokenizer"]
    assert "by_category" in payload["retention"]
    assert "lexical proxy" in payload["caveat"]


def test_report_discloses_when_counts_are_approximate():
    result = measure.measure(SOURCE, FAITHFUL_CRUMB)
    report = measure.format_report(result, source_label="s", crumb_label="c")
    if not result.exact:
        assert "approximate" in report
    assert "lexical proxy" in report


# ── thresholds ───────────────────────────────────────────────────────

def test_thresholds_pass_when_met():
    result = measure.measure(SOURCE, FAITHFUL_CRUMB)
    assert measure.check_thresholds(result, min_retention=10.0) == []


def test_thresholds_fail_with_a_reason():
    result = measure.measure(SOURCE, LOSSY_CRUMB)
    failures = measure.check_thresholds(result, min_retention=90.0)
    assert len(failures) == 1
    assert "retention" in failures[0]


# ── CLI wiring ───────────────────────────────────────────────────────

def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "crumb_cli.py"), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_measure_emits_json(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text(SOURCE, encoding="utf-8")
    crumb = tmp_path / "a.crumb"
    crumb.write_text(FAITHFUL_CRUMB, encoding="utf-8")

    result = _run("measure", str(crumb), "--source", str(src), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["retention"]["facts_total"] > 0


def test_cli_measure_gate_exits_nonzero(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text(SOURCE, encoding="utf-8")
    crumb = tmp_path / "lossy.crumb"
    crumb.write_text(LOSSY_CRUMB, encoding="utf-8")

    result = _run("measure", str(crumb), "--source", str(src), "--min-retention", "90")
    assert result.returncode == 1
    assert "FAIL" in result.stderr


def test_committed_example_still_matches_the_documented_numbers():
    """The README quotes this fixture. If extraction drifts, the docs lie."""
    source = REPO_ROOT / "examples" / "transcript-checkout-openai.json"
    crumb = REPO_ROOT / "examples" / "transcript-checkout.crumb"

    result = _run("from-messages", "-i", str(source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == crumb.read_text(encoding="utf-8"), (
        "regenerate examples/transcript-checkout.crumb — extraction changed"
    )

    _, messages = transcripts.load_messages(source.read_text(encoding="utf-8"))
    measured = measure.measure(
        transcripts.transcript_text(messages), crumb.read_text(encoding="utf-8")
    )
    assert measured.saved_pct > 85, f"README claims ~89% savings, got {measured.saved_pct:.1f}%"
    assert measured.retention > 0.85, f"README claims ~92% retention, got {measured.retention:.2%}"


def test_cli_measure_flattens_a_json_transcript_source(tmp_path):
    """JSON punctuation must not be counted as source tokens."""
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"messages": [
        {"role": "user", "content": "Fix src/checkout/CartService.ts please, it 503s."}
    ]}), encoding="utf-8")
    crumb = tmp_path / "a.crumb"
    crumb.write_text(FAITHFUL_CRUMB, encoding="utf-8")

    result = _run("measure", str(crumb), "--source", str(src), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    raw_tokens = len(src.read_text(encoding="utf-8")) // 4
    assert payload["tokens"]["source"] < raw_tokens

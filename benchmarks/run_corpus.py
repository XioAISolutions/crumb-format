#!/usr/bin/env python3
"""Measure every transcript in the corpus and report token savings + retention.

This turns `crumb measure` from a thing you can run into a thing that runs
itself. Two properties it exists to hold:

* **Format independence.** The five ``short-*`` files are the same conversation
  exported five different ways. If OpenAI and ChatGPT-export inputs produce
  different handoffs, that is a reader bug, and nothing else would surface it.
* **No silent regression.** Numbers are compared against a committed baseline,
  so a change to the extractor that quietly drops facts fails the build instead
  of shipping.

    python benchmarks/run_corpus.py            # human table
    python benchmarks/run_corpus.py --json     # machine-readable
    python benchmarks/run_corpus.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import measure as measure_mod  # noqa: E402
from cli import transcripts  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus"
BASELINE = Path(__file__).resolve().parent / "baseline.json"

# Retention is a lexical proxy and falls with transcript length by design: a
# 50k-token session cannot be carried losslessly in a 2.4k handoff. The floors
# below are regression guards, not targets — they are set beneath the measured
# values so that a real drop trips them while normal variation does not.
TOLERANCE = {"saved_pct": 3.0, "retention": 0.05}


def measure_one(path: Path) -> dict:
    fmt, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
    crumb = transcripts.to_crumb(messages, fmt=fmt)
    result = measure_mod.measure(transcripts.transcript_text(messages), crumb)
    return {
        "file": path.name,
        "format": fmt,
        "messages": len(messages),
        "source_tokens": result.source_tokens,
        "crumb_tokens": result.crumb_tokens,
        "saved_pct": round(result.saved_pct, 1),
        "ratio": round(result.ratio, 1),
        "retention": round(result.retention, 3),
        "tokenizer": result.tokenizer,
    }


def run() -> list[dict]:
    files = sorted(CORPUS.glob("*.json")) + sorted(CORPUS.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no corpus files in {CORPUS} — run benchmarks/build_corpus.py")
    return [measure_one(p) for p in files]


def format_table(rows: list[dict]) -> str:
    header = f"{'file':<28} {'format':<24} {'msgs':>5} {'tokens':>16} {'saved':>7} {'retain':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['file']:<28} {r['format']:<24} {r['messages']:>5} "
            f"{r['source_tokens']:>7,}→{r['crumb_tokens']:<8,} "
            f"{r['saved_pct']:>6.1f}% {r['retention'] * 100:>6.1f}%"
        )
    lines.append("")
    lines.append(f"Token counts via {rows[0]['tokenizer']}.")
    lines.append("Retention is a lexical proxy — it reports which load-bearing tokens survived,")
    lines.append("not whether the next model succeeds. It falls with length by design.")
    return "\n".join(lines)


def check_against_baseline(rows: list[dict]) -> list[str]:
    if not BASELINE.is_file():
        return [f"no baseline at {BASELINE} — run with --update-baseline"]
    baseline = {b["file"]: b for b in json.loads(BASELINE.read_text(encoding="utf-8"))["results"]}
    failures: list[str] = []
    for row in rows:
        prior = baseline.get(row["file"])
        if prior is None:
            failures.append(f"{row['file']}: not in baseline — regenerate it")
            continue
        if row["format"] != prior["format"]:
            failures.append(
                f"{row['file']}: detected as {row['format']}, baseline says {prior['format']}")
        if row["saved_pct"] < prior["saved_pct"] - TOLERANCE["saved_pct"]:
            failures.append(
                f"{row['file']}: savings fell {prior['saved_pct']}% → {row['saved_pct']}%")
        if row["retention"] < prior["retention"] - TOLERANCE["retention"]:
            failures.append(
                f"{row['file']}: retention fell {prior['retention']} → {row['retention']}")
    return failures


def check_format_independence(rows: list[dict]) -> list[str]:
    """The five short files are one conversation exported five ways."""
    short = [r for r in rows if r["file"].startswith("short-")]
    if len(short) < 2:
        return []
    retentions = {r["retention"] for r in short}
    if max(retentions) - min(retentions) > 0.10:
        detail = ", ".join(f"{r['format']}={r['retention']}" for r in short)
        return [f"same conversation retains differently by format: {detail}"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Overwrite baseline.json with the current numbers.")
    args = parser.parse_args(argv)

    rows = run()

    if args.update_baseline:
        BASELINE.write_text(
            json.dumps({"tolerance": TOLERANCE, "results": rows}, indent=2) + "\n",
            encoding="utf-8")
        print(f"baseline written to {BASELINE}")
        return 0

    print(json.dumps(rows, indent=2) if args.json else format_table(rows))

    failures = check_against_baseline(rows) + check_format_independence(rows)
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

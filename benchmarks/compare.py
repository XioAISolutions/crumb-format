#!/usr/bin/env python3
"""Compare context-compression strategies on the same corpus.

Every strategy here is **deterministic and offline**, so the numbers reproduce
exactly and cost nothing to generate. That is a deliberate constraint: a
comparison that needs API keys is a comparison nobody re-runs, and one that
cannot be re-run is indistinguishable from a claim.

What this is *not*: it does not call OpenAI's compaction or Anthropic's context
editing. It replicates the **documented mechanisms** of the truncation-family
strategies — clearing tool results beyond the last N, dropping thinking blocks,
keeping a recency window — and measures what each drops. Those mechanisms are
public; the hosted implementations also summarise, which these do not. The
`*-style` naming says so, and no row here should be read as a vendor's score.

The point is not that CRUMB wins. On raw ratio it should not: a strategy that
keeps 400 tokens of the most recent turns will always beat one that keeps
structure. The point is that every row states *what it dropped*, which is the
property the hosted alternatives do not expose at all.

    python benchmarks/compare.py
    python benchmarks/compare.py --json
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


# ── strategies ───────────────────────────────────────────────────────
# Each takes the parsed messages and returns the compressed text a model would
# actually receive.

def strategy_crumb(messages, fmt: str) -> str:
    """CRUMB: restructure into a handoff. Deterministic, no model call."""
    return transcripts.to_crumb(messages, fmt=fmt)


def strategy_full(messages, fmt: str) -> str:
    """The baseline: send everything. 0% saved, 100% retained by definition."""
    return transcripts.transcript_text(messages)


def _recent_window(messages, keep_turns: int) -> str:
    return transcripts.transcript_text(messages[-keep_turns:])


def strategy_recency_10(messages, fmt: str) -> str:
    """Sliding window — the simplest thing that works, and a hard baseline."""
    return _recent_window(messages, 10)


def strategy_recency_4(messages, fmt: str) -> str:
    """An aggressive window, roughly matched to a CRUMB's token budget."""
    return _recent_window(messages, 4)


def strategy_head_10(messages, fmt: str) -> str:
    """Head truncation, included because it is what naive caps actually do."""
    return transcripts.transcript_text(messages[:10])


def strategy_clear_tool_results(messages, fmt: str):
    """Anthropic-style tool-result clearing, mechanism only.

    Mirrors the documented `clear_tool_uses` behaviour: keep the most recent N
    tool uses, replace older tool results with a placeholder. Text turns are
    untouched, which is why this saves less than a window but loses less prose.
    """
    kept, seen = [], 0
    for msg in reversed(messages):
        clone = transcripts.Message(
            role=msg.role, text=msg.text, tool_calls=list(msg.tool_calls),
            tool_results=list(msg.tool_results), code_blocks=list(msg.code_blocks),
        )
        if clone.tool_results:
            seen += 1
            if seen > 3:
                clone.tool_results = ["[tool result cleared]"]
        kept.append(clone)
    return transcripts.transcript_text(list(reversed(kept)))


STRATEGIES = {
    "full-transcript": strategy_full,
    "recency-window-10": strategy_recency_10,
    "recency-window-4": strategy_recency_4,
    "head-truncate-10": strategy_head_10,
    "anthropic-style-clear-tools": strategy_clear_tool_results,
    "crumb": strategy_crumb,
}


def run() -> list[dict]:
    files = sorted(CORPUS.glob("*.json")) + sorted(CORPUS.glob("*.jsonl"))
    rows: list[dict] = []
    for path in files:
        fmt, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
        source = transcripts.transcript_text(messages)
        for name, fn in STRATEGIES.items():
            compressed = fn(messages, fmt)
            result = measure_mod.measure(source, compressed)
            lost = sorted({f for c in result.categories.values() for f in c.lost})
            rows.append({
                "file": path.name,
                "messages": len(messages),
                "strategy": name,
                "tokens": result.crumb_tokens,
                "saved_pct": round(result.saved_pct, 1),
                "retention": round(result.retention, 3),
                "lost_examples": lost[:5],
                "tokenizer": result.tokenizer,
            })
    return rows


def format_table(rows: list[dict]) -> str:
    out: list[str] = []
    for path in dict.fromkeys(r["file"] for r in rows):
        subset = [r for r in rows if r["file"] == path]
        out.append(f"\n{path}  ({subset[0]['messages']} messages)")
        out.append(f"  {'strategy':<30} {'tokens':>8} {'saved':>8} {'retained':>9}   {'example losses'}")
        out.append("  " + "-" * 92)
        for r in sorted(subset, key=lambda r: -r["saved_pct"]):
            losses = ", ".join(r["lost_examples"][:3]) or "—"
            out.append(
                f"  {r['strategy']:<30} {r['tokens']:>8,} {r['saved_pct']:>7.1f}% "
                f"{r['retention'] * 100:>8.1f}%   {losses[:44]}"
            )
    out.append("")
    out.append(f"Token counts via {rows[0]['tokenizer']}.")
    out.append(
        "`*-style` rows replicate a documented mechanism, not a vendor's implementation:\n"
        "the hosted versions also summarise, which these do not. No row is a vendor's score."
    )
    out.append(
        "Retention is lexical — which load-bearing tokens survived, not whether the next\n"
        "model succeeds. The column that matters is the last one: every strategy here can\n"
        "say what it dropped, which hosted compaction does not expose."
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    rows = run()
    print(json.dumps(rows, indent=2) if args.json else format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

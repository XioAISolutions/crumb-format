#!/usr/bin/env python3
"""measure_drift.py — quantify semantic drift across MeTalk levels 0-5.

Two backends are supported:

  ngram     (default, no deps)  Lexical similarity using character 4-gram
                                cosine. Fast and deterministic. NOT a
                                semantic measure — it tells you how
                                recoverable the consonant skeleton is.

  st        sentence-transformers (`pip install crumb-format[embeddings]`)
                                Real semantic cosine similarity using
                                all-MiniLM-L6-v2.

Usage:
    python scripts/measure_drift.py                       # ngram, examples/
    python scripts/measure_drift.py --backend st          # semantic
    python scripts/measure_drift.py --dir fixtures/valid  # different corpus
    python scripts/measure_drift.py --md > docs/vowel-drift-benchmark.md
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.metalk import encode as metalk_encode  # noqa: E402


# ── Similarity backends ────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 4) -> Counter:
    text = text.lower()
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def ngram_similarity(a: str, b: str, n: int = 4) -> float:
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ga or not gb:
        return 0.0
    keys = set(ga) | set(gb)
    dot = sum(ga[k] * gb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ga.values()))
    nb = math.sqrt(sum(v * v for v in gb.values()))
    return dot / (na * nb) if na and nb else 0.0


def make_st_similarity():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print("error: sentence-transformers not installed. "
              "Install with: pip install crumb-format[embeddings]",
              file=sys.stderr)
        sys.exit(2)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def sim(a: str, b: str) -> float:
        ea, eb = model.encode([a, b])
        dot = float(sum(x * y for x, y in zip(ea, eb)))
        na = math.sqrt(float(sum(x * x for x in ea)))
        nb = math.sqrt(float(sum(y * y for y in eb)))
        return dot / (na * nb) if na and nb else 0.0
    return sim


# ── Token estimate (matches cli/metalk._estimate_tokens) ─────────────

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Driver ───────────────────────────────────────────────────────────

LEVELS = [1, 2, 3, 4, 5]


def measure_file(path: Path, sim: Callable[[str, str], float]) -> dict:
    original = path.read_text(encoding="utf-8")
    rows = {"file": path.name, "original_tokens": estimate_tokens(original)}
    for level in LEVELS:
        try:
            encoded = metalk_encode(original, level=level)
        except Exception as exc:
            rows[f"L{level}"] = {"error": str(exc)}
            continue
        rows[f"L{level}"] = {
            "tokens": estimate_tokens(encoded),
            "pct_saved": round(
                ((rows["original_tokens"] - estimate_tokens(encoded))
                 / rows["original_tokens"]) * 100, 1
            ),
            "sim": round(sim(original, encoded), 3),
        }
    return rows


def render_table(rows: list[dict], backend: str) -> str:
    """Markdown table: file × level → sim / pct_saved."""
    lines = [
        f"# Vowel-Strip Drift Benchmark ({backend})",
        "",
        "Cosine similarity vs original, per MeTalk level.",
        f"`{backend}` backend: "
        + ("char 4-gram lexical similarity (no model)" if backend == "ngram"
           else "all-MiniLM-L6-v2 semantic embedding"),
        "",
        "| File | Tokens | L1 sim/save | L2 sim/save | L3 sim/save | L4 sim/save | L5 sim/save |",
        "|------|--------|-------------|-------------|-------------|-------------|-------------|",
    ]
    sim_totals = {f"L{level}": [] for level in LEVELS}
    save_totals = {f"L{level}": [] for level in LEVELS}
    for row in rows:
        cells = [row["file"], str(row["original_tokens"])]
        for level in LEVELS:
            key = f"L{level}"
            cell = row.get(key, {})
            if "error" in cell:
                cells.append(f"err: {cell['error'][:20]}")
            else:
                cells.append(f"{cell['sim']:.3f} / {cell['pct_saved']}%")
                sim_totals[key].append(cell['sim'])
                save_totals[key].append(cell['pct_saved'])
        lines.append("| " + " | ".join(cells) + " |")
    # Aggregate row
    avg_cells = ["**average**", ""]
    for level in LEVELS:
        key = f"L{level}"
        if sim_totals[key]:
            asim = sum(sim_totals[key]) / len(sim_totals[key])
            asave = sum(save_totals[key]) / len(save_totals[key])
            avg_cells.append(f"**{asim:.3f} / {asave:.1f}%**")
        else:
            avg_cells.append("—")
    lines.append("| " + " | ".join(avg_cells) + " |")
    lines.append("")
    lines.append("**Reading the table:** `sim` is cosine similarity vs the "
                 "original (1.0 = identical). `save` is the % token reduction "
                 "vs the original. Higher sim with higher save is better.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["ngram", "st"], default="ngram",
                        help="Similarity backend (default: ngram).")
    parser.add_argument("--dir", default="examples",
                        help="Directory of .crumb files to benchmark (default: examples).")
    parser.add_argument("--md", action="store_true",
                        help="Emit a markdown report (suitable for docs/vowel-drift-benchmark.md).")
    args = parser.parse_args()

    target = ROOT / args.dir
    if not target.exists():
        print(f"error: {target} not found", file=sys.stderr)
        return 1

    files = sorted(target.glob("*.crumb"))
    if not files:
        print(f"error: no .crumb files under {target}", file=sys.stderr)
        return 1

    sim = ngram_similarity if args.backend == "ngram" else make_st_similarity()

    rows = [measure_file(f, sim) for f in files]

    if args.md:
        print(render_table(rows, args.backend))
    else:
        for row in rows:
            print(f"\n{row['file']}  ({row['original_tokens']} tokens)")
            for level in LEVELS:
                cell = row[f"L{level}"]
                if "error" in cell:
                    print(f"  L{level}: error: {cell['error']}")
                else:
                    print(f"  L{level}: sim={cell['sim']:.3f}  "
                          f"tokens={cell['tokens']}  saved={cell['pct_saved']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

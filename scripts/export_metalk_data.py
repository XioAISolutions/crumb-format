#!/usr/bin/env python3
"""Export MeTalk dictionaries as JSON for the in-browser JS port.

Python (`cli/metalk.py`) is the source of truth. This script serialises
every dict/frozenset needed by the algorithm into `web/metalk-data.json`.
The JS port loads that file at runtime.

A test (`tests/test_js_port.py`) regenerates the JSON in memory and
asserts it matches the on-disk copy so drift is caught at CI time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import metalk  # noqa: E402
from cli import vowelstrip  # noqa: E402

OUT = ROOT / "web" / "metalk-data.json"


def build() -> dict:
    return {
        "structural": metalk.STRUCTURAL,
        "section_map": metalk.SECTION_MAP,
        "header_key_map": metalk.HEADER_KEY_MAP,
        "abbrev": metalk.ABBREV,
        "strip_words": sorted(metalk.STRIP_WORDS),
        "phrase_rewrites": metalk.PHRASE_REWRITES,
        "protected_words": sorted(vowelstrip.PROTECTED_WORDS),
        "sentence_punct": sorted(vowelstrip._SENTENCE_PUNCT),
        "opaque_chars": sorted(vowelstrip._OPAQUE_CHARS),
        "vowels": sorted(vowelstrip.VOWELS),
    }


def main() -> int:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({sum(len(v) if hasattr(v, '__len__') else 1 for v in data.values())} entries across {len(data)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

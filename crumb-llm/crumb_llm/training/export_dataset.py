"""Export a directory of CRUMB files as a JSONL training dataset.

Each line is one JSON object. We emit instruction/response style records so
the corpus is immediately usable for SFT or distillation: the *instruction* is
a CrumbLLM task prompt, the *input* is the raw CRUMB, and *output* is left
empty (to be filled by a teacher model or a human). We also keep a ``text``
field with the raw CRUMB for plain language-modeling.

No heavy dependencies: this is pure stdlib.
"""

from __future__ import annotations

import json
from pathlib import Path

from crumb_llm.crumb.loader import load_crumb

# The instructions we want a scratch/SFT model to learn to perform.
_INSTRUCTIONS = {
    "summarize": "Summarize this CRUMB project memory.",
    "risks": "List and classify the risks in this CRUMB memory.",
    "next": "List the next actions implied by this CRUMB memory.",
}


def export_dataset(
    crumb_dir: str | Path,
    out: str | Path,
    instructions: bool = True,
) -> int:
    """Walk ``crumb_dir`` for ``.crumb`` files and write JSONL to ``out``.

    Returns the number of records written.
    """
    root = Path(crumb_dir)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for crumb_path in sorted(root.rglob("*.crumb")):
            try:
                doc = load_crumb(crumb_path)
            except Exception:
                continue  # let crumb-format's linter handle malformed files
            meta = {
                "source_path": str(crumb_path),
                "kind": doc.kind,
            }
            # Plain LM record.
            fh.write(
                json.dumps({"text": doc.raw_text, "meta": meta}, ensure_ascii=False)
                + "\n"
            )
            records += 1
            if instructions:
                for _task, instruction in _INSTRUCTIONS.items():
                    fh.write(
                        json.dumps(
                            {
                                "instruction": instruction,
                                "input": doc.raw_text,
                                "output": "",
                                "meta": meta,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    records += 1
    return records

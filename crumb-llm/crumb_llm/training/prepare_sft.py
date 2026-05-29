"""Prepare an exported JSONL dataset for supervised fine-tuning.

Reads the JSONL produced by ``export_dataset`` and turns instruction/input/
output records into flat SFT strings of the form::

    <instruction>\n\n<input>\n\n<output><|endoftext|>

Tokenization is optional and only happens if ``tiktoken`` is installed. The
plain-text SFT file is always produced.

Cannibalized from train-llm-from-scratch's SFT preparation step.
"""

from __future__ import annotations

import json
from pathlib import Path

EOT = "<|endoftext|>"


def _format_record(rec: dict) -> str | None:
    instruction = (rec.get("instruction") or "").strip()
    input_text = (rec.get("input") or "").strip()
    output = (rec.get("output") or "").strip()
    if not instruction or not output:
        # Skip unlabeled records — they have no target to learn from.
        return None
    parts = [instruction]
    if input_text:
        parts.append(input_text)
    parts.append(output)
    return "\n\n".join(parts) + EOT


def prepare_sft(
    dataset_jsonl: str | Path,
    out: str | Path,
    tokenize: bool = False,
) -> dict:
    """Convert exported JSONL into SFT examples.

    Returns a summary dict with counts. When ``tokenize`` is True and
    ``tiktoken`` is available, also writes a ``.tokens.json`` file.
    """
    src = Path(dataset_jsonl)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples: list[str] = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            formatted = _format_record(rec)
            if formatted:
                examples.append(formatted)

    out_path.write_text("\n".join(examples), encoding="utf-8")
    summary = {"examples": len(examples), "out": str(out_path), "tokenized": False}

    if tokenize:
        try:
            import tiktoken
        except Exception:
            summary["tokenize_error"] = (
                "tiktoken not installed (pip install 'crumb-llm[scratch]')"
            )
            return summary
        enc = tiktoken.get_encoding("r50k_base")
        tokens: list[int] = []
        for ex in examples:
            tokens.extend(enc.encode(ex, allowed_special={EOT}))
        tok_path = out_path.with_suffix(".tokens.json")
        tok_path.write_text(json.dumps(tokens), encoding="utf-8")
        summary["tokenized"] = True
        summary["tokens"] = len(tokens)
        summary["tokens_out"] = str(tok_path)

    return summary

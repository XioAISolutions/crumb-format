#!/usr/bin/env python3
"""Does the next model still *use* what compression kept?

`crumb measure` reports **lexical** retention: whether a load-bearing token
survives into the compressed form. That is a recoverability metric, and the
literature is explicit that recoverability is not the same question as whether
the compressed context still works. Compaction can fail through attention
rather than deletion — information present but unused, "lost in compaction".
A format that scores well on retention and badly here would look fine under
every number this repository published before this file existed.

So this harness asks the harder question, in the shape the prior work uses:

  1. Sample load-bearing facts across the **whole** transcript, stratified into
     thirds, so no strategy scores merely on where a probe came from.
  2. Build **cloze probes** from them: the sentence that carried the fact, with
     the fact blanked out. Answering one requires finding the fact in whatever
     context the strategy produced.
  3. Answer every probe under each condition — full transcript, crumb,
     truncation — and grade by normalised exact match.
  4. Report success **with** cost, because completion alone is not a
     comparison: a strategy that keeps everything trivially wins on success
     and loses on tokens.

Paired conditions are the point. A probe that the full transcript answers and
the compressed context does not is a concrete, inspectable failure — the same
pairing ACON uses to find what a compressor should have kept.

**Probe generation and grading are deterministic.** No LLM writes the
questions and no LLM judges the answers, so a run reproduces exactly and two
people comparing results are comparing the same thing. Only the answering step
needs a model.

    python benchmarks/task_success.py --provider oracle      # harness self-test
    python benchmarks/task_success.py --provider ollama --model llama3
    python benchmarks/task_success.py --provider anthropic --model claude-sonnet-5

`oracle` is **not a result.** It answers by substring search over the context,
so it measures pure recoverability — the ceiling a real model cannot exceed.
It exists to prove the harness works and to bound what any model could score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli import measure as measure_mod  # noqa: E402
from cli import transcripts  # noqa: E402
import compare  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus"

# Probes are sampled across the WHOLE transcript, stratified into thirds.
#
# The first version of this file drew them only from the early turns, on the
# reasoning that those are what compression drops. That silently rigged the
# comparison: `head-truncate-10` *keeps* the first ten turns, so it scored on
# probes it was structurally guaranteed to answer, while every recency strategy
# scored zero for the same structural reason. The number measured where a probe
# came from, not whether a strategy preserved anything worth having.
#
# Sampling all three positions removes that bias and makes the result readable:
# each strategy wins in the region it keeps, and the overall figure is the one
# that answers "could the next agent continue the work".
POSITIONS = ("early", "middle", "late")

# Facts shorter than this match too easily by chance in a long context.
MIN_FACT_LENGTH = 4

# A real session repeats one shape of identifier many times (module_00,
# module_01, ...). Without a cap, one family supplies every probe and the
# benchmark measures a single fact type.
MAX_PER_FAMILY = 2


@dataclass
class Probe:
    """One cloze question whose answer lives in the dropped region."""

    question: str
    expected: str
    category: str
    turn: int
    position: str = "early"

    def prompt(self, context: str) -> str:
        return (
            "You are resuming another agent's session. Using ONLY the context "
            "below, fill in the blank.\n"
            "Answer with the missing value alone — no explanation, no "
            "punctuation, no full sentence. If the context does not contain "
            "it, answer exactly: UNKNOWN\n\n"
            f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
            f"{self.question}\n\nMissing value:"
        )


@dataclass
class ConditionResult:
    condition: str
    tokens: int = 0
    answered: int = 0
    total: int = 0
    missed: List[str] = field(default_factory=list)

    @property
    def success(self) -> float:
        return (self.answered / self.total) if self.total else 0.0


# ── probe construction ───────────────────────────────────────────────

SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def _position_of(turn: int, total: int) -> str:
    """Which third of the conversation a turn falls in."""
    if total <= 0:
        return "early"
    third = turn / total
    if third < 1 / 3:
        return "early"
    return "middle" if third < 2 / 3 else "late"


def _family(fact: str) -> str:
    """Collapse an enumerated identifier family: module_07 -> module_##."""
    return re.sub(r"\d+", "#", fact)


def build_probes(messages: Sequence[transcripts.Message], limit: int = 12) -> List[Probe]:
    """Cloze probes over load-bearing facts, stratified across the transcript.

    Deterministic: same transcript in, same probes out, so a published number
    can be re-derived by anyone holding the corpus.
    """
    total = len(messages)
    candidates: Dict[str, List[Probe]] = {p: [] for p in POSITIONS}
    seen: set[str] = set()
    family_counts: Dict[str, int] = {}

    for turn, message in enumerate(messages):
        text = message.text
        if not text.strip():
            continue
        for category, facts in measure_mod.extract_facts(text).items():
            for fact in sorted(facts):
                if len(fact) < MIN_FACT_LENGTH or fact in seen:
                    continue
                family = _family(fact)
                if family_counts.get(family, 0) >= MAX_PER_FAMILY:
                    continue
                sentence = _sentence_containing(text, fact)
                if not sentence:
                    continue
                blanked = re.sub(re.escape(fact), "____", sentence, count=1, flags=re.IGNORECASE)
                if "____" not in blanked:
                    continue
                seen.add(fact)
                family_counts[family] = family_counts.get(family, 0) + 1
                position = _position_of(turn, total)
                candidates[position].append(Probe(
                    question=f"Complete this line from the session:\n  {blanked.strip()}",
                    expected=fact,
                    category=category,
                    turn=turn,
                    position=position,
                ))

    # Round-robin across positions so no third dominates the sample.
    per_position = max(1, limit // len(POSITIONS))
    chosen: List[Probe] = []
    for position in POSITIONS:
        chosen.extend(candidates[position][:per_position])
    if len(chosen) < limit:  # backfill from whichever positions have more
        pool = [p for position in POSITIONS for p in candidates[position][per_position:]]
        pool.sort(key=lambda p: (p.turn, p.expected))
        chosen.extend(pool[: limit - len(chosen)])

    chosen.sort(key=lambda p: (p.turn, p.category, p.expected))
    return chosen[:limit]


def _sentence_containing(text: str, fact: str) -> str:
    lowered = fact.lower()
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if lowered in sentence.lower():
            # A sentence that is only the fact gives the model nothing to
            # locate it by once blanked.
            if len(sentence) > len(fact) + 12:
                return sentence
    return ""


# ── grading ──────────────────────────────────────────────────────────

def normalise_answer(text: str) -> str:
    """Strip the wrapping a model adds without changing the answer itself."""
    answer = text.strip().splitlines()[0] if text.strip() else ""
    answer = answer.strip().strip("`\"'*")
    answer = re.sub(r"^(the |a |an |answer:?\s*|missing value:?\s*)", "", answer, flags=re.IGNORECASE)
    return answer.strip().rstrip(".,;:!?").strip().lower()


def grades(answer: str, expected: str) -> bool:
    given = normalise_answer(answer)
    want = expected.strip().lower()
    if not given or given == "unknown":
        return False
    # Exact, or the expected value surrounded by the model's own phrasing.
    return given == want or want == normalise_answer(given)


# ── providers ────────────────────────────────────────────────────────
# Answering is the only step that needs a model. Each provider takes a prompt
# and returns raw text; everything else in this file is deterministic.

class ProviderError(RuntimeError):
    """The model could not be reached or refused the request."""


def provider_oracle(prompt: str, model: str, probe: Probe) -> str:
    """Substring lookup, not a model. The recoverability ceiling.

    It is handed the answer key — that is the point, and the signature says so
    rather than smuggling it in. It answers a probe if and only if the expected
    value is literally present in the context, which is the best any model
    could do. So it bounds every real result and proves the harness end to end,
    but it is not a score, and `format_report` refuses to present it as one.
    """
    context = prompt.split("--- CONTEXT ---", 1)[-1].split("--- END CONTEXT ---", 1)[0]
    return probe.expected if probe.expected.lower() in context.lower() else "UNKNOWN"


def _http_json(url: str, payload: dict, headers: Dict[str, str], timeout: int = 120) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"{url} returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"could not reach {url}: {exc}") from exc


def provider_ollama(prompt: str, model: str, probe: Probe) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    data = _http_json(f"{host}/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0},
    })
    return str(data.get("response", ""))


def provider_anthropic(prompt: str, model: str, probe: Probe) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ProviderError("ANTHROPIC_API_KEY is not set")
    data = _http_json("https://api.anthropic.com/v1/messages", {
        "model": model, "max_tokens": 64, "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }, {"x-api-key": key, "anthropic-version": "2023-06-01"})
    blocks = data.get("content", [])
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))


def provider_openai(prompt: str, model: str, probe: Probe) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY is not set")
    data = _http_json("https://api.openai.com/v1/chat/completions", {
        "model": model, "temperature": 0.0, "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }, {"Authorization": f"Bearer {key}"})
    return str(data["choices"][0]["message"]["content"])


PROVIDERS: Dict[str, Callable[[str, str, "Probe"], str]] = {
    "oracle": provider_oracle,
    "ollama": provider_ollama,
    "anthropic": provider_anthropic,
    "openai": provider_openai,
}

# Conditions compared. Named from compare.STRATEGIES so the two harnesses
# cannot drift into measuring different things under the same name.
CONDITIONS = ("full-transcript", "crumb", "head-truncate-10", "recency-window-10")


def run(
    provider: str = "oracle",
    model: str = "",
    conditions: Sequence[str] = CONDITIONS,
    corpus: Path | None = None,
    limit: int = 12,
) -> List[dict]:
    answer = PROVIDERS[provider]
    corpus_dir = corpus or CORPUS
    files = sorted(corpus_dir.glob("*.json")) + sorted(corpus_dir.glob("*.jsonl"))

    rows: List[dict] = []
    for path in files:
        fmt, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
        probes = build_probes(messages, limit=limit)
        if not probes:
            continue
        for name in conditions:
            context = compare.STRATEGIES[name](messages, fmt)
            tokens, tokenizer = measure_mod.count_tokens(context)
            result = ConditionResult(condition=name, tokens=tokens, total=len(probes))
            by_position: Dict[str, List[int]] = {p: [0, 0] for p in POSITIONS}
            for probe in probes:
                try:
                    reply = answer(probe.prompt(context), model, probe)
                except ProviderError as exc:
                    raise SystemExit(f"task_success: {exc}")
                bucket = by_position[probe.position]
                bucket[1] += 1
                if grades(reply, probe.expected):
                    result.answered += 1
                    bucket[0] += 1
                else:
                    result.missed.append(probe.expected)
            rows.append({
                "file": path.name,
                "condition": name,
                "probes": result.total,
                "answered": result.answered,
                "success": round(result.success, 3),
                "tokens": tokens,
                "tokenizer": tokenizer,
                "by_position": {k: {"answered": v[0], "probes": v[1]}
                                for k, v in by_position.items() if v[1]},
                "missed": result.missed[:5],
                "provider": provider,
                "model": model or None,
            })
    return rows


def format_report(rows: List[dict]) -> str:
    if not rows:
        return "No probes could be built from this corpus."
    provider = rows[0]["provider"]
    out: List[str] = []
    for path in dict.fromkeys(r["file"] for r in rows):
        subset = [r for r in rows if r["file"] == path]
        out.append(f"\n{path}  ({subset[0]['probes']} probes, sampled across the transcript)")
        out.append(
            f"  {'condition':<24} {'tokens':>8} {'answered':>9} {'success':>8}  "
            f"{'early':>6} {'mid':>6} {'late':>6}   missed"
        )
        out.append("  " + "-" * 104)
        for r in sorted(subset, key=lambda r: -r["success"]):
            missed = ", ".join(r["missed"][:2]) or "—"
            pos = r.get("by_position", {})
            def cell(name: str) -> str:
                v = pos.get(name)
                return f"{v['answered']}/{v['probes']}" if v else "—"
            out.append(
                f"  {r['condition']:<24} {r['tokens']:>8,} "
                f"{r['answered']:>4}/{r['probes']:<4} {r['success'] * 100:>7.1f}%  "
                f"{cell('early'):>6} {cell('middle'):>6} {cell('late'):>6}   {missed[:26]}"
            )
    out.append("")
    if provider == "oracle":
        out.append(
            "PROVIDER=oracle — this is NOT a model result. The oracle answers by substring\n"
            "lookup, so it reports pure recoverability: the ceiling a real model cannot\n"
            "exceed. Re-run with --provider ollama/anthropic/openai for a task-success\n"
            "number, and only that number belongs in a claim."
        )
    else:
        out.append(
            f"Provider {provider}, model {rows[0]['model']}. Probes and grading are\n"
            "deterministic; only the answering step used a model, at temperature 0."
        )
    out.append(
        "Probes are sampled across the whole transcript in thirds, so no strategy scores\n"
        "on where a probe came from: head-truncation keeps the early turns and recency\n"
        "keeps the late ones, and the per-position columns show exactly that.\n"
        "Success is reported with tokens because completion alone is not a comparison."
    )
    return "\n".join(out)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="oracle", choices=sorted(PROVIDERS))
    parser.add_argument("--model", default="", help="Model name for the chosen provider.")
    parser.add_argument("--limit", type=int, default=12, help="Max probes per transcript.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--dump-probes", action="store_true",
                        help="Print the generated probes and exit, without answering them.")
    args = parser.parse_args(argv)

    if args.dump_probes:
        files = sorted(CORPUS.glob("*.json")) + sorted(CORPUS.glob("*.jsonl"))
        dump = []
        for path in files:
            _, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
            for probe in build_probes(messages, limit=args.limit):
                dump.append({"file": path.name, "turn": probe.turn,
                             "category": probe.category, "expected": probe.expected,
                             "question": probe.question})
        print(json.dumps(dump, indent=2))
        return 0

    if args.provider != "oracle" and not args.model:
        parser.error(f"--model is required for --provider {args.provider}")

    rows = run(provider=args.provider, model=args.model, limit=args.limit)
    print(json.dumps(rows, indent=2) if args.json else format_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# CRUMB Context-Handoff Benchmark — submission metadata

A single page holding what a benchmark directory or index typically asks for,
so a submission is copy-paste rather than a rewrite. Keep it accurate: every
claim below is checkable from this repository in under a minute, and that is
the only reason it is worth submitting.

| Field | Value |
| --- | --- |
| **Name** | CRUMB Context-Handoff Benchmark |
| **What it measures** | What context compression costs when a session is handed from one agent to another — token savings, lexical fact retention, and probe-based task success |
| **Task type** | Context compression / long-horizon agent handoff / long-context utilisation |
| **Modality** | Text (agent conversation transcripts) |
| **Subject of evaluation** | **Compression strategies, not models.** A model appears only as the answerer in the task-success harness |
| **Corpus** | 7 committed transcripts across 5 ingest formats: OpenAI Chat Completions, Anthropic Messages, ChatGPT export, Claude.ai export, Claude Code JSONL. Includes a 159-message long session and a tool-traffic-dominated session |
| **Metrics** | (1) token savings vs. full transcript; (2) lexical fact retention by category — paths, identifiers, URLs, error codes, numbers, quoted strings; (3) probe task success with per-position breakdown; all reported alongside token cost |
| **Baselines** | full transcript, head truncation, recency windows (4 and 10), Anthropic-style tool-result clearing, CRUMB |
| **Determinism** | Probe generation, grading, and all compression strategies are deterministic. Only the task-success answering step uses a model, at temperature 0 |
| **API key required** | **No** for the compression comparison and the recoverability ceiling. Yes only for a real-model task-success run |
| **Runtime** | ~2 seconds for the full offline comparison |
| **Licence** | MIT |
| **Repository** | https://github.com/XioAISolutions/crumb-format |
| **Entry points** | `benchmarks/compare.py`, `benchmarks/task_success.py` |
| **Third-party submissions** | Yes — `benchmarks/external/` accepts real output from any handoff tool, measured through the same code |
| **Leaderboard** | None. This repository publishes no ranking |

## Reproduction

```bash
git clone https://github.com/XioAISolutions/crumb-format
cd crumb-format
python benchmarks/compare.py                                # compression comparison, offline
python benchmarks/task_success.py --provider oracle         # recoverability ceiling
python benchmarks/task_success.py --provider ollama --model llama3   # a real result
```

No install step: the CLI is stdlib-only and runs from the checkout.

## What this benchmark does *not* claim

Stated up front because a directory entry that overclaims is worse than no
entry:

- **No model is scored.** This evaluates compression strategies. It is not a
  model leaderboard and should not be indexed as one.
- **No task-success result has been published yet.** The committed numbers are
  a recoverability *ceiling* from a substring oracle, not a model run.
- **Lexical retention is a proxy** for information survival, not semantic
  fidelity, and every report says so.
- **CRUMB does not win on every axis, and the harness proves it.** A recency
  window matches CRUMB at 56% of the tokens on the long session and beats it on
  the tool-heavy one. The test suite fails if no strategy ever beats CRUMB —
  see `tests/test_compare.py` and `tests/test_task_success.py`.
- **n is small** (5–6 probes per transcript) and no confidence intervals are
  reported, because none would be meaningful at that n.

## Positioning against existing work

Not a competitor to token-level compressors such as LLMLingua, which optimise a
different objective (perplexity-scored token dropping). Closest in spirit to
[ACON](https://arxiv.org/abs/2510.00615) and the long-horizon agent-memory
evaluations, and it deliberately reuses their framing: paired conditions, and
success reported with cost.

The distinguishing property is **inspectability** — every row names the specific
facts it dropped, the corpus is committed rather than generated at runtime, and
the whole comparison reproduces offline in two seconds with no key. Full
methodology and limitations: [`docs/TASK_SUCCESS.md`](../docs/TASK_SUCCESS.md)
and [`docs/BENCHMARK.md`](../docs/BENCHMARK.md).

# Benchmark corpus

Six committed transcripts and a runner that measures what compressing each one
actually costs.

```bash
python benchmarks/run_corpus.py           # human table
python benchmarks/run_corpus.py --json    # machine-readable
```

```text
file                         format                    msgs           tokens   saved  retain
--------------------------------------------------------------------------------------------
short-anthropic.json         anthropic-messages          39   2,804→395        85.9%  100.0%
short-chatgpt-export.json    chatgpt-export              39   2,804→393        86.0%  100.0%
short-claude-export.json     claude-export               39   2,804→393        86.0%  100.0%
short-openai.json            openai-messages             39   2,804→394        85.9%  100.0%
short-claude-code.jsonl      claude-code-transcript      39   2,804→397        85.8%  100.0%
long-claude-code.jsonl       claude-code-transcript     159   3,705→547        85.2%   30.5%
```

## What the corpus is for

**Format independence.** The five `short-*` files are *the same conversation*
exported five different ways. They land within four tokens of each other at
identical retention — which is the property CRUMB needs and the one nothing
else would catch. A reader bug in any single format shows up here as divergence.

**Regression detection.** Every result is compared against `baseline.json`.
An extractor change that quietly drops facts fails `tests/test_benchmarks.py`
rather than shipping. The guards are themselves tested for their ability to go
red; a check that cannot fail is decoration.

**Length.** Every extraction fault found so far appeared only at length — fixed
item caps discarding most of a long session, head-truncation describing stale
state, markdown headings and narration mistaken for content. A 40-turn fixture
exercises none of that, so the corpus carries a 159-turn one.

## Reading the retention column

Retention is the share of load-bearing tokens — paths, URLs, identifiers, error
codes, numbers with units — that survive into the handoff. It is a **lexical
proxy**: it reports *what* was dropped, never whether the next model can still
do the work.

It falls with length by design. A 50k-token session cannot be carried
losslessly in a 2.4k handoff, and the honest response is to report that rather
than raise the caps until the number looks better — which would buy a prettier
metric and a bloated crumb. The long fixture sits at 30.5% and is expected to.

Whether the *right* 30.5% survived is a question only a task-success benchmark
answers, and that is the natural next thing to build here.

## Regenerating

The corpus is committed rather than generated at test time, so `baseline.json`
describes fixed inputs. `build_corpus.py` exists so it can be regenerated and
reviewed as a diff:

```bash
python benchmarks/build_corpus.py              # deterministic; no diff if unchanged
python benchmarks/run_corpus.py --update-baseline
```

Update the baseline only when a change is *meant* to move the numbers, and say
why in the commit. That file is the record of what the tool was measured to do.

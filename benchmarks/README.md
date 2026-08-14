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

## Comparing strategies

`compare.py` runs several context-compression strategies over the same corpus
and reports, for each, what it dropped:

```bash
python benchmarks/compare.py
```

```text
long-claude-code.jsonl  (159 messages)
  strategy                         tokens    saved  retained   example losses
  recency-window-4                    169    95.4%      6.1%   module_00, module_01, module_02
  head-truncate-10                    251    93.2%      6.1%   1200ms, 502, https://example.com/docs/ledger
  recency-window-10                   306    91.7%      9.8%   module_00, module_01, module_02
  crumb                               547    85.2%     30.5%   module_00, module_01, module_02
  full-transcript                   3,705     0.0%    100.0%   —
```

Every strategy is **deterministic and offline**, so the numbers reproduce
exactly and cost nothing. A comparison that needs API keys is one nobody
re-runs, and one that cannot be re-run is indistinguishable from a claim.

**`*-style` rows replicate a documented mechanism, not a vendor's
implementation.** The hosted versions also summarise; these do not. No row here
is a vendor's score, and the table says so.

### What it shows, including where CRUMB loses

On **long sessions**, CRUMB retains 30.5% where truncation retains 6–10% at
comparable compression — and head-truncation loses the root cause outright
(`1200ms`, `502`, the spec URL), because in a real session the conclusion is at
the end.

On **tool-heavy sessions**, Anthropic-style tool-result clearing retains **62%
where CRUMB retains 27%**, at far lower compression (17.8% vs 66.8% saved).
CRUMB does not win that one, and the harness is tested for its ability to say
so — `tests/test_compare.py` fails if no strategy beats CRUMB on any axis
anywhere, because a comparison built so its author always wins is marketing.

The column that actually differentiates is the last one. Every strategy here
can state what it dropped; hosted compaction does not expose that at all.

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

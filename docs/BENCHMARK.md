# What context compression actually costs

Six strategies, seven transcripts, one question: **when you compress an agent
session, what do you lose?**

Every number here is reproducible on your machine in under two seconds, with no
API key:

```bash
git clone https://github.com/XioAISolutions/crumb-format
cd crumb-format && pip install -e .
python benchmarks/compare.py
```

---

## The headline

On a long session, the strategy everyone reaches for first — keep the most
recent turns — throws away almost everything that made the session worth
keeping.

```text
long-claude-code.jsonl  (159 messages)
  strategy                         tokens    saved  retained   example losses
  recency-window-4                    169    95.4%      6.1%   module_00, module_01, module_02
  head-truncate-10                    251    93.2%      6.1%   1200ms, 502, https://…/ledger
  recency-window-10                   306    91.7%      9.8%   module_00, module_01, module_02
  crumb                               547    85.2%     30.5%   module_00, module_01, module_02
  full-transcript                   3,705     0.0%    100.0%   —
```

A structured handoff retains **3–5× more load-bearing detail** than truncation
at comparable compression.

Look at the `head-truncate-10` row specifically. It drops `1200ms`, `502`, and
the spec URL — which is to say it drops **the entire diagnosis**. In a real
session the conclusion is at the end, so head-truncation reliably keeps the
part where you were still confused and discards the part where you worked it
out.

## Where the structured handoff loses

```text
tools-claude-code.jsonl  (63 messages)
  strategy                         tokens    saved  retained
  recency-window-4                    162    89.2%     12.4%
  crumb                               497    66.8%     27.1%
  anthropic-style-clear-tools       1,232    17.8%     62.0%
  full-transcript                   1,499     0.0%    100.0%
```

On a tool-heavy session, clearing old tool results retains **62% where CRUMB
retains 27%** — at a quarter of the compression. If you can afford the tokens,
that is the better trade, and no amount of structure changes it.

This row is in the benchmark on purpose. `tests/test_compare.py` fails the
build if *no* strategy beats CRUMB on any axis anywhere, because a comparison
built so its author always wins is marketing, not evidence.

## Short sessions

```text
short-*.json  (39 messages, the same conversation in 5 export formats)
  recency-window-4                    340    87.9%     81.8%
  crumb                            393–397    85.9%    100.0%
  head-truncate-10                    738    73.7%     54.5%
  recency-window-10                   763    72.8%     90.9%
```

Recency wins on ratio by 2 points and loses 18% of the facts. The interesting
result is the spread across the five `crumb` rows: **393, 393, 394, 395, 397
tokens** — the same conversation exported by OpenAI, Anthropic, ChatGPT,
Claude.ai and Claude Code, landing within four tokens of each other at
identical retention. Compression that depends on which vendor exported your
transcript is not compression you can rely on.

## Method

**Retention** is the share of load-bearing tokens — file paths, URLs,
identifiers, error codes, numbers with units, quoted strings — that survive
into the compressed form. It is computed by extracting those tokens from the
source and checking containment in the output.

**It is a lexical proxy.** It tells you *what* was dropped. It does not tell
you whether the next model can still do the work. Treat it as a regression
guard, not a score. A task-success benchmark is the honest next step and is not
claimed here.

**Token counts** come from `tiktoken` when installed, and from an explicitly
labelled `chars/4` heuristic when not. The report always names which produced
the number.

### The `*-style` rows are mechanisms, not vendors

`anthropic-style-clear-tools` replicates the *documented mechanism* of tool-result
clearing — keep the most recent N tool uses, replace older results with a
placeholder. It is **not** a call to Anthropic's API, and the hosted version
also summarises, which this does not. No row here is a vendor's score, and none
should be quoted as one.

This constraint is deliberate: every strategy is deterministic and offline, so
the numbers reproduce exactly and cost nothing. A comparison that needs API
keys is a comparison nobody re-runs, and one that cannot be re-run is
indistinguishable from a claim.

## Why retention falls with length

The long session retains 30.5%, the short ones 100%. That is not a defect being
hidden — a 50k-token session cannot be carried losslessly in a 2.4k handoff,
and the arithmetic does not care how the compressor is written.

The caps that decide *which* facts survive could be raised until that number
looked respectable. They deliberately are not. That would buy a prettier metric
and a bloated handoff, and the point of publishing a benchmark is to report
what the tool does rather than what it would be nice for it to do.

## Audit your own compression

None of this is specific to CRUMB. The same measurement runs against any
compressed context — a provider's compaction output, an LLM summary, a
truncated transcript:

```bash
crumb verify their-compaction-output.txt --source session.jsonl --label "vendor compaction"
```

```text
vendor compaction audit — their-compaction-output.txt vs session.jsonl
  Saved:             97.2%  (35.7x smaller)
  Fact retention:    41.0%  (25/61 load-bearing tokens kept)
    paths           43.8%  (14/32)  lost: src/api/handler.py, …
```

Hosted compaction generally cannot tell you this, because you cannot read the
compressed form to check. That is the gap worth caring about — more than the
ratio.

## Corpus

Seven transcripts in `benchmarks/corpus/`, committed rather than generated, so
the baseline describes fixed inputs:

| File | Shape |
| --- | --- |
| `short-openai.json`, `short-anthropic.json`, `short-chatgpt-export.json`, `short-claude-export.json`, `short-claude-code.jsonl` | one 39-message conversation, five export formats |
| `long-claude-code.jsonl` | 159 messages; conclusion at the end |
| `tools-claude-code.jsonl` | 63 messages, tool-traffic dominated |

The tool-heavy fixture exists because a sample of a real Claude Code transcript
ran 427 `tool_use` and 426 `tool_result` blocks against 162 text blocks. Tool
traffic is the majority of a real agent session; a corpus without it makes
tool-clearing strategies look free.

## Measuring other tools, not just other strategies

Everything above is a strategy *this repository implements*, and that is a real
limit. Replicating a competitor's documented mechanism and then scoring it is
not the same as measuring the competitor — which is why those rows are labelled
`*-style` and why none of them carries a vendor's name as though it were a
vendor's result.

The corpus is what makes the honest version possible. It holds **committed
transcripts**, so any tool that turns a conversation into a handoff can be run
over the same bytes, and two people can compare results without either trusting
the other's harness. Publishing the inputs matters more than publishing a score.

`benchmarks/external/<tool>/<corpus-stem>.txt` is where real output goes. Drop
in what the tool would actually hand the next agent, add a `SOURCE.md` naming
the version and command, and re-run `compare.py`. Those rows are measured by
the same `cli/measure.py` as everything else and carry no `-style` suffix,
because they are not approximations. The full protocol is in
[`benchmarks/external/README.md`](../benchmarks/external/README.md).

**Cases nobody has supplied are printed as `—`, never skipped.** An unmeasured
transcript is a fact about how far the comparison has actually got, and hiding
it would let the table imply coverage it does not have.
`tests/test_compare.py` fails if a missing case is silently dropped.

### What is not measured yet

| Tool | Status |
| --- | --- |
| `ai-memory` | not measured — session pages are plain files (`sessions/<id>.md`), so this is capturable; the handoff records themselves live in SQLite behind MCP |
| OpenAI compaction, Anthropic context editing | **not capturable** — the provider compacts server-side and never returns the compacted context, which is why only the documented mechanisms are replicated above |

That first row is an open invitation, not an oversight. If you maintain a
handoff tool and think this measures you unfairly, the corpus and the
measurement code are both here — open a PR with your artifacts. Results that
show CRUMB losing get merged; `tests/test_compare.py` already fails if no
strategy beats CRUMB on any axis anywhere, because a comparison its author
always wins is marketing rather than measurement.

## Reproducing and extending

```bash
python benchmarks/compare.py --json          # machine-readable
python benchmarks/run_corpus.py              # CRUMB only, vs committed baseline
python benchmarks/build_corpus.py            # regenerate the corpus (deterministic)
```

Adding a strategy is a function in `benchmarks/compare.py` that takes messages
and returns the text a model would receive. If you think a strategy would beat
these, add it — the harness is built to let it.

# Does the next model still *use* what compression kept?

`crumb measure` reports **lexical retention**: whether a load-bearing token —
a path, an error code, an identifier — survives into the compressed form. Every
report this repository publishes says, in those words, that this is a proxy.

It is a *recoverability* metric, and the prior work is explicit that
recoverability is not the same question as whether the compressed context still
works. Compaction can fail through attention rather than deletion: information
that is present but unused, which the literature calls **"lost in
compaction"**. A format could score well on retention and badly on the thing
anyone actually cares about, and nothing in this repository would have noticed.

`benchmarks/task_success.py` asks the harder question.

## Method

1. **Sample load-bearing facts across the whole transcript**, stratified into
   thirds (early / middle / late).
2. **Build cloze probes**: the sentence that carried the fact, with the fact
   blanked out. Answering one requires locating that fact in whatever context
   the strategy produced.
3. **Answer every probe under each condition** — full transcript, crumb,
   head-truncation, recency window — at temperature 0.
4. **Grade by normalised exact match**, and report success **with token cost**.

Probe generation and grading are **deterministic**. No model writes the
questions and no model judges the answers, so a run reproduces exactly and two
people comparing results are comparing the same thing. Only the answering step
needs a model.

```bash
python benchmarks/task_success.py --provider oracle              # harness self-test
python benchmarks/task_success.py --provider ollama   --model llama3
python benchmarks/task_success.py --provider anthropic --model claude-sonnet-5
python benchmarks/task_success.py --dump-probes                  # inspect the questions
```

### Why stratified sampling is load-bearing

The first version of this harness drew probes only from the early turns, on the
reasoning that early turns are what compression drops. That silently rigged the
comparison. `head-truncate-10` *keeps* the first ten turns, so it scored on
probes it was structurally guaranteed to answer, and every recency strategy
scored zero for the same structural reason. The number measured **where a probe
came from**, not what a strategy preserved.

`tests/test_task_success.py::test_probes_are_not_all_from_one_position` fails
if that returns. It is mutation-tested: reintroducing early-only sampling turns
it red.

A per-family cap exists for the same reason — a real session repeats
`module_00, module_01, …`, and without a cap one identifier shape supplies
every probe.

## What the oracle shows

The `oracle` provider is **not a model**. It answers by substring lookup, so it
reports pure recoverability: the ceiling no model can exceed. It exists to
prove the harness end to end and to bound any real result.

Even at that ceiling, CRUMB does not win:

```
long-claude-code.jsonl  (6 probes, sampled across the transcript)
  condition                  tokens  answered  success   early    mid   late
  full-transcript             3,705    6/6      100.0%     2/2      —    4/4
  crumb                         547    4/6       66.7%     0/2      —    4/4
  recency-window-10             306    4/6       66.7%     0/2      —    4/4
  head-truncate-10              251    2/6       33.3%     2/2      —    0/4

tools-claude-code.jsonl  (5 probes, sampled across the transcript)
  full-transcript             1,499    5/5      100.0%       —      —    5/5
  recency-window-10             297    5/5      100.0%       —      —    5/5
  crumb                         497    4/5       80.0%       —      —    4/5
  head-truncate-10              239    1/5       20.0%       —      —    1/5
```

**A recency window matches CRUMB on the long session at 56% of the tokens, and
beats it outright on the tool-heavy one.** That is the opposite of what
`compare.py`'s lexical retention suggests, where CRUMB leads truncation 30.5%
to 6.1% on the same long transcript.

Both numbers are real. They disagree because they measure different things over
different denominators: retention scores every fact in the source, while these
probes are a small stratified sample. The disagreement is the most useful thing
in this document — it is precisely why a retention number should never have
been presented as a stand-in for task success, and why this harness exists.

## Limitations, stated plainly

- **These are not model results.** No run against a real model has been
  published. The table above is the recoverability ceiling. Until a model run
  exists, no task-success claim should be made from this repository.
- **n is small.** Five to six probes per transcript. Fine for detecting a large
  effect, useless for a small one. No confidence intervals are reported because
  none would be meaningful at this n.
- **The middle third is often empty.** The committed corpus is partly
  synthetic, and its middle turns largely repeat facts introduced earlier,
  which deduplication removes. A corpus of real, varied sessions would fix this
  and is the most valuable single improvement available.
- **Cloze probes test fact recovery, not reasoning.** They are a lower bound on
  what "continuing the work" requires. A model can fill every blank and still
  fail to resume the task.
- **Exact match under-credits a correct paraphrase.** Deliberate: an LLM judge
  would make the result non-reproducible, which costs more than it buys.

## Prior art

This design borrows from published work rather than inventing a methodology:

- **[ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/abs/2510.00615)**
  — paired trajectories where full context succeeds and compressed context
  fails, used to find what a compressor should have kept. The paired-condition
  structure here is the same idea.
- **[What Does Context Compression Cost an Agent?](https://arxiv.org/html/2608.16370)**
  — task completion alone is insufficient; success must be paired with cost.
  Hence the token column beside every success figure.
- **LLMLingua** — token-level prompt compression scored by a small model's
  perplexity. A different mechanism to CRUMB's semantic restructuring, and not
  a competitor on this axis; CRUMB should not claim to beat token-level
  compressors on compression ratio.
- **[MemGym](https://arxiv.org/pdf/2605.20833)** and long-horizon memory
  environments — the direction this harness should grow toward: real tasks with
  real success criteria, rather than fact recovery.

## Contributing a result

Run the harness against a real model and open a PR with the JSON output and the
exact command. Results that show CRUMB losing get merged — the suite already
fails if no condition ever matches or beats CRUMB, for the same reason
`benchmarks/external/` prints unmeasured cases as `—` instead of hiding them.

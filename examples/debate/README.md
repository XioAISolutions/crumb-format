# Multi-model debate with CRUMB

A worked example of the **debate** handoff pattern: the same question goes to N models, each emits a `kind=mem` answer, and a reducer produces a final `kind=mem` that preserves real disagreement instead of averaging it away.

## The flow

```
architect (human)
  |
  |  01-question.crumb              kind=task
  |
  +-----+-----------+-----------+
        |           |           |
        v           v           v
   [Claude]     [GPT]       [Gemini]
        |           |           |
        v           v           v
  02-answer-   03-answer-   04-answer-
  claude.crumb gpt.crumb    gemini.crumb
  kind=mem     kind=mem     kind=mem
        |           |           |
        +-----+-----+-----------+
              |
              v
        [reducer agent]
              |
              v
        05-reduced.crumb           kind=mem
```

## Why this shape beats a single-model answer

- **Disagreement surfaces architecture choices you wouldn't see from one voice.** In this example, all three models have idempotency and dead-lettering in common, but disagree sharply on *where* retries should live (handler vs. queue). The reducer preserves that tension.
- **Citations become a cross-check.** Each responder cites prior art (AWS, Google SRE, Newman). If one model cites a source the others don't recognize, that's a signal to verify.
- **The reducer is auditable.** `05-reduced.crumb` carries `ext.debate.answers=<id1,id2,id3>` — any auditor can pull up the inputs to the synthesis and check the reducer didn't omit an important view.

## The extension convention

Each answer crumb carries:

```text
extensions=ext.debate.question.v1
ext.debate.question=<id of the question crumb>
```

The reducer carries both `ext.debate.question=` and `ext.debate.answers=<comma-separated ids>`. Namespaced (`ext.debate.*`), so any CRUMB parser ignores them safely per SPEC §8.1, but a debate-aware tool can walk them.

## Protocol rules for responders

1. MUST emit `kind=mem` (not `task`) — an answer is durable output, not a new request.
2. MUST NOT edit another responder's crumb. Each answer stands alone.
3. MUST cite at least one verifiable prior-art source.
4. SHOULD surface disagreement with other responders explicitly when relevant — the reducer relies on this to avoid a synthesis that papers over real tradeoffs.

## Protocol rules for the reducer

1. MUST reference every answer crumb it considered in `ext.debate.answers=`.
2. MUST preserve genuine disagreement under a "disagreement" heading. Averaging-it-away is a debate-pattern failure.
3. SHOULD flag residual risks — areas none of the responders covered — as their own follow-up `task` or callout in the synthesis.
4. SHOULD NOT introduce new facts that were not in any responder's answer.

## Validate the whole set

```bash
crumb validate examples/debate/*.crumb
```

## When to use this vs. orchestration

- Use **debate** (this dir) when the question is open-ended and disagreement is valuable.
- Use **orchestration** (`examples/orchestration/`) when the work decomposes into specialized sequential steps.
- Combine them: a debate can be one stage *inside* an orchestration pipeline — e.g., "retrieve" then "debate the synthesis" then "cite."

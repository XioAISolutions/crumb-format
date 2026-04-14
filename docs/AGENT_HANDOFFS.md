# Agent handoff patterns

`docs/HANDOFF_PATTERNS.md` covers the four single-crumb shapes (task / mem / map / cross-tool). This document is the next layer up: the patterns that emerge when **multiple agents** pass crumbs to each other to do real work.

Every pattern here uses only `v=1.1` CRUMB — no new headers, no new sections. The multi-agent shape lives in how crumbs reference each other via namespaced extensions (SPEC §8.1).

---

## The upstream-pointer convention

Every pattern below uses a shared idiom for saying "this crumb is a reply to that crumb":

```text
extensions=ext.<project>.upstream.v1
ext.<project>.upstream=<id of the prior crumb>
```

- Namespaced under `ext.<project>.*` so any parser ignores it safely.
- The pointer is **by `id=`**, not by filename or path — a chain can survive rename and transport.
- Walking pointers lets a replay tool reconstruct the full chain.
- A final "answer" crumb MAY additionally carry `ext.<project>.chain=<id1,id2,...>` listing every crumb in the chain, so a single artifact is enough for audit without walking.

You don't have to use the word "upstream." Pick a verb that fits your domain (`ext.debate.question=`, `ext.rag.source=`, `ext.workflow.parent=`). The rule is: namespaced, pointer-by-id, stable across transports.

---

## Pattern 1 — Linear pipeline

Specialized agents do sequential steps. Each agent's output is the next agent's input.

```
user --task--> A --task--> B --task--> C --mem--> user
```

**When to use.** The work decomposes cleanly into stages, each with a different specialization (ingest, retrieve, cite; parse, plan, execute; research, draft, review).

**Crumb types by position:**
- Entry points (user → A, A → B, B → C) are `kind=task`. "Please do this next step."
- The final result is `kind=mem`. "Here is the durable output."
- Intermediate results that carry raw data (e.g., retrieved chunks) are `kind=mem` even mid-chain — `mem` signals data-not-request.

**Worked example.** [`examples/orchestration/`](../examples/orchestration/) — a 3-agent RAG pipeline (ingest → retrieve → cite) with all five crumbs inlined and validated.

**Gotchas.**
- Don't collapse stages into one crumb for "efficiency." The whole point is that each stage is independently auditable and replaceable.
- Don't put generated text in a retrieval-stage crumb. Retrieval emits raw source chunks; generation is a separate stage that cites them.

---

## Pattern 2 — Supervisor / worker (fan-out, fan-in)

One coordinator agent dispatches N parallel subtasks, then consolidates the replies.

```
              +--task--> worker 1 --mem--+
              |                          |
supervisor ---+--task--> worker 2 --mem--+---> supervisor --mem--> user
              |                          |
              +--task--> worker 3 --mem--+
```

**When to use.** The work is naturally parallel and the parts don't depend on each other at dispatch time. Examples: running the same analysis across N files, asking the same research question of N data sources, fan-out web scraping.

**Crumb types:**
- Every subtask is a separate `kind=task` crumb with its own `id=`. Workers see only their own crumb.
- Every reply is a `kind=mem` carrying `ext.<project>.parent=<supervisor-task-id>`.
- The supervisor's final consolidated output is a `kind=mem` carrying `ext.<project>.children=<id1,id2,...>`.

**Gotcha.** If workers can fail independently, the supervisor's consolidated `mem` MUST explicitly say which children completed and which didn't. Silently omitting a failed child destroys the audit trail.

---

## Pattern 3 — Debate / reducer

N models answer the same question. A reducer synthesizes, preserving real disagreement.

```
                    +--> [model A] --mem--+
                    |                     |
user --task---------+--> [model B] --mem--+---> reducer --mem--> user
                    |                     |
                    +--> [model C] --mem--+
```

**When to use.** The question is open-ended; disagreement between answers is a signal, not a bug; you care about seeing multiple credible positions. Examples: architecture tradeoffs, risk assessment, legal/compliance interpretation where reasonable readers differ.

**How it differs from supervisor/worker:**
- In supervisor/worker, the N tasks are *different*. In debate, the N tasks are *the same question* sent to different responders.
- The reducer's job is synthesis with preservation of disagreement, not consolidation of non-overlapping work.

**Worked example.** [`examples/debate/`](../examples/debate/) — three models debate a webhook-retry-policy question, with a reducer that preserves their disagreement on *where* retries should live.

**Protocol rules:**
- Responders MUST emit `kind=mem`, not `task` (it's durable output, not a request).
- Responders MUST NOT edit other responders' crumbs.
- The reducer MUST reference every answer it considered in `ext.debate.answers=`.
- The reducer MUST preserve genuine disagreement — averaging-it-away is a debate failure.
- The reducer SHOULD flag *residual risk* (points none of the responders covered) as a follow-up.

---

## Pattern 4 — Long-running state

An agent loop runs across many sessions. A `kind=mem` crumb, stored under version control, acts as shared state. Every session reads it on wake-up and writes updates on shutdown.

```
session 1 --reads--> [state.crumb] --writes--> session 1
                         |
                         v
session 2 --reads--> [state.crumb] --writes--> session 2
                         |
                         v
                       ...
```

**When to use.** Preferences, long-horizon goals, accumulated observations about a project, or anything that should survive a process restart.

**Crumb types:**
- The state is a `kind=mem` with `[consolidated]` as the stable surface and `[raw]` as the untouched append log.
- Each session's updates are dream-passed into `[consolidated]` periodically (see `DREAMING.md`).

**This is what Palace does natively.** `crumb palace` stores a directory of `kind=mem` crumbs organized by wings and halls, each updated across sessions. Use Palace if you want a maintained implementation; use the raw `mem` pattern if you want it simpler.

**Gotcha.** Don't let `[raw]` grow unboundedly. Run `crumb dream` on a cadence that matches your write volume so consolidation stays within `max_index_tokens`.

---

## Pattern 5 — Nested patterns

These patterns compose. A supervisor (Pattern 2) can dispatch a debate (Pattern 3) for each subtask. A linear pipeline (Pattern 1) can have a debate as one of its stages. Long-running state (Pattern 4) can be the memory layer that every agent in the pipeline reads from.

When you nest, the naming convention matters:

- Use one `ext.<namespace>.*` per pattern layer.
  - `ext.rag.upstream=` for the pipeline layer.
  - `ext.debate.question=` for the debate nested inside.
- The final output crumb can carry both. A parser that understands `rag` but not `debate` still reconstructs the pipeline chain; a parser that understands `debate` sees the synthesis metadata.

---

## What CRUMB does not prescribe

- **Transport.** Crumbs are plain text. Shovel them over stdin, files, message queues, HTTP, whatever you use. CRUMB has no opinion.
- **Concurrency.** Whether you fan out to N workers in parallel or serially is a scheduling choice. The crumb shape is identical.
- **Retries.** If a task crumb fails, re-dispatch or not — the protocol doesn't care. What matters is that every new dispatch emits a fresh `id=` so replay is unambiguous.
- **Authorization.** Who is allowed to emit what is AgentAuth's job (`crumb passport`, `crumb policy`). The handoff pattern describes shape, not access.

---

## Minimal starter

The smallest useful multi-agent setup is Pattern 1 with two stages. Pseudocode:

```python
from cli.handoff import emit_task, emit_mem

# Stage 1 receives the user's question as a task crumb, produces retrieval output.
t1 = emit_task(
    title="Retrieve chunks for question X",
    goal="Return top-K chunks relevant to the user's question.",
    context=["Question: " + question, "Corpus: 12 regulation PDFs"],
    constraints=["Retrieval only, no generation"],
    source="rag.ingest",
    upstream=None,
)
# ... retrieve agent processes t1 and emits ...
m1 = emit_mem(
    title="Retrieved chunks for X",
    consolidated=chunks,
    source="rag.retrieve",
    upstream=t1["id"],
)
# Stage 2 receives m1 + a new task and produces the cited answer.
```

See [`cli/handoff.py`](../cli/handoff.py) for the helper module.

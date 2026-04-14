# Multi-agent orchestration with CRUMB

A fully-worked 3-agent retrieval-augmented-generation (RAG) pipeline where every agent-to-agent message is a valid `.crumb` file. This is the shape you want when an AI system is made of specialized agents that hand work off to each other, and when you need an audit trail that survives the process exiting.

## The pipeline

```
user
  |  01-ingest-request.crumb       kind=task
  v
[ingest agent]
  |  02-retrieve-request.crumb     kind=task
  v
[retrieve agent]
  |  03-retrieve-result.crumb      kind=mem   <-- chunks only, no generated text
  v
[cite agent receives:
   - 02-retrieve-request.crumb (the goal)
   - 03-retrieve-result.crumb (the source material)
 and produces:]
  |  04-cite-request.crumb         kind=task   <-- self-dispatch for the cite step
  v
[cite agent]
  |  05-final-answer.crumb         kind=mem
  v
user
```

Each arrow is a copy-paste-able CRUMB that could be logged, diffed, replayed, or handed to a different model. Nothing is held in ephemeral agent memory.

## Why this shape

- **Every handoff is auditable.** `05-final-answer.crumb` carries an `ext.compliance-ai.chain=` header listing the full provenance chain. An auditor can replay the pipeline from the logs.
- **Every step is typed.** A `task` crumb means "please do this." A `mem` crumb means "here is durable output." Mixing them is a protocol error you can lint for.
- **No hallucinated citations.** The retrieve agent emits raw chunks as `[consolidated]`. The cite agent is constrained to cite only what the upstream crumb actually contains. `crumb lint` can grep for citation tokens that don't appear in the upstream mem crumb.
- **Agent swaps are free.** If you replace the cite agent with a different model, you don't have to rewrite glue code — you just point it at the same two upstream crumbs.

## The upstream-pointer convention

Every non-root crumb in the chain carries:

```text
extensions=ext.<project>.upstream.v1
ext.<project>.upstream=<id of the crumb being replied to>
```

This is a namespaced extension (SPEC §8.1) — any parser ignores it safely, but an orchestrator can traverse it to rebuild the chain. The final answer additionally carries `ext.<project>.chain=id1,id2,…` so a single crumb is enough to reconstruct the whole pipeline without walking pointers.

## Validate the whole pipeline

```bash
crumb validate examples/orchestration/*.crumb
```

Every crumb in this directory is v1.1-conformant. If you're building a similar orchestrator, treat these files as the shape your agents should emit.

## Extending this pattern

- **Supervisor/worker.** The ingest agent can emit N parallel `task` crumbs for N workers, then consolidate their `mem` replies. See `docs/AGENT_HANDOFFS.md` §2.
- **Debate.** Emit the same `task` crumb to multiple models, collect their `mem` answers, then run a reducer. See `examples/debate/`.
- **Long-running state.** Wrap the pipeline in a persistent `mem` crumb updated across sessions — this is what Palace does natively (`crumb palace`).

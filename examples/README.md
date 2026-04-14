# Example crumbs 🍞

Copy, edit, and paste these into any AI tool — they're all valid v1.1 crumbs.

## By kind

### `kind=task` — hand off work
- [`task-feature-continuation.crumb`](task-feature-continuation.crumb) — continue a feature already in progress
- [`task-bug-fix.crumb`](task-bug-fix.crumb) — hand off a bug-fix task cleanly
- [`task-cross-tool-feature-handoff.crumb`](task-cross-tool-feature-handoff.crumb) — move work between Cursor, Claude, ChatGPT
- [`task-content-repurpose-handoff.crumb`](task-content-repurpose-handoff.crumb) — re-use crumb context for marketing/docs

### `kind=mem` — durable preferences and knowledge
- [`mem-user-preferences.crumb`](mem-user-preferences.crumb) — user or project preferences that survive sessions
- [`mem-terse-output.crumb`](mem-terse-output.crumb) — preference crumb for token-efficient AI output (no preambles, filler, or sign-offs); pairs with MeTalk wire compression

### `kind=map` — codebase overview
- [`map-repo-onboarding.crumb`](map-repo-onboarding.crumb) — compact repo structure and invariants
- [`map-client-takeover.crumb`](map-client-takeover.crumb) — inherit an unfamiliar project

### `kind=log` — append-only session transcript
- [`log-deployment.crumb`](log-deployment.crumb) — timestamped events from a deploy

### `kind=todo` — track work items
- [`todo-sprint.crumb`](todo-sprint.crumb) — checkbox-style task list

### `kind=wake` — session bootstrap from Palace
- [`wake-session.crumb`](wake-session.crumb) — instant-context crumb emitted by `crumb wake`

## Walkthroughs

- [`lifecycle-demo.md`](lifecycle-demo.md) — full append → dream → diff lifecycle for a mem crumb
- [`metalk-demo.md`](metalk-demo.md) — before/after token compression at each MeTalk level

## Multi-agent patterns

- [`orchestration/`](orchestration/) — worked 3-agent RAG pipeline (ingest → retrieve → cite) with every handoff crumb inline
- [`debate/`](debate/) — three models debate the same question, a reducer preserves the disagreement
- See [`docs/AGENT_HANDOFFS.md`](../docs/AGENT_HANDOFFS.md) for the full pattern catalogue

## Best practice

Start with the smallest crumb that still lets the next AI act confidently.

- `task` when you know the next action
- `mem` for stable preferences and guardrails
- `map` for repo structure and invariants
- `log` for timestamped, immutable events
- `todo` for tracked work items
- `wake` for spinning up a new AI session from your Palace

Pass the crumb, not the whole loaf. 🍞

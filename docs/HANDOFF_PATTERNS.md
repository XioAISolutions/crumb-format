# Handoff patterns

CRUMB works best when you choose the right shape for the handoff.

## 1. Feature continuation

Use `kind=task` when one AI already started implementation and another AI needs to continue.

Recommended sections:
- `[goal]`
- `[context]`
- `[constraints]`
- optional `[notes]`

## 2. Repo onboarding

Use `kind=map` when a new AI needs to understand a codebase quickly.

Recommended sections:
- `[project]`
- `[modules]`
- `[invariants]`
- optional `[flows]`

## 3. Durable preferences

Use `kind=mem` when the information should survive across many sessions.

Recommended sections:
- `[consolidated]`
- optional `[dream]`
- optional `[raw]` before the next consolidation pass

## 4. Cross-tool handoff

A simple pattern that works well:
1. summarize the current work into a `.crumb`
2. paste the `.crumb` into the next AI
3. ask it to continue from the crumb, not from assumptions

That is the whole point: small, portable, clear.

# Example crumbs

These examples are meant to be copied, edited, and pasted into another AI.

## Included examples

- `task-feature-continuation.crumb` — continue a feature already in progress
- `task-bug-fix.crumb` — hand off a bug-fix task cleanly
- `mem-user-preferences.crumb` — durable user or project preferences
- `map-repo-onboarding.crumb` — a compact repo or project map
- `task-packed-auth-context.crumb` — packed handoff built under a token budget
- `mem-mempalace-auth-migration.crumb` — MemPalace bridge export as a durable memory crumb

## Best practice

Start with the smallest crumb that still lets the next AI act confidently.

- Use `task` when you know the next action
- Use `mem` for stable preferences and guardrails
- Use `map` for repo structure and invariants
- Use `pack` when you need to collapse multiple crumbs into one budgeted handoff
- Use `bridge` when context must move across memory systems

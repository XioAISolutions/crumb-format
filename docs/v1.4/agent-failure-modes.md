# v1.4 candidate — Canonical `[checks]` names for agent failure modes

**Status:** Draft. Non-normative. Companion to [`typed-checks.md`](typed-checks.md): once the threshold grammar is settled, the *names* on the left side of `::` should also have a normative vocabulary so cross-tool consumers can act on them. Inspired by the failure-mode taxonomy that [HALO](https://github.com/context-labs/halo) and similar agent-introspection tools surface from execution traces.

## Why

`[checks]` already supports any `name :: status` line:

```
[checks]
- whatever_string :: pass
- another_thing   :: fail
```

That's flexible but not portable. A receiver of a CRUMB log emitted by jcode can't reliably ask "did this session hallucinate any tool calls?" because every emitter picks its own name. The minimal-basis filter in [`v1.4-scoping.md`](../v1.4-scoping.md) says: don't grow the wire format to fix this — establish a **canonical vocabulary** instead, and let unknown names continue to validate.

## Proposal in one paragraph

Define a small set of canonical `[checks]` names for common agent failure modes. Senders MAY use other names; receivers SHOULD recognize the canonical ones when they appear. v1.4 validators do not enforce these names — they remain free-form. The vocabulary is normative-by-convention, not by parser behavior.

## Initial vocabulary (10 names)

Each name is `snake_case`, scoped enough to be unambiguous, and short enough to type by hand. All examples use `:: detected` for boolean presence and `:: count=N` for numeric. Senders MAY combine with the typed-checks annotations from [`typed-checks.md`](typed-checks.md) (`value=`, `threshold=`).

| Name | Meaning | Typical line |
|---|---|---|
| `hallucinated_tool_call` | Model invoked a tool name that doesn't exist in the harness | `- hallucinated_tool_call :: detected count=1 tool=foo_bar` |
| `refusal_loop` | Model refused N+ consecutive turns | `- refusal_loop :: detected count=3 reason=policy_filter` |
| `tool_error_unhandled` | Tool returned an error, model didn't recover or acknowledge | `- tool_error_unhandled :: detected count=1 tool=grep` |
| `semantic_drift` | Output stopped addressing the stated goal | `- semantic_drift :: detected note=last 3 turns off-topic` |
| `token_budget_exceeded` | Output exceeded a declared token budget | `- token_budget_exceeded :: fail value=12450 threshold=8000 unit=tokens` |
| `invalid_handoff_target` | A `[handoff]` line targets an unknown receiver | `- invalid_handoff_target :: detected count=1 target=ship_v2` |
| `circular_reference` | A `refs=` chain or `[handoff] after=` graph contains a cycle | `- circular_reference :: detected note=task-a → task-b → task-a` |
| `truncated_output` | Model output was cut off mid-token by max_tokens | `- truncated_output :: detected` |
| `prompt_injection_suspected` | Tool output contained instructions overriding the system prompt | `- prompt_injection_suspected :: detected source=web_fetch` |
| `unauthorized_tool_call` | Tool invocation violated the AgentAuth `[guardrails]` policy | `- unauthorized_tool_call :: detected count=1 tool=shell-exec` |

These cover the failure shapes that production agent harnesses (OpenAI Agents SDK, jcode, Claude Code, Cursor) actually surface. The list is closed for v1.4 — additions get their own scoping.

## Worked example

A `kind=log` crumb produced from a HALO trace using `crumb from-halo`, manually annotated by a reviewer with canonical check names:

```text
BEGIN CRUMB
v=1.3
kind=log
title=Session 4827 — agent harness review
source=halo
trace_id=halo-4827
---
[entries]
- agent.session.start  duration=500ms  model=claude-3-5
- tool.call  duration=1000ms  tool_name=grep
- tool.call  :: error  tool_name=hallucinated_function  note='tool not found'
- agent.refusal  :: error  note='model refused'
- agent.session.end

[checks]
- hallucinated_tool_call    :: detected   count=1   tool=hallucinated_function
- tool_error_unhandled      :: detected   count=1
- refusal_loop              :: detected   count=1   reason=policy_filter
- semantic_drift            :: pass
- token_budget_exceeded     :: pass       value=2400   threshold=8000   unit=tokens
END CRUMB
```

A consumer (CI gate, dashboard, fleet manager) can now reliably answer "are there hallucinations in this trace?" by looking at `[checks]` for `hallucinated_tool_call`. No prompt engineering, no string heuristics.

## What this is not

- **Not a parser change.** v1.4 validators do not reject unknown names or check for these names. They remain free-form annotations.
- **Not a HALO dependency.** This vocabulary stands alone. HALO is just one tool that emits these failure modes; jcode, Cursor, Claude Code, OpenAI Agents SDK all have analogous categories. Anyone can adopt.
- **Not exhaustive.** Ten names. Expansions need their own scoping doc and a real receiver-side use case (the minimal-basis test).

## Open question

Is `:: detected` the right second-half token? Alternatives:
- `:: present` (more neutral but less alarming for failure modes)
- `:: yes` (shortest)
- `:: 1` (numeric)

Recommend `:: detected` — it reads correctly out loud ("hallucinated_tool_call detected"), survives translation, and pairs naturally with the count annotation.

## Sequencing

Bundle this with the typed-checks and deadlines drafts as the v1.4 normative bump:
1. SPEC §15 normative paragraph for typed-checks (already drafted)
2. SPEC §11.1 normative paragraph for deadlines (already drafted, impl shipped in v0.9.0)
3. SPEC §15 appendix for canonical failure-mode names (this doc)

Total v1.4 ship surface: ~3 SPEC paragraphs, ~150 LOC validator changes, ~120 LOC tests. All additive.

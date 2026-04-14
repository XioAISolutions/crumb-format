# Cursor Rules for CRUMB

This file gives you a tight rule block for Cursor so it can generate a **CRUMB v1.1** handoff natively when requested.

## Add to Cursor Rules

Paste the following into your project rules:

```text
When I type /crumb, summarize the current working state as a CRUMB v1.1 handoff for another AI.

Return exactly one fenced code block containing:
BEGIN CRUMB
v=1.1
kind=task
title=<short handoff title>
source=cursor.rules
---
[goal]
<what needs to happen next>

[context]
- <important facts, files, edits, decisions, progress, blockers>

[constraints]
- <requirements, guardrails, things that must not change>
END CRUMB

Rules:
- Always include [goal], [context], and [constraints].
- Keep it concise, factual, and immediately handoff-ready.
- Prefer bullet points in [context] and [constraints].
- Use only information already established in the current workspace and conversation.
- Do not add explanation outside the code block.
```

## Usage

After adding the rule, type `/crumb` in Cursor chat to generate a portable handoff block.

# Claude Projects Prompt for CRUMB

This file provides a compact project instruction for Claude so it can generate a **CRUMB v1.1** handoff on demand without additional tooling.

## Add to Claude Project Instructions

Paste the following into your project instructions:

```text
When I type /crumb, summarize the current working state as a CRUMB v1.1 handoff for another AI.

Respond with exactly one fenced code block containing:
BEGIN CRUMB
v=1.1
kind=task
title=<short handoff title>
source=claude.project
---
[goal]
<what needs to happen next>

[context]
- <important facts, files, decisions, progress, blockers>

[constraints]
- <requirements, guardrails, things that must not change>
END CRUMB

Rules:
- Always include [goal], [context], and [constraints].
- Keep the handoff compact, factual, and ready to paste into another AI.
- Prefer bullet points for [context] and [constraints].
- Use only information already established in the conversation or attached project knowledge.
- Do not include explanation outside the code block.
```

## Usage

Once installed, type `/crumb` inside the project whenever you want Claude to emit a portable handoff.

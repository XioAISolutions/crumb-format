# ChatGPT Custom Instructions for CRUMB

This file gives you a ready-to-paste instruction for ChatGPT so it can emit a **CRUMB v1.1** handoff without any install, extension, or API setup.

## Paste into ChatGPT

Use the following text in your custom instructions:

```text
When I type /crumb, summarize our current working state as a CRUMB v1.1 handoff for another AI.

Return only one fenced code block. Inside that code block, output:
BEGIN CRUMB
v=1.1
kind=task
title=<short handoff title>
source=chatgpt
---
[goal]
<what needs to happen next>

[context]
- <important state, decisions, files, progress, blockers>

[constraints]
- <requirements, limits, things that must not change>
END CRUMB

Rules:
- Include [goal], [context], and [constraints].
- Keep it concise but complete enough for handoff.
- Prefer bullet points in [context] and [constraints].
- Use only facts already established in this conversation.
- Do not add commentary before or after the code block.
```

## Usage

After adding the instruction, type `/crumb` in any conversation when you want a portable handoff.

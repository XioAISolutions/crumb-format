# Before / After demo 🍞

Use this for your first public demo.

## Before

Show a messy block of notes or transcript fragments like:

- pricing page half rebuilt in Claude
- mobile spacing still broken
- CTA buttons not wired
- old testimonials should not come back
- keep premium feel
- do not add dependencies
- need something launchable today

## After

Show this `.crumb`:

```text
BEGIN CRUMB
v=1.1
kind=task
title=Continue pricing page rebuild in Cursor
source=claude.chat
project=xioai-site
max_index_tokens=640
max_total_tokens=2400
---
[goal]
Continue the pricing page rebuild in Cursor and finish the section structure, CTA wiring, and responsive cleanup.

[context]
- Work started in Claude.
- New pricing page already has hero, plan cards, and FAQ shell.
- Mobile spacing is still uneven below 768px.
- CTA buttons are present but not wired to the correct booking and signup targets.
- Old testimonials section should not be brought back.
- Brand tone should feel premium, minimal, and direct.

[constraints]
- Keep the existing visual direction.
- Do not add new dependencies.
- Reuse current components where possible.
- Preserve fast load performance.
- Keep copy concise and sales-forward.
END CRUMB
```

## Final frame

Show the second AI continuing cleanly from the crumb.

That is the product story in one screen:

**switch tools without losing the plot.**

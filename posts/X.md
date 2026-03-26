I kept running into the same problem:

You do work in one AI tool.
You switch tools.
You lose the plot.

So I made **CRUMB** 🍞

A tiny, copy-paste AI handoff format.

Instead of pasting giant transcripts, you pass a small block with:
- the goal
- the context that matters
- the constraints that must hold

Example:

```text
BEGIN CRUMB
v=1.1
kind=task
title=Continue dark mode work
source=chatgpt.chat
---
[goal]
Finish the dark mode feature without changing app navigation.

[context]
- Theme context exists.
- Settings toggle UI is half-done.
- Persistence is not wired yet.

[constraints]
- Keep Expo setup unchanged.
- No new dependencies.
END CRUMB
```

Repo: https://github.com/XioAISolutions/crumb-format

Switch AIs without losing the plot.
Pass the crumb. 🍞

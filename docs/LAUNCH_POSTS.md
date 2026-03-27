# Launch Posts

## Show HN

**Title:** Show HN: CRUMB -- a copy-paste format for handing off work between AI tools

**Body:**

I kept losing context when switching between AI tools. Start a task in Cursor, need to continue in Claude, and you're either pasting a wall of chat history or starting over.

CRUMB is a small plain-text format for these handoffs. It looks like this:

```
BEGIN CRUMB
v=1.1
kind=task
title=Fix login redirect bug
source=cursor.agent
---
[goal]
Fix the bug where authenticated users are redirected back to /login after refresh.

[context]
- App uses JWT cookie auth
- Redirect loop happens only on full page refresh
- Middleware reads auth state before cookie parsing is complete

[constraints]
- Do not change the login UI
- Preserve existing cookie names
END CRUMB
```

You paste that into the next AI and it picks up the work. No plugins, no integrations, no accounts -- just text.

Three kinds: `task` (what to do next), `mem` (preferences that survive across sessions), `map` (repo structure for onboarding an AI to your codebase).

The repo has a spec, a CLI for generating crumbs from the terminal, validators in Python and Node, and ready-to-paste examples. There's also a custom instruction snippet you can add to any AI so it generates CRUMBs when you say "crumb it."

What I'm most curious about: do other people actually switch between AI tools mid-task, or am I the only one doing this?

---

## Reddit (r/programming or r/artificial)

**Title:** I built a tiny format for handing off work between AI tools without losing context

**Body:**

Problem I kept hitting: I'd be deep into something with Cursor, realize I need Claude for a different part, and then spend 5 minutes re-explaining what I was doing. Or I'd paste the entire chat log and watch it hallucinate from the noise.

So I made CRUMB -- a structured text block you copy-paste between tools. Goal, context, constraints. That's it. ~15 lines, plain text, works everywhere.

```
BEGIN CRUMB
v=1.1
kind=task
title=Fix login redirect bug
source=cursor.agent
---
[goal]
Fix the bug where authenticated users are redirected back to /login after refresh.

[context]
- App uses JWT cookie auth
- Middleware reads auth state before cookie parsing is complete

[constraints]
- Do not change the login UI
- Preserve existing cookie names
END CRUMB
```

Paste that into any AI and it skips straight to the work.

There are three kinds:
- **task** -- "here's what to do next"
- **mem** -- "here are my preferences" (survives across sessions)
- **map** -- "here's how this codebase is structured"

The killer feature for me: you can add a custom instruction to any AI that says "when I say 'crumb it', generate a CRUMB." Then switching tools is literally: say "crumb it," copy, paste into next tool, keep going.

No dependencies, no plugins, no API. Just text.

GitHub: [link]

Curious if anyone else has this problem or if you've found other ways to handle it.

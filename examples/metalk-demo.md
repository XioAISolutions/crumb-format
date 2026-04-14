# MeTalk compression demo

MeTalk ("caveman speak") is CRUMB's optional token compression pass for AI-to-AI handoffs. It has three levels:

- **Level 1** — dictionary substitutions only (lossless, fully reversible)
- **Level 2** — dictionary + grammar stripping (lossy, meaning-preserving) — **default**
- **Level 3** — dictionary + grammar + aggressive condensing

MeTalk-encoded crumbs carry an `mt=N` header so receiving tools can detect the format and decode Level 1 substitutions back to full English.

---

## Source crumb

Starting from [`examples/task-bug-fix.crumb`](task-bug-fix.crumb):

```
BEGIN CRUMB
v=1.1
kind=task
title=Fix login redirect bug
source=cursor.agent
project=web-app
max_index_tokens=512
max_total_tokens=2048
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
- Add a regression check before merging
END CRUMB
```

**~127 tokens.**

---

## Level 1 — lossless (`crumb metalk … --level 1`)

Abbreviates `BEGIN CRUMB`→`BC`, header keys (`kind`→`k`, `source`→`src`, `title`→`t`, …), section names (`[goal]`→`[g]`, `[context]`→`[cx]`, …), and tech terms in the body (`authentication`→`auth`, `middleware`→`mw`, …). Full English elsewhere is preserved.

```
BC
v=1.1
k=task
t=Fix login redirect bug
src=cursor.agent
pj=web-app
mit=512
mtt=2048
mt=1
---
[g]
Fix the bug where authenticated users are redirected back to /login after refresh.

[cx]
- App uses JWT cookie auth
- Redirect loop happens only on full page refresh
- Mw reads auth state before cookie parsing is complete

[ct]
- Do not change the login UI
- Preserve existing cookie names
- Add a regression check before merging
EC
```

**~108 tokens (15.0% saved, 1.18x).** Round-trip with `--decode` returns the original.

---

## Level 2 — default (`crumb metalk …`)

Adds grammar stripping: drops articles (`the`, `a`, `an`), filler (`just`, `very`, `really`), and rewrites verbose phrases (`do not`→`don't`, `in order to`→`to`).

```
BC
v=1.1
k=task
t=Fix login redirect bug
src=cursor.agent
pj=web-app
mit=512
mtt=2048
mt=2
---
[g]
Fix bug where authenticated users are redirected back to /login after refresh.

[cx]
- App uses JWT cookie auth
- Redirect loop happens only on full page refresh
- Mw reads auth state before cookie parsing is complete

[ct]
- don't change login UI
- Preserve existing cookie names
- Add regression check before merging
EC
```

**~105 tokens (17.3% saved, 1.21x).** `--decode` recovers dictionary subs but cannot restore stripped articles — the meaning is preserved, the grammar is terse.

---

## Level 3 — aggressive (`crumb metalk … --level 3`)

Same as Level 2 plus sentence condensing: drops trailing periods on bullets, removes empty bullets, collapses blank lines. Best for dense, fact-heavy crumbs.

For this particular example Level 3 produces the same output as Level 2 because the source is already tight. On larger, prose-heavy crumbs Level 3 typically adds another 5–15% savings.

---

## Chaining with two-stage compression

MeTalk composes with the existing `crumb compress` pipeline:

```bash
crumb compress task.crumb --metalk --metalk-level 2 -o tight.crumb
```

That runs:

1. **Stage 1** — semantic deduplication
2. **Stage 2** — signal-based pruning
3. **Stage 3** — MeTalk (optional)

`crumb bench task.crumb` shows all three stages' projected savings.

---

## When to use which level

| Scenario | Level |
|---|---|
| Archiving a crumb and need a perfect round-trip | 1 |
| Handing off to another AI over a token-constrained channel | 2 (default) |
| Stuffing maximum context into a tight budget | 3 |
| Producing a crumb for human review | skip MeTalk |

Receiving AI tools should call `crumb metalk --decode` first to restore dictionary substitutions before parsing. Grammar-stripped output is intended to be read directly by LLMs, which tolerate missing articles without loss of meaning.

---

## Two-layer token efficiency

MeTalk compresses one layer of the token stack — the crumb payload on the wire. There's a second, complementary layer: the AI's generated output itself. Both are about stripping ceremony and letting signal travel denser.

- **Output-layer compression** — shape how the model *writes* (tell it to drop articles, filler, preambles, and sign-offs). See [`mem-terse-output.crumb`](mem-terse-output.crumb) for a loadable preference crumb.
- **Wire-layer compression (MeTalk)** — shape how the crumb *travels* between tools. Lossless at Level 1, meaning-preserving at Level 2/3.

The two compose. Use a terse-output preference at the session level; use MeTalk at the handoff level.

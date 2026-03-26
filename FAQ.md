# FAQ 🍞

## What is CRUMB?

CRUMB is a text-first AI handoff format.

It gives work a small, structured shape so you can move it between AI tools without re-explaining everything.

## Why not just paste the whole chat?

Because raw chats are noisy.

They mix:
- dead ends
- repeated context
- conflicting ideas
- filler the next AI has to re-parse

CRUMB keeps the parts that matter for the next step.

## Is CRUMB a compression format?

Not primarily.

The core value is **handoff quality**, not byte savings.

Optional binary transport can exist later, but the canonical form is plain text.

## Does an AI need special support to read CRUMB?

No.

That is one of the main goals.

A `.crumb` should be understandable by a human and by an LLM as plain text.

## What are the main kinds?

- `kind=task` — what to do next
- `kind=mem` — long-term consolidated memory
- `kind=map` — project or repo map

## What does “dreaming” mean here?

Dreaming is the consolidation pass.

It means turning noisy notes, recent tasks, or raw memory into a smaller, cleaner, more trustworthy crumb.

## Should I store everything in CRUMB?

No.

CRUMB is strongest when it stores the **small set of things that matter for the next step**.

## Can I commit `.crumb` files to Git?

Yes.

They are plain text and diff-friendly.

## What is the smallest useful demo?

1. Start work in one AI tool
2. Turn the current state into a `.crumb`
3. Paste that `.crumb` into another AI
4. Show that the second AI continues cleanly

Pass the crumb, not the whole loaf. 🍞

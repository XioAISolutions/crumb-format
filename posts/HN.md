# Show HN: CRUMB — a copy-paste AI handoff format

I kept hitting the same workflow problem:

Work starts in one AI tool, then needs to continue in another. The usual handoff is either:
- a giant raw transcript
- a loose summary that drops important constraints
- or re-explaining everything manually

CRUMB is an attempt to make that handoff small, readable, and copy-pasteable.

It is text-first, not binary-first.

A `.crumb` is a short structured block with things like:
- goal
- context
- constraints
- consolidated memory
- project map

The idea is that a human can read it, a tiny script can validate it, and an LLM can use it immediately without special decoding.

Repo:
https://github.com/XioAISolutions/crumb-format

Would love feedback on:
- whether this is meaningfully better than raw copy-paste
- where it breaks
- whether task / mem / map is the right top-level split
- what real handoff examples are still missing

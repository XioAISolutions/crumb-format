# DREAMING.md — Consolidation for .crumb

This document defines how to perform “dream” / consolidation passes over `.crumb` files.

The goal is simple:

> Turn noisy interaction history into a **small, trustworthy, portable** `.crumb`.

---

## Why consolidate?

Without consolidation:
- long-term memory files become noisy and contradictory
- obsolete facts linger
- new tools cannot tell which parts of a giant transcript still matter
- every new AI has to re-parse the whole history

With consolidation:
- high-signal facts are kept
- low-signal noise is dropped
- contradictions are resolved in favor of newer truth
- the result fits into a small token budget and travels well between tools

`.crumb` is designed to store the **result** of consolidation, not raw logs.

---

## Four-phase dream pass

1. **Orient** — understand existing memory
2. **Gather signal** — find what changed or drifted
3. **Consolidate** — merge, correct, and promote important facts
4. **Prune to budget** — keep the index small and sharp

### Orient
Read the headers and key sections of existing `kind=mem` and `kind=map` crumbs. You do not need to read full chat transcripts here.

### Gather signal
Prefer:
1. recent `[raw]` or `[logs]`
2. recent `kind=task` crumbs
3. narrow transcript grep only if needed

### Consolidate
For each new signal:
- merge if it clarifies an existing fact
- replace if it contradicts an outdated fact
- add if it is orthogonal and important

Move processed `[raw]` items into `[consolidated]`.

### Prune to budget
If above `max_index_tokens`, trim stale or low-value entries first and preserve:
- current architecture
- current dependencies
- current invariants
- current safety constraints

---

## Updating dream metadata

After a successful pass, update:

```text
dream_pass=2026-03-25T21:00:00Z
dream_sessions=12
```

Recommended `[dream]` section:

```text
[dream]
- last_pass: 2026-03-25T21:00:00Z
- sessions_seen: 12
- notes:
  - Promoted new auth config details to [consolidated].
  - Removed outdated references to MongoDB.
```

---

## Tool guidance

If consuming `.crumb`:
- prefer recent `dream_pass` for long-term memory
- include header + key sections first
- only include raw/log sections when explicitly needed

If emitting `.crumb`:
- keep `[raw]` small and append-only
- run dream passes periodically
- do not fear deleting obsolete facts

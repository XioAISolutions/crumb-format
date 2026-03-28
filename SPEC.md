# .crumb Specification (v1.1)

**Status:** Draft  
**Category:** AI handoff format  
**Goal:** A tiny, human-readable file format for portable AI context handoffs between tools.

---

## 1. Design goals

`.crumb` is designed to:

- Capture the **state of work** in a small, structured text block.
- Be **portable** across AIs, tools, and sessions (copy/paste, commit, attach).
- Encode **budgets** and **priorities** so tools can load context intelligently.
- Represent **consolidated** memory (after a /dream-style pass), not just raw logs.
- Stay **simple enough to hand-edit** and robust enough to parse.

If it can’t fit on one mobile screen or be written by hand, it has failed.

---

## 2. File structure

A `.crumb` file has four parts:

1. Opening marker: `BEGIN CRUMB`
2. Header (`key=value`, one per line)
3. Body (section blocks)
4. Closing marker: `END CRUMB`

### 2.1 Grammar (informal)

```text
file        := "BEGIN CRUMB" NL header NL "---" NL sections? "END CRUMB" NL?
header      := header_line*
header_line := key "=" value NL
key         := 1+ non-space, non-`=`
value       := rest of line (trim trailing whitespace)

sections    := section+
section     := "[" section_name "]" NL section_body NL?
section_name:= 1+ non-`]`
section_body:= 0+ lines until next section or END CRUMB

NL          := "\\n" or "\\r\\n"
```

Parsers SHOULD ignore unknown header keys and unknown section names.

---

## 3. Header fields

All keys are lowercase ASCII with no spaces.

### 3.1 Required fields

- `v`
- `kind`
- `source`

Allowed `kind` values:
- `task`
- `mem`
- `map`

### 3.2 Recommended fields

- `title`
- `dream_pass`
- `dream_sessions`
- `max_index_tokens`
- `max_total_tokens`

### 3.3 Optional fields

- `id`
- `project`
- `env`
- `tags`
- `url` — link to the CRUMB spec or project; helps recipients understand the format

---

## 4. Sections

### 4.1 `kind=task`
Required sections:
- `[goal]`
- `[context]`
- `[constraints]`

Optional sections:
- `[logs]`
- `[notes]`
- `[raw_sessions]`

### 4.2 `kind=mem`
Required sections:
- `[consolidated]`

Optional sections:
- `[raw]`
- `[dream]`

### 4.3 `kind=map`
Required sections:
- `[project]`
- `[modules]`

Optional sections:
- `[invariants]`
- `[flows]`
- `[dependencies]`

---

## 5. Budgets and loading rules

`max_index_tokens` and `max_total_tokens` are advisory budgets, not exact counts.

Recommended loading behavior:
- Always load header and key sections (`goal`, `constraints`, `consolidated`, `project`)
- Load `context`, `modules`, and `flows` next if budget allows
- Only grep `logs`, `raw_sessions`, and `raw` unless explicitly needed

---

## 6. Consolidation (dream) semantics

When a dream pass runs, it SHOULD:
- merge near-duplicate facts
- prefer newer facts when conflicts occur
- remove stale or incorrect entries
- move processed items from `[raw]` into `[consolidated]`
- trim `[consolidated]` to fit within `max_index_tokens` where possible

It SHOULD update:
- `dream_pass`
- `dream_sessions`
- `[dream]` notes

---

## 7. Transport and binary compression (optional)

The canonical form of `.crumb` is plain text. Binary transports such as `.crumb.bin` are allowed, but the text form remains the source of truth for interoperability.

---

## 8. Compatibility

Parsers SHOULD ignore unknown headers and sections. Writers SHOULD add new sections rather than changing the meaning of old ones and bump `v` for breaking changes.

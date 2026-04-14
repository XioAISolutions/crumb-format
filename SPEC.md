# .crumb Specification (v1.1)

**Status:** Stable
**Category:** AI handoff format
**Goal:** A tiny, human-readable protocol for portable AI context handoffs between tools and memory systems.

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

### 2.1 Grammar (ABNF)

The grammar below uses ABNF as defined in [RFC 5234](https://www.rfc-editor.org/rfc/rfc5234) with the core rules `ALPHA`, `DIGIT`, `HTAB`, `SP`, `VCHAR`, `WSP`. CRLF and LF line endings are both permitted; parsers MUST treat `CRLF` and `LF` as equivalent line terminators.

```abnf
file          = begin-marker NL header-block NL separator NL [body] end-marker [NL]

begin-marker  = "BEGIN CRUMB"
end-marker    = "END CRUMB"
separator     = "---"

header-block  = *header-line
header-line   = key "=" value NL
key           = 1*key-char
key-char      = ALPHA / DIGIT / "-" / "_" / "." / "x"   ; lowercase ASCII; "x-" and "ext." prefixes reserved for namespaced extensions
value         = *VCHAR-or-SP                            ; trimmed of leading/trailing WSP

body          = section *(blank-line / section)
section       = section-header NL section-body
section-header= "[" section-name "]"
section-name  = 1*(ALPHA / DIGIT / "_" / "-")
section-body  = *body-line
body-line     = (1*VCHAR-or-SP / blank-line) NL

blank-line    = *WSP NL
NL            = CRLF / LF
VCHAR-or-SP   = VCHAR / SP / HTAB
```

**Conformance notes:**

- Parsers MUST reject files lacking either marker.
- Parsers MUST reject files lacking the `---` separator.
- Parsers MUST reject duplicate header keys (last-write-wins is a writer-side convention only).
- Parsers SHOULD ignore unknown header keys and unknown section names.
- Parsers SHOULD treat header keys case-sensitively as lowercase ASCII; producers MUST emit lowercase keys.
- Whitespace inside a header value is preserved verbatim except for trim of leading/trailing whitespace.
- A section with no `body-line` content is "empty"; whether emptiness is an error depends on whether the section is required for the kind (see §4).

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
- `log`
- `todo`
- `wake`

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
- `extensions`
- `url` — link to the CRUMB spec or project; helps recipients understand the format

Namespaced extension headers are also allowed, for example:

- `x-crumb-pack.strategy`
- `ext.acme.priority`

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

### 4.4 `kind=log`
Append-only session transcript. Never consolidated — entries are immutable.

Required sections:
- `[entries]` — timestamped lines in `- [ISO8601] text` format

### 4.5 `kind=todo`
Foresight/prospective memory. Tracks work items with checkbox state.

Required sections:
- `[tasks]` — entries in `- [ ] task` (open) or `- [x] task` (done) format

Optional sections:
- `[archived]` — completed tasks moved here by a dream pass

### 4.6 `kind=wake`
Session bootstrap crumb. Emitted by `crumb wake` to give a new AI session instant context from a Palace without requiring the user to re-explain.

Required sections:
- `[identity]` — who this palace belongs to, wing/room counts, summary stats

Optional sections:
- `[facts]` — top facts harvested from the palace's `facts` halls
- `[rooms]` — per-wing index of halls and room counts
- `[gaps]` — top knowledge gaps (when produced with `--reflect`)

Wake crumbs are ephemeral — they are regenerated from palace contents on demand and are not consolidated or dream-processed.

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

### 8.1 Extension model

CRUMB preserves `v=1.1` compatibility by treating extensions as additive metadata.

Rules:

- unknown headers MUST NOT break parsing
- unknown sections MUST NOT break parsing
- `extensions=` SHOULD contain namespaced identifiers such as `crumb.pack.v1`
- custom extension headers SHOULD be namespaced (for example `x-crumb-pack.strategy=hybrid`)
- extensions MUST preserve human readability

Examples:

```text
id=crumb-pack-abc123
tags=pack, auth
extensions=crumb.pack.v1, bridge.mempalace.export.v1
max_total_tokens=1800
max_index_tokens=900
x-crumb-pack.strategy=hybrid
```

---

## 9. Stability and versioning

The `v` header declares the format version a producer is targeting. CRUMB follows a loose semantic versioning scheme aligned with the protocol surface, not with any single tooling implementation.

### 9.1 What is stable at `v=1.1`

The following are **frozen** for the entire `v=1.x` line. Conforming parsers written against `v=1.1` MUST continue to parse all future `v=1.x` documents:

- The four-part file structure (`BEGIN CRUMB`, header block, `---` separator, body, `END CRUMB`).
- The `key=value` header syntax and the lowercase ASCII key character set defined in §2.1.
- The `[section-name]` body syntax.
- The required header set: `v`, `kind`, `source`.
- The currently defined `kind` values (`task`, `mem`, `map`, `log`, `todo`, `wake`) and their required sections (§4).
- The `CRLF`/`LF` line-terminator equivalence rule.
- The "ignore unknown headers and sections" rule (§8).
- The extension namespacing rules (§8.1).

Producers and consumers MAY rely on these guarantees indefinitely within the `v=1.x` line.

### 9.2 What may change inside `v=1.x`

Minor versions (`v=1.2`, `v=1.3`, …) MAY:

- introduce new optional headers (for example, a new advisory budget),
- introduce new optional sections within existing kinds,
- introduce new `kind` values,
- add new conformance notes that further constrain previously underspecified behavior,
- tighten security or extension guidance.

Minor versions MUST NOT:

- change the meaning of any existing header or section,
- remove any required header or section,
- change the file structure, line-terminator rules, or key character set,
- repurpose any reserved namespace prefix (`x-`, `ext.`).

A `v=1.x` document with `x > 1` SHOULD remain parseable by a `v=1.1` parser, with the new optional headers and sections being silently ignored per §8.

### 9.3 Deprecation policy

If a future minor version deprecates an existing optional feature:

1. The feature MUST first be marked `Deprecated` in this specification for at least one minor version before any tooling stops emitting it.
2. Parsers MUST continue to accept the deprecated feature for the remainder of the `v=1.x` line.
3. Removal is only permitted at a major version bump (`v=2.0`).

### 9.4 Breaking changes (`v=2.0` and beyond)

A new major version (`v=2.0`) is reserved for changes that would break a `v=1.x` parser — for example, altering the file structure, redefining a `kind`, or introducing a non-text canonical form.

Major version work happens in a separate specification document (`SPEC-2.md`) and is not delivered as a silent change to this file.

### 9.5 Tooling vs. format versioning

The `v=` header tracks the **format**, not the reference tooling. The `crumb-format` Python package, the MCP server tool names, and the REST/A2A endpoints have their own versioning surface, documented in `docs/STABILITY.md`. Implementations MUST NOT couple `v=` to their own release version.

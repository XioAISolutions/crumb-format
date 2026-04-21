# .crumb Specification (v1.2)

**Status:** Draft  
**Category:** AI handoff format  
**Goal:** A tiny, human-readable protocol for portable AI context handoffs between tools and memory systems.

**Version compatibility:** v1.2 is backward-compatible with v1.1. A v1.1 parser will accept a v1.2 file by ignoring unknown headers and sections (per §8). A v1.2 parser accepts both `v=1.1` and `v=1.2`. Every v1.2 addition is optional and purely additive.

**Efficiency layers (§§13–16):** content-addressed refs, priority annotations, delta crumbs, and budget-aware packing. All additive — a v1.2 consumer that ignores them is still a compliant consumer.

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
- `refs` — comma-separated cross-crumb references (v1.2, see §9)

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

## 9. Cross-crumb references (v1.2)

A CRUMB can point at other CRUMBs by id. This turns an isolated handoff into a navigable graph without breaking the copy-paste model.

### 9.1 `refs` header

Comma-separated list of references. Parsers MUST NOT fail on unresolvable refs — references are advisory pointers, not hard dependencies.

```text
refs=mem-prefs-abc123, map-web-app-2026q2
```

### 9.2 `[refs]` section (optional)

When a reference needs more than a bare id (resolution hints, role labels, notes), the writer MAY add a `[refs]` section. One ref per line. Format is advisory:

```text
[refs]
- mem-prefs-abc123  role=style  why=caller prefers concise commits
- map-web-app-2026q2  role=terrain  note=only [modules] is required for this task
```

Resolution strategy (filesystem path, content hash, URL, registry lookup) is intentionally left to the implementation. See [`docs/v1.2-ref-resolution.md`](docs/v1.2-ref-resolution.md) for the open design question.

### 9.3 Cycles

Refs MAY form cycles. Consumers SHOULD walk refs with a depth limit and a visited-set.

---

## 10. Foldable sections (v1.2)

A single section sometimes needs both a short form for token-tight contexts and a long form for full fidelity. v1.2 expresses this with **namespaced section names**, not true nesting — the flat grammar in §2.1 is preserved.

### 10.1 Naming convention

```text
[fold:NAME/summary]
short form here

[fold:NAME/full]
long form here
```

`NAME` is a bare identifier: letters, digits, `-`, `_`. No slashes except the one separating `NAME` from the variant.

### 10.2 Pairing rule

If `[fold:NAME/full]` is present, `[fold:NAME/summary]` MUST also be present. A `/full` without a `/summary` is a validation error. The reverse is allowed — a lone `/summary` is the degenerate case.

### 10.3 Fold-satisfies-required

When the required section for a kind is `[X]`, a writer MAY replace it with the pair `[fold:X/summary]` + `[fold:X/full]`. Validators MUST accept this substitution. A plain `[X]` and a fold pair MUST NOT coexist — choose one form per section per file.

Example for `kind=task`:

```text
[goal]
...

[fold:context/summary]
High-severity login bug, JWT middleware, no UI changes.

[fold:context/full]
Full transcript of the investigation, 40 lines of repro, stack trace...

[constraints]
...
```

### 10.4 Selection heuristic

Which variant a consumer loads is not mandated. Token-budget-aware loaders SHOULD prefer `/summary` under pressure and upgrade to `/full` when budget allows. See [`docs/v1.2-fold-heuristic.md`](docs/v1.2-fold-heuristic.md) for the open design question.

---

## 11. Handoff primitive (v1.2)

A `[handoff]` section is an optional, explicit "next AI do this" block. It does not replace `[goal]` — `[goal]` is what the work is, `[handoff]` is what the next agent should pick up first.

### 11.1 Structure

One action per line. A line MAY be a bare instruction or a namespaced form:

```text
[handoff]
- to=any  do=reproduce the failing test in tests/test_auth.py
- to=any  do=propose a fix without landing it
- to=human  do=approve the fix before merge
```

Recognized keys (all optional, all advisory): `to`, `do`, `why`, `deadline`, `ref`. Consumers that don't understand the namespaced form SHOULD treat the whole line as a bullet.

### 11.2 Ordering

Earlier lines have priority over later lines. A consumer SHOULD execute top-down.

### 11.3 Completion signal

A line starting with `- [x]` is treated as already completed context, not pending work — same convention as `[tasks]` in `kind=todo`.

---

## 12. Typed content annotations (v1.2)

A section's first non-blank line MAY start with `@type:` to tag the content type. This lets consumers render or parse the body correctly without guessing.

### 12.1 Syntax

```text
[context]
@type: code/python
def login(user): ...
```

The value is an advisory media-type-ish string. Suggested forms:

- `text/markdown` (default if omitted)
- `text/plain`
- `code/LANG` — e.g. `code/python`, `code/typescript`
- `diff/unified`
- `json`, `yaml`, `toml`

### 12.2 Rules

- `@type:` MUST appear as the first non-blank line of the section to count.
- An empty `@type:` value is a validation error.
- Unknown type values MUST NOT break parsing — consumers fall back to plain text.
- `@type:` SHOULD NOT be used on required sections where prose is the dominant form (`[goal]`, `[consolidated]`, `[identity]`) — prefer a dedicated section instead.

---

## 13. Content-addressed refs (v1.2)

A ref MAY be a content-addressed digest instead of a bare id. The receiver can then elide content it has already seen, the way a KV cache reuses already-computed key/value vectors across turns.

### 13.1 Syntax

```text
refs=sha256:9dbaddaac199f037b3cff1dff42bb46928a8cd26f86e3b197846a160d76626fc
```

Digest rules:

- Only `sha256:` is defined in v1.2.
- The hex part MUST be 16–64 lowercase hex characters. 64 is canonical; shorter forms are prefix matches (a receiver with a shorter seen-digest MAY still claim a match).
- `refs=` MAY mix digest refs and id refs.

### 13.2 Canonical form for hashing

The digest is computed over a normalized re-render of the crumb:

- parse and re-render via the standard renderer (sections emitted in document order)
- drop volatile headers: `id`, `dream_pass`, `dream_sessions`, and `refs` itself

This keeps a crumb's identity tied to its body, not to who last indexed it or what it currently points at.

### 13.3 Receiver "seen set"

A consumer MAY maintain a seen set — a persisted set of digests it has already loaded this session — and elide refs that match it. CRUMB's reference CLI stores this at `$CRUMB_SEEN_FILE` (default: `~/.crumb/seen`). The seen set is advisory; senders MUST NOT assume it exists.

---

## 14. Priority annotations (v1.2)

A section's body MAY start with `@priority: N` (optionally after `@type:`). Budget-aware packers drop the lowest-priority optional sections first.

### 14.1 Syntax

```text
[notes]
@priority: 2
- Low-priority scratch pad.

[rationale]
@priority: 8
- High-priority explanation.
```

### 14.2 Rules

- `N` MUST be an integer between 1 and 10 inclusive. `10` is highest priority, `1` is lowest.
- `@priority:` MUST appear as the first non-blank line of the section body, OR as the second such line when preceded by `@type:`.
- Required sections implicitly have priority `10` regardless of annotation and MUST NOT be dropped by a budget packer.
- Default priority for unmarked sections:
  - `[fold:X/summary]` → 9
  - optional non-fold sections → 5
  - `[fold:X/full]` → 4
- Unknown / missing annotations MUST NOT break parsing.

---

## 15. Delta crumbs (v1.2)

A `kind=delta` crumb carries only what changed between two crumbs, analogous to a residual after a polar projection. A receiver with the base reconstructs the target without a full resend.

### 15.1 Header

```text
v=1.2
kind=delta
base=sha256:<digest>
target=sha256:<digest>   # optional but recommended
source=...
```

`base=` is REQUIRED. `target=` is OPTIONAL but strongly recommended — it lets the receiver verify reconstruction.

### 15.2 `[changes]` section

Required. One operation per line:

```text
[changes]
- +[section] new text
- -[section] removed text
- ~[section] new text :: replaces :: old text
- ~[@headers] key=new :: replaces :: key=old
- +[@headers] key=value
- -[@headers] key=value
```

The pseudo-section `@headers` carries header diffs. Excluded keys (never diffed): `v`, `kind`, `base`, `target`, `id`. The text portion is preserved verbatim so bullet-vs-prose formatting roundtrips cleanly.

### 15.3 Apply semantics

When applying a delta to a base:

1. Parse both. Verify `base` digest matches the supplied base if present.
2. Apply `@headers` ops to the header map.
3. For each section op, replace/insert/remove lines within the named section by exact-line match.
4. Render the result. If `target=` was declared, verify the reconstructed digest matches.

Producers SHOULD emit operations in the order they should be applied. Consumers that cannot resolve a `-` operation (line missing from base) SHOULD continue with a warning rather than abort — delta apply is best-effort.

---

## 16. Budget-aware packing (v1.2)

A consumer MAY compress a crumb to fit a token budget by composing the additive primitives above. The order is intentional — the sender's declared priorities come before any lossy compression:

1. **Elide seen refs** (§13.3) — drop digest refs the receiver has confirmed.
2. **Drop `[fold:X/full]` variants** (§10) — the `/summary` survives.
3. **Drop lowest-priority optional sections** (§14) — never drop required sections.
4. **Escalate MeTalk** (levels 1 → 2 → 3) — dictionary, grammar strip, aggressive condensing.
5. If still over budget, fail loudly rather than truncate required content.

The reference CLI exposes this as `crumb squeeze --budget N`. A consumer MAY implement a subset — the ordering is prescriptive, but every step is optional.

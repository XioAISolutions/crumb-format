# .crumb Specification (v1.4)

**Status:** Stable
**Category:** AI handoff format
**Goal:** A tiny, human-readable protocol for portable AI context handoffs between tools and memory systems.

**Version compatibility:** v1.4 is backward-compatible with v1.1, v1.2, and v1.3. A v1.3 parser accepts a v1.4 file by ignoring unknown headers and sections (per §8). A v1.4 parser accepts `v ∈ {1.1, 1.2, 1.3, 1.4}`. Every v1.4 addition is optional and purely additive.

**Efficiency layers (§§13–16):** content-addressed refs, priority annotations, delta crumbs, and budget-aware packing. All additive — a v1.2 consumer that ignores them is still a compliant consumer.

**v1.3 additions (§§17–23):** normative ref resolution, normative fold selection, `[handoff]` dependencies, structured `[constraints]`, new optional sections (`[checks]`, `[guardrails]`, `[capabilities]`, `[script]`, `[workflow]`), `[invariants]` extended to `kind=task`, and a new `kind=agent` for reusable personas.

**v1.4 additions (§§11.4 and 21.1.1–21.1.2):** normative ISO-8601 format for `[handoff] deadline=`, normative typed `[checks]` thresholds (`value=`/`threshold=`/`op=`/`unit=` with sender consistency rule), and a closed-list canonical vocabulary for common agent failure-mode names. All three are warn-not-reject — sender freedom preserved, receivers gain a stable contract.

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
- `delta` (v1.2; see §15)
- `agent` (v1.3; see §23)

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

### 11.4 Normative `deadline=` format (v1.4)

When a `[handoff]` line carries a `deadline=` annotation, the value MUST be ISO-8601 in one of two forms:

- **Date-only:** `YYYY-MM-DD` (e.g. `2026-04-30`). Implicit timezone is the receiver's local zone. Use when time-of-day doesn't matter.
- **Datetime:** `YYYY-MM-DDTHH:MM:SS<tz>` where `<tz>` is `Z` or `±HH:MM` (e.g. `2026-04-30T17:00:00Z`, `2026-04-30T15:00:00+02:00`). The timezone suffix is REQUIRED — bare datetimes like `2026-04-30T15:00:00` are malformed.

Other ISO-8601 variants (no-seconds `HH:MM`, fractional seconds, second-precision offsets) are explicitly NOT permitted; the format is intentionally narrow so cross-language parsers behave identically.

**Validator behavior.** A v1.4 validator MUST emit a `WARN` for malformed `deadline=` values; it MUST NOT raise a parse error. Free-form `deadline=` values (the v1.3 behavior) continue to validate, just with a warning. `crumb lint --strict` MAY promote the warning to a non-zero exit.

**Past-due is not malformed.** A `deadline=` in the past is a valid handoff with an overdue annotation. Surfacing it is `crumb lint --check-deadlines`'s job, not the validator's.

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

---

## 17. Normative ref resolution (v1.3)

v1.2 §9 left `refs` resolution to implementations. v1.3 normatively specifies the default.

### 17.1 Resolution order

A conforming consumer MUST attempt resolution in this order and stop at the first hit:

1. **Bare id → local directory.** Look for `<id>.crumb` in a consumer-configured search path. Default search path: the current working directory, then `$CRUMB_HOME` (default `~/.crumb/`).
2. **`sha256:<hex>` digest → content store.** If a consumer maintains a content-addressed store (`$CRUMB_STORE`, default `~/.crumb/store/`), look up by digest.
3. **URL → fetch.** Only if network is explicitly enabled by the consumer. Default: disabled.
4. **Registry lookup.** Only if a registry is explicitly configured. Default: none.

A consumer MUST NOT fail a parse on an unresolvable ref. Refs are advisory.

### 17.2 Cycles

Consumers MUST walk refs with a visited-set. Default depth limit: **5**. Consumers MAY configure.

### 17.3 `crumb lint` contract

`crumb lint` SHOULD warn on unresolvable refs when invoked with `--check-refs`. Parsers MUST NOT warn.

---

## 18. Normative fold selection (v1.3)

v1.2 §10 left fold variant selection to implementations. v1.3 normatively specifies the default.

### 18.1 Size-greedy with summary floor

A conforming budget-aware consumer MUST:

1. Count every `[fold:X/summary]` as mandatory.
2. For each fold, attempt to upgrade `/summary` to `/full` in **declaration order** until the remaining token budget would be exceeded.
3. Never load both `/summary` and `/full` for the same NAME.

### 18.2 Writer override

A writer MAY declare `fold_priority=` as an optional header listing fold NAMEs in upgrade order:

```text
fold_priority=context, constraints
```

When present, consumers MUST honor this order over declaration order.

### 18.3 Both variants loaded

Forbidden. A consumer that loads both variants for the same NAME is non-conforming.

---

## 19. `[handoff]` dependencies (v1.3)

v1.2 §11 defines top-down ordering. v1.3 adds optional explicit dependencies.

### 19.1 Syntax

Two new optional keys on a `[handoff]` line:

- `id=<token>` — stable identifier. Tokens match `[a-zA-Z0-9_-]+`.
- `after=<token>[,<token>...]` — comma-separated dependency list.

```text
[handoff]
- id=repro   to=any    do=reproduce the failing test
- id=fix     to=any    do=propose a fix                  after=repro
- id=test    to=any    do=add regression test            after=fix
- id=review  to=human  do=approve before merge           after=test
```

### 19.2 Execution semantics

- A step with unmet `after=` dependencies is blocked and MUST NOT run.
- Earlier-line priority (v1.2 §11.2) applies as a tiebreaker among ready steps.
- A `- [x]` completed line satisfies dependencies on its id.
- Cycles MUST be detected. Consumers SHOULD warn and fall back to linear order.

### 19.3 Backward compatibility

A v1.2 consumer that sees `id=` / `after=` as unknown keys treats the whole line as a bullet (v1.2 §11.1).

---

## 20. Structured `[constraints]` lines (v1.3)

`[constraints]` bodies remain free prose. v1.3 formalizes an **optional** structured bullet form. Mixed bodies (prose + structured) are valid.

### 20.1 Syntax

```text
[constraints]
- Do not change the login UI                   # prose, unchanged
- deny=filesystem.write(/etc/**)               # structured
- require=regression_test_before_merge  why=audit
- prefer=incremental_patches
```

Recognized keys (all optional, all advisory):

- `deny=<expr>` — an action the receiving agent SHOULD NOT take.
- `require=<expr>` — a precondition the receiving agent SHOULD satisfy.
- `prefer=<expr>` — a soft preference.
- `why=<text>` — rationale.

Unknown keys do not invalidate a line.

---

## 21. New optional sections (v1.3)

All sections in this block are optional and carrier-only — CRUMB parsers do not execute or enforce their contents.

### 21.1 `[checks]`

Verification results at handoff time. One check per line in `name :: status` form, with optional trailing `key=value` annotations:

```text
[checks]
- tests.test_auth.py :: pass
- coverage :: 87%      threshold=85
- lint :: fail         note=unused import in auth.py:12
```

#### 21.1.1 Typed thresholds (v1.4 normative)

Four annotations are reserved with normative semantics:

| Annotation | Type | Meaning |
|---|---|---|
| `value=` | numeric or string | Observed value of the check. |
| `threshold=` | numeric or string | Sender-declared bound. |
| `op=` | one of `>=`, `<=`, `==`, `!=`, `>`, `<` | Comparison applied between `value` and `threshold`. Default: `>=`. |
| `unit=` | string | Documentation only; no semantic. Examples: `%`, `ms`, `MB`. |

`value=` and `threshold=` are numeric when both match `^-?\d+(\.\d+)?$`; otherwise they are strings (and `op=` is restricted to `==`/`!=`).

**Sender consistency rule.** When all of `value=`, `threshold=`, and a usable `op=` are present, the line's `<status>` MUST be:

- `pass` when `value <op> threshold` evaluates true,
- `fail` when it evaluates false.

`warn`, `skip`, and `pending` opt out of this rule (the sender can still observe a value but elect not to gate on it).

```text
[checks]
- coverage     :: pass    value=87       threshold=85    op=>=    unit=%
- latency_p99  :: warn    value=120      threshold=100   op=<=    unit=ms     note=regressed since main
- bundle_size  :: fail    value=1840     threshold=1500  op=<=    unit=KB
```

**Validator behavior.** A v1.4 validator MUST emit a `WARN` when status disagrees with the comparison; it MUST NOT raise a parse error. Other ISO-8601-style annotations on the same line are unaffected.

#### 21.1.2 Canonical failure-mode names (v1.4 normative-by-convention)

A closed-list vocabulary is defined for common agent failure modes so cross-tool consumers can act on `[checks]` lines without string heuristics:

| Name | When to use |
|---|---|
| `hallucinated_tool_call` | Tool name doesn't exist in the harness. |
| `refusal_loop` | Model refused N+ consecutive turns. |
| `tool_error_unhandled` | Tool returned an error and the model didn't recover. |
| `semantic_drift` | Output stopped addressing the stated goal. |
| `token_budget_exceeded` | Output exceeded a declared token budget. |
| `invalid_handoff_target` | A `[handoff]` line targets an unknown receiver. |
| `circular_reference` | A `refs=` chain or `[handoff] after=` graph contains a cycle. |
| `truncated_output` | Model output was cut off by max_tokens. |
| `prompt_injection_suspected` | Tool output contained instructions overriding the system prompt. |
| `unauthorized_tool_call` | Tool invocation violated the `[guardrails]` policy. |

Senders MAY use other names (validators continue to accept any). Receivers SHOULD recognize these canonical names when they appear. The list is closed for v1.4; future additions need a new spec amendment.

### 21.2 `[guardrails]`

Structured enforcement hints. CRUMB parsers do not enforce; downstream runtimes (AgentAuth, MCP policy engines, CI guards) MAY translate lines into real policy.

Recognized keys:

- `type=<tool|scope|verify|approval>`
- `deny=`, `require=`, `max=`, `min=`, `who=`, `action=`
- `why=<text>`

```text
[guardrails]
- type=tool      deny=shell-exec      why=security boundary
- type=approval  action=merge         who=human
```

### 21.3 `[capabilities]`

Handoff-time sender self-description. Recognized keys:

- `can=<expr>[,<expr>...]`
- `cannot=<expr>[,<expr>...]`
- `prefers=<expr>[,<expr>...]`

```text
[capabilities]
- can=read_files, run_tests, git_commit
- cannot=deploy_prod, shell_exec
- prefers=incremental_patches
```

When an AgentAuth passport is also present for the sender, the passport is authoritative.

### 21.4 `[script]`

Opaque executable-intent block. The body MUST begin with `@type: <lang>`. CRUMB parsers MUST NOT execute `[script]` bodies.

```text
[script]
@type: weave
@action: validate
---
agent.can("shell-exec") -> false
agent.must("run_tests") -> true
```

`crumb lint` SHOULD warn on `[script]` bodies larger than 2 KB.

### 21.5 `[workflow]`

Numbered multi-step state machine for orchestration use cases that outgrow `[handoff]`. Recognized keys on each line:

- `id=<token>` — stable id (defaults to the numeric prefix).
- `status=<pending|blocked|in_progress|done|skipped>`
- `owner=<expr>`
- `depends_on=<id>[,<id>...]`
- `why=<text>`

```text
[workflow]
1. reproduce_bug     status=pending     owner=any
2. write_test        status=blocked     owner=any      depends_on=1
3. implement_fix     status=blocked     owner=any      depends_on=2
4. human_approval    status=blocked     owner=human    depends_on=3
```

Cycle detection and unknown-dep rejection apply as in §19.2.

### 21.6 When to pick which

- `[handoff]` — linear or lightly-branched next steps (3–5 items).
- `[workflow]` — non-trivial ownership or cross-session status tracking.
- Both MAY coexist.

---

## 22. `[invariants]` on `kind=task` (v1.3)

`[invariants]` already exists on `kind=map` (§4.3) meaning architectural truths. v1.3 extends it to `kind=task` as an optional section capturing runtime assertions the receiving agent must maintain while performing the task.

```text
[invariants]
- auth.middleware completes before auth.state read
- tests.test_auth.py passes
- no new dependencies introduced
```

Distinctions:

- `[constraints]` — what the receiver must not do.
- `[invariants]` — what must remain true throughout the work.
- `[checks]` — what was verified before handoff.

Overlap is permitted.

---

## 23. `kind=agent` (v1.3)

A `kind=agent` crumb describes a reusable agent persona — identity, style, standing orders — without tying it to a specific task.

### 23.1 Required sections

- `[identity]` — role, style, scope.

### 23.2 Optional sections

- `[rules]` — standing orders.
- `[knowledge]` — expertise and learning areas.
- `[capabilities]` — see §21.3.
- `[guardrails]` — see §21.2.

### 23.3 Example

```text
BEGIN CRUMB
v=1.3
kind=agent
id=code-reviewer-v2
source=human.notes
---
[identity]
role=senior_reviewer
style=focus_on_edge_cases

[rules]
- never approve without tests

[knowledge]
- expert=python, typescript
- learning=rust
END CRUMB
```

### 23.4 Usage

A task crumb references an agent crumb via `refs=<agent-id>`. The receiving runtime loads the agent crumb first to establish persona, then processes the task.

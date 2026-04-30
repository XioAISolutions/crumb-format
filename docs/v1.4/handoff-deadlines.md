# v1.4 candidate — `[handoff]` `deadline=` deadlines normative

**Status:** Draft. Non-normative. Successor of [`typed-checks.md`](typed-checks.md). The narrowest of the v1.4 candidates from the scoping doc — pure additive convention with no schema change. The current SPEC permits a free-form `deadline=` on `[handoff]` lines but says nothing about format; v1.4 fixes that.

## Why

`[handoff]` already supports `deadline=` per SPEC §11.1, but the value is whatever the sender writes:

```text
[handoff]
- to=human  do=approve before merge   deadline=Friday
- to=any    do=fix before launch      deadline=2026-04-30
- to=any    do=ship                   deadline=ship-week
```

Three problems:

1. **Receivers can't parse it.** A consumer that wants to flag overdue handoffs has no machine-readable shape to operate on.
2. **No validation.** A typo like `deadline=2026-13-01` validates clean.
3. **Lint can't help.** `crumb lint` could highlight overdue handoffs but has nothing to compare against.

## Proposal in one paragraph

Pick **ISO-8601** as the normative format for `deadline=`. Validators warn on malformed values; never reject. `crumb lint` gains `--check-deadlines` that surfaces past-due handoffs. v1.3 parsers continue to accept any `deadline=` value as a free-form annotation, so adoption is gradual.

## Grammar (proposed v1.4)

The `deadline=` annotation on `[handoff]` lines MUST be one of:

| Form | Example | Notes |
|---|---|---|
| Date-only | `2026-04-30` | `YYYY-MM-DD`. Implicit timezone is the receiver's local zone. Use this when the time of day doesn't matter. |
| Datetime | `2026-04-30T15:00:00Z` | `YYYY-MM-DDTHH:MM:SS` with **required** timezone suffix (`Z` or `±HH:MM`). |

Anything else is malformed. Validators emit `WARN` on malformed values and continue. `crumb lint --strict --check-deadlines` promotes the warn to error.

### Why both forms

- Date-only is what humans actually write ("ship by Friday"). Forcing a time is friction.
- Datetimes need timezones because handoffs travel across time zones constantly. A bare `2026-04-30T15:00:00` is ambiguous and validators must reject it.

### Why ISO-8601

- One unambiguous global standard. Sortable as text.
- Already used informally in v1.3 examples (e.g. `created=2026-04-15T10:30:00Z`).
- Stdlib parseable in every language we care about (`datetime.fromisoformat` in Python 3.11+, `Date.parse` in JS, `chrono::DateTime::parse_from_rfc3339` in Rust).

### Why warn-not-reject

A handoff with a past deadline is not invalid — it's just overdue. The CRUMB document is still a valid handoff, the receiver just knows it's late. Rejecting would block a workflow we explicitly want to support: post-mortems on missed deadlines.

## Worked example

```text
BEGIN CRUMB
v=1.3
kind=task
title=Ship v0.9 release
source=ci.github
---
[goal]
Tag and publish v0.9 of crumb-format.

[context]
- All v1.4 draft features merged
- CHANGELOG written
- PyPI artifacts not yet built

[constraints]
- Don't ship if pytest is red on main
- Don't ship without CHANGELOG verified

[handoff]
- id=verify   to=any     do=run scripts/publish.sh --test on TestPyPI    deadline=2026-04-30
- id=tag      to=human   do=tag v0.9.0 and push                          deadline=2026-04-30T17:00:00Z   after=verify
- id=publish  to=human   do=run scripts/publish.sh                       deadline=2026-05-01T00:00:00Z   after=tag
END CRUMB
```

A v1.4-aware consumer can iterate `[handoff]` lines, parse `deadline=` into a real datetime, sort by urgency, surface overdue items.

## `crumb lint --check-deadlines`

New flag in `cli/linting.py`. Walks every `[handoff]` line. For each `deadline=`:

- **Malformed:** `WARN handoff line N: deadline=<value> is not ISO-8601 (expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS±HH:MM)`
- **Past:** `WARN handoff line N: deadline=<value> is overdue by <duration>`
- **Future:** silent (or under `--verbose`, an info line)

`--strict` promotes WARN to ERROR (exit 2, matching the existing lint exit code convention).

## Migration

Zero-cost. v1.3 senders that write a free-form `deadline=` continue to validate. v1.3 senders that adopt ISO-8601 voluntarily get nothing (no v1.3 parser checks the format) but their crumbs become useful when v1.4 receivers arrive.

For receivers, the path is: ignore `deadline=` (v1.3 behavior), then opt in to lint checks (v1.4), then surface them in UI (v1.5+).

## Reference implementation surface

When v1.4 lands normatively:

1. `validators/validate.py` — new `_validate_handoff_deadlines()` walking `[handoff]` lines, calling `datetime.fromisoformat` on each `deadline=`, raising on malformed dates only.
2. `validators/validate.js` — mirror using `Date.parse()` plus the explicit-timezone check.
3. `cli/linting.py` — `--check-deadlines` flag wires into the existing lint pass.

Estimated change: ~80 LOC across the three files. Tests: `tests/test_v14_deadlines.py` ~60 LOC, 6 cases (valid date, valid datetime with Z, valid datetime with offset, malformed month, missing timezone on datetime, past deadline → lint warning).

The wire-format version bump itself is a separate concern — SPEC §11.1 gains one paragraph. Per SPEC §8 backward compat holds: a v1.3 parser ignoring `deadline=` reads the same crumb just fine.

## Open questions

- **`until=` alias?** Some prior art in scoping doc used `until=` as a shorter alternative. Recommend declining — one canonical name (`deadline=`) is simpler. If a sender wants `until=`, they can keep using it as a free-form annotation; v1.4 just makes one specific name normative.
- **Recurring deadlines?** Out of scope. CRUMB is one-shot handoff; recurring belongs in workflow tooling.
- **Relative offsets?** `deadline=+24h`, `deadline=tomorrow`. Out of scope. Sender resolves to ISO-8601 at write time; if they want relative semantics they precompute.
- **Deadline on `[workflow]` steps?** v1.3 `[workflow]` lines already accept arbitrary `key=value`. Same v1.4 normative format would apply. Add a one-paragraph note to the SPEC update; no separate doc needed.

## What this doc explicitly is not

Not a SPEC.md amendment. Not a parser change. Not a normative recommendation today. Same shape as `typed-checks.md` — locks in the design before code lands so v1.4 doesn't accumulate unspecified-but-tolerated annotation conventions.

## Sequencing

This proposal is **smaller than typed-checks**. Recommend bundling both into the v1.4 normative bump:

1. SPEC §11.1 paragraph for `deadline=` ISO-8601 normative.
2. SPEC §15 paragraph for typed `[checks]` thresholds normative.
3. Validators: ~140 LOC across both for Python + JS mirroring.
4. Tests: ~140 LOC across both.
5. `crumb lint` flags: `--check-deadlines`, `--check-thresholds`.
6. CHANGELOG entry + version bump.

Total v1.4 release: ~400 LOC, all additive, all behind opt-in lint flags. Backward compatible.

# v1.4 candidate — Typed `[checks]` thresholds

**Status:** Draft. Non-normative. This doc is a proposal for what v1.4 of the wire format would say about `[checks]`. It compiles and parses today as v1.3 (annotations a v1.3 parser would ignore), so it can ship in this doc form before any wire-format bump.

**Lineage:** §44.2 of `docs/v1.4-scoping.md`. Picked up first because it's narrow, additive, and has a clear consumer story (CI integrations).

## Why

v1.3 `[checks]` is normative on the line shape — `<name> :: <status>` with optional trailing `key=value` annotations — but consumers can't act on a threshold without string-parsing the annotation. Three concrete pains:

1. **CI gating.** A consumer that wants to know "does this crumb pass our coverage gate?" has to parse `coverage :: 87% threshold=85`, normalize the units, and choose a comparison operator. Every consumer reinvents that.
2. **Status inconsistency.** A check can be marked `pass` while its `value < threshold`. Validators don't catch the contradiction.
3. **Heterogeneous types.** Some checks are percentages, some are durations, some are version ranges. There's no signal in the line about which is which.

## Proposal in one paragraph

Promote four annotations to **named** in v1.4: `value=`, `threshold=`, `op=`, and `unit=`. Define a small consistency rule: when both `value=` and `threshold=` are present and `op=` resolves to a comparison, the line's `<status>` MUST agree with the comparison. Validators warn-or-fail on mismatch. Everything else stays the same. v1.3 parsers continue to read these as ignored annotations.

## Grammar (proposed v1.4)

A `[checks]` entry remains:

```
- <name> :: <status>  [annotations]*
```

Five enumerated status values become normative (already de-facto):

```
pass | fail | warn | skip | pending
```

Four annotations become named:

| Annotation | Type | Meaning |
|---|---|---|
| `value=` | numeric or string | Observed value of the check. |
| `threshold=` | numeric or string | Sender-declared bound. |
| `op=` | one of `>=`, `<=`, `==`, `!=`, `>`, `<` | Comparison applied between `value` and `threshold`. Default `>=`. |
| `unit=` | string | Documentation only; no semantic. Examples: `%`, `ms`, `ns`, `MB`, `count`. |

Existing free-form annotations (`note=`, `since=`, etc.) are unaffected and continue to be permitted.

### Numeric form

`value=` and `threshold=` parse as numeric when both match `^-?\d+(\.\d+)?$`. The unit travels separately:

```
- coverage      :: pass    value=87     threshold=85    op=>=    unit=%
- latency_p99   :: pass    value=42     threshold=100   op=<=    unit=ms
- error_rate    :: fail    value=0.4    threshold=0.1   op=<=    unit=%
```

### String form

When either side is non-numeric, `op=` is restricted to `==` and `!=`:

```
- version       :: pass    value=1.2.3   threshold=1.2.3   op===
- region        :: fail    value=us-east threshold=eu-west op===
```

Range checks are out of scope for v1.4 — pick a third tool. (Future doc may add `lower=`/`upper=`.)

## Consistency rule

When all of `value=`, `threshold=`, and a usable `op=` are present, the `<status>` MUST be:

- `pass` when the comparison `value <op> threshold` evaluates true.
- `fail` when it evaluates false.

A check MAY still emit `warn`, `skip`, or `pending` regardless — those statuses opt out of the consistency check. This preserves the use case "I observed 87% but I'm warning until next sprint."

Validator behavior:

- v1.3 validator: ignores all of this. Continues to accept the line.
- v1.4 validator (proposed): emits a `WARN` when status disagrees with the comparison. Promotes to `ERROR` under `--strict`. Never emits an error if the check elides one of `value=` / `threshold=` / `op=` — the consistency rule applies only when the sender provides all three.

`crumb lint` gains a `--check-thresholds` flag that surfaces these warnings in CI-friendly output.

## Worked example

A v1.3 file using v1.4 conventions today (validates as v=1.3 because the annotations are ignored):

```text
BEGIN CRUMB
v=1.3
kind=task
title=Release gate
source=ci.github
---
[goal]
Decide whether to ship.

[context]
- Build green
- Coverage stable

[constraints]
- Don't ship if any blocker check fails

[checks]
- tests        :: pass
- coverage     :: pass    value=87       threshold=85    op=>=    unit=%
- latency_p99  :: warn    value=120      threshold=100   op=<=    unit=ms      note=regressed since main
- bundle_size  :: fail    value=1840     threshold=1500  op=<=    unit=KB
END CRUMB
```

A consumer that understands v1.4 typed checks can:

1. Iterate `[checks]` lines.
2. For each line with `value` / `threshold` / `op`, classify into `passing`, `failing`, `informational`.
3. Surface a single boolean "ready to ship" by AND-ing all `<status> != fail` (or any user-defined gate).

A consumer that doesn't understand v1.4 sees the original `<status>` field and acts on that, which the sender is already required to set consistently.

## Migration

Zero-cost. v1.3 senders that don't set `value`/`threshold`/`op` continue to validate. v1.3 receivers that don't read those fields continue to consume.

For senders that adopt the conventions early (today, before v1.4 is normative): emit `v=1.3`, document the annotations in your project README, and wait for parsers to upgrade. Sender-side consistency (status agrees with comparison) is a sender responsibility either way.

## Reference implementation surface

When v1.4 lands normatively, three small changes in `crumb-format`:

1. `validators/validate.py` — new `_validate_checks_consistency()` function, emits `WARN` per offending line.
2. `validators/validate.js` — mirror.
3. `cli/linting.py` — new `--check-thresholds` flag wires through to the validator pass.

Estimated change set: ~60 LOC across the three files. Plus tests in `tests/test_v14_typed_checks.py` (~80 LOC, 8 cases: numeric pass, numeric fail, default op, explicit op, string op, status-comparison agreement, status-comparison disagreement, missing-threshold-skipped).

The wire-format version bump itself is a separate concern — the SPEC §`<TBD>` text adds a paragraph to §15 (`[checks]`), `SUPPORTED_VERSIONS` adds `"1.4"`, and per SPEC §8 backward compat holds.

## Open questions

- **Unit normalization.** Should `value=42 unit=ms threshold=0.1 unit=s` be comparable? Current proposal: no — `value` and `threshold` must share the same `unit=`, and validators don't normalize. A separate v1.5 doc could add a normalization table if the use case shows up.
- **Boolean checks.** Some consumers want `value=true threshold=true op===`. Allowed under the string form, but feels heavy. Possibly add a v1.5 shorthand.
- **Multiple thresholds.** "Pass at 85, warn at 75". Out of scope; the sender already chooses one threshold and one status; layering adds complexity for marginal gain.

## What this doc explicitly is not

Not a SPEC.md amendment. Not a parser change. Not a normative recommendation. The intent is to lock in the design before any code lands so v1.4 doesn't accumulate the v1.2-style "annotations technically allowed but semantics vague" debt.

When the project is ready to ship v1.4, this doc becomes a SPEC §15 amendment and a small code patch. Until then, it's a cross-team agreement so anyone emitting `[checks]` can use the proposed conventions and have downstream tools converge on them.

# CRUMB conformance suite

Executable cases for the CRUMB wire format. An implementation is conformant if
it **accepts every `accept` case and rejects every `reject` case**.

Only the accept/reject decision is normative. Rejection *messages* are
deliberately unconstrained — phrase errors however suits your users.

## Why this exists

CRUMB called itself a standard while carrying three independent parsers across
three repositories, with no shared code and no test comparing them:

| Implementation | Emitted |
| --- | --- |
| `crumb-format` | `v=1.4` |
| `CrumbContext` | `v=1.3`, hand-written as a string literal |
| `CrumbLLM` | forked 334-line parser accepting 1.1–1.4 |
| `CrumbLLM` EJA schemas | required `crumb_version: "1.5"` — a version no spec defines |

Nothing detected the drift, because nothing compared them. A format whose own
ecosystem cannot agree on the current version number is a convention with
variants, not a standard.

The suite found a real interop bug the first time two implementations ran the
same cases: `validators/validate.js` accepted a malformed `sha256:` content ref
and accepted a `kind=delta` crumb with no `base` header — both of which the
reference parser rejects. A delta with no base cannot be applied to anything.
Both are fixed; both are now cases here.

## Running it

**Python** (the reference implementation, `crumb_core`):

```bash
python -m pytest tests/test_conformance.py -v
```

**Node** (`validators/validate.js`) — run over every case, expecting exit 0 for
`accept` and non-zero for `reject`. The pytest suite above drives both, so a
disagreement between them fails the build.

**Your implementation**: read `manifest.json`, and for each case parse
`cases/<id>.crumb`. The manifest gives you `expect`, the `spec` section the
case derives from, and a `reason` explaining what the case protects.

```json
{
  "id": "reject-delta-without-base",
  "file": "cases/reject-delta-without-base.crumb",
  "expect": "reject",
  "spec": "15",
  "reason": "A delta with no base cannot be applied."
}
```

## What is covered

31 cases across the normative rules:

- **Structure** (§2) — `BEGIN`/`END` markers, the `---` separator, `key=value`
  headers, body content outside any section, surrounding blank lines
- **Headers** (§3) — the three required headers, the supported-version set
  (`1.1`–`1.4` accepted; `1.0`, `1.5`, `2.0` and garbage rejected), the `kind`
  whitelist
- **Additive compatibility** (§8) — unknown headers and unknown sections are
  *ignored*, not rejected. This is the property that lets a v1.3 parser read a
  v1.4 file, and it is the one most easily broken by a stricter rewrite.
- **Required sections** (§4) — per-kind requirements, and the rule that a
  present-but-empty required section is a rejection
- **Folds** (§10) — a `fold:X/summary` + `fold:X/full` pair satisfies a
  required section; `/full` without `/summary` is rejected
- **Refs** (§9, §13) — id refs, empty-header rejection, `sha256:` digest shape
- **Deltas** (§15) — `base` header required, `[changes]` must be non-empty

Round-tripping is also asserted: `render(parse(x))` must itself parse, and to
the same headers and section names.

## Adding a case

Add the `.crumb` file to `cases/`, then an entry to `manifest.json` citing the
SPEC section and a reason of substance — `tests/test_conformance.py` enforces
both. New cases should encode a rule the spec already states. If the spec does
not state it, amend the spec first.

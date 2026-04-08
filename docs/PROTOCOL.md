# CRUMB Protocol

CRUMB is an open context interchange protocol for AI workflows.

The protocol goal is not “store everything forever.” The goal is to move the right context between tools in a form that is:

- portable
- deterministic
- human-readable
- diffable
- safe to validate

## Protocol shape

Canonical CRUMBs are plain text:

1. `BEGIN CRUMB`
2. header lines (`key=value`)
3. `---`
4. named sections (`[goal]`, `[context]`, and so on)
5. `END CRUMB`

The source of truth is always the text artifact, not an opaque local database.

## Core kinds

- `task` — goal, context, constraints for immediate handoff
- `mem` — durable facts and preferences
- `map` — project and module structure
- `log` — transcript or evidence trail
- `todo` — open and completed work items

## Extension model

CRUMB keeps `v=1.1` compatibility in this release.

The extension model follows three rules:

1. Existing parsers ignore unknown headers and sections.
2. New metadata should prefer optional headers over changing old section semantics.
3. New extension names must be namespaced and documented.

### Stable optional headers

These are part of the documented extension surface:

- `id=`
- `url=`
- `tags=`
- `extensions=`
- `max_total_tokens=`
- `max_index_tokens=`

### `extensions=` header

`extensions=` is a comma-separated list of extension identifiers:

```text
extensions=crumb.pack.v1, bridge.mempalace.export.v1
```

Extension identifiers should be namespaced. Good examples:

- `crumb.pack.v1`
- `bridge.mempalace.export.v1`
- `acme.audit.v1`

Bad examples:

- `pack`
- `v1`
- `custom`

### Namespaced custom headers

If a workflow needs extra headers, use namespaced keys such as:

```text
x-crumb-pack.strategy=hybrid
ext.acme.priority=high
```

Old readers should preserve them even if they do not understand them.

## Determinism

Deterministic behavior matters because CRUMBs are frequently:

- committed to git
- reviewed in pull requests
- generated in CI
- passed between tools that should agree on the same output

CRUMB therefore prefers:

- stable header ordering where practical
- explicit budgets
- file-based fixtures
- deterministic pack assembly by default

Optional local-model compression is allowed, but it should be additive and degrade gracefully.

## Human readability

Human readability is not a nice-to-have. It is a protocol constraint.

A CRUMB should be easy to:

- skim in a terminal
- paste into a chat window
- audit for secrets
- edit by hand
- diff in git

If a feature weakens those properties, it should be optional or rejected.

## Conformance

Protocol conformance is enforced with:

- reference validators in Python and Node
- golden fixtures under `fixtures/`
- CLI validation via `crumb validate`
- CI validation via the included workflow and GitHub Action

See [SPEC.md](../SPEC.md) for the formal grammar and [docs/COMPATIBILITY.md](COMPATIBILITY.md) for compatibility guarantees.

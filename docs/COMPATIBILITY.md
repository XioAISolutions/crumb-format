# Compatibility

CRUMB keeps `v=1.1` compatibility in this release.

That means old CRUMBs still work, current commands still work, and new metadata is added in a way that old readers can safely ignore.

## Compatibility rules

### Parsers

Parsers should:

- require `v`, `kind`, and `source`
- enforce the required sections for the chosen `kind`
- ignore unknown headers
- ignore unknown sections

### Writers

Writers should:

- keep the canonical text structure
- avoid changing the meaning of existing headers or sections
- prefer adding optional headers or new sections for extensions
- bump `v` only for breaking changes

## Extension-safe patterns

Safe:

```text
extensions=crumb.pack.v1
x-crumb-pack.strategy=hybrid
[sources]
- task.crumb
```

Unsafe:

- reinterpreting `[context]` to mean something else
- making old required sections optional
- changing `v=1.1` parsing rules without a version bump

## Golden fixtures

Conformance fixtures live in:

- `fixtures/valid/`
- `fixtures/invalid/`
- `fixtures/extensions/`

Each fixture has:

- an input `.crumb`
- an expected parse JSON or expected validation result

These fixtures are exercised through:

- Python tests
- the Python validator
- the Node validator
- CI in `.github/workflows/validate-examples.yml`

## Validator coverage

The reference validators now cover:

- `task`
- `mem`
- `map`
- `log`
- `todo`

Use them as the baseline behavior when implementing CRUMB elsewhere.

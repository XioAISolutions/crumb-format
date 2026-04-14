# Bridges

Bridge adapters let CRUMB sit on top of storage and retrieval systems without becoming trapped inside them.

The adapter rule is simple:

- canonical artifact in: `.crumb`
- backend-specific representation out: adapter layer

That keeps CRUMB protocol-first.

## Current backend: MemPalace

CRUMB now ships an adapter-shaped MemPalace bridge:

```bash
crumb bridge mempalace export --query "auth migration" --as task -o out/
crumb bridge mempalace import handoff.crumb -o mempalace-import.json
```

## Export

`crumb bridge mempalace export` can work in two modes:

1. Query the local MemPalace CLI directly
2. Consume a saved text export with `--input`

Supported flags:

- `--query`
- `--input`
- `--wing`
- `--room`
- `--entity`
- `--hall`
- `--as task|mem|log`
- `-o / --output`

### Export output kinds

- `--as task` — actionable handoff
- `--as mem` — durable facts
- `--as log` — transcript/evidence artifact

Generated CRUMBs include:

- `source=mempalace.bridge`
- `extensions=bridge.mempalace.export.v1`

### Graceful failure

If MemPalace is not installed and `--input` is not provided, CRUMB fails with a clear message explaining that the user can either:

- install MemPalace
- or pass a saved text export

## Import

`crumb bridge mempalace import` currently produces an adapter-ready JSON bundle.

That is intentional.

Direct writes into MemPalace internals are not safe enough yet to present as a default workflow. The current import command therefore prioritizes:

- clean structural mapping
- stable JSON output
- explicit documentation of what is and is not implemented

### Current import mapping

| CRUMB kind | Default MemPalace hall |
| --- | --- |
| `task` | `hall_events` |
| `mem` | `hall_facts` |
| `map` | `hall_discoveries` |
| `log` | `hall_events` |
| `todo` | `hall_advice` |

The output bundle keeps:

- wing
- room
- entity
- title
- original headers
- section lines

## Adapter architecture

Bridge code is adapter-based so future backends can slot in without rewriting the CLI contract:

- `mempalace`
- `sqlite`
- `vector`
- `plain-files`

The intended adapter contract is:

- `export(...) -> CRUMBs`
- `import(...) -> backend bundle or write report`

That keeps bridge logic composable and honest about what is implemented.

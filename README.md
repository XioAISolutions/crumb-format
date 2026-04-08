# CRUMB

CRUMB is the open context interchange standard for AI workflows.

It is a small, human-readable, local-first protocol for moving context between ChatGPT, Claude, Cursor, Gemini, MCP tools, local models, git workflows, and memory systems without getting trapped inside any one product.

CRUMB is designed to be:

- portable across tools and sessions
- deterministic by default
- compact under token budgets
- easy to validate, diff, commit, and hand-edit
- backward-compatible with existing `v=1.1` files

If you want a useful mental model, treat CRUMB like USB-C for agent context: a stable, boring transport layer that many systems can plug into.

## Why protocol beats siloed memory

Most AI memory products optimize for storing and retrieving data inside their own system.

CRUMB solves a different problem:

- you need a portable artifact that survives switching tools
- you need the next tool to act immediately instead of re-parsing a raw transcript
- you need context packs that fit token budgets
- you need import/export paths across local files, git, MCP tools, and memory backends

CRUMB does not replace retrieval systems. It sits above them.

## Quick example

```text
BEGIN CRUMB
v=1.1
kind=task
title=Fix auth redirect loop
source=cursor.agent
project=web-app
---
[goal]
Fix the auth redirect loop that happens after a hard refresh.

[context]
- App uses JWT cookie auth
- Redirect loop only happens on full page refresh
- Middleware reads auth state before cookie parsing stabilizes

[constraints]
- Do not change the login UI
- Preserve existing cookie names
- Add a regression check before merging
END CRUMB
```

That artifact is readable by a human, parsable by tools, easy to paste into another AI, and safe to keep in git.

## What CRUMB supports now

- `task`, `mem`, `map`, `log`, and `todo` CRUMBs
- deterministic pack building under token budgets with `crumb pack`
- bridge adapters with `crumb bridge mempalace ...`
- validation plus golden fixtures for conformance
- safety linting with `crumb lint`
- local compression with MeTalk and optional Ollama flows
- MCP server, REST API, browser extension, VS Code extension, git/CI integration

## Install

### pip

```bash
pip install crumb-format
```

### Homebrew

```bash
brew tap XioAISolutions/tap
brew install crumb
```

Then verify the install:

```bash
crumb --help
crumb validate examples/*.crumb
```

Homebrew release notes and tap automation live in [docs/HOMEBREW.md](docs/HOMEBREW.md).

## Quick start

### Create and validate a plain task crumb

```bash
crumb new task \
  --title "Fix auth redirect" \
  --goal "Fix the refresh redirect loop" \
  --context "JWT cookie auth" "Only happens on hard refresh" \
  --constraints "Keep cookie names stable" "Add a regression test" \
  -o handoff.crumb

crumb validate handoff.crumb
```

### Build a packed handoff under a budget

```bash
crumb pack \
  --dir ./crumbs \
  --query "auth redirect refresh" \
  --kind task \
  --mode implement \
  --max-total-tokens 1800 \
  --strategy hybrid \
  -o handoff.crumb
```

### Export a MemPalace search into CRUMBs

```bash
crumb bridge mempalace export \
  --query "auth migration" \
  --as task \
  -o out/
```

If MemPalace is unavailable, use a saved text export instead:

```bash
crumb bridge mempalace export \
  --input mempalace-search.txt \
  --query "auth migration" \
  --as mem \
  -o out/
```

### Lint for secrets and oversize logs

```bash
crumb lint handoff.crumb --secrets --strict
crumb lint handoff.crumb --secrets --redact
```

## Core commands

| Command | What it does |
| --- | --- |
| `crumb new` | Create a plain `task`, `mem`, `map`, `log`, or `todo` CRUMB |
| `crumb validate` | Validate one or more `.crumb` files |
| `crumb pack` | Assemble a deterministic context pack under a token budget |
| `crumb bridge mempalace export` | Turn MemPalace retrieval output into CRUMBs |
| `crumb bridge mempalace import` | Convert CRUMBs into an adapter-ready MemPalace bundle |
| `crumb lint` | Detect secrets, suspicious headers, budget overruns, and giant raw logs |
| `crumb search` | Search a CRUMB library by keyword, fuzzy match, or ranked relevance |
| `crumb export` / `crumb import` | Convert between CRUMB and JSON/Markdown surfaces |
| `crumb metalk` | Apply caveman compression to reduce handoff size |

Run `crumb --help` for the complete CLI surface.

## Protocol extensions

CRUMB stays on `v=1.1` for this release.

The parser remains permissive:

- unknown headers are preserved
- unknown sections are preserved
- old files still parse
- new tooling can add optional metadata without breaking older readers

The documented extension model now includes:

- optional core headers such as `id=`, `url=`, `tags=`, `extensions=`, `max_total_tokens=`, and `max_index_tokens=`
- namespaced extension names inside `extensions=`, for example `crumb.pack.v1`
- namespaced custom headers such as `x-crumb-pack.strategy=hybrid`

See [SPEC.md](SPEC.md), [docs/PROTOCOL.md](docs/PROTOCOL.md), and [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Context packs

`crumb pack` is the protocol-grade way to turn a directory of CRUMBs into one final handoff artifact.

It can:

- gather relevant CRUMBs from files
- incorporate repo tree and git diff signals when available
- rank results with `keyword`, `ranked`, `recent`, or `hybrid` strategies
- shape output for `implement`, `debug`, or `review` workflows
- rewrite the selected facts into a mode-aware brief instead of emitting a flat ranked dump
- keep the highest-signal entries first
- enforce `max_total_tokens`
- optionally run a final local Ollama compression pass

See [docs/PACKS.md](docs/PACKS.md).

## Bridges

Bridge adapters make CRUMB importable and exportable from storage/retrieval systems without coupling the protocol to any single backend.

Current adapter:

- `mempalace`

Planned adapter families:

- `sqlite`
- `vector`
- `plain-files`

See [docs/BRIDGES.md](docs/BRIDGES.md).

## Validation and conformance

CRUMB ships:

- Python and Node reference validators in [validators/](validators/)
- golden fixtures in [fixtures/](fixtures/)
- a GitHub Action and CI workflow that validate examples and fixtures

The goal is simple: outside tools should be able to implement CRUMB without reverse-engineering one CLI.

## Safety and local-first operation

Core CRUMB workflows do not require cloud APIs.

- deterministic pack building is local-first
- validators and linting are local
- bridge import/export can operate on saved text exports
- optional Ollama support stays local to `http://localhost:11434`

Security guidance and lint rules live in [docs/SECURITY.md](docs/SECURITY.md).

## Integrations

### MCP server

Expose CRUMB operations to Claude Desktop, Cursor, and other MCP clients:

```bash
python3 mcp/server.py
```

See [mcp/README.md](mcp/README.md).

### REST API

The repo includes a minimal local API server for validation, parse, and render flows:

```bash
python3 api/server.py
```

See [api/README.md](api/README.md).

### Browser extension

The browser extension can capture visible AI chat context and turn it into CRUMBs. See [browser-extension/README.md](browser-extension/README.md).

### VS Code extension

The VS Code extension ships syntax highlighting and snippets for all CRUMB kinds. See [vscode-extension/README.md](vscode-extension/README.md).

### Git / CI

- composite validation action: [action.yml](action.yml)
- example validation workflow: [.github/workflows/validate-examples.yml](.github/workflows/validate-examples.yml)
- pre-commit support: validate `.crumb` files before merge

## Examples

Examples live in [examples/](examples/):

- plain task crumb
- memory crumb
- map crumb
- packed task crumb
- MemPalace bridge export crumb

## Repository map

- [SPEC.md](SPEC.md) — formal CRUMB v1.1 grammar and rules
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — protocol framing and extension model
- [docs/PACKS.md](docs/PACKS.md) — context pack strategies and budget behavior
- [docs/BRIDGES.md](docs/BRIDGES.md) — bridge adapter architecture and MemPalace workflows
- [docs/SECURITY.md](docs/SECURITY.md) — linting, redaction, and safety guidance
- [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) — compatibility rules and fixture suite
- [examples/](examples/) — copy-pasteable CRUMB artifacts
- [fixtures/](fixtures/) — golden conformance fixtures
- [cli/crumb.py](cli/crumb.py) — reference CLI
- [validators/](validators/) — reference validators

## Design constraints

CRUMB deliberately avoids a few traps:

- it is not a cloud-only product
- it is not a database-first silo
- it does not trade readability for compression by default
- it does not break old files just to add new metadata

That is the point. Portable context wins when the protocol is smaller than the tools built around it.

# CRUMB Capsules, Relay, and Brain Bridge

## CRUMB Capsules

CRUMB Capsules are the sharable, resumable layer on top of the core CRUMB format.

A capsule takes a `.crumb` file and emits a small bundle:

- a preview card in Markdown
- a dark HTML share card
- a MeTalk-compressed resume payload
- a JSON metadata snapshot

This gives CRUMB a more viral surface without changing the underlying format.

### Why Capsules exist

Raw CRUMBs are already portable.
Capsules make them:

- easier to preview before pasting
- easier to share in issues, Slack, docs, and landing pages
- easier to resume in a target AI
- easier to screenshot and spread

### Usage

```bash
python capsule_cli.py create examples/task-bug-fix.crumb --target claude
```

Output files are written to `dist/capsules/` by default.

### Outputs

- `*.capsule.md` — preview and resume instructions
- `*.capsule.html` — dark visual capsule card
- `*.capsule.txt` — MeTalk resume payload
- `*.capsule.json` — metadata snapshot

### Design principles

- additive, not breaking
- based on CRUMB v1.1
- human-readable previews
- MeTalk for transport efficiency
- target-aware labels without target lock-in

### Next steps

- wire capsule generation into the main `crumb` CLI
- generate share links and QR codes
- add browser-extension one-click capsule export
- add signed integrity metadata

## CRUMB Relay

CRUMB Relay is the timeline view for handoffs.

It scans a directory of `.crumb` files and builds a lightweight event chain:
what moved, where it came from, what kind of crumb it was, and the first
meaningful preview line.

### Why Relay matters

Relay turns invisible AI handoffs into a visible chain of work.
That makes the product easier to debug, easier to trust, and easier to show.

### Usage

```bash
python capsule_cli.py relay .
```

Or emit JSON for downstream tooling:

```bash
python capsule_cli.py relay . --format json
```

### Current event model

Relay currently derives a timeline from available CRUMB fields such as:

- `dream_pass`
- `issued`
- `generated`
- title / kind / source
- the first visible goal or memory line

### Next steps

- wire Relay into the browser extension
- show AI-to-AI hop chains visually
- attach repo diff references
- publish Relay views as shareable pages

## CRUMB Brain Bridge

The brain bridge keeps CRUMB and the long-term memory layer separate but connected.

### Current implementation

This branch ships a filesystem adapter as the first working bridge:

- save `.crumb` files into a workspace-scoped brain directory
- maintain a simple `index.json`
- recall task or memory context into a fresh CRUMB

### Why this approach

The brain project is still evolving.
A filesystem adapter keeps the bridge real and usable now without forcing
CRUMB into a heavyweight backend too early.

### Usage

Save into the brain:

```bash
python brain_bridge_cli.py save examples/task-bug-fix.crumb --workspace demo
```

Recall as a new task crumb:

```bash
python brain_bridge_cli.py recall "auth redirect" --workspace demo --kind task
```

Recall as a memory crumb:

```bash
python brain_bridge_cli.py recall "auth redirect" --workspace demo --kind mem
```

### Planned upgrades

- main `crumb bridge brain` command
- better ranking and hybrid retrieval
- workspace / project / user scoping
- adapters for richer backends when the brain runtime settles

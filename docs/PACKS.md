# Context Packs

`crumb pack` builds one final CRUMB handoff from a directory of existing CRUMBs and local repository signals.

The command is designed for the moment when you do not want to paste a raw chat log or hand-curate context by hand.

## Command

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

## Supported flags

- `--dir` — source directory containing `.crumb` files
- `--query` — retrieval and ranking query
- `--project` — optional project filter/header
- `--kind task|mem|map` — output kind
- `--mode implement|debug|review` — shape the packed handoff for implementation, debugging, or review
- `--max-total-tokens` — estimated budget for the final packed artifact
- `--strategy keyword|ranked|recent|hybrid` — file ranking strategy
- `--output` / `-o` — output path
- `--ollama` — optional local-model compression pass
- `--ollama-model` — optional Ollama model override

## What `pack` includes

By default, `pack` combines:

- relevant CRUMBs from the target directory
- task/mem/map/log/todo lines with score-based ranking
- git diff summaries when the directory is inside a git repo
- repo tree hints for module-level context

## Modes

- `implement` — prefers task and memory crumbs, keeps context short, and rewrites the pack into an execution brief with labels such as `Start in`, `Current baseline`, and `Next step`
- `debug` — gives more trust to logs and symptoms, surfaces regression/testing evidence earlier, and rewrites the pack into a diagnosis brief with labels such as `Observed symptom`, `Likely cause`, and `Next check`
- `review` — pushes changed-file scope and review guardrails upward, and rewrites the pack into a merge-review brief with labels such as `Review scope`, `Affected module`, `Current invariant`, and `Missing check`

The mode is encoded in the packed artifact via `x-crumb-pack.mode=...`.

The exact mix depends on the output kind.

### `kind=task`

Produces:

- `[goal]`
- `[context]`
- `[constraints]`

Selection rules:

- task goals and constraints get the highest priority
- mem preferences can become constraints
- map and log content become context
- open todo items become context
- git diff and repo tree signals augment context
- the final context is mode-aware, so the same source corpus can render as an implementation brief, a debugging brief, or a review brief without changing the underlying facts

### `kind=mem`

Produces:

- `[consolidated]`

Selection rules:

- durable facts and preferences are preferred
- repeated lines are deduplicated
- lower-signal lines are pruned first

### `kind=map`

Produces:

- `[project]`
- `[modules]`

Selection rules:

- existing map crumbs win
- repo tree and changed-file signals fill gaps

## Strategies

- `keyword` — exact term matching
- `ranked` — relevance plus information density
- `recent` — newest files first, with light query bias
- `hybrid` — combined keyword + ranked + recency scoring

`hybrid` is the default because it behaves well for mixed corpora without hiding the scoring model.

## Budget behavior

`pack` is deterministic by default:

1. rank source files
2. extract candidate lines
3. deduplicate
4. keep the highest-signal entries first
5. prune low-signal lines until the artifact fits

If the required sections still cannot fit, the command fails clearly instead of emitting an invalid CRUMB.

## Optional local-model compression

When `--ollama` is set, `pack` performs the deterministic assembly first and then asks a local Ollama model to compress the result while preserving:

- `v=1.1`
- the output kind
- required sections
- human readability

If Ollama is unavailable, `pack` fails with a clear local-only error message.

## Output metadata

Packed CRUMBs include:

- `source=crumb.pack`
- a deterministic `id=`
- `max_total_tokens=...`
- `extensions=crumb.pack.v1`

Additional workflow metadata should use namespaced headers such as `x-crumb-pack.strategy=hybrid`.

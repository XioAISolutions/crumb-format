# Changelog

## v0.6.0

### v1.3 wire format

First release to bump the wire format from `v=1.2` to `v=1.3`. All additions are optional and purely additive. A v1.2 parser accepts a v1.3 file by ignoring unknown headers and sections (per SPEC §8); a v1.3 parser accepts `v ∈ {1.1, 1.2, 1.3}`.

**Closed v1.2 open questions**

- **Ref resolution (§1)** — normative order: bare id → local dir, `sha256:` → content store, URL (opt-in), registry (opt-in). Default depth limit 5 with visited-set cycle handling. New module `cli/ref_resolver.py`.
- **Fold heuristic (§2)** — size-greedy with summary floor. Writer override via new optional `fold_priority=` header. New helper `squeeze.select_folds_size_greedy()`.

**New primitives**

- **`kind=agent`** — reusable agent personas. Required `[identity]`; optional `[rules]`, `[knowledge]`, `[capabilities]`, `[guardrails]`.
- **`[handoff]` dependencies** — optional `id=<token>` and `after=<id>[,...]` on handoff lines for non-linear graphs. Cycle detection and unknown-dep rejection in the parser.
- **`[workflow]` section** — numbered steps with `status=`, `owner=`, `depends_on=`. Same cycle detection as `[handoff]`.
- **`[checks]` section** — verification results in `name :: status` form with trailing `key=value` annotations.
- **`[guardrails]` section** — structured enforcement hints (`type=`, `deny=`, `require=`, `who=`, `action=`) for downstream runtimes. Parsers do not enforce.
- **`[capabilities]` section** — handoff-time self-description (`can=`, `cannot=`, `prefers=`).
- **`[script]` section** — executable-intent carrier with required `@type:` first line. Parsers do not execute.
- **`[invariants]` on `kind=task`** — previously map-only; now allowed on task crumbs too.
- **Structured `[constraints]` lines** — optional `deny=`, `require=`, `prefer=`, `why=` keys alongside prose bullets. Unknown keys do not invalidate a line.

**Parser & validator**

- `cli/crumb.py`, `validators/validate.py`, `validators/validate.js` accept `v=1.3` and `kind=agent`, validate handoff/workflow dependency graphs, cycle-check both, and enforce `@type:` on `[script]`.
- `CLI_VERSION` bumped to `0.6.0`.

**Examples & tests**

- `examples/v13-agent.crumb`, `v13-handoff-deps.crumb`, `v13-checks.crumb`, `v13-guardrails.crumb`, `v13-workflow.crumb`, `v13-script.crumb`, `v13-fold-priority.crumb`.
- `tests/test_v13.py` — 35 new cases covering kind=agent, handoff deps, workflow, fold_priority, checks, script, ref resolver, size-greedy folds.

**Validator mirror fix**

- The Node validator (`validators/validate.js`) was missing `kind=delta` support from v0.5.0; now mirrors the Python validator's REQUIRED_SECTIONS entry.

## v0.5.0

### Efficiency layers (SPEC §§13-16)

Four additive layers that let a sender fit more signal into a fixed token budget without changing v1.2 compatibility. Inspired by the KV-cache-quantization playbook: compose decomposition, sparse references, delta encoding, and dictionary compression to push the token/information ratio down.

- **Content-addressed refs** — `refs=sha256:<hex>` entries identify a CRUMB by a stable digest computed over its canonical form (volatile `id`, `dream_pass`, `dream_sessions`, and `refs` headers excluded). A receiver's "seen set" (`$CRUMB_SEEN_FILE`, default `~/.crumb/seen`) lets the sender elide content the receiver already holds.
- **Priority annotations** — `@priority: 1..10` on a section body tells a budget-aware consumer which optional sections to drop first. Required sections are always priority 10.
- **Delta crumbs** — `kind=delta`, `base=sha256:...`, `target=sha256:...` carries only the `+` / `-` / `~` operations needed to reconstruct the target. Headers diffs travel in a pseudo-section `@headers`. Apply is verify-by-default: the reconstructed digest is checked against `target=`.
- **Budget-aware packing** — a prescriptive ordering that composes the above: elide seen refs → drop `[fold:X/full]` → drop lowest-priority optional sections → escalate MeTalk (1 → 2 → 3) → fail loudly if required content still won't fit.

### New CLI commands

- `crumb squeeze FILE --budget N` — apply the budget-aware packing order end to end. `--seen`, `--seen-hash`, `--no-seen`, `--metalk-max-level`, `--dry-run` for report-only.
- `crumb hash FILE [--short N]` — print a CRUMB's `sha256:<hex>` content digest.
- `crumb seen {add,remove,list,check,clear}` — manage the receiver-side seen set.
- `crumb delta BASE TARGET` — compute a `kind=delta` crumb.
- `crumb apply BASE DELTA` — reconstruct a target from its base + delta (verify by default).

### Parser & examples

- `cli/crumb.py` now recognises `kind=delta`, validates `@priority:` integer values and `refs=sha256:...` digest format, and parses `[changes]` entries with the `+` / `-` / `~` prefix grammar.
- New examples: `examples/v12-priority.crumb`, `examples/v12-content-ref.crumb`, `examples/v12-delta.crumb`.
- Tests: `tests/test_squeeze.py` covers all four layers (30 new cases).

All additions are v1.2-compatible. A v1.2 consumer that ignores §§13–16 is still a compliant consumer.

## v0.4.0

First release to bump the wire format itself from `v=1.1` to `v=1.2`. All four additions are optional, purely additive, and a v1.1 parser accepts a v1.2 file by ignoring unknown headers and sections (per `SPEC.md §8`).

### New Format Primitives

- **Cross-crumb references** — optional `refs=` header and `[refs]` section let a CRUMB point at other CRUMBs by id, turning an isolated handoff into a navigable graph. Resolution scheme left open for v1.3; see `docs/v1.2-ref-resolution.md`.
- **Foldable sections** — namespaced `[fold:NAME/summary]` + `[fold:NAME/full]` pairs carry short and long forms of a single logical section while preserving the flat grammar. A fold pair substitutes for the plain required section (fold-satisfies-required rule). Consumer selection heuristic left open for v1.3; see `docs/v1.2-fold-heuristic.md`.
- **`[handoff]` primitive** — optional explicit "next AI do this" block with advisory `to`/`do`/`why`/`deadline`/`ref` keys, distinct from `[goal]`.
- **Typed content annotations** — a section's first line MAY start with `@type: code/LANG` (or `diff/unified`, `json`, `yaml`, `toml`, `text/markdown`, `text/plain`) to tag content for consumers.

### Parser & Validator Updates

- `cli/crumb.py` accepts `v ∈ {1.1, 1.2}`, enforces the fold-satisfies-required rule, and validates v1.2 primitives as additive.
- Reference validators `validators/validate.py` and `validators/validate.js` updated to match.
- All v1.1 CRUMBs continue to validate unchanged.

### Examples & Docs

- `examples/v12-refs.crumb`, `v12-fold.crumb`, `v12-handoff.crumb`, `v12-typed-content.crumb` — one example per v1.2 feature.
- `docs/v1.2-ref-resolution.md`, `docs/v1.2-fold-heuristic.md` — open design docs framing the two deferred decisions for v1.3.

### Standalone Posture

Explicit decision: no bridge code to Weft/WeaveMind, LangGraph, n8n, or external orchestration runtimes. CRUMB stays plain text that travels via copy-paste, git, and clipboard. Existing bridges (`openai-threads`, `langchain-memory`, `crewai-task`, `autogen`, `claude-project`, MemPalace) continue unchanged.

### Unchanged from 0.3.0

Palace, Reflect, MeTalk, Wake, AgentAuth (passport/policy/audit/webhooks), deterministic `crumb pack`, MemPalace bridge, `crumb lint`, REST/A2A bridges, MCP servers, golden fixture suite, 41+ CLI commands — all ship unchanged in 0.4.0.

## v0.3.0

Protocol-grade release. CRUMB 0.3.0 turns the project from a useful handoff tool into a protocol-grade context workflow while keeping CRUMB file compatibility at `v=1.1`. The 0.3 track (deterministic packs, adapter bridges, safety linting, golden fixtures, extension model) is now merged into the main line alongside v0.2's Palace, Reflect, Wake, AgentAuth, and MeTalk surfaces.

### New Features

**Deterministic Context Packs**

- `crumb pack` builds a task/mem/map CRUMB from a directory of crumbs under a token budget
- Four ranking strategies: keyword, ranked (TF-IDF), recent, hybrid (default)
- Output shaping via `--mode implement|debug|review` — context sections shaped for the task you're handing off
- Optional `--ollama` final compression pass using a local model
- Git-aware context pickup: diff summaries, repo tree, recent files

**Bridge Adapter Surface**

- `crumb bridge mempalace export` — pull context from MemPalace (or saved export) into a new CRUMB
- `crumb bridge mempalace import` — convert CRUMB files into a MemPalace-ready adapter bundle
- Peer of the existing format bridges (`bridge export --to openai-threads`, etc.) — both coexist under the same subcommand
- Adapter abstract base class (`BridgeAdapter`) for future backends

**Safety Linting**

- `crumb lint` scans CRUMBs for credentials (OpenAI, GitHub, AWS, Slack, JWT, bearer, generic) with regex-based patterns
- `--redact` rewrites matches to `[REDACTED:label]` in-place or to `--output`
- `--max-size` warnings for oversized raw logs and overall CRUMBs
- Budget checks against `max_total_tokens` / `max_index_tokens` headers
- Namespaced header validation; `--strict` exits non-zero on any warning

**Extension Model**

- Documented optional headers: `id`, `url`, `tags`, `extensions`, `max_total_tokens`, `max_index_tokens`
- Namespaced extension names (`x-*`, `ext.*`, reverse-dns) with warnings in `crumb lint` for non-namespaced use
- `append_extension()` helper for programmatic extension declaration

**Golden Fixture Suite**

- `fixtures/valid/`, `fixtures/invalid/`, `fixtures/extensions/` with expected JSON / expected error files
- Python and Node validators (`validators/validate.py`, `validators/validate.js`) now walk directories and expand globs
- CI exercises the full fixture suite on every push and PR
- Enables third-party CRUMB implementations to verify conformance

### Improvements

- `crumb validate <dir>` and `crumb validate '*.crumb'` now work (glob + directory expansion)
- Git-aware helpers (`_git_repo_root`, `_build_repo_tree`, `.gitignore` handling) available to the scanner surface
- Two new MCP tools: `crumb_pack`, `crumb_lint` (tool list grows from 20 to 22)
- SPEC.md adds §8.1 documenting the extension model
- `log`, `todo`, and `wake` kinds now recognized by both reference validators

### Stats

- 323 tests passing (up from 291 in v0.2.0)
- 8 new modules: `cli/pack.py`, `cli/linting.py`, `cli/extensions.py`, `cli/local_ai.py`, `cli/mempalace_bridge.py`, plus fixture helpers
- Package layout unchanged: cli, agentauth, mcp, api, a2a

## v0.2.0

Major expansion: CRUMB grows from a simple handoff format into a full AI knowledge system with 41 CLI commands, persistent spatial memory, self-learning gap detection, agent governance, and cross-tool interoperability.

### New Features

**Palace Spatial Memory**

- Hierarchical knowledge base: wings (people/projects) > halls (facts/events/discoveries/preferences/advice) > rooms (topics)
- Pure filesystem storage -- every room is a `.crumb` file, grep-able, git-able, diff-able
- Auto-classification of observations into halls via rule-based keyword/regex patterns
- Cross-wing tunnel detection for related topics
- Session wake-up crumbs (`crumb wake`) for instant AI context bootstrap (~170 tokens)

**Self-Learning Gap Detection**

- `crumb reflect` scores palace health 0-100 with letter grades
- Detects 7 gap types: empty halls, thin wings, stale rooms, missing cross-wing halls, undocumented preferences, no discoveries, empty palace
- Actionable suggestions with exact commands to fill each gap
- `crumb palace wiki` generates a structured knowledge index
- `crumb wake --reflect` injects top knowledge gaps into session wake-ups

**MeTalk Caveman Compression**

- Three compression levels for AI-to-AI communication
- Level 1: dictionary substitution (lossless, reversible)
- Level 2: dict + grammar stripping (~40% token savings)
- Level 3: aggressive condensing (~50-60% token savings)
- Chainable with existing two-stage compression (`crumb compress --metalk`)

**AgentAuth -- Agent Identity & Governance**

- Cryptographic agent passports with registration, inspection, revocation
- Tool authorization policies with glob-pattern allow/deny rules
- Credential broker for secure secret access
- Full audit trail with risk scoring and evidence export
- Shadow AI scanner to discover unauthorized agents in projects
- Compliance reports (general, EU AI Act, SOC2)
- HTML dashboard for agent fleet overview

**Cross-AI Interoperability**

- REST API server (OpenAPI 3.1) with 15+ endpoints
- Google A2A protocol bridge (agent card, task handler)
- Format bridges: openai-threads, langchain-memory, crewai-task, autogen, claude-project
- Event webhooks for agent activity monitoring

**New CLI Commands**

- `crumb receive` -- clipboard/file/stdin intake with validation and palace auto-filing
- `crumb context` -- generate task crumbs from git state, palace facts, and open TODOs
- `crumb metalk` -- MeTalk compression encode/decode
- `crumb palace init/add/list/search/tunnel/stats/wiki` -- full palace management
- `crumb classify` -- standalone hall classification with explain mode
- `crumb wake` -- session bootstrap crumb generation
- `crumb reflect` -- knowledge gap analysis
- `crumb passport register/inspect/revoke/list` -- agent identity
- `crumb policy set/test` -- tool authorization
- `crumb audit export/feed` -- audit trail
- `crumb scan` -- shadow AI scanner
- `crumb comply` -- compliance reports
- `crumb dashboard` -- HTML agent dashboard
- `crumb bridge export/import/list` -- format conversion
- `crumb webhook add/list/remove/test` -- event hooks
- `crumb init --all` -- seed all AI tools at once (Claude, Cursor, Copilot, Gemini, etc.)

**MCP Servers**

- CRUMB MCP server for Claude Desktop/Cursor/Claude Code integration
- AgentAuth MCP server with 13 tools for agent governance

### Improvements

- Two-stage compression renamed for clarity (dedup + signal pruning)
- `crumb compress --metalk` adds MeTalk as optional Stage 3
- `crumb bench` updated with MeTalk stats
- `--version` flag added to CLI
- New examples: log, todo, and wake crumb files
- Quickstart guide (`docs/QUICKSTART.md`)

### Infrastructure

- GitHub Actions CI workflow for pytest on Python 3.10/3.11/3.12
- CRUMB Bench reusable workflow for PR quality gates
- Shadow AI scan workflow
- PyPI and ClawHub publish workflows
- Pre-commit hook for `.crumb` validation
- Fixed packaging: `api` and `a2a` modules now included in pip install

### Stats

- 41 CLI commands (up from ~12 in v0.1)
- 291 tests passing
- 6 Python packages: cli, agentauth, mcp, api, a2a, validators

## v0.1.0

Initial release. Core CRUMB format with parse, validate, create, inspect, search, merge, compact, diff, compress, bench, export, import, and handoff commands.

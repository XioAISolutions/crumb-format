# Changelog

## v0.4.0

Spec-freeze release. CRUMB 0.4.0 declares the `v=1.1` format **Stable**, ships a formal ABNF grammar, a conformance manifest for third-party implementations, an explicit threat model, and a tooling-stability policy. No file-format changes — every `v=1.1` document that worked under 0.3 still works.

### New Features

**Spec freeze**
- `SPEC.md` status promoted from **Draft** to **Stable**
- §2.1 grammar replaced with formal **ABNF (RFC 5234)**, plus seven explicit conformance notes (parsers MUST reject missing markers, MUST reject missing `---`, MUST reject duplicate header keys, SHOULD ignore unknown headers/sections, MUST emit lowercase keys, etc.)
- New §9 **"Stability and versioning"** locks the `v=1.x` line: structure, key syntax, six kinds, and CRLF/LF equivalence are frozen for all minor versions
- New §9.3 deprecation policy: at least one minor version of warning before removal; removals only at `v=2.0`

**Tooling stability policy**
- New `docs/STABILITY.md` documents the public surface for CLI, Python API, MCP, REST, A2A, and bridge formats
- Two independent versioning surfaces declared: format (`v=` header) and tooling (`crumb-format` package)
- Per-surface stability promises: stable subcommand names, stable JSON output, stable MCP tool names/schemas, stable OpenAPI contract

**Conformance manifest**
- New `fixtures/manifest.json` enumerates every fixture with id, kind, category, expected output, and conformance level
- Two conformance levels declared: `core` (5 fixtures, all v=1.1 essentials) and `extensions` (adds 2 namespaced extensions)
- Stable fixture IDs for the `v=1.x` line — additions allowed in minor releases, no renames or removals without a major bump

**Threat model**
- New `docs/THREAT_MODEL.md` enumerates 7 named threats (T1–T7): secret leakage, parser DoS, prompt injection, transit tampering, stolen passports, namespace squatting, supply-chain
- Documents trust boundaries, in-scope assets, residual risks, and explicit out-of-scope threats
- Reserves an optional `signature=` namespaced extension for future end-to-end integrity work

### Improvements

- `pyproject.toml` description updated to reflect spec-freeze posture
- CLI version bumped to `crumb 0.4.0`
- REST API `VERSION` constant bumped to `0.4.0`

### No breaking changes

Every CLI command, MCP tool, REST endpoint, and `.crumb` file accepted by 0.3 is still accepted by 0.4. `v=1.1` documents remain the canonical form.

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

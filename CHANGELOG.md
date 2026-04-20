# Changelog

## v0.5.0

Prompt-compression release. Adds the chat-box-style playground UI, a pure-JS port of the whole MeTalk + vowel-strip pipeline, a browser-extension popup that runs the algorithm fully in-browser, and the embedding-drift research tooling from the original vowel-strip work.

### Playground — chat-box style prompt compressor

- **`crumb playground`** subcommand boots the existing stdlib API server and opens a browser-based UI (`web/playground.html`) for live MeTalk L1-L5 compression. Paste a prompt or `.crumb` on the left, slide the level, copy the result. Auto-detects whether input is a CRUMB or plain prose.
- **`POST /metalk/compress`** endpoint added to `api/server.py` for programmatic use. Accepts `{text, level, vowel_min_length, adaptive_threshold, mode}`, returns `{encoded, stats}` with token counts, savings %, ratio, vowel retention, and detected mode.
- **Static file serving** added to `api/server.py` for the `web/` directory (with directory-traversal protection). `GET /` serves the playground.

### MeTalk vowel-strip (Layers 4 & 5)

- **Layer 4 — skeleton**: rule-based interior vowel removal added on top of the existing L1-L3 dictionary/grammar/condense pipeline (`cli/vowelstrip.py`). Skips section headers, header values, fenced code, v1.2 `@type: code/*` blocks, URLs, snake_case identifiers, file paths, contractions, all-caps acronyms, and a small `PROTECTED_WORDS` allowlist of confusable consonant skeletons.
- **Layer 5 — adaptive**: same strip, but per-line drift is measured against a sentence-transformers embedding and the strip is kept only if cosine similarity stays above `--adaptive-threshold` (default 0.85). Requires the optional `[embeddings]` extra; falls back to L4 with a warning when not installed.
- **`crumb vowelstrip` subcommand**: standalone use against any `.crumb` or plain text file, with `--min-length`, `--adaptive`, `--threshold`, `--plain` flags.
- **`crumb metalk --level 4` / `--level 5`** plus `--vowel-min-length` and `--adaptive-threshold` flags. The `compress`, `context`, and `wake` commands accept the new levels too.
- **Drift measurement harness**: `scripts/measure_drift.py` ranks each MeTalk level against an `examples/` corpus using either char-4-gram lexical similarity (default, no deps) or `sentence-transformers` semantic similarity. `--md` emits a markdown report.
- **Docs**: `docs/vowel-drift.md` methodology + tuning, `docs/vowel-drift-benchmark.md` bundled ngram benchmark across the 17 example crumbs.
- **Wire format**: encoded crumbs carry a new `vs=N` header recording the min-length threshold; `decode()` strips both `mt=` and `vs=` headers.
- **Optional dependency**: `pip install crumb-format[embeddings]` installs `sentence-transformers` for L5 and the semantic drift backend.

### Compare view + `/metalk/compare` endpoint

- **Playground "Compare all" tab**: renders L1-L5 side-by-side against the same input, highlights the level with the highest savings, per-card Copy button.
- **`POST /metalk/compare`** runs all five levels in one request and returns `{levels: [{level, encoded, stats}, ...], original_tokens}`.
- Server-side plain-text handling now wraps at all levels (previously L1-3 only) so L4/L5 plain prose gets the full dict + grammar + condense + vowel-strip pipeline — visible e.g. "authentication middleware configuration" → `Pls fix ath mw cfg.` at L4 (58% saved vs 50% at L1-3).

### Pure-JS port (runs fully in-browser)

- **`web/metalk.js`**: complete port of `cli/metalk.py` and `cli/vowelstrip.py` to a single UMD module — encode/decode/stripText/stripLine/compressionStats. Loads dictionary data from `web/metalk-data.json`.
- **`scripts/export_metalk_data.py`**: exports Python dicts/frozensets to the JSON file; Python remains the single source of truth.
- **`tests/test_js_port.py`**: drift-guard test that regenerates the JSON in memory and compares against the checked-in file, plus a Node-driver parametric test that asserts JS encode output matches Python byte-for-byte across every example crumb at L1-L4.
- **Playground falls back to the JS port** when the server is unreachable — the page now works served from `file://` or a static host.

### Browser extension v1.2.0 — prompt-compressor popup

- Rewritten `browser-extension/popup.html` + `popup.js` as a compact mini-playground: preset tabs, paste-from-clipboard, pull-current-page-selection (via `chrome.scripting.executeScript`), live debounced compression, `Cmd/Ctrl+Enter` to copy, `Cmd/Ctrl+1..5` to switch preset.
- Bundles `metalk.js` + `metalk-data.json`; **everything runs client-side** — no server round-trip, no network traffic. Works offline.
- Manifest updated to v1.2.0; added `scripting` and `storage` permissions, declared `metalk-data.json` as a web-accessible resource.
- Existing "Copy as CRUMB" context menu for ChatGPT / Claude / Gemini is unchanged.

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

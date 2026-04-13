# Changelog

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

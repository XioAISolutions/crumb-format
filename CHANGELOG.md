# Changelog

## v1.0.0

### v1.4 wire format normative + 1.0 release marker

The wire format is now stable. v=1.4 lands normatively, the SPEC is
labeled "Stable" rather than "Draft", and the project moves to its
first 1.0 release.

**Wire-format additions** (all backward-compatible per SPEC §8 — a
v1.3 parser accepts a v1.4 file by ignoring unknown semantics):

- **§11.4 — `[handoff] deadline=` ISO-8601 normative.** Two accepted
  forms: date-only `YYYY-MM-DD` (receiver-local) or datetime
  `YYYY-MM-DDTHH:MM:SS<tz>` with required `Z` or `±HH:MM` suffix.
  Other ISO-8601 variants explicitly NOT permitted (no fractional
  seconds, no second-precision offsets, no missing seconds).
  Validator emits WARN on malformed values, never raises.
- **§21.1.1 — typed `[checks]` thresholds.** Four annotations
  reserved with normative semantics: `value=`, `threshold=`, `op=`
  (default `>=`), `unit=`. When all of value/threshold/op are
  present, status MUST agree with the comparison (`pass` for true,
  `fail` for false). `warn`/`skip`/`pending` opt out.
- **§21.1.2 — canonical failure-mode names.** Ten closed-list names
  (`hallucinated_tool_call`, `refusal_loop`, `tool_error_unhandled`,
  `semantic_drift`, `token_budget_exceeded`, `invalid_handoff_target`,
  `circular_reference`, `truncated_output`, `prompt_injection_suspected`,
  `unauthorized_tool_call`) are normative-by-convention. Validators
  continue to accept any name; receivers SHOULD recognize these.

**Validators**

- `validators/validate.py` and `validators/validate.js` both add
  `"1.4"` to `SUPPORTED_VERSIONS`. v=1.5+ continues to reject
  (whitelist enforced; sanity test added).
- `cli/crumb.py` matches.

**Templates**

- `crumb new <kind>` defaults to emitting `v=1.4` for all 6 kinds.
- v=1.1, v=1.2, v=1.3 continue to validate on the parser side.

**Examples**

- `examples/v14-release-gate.crumb` — worked example combining
  normative deadlines, typed checks, and canonical failure-mode
  names. Validates clean against both validators.

**Tests**

- New `tests/test_v14_normative.py` (20 cases): version acceptance
  whitelist, template emission, SPEC.md amendments cross-checked
  against the runtime canonical-names list, Node validator parity.
- 637 tests passing total (was 614 at v0.11.0).

**1.0 commitment**

Post-1.0, the wire-format spec is treated as stable. Future
additions must be backward-compatible (additive headers and
sections only, per §8) or wait for a v2.0. CLI surface continues
to evolve under semver minor bumps.

CLI_VERSION → 1.0.0. pyproject.toml version → 1.0.0.

## v0.11.0

### Simplification + neutral naming

No wire-format change. This release removes accumulated complexity
and external-product affiliations from the user-facing CLI.

**Removed: 9 deprecated aliases.** The v0.7 deprecations are gone:
`compress`, `compact`, `squeeze`, `share`, `dashboard`, `todo-add`,
`todo-done`, `todo-list`, `todo-dream`. Their replacements have been
shipping for 4 releases:
- `compress` / `compact` / `squeeze` → `crumb optimize --mode {signal,minimal,budget}`
- `share` → `crumb handoff` (clipboard) or paste directly
- `dashboard` → `crumb audit export --format html`
- `todo-*` → `crumb todo {add,done,list,dream}`

**Renamed: `cli/mempalace_bridge.py` → `cli/memory_bridge.py`** to
match the actual scope (a memory-bridge framework with adapter
registry; MemPalace is just one adapter). `cli/mempalace_bridge.py`
remains as a one-line compat shim that re-exports from the new
location; remove in v0.12.

**Demoted: `crumb from-halo` is now a hidden alias of `crumb from-otel`.**
HALO traces are standard OTEL JSONL — `from-otel` is the canonical
name. `from-halo` still works but is hidden from `--help` and the
grouped command index. The bridge code is unchanged; only the surface
naming is now neutral.

**Two-tier `--help`.** Default `crumb --help` now shows only the
five core commands a new user reaches for (`new`, `validate`,
`handoff`, `receive`, `lint`) plus a pointer to `--help-all`. The
full grouped command index (~40 commands) moved behind
`crumb --help-all`. Surface-area complexity is opt-in.

**README rewrite.** Front-loads "Add crumb it to your AI" as Step 1
(no install required) and "Try it" as Step 2 (paste-only). The CLI
install moves to "Optional — install the CLI for power tooling"
much further down. Tagline becomes "The copy-paste AI handoff
format. No install required." Stale v0.6.0 release banner removed.

**Tests.** 604 passing. Updated `TestRemovedAliases` to confirm the
deprecated names now reject as unknown subcommands (exit 2).
Updated `TestGroupedHelp` to assert the new core-vs-all tier split.

CLI_VERSION → 0.11.0.

## v0.10.0

### HALO/OTEL bridge, v1.4 surface completion, release fix-ups

No wire-format change. No new dependencies. Format stays at v=1.3.

**HALO/OTEL bridge** (landed via #25)

- `cli/halo_bridge.py` — permissive OTEL JSONL parser. Handles flat-dict
  and OTLP attribute shapes, snake_case/camelCase keys, OTLP enum status
  codes (numeric + string), out-of-range timestamps, scalar status
  payloads, scalar `events` values, missing fields, and 4 envelope
  shapes (`resourceSpans` / `scopeSpans` / legacy
  `instrumentationLibrarySpans` / bare `spans` batches). Skips
  non-span objects (debug logs) via a `_looks_like_span` gate. 20
  Codex-caught defects, every one a real corner case from real OTEL
  exporters.
- `crumb from-halo <file>` and `crumb from-otel <file>` — same
  parser, two entry points. Listed under Create in grouped `--help`.
- `examples/halo-trace-to-log.crumb` — worked example.
- `docs/integrations/halo.md` — pipeline pattern doc.
- `docs/v1.4/agent-failure-modes.md` — 10 canonical `[checks]`
  names for common agent failure modes.

**v1.4 surface completion**

- `cli/failure_modes.py` + `crumb lint --check-failure-modes` —
  wires the canonical vocabulary into runtime. Emits INFO findings
  for canonical names; suggests canonical replacements for ad-hoc
  names via a 21+ pattern heuristic table.
- `validators/validate.js` — JS deadline parser mirrors
  `cli/deadlines.py`. Same form-first dispatch, same calendar
  round-trip via `Date.UTC`, same explicit `Z`/`±HH:MM` regex.
  Exported alongside `parseCrumb`.

**Release fix-ups**

- Deprecation calendar bumped v0.9 → v0.10. The v0.7 deprecations
  (compress, compact, squeeze, share, dashboard, todo-add/-done/
  -list/-dream) now scheduled for removal in v0.10. All shims still
  function with `[deprecated]` stderr hints.
- README "Try it right now" example modernized v=1.1 → v=1.3 with
  a back-compat note that all three versions validate.
- `crumb delta` same-file is now exit 0 with `Note:` instead of
  exit 1 with `Error:`. No-op is not a failure for CI scripts
  diffing for change detection.

**Tests.** 595 passing (was 500 at v0.9.0). New: 40 halo_bridge,
33 failure_modes, 16 JS deadline harness, 1 delta-no-op. All four
new modules ship in the wheel.

CLI_VERSION → 0.10.0.

## v0.9.0

### v1.4 deadlines — first implementation behind opt-in flag

No wire-format change. Format stays at v=1.3. This release implements
the v0.8.x deadlines design doc (`docs/v1.4/handoff-deadlines.md`) and
lets adopters opt in via a new lint flag before v1.4 lands normatively.

**`cli/deadlines.py`** — strict ISO-8601 parser for `[handoff] deadline=`
annotations. Two accepted forms:
- Date-only `YYYY-MM-DD` (receiver-local).
- Datetime `YYYY-MM-DDTHH:MM:SS` with required `Z` or `±HH:MM` suffix.

Anything else raises `DeadlineParseError`. Public surface:
`parse_deadline`, `is_overdue`, `check_deadline_lines`.

**`crumb lint --check-deadlines`** — new opt-in flag. Walks `[handoff]`
lines; emits `WARN malformed_deadline` for non-ISO-8601 values and
`WARN overdue_deadline` for past-due deadlines. Off by default.
`--strict` promotes to exit 1 (matching the existing strict-warning
convention; exit 2 stays reserved for parse failures, per the v0.7.0
unification).

**Backward compat.** Free-form `deadline=` values continue to validate.
Receivers that don't pass `--check-deadlines` see no behavior change.

**Tests.** 28 new cases in `tests/test_deadlines.py`. Each Codex-found
defect from PR #23's review (10 across 6 rounds) plus PR #24's review
(naive-`now` comparison crash) maps to a dedicated test. 500 tests
passing (was 472).

**Out of scope.** JS validator mirror, SPEC.md amendment, wire-format
bump. v1.4 is still scoping; ship a Python-side reference impl now
and mirror to JS in a follow-up.

## v0.8.0

### Guardrails bridge, MCP v1.3 surface, CI bench fix

No wire-format changes. v0.8.0 makes a normative SPEC SHOULD actually
work, exposes the v1.3 surface through MCP, and fixes the CI workflows
that were silently producing N/A bench scores for newer crumbs.

**`[guardrails]` → AgentAuth bridge (SPEC §21.2 implemented)**

Was aspirational in v0.6.0 ("AgentAuth-aware runtimes SHOULD translate"),
now there is code:

- `cli/guardrails.py` — `parse_guardrail_line`, `translate_guardrails`,
  `apply_guardrails_to_policy`. Dry-run by default.
- `crumb guardrails <file>` — preview the translation. `--apply --agent-name <name>`
  actually sets policy via `agentauth.ToolPolicy`.
- 11 new tests in `tests/test_guardrails.py` covering parsing, bucketing,
  dry-run, real application, and no-op paths.

Smoke against `examples/v13-guardrails.crumb`:
```
[DRY-RUN] agent_name=unknown-agent
  deny:      shell-exec
  require:   tests
  approval:  merge by human
  scope:     files=5
```

Listed under Governance in `crumb --help`.

**MCP surface exposes v1.3**

`mcp/server.py`:
- `crumb_new` accepts `kind=agent` plus `agent_id` / `identity` / `rules` / `knowledge`.
- `crumb_lint` accepts `check_refs`.
- New tools: `crumb_resolve`, `crumb_guardrails`.
- Tool count went from 22 → 24.

**CI workflow fixes**

- `bench-pr.yml`, `auto-crumb-pr.yml`, `auto-crumb-template.yml` all installed
  via `pip install crumb-format`, which pinned to the last PyPI release.
  Result: `crumb bench` returned `N/A` for every crumb using features newer
  than that release (v=1.2, v=1.3, kind=agent, kind=delta). All three
  workflows now install from the local checkout when running inside the
  `crumb-format` repo (detected via `pyproject.toml` name check), and from
  PyPI when called from another repo. External callers keep working.
- `auto-crumb-template.yml` was triggering on `[opened, synchronize]`,
  which made every push to a PR auto-commit a new crumb (which was itself
  a push that triggered another commit). Trigger scoped to `[opened, reopened]`.

**v1.4 scoping doc**

`docs/v1.4-scoping.md` is a non-normative parking lot for next-bump
candidates: `kind=review`, typed `[checks]` thresholds, `kind=receipt`,
`[handoff] until=` deadlines normative. Explicitly scoping-only — nothing
committed.

**Tests**

466 passing (was 455). Adds 11 guardrails tests. All pre-v0.8 deprecation
shims still honored — `compress`, `compact`, `squeeze`, `share`,
`dashboard`, `todo-add`/`-done`/`-list`/`-dream` continue to function with
their `[deprecated]` stderr hint. Removal still scheduled for the release
after v0.8 (i.e. v0.9), since v0.7 was the announce release and v0.8 is
the second release of the deprecation window.

## v0.7.0

### Usability and simplicity pass

No wire-format changes. This release is about reducing surface friction and
fixing places where the repo contradicted its own docs.

**Dissonance fixes**

- `crumb new <kind>` templates now emit `v=1.3` for every kind (was `v=1.1`
  for everything except `agent`). README documented v1.3 features the default
  emitter wouldn't produce.
- Unknown-kind errors now enumerate valid kinds. `unknown kind: 'frogpile'`
  → `unknown kind: 'frogpile'. valid: agent, delta, log, map, mem, task, todo, wake`.
- AgentAuth's `.crumb-auth/` storage now prints a one-time stderr notice on
  first creation so users aren't surprised by a new directory tree appearing
  silently. Suppress with `CRUMB_QUIET=1`.
- `crumb validate` exit code on parse error is now `2` (was `1`), matching
  `crumb lint` and pytest convention. Scripts pinned to exit `1` need updating.
- Missing-section error simplified: `missing required section for kind=task: [goal]`
  (no fold-syntax tail). Run `crumb validate --hint <file>` to see the fold
  alternative when relevant.
- `crumb handoff` now exits `1` and warns explicitly when every clipboard
  tool fails (pbcopy / xclip / xsel / clip.exe), rather than printing the
  crumb to stderr and silently exiting `0`.
- `crumb optimize --mode budget` failure suggests recovery (`--metalk-max-level`,
  `--seen`, raise budget) instead of just "cannot squeeze".

**Surface simplification**

- New `crumb todo {add,done,list,dream}` nested form (mirrors `palace`,
  `passport`, etc.). Old `todo-add` / `todo-done` / `todo-list` / `todo-dream`
  remain as deprecated aliases for one release; they print a `[deprecated]`
  hint to stderr and call through to the new handler.
- New `crumb optimize --mode {minimal,signal,budget}` replaces the three
  separate `compact` / `compress` / `squeeze` commands. Old commands are
  deprecated aliases for one release.
- `crumb share` and `crumb dashboard` are now deprecated (removal scheduled
  for v0.8). Replacements: `crumb handoff` (clipboard) and
  `crumb audit export --format html` respectively.
- `crumb --help` now opens with a grouped command index (Create, Inspect,
  Edit, Optimize, Handoff, Memory, Format, Governance, Todo, Other) instead
  of an alphabetical dump.

**Onboarding**

- README front-loads the "Add 'crumb it' to your AI" prompt above the v1.2
  / v1.3 spec deep-dives so users adopting via AI custom instructions see
  the most relevant content first.
- README's "Configuration" subsection documents `CRUMB_HOME`, `CRUMB_STORE`,
  `CRUMB_SEEN_FILE`, and `CRUMB_QUIET`.
- `docs/QUICKSTART.md` adds a "0. (Optional) One-time setup" block calling
  out `crumb palace init` for the Palace / Wake / Receive flows.
- `examples/v13-script.crumb` replaced. The original Weave-DSL example was
  abstract; the new one carries a literal shell verification command
  (`pytest tests/test_auth.py -q`) that a receiver can run to reproduce the
  sender's checks.

**Tests**

- New `tests/test_usability.py` with 21 cases covering template versions,
  deprecation aliases, optimize modes, exit codes, unknown-kind enumeration,
  AgentAuth notice, and grouped-help rendering.
- 455 tests passing (was 445).

**Deprecations (removal scheduled for v0.9 — see v0.8.0 note above)**

`compress`, `compact`, `squeeze`, `share`, `dashboard`, `todo-add`,
`todo-done`, `todo-list`, `todo-dream`. All print `[deprecated]` hints and
continue to function.

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

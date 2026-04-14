# Stability and Versioning

This document is the operational companion to `SPEC.md` §9. It declares what the CRUMB project promises to keep stable, what is allowed to change, and how breaking changes are signalled.

There are **two** independent versioning surfaces:

1. **The format** — tracked by the `v=` header inside every `.crumb` document.
2. **The tooling** — tracked by the `crumb-format` Python package version, the MCP server, the REST/A2A bridges, and the AgentAuth SDK.

The `v=` header is **not** coupled to the package version. A `crumb-format==2.4.0` release may still write `v=1.1` documents.

---

## 1. Format stability (`v=1.x`)

See `SPEC.md` §9 for the normative rules. In short:

- Every guarantee made at `v=1.1` holds for the full `v=1.x` line.
- Minor versions (`v=1.2`, `v=1.3`, …) only **add** optional headers, sections, or kinds.
- Breaking changes require `v=2.0` and ship in a separate `SPEC-2.md`.
- Deprecation requires at least one minor version of warning before any tool stops emitting the deprecated feature.

A `v=1.1`-only parser will continue to parse every future `v=1.x` document, ignoring unknown headers and sections per §8.

---

## 2. Tooling stability

The `crumb-format` Python package follows [SemVer 2.0](https://semver.org). The public surface — what we promise not to break in a minor release — is enumerated below. Anything not listed here is internal and may change without notice.

### 2.1 CLI (`crumb`)

**Stable at 1.0:**

- The set of subcommands documented in `crumb --help`.
- The names and types of all documented flags for each subcommand.
- The exit codes: `0` (success), `1` (validation/runtime failure), `2` (argument error).
- The structure of `--format json` output for any subcommand that supports it.

**Not stable:**

- Human-readable text output formatting (colors, exact wording, table layout).
- Internal helper modules under `cli/` not exposed as commands.

Adding new subcommands or new optional flags is a **minor** release. Renaming or removing a documented subcommand or flag requires a **major** release with at least one minor version of deprecation warning.

### 2.2 Python API

The following modules are stable public API:

- `cli.crumb` — the `parse_crumb`, `render_crumb`, and `validate_crumb` functions (signatures, return shapes).
- `cli.errors` *(added 0.7.0)* — `CrumbError` base class and its ten subclasses (`MissingMarkerError`, `MissingEndMarkerError`, `MissingSeparatorError`, `InvalidHeaderLineError`, `MissingHeaderError`, `BadVersionError`, `UnknownKindError`, `OrphanBodyError`, `MissingSectionError`, `EmptySectionError`). All subclass `ValueError` for backwards compatibility.
- `cli.extensions` — `SPEC_URL`, `append_extension`, `parse_extensions`.
- `cli.metalk` — `encode`, `decode`, `compression_stats`.
- `cli.handoff` — `emit_task`, `emit_mem`, `walk_chain`, `validate_chain`, `new_id`, `ChainError`.
- `agentauth` — `AgentPassport`, `ToolPolicy`, `CredentialBroker`, `protect`.

Internal helpers (anything prefixed `_`) and submodules not listed above are not part of the public API.

#### 2.2.1 Parser error codes (added 0.7.0)

The string constants exported from `cli.errors` are part of the stable surface — third-party tooling may compare `exc.code` against them:

- `E_MISSING_MARKER`, `E_MISSING_END_MARKER`, `E_MISSING_SEPARATOR`
- `E_INVALID_HEADER_LINE`, `E_MISSING_HEADER`, `E_BAD_VERSION`, `E_UNKNOWN_KIND`
- `E_ORPHAN_BODY`, `E_MISSING_SECTION`, `E_EMPTY_SECTION`

New error codes may be added in minor releases. Existing codes are not renamed or reused before a major bump. The exception *message strings* raised by `parse_crumb()` are also frozen for the 0.x line — they appear in CLI output, fixture `.expected.txt` files, and downstream log messages.

#### 2.2.2 Parsed-shape JSON Schema (added 0.7.0)

`schemas/crumb.schema.json` is a formal Draft 2020-12 contract for the dict returned by `parse_crumb()`. It complements the text-format ABNF in `SPEC.md` §2.1. New optional headers/sections may be added in minor releases; the required set is frozen until `v=2.0`.

### 2.3 MCP server

The tool **names** and **input schemas** declared by `mcp/server.py` and `mcp/agentauth_server.py` are stable. Adding new tools is a minor release. Renaming or removing a tool, or breaking an input schema, requires a major release.

### 2.4 REST API

`api/server.py` exposes an OpenAPI 3.1 contract. The endpoint paths, HTTP methods, request/response schemas, and status codes documented in the spec are stable. Adding new endpoints or new optional fields is a minor release.

### 2.5 A2A bridge

The agent card schema and task-handler contract in `a2a/` follow the upstream Google A2A protocol. We track A2A protocol version compatibility in `a2a/README.md`.

### 2.6 Bridge formats

Each named format under `crumb bridge` (`openai-threads`, `langchain-memory`, `crewai-task`, `autogen`, `claude-project`, `mempalace`) is a stable target. The output schema for each format is captured by golden fixtures in `fixtures/`. Format **additions** are a minor release. Format **removals** are major.

---

## 3. Deprecation policy

When a feature is deprecated:

1. The next minor release marks it `Deprecated` in `SPEC.md`, `CHANGELOG.md`, and inline (e.g., a deprecation warning printed by the CLI).
2. The feature continues to work for at least one full minor cycle.
3. Removal happens at the next major version.

Security-critical removals (e.g., a vulnerable header or section) may bypass the deprecation cycle but MUST be called out explicitly in `CHANGELOG.md` and `docs/SECURITY.md`.

---

## 4. Versioning conventions

| Surface             | Versioning                                  | Where it lives                |
|---------------------|---------------------------------------------|-------------------------------|
| Format              | `v=MAJOR.MINOR` inside every `.crumb`       | `SPEC.md`, header field `v`   |
| `crumb-format` pkg  | SemVer 2.0 (`MAJOR.MINOR.PATCH`)            | `pyproject.toml`              |
| MCP tool surface    | Tracks `crumb-format` package version       | `mcp/server.py` `__version__` |
| REST API            | Tracks `crumb-format` package version       | `api/server.py` `VERSION`     |
| AgentAuth SDK       | Tracks `crumb-format` package version       | `agentauth/__init__.py`       |

Pre-1.0 packages (`0.x.y`) follow the SemVer pre-1.0 convention: minor bumps may break public API, patch bumps are additive or fix-only. From 1.0 onward, the rules in §2 apply strictly.

---

## 5. How we communicate changes

Every release ships:

- A `CHANGELOG.md` entry under the new version, with **Added / Changed / Deprecated / Removed / Fixed / Security** subsections.
- A git tag of the form `vMAJOR.MINOR.PATCH`.
- For format changes, a corresponding update to `SPEC.md` (and `SPEC-2.md` for v2 work).

If you depend on CRUMB and want to be notified of stability-affecting changes, watch the `CHANGELOG.md` file or subscribe to GitHub releases for `XioAISolutions/crumb-format`.

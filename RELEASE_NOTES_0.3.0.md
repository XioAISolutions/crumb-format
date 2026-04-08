# CRUMB 0.3.0 Release Notes

## Summary

CRUMB 0.3.0 turns the project from a useful handoff tool into a protocol-grade context workflow.

This release keeps CRUMB file compatibility at `v=1.1`, while adding deterministic context packing, bridge adapters, safety linting, a documented extension model, golden fixtures, and stronger protocol docs.

## Highlights

- `crumb pack` builds deterministic context packs under token budgets
- `crumb bridge mempalace export|import` adds the first adapter-based bridge surface
- `crumb lint` catches secrets, suspicious headers, oversized raw logs, and budget overruns
- protocol fixtures and reference validators make external implementations easier to test
- pack output is now mode-aware for `implement`, `debug`, and `review` workflows

## Included in 0.3.0

### Protocol workflow

- Added `crumb pack` for deterministic context pack assembly from CRUMBs plus local repo signals
- Added support for output shaping via `--mode implement|debug|review`
- Added optional local Ollama compression after deterministic pack assembly

### Bridges

- Added `crumb bridge mempalace export`
- Added `crumb bridge mempalace import`
- Established adapter boundaries for future bridge backends

### Safety and conformance

- Added `crumb lint` with secret scanning, redaction support, size warnings, and strict mode
- Added golden fixtures under `fixtures/valid`, `fixtures/invalid`, and `fixtures/extensions`
- Expanded Python and Node validators to cover the extension surface

### Spec and docs

- Documented the extension model for optional headers like `id`, `url`, `tags`, `extensions`, `max_total_tokens`, and `max_index_tokens`
- Repositioned the README and protocol docs around CRUMB as an open context interchange standard
- Added dedicated docs for packs, bridges, security, protocol semantics, and compatibility

## Verification

- `pytest -q tests/test_protocol_features.py tests/test_crumb.py tests/test_metalk.py`
- protocol acceptance flow:
  - `crumb pack`
  - `crumb lint --strict`
  - `crumb bridge mempalace export`
  - `crumb validate`
- Python and Node validators both pass valid and extension fixtures and reject invalid fixtures

## Suggested short release text

CRUMB 0.3.0 adds deterministic context packs, bridge adapters, safety linting, an extension model, and golden fixtures while keeping CRUMB files compatible at `v=1.1`.

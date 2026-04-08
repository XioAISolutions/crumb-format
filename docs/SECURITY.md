# Security and Safety

CRUMB is designed for portable context, which means it is easy to copy, paste, share, commit, and attach.

That portability is useful, but it also means safety checks matter.

## `crumb lint`

Use `crumb lint` before sharing or committing sensitive CRUMBs:

```bash
crumb lint handoff.crumb --secrets --strict
```

Supported checks:

- likely secrets and credentials
- suspicious or malformed headers
- large raw logs
- token-budget overruns from `max_total_tokens` and `max_index_tokens`

## Exit codes

- `0` — no blocking issues
- `1` — security findings or warnings under `--strict`
- `2` — parse or file-level failure

## Secret detection

Initial secret linting covers likely:

- OpenAI keys
- GitHub tokens
- AWS access keys
- Slack tokens
- Bearer tokens
- JWTs
- generic `api_key=...`, `token=...`, `password=...`, `secret=...` assignments

These are heuristic warnings, not perfect proofs. Treat them as a safety net, not a substitute for review.

## Redaction

To redact obvious credentials:

```bash
crumb lint handoff.crumb --secrets --redact
```

Behavior:

- by default, redaction writes back to the source file
- if `--output` is provided, CRUMB writes the redacted result there instead

## Sharing guidance

Before pasting a CRUMB into a hosted AI or public issue:

1. Run `crumb lint --secrets`.
2. Remove or redact raw credentials.
3. Avoid dumping giant `[entries]`, `[logs]`, or `[raw_sessions]` sections unless they are truly needed.
4. Prefer packed task/mem/map artifacts over raw transcripts.

## Local-first safety

Core CRUMB functionality does not require remote APIs.

- validation is local
- linting is local
- pack building is deterministic and local
- bridge export can run from saved text
- optional compression can use Ollama locally

That local-first bias is deliberate.

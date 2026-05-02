# Integrating with HALO

[HALO](https://github.com/context-labs/halo) is a self-improvement loop for AI agent harnesses. It collects OpenTelemetry-style execution traces, has a reasoning model surface systemic failure patterns, and produces reports a coding agent can act on.

CRUMB is the wire format for those reports — paste-able, structured, validator-checked. v0.9+ ships two CLI subcommands that turn a HALO `traces.jsonl` into a `kind=log` CRUMB you can hand off to any AI:

- `crumb from-halo <traces.jsonl>` — HALO-flavored defaults (source label `halo`, "HALO trace from..." title).
- `crumb from-otel <traces.jsonl>` — generic OpenTelemetry input with the same parser; use this for any OTEL-emitting agent (HALO, OpenAI Agents SDK, custom harnesses).

## Pipeline

```
                 traces.jsonl
                      │
                      ▼
           ┌──────────────────────┐
           │  HALO RLM analysis   │   (HALO's job — not CRUMB's)
           │  (failure modes,     │
           │   patterns, fixes)   │
           └──────────────────────┘
                      │
                      ▼
                 report + traces
                      │
                      ▼
           ┌──────────────────────┐
           │  crumb from-halo     │   (this repo)
           │  → kind=log .crumb   │
           └──────────────────────┘
                      │
                      ▼
              paste into Claude / Cursor / ChatGPT / jcode / ...
                      │
                      ▼
                  fix lands
```

## Worked example

The repo ships a synthetic OTEL fixture and the resulting crumb:

```bash
crumb from-halo tests/fixtures/halo-traces.jsonl
```

Output (this exact file is checked in at [`examples/halo-trace-to-log.crumb`](../../examples/halo-trace-to-log.crumb)):

```text
BEGIN CRUMB
v=1.3
kind=log
title=HALO trace from halo-traces.jsonl
source=halo
trace_id=trace-abc123
started_at=2023-11-14T22:13:20+00:00
ended_at=2023-11-14T22:13:30+00:00
total_duration_ms=10100
span_count=5
error_count=2
---
[entries]
- agent.session.start  duration=500ms  model=claude-3-5-sonnet  agent_name=reviewer-v2
- tool.call  duration=1000ms  tool_name=grep
- tool.call  :: error  duration=1500ms  tool_name=hallucinated_function  note='tool not found: hallucinated_function'
- agent.refusal  :: error  duration=1000ms  note='model refused'
- agent.session.end  duration=100ms  agent_name=reviewer-v2
END CRUMB
```

Five spans collapse into one paste-able CRUMB. Failure-status spans are surfaced inline (`:: error`). Common agent attributes (`model`, `tool.name`, `agent_name`) ride along; everything else is dropped to keep the bullet list scannable.

## What gets surfaced inline

The bridge is opinionated about which OTEL attributes are worth surfacing on each bullet. Currently:

- `model`, `model_name` — which LLM ran
- `tool.name`, `tool_name` — which tool was called
- `agent_name` — which agent executed the span
- Status code (only when not `OK`) and a truncated status message

The full attribute dict is lossy by design — this is a CRUMB, not a database export. If you need everything, keep the original JSONL.

## Adding canonical failure-mode names

After running `crumb from-halo`, you'll often want to annotate the resulting log crumb with the failure modes HALO's analysis surfaced. The v1.4 draft [`agent-failure-modes.md`](../v1.4/agent-failure-modes.md) defines a canonical vocabulary so cross-tool consumers can act on them. Quick example to append after `[entries]`:

```text
[checks]
- hallucinated_tool_call    :: detected   count=1   tool=hallucinated_function
- tool_error_unhandled      :: detected   count=1
- refusal_loop              :: detected   count=1   reason=policy_filter
```

Validators today ignore these names (free-form). v1.4 normalizes the vocabulary so receivers can rely on it.

## Why not a `kind=trace` primitive?

Asked and answered in the v1.4 scoping doc's "minimal sufficient basis" filter. `kind=log` already exists; it's a sequence of timestamped entries; that's exactly what an OTEL trace is. Inventing `kind=trace` would grow the wire-format basis without adding expressive power. Rejected.

## What HALO keeps

The bridge does **not** try to reimplement HALO's RLM analysis. CRUMB is a wire format, not an orchestration runtime — that's the [SPEC §1 standalone posture](../../SPEC.md). Use HALO for what HALO does well (failure-pattern analysis, report generation), use CRUMB for what CRUMB does well (paste-able structured handoff).

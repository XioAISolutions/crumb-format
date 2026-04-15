# Agent Integrity Layer

CRUMB now has a lightweight agent integrity layer for checking an AI decision before it writes memory, edits files, calls tools, or hands work to another model.

This deliberately cannibalizes the useful engineering pattern from experimental CL1/LLM encoder repos without adopting speculative consciousness claims.

## What it does

The layer runs deterministic canaries around an agent decision:

- output presence and usefulness
- physical logic canary, including the car-wash quantization check
- explicit instruction-following check
- simple contradiction detection
- destructive tool-use safety check
- memory write validation and secret detection
- high-risk file edit scoring
- rolling drift monitoring for long-running sessions

## Python API

```python
from cli.agent_integrity import run_agent_integrity_check

result = run_agent_integrity_check(
    input_text="Should I walk or drive to the car wash if I need my car washed?",
    proposed_action="answer user question",
    model_output="Drive, because the car must be at the car wash to be washed.",
)

print(result.to_dict())
```

Stable result shape:

```json
{
  "passed": true,
  "score": 1.0,
  "failures": [],
  "recommendation": "proceed"
}
```

Recommendations are:

- `proceed` — safe enough to continue
- `retry` — rerun or improve the model output
- `ask_user` — human confirmation recommended
- `block` — do not execute the action

## CLI usage

```bash
python -m cli.agent_integrity \
  --input "Should I walk or drive to the car wash if I need my car washed?" \
  --action "answer user" \
  --output "Drive, because the car must be at the car wash." \
  --json
```

## Where to use it

### Before memory writes

Run this before `crumb append`, `crumb dream`, Palace filing, or any memory consolidation pass.

### Before file edits

Run this before Codex, Claude, OpenClaw, or a local desktop agent applies patches.

### Before tool calls

Run this before deleting files, deploying, editing `.env`, changing workflows, touching migrations, or executing shell commands.

### In long-running agents

Use `DriftMonitor` to watch the rolling integrity score:

```python
from cli.agent_integrity import DriftMonitor, run_agent_integrity_check

monitor = DriftMonitor(window=20)
result = run_agent_integrity_check(input_text, proposed_action, model_output)
state = monitor.observe(result)

if state["status"] == "block":
    raise RuntimeError("Agent drift detected; stop execution.")
```

## Design rule

This is not a truth engine. It is a cheap failure filter.

It should catch dumb, dangerous, or contradictory outputs before they become memory, file edits, or tool actions.

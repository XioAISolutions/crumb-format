"""Tests that web/index.html emits spec-valid v=1.2 CRUMB across kinds.

The web generator is pure JS embedded in a single HTML file; it gets
shipped standalone and users open it in a browser. Before 0.4.0 it
emitted TOML-style `[crumb]` / `v = 1.1` output that never validated.
This test spawns Node with a minimal DOM shim, feeds sample inputs,
and runs the reference Python validator on every kind.

Skipped if `node` is not on PATH so CI environments without Node still
pass.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "scripts" / "run_web_generator.js"
sys.path.insert(0, str(REPO_ROOT / "validators"))
import validate  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)


def _split_kinds(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current_kind: str | None = None
    buf: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("===KIND="):
            if current_kind is not None:
                out[current_kind] = "\n".join(buf)
            current_kind = line.strip("= ").split("=")[1]
            buf = []
        else:
            buf.append(line)
    if current_kind is not None:
        out[current_kind] = "\n".join(buf)
    return out


def _run(sample: str) -> dict[str, str]:
    result = subprocess.run(
        ["node", str(HARNESS)],
        input=sample,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"harness failed: {result.stderr}\nstdout:{result.stdout}"
    )
    return _split_kinds(result.stdout)


SHORT_SAMPLE = "hello world, debug login bug"

NORMAL_SAMPLE = """\
User: I am building a checkout flow. We decided to use Stripe over PayPal.
Must not break existing cart flow. TODO: wire up webhook at api/stripe_webhook.py
Assistant: Here is a handler
```python
def handle(req): ...
```
User: Also need to add tests at tests/test_checkout.py
"""

LONG_SAMPLE = ("User: let's debug the auth bug. " * 200).strip() + "\nTODO: ship the fix."


@pytest.mark.parametrize("sample", [SHORT_SAMPLE, NORMAL_SAMPLE, LONG_SAMPLE])
@pytest.mark.parametrize("kind", ["task", "mem", "map"])
def test_web_generator_output_validates(sample: str, kind: str) -> None:
    outputs = _run(sample)
    assert kind in outputs, f"kind={kind} missing from harness output"
    validate.parse_crumb(outputs[kind])  # raises ValidationError if invalid


def test_task_includes_handoff_primitive() -> None:
    outputs = _run(NORMAL_SAMPLE)
    parsed = validate.parse_crumb(outputs["task"])
    assert "handoff" in parsed["sections"], (
        "[handoff] primitive must be present on task kind — it's the core v1.2 value add"
    )
    assert any(line.strip() for line in parsed["sections"]["handoff"])


def test_long_input_produces_fold_pair() -> None:
    outputs = _run(LONG_SAMPLE)
    parsed = validate.parse_crumb(outputs["task"])
    sections = parsed["sections"]
    assert "fold:context/summary" in sections
    assert "fold:context/full" in sections
    assert "context" not in sections, "fold pair and plain [context] must not coexist"


def test_headers_are_spec_shape() -> None:
    outputs = _run(NORMAL_SAMPLE)
    body = outputs["task"].splitlines()
    assert body[0] == "BEGIN CRUMB"
    assert body[-1] == "END CRUMB"
    header_lines = body[1 : body.index("---")]
    keys = {line.split("=", 1)[0] for line in header_lines if "=" in line}
    assert {"v", "kind", "source"}.issubset(keys)
    v_line = next(line for line in header_lines if line.startswith("v="))
    assert v_line == "v=1.2", "web generator should emit v=1.2"

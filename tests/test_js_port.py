"""Test that the in-browser JS port matches the Python implementation.

- Regenerates the data JSON in memory and asserts it matches the file on disk
  (so dictionary drift between Python and JS gets caught).
- Runs the JS port via Node over a set of fixtures and compares each level's
  output byte-for-byte with the Python encoder.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import metalk  # noqa: E402
from scripts.export_metalk_data import build as build_data  # noqa: E402


DATA_PATH = ROOT / "web" / "metalk-data.json"
JS_PATH = ROOT / "web" / "metalk.js"
NODE = shutil.which("node")


def test_data_json_matches_python_source():
    """Drift guard: if the Python dicts change, the JSON must be regenerated."""
    fresh = build_data()
    on_disk = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    # Normalize both through json.dumps(sort_keys=True) for stable comparison.
    assert json.dumps(fresh, sort_keys=True) == json.dumps(on_disk, sort_keys=True), (
        "web/metalk-data.json is stale. Regenerate with: "
        "python scripts/export_metalk_data.py"
    )


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_js_port_matches_python(level, tmp_path):
    """For every example crumb, JS encode(level) must equal Python encode(level)."""
    fixtures = sorted((ROOT / "examples").glob("*.crumb"))
    assert fixtures, "no example crumbs found"

    # Build a tiny Node driver that loads metalk.js and encodes each fixture.
    driver = tmp_path / "driver.js"
    driver.write_text(f"""
const fs = require("fs");
const path = require("path");
// Load metalk.js as CommonJS
global.self = global;
global.fetch = async (url) => ({{
  ok: true,
  json: async () => JSON.parse(fs.readFileSync("{DATA_PATH}", "utf-8")),
  status: 200
}});
const Metalk = require("{JS_PATH}");
(async () => {{
  await Metalk.load();
  const inputs = JSON.parse(fs.readFileSync(process.argv[2], "utf-8"));
  const out = {{}};
  for (const name in inputs) {{
    out[name] = Metalk.encode(inputs[name], {level}, {{ vowel_min_length: 4 }});
  }}
  process.stdout.write(JSON.stringify(out));
}})();
""", encoding="utf-8")

    inputs = {f.name: f.read_text(encoding="utf-8") for f in fixtures}
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")

    result = subprocess.run(
        [NODE, str(driver), str(inputs_path)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "NODE_PATH": "/opt/node22/lib/node_modules"},
    )
    assert result.returncode == 0, f"node driver failed: {result.stderr}"
    js_outputs = json.loads(result.stdout)

    mismatches = []
    for name, original in inputs.items():
        py_encoded = metalk.encode(original, level=level)
        js_encoded = js_outputs[name]
        if py_encoded != js_encoded:
            mismatches.append((name, py_encoded, js_encoded))

    if mismatches:
        msg = [f"{len(mismatches)}/{len(inputs)} fixtures differ at L{level}:"]
        for name, py, js in mismatches[:3]:
            msg.append(f"\n--- {name} ---")
            msg.append(f"PY: {py[:200]!r}")
            msg.append(f"JS: {js[:200]!r}")
        pytest.fail("\n".join(msg))

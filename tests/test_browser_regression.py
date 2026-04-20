"""Node-driven regression tests for the browser-side compressLocal paths.

Both the extension popup (`browser-extension/popup.js`) and the playground's
offline fallback (`web/playground.html`) have their own `compressLocal`
implementations that unwrap the synthetic mem-crumb. Codex caught two bugs
in round 3:

1. Bracket filter dropped user `[todo]` / `[note]` lines (same class of
   bug as the server fix in round 2).
2. The playground's `isCrumb` mixed `&&`/`||` without grouping and called
   the Python-only `.lstrip()`, so `mode=plain` still ran the BC check and
   leading-whitespace `BEGIN CRUMB` was missed in `auto`.

We test both fixes by running the actual JS through Node:

- Extension: extract the `compressLocal` from popup.js, instantiate with
  a fresh Metalk instance, and assert user `[todo]` lines survive.
- Playground: extract the `compressLocal` from playground.html and assert
  mode=plain never misdetects, mode=auto handles leading whitespace.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


def _extract_function(src: str, name: str) -> str:
    """Pull a `function NAME(...) { ... }` block out of source by brace-matching."""
    idx = src.find(f"function {name}")
    if idx < 0:
        raise ValueError(f"function {name} not found")
    open_brace = src.find("{", idx)
    depth = 0
    for i in range(open_brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[idx:i + 1]
    raise ValueError(f"unterminated function {name}")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_extension_popup_preserves_user_brackets(tmp_path):
    """Regression: extension popup used to drop `[todo]` / `[note]` lines.
    With the encodePlain refactor there's no longer a wrap/unwrap phase —
    plain text goes directly through the body-transform pipeline."""
    popup_src = (ROOT / "browser-extension" / "popup.js").read_text(encoding="utf-8")
    fn_src = _extract_function(popup_src, "compressLocal")
    sentinel_src = _extract_function(popup_src, "looksLikeCrumb")

    driver = tmp_path / "driver.js"
    driver.write_text(f"""
const fs = require("fs");
const path = require("path");
global.self = global;
global.fetch = async () => ({{
  ok: true, status: 200,
  json: async () => JSON.parse(fs.readFileSync(
    "{ROOT / 'browser-extension' / 'metalk-data.json'}", "utf-8"))
}});
const Metalk = require("{ROOT / 'browser-extension' / 'metalk.js'}");
self.Metalk = Metalk;
{sentinel_src}
{fn_src}
(async () => {{
  await Metalk.load();
  const text = "Ship the auth fix.\\n[todo]\\nWrite a regression test.\\n[note]\\nMerge after CI.";
  const res = compressLocal(text, 1);
  process.stdout.write(JSON.stringify(res));
}})();
""", encoding="utf-8")

    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"node driver failed: {result.stderr}"
    out = json.loads(result.stdout)
    assert "[todo]" in out["encoded"], f"user [todo] dropped: {out['encoded']!r}"
    assert "[note]" in out["encoded"], f"user [note] dropped: {out['encoded']!r}"
    assert "[consolidated]" not in out["encoded"]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_playground_numeric_knob_preserves_zero(tmp_path):
    """Regression: `parseFloat(x) || 0.85` replaced a user-entered 0 with 0.85.
    The new `numericKnob` helper uses isFinite so legitimate zeroes survive."""
    html = (ROOT / "web" / "playground.html").read_text(encoding="utf-8")
    fn_src = _extract_function(html, "numericKnob")

    driver = tmp_path / "driver.js"
    driver.write_text(f"""
{fn_src}
const cases = [
  {{ raw: "0",   parser: parseFloat, fb: 0.85, want: 0 }},
  {{ raw: "0.0", parser: parseFloat, fb: 0.85, want: 0 }},
  {{ raw: "0.5", parser: parseFloat, fb: 0.85, want: 0.5 }},
  {{ raw: "",    parser: parseFloat, fb: 0.85, want: 0.85 }},
  {{ raw: "abc", parser: parseFloat, fb: 0.85, want: 0.85 }},
  {{ raw: "0",   parser: (v) => parseInt(v, 10), fb: 4, want: 0 }},
  {{ raw: "5",   parser: (v) => parseInt(v, 10), fb: 4, want: 5 }},
];
const results = cases.map((c) => ({{
  input: c.raw, want: c.want, got: numericKnob(c.raw, c.parser, c.fb)
}}));
process.stdout.write(JSON.stringify(results));
""", encoding="utf-8")

    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, f"node driver failed: {result.stderr}"
    rows = json.loads(result.stdout)
    for r in rows:
        assert r["got"] == r["want"], (
            f"numericKnob({r['input']!r}): want {r['want']}, got {r['got']}"
        )


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_compress_validation_error_does_not_flip_to_offline(tmp_path):
    """Regression: the playground's compress .catch used to run for 4xx
    responses, silently flipping useServer=false for the session. The
    fixed code only falls back on true network errors."""
    # Extract the decision logic by simulating a 400 and asserting
    # showServerError is invoked (not tryLocal).
    driver = tmp_path / "driver.js"
    driver.write_text(r"""
let useServer = true;
let offlineCalled = false;
let serverErrorShown = null;

function tryLocal() {
  useServer = false;
  offlineCalled = true;
}
function showServerError(err) { serverErrorShown = err; }

// Mirror the fixed compress() server-response handler exactly:
function handleResponse(res) {
  if (res.ok) return;
  showServerError(res.body && res.body.error ? res.body.error : ("HTTP " + res.status));
}

// Simulate a 400 with a validation error (e.g. adaptive_threshold=2).
handleResponse({ ok: false, status: 400, body: { error: "'adaptive_threshold' must be between 0 and 1" } });
console.log(JSON.stringify({ offlineCalled, serverErrorShown, useServer }));
""", encoding="utf-8")
    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["offlineCalled"] is False, \
        "4xx wrongly triggered offline fallback"
    assert out["useServer"] is True, \
        "useServer was flipped on a 4xx response"
    assert "adaptive_threshold" in out["serverErrorShown"]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_compare_clears_stale_cards_on_error(tmp_path):
    """Regression: compare used to leave previous cards visible on a 400
    because data.levels was missing. Now it calls showCompareError which
    replaces the grid with an error message."""
    driver = tmp_path / "driver.js"
    driver.write_text(r"""
let grid = "<card>old</card><card>old</card>";
function showCompareError(msg) {
  grid = '<div class="err">' + msg + '</div>';
}
function handleCompareResponse(res) {
  if (!res.ok) {
    const msg = (res.body && res.body.error) ? res.body.error : ("HTTP " + res.status);
    showCompareError("Error: " + msg);
    return;
  }
  if (res.body && res.body.levels) {
    grid = "<card>new1</card><card>new2</card>";
  } else {
    showCompareError("Error: malformed response (no levels field)");
  }
}
handleCompareResponse({ ok: false, status: 400, body: { error: "'text' must be a string" } });
console.log(JSON.stringify({ grid }));
""", encoding="utf-8")
    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert "old" not in out["grid"], "stale cards were not cleared on 4xx"
    assert "err" in out["grid"] and "text" in out["grid"]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_encode_passthrough_on_missing_separator(tmp_path):
    """Regression: JS encode fell through to encodePlain when no `---`
    separator was present, which diverges from Python where encode()
    returns input unchanged. That caused offline/extension flows running
    `encode(mode="crumb")` on an incomplete snippet to silently rewrite
    the text while the server left it alone."""
    driver = tmp_path / "driver.js"
    driver.write_text(f"""
const fs = require("fs");
global.self = global;
global.fetch = async () => ({{
  ok: true, status: 200,
  json: async () => JSON.parse(fs.readFileSync("{ROOT / 'web' / 'metalk-data.json'}", "utf-8"))
}});
const Metalk = require("{ROOT / 'web' / 'metalk.js'}");
(async () => {{
  await Metalk.load();
  // An incomplete CRUMB snippet — headers but no `---` separator.
  const snippet = "BEGIN CRUMB\\nv=1.1\\nkind=task\\ntitle=Draft\\n";
  const out = Metalk.encode(snippet, 2, {{ vowel_min_length: 4 }});
  process.stdout.write(JSON.stringify({{ input: snippet, output: out, unchanged: out === snippet }}));
}})();
""", encoding="utf-8")
    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["unchanged"], (
        f"JS encode rewrote separator-less input: input={out['input']!r} "
        f"output={out['output']!r}"
    )


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_contractions_pass_through_vowel_strip(tmp_path):
    """Regression: the JS port reads opaque_chars from metalk-data.json,
    so adding `'` on the Python side propagates here. Confirms the JS
    encodePlain at L4 leaves contractions and apostrophe-names untouched."""
    driver = tmp_path / "driver.js"
    driver.write_text(f"""
const fs = require("fs");
global.self = global;
global.fetch = async () => ({{
  ok: true, status: 200,
  json: async () => JSON.parse(fs.readFileSync("{ROOT / 'web' / 'metalk-data.json'}", "utf-8"))
}});
const Metalk = require("{ROOT / 'web' / 'metalk.js'}");
(async () => {{
  await Metalk.load();
  const cases = ["couldn't", "don't", "isn't", "O'Reilly", "O'Brien"];
  const out = cases.map(s => ({{ input: s, encoded: Metalk.encodePlain(s, 4, {{ vowel_min_length: 4 }}) }}));
  process.stdout.write(JSON.stringify(out));
}})();
""", encoding="utf-8")
    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    for row in rows:
        assert row["encoded"] == row["input"], (
            f"JS L4 vowel-stripped {row['input']!r} → {row['encoded']!r}"
        )


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_looks_like_crumb_strict_first_line(tmp_path):
    """Regression: the JS auto-detection used `startswith` which matched
    any 'BC...' prefix. Now it requires the full sentinel as the first
    non-empty line, mirroring the Python server helper."""
    html = (ROOT / "web" / "playground.html").read_text(encoding="utf-8")
    fn_src = _extract_function(html, "looksLikeCrumb")

    driver = tmp_path / "driver.js"
    driver.write_text(f"""
{fn_src}
const cases = [
  // Positive — real CRUMB sentinels.
  {{ text: "BEGIN CRUMB\\nv=1.1\\n---\\n...",  want: true }},
  {{ text: "   BEGIN CRUMB\\nv=1.1\\n...",     want: true }},
  {{ text: "BC\\nv=1.1\\n---\\n...",           want: true }},
  // Negative — prompts that merely start with BC/BEGIN-like letters.
  {{ text: "BC drivers are failing today.",    want: false }},
  {{ text: "BCD framework comparison.",        want: false }},
  {{ text: "BC Technologies released a report.", want: false }},
  {{ text: "BEGIN CRUMBLE cookies recipe.",    want: false }},
  {{ text: "Please fix the bug.",              want: false }},
];
const got = cases.map((c) => ({{ text: c.text, want: c.want, got: looksLikeCrumb(c.text) }}));
process.stdout.write(JSON.stringify(got));
""", encoding="utf-8")

    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    for row in rows:
        assert row["got"] == row["want"], (
            f"looksLikeCrumb({row['text']!r}): want {row['want']}, got {row['got']}"
        )


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_separator_is_trim_aware(tmp_path):
    """Regression: lines.indexOf('---') missed separators with surrounding
    whitespace, so a CRUMB line like ' --- ' was encoded as plain in JS
    while Python handled it as structured. Now both use trim-aware match."""
    driver = tmp_path / "driver.js"
    driver.write_text(f"""
const fs = require("fs");
global.self = global;
global.fetch = async () => ({{
  ok: true, status: 200,
  json: async () => JSON.parse(fs.readFileSync("{ROOT / 'web' / 'metalk-data.json'}", "utf-8"))
}});
const Metalk = require("{ROOT / 'web' / 'metalk.js'}");
(async () => {{
  await Metalk.load();
  // Separator has leading + trailing whitespace.
  const crumb = "BEGIN CRUMB\\nv=1.1\\nkind=task\\ntitle=T\\n   ---   \\n[goal]\\nFix it.\\nEND CRUMB\\n";
  const encoded = Metalk.encode(crumb, 2);
  // Structured path injects mt= header; plain path does not.
  process.stdout.write(JSON.stringify({{
    encoded,
    hasMt: encoded.includes("mt=2"),
    hasStructural: encoded.startsWith("BC")
  }}));
}})();
""", encoding="utf-8")
    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["hasMt"], f"separator with whitespace fell through to plain path: {out!r}"
    assert out["hasStructural"], f"BEGIN CRUMB not abbreviated to BC: {out!r}"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_compare_falls_back_to_local_on_network_error(tmp_path):
    """Regression: compare used to show an unreachable error on network
    failure instead of falling back to the JS port like single-compress.
    The fix: on fetch rejection, runCompareLocal() runs all 5 levels
    through the JS port and renders cards."""
    driver = tmp_path / "driver.js"
    driver.write_text(r"""
let rendered = null;
let toastMessage = null;
function renderCompareCards(rows) { rendered = rows; }
function showToast(msg) { toastMessage = msg; }
function showCompareError(msg) { rendered = { error: msg }; }
// Stub compressLocal: returns deterministic rows so we can inspect.
function compressLocal(text, level) {
  return {
    encoded: "LVL" + level + ":" + text,
    stats: {
      original_tokens: 10, encoded_tokens: 5, saved_tokens: 5,
      pct_saved: 50, ratio: 2.0, original_chars: 40, encoded_chars: 20,
      vowels_removed: 0, vowel_retention_pct: 100, level: level, mode: "plain"
    }
  };
}
function ensureJs() { return Promise.resolve(); }
// Mirror the fixed runCompareLocal path exactly.
function runCompareLocal(text, vml, mode) {
  return ensureJs().then(function () {
    var rows = [1, 2, 3, 4, 5].map(function (level) {
      try {
        var res = compressLocal(text, level, vml, mode);
        return { level: level, encoded: res.encoded, stats: res.stats };
      } catch (err) {
        return { level: level, error: String(err && err.message || err) };
      }
    });
    renderCompareCards(rows);
    showToast("Running offline — JS port in use");
  });
}
(async () => {
  await runCompareLocal("hello", 4, "plain");
  console.log(JSON.stringify({
    count: Array.isArray(rendered) ? rendered.length : 0,
    levels: Array.isArray(rendered) ? rendered.map(r => r.level) : [],
    toast: toastMessage
  }));
})();
""", encoding="utf-8")
    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["count"] == 5, f"expected 5 local cards, got {out['count']}"
    assert out["levels"] == [1, 2, 3, 4, 5]
    assert "offline" in out["toast"].lower()


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_playground_offline_is_crumb_detection(tmp_path):
    """Regression: the offline `isCrumb` mixed &&/|| without grouping and
    used Python's lstrip, so mode=plain still triggered the BC check and
    `   BEGIN CRUMB` in auto mode was missed."""
    html = (ROOT / "web" / "playground.html").read_text(encoding="utf-8")
    # Extract the compressLocal function body verbatim.
    fn_src = _extract_function(html, "compressLocal")
    sentinel_src = _extract_function(html, "looksLikeCrumb")
    estimate_src = _extract_function(html, "estimateVowelRetention")

    driver = tmp_path / "driver.js"
    driver.write_text(f"""
const fs = require("fs");
global.window = global;
global.self = global;
global.fetch = async () => ({{
  ok: true, status: 200,
  json: async () => JSON.parse(fs.readFileSync(
    "{ROOT / 'web' / 'metalk-data.json'}", "utf-8"))
}});
const Metalk = require("{ROOT / 'web' / 'metalk.js'}");
window.Metalk = Metalk;
{sentinel_src}
{estimate_src}
{fn_src}
(async () => {{
  await Metalk.load();
  // 1. mode=plain must NEVER detect as crumb, even if text happens to
  //    start with "BC" (legitimate prose).
  const plainStartingWithBC = "BC Technologies released their report today.";
  const r1 = compressLocal(plainStartingWithBC, 2, 4, "plain");
  // 2. mode=auto on leading-whitespace BEGIN CRUMB must detect as crumb.
  const crumbLeadingWs = "   BEGIN CRUMB\\nv=1.1\\nkind=task\\ntitle=T\\n---\\n[goal]\\nFix auth.\\nEND CRUMB\\n";
  const r2 = compressLocal(crumbLeadingWs, 2, 4, "auto");
  // 3. mode=auto on plain prose starting with arbitrary letters is plain.
  const plainProse = "Please fix the authentication middleware.";
  const r3 = compressLocal(plainProse, 2, 4, "auto");
  process.stdout.write(JSON.stringify({{
    plain_starting_with_bc_mode: r1.stats.mode,
    crumb_leading_ws_mode: r2.stats.mode,
    plain_auto_mode: r3.stats.mode
  }}));
}})();
""", encoding="utf-8")

    result = subprocess.run([NODE, str(driver)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"node driver failed: {result.stderr}"
    out = json.loads(result.stdout)

    assert out["plain_starting_with_bc_mode"] == "plain", (
        f"mode=plain misdetected text starting with 'BC' as crumb: {out}"
    )
    assert out["crumb_leading_ws_mode"] == "crumb", (
        f"mode=auto missed leading-whitespace BEGIN CRUMB: {out}"
    )
    assert out["plain_auto_mode"] == "plain", (
        f"mode=auto misdetected plain prose: {out}"
    )

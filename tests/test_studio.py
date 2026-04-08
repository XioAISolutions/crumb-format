import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "cli") not in sys.path:
    sys.path.insert(0, str(ROOT / "cli"))

import crumb
from studio.app import main as studio_main
from studio.engine import DEFAULT_SOURCE, SUPPORTED_MODES, build_studio_output
from studio.history import HistoryStore


SAMPLE_INPUT = """\
user: Need to fix the auth middleware before launch.
assistant: Preserve the cookie names and add regression coverage.
assistant: Update middleware.ts and tests/auth.spec.ts before shipping.
```ts
middleware.ts
```
"""


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_build_studio_output_generates_parseable_crumb(mode):
    result = build_studio_output(SAMPLE_INPUT, mode=mode, title=f"{mode} output", source="studio.test")
    parsed = crumb.parse_crumb(result.output_text)

    assert parsed["headers"]["kind"] == mode
    assert parsed["headers"]["title"] == f"{mode} output"
    assert parsed["headers"]["source"] == "studio.test"
    assert result.stats.input_chars > 0
    assert result.stats.output_chars > 0
    assert result.stats.output_lines > 0
    assert result.output_markdown
    assert result.output_json


def test_build_studio_output_uses_default_source():
    result = build_studio_output(SAMPLE_INPUT, mode="task")
    parsed = crumb.parse_crumb(result.output_text)

    assert parsed["headers"]["source"] == DEFAULT_SOURCE


def test_build_studio_output_rejects_empty_input():
    with pytest.raises(ValueError, match="Paste some raw context"):
        build_studio_output("   \n", mode="task")


def test_history_store_round_trip(tmp_path):
    store = HistoryStore(path=tmp_path / "history.json", limit=5)
    item = build_studio_output(SAMPLE_INPUT, mode="mem").to_history_item()

    items = store.add(item)
    assert items[0]["id"] == item["id"]
    assert store.get(item["id"])["outputText"] == item["outputText"]

    store.clear()
    assert store.list_items() == []


def test_studio_smoke_test(capsys):
    studio_main(["--smoke-test"])
    captured = capsys.readouterr()

    assert '"input_chars"' in captured.out
    assert "BEGIN CRUMB" in captured.out


def test_cli_studio_smoke_test(capsys):
    crumb.main(["studio", "--smoke-test"])
    captured = capsys.readouterr()

    assert '"output_tokens"' in captured.out
    assert "END CRUMB" in captured.out

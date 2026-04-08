from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

from cli import crumb as crumb_engine
from studio.engine import DEFAULT_SOURCE, SUPPORTED_MODES, build_studio_output
from studio.history import HistoryStore


def _app_version() -> str:
    try:
        return importlib.metadata.version("crumb-format")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _asset_path(*parts: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / "studio" / "static" / Path(*parts)
    return Path(__file__).resolve().parent / "static" / Path(*parts)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "crumb-studio-output"


class StudioApi:
    def __init__(self) -> None:
        self.history = HistoryStore()
        self.window: Any | None = None
        self.webview_module: Any | None = None

    def bind_window(self, window: Any, webview_module: Any) -> None:
        self.window = window
        self.webview_module = webview_module

    def bootstrap(self) -> dict[str, Any]:
        return {
            "ok": True,
            "appVersion": _app_version(),
            "defaultSource": DEFAULT_SOURCE,
            "modes": list(SUPPORTED_MODES),
            "history": self.history.list_items(),
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = build_studio_output(
                raw_text=str(payload.get("inputText", "")),
                mode=str(payload.get("mode", "task")),
                title=str(payload.get("title", "")).strip() or None,
                source=str(payload.get("source", "")).strip() or DEFAULT_SOURCE,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        history_items = self.history.add(result.to_history_item())
        return {"ok": True, "result": result.to_payload(), "history": history_items}

    def copy_output(self, output_text: str) -> dict[str, Any]:
        text = (output_text or "").strip()
        if not text:
            return {"ok": False, "error": "There is no output to copy yet."}
        if crumb_engine._copy_to_clipboard(text):
            return {"ok": True, "message": "Structured output copied to the clipboard."}
        return {"ok": False, "error": "Clipboard access is not available on this machine."}

    def save_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.window is None or self.webview_module is None:
            return {"ok": False, "error": "Desktop file save is only available inside the Studio app window."}

        output_text = str(payload.get("outputText", "")).strip()
        if not output_text:
            return {"ok": False, "error": "Generate output before saving a .crumb file."}

        title = str(payload.get("title", "")).strip() or "crumb-studio-output"
        suggested_name = f"{_slugify(title)}.crumb"
        path = self.window.create_file_dialog(
            self.webview_module.SAVE_DIALOG,
            save_filename=suggested_name,
            file_types=("CRUMB files (*.crumb)", "Text files (*.txt)"),
        )
        if not path:
            return {"ok": False, "cancelled": True}
        if isinstance(path, (list, tuple)):
            path = path[0]

        target = Path(path)
        target.write_text(output_text, encoding="utf-8")
        return {"ok": True, "path": str(target)}

    def export_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.window is None or self.webview_module is None:
            return {"ok": False, "error": "Desktop export is only available inside the Studio app window."}

        output_text = str(payload.get("outputText", "")).strip()
        export_format = str(payload.get("format", "markdown")).strip().lower()
        title = str(payload.get("title", "")).strip() or "crumb-studio-output"
        if not output_text:
            return {"ok": False, "error": "Generate output before exporting it."}

        parsed = crumb_engine.parse_crumb(output_text)
        if export_format == "markdown":
            content = crumb_engine.crumb_to_markdown(parsed)
            suffix = ".md"
            file_types = ("Markdown files (*.md)", "Text files (*.txt)")
        elif export_format == "json":
            content = crumb_engine.crumb_to_json(parsed)
            suffix = ".json"
            file_types = ("JSON files (*.json)", "Text files (*.txt)")
        else:
            content = output_text
            suffix = ".txt"
            file_types = ("Text files (*.txt)", "All files (*.*)")

        suggested_name = f"{_slugify(title)}{suffix}"
        path = self.window.create_file_dialog(
            self.webview_module.SAVE_DIALOG,
            save_filename=suggested_name,
            file_types=file_types,
        )
        if not path:
            return {"ok": False, "cancelled": True}
        if isinstance(path, (list, tuple)):
            path = path[0]

        target = Path(path)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(target), "format": export_format}

    def get_history(self) -> dict[str, Any]:
        return {"ok": True, "history": self.history.list_items()}

    def load_history_item(self, item_id: str) -> dict[str, Any]:
        item = self.history.get(item_id)
        if not item:
            return {"ok": False, "error": "That history item is no longer available."}
        return {"ok": True, "item": item}

    def clear_history(self) -> dict[str, Any]:
        self.history.clear()
        return {"ok": True, "history": []}


def _run_smoke_test() -> int:
    sample = (
        "user: Need a compact handoff for the auth cleanup.\n"
        "assistant: We should preserve cookie names and add regression coverage.\n"
        "```ts\nmiddleware.ts\n```\n"
    )
    result = build_studio_output(sample, "task", title="Smoke test", source="studio.smoke")
    print(json.dumps(result.to_payload()["stats"], indent=2))
    print(result.output_text)
    return 0


def launch_studio(debug: bool = False) -> int:
    try:
        import webview
    except ImportError:
        print(
            "CRUMB Studio needs the optional desktop dependencies.\n"
            "Install them with: python -m pip install 'crumb-format[studio]'",
            file=sys.stderr,
        )
        return 1

    api = StudioApi()
    window = webview.create_window(
        title="CRUMB Studio",
        url=_asset_path("index.html").as_uri(),
        js_api=api,
        width=1440,
        height=920,
        min_size=(1120, 720),
        background_color="#0b0d12",
    )
    api.bind_window(window, webview)
    webview.start(debug=debug)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the CRUMB Studio desktop app.")
    parser.add_argument("--debug", action="store_true", help="Enable the desktop webview debug mode.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the Studio engine smoke test and exit.")
    args = parser.parse_args(argv)

    if args.smoke_test:
        return _run_smoke_test()
    return launch_studio(debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

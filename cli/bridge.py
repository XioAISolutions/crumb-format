"""Bridge adapters for moving CRUMBs in and out of external memory systems."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from extensions import SPEC_URL, append_extension


def _crumb():
    return importlib.import_module("crumb")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "item"


def _flatten_lines(lines: list[str]) -> list[str]:
    result = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("```"):
            continue
        stripped = re.sub(r"^[*\-•\d.\]\)\s]+", "", stripped).strip()
        if stripped:
            result.append(stripped)
    return result


def _read_bridge_input(path_or_dash: str | None) -> str:
    crumb = _crumb()
    return crumb.read_text(path_or_dash or "-")


def _write_output(path: str, content: str) -> None:
    crumb = _crumb()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crumb.write_text(path, content)


class BridgeAdapter(ABC):
    name = "bridge"

    @abstractmethod
    def export(self, args: argparse.Namespace) -> list[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def import_crumbs(self, args: argparse.Namespace) -> str:
        raise NotImplementedError


class MempalaceAdapter(BridgeAdapter):
    name = "mempalace"

    def _ensure_cli(self) -> str:
        command = shutil.which("mempalace")
        if not command:
            raise RuntimeError(
                "MemPalace CLI is not installed or not on PATH. Install it with `pip install mempalace`, "
                "or pass --input with a saved MemPalace export."
            )
        return command

    def _run_search(self, args: argparse.Namespace) -> str:
        command = [self._ensure_cli(), "search", args.query]
        if args.wing:
            command.append(f"--wing={args.wing}")
        if getattr(args, "hall", None):
            command.append(f"--hall={args.hall}")

        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "MemPalace search failed"
            raise RuntimeError(
                f"MemPalace search failed: {detail}. If your installed MemPalace CLI uses different flags, "
                "run the search manually and pass the saved text with --input."
            )
        return completed.stdout

    def _load_source_text(self, args: argparse.Namespace) -> str:
        if args.input:
            return _read_bridge_input(args.input)
        if not args.query:
            raise RuntimeError("Provide either --query to search MemPalace or --input with a text export.")
        return self._run_search(args)

    def _extract_records(self, text: str, args: argparse.Namespace) -> list[str]:
        lines = _flatten_lines(text.splitlines())
        if args.room:
            lines = [line for line in lines if args.room.lower() in line.lower()]
        if args.entity:
            lines = [line for line in lines if args.entity.lower() in line.lower()] or lines
        return lines

    def _build_export_crumb(self, lines: list[str], args: argparse.Namespace) -> str:
        crumb = _crumb()
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        query_label = args.query or args.entity or args.room or "mempalace export"
        title = args.title or f"MemPalace export: {query_label}"
        headers = {
            "v": "1.1",
            "kind": args.as_kind,
            "title": title,
            "source": "mempalace.bridge",
            "url": SPEC_URL,
            "tags": "bridge, mempalace",
            "id": f"mempalace-{_slugify(query_label)}",
        }
        if args.wing:
            headers["project"] = args.wing
        append_extension(headers, "bridge.mempalace.export.v1")

        if args.as_kind == "task":
            sections = {
                "goal": [f"Continue the work described by the retrieved MemPalace context for: {query_label}."],
                "context": [f"- {line}" for line in lines[:24]] or ["- No matching memories were returned."],
                "constraints": [
                    "- Treat retrieved memories as supporting context, not ground truth.",
                    "- Preserve human readability and keep the resulting handoff portable across tools.",
                ],
            }
        elif args.as_kind == "mem":
            sections = {
                "consolidated": [f"- {line}" for line in lines[:32]] or ["- No durable facts were retrieved."],
            }
        else:
            entries = []
            for line in lines[:32]:
                if re.match(r"^- \[\d{4}-\d{2}-\d{2}T", line):
                    entries.append(line)
                else:
                    entries.append(f"- [{now}] {line}")
            sections = {"entries": entries or [f"- [{now}] No matching memories were returned."]}

        return crumb.render_crumb(headers, sections)

    def export(self, args: argparse.Namespace) -> list[tuple[str, str]]:
        text = self._load_source_text(args)
        lines = self._extract_records(text, args)
        content = self._build_export_crumb(lines, args)

        output_path = Path(args.output)
        if args.output == "-" or (not output_path.exists() and output_path.suffix == ".crumb"):
            filename = args.output
        else:
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = output_dir / f"mempalace-{args.as_kind}-{_slugify(args.query or args.entity or args.room or 'export')}.crumb"
        return [(str(filename), content)]

    def import_crumbs(self, args: argparse.Namespace) -> str:
        crumb = _crumb()
        records = []
        wing = args.wing or "default"
        for filepath in args.files:
            parsed = crumb.parse_crumb(Path(filepath).read_text(encoding="utf-8"))
            headers = parsed["headers"]
            sections = parsed["sections"]
            kind = headers["kind"]
            title = headers.get("title", Path(filepath).stem)
            room = args.room or headers.get("project") or _slugify(title)
            hall = {
                "task": "hall_events",
                "mem": "hall_facts",
                "map": "hall_discoveries",
                "log": "hall_events",
                "todo": "hall_advice",
            }.get(kind, "hall_facts")

            lines = []
            for section_name, section_lines in sections.items():
                for raw in section_lines:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    lines.append({"section": section_name, "text": stripped})

            records.append(
                {
                    "backend": "mempalace",
                    "mode": "adapter-ready",
                    "wing": wing,
                    "room": room,
                    "hall": hall,
                    "entity": args.entity or headers.get("id") or title,
                    "title": title,
                    "kind": kind,
                    "source_file": filepath,
                    "metadata": headers,
                    "lines": lines,
                }
            )

        return json.dumps(
            {
                "adapter": "mempalace",
                "version": 1,
                "supported": {
                    "direct_write": False,
                    "output_format": "adapter-ready JSON bundle",
                },
                "records": records,
            },
            indent=2,
        ) + "\n"


ADAPTERS: dict[str, BridgeAdapter] = {
    "mempalace": MempalaceAdapter(),
}


def get_adapter(name: str) -> BridgeAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(ADAPTERS))
        raise RuntimeError(f"Unknown bridge backend '{name}'. Supported backends: {supported}") from exc


def run_bridge_export(args: argparse.Namespace) -> None:
    adapter = get_adapter(args.backend)
    outputs = adapter.export(args)
    for path, content in outputs:
        if path == "-":
            _crumb().write_text("-", content)
        else:
            _write_output(path, content)
            print(f"Wrote {path}")


def run_bridge_import(args: argparse.Namespace) -> None:
    adapter = get_adapter(args.backend)
    content = adapter.import_crumbs(args)
    output = Path(args.output)
    if args.output == "-":
        _crumb().write_text("-", content)
        return
    if output.exists() and output.is_dir():
        target = output / f"{args.backend}-import.json"
    elif output.suffix.lower() != ".json":
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"{args.backend}-import.json"
    else:
        target = output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"Wrote {target}")

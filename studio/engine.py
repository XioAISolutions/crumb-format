from __future__ import annotations

import datetime
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from cli import crumb as crumb_engine

SUPPORTED_MODES = ("task", "mem", "map", "log", "todo")
DEFAULT_SOURCE = "crumb.studio"

ACTION_HINTS = (
    "fix",
    "ship",
    "add",
    "update",
    "create",
    "review",
    "verify",
    "investigate",
    "document",
    "follow up",
    "need to",
    "must",
    "todo",
    "next",
)
CONSTRAINT_HINTS = (
    "must",
    "should not",
    "do not",
    "don't",
    "without",
    "preserve",
    "keep",
    "avoid",
    "cannot",
    "can't",
    "only",
)
PATH_PATTERN = re.compile(
    r"(?:(?:[\w.-]+/)+[\w.-]+|\b[\w.-]+\.(?:py|js|jsx|ts|tsx|md|json|toml|yaml|yml|rs|go|swift|css|html)\b)"
)


@dataclass
class StudioStats:
    input_chars: int
    output_chars: int
    input_tokens: int
    output_tokens: int
    input_lines: int
    output_lines: int
    char_delta: int
    token_delta: int
    output_ratio: float


@dataclass
class StudioResult:
    id: str
    created_at: str
    mode: str
    title: str
    source: str
    input_text: str
    output_text: str
    output_markdown: str
    output_json: str
    stats: StudioStats

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "mode": self.mode,
            "title": self.title,
            "source": self.source,
            "inputText": self.input_text,
            "outputText": self.output_text,
            "outputMarkdown": self.output_markdown,
            "outputJson": self.output_json,
            "stats": asdict(self.stats),
        }

    def to_history_item(self) -> dict[str, Any]:
        preview_lines = [line for line in self.output_text.splitlines() if line.strip()]
        input_lines = [line for line in self.input_text.splitlines() if line.strip()]
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "mode": self.mode,
            "title": self.title,
            "source": self.source,
            "inputPreview": " ".join(input_lines[:2])[:160],
            "outputPreview": " ".join(preview_lines[:3])[:180],
            "inputText": self.input_text,
            "outputText": self.output_text,
            "stats": asdict(self.stats),
        }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "crumb-studio-output"


def _clean_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(
        r"^(?:user|assistant|system|ai|claude|chatgpt|cursor|codex|copilot|gemini)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _non_empty_lines(raw_text: str) -> list[str]:
    lines = []
    for raw_line in raw_text.splitlines():
        cleaned = _clean_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    return lines


def _dedupe(lines: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        normalized = line.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(line)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _shorten(text: str, limit: int = 78) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 1].rsplit(" ", 1)[0].strip()
    return f"{trimmed or text[: limit - 1]}…"


def _derive_title(raw_text: str, mode: str, explicit_title: str | None) -> str:
    if explicit_title and explicit_title.strip():
        return _shorten(explicit_title.strip(), 72)

    for line in _non_empty_lines(raw_text):
        if line.startswith("```"):
            continue
        if len(line) >= 5:
            prefix = {
                "task": "Task",
                "mem": "Memory",
                "map": "Map",
                "log": "Log",
                "todo": "Todo",
            }[mode]
            return _shorten(f"{prefix}: {line}", 72)

    return {
        "task": "Task handoff",
        "mem": "Memory capture",
        "map": "Project map",
        "log": "Session log",
        "todo": "Action list",
    }[mode]


def _collect_signals(raw_text: str) -> dict[str, Any]:
    user_lines, ai_lines, code_blocks, decisions = crumb_engine.parse_chat_lines(raw_text)
    plain_lines = _non_empty_lines(raw_text)
    cleaned_user = [_clean_line(line) for line in user_lines]
    cleaned_ai = [_clean_line(line) for line in ai_lines]
    decisions = _dedupe([_clean_line(line) for line in decisions], limit=8)
    return {
        "plain_lines": _dedupe(plain_lines),
        "user_lines": _dedupe([line for line in cleaned_user if line]),
        "ai_lines": _dedupe([line for line in cleaned_ai if line]),
        "code_blocks": code_blocks,
        "decisions": decisions,
    }


def _extract_constraints(lines: list[str]) -> list[str]:
    matches = [
        f"- {line}"
        for line in lines
        if any(hint in line.lower() for hint in CONSTRAINT_HINTS)
    ]
    matches = _dedupe(matches, limit=5)
    if matches:
        return matches
    return [
        "- Preserve the important behavior already captured in the source context.",
        "- Keep the output compact enough to paste into another AI tool quickly.",
    ]


def _extract_actions(lines: list[str]) -> list[str]:
    actionable = []
    for line in lines:
        lowered = line.lower()
        if any(hint in lowered for hint in ACTION_HINTS):
            actionable.append(line)
            continue
        if lowered.startswith(("ship ", "fix ", "add ", "update ", "review ", "document ")):
            actionable.append(line)
    return _dedupe(actionable, limit=8)


def _extract_goal(signals: dict[str, Any], mode: str) -> str:
    if signals["decisions"]:
        return _shorten(signals["decisions"][0], 120)

    candidates = signals["user_lines"] or signals["plain_lines"]
    if candidates:
        first = candidates[0]
        if mode == "task":
            return _shorten(first, 120)
        return _shorten(f"Capture and structure: {first}", 120)

    return {
        "task": "Continue and complete the work captured in the source text.",
        "mem": "Capture the durable decisions and preferences from the source text.",
        "map": "Summarize the project structure and key moving parts.",
        "log": "Record the important events from the source text.",
        "todo": "Extract the next concrete tasks from the source text.",
    }[mode]


def _extract_modules(signals: dict[str, Any]) -> list[str]:
    modules: list[str] = []
    for line in signals["plain_lines"]:
        for match in PATH_PATTERN.findall(line):
            modules.append(match)
        if "/" in line and len(line.split()) <= 8:
            modules.append(line)
    if signals["code_blocks"]:
        for block in signals["code_blocks"][:3]:
            preview_lines = [line for line in block["code"].splitlines() if line.strip()]
            if preview_lines:
                modules.append(preview_lines[0][:80])
    if not modules:
        keywords = []
        for line in signals["plain_lines"][:10]:
            keywords.extend(sorted(crumb_engine.extract_keywords(line)))
        modules = [f"topic:{keyword}" for keyword in _dedupe(keywords, limit=8)]
    return [f"- {module}" for module in _dedupe(modules, limit=8)] or ["- core workflow", "- open questions"]


def _extract_log_entries(signals: dict[str, Any]) -> list[str]:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = signals["plain_lines"][:10] or ["No clear events were extracted from the source text."]
    return [f"- [{now}] {entry}" for entry in entries]


def _extract_memory_entries(signals: dict[str, Any]) -> list[str]:
    entries = []
    entries.extend(signals["decisions"])
    entries.extend(
        line
        for line in signals["plain_lines"]
        if any(keyword in line.lower() for keyword in ("prefer", "always", "never", "use ", "keep ", "avoid", "important"))
    )
    entries.extend(signals["user_lines"][:8])
    deduped = _dedupe([f"- {entry}" for entry in entries if entry], limit=10)
    return deduped or ["- No durable memory items were confidently extracted yet."]


def _extract_todo_entries(signals: dict[str, Any]) -> list[str]:
    actions = _extract_actions(signals["plain_lines"] + signals["user_lines"] + signals["ai_lines"])
    if not actions:
        actions = signals["plain_lines"][:6]
    tasks = []
    for action in actions:
        action = action.rstrip(".")
        if not action:
            continue
        tasks.append(f"- [ ] {action}")
    return _dedupe(tasks, limit=8) or ["- [ ] Review the source context and identify the next action."]


def _task_sections(signals: dict[str, Any], goal: str) -> dict[str, list[str]]:
    context_lines: list[str] = []
    if signals["decisions"]:
        context_lines.append("- Decisions already made:")
        context_lines.extend([f"  - {line}" for line in signals["decisions"][:4]])
    if signals["code_blocks"]:
        context_lines.append(f"- Code artifacts mentioned: {len(signals['code_blocks'])}")
        for block in signals["code_blocks"][:3]:
            preview_lines = [line for line in block["code"].splitlines() if line.strip()]
            if preview_lines:
                context_lines.append(f"  - {preview_lines[0][:90]}")
    source_lines = signals["user_lines"] + signals["ai_lines"] + signals["plain_lines"]
    if source_lines:
        context_lines.append("- Source highlights:")
        context_lines.extend([f"  - {line}" for line in _dedupe(source_lines, limit=8)])
    if not context_lines:
        context_lines = ["- Source text provided, but no strong signals were extracted."]
    return {
        "goal": [goal, ""],
        "context": context_lines + [""],
        "constraints": _extract_constraints(signals["plain_lines"]) + [""],
    }


def _mem_sections(signals: dict[str, Any]) -> dict[str, list[str]]:
    return {"consolidated": _extract_memory_entries(signals) + [""]}


def _map_sections(signals: dict[str, Any]) -> dict[str, list[str]]:
    summary_lines = signals["plain_lines"][:2]
    project_summary = " ".join(summary_lines) if summary_lines else _extract_goal(signals, "map")
    return {
        "project": [_shorten(project_summary, 180), ""],
        "modules": _extract_modules(signals) + [""],
    }


def _log_sections(signals: dict[str, Any]) -> dict[str, list[str]]:
    return {"entries": _extract_log_entries(signals) + [""]}


def _todo_sections(signals: dict[str, Any]) -> dict[str, list[str]]:
    return {"tasks": _extract_todo_entries(signals) + [""]}


def _build_stats(raw_text: str, output_text: str) -> StudioStats:
    input_tokens = crumb_engine.estimate_tokens(raw_text)
    output_tokens = crumb_engine.estimate_tokens(output_text)
    return StudioStats(
        input_chars=len(raw_text),
        output_chars=len(output_text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_lines=len([line for line in raw_text.splitlines() if line.strip()]),
        output_lines=len([line for line in output_text.splitlines() if line.strip()]),
        char_delta=len(raw_text) - len(output_text),
        token_delta=input_tokens - output_tokens,
        output_ratio=round((len(output_text) / max(len(raw_text), 1)), 3),
    )


def build_studio_output(
    raw_text: str,
    mode: str,
    title: str | None = None,
    source: str | None = None,
) -> StudioResult:
    mode = (mode or "task").strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'.")

    cleaned_input = raw_text.strip()
    if not cleaned_input:
        raise ValueError("Paste some raw context into the left pane before running CRUMB Studio.")

    signals = _collect_signals(cleaned_input)
    resolved_title = _derive_title(cleaned_input, mode, title)
    resolved_source = (source or DEFAULT_SOURCE).strip() or DEFAULT_SOURCE
    goal = _extract_goal(signals, mode)

    if mode == "task":
        sections = _task_sections(signals, goal)
        headers = {"v": "1.1", "kind": "task", "title": resolved_title, "source": resolved_source}
    elif mode == "mem":
        sections = _mem_sections(signals)
        headers = {"v": "1.1", "kind": "mem", "title": resolved_title, "source": resolved_source}
    elif mode == "map":
        sections = _map_sections(signals)
        headers = {
            "v": "1.1",
            "kind": "map",
            "title": resolved_title,
            "source": resolved_source,
            "project": _slugify(resolved_title),
        }
    elif mode == "log":
        sections = _log_sections(signals)
        headers = {"v": "1.1", "kind": "log", "title": resolved_title, "source": resolved_source}
    else:
        sections = _todo_sections(signals)
        headers = {"v": "1.1", "kind": "todo", "title": resolved_title, "source": resolved_source}

    output_text = crumb_engine.render_crumb(headers, sections)
    parsed = crumb_engine.parse_crumb(output_text)
    created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return StudioResult(
        id=uuid.uuid4().hex,
        created_at=created_at,
        mode=mode,
        title=resolved_title,
        source=resolved_source,
        input_text=cleaned_input,
        output_text=output_text,
        output_markdown=crumb_engine.crumb_to_markdown(parsed),
        output_json=crumb_engine.crumb_to_json(parsed),
        stats=_build_stats(cleaned_input, output_text),
    )

#!/usr/bin/env python3
"""Minimal CLI for creating, validating, and inspecting .crumb handoff files."""

import argparse
import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, List


REQUIRED_HEADERS = ["v", "kind", "source"]
REQUIRED_SECTIONS = {
    "task": ["goal", "context", "constraints"],
    "mem": ["consolidated"],
    "map": ["project", "modules"],
}


def read_text(path: str | None) -> str:
    if path is None or path == '-':
        return sys.stdin.read()
    return Path(path).read_text(encoding='utf-8')


def write_text(path: str | None, content: str) -> None:
    if path is None or path == '-':
        sys.stdout.write(content)
        return
    Path(path).write_text(content, encoding='utf-8')


def parse_crumb(text: str) -> Dict[str, object]:
    """Parse a .crumb file and return headers and sections.

    Raises ValueError on any structural problem.
    """
    lines = [line.rstrip("\n") for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0] != "BEGIN CRUMB":
        raise ValueError("missing BEGIN CRUMB marker")
    if lines[-1] != "END CRUMB":
        raise ValueError("missing END CRUMB marker")
    try:
        sep_index = lines.index("---")
    except ValueError:
        raise ValueError("missing header separator ---")
    headers: Dict[str, str] = {}
    for line in lines[1:sep_index]:
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"invalid header line: {line!r}")
        key, value = line.split("=", 1)
        headers[key.strip()] = value.strip()
    for key in REQUIRED_HEADERS:
        if key not in headers:
            raise ValueError(f"missing required header: {key}")
    if headers["v"] != "1.1":
        raise ValueError(f"unsupported version: {headers['v']}")
    kind = headers["kind"]
    if kind not in REQUIRED_SECTIONS:
        raise ValueError(f"unknown kind: {kind}")
    sections: Dict[str, List[str]] = {}
    current_section: str | None = None
    for line in lines[sep_index + 1 : -1]:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip().lower()
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            if stripped:
                raise ValueError("body content found before first section")
            continue
        sections[current_section].append(line)
    for section in REQUIRED_SECTIONS[kind]:
        if section not in sections:
            raise ValueError(f"missing required section for kind={kind}: [{section}]")
        if not any(item.strip() for item in sections[section]):
            raise ValueError(f"section [{section}] is empty")
    return {"headers": headers, "sections": sections}


# ── new ──────────────────────────────────────────────────────────────

TEMPLATES = {
    "task": dedent("""\
        BEGIN CRUMB
        v=1.1
        kind=task
        title={title}
        source={source}
        ---
        [goal]
        {goal}

        [context]
        {context}

        [constraints]
        {constraints}
        END CRUMB
    """),
    "mem": dedent("""\
        BEGIN CRUMB
        v=1.1
        kind=mem
        title={title}
        source={source}
        ---
        [consolidated]
        {consolidated}
        END CRUMB
    """),
    "map": dedent("""\
        BEGIN CRUMB
        v=1.1
        kind=map
        title={title}
        source={source}
        project={project_name}
        ---
        [project]
        {project_desc}

        [modules]
        {modules}
        END CRUMB
    """),
}

PLACEHOLDERS = {
    "task": {
        "goal": "<what needs to happen next>",
        "context": "<key facts, decisions, and current state>",
        "constraints": "<what must not change>",
    },
    "mem": {
        "consolidated": "<durable facts, preferences, decisions>",
    },
    "map": {
        "project_desc": "<one-line project description>",
        "modules": "<key files and directories>",
    },
}


def cmd_new(args: argparse.Namespace) -> None:
    kind = args.kind
    title = args.title or ""
    source = args.source or ""

    values = {"title": title, "source": source}

    if kind == "task":
        values["goal"] = args.goal or PLACEHOLDERS["task"]["goal"]
        if args.context:
            values["context"] = "\n".join(f"- {c}" for c in args.context)
        else:
            values["context"] = PLACEHOLDERS["task"]["context"]
        if args.constraints:
            values["constraints"] = "\n".join(f"- {c}" for c in args.constraints)
        else:
            values["constraints"] = PLACEHOLDERS["task"]["constraints"]
    elif kind == "mem":
        if args.entries:
            values["consolidated"] = "\n".join(f"- {e}" for e in args.entries)
        else:
            values["consolidated"] = PLACEHOLDERS["mem"]["consolidated"]
    elif kind == "map":
        values["project_name"] = args.project or ""
        values["project_desc"] = args.description or PLACEHOLDERS["map"]["project_desc"]
        if args.modules:
            values["modules"] = "\n".join(f"- {m}" for m in args.modules)
        else:
            values["modules"] = PLACEHOLDERS["map"]["modules"]

    crumb = TEMPLATES[kind].format(**values)
    write_text(args.output, crumb)


# ── from-chat ────────────────────────────────────────────────────────

AI_PREFIXES = ('assistant:', 'ai:', 'claude:', 'gpt:', 'chatgpt:', 'copilot:', 'gemini:', 'system:')


def cmd_from_chat(args: argparse.Namespace) -> None:
    raw = read_text(args.input)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    user_lines = [line for line in lines if not line.lower().startswith(AI_PREFIXES)]
    ai_lines = [line for line in lines if line not in user_lines]

    goal = args.goal.strip() if args.goal else 'Continue this work from where the last assistant left off.'
    title = args.title or 'Continue previous session'
    source = args.source or 'chat.log'

    context_lines = []
    if user_lines:
        context_lines.append('- User conversation (selected lines):')
        context_lines.extend([f'  - {line}' for line in user_lines[:8]])
    if ai_lines:
        context_lines.append('- Previous AI responses (selected lines):')
        context_lines.extend([f'  - {line}' for line in ai_lines[-8:]])
    if not context_lines:
        context_lines = ['- Conversation context omitted.']

    constraints = [f'- {c.strip()}' for c in (args.constraints or []) if c.strip()]
    if not constraints:
        constraints = ['- No additional constraints; follow best practices.']

    crumb = dedent(f'''\
BEGIN CRUMB
v=1.1
kind=task
title={title}
source={source}
---
[goal]
{goal}

[context]
''')
    crumb += '\n'.join(context_lines) + '\n\n[constraints]\n'
    crumb += '\n'.join(constraints) + '\nEND CRUMB\n'
    write_text(args.output, crumb)


# ── validate ─────────────────────────────────────────────────────────

def cmd_validate(args: argparse.Namespace) -> None:
    errors = 0
    for path in args.files:
        try:
            text = Path(path).read_text(encoding='utf-8')
            parsed = parse_crumb(text)
            kind = parsed['headers']['kind']
            title = parsed['headers'].get('title', '')
            label = f"  kind={kind}"
            if title:
                label += f"  title={title}"
            print(f"OK  {path}{label}")
        except Exception as exc:
            print(f"ERR {path}  {exc}", file=sys.stderr)
            errors += 1
    sys.exit(1 if errors else 0)


# ── inspect ──────────────────────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace) -> None:
    text = read_text(args.file)
    try:
        parsed = parse_crumb(text)
    except ValueError as exc:
        print(f"Parse error: {exc}", file=sys.stderr)
        sys.exit(1)

    headers = parsed['headers']
    sections = parsed['sections']

    print("Headers:")
    for key, value in headers.items():
        print(f"  {key} = {value}")

    print(f"\nSections ({len(sections)}):")
    for name, lines in sections.items():
        content_lines = [l for l in lines if l.strip()]
        print(f"  [{name}] ({len(content_lines)} lines)")
        if not args.headers_only:
            for line in content_lines:
                print(f"    {line}")

    kind = headers['kind']
    required = set(REQUIRED_SECTIONS.get(kind, []))
    present = set(sections.keys())
    missing = required - present
    extra = present - required
    if missing:
        print(f"\nMissing required: {', '.join(f'[{s}]' for s in sorted(missing))}")
    if extra:
        print(f"\nOptional sections: {', '.join(f'[{s}]' for s in sorted(extra))}")


# ── argument parser ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='crumb',
        description='Create, validate, and inspect .crumb handoff files.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # new
    new = sub.add_parser('new', help='Create a new .crumb file.')
    new.add_argument('kind', choices=['task', 'mem', 'map'], help='Kind of crumb to create.')
    new.add_argument('--title', '-t', help='Title for the crumb.')
    new.add_argument('--source', '-s', help='Source label (e.g. claude.chat, cursor.agent).')
    new.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    # task-specific
    new.add_argument('--goal', help='Goal text (task only).')
    new.add_argument('--context', nargs='*', help='Context items (task only).')
    new.add_argument('--constraints', '-c', nargs='*', help='Constraint items (task only).')
    # mem-specific
    new.add_argument('--entries', '-e', nargs='*', help='Consolidated entries (mem only).')
    # map-specific
    new.add_argument('--project', '-p', help='Project name (map only).')
    new.add_argument('--description', '-d', help='Project description (map only).')
    new.add_argument('--modules', '-m', nargs='*', help='Module entries (map only).')
    new.set_defaults(func=cmd_new)

    # from-chat
    from_chat = sub.add_parser('from-chat', help='Convert a chat log into a task crumb.')
    from_chat.add_argument('--input', '-i', default='-', help='Input file or - for stdin.')
    from_chat.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    from_chat.add_argument('--title', help='Title for the crumb.')
    from_chat.add_argument('--source', help='Source label (e.g. chatgpt.chat, claude.chat).')
    from_chat.add_argument('--goal', help='Override the default goal text.')
    from_chat.add_argument('--constraints', '-c', nargs='*', help='Constraints as separate arguments.')
    from_chat.set_defaults(func=cmd_from_chat)

    # validate
    validate = sub.add_parser('validate', help='Validate one or more .crumb files.')
    validate.add_argument('files', nargs='+', help='.crumb files to validate.')
    validate.set_defaults(func=cmd_validate)

    # inspect
    inspect_cmd = sub.add_parser('inspect', help='Parse and display a .crumb file.')
    inspect_cmd.add_argument('file', nargs='?', help='.crumb file to inspect (default: stdin).')
    inspect_cmd.add_argument('--headers-only', '-H', action='store_true', help='Show only headers and section names.')
    inspect_cmd.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

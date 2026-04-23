#!/usr/bin/env python3
"""Minimal CLI for creating, validating, inspecting, and managing .crumb handoff files."""

import argparse
import datetime
import difflib
import fnmatch
import glob
import json
import os
import platform
import re
import base64
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import Dict, List


REQUIRED_HEADERS = ["v", "kind", "source"]
REQUIRED_SECTIONS = {
    "task": ["goal", "context", "constraints"],
    "mem": ["consolidated"],
    "map": ["project", "modules"],
    "log": ["entries"],
    "todo": ["tasks"],
    "passport": ["identity", "permissions"],
    "audit": ["goal", "actions", "verdict"],
    "wake": ["identity"],
    "delta": ["changes"],
    "agent": ["identity"],
}
CLI_VERSION = "0.6.0"
SUPPORTED_VERSIONS = {"1.1", "1.2", "1.3"}
FOLD_SECTION_RE = re.compile(r"^fold:([^/]+)/(summary|full)$")
CONTENT_REF_RE = re.compile(r"^sha256:[0-9a-f]{16,64}$")
DELTA_CHANGE_RE = re.compile(r"^\s*-\s*([+\-~])\[(@?[a-z0-9_:/-]+)\]\s*(.*)$", re.IGNORECASE)
DELTA_HEADERS_SECTION = "@headers"
HANDOFF_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
WORKFLOW_LINE_RE = re.compile(r"^\s*-?\s*(\d+)[.)]\s*(.+)$")


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
    if headers["v"] not in SUPPORTED_VERSIONS:
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
        fold_summary = f"fold:{section}/summary"
        fold_full = f"fold:{section}/full"
        if section in sections:
            if not any(item.strip() for item in sections[section]):
                raise ValueError(f"section [{section}] is empty")
        elif fold_summary in sections or fold_full in sections:
            for variant_name in (fold_summary, fold_full):
                if variant_name in sections and not any(
                    item.strip() for item in sections[variant_name]
                ):
                    raise ValueError(f"section [{variant_name}] is empty")
        else:
            raise ValueError(
                f"missing required section for kind={kind}: [{section}] "
                f"(or [fold:{section}/summary] + [fold:{section}/full])"
            )

    _validate_v12_additive(headers, sections)
    _validate_v13_additive(headers, sections)

    return {"headers": headers, "sections": sections}


def _validate_v12_additive(
    headers: Dict[str, str], sections: Dict[str, List[str]]
) -> None:
    """Additive v1.2 validation. Does not reject v1.1 files."""

    if "refs" in headers:
        refs_value = headers["refs"].strip()
        if not refs_value:
            raise ValueError("refs header must not be empty when present")
        for ref in (r.strip() for r in refs_value.split(",")):
            if not ref:
                raise ValueError("refs header contains an empty entry")
            if ref.startswith("sha256:") and not CONTENT_REF_RE.match(ref):
                raise ValueError(
                    f"refs entry {ref!r} has a malformed sha256: digest"
                )

    if "refs" in sections and not any(line.strip() for line in sections["refs"]):
        raise ValueError("[refs] section is empty; omit it instead")

    if "handoff" in sections and not any(
        line.strip() for line in sections["handoff"]
    ):
        raise ValueError("[handoff] section is empty; omit it instead")

    fold_pairs: Dict[str, set] = {}
    for section_name in sections:
        match = FOLD_SECTION_RE.match(section_name)
        if not match:
            continue
        fold_name, variant = match.group(1), match.group(2)
        fold_pairs.setdefault(fold_name, set()).add(variant)

    for fold_name, variants in fold_pairs.items():
        if "full" in variants and "summary" not in variants:
            raise ValueError(
                f"fold:{fold_name} declares /full without a paired /summary"
            )

    for section_name, body in sections.items():
        meaningful = [line for line in body if line.strip()]
        for idx, line in enumerate(meaningful[:2]):
            stripped = line.strip()
            if stripped.startswith("@type:") and idx == 0:
                content_type = stripped.split(":", 1)[1].strip()
                if not content_type:
                    raise ValueError(
                        f"@type annotation has empty value in [{section_name}]"
                    )
            if stripped.startswith("@priority:"):
                raw = stripped.split(":", 1)[1].strip()
                if not raw:
                    raise ValueError(
                        f"@priority annotation has empty value in [{section_name}]"
                    )
                try:
                    score = int(raw)
                except ValueError:
                    raise ValueError(
                        f"@priority value in [{section_name}] must be an integer 1-10"
                    )
                if not 1 <= score <= 10:
                    raise ValueError(
                        f"@priority value in [{section_name}] must be between 1 and 10"
                    )

    if headers.get("kind") == "delta":
        if "base" not in headers or not headers["base"].strip():
            raise ValueError("kind=delta requires a 'base' header identifying the parent crumb")
        changes = [line for line in sections.get("changes", []) if line.strip()]
        if not changes:
            raise ValueError("kind=delta requires at least one entry in [changes]")
        for line in changes:
            stripped = line.strip()
            if stripped.startswith("@"):
                continue
            if not DELTA_CHANGE_RE.match(line):
                raise ValueError(
                    f"malformed [changes] entry: {stripped!r} "
                    "(expected '- +[section] text', '- -[section] text', or '- ~[section] text')"
                )


def _parse_kv_line(line: str) -> Dict[str, str]:
    """Parse 'key=value  key=value' style trailing annotations on a bullet line."""
    tokens: Dict[str, str] = {}
    body = line.strip()
    if body.startswith("- "):
        body = body[2:]
    elif body.startswith("-"):
        body = body[1:]
    for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)", body):
        tokens[match.group(1)] = match.group(2)
    return tokens


def _validate_v13_additive(
    headers: Dict[str, str], sections: Dict[str, List[str]]
) -> None:
    """Additive v1.3 validation. Does not reject v1.1 or v1.2 files."""
    if "fold_priority" in headers:
        value = headers["fold_priority"].strip()
        if not value:
            raise ValueError("fold_priority header must not be empty when present")
        for name in (n.strip() for n in value.split(",")):
            if not name:
                raise ValueError("fold_priority contains an empty entry")
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                raise ValueError(f"fold_priority entry {name!r} has invalid characters")

    if "handoff" in sections:
        step_ids: Dict[str, int] = {}
        deps: Dict[str, List[str]] = {}
        position = 0
        for line in sections["handoff"]:
            stripped = line.strip()
            if not stripped or stripped.startswith("- [x]"):
                continue
            if not stripped.startswith("-"):
                continue
            position += 1
            tokens = _parse_kv_line(stripped)
            step_id = tokens.get("id", str(position))
            if not HANDOFF_ID_RE.match(step_id):
                raise ValueError(
                    f"[handoff] id={step_id!r} must match [a-zA-Z0-9_-]+"
                )
            if step_id in step_ids:
                raise ValueError(f"[handoff] duplicate id={step_id!r}")
            step_ids[step_id] = position
            after = tokens.get("after", "")
            if after:
                deps[step_id] = [
                    d.strip() for d in after.split(",") if d.strip()
                ]
        for step_id, refs in deps.items():
            for ref in refs:
                if ref not in step_ids:
                    raise ValueError(
                        f"[handoff] id={step_id!r} has unknown after= dependency {ref!r}"
                    )
        _detect_dep_cycle(deps, label="[handoff]")

    if "workflow" in sections:
        step_ids: Dict[str, int] = {}
        deps: Dict[str, List[str]] = {}
        for line in sections["workflow"]:
            stripped = line.strip()
            if not stripped:
                continue
            match = WORKFLOW_LINE_RE.match(stripped)
            if not match:
                if stripped.startswith("-"):
                    continue
                raise ValueError(
                    f"[workflow] line must be numbered (e.g. '1. reproduce_bug'): {stripped!r}"
                )
            num, rest = match.group(1), match.group(2)
            tokens = _parse_kv_line("- " + rest)
            step_id = tokens.get("id", num)
            if not HANDOFF_ID_RE.match(step_id):
                raise ValueError(
                    f"[workflow] id={step_id!r} must match [a-zA-Z0-9_-]+"
                )
            if step_id in step_ids:
                raise ValueError(f"[workflow] duplicate id={step_id!r}")
            step_ids[step_id] = int(num)
            depends = tokens.get("depends_on", "")
            if depends:
                deps[step_id] = [
                    d.strip() for d in depends.split(",") if d.strip()
                ]
        for step_id, refs in deps.items():
            for ref in refs:
                if ref not in step_ids:
                    raise ValueError(
                        f"[workflow] id={step_id!r} has unknown depends_on {ref!r}"
                    )
        _detect_dep_cycle(deps, label="[workflow]")

    if "script" in sections:
        meaningful = [line for line in sections["script"] if line.strip()]
        if meaningful and not meaningful[0].strip().startswith("@type:"):
            raise ValueError("[script] section must begin with @type: <lang>")

    if "checks" in sections:
        for line in sections["checks"]:
            stripped = line.strip()
            if not stripped or not stripped.startswith("-"):
                continue
            body = stripped[1:].strip()
            if "::" not in body:
                raise ValueError(
                    f"[checks] line must use 'name :: status' format: {stripped!r}"
                )


def _detect_dep_cycle(deps: Dict[str, List[str]], label: str) -> None:
    """Raise if deps contains a cycle. Uses DFS coloring."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {k: WHITE for k in deps}

    def visit(node: str) -> None:
        if color.get(node, WHITE) == GRAY:
            raise ValueError(f"{label} dependency cycle through {node!r}")
        if color.get(node, WHITE) == BLACK:
            return
        color[node] = GRAY
        for child in deps.get(node, []):
            if child in deps:
                visit(child)
        color[node] = BLACK

    for node in list(deps):
        if color[node] == WHITE:
            visit(node)


def render_crumb(headers: Dict[str, str], sections: Dict[str, List[str]]) -> str:
    """Render headers and sections back into a .crumb file string."""
    lines = ["BEGIN CRUMB"]
    for key, value in headers.items():
        lines.append(f"{key}={value}")
    lines.append("---")
    for name, body in sections.items():
        lines.append(f"[{name}]")
        lines.extend(body)
        if body and body[-1].strip():
            lines.append("")
    lines.append("END CRUMB")
    return "\n".join(lines) + "\n"


def normalize_entry(text: str) -> str:
    """Normalize a bullet entry for deduplication comparison."""
    text = text.strip()
    while text.startswith(('-', '*', ' ')):
        text = text.lstrip('-* ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from an entry (stopwords removed)."""
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or',
        'nor', 'not', 'no', 'so', 'if', 'that', 'this', 'it', 'its',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very',
        'just', 'about', 'also', 'use', 'using', 'prefers', 'prefer',
        'wants', 'like', 'likes',
    }
    words = set(re.findall(r'[a-z0-9]+', text.lower()))
    return words - STOPWORDS


def score_entry(entry: str, all_entries: list, entry_keywords: dict) -> float:
    """Score an entry by information density (signal ranking).

    Higher score = more unique information = higher priority to keep.
    Factors: keyword uniqueness across all entries, entry specificity (length),
    and whether it contains concrete nouns (technical terms, proper nouns).
    """
    norm = normalize_entry(entry)
    keywords = entry_keywords.get(norm, set())
    if not keywords:
        return 0.0

    # Keyword uniqueness: how many other entries share these keywords?
    # Rare keywords = high signal (like rare tokens in TF-IDF)
    keyword_counts = Counter()
    for e in all_entries:
        e_norm = normalize_entry(e)
        for kw in entry_keywords.get(e_norm, set()):
            keyword_counts[kw] += 1

    uniqueness = 0.0
    for kw in keywords:
        count = keyword_counts.get(kw, 1)
        uniqueness += 1.0 / count  # rare keywords score higher

    # Specificity: longer entries with more keywords carry more information
    specificity = len(keywords) * 0.5

    # Technical term bonus: entries with specific terms (numbers, paths, versions)
    tech_bonus = 0.0
    if re.search(r'\d', entry):
        tech_bonus += 1.0  # contains numbers
    if re.search(r'[/\\.]', entry):
        tech_bonus += 1.0  # contains paths or extensions
    if re.search(r'v\d|[A-Z][a-z]+[A-Z]', entry):
        tech_bonus += 1.0  # version numbers or camelCase

    return uniqueness + specificity + tech_bonus


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
    "log": dedent("""\
        BEGIN CRUMB
        v=1.1
        kind=log
        title={title}
        source={source}
        ---
        [entries]
        {entries}
        END CRUMB
    """),
    "todo": dedent("""\
        BEGIN CRUMB
        v=1.1
        kind=todo
        title={title}
        source={source}
        ---
        [tasks]
        {tasks}
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
    "log": {
        "entries": "<timestamped session log entries>",
    },
    "todo": {
        "tasks": "- [ ] <task description>",
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
    elif kind == "log":
        if args.entries:
            now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            values["entries"] = "\n".join(f"- [{now}] {e}" for e in args.entries)
        else:
            values["entries"] = PLACEHOLDERS["log"]["entries"]
    elif kind == "todo":
        if args.entries:
            values["tasks"] = "\n".join(f"- [ ] {e}" for e in args.entries)
        else:
            values["tasks"] = PLACEHOLDERS["todo"]["tasks"]

    crumb = TEMPLATES[kind].format(**values)
    write_text(args.output, crumb)


# ── from-chat ────────────────────────────────────────────────────────

AI_PREFIXES = ('assistant:', 'ai:', 'claude:', 'gpt:', 'chatgpt:', 'copilot:', 'gemini:', 'system:')

DECISION_PATTERNS = [
    re.compile(r"(?:let's|we'll|we should|decided to|going with|chose|picking|switched to|using)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:decision|conclusion|agreed|settled on)[:\s]+(.+)", re.IGNORECASE),
]


def parse_chat_lines(raw: str) -> tuple:
    """Parse chat text into structured segments: user lines, AI lines, code blocks, decisions."""
    lines = raw.splitlines()
    user_lines = []
    ai_lines = []
    code_blocks = []
    decisions = []

    in_code_block = False
    current_code = []
    code_lang = ''

    for line in lines:
        stripped = line.strip()

        # Code block detection
        if stripped.startswith('```'):
            if in_code_block:
                code_blocks.append({'lang': code_lang, 'code': '\n'.join(current_code)})
                current_code = []
                in_code_block = False
            else:
                in_code_block = True
                code_lang = stripped[3:].strip()
            continue

        if in_code_block:
            current_code.append(line)
            continue

        if not stripped:
            continue

        # Decision extraction
        for pat in DECISION_PATTERNS:
            m = pat.search(stripped)
            if m:
                decisions.append(m.group(0).strip())
                break

        # Classify as user or AI
        if stripped.lower().startswith(AI_PREFIXES):
            ai_lines.append(stripped)
        else:
            user_lines.append(stripped)

    return user_lines, ai_lines, code_blocks, decisions


def cmd_from_chat(args: argparse.Namespace) -> None:
    raw = read_text(args.input)
    user_lines, ai_lines, code_blocks, decisions = parse_chat_lines(raw)

    goal = args.goal.strip() if args.goal else 'Continue this work from where the last assistant left off.'
    title = args.title or 'Continue previous session'
    source = args.source or 'chat.log'

    context_lines = []

    # Decisions first — highest signal
    if decisions:
        context_lines.append('- Decisions made:')
        for d in decisions[:6]:
            context_lines.append(f'  - {d}')

    # Code blocks — concrete artifacts
    if code_blocks:
        context_lines.append(f'- Code discussed ({len(code_blocks)} block{"s" if len(code_blocks) != 1 else ""}):')
        for block in code_blocks[:3]:
            lang_label = f' ({block["lang"]})' if block["lang"] else ''
            preview = block["code"].strip().split('\n')[0][:80]
            context_lines.append(f'  - {preview}...{lang_label}')

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

    # Build output based on kind
    kind = args.kind

    if kind == 'mem' and decisions:
        # Extract decisions as mem entries
        entries = [f'- {d}' for d in decisions]
        headers = {
            'v': '1.1', 'kind': 'mem', 'title': title,
            'source': source,
        }
        sections = {'consolidated': entries + ['']}
        crumb_text = render_crumb(headers, sections)
    else:
        crumb_text = dedent(f'''\
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
        crumb_text += '\n'.join(context_lines) + '\n\n[constraints]\n'
        crumb_text += '\n'.join(constraints) + '\nEND CRUMB\n'

    write_text(args.output, crumb_text)


# ── from-git ────────────────────────────────────────────────────────

def _git_run(*cmd: str) -> str:
    """Run a git command and return stripped stdout. Raises SystemExit on failure."""
    result = subprocess.run(
        ['git'] + list(cmd),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ''
    return result.stdout.strip()


def _detect_base_branch() -> str:
    """Auto-detect the base branch: try main, then master, then fall back to None."""
    for candidate in ('main', 'master'):
        result = subprocess.run(
            ['git', 'rev-parse', '--verify', candidate],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return candidate
    return ''


def cmd_from_git(args: argparse.Namespace) -> None:
    """Generate a task crumb from recent git activity."""
    # Check that we are inside a git repo
    check = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        print("Error: not inside a git repository.", file=sys.stderr)
        sys.exit(1)

    commits = args.commits
    base = args.branch
    source = args.source or 'git'

    # Current branch
    current_branch = _git_run('branch', '--show-current') or 'HEAD'

    # Auto-detect base branch if not provided
    if not base:
        base = _detect_base_branch()

    # Recent commit messages (oneline)
    recent_commits = _git_run('log', '--oneline', f'-{commits}')
    commit_lines = [l for l in recent_commits.splitlines() if l.strip()] if recent_commits else []

    # Latest commit subject for goal inference
    latest_subject = _git_run('log', '-1', '--format=%s')

    # Changed files and diff stat
    if base:
        changed_files = _git_run('diff', '--name-only', f'{base}...HEAD')
        diff_stat = _git_run('diff', '--stat', f'{base}...HEAD')
    else:
        # No base branch found — compare against HEAD~N
        changed_files = _git_run('diff', '--name-only', f'HEAD~{commits}', 'HEAD')
        diff_stat = _git_run('diff', '--stat', f'HEAD~{commits}', 'HEAD')

    changed_file_list = [f for f in changed_files.splitlines() if f.strip()] if changed_files else []

    # --- Infer goal ---
    if args.title:
        goal = args.title
    elif current_branch not in ('main', 'master', 'HEAD', ''):
        # Feature branch — use branch name as goal hint
        branch_label = current_branch.replace('-', ' ').replace('_', ' ').replace('/', ': ')
        goal = f"Continue work on: {branch_label} (latest: {latest_subject})"
    elif latest_subject:
        goal = f"Continue from: {latest_subject}"
    else:
        goal = "Continue recent work in this repository."

    title = args.title or goal[:80]

    # --- Build context ---
    context_lines = []
    context_lines.append(f'- Branch: {current_branch}')

    if base:
        context_lines.append(f'- Base branch: {base}')

    if commit_lines:
        context_lines.append(f'- Recent commits ({len(commit_lines)}):')
        for cl in commit_lines:
            context_lines.append(f'  - {cl}')

    if changed_file_list:
        context_lines.append(f'- Changed files ({len(changed_file_list)}):')
        for cf in changed_file_list[:20]:
            context_lines.append(f'  - {cf}')
        if len(changed_file_list) > 20:
            context_lines.append(f'  - ... and {len(changed_file_list) - 20} more')

    if diff_stat:
        context_lines.append('- Diff summary:')
        for line in diff_stat.splitlines()[-3:]:
            context_lines.append(f'  - {line.strip()}')

    if not context_lines:
        context_lines = ['- No git context available.']

    # --- Infer constraints ---
    constraint_lines = []

    # Check for failing tests by looking for common test failure indicators
    test_result = subprocess.run(
        ['git', 'log', '-1', '--format=%B'],
        capture_output=True, text=True,
    )
    last_body = test_result.stdout.lower() if test_result.returncode == 0 else ''
    if any(kw in last_body for kw in ('fixme', 'wip', 'todo', 'broken', 'failing')):
        constraint_lines.append('- Warning: latest commit may contain incomplete work (WIP/TODO detected in message).')

    # Check for merge conflicts
    conflict_check = subprocess.run(
        ['git', 'diff', '--check'],
        capture_output=True, text=True,
    )
    if conflict_check.returncode != 0 and 'conflict' in conflict_check.stdout.lower():
        constraint_lines.append('- Merge conflicts detected — resolve before continuing.')

    if not constraint_lines:
        constraint_lines.append('- Review before merging.')

    # --- Render crumb ---
    crumb_text = render_crumb(
        headers={
            'v': '1.1',
            'kind': 'task',
            'title': title,
            'source': source,
        },
        sections={
            'goal': [goal],
            'context': context_lines,
            'constraints': constraint_lines,
        },
    )

    write_text(args.output, crumb_text)


# ── validate ─────────────────────────────────────────────────────────

def cmd_validate(args: argparse.Namespace) -> None:
    errors = 0
    expanded: List[str] = []
    for raw in args.files:
        if any(ch in raw for ch in '*?[]'):
            matches = sorted(glob.glob(raw))
            expanded.extend(matches or [raw])
            continue
        candidate = Path(raw)
        if candidate.is_dir():
            expanded.extend(str(p) for p in sorted(candidate.rglob('*.crumb')))
            continue
        expanded.append(raw)
    for path in expanded:
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

    # Token stats
    tokens = estimate_tokens(text)
    total_lines = sum(len([l for l in lines if l.strip()]) for lines in sections.values())
    print(f"\nToken cost: ~{tokens} tokens ({len(text)} chars)")
    print(f"Content density: {total_lines} lines across {len(sections)} sections")

    # Info density score
    all_content = ' '.join(' '.join(lines) for lines in sections.values())
    kw = extract_keywords(all_content)
    density = len(kw) / max(tokens, 1) * 100
    print(f"Keyword density: {len(kw)} unique keywords ({density:.1f} per 100 tokens)")


# ── append ───────────────────────────────────────────────────────────

def cmd_append(args: argparse.Namespace) -> None:
    """Append raw observations to a mem crumb's [raw] section."""
    path = Path(args.file)
    text = path.read_text(encoding='utf-8')
    parsed = parse_crumb(text)

    if parsed['headers']['kind'] != 'mem':
        print(f"Error: {args.file} is kind={parsed['headers']['kind']}, expected kind=mem", file=sys.stderr)
        sys.exit(1)

    if 'raw' not in parsed['sections']:
        parsed['sections']['raw'] = ['']

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for entry in args.entries:
        parsed['sections']['raw'].append(f"- [{timestamp}] {entry}")
    parsed['sections']['raw'].append('')

    output = render_crumb(parsed['headers'], parsed['sections'])
    path.write_text(output, encoding='utf-8')
    print(f"Appended {len(args.entries)} entries to [raw] in {args.file}")
    run_hook('post_append', {'file': args.file, 'count': str(len(args.entries))})


# ── dream ────────────────────────────────────────────────────────────

def cmd_dream(args: argparse.Namespace) -> None:
    """Run a consolidation pass: deduplicate, merge [raw] into [consolidated], prune to budget."""
    path = Path(args.file)
    text = path.read_text(encoding='utf-8')
    parsed = parse_crumb(text)

    if parsed['headers']['kind'] != 'mem':
        print(f"Error: {args.file} is kind={parsed['headers']['kind']}, expected kind=mem", file=sys.stderr)
        sys.exit(1)

    headers = parsed['headers']
    sections = parsed['sections']

    # Collect all entries from [consolidated] and [raw]
    existing = [l.strip() for l in sections.get('consolidated', []) if l.strip()]
    raw = [l.strip() for l in sections.get('raw', []) if l.strip()]

    # Strip timestamps from raw entries for merging: "- [2026-03-28T...] fact" → "- fact"
    cleaned_raw = []
    for entry in raw:
        stripped = re.sub(r'^-\s*\[\d{4}-\d{2}-\d{2}T[^\]]*\]\s*', '- ', entry)
        cleaned_raw.append(stripped)

    # Deduplicate: normalize and track seen entries
    seen = set()
    merged = []

    # Raw entries take priority (newer truth wins), so process them first
    for entry in cleaned_raw:
        norm = normalize_entry(entry)
        if norm and norm not in seen:
            seen.add(norm)
            merged.append(entry)

    # Then add existing entries that aren't duplicated by raw
    for entry in existing:
        norm = normalize_entry(entry)
        if norm and norm not in seen:
            seen.add(norm)
            merged.append(entry)

    # Prune to budget using signal scoring
    # Instead of dropping from the end, score entries by information density
    # and drop lowest-signal entries first
    budget = int(headers.get('max_index_tokens', '0'))
    pruned = 0
    if budget > 0 and estimate_tokens('\n'.join(merged)) > budget:
        # Build keyword index for scoring
        entry_kw = {normalize_entry(e): extract_keywords(e) for e in merged}
        # Score and sort: keep highest-signal entries
        scored = [(score_entry(e, merged, entry_kw), i, e) for i, e in enumerate(merged)]
        while scored and estimate_tokens('\n'.join(s[2] for s in scored)) > budget:
            # Remove lowest-scoring entry
            scored.sort(key=lambda x: (x[0], -x[1]))  # lowest score first, oldest first on tie
            scored.pop(0)
            pruned += 1
        # Restore original order for surviving entries
        scored.sort(key=lambda x: x[1])
        merged = [s[2] for s in scored]

    # Update sections
    sections['consolidated'] = [f"{e}" for e in merged] + ['']
    sections.pop('raw', None)

    # Update dream metadata
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    headers['dream_pass'] = now
    prev_sessions = int(headers.get('dream_sessions', '0'))
    headers['dream_sessions'] = str(prev_sessions + 1)

    # Update [dream] section
    notes = [f"- last_pass: {now}"]
    notes.append(f"- sessions_seen: {prev_sessions + 1}")
    notes.append(f"- entries_before: {len(existing) + len(raw)}")
    notes.append(f"- entries_after: {len(merged)}")
    notes.append(f"- deduplicated: {len(existing) + len(cleaned_raw) - len(merged)}")
    if pruned:
        notes.append(f"- pruned_for_budget: {pruned}")
    sections['dream'] = notes + ['']

    output = render_crumb(headers, sections)
    original_tokens = estimate_tokens(text)
    output_tokens = estimate_tokens(output)
    ratio = ((original_tokens - output_tokens) / original_tokens * 100) if original_tokens > 0 else 0
    if args.dry_run:
        sys.stdout.write(output)
    else:
        path.write_text(output, encoding='utf-8')
        print(f"Dream pass complete on {args.file}")
        print(f"  {len(existing)} existing + {len(raw)} raw → {len(merged)} consolidated")
        if pruned:
            print(f"  Pruned {pruned} entries to fit budget ({budget} tokens)")
        print(f"  Compression: {original_tokens} → {output_tokens} tokens ({ratio:.0f}% reduction)")
        run_hook('post_dream', {'file': args.file, 'entries': str(len(merged))})


# ── repo / gitignore helpers (used by pack and other scanners) ───────

COMMON_IGNORED_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
}
COMMON_IGNORED_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}


def _git_completed(args: list, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )


def _git_repo_root(path: Path) -> Path | None:
    result = _git_completed(["rev-parse", "--show-toplevel"], cwd=path)
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root).resolve() if root else None


def _load_gitignore_patterns(root: Path) -> list:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for raw in gitignore.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line)
    return patterns


def _matches_gitignore_pattern(relative_path: str, pattern: str, is_dir: bool) -> bool:
    rel = relative_path.replace("\\", "/")
    normalized = pattern.rstrip("/")
    if not normalized:
        return False
    dir_only = pattern.endswith("/")
    if dir_only and not is_dir:
        return False
    if "/" in normalized:
        candidate = normalized.lstrip("/")
        return fnmatch.fnmatch(rel, candidate) or rel.startswith(candidate + "/")
    parts = [part for part in rel.split("/") if part]
    return any(fnmatch.fnmatch(part, normalized) for part in parts)


def _is_ignored_path(
    path: Path,
    root: Path,
    repo_root: Path | None,
    gitignore_patterns: list,
    cache: dict,
) -> bool:
    key = str(path.resolve())
    if key in cache:
        return cache[key]

    if path.name in COMMON_IGNORED_NAMES or any(path.name.endswith(suffix) for suffix in COMMON_IGNORED_SUFFIXES):
        cache[key] = True
        return True

    relative_to_root = path.resolve().relative_to(root.resolve()).as_posix()
    for pattern in gitignore_patterns:
        if _matches_gitignore_pattern(relative_to_root, pattern, path.is_dir()):
            cache[key] = True
            return True

    if repo_root is not None:
        try:
            repo_relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            repo_relative = path.name
        result = _git_completed(["check-ignore", repo_relative], cwd=repo_root)
        if result.returncode == 0:
            cache[key] = True
            return True

    cache[key] = False
    return False


def _build_repo_tree(root: Path) -> list:
    root = root.resolve()
    repo_root = _git_repo_root(root)
    gitignore_patterns = _load_gitignore_patterns(root)
    ignore_cache: dict = {}
    lines = [f"- {root.name}/"]

    for current_root, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current_root).resolve()
        depth = len(current_path.relative_to(root).parts)
        indent = "  " * depth

        visible_dirs = []
        for dirname in sorted(dirs):
            candidate = current_path / dirname
            if not _is_ignored_path(candidate, root, repo_root, gitignore_patterns, ignore_cache):
                visible_dirs.append(dirname)
        dirs[:] = visible_dirs

        for dirname in dirs:
            lines.append(f"{indent}- {dirname}/")

        for filename in sorted(files):
            candidate = current_path / filename
            if _is_ignored_path(candidate, root, repo_root, gitignore_patterns, ignore_cache):
                continue
            lines.append(f"{indent}- {filename}")

    return lines


# ── search ───────────────────────────────────────────────────────────

def _load_crumb_files(search_dir: Path) -> list:
    """Load and parse all .crumb files in a directory."""
    results = []
    for path in sorted(search_dir.rglob('*.crumb')):
        try:
            text = path.read_text(encoding='utf-8')
            parsed = parse_crumb(text)
            results.append((path, text, parsed))
        except (ValueError, Exception):
            continue
    return results


def _search_keyword(query_terms: list, crumb_files: list) -> list:
    """Keyword search: exact term matching with frequency scoring."""
    results = []
    for path, text, parsed in crumb_files:
        headers = parsed['headers']
        sections = parsed['sections']

        all_text = ' '.join(' '.join(lines) for lines in sections.values()).lower()
        header_text = ' '.join(f"{k} {v}" for k, v in headers.items()).lower()
        full_text = header_text + ' ' + all_text

        score = 0
        matched_terms = []
        for term in query_terms:
            count = full_text.count(term)
            if count > 0:
                score += count
                matched_terms.append(term)

        if not matched_terms:
            continue

        if len(matched_terms) == len(query_terms):
            score += 10

        matching_sections = []
        for name, lines in sections.items():
            section_text = ' '.join(lines).lower()
            if any(term in section_text for term in query_terms):
                matching_sections.append(name)

        results.append({
            'path': path, 'score': score, 'kind': headers['kind'],
            'title': headers.get('title', ''), 'matched_terms': matched_terms,
            'matching_sections': matching_sections,
        })
    return results


def _search_fuzzy(query_terms: list, crumb_files: list) -> list:
    """Fuzzy search: uses difflib for approximate matching."""
    query = ' '.join(query_terms)
    results = []
    for path, text, parsed in crumb_files:
        headers = parsed['headers']
        sections = parsed['sections']

        # Collect all text lines for fuzzy matching
        all_lines = []
        for name, lines in sections.items():
            for line in lines:
                if line.strip():
                    all_lines.append((name, line.strip()))

        # Also match against title and headers
        title = headers.get('title', '')
        if title:
            all_lines.append(('title', title))

        best_ratio = 0.0
        best_section = ''
        matched_terms = []

        for section_name, line in all_lines:
            ratio = difflib.SequenceMatcher(None, query.lower(), line.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_section = section_name

            # Also check individual term fuzzy matches
            words = line.lower().split()
            for term in query_terms:
                matches = difflib.get_close_matches(term, words, n=1, cutoff=0.6)
                if matches and term not in matched_terms:
                    matched_terms.append(term)

        if best_ratio < 0.3 and not matched_terms:
            continue

        score = int(best_ratio * 100)
        matching_sections = [best_section] if best_section else []

        results.append({
            'path': path, 'score': score, 'kind': headers['kind'],
            'title': headers.get('title', ''), 'matched_terms': matched_terms or query_terms,
            'matching_sections': matching_sections,
        })
    return results


def _search_ranked(query_terms: list, crumb_files: list) -> list:
    """Ranked search: uses TF-IDF keyword scoring for information-density ranking."""
    results = []
    for path, text, parsed in crumb_files:
        headers = parsed['headers']
        sections = parsed['sections']

        all_entries = []
        for name, lines in sections.items():
            for line in lines:
                if line.strip():
                    all_entries.append(line.strip())

        if not all_entries:
            continue

        # Build keyword index
        entry_kw = {normalize_entry(e): extract_keywords(e) for e in all_entries}
        query_kw = set(query_terms)

        # Score each entry by: query relevance × information density
        total_score = 0.0
        matched_terms = []
        matching_sections = []

        for name, lines in sections.items():
            section_matched = False
            for line in lines:
                if not line.strip():
                    continue
                entry = line.strip()
                kw = entry_kw.get(normalize_entry(entry), set())
                overlap = kw & query_kw
                if overlap:
                    # Query relevance
                    relevance = len(overlap) / len(query_kw)
                    # Information density from score_entry
                    density = score_entry(entry, all_entries, entry_kw)
                    total_score += relevance * density
                    matched_terms.extend(t for t in overlap if t not in matched_terms)
                    section_matched = True
            if section_matched:
                matching_sections.append(name)

        if total_score == 0:
            continue

        results.append({
            'path': path, 'score': int(total_score * 10), 'kind': headers['kind'],
            'title': headers.get('title', ''), 'matched_terms': matched_terms,
            'matching_sections': matching_sections,
        })
    return results


def cmd_search(args: argparse.Namespace) -> None:
    """Search across .crumb files by keyword, fuzzy match, or ranked relevance."""
    query_terms = args.query.lower().split()
    search_dir = Path(args.dir)
    method = args.method

    if not search_dir.is_dir():
        print(f"Error: {args.dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    crumb_files = _load_crumb_files(search_dir)
    if not crumb_files:
        print(f"No .crumb files found in {args.dir}")
        return

    if method == 'fuzzy':
        results = _search_fuzzy(query_terms, crumb_files)
    elif method == 'ranked':
        results = _search_ranked(query_terms, crumb_files)
    else:
        results = _search_keyword(query_terms, crumb_files)

    results.sort(key=lambda r: r['score'], reverse=True)

    if not results:
        print(f"No matches for: {args.query}")
        return

    limit = args.limit or len(results)
    for r in results[:limit]:
        title_part = f"  title={r['title']}" if r['title'] else ""
        sections_part = f"  in [{', '.join(r['matching_sections'])}]" if r['matching_sections'] else ""
        print(f"  {r['score']:3d}  {r['path']}  kind={r['kind']}{title_part}{sections_part}")


# ── merge ────────────────────────────────────────────────────────────

def cmd_merge(args: argparse.Namespace) -> None:
    """Merge multiple mem crumbs into one consolidated file."""
    all_entries = []
    sources = []

    for filepath in args.files:
        text = Path(filepath).read_text(encoding='utf-8')
        parsed = parse_crumb(text)
        if parsed['headers']['kind'] != 'mem':
            print(f"Skipping {filepath}: kind={parsed['headers']['kind']}, expected mem", file=sys.stderr)
            continue
        sources.append(parsed['headers'].get('source', 'unknown'))

        for section_name in ('consolidated', 'raw'):
            for line in parsed['sections'].get(section_name, []):
                if line.strip():
                    all_entries.append(line.strip())

    # Deduplicate
    seen = set()
    merged = []
    for entry in all_entries:
        norm = normalize_entry(entry)
        if norm and norm not in seen:
            seen.add(norm)
            merged.append(entry)

    title = args.title or "Merged memory"
    source = ', '.join(sorted(set(sources))) if sources else "merged"
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    headers = {
        'v': '1.1',
        'kind': 'mem',
        'title': title,
        'source': source,
        'dream_pass': now,
        'dream_sessions': '1',
    }
    sections = {
        'consolidated': merged + [''],
        'dream': [
            f"- last_pass: {now}",
            f"- merged_from: {len(args.files)} files",
            f"- entries_before: {len(all_entries)}",
            f"- entries_after: {len(merged)}",
            f"- deduplicated: {len(all_entries) - len(merged)}",
            '',
        ],
    }

    output = render_crumb(headers, sections)
    write_text(args.output, output)
    if args.output != '-':
        print(f"Merged {len(args.files)} files → {len(merged)} entries in {args.output}")


# ── compact ──────────────────────────────────────────────────────────

# Eliminate overhead. Strip a crumb to its minimum
# viable form — required headers, required sections, nothing else.

OPTIONAL_HEADERS = {'title', 'dream_pass', 'dream_sessions', 'max_index_tokens',
                    'max_total_tokens', 'id', 'project', 'env', 'tags', 'url'}


def cmd_compact(args: argparse.Namespace) -> None:
    """Strip a crumb to minimum viable form. Removes optional headers and sections."""
    text = read_text(args.file)
    parsed = parse_crumb(text)
    headers = parsed['headers']
    sections = parsed['sections']
    kind = headers['kind']

    # Keep only required headers + title (title is too useful to strip)
    compact_headers = {}
    keep_headers = set(REQUIRED_HEADERS) | {'title'}
    if kind == 'map':
        keep_headers.add('project')
    for key in headers:
        if key in keep_headers:
            compact_headers[key] = headers[key]

    # Keep only required sections, strip empty lines within sections
    required = set(REQUIRED_SECTIONS.get(kind, []))
    compact_sections = {}
    for name in sections:
        if name in required:
            compact_sections[name] = [l for l in sections[name] if l.strip()]
            if compact_sections[name]:
                compact_sections[name].append('')

    output = render_crumb(compact_headers, compact_sections)

    original_tokens = estimate_tokens(text)
    compact_tokens = estimate_tokens(output)
    reduction = ((original_tokens - compact_tokens) / original_tokens * 100) if original_tokens > 0 else 0

    write_text(args.output, output)
    if args.output != '-':
        print(f"Compacted {args.file}: {original_tokens} → {compact_tokens} tokens ({reduction:.0f}% reduction)")


# ── compress ────────────────────────────────────────────────────────

def _semantic_dedup(entries: list) -> list:
    """Stage 1: Semantic deduplication — merge near-duplicate entries.

    Normalizes entries and merges those with high similarity.
    """
    if not entries:
        return []

    result = []
    seen_norms = set()
    for entry in entries:
        norm = normalize_entry(entry)
        if not norm:
            continue
        # Exact dedup
        if norm in seen_norms:
            continue
        # Fuzzy dedup: check similarity against existing entries
        is_dup = False
        for existing_norm in seen_norms:
            ratio = difflib.SequenceMatcher(None, norm, existing_norm).ratio()
            if ratio > 0.8:  # 80% similar = duplicate
                is_dup = True
                break
        if not is_dup:
            seen_norms.add(norm)
            result.append(entry)
    return result


def _signal_prune(entries: list, target_ratio: float = 0.5) -> tuple:
    """Stage 2: Signal-scored pruning — keep high-signal, drop low-signal.

    Reduces each entry to a keep/drop decision based on information density
    scoring. Returns (kept_entries, pruned_count).
    """
    if not entries or target_ratio >= 1.0:
        return entries, 0

    target_count = max(1, int(len(entries) * target_ratio))
    if len(entries) <= target_count:
        return entries, 0

    # Build keyword index
    entry_kw = {normalize_entry(e): extract_keywords(e) for e in entries}
    # Score each entry
    scored = [(score_entry(e, entries, entry_kw), i, e) for i, e in enumerate(entries)]
    # Sort by score descending, keep top entries
    scored.sort(key=lambda x: (-x[0], x[1]))
    kept = scored[:target_count]
    # Restore original order
    kept.sort(key=lambda x: x[1])

    return [s[2] for s in kept], len(entries) - target_count


def cmd_compress(args: argparse.Namespace) -> None:
    """Two-stage context compression.

    Stage 1: Semantic deduplication — normalize and merge near-duplicate
    entries across all sections.
    Stage 2: Signal-scored pruning — score entries by information density
    and drop lowest-signal entries to hit target ratio.
    """
    text = read_text(args.file)
    parsed = parse_crumb(text)
    headers = parsed['headers']
    sections = parsed['sections']
    kind = headers['kind']
    original_tokens = estimate_tokens(text)

    target = args.target  # target ratio: 0.0 = max compression, 1.0 = no compression

    stats = {'stage1_removed': 0, 'stage2_removed': 0}

    # Apply two-stage compression to content sections
    for name, lines in sections.items():
        entries = [l for l in lines if l.strip()]
        if len(entries) < 2:
            continue

        # Stage 1: Semantic dedup
        deduped = _semantic_dedup(entries)
        stats['stage1_removed'] += len(entries) - len(deduped)

        # Stage 2: Signal pruning (only if target < 1.0)
        pruned, pruned_count = _signal_prune(deduped, target)
        stats['stage2_removed'] += pruned_count

        sections[name] = pruned + [''] if pruned else ['']

    # Also strip optional headers
    keep_headers = set(REQUIRED_HEADERS) | {'title'}
    if kind == 'map':
        keep_headers.add('project')
    compact_headers = {k: v for k, v in headers.items() if k in keep_headers}

    output = render_crumb(compact_headers, sections)

    # Stage 3 (optional): MeTalk caveman compression
    metalk_stats = None
    if getattr(args, 'metalk', False):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cli.metalk import encode as metalk_encode, compression_stats as metalk_cs
        pre_mt = output
        output = metalk_encode(output, level=getattr(args, 'metalk_level', 2))
        metalk_stats = metalk_cs(pre_mt, output)

    output_tokens = estimate_tokens(output)
    ratio = ((original_tokens - output_tokens) / original_tokens * 100) if original_tokens > 0 else 0

    write_text(args.output, output)

    if args.output != '-':
        print(f"CRUMB compression on {args.file}:")
        print(f"  Stage 1 (semantic dedup):  {stats['stage1_removed']} entries merged")
        print(f"  Stage 2 (signal pruning):  {stats['stage2_removed']} low-signal entries dropped")
        if metalk_stats:
            print(f"  Stage 3 (MeTalk):          {metalk_stats['pct_saved']}% additional reduction")
        print(f"  Result: {original_tokens} → {output_tokens} tokens ({ratio:.0f}% reduction)")
        multiplier = original_tokens / max(output_tokens, 1)
        print(f"  Compression ratio: {multiplier:.1f}x")


# ── bench ───────────────────────────────────────────────────────────

def cmd_bench(args: argparse.Namespace) -> None:
    """Benchmark a crumb's compression efficiency and information density."""
    text = read_text(args.file)
    parsed = parse_crumb(text)
    headers = parsed['headers']
    sections = parsed['sections']
    kind = headers['kind']

    tokens = estimate_tokens(text)
    chars = len(text)
    total_lines = sum(len([l for l in lines if l.strip()]) for lines in sections.values())

    # Keyword analysis
    all_content = ' '.join(' '.join(lines) for lines in sections.values())
    keywords = extract_keywords(all_content)
    keyword_density = len(keywords) / max(tokens, 1) * 100

    # Compute compressibility: simulate two-stage compress
    sim_sections = {}
    original_entries = 0
    after_stage1 = 0
    after_stage2 = 0
    for name, lines in sections.items():
        entries = [l for l in lines if l.strip()]
        original_entries += len(entries)
        deduped = _semantic_dedup(entries)
        after_stage1 += len(deduped)
        pruned, _ = _signal_prune(deduped, 0.5)
        after_stage2 += len(pruned)
        sim_sections[name] = pruned + [''] if pruned else ['']

    keep_h = set(REQUIRED_HEADERS) | {'title'}
    if kind == 'map':
        keep_h.add('project')
    sim_headers = {k: v for k, v in headers.items() if k in keep_h}
    compressed = render_crumb(sim_headers, sim_sections)
    compressed_tokens = estimate_tokens(compressed)
    max_ratio = tokens / max(compressed_tokens, 1)

    # MeTalk stage 3 projection
    from cli.metalk import encode as _mt_encode
    metalked = _mt_encode(compressed, level=2)
    metalked_tokens = estimate_tokens(metalked)
    metalk_saved_pct = ((compressed_tokens - metalked_tokens) / max(compressed_tokens, 1)) * 100
    full_ratio = tokens / max(metalked_tokens, 1)

    # Score components
    density_score = min(keyword_density * 5, 25)  # max 25
    compression_score = min(max_ratio * 5, 25)  # max 25
    structure_score = 25 if not (set(REQUIRED_SECTIONS.get(kind, [])) - set(sections.keys())) else 10
    conciseness_score = min(25, max(0, 25 - (tokens - 100) / 40))  # smaller = better, max 25

    total = density_score + compression_score + structure_score + conciseness_score

    # Grade
    if total >= 85:
        grade = 'A'
    elif total >= 70:
        grade = 'B'
    elif total >= 55:
        grade = 'C'
    elif total >= 40:
        grade = 'D'
    else:
        grade = 'F'

    print(f"CRUMB Bench — {args.file}")
    print(f"{'=' * 50}")
    print(f"  Kind:              {kind}")
    print(f"  Token cost:        ~{tokens} tokens ({chars} chars)")
    print(f"  Content:           {total_lines} lines, {len(sections)} sections")
    print(f"  Unique keywords:   {len(keywords)}")
    print(f"  Keyword density:   {keyword_density:.1f} per 100 tokens")
    print(f"  Max compression:   {max_ratio:.1f}x ({tokens} → {compressed_tokens} tokens)")
    print(f"  Dedup potential:   {original_entries} → {after_stage1} entries (stage 1)")
    print(f"  Prune potential:   {after_stage1} → {after_stage2} entries (stage 2)")
    print(f"  MeTalk potential:  {full_ratio:.1f}x ({tokens} → {metalked_tokens} tokens, +{metalk_saved_pct:.0f}% over stage 2)")
    print(f"{'=' * 50}")
    print(f"  Density:           {density_score:.0f}/25")
    print(f"  Compressibility:   {compression_score:.0f}/25")
    print(f"  Structure:         {structure_score:.0f}/25")
    print(f"  Conciseness:       {conciseness_score:.0f}/25")
    print(f"{'=' * 50}")
    print(f"  SCORE: {total:.0f}/100  Grade: {grade}")


# ── diff ─────────────────────────────────────────────────────────────

def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two .crumb files and show what changed."""
    text_a = Path(args.file_a).read_text(encoding='utf-8')
    text_b = Path(args.file_b).read_text(encoding='utf-8')
    parsed_a = parse_crumb(text_a)
    parsed_b = parse_crumb(text_b)

    headers_a = parsed_a['headers']
    headers_b = parsed_b['headers']
    sections_a = parsed_a['sections']
    sections_b = parsed_b['sections']

    changes = 0

    # Compare headers
    all_keys = sorted(set(list(headers_a.keys()) + list(headers_b.keys())))
    header_diffs = []
    for key in all_keys:
        val_a = headers_a.get(key)
        val_b = headers_b.get(key)
        if val_a != val_b:
            if val_a is None:
                header_diffs.append(f"  + {key}={val_b}")
            elif val_b is None:
                header_diffs.append(f"  - {key}={val_a}")
            else:
                header_diffs.append(f"  - {key}={val_a}")
                header_diffs.append(f"  + {key}={val_b}")
            changes += 1

    if header_diffs:
        print("Headers:")
        for line in header_diffs:
            print(line)
        print()

    # Compare sections
    all_sections = sorted(set(list(sections_a.keys()) + list(sections_b.keys())))
    for section in all_sections:
        entries_a = set(normalize_entry(l) for l in sections_a.get(section, []) if l.strip())
        entries_b = set(normalize_entry(l) for l in sections_b.get(section, []) if l.strip())
        raw_a = {normalize_entry(l): l.strip() for l in sections_a.get(section, []) if l.strip()}
        raw_b = {normalize_entry(l): l.strip() for l in sections_b.get(section, []) if l.strip()}

        added = entries_b - entries_a
        removed = entries_a - entries_b

        if not added and not removed:
            continue

        changes += len(added) + len(removed)
        print(f"[{section}]:")
        for norm in sorted(removed):
            print(f"  - {raw_a[norm]}")
        for norm in sorted(added):
            print(f"  + {raw_b[norm]}")
        print()

    if changes == 0:
        print("No differences found.")
    else:
        print(f"{changes} change(s) total.")


# ── init ─────────────────────────────────────────────────────────────

SENDER_INSTRUCTION = dedent("""\
    When I say "crumb it", generate a CRUMB summarizing the current state.

    For tasks and handoffs, use kind=task:
      BEGIN CRUMB
      v=1.1
      kind=task
      title=<short description>
      source=<this tool>
      url=https://github.com/XioAISolutions/crumb-format
      ---
      [goal]       <what needs to happen next>
      [context]    <key facts, decisions, current state>
      [constraints] <what must not change>
      END CRUMB

    For preferences and memory, use kind=mem:
      BEGIN CRUMB
      v=1.1
      kind=mem
      title=<topic>
      source=<this tool>
      ---
      [consolidated] <durable facts, preferences, decisions>
      END CRUMB

    For repo/project overviews, use kind=map:
      BEGIN CRUMB
      v=1.1
      kind=map
      title=<project name>
      source=<this tool>
      ---
      [project]  <one-line description>
      [modules]  <key files and directories>
      END CRUMB
""")

RECEIVER_INSTRUCTION = dedent("""\
    When you receive text containing BEGIN CRUMB / END CRUMB markers, treat it
    as a structured handoff. Parse the headers (kind, title, source) and sections
    ([goal], [context], [constraints], [consolidated], [project], [modules]).
    Act on the crumb directly — do not ask the user to re-explain what is already
    in the crumb. If the crumb is kind=task, start working on the goal. If
    kind=mem, use the consolidated facts as context. If kind=map, use it to
    understand the codebase.
""")

CLAUDE_MD_SNIPPET = dedent("""\
    ## CRUMB handoffs

    This project uses [CRUMB](https://github.com/XioAISolutions/crumb-format)
    for AI handoffs. When switching between AI tools:

    - Say "crumb it" to generate a structured handoff
    - Paste a received crumb to continue work from another AI
    - Store .crumb files in the `crumbs/` directory

    When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it
    directly. When asked to "crumb it", generate a crumb summarizing the current
    goal, context, and constraints.
""")

CURSORRULES_SNIPPET = dedent("""\
    # CRUMB handoffs
    # This project uses CRUMB (https://github.com/XioAISolutions/crumb-format)
    # for structured AI handoffs.
    #
    # When the user says "crumb it", generate a CRUMB block summarizing the
    # current task state (goal, context, constraints).
    #
    # When you receive a BEGIN CRUMB / END CRUMB block, parse it as a structured
    # handoff and act on it directly without asking the user to re-explain.
    #
    # CRUMB format:
    #   BEGIN CRUMB
    #   v=1.1
    #   kind=task|mem|map|log|todo
    #   title=<short description>
    #   source=cursor.agent
    #   url=https://github.com/XioAISolutions/crumb-format
    #   ---
    #   [goal]       <what needs to happen next>
    #   [context]    <key facts, decisions, current state>
    #   [constraints] <what must not change>
    #   END CRUMB
""")

WINDSURF_RULES_SNIPPET = dedent("""\
    # CRUMB handoffs
    # This project uses CRUMB (https://github.com/XioAISolutions/crumb-format)
    # for structured AI handoffs between tools.
    #
    # "crumb it" = generate a CRUMB block summarizing current work state.
    # BEGIN CRUMB / END CRUMB blocks = structured handoffs, act on them directly.
    #
    # Kinds: task (goal/context/constraints), mem (consolidated preferences),
    #        map (project/modules), log (timestamped entries), todo (checkbox tasks)
    # Always set source=windsurf.agent and include the url header.
""")

CHATGPT_RULES_SNIPPET = dedent("""\
    ## CRUMB handoffs

    This conversation may involve CRUMB handoffs
    (https://github.com/XioAISolutions/crumb-format).

    When the user says "crumb it", generate a structured handoff:

    BEGIN CRUMB
    v=1.1
    kind=task
    title=<short description>
    source=chatgpt
    url=https://github.com/XioAISolutions/crumb-format
    ---
    [goal]
    <what needs to happen next>

    [context]
    <key facts, decisions, current state>

    [constraints]
    <what must not change>
    END CRUMB

    When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it
    directly — do not ask the user to re-explain what is already in the crumb.

    Other kinds: mem (preferences), map (codebase), log (transcript), todo (tasks).
""")

GEMINI_SNIPPET = dedent("""\
    {
      "crumb_handoffs": {
        "description": "This project uses CRUMB (https://github.com/XioAISolutions/crumb-format) for structured AI handoffs.",
        "instructions": [
          "When the user says 'crumb it', generate a CRUMB block summarizing the current task state.",
          "When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it directly.",
          "Always set source=gemini.agent and include the url header.",
          "CRUMB kinds: task (goal/context/constraints), mem (consolidated preferences), map (project/modules), log (entries), todo (tasks).",
          "AgentAuth: If a kind=passport crumb is present, respect its [identity] and [permissions] sections to enforce agent authorization boundaries."
        ],
        "format_example": "BEGIN CRUMB\\nv=1.1\\nkind=task\\ntitle=<short description>\\nsource=gemini.agent\\nurl=https://github.com/XioAISolutions/crumb-format\\n---\\n[goal]\\n<what needs to happen>\\n[context]\\n<key facts>\\n[constraints]\\n<what must not change>\\nEND CRUMB"
      }
    }
""")

COPILOT_SNIPPET = dedent("""\
    ## CRUMB handoffs

    This project uses [CRUMB](https://github.com/XioAISolutions/crumb-format)
    for structured AI handoffs between tools.

    - When the user says "crumb it", generate a structured CRUMB handoff block
      summarizing the current goal, context, and constraints.
    - When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it
      directly without asking the user to re-explain.
    - Always set `source=copilot.agent` and include the `url` header.
    - CRUMB kinds: task, mem, map, log, todo.
    - AgentAuth passport: If a kind=passport crumb exists, respect its [identity]
      and [permissions] sections to enforce agent authorization boundaries.

    Format:

        BEGIN CRUMB
        v=1.1
        kind=task
        title=<short description>
        source=copilot.agent
        url=https://github.com/XioAISolutions/crumb-format
        ---
        [goal]
        <what needs to happen next>
        [context]
        <key facts, decisions, current state>
        [constraints]
        <what must not change>
        END CRUMB
""")

CODY_SNIPPET = dedent("""\
    {
      "customInstructions": {
        "crumb_handoffs": "This project uses CRUMB (https://github.com/XioAISolutions/crumb-format) for structured AI handoffs. When the user says 'crumb it', generate a CRUMB block: BEGIN CRUMB / v=1.1 / kind=task / title=<desc> / source=cody.agent / url=https://github.com/XioAISolutions/crumb-format / --- / [goal] / [context] / [constraints] / END CRUMB. When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it directly. Kinds: task, mem, map, log, todo. AgentAuth: respect kind=passport crumbs with [identity] and [permissions] for agent authorization."
      }
    }
""")

CONTINUE_DEV_SNIPPET = dedent("""\
    {
      "systemMessage": "This project uses CRUMB (https://github.com/XioAISolutions/crumb-format) for structured AI handoffs. When the user says 'crumb it', generate a CRUMB block summarizing current work: BEGIN CRUMB, v=1.1, kind=task, title=<desc>, source=continue.agent, url=https://github.com/XioAISolutions/crumb-format, ---, [goal], [context], [constraints], END CRUMB. When you receive a BEGIN CRUMB / END CRUMB block, parse and act on it directly. Other kinds: mem (preferences), map (codebase), log (transcript), todo (tasks). AgentAuth: if a kind=passport crumb is present, respect its [identity] and [permissions] sections to enforce agent authorization boundaries."
    }
""")

AIDER_SNIPPET = dedent("""\
    # CRUMB handoffs
    # This project uses CRUMB (https://github.com/XioAISolutions/crumb-format)
    # for structured AI handoffs between tools.
    #
    # Conventions:
    # - When the user says "crumb it", generate a CRUMB block summarizing
    #   current task state (goal, context, constraints).
    # - When you receive a BEGIN CRUMB / END CRUMB block, parse it as a
    #   structured handoff and act on it directly.
    # - Always set source=aider.agent and include the url header.
    # - CRUMB kinds: task, mem, map, log, todo.
    # - AgentAuth: respect kind=passport crumbs with [identity] and
    #   [permissions] for agent authorization boundaries.
    #
    # CRUMB format:
    #   BEGIN CRUMB
    #   v=1.1
    #   kind=task
    #   title=<short description>
    #   source=aider.agent
    #   url=https://github.com/XioAISolutions/crumb-format
    #   ---
    #   [goal]       <what needs to happen next>
    #   [context]    <key facts, decisions, current state>
    #   [constraints] <what must not change>
    #   END CRUMB
""")

REPLIT_SNIPPET = dedent("""\
    # CRUMB handoffs
    # This project uses CRUMB (https://github.com/XioAISolutions/crumb-format)
    # for structured AI handoffs between AI tools.
    #
    # When the user says "crumb it", generate a CRUMB block:
    #   BEGIN CRUMB
    #   v=1.1
    #   kind=task
    #   title=<short description>
    #   source=replit.agent
    #   url=https://github.com/XioAISolutions/crumb-format
    #   ---
    #   [goal]       <what needs to happen next>
    #   [context]    <key facts, decisions, current state>
    #   [constraints] <what must not change>
    #   END CRUMB
    #
    # When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on
    # it directly without asking the user to re-explain.
    # Kinds: task, mem, map, log, todo.
    # AgentAuth: respect kind=passport crumbs with [identity] and [permissions].
""")

DEVIN_SNIPPET = dedent("""\
    ## CRUMB handoffs

    This project uses [CRUMB](https://github.com/XioAISolutions/crumb-format)
    for structured AI handoffs.

    - When asked to "crumb it", generate a CRUMB block summarizing the current
      goal, context, and constraints. Always set `source=devin.agent`.
    - When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it
      directly — do not ask the user to re-explain what is in the crumb.
    - Store .crumb files in the `crumbs/` directory.
    - CRUMB kinds: task (goal/context/constraints), mem (consolidated preferences),
      map (project/modules), log (timestamped entries), todo (checkbox tasks).
    - AgentAuth passport: If a kind=passport crumb exists, respect its [identity]
      and [permissions] sections to enforce agent authorization boundaries.

    Format:

        BEGIN CRUMB
        v=1.1
        kind=task
        title=<short description>
        source=devin.agent
        url=https://github.com/XioAISolutions/crumb-format
        ---
        [goal]
        <what needs to happen next>
        [context]
        <key facts, decisions, current state>
        [constraints]
        <what must not change>
        END CRUMB
""")

BOLT_SNIPPET = dedent("""\
    {
      "instructions": {
        "crumb_handoffs": {
          "description": "This project uses CRUMB (https://github.com/XioAISolutions/crumb-format) for structured AI handoffs.",
          "on_crumb_it": "Generate a CRUMB block: BEGIN CRUMB / v=1.1 / kind=task / title=<desc> / source=bolt.agent / url=https://github.com/XioAISolutions/crumb-format / --- / [goal] / [context] / [constraints] / END CRUMB",
          "on_receive": "When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it directly.",
          "kinds": "task (goal/context/constraints), mem (preferences), map (project/modules), log (entries), todo (tasks)",
          "agent_auth": "Respect kind=passport crumbs with [identity] and [permissions] for agent authorization."
        }
      }
    }
""")

LOVABLE_SNIPPET = dedent("""\
    {
      "instructions": {
        "crumb_handoffs": {
          "description": "This project uses CRUMB (https://github.com/XioAISolutions/crumb-format) for structured AI handoffs.",
          "on_crumb_it": "Generate a CRUMB block: BEGIN CRUMB / v=1.1 / kind=task / title=<desc> / source=lovable.agent / url=https://github.com/XioAISolutions/crumb-format / --- / [goal] / [context] / [constraints] / END CRUMB",
          "on_receive": "When you receive a BEGIN CRUMB / END CRUMB block, parse it and act on it directly.",
          "kinds": "task (goal/context/constraints), mem (preferences), map (project/modules), log (entries), todo (tasks)",
          "agent_auth": "Respect kind=passport crumbs with [identity] and [permissions] for agent authorization."
        }
      }
    }
""")


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize CRUMB in a project: generate map crumb and integration snippets."""
    project_dir = Path(args.dir)
    crumbs_dir = project_dir / 'crumbs'
    crumbs_dir.mkdir(exist_ok=True)

    # Detect project name from directory
    project_name = args.project or project_dir.resolve().name

    # Generate map crumb from directory structure
    modules = []
    for item in sorted(project_dir.iterdir()):
        if item.name.startswith('.') or item.name == 'crumbs' or item.name == '__pycache__' or item.name == 'node_modules':
            continue
        if item.is_dir():
            modules.append(f"- {item.name}/")
        elif item.is_file() and item.suffix in ('.py', '.js', '.ts', '.go', '.rs', '.md', '.toml', '.json', '.yaml', '.yml'):
            modules.append(f"- {item.name}")

    if not modules:
        modules = ['- <add key files and directories>']

    map_headers = {
        'v': '1.1',
        'kind': 'map',
        'title': f'{project_name} project map',
        'source': 'crumb.init',
        'project': project_name,
        'url': 'https://github.com/XioAISolutions/crumb-format',
    }
    map_sections = {
        'project': [args.description or '<one-line project description>', ''],
        'modules': modules + [''],
    }
    map_crumb = render_crumb(map_headers, map_sections)

    map_path = crumbs_dir / 'map.crumb'
    map_path.write_text(map_crumb, encoding='utf-8')
    print(f"  Created {map_path}")

    # Write integration snippets
    def _write_or_append(filepath, snippet, marker='CRUMB'):
        if filepath.exists():
            existing = filepath.read_text(encoding='utf-8')
            if marker not in existing:
                filepath.write_text(existing.rstrip() + '\n\n' + snippet, encoding='utf-8')
                print(f"  Appended CRUMB section to {filepath}")
            else:
                print(f"  Skipped {filepath} (already has CRUMB section)")
        else:
            filepath.write_text(snippet, encoding='utf-8')
            print(f"  Created {filepath}")

    if args.claude_md:
        _write_or_append(project_dir / 'CLAUDE.md', CLAUDE_MD_SNIPPET)

    if args.cursor_rules:
        cursor_dir = project_dir / '.cursor'
        cursor_dir.mkdir(exist_ok=True)
        _write_or_append(cursor_dir / 'rules', CURSORRULES_SNIPPET)

    if args.windsurf_rules:
        _write_or_append(project_dir / '.windsurfrules', WINDSURF_RULES_SNIPPET)

    if args.chatgpt_rules:
        print(f"\n--- ChatGPT Custom Instructions (paste into Settings > Custom Instructions) ---\n")
        print(CHATGPT_RULES_SNIPPET)

    if args.gemini:
        gemini_dir = project_dir / '.gemini'
        gemini_dir.mkdir(exist_ok=True)
        _write_or_append(gemini_dir / 'settings.json', GEMINI_SNIPPET)

    if args.copilot:
        github_dir = project_dir / '.github'
        github_dir.mkdir(exist_ok=True)
        _write_or_append(github_dir / 'copilot-instructions.md', COPILOT_SNIPPET)

    if args.cody:
        sg_dir = project_dir / '.sourcegraph'
        sg_dir.mkdir(exist_ok=True)
        _write_or_append(sg_dir / 'cody.json', CODY_SNIPPET)

    if args.continue_dev:
        cont_dir = project_dir / '.continue'
        cont_dir.mkdir(exist_ok=True)
        _write_or_append(cont_dir / 'config.json', CONTINUE_DEV_SNIPPET)

    if args.aider:
        _write_or_append(project_dir / '.aider.conf.yml', AIDER_SNIPPET)

    if args.replit:
        _write_or_append(project_dir / '.replit', REPLIT_SNIPPET)

    if args.devin:
        _write_or_append(project_dir / 'devin.md', DEVIN_SNIPPET)

    if args.bolt:
        bolt_dir = project_dir / '.bolt'
        bolt_dir.mkdir(exist_ok=True)
        _write_or_append(bolt_dir / 'config.json', BOLT_SNIPPET)

    if args.lovable:
        lovable_dir = project_dir / '.lovable'
        lovable_dir.mkdir(exist_ok=True)
        _write_or_append(lovable_dir / 'config.json', LOVABLE_SNIPPET)

    if args.all_rules:
        _write_or_append(project_dir / 'CLAUDE.md', CLAUDE_MD_SNIPPET)
        cursor_dir = project_dir / '.cursor'
        cursor_dir.mkdir(exist_ok=True)
        _write_or_append(cursor_dir / 'rules', CURSORRULES_SNIPPET)
        _write_or_append(project_dir / '.windsurfrules', WINDSURF_RULES_SNIPPET)
        print(f"\n--- ChatGPT Custom Instructions (paste into Settings > Custom Instructions) ---\n")
        print(CHATGPT_RULES_SNIPPET)
        gemini_dir = project_dir / '.gemini'
        gemini_dir.mkdir(exist_ok=True)
        _write_or_append(gemini_dir / 'settings.json', GEMINI_SNIPPET)
        github_dir = project_dir / '.github'
        github_dir.mkdir(exist_ok=True)
        _write_or_append(github_dir / 'copilot-instructions.md', COPILOT_SNIPPET)
        sg_dir = project_dir / '.sourcegraph'
        sg_dir.mkdir(exist_ok=True)
        _write_or_append(sg_dir / 'cody.json', CODY_SNIPPET)
        cont_dir = project_dir / '.continue'
        cont_dir.mkdir(exist_ok=True)
        _write_or_append(cont_dir / 'config.json', CONTINUE_DEV_SNIPPET)
        _write_or_append(project_dir / '.aider.conf.yml', AIDER_SNIPPET)
        _write_or_append(project_dir / '.replit', REPLIT_SNIPPET)
        _write_or_append(project_dir / 'devin.md', DEVIN_SNIPPET)
        bolt_dir = project_dir / '.bolt'
        bolt_dir.mkdir(exist_ok=True)
        _write_or_append(bolt_dir / 'config.json', BOLT_SNIPPET)
        lovable_dir = project_dir / '.lovable'
        lovable_dir.mkdir(exist_ok=True)
        _write_or_append(lovable_dir / 'config.json', LOVABLE_SNIPPET)

    # Print custom instruction snippets (unless --all already printed)
    if not args.all_rules:
        print(f"\n--- Sender instruction (add to your AI's custom instructions) ---\n")
        print(SENDER_INSTRUCTION)
        print(f"--- Receiver instruction (add to the receiving AI) ---\n")
        print(RECEIVER_INSTRUCTION)

    print(f"Done. CRUMB initialized in {project_dir}")
    print(f"  - Map crumb: {map_path}")
    all_flags = [args.claude_md, args.cursor_rules, args.windsurf_rules, args.chatgpt_rules,
                 args.gemini, args.copilot, args.cody, args.continue_dev, args.aider,
                 args.replit, args.devin, args.bolt, args.lovable, args.all_rules]
    if not any(all_flags):
        print(f"  Tip: run with --all to seed all AI tools at once")


# ── log (append-only session transcript) ────────────────────────────

def cmd_log(args: argparse.Namespace) -> None:
    """Append timestamped entries to a log crumb. Creates the file if it doesn't exist."""
    path = Path(args.file)

    if path.exists():
        text = path.read_text(encoding='utf-8')
        parsed = parse_crumb(text)
        if parsed['headers']['kind'] != 'log':
            print(f"Error: {args.file} is kind={parsed['headers']['kind']}, expected kind=log", file=sys.stderr)
            sys.exit(1)
    else:
        # Create a new log crumb
        title = args.title or path.stem
        parsed = {
            'headers': {'v': '1.1', 'kind': 'log', 'title': title, 'source': args.source or 'cli'},
            'sections': {'entries': ['']},
        }

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for entry in args.entries:
        parsed['sections']['entries'].append(f"- [{timestamp}] {entry}")
    parsed['sections']['entries'].append('')

    output = render_crumb(parsed['headers'], parsed['sections'])
    path.write_text(output, encoding='utf-8')
    print(f"Logged {len(args.entries)} entries to {args.file}")


# ── todo (foresight memory) ─────────────────────────────────────────

def cmd_todo_add(args: argparse.Namespace) -> None:
    """Add tasks to a todo crumb. Creates the file if it doesn't exist."""
    path = Path(args.file)

    if path.exists():
        text = path.read_text(encoding='utf-8')
        parsed = parse_crumb(text)
        if parsed['headers']['kind'] != 'todo':
            print(f"Error: {args.file} is kind={parsed['headers']['kind']}, expected kind=todo", file=sys.stderr)
            sys.exit(1)
    else:
        title = args.title or path.stem
        parsed = {
            'headers': {'v': '1.1', 'kind': 'todo', 'title': title, 'source': args.source or 'cli'},
            'sections': {'tasks': ['']},
        }

    for task in args.tasks:
        parsed['sections']['tasks'].append(f"- [ ] {task}")
    parsed['sections']['tasks'].append('')

    output = render_crumb(parsed['headers'], parsed['sections'])
    path.write_text(output, encoding='utf-8')
    print(f"Added {len(args.tasks)} tasks to {args.file}")


def cmd_todo_done(args: argparse.Namespace) -> None:
    """Mark tasks as done in a todo crumb by substring match."""
    path = Path(args.file)
    text = path.read_text(encoding='utf-8')
    parsed = parse_crumb(text)

    if parsed['headers']['kind'] != 'todo':
        print(f"Error: {args.file} is kind={parsed['headers']['kind']}, expected kind=todo", file=sys.stderr)
        sys.exit(1)

    query = args.query.lower()
    completed = 0
    new_tasks = []
    for line in parsed['sections']['tasks']:
        if line.strip().startswith('- [ ]') and query in line.lower():
            new_tasks.append(line.replace('- [ ]', '- [x]', 1))
            completed += 1
        else:
            new_tasks.append(line)

    if completed == 0:
        print(f"No open tasks matching '{args.query}' found.")
        return

    parsed['sections']['tasks'] = new_tasks
    output = render_crumb(parsed['headers'], parsed['sections'])
    path.write_text(output, encoding='utf-8')
    print(f"Completed {completed} task(s) matching '{args.query}'")


def cmd_todo_list(args: argparse.Namespace) -> None:
    """List tasks from a todo crumb, optionally filtering by status."""
    text = read_text(args.file)
    parsed = parse_crumb(text)

    if parsed['headers']['kind'] != 'todo':
        print(f"Error: not a todo crumb (kind={parsed['headers']['kind']})", file=sys.stderr)
        sys.exit(1)

    show_all = args.show_all
    title = parsed['headers'].get('title', '')
    if title:
        print(f"Todo: {title}")

    open_count = 0
    done_count = 0
    for line in parsed['sections']['tasks']:
        stripped = line.strip()
        if stripped.startswith('- [x]'):
            done_count += 1
            if show_all:
                print(f"  {stripped}")
        elif stripped.startswith('- [ ]'):
            open_count += 1
            print(f"  {stripped}")

    print(f"\n  {open_count} open, {done_count} done")


def cmd_todo_dream(args: argparse.Namespace) -> None:
    """Archive completed tasks from [tasks] to [archived], keeping the todo crumb clean."""
    path = Path(args.file)
    text = path.read_text(encoding='utf-8')
    parsed = parse_crumb(text)

    if parsed['headers']['kind'] != 'todo':
        print(f"Error: {args.file} is kind={parsed['headers']['kind']}, expected kind=todo", file=sys.stderr)
        sys.exit(1)

    open_tasks = []
    archived = []
    for line in parsed['sections']['tasks']:
        stripped = line.strip()
        if stripped.startswith('- [x]'):
            archived.append(stripped)
        else:
            open_tasks.append(line)

    if not archived:
        print("No completed tasks to archive.")
        return

    parsed['sections']['tasks'] = open_tasks if any(l.strip() for l in open_tasks) else ['']
    if 'archived' not in parsed['sections']:
        parsed['sections']['archived'] = ['']
    parsed['sections']['archived'] = archived + parsed['sections']['archived']

    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    parsed['headers']['dream_pass'] = now

    output = render_crumb(parsed['headers'], parsed['sections'])
    path.write_text(output, encoding='utf-8')
    print(f"Archived {len(archived)} completed tasks from {args.file}")


# ── watch ───────────────────────────────────────────────────────────

def cmd_watch(args: argparse.Namespace) -> None:
    """Watch a .crumb file or directory and auto-dream when raw entries exceed threshold."""
    import time

    target = Path(args.target)
    threshold = args.threshold
    interval = args.interval

    if target.is_file():
        files_to_watch = [target]
    elif target.is_dir():
        files_to_watch = sorted(target.rglob('*.crumb'))
    else:
        print(f"Error: {args.target} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Watching {len(files_to_watch)} file(s), auto-dream threshold={threshold} raw entries, interval={interval}s")
    print("Press Ctrl+C to stop.\n")

    # Track file modification times
    mtimes = {}
    for f in files_to_watch:
        mtimes[f] = f.stat().st_mtime if f.exists() else 0

    try:
        while True:
            # Re-scan directory if watching a dir
            if target.is_dir():
                files_to_watch = sorted(target.rglob('*.crumb'))

            for f in files_to_watch:
                if not f.exists():
                    continue
                current_mtime = f.stat().st_mtime
                if current_mtime == mtimes.get(f, 0):
                    continue

                mtimes[f] = current_mtime

                try:
                    text = f.read_text(encoding='utf-8')
                    parsed = parse_crumb(text)
                except (ValueError, Exception):
                    continue

                kind = parsed['headers']['kind']

                # Auto-dream for mem crumbs with enough raw entries
                if kind == 'mem' and 'raw' in parsed['sections']:
                    raw_entries = [l for l in parsed['sections']['raw'] if l.strip()]
                    if len(raw_entries) >= threshold:
                        print(f"[auto-dream] {f.name}: {len(raw_entries)} raw entries (threshold={threshold})")

                        # Create a namespace that looks like dream args
                        class DreamArgs:
                            file = str(f)
                            dry_run = False
                        cmd_dream(DreamArgs())

                # Run on-change hook
                run_hook('on_change', {'file': str(f), 'kind': kind})

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


# ── export ──────────────────────────────────────────────────────────

def crumb_to_json(parsed: Dict) -> str:
    """Convert a parsed crumb to JSON."""
    obj = {
        'headers': parsed['headers'],
        'sections': {name: [l for l in lines if l.strip()] for name, lines in parsed['sections'].items()},
    }
    return json.dumps(obj, indent=2)


def crumb_to_markdown(parsed: Dict) -> str:
    """Convert a parsed crumb to markdown."""
    headers = parsed['headers']
    sections = parsed['sections']
    kind = headers['kind']
    title = headers.get('title', f'{kind} crumb')
    lines = [f"# {title}", ""]

    # Metadata table
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    for key, value in headers.items():
        if key != 'title':
            lines.append(f"| {key} | {value} |")
    lines.append("")

    for name, body in sections.items():
        content_lines = [l for l in body if l.strip()]
        if not content_lines:
            continue
        lines.append(f"## {name.capitalize()}")
        lines.append("")
        for line in content_lines:
            lines.append(line)
        lines.append("")

    return "\n".join(lines)


def crumb_to_clipboard(parsed: Dict) -> str:
    """Format a crumb for pasting into AI chat windows."""
    headers = parsed['headers']
    sections = parsed['sections']
    kind = headers['kind']
    title = headers.get('title', '')

    lines = [f"[CRUMB handoff — {kind}: {title}]", ""]

    if kind == 'task':
        goal_lines = [l.strip() for l in sections.get('goal', []) if l.strip()]
        lines.append(f"Goal: {' '.join(goal_lines)}")
        lines.append("")
        ctx = [l.strip() for l in sections.get('context', []) if l.strip()]
        if ctx:
            lines.append("Context:")
            lines.extend(f"  {l}" for l in ctx)
            lines.append("")
        constraints = [l.strip() for l in sections.get('constraints', []) if l.strip()]
        if constraints:
            lines.append("Constraints:")
            lines.extend(f"  {l}" for l in constraints)
    elif kind == 'mem':
        entries = [l.strip() for l in sections.get('consolidated', []) if l.strip()]
        lines.append("Known facts:")
        lines.extend(f"  {l}" for l in entries)
    elif kind == 'map':
        desc = [l.strip() for l in sections.get('project', []) if l.strip()]
        if desc:
            lines.append(f"Project: {' '.join(desc)}")
            lines.append("")
        mods = [l.strip() for l in sections.get('modules', []) if l.strip()]
        if mods:
            lines.append("Key modules:")
            lines.extend(f"  {l}" for l in mods)
    elif kind == 'log':
        entries = [l.strip() for l in sections.get('entries', []) if l.strip()]
        lines.append("Session log:")
        lines.extend(f"  {l}" for l in entries)
    elif kind == 'todo':
        tasks = [l.strip() for l in sections.get('tasks', []) if l.strip()]
        lines.append("Tasks:")
        lines.extend(f"  {l}" for l in tasks)

    lines.append("")
    lines.append(f"(Generated by CRUMB — https://github.com/XioAISolutions/crumb-format)")
    return "\n".join(lines)


def cmd_export(args: argparse.Namespace) -> None:
    """Export a .crumb file to another format."""
    text = read_text(args.file)
    parsed = parse_crumb(text)

    fmt = args.format
    if fmt == 'json':
        output = crumb_to_json(parsed)
    elif fmt == 'markdown':
        output = crumb_to_markdown(parsed)
    elif fmt == 'clipboard':
        output = crumb_to_clipboard(parsed)
    else:
        print(f"Error: unknown format '{fmt}'", file=sys.stderr)
        sys.exit(1)

    write_text(args.output, output)
    if args.output != '-':
        print(f"Exported {args.file} as {fmt} → {args.output}")


# ── import ──────────────────────────────────────────────────────────

def cmd_import(args: argparse.Namespace) -> None:
    """Import from JSON or markdown into a .crumb file."""
    text = read_text(args.input)
    fmt = getattr(args, 'from')

    if fmt == 'json':
        obj = json.loads(text)
        headers = obj.get('headers', {})
        sections_raw = obj.get('sections', {})
        # Ensure required headers
        if 'v' not in headers:
            headers['v'] = '1.1'
        if 'kind' not in headers:
            print("Error: JSON must have headers.kind", file=sys.stderr)
            sys.exit(1)
        if 'source' not in headers:
            headers['source'] = 'import.json'
        # Convert section lists to proper format
        sections = {}
        for name, entries in sections_raw.items():
            if isinstance(entries, list):
                sections[name] = entries + ['']
            else:
                sections[name] = [str(entries), '']
        output = render_crumb(headers, sections)
    elif fmt == 'markdown':
        # Parse markdown with # title, ## sections, | table | headers |
        lines = text.splitlines()
        title = ''
        headers = {'v': '1.1', 'source': 'import.markdown'}
        sections = {}
        current_section = None

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('# ') and not stripped.startswith('## '):
                title = stripped[2:].strip()
                headers['title'] = title
            elif stripped.startswith('## '):
                current_section = stripped[3:].strip().lower()
                sections[current_section] = []
            elif stripped.startswith('|') and '|' in stripped[1:]:
                # Table row — extract header metadata
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                if len(cells) == 2 and cells[0] and not cells[0].startswith('-'):
                    key, val = cells[0].lower(), cells[1]
                    if key in ('kind', 'v', 'source', 'project', 'env', 'tags', 'url',
                               'dream_pass', 'dream_sessions', 'max_index_tokens', 'max_total_tokens'):
                        headers[key] = val
            elif current_section is not None and stripped:
                sections[current_section].append(line)

        if 'kind' not in headers:
            print("Error: markdown must have a 'kind' field in the metadata table", file=sys.stderr)
            sys.exit(1)

        # Add trailing blank lines
        for name in sections:
            sections[name].append('')

        output = render_crumb(headers, sections)
    else:
        print(f"Error: unknown format '{fmt}'", file=sys.stderr)
        sys.exit(1)

    # Validate the result
    try:
        parse_crumb(output)
    except ValueError as exc:
        print(f"Warning: imported crumb has validation issues: {exc}", file=sys.stderr)

    write_text(args.output, output)
    if args.output != '-':
        print(f"Imported {fmt} → {args.output}")


# ── hooks (.crumbrc) ───────────────────────────────────────────────

def load_hooks(project_dir: str = '.') -> Dict[str, str]:
    """Load hooks from .crumbrc file. Returns {hook_name: shell_command}."""
    rc_path = Path(project_dir) / '.crumbrc'
    if not rc_path.exists():
        return {}

    hooks = {}
    in_hooks = False
    for line in rc_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped == '[hooks]':
            in_hooks = True
            continue
        if stripped.startswith('[') and stripped.endswith(']'):
            in_hooks = False
            continue
        if in_hooks and '=' in stripped and not stripped.startswith('#'):
            key, val = stripped.split('=', 1)
            hooks[key.strip()] = val.strip()
    return hooks


def run_hook(hook_name: str, context: Dict[str, str] = None) -> bool:
    """Run a hook if defined. Returns True if hook ran successfully or wasn't defined."""
    hooks = load_hooks()
    if hook_name not in hooks:
        return True

    cmd = hooks[hook_name]
    env = os.environ.copy()
    if context:
        for key, val in context.items():
            env[f"CRUMB_{key.upper()}"] = str(val)

    try:
        result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Hook '{hook_name}' failed (exit {result.returncode})", file=sys.stderr)
            if result.stderr:
                print(f"  {result.stderr.strip()}", file=sys.stderr)
            return False
        if result.stdout.strip():
            print(f"[hook:{hook_name}] {result.stdout.strip()}")
        return True
    except subprocess.TimeoutExpired:
        print(f"Hook '{hook_name}' timed out after 30s", file=sys.stderr)
        return False


def cmd_hooks(args: argparse.Namespace) -> None:
    """Show configured hooks from .crumbrc."""
    hooks = load_hooks(args.dir)
    if not hooks:
        print("No hooks configured.")
        print(f"\nCreate a .crumbrc file with a [hooks] section:")
        print(f"  [hooks]")
        print(f"  post_dream = git add *.crumb && git commit -m 'dream pass'")
        print(f"  post_append = echo 'Entry added'")
        print(f"  pre_validate = python scripts/lint.py")
        return

    print("Configured hooks:")
    for name, cmd in hooks.items():
        print(f"  {name} = {cmd}")


# ── template registry ──────────────────────────────────────────────

TEMPLATE_DIR = Path.home() / '.crumb' / 'templates'

BUILTIN_TEMPLATES = {
    'bug-fix': {
        'kind': 'task',
        'description': 'Hand off a bug fix between AI tools',
        'content': dedent("""\
            BEGIN CRUMB
            v=1.1
            kind=task
            title=Fix: <bug description>
            source=<your-tool>
            ---
            [goal]
            Fix the bug described below and verify the fix with tests.

            [context]
            - Bug: <what's broken>
            - Expected: <what should happen>
            - Actual: <what happens instead>
            - Reproduction steps: <how to trigger>
            - Relevant files: <paths>

            [constraints]
            - Don't break existing tests
            - Keep the fix minimal — no refactoring
            END CRUMB
        """),
    },
    'feature': {
        'kind': 'task',
        'description': 'Hand off a feature implementation',
        'content': dedent("""\
            BEGIN CRUMB
            v=1.1
            kind=task
            title=Feature: <feature name>
            source=<your-tool>
            ---
            [goal]
            Implement the feature described below.

            [context]
            - Feature: <what to build>
            - Acceptance criteria: <when is it done>
            - Related code: <paths and patterns>
            - Prior decisions: <anything already decided>

            [constraints]
            - Follow existing code style
            - Add tests for new functionality
            END CRUMB
        """),
    },
    'code-review': {
        'kind': 'task',
        'description': 'Hand off code for review',
        'content': dedent("""\
            BEGIN CRUMB
            v=1.1
            kind=task
            title=Review: <what to review>
            source=<your-tool>
            ---
            [goal]
            Review the following changes for correctness, security, and style.

            [context]
            - Changed files: <paths>
            - What changed: <summary of changes>
            - Why: <motivation for the change>

            [constraints]
            - Focus on bugs and security issues first
            - Style nits are low priority
            END CRUMB
        """),
    },
    'onboarding': {
        'kind': 'map',
        'description': 'Onboard an AI to a codebase',
        'content': dedent("""\
            BEGIN CRUMB
            v=1.1
            kind=map
            title=<project> onboarding
            source=<your-tool>
            project=<project-name>
            ---
            [project]
            <one-line project description>

            [modules]
            - src/         — source code
            - tests/       — test suite
            - docs/        — documentation

            [invariants]
            - <rule that must always hold>
            - <another invariant>
            END CRUMB
        """),
    },
    'preferences': {
        'kind': 'mem',
        'description': 'Store user or team preferences',
        'content': dedent("""\
            BEGIN CRUMB
            v=1.1
            kind=mem
            title=<your name> preferences
            source=<your-tool>
            ---
            [consolidated]
            - Preferred language: <lang>
            - Code style: <style preferences>
            - Testing approach: <how you test>
            - Communication style: <terse/verbose>
            END CRUMB
        """),
    },
    'migration': {
        'kind': 'task',
        'description': 'Hand off a migration or upgrade task',
        'content': dedent("""\
            BEGIN CRUMB
            v=1.1
            kind=task
            title=Migrate: <what to migrate>
            source=<your-tool>
            ---
            [goal]
            Complete the migration described below.

            [context]
            - From: <current state/version>
            - To: <target state/version>
            - Migration steps completed so far: <what's done>
            - Remaining steps: <what's left>
            - Blockers: <any issues>

            [constraints]
            - Zero downtime required
            - Maintain backwards compatibility during rollout
            END CRUMB
        """),
    },
}


def cmd_template(args: argparse.Namespace) -> None:
    """Manage crumb templates."""
    action = args.action

    if action == 'list':
        print("Built-in templates:")
        for name, tmpl in sorted(BUILTIN_TEMPLATES.items()):
            print(f"  {name:16s} ({tmpl['kind']:4s})  {tmpl['description']}")

        # User templates
        if TEMPLATE_DIR.exists():
            user_templates = sorted(TEMPLATE_DIR.glob('*.crumb'))
            if user_templates:
                print(f"\nUser templates (~/.crumb/templates/):")
                for f in user_templates:
                    name = f.stem
                    try:
                        parsed = parse_crumb(f.read_text(encoding='utf-8'))
                        kind = parsed['headers']['kind']
                        title = parsed['headers'].get('title', '')
                        print(f"  {name:16s} ({kind:4s})  {title}")
                    except ValueError:
                        print(f"  {name:16s} (invalid)")

    elif action == 'use':
        name = args.name
        if not name:
            print("Error: template name required. Run 'crumb template list' to see options.", file=sys.stderr)
            sys.exit(1)

        # Check user templates first, then built-in
        user_path = TEMPLATE_DIR / f'{name}.crumb'
        if user_path.exists():
            content = user_path.read_text(encoding='utf-8')
        elif name in BUILTIN_TEMPLATES:
            content = BUILTIN_TEMPLATES[name]['content']
        else:
            print(f"Error: unknown template '{name}'", file=sys.stderr)
            print(f"Run 'crumb template list' to see available templates.", file=sys.stderr)
            sys.exit(1)

        write_text(args.output, content)
        if args.output != '-':
            print(f"Created {args.output} from template '{name}'")

    elif action == 'add':
        name = args.name
        source = args.source_file
        if not name or not source:
            print("Error: both name and source file required.", file=sys.stderr)
            print("Usage: crumb template add <name> <file.crumb>", file=sys.stderr)
            sys.exit(1)

        # Validate the source crumb
        text = Path(source).read_text(encoding='utf-8')
        try:
            parse_crumb(text)
        except ValueError as exc:
            print(f"Error: {source} is not a valid crumb: {exc}", file=sys.stderr)
            sys.exit(1)

        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        dest = TEMPLATE_DIR / f'{name}.crumb'
        shutil.copy2(source, dest)
        print(f"Saved template '{name}' to {dest}")


# ── share ───────────────────────────────────────────────────────────

def cmd_share(args: argparse.Namespace) -> None:
    """Share a .crumb file via GitHub Gist or as a data URI fallback."""
    filepath = args.file
    text = Path(filepath).read_text(encoding='utf-8')

    # Extract title from the crumb for the gist description
    title = ''
    try:
        parsed = parse_crumb(text)
        title = parsed['headers'].get('title', '')
    except ValueError:
        pass

    description = f"CRUMB handoff: {title} — https://github.com/XioAISolutions/crumb-format"

    # Try gh gist create first
    try:
        result = subprocess.run(
            ["gh", "gist", "create", "--public", "-d", description, filepath],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            print(url)
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: generate a self-contained data URI
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title or 'CRUMB handoff'}</title>
<style>
body {{ font-family: monospace; max-width: 800px; margin: 2em auto; padding: 1em; background: #1e1e2e; color: #cdd6f4; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; background: #313244; padding: 1em; border-radius: 8px; }}
footer {{ margin-top: 2em; color: #6c7086; text-align: center; }}
</style></head><body>
<h2>{title or 'CRUMB handoff'}</h2>
<pre>{text}</pre>
<footer>Get CRUMB: pip install crumb-format</footer>
</body></html>"""
    encoded = base64.b64encode(html.encode('utf-8')).decode('ascii')
    data_uri = f"data:text/html;base64,{encoded}"
    print(data_uri)


# ── handoff ─────────────────────────────────────────────────────────

def _copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using platform-appropriate tool. Returns True on success."""
    system = platform.system()
    cmds = []
    if system == 'Darwin':
        cmds = [['pbcopy']]
    elif system == 'Linux':
        # Check for WSL
        if 'microsoft' in platform.uname().release.lower():
            cmds = [['clip.exe']]
        else:
            cmds = [['xclip', '-selection', 'clipboard'], ['xsel', '--clipboard', '--input']]
    elif system == 'Windows':
        cmds = [['clip.exe']]

    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, input=text, text=True, capture_output=True, timeout=5)
            if proc.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def cmd_handoff(args: argparse.Namespace) -> None:
    """Copy a .crumb file to clipboard for pasting into an AI tool."""
    filepath = args.file
    text = Path(filepath).read_text(encoding='utf-8')

    target = args.target

    messages = {
        'claude': 'Crumb copied! Open Claude and paste.',
        'cursor': 'Crumb copied! Open Cursor and paste into chat.',
        'chatgpt': 'Crumb copied! Open ChatGPT and paste.',
        'gemini': 'Crumb copied! Open Gemini and paste.',
    }

    if _copy_to_clipboard(text):
        if target:
            print(messages.get(target, f'Crumb copied! Open {target} and paste.'))
        else:
            print('Crumb copied to clipboard! Paste into any AI tool.')
    else:
        print(text)
        print('\n---\nCopy the above and paste into your AI tool', file=sys.stderr)


def _paste_from_clipboard() -> str | None:
    """Read text from clipboard using platform-appropriate tool."""
    system = platform.system()
    cmds = []
    if system == 'Darwin':
        cmds = [['pbpaste']]
    elif system == 'Linux':
        if 'microsoft' in platform.uname().release.lower():
            cmds = [['powershell.exe', '-c', 'Get-Clipboard']]
        else:
            cmds = [['xclip', '-selection', 'clipboard', '-o'],
                    ['xsel', '--clipboard', '--output']]
    elif system == 'Windows':
        cmds = [['powershell.exe', '-c', 'Get-Clipboard']]

    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                return proc.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def cmd_receive(args: argparse.Namespace) -> None:
    """Receive a crumb: read from clipboard or stdin, validate, optionally file to palace."""
    # Get text from clipboard, file, or stdin
    if args.file and args.file != '-':
        text = Path(args.file).read_text(encoding='utf-8')
        source_label = args.file
    else:
        pasted = _paste_from_clipboard()
        if pasted and 'BEGIN CRUMB' in pasted:
            text = pasted
            source_label = 'clipboard'
        else:
            print('Reading from stdin (paste crumb, then Ctrl+D)...', file=sys.stderr)
            text = sys.stdin.read()
            source_label = 'stdin'

    # Validate
    try:
        parsed = parse_crumb(text)
    except ValueError as e:
        print(f'Invalid crumb from {source_label}: {e}', file=sys.stderr)
        sys.exit(1)

    headers = parsed['headers']
    kind = headers['kind']
    title = headers.get('title', '(untitled)')
    print(f'Received: kind={kind}  title={title}  from={source_label}')

    # Save to file if requested
    if args.output and args.output != '-':
        write_text(args.output, text)
        print(f'Saved to {args.output}')

    # Auto-file to palace if requested
    if args.palace:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cli.palace import find_palace, add_observation, rebuild_index
        from cli.classify import classify

        root = find_palace(Path('.'))
        if root is None:
            print('No palace found — skipping palace filing. Run `crumb palace init` first.',
                  file=sys.stderr)
        else:
            wing = args.wing or headers.get('source', 'incoming').split('.')[0]
            room_name = headers.get('title', 'received').lower().replace(' ', '-')[:40]

            # Gather text from key sections
            bullets = []
            for section_name in ('goal', 'consolidated', 'context'):
                for line in parsed['sections'].get(section_name, []):
                    stripped = line.strip()
                    if stripped and stripped.startswith('-'):
                        bullets.append(stripped.lstrip('- ').strip())
                    elif stripped:
                        bullets.append(stripped)

            if not bullets:
                bullets = [f'Received {kind} crumb: {title}']

            hall = args.hall or classify(' '.join(bullets[:3]))

            for bullet in bullets[:10]:
                add_observation(root, wing, hall, room_name, bullet)
            rebuild_index(root)
            print(f'Filed {len(bullets[:10])} observations → {wing}/{hall}/{room_name}')

    # Show summary
    sections = parsed['sections']
    for name, body in sections.items():
        content_lines = [l for l in body if l.strip()]
        print(f'  [{name}] {len(content_lines)} lines')


def cmd_context(args: argparse.Namespace) -> None:
    """Generate a crumb from the current project state — git, palace, and todo crumbs."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    context_lines = []
    constraint_lines = []

    # --- Git state ---
    git_available = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        capture_output=True, text=True,
    ).returncode == 0

    if git_available:
        branch = _git_run('branch', '--show-current') or 'HEAD'
        context_lines.append(f'- Branch: {branch}')

        # Status summary
        status = _git_run('status', '--porcelain')
        if status:
            status_lines = [l for l in status.splitlines() if l.strip()]
            modified = sum(1 for l in status_lines if l.startswith(' M') or l.startswith('M'))
            added = sum(1 for l in status_lines if l.startswith('A') or l.startswith('??'))
            context_lines.append(f'- Working tree: {modified} modified, {added} untracked')
        else:
            context_lines.append('- Working tree: clean')

        # Recent commits
        recent = _git_run('log', '--oneline', f'-{args.commits}')
        if recent:
            context_lines.append(f'- Recent commits:')
            for line in recent.splitlines()[:args.commits]:
                if line.strip():
                    context_lines.append(f'  - {line.strip()}')

        # Uncommitted diff summary
        diff_stat = _git_run('diff', '--stat')
        if diff_stat:
            summary_line = diff_stat.strip().splitlines()[-1] if diff_stat.strip() else ''
            if summary_line:
                context_lines.append(f'- Uncommitted changes: {summary_line.strip()}')

        # Check for WIP signals
        last_msg = _git_run('log', '-1', '--format=%s') or ''
        if any(kw in last_msg.lower() for kw in ('wip', 'fixme', 'todo', 'broken')):
            constraint_lines.append('- Latest commit may be incomplete (WIP detected)')

    # --- Palace facts ---
    from cli.palace import find_palace, list_rooms

    palace_root = find_palace(Path('.'))
    if palace_root:
        rooms = list_rooms(palace_root, hall='facts')
        if rooms:
            context_lines.append('- Key facts from palace:')
            fact_count = 0
            for w, h, r, p in rooms:
                if fact_count >= args.max_facts:
                    break
                try:
                    body = p.read_text()
                except OSError:
                    continue
                in_section = False
                for line in body.splitlines():
                    s = line.strip()
                    if s == '[consolidated]':
                        in_section = True
                        continue
                    if s.startswith('[') and s.endswith(']'):
                        in_section = False
                        continue
                    if in_section and s.startswith('-') and fact_count < args.max_facts:
                        context_lines.append(f'  - {w}/{r}: {s.lstrip("- ").strip()}')
                        fact_count += 1

    # --- Todo crumbs in current dir ---
    todo_files = list(Path('.').glob('*.crumb'))
    open_todos = []
    for tf in todo_files:
        try:
            parsed = parse_crumb(tf.read_text())
        except (ValueError, OSError):
            continue
        if parsed['headers'].get('kind') != 'todo':
            continue
        for line in parsed['sections'].get('tasks', []):
            s = line.strip()
            if s.startswith('- [ ]'):
                open_todos.append(s[5:].strip())

    if open_todos:
        context_lines.append(f'- Open TODOs ({len(open_todos)}):')
        for t in open_todos[:5]:
            context_lines.append(f'  - {t}')
        if len(open_todos) > 5:
            context_lines.append(f'  - ... and {len(open_todos) - 5} more')

    # --- Build the goal ---
    if args.goal:
        goal = args.goal
    elif git_available:
        latest = _git_run('log', '-1', '--format=%s') or ''
        branch = _git_run('branch', '--show-current') or ''
        if branch and branch not in ('main', 'master', 'HEAD'):
            goal = f'Continue work on {branch}'
        elif latest:
            goal = f'Continue from: {latest}'
        else:
            goal = 'Continue work on this project'
    else:
        goal = 'Continue work on this project'

    title = args.title or goal[:80]

    if not constraint_lines:
        constraint_lines.append('- Review context before continuing')

    crumb_text = render_crumb(
        headers={
            'v': '1.1',
            'kind': 'task',
            'title': title,
            'source': args.source or 'crumb.context',
        },
        sections={
            'goal': [goal],
            'context': context_lines or ['- No project context available'],
            'constraints': constraint_lines,
        },
    )

    # Optional MeTalk compression
    if args.metalk:
        from cli.metalk import encode
        crumb_text = encode(crumb_text, level=args.metalk_level)

    if args.clipboard:
        if _copy_to_clipboard(crumb_text):
            print('Context crumb copied to clipboard!', file=sys.stderr)
        else:
            print(crumb_text)
            print('\n---\nCopy the above and paste into your next AI tool', file=sys.stderr)
    else:
        write_text(args.output, crumb_text)


# ── Agent Passport commands ──────────────────────────────────────────


def cmd_passport(args: argparse.Namespace) -> None:
    """Agent identity management: register, inspect, revoke, list."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agentauth import AgentPassport

    action = args.passport_action
    mgr = AgentPassport()

    if action == 'register':
        result = mgr.register(
            name=args.name,
            framework=args.framework,
            owner=args.owner,
            tools_allowed=args.tools_allowed if args.tools_allowed else None,
            tools_denied=args.tools_denied if args.tools_denied else None,
            ttl_days=args.ttl_days,
        )
        print(f"Registered agent '{result['name']}' — id: {result['agent_id']}")
        print(f"Passport saved to: {result['passport_path']}")
        if args.output:
            data = mgr.inspect(result['agent_id'])
            if data:
                content = render_crumb(data['headers'], data['sections'])
                write_text(args.output, content)

    elif action == 'inspect':
        data = mgr.inspect(args.agent_id)
        if data is None:
            print(f"Passport not found: {args.agent_id}", file=sys.stderr)
            sys.exit(1)
        print(render_crumb(data['headers'], data['sections']))

    elif action == 'revoke':
        ok = mgr.revoke(args.agent_id)
        if ok:
            print(f"Passport {args.agent_id} revoked.")
        else:
            print(f"Could not revoke {args.agent_id} (not found or already revoked).",
                  file=sys.stderr)
            sys.exit(1)

    elif action == 'list':
        agents = mgr.list_all(status_filter=args.status)
        if not agents:
            print("No agents found.")
            return
        print(f"{'ID':<16} {'Name':<30} {'Status':<10} {'Expires'}")
        print("-" * 76)
        for a in agents:
            print(f"{a['agent_id']:<16} {a['name']:<30} {a['status']:<10} {a.get('expires', 'n/a')}")


def cmd_policy(args: argparse.Namespace) -> None:
    """Tool authorization policy: set, test."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agentauth import ToolPolicy

    action = args.policy_action
    policy = ToolPolicy()

    if action == 'set':
        result = policy.set_policy(
            agent_name=args.agent_name,
            tools_allowed=args.allow if args.allow else None,
            tools_denied=args.deny if args.deny else None,
            max_actions_per_session=args.max_actions,
        )
        print(f"Policy updated for {args.agent_name}.")

    elif action == 'test':
        result = policy.test(agent_name=args.agent_name, tool=args.tool)
        if result['allowed']:
            print(f"\033[32mALLOW\033[0m {args.tool} — {result['reason']}")
        else:
            print(f"\033[31mDENY\033[0m {args.tool} — {result['reason']}")


def cmd_audit(args: argparse.Namespace) -> None:
    """Audit trail management: export, feed."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agentauth import AuditLogger

    action = args.audit_action
    logger = AuditLogger()

    if action == 'export':
        evidence = logger.export_evidence(
            agent_id=args.agent,
            since=args.since,
            output_format=args.format,
        )
        if args.output and args.output != '-':
            write_text(args.output, evidence)
        else:
            print(evidence)

    elif action == 'feed':
        lines = logger.feed(agent_id=args.agent)
        if not lines:
            print("No audit entries found.")
            return
        for line in lines:
            print(line)


# ── argument parser ──────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Shadow AI Scanner
# ---------------------------------------------------------------------------

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# AI tool config file patterns (glob-style, resolved relative to scan root)
_AI_CONFIG_GLOBS = [
    ".cursor/rules",
    ".cursor/rules/**",
    ".claude/*",
    ".windsurf*",
    "CLAUDE.md",
    ".github/copilot*",
    ".github/copilot*/**",
    ".continue/*",
    ".continue/**",
    ".aider*",
    ".cody*",
]

# Environment variable names that indicate AI API keys
_AI_ENV_KEY_PATTERNS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GOOGLE_AI_KEY",
    "GOOGLE_API_KEY",
    "COHERE_API_KEY",
    "HUGGINGFACE_API_KEY",
    "HF_TOKEN",
    "REPLICATE_API_TOKEN",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "PERPLEXITY_API_KEY",
    "AI21_API_KEY",
    "DEEPSEEK_API_KEY",
]

# AI SDK package names to look for in dependency manifests
_AI_SDK_PACKAGES = [
    "openai",
    "anthropic",
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-openai",
    "langchain-anthropic",
    "langgraph",
    "crewai",
    "autogen",
    "pyautogen",
    "llama-index",
    "llama_index",
    "llamaindex",
    "transformers",
    "huggingface-hub",
    "cohere",
    "google-generativeai",
    "google-genai",
    "replicate",
    "mistralai",
    "groq",
    "together",
    "ai21",
    "deepseek",
    "litellm",
    "guidance",
    "semantic-kernel",
    "promptflow",
    "dspy",
    "dspy-ai",
    "haystack-ai",
    "vllm",
    "claude-agent-sdk",
    "@anthropic-ai/sdk",
    "@openai/api",
    "@langchain/core",
    "@langchain/community",
    "@langchain/openai",
    "@langchain/anthropic",
    "llamaindex",  # npm
    "cohere-ai",
    "@google/generative-ai",
    "@mistralai/mistralai",
    "groq-sdk",
]

# Import patterns for code scanning (Python / JS / TS)
_AI_IMPORT_RES = [
    re.compile(r"^\s*import\s+(openai|anthropic|langchain|crewai|autogen|cohere|groq|litellm|replicate|mistralai|together|guidance|dspy|haystack)\b"),
    re.compile(r"^\s*from\s+(openai|anthropic|langchain|crewai|autogen|cohere|groq|litellm|replicate|mistralai|together|guidance|dspy|haystack)\b"),
    re.compile(r"""^\s*(?:const|let|var|import)\s+.*(?:require|from)\s*\(?\s*['"](@?(?:openai|anthropic|langchain|cohere-ai|groq-sdk|@google/generative-ai|@mistralai/mistralai|@anthropic-ai/sdk|@openai/api|@langchain/\w+|llamaindex))['"]"""),
]

# MCP config file names
_MCP_CONFIG_NAMES = [
    "claude_desktop_config.json",
    "mcp.json",
    ".mcp.json",
    "mcp_config.json",
]


def _scan_config_files(root: Path) -> List[Dict]:
    """Scan for AI tool configuration files."""
    findings: List[Dict] = []
    for pattern in _AI_CONFIG_GLOBS:
        for match in sorted(root.glob(pattern)):
            if match.is_file():
                findings.append({
                    "type": "ai_config",
                    "path": str(match.relative_to(root)),
                    "detail": f"AI tool configuration file: {match.name}",
                    "risk_level": "medium",
                })
    # de-duplicate by path
    seen: set = set()
    deduped: List[Dict] = []
    for f in findings:
        if f["path"] not in seen:
            seen.add(f["path"])
            deduped.append(f)
    return deduped


def _scan_env_files(root: Path) -> List[Dict]:
    """Scan .env* files for AI API keys."""
    findings: List[Dict] = []
    for envfile in sorted(root.rglob(".env*")):
        if not envfile.is_file():
            continue
        # skip directories and binary-looking files
        try:
            text = envfile.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            for pattern in _AI_ENV_KEY_PATTERNS:
                if key.upper() == pattern:
                    findings.append({
                        "type": "api_key",
                        "path": str(envfile.relative_to(root)),
                        "detail": f"AI API key found: {key}",
                        "risk_level": "critical",
                    })
    return findings


def _scan_dependencies(root: Path) -> List[Dict]:
    """Scan dependency manifests for AI SDK packages."""
    findings: List[Dict] = []
    sdk_set_lower = {p.lower() for p in _AI_SDK_PACKAGES}

    # --- requirements*.txt ---
    for req in sorted(root.rglob("requirements*.txt")):
        if not req.is_file():
            continue
        try:
            text = req.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # extract package name (before any version specifier)
            pkg = re.split(r"[>=<!\[;@\s]", line)[0].strip().lower()
            if pkg in sdk_set_lower:
                findings.append({
                    "type": "ai_dependency",
                    "path": str(req.relative_to(root)),
                    "detail": f"AI SDK dependency: {pkg}",
                    "risk_level": "high",
                })

    # --- pyproject.toml ---
    for pp in sorted(root.rglob("pyproject.toml")):
        if not pp.is_file():
            continue
        try:
            text = pp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pkg in _AI_SDK_PACKAGES:
            # loose check: package name appears in a dependency context
            if re.search(r'(?:^|\s|"|' + "'" + r')' + re.escape(pkg.lower()) + r'(?:\s|[>=<"\']|$)', text.lower(), re.MULTILINE):
                findings.append({
                    "type": "ai_dependency",
                    "path": str(pp.relative_to(root)),
                    "detail": f"AI SDK dependency: {pkg}",
                    "risk_level": "high",
                })

    # --- package.json ---
    for pj in sorted(root.rglob("package.json")):
        if not pj.is_file():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        all_deps: dict = {}
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            all_deps.update(data.get(section, {}))
        for dep_name in all_deps:
            if dep_name.lower() in sdk_set_lower:
                findings.append({
                    "type": "ai_dependency",
                    "path": str(pj.relative_to(root)),
                    "detail": f"AI SDK dependency: {dep_name}",
                    "risk_level": "high",
                })

    # --- Gemfile ---
    for gf in sorted(root.rglob("Gemfile")):
        if not gf.is_file():
            continue
        try:
            text = gf.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = re.match(r"""gem\s+['"]([^'"]+)['"]""", stripped)
            if m:
                gem = m.group(1).strip().lower()
                if gem in sdk_set_lower:
                    findings.append({
                        "type": "ai_dependency",
                        "path": str(gf.relative_to(root)),
                        "detail": f"AI SDK dependency (gem): {gem}",
                        "risk_level": "high",
                    })

    return findings


def _scan_mcp_configs(root: Path) -> List[Dict]:
    """Scan for MCP server configuration files."""
    findings: List[Dict] = []
    for name in _MCP_CONFIG_NAMES:
        for match in sorted(root.rglob(name)):
            if match.is_file():
                detail = f"MCP server configuration: {match.name}"
                # peek inside for mcpServers key
                try:
                    data = json.loads(match.read_text(encoding="utf-8", errors="replace"))
                    servers = data.get("mcpServers", {})
                    if servers:
                        detail += f" ({len(servers)} server(s): {', '.join(list(servers)[:5])})"
                except Exception:
                    pass
                findings.append({
                    "type": "mcp_config",
                    "path": str(match.relative_to(root)),
                    "detail": detail,
                    "risk_level": "high",
                })
    return findings


def _scan_code_imports(root: Path) -> List[Dict]:
    """Scan .py/.js/.ts files for AI SDK imports."""
    findings: List[Dict] = []
    seen: set = set()  # (path, pkg) dedup
    extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"}
    skip_dirs = {"node_modules", ".venv", "venv", "__pycache__", ".git", ".tox", "dist", "build"}

    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file() or fpath.suffix not in extensions:
            continue
        # skip heavy directories
        parts = set(fpath.relative_to(root).parts)
        if parts & skip_dirs:
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            for pat in _AI_IMPORT_RES:
                m = pat.match(line)
                if m:
                    pkg = m.group(1)
                    key = (str(fpath.relative_to(root)), pkg)
                    if key not in seen:
                        seen.add(key)
                        findings.append({
                            "type": "code_import",
                            "path": str(fpath.relative_to(root)),
                            "detail": f"AI SDK import: {pkg}",
                            "risk_level": "low",
                        })
    return findings


def _format_scan_text(findings: List[Dict], root: Path) -> str:
    """Format scan findings as human-readable text."""
    if not findings:
        return "Shadow AI scan complete: no findings.\n"

    lines: List[str] = []
    lines.append(f"Shadow AI Scan Report")
    lines.append(f"Scanned: {root}")
    lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total findings: {len(findings)}")
    lines.append("")

    # summary by risk level
    risk_counts: Dict[str, int] = {}
    for f in findings:
        risk_counts[f["risk_level"]] = risk_counts.get(f["risk_level"], 0) + 1
    for level in ("critical", "high", "medium", "low"):
        if level in risk_counts:
            lines.append(f"  {level.upper():10s}: {risk_counts[level]}")
    lines.append("")
    lines.append("-" * 72)

    # group by risk level
    for level in ("critical", "high", "medium", "low"):
        group = [f for f in findings if f["risk_level"] == level]
        if not group:
            continue
        lines.append(f"\n[{level.upper()}]")
        for f in group:
            lines.append(f"  {f['type']:16s} {f['path']}")
            lines.append(f"                   {f['detail']}")
    lines.append("")
    return "\n".join(lines)


def _format_scan_json(findings: List[Dict], root: Path) -> str:
    """Format scan findings as JSON."""
    report = {
        "scan_root": str(root),
        "date": datetime.datetime.now().isoformat(),
        "total_findings": len(findings),
        "findings": findings,
    }
    return json.dumps(report, indent=2)


def _format_scan_crumb(findings: List[Dict], root: Path) -> str:
    """Format scan findings as a .crumb audit file."""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    actions: List[str] = []
    for f in findings:
        actions.append(f"[{f['risk_level'].upper()}] {f['type']} | {f['path']} | {f['detail']}")

    max_risk = "low"
    for f in findings:
        if RISK_ORDER.get(f["risk_level"], 0) > RISK_ORDER.get(max_risk, 0):
            max_risk = f["risk_level"]

    verdict = "pass" if not findings else ("fail" if max_risk in ("critical", "high") else "review")

    lines = [
        "BEGIN CRUMB",
        "v = 1.1",
        "kind = audit",
        f"source = crumb.scan",
        f"title = Shadow AI Scan - {root}",
        f"ts = {now}",
        "---",
        "[goal]",
        f"Discover unauthorized/unregistered AI agents in {root}",
        "",
        "[actions]",
    ]
    for a in actions:
        lines.append(f"- {a}")
    if not actions:
        lines.append("- No findings")
    lines.append("")
    lines.append("[verdict]")
    lines.append(verdict)
    lines.append("")
    lines.append("END CRUMB")
    return "\n".join(lines)


def cmd_scan(args: argparse.Namespace) -> None:
    """Shadow AI scanner: discover unauthorized AI agents in a project directory."""
    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    min_risk = RISK_ORDER.get(args.min_risk, 0)

    # Collect all findings
    findings: List[Dict] = []
    findings.extend(_scan_config_files(root))
    findings.extend(_scan_env_files(root))
    findings.extend(_scan_dependencies(root))
    findings.extend(_scan_mcp_configs(root))
    findings.extend(_scan_code_imports(root))

    # Filter by minimum risk level
    findings = [f for f in findings if RISK_ORDER.get(f["risk_level"], 0) >= min_risk]

    # Sort by risk level descending, then path
    findings.sort(key=lambda f: (-RISK_ORDER.get(f["risk_level"], 0), f["path"]))

    # Format output
    fmt = args.format
    if fmt == "json":
        output = _format_scan_json(findings, root)
    elif fmt == "crumb":
        output = _format_scan_crumb(findings, root)
    else:
        output = _format_scan_text(findings, root)

    print(output)


def cmd_comply(args: argparse.Namespace) -> None:
    """Generate compliance report from agent passport and audit data."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agentauth import AgentPassport, ToolPolicy, AuditLogger
    from agentauth.store import PassportStore

    store = PassportStore()
    mgr = AgentPassport(store=store)
    policy = ToolPolicy(store=store)
    logger = AuditLogger(store=store)

    framework = args.framework
    now = datetime.datetime.now(datetime.timezone.utc)

    # ── Gather data ──────────────────────────────────────────
    agents = mgr.list_all()
    total = len(agents)
    active = [a for a in agents if a['status'] == 'active']
    revoked = [a for a in agents if a['status'] == 'revoked']
    expired = []
    for a in agents:
        exp = a.get('expires', '')
        if exp:
            try:
                exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
                if now > exp_date:
                    expired.append(a)
            except ValueError:
                pass

    # Policy coverage
    agents_with_policy = []
    agents_without_policy = []
    for a in agents:
        name = a.get('name', '')
        p = store.load_policy(name)
        if p:
            agents_with_policy.append(a)
        else:
            agents_without_policy.append(a)

    # Audit data
    audit_paths = store.list_audits()
    total_actions = 0
    allow_count = 0
    deny_count = 0
    audit_records = []
    for p in audit_paths:
        try:
            data = parse_crumb(p.read_text(encoding='utf-8'))
            actions = data['sections'].get('actions', [])
            for line in actions:
                stripped = line.strip()
                if not stripped or stripped.startswith('('):
                    continue
                total_actions += 1
                if 'ALLOW' in stripped:
                    allow_count += 1
                elif 'DENY' in stripped:
                    deny_count += 1
            verdict_lines = data['sections'].get('verdict', [])
            record = {'headers': data['headers'], 'verdict': {}}
            for vl in verdict_lines:
                vl = vl.strip()
                if ':' in vl:
                    k, v = vl.split(':', 1)
                    record['verdict'][k.strip()] = v.strip()
            audit_records.append(record)
        except (ValueError, KeyError):
            continue

    # Compliance score
    score = 100
    findings = []

    if total == 0:
        score = 0
        findings.append(('CRITICAL', 'No agents registered — no governance in place'))
    else:
        # Deduct for missing policies
        if agents_without_policy:
            pct = len(agents_without_policy) / total * 30
            score -= pct
            findings.append(('HIGH', f'{len(agents_without_policy)}/{total} agents lack tool authorization policies'))

        # Deduct for expired passports
        if expired:
            pct = len(expired) / total * 20
            score -= pct
            findings.append(('HIGH', f'{len(expired)} agent passport(s) expired — renew or revoke'))

        # Deduct for high deny rate
        if total_actions > 0 and deny_count / total_actions > 0.2:
            score -= 15
            findings.append(('MEDIUM', f'High deny rate: {deny_count}/{total_actions} actions denied ({deny_count/total_actions*100:.0f}%)'))

        # Deduct if no audit data
        if not audit_records:
            score -= 20
            findings.append(('HIGH', 'No audit trail data — enable session logging'))

    score = max(0, min(100, round(score)))

    # Framework-specific findings
    if framework == 'eu-ai-act':
        if not agents_with_policy:
            findings.append(('CRITICAL', 'EU AI Act Art. 9: No risk management policies defined'))
        if not audit_records:
            findings.append(('CRITICAL', 'EU AI Act Art. 12: No automatic logging/audit trail'))
        findings.append(('INFO', 'EU AI Act compliance assessment — review Art. 6-15 requirements'))
    elif framework == 'soc2':
        if agents_without_policy:
            findings.append(('HIGH', 'SOC2 CC6.1: Access control policies incomplete'))
        if not audit_records:
            findings.append(('HIGH', 'SOC2 CC7.2: System monitoring not evidenced'))
        findings.append(('INFO', 'SOC2 Type II assessment — ensure continuous monitoring'))

    # ── Format output ────────────────────────────────────────
    fmt = args.format

    if fmt == 'json':
        import json as _json
        report = {
            'generated': now.isoformat(),
            'framework': framework,
            'compliance_score': score,
            'summary': {
                'total_agents': total,
                'active': len(active),
                'revoked': len(revoked),
                'expired': len(expired),
                'with_policy': len(agents_with_policy),
                'without_policy': len(agents_without_policy),
                'total_actions': total_actions,
                'allowed_actions': allow_count,
                'denied_actions': deny_count,
                'audit_sessions': len(audit_records),
            },
            'agents': agents,
            'findings': [{'severity': s, 'message': m} for s, m in findings],
        }
        output = _json.dumps(report, indent=2)

    elif fmt == 'html':
        score_color = '#4caf50' if score >= 80 else '#ff9800' if score >= 50 else '#f44336'
        fw_label = {'general': 'General', 'eu-ai-act': 'EU AI Act', 'soc2': 'SOC2 Type II'}.get(framework, framework)

        findings_html = ''
        for sev, msg in findings:
            sev_colors = {'CRITICAL': '#f44336', 'HIGH': '#ff5722', 'MEDIUM': '#ff9800', 'INFO': '#2196f3'}
            c = sev_colors.get(sev, '#999')
            findings_html += f'<tr><td><span style="background:{c};color:#fff;padding:2px 8px;border-radius:3px;font-size:12px;">{sev}</span></td><td>{msg}</td></tr>\n'

        agent_rows = ''
        for a in agents:
            st = a.get('status', '')
            sc = '#4caf50' if st == 'active' else '#f44336' if st == 'revoked' else '#999'
            has_pol = 'Yes' if any(ap['name'] == a['name'] for ap in agents_with_policy) else '<span style="color:#f44336">No</span>'
            agent_rows += f'<tr><td>{a.get("agent_id","")}</td><td>{a.get("name","")}</td><td><span style="color:{sc}">{st}</span></td><td>{a.get("issued","")}</td><td>{a.get("expires","")}</td><td>{has_pol}</td></tr>\n'

        output = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AgentAuth Compliance Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
.container {{ max-width: 900px; margin: 0 auto; }}
h1 {{ color: #fff; border-bottom: 2px solid #333; padding-bottom: 10px; }}
h2 {{ color: #aaa; margin-top: 30px; }}
.score-box {{ display: inline-block; background: {score_color}; color: #fff; font-size: 48px; font-weight: bold; padding: 20px 40px; border-radius: 12px; }}
.meta {{ color: #888; font-size: 14px; margin: 10px 0; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
.stat {{ background: #16213e; padding: 16px; border-radius: 8px; text-align: center; }}
.stat .num {{ font-size: 28px; font-weight: bold; color: #fff; }}
.stat .label {{ font-size: 12px; color: #888; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #2a2a4a; }}
th {{ color: #888; font-size: 12px; text-transform: uppercase; }}
.footer {{ margin-top: 40px; padding-top: 10px; border-top: 1px solid #333; color: #666; font-size: 12px; }}
</style></head><body>
<div class="container">
<h1>AgentAuth Compliance Report</h1>
<p class="meta">Framework: {fw_label} | Generated: {now.strftime("%Y-%m-%d %H:%M UTC")}</p>

<h2>Compliance Score</h2>
<div class="score-box">{score}</div>
<span style="margin-left:16px;color:#888;">/ 100</span>

<h2>Executive Summary</h2>
<div class="stat-grid">
<div class="stat"><div class="num">{total}</div><div class="label">Total Agents</div></div>
<div class="stat"><div class="num" style="color:#4caf50">{len(active)}</div><div class="label">Active</div></div>
<div class="stat"><div class="num" style="color:#f44336">{len(revoked)}</div><div class="label">Revoked</div></div>
<div class="stat"><div class="num" style="color:#ff9800">{len(expired)}</div><div class="label">Expired</div></div>
</div>
<div class="stat-grid">
<div class="stat"><div class="num">{total_actions}</div><div class="label">Total Actions</div></div>
<div class="stat"><div class="num" style="color:#4caf50">{allow_count}</div><div class="label">Allowed</div></div>
<div class="stat"><div class="num" style="color:#f44336">{deny_count}</div><div class="label">Denied</div></div>
<div class="stat"><div class="num">{len(audit_records)}</div><div class="label">Audit Sessions</div></div>
</div>

<h2>Agent Inventory</h2>
<table><tr><th>ID</th><th>Name</th><th>Status</th><th>Issued</th><th>Expires</th><th>Policy</th></tr>
{agent_rows if agent_rows else '<tr><td colspan="6" style="color:#888;">No agents registered</td></tr>'}
</table>

<h2>Policy Coverage</h2>
<div style="background:#16213e;border-radius:8px;padding:16px;margin:10px 0;">
<div style="display:flex;align-items:center;gap:8px;">
<div style="flex:1;background:#333;border-radius:4px;height:24px;overflow:hidden;">
<div style="background:#4caf50;height:100%;width:{len(agents_with_policy)/total*100 if total else 0:.0f}%;"></div>
</div>
<span>{len(agents_with_policy)}/{total} agents have policies</span>
</div>
</div>

<h2>Findings & Recommendations</h2>
<table>
{findings_html if findings_html else '<tr><td colspan="2" style="color:#4caf50;">No findings — all checks passed</td></tr>'}
</table>

<div class="footer">
AgentAuth Compliance Report v1.0 | CRUMB Format v1.1 | {fw_label} Framework<br>
Generated by <code>crumb comply</code> — https://github.com/XioAISolutions/crumb-format
</div>
</div></body></html>'''

    else:  # text
        lines = []
        lines.append('=' * 70)
        lines.append('AGENTAUTH COMPLIANCE REPORT')
        lines.append('=' * 70)
        fw_label = {'general': 'General', 'eu-ai-act': 'EU AI Act', 'soc2': 'SOC2 Type II'}.get(framework, framework)
        lines.append(f'Framework:  {fw_label}')
        lines.append(f'Generated:  {now.strftime("%Y-%m-%d %H:%M UTC")}')
        lines.append(f'Score:      {score}/100')
        lines.append('')

        lines.append('EXECUTIVE SUMMARY')
        lines.append('-' * 40)
        lines.append(f'  Total agents:      {total}')
        lines.append(f'  Active:            {len(active)}')
        lines.append(f'  Revoked:           {len(revoked)}')
        lines.append(f'  Expired:           {len(expired)}')
        lines.append(f'  With policy:       {len(agents_with_policy)}')
        lines.append(f'  Without policy:    {len(agents_without_policy)}')
        lines.append(f'  Total actions:     {total_actions}')
        lines.append(f'  Allowed:           {allow_count}')
        lines.append(f'  Denied:            {deny_count}')
        lines.append(f'  Audit sessions:    {len(audit_records)}')
        lines.append('')

        lines.append('AGENT INVENTORY')
        lines.append('-' * 40)
        if agents:
            lines.append(f'  {"ID":<16} {"Name":<24} {"Status":<10} {"Expires":<12} Policy')
            for a in agents:
                has_pol = 'Yes' if any(ap['name'] == a['name'] for ap in agents_with_policy) else 'NO'
                lines.append(f'  {a.get("agent_id",""):<16} {a.get("name",""):<24} {a.get("status",""):<10} {a.get("expires","n/a"):<12} {has_pol}')
        else:
            lines.append('  (no agents registered)')
        lines.append('')

        lines.append('FINDINGS & RECOMMENDATIONS')
        lines.append('-' * 40)
        if findings:
            for sev, msg in findings:
                lines.append(f'  [{sev}] {msg}')
        else:
            lines.append('  No findings — all checks passed.')
        lines.append('')
        lines.append('=' * 70)
        output = '\n'.join(lines)

    if args.output and args.output != '-':
        write_text(args.output, output)
        print(f'Compliance report written to {args.output}', file=sys.stderr)
    else:
        print(output)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Generate a self-contained HTML dashboard for agent auth overview."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agentauth import AgentPassport, ToolPolicy, AuditLogger
    from agentauth.store import PassportStore

    store = PassportStore()
    passport_mgr = AgentPassport(store)
    logger = AuditLogger(store)

    # ── Gather data ───────────────────────────────────────────
    all_agents = passport_mgr.list_all(status_filter="all")
    now = datetime.datetime.now(datetime.timezone.utc)
    now_str = now.strftime("%Y-%m-%d")
    generated_at = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    total = len(all_agents)
    active = 0
    revoked = 0
    expired = 0
    for a in all_agents:
        st = a.get("status", "unknown")
        if st == "revoked":
            revoked += 1
        elif a.get("expires", "") and a["expires"] < now_str:
            expired += 1
        elif st == "active":
            active += 1

    # Enrich agents with framework/owner from passport data
    enriched_agents = []
    for a in all_agents:
        agent_id = a["agent_id"]
        data = passport_mgr.inspect(agent_id)
        framework = ""
        owner = ""
        if data:
            h = data["headers"]
            framework = h.get("agent_framework", "")
            owner = ""
            # Try to get owner from identity section
            for line in data["sections"].get("identity", []):
                stripped = line.strip()
                if stripped.startswith("owner:"):
                    owner = stripped.split(":", 1)[1].strip()
                    break
        # Determine effective status
        st = a.get("status", "unknown")
        if st != "revoked" and a.get("expires", "") and a["expires"] < now_str:
            effective_status = "expired"
        else:
            effective_status = st
        enriched_agents.append({
            "agent_id": agent_id,
            "name": a.get("name", ""),
            "framework": framework,
            "owner": owner,
            "status": effective_status,
            "issued": a.get("issued", ""),
            "expires": a.get("expires", ""),
        })

    # Policy coverage
    policy_files = set()
    for p in store.policies_dir.glob("*.json"):
        policy_files.add(p.stem)
    agents_with_policy = []
    agents_without_policy = []
    for a in enriched_agents:
        name = a["name"]
        if name in policy_files or a["agent_id"] in policy_files:
            agents_with_policy.append(a["name"] or a["agent_id"])
        else:
            agents_without_policy.append(a["name"] or a["agent_id"])

    # Audit feed
    audit_lines = logger.feed()
    recent_audit = audit_lines[-30:] if len(audit_lines) > 30 else audit_lines
    recent_audit.reverse()  # newest first

    # Risk summary — count verdicts
    allow_count = 0
    deny_count = 0
    for line in audit_lines:
        if "] ALLOW " in line or "] ALLOW\t" in line:
            allow_count += 1
        elif "] DENY " in line or "] DENY\t" in line:
            deny_count += 1
    total_actions = allow_count + deny_count

    # Per-agent risk (deny ratio)
    agent_deny_map: dict[str, list[int, int]] = {}
    for line in audit_lines:
        # Lines look like: [agent_id/session_id] [timestamp] VERDICT tool: detail
        if line.startswith("["):
            bracket_end = line.find("]")
            if bracket_end > 0:
                inner = line[1:bracket_end]
                aid = inner.split("/")[0] if "/" in inner else inner
                if aid not in agent_deny_map:
                    agent_deny_map[aid] = [0, 0]  # [total, denied]
                agent_deny_map[aid][0] += 1
                if "] DENY " in line or "] DENY\t" in line:
                    agent_deny_map[aid][1] += 1

    # ── Build HTML ────────────────────────────────────────────
    def _esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))

    # Agent table rows
    agent_rows = ""
    for a in enriched_agents:
        status_class = a["status"]
        agent_rows += (
            f'<tr class="status-{_esc(status_class)}">'
            f'<td class="mono">{_esc(a["agent_id"])}</td>'
            f'<td>{_esc(a["name"])}</td>'
            f'<td>{_esc(a["framework"])}</td>'
            f'<td>{_esc(a["owner"])}</td>'
            f'<td><span class="badge badge-{_esc(status_class)}">{_esc(a["status"])}</span></td>'
            f'<td>{_esc(a["issued"])}</td>'
            f'<td>{_esc(a["expires"])}</td>'
            f'</tr>\n'
        )

    # Policy coverage items
    policy_html = ""
    for name in agents_with_policy:
        policy_html += f'<div class="policy-item policy-yes"><span class="policy-dot">&#9679;</span> {_esc(name)}</div>\n'
    for name in agents_without_policy:
        policy_html += f'<div class="policy-item policy-no"><span class="policy-dot">&#9675;</span> {_esc(name)}</div>\n'
    if not policy_html:
        policy_html = '<div class="empty-state">No agents registered.</div>'

    policy_covered = len(agents_with_policy)
    policy_total = len(agents_with_policy) + len(agents_without_policy)
    policy_pct = round(policy_covered / policy_total * 100) if policy_total else 0

    # Audit feed HTML
    audit_html = ""
    for line in recent_audit:
        css = "audit-allow" if "ALLOW" in line else "audit-deny" if "DENY" in line else ""
        audit_html += f'<div class="audit-line {css}">{_esc(line)}</div>\n'
    if not audit_html:
        audit_html = '<div class="empty-state">No audit activity recorded.</div>'

    # Risk bars
    risk_bars_html = ""
    for aid, (t, d) in sorted(agent_deny_map.items()):
        pct = round(d / t * 100) if t else 0
        bar_color = "#4ade80" if pct < 10 else "#facc15" if pct < 30 else "#f87171"
        name_label = aid
        for a in enriched_agents:
            if a["agent_id"] == aid:
                name_label = a["name"] or aid
                break
        risk_bars_html += (
            f'<div class="risk-row">'
            f'<span class="risk-label">{_esc(name_label)}</span>'
            f'<div class="risk-bar-bg">'
            f'<div class="risk-bar-fill" style="width:{pct}%;background:{bar_color};"></div>'
            f'</div>'
            f'<span class="risk-pct">{pct}%</span>'
            f'</div>\n'
        )
    if not risk_bars_html:
        # Show a summary bar for overall if no per-agent data
        if total_actions > 0:
            deny_pct = round(deny_count / total_actions * 100)
            bar_color = "#4ade80" if deny_pct < 10 else "#facc15" if deny_pct < 30 else "#f87171"
            risk_bars_html = (
                f'<div class="risk-row">'
                f'<span class="risk-label">Overall</span>'
                f'<div class="risk-bar-bg">'
                f'<div class="risk-bar-fill" style="width:{deny_pct}%;background:{bar_color};"></div>'
                f'</div>'
                f'<span class="risk-pct">{deny_pct}%</span>'
                f'</div>\n'
            )
        else:
            risk_bars_html = '<div class="empty-state">No actions recorded yet.</div>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentAuth Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #0f1117; color: #e2e8f0; line-height: 1.6; padding: 24px;
  }}
  h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #f1f5f9; }}
  .subtitle {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 28px; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: #cbd5e1; margin-bottom: 14px; }}

  /* Cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{
    background: #1e2030; border: 1px solid #2a2d3e; border-radius: 10px;
    padding: 20px; text-align: center;
  }}
  .card .num {{ font-size: 2.2rem; font-weight: 700; }}
  .card .label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }}
  .card.total .num {{ color: #818cf8; }}
  .card.active .num {{ color: #4ade80; }}
  .card.revoked .num {{ color: #f87171; }}
  .card.expired .num {{ color: #facc15; }}

  /* Table */
  .table-wrap {{ overflow-x: auto; margin-bottom: 32px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{
    background: #1a1c2e; color: #94a3b8; font-weight: 600; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.06em; padding: 10px 12px; text-align: left;
    border-bottom: 2px solid #2a2d3e; cursor: pointer; user-select: none; white-space: nowrap;
  }}
  th:hover {{ color: #e2e8f0; }}
  th .sort-arrow {{ margin-left: 4px; font-size: 0.6rem; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #1e2030; }}
  tr:hover td {{ background: #1a1c2e; }}
  .mono {{ font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace; font-size: 0.82rem; }}

  /* Badges */
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .badge-active {{ background: #064e3b; color: #4ade80; }}
  .badge-revoked {{ background: #450a0a; color: #f87171; }}
  .badge-expired {{ background: #422006; color: #facc15; }}
  .badge-unknown {{ background: #1e293b; color: #94a3b8; }}

  /* Two-column layout */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
  @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .panel {{
    background: #1e2030; border: 1px solid #2a2d3e; border-radius: 10px; padding: 20px;
  }}

  /* Policy coverage */
  .policy-bar-bg {{
    height: 8px; background: #2a2d3e; border-radius: 4px; margin-bottom: 14px; overflow: hidden;
  }}
  .policy-bar-fill {{ height: 100%; border-radius: 4px; background: #818cf8; transition: width 0.3s; }}
  .policy-summary {{ font-size: 0.8rem; color: #94a3b8; margin-bottom: 12px; }}
  .policy-item {{ font-size: 0.85rem; padding: 3px 0; }}
  .policy-yes {{ color: #4ade80; }}
  .policy-no {{ color: #f87171; }}
  .policy-dot {{ margin-right: 6px; }}

  /* Audit feed */
  .audit-feed {{ max-height: 340px; overflow-y: auto; }}
  .audit-line {{
    font-family: "SF Mono", "Fira Code", monospace; font-size: 0.78rem;
    padding: 4px 8px; border-radius: 4px; margin-bottom: 3px; white-space: pre-wrap; word-break: break-all;
  }}
  .audit-allow {{ background: #0a2e1f; color: #86efac; }}
  .audit-deny {{ background: #2d0a0a; color: #fca5a5; }}

  /* Risk bars */
  .risk-row {{ display: flex; align-items: center; margin-bottom: 10px; }}
  .risk-label {{ width: 140px; font-size: 0.85rem; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .risk-bar-bg {{ flex: 1; height: 16px; background: #2a2d3e; border-radius: 4px; margin: 0 12px; overflow: hidden; }}
  .risk-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; min-width: 2px; }}
  .risk-pct {{ width: 48px; text-align: right; font-size: 0.82rem; font-weight: 600; flex-shrink: 0; }}

  .empty-state {{ color: #64748b; font-size: 0.85rem; font-style: italic; padding: 12px 0; }}

  /* Filter */
  .filter-bar {{ margin-bottom: 14px; }}
  .filter-bar input {{
    background: #161825; border: 1px solid #2a2d3e; color: #e2e8f0; border-radius: 6px;
    padding: 7px 12px; font-size: 0.85rem; width: 260px; outline: none;
  }}
  .filter-bar input:focus {{ border-color: #818cf8; }}
</style>
</head>
<body>
<h1>AgentAuth Dashboard</h1>
<div class="subtitle">Generated {_esc(generated_at)}</div>

<h2>Agent Overview</h2>
<div class="cards">
  <div class="card total"><div class="num">{total}</div><div class="label">Total Agents</div></div>
  <div class="card active"><div class="num">{active}</div><div class="label">Active</div></div>
  <div class="card revoked"><div class="num">{revoked}</div><div class="label">Revoked</div></div>
  <div class="card expired"><div class="num">{expired}</div><div class="label">Expired</div></div>
</div>

<h2>Agent Table</h2>
<div class="filter-bar"><input type="text" id="agentFilter" placeholder="Filter agents..." oninput="filterTable()"></div>
<div class="table-wrap">
<table id="agentTable">
<thead>
<tr>
  <th onclick="sortTable(0)">ID <span class="sort-arrow">&#9650;&#9660;</span></th>
  <th onclick="sortTable(1)">Name <span class="sort-arrow">&#9650;&#9660;</span></th>
  <th onclick="sortTable(2)">Framework <span class="sort-arrow">&#9650;&#9660;</span></th>
  <th onclick="sortTable(3)">Owner <span class="sort-arrow">&#9650;&#9660;</span></th>
  <th onclick="sortTable(4)">Status <span class="sort-arrow">&#9650;&#9660;</span></th>
  <th onclick="sortTable(5)">Issued <span class="sort-arrow">&#9650;&#9660;</span></th>
  <th onclick="sortTable(6)">Expires <span class="sort-arrow">&#9650;&#9660;</span></th>
</tr>
</thead>
<tbody>
{agent_rows}</tbody>
</table>
</div>

<div class="grid-2">
  <div class="panel">
    <h2>Policy Coverage</h2>
    <div class="policy-summary">{policy_covered} of {policy_total} agents have policies ({policy_pct}%)</div>
    <div class="policy-bar-bg"><div class="policy-bar-fill" style="width:{policy_pct}%"></div></div>
    {policy_html}
  </div>
  <div class="panel">
    <h2>Risk Summary</h2>
    <div class="policy-summary">Denial rate per agent (higher = more blocked actions)</div>
    {risk_bars_html}
  </div>
</div>

<h2>Recent Audit Activity</h2>
<div class="panel">
  <div class="audit-feed">
    {audit_html}
  </div>
</div>

<script>
var sortDir = {{}};
function sortTable(col) {{
  var table = document.getElementById("agentTable");
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.rows);
  var dir = sortDir[col] === "asc" ? "desc" : "asc";
  sortDir[col] = dir;
  rows.sort(function(a, b) {{
    var va = a.cells[col].textContent.trim().toLowerCase();
    var vb = b.cells[col].textContent.trim().toLowerCase();
    if (va < vb) return dir === "asc" ? -1 : 1;
    if (va > vb) return dir === "asc" ? 1 : -1;
    return 0;
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
}}
function filterTable() {{
  var q = document.getElementById("agentFilter").value.toLowerCase();
  var rows = document.getElementById("agentTable").tBodies[0].rows;
  for (var i = 0; i < rows.length; i++) {{
    var text = rows[i].textContent.toLowerCase();
    rows[i].style.display = text.indexOf(q) >= 0 ? "" : "none";
  }}
}}
</script>
</body>
</html>'''

    output_path = args.output
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Dashboard written to {output_path}")


# ── MeTalk compression ───────────────────────────────────────────────


def cmd_metalk(args: argparse.Namespace) -> None:
    """Encode or decode a crumb using MeTalk caveman compression."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cli.metalk import encode, decode, compression_stats

    text = read_text(args.file)

    if args.decode:
        result = decode(text)
        if args.output and args.output != '-':
            write_text(args.output, result)
            print('MeTalk decoded.', file=sys.stderr)
        else:
            print(result)
    else:
        level = args.level
        result = encode(text, level=level)
        stats = compression_stats(text, result)
        if args.output and args.output != '-':
            write_text(args.output, result)
            print(f"MeTalk (level {level}): "
                  f"{stats['original_tokens']}→{stats['encoded_tokens']} tokens "
                  f"({stats['pct_saved']}% saved, {stats['ratio']}x)",
                  file=sys.stderr)
        else:
            print(result)
            print(f"\n--- MeTalk (level {level}): "
                  f"{stats['original_tokens']}→{stats['encoded_tokens']} tokens "
                  f"({stats['pct_saved']}% saved, {stats['ratio']}x) ---",
                  file=sys.stderr)


# ── Format bridges ───────────────────────────────────────────────────

def _bridge_export_openai(data):
    """CRUMB → OpenAI Assistant thread messages."""
    import json as _json
    headers = data['headers']
    sections = data['sections']
    messages = []
    # System message with goal/context
    goal = '\n'.join(sections.get('goal', []))
    context = '\n'.join(sections.get('context', []))
    constraints = '\n'.join(sections.get('constraints', []))
    consolidated = '\n'.join(sections.get('consolidated', []))
    if goal or context or consolidated:
        sys_content = ''
        if goal:
            sys_content += f"Goal: {goal.strip()}\n"
        if context:
            sys_content += f"Context: {context.strip()}\n"
        if constraints:
            sys_content += f"Constraints: {constraints.strip()}\n"
        if consolidated:
            sys_content += consolidated.strip()
        messages.append({"role": "system", "content": sys_content.strip()})
    # Raw observations as user messages
    for line in sections.get('raw', []):
        line = line.strip()
        if line:
            messages.append({"role": "user", "content": line})
    result = {
        "messages": messages,
        "metadata": {
            "title": headers.get('title', ''),
            "kind": headers.get('kind', ''),
            "source": headers.get('source', ''),
        }
    }
    return _json.dumps(result, indent=2)


def _bridge_export_langchain(data):
    """CRUMB → LangChain ConversationBufferMemory."""
    import json as _json
    sections = data['sections']
    messages = []
    consolidated = sections.get('consolidated', [])
    raw = sections.get('raw', [])
    for line in consolidated:
        line = line.strip()
        if line:
            messages.append({"type": "ai", "content": line})
    for line in raw:
        line = line.strip()
        if line:
            messages.append({"type": "human", "content": line})
    return _json.dumps({"chat_memory": {"messages": messages}}, indent=2)


def _bridge_export_crewai(data):
    """CRUMB → CrewAI task definition."""
    import json as _json
    headers = data['headers']
    sections = data['sections']
    goal = '\n'.join(l.strip() for l in sections.get('goal', []))
    context = '\n'.join(l.strip() for l in sections.get('context', []))
    constraints = '\n'.join(l.strip() for l in sections.get('constraints', []))
    return _json.dumps({
        "description": goal or headers.get('title', ''),
        "expected_output": constraints or "Complete the task as specified.",
        "agent": headers.get('source', ''),
        "context": context,
    }, indent=2)


def _bridge_export_autogen(data):
    """CRUMB → AutoGen conversation messages."""
    import json as _json
    headers = data['headers']
    sections = data['sections']
    messages = []
    goal = '\n'.join(l.strip() for l in sections.get('goal', []))
    if goal:
        messages.append({"role": "user", "content": goal, "name": "user"})
    context = '\n'.join(l.strip() for l in sections.get('context', []))
    if context:
        messages.append({"role": "assistant", "content": f"Context: {context}", "name": headers.get('source', 'assistant')})
    for line in sections.get('consolidated', []):
        line = line.strip()
        if line:
            messages.append({"role": "assistant", "content": line, "name": "memory"})
    return _json.dumps(messages, indent=2)


def _bridge_export_claude_project(data):
    """CRUMB → Claude Project custom instructions."""
    headers = data['headers']
    sections = data['sections']
    lines = []
    lines.append(f"# {headers.get('title', 'Project Context')}")
    lines.append('')
    for section_name in ('goal', 'context', 'constraints', 'consolidated', 'project', 'modules'):
        content = sections.get(section_name, [])
        if content:
            lines.append(f"## {section_name.title()}")
            for l in content:
                lines.append(l.rstrip())
            lines.append('')
    lines.append('---')
    lines.append('When I say "crumb it", generate a CRUMB summarizing the current state.')
    lines.append('Format: BEGIN CRUMB / v=1.1 / headers / --- / sections / END CRUMB')
    return '\n'.join(lines)


def _bridge_import_openai(text):
    """OpenAI thread JSON → CRUMB."""
    import json as _json
    data = _json.loads(text)
    messages = data.get('messages', data if isinstance(data, list) else [])
    meta = data.get('metadata', {})
    goal_parts = []
    context_parts = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'system':
            context_parts.append(content)
        elif role == 'user':
            goal_parts.append(content)
    headers = {'v': '1.1', 'kind': meta.get('kind', 'task'), 'title': meta.get('title', 'Imported from OpenAI')}
    sections = {}
    if goal_parts:
        sections['goal'] = [f'  {g}' for g in goal_parts]
    if context_parts:
        sections['context'] = [f'  {c}' for c in context_parts]
    return render_crumb(headers, sections)


def _bridge_import_langchain(text):
    """LangChain memory JSON → CRUMB."""
    import json as _json
    data = _json.loads(text)
    messages = data.get('chat_memory', {}).get('messages', [])
    consolidated = []
    raw = []
    for msg in messages:
        t = msg.get('type', '')
        c = msg.get('content', '')
        if t == 'ai':
            consolidated.append(f'  {c}')
        else:
            raw.append(f'  {c}')
    headers = {'v': '1.1', 'kind': 'mem', 'title': 'Imported from LangChain'}
    sections = {}
    if consolidated:
        sections['consolidated'] = consolidated
    if raw:
        sections['raw'] = raw
    return render_crumb(headers, sections)


def _bridge_import_crewai(text):
    """CrewAI task JSON → CRUMB."""
    import json as _json
    data = _json.loads(text)
    headers = {'v': '1.1', 'kind': 'task', 'title': data.get('description', 'CrewAI Task')[:80]}
    if data.get('agent'):
        headers['source'] = data['agent']
    sections = {
        'goal': [f"  {data.get('description', '')}"],
    }
    if data.get('context'):
        sections['context'] = [f"  {data['context']}"]
    if data.get('expected_output'):
        sections['constraints'] = [f"  Expected output: {data['expected_output']}"]
    return render_crumb(headers, sections)


def _bridge_import_autogen(text):
    """AutoGen message array → CRUMB."""
    import json as _json
    messages = _json.loads(text)
    if not isinstance(messages, list):
        messages = [messages]
    goal_parts = []
    context_parts = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'user':
            goal_parts.append(f'  {content}')
        else:
            context_parts.append(f'  {content}')
    headers = {'v': '1.1', 'kind': 'task', 'title': 'Imported from AutoGen'}
    sections = {}
    if goal_parts:
        sections['goal'] = goal_parts
    if context_parts:
        sections['context'] = context_parts
    return render_crumb(headers, sections)


BRIDGE_EXPORTERS = {
    'openai-threads': _bridge_export_openai,
    'langchain-memory': _bridge_export_langchain,
    'crewai-task': _bridge_export_crewai,
    'autogen': _bridge_export_autogen,
    'claude-project': _bridge_export_claude_project,
}

BRIDGE_IMPORTERS = {
    'openai-threads': _bridge_import_openai,
    'langchain-memory': _bridge_import_langchain,
    'crewai-task': _bridge_import_crewai,
    'autogen': _bridge_import_autogen,
}


def cmd_bridge(args: argparse.Namespace) -> None:
    """Convert between CRUMB and other AI agent formats."""
    action = args.bridge_action

    if action == 'export':
        fmt = args.to
        if fmt not in BRIDGE_EXPORTERS:
            print(f"Unknown export format: {fmt}", file=sys.stderr)
            sys.exit(1)
        text = read_text(args.input)
        data = parse_crumb(text)
        result = BRIDGE_EXPORTERS[fmt](data)
        if args.output and args.output != '-':
            write_text(args.output, result)
        else:
            print(result)

    elif action == 'import':
        fmt = args.source_format
        if fmt not in BRIDGE_IMPORTERS:
            print(f"Unknown import format: {fmt}", file=sys.stderr)
            sys.exit(1)
        text = read_text(args.input)
        result = BRIDGE_IMPORTERS[fmt](text)
        if args.output and args.output != '-':
            write_text(args.output, result)
        else:
            print(result)

    elif action == 'list':
        print("Export formats:")
        for k in BRIDGE_EXPORTERS:
            print(f"  {k}")
        print("\nImport formats:")
        for k in BRIDGE_IMPORTERS:
            print(f"  {k}")

    elif action == 'mempalace':
        from cli import mempalace_bridge
        sub_action = getattr(args, 'bridge_action2', None) or getattr(args, 'mempalace_action', None)
        try:
            if sub_action == 'export':
                mempalace_bridge.run_bridge_export(args)
            elif sub_action == 'import':
                mempalace_bridge.run_bridge_import(args)
            else:
                print(f"Unknown mempalace action: {sub_action}", file=sys.stderr)
                sys.exit(1)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)


# ── Pack / Lint commands ─────────────────────────────────────────────

def cmd_pack(args: argparse.Namespace) -> None:
    """Assemble a deterministic CRUMB context pack from a directory of crumbs."""
    from cli import pack
    from cli.local_ai import LocalAIError

    try:
        pack.run_pack(args)
    except LocalAIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_lint(args: argparse.Namespace) -> None:
    """Lint CRUMBs for secrets, oversized raw logs, suspicious headers, and budget issues."""
    from cli import linting

    try:
        linting.run_lint(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


# ── squeeze (budget-aware packer) ─────────────────────────────────────

def cmd_squeeze(args: argparse.Namespace) -> None:
    from cli import hashing, squeeze as squeeze_mod

    text = read_text(args.file)
    seen: set[str] = set()
    if args.no_seen:
        seen = set()
    elif args.seen:
        seen = hashing.load_seen(args.seen)
    else:
        seen = hashing.load_seen()
    for extra in args.seen_hash or []:
        seen.add(extra.strip())

    try:
        rendered, report = squeeze_mod.squeeze_crumb(
            text,
            budget=args.budget,
            seen=seen,
            metalk_max_level=args.metalk_max_level,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(squeeze_mod.format_report(report))
        return

    write_text(args.output, rendered)
    if args.output != '-':
        print(squeeze_mod.format_report(report))


# ── hash ──────────────────────────────────────────────────────────────

def cmd_hash(args: argparse.Namespace) -> None:
    from cli import hashing

    text = read_text(args.file)
    try:
        digest = hashing.content_hash(text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.short:
        digest = hashing.short_hash(digest, length=args.short)
    print(digest)


# ── seen set ──────────────────────────────────────────────────────────

def cmd_seen(args: argparse.Namespace) -> None:
    from cli import hashing

    action = args.seen_action
    store = getattr(args, 'store', None)

    if action == 'list':
        for entry in sorted(hashing.load_seen(store)):
            print(entry)
        return

    if action == 'add':
        digests = list(args.digests or [])
        for path in args.from_file or []:
            text = read_text(path)
            digests.append(hashing.content_hash(text))
        if not digests:
            print("Error: provide at least one sha256:<hex> digest or --from-file path", file=sys.stderr)
            sys.exit(1)
        updated = hashing.add_seen(digests, store)
        print(f"Seen set now holds {len(updated)} digest(s).")
        return

    if action == 'remove':
        updated = hashing.remove_seen(args.digests or [], store)
        print(f"Seen set now holds {len(updated)} digest(s).")
        return

    if action == 'clear':
        hashing.clear_seen(store)
        print("Seen set cleared.")
        return

    if action == 'check':
        seen = hashing.load_seen(store)
        exit_code = 0
        for digest in args.digests or []:
            present = hashing.digest_matches_set(digest, seen)
            print(f"{digest}: {'seen' if present else 'missing'}")
            if not present:
                exit_code = 1
        sys.exit(exit_code)


# ── delta / apply ─────────────────────────────────────────────────────

def cmd_delta(args: argparse.Namespace) -> None:
    from cli import delta as delta_mod

    base_text = read_text(args.base)
    target_text = read_text(args.target)
    try:
        rendered = delta_mod.build_delta_crumb(
            base_text,
            target_text,
            source=args.source or 'crumb.delta',
            title=args.title,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    write_text(args.output, rendered)
    if args.output != '-':
        print(f"Wrote delta crumb → {args.output} (~{estimate_tokens(rendered)} tokens)")


def cmd_apply(args: argparse.Namespace) -> None:
    from cli import delta as delta_mod

    base_text = read_text(args.base)
    delta_text = read_text(args.delta)
    try:
        rendered = delta_mod.apply_delta(base_text, delta_text, verify=not args.no_verify)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    write_text(args.output, rendered)
    if args.output != '-':
        print(f"Reconstructed target → {args.output} (~{estimate_tokens(rendered)} tokens)")


# ── Webhook commands ─────────────────────────────────────────────────

def cmd_webhook(args: argparse.Namespace) -> None:
    """Manage webhook subscriptions for AgentAuth events."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agentauth.webhooks import WebhookManager

    mgr = WebhookManager()
    action = args.webhook_action

    if action == 'add':
        wh = mgr.register(args.url, args.events)
        print(f"Webhook registered: {wh['id']}")
        print(f"  URL: {wh['url']}")
        print(f"  Events: {', '.join(wh['events'])}")

    elif action == 'list':
        hooks = mgr.list_hooks()
        if not hooks:
            print("No webhooks registered.")
            return
        print(f"{'ID':<16} {'URL':<40} Events")
        print("-" * 76)
        for h in hooks:
            print(f"{h['id']:<16} {h['url']:<40} {', '.join(h['events'])}")

    elif action == 'remove':
        ok = mgr.remove(args.webhook_id)
        if ok:
            print(f"Webhook {args.webhook_id} removed.")
        else:
            print(f"Webhook not found: {args.webhook_id}", file=sys.stderr)
            sys.exit(1)

    elif action == 'test':
        result = mgr.test(args.webhook_id)
        if result['success']:
            print(f"Test event sent to {result['url']} — status: {result.get('status_code', 'ok')}")
        else:
            print(f"Test failed: {result.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)


# ── Palace / Classify / Wake ─────────────────────────────────────────

def cmd_palace(args: argparse.Namespace) -> None:
    """Hierarchical spatial memory: wings / halls / rooms / tunnels."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cli import palace

    action = args.palace_action

    if action == 'init':
        target = Path(args.path or '.').resolve()
        root = palace.init_palace(target)
        print(f'Palace initialized at {root}')
        return

    root = palace.find_palace(Path(args.path or '.') if hasattr(args, 'path') and args.path else None)
    if root is None:
        print('error: no .crumb-palace found. Run `crumb palace init` first.', file=sys.stderr)
        sys.exit(1)

    if action == 'add':
        hall = args.hall
        if not hall:
            from cli.classify import classify
            hall = classify(args.text)
        path = palace.add_observation(root, args.wing, hall, args.room, args.text)
        palace.rebuild_index(root)
        relpath = path.relative_to(root.parent)
        print(f'Filed in {relpath} (hall={hall})')

    elif action == 'list':
        rooms = palace.list_rooms(root, wing=getattr(args, 'wing', None),
                                  hall=getattr(args, 'hall', None))
        if not rooms:
            print('(no rooms yet)')
            return
        for wing, hall, room, _ in rooms:
            print(f'{wing}/{hall}/{room}')

    elif action == 'search':
        results = palace.palace_search(root, args.query,
                                       wing=getattr(args, 'wing', None),
                                       hall=getattr(args, 'hall', None))
        if not results:
            print('no matches')
            sys.exit(1)
        for wing, hall, room, _, hits in results:
            print(f'{hits:3}x  {wing}/{hall}/{room}')

    elif action == 'tunnel':
        path = palace.rebuild_tunnels(root)
        print(f'Tunnels rebuilt → {path}')

    elif action == 'stats':
        stats = palace.palace_stats(root)
        print(f'Wings:  {stats["wings"]}')
        print(f'Rooms:  {stats["rooms"]}')
        print('By hall:')
        for h, c in stats['by_hall'].items():
            print(f'  {h:14} {c}')

    elif action == 'wiki':
        output = getattr(args, 'output', '-') or '-'
        cmd_palace_wiki(root, output)


def cmd_classify(args: argparse.Namespace) -> None:
    """Rule-based memory hall classifier."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cli.classify import classify, classify_batch, explain

    if args.text:
        if getattr(args, 'explain', False):
            info = explain(args.text)
            print(f'hall: {info["hall"]}')
            print(f'scores: {info["scores"]}')
            for h, pats in info['matched_patterns'].items():
                if pats:
                    print(f'  {h}: {pats}')
        else:
            print(classify(args.text))
    elif args.file:
        text = Path(args.file).read_text(encoding='utf-8')
        for hall, line in classify_batch(text.splitlines()):
            print(f'{hall:14} {line}')
    else:
        print('error: provide --text or --file', file=sys.stderr)
        sys.exit(1)


def cmd_wake(args: argparse.Namespace) -> None:
    """Emit a session wake-up crumb from the palace."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cli.palace import find_palace, build_wake_crumb

    root = find_palace(Path(args.path) if args.path else None)
    if root is None:
        print('error: no .crumb-palace found. Run `crumb palace init` first.', file=sys.stderr)
        sys.exit(1)

    wake_text = build_wake_crumb(root, max_facts=args.max_facts)

    # Inject reflection insights if palace has gaps
    if getattr(args, 'reflect', False):
        from cli.reflect import reflect
        report = reflect(root)
        if report.gaps:
            # Insert gap summary before END CRUMB
            gap_lines = [f"- [{g.priority}] {g.detail}" for g in report.gaps[:3]]
            gap_section = "\n[gaps]\n" + "\n".join(gap_lines) + "\n"
            wake_text = wake_text.replace("\nEND CRUMB", gap_section + "\nEND CRUMB")

    if args.metalk:
        from cli.metalk import encode
        wake_text = encode(wake_text, level=args.metalk_level)

    if args.output and args.output != '-':
        write_text(args.output, wake_text)
        print(f'Wake crumb written to {args.output}', file=sys.stderr)
    else:
        print(wake_text, end='')


def cmd_reflect(args: argparse.Namespace) -> None:
    """Analyze palace health and identify knowledge gaps."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cli.palace import find_palace
    from cli.reflect import reflect, render_report, render_report_crumb

    root = find_palace(Path(args.path) if args.path else None)
    if root is None:
        print('error: no .crumb-palace found. Run `crumb palace init` first.', file=sys.stderr)
        sys.exit(1)

    report = reflect(root, stale_days=args.stale_days)

    if args.format == 'crumb':
        output = render_report_crumb(report)
    else:
        output = render_report(report)

    if args.output and args.output != '-':
        write_text(args.output, output)
        print(f'Reflection written to {args.output}', file=sys.stderr)
    else:
        print(output)


def cmd_palace_wiki(root, args_output: str) -> None:
    """Generate a structured wiki index from palace contents."""
    from cli.palace import list_wings, list_halls, list_rooms, palace_stats, HALLS
    from cli.reflect import reflect

    stats = palace_stats(root)
    report = reflect(root)
    rooms = list_rooms(root)
    wings = list_wings(root)

    lines = [
        "BEGIN CRUMB",
        "v=1.1",
        "kind=map",
        f"title=Palace Wiki — {stats['wings']} wings, {stats['rooms']} rooms",
        "source=palace.wiki",
        "project=palace",
        "---",
        "[project]",
        f"Knowledge base wiki. Health: {report.health_score}/100 ({report.grade})",
        f"Generated by crumb palace wiki.",
        "",
        "[modules]",
    ]

    if not wings:
        lines.append("- (palace is empty — run `crumb palace add` to start)")
    else:
        for w in wings:
            wing_rooms = list_rooms(root, wing=w)
            wing_halls = list_halls(root, w)
            lines.append(f"- {w} ({len(wing_rooms)} rooms across {len(wing_halls)} halls):")

            # Group by hall
            by_hall: dict = {}
            for _, h, r, p in wing_rooms:
                by_hall.setdefault(h, []).append((r, p))

            for h in HALLS:
                if h not in by_hall:
                    continue
                room_names = [r for r, _ in sorted(by_hall[h])]
                lines.append(f"  - {h}: {', '.join(room_names)}")

            # Note missing halls
            missing = set(HALLS) - set(by_hall.keys())
            if missing:
                lines.append(f"  - (gaps: {', '.join(sorted(missing))})")

    # Append top gaps as actionable items
    if report.gaps:
        lines.append("")
        lines.append("- Top actions:")
        for gap in report.gaps[:5]:
            lines.append(f"  - [{gap.priority}] {gap.detail}")

    lines += ["", "END CRUMB"]
    wiki_text = "\n".join(lines) + "\n"

    if args_output and args_output != '-':
        write_text(args_output, wiki_text)
        print(f'Wiki written to {args_output}', file=sys.stderr)
    else:
        print(wiki_text, end='')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='crumb',
        description='Create, validate, inspect, and manage .crumb handoff files.',
    )
    parser.add_argument('--version', action='version', version=f'crumb {CLI_VERSION}')
    sub = parser.add_subparsers(dest='command', required=True)

    # new
    new = sub.add_parser('new', help='Create a new .crumb file.')
    new.add_argument('kind', choices=['task', 'mem', 'map', 'log', 'todo'], help='Kind of crumb to create.')
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
    from_chat = sub.add_parser('from-chat', help='Convert a chat log into a task or mem crumb.')
    from_chat.add_argument('--input', '-i', default='-', help='Input file or - for stdin.')
    from_chat.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    from_chat.add_argument('--title', help='Title for the crumb.')
    from_chat.add_argument('--source', help='Source label (e.g. chatgpt.chat, claude.chat).')
    from_chat.add_argument('--goal', help='Override the default goal text.')
    from_chat.add_argument('--kind', '-k', choices=['task', 'mem'], default='task',
                           help='Output kind: task (default) or mem (extracts decisions).')
    from_chat.add_argument('--constraints', '-c', nargs='*', help='Constraints as separate arguments.')
    from_chat.set_defaults(func=cmd_from_chat)

    # from-git
    from_git = sub.add_parser('from-git', help='Generate a task crumb from recent git activity.')
    from_git.add_argument('--commits', type=int, default=5, help='Number of recent commits to include (default: 5).')
    from_git.add_argument('--branch', help='Base branch to compare against (default: auto-detect main/master).')
    from_git.add_argument('--title', help='Override the auto-generated title.')
    from_git.add_argument('--source', help='Override source label (default: git).')
    from_git.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    from_git.set_defaults(func=cmd_from_git)

    # validate
    validate = sub.add_parser('validate', help='Validate one or more .crumb files.')
    validate.add_argument('files', nargs='+', help='.crumb files to validate.')
    validate.set_defaults(func=cmd_validate)

    # inspect
    inspect_cmd = sub.add_parser('inspect', help='Parse and display a .crumb file.')
    inspect_cmd.add_argument('file', nargs='?', help='.crumb file to inspect (default: stdin).')
    inspect_cmd.add_argument('--headers-only', '-H', action='store_true', help='Show only headers and section names.')
    inspect_cmd.set_defaults(func=cmd_inspect)

    # append
    append_cmd = sub.add_parser('append', help='Append raw observations to a mem crumb.')
    append_cmd.add_argument('file', help='Path to an existing kind=mem .crumb file.')
    append_cmd.add_argument('entries', nargs='+', help='Observations to append to [raw].')
    append_cmd.set_defaults(func=cmd_append)

    # dream
    dream = sub.add_parser('dream', help='Run a consolidation pass on a mem crumb.')
    dream.add_argument('file', help='Path to a kind=mem .crumb file.')
    dream.add_argument('--dry-run', action='store_true', help='Print result to stdout instead of writing.')
    dream.set_defaults(func=cmd_dream)

    # search
    search = sub.add_parser('search', help='Search .crumb files by keyword, fuzzy, or ranked.')
    search.add_argument('query', help='Search query (space-separated terms).')
    search.add_argument('--dir', default='.', help='Directory to search (default: current).')
    search.add_argument('--method', '-m', choices=['keyword', 'fuzzy', 'ranked'], default='keyword',
                        help='Search method: keyword (exact), fuzzy (approximate), ranked (TF-IDF).')
    search.add_argument('--limit', '-n', type=int, help='Max results to show.')
    search.set_defaults(func=cmd_search)

    # merge
    merge = sub.add_parser('merge', help='Merge multiple mem crumbs into one.')
    merge.add_argument('files', nargs='+', help='Mem .crumb files to merge.')
    merge.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    merge.add_argument('--title', '-t', help='Title for the merged crumb.')
    merge.set_defaults(func=cmd_merge)

    # compact
    compact = sub.add_parser('compact', help='Strip a crumb to minimum viable form.')
    compact.add_argument('file', nargs='?', help='.crumb file to compact (default: stdin).')
    compact.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    compact.set_defaults(func=cmd_compact)

    # diff
    diff = sub.add_parser('diff', help='Compare two .crumb files.')
    diff.add_argument('file_a', help='First .crumb file.')
    diff.add_argument('file_b', help='Second .crumb file.')
    diff.set_defaults(func=cmd_diff)

    # init
    init = sub.add_parser('init', help='Initialize CRUMB in a project.')
    init.add_argument('--dir', default='.', help='Project directory (default: current).')
    init.add_argument('--project', '-p', help='Project name (default: directory name).')
    init.add_argument('--description', '-d', help='One-line project description.')
    init.add_argument('--claude-md', action='store_true', help='Create/update CLAUDE.md with CRUMB instructions.')
    init.add_argument('--cursor-rules', action='store_true', help='Create .cursor/rules with CRUMB instructions.')
    init.add_argument('--windsurf-rules', action='store_true', help='Create .windsurfrules with CRUMB instructions.')
    init.add_argument('--chatgpt-rules', action='store_true', help='Print ChatGPT custom instructions.')
    init.add_argument('--gemini', action='store_true', help='Create .gemini/settings.json with CRUMB instructions.')
    init.add_argument('--copilot', action='store_true', help='Create .github/copilot-instructions.md with CRUMB instructions.')
    init.add_argument('--cody', action='store_true', help='Create .sourcegraph/cody.json with CRUMB instructions.')
    init.add_argument('--continue-dev', action='store_true', help='Create .continue/config.json with CRUMB system message.')
    init.add_argument('--aider', action='store_true', help='Create .aider.conf.yml with CRUMB conventions.')
    init.add_argument('--replit', action='store_true', help='Create .replit with CRUMB instructions.')
    init.add_argument('--devin', action='store_true', help='Create devin.md with CRUMB instructions.')
    init.add_argument('--bolt', action='store_true', help='Create .bolt/config.json with CRUMB instructions.')
    init.add_argument('--lovable', action='store_true', help='Create .lovable/config.json with CRUMB instructions.')
    init.add_argument('--all', dest='all_rules', action='store_true', help='Seed all AI tools at once.')
    init.set_defaults(func=cmd_init)

    # log
    log_cmd = sub.add_parser('log', help='Append timestamped entries to a log crumb.')
    log_cmd.add_argument('file', help='Path to a log crumb (created if missing).')
    log_cmd.add_argument('entries', nargs='+', help='Entries to log.')
    log_cmd.add_argument('--title', '-t', help='Title (for new log crumbs).')
    log_cmd.add_argument('--source', '-s', help='Source label.')
    log_cmd.set_defaults(func=cmd_log)

    # todo-add
    todo_add_cmd = sub.add_parser('todo-add', help='Add tasks to a todo crumb.')
    todo_add_cmd.add_argument('file', help='Path to a todo crumb (created if missing).')
    todo_add_cmd.add_argument('tasks', nargs='+', help='Tasks to add.')
    todo_add_cmd.add_argument('--title', '-t', help='Title (for new todo crumbs).')
    todo_add_cmd.add_argument('--source', '-s', help='Source label.')
    todo_add_cmd.set_defaults(func=cmd_todo_add)

    # todo-done
    todo_done_cmd = sub.add_parser('todo-done', help='Mark tasks as done by substring match.')
    todo_done_cmd.add_argument('file', help='Path to a todo crumb.')
    todo_done_cmd.add_argument('query', help='Substring to match against open tasks.')
    todo_done_cmd.set_defaults(func=cmd_todo_done)

    # todo-list
    todo_list_cmd = sub.add_parser('todo-list', help='List tasks from a todo crumb.')
    todo_list_cmd.add_argument('file', nargs='?', help='Path to a todo crumb.')
    todo_list_cmd.add_argument('--all', '-a', dest='show_all', action='store_true', help='Show completed tasks too.')
    todo_list_cmd.set_defaults(func=cmd_todo_list)

    # todo-dream
    todo_dream_cmd = sub.add_parser('todo-dream', help='Archive completed tasks to [archived] section.')
    todo_dream_cmd.add_argument('file', help='Path to a todo crumb.')
    todo_dream_cmd.set_defaults(func=cmd_todo_dream)

    # watch
    watch_cmd = sub.add_parser('watch', help='Watch crumbs and auto-dream when raw exceeds threshold.')
    watch_cmd.add_argument('target', help='File or directory to watch.')
    watch_cmd.add_argument('--threshold', type=int, default=5, help='Raw entries before auto-dream (default: 5).')
    watch_cmd.add_argument('--interval', type=int, default=3, help='Poll interval in seconds (default: 3).')
    watch_cmd.set_defaults(func=cmd_watch)

    # export
    export_cmd = sub.add_parser('export', help='Export a .crumb to another format.')
    export_cmd.add_argument('file', nargs='?', help='.crumb file to export (default: stdin).')
    export_cmd.add_argument('--format', '-f', required=True, choices=['json', 'markdown', 'clipboard'],
                            help='Output format.')
    export_cmd.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    export_cmd.set_defaults(func=cmd_export)

    # import
    import_cmd = sub.add_parser('import', help='Import from JSON or markdown into a .crumb.')
    import_cmd.add_argument('--from', required=True, choices=['json', 'markdown'],
                            help='Source format.', dest='from')
    import_cmd.add_argument('--input', '-i', default='-', help='Input file or - for stdin.')
    import_cmd.add_argument('--output', '-o', default='-', help='Output file or - for stdout.')
    import_cmd.set_defaults(func=cmd_import)

    # hooks
    hooks_cmd = sub.add_parser('hooks', help='Show configured hooks from .crumbrc.')
    hooks_cmd.add_argument('--dir', default='.', help='Project directory (default: current).')
    hooks_cmd.set_defaults(func=cmd_hooks)

    # template
    template_cmd = sub.add_parser('template', help='Manage crumb templates.')
    template_cmd.add_argument('action', choices=['list', 'use', 'add'], help='Template action.')
    template_cmd.add_argument('name', nargs='?', help='Template name (for use/add).')
    template_cmd.add_argument('source_file', nargs='?', help='Source .crumb file (for add).')
    template_cmd.add_argument('--output', '-o', default='-', help='Output file (for use).')
    template_cmd.set_defaults(func=cmd_template)

    # share
    # compress
    compress_cmd = sub.add_parser('compress', help='Two-stage context compression (dedup + signal pruning).')
    compress_cmd.add_argument('file', help='.crumb file to compress.')
    compress_cmd.add_argument('-o', '--output', default='-', help='Output path (default: stdout).')
    compress_cmd.add_argument('--target', type=float, default=0.5,
                              help='Target retention ratio 0.0-1.0 (default: 0.5 = keep top 50%%).')
    compress_cmd.add_argument('--metalk', action='store_true',
                              help='Apply MeTalk caveman compression as Stage 3.')
    compress_cmd.add_argument('--metalk-level', type=int, choices=[1, 2, 3], default=2,
                              help='MeTalk level (default: 2).')
    compress_cmd.set_defaults(func=cmd_compress)

    # bench
    bench_cmd = sub.add_parser('bench', help='Benchmark compression efficiency and information density.')
    bench_cmd.add_argument('file', help='.crumb file to benchmark.')
    bench_cmd.set_defaults(func=cmd_bench)

    share_cmd = sub.add_parser('share', help='Share a .crumb file via GitHub Gist or data URI.')
    share_cmd.add_argument('file', help='.crumb file to share.')
    share_cmd.set_defaults(func=cmd_share)

    # handoff
    handoff_cmd = sub.add_parser('handoff', help='Copy a .crumb to clipboard for pasting into an AI tool.')
    handoff_cmd.add_argument('file', help='.crumb file to hand off.')
    handoff_cmd.add_argument('--target', choices=['claude', 'cursor', 'chatgpt', 'gemini'],
                             help='Target AI tool (optional).')
    handoff_cmd.set_defaults(func=cmd_handoff)

    # --- Receive ---
    receive_cmd = sub.add_parser('receive', help='Receive a crumb from clipboard/file/stdin, validate, optionally file to palace.')
    receive_cmd.add_argument('--file', help='Read crumb from this file instead of clipboard.')
    receive_cmd.add_argument('-o', '--output', default=None, help='Save received crumb to file.')
    receive_cmd.add_argument('--palace', action='store_true', help='Auto-file observations into the palace.')
    receive_cmd.add_argument('--wing', help='Palace wing (default: derived from source header).')
    receive_cmd.add_argument('--hall', choices=['facts', 'events', 'discoveries', 'preferences', 'advice'],
                             help='Palace hall (default: auto-classified).')
    receive_cmd.set_defaults(func=cmd_receive)

    # --- Context ---
    context_cmd = sub.add_parser('context', help='Generate a task crumb from current project state (git, palace, todos).')
    context_cmd.add_argument('--commits', type=int, default=5, help='Number of recent commits to include (default: 5).')
    context_cmd.add_argument('--goal', help='Override the auto-detected goal.')
    context_cmd.add_argument('--title', help='Override the auto-generated title.')
    context_cmd.add_argument('--source', help='Override source label (default: crumb.context).')
    context_cmd.add_argument('-o', '--output', default='-', help='Output file or - for stdout.')
    context_cmd.add_argument('--metalk', action='store_true', help='Apply MeTalk compression.')
    context_cmd.add_argument('--metalk-level', type=int, choices=[1, 2, 3], default=2,
                             help='MeTalk level (default: 2).')
    context_cmd.add_argument('--clipboard', action='store_true', help='Copy result to clipboard instead of printing.')
    context_cmd.add_argument('--max-facts', type=int, default=8, help='Max palace facts to include (default: 8).')
    context_cmd.set_defaults(func=cmd_context)

    # --- Agent Passport ---
    passport_cmd = sub.add_parser('passport', help='Agent identity management.')
    passport_sub = passport_cmd.add_subparsers(dest='passport_action', required=True)

    reg = passport_sub.add_parser('register', help='Register an agent, issue passport.')
    reg.add_argument('name', help='Agent name.')
    reg.add_argument('--framework', default='unknown')
    reg.add_argument('--owner', default='')
    reg.add_argument('--tools-allowed', nargs='*', default=[])
    reg.add_argument('--tools-denied', nargs='*', default=[])
    reg.add_argument('--ttl-days', type=int, default=90)
    reg.add_argument('-o', '--output', default=None)
    passport_cmd.set_defaults(func=cmd_passport)

    insp = passport_sub.add_parser('inspect', help='View agent passport.')
    insp.add_argument('agent_id', help='Agent ID or name.')

    rev = passport_sub.add_parser('revoke', help='Revoke passport (kill switch).')
    rev.add_argument('agent_id', help='Agent ID to revoke.')

    lst = passport_sub.add_parser('list', help='List registered agents.')
    lst.add_argument('--status', choices=['active', 'revoked', 'expired', 'all'], default='all')

    # --- Policy ---
    policy_cmd = sub.add_parser('policy', help='Tool authorization policy.')
    policy_sub = policy_cmd.add_subparsers(dest='policy_action', required=True)

    ps = policy_sub.add_parser('set', help='Set tool authorization rules.')
    ps.add_argument('agent_name')
    ps.add_argument('--allow', nargs='*', default=[])
    ps.add_argument('--deny', nargs='*', default=[])
    ps.add_argument('--max-actions', type=int, default=1000)
    policy_cmd.set_defaults(func=cmd_policy)

    pt = policy_sub.add_parser('test', help='Simulate action against policy.')
    pt.add_argument('agent_name')
    pt.add_argument('tool')

    # --- Audit ---
    audit_cmd = sub.add_parser('audit', help='Audit trail management.')
    audit_sub = audit_cmd.add_subparsers(dest='audit_action', required=True)

    ae = audit_sub.add_parser('export', help='Export audit evidence pack.')
    ae.add_argument('--agent', default=None)
    ae.add_argument('--since', default=None)
    ae.add_argument('-f', '--format', choices=['crumb', 'json', 'csv'], default='crumb')
    ae.add_argument('-o', '--output', default='-')
    audit_cmd.set_defaults(func=cmd_audit)

    af = audit_sub.add_parser('feed', help='Live action feed.')
    af.add_argument('--agent', default=None)

    # --- Shadow AI Scanner ---
    scan_cmd = sub.add_parser('scan', help='Shadow AI scanner: discover unauthorized AI agents in a project.')
    scan_cmd.add_argument('--path', default='.', help='Directory to scan (default: current directory).')
    scan_cmd.add_argument('--format', '-f', choices=['text', 'json', 'crumb'], default='text',
                          help='Output format (default: text).')
    scan_cmd.add_argument('--min-risk', choices=['low', 'medium', 'high', 'critical'], default='low',
                          help='Minimum risk level to report (default: low).')
    scan_cmd.set_defaults(func=cmd_scan)

    # --- Compliance Report ---
    comply_cmd = sub.add_parser('comply', help='Generate compliance report from agent data.')
    comply_cmd.add_argument('-f', '--format', choices=['text', 'json', 'html'], default='text')
    comply_cmd.add_argument('-o', '--output', default='-')
    comply_cmd.add_argument('--framework', choices=['general', 'eu-ai-act', 'soc2'], default='general')
    comply_cmd.set_defaults(func=cmd_comply)

    # --- Dashboard ---
    dash_cmd = sub.add_parser('dashboard', help='Generate agent dashboard (HTML).')
    dash_cmd.add_argument('-o', '--output', default='agentauth-dashboard.html')
    dash_cmd.set_defaults(func=cmd_dashboard)

    # --- Format Bridges ---
    bridge_cmd = sub.add_parser('bridge', help='Convert between CRUMB and other AI formats.')
    bridge_sub = bridge_cmd.add_subparsers(dest='bridge_action', required=True)

    br_export = bridge_sub.add_parser('export', help='Export CRUMB to another format.')
    br_export.add_argument('--to', required=True,
                           choices=['openai-threads', 'langchain-memory', 'crewai-task', 'autogen', 'claude-project'],
                           help='Target format.')
    br_export.add_argument('input', help='Input .crumb file.')
    br_export.add_argument('-o', '--output', default='-', help='Output file.')
    bridge_cmd.set_defaults(func=cmd_bridge)

    br_import = bridge_sub.add_parser('import', help='Import from another format to CRUMB.')
    br_import.add_argument('--from', dest='source_format', required=True,
                           choices=['openai-threads', 'langchain-memory', 'crewai-task', 'autogen'],
                           help='Source format.')
    br_import.add_argument('input', help='Input file.')
    br_import.add_argument('-o', '--output', default='-', help='Output .crumb file.')

    br_list = bridge_sub.add_parser('list', help='List available bridge formats.')

    # Mempalace adapter subtree (bridge mempalace {export,import})
    br_mempalace = bridge_sub.add_parser('mempalace', help='MemPalace bridge adapter for CRUMB import/export.')
    mp_sub = br_mempalace.add_subparsers(dest='mempalace_action', required=True)

    mp_export = mp_sub.add_parser('export', help='Export MemPalace context as one or more CRUMBs.')
    mp_export.add_argument('--query', help='Search query to run against MemPalace.')
    mp_export.add_argument('--input', help='Saved MemPalace export text or - for stdin.')
    mp_export.add_argument('--wing', help='Optional MemPalace wing filter.')
    mp_export.add_argument('--room', help='Optional room filter applied to retrieved lines.')
    mp_export.add_argument('--entity', help='Optional entity filter applied to retrieved lines.')
    mp_export.add_argument('--hall', help='Optional MemPalace hall filter when using the CLI backend.')
    mp_export.add_argument('--project', help='Optional project header override.')
    mp_export.add_argument('--title', help='Optional title override.')
    mp_export.add_argument('--as', dest='as_kind', required=True, choices=['task', 'mem', 'log'],
                           help='Output CRUMB kind to emit.')
    mp_export.add_argument('--output', '-o', required=True,
                           help='Output file path or directory for generated .crumb files.')
    mp_export.set_defaults(backend='mempalace')

    mp_import = mp_sub.add_parser('import', help='Convert .crumb files into a MemPalace-compatible adapter bundle.')
    mp_import.add_argument('files', nargs='+', help='One or more .crumb files to convert.')
    mp_import.add_argument('--wing', help='Target MemPalace wing (default: default).')
    mp_import.add_argument('--room', help='Optional room override.')
    mp_import.add_argument('--entity', help='Optional entity override.')
    mp_import.add_argument('--output', '-o', required=True,
                           help='Output JSON file path or directory for the adapter bundle.')
    mp_import.set_defaults(backend='mempalace')

    # --- Pack ---
    pack_cmd = sub.add_parser('pack', help='Assemble a deterministic CRUMB context pack from a directory of crumbs.')
    pack_cmd.add_argument('--dir', required=True, help='Directory containing source .crumb files.')
    pack_cmd.add_argument('--query', required=True, help='Query describing the handoff you want to build.')
    pack_cmd.add_argument('--project', help='Optional project filter/header to apply while selecting crumbs.')
    pack_cmd.add_argument('--kind', required=True, choices=['task', 'mem', 'map'],
                          help='Output kind for the packed CRUMB.')
    pack_cmd.add_argument('--mode', choices=['implement', 'debug', 'review'], default='implement',
                          help='Pack shaping mode: implement (default), debug, or review.')
    pack_cmd.add_argument('--max-total-tokens', type=int, required=True,
                          help='Estimated token budget for the final packed CRUMB.')
    pack_cmd.add_argument('--strategy', choices=['keyword', 'ranked', 'recent', 'hybrid'], default='hybrid',
                          help='Ranking strategy for selecting and merging context (default: hybrid).')
    pack_cmd.add_argument('--title', help='Optional title override for the packed CRUMB.')
    pack_cmd.add_argument('--ollama', '--use-local', dest='ollama', action='store_true',
                          help='Optionally run a final local-model compression pass with Ollama.')
    pack_cmd.add_argument('--ollama-model', default='llama3.2:3b',
                          help='Local Ollama model to use when --ollama is set (default: llama3.2:3b).')
    pack_cmd.add_argument('--output', '-o', required=True, help='Output .crumb file path.')
    pack_cmd.set_defaults(func=cmd_pack)

    # --- Squeeze (budget-aware packer) ---
    squeeze_cmd = sub.add_parser('squeeze',
                                 help='Compress a CRUMB to fit a token budget (drops folds, elides seen refs, escalates MeTalk).')
    squeeze_cmd.add_argument('file', help='.crumb file to squeeze.')
    squeeze_cmd.add_argument('--budget', type=int, required=True,
                             help='Target estimated token budget.')
    squeeze_cmd.add_argument('--seen', help='Path to a seen-set file (defaults to ~/.crumb/seen or $CRUMB_SEEN_FILE).')
    squeeze_cmd.add_argument('--seen-hash', action='append', default=[],
                             help='Extra sha256:<hex> digests to treat as already-seen. Repeatable.')
    squeeze_cmd.add_argument('--no-seen', action='store_true',
                             help='Ignore any persisted seen set.')
    squeeze_cmd.add_argument('--metalk-max-level', type=int, choices=[1, 2, 3], default=3,
                             help='Highest MeTalk level to apply if folds + priorities are not enough (default: 3).')
    squeeze_cmd.add_argument('--dry-run', action='store_true',
                             help='Print the compression report without writing output.')
    squeeze_cmd.add_argument('-o', '--output', default='-', help='Output file or - for stdout.')
    squeeze_cmd.set_defaults(func=cmd_squeeze)

    # --- Hash (content-addressed digest) ---
    hash_cmd = sub.add_parser('hash', help='Print the sha256 content digest of a CRUMB.')
    hash_cmd.add_argument('file', help='.crumb file to hash.')
    hash_cmd.add_argument('--short', type=int,
                          help='Print a shortened sha256:<hex> digest truncated to this many hex chars.')
    hash_cmd.set_defaults(func=cmd_hash)

    # --- Seen set (content-addressed ref registry) ---
    seen_cmd = sub.add_parser('seen', help='Manage the content-addressed seen-set registry.')
    seen_sub = seen_cmd.add_subparsers(dest='seen_action', required=True)
    seen_add = seen_sub.add_parser('add', help='Add sha256:<hex> digests (or digests derived from files) to the seen set.')
    seen_add.add_argument('digests', nargs='*', help='sha256:<hex> digests to add.')
    seen_add.add_argument('--from-file', action='append', default=[],
                          help='Derive a digest from this .crumb file and add it. Repeatable.')
    seen_add.add_argument('--store', help='Override the seen-set file path.')
    seen_remove = seen_sub.add_parser('remove', help='Remove digests from the seen set.')
    seen_remove.add_argument('digests', nargs='+')
    seen_remove.add_argument('--store')
    seen_list = seen_sub.add_parser('list', help='List digests in the seen set.')
    seen_list.add_argument('--store')
    seen_check = seen_sub.add_parser('check', help='Exit non-zero if any digest is missing from the seen set.')
    seen_check.add_argument('digests', nargs='+')
    seen_check.add_argument('--store')
    seen_clear = seen_sub.add_parser('clear', help='Empty the seen set.')
    seen_clear.add_argument('--store')
    seen_cmd.set_defaults(func=cmd_seen)

    # --- Delta / Apply ---
    delta_cmd = sub.add_parser('delta', help='Compute a kind=delta CRUMB between two CRUMBs.')
    delta_cmd.add_argument('base', help='Base .crumb file.')
    delta_cmd.add_argument('target', help='Target .crumb file.')
    delta_cmd.add_argument('--title', help='Optional title for the delta crumb.')
    delta_cmd.add_argument('--source', help='Source header (default: crumb.delta).')
    delta_cmd.add_argument('-o', '--output', default='-', help='Output file or - for stdout.')
    delta_cmd.set_defaults(func=cmd_delta)

    apply_cmd = sub.add_parser('apply', help='Apply a kind=delta CRUMB to a base CRUMB and reconstruct the target.')
    apply_cmd.add_argument('base', help='Base .crumb file.')
    apply_cmd.add_argument('delta', help='Delta .crumb file (kind=delta).')
    apply_cmd.add_argument('--no-verify', action='store_true',
                           help='Skip sha256 verification of base and reconstructed target digests.')
    apply_cmd.add_argument('-o', '--output', default='-', help='Output file or - for stdout.')
    apply_cmd.set_defaults(func=cmd_apply)

    # --- Lint ---
    lint_cmd = sub.add_parser('lint', help='Lint CRUMBs for secrets, oversized raw logs, suspicious headers, and budget issues.')
    lint_cmd.add_argument('files', nargs='+', help='One or more .crumb files to lint.')
    lint_cmd.add_argument('--secrets', action='store_true', help='Enable secret detection checks.')
    lint_cmd.add_argument('--redact', action='store_true', help='Redact obvious credentials in-place unless --output is set.')
    lint_cmd.add_argument('--max-size', type=int,
                          help='Warn when estimated total or raw section tokens exceed this value.')
    lint_cmd.add_argument('--strict', action='store_true', help='Return a non-zero exit code for warnings.')
    lint_cmd.add_argument('--output', help='Optional output file or directory for redacted content.')
    lint_cmd.set_defaults(func=cmd_lint)

    # --- Webhooks ---
    wh_cmd = sub.add_parser('webhook', help='Manage AgentAuth event webhooks.')
    wh_sub = wh_cmd.add_subparsers(dest='webhook_action', required=True)

    wh_add = wh_sub.add_parser('add', help='Register a webhook.')
    wh_add.add_argument('url', help='Webhook URL.')
    wh_add.add_argument('--events', nargs='+', required=True,
                        help='Events to subscribe to (e.g. passport.revoked policy.denied).')
    wh_cmd.set_defaults(func=cmd_webhook)

    wh_list = wh_sub.add_parser('list', help='List registered webhooks.')
    wh_remove = wh_sub.add_parser('remove', help='Remove a webhook.')
    wh_remove.add_argument('webhook_id', help='Webhook ID to remove.')
    wh_test = wh_sub.add_parser('test', help='Send a test event to a webhook.')
    wh_test.add_argument('webhook_id', help='Webhook ID to test.')

    # --- MeTalk ---
    mt_cmd = sub.add_parser('metalk', help='MeTalk caveman compression — reduce tokens for AI-to-AI communication.')
    mt_cmd.add_argument('file', help='.crumb file to encode/decode.')
    mt_cmd.add_argument('--level', type=int, choices=[1, 2, 3], default=2,
                        help='1=dict only (lossless), 2=dict+grammar strip, 3=aggressive (default: 2).')
    mt_cmd.add_argument('--decode', action='store_true', help='Decode MeTalk back to full form.')
    mt_cmd.add_argument('-o', '--output', default='-', help='Output path.')
    mt_cmd.set_defaults(func=cmd_metalk)

    # --- Palace ---
    pal_cmd = sub.add_parser('palace', help='Hierarchical spatial memory: wings / halls / rooms / tunnels.')
    pal_sub = pal_cmd.add_subparsers(dest='palace_action', required=True)

    pal_init = pal_sub.add_parser('init', help='Initialize a .crumb-palace in the current directory.')
    pal_init.add_argument('--path', help='Parent directory (default: cwd).')

    pal_add = pal_sub.add_parser('add', help='File an observation into a room (creates room if missing).')
    pal_add.add_argument('text', help='Observation text.')
    pal_add.add_argument('--wing', required=True, help='Wing name (person/project/topic).')
    pal_add.add_argument('--room', required=True, help='Room name (specific topic).')
    pal_add.add_argument('--hall', choices=['facts', 'events', 'discoveries', 'preferences', 'advice'],
                         help='Hall (auto-classified via `crumb classify` if omitted).')
    pal_add.add_argument('--path', help='Start directory for palace lookup.')

    pal_list = pal_sub.add_parser('list', help='List rooms.')
    pal_list.add_argument('--wing', help='Filter by wing.')
    pal_list.add_argument('--hall', choices=['facts', 'events', 'discoveries', 'preferences', 'advice'])
    pal_list.add_argument('--path', help='Start directory.')

    pal_search = pal_sub.add_parser('search', help='Substring search across room bodies.')
    pal_search.add_argument('query', help='Search query.')
    pal_search.add_argument('--wing', help='Restrict to one wing.')
    pal_search.add_argument('--hall', choices=['facts', 'events', 'discoveries', 'preferences', 'advice'])
    pal_search.add_argument('--path', help='Start directory.')

    pal_tunnel = pal_sub.add_parser('tunnel', help='Rebuild cross-wing tunnel index.')
    pal_tunnel.add_argument('--path', help='Start directory.')

    pal_stats = pal_sub.add_parser('stats', help='Show palace statistics.')
    pal_stats.add_argument('--path', help='Start directory.')

    pal_wiki = pal_sub.add_parser('wiki', help='Generate a structured knowledge index from palace contents.')
    pal_wiki.add_argument('--path', help='Start directory.')
    pal_wiki.add_argument('-o', '--output', default='-', help='Output file or - for stdout.')

    pal_cmd.set_defaults(func=cmd_palace)

    # --- Classify ---
    cls_cmd = sub.add_parser('classify', help='Rule-based memory hall classifier.')
    cls_cmd.add_argument('--text', help='Text to classify.')
    cls_cmd.add_argument('--file', help='File of lines to classify (one per line).')
    cls_cmd.add_argument('--explain', action='store_true', help='Show matched patterns and scores.')
    cls_cmd.set_defaults(func=cmd_classify)

    # --- Wake ---
    wake_cmd = sub.add_parser('wake', help='Emit a session wake-up crumb from the palace.')
    wake_cmd.add_argument('--path', help='Start directory for palace lookup.')
    wake_cmd.add_argument('-o', '--output', default='-', help='Output file or - for stdout.')
    wake_cmd.add_argument('--max-facts', type=int, default=8, help='Max facts to include (default: 8).')
    wake_cmd.add_argument('--metalk', action='store_true', help='Pipe output through MeTalk compression.')
    wake_cmd.add_argument('--metalk-level', type=int, choices=[1, 2, 3], default=2)
    wake_cmd.add_argument('--reflect', action='store_true',
                          help='Include top knowledge gaps in the wake crumb.')
    wake_cmd.set_defaults(func=cmd_wake)

    # --- Reflect ---
    reflect_cmd = sub.add_parser('reflect', help='Analyze palace health and identify knowledge gaps (self-learning).')
    reflect_cmd.add_argument('--path', help='Start directory for palace lookup.')
    reflect_cmd.add_argument('-o', '--output', default='-', help='Output file or - for stdout.')
    reflect_cmd.add_argument('--format', '-f', choices=['text', 'crumb'], default='text',
                             help='Output format (default: text).')
    reflect_cmd.add_argument('--stale-days', type=int, default=30,
                             help='Days before a room is considered stale (default: 30).')
    reflect_cmd.set_defaults(func=cmd_reflect)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

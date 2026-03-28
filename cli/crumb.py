#!/usr/bin/env python3
"""Minimal CLI for creating, validating, inspecting, and managing .crumb handoff files."""

import argparse
import datetime
import difflib
import glob
import json
import os
import re
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
    """Score an entry by information density (TurboQuant-inspired signal ranking).

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

    # Prune to budget using signal scoring (TurboQuant-inspired)
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
    if args.dry_run:
        sys.stdout.write(output)
    else:
        path.write_text(output, encoding='utf-8')
        print(f"Dream pass complete on {args.file}")
        print(f"  {len(existing)} existing + {len(raw)} raw → {len(merged)} consolidated")
        if pruned:
            print(f"  Pruned {pruned} entries to fit budget ({budget} tokens)")
        run_hook('post_dream', {'file': args.file, 'entries': str(len(merged))})


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

# TurboQuant-inspired: eliminate overhead. Strip a crumb to its minimum
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
    if args.claude_md:
        claude_md_path = project_dir / 'CLAUDE.md'
        if claude_md_path.exists():
            existing = claude_md_path.read_text(encoding='utf-8')
            if 'CRUMB' not in existing:
                claude_md_path.write_text(existing.rstrip() + '\n\n' + CLAUDE_MD_SNIPPET, encoding='utf-8')
                print(f"  Appended CRUMB section to {claude_md_path}")
            else:
                print(f"  Skipped {claude_md_path} (already has CRUMB section)")
        else:
            claude_md_path.write_text(CLAUDE_MD_SNIPPET, encoding='utf-8')
            print(f"  Created {claude_md_path}")

    # Print custom instruction snippets
    print(f"\n--- Sender instruction (add to your AI's custom instructions) ---\n")
    print(SENDER_INSTRUCTION)
    print(f"--- Receiver instruction (add to the receiving AI) ---\n")
    print(RECEIVER_INSTRUCTION)

    print(f"Done. CRUMB initialized in {project_dir}")
    print(f"  - Map crumb: {map_path}")
    print(f"  - Add sender instruction to your AI's custom instructions")
    print(f"  - Add receiver instruction to the AI that receives crumbs")


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


# ── argument parser ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='crumb',
        description='Create, validate, inspect, and manage .crumb handoff files.',
    )
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
    init.add_argument('--claude-md', action='store_true', help='Also create/update CLAUDE.md with CRUMB instructions.')
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

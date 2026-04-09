"""Helpers for CRUMB Capsules, Relay timelines, and filesystem brain bridges.

This module is intentionally additive. It does not change the core CRUMB
format or existing commands. It gives the repo a sharable, resumable layer
that can later be wired into the main CLI.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any

from crumb import parse_crumb, render_crumb, estimate_tokens
from metalk import encode as metalk_encode


def _safe_slug(text: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '-' for ch in text).strip('-')
    while '--' in cleaned:
        cleaned = cleaned.replace('--', '-')
    return cleaned or 'crumb-capsule'


def _first_nonempty(lines: list[str]) -> str:
    for line in lines:
        if line.strip():
            return line.strip()
    return ''


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def build_capsule(path: str, target: str | None = None, metalk_level: int = 2) -> dict[str, Any]:
    crumb_path = Path(path)
    text = crumb_path.read_text(encoding='utf-8')
    parsed = parse_crumb(text)
    headers = parsed['headers']
    sections = parsed['sections']

    title = headers.get('title') or crumb_path.stem
    kind = headers['kind']
    goal = _first_nonempty(sections.get('goal', []))
    context_lines = [line.strip() for line in sections.get('context', []) if line.strip()][:5]
    constraint_lines = [line.strip() for line in sections.get('constraints', []) if line.strip()][:5]

    encoded = metalk_encode(text, level=metalk_level)
    capsule_id = headers.get('id') or _sha12(text)

    preview = {
        'goal': goal,
        'context': context_lines,
        'constraints': constraint_lines,
    }

    return {
        'id': capsule_id,
        'title': title,
        'slug': _safe_slug(title),
        'kind': kind,
        'source': headers.get('source', ''),
        'target': target or 'any',
        'path': str(crumb_path),
        'original': text,
        'encoded': encoded,
        'original_tokens': estimate_tokens(text),
        'encoded_tokens': estimate_tokens(encoded),
        'compression_ratio': round(
            estimate_tokens(text) / max(estimate_tokens(encoded), 1), 2
        ),
        'headers': dict(headers),
        'preview': preview,
    }


def capsule_to_markdown(capsule: dict[str, Any]) -> str:
    preview = capsule['preview']
    lines = [
        f"# CRUMB Capsule — {capsule['title']}",
        '',
        f"- id: `{capsule['id']}`",
        f"- kind: `{capsule['kind']}`",
        f"- source: `{capsule['source']}`",
        f"- target: `{capsule['target']}`",
        f"- original tokens: ~{capsule['original_tokens']}",
        f"- capsule tokens: ~{capsule['encoded_tokens']}",
        f"- compression ratio: {capsule['compression_ratio']}x",
        '',
        '## Preview',
    ]
    if preview['goal']:
        lines.extend(['', '**Goal**', preview['goal']])
    if preview['context']:
        lines.extend(['', '**Context**'])
        lines.extend(f"- {line}" for line in preview['context'])
    if preview['constraints']:
        lines.extend(['', '**Constraints**'])
        lines.extend(f"- {line}" for line in preview['constraints'])

    lines.extend([
        '',
        '## Resume payload',
        '',
        '```text',
        capsule['encoded'].rstrip(),
        '```',
        '',
        '## Resume targets',
        '',
        '- ChatGPT: paste the payload into a new chat',
        '- Claude: paste the payload into a new chat or project',
        '- Cursor: paste into agent chat',
        '- Gemini: paste into a new session',
        '',
    ])
    return '\n'.join(lines)


def capsule_to_html(capsule: dict[str, Any]) -> str:
    preview = capsule['preview']

    def bullets(items: list[str]) -> str:
        if not items:
            return '<li><em>None</em></li>'
        return ''.join(f'<li>{html.escape(item)}</li>' for item in items)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>CRUMB Capsule — {html.escape(capsule['title'])}</title>
  <style>
    body {{
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      max-width: 920px;
      margin: 40px auto;
      padding: 0 20px 40px;
      background: #0b0f14;
      color: #e8eef6;
    }}
    .card {{
      background: #121923;
      border: 1px solid #263243;
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }}
    .muted {{ color: #97a6ba; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0 24px;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border: 1px solid #33465d;
      border-radius: 999px;
      font-size: 12px;
      color: #b8c6d8;
      margin-right: 8px;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #0f141c;
      border: 1px solid #263243;
      border-radius: 12px;
      padding: 16px;
      overflow-x: auto;
    }}
    h1, h2, h3 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <div>
      <span class=\"pill\">{html.escape(capsule['kind'])}</span>
      <span class=\"pill\">target: {html.escape(capsule['target'])}</span>
      <span class=\"pill\">id: {html.escape(capsule['id'])}</span>
    </div>
    <h1>{html.escape(capsule['title'])}</h1>
    <p class=\"muted\">Portable AI handoff capsule generated from CRUMB.</p>

    <div class=\"meta\">
      <div><strong>Source</strong><br><span class=\"muted\">{html.escape(capsule['source'])}</span></div>
      <div><strong>Original tokens</strong><br><span class=\"muted\">~{capsule['original_tokens']}</span></div>
      <div><strong>Capsule tokens</strong><br><span class=\"muted\">~{capsule['encoded_tokens']}</span></div>
      <div><strong>Compression ratio</strong><br><span class=\"muted\">{capsule['compression_ratio']}x</span></div>
    </div>

    <h2>Goal</h2>
    <p>{html.escape(preview['goal'] or 'No explicit goal in this crumb.')}</p>

    <h2>Context</h2>
    <ul>{bullets(preview['context'])}</ul>

    <h2>Constraints</h2>
    <ul>{bullets(preview['constraints'])}</ul>

    <h2>Resume payload</h2>
    <pre>{html.escape(capsule['encoded'])}</pre>
  </div>
</body>
</html>
"""


def write_capsule_bundle(
    crumb_path: str,
    output_dir: str,
    target: str | None = None,
    metalk_level: int = 2,
) -> dict[str, str]:
    capsule = build_capsule(crumb_path, target=target, metalk_level=metalk_level)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = capsule['slug']
    markdown_path = out_dir / f'{stem}.capsule.md'
    html_path = out_dir / f'{stem}.capsule.html'
    payload_path = out_dir / f'{stem}.capsule.txt'
    meta_path = out_dir / f'{stem}.capsule.json'

    markdown_path.write_text(capsule_to_markdown(capsule), encoding='utf-8')
    html_path.write_text(capsule_to_html(capsule), encoding='utf-8')
    payload_path.write_text(capsule['encoded'], encoding='utf-8')
    meta_path.write_text(json.dumps(capsule, indent=2), encoding='utf-8')

    return {
        'markdown': str(markdown_path),
        'html': str(html_path),
        'payload': str(payload_path),
        'json': str(meta_path),
    }


def build_relay(directory: str) -> dict[str, Any]:
    root = Path(directory)
    events = []
    for path in sorted(root.rglob('*.crumb')):
        try:
            parsed = parse_crumb(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        headers = parsed['headers']
        sections = parsed['sections']
        timestamp = headers.get('dream_pass') or headers.get('issued') or headers.get('generated') or ''
        preview = _first_nonempty(sections.get('goal', [])) or _first_nonempty(
            sections.get('consolidated', [])
        ) or _first_nonempty(sections.get('project', []))
        events.append(
            {
                'path': str(path),
                'title': headers.get('title', path.stem),
                'kind': headers.get('kind', 'unknown'),
                'source': headers.get('source', ''),
                'timestamp': timestamp,
                'preview': preview,
            }
        )
    return {'count': len(events), 'events': events}


def relay_to_markdown(relay: dict[str, Any]) -> str:
    lines = ['# CRUMB Relay', '', f"- events: {relay['count']}", '']
    if not relay['events']:
        lines.append('No CRUMB events found.')
        return '\n'.join(lines)

    for idx, event in enumerate(relay['events'], start=1):
        label = event['timestamp'] or f'event-{idx}'
        lines.extend(
            [
                f"## {idx}. {event['title']}",
                '',
                f"- when: {label}",
                f"- kind: {event['kind']}",
                f"- source: {event['source']}",
                f"- path: `{event['path']}`",
                f"- preview: {event['preview'] or 'n/a'}",
                '',
            ]
        )
    return '\n'.join(lines)


def save_to_brain(crumb_path: str, brain_dir: str, workspace: str = 'default') -> dict[str, Any]:
    source = Path(crumb_path)
    if not source.exists():
        raise FileNotFoundError(f'crumb not found: {source}')

    parsed = parse_crumb(source.read_text(encoding='utf-8'))
    title = parsed['headers'].get('title', source.stem)

    root = Path(brain_dir) / workspace
    memory_dir = root / 'crumbs'
    memory_dir.mkdir(parents=True, exist_ok=True)

    dest = memory_dir / source.name
    shutil.copy2(source, dest)

    index_path = root / 'index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
    else:
        index = {'workspace': workspace, 'items': []}

    item = {
        'title': title,
        'kind': parsed['headers']['kind'],
        'source': parsed['headers'].get('source', ''),
        'path': str(dest),
        'id': parsed['headers'].get('id', _sha12(dest.read_text(encoding='utf-8'))),
    }
    index['items'] = [existing for existing in index['items'] if existing.get('path') != str(dest)]
    index['items'].append(item)
    index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')
    return item


def recall_from_brain(
    brain_dir: str,
    query: str,
    workspace: str = 'default',
    kind: str = 'task',
    top_k: int = 5,
) -> str:
    root = Path(brain_dir) / workspace / 'crumbs'
    if not root.exists():
        raise FileNotFoundError(f'brain workspace not found: {root}')

    query_terms = [term for term in query.lower().split() if term]
    scored: list[tuple[int, Path, dict[str, Any]]] = []

    for path in sorted(root.glob('*.crumb')):
        try:
            text = path.read_text(encoding='utf-8')
            parsed = parse_crumb(text)
        except Exception:
            continue
        searchable = json.dumps(parsed, ensure_ascii=False).lower()
        score = sum(searchable.count(term) for term in query_terms)
        if score:
            scored.append((score, path, parsed))

    scored.sort(key=lambda item: item[0], reverse=True)
    hits = scored[:top_k]

    if kind == 'mem':
        headers = {
            'v': '1.1',
            'kind': 'mem',
            'title': f'Brain recall: {query}',
            'source': 'brain.bridge',
        }
        consolidated = []
        for score, path, parsed in hits:
            title = parsed['headers'].get('title', path.stem)
            consolidated.append(f"- {title} (score={score}, path={path.name})")
            for section_name in ('consolidated', 'goal', 'context'):
                for line in parsed['sections'].get(section_name, []):
                    if line.strip():
                        consolidated.append(f"  - {line.strip()}")
                        if len(consolidated) >= 18:
                            break
                if len(consolidated) >= 18:
                    break
            if len(consolidated) >= 18:
                break
        if not consolidated:
            consolidated = [f'- No stored memories matched: {query}']
        return render_crumb(headers, {'consolidated': consolidated + ['']})

    context = []
    for score, path, parsed in hits:
        context.append(f"- Match: {parsed['headers'].get('title', path.stem)} (score={score})")
        for section_name in ('goal', 'context', 'consolidated', 'project'):
            for line in parsed['sections'].get(section_name, []):
                if line.strip():
                    context.append(f"  - {line.strip()}")
                    if len(context) >= 20:
                        break
            if len(context) >= 20:
                break
        if len(context) >= 20:
            break

    if not context:
        context = [f'- No stored memories matched: {query}']

    headers = {
        'v': '1.1',
        'kind': 'task',
        'title': f'Brain recall: {query}',
        'source': 'brain.bridge',
    }
    sections = {
        'goal': [f'Resume work related to: {query}', ''],
        'context': context + [''],
        'constraints': [
            '- Treat retrieved items as prior context, not guaranteed truth.',
            '- Verify important details against live files before acting.',
            '',
        ],
    }
    return render_crumb(headers, sections)

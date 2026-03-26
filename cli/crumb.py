#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from textwrap import dedent

def read_text(path: str | None) -> str:
    if path is None or path == '-':
        return sys.stdin.read()
    return Path(path).read_text(encoding='utf-8')

def write_text(path: str | None, content: str) -> None:
    if path is None or path == '-':
        sys.stdout.write(content)
        return
    Path(path).write_text(content, encoding='utf-8')

def cmd_from_chat(args: argparse.Namespace) -> None:
    raw = read_text(args.input)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    user_lines = [line for line in lines if not line.lower().startswith(('assistant:', 'ai:', 'claude:', 'gpt:', 'system:'))]
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

def cmd_validate(args: argparse.Namespace) -> None:
    text = read_text(args.input).strip()
    ok = text.startswith('BEGIN CRUMB') and text.endswith('END CRUMB')
    if ok:
        print('OK: Looks like a valid .crumb block.')
        sys.exit(0)
    print('ERROR: Invalid .crumb block.', file=sys.stderr)
    sys.exit(1)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='crumb', description='Minimal CLI for creating and validating .crumb handoff files.')
    sub = parser.add_subparsers(dest='command', required=True)

    from_chat = sub.add_parser('from-chat', help='Convert a messy chat log into a .crumb handoff.')
    from_chat.add_argument('--input', '-i', default='-', help='Input chat text file or - for stdin.')
    from_chat.add_argument('--output', '-o', default='-', help='Output .crumb file or - for stdout.')
    from_chat.add_argument('--title', help='Optional title for the crumb.')
    from_chat.add_argument('--source', help='Optional source label, e.g. chatgpt.chat or claude.chat.')
    from_chat.add_argument('--goal', help='Override the default goal text.')
    from_chat.add_argument('--constraints', '-c', nargs='*', help='Optional constraints as separate arguments.')
    from_chat.set_defaults(func=cmd_from_chat)

    validate = sub.add_parser('validate', help='Lightweight validation for a .crumb block.')
    validate.add_argument('--input', '-i', default='-', help='Input .crumb file or - for stdin.')
    validate.set_defaults(func=cmd_validate)
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == '__main__':
    main()

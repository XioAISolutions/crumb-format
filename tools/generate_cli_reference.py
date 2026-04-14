#!/usr/bin/env python3
"""Regenerate ``docs/CLI.md`` from the live argparse tree.

Run::

    python tools/generate_cli_reference.py             # writes docs/CLI.md
    python tools/generate_cli_reference.py --check     # non-zero exit on drift

The ``--check`` mode is what CI / tests use to detect drift — any change
to the CLI surface that forgets to regenerate the doc will fail.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.crumb import build_parser  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "docs" / "CLI.md"


def _format_action(action: argparse.Action) -> str:
    """Return a human-readable flag signature for one argparse action."""
    names = ", ".join(action.option_strings) if action.option_strings else action.dest
    meta = ""
    if action.nargs is None and action.option_strings and not isinstance(
        action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._CountAction)
    ):
        meta = f" {action.metavar or action.dest.upper()}"
    elif action.nargs == "?":
        meta = f" [{action.metavar or action.dest.upper()}]"
    elif action.nargs == "*":
        meta = f" [{action.metavar or action.dest.upper()}...]"
    elif action.nargs == "+":
        meta = f" {action.metavar or action.dest.upper()}..."
    line = f"`{names}{meta}`"
    if action.help and action.help != argparse.SUPPRESS:
        line += f" — {action.help}"
    if action.default not in (None, False, argparse.SUPPRESS) and action.option_strings:
        line += f" (default: `{action.default}`)"
    return line


def _render_subparser(name: str, parser: argparse.ArgumentParser) -> str:
    out = io.StringIO()
    out.write(f"### `crumb {name}`\n\n")
    desc = (parser.description or "").strip()
    if desc:
        out.write(f"{desc}\n\n")

    positional = [a for a in parser._actions if not a.option_strings and a.dest != "help"]
    optional = [
        a
        for a in parser._actions
        if a.option_strings and "-h" not in a.option_strings
    ]

    # Skip the subparsers pseudo-action if present
    positional = [
        a for a in positional
        if not isinstance(a, argparse._SubParsersAction)
    ]

    if positional:
        out.write("**Arguments**\n\n")
        for a in positional:
            out.write(f"- {_format_action(a)}\n")
        out.write("\n")

    if optional:
        out.write("**Options**\n\n")
        for a in optional:
            out.write(f"- {_format_action(a)}\n")
        out.write("\n")

    # Handle nested subparsers (e.g. `crumb palace init`, `crumb passport register`)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for nested_name, nested_parser in action.choices.items():
                out.write(f"#### `crumb {name} {nested_name}`\n\n")
                nested_desc = (nested_parser.description or "").strip()
                if nested_desc:
                    out.write(f"{nested_desc}\n\n")
                nested_pos = [
                    a for a in nested_parser._actions
                    if not a.option_strings and a.dest != "help"
                    and not isinstance(a, argparse._SubParsersAction)
                ]
                nested_opt = [
                    a for a in nested_parser._actions
                    if a.option_strings and "-h" not in a.option_strings
                ]
                if nested_pos:
                    out.write("**Arguments**\n\n")
                    for a in nested_pos:
                        out.write(f"- {_format_action(a)}\n")
                    out.write("\n")
                if nested_opt:
                    out.write("**Options**\n\n")
                    for a in nested_opt:
                        out.write(f"- {_format_action(a)}\n")
                    out.write("\n")

    return out.getvalue()


def generate() -> str:
    parser = build_parser()
    out = io.StringIO()
    out.write("# CRUMB CLI reference\n\n")
    out.write(
        "_This document is auto-generated from the argparse tree by "
        "`tools/generate_cli_reference.py`. Do not edit by hand — rerun the "
        "generator after changing the CLI surface._\n\n"
    )
    out.write(
        "See [`docs/STABILITY.md`](STABILITY.md) for which pieces of this "
        "surface are frozen for the 0.x line.\n\n"
    )

    # Find the top-level subparsers action.
    top_sub = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            top_sub = action
            break
    if top_sub is None:
        raise RuntimeError("No subparsers in top-level parser — CLI is malformed.")

    out.write("## Top-level usage\n\n")
    out.write("```\n")
    out.write(parser.format_usage().rstrip() + "\n")
    out.write("```\n\n")

    out.write(f"## Subcommands ({len(top_sub.choices)})\n\n")
    for name in sorted(top_sub.choices.keys()):
        out.write(f"- [`crumb {name}`](#crumb-{name})\n")
    out.write("\n---\n\n")

    for name in sorted(top_sub.choices.keys()):
        out.write(_render_subparser(name, top_sub.choices[name]))
        out.write("\n")

    return out.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the generated output differs from the on-disk file.",
    )
    args = ap.parse_args()

    new_text = generate()

    if args.check:
        existing = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if existing != new_text:
            print(
                f"drift: {OUTPUT_PATH} is out of date. "
                "Rerun `python tools/generate_cli_reference.py` and commit.",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT_PATH}: up to date")
        return 0

    OUTPUT_PATH.write_text(new_text, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(new_text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

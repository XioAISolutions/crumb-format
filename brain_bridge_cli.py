"""Filesystem bridge for the CRUMB brain project layer.

This is intentionally simple: it gives CRUMB a real save/recall bridge
without forcing the project into a heavyweight backend before the runtime
architecture is settled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from capsule_tools import recall_from_brain, save_to_brain


def cmd_save(args: argparse.Namespace) -> None:
    item = save_to_brain(args.file, args.brain_dir, workspace=args.workspace)
    print("Saved to brain workspace:")
    for key, value in item.items():
        print(f"  {key}: {value}")


def cmd_recall(args: argparse.Namespace) -> None:
    crumb = recall_from_brain(
        args.brain_dir,
        args.query,
        workspace=args.workspace,
        kind=args.kind,
        top_k=args.top_k,
    )
    if args.output:
        Path(args.output).write_text(crumb, encoding='utf-8')
        print(f"Wrote recall crumb to {args.output}")
    else:
        print(crumb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='brain-bridge',
        description='Save CRUMBs into a filesystem brain workspace and recall them later.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    save = sub.add_parser('save', help='Save a .crumb file into the brain workspace')
    save.add_argument('file', help='.crumb file to save')
    save.add_argument('--brain-dir', default='.brain', help='Brain root directory')
    save.add_argument('--workspace', default='default', help='Workspace name')
    save.set_defaults(func=cmd_save)

    recall = sub.add_parser('recall', help='Recall context from the brain workspace into a new crumb')
    recall.add_argument('query', help='Query to recall')
    recall.add_argument('--brain-dir', default='.brain', help='Brain root directory')
    recall.add_argument('--workspace', default='default', help='Workspace name')
    recall.add_argument('--kind', choices=['task', 'mem'], default='task')
    recall.add_argument('--top-k', type=int, default=5)
    recall.add_argument('--output', '-o', help='Write the recall crumb to a file')
    recall.set_defaults(func=cmd_recall)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

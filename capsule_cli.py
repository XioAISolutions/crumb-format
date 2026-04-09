"""Standalone helper CLI for CRUMB Capsules and Relay timelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from capsule_tools import build_relay, relay_to_markdown, write_capsule_bundle


def cmd_create(args: argparse.Namespace) -> None:
    outputs = write_capsule_bundle(
        args.file,
        args.output_dir,
        target=args.target,
        metalk_level=args.metalk_level,
    )
    print("Created capsule bundle:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


def cmd_relay(args: argparse.Namespace) -> None:
    relay = build_relay(args.dir)
    if args.format == 'json':
        print(json.dumps(relay, indent=2))
    else:
        print(relay_to_markdown(relay))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='capsule',
        description='Build sharable CRUMB Capsules and Relay timelines.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    create = sub.add_parser('create', help='Create a capsule bundle from a .crumb file.')
    create.add_argument('file', help='.crumb file to convert into a capsule')
    create.add_argument('--output-dir', '-o', default='dist/capsules', help='Where to write the bundle')
    create.add_argument('--target', choices=['any', 'chatgpt', 'claude', 'cursor', 'gemini'], default='any')
    create.add_argument('--metalk-level', type=int, choices=[1, 2, 3], default=2)
    create.set_defaults(func=cmd_create)

    relay = sub.add_parser('relay', help='Build a relay timeline from a directory of .crumb files.')
    relay.add_argument('dir', help='Directory to scan')
    relay.add_argument('--format', choices=['markdown', 'json'], default='markdown')
    relay.set_defaults(func=cmd_relay)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

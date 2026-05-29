"""``crumblm`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crumb_llm import __version__, config as config_mod
from crumb_llm.crumb.loader import load_crumb
from crumb_llm.crumb.pack_reader import read_pack
from crumb_llm.crumb.writer import dumps_json, write_output
from crumb_llm.engine import CrumbEngine
from crumb_llm.models import CrumbDoc, CrumbPack, ProviderConfig
from crumb_llm.providers import get_provider


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_engine() -> CrumbEngine:
    """Build the engine from saved config, falling back to the mock provider."""
    cfg = config_mod.load_config()
    return CrumbEngine(provider=get_provider(cfg))


def _load_source(path: str) -> CrumbDoc | CrumbPack:
    """Load a <crumb_or_pack> argument: a .crumb file or a pack directory."""
    p = Path(path)
    if p.is_dir():
        return read_pack(p)
    return load_crumb(p)


def _emit(result, out: str | None, as_json: bool = False) -> int:
    """Write a result and print warnings to stderr. Returns an exit code."""
    if as_json:
        write_output(dumps_json(result.to_dict()), out)
    else:
        body = result.text
        if result.data is not None and not body.strip():
            body = dumps_json(result.data)
        write_output(body, out)

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    # Provider failure is the one warning class that should fail the command.
    if any(w.startswith("provider error") for w in result.warnings):
        return 1
    return 0


# ---------------------------------------------------------------------------
# command handlers
# ---------------------------------------------------------------------------

def cmd_analyze(args) -> int:
    doc = load_crumb(args.crumb_file)
    result = _build_engine().analyze(doc)
    return _emit(result, args.out)


def cmd_analyze_pack(args) -> int:
    pack = read_pack(args.pack_dir)
    if len(pack) == 0:
        print(f"warning: no .crumb files found in {args.pack_dir}", file=sys.stderr)
    result = _build_engine().analyze_pack(pack)
    return _emit(result, args.out)


def cmd_summarize(args) -> int:
    source = _load_source(args.source)
    result = _build_engine().summarize(source)
    return _emit(result, args.out)


def cmd_risks(args) -> int:
    source = _load_source(args.source)
    as_json = args.format == "json"
    result = _build_engine().risks(source, as_json=as_json)
    return _emit(result, args.out, as_json=as_json)


def cmd_next(args) -> int:
    source = _load_source(args.source)
    result = _build_engine().next_actions(source)
    return _emit(result, args.out)


def cmd_improve(args) -> int:
    doc = load_crumb(args.crumb_file)
    result = _build_engine().improve_handoff(doc)
    return _emit(result, args.out)


def _save_provider(cfg: ProviderConfig) -> int:
    path = config_mod.save_config(cfg)
    print(f"Saved provider config to {path}")
    env = config_mod.PROVIDER_ENV.get(cfg.provider)
    if env:
        present = "set" if _env_present(env) else "NOT set"
        print(f"Reminder: export {env} (currently {present}). "
              "Secrets are never written to config.")
    return 0


def _env_present(name: str) -> bool:
    import os

    return bool(os.environ.get(name))


def cmd_setup(args) -> int:
    if args.target == "openai":
        return _save_provider(ProviderConfig(provider="openai", model=args.model or ""))
    if args.target == "anthropic":
        return _save_provider(
            ProviderConfig(provider="anthropic", model=args.model or "")
        )
    if args.target == "local":
        backend = args.backend
        if backend not in {"ollama", "lmstudio"}:
            print("error: --backend must be 'ollama' or 'lmstudio'", file=sys.stderr)
            return 2
        provider = "ollama" if backend == "ollama" else "lmstudio"
        return _save_provider(
            ProviderConfig(
                provider=provider, model=args.model or "", backend=backend
            )
        )
    print(f"error: unknown setup target: {args.target}", file=sys.stderr)
    return 2


def cmd_status(args) -> int:
    info = config_mod.status()
    print(dumps_json(info))
    return 0


def cmd_export_dataset(args) -> int:
    from crumb_llm.training.export_dataset import export_dataset

    count = export_dataset(args.crumb_dir, args.out)
    print(f"Wrote {count} records to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crumblm",
        description=(
            "CrumbLLM — AI analysis over CRUMB files and packs. Reads project "
            "memory and produces summaries, risks, next actions, and improved "
            "handoffs."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crumblm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Analyze a single CRUMB file")
    p.add_argument("crumb_file")
    p.add_argument("--out", help="Write output to a file instead of stdout")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("analyze-pack", help="Analyze a CRUMB pack (directory)")
    p.add_argument("pack_dir")
    p.add_argument("--out")
    p.set_defaults(func=cmd_analyze_pack)

    p = sub.add_parser("summarize", help="Summarize a CRUMB file or pack")
    p.add_argument("source")
    p.add_argument("--out")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("risks", help="Classify risks in a CRUMB file or pack")
    p.add_argument("source")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--out")
    p.set_defaults(func=cmd_risks)

    p = sub.add_parser("next", help="List next actions for a CRUMB file or pack")
    p.add_argument("source")
    p.add_argument("--out")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("improve", help="Improve a CRUMB handoff")
    p.add_argument("crumb_file")
    p.add_argument("--out")
    p.set_defaults(func=cmd_improve)

    p = sub.add_parser("setup", help="Configure a provider (no secrets stored)")
    p.add_argument("target", choices=["openai", "anthropic", "local"])
    p.add_argument("--model", default="")
    p.add_argument(
        "--backend",
        choices=["ollama", "lmstudio"],
        help="Required for `setup local`",
    )
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("status", help="Show provider config and key presence")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("export-dataset", help="Export CRUMB dir as JSONL dataset")
    p.add_argument("crumb_dir")
    p.add_argument("--out", default="data/crumb_training.jsonl")
    p.set_defaults(func=cmd_export_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # surface, never silently swallow
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

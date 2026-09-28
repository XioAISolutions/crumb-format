"""``crumb-wavelm`` CLI — the user-facing command for the standalone package.

Subcommands:

    crumb-wavelm info
        Show installed version + runtime info.

    crumb-wavelm train [--config tiny|small|...|path] [--data PATH] [--steps N]
        Train a Wave-Field LM. Forwards to crumb_wavelm.train.

    crumb-wavelm generate --ckpt DIR --prompt "..."
        One-shot text generation from a checkpoint.

    crumb-wavelm chat --ckpt DIR [--template chatml|crumb] [--system "..."]
        Interactive REPL using the configured chat template.

    crumb-wavelm serve --ckpt DIR [--port 8090]
        Start the HTTP server (OpenAI-compatible /v1/completions).

    crumb-wavelm quantize --ckpt DIR [--out DIR]
        Int8-dynamic-quantize the Linear layers of a checkpoint and
        save a smaller copy. CPU-only.

    crumb-wavelm index CRUMB_DIR -o crumb_index.json
        Build a spectral context-pulling index from .crumb files.

    crumb-wavelm pull "query" --index-file crumb_index.json
        Pull relevant crumb sections for a query.

    crumb-wavelm bench|compare|perplexity|demo|export ...
        Research and packaging utilities for the standalone package.

    crumb-wavelm models
        List registered local/remote checkpoints.

    crumb-wavelm download MODEL_ID [--out DIR]
        Download or copy a registered checkpoint into the local cache.

    crumb-wavelm register MODEL_ID --path DIR
        Add a local checkpoint directory to the user registry.

Each subcommand delegates to its dedicated module so existing
``python -m crumb_wavelm.<module>`` invocations keep working.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _cmd_info(args: argparse.Namespace) -> int:
    from . import __version__, _TORCH_HINT
    print(f"crumb-wavelm v{__version__}")
    try:
        import torch
    except ImportError:
        print("  torch:  not installed")
        print("  status: model operations unavailable")
        print()
        print(_TORCH_HINT)
        return 0

    print(f"  torch:  {torch.__version__}")
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    qe = ", ".join(torch.backends.quantized.supported_engines)
    print(f"  quant:  {qe}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import load_config, train

    cfg = load_config(args.config)
    if args.arch:
        cfg["arch"] = args.arch
    train(
        config=cfg,
        data_path=args.data,
        steps=args.steps,
        out_dir=args.out,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        grad_accum=args.grad_accum,
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .sample import generate

    prompt = args.prompt
    if args.from_crumb:
        from .crumb_adapter import CrumbPriorBuilder

        text = Path(args.from_crumb).read_text(encoding="utf-8", errors="replace")
        priors = CrumbPriorBuilder().build(text)
        prompt = "".join(chr(b) if b < 128 else "" for b in priors.input_ids[0].tolist())
    text = generate(
        ckpt_dir=args.ckpt,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .serve import serve
    serve(
        ckpt_dir=args.ckpt,
        port=args.port,
        host=args.host,
        index_dir=args.index,
        max_concurrent=args.max_concurrent,
        production=args.production,
    )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import benchmark, format_rows

    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    rows = benchmark(
        lens=lens,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        field_size=args.field_size,
        batch_size=args.batch,
    )
    print(format_rows(rows))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare

    compare(
        config_name=args.config,
        data_path=args.data,
        steps=args.steps,
        out_dir=args.out,
        seed=args.seed,
        log_every=args.log_every,
    )
    return 0


def _cmd_perplexity(args: argparse.Namespace) -> int:
    import torch

    from .sample import load_checkpoint

    model, tok = load_checkpoint(args.ckpt)
    text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    ids = torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)
    total_tokens = ids.size(1)
    max_ctx = getattr(model.cfg, "field_size", None) or getattr(model.cfg, "block_size", total_tokens)
    ids = ids[:, :max_ctx]
    with torch.no_grad():
        x = ids[:, :-1]
        y = ids[:, 1:]
        out = model(x, targets=y)
    loss = out["loss"].item()
    print(f"file:       {args.file}")
    print(f"tokens:     {total_tokens}  (scored: {y.size(1)})")
    print(f"loss:       {loss:.4f}")
    print(f"perplexity: {math.exp(loss):.2f}")
    print(f"bpc:        {loss / math.log(2):.3f}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import main as demo_main

    demo_argv: list[str] = []
    if args.ckpt:
        demo_argv.extend(["--ckpt", args.ckpt])
    if args.data:
        demo_argv.extend(["--data", args.data])
    demo_argv.extend(["--steps", str(args.steps)])
    demo_argv.extend(["--config", args.config])
    if args.compare:
        demo_argv.append("--compare")
    demo_main(demo_argv)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .hub import save_for_hub
    from .sample import load_checkpoint

    model, tok = load_checkpoint(args.ckpt)
    out_dir = args.output or str(Path(args.ckpt) / "hub")
    save_for_hub(model, tok, out_dir, model_name=args.name)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from .context_pull import build_index

    idx = build_index(args.directory, field_size=args.field_size)
    idx.save(args.output)
    print(f"Indexed {len(idx.sections)} sections -> {args.output}")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    from .context_pull import CrumbIndex, pull_context

    idx = CrumbIndex.load(args.index_file)
    pulled = pull_context(args.query, idx, max_tokens=args.max_tokens, top_k=args.top_k)
    if pulled:
        print(pulled)
    else:
        print("No relevant sections found.")
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    import shutil

    import torch

    from .quantize import quantize_dynamic_linear, model_size_mb
    from .sample import load_checkpoint
    src = Path(args.ckpt)
    dst = Path(args.out) if args.out else src.parent / (src.name + "-int8")
    dst.mkdir(parents=True, exist_ok=True)

    model, tok = load_checkpoint(src)
    before = model_size_mb(model)
    qmodel = quantize_dynamic_linear(model)
    after = model_size_mb(qmodel)

    # Mirror the ckpt directory: copy the sidecar files, replace ckpt.pt.
    for f in src.iterdir():
        if f.name == "ckpt.pt":
            continue
        shutil.copy2(f, dst / f.name)
    # Quantized state dicts can't reload through the float WaveFieldLM
    # constructor — store the full quantized module under a clearly named
    # file. Loading is via torch.load + a runtime swap path.
    torch.save({
        "quantized_module": qmodel,
        "tokenizer_type": getattr(tok, "name", "byte"),
    }, dst / "ckpt-int8.pt")
    print(f"[quantize] {src} -> {dst}")
    print(f"  size: {before:.2f} MB -> {after:.2f} MB ({100*(1-after/before):.1f}% smaller)")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    import torch

    from .chat import apply_chat_template, stop_tokens_for
    from .sample import load_checkpoint

    model, tok = load_checkpoint(args.ckpt)
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    stops = stop_tokens_for(args.template)

    print(f"crumb-wavelm chat — template={args.template}, ckpt={args.ckpt}")
    print("type /exit to quit, /reset to clear history, /stop_words to show stop tokens.\n")
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/exit":
            return 0
        if user == "/reset":
            messages = ([] if not args.system
                        else [{"role": "system", "content": args.system}])
            print("(history cleared)")
            continue
        if user == "/stop_words":
            print(f"stop tokens: {stops}")
            continue

        messages.append({"role": "user", "content": user})
        prompt = apply_chat_template(messages, template=args.template)
        ids = torch.tensor(tok.encode(prompt), dtype=torch.long).unsqueeze(0)
        out = model.generate(
            ids, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            eos_token_id=getattr(tok, "eos_id", None),
        )
        completion_ids = out[0, ids.shape[1]:].tolist()
        text = tok.decode(completion_ids)
        for stop in stops:
            if stop in text:
                text = text.split(stop)[0]
                break
        text = text.strip()
        print(text)
        messages.append({"role": "assistant", "content": text})
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .registry import RegistryError, download_model

    try:
        path = download_model(
            args.model_id,
            out_dir=args.out,
            registry=args.registry,
            cache_dir=args.cache_dir,
            force=args.force,
        )
    except RegistryError as exc:
        print(f"download: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .registry import RegistryError, list_models

    try:
        models = list_models(args.registry, include_planned=not args.downloadable)
    except RegistryError as exc:
        print(f"models: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(models, indent=2, sort_keys=True))
        return 0
    if not models:
        print("No models registered.")
        return 0
    for model in models:
        marker = "downloadable" if model.get("downloadable") else model.get("status", "listed")
        aliases = model.get("aliases") or []
        alias_text = f" ({', '.join(aliases)})" if aliases else ""
        print(f"{model['id']}{alias_text}  [{marker}]")
        if model.get("description"):
            print(f"  {model['description']}")
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    from .registry import RegistryError, register_local_model

    aliases = []
    for value in args.alias:
        aliases.extend(part.strip() for part in value.split(",") if part.strip())
    try:
        path = register_local_model(
            args.model_id,
            args.path,
            name=args.name,
            description=args.description,
            registry_path=args.registry,
            aliases=aliases,
        )
    except RegistryError as exc:
        print(f"register: {exc}", file=sys.stderr)
        return 1
    print(f"registered {args.model_id} in {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crumb-wavelm", description="Crumb LLM CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Show version + runtime info.").set_defaults(func=_cmd_info)

    p = sub.add_parser("train", help="Train a Wave-Field LM.")
    p.add_argument("--config", default="tiny", help="Built-in config name or JSON path.")
    p.add_argument("--data", default=None, help="Path to text file or directory.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--out", default="crumb_llm/checkpoints/run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--arch", default=None, choices=["wave_field", "transformer"])
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("generate", help="One-shot generation from a checkpoint.")
    p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    p.add_argument("--prompt", default="", help="Initial prompt text.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--from-crumb", default=None, help="Read prompt from a .crumb file.")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("serve", help="Start the HTTP server.")
    p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--index", default=None, help="Directory of .crumb files for context pulling.")
    p.add_argument("--production", action="store_true", help="Enable API-key auth and tier limits.")
    p.add_argument("--max-concurrent", type=int, default=4)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("bench", help="Benchmark wave-field vs transformer forward pass.")
    p.add_argument("--lens", default="256,1024,4096", help="Comma-separated sequence lengths.")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--field-size", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("compare", help="Train wave-field and transformer, then report the gap.")
    p.add_argument("--config", default="tiny")
    p.add_argument("--data", default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--out", default="/tmp/crumb_llm_compare")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("perplexity", help="Score a crumb or text file under a checkpoint.")
    p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    p.add_argument("file", help="Input .crumb/.txt/.md file.")
    p.set_defaults(func=_cmd_perplexity)

    p = sub.add_parser("demo", help="End-to-end demo: train, generate, context-pull.")
    p.add_argument("--ckpt", default=None, help="Skip training and use an existing checkpoint.")
    p.add_argument("--data", default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--config", default="tiny")
    p.add_argument("--compare", action="store_true")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("export", help="Export a checkpoint for HuggingFace Hub upload.")
    p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    p.add_argument("--name", default="crumb-llm-tiny", help="Model name for Hub.")
    p.add_argument("-o", "--output", default=None, help="Output directory.")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("index", help="Build a context-pulling index from .crumb files.")
    p.add_argument("directory", help="Directory containing .crumb files.")
    p.add_argument("-o", "--output", default="crumb_index.json", help="Output index file.")
    p.add_argument("--field-size", type=int, default=256)
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("pull", help="Pull relevant crumb sections for a query.")
    p.add_argument("query", help="Query text to match against the index.")
    p.add_argument("--index-file", required=True, help="Path to a crumb_index.json file.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=_cmd_pull)

    p = sub.add_parser("chat", help="Interactive REPL.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--template", default="chatml", choices=["chatml", "crumb"])
    p.add_argument("--system", default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("quantize", help="Int8 quantize a checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_quantize)

    p = sub.add_parser("models", help="List registered checkpoints.")
    p.add_argument("--registry", action="append", default=[], help="Additional registry JSON path or URL.")
    p.add_argument("--downloadable", action="store_true", help="Only show downloadable models.")
    p.add_argument("--json", action="store_true", help="Print JSON.")
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser("download", help="Download or copy a registered checkpoint.")
    p.add_argument("model_id")
    p.add_argument("--out", default=None)
    p.add_argument("--registry", action="append", default=[], help="Additional registry JSON path or URL.")
    p.add_argument("--cache-dir", default=None, help="Override model cache directory.")
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("register", help="Register a local checkpoint directory.")
    p.add_argument("model_id")
    p.add_argument("--path", required=True, help="Checkpoint or Hub-format directory.")
    p.add_argument("--name", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--alias", action="append", default=[], help="Alias, or comma-separated aliases.")
    p.add_argument("--registry", default=None, help="Registry file to update.")
    p.set_defaults(func=_cmd_register)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

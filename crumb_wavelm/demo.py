"""End-to-end demo: train, generate, serve, context-pull.

Shows Crumb LLM as a working language model, not just a research toy.

Usage:
    python -m crumb_wavelm.demo                    # full demo (train + generate)
    python -m crumb_wavelm.demo --ckpt /path       # skip training, just demo
    python -m crumb_wavelm.demo --compare           # include transformer comparison
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .train import load_config, build_model, build_tokenizer, train, cosine_with_warmup, eval_loss
from .sample import load_checkpoint, generate
from .cache import generate_cached
from .data import load_text_corpus, make_token_stream, train_eval_split
from .context_pull import build_index, pull_context


def demo_generate(ckpt_dir: str | Path, prompts: list[str] | None = None) -> None:
    """Generate text from a trained checkpoint and print results."""
    model, tok = load_checkpoint(ckpt_dir)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    prompts = prompts or [
        "First Citizen:\nBefore we proceed any further",
        "ROMEO:\nBut, soft! what light through",
        "To be, or not to be",
        "The wave field carries",
    ]

    print(f"\n{'='*60}")
    print(f"  Crumb LLM — Generation Demo")
    print(f"  params: {n_params:,}  field_size: {getattr(model.cfg, 'field_size', '?')}")
    print(f"{'='*60}\n")

    for prompt in prompts:
        print(f"PROMPT: {prompt}")
        print(f"{'─'*50}")

        # Standard generation.
        ids = torch.tensor(tok.encode(prompt), dtype=torch.long).unsqueeze(0)
        t0 = time.time()
        out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=40)
        dt = time.time() - t0
        text = tok.decode(out[0].tolist())
        n_gen = out.shape[1] - ids.shape[1]

        print(text)
        print(f"\n[{n_gen} tokens, {dt*1000:.0f}ms, {n_gen/max(dt,0.001):.0f} tok/s]")

        # Cached generation (should match but faster on long sequences).
        t0 = time.time()
        out_cached = generate_cached(model, ids, max_new_tokens=50, temperature=0.8, top_k=40)
        dt_cached = time.time() - t0
        n_cached = out_cached.shape[1] - ids.shape[1]
        print(f"[cached: {n_cached} tokens, {dt_cached*1000:.0f}ms]")
        print()

    # Perplexity on held-out text.
    print(f"{'='*60}")
    print(f"  Perplexity Evaluation")
    print(f"{'='*60}")
    field_size = getattr(model.cfg, 'field_size', 256)
    eval_text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n"
    eval_ids = torch.tensor(tok.encode(eval_text), dtype=torch.long).unsqueeze(0)
    eval_ids = eval_ids[:, :field_size]
    with torch.no_grad():
        out = model(eval_ids[:, :-1], targets=eval_ids[:, 1:])
    loss = out["loss"].item()
    print(f"  eval text: {len(eval_text)} chars, {eval_ids.size(1)} tokens")
    print(f"  loss: {loss:.4f}")
    print(f"  perplexity: {math.exp(min(loss, 20)):.2f}")
    print(f"  bits/char: {loss / math.log(2):.3f}")
    print()


def demo_context_pull(crumb_dir: str | Path = "examples") -> None:
    """Demo context pulling from a crumb library."""
    crumb_path = Path(crumb_dir)
    if not crumb_path.exists():
        print("[demo] no examples/ directory — skipping context-pull demo")
        return

    print(f"{'='*60}")
    print(f"  Context Pulling Demo")
    print(f"{'='*60}")

    idx = build_index(crumb_path, field_size=128)
    print(f"  indexed {len(idx.sections)} sections from {crumb_path}\n")

    queries = [
        "Fix the login redirect bug",
        "What are the user's preferences?",
        "Deploy the application",
    ]
    for q in queries:
        print(f"  QUERY: {q}")
        pulled = pull_context(q, idx, max_tokens=100, top_k=3)
        if pulled:
            lines = pulled.split("\n")
            for line in lines[:5]:
                print(f"    {line}")
            if len(lines) > 5:
                print(f"    ... ({len(lines) - 5} more lines)")
        else:
            print("    (no relevant sections)")
        print()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Crumb LLM end-to-end demo.")
    ap.add_argument("--ckpt", default=None, help="Skip training, use existing checkpoint.")
    ap.add_argument("--data", default=None, help="Training corpus path.")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--compare", action="store_true", help="Include transformer comparison.")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args(argv)

    ckpt_dir = args.ckpt

    if ckpt_dir is None:
        # Train a model first.
        print(f"{'='*60}")
        print(f"  Training Crumb LLM ({args.steps} steps)")
        print(f"{'='*60}\n")
        ckpt_dir = "/tmp/crumb_llm_demo"
        cfg = load_config(args.config)
        metrics = train(
            config=cfg,
            data_path=args.data,
            steps=args.steps,
            out_dir=ckpt_dir,
            log_every=max(1, args.steps // 10),
            eval_every=max(1, args.steps // 4),
        )
        print(f"\n  Final loss: {metrics['final_loss']:.4f}")
        print(f"  Final BPC: {metrics['final_bpc']:.3f}")
        print(f"  Wall time: {metrics['wallclock_seconds']:.1f}s")

    if not args.skip_generate:
        demo_generate(ckpt_dir)

    demo_context_pull()

    if args.compare:
        print(f"\n{'='*60}")
        print(f"  Transformer Comparison")
        print(f"{'='*60}\n")
        from .compare import compare
        compare(
            config_name=args.config,
            data_path=args.data,
            steps=min(args.steps, 1000),
        )


if __name__ == "__main__":
    main()

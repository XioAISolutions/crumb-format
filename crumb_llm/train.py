"""Tiny training loop for WaveFieldLM.

Designed to converge to a usable (toy) checkpoint in minutes on a CPU
for the ``tiny`` config. Larger configs work on a single GPU.

Usage:
    python -m crumb_llm.train --config tiny --steps 2000
    python -m crumb_llm.train --config small --steps 50000 --data path/to/text

The same trainer powers ``crumb wave train`` once the CLI hook is in.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .baseline import TinyTransformerLM, TransformerConfig
from .data import load_text_corpus, make_token_stream, train_eval_split
from .model import WaveFieldLM, WaveFieldConfig
from .tokenizer import ByteTokenizer, CharTokenizer


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


# ── Helpers ──────────────────────────────────────────────────────────


def load_config(name: str) -> dict:
    """Resolve ``name`` to a config dict. ``name`` may be a built-in or a path."""
    p = Path(name)
    if not p.exists():
        p = CONFIG_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"config not found: {name}")
    return json.loads(p.read_text())


def cosine_with_warmup(optimizer, warmup: int, total: int):
    def lr(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return LambdaLR(optimizer, lr_lambda=lr)


def build_model(cfg: dict, vocab_size: int):
    arch = cfg.get("arch", "wave_field")
    if arch == "wave_field":
        mc = WaveFieldConfig(
            vocab_size=vocab_size,
            dim=cfg["dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            field_size=cfg["field_size"],
            ffn_mult=cfg.get("ffn_mult", 8.0 / 3.0),
            causal=cfg.get("causal", True),
            kernel_mode=cfg.get("kernel_mode", "freq"),
            dropout=cfg.get("dropout", 0.0),
            tie_embeddings=cfg.get("tie_embeddings", True),
            boundary=cfg.get("boundary", "periodic"),
            dispersion=cfg.get("dispersion", False),
            interference_mixer=cfg.get("interference_mixer", False),
            adaptive_kernels=cfg.get("adaptive_kernels", False),
            spectral_gate=cfg.get("spectral_gate", False),
            resonance_memory=cfg.get("resonance_memory", False),
            hybrid_gate=cfg.get("hybrid_gate", False),
            hybrid_window=cfg.get("hybrid_window", 64),
            use_v2_blocks=cfg.get("use_v2_blocks", False),
            v2_learned_scatter=cfg.get("v2_learned_scatter", True),
            v2_query_freq_gate=cfg.get("v2_query_freq_gate", True),
            v2_short_conv=cfg.get("v2_short_conv", True),
            v2_position_mod=cfg.get("v2_position_mod", True),
            v2_rotary=cfg.get("v2_rotary", True),
        )
        return WaveFieldLM(mc), mc
    if arch == "transformer":
        tc = TransformerConfig(
            vocab_size=vocab_size,
            dim=cfg["dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            block_size=cfg["block_size"],
            ffn_mult=cfg.get("ffn_mult", 8.0 / 3.0),
            dropout=cfg.get("dropout", 0.0),
            tie_embeddings=cfg.get("tie_embeddings", True),
        )
        return TinyTransformerLM(tc), tc
    raise ValueError(f"unknown arch: {arch!r}")


def build_tokenizer(cfg: dict, corpus: str):
    tname = cfg.get("tokenizer", "byte")
    if tname == "byte":
        return ByteTokenizer()
    if tname == "char":
        return CharTokenizer.fit(corpus)
    raise ValueError(f"unknown tokenizer: {tname!r}")


# ── Eval ─────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_loss(model, stream, batches: int = 4, batch_size: int = 8) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = stream.batch(batch_size)
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        if isinstance(model, WaveFieldLM):
            out = model(x, targets=y)
        else:
            out = model(x, targets=y)
        total += out["loss"].item()
    model.train()
    return total / batches


# ── Main trainer ─────────────────────────────────────────────────────


def train(
    config: dict,
    data_path: str | None,
    steps: int,
    out_dir: str | Path,
    seed: int = 0,
    log_every: int = 50,
    eval_every: int = 500,
    grad_accum: int = 1,
) -> dict:
    """Run training. Returns a small dict of final metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)

    text = load_text_corpus(data_path)
    train_text, eval_text = train_eval_split(text, eval_frac=0.05)
    tok = build_tokenizer(config, train_text)

    block_size = config.get("block_size", config.get("field_size", 256))
    train_stream = make_token_stream(train_text, tok, block_size=block_size)
    eval_stream = make_token_stream(eval_text or train_text[-1000:], tok, block_size=block_size)

    model, mc = build_model(config, vocab_size=tok.vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] arch={config.get('arch', 'wave_field')} params={n_params:,} device={device}")

    optim = AdamW(
        model.parameters(),
        lr=config.get("lr", 3e-4),
        betas=(0.9, 0.95),
        weight_decay=config.get("weight_decay", 0.1),
    )
    sched = cosine_with_warmup(optim, warmup=config.get("warmup", min(100, steps // 10)), total=steps)

    bs = config.get("batch_size", 8)
    t0 = time.time()
    final_loss = float("nan")
    for step in range(1, steps + 1):
        loss_accum = 0.0
        optim.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = train_stream.batch(bs)
            x, y = x.to(device), y.to(device)
            out = model(x, targets=y)
            loss = out["loss"] / grad_accum
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        final_loss = loss_accum
        if step % log_every == 0 or step == 1:
            dt = time.time() - t0
            print(f"[train] step={step:>6d}  loss={final_loss:.4f}  bpc={final_loss / math.log(2):.3f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={dt:.1f}s")
        if eval_every and step % eval_every == 0:
            el = eval_loss(model, eval_stream, batches=4, batch_size=bs)
            print(f"[eval ] step={step:>6d}  loss={el:.4f}  bpc={el / math.log(2):.3f}")

    # Final save.
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": asdict(mc),
        "tokenizer_type": config.get("tokenizer", "byte"),
        "arch": config.get("arch", "wave_field"),
    }
    ckpt_path = out_dir / "ckpt.pt"
    torch.save(ckpt, ckpt_path)
    tok.save(out_dir / "tokenizer.json")
    metrics = {
        "final_loss": final_loss,
        "final_bpc": final_loss / math.log(2),
        "params": n_params,
        "steps": steps,
        "wallclock_seconds": time.time() - t0,
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] saved checkpoint → {ckpt_path}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a Wave-Field LM.")
    ap.add_argument("--config", default="tiny", help="Built-in config name or path to JSON.")
    ap.add_argument("--data", default=None, help="Path to text file or directory.")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="crumb_llm/checkpoints/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--arch", default=None, help="Override arch in config (wave_field|transformer).")
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    main()

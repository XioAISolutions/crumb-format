"""Head-to-head comparison: Crumb LLM vs Transformer baseline.

Trains both architectures with identical hyperparameters on the same
corpus and reports the final perplexity gap. This is the experiment
that validates (or refutes) the "within 5% of transformer" claim.

Usage:
    python -m crumb_wavelm.compare --steps 2000 --data examples/
    python -m crumb_wavelm.compare --config small --steps 50000 --data /path/to/corpus
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .data import load_text_corpus, make_token_stream, train_eval_split
from .train import build_model, build_tokenizer, cosine_with_warmup, eval_loss, load_config

import torch
from torch.optim import AdamW


def compare(
    config_name: str = "tiny",
    data_path: str | None = None,
    steps: int = 1000,
    out_dir: str | Path = "/tmp/crumb_llm_compare",
    seed: int = 0,
    log_every: int = 100,
) -> dict:
    """Train wave-field and transformer, report perplexity gap."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_name)
    text = load_text_corpus(data_path)
    train_text, eval_text = train_eval_split(text, eval_frac=0.05)

    results = {}

    for arch_name in ["wave_field", "transformer"]:
        print(f"\n{'='*60}")
        print(f"  Training: {arch_name}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        import random
        random.seed(seed)

        arch_cfg = dict(cfg)
        arch_cfg["arch"] = arch_name
        if arch_name == "transformer" and "block_size" not in arch_cfg:
            arch_cfg["block_size"] = arch_cfg.get("field_size", 256)

        tok = build_tokenizer(arch_cfg, train_text)
        block_size = arch_cfg.get("block_size", arch_cfg.get("field_size", 256))
        train_stream = make_token_stream(train_text, tok, block_size=block_size)
        eval_stream = make_token_stream(eval_text or train_text[-1000:], tok, block_size=block_size)

        model, mc = build_model(arch_cfg, vocab_size=tok.vocab_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params: {n_params:,}  device: {device}")

        optim = AdamW(
            model.parameters(),
            lr=arch_cfg.get("lr", 3e-4),
            betas=(0.9, 0.95),
            weight_decay=arch_cfg.get("weight_decay", 0.1),
        )
        sched = cosine_with_warmup(
            optim,
            warmup=arch_cfg.get("warmup", min(100, steps // 10)),
            total=steps,
        )

        bs = arch_cfg.get("batch_size", 8)
        t0 = time.time()
        train_losses = []
        for step in range(1, steps + 1):
            optim.zero_grad(set_to_none=True)
            x, y = train_stream.batch(bs)
            x, y = x.to(device), y.to(device)
            out = model(x, targets=y)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            train_losses.append(loss.item())
            if step % log_every == 0 or step == 1:
                el = eval_loss(model, eval_stream, batches=4, batch_size=bs)
                bpc = el / math.log(2)
                ppl = math.exp(el)
                dt = time.time() - t0
                print(f"  [{arch_name}] step={step:>6d}  train={loss.item():.4f}  "
                      f"eval={el:.4f}  ppl={ppl:.2f}  bpc={bpc:.3f}  {dt:.1f}s")

        final_eval = eval_loss(model, eval_stream, batches=8, batch_size=bs)
        results[arch_name] = {
            "params": n_params,
            "final_train_loss": train_losses[-1],
            "final_eval_loss": final_eval,
            "final_ppl": math.exp(final_eval),
            "final_bpc": final_eval / math.log(2),
            "wallclock_s": time.time() - t0,
        }

    # Report.
    print(f"\n{'='*60}")
    print("  COMPARISON RESULTS")
    print(f"{'='*60}")
    wf = results["wave_field"]
    tf = results["transformer"]
    gap_ppl = (wf["final_ppl"] - tf["final_ppl"]) / tf["final_ppl"] * 100
    print(f"  Wave-Field:   ppl={wf['final_ppl']:.2f}  bpc={wf['final_bpc']:.3f}  "
          f"params={wf['params']:,}  time={wf['wallclock_s']:.1f}s")
    print(f"  Transformer:  ppl={tf['final_ppl']:.2f}  bpc={tf['final_bpc']:.3f}  "
          f"params={tf['params']:,}  time={tf['wallclock_s']:.1f}s")
    print(f"  PPL gap:      {gap_ppl:+.1f}%  {'(within 5%)' if abs(gap_ppl) < 5 else '(>5% gap)'}")

    results["ppl_gap_pct"] = gap_ppl
    report_path = out_dir / "comparison.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {report_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare Crumb LLM vs Transformer baseline.")
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default="/tmp/crumb_llm_compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)
    compare(
        config_name=args.config,
        data_path=args.data,
        steps=args.steps,
        out_dir=args.out,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

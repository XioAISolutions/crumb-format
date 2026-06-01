#!/usr/bin/env python3
"""One-command GPU training for Crumb LLM.

Gets coherent Shakespeare generation in ~10 minutes on any modern GPU.

Usage (Colab, Lambda, RunPod, local):
    pip install 'crumb-format[llm]'
    python scripts/train_gpu.py

Or with custom data:
    python scripts/train_gpu.py --data /path/to/corpus.txt --steps 20000
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(path: str = "/tmp/shakespeare.txt") -> str:
    p = Path(path)
    if p.exists():
        print(f"[data] using cached {p} ({p.stat().st_size:,} bytes)")
        return str(p)
    print(f"[data] downloading TinyShakespeare...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, str(p))
    print(f"[data] saved {p.stat().st_size:,} bytes")
    return str(p)


def main():
    ap = argparse.ArgumentParser(description="Train Crumb LLM on GPU (or CPU)")
    ap.add_argument("--data", default=None, help="Text file. Default: TinyShakespeare.")
    ap.add_argument("--config", default="small", help="Config name (tiny/small/medium/v2).")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--out", default="/tmp/crumb_llm_gpu")
    ap.add_argument("--demo", action="store_true", help="Run demo after training.")
    args = ap.parse_args()

    data = args.data or download_shakespeare()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║              Crumb LLM — GPU Training                    ║
╠══════════════════════════════════════════════════════════╣
║  config:  {args.config:<46s} ║
║  data:    {data:<46s} ║
║  steps:   {args.steps:<46d} ║
║  output:  {args.out:<46s} ║
╚══════════════════════════════════════════════════════════╝
""")

    cmd = [
        sys.executable, "-m", "crumb_wavelm.train",
        "--config", args.config,
        "--steps", str(args.steps),
        "--data", data,
        "--out", args.out,
        "--log-every", str(max(1, args.steps // 20)),
        "--eval-every", str(max(1, args.steps // 5)),
    ]
    subprocess.run(cmd, check=True)

    if args.demo:
        print("\n\n" + "="*60)
        print("  Generation Demo")
        print("="*60 + "\n")
        import torch
        from crumb_wavelm.sample import load_checkpoint

        model, tok = load_checkpoint(args.out)
        model.eval()
        prompts = [
            "First Citizen:\nBefore we proceed",
            "ROMEO:\nBut, soft! what light through",
            "To be, or not to be, that is",
        ]
        for p in prompts:
            ids = torch.tensor(tok.encode(p), dtype=torch.long).unsqueeze(0)
            device = next(model.parameters()).device
            ids = ids.to(device)
            out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=40)
            text = tok.decode(out[0].tolist())
            print(f"PROMPT: {p.strip()}")
            print("-" * 50)
            print(text[:400])
            print()

    print(f"\nCheckpoint saved: {args.out}/")
    print(f"Generate: crumb llm generate --ckpt {args.out} --prompt 'ROMEO:'")
    print(f"Serve:    crumb llm serve --ckpt {args.out} --index examples/")
    print(f"Export:   crumb llm export --ckpt {args.out} --name my-crumb-llm")


if __name__ == "__main__":
    main()

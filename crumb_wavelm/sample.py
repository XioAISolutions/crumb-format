"""Load a checkpoint and generate text.

Usage:
    python -m crumb_wavelm.sample --ckpt path/to/run \
        --prompt "BEGIN CRUMB" --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .baseline import TinyTransformerLM, TransformerConfig
from .model import WaveFieldLM, WaveFieldConfig
from .tokenizer import ByteTokenizer, CharTokenizer


def load_checkpoint(ckpt_dir: str | Path):
    p = Path(ckpt_dir)
    ckpt = torch.load(p / "ckpt.pt", map_location="cpu", weights_only=False)
    tok_path = p / "tokenizer.json"
    tok_type = ckpt.get("tokenizer_type", "byte")
    if tok_type == "byte":
        tok = ByteTokenizer.load(tok_path) if tok_path.exists() else ByteTokenizer()
    elif tok_type == "char":
        tok = CharTokenizer.load(tok_path)
    else:
        raise ValueError(f"unknown tokenizer type: {tok_type!r}")
    arch = ckpt.get("arch", "wave_field")
    if arch == "wave_field":
        cfg = WaveFieldConfig(**ckpt["model_config"])
        model = WaveFieldLM(cfg)
    elif arch == "transformer":
        cfg = TransformerConfig(**ckpt["model_config"])
        model = TinyTransformerLM(cfg)
    else:
        raise ValueError(f"unknown arch: {arch!r}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


def generate(
    ckpt_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = 40,
    seed: int | None = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    model, tok = load_checkpoint(ckpt_dir)
    ids = torch.tensor(tok.encode(prompt), dtype=torch.long).unsqueeze(0)
    if isinstance(model, WaveFieldLM):
        out_ids = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    else:
        # Quick & dirty AR sampling for the baseline.
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = ids[:, -model.cfg.block_size :]
                logits = model(ctx)["logits"][:, -1, :] / max(temperature, 1e-6)
                if top_k:
                    v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    logits = torch.where(
                        logits < v[..., [-1]], torch.full_like(logits, float("-inf")), logits
                    )
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, nxt], dim=1)
            out_ids = ids
    return tok.decode(out_ids[0].tolist())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sample from a Wave-Field LM checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    ap.add_argument("--prompt", default="", help="Initial prompt text.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    text = generate(
        ckpt_dir=args.ckpt,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()

"""Minimal from-scratch decoder-only transformer (EXPERIMENTAL).

This is a tiny, character-level GPT used only by the experimental ``scratch``
provider. It is cannibalized from FareedKhan-dev/train-llm-from-scratch and
deliberately kept small and dependency-light.

It NEVER runs automatically. Training only happens when a user explicitly calls
``train_scratch(...)``. torch is imported lazily so the rest of CrumbLLM works
without it.
"""

from __future__ import annotations

import json
from pathlib import Path


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "torch is required for scratch training/inference. "
            "Install with: pip install 'crumb-llm[scratch]'"
        ) from exc


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_model(vocab_size: int, n_embed: int, n_head: int, n_blocks: int,
                 context_length: int):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embed)
            self.attn = nn.MultiheadAttention(
                n_embed, n_head, batch_first=True
            )
            self.ln2 = nn.LayerNorm(n_embed)
            self.mlp = nn.Sequential(
                nn.Linear(n_embed, 4 * n_embed),
                nn.GELU(),
                nn.Linear(4 * n_embed, n_embed),
            )

        def forward(self, x):
            t = x.size(1)
            mask = torch.triu(
                torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
            )
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_length = context_length
            self.tok = nn.Embedding(vocab_size, n_embed)
            self.pos = nn.Embedding(context_length, n_embed)
            self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
            self.ln_f = nn.LayerNorm(n_embed)
            self.head = nn.Linear(n_embed, vocab_size, bias=False)

        def forward(self, idx, targets=None):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None, :, :]
            for blk in self.blocks:
                x = blk(x)
            logits = self.head(self.ln_f(x))
            loss = None
            if targets is not None:
                import torch.nn.functional as F

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=1.0):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.context_length:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, nxt], dim=1)
            return idx

    return TinyGPT()


# ---------------------------------------------------------------------------
# Char tokenizer (kept inside the checkpoint for portability)
# ---------------------------------------------------------------------------

def _build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


# ---------------------------------------------------------------------------
# Train / load
# ---------------------------------------------------------------------------

def train_scratch(
    sft_text_path: str | Path,
    out_path: str | Path,
    steps: int = 500,
    batch_size: int = 8,
    context_length: int = 128,
    n_embed: int = 128,
    n_head: int = 4,
    n_blocks: int = 2,
    lr: float = 3e-4,
    confirm: bool = False,
) -> dict:
    """Train a tiny scratch model. EXPERIMENTAL and opt-in.

    ``confirm`` must be True to actually train — this guards against accidental
    training (CrumbLLM never trains by default).
    """
    if not confirm:
        raise RuntimeError(
            "Refusing to train: scratch training is experimental and opt-in. "
            "Pass confirm=True (or `--confirm` on the CLI) to proceed."
        )
    _require_torch()
    import torch

    text = Path(sft_text_path).read_text(encoding="utf-8")
    if not text:
        raise ValueError("training corpus is empty")
    stoi, itos = _build_vocab(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    model = _build_model(len(stoi), n_embed, n_head, n_blocks, context_length)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_batch():
        ix = torch.randint(0, max(1, len(data) - context_length - 1), (batch_size,))
        x = torch.stack([data[i:i + context_length] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])
        return x, y

    model.train()
    last_loss = float("nan")
    for step in range(steps):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": len(stoi),
            "n_embed": n_embed,
            "n_head": n_head,
            "n_blocks": n_blocks,
            "context_length": context_length,
        },
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    return {"steps": steps, "final_loss": last_loss, "out": str(out)}


def load_and_generate(
    checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
) -> str:
    """Load a scratch checkpoint and generate a continuation of ``prompt``."""
    _require_torch()
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()}

    model = _build_model(
        cfg["vocab_size"], cfg["n_embed"], cfg["n_head"],
        cfg["n_blocks"], cfg["context_length"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    seed = [stoi[c] for c in prompt if c in stoi] or [0]
    idx = torch.tensor([seed], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return "".join(itos.get(int(i), "") for i in out[0].tolist())


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(json.dumps({"note": "use `crumblm` CLI or import train_scratch()"}))
    sys.exit(0)

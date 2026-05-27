"""Tiny causal-attention baseline for apples-to-apples comparison.

The Wave-Field paper claims "within 5% of standard transformer quality";
to substantiate or refute that claim *for the configs that ship in this
package*, we need a baseline with the same dim / depth / FFN shape and
the same training loop. This module provides one.

Everything matches ``WaveFieldLM`` except the mixing op: dot-product
attention with a causal mask. Per-step cost is O(B · H · N² · d_head),
which is exactly the cost the wave-field architecture is trying to
beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import RMSNorm, SwiGLUFFN


@dataclass
class TransformerConfig:
    vocab_size: int = 256
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    block_size: int = 256          # max context — used for the causal mask
    ffn_mult: float = 8.0 / 3.0
    dropout: float = 0.0
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must divide n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Built-in flash-aware kernel; is_causal handles the triangular mask.
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        return self.drop(self.proj(attn))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg.dim, cfg.n_heads, cfg.dropout)
        self.norm_ffn = RMSNorm(cfg.dim)
        self.ffn = SwiGLUFFN(cfg.dim, hidden_mult=cfg.ffn_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TinyTransformerLM(nn.Module):
    """Baseline. Identical embedding/FFN/norm shape to WaveFieldLM."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.block_size, cfg.dim)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_out = RMSNorm(cfg.dim)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> dict:
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        logits = x @ self.embed.weight.t() if self.lm_head is None else self.lm_head(x)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return out

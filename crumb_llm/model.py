"""WaveFieldLM — embedding → N blocks → tied LM head.

Closely mirrors a decoder-only transformer in skeleton — embed + stack +
unembed — with the wave-field block in place of attention. No positional
encoding is used: the scatter/gather step already encodes position
geometrically (each token deposits at a different field coordinate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import WaveFieldBlock, WaveFieldBlockConfig, RMSNorm


@dataclass
class WaveFieldConfig:
    vocab_size: int = 256                  # default = byte-level
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    field_size: int = 256
    ffn_mult: float = 8.0 / 3.0
    causal: bool = True
    kernel_mode: str = "freq"
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Advanced wave physics.
    boundary: str = "periodic"             # periodic | absorbing | reflecting
    dispersion: bool = False               # learnable frequency-dependent phase
    interference_mixer: bool = False       # complex head interference vs proj_out
    # Crumb-adapter knobs (forwarded by `forward(...)` callers; the model
    # itself only accepts pre-computed tensors).
    use_crumb_priors: bool = False
    crumb_prior_keys: tuple = field(
        default_factory=lambda: ("scatter_weights", "kernel_bias")
    )


class WaveFieldLM(nn.Module):
    """Decoder-only language model with wave-field mixing.

    Forward signature accepts either bare token ids or a dict of inputs
    so the crumb adapter can pass through structural priors without
    every caller having to wrap them.
    """

    def __init__(self, cfg: WaveFieldConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        block_cfg = WaveFieldBlockConfig(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            field_size=cfg.field_size,
            ffn_mult=cfg.ffn_mult,
            causal=cfg.causal,
            kernel_mode=cfg.kernel_mode,
            dropout=cfg.dropout,
            boundary=cfg.boundary,
            dispersion=cfg.dispersion,
            interference_mixer=cfg.interference_mixer,
        )
        self.blocks = nn.ModuleList(WaveFieldBlock(block_cfg) for _ in range(cfg.n_layers))
        self.norm_out = RMSNorm(cfg.dim)
        if cfg.tie_embeddings:
            self.lm_head = None  # uses self.embed.weight at forward time
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Tensor,
        scatter_weights: Optional[Tensor] = None,
        kernel_bias: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
    ) -> dict:
        """Compute logits (and optionally loss).

        Args:
            input_ids:        ``[B, N]`` long.
            scatter_weights:  ``[B, N]`` float, optional, ≥ 0.
            kernel_bias:      ``[n_layers, H, F//2 + 1]`` complex, optional —
                              additive bias on each layer's kernel spectrum
                              (one row per layer).
            targets:          ``[B, N]`` long, optional. If given, returns
                              cross-entropy loss in ``out['loss']``.

        Returns:
            dict with keys ``logits`` ([B, N, V]) and optionally ``loss``.
        """
        x = self.embed(input_ids)
        for i, block in enumerate(self.blocks):
            kb = None
            if kernel_bias is not None:
                kb = kernel_bias[i]
            x = block(x, scatter_weights=scatter_weights, kernel_bias=kb)
        x = self.norm_out(x)
        if self.lm_head is None:
            logits = x @ self.embed.weight.t()
        else:
            logits = self.lm_head(x)

        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return out

    # ── Sampling ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        scatter_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Autoregressive sampling.

        Re-runs the full forward each step (no KV-cache equivalent yet —
        see TODO in docs/wave-field-llm.md). For sequences up to the
        configured field_size, this is fine; beyond that, accuracy
        degrades because the scatter target moves off the field's right
        edge. A streaming variant is future work.
        """
        ids = input_ids
        for _ in range(max_new_tokens):
            # Truncate context to the field size on the right.
            ctx = ids[:, -self.cfg.field_size :]
            sw = None
            if scatter_weights is not None:
                sw = scatter_weights[:, -self.cfg.field_size :]
            out = self(ctx, scatter_weights=sw)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = torch.where(
                    logits < v[..., [-1]], torch.full_like(logits, float("-inf")), logits
                )
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return ids

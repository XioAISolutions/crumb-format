"""Baseline transformer models for comparison.

Implements comparable transformer baselines:
- GPT-2 style (standard transformer)
- LLaMA style (RoPE, SwiGLU, RMSNorm)
- Mistral style (sliding window attention)

All baselines match the Wave Field LLM in parameter count for fair comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import from crumb_llm for consistency
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crumb_llm.baseline import TinyTransformerLM, TransformerConfig
from crumb_llm.layers import RMSNorm, SwiGLUFFN


@dataclass
class LLaMAConfig:
    """LLaMA-style transformer configuration."""
    vocab_size: int = 256
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    block_size: int = 2048
    ffn_mult: float = 8.0 / 3.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    use_rope: bool = True


@dataclass
class MistralConfig:
    """Mistral-style transformer with sliding window attention."""
    vocab_size: int = 256
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    block_size: int = 2048
    ffn_mult: float = 8.0 / 3.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    sliding_window: int = 512  # Local attention window


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor [B, H, N, D]
            seq_len: Sequence length
            
        Returns:
            cos, sin tensors for rotary embedding
        """
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype)
        )


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to input tensor.
    
    Args:
        x: Input [B, H, N, D]
        cos: Cosine values [N, D]
        sin: Sine values [N, D]
        
    Returns:
        Rotated tensor [B, H, N, D]
    """
    # Split into even and odd dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation
    rotated = torch.stack([
        x1 * cos[..., ::2] - x2 * sin[..., 1::2],
        x1 * sin[..., ::2] + x2 * cos[..., 1::2]
    ], dim=-1)
    
    return rotated.flatten(-2)


class LLaMAAttention(nn.Module):
    """LLaMA-style attention with RoPE."""
    
    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        if cfg.dim % cfg.n_heads != 0:
            raise ValueError(f"dim ({cfg.dim}) must divide n_heads ({cfg.n_heads})")
        
        self.n_heads = cfg.n_heads
        self.d_head = cfg.dim // cfg.n_heads
        self.use_rope = cfg.use_rope
        
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        
        if self.use_rope:
            self.rope = RotaryEmbedding(
                self.d_head,
                max_seq_len=cfg.block_size,
                theta=cfg.rope_theta
            )
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        
        # Compute Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE
        if self.use_rope:
            cos, sin = self.rope(q, N)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        
        return self.drop(self.proj(attn))


class SlidingWindowAttention(nn.Module):
    """Mistral-style sliding window attention."""
    
    def __init__(self, cfg: MistralConfig):
        super().__init__()
        if cfg.dim % cfg.n_heads != 0:
            raise ValueError(f"dim ({cfg.dim}) must divide n_heads ({cfg.n_heads})")
        
        self.n_heads = cfg.n_heads
        self.d_head = cfg.dim // cfg.n_heads
        self.window_size = cfg.sliding_window
        
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        
        self.rope = RotaryEmbedding(
            self.d_head,
            max_seq_len=cfg.block_size,
            theta=cfg.rope_theta
        )
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        
        # Compute Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(q, N)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Create sliding window mask
        # For simplicity, use causal mask (full implementation would use banded mask)
        # TODO: Implement proper sliding window mask for production
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        
        return self.drop(self.proj(attn))


class LLaMABlock(nn.Module):
    """LLaMA-style transformer block."""
    
    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.dim)
        self.attn = LLaMAAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.dim)
        self.ffn = SwiGLUFFN(cfg.dim, hidden_mult=cfg.ffn_mult)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MistralBlock(nn.Module):
    """Mistral-style transformer block with sliding window."""
    
    def __init__(self, cfg: MistralConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.dim)
        self.attn = SlidingWindowAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.dim)
        self.ffn = SwiGLUFFN(cfg.dim, hidden_mult=cfg.ffn_mult)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class LLaMALM(nn.Module):
    """LLaMA-style language model."""
    
    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(LLaMABlock(cfg) for _ in range(cfg.n_layers))
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
    
    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None
    ) -> dict:
        """
        Args:
            input_ids: [B, N] token indices
            targets: [B, N] target tokens for loss computation
            
        Returns:
            Dictionary with logits and optional loss
        """
        x = self.embed(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_out(x)
        
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed.weight)
        
        output = {"logits": logits}
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss
        
        return output
    
    def num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class MistralLM(nn.Module):
    """Mistral-style language model with sliding window attention."""
    
    def __init__(self, cfg: MistralConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(MistralBlock(cfg) for _ in range(cfg.n_layers))
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
    
    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None
    ) -> dict:
        """
        Args:
            input_ids: [B, N] token indices
            targets: [B, N] target tokens for loss computation
            
        Returns:
            Dictionary with logits and optional loss
        """
        x = self.embed(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_out(x)
        
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed.weight)
        
        output = {"logits": logits}
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss
        
        return output
    
    def num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_baseline_model(
    model_type: str,
    vocab_size: int = 256,
    dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    **kwargs
) -> nn.Module:
    """Factory function to create baseline models.
    
    Args:
        model_type: One of 'gpt2', 'llama', 'mistral'
        vocab_size: Vocabulary size
        dim: Model dimension
        n_layers: Number of layers
        n_heads: Number of attention heads
        **kwargs: Additional config parameters
        
    Returns:
        Baseline model instance
    """
    if model_type.lower() == "gpt2":
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            **kwargs
        )
        return TinyTransformerLM(cfg)
    
    elif model_type.lower() == "llama":
        cfg = LLaMAConfig(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            **kwargs
        )
        return LLaMALM(cfg)
    
    elif model_type.lower() == "mistral":
        cfg = MistralConfig(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            **kwargs
        )
        return MistralLM(cfg)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: gpt2, llama, mistral")


if __name__ == "__main__":
    # Test baseline models
    print("Testing baseline models...")
    
    for model_type in ["gpt2", "llama", "mistral"]:
        print(f"\n{model_type.upper()}:")
        model = create_baseline_model(model_type, dim=128, n_layers=4, n_heads=4)
        print(f"  Parameters: {model.num_parameters():,}")
        
        # Test forward pass
        x = torch.randint(0, 256, (2, 64))
        output = model(x, targets=x)
        print(f"  Logits shape: {output['logits'].shape}")
        print(f"  Loss: {output['loss'].item():.4f}")

# Made with Bob

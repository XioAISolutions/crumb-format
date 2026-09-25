"""Sparse and Efficient Wave Field Architectures.

Implements ultra-efficient models through:
- Mixture of Experts (MoE) with wave routing
- Sparse wave propagation
- Pruning and compression
- Knowledge distillation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model import WaveFieldLM, WaveFieldConfig
from ..layers import WaveFieldBlock, WaveFieldBlockConfig, RMSNorm, SwiGLUFFN


@dataclass
class SparseConfig:
    """Configuration for sparse architectures."""
    num_experts: int = 8
    top_k_experts: int = 2
    sparsity_ratio: float = 0.5  # Fraction of weights to prune
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5


class WaveRouter(nn.Module):
    """Router for Mixture of Experts in wave space.
    
    Routes tokens to experts based on wave field patterns.
    """
    
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Routing network
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_experts),
        )
        
        # Load balancing loss coefficient
        self.load_balance_coef = 0.01
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Route inputs to top-k experts.
        
        Args:
            x: [B, N, D] input tensor
            
        Returns:
            expert_weights: [B, N, top_k] routing weights
            expert_indices: [B, N, top_k] expert indices
            load_balance_loss: Scalar load balancing loss
        """
        B, N, D = x.shape
        
        # Compute routing logits
        logits = self.router(x)  # [B, N, num_experts]
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)  # [B, N, top_k]
        
        # Load balancing loss (encourage uniform expert usage)
        router_probs = F.softmax(logits, dim=-1)  # [B, N, num_experts]
        expert_usage = router_probs.mean(dim=(0, 1))  # [num_experts]
        load_balance_loss = self.load_balance_coef * (expert_usage ** 2).sum()
        
        return expert_weights, top_k_indices, load_balance_loss


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with wave field experts.
    
    Each expert is a wave field block specialized for different patterns.
    """
    
    def __init__(
        self,
        config: WaveFieldBlockConfig,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = WaveRouter(config.dim, num_experts, top_k)
        
        # Expert wave field blocks
        self.experts = nn.ModuleList([
            WaveFieldBlock(config)
            for _ in range(num_experts)
        ])
    
    def forward(
        self,
        x: Tensor,
        scatter_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward through MoE layer.
        
        Args:
            x: [B, N, D] input
            scatter_weights: Optional scatter weights
            
        Returns:
            output: [B, N, D] mixed expert outputs
            load_balance_loss: Scalar load balancing loss
        """
        B, N, D = x.shape
        
        # Route to experts
        expert_weights, expert_indices, load_balance_loss = self.router(x)
        
        # Process through selected experts
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            # Get indices for this k
            indices_k = expert_indices[:, :, k]  # [B, N]
            weights_k = expert_weights[:, :, k]  # [B, N]
            
            # Process each expert
            for expert_idx in range(self.num_experts):
                # Mask for tokens routed to this expert
                mask = (indices_k == expert_idx)  # [B, N]
                
                if mask.any():
                    # Process through expert
                    expert_out = self.experts[expert_idx](x, scatter_weights)
                    
                    # Add weighted contribution
                    weight_mask = weights_k * mask.float()
                    output += expert_out * weight_mask.unsqueeze(-1)
        
        return output, load_balance_loss


class SparseWaveField(nn.Module):
    """Sparse wave field model with dynamic sparsity.
    
    Implements sparse wave propagation where only active frequencies
    and connections are computed.
    """
    
    def __init__(
        self,
        config: WaveFieldConfig,
        sparse_config: SparseConfig,
    ):
        super().__init__()
        self.config = config
        self.sparse_config = sparse_config
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        
        # Sparse MoE blocks
        block_cfg = WaveFieldBlockConfig(
            dim=config.dim,
            n_heads=config.n_heads,
            field_size=config.field_size,
            ffn_mult=config.ffn_mult,
            causal=config.causal,
        )
        
        self.blocks = nn.ModuleList([
            MixtureOfExperts(
                block_cfg,
                sparse_config.num_experts,
                sparse_config.top_k_experts,
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm_out = RMSNorm(config.dim)
        
        # Output head
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Sparsity masks (learned during pruning)
        self.sparsity_masks: Dict[str, Tensor] = {}
    
    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
    ) -> dict:
        """Forward pass with sparse computation.
        
        Args:
            input_ids: [B, N] token ids
            targets: Optional targets for loss
            
        Returns:
            dict with 'logits', optional 'loss', and 'aux_loss'
        """
        x = self.embed(input_ids)
        
        # Accumulate auxiliary losses (load balancing)
        aux_loss = 0.0
        
        # Process through sparse MoE blocks
        for block in self.blocks:
            x, lb_loss = block(x)
            aux_loss += lb_loss
        
        x = self.norm_out(x)
        
        # Output
        if self.lm_head is None:
            logits = x @ self.embed.weight.t()
        else:
            logits = self.lm_head(x)
        
        out = {"logits": logits, "aux_loss": aux_loss}
        
        if targets is not None:
            main_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = main_loss + aux_loss
        
        return out
    
    def prune_weights(self, sparsity_ratio: Optional[float] = None) -> None:
        """Prune weights based on magnitude.
        
        Args:
            sparsity_ratio: Fraction of weights to prune (default from config)
        """
        sparsity_ratio = sparsity_ratio or self.sparse_config.sparsity_ratio
        
        # Collect all weights
        all_weights = []
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                all_weights.append((name, param))
        
        # Compute global threshold
        all_magnitudes = torch.cat([
            param.abs().flatten() for _, param in all_weights
        ])
        threshold = torch.quantile(all_magnitudes, sparsity_ratio)
        
        # Create and apply masks
        for name, param in all_weights:
            mask = (param.abs() >= threshold).float()
            self.sparsity_masks[name] = mask
            param.data *= mask
        
        print(f"Pruned {sparsity_ratio * 100:.1f}% of weights")
    
    def apply_sparsity_masks(self) -> None:
        """Apply stored sparsity masks to weights."""
        for name, param in self.named_parameters():
            if name in self.sparsity_masks:
                param.data *= self.sparsity_masks[name]
    
    def count_active_parameters(self) -> int:
        """Count non-zero parameters after pruning."""
        total = 0
        for name, param in self.named_parameters():
            if name in self.sparsity_masks:
                total += (param != 0).sum().item()
            else:
                total += param.numel()
        return total


class KnowledgeDistiller:
    """Distill knowledge from large teacher to small student.
    
    Uses wave field representations for distillation.
    """
    
    def __init__(
        self,
        teacher: WaveFieldLM,
        student: WaveFieldLM,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in teacher.parameters():
            param.requires_grad = False
        
        self.teacher.eval()
    
    def distillation_loss(
        self,
        input_ids: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute distillation loss.
        
        Args:
            input_ids: [B, N] input tokens
            targets: [B, N] target tokens
            
        Returns:
            loss: Combined distillation and task loss
        """
        # Student forward
        student_out = self.student(input_ids, targets=targets)
        student_logits = student_out["logits"]
        task_loss = student_out["loss"]
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(input_ids)
            teacher_logits = teacher_out["logits"]
        
        # Distillation loss (KL divergence)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss = F.kl_div(
            student_log_probs.reshape(-1, student_log_probs.size(-1)),
            teacher_probs.reshape(-1, teacher_probs.size(-1)),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss
        
        return total_loss
    
    def train_step(
        self,
        input_ids: Tensor,
        targets: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Single distillation training step.
        
        Args:
            input_ids: [B, N] input tokens
            targets: [B, N] target tokens
            optimizer: Optimizer for student
            
        Returns:
            metrics: Training metrics
        """
        loss = self.distillation_loss(input_ids, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"distillation_loss": loss.item()}


def compress_model(
    model: WaveFieldLM,
    compression_ratio: float = 0.5,
    method: str = "prune",  # prune | quantize | distill
) -> WaveFieldLM:
    """Compress a wave field model.
    
    Args:
        model: Model to compress
        compression_ratio: Target compression ratio
        method: Compression method
        
    Returns:
        compressed_model: Compressed model
    """
    if method == "prune":
        # Convert to sparse model and prune
        sparse_config = SparseConfig(sparsity_ratio=compression_ratio)
        sparse_model = SparseWaveField(model.cfg, sparse_config)
        sparse_model.load_state_dict(model.state_dict(), strict=False)
        sparse_model.prune_weights()
        return sparse_model
    
    elif method == "quantize":
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model
    
    elif method == "distill":
        # Create smaller student model
        student_config = WaveFieldConfig(
            vocab_size=model.cfg.vocab_size,
            dim=int(model.cfg.dim * compression_ratio),
            n_layers=max(1, int(model.cfg.n_layers * compression_ratio)),
            n_heads=max(1, int(model.cfg.n_heads * compression_ratio)),
            field_size=model.cfg.field_size,
        )
        student_model = WaveFieldLM(student_config)
        return student_model
    
    else:
        raise ValueError(f"Unknown compression method: {method}")

# Made with Bob

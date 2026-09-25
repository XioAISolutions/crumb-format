"""Optimizer configuration and learning rate scheduling for Wave Field LLM.

Provides:
- Parameter grouping (no weight decay for biases, layer norms)
- AdamW optimizer setup
- Learning rate schedulers (cosine, linear, constant with warmup)
- Gradient clipping utilities
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and learning rate scheduling."""
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    
    # Learning rate schedule
    scheduler: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 100
    max_steps: int = 10000
    min_lr_ratio: float = 0.1  # minimum LR as fraction of max LR
    
    # Gradient clipping
    grad_clip: float = 1.0
    grad_clip_norm_type: float = 2.0


def configure_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    filter_requires_grad: bool = True,
) -> AdamW:
    """Create AdamW optimizer with parameter grouping.
    
    Parameters without weight decay:
    - Biases
    - Layer norm parameters (weight and bias)
    - Embedding parameters
    
    Args:
        model: The model to optimize
        config: Optimizer configuration
        filter_requires_grad: Only include parameters that require gradients
        
    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters into groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if filter_requires_grad and not param.requires_grad:
            continue
            
        # No weight decay for biases, norms, and embeddings
        if (
            param.ndim < 2  # biases and norms (1D)
            or "norm" in name.lower()
            or "ln" in name.lower()
            or "bias" in name.lower()
            or "embed" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )
    
    return optimizer


def get_lr_scheduler(
    optimizer: Optimizer,
    config: OptimizerConfig,
) -> LambdaLR:
    """Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        config: Optimizer configuration with scheduler settings
        
    Returns:
        Learning rate scheduler
    """
    if config.scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
    elif config.scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
    elif config.scheduler == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            warmup_steps=config.warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine learning rate schedule with linear warmup.
    
    LR increases linearly from 0 to max during warmup, then follows
    a cosine decay to min_lr_ratio * max_lr.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        # Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        
        # Cosine annealing from 1.0 to min_lr_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear learning rate schedule with linear warmup.
    
    LR increases linearly from 0 to max during warmup, then decays
    linearly to min_lr_ratio * max_lr.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        # Linear decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
) -> LambdaLR:
    """Constant learning rate with linear warmup.
    
    LR increases linearly from 0 to max during warmup, then stays constant.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradients by global norm.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2.0 for L2 norm)
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
    )


def get_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute global gradient norm without clipping.
    
    Args:
        parameters: Model parameters
        
    Returns:
        Global gradient norm
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Made with Bob

"""Evaluation metrics for Wave Field LLM training.

Provides:
- Perplexity calculation
- Token accuracy
- Bits per character/byte
- CRUMB-specific metrics (section boundary detection, cross-reference resolution)
- Logging utilities
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MetricsAccumulator:
    """Accumulate metrics over multiple batches."""
    
    total_loss: float = 0.0
    total_tokens: int = 0
    total_correct: int = 0
    total_samples: int = 0
    
    # CRUMB-specific
    section_boundary_correct: int = 0
    section_boundary_total: int = 0
    ref_resolution_correct: int = 0
    ref_resolution_total: int = 0
    
    # Auxiliary losses
    diversity_loss: float = 0.0
    smoothness_loss: float = 0.0
    
    def update(
        self,
        loss: float,
        logits: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
        diversity_loss: Optional[float] = None,
        smoothness_loss: Optional[float] = None,
    ) -> None:
        """Update metrics with a batch.
        
        Args:
            loss: Batch loss
            logits: [B, N, V] predictions
            targets: [B, N] target IDs
            mask: [B, N] optional mask
            diversity_loss: Optional diversity loss value
            smoothness_loss: Optional smoothness loss value
        """
        B, N, V = logits.shape
        
        # Loss
        self.total_loss += loss * B * N
        
        # Tokens
        if mask is not None:
            num_tokens = mask.sum().item()
        else:
            num_tokens = B * N
        self.total_tokens += num_tokens
        
        # Accuracy
        preds = logits.argmax(dim=-1)  # [B, N]
        if mask is not None:
            correct = ((preds == targets) * mask).sum().item()
        else:
            correct = (preds == targets).sum().item()
        self.total_correct += correct
        
        # Samples
        self.total_samples += B
        
        # Auxiliary losses
        if diversity_loss is not None:
            self.diversity_loss += diversity_loss * B
        if smoothness_loss is not None:
            self.smoothness_loss += smoothness_loss * B
    
    def compute(self) -> dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        if self.total_tokens == 0:
            return {}
        
        avg_loss = self.total_loss / self.total_tokens
        accuracy = self.total_correct / self.total_tokens
        perplexity = math.exp(min(avg_loss, 20.0))
        bpc = avg_loss / math.log(2)
        
        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "bpc": bpc,
        }
        
        # Auxiliary losses
        if self.total_samples > 0:
            if self.diversity_loss > 0:
                metrics["diversity_loss"] = self.diversity_loss / self.total_samples
            if self.smoothness_loss > 0:
                metrics["smoothness_loss"] = self.smoothness_loss / self.total_samples
        
        # CRUMB-specific metrics
        if self.section_boundary_total > 0:
            metrics["section_boundary_acc"] = (
                self.section_boundary_correct / self.section_boundary_total
            )
        if self.ref_resolution_total > 0:
            metrics["ref_resolution_acc"] = (
                self.ref_resolution_correct / self.ref_resolution_total
            )
        
        return metrics
    
    def reset(self) -> None:
        """Reset all counters."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.total_correct = 0
        self.total_samples = 0
        self.section_boundary_correct = 0
        self.section_boundary_total = 0
        self.ref_resolution_correct = 0
        self.ref_resolution_total = 0
        self.diversity_loss = 0.0
        self.smoothness_loss = 0.0


def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity.
    
    Args:
        loss: Cross-entropy loss (nats)
        
    Returns:
        Perplexity = exp(loss)
    """
    return math.exp(min(loss, 20.0))


def compute_bits_per_char(loss: float) -> float:
    """Convert cross-entropy loss to bits per character.
    
    Args:
        loss: Cross-entropy loss (nats)
        
    Returns:
        Bits per character = loss / log(2)
    """
    return loss / math.log(2)


def compute_token_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
) -> float:
    """Compute token-level accuracy.
    
    Args:
        logits: [B, N, V] predictions
        targets: [B, N] target IDs
        mask: [B, N] optional mask
        
    Returns:
        Accuracy as a float in [0, 1]
    """
    preds = logits.argmax(dim=-1)
    
    if mask is not None:
        correct = ((preds == targets) * mask).sum().item()
        total = mask.sum().item()
    else:
        correct = (preds == targets).sum().item()
        total = targets.numel()
    
    return correct / max(total, 1)


def compute_top_k_accuracy(
    logits: Tensor,
    targets: Tensor,
    k: int = 5,
    mask: Optional[Tensor] = None,
) -> float:
    """Compute top-k accuracy.
    
    Args:
        logits: [B, N, V] predictions
        targets: [B, N] target IDs
        k: Number of top predictions to consider
        mask: [B, N] optional mask
        
    Returns:
        Top-k accuracy as a float in [0, 1]
    """
    # Get top-k predictions
    top_k_preds = logits.topk(k, dim=-1).indices  # [B, N, k]
    
    # Check if target is in top-k
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1)  # [B, N]
    
    if mask is not None:
        correct = correct * mask
        total = mask.sum().item()
    else:
        total = targets.numel()
    
    return correct.sum().item() / max(total, 1)


def detect_section_boundaries(
    tokens: Tensor,
    section_token_id: int,
) -> Tensor:
    """Detect section boundary positions in token sequences.
    
    Args:
        tokens: [B, N] token IDs
        section_token_id: Token ID for section separator
        
    Returns:
        [B, N] boolean mask where True indicates section boundary
    """
    return tokens == section_token_id


def compute_section_boundary_accuracy(
    logits: Tensor,
    targets: Tensor,
    section_token_id: int,
) -> tuple[int, int]:
    """Compute accuracy of section boundary detection.
    
    Args:
        logits: [B, N, V] predictions
        targets: [B, N] target IDs
        section_token_id: Token ID for section separator
        
    Returns:
        Tuple of (correct, total) section boundary predictions
    """
    preds = logits.argmax(dim=-1)
    
    # Find actual section boundaries
    is_boundary = targets == section_token_id
    
    # Check if predicted correctly
    correct = ((preds == targets) * is_boundary).sum().item()
    total = is_boundary.sum().item()
    
    return correct, total


class MetricsLogger:
    """Logger for training metrics with moving averages."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: dict[str, list[float]] = {}
    
    def log(self, metrics: dict[str, float]) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
            
            # Keep only recent history
            if len(self.history[name]) > self.window_size:
                self.history[name] = self.history[name][-self.window_size:]
    
    def get_average(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """Get moving average of a metric.
        
        Args:
            name: Metric name
            window: Window size (default: use logger's window_size)
            
        Returns:
            Moving average or None if metric not found
        """
        if name not in self.history or not self.history[name]:
            return None
        
        window = window or self.window_size
        recent = self.history[name][-window:]
        return sum(recent) / len(recent)
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest value or None if metric not found
        """
        if name not in self.history or not self.history[name]:
            return None
        return self.history[name][-1]
    
    def format_metrics(
        self,
        metrics: dict[str, float],
        precision: int = 4,
    ) -> str:
        """Format metrics as a string.
        
        Args:
            metrics: Dictionary of metrics
            precision: Number of decimal places
            
        Returns:
            Formatted string
        """
        parts = []
        for name, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{name}={value:.{precision}f}")
            else:
                parts.append(f"{name}={value}")
        return " ".join(parts)
    
    def reset(self) -> None:
        """Clear all history."""
        self.history.clear()

# Made with Bob

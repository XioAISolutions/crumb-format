"""Loss functions for Wave Field LLM training.

Provides:
- Cross-entropy loss (primary)
- Spectral diversity loss (encourage different frequencies per head)
- Field smoothness loss (penalize high-frequency noise)
- Combined loss with configurable weights
- Per-token loss masking for padding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossConfig:
    """Configuration for loss computation."""
    
    # Primary loss
    label_smoothing: float = 0.0
    ignore_index: int = -100
    
    # Auxiliary losses
    spectral_diversity_weight: float = 0.0
    field_smoothness_weight: float = 0.0
    
    # Spectral diversity settings
    diversity_temperature: float = 1.0
    
    # Field smoothness settings
    smoothness_order: int = 2  # order of derivative to penalize


class WaveFieldLoss(nn.Module):
    """Combined loss for Wave Field LLM training.
    
    Computes:
    1. Cross-entropy loss (primary)
    2. Spectral diversity loss (optional, encourages heads to use different frequencies)
    3. Field smoothness loss (optional, penalizes high-frequency noise)
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        field_states: Optional[Tensor] = None,
        head_spectra: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Compute combined loss.
        
        Args:
            logits: [B, N, V] model predictions
            targets: [B, N] target token IDs
            field_states: [B, H, F, D] field states (for smoothness loss)
            head_spectra: [B, H, F//2+1] head frequency spectra (for diversity loss)
            mask: [B, N] optional mask (1 = compute loss, 0 = ignore)
            
        Returns:
            Dictionary with 'loss' and individual loss components
        """
        # Primary cross-entropy loss
        ce_loss = self.cross_entropy_loss(logits, targets, mask)
        
        total_loss = ce_loss
        losses = {"ce_loss": ce_loss}
        
        # Spectral diversity loss
        if self.config.spectral_diversity_weight > 0 and head_spectra is not None:
            div_loss = self.spectral_diversity_loss(head_spectra)
            total_loss = total_loss + self.config.spectral_diversity_weight * div_loss
            losses["diversity_loss"] = div_loss
        
        # Field smoothness loss
        if self.config.field_smoothness_weight > 0 and field_states is not None:
            smooth_loss = self.field_smoothness_loss(field_states)
            total_loss = total_loss + self.config.field_smoothness_weight * smooth_loss
            losses["smoothness_loss"] = smooth_loss
        
        losses["loss"] = total_loss
        return losses
    
    def cross_entropy_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute cross-entropy loss with optional label smoothing and masking.
        
        Args:
            logits: [B, N, V] predictions
            targets: [B, N] target IDs
            mask: [B, N] optional mask
            
        Returns:
            Scalar loss
        """
        B, N, V = logits.shape
        
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing,
            reduction="none",
        )
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat
            return loss.sum() / mask_flat.sum().clamp(min=1.0)
        
        return loss.mean()
    
    def spectral_diversity_loss(self, head_spectra: Tensor) -> Tensor:
        """Encourage different heads to use different frequency ranges.
        
        Computes pairwise cosine similarity between head spectra and
        penalizes high similarity (we want heads to be diverse).
        
        Args:
            head_spectra: [B, H, F//2+1] complex or [B, H, F//2+1, 2] real
            
        Returns:
            Scalar diversity loss (lower = more diverse)
        """
        # Convert to magnitude if complex
        if head_spectra.is_complex():
            spectra = head_spectra.abs()
        elif head_spectra.shape[-1] == 2:
            # Real representation [B, H, F, 2]
            spectra = torch.sqrt(head_spectra[..., 0]**2 + head_spectra[..., 1]**2)
        else:
            spectra = head_spectra
        
        # Normalize spectra
        spectra = F.normalize(spectra, p=2, dim=-1)  # [B, H, F]
        
        # Compute pairwise cosine similarity
        # [B, H, F] @ [B, F, H] -> [B, H, H]
        similarity = torch.bmm(spectra, spectra.transpose(1, 2))
        
        # Mask out diagonal (self-similarity)
        H = similarity.shape[1]
        mask = ~torch.eye(H, dtype=torch.bool, device=similarity.device)
        mask = mask.unsqueeze(0).expand_as(similarity)
        
        # Average off-diagonal similarities
        off_diag_sim = similarity[mask].view(similarity.shape[0], H, H - 1)
        
        # Apply temperature and compute loss
        # We want to minimize similarity, so loss = mean(similarity)
        loss = (off_diag_sim / self.config.diversity_temperature).mean()
        
        return loss
    
    def field_smoothness_loss(self, field_states: Tensor) -> Tensor:
        """Penalize high-frequency noise in field states.
        
        Computes finite differences and penalizes large variations,
        encouraging smooth field evolution.
        
        Args:
            field_states: [B, H, F, D] field states
            
        Returns:
            Scalar smoothness loss (lower = smoother)
        """
        # Compute finite differences along field dimension
        if self.config.smoothness_order == 1:
            # First-order: |f[i+1] - f[i]|
            diff = field_states[:, :, 1:, :] - field_states[:, :, :-1, :]
        elif self.config.smoothness_order == 2:
            # Second-order: |f[i+1] - 2*f[i] + f[i-1]|
            diff = (
                field_states[:, :, 2:, :]
                - 2 * field_states[:, :, 1:-1, :]
                + field_states[:, :, :-2, :]
            )
        else:
            raise ValueError(f"Unsupported smoothness order: {self.config.smoothness_order}")
        
        # L2 norm of differences
        loss = (diff ** 2).mean()
        
        return loss


def compute_perplexity(loss: Tensor) -> Tensor:
    """Convert cross-entropy loss to perplexity.
    
    Args:
        loss: Cross-entropy loss (scalar or tensor)
        
    Returns:
        Perplexity = exp(loss)
    """
    # Clamp to prevent overflow
    loss_clamped = torch.clamp(loss, max=20.0)
    return torch.exp(loss_clamped)


def compute_bits_per_token(loss: Tensor) -> Tensor:
    """Convert cross-entropy loss to bits per token.
    
    Args:
        loss: Cross-entropy loss (nats)
        
    Returns:
        Bits per token = loss / log(2)
    """
    return loss / torch.log(torch.tensor(2.0, device=loss.device))


def masked_loss(
    loss: Tensor,
    mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Apply mask to loss and reduce.
    
    Args:
        loss: [B, N] or [B*N] loss values
        mask: [B, N] or [B*N] mask (1 = include, 0 = exclude)
        reduction: 'mean' | 'sum' | 'none'
        
    Returns:
        Reduced loss
    """
    masked = loss * mask
    
    if reduction == "none":
        return masked
    elif reduction == "sum":
        return masked.sum()
    elif reduction == "mean":
        return masked.sum() / mask.sum().clamp(min=1.0)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

# Made with Bob

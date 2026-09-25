"""Checkpoint management for Wave Field LLM training.

Provides:
- Save/load model state
- Save/load optimizer state
- Save/load training state (epoch, step, best metrics)
- Checkpoint rotation (keep best N checkpoints)
- Resume training from checkpoint
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    
    step: int
    epoch: int
    loss: float
    learning_rate: float
    
    # Optional metrics
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    eval_loss: Optional[float] = None
    
    # Training state
    total_tokens: int = 0
    wallclock_seconds: float = 0.0
    
    # Model info
    model_config: Optional[dict] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manage model checkpoints with rotation and best-model tracking."""
    
    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_best_n: int = 3,
        keep_last_n: int = 2,
        metric_name: str = "loss",
        metric_mode: str = "min",
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
            keep_last_n: Number of most recent checkpoints to keep
            metric_name: Metric to use for "best" selection
            metric_mode: 'min' or 'max' for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best_n = keep_best_n
        self.keep_last_n = keep_last_n
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        # Track checkpoints
        self.best_checkpoints: list[tuple[float, Path]] = []  # (metric, path)
        self.recent_checkpoints: list[Path] = []
        
        # Load existing checkpoint info if available
        self._load_checkpoint_index()
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        metadata: CheckpointMetadata,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            metadata: Checkpoint metadata
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint filename
        if is_best:
            filename = "best.pt"
        else:
            filename = f"checkpoint_step_{metadata.step}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata.to_dict(),
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update tracking
        if is_best:
            metric_value = getattr(metadata, self.metric_name, metadata.loss)
            self._add_best_checkpoint(metric_value, checkpoint_path)
        else:
            self._add_recent_checkpoint(checkpoint_path)
        
        # Save checkpoint index
        self._save_checkpoint_index()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> CheckpointMetadata:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint to
            
        Returns:
            Checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load metadata
        metadata = CheckpointMetadata.from_dict(checkpoint["metadata"])
        
        return metadata
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if self.recent_checkpoints:
            return self.recent_checkpoints[-1]
        
        # Fallback: search directory
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        
        if checkpoints:
            return checkpoints[-1]
        
        return None
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None if no checkpoints exist
        """
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return best_path
        
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        
        return None
    
    def _add_best_checkpoint(self, metric_value: float, path: Path) -> None:
        """Add checkpoint to best list and rotate if needed."""
        # Add to list
        self.best_checkpoints.append((metric_value, path))
        
        # Sort by metric
        reverse = self.metric_mode == "max"
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse)
        
        # Keep only top N
        if len(self.best_checkpoints) > self.keep_best_n:
            # Remove worst checkpoint
            _, removed_path = self.best_checkpoints.pop()
            if removed_path.exists() and removed_path.name != "best.pt":
                removed_path.unlink()
    
    def _add_recent_checkpoint(self, path: Path) -> None:
        """Add checkpoint to recent list and rotate if needed."""
        self.recent_checkpoints.append(path)
        
        # Keep only last N
        if len(self.recent_checkpoints) > self.keep_last_n:
            removed_path = self.recent_checkpoints.pop(0)
            if removed_path.exists():
                removed_path.unlink()
    
    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index = {
            "best_checkpoints": [
                {"metric": m, "path": str(p)}
                for m, p in self.best_checkpoints
            ],
            "recent_checkpoints": [str(p) for p in self.recent_checkpoints],
        }
        
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    
    def _load_checkpoint_index(self) -> None:
        """Load checkpoint index from disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        
        if not index_path.exists():
            return
        
        try:
            with open(index_path) as f:
                index = json.load(f)
            
            self.best_checkpoints = [
                (item["metric"], Path(item["path"]))
                for item in index.get("best_checkpoints", [])
                if Path(item["path"]).exists()
            ]
            
            self.recent_checkpoints = [
                Path(p)
                for p in index.get("recent_checkpoints", [])
                if Path(p).exists()
            ]
        except (json.JSONDecodeError, KeyError):
            # Corrupted index, ignore
            pass


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Save a checkpoint (simple interface).
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer to save
        metadata: Optional metadata dictionary
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load a checkpoint (simple interface).
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def resume_training(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    device: Optional[torch.device] = None,
) -> tuple[int, dict[str, Any]]:
    """Resume training from latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to load checkpoint to
        
    Returns:
        Tuple of (start_step, metadata)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find latest checkpoint
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    
    if not checkpoints:
        # Check for best.pt
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            checkpoints = [best_path]
        else:
            return 0, {}
    
    latest = checkpoints[-1]
    
    # Load checkpoint
    checkpoint = load_checkpoint(latest, model, optimizer, device)
    
    metadata = checkpoint.get("metadata", {})
    start_step = metadata.get("step", 0) + 1
    
    return start_step, metadata

# Made with Bob

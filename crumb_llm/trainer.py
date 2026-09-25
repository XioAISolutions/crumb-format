"""Comprehensive trainer for Wave Field LLM with all Phase 2 features.

Provides:
- WaveFieldTrainer class with full training loop
- Distributed training support (DDP)
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling
- Checkpoint management
- Validation and early stopping
- Comprehensive logging
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import CheckpointManager, CheckpointMetadata
from .loss import LossConfig, WaveFieldLoss
from .metrics import MetricsAccumulator, MetricsLogger
from .optimizer import OptimizerConfig, configure_optimizer, get_lr_scheduler, clip_gradients


@dataclass
class TrainerConfig:
    """Configuration for Wave Field LLM trainer."""
    
    # Training
    max_steps: int = 10000
    max_epochs: Optional[int] = None
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Mixed precision
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 | float16
    
    # Gradient checkpointing
    gradient_checkpointing: bool = False
    
    # Validation
    eval_every: int = 500
    eval_steps: int = 100
    
    # Logging
    log_every: int = 10
    log_grad_norm: bool = True
    
    # Checkpointing
    save_every: int = 1000
    checkpoint_dir: str = "./checkpoints"
    keep_best_n: int = 3
    keep_last_n: int = 2
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "eval_loss"
    early_stopping_mode: str = "min"
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # Curriculum learning
    curriculum_learning: bool = False
    curriculum_start_length: int = 128
    curriculum_end_length: int = 512
    curriculum_steps: int = 5000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 42


class WaveFieldTrainer:
    """Comprehensive trainer for Wave Field LLM."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: TrainerConfig,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            eval_loader: Validation data loader (optional)
            config: Trainer configuration
        """
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Setup device
        self.device = torch.device(config.device)
        
        # Setup distributed training
        if config.distributed:
            self._setup_distributed()
            model = model.to(self.device)
            model = DDP(model, device_ids=[config.local_rank])
        else:
            model = model.to(self.device)
        
        self.model = model
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing and hasattr(model, 'blocks'):
            self._setup_gradient_checkpointing()
        
        # Setup optimizer and scheduler
        self.optimizer = configure_optimizer(model, config.optimizer)
        self.scheduler = get_lr_scheduler(self.optimizer, config.optimizer)
        
        # Setup loss function
        self.loss_fn = WaveFieldLoss(config.loss)
        
        # Setup mixed precision
        self.use_amp = config.mixed_precision and config.device == "cuda"
        if self.use_amp:
            amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
            self.amp_dtype = amp_dtype
            self.scaler = GradScaler(enabled=True)
        else:
            self.amp_dtype = torch.float32
            self.scaler = GradScaler(enabled=False)
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_best_n=config.keep_best_n,
            keep_last_n=config.keep_last_n,
            metric_name=config.early_stopping_metric,
            metric_mode=config.early_stopping_mode,
        )
        
        # Setup metrics tracking
        self.metrics_logger = MetricsLogger(window_size=100)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
    def _setup_distributed(self) -> None:
        """Setup distributed training."""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        
        self.config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.config.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(self.config.local_rank)
    
    def _setup_gradient_checkpointing(self) -> None:
        """Setup gradient checkpointing for memory efficiency."""
        from torch.utils.checkpoint import checkpoint
        
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        for block in model.blocks:
            block._orig_forward = block.forward
            
            def make_ckpt_fn(b):
                def ckpt_forward(*args, **kwargs):
                    return checkpoint(b._orig_forward, *args, use_reentrant=False, **kwargs)
                return ckpt_forward
            
            block.forward = make_ckpt_fn(block)
    
    def train(self) -> dict:
        """Run training loop.
        
        Returns:
            Dictionary of final metrics
        """
        self.model.train()
        
        # Progress bar
        if self.is_main_process():
            pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            epoch_metrics = self._train_epoch()
            
            # Update progress bar
            if self.is_main_process():
                pbar.update(epoch_metrics.get('steps', 0))
                pbar.set_postfix(epoch_metrics)
            
            # Check early stopping
            if self.config.early_stopping and self._should_stop_early():
                if self.is_main_process():
                    print(f"Early stopping triggered at step {self.global_step}")
                break
            
            self.epoch += 1
            
            # Check max epochs
            if self.config.max_epochs and self.epoch >= self.config.max_epochs:
                break
        
        if self.is_main_process():
            pbar.close()
        
        # Final evaluation
        if self.eval_loader is not None:
            final_metrics = self.evaluate()
        else:
            final_metrics = {}
        
        # Save final checkpoint
        if self.is_main_process():
            self._save_checkpoint(is_best=False)
        
        return final_metrics
    
    def _train_epoch(self) -> dict:
        """Train for one epoch.
        
        Returns:
            Dictionary of epoch metrics
        """
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            if self.global_step >= self.config.max_steps:
                break
            
            # Training step
            loss, metrics = self._train_step(batch)
            
            epoch_loss += loss
            epoch_steps += 1
            
            # Logging
            if self.global_step % self.config.log_every == 0 and self.is_main_process():
                self._log_metrics(metrics)
            
            # Validation
            if self.eval_loader and self.global_step % self.config.eval_every == 0:
                eval_metrics = self.evaluate()
                
                if self.is_main_process():
                    self._log_metrics(eval_metrics, prefix="eval")
                    
                    # Check if best model
                    is_best = self._is_best_model(eval_metrics)
                    if is_best:
                        self.best_metric = eval_metrics[self.config.early_stopping_metric]
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                
                self.model.train()
            
            # Checkpointing
            if self.global_step % self.config.save_every == 0 and self.is_main_process():
                self._save_checkpoint(is_best=False)
            
            self.global_step += 1
        
        return {
            "loss": epoch_loss / max(epoch_steps, 1),
            "steps": epoch_steps,
        }
    
    def _train_step(self, batch: dict) -> tuple[float, dict]:
        """Execute one training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Curriculum learning
        if self.config.curriculum_learning:
            seq_len = self._get_curriculum_length()
            batch = self._truncate_batch(batch, seq_len)
        
        # Gradient accumulation
        loss_accum = 0.0
        
        for micro_step in range(self.config.gradient_accumulation_steps):
            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(batch["input_ids"], targets=batch["labels"])
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            loss_accum += loss.item()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = clip_gradients(
            self.model.parameters(),
            max_norm=self.config.optimizer.grad_clip,
            norm_type=self.config.optimizer.grad_clip_norm_type,
        )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Scheduler step
        self.scheduler.step()
        
        # Update token count
        self.total_tokens += batch["input_ids"].numel()
        
        # Compute metrics
        metrics = {
            "loss": loss_accum,
            "lr": self.scheduler.get_last_lr()[0],
            "step": self.global_step,
            "tokens": self.total_tokens,
        }
        
        if self.config.log_grad_norm:
            metrics["grad_norm"] = grad_norm
        
        return loss_accum, metrics
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate model on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        metrics_acc = MetricsAccumulator()
        
        eval_steps = min(self.config.eval_steps, len(self.eval_loader))
        
        for step, batch in enumerate(self.eval_loader):
            if step >= eval_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(batch["input_ids"], targets=batch["labels"])
            
            # Update metrics
            metrics_acc.update(
                loss=outputs["loss"].item(),
                logits=outputs["logits"],
                targets=batch["labels"],
            )
        
        metrics = metrics_acc.compute()
        metrics["eval_loss"] = metrics["loss"]
        
        return metrics
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save checkpoint."""
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        metadata = CheckpointMetadata(
            step=self.global_step,
            epoch=self.epoch,
            loss=self.metrics_logger.get_latest("loss") or 0.0,
            learning_rate=self.scheduler.get_last_lr()[0],
            total_tokens=self.total_tokens,
            wallclock_seconds=time.time() - self.start_time,
        )
        
        self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=self.optimizer,
            metadata=metadata,
            is_best=is_best,
        )
    
    def _log_metrics(self, metrics: dict, prefix: str = "train") -> None:
        """Log metrics."""
        self.metrics_logger.log(metrics)
        
        # Format and print
        formatted = self.metrics_logger.format_metrics(metrics)
        print(f"[{prefix}] step={self.global_step} {formatted}")
    
    def _is_best_model(self, metrics: dict) -> bool:
        """Check if current model is the best so far."""
        metric_value = metrics.get(self.config.early_stopping_metric)
        
        if metric_value is None:
            return False
        
        if self.config.early_stopping_mode == "min":
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria is met."""
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _get_curriculum_length(self) -> int:
        """Get current sequence length for curriculum learning."""
        if self.global_step >= self.config.curriculum_steps:
            return self.config.curriculum_end_length
        
        progress = self.global_step / self.config.curriculum_steps
        length = (
            self.config.curriculum_start_length +
            (self.config.curriculum_end_length - self.config.curriculum_start_length) * progress
        )
        
        return int(length)
    
    def _truncate_batch(self, batch: dict, seq_len: int) -> dict:
        """Truncate batch to specified sequence length."""
        return {
            k: v[:, :seq_len] if isinstance(v, torch.Tensor) and v.ndim >= 2 else v
            for k, v in batch.items()
        }
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (for logging/saving)."""
        return not self.config.distributed or self.config.local_rank == 0
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=self.optimizer,
            device=self.device,
        )
        
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self.total_tokens = metadata.total_tokens
        
        # Adjust scheduler
        for _ in range(self.global_step):
            self.scheduler.step()
        
        if self.is_main_process():
            print(f"Resumed from checkpoint at step {self.global_step}")

# Made with Bob

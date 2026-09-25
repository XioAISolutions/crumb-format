#!/usr/bin/env python3
"""
Wave Field LLM Model Zoo - Training Pipeline

Complete training orchestration system with multi-stage training,
distributed training support, checkpoint management, and comprehensive logging.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crumb_llm.model import WaveFieldLLM
from crumb_llm.trainer import Trainer
from crumb_llm.data_enhanced import create_dataloaders
from crumb_llm.optimizer import create_optimizer
from crumb_llm.checkpoint import CheckpointManager
from crumb_llm.metrics import MetricsTracker

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML."""
    
    # Model configuration
    model_name: str
    model_version: str
    architecture: Dict[str, Any]
    wave_field: Dict[str, Any]
    
    # Training configuration
    total_tokens: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    warmup_steps: int
    max_steps: int
    lr_decay_style: str
    mixed_precision: str
    
    # Checkpointing
    save_interval: int
    eval_interval: int
    keep_last_n_checkpoints: int
    
    # Early stopping
    early_stopping_patience: int
    early_stopping_metric: str
    
    # Dataset configuration
    dataset_mix: Dict[str, float]
    seq_length: int
    num_workers: int
    val_split: float
    val_samples: int
    
    # Hardware configuration
    min_gpus: int
    recommended_gpus: int
    
    # Logging configuration
    log_interval: int
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    tensorboard: bool = True
    log_dir: str = "logs"
    
    # Distributed training
    distributed_backend: str = "nccl"
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    
    # Optional features
    resume_from: Optional[str] = None
    stages: Optional[List[Dict[str, Any]]] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            model_name=config['model']['name'],
            model_version=config['model']['version'],
            architecture=config['model']['architecture'],
            wave_field=config['model']['wave_field'],
            total_tokens=config['training']['total_tokens'],
            batch_size=config['training']['batch_size'],
            gradient_accumulation=config['training']['gradient_accumulation'],
            learning_rate=config['training']['learning_rate'],
            min_learning_rate=config['training']['min_learning_rate'],
            weight_decay=config['training']['weight_decay'],
            beta1=config['training']['beta1'],
            beta2=config['training']['beta2'],
            grad_clip=config['training']['grad_clip'],
            warmup_steps=config['training']['warmup_steps'],
            max_steps=config['training']['max_steps'],
            lr_decay_style=config['training']['lr_decay_style'],
            mixed_precision=config['training']['mixed_precision'],
            save_interval=config['training']['save_interval'],
            eval_interval=config['training']['eval_interval'],
            keep_last_n_checkpoints=config['training']['keep_last_n_checkpoints'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            early_stopping_metric=config['training']['early_stopping_metric'],
            dataset_mix=config['dataset']['mix'],
            seq_length=config['dataset']['seq_length'],
            num_workers=config['dataset']['num_workers'],
            val_split=config['dataset']['val_split'],
            val_samples=config['dataset']['val_samples'],
            min_gpus=config['hardware']['min_gpus'],
            recommended_gpus=config['hardware']['recommended_gpus'],
            log_interval=config['logging']['log_interval'],
            wandb_project=config['logging'].get('wandb_project'),
            wandb_entity=config['logging'].get('wandb_entity'),
            tensorboard=config['logging'].get('tensorboard', True),
            log_dir=config['logging']['log_dir'],
            distributed_backend=config.get('distributed', {}).get('backend', 'nccl'),
            use_fsdp=config.get('distributed', {}).get('use_fsdp', False),
            fsdp_sharding_strategy=config.get('distributed', {}).get('fsdp_sharding_strategy', 'FULL_SHARD'),
            resume_from=config['training'].get('resume_from'),
            stages=config['training'].get('stages'),
        )


class TrainingPipeline:
    """Complete training pipeline orchestrator."""
    
    def __init__(
        self,
        config: TrainingConfig,
        output_dir: str,
        num_gpus: int = 1,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_gpus = num_gpus
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = local_rank == 0
        
        # Initialize distributed training if needed
        if world_size > 1:
            self._init_distributed()
        
        # Set device
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Initialize logging
        self._init_logging()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.output_dir / "checkpoints",
            keep_last_n=config.keep_last_n_checkpoints,
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._resume_from_checkpoint(config.resume_from)
    
    def _init_distributed(self):
        """Initialize distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.distributed_backend,
                init_method='env://',
                world_size=self.world_size,
                rank=self.local_rank,
            )
        
        torch.cuda.set_device(self.local_rank)
        logger.info(f"Initialized distributed training: rank {self.local_rank}/{self.world_size}")
    
    def _init_logging(self):
        """Initialize logging systems."""
        if not self.is_main_process:
            return
        
        # Create log directory
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases
        if WANDB_AVAILABLE and self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"{self.config.model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.config),
            )
            logger.info("Initialized Weights & Biases logging")
        
        # Initialize TensorBoard
        if TENSORBOARD_AVAILABLE and self.config.tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
            logger.info("Initialized TensorBoard logging")
        else:
            self.tensorboard_writer = None
    
    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
        logger.info(f"Creating model: {self.config.model_name}")
        
        model = WaveFieldLLM(
            vocab_size=self.config.architecture['vocab_size'],
            dim=self.config.architecture['dim'],
            n_layers=self.config.architecture['n_layers'],
            n_heads=self.config.architecture['n_heads'],
            field_size=self.config.architecture['field_size'],
            max_seq_len=self.config.architecture['max_seq_len'],
            dropout=self.config.architecture['dropout'],
        )
        
        model = model.to(self.device)
        
        # Wrap with distributed training
        if self.world_size > 1:
            if self.config.use_fsdp:
                # Use FSDP for large models
                model = FSDP(
                    model,
                    sharding_strategy=self.config.fsdp_sharding_strategy,
                    auto_wrap_policy=transformer_auto_wrap_policy,
                    device_id=self.local_rank,
                )
                logger.info("Wrapped model with FSDP")
            else:
                # Use DDP for smaller models
                model = DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
                logger.info("Wrapped model with DDP")
        
        # Log model info
        if self.is_main_process:
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {num_params:,}")
            logger.info(f"Model size: {num_params * 4 / 1e9:.2f} GB (fp32)")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer and learning rate scheduler."""
        return create_optimizer(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
        )
    
    def _create_dataloaders(self) -> Tuple[Any, Any]:
        """Create training and validation data loaders."""
        logger.info("Creating data loaders")
        
        train_loader, val_loader = create_dataloaders(
            dataset_mix=self.config.dataset_mix,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            num_workers=self.config.num_workers,
            val_split=self.config.val_split,
            world_size=self.world_size,
            rank=self.local_rank,
        )
        
        return train_loader, val_loader
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, (DDP, FSDP)):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training")
        logger.info(f"Total steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation}")
        
        # Training loop
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            epoch_loss = self._train_epoch()
            
            # Validation
            if self.global_step % self.config.eval_interval == 0:
                val_metrics = self._validate()
                
                # Check for improvement
                current_metric = val_metrics.get(self.config.early_stopping_metric, float('inf'))
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    
                    # Save best checkpoint
                    if self.is_main_process:
                        self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                    break
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0 and self.is_main_process:
                self._save_checkpoint()
        
        logger.info("Training completed")
        
        # Final validation
        if self.is_main_process:
            final_metrics = self._validate()
            logger.info(f"Final validation metrics: {final_metrics}")
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            disable=not self.is_main_process,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = self._train_step(batch)
            epoch_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss:.4f}",
                'step': self.global_step,
            })
            
            # Logging
            if self.global_step % self.config.log_interval == 0 and self.is_main_process:
                self._log_metrics({
                    'train/loss': loss,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.epoch,
                })
            
            # Check if max steps reached
            if self.global_step >= self.config.max_steps:
                break
        
        return epoch_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels)
        loss = outputs['loss'] / self.config.gradient_accumulation
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation == 0:
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item() * self.config.gradient_accumulation
    
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        logger.info("Running validation")
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=not self.is_main_process):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, labels=labels)
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {
            'val_loss': avg_loss,
            'val_perplexity': torch.exp(torch.tensor(avg_loss)).item(),
        }
        
        if self.is_main_process:
            self._log_metrics({f'val/{k}': v for k, v in metrics.items()})
            logger.info(f"Validation metrics: {metrics}")
        
        self.model.train()
        return metrics
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, (DDP, FSDP)) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'config': vars(self.config),
        }
        
        if is_best:
            checkpoint_path = self.output_dir / "checkpoints" / "checkpoint-best.pt"
        else:
            checkpoint_path = self.output_dir / "checkpoints" / f"checkpoint-step-{self.global_step}.pt"
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Manage checkpoint rotation
        self.checkpoint_manager.save_checkpoint(checkpoint_path, self.global_step)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all logging systems."""
        # Weights & Biases
        if WANDB_AVAILABLE and self.config.wandb_project:
            wandb.log(metrics, step=self.global_step)
        
        # TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, self.global_step)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.world_size > 1:
            dist.destroy_process_group()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if WANDB_AVAILABLE and self.config.wandb_project:
            wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wave Field LLM Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainingConfig.from_yaml(args.config)
    
    # Check GPU availability
    if args.num_gpus > 1 and not torch.cuda.is_available():
        logger.error("CUDA not available but multiple GPUs requested")
        sys.exit(1)
    
    # Get world size from environment (for distributed training)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        config=config,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        local_rank=args.local_rank,
        world_size=world_size,
    )
    
    try:
        # Run training
        pipeline.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        pipeline.cleanup()


if __name__ == "__main__":
    main()

# Made with Bob

#!/usr/bin/env python3
"""
Complete CRUMB LLM Training Example

Demonstrates comprehensive training with all features:
- Data loading and preprocessing
- Model configuration
- Training loop with validation
- Checkpoint management
- Evaluation and metrics

Usage:
    python examples/train_model.py --data training_data.txt --steps 50000
"""

import argparse
import torch
from torch.utils.data import DataLoader
from crumb_llm import WaveFieldLM, WaveFieldConfig
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig
from crumb_llm.optimizer import OptimizerConfig
from crumb_llm.loss import LossConfig
from crumb_llm.data_enhanced import TextDataset, DataConfig, create_dataloader
from crumb_llm.tokenizer import ByteTokenizer
from crumb_llm.checkpoint import CheckpointManager

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CRUMB LLM')
    
    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--eval-data', type=str, default=None,
                       help='Path to evaluation data file')
    
    # Model
    parser.add_argument('--config', type=str, default='small',
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model configuration')
    
    # Training
    parser.add_argument('--steps', type=int, default=50000,
                       help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--grad-accum', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Optimization
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Use gradient checkpointing')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile()')
    
    # Checkpointing
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--eval-every', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    return parser.parse_args()

def load_data(args, tokenizer):
    """Load and prepare training data."""
    print("📚 Loading data...")
    
    # Load training data
    with open(args.data, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    print(f"   Training data: {len(train_text):,} characters")
    
    # Create training dataset
    train_dataset = TextDataset(
        train_text,
        tokenizer,
        block_size=512,
        random_crop=True,
    )
    
    # Create training dataloader
    data_config = DataConfig(
        batch_size=args.batch_size,
        dynamic_batching=True,
        max_tokens_per_batch=args.batch_size * 512,
        num_workers=4,
        pin_memory=True,
    )
    
    train_loader = create_dataloader(train_dataset, data_config, is_train=True)
    
    # Load evaluation data if provided
    eval_loader = None
    if args.eval_data:
        with open(args.eval_data, 'r', encoding='utf-8') as f:
            eval_text = f.read()
        
        print(f"   Evaluation data: {len(eval_text):,} characters")
        
        eval_dataset = TextDataset(
            eval_text,
            tokenizer,
            block_size=512,
            random_crop=False,
        )
        
        eval_loader = create_dataloader(eval_dataset, data_config, is_train=False)
    
    return train_loader, eval_loader

def create_model(args):
    """Create model from configuration."""
    print(f"🏗️  Creating {args.config} model...")
    
    # Get configuration
    if args.config == 'tiny':
        config = WaveFieldConfig(
            vocab_size=256, dim=256, n_layers=4, n_heads=4,
            field_size=512, block_size=512
        )
    elif args.config == 'small':
        config = WaveFieldConfig(
            vocab_size=256, dim=512, n_layers=8, n_heads=8,
            field_size=1024, block_size=512
        )
    elif args.config == 'medium':
        config = WaveFieldConfig(
            vocab_size=256, dim=1024, n_layers=16, n_heads=16,
            field_size=2048, block_size=1024
        )
    else:  # large
        config = WaveFieldConfig(
            vocab_size=256, dim=2048, n_layers=24, n_heads=24,
            field_size=4096, block_size=2048
        )
    
    # Create model
    model = WaveFieldLM(config)
    
    # Compile if requested
    if args.compile and hasattr(torch, 'compile'):
        print("   Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"   Dimension: {config.dim}")
    print(f"   Layers: {config.n_layers}")
    print(f"   Heads: {config.n_heads}")
    print(f"   Field size: {config.field_size}")
    
    return model, config

def main():
    """Main training function."""
    args = parse_args()
    
    print("🌊 CRUMB LLM Training")
    print("=" * 70)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Load data
    train_loader, eval_loader = load_data(args, tokenizer)
    
    # Create model
    model, model_config = create_model(args)
    
    # Configure training
    print("\n⚙️  Configuring training...")
    trainer_config = TrainerConfig(
        max_steps=args.steps,
        optimizer=OptimizerConfig(
            learning_rate=args.learning_rate,
            weight_decay=0.1,
            scheduler='cosine',
            warmup_steps=args.warmup_steps,
            max_steps=args.steps,
        ),
        loss=LossConfig(
            label_smoothing=0.1,
            spectral_diversity_weight=0.01,
            field_smoothness_weight=0.01,
        ),
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.output_dir,
    )
    
    print(f"   Steps: {args.steps:,}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Gradient accumulation: {args.grad_accum}")
    print(f"   Mixed precision: {args.mixed_precision}")
    print(f"   Gradient checkpointing: {args.gradient_checkpointing}")
    
    # Create trainer
    trainer = WaveFieldTrainer(
        model,
        train_loader,
        eval_loader,
        trainer_config,
    )
    
    # Resume if requested
    if args.resume:
        print(f"\n🔄 Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n🎯 Starting training...")
    print("=" * 70)
    
    metrics = trainer.train()
    
    # Print final results
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"Final loss: {metrics['final_loss']:.4f}")
    if 'best_eval_loss' in metrics:
        print(f"Best eval loss: {metrics['best_eval_loss']:.4f}")
    print(f"Total steps: {metrics['total_steps']:,}")
    print(f"Checkpoints saved to: {args.output_dir}")
    
    # Save final model
    final_path = f"{args.output_dir}/final_model.pt"
    torch.save({
        'model': model.state_dict(),
        'config': model_config,
        'tokenizer': tokenizer,
        'metrics': metrics,
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    print("\n🎉 Done!")
    print("\nNext steps:")
    print(f"  - Generate text: python examples/generate_text.py --model {final_path}")
    print(f"  - Start API server: python -m crumb_llm.serve --model {final_path}")
    print(f"  - Run benchmarks: python -m benchmarks.benchmark_suite --model {final_path}")

if __name__ == '__main__':
    main()

# Made with Bob

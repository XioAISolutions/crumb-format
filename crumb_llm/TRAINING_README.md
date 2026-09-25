# Wave Field LLM Training Infrastructure (Phase 2)

Complete training infrastructure for Wave Field LLM with production-ready features.

## Overview

This Phase 2 implementation provides comprehensive training capabilities including:

- ✅ **Optimizer Configuration**: Parameter grouping, multiple schedulers (cosine, linear, constant)
- ✅ **Loss Functions**: Cross-entropy with auxiliary losses (spectral diversity, field smoothness)
- ✅ **Metrics**: Perplexity, accuracy, bits-per-token, CRUMB-specific metrics
- ✅ **Checkpoint Management**: Save/load with rotation, best model tracking
- ✅ **Comprehensive Trainer**: Full training loop with all features
- ✅ **Data Loading**: TextDataset, CRUMBDataset, dynamic batching
- ✅ **Distributed Training**: DDP support for multi-GPU training
- ✅ **Mixed Precision**: AMP with BF16/FP16 support
- ✅ **Advanced Features**: Gradient accumulation, checkpointing, curriculum learning, early stopping

## Module Structure

```
crumb_llm/
├── optimizer.py          # Optimizer configuration and LR scheduling
├── loss.py              # Loss functions (CE + auxiliary)
├── metrics.py           # Evaluation metrics and logging
├── checkpoint.py        # Checkpoint management
├── trainer.py           # Comprehensive WaveFieldTrainer class
├── data_enhanced.py     # Enhanced data loading (TextDataset, CRUMBDataset)
└── train.py            # Simple training script (existing)
```

## Quick Start

### 1. Simple Training (Existing Interface)

Use the existing simple trainer for quick experiments:

```bash
python -m crumb_llm.train --config tiny --steps 1000
```

### 2. Comprehensive Training (New Interface)

Use the new comprehensive trainer with all Phase 2 features:

```python
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig
from crumb_llm.model import WaveFieldLM, WaveFieldConfig
from crumb_llm.optimizer import OptimizerConfig
from crumb_llm.loss import LossConfig

# Create model
model_config = WaveFieldConfig(vocab_size=256, dim=512, n_layers=8, n_heads=8)
model = WaveFieldLM(model_config)

# Configure training
trainer_config = TrainerConfig(
    max_steps=50000,
    optimizer=OptimizerConfig(learning_rate=3e-4, scheduler="cosine"),
    loss=LossConfig(spectral_diversity_weight=0.01),
    mixed_precision=True,
    checkpoint_dir="./checkpoints",
)

# Train
trainer = WaveFieldTrainer(model, train_loader, eval_loader, trainer_config)
metrics = trainer.train()
```

## Features

### Optimizer Configuration

**Parameter Grouping**: Automatically separates parameters into decay/no-decay groups:
- Weight decay applied to: weights (2D+ tensors)
- No weight decay for: biases, layer norms, embeddings

**Learning Rate Schedulers**:
- Cosine decay with warmup (default)
- Linear decay with warmup
- Constant with warmup

```python
from crumb_llm.optimizer import OptimizerConfig, configure_optimizer, get_lr_scheduler

config = OptimizerConfig(
    learning_rate=3e-4,
    weight_decay=0.1,
    scheduler="cosine",
    warmup_steps=1000,
    max_steps=50000,
)

optimizer = configure_optimizer(model, config)
scheduler = get_lr_scheduler(optimizer, config)
```

### Loss Functions

**Primary Loss**: Cross-entropy with label smoothing

**Auxiliary Losses**:
- **Spectral Diversity**: Encourages different heads to use different frequency ranges
- **Field Smoothness**: Penalizes high-frequency noise in field states

```python
from crumb_llm.loss import LossConfig, WaveFieldLoss

config = LossConfig(
    label_smoothing=0.1,
    spectral_diversity_weight=0.01,
    field_smoothness_weight=0.01,
)

loss_fn = WaveFieldLoss(config)
losses = loss_fn(logits, targets, field_states=fields, head_spectra=spectra)
```

### Metrics

**Standard Metrics**:
- Loss (cross-entropy)
- Perplexity
- Token accuracy
- Bits per character/token

**CRUMB-Specific Metrics**:
- Section boundary detection accuracy
- Cross-reference resolution accuracy

```python
from crumb_llm.metrics import MetricsAccumulator

acc = MetricsAccumulator()
acc.update(loss, logits, targets)
metrics = acc.compute()  # Returns dict with all metrics
```

### Checkpoint Management

**Features**:
- Automatic checkpoint rotation (keep best N + last N)
- Best model tracking by any metric
- Resume training from checkpoint
- Metadata storage (step, epoch, metrics)

```python
from crumb_llm.checkpoint import CheckpointManager, CheckpointMetadata

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    keep_best_n=3,
    keep_last_n=2,
    metric_name="eval_loss",
    metric_mode="min",
)

# Save checkpoint
metadata = CheckpointMetadata(step=1000, epoch=1, loss=2.5, learning_rate=3e-4)
manager.save_checkpoint(model, optimizer, metadata, is_best=True)

# Load checkpoint
metadata = manager.load_checkpoint("./checkpoints/best.pt", model, optimizer)
```

### Comprehensive Trainer

**Features**:
- Full training loop with validation
- Distributed training (DDP)
- Mixed precision (AMP)
- Gradient accumulation
- Gradient checkpointing
- Curriculum learning
- Early stopping
- Comprehensive logging

```python
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig

config = TrainerConfig(
    max_steps=50000,
    gradient_accumulation_steps=4,
    mixed_precision=True,
    gradient_checkpointing=True,
    curriculum_learning=True,
    early_stopping=True,
    eval_every=500,
    save_every=1000,
)

trainer = WaveFieldTrainer(model, train_loader, eval_loader, config)
metrics = trainer.train()
```

### Data Loading

**TextDataset**: For plain text with sliding window or random cropping

**CRUMBDataset**: For CRUMB-structured documents with structure-aware sampling

**Dynamic Batching**: Groups sequences by length for efficiency

```python
from crumb_llm.data_enhanced import TextDataset, CRUMBDataset, create_dataloader, DataConfig

# Text dataset
dataset = TextDataset(text, tokenizer, block_size=512, random_crop=True)

# CRUMB dataset
dataset = CRUMBDataset(crumb_files, tokenizer, block_size=512, preserve_structure=True)

# Create dataloader with dynamic batching
config = DataConfig(
    batch_size=32,
    dynamic_batching=True,
    max_tokens_per_batch=16384,
)
loader = create_dataloader(dataset, config, is_train=True)
```

## Training Strategies

### Single GPU Training

```bash
python -m crumb_llm.train \
  --config small \
  --data /path/to/data \
  --steps 50000 \
  --batch-size 32 \
  --grad-accum 4
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 -m crumb_llm.train \
  --config medium \
  --data /path/to/data \
  --steps 100000 \
  --batch-size 16
```

### Mixed Precision Training

Automatically enabled on CUDA with BF16 support:

```python
config = TrainerConfig(
    mixed_precision=True,
    amp_dtype="bfloat16",  # or "float16"
)
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
config = TrainerConfig(
    batch_size=8,
    gradient_accumulation_steps=8,  # effective batch = 64
)
```

### Curriculum Learning

Progressive sequence length:

```python
config = TrainerConfig(
    curriculum_learning=True,
    curriculum_start_length=128,
    curriculum_end_length=512,
    curriculum_steps=10000,
)
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_training_infrastructure.py -v
```

Tests cover:
- Optimizer configuration and parameter grouping
- Loss computation (primary + auxiliary)
- Metrics accumulation and computation
- Checkpoint save/load
- Trainer initialization

## Performance

Expected throughput on A100 (40GB):

| Model | Batch Size | Tokens/sec | Memory |
|-------|------------|------------|--------|
| Tiny (230K) | 128 | ~50K | 2 GB |
| Small (25M) | 32 | ~15K | 8 GB |
| Medium (350M) | 16 | ~5K | 24 GB |
| Large (1.5B) | 4 | ~1.5K | 38 GB |

## API Reference

### Optimizer

```python
from crumb_llm.optimizer import (
    OptimizerConfig,
    configure_optimizer,
    get_lr_scheduler,
    clip_gradients,
)
```

### Loss

```python
from crumb_llm.loss import (
    LossConfig,
    WaveFieldLoss,
    compute_perplexity,
    compute_bits_per_token,
)
```

### Metrics

```python
from crumb_llm.metrics import (
    MetricsAccumulator,
    MetricsLogger,
    compute_token_accuracy,
    compute_top_k_accuracy,
)
```

### Checkpoint

```python
from crumb_llm.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    save_checkpoint,
    load_checkpoint,
    resume_training,
)
```

### Trainer

```python
from crumb_llm.trainer import (
    WaveFieldTrainer,
    TrainerConfig,
)
```

### Data

```python
from crumb_llm.data_enhanced import (
    TextDataset,
    CRUMBDataset,
    DataConfig,
    create_dataloader,
    load_dataset,
)
```

## Examples

See `docs/TRAINING_GUIDE.md` for comprehensive examples and tutorials.

## Troubleshooting

### NaN/Inf Loss
- Reduce learning rate
- Increase warmup steps
- Check gradient clipping (default: 1.0)

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Enable mixed precision

### Slow Training
- Increase batch size
- Enable mixed precision
- Use more data workers
- Check GPU utilization

## Contributing

When adding new features:

1. Add tests to `tests/test_training_infrastructure.py`
2. Update documentation in `docs/TRAINING_GUIDE.md`
3. Follow existing code style and type hints
4. Ensure backward compatibility with existing `train.py`

## License

Same as parent project (see LICENSE file).

---

**Phase**: 2 (Training Infrastructure)  
**Status**: Complete  
**Last Updated**: 2026-05-28
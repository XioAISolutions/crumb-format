# Wave Field LLM Training Guide

Complete guide for training Wave Field LLM models using the Phase 2 training infrastructure.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Configuration](#training-configuration)
3. [Data Preparation](#data-preparation)
4. [Training Commands](#training-commands)
5. [Distributed Training](#distributed-training)
6. [Advanced Features](#advanced-features)
7. [Monitoring and Debugging](#monitoring-and-debugging)

## Quick Start

### Basic Training

Train a tiny model on synthetic data (for testing):

```bash
python -m crumb_llm.train \
  --config tiny \
  --steps 1000 \
  --out ./checkpoints/tiny_run
```

### Training on Custom Data

Train on your own text data:

```bash
python -m crumb_llm.train \
  --config small \
  --data /path/to/your/data.txt \
  --steps 50000 \
  --out ./checkpoints/small_run \
  --batch-size 32 \
  --grad-accum 4
```

### Training on CRUMB Documents

Train on CRUMB-structured documents:

```bash
python -m crumb_llm.train \
  --config small \
  --data /path/to/crumb/directory \
  --steps 50000 \
  --out ./checkpoints/crumb_run
```

## Training Configuration

### Built-in Configs

The following configs are available in `crumb_llm/configs/`:

- **tiny.json**: 230K params, for quick testing (CPU-friendly)
- **small.json**: 25M params, for single GPU training
- **medium.json**: 350M params, for multi-GPU training
- **large.json**: 1.5B params, for distributed training

### Custom Configuration

Create a custom config JSON file:

```json
{
  "arch": "wave_field",
  "vocab_size": 256,
  "dim": 512,
  "n_layers": 12,
  "n_heads": 8,
  "field_size": 1024,
  "block_size": 512,
  "ffn_mult": 2.67,
  "dropout": 0.1,
  "tie_embeddings": true,
  
  "tokenizer": "byte",
  "batch_size": 32,
  "lr": 3e-4,
  "weight_decay": 0.1,
  "warmup": 1000,
  
  "mixed_precision": true,
  "gradient_checkpointing": false
}
```

Use it with:

```bash
python -m crumb_llm.train --config /path/to/config.json --steps 100000
```

## Data Preparation

### Text Files

Any UTF-8 text file works:

```bash
python -m crumb_llm.train --data corpus.txt --steps 50000
```

### Directory of Files

Point to a directory containing `.txt`, `.md`, or `.crumb` files:

```bash
python -m crumb_llm.train --data /path/to/corpus/ --steps 50000
```

### CRUMB Documents

For CRUMB-structured documents, the trainer automatically:
- Preserves section boundaries
- Respects priority annotations
- Handles fold structures

```bash
python -m crumb_llm.train \
  --data /path/to/crumb/docs/ \
  --config small \
  --steps 50000
```

## Training Commands

### Using the New Trainer (Recommended)

The new comprehensive trainer with all Phase 2 features:

```python
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig
from crumb_llm.model import WaveFieldLM, WaveFieldConfig
from crumb_llm.data_enhanced import load_dataset, create_dataloader, DataConfig
from crumb_llm.optimizer import OptimizerConfig
from crumb_llm.loss import LossConfig

# Model config
model_config = WaveFieldConfig(
    vocab_size=256,
    dim=512,
    n_layers=8,
    n_heads=8,
    field_size=1024,
)
model = WaveFieldLM(model_config)

# Data config
data_config = DataConfig(
    data_path="/path/to/data",
    block_size=512,
    batch_size=32,
)
train_dataset = load_dataset(data_config, split="train")
eval_dataset = load_dataset(data_config, split="eval")
train_loader = create_dataloader(train_dataset, data_config, is_train=True)
eval_loader = create_dataloader(eval_dataset, data_config, is_train=False)

# Trainer config
trainer_config = TrainerConfig(
    max_steps=50000,
    gradient_accumulation_steps=4,
    optimizer=OptimizerConfig(
        learning_rate=3e-4,
        weight_decay=0.1,
        scheduler="cosine",
        warmup_steps=1000,
    ),
    loss=LossConfig(
        label_smoothing=0.1,
        spectral_diversity_weight=0.01,
        field_smoothness_weight=0.01,
    ),
    mixed_precision=True,
    eval_every=500,
    save_every=1000,
    checkpoint_dir="./checkpoints",
)

# Train
trainer = WaveFieldTrainer(model, train_loader, eval_loader, trainer_config)
metrics = trainer.train()
```

### Using the Simple Trainer

For quick experiments, use the existing simple trainer:

```bash
python -m crumb_llm.train \
  --config tiny \
  --steps 1000 \
  --log-every 50 \
  --eval-every 200
```

## Distributed Training

### Single Node, Multiple GPUs

Using PyTorch DDP:

```bash
torchrun --nproc_per_node=4 -m crumb_llm.train \
  --config medium \
  --data /path/to/data \
  --steps 100000 \
  --batch-size 16 \
  --grad-accum 4
```

### Multiple Nodes

On each node:

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m crumb_llm.train \
  --config large \
  --data /path/to/data \
  --steps 500000
```

## Advanced Features

### Mixed Precision Training

Automatically enabled on CUDA with BF16 support:

```python
trainer_config = TrainerConfig(
    mixed_precision=True,
    amp_dtype="bfloat16",  # or "float16"
)
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
trainer_config = TrainerConfig(
    batch_size=8,
    gradient_accumulation_steps=8,  # effective batch size = 64
)
```

### Gradient Checkpointing

Save memory for large models:

```python
trainer_config = TrainerConfig(
    gradient_checkpointing=True,
)
```

### Curriculum Learning

Progressive sequence length training:

```python
trainer_config = TrainerConfig(
    curriculum_learning=True,
    curriculum_start_length=128,
    curriculum_end_length=512,
    curriculum_steps=10000,
)
```

### Early Stopping

Stop training when validation loss plateaus:

```python
trainer_config = TrainerConfig(
    early_stopping=True,
    early_stopping_patience=5,
    early_stopping_metric="eval_loss",
    early_stopping_mode="min",
)
```

### Auxiliary Losses

Enable spectral diversity and field smoothness losses:

```python
loss_config = LossConfig(
    spectral_diversity_weight=0.01,  # Encourage different frequencies per head
    field_smoothness_weight=0.01,    # Penalize high-frequency noise
)
```

### Dynamic Batching

Group sequences by length for efficiency:

```python
data_config = DataConfig(
    dynamic_batching=True,
    max_tokens_per_batch=16384,
)
```

## Monitoring and Debugging

### Training Logs

The trainer logs:
- Loss (cross-entropy, auxiliary)
- Learning rate
- Gradient norm
- Tokens per second
- Perplexity
- Accuracy

Example output:
```
[train] step=100 loss=3.2456 perplexity=25.67 accuracy=0.3421 lr=3.00e-04 grad_norm=1.23
[eval] step=500 loss=2.8934 perplexity=18.09 accuracy=0.4123
```

### Checkpoints

Checkpoints are saved to `checkpoint_dir`:
- `best.pt`: Best model by validation loss
- `checkpoint_step_N.pt`: Regular checkpoints
- `checkpoint_index.json`: Checkpoint metadata

### Resume Training

Resume from the latest checkpoint:

```python
trainer = WaveFieldTrainer(model, train_loader, eval_loader, config)
trainer.resume_from_checkpoint("./checkpoints/checkpoint_step_10000.pt")
metrics = trainer.train()
```

### TensorBoard (Future)

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./checkpoints/logs
```

## Performance Tips

### GPU Utilization

1. **Increase batch size**: Use gradient accumulation if OOM
2. **Enable mixed precision**: 2× speedup on modern GPUs
3. **Use gradient checkpointing**: Trade compute for memory
4. **Optimize data loading**: Use `num_workers > 0` and `pin_memory=True`

### Training Speed

Expected throughput on A100 (40GB):

| Model Size | Batch Size | Tokens/sec | Memory |
|------------|------------|------------|--------|
| Tiny (230K) | 128 | ~50K | 2 GB |
| Small (25M) | 32 | ~15K | 8 GB |
| Medium (350M) | 16 | ~5K | 24 GB |
| Large (1.5B) | 4 | ~1.5K | 38 GB |

### Memory Optimization

If you run out of memory:

1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Reduce sequence length
5. Use mixed precision (BF16/FP16)

## Troubleshooting

### NaN/Inf Loss

If training diverges:

1. Reduce learning rate
2. Increase warmup steps
3. Enable gradient clipping (default: 1.0)
4. Check data for corrupted samples
5. Reduce mixed precision to FP32

### Slow Training

If training is slower than expected:

1. Check GPU utilization (`nvidia-smi`)
2. Increase batch size
3. Enable mixed precision
4. Use more data workers
5. Profile with PyTorch profiler

### Out of Memory

If you get OOM errors:

1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Reduce model size
5. Use distributed training

## Example Training Scripts

### Train Small Model

```bash
#!/bin/bash
python -m crumb_llm.train \
  --config small \
  --data ./data/corpus.txt \
  --steps 50000 \
  --batch-size 32 \
  --grad-accum 4 \
  --out ./checkpoints/small_run \
  --log-every 50 \
  --eval-every 500 \
  --seed 42
```

### Train Medium Model (Multi-GPU)

```bash
#!/bin/bash
torchrun --nproc_per_node=4 -m crumb_llm.train \
  --config medium \
  --data ./data/corpus/ \
  --steps 100000 \
  --batch-size 16 \
  --grad-accum 8 \
  --out ./checkpoints/medium_run \
  --log-every 20 \
  --eval-every 1000
```

### Fine-tune on CRUMB Documents

```bash
#!/bin/bash
python -m crumb_llm.train \
  --config small \
  --data ./crumb_docs/ \
  --steps 10000 \
  --batch-size 16 \
  --out ./checkpoints/crumb_finetune \
  --resume ./checkpoints/pretrained/best.pt
```

## Next Steps

After training:

1. **Evaluate**: Use `crumb_llm.bench` to benchmark your model
2. **Generate**: Use `crumb_llm.sample` to generate text
3. **Serve**: Use `crumb_llm.serve` to deploy as an API
4. **Export**: Use `crumb_llm.hub` to push to HuggingFace Hub

## References

- [Wave Field LLM Specification](WAVE_FIELD_LLM_SPEC.md)
- [Model Architecture](crumb-llm-architecture.md)
- [API Documentation](../crumb_llm/)

---

**Last Updated**: 2026-05-28  
**Version**: Phase 2 Training Infrastructure
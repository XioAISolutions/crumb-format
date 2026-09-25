## Model Zoo Training Scripts

Ready-to-use scripts for training and evaluating Wave Field LLM models.

### Training Scripts

#### Universal Training Script
```bash
./train_model.sh <config_name> [num_gpus] [output_dir]
```

Examples:
```bash
# Train tiny model on 2 GPUs
./train_model.sh tiny 2

# Train small model on 8 GPUs
./train_model.sh small 8 models/my-small-model

# Train medium model on 16 GPUs
./train_model.sh medium 16

# Train large model on 32 GPUs
./train_model.sh large 32

# Train specialized models
./train_model.sh small-code 8
./train_model.sh medium-crumb 16
./train_model.sh large-instruct 32
```

### Evaluation Script
```bash
./evaluate_model.sh <model_path> <output_file> [benchmarks...]
```

Examples:
```bash
# Evaluate on default benchmarks
./evaluate_model.sh models/wavefield-small results/eval.json

# Evaluate on specific benchmarks
./evaluate_model.sh models/wavefield-small results/eval.json mmlu hellaswag humaneval

# Full evaluation
./evaluate_model.sh models/wavefield-large results/eval.json mmlu hellaswag arc_easy arc_challenge winogrande truthfulqa humaneval mbpp
```

### Quick Start

1. **Train a model:**
   ```bash
   chmod +x model_zoo/scripts/*.sh
   ./model_zoo/scripts/train_model.sh small 8
   ```

2. **Evaluate the model:**
   ```bash
   ./model_zoo/scripts/evaluate_model.sh models/small results/eval.json
   ```

3. **Generate model card:**
   ```bash
   python -m model_zoo.model_cards.generator \
     --config model_zoo/configs/small.yaml \
     --training-log logs/wavefield-small/training.json \
     --eval-results results/eval.json \
     --output model_zoo/model_cards/wavefield-small.md
   ```

4. **Register model:**
   ```bash
   python -c "
   from model_zoo.registry import ModelRegistry
   registry = ModelRegistry()
   registry.register(
       name='wavefield-small',
       version='1.0.0',
       checkpoint_path='models/small/checkpoint-best.pt',
       config_path='model_zoo/configs/small.yaml',
       eval_results_path='results/eval.json'
   )
   "
   ```

### Configuration

All training configurations are in `model_zoo/configs/`:
- `tiny.yaml` - 230K parameters
- `small.yaml` - 25M parameters
- `medium.yaml` - 350M parameters
- `large.yaml` - 1.5B parameters
- `small-code.yaml` - Code-specialized
- `medium-crumb.yaml` - CRUMB-specialized
- `large-instruct.yaml` - Instruction-tuned

### Requirements

- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory (tiny/small)
- 24GB+ GPU memory (medium)
- 80GB+ GPU memory (large)

### Distributed Training

The scripts automatically use distributed training when multiple GPUs are specified:
- Uses `torchrun` for multi-GPU training
- Supports DDP and FSDP
- Automatic gradient accumulation
- Mixed precision training (fp16/bf16)

### Monitoring

Training progress is logged to:
- Console output
- TensorBoard (logs directory)
- Weights & Biases (if configured)

View TensorBoard:
```bash
tensorboard --logdir logs/
```

### Troubleshooting

**Out of memory:**
- Reduce batch size in config
- Increase gradient accumulation
- Enable gradient checkpointing
- Use FSDP for large models

**Slow training:**
- Check data loading (increase num_workers)
- Enable mixed precision
- Use flash attention
- Optimize data pipeline

**Poor convergence:**
- Adjust learning rate
- Increase warmup steps
- Check data quality
- Review loss curves
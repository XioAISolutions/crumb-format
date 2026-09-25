# CRUMB LLM API Reference

Complete API documentation for CRUMB LLM. This reference covers all public classes, functions, and modules.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Model Architecture](#model-architecture)
3. [Training](#training)
4. [Inference](#inference)
5. [Data Loading](#data-loading)
6. [Utilities](#utilities)
7. [CLI Commands](#cli-commands)

## Core Modules

### `crumb_llm.kernels`

Wave kernel construction and FFT-based convolution.

#### `wave_kernel_time(alpha, omega, phi, size)`

Build wave kernel in time domain.

**Parameters:**
- `alpha` (float or Tensor): Damping coefficient (α > 0)
- `omega` (float or Tensor): Angular frequency
- `phi` (float or Tensor): Phase shift
- `size` (int): Kernel size (field size)

**Returns:**
- `Tensor`: Wave kernel of shape `[size]`

**Example:**
```python
import torch
from crumb_llm import wave_kernel_time

kernel = wave_kernel_time(alpha=0.1, omega=2.0, phi=0.0, size=1024)
# kernel.shape: [1024]
```

#### `wave_kernel_freq(alpha, omega, phi, size)`

Build wave kernel in frequency domain (more efficient).

**Parameters:**
- Same as `wave_kernel_time`

**Returns:**
- `Tensor`: Complex frequency-domain kernel of shape `[size//2 + 1]`

**Example:**
```python
kernel_freq = wave_kernel_freq(alpha=0.1, omega=2.0, phi=0.0, size=1024)
# kernel_freq.shape: [513] (complex)
```

#### `fft_convolve(x, kernel_freq)`

Perform FFT-based convolution.

**Parameters:**
- `x` (Tensor): Input tensor `[..., N]`
- `kernel_freq` (Tensor): Frequency-domain kernel `[N//2 + 1]`

**Returns:**
- `Tensor`: Convolved output, same shape as `x`

**Example:**
```python
from crumb_llm import fft_convolve

field = torch.randn(4, 1024, 512)  # [B, F, D]
kernel = wave_kernel_freq(0.1, 2.0, 0.0, 1024)
output = fft_convolve(field, kernel)
# output.shape: [4, 1024, 512]
```

### `crumb_llm.scatter_gather`

Scatter/gather operations for token-field conversion.

#### `scatter_linear(tokens, positions, field_size, weights=None)`

Scatter tokens onto continuous field using linear interpolation.

**Parameters:**
- `tokens` (Tensor): Token embeddings `[B, N, D]`
- `positions` (Tensor): Token positions `[B, N]` (0 to N-1)
- `field_size` (int): Size of output field
- `weights` (Tensor, optional): Scatter weights `[B, N]`

**Returns:**
- `Tensor`: Field state `[B, field_size, D]`

**Example:**
```python
from crumb_llm import scatter_linear

tokens = torch.randn(4, 128, 512)  # [B, N, D]
positions = torch.arange(128).expand(4, -1)
field = scatter_linear(tokens, positions, field_size=256)
# field.shape: [4, 256, 512]
```

#### `gather_linear(field, positions)`

Gather token states from field using linear interpolation.

**Parameters:**
- `field` (Tensor): Field state `[B, F, D]`
- `positions` (Tensor): Token positions `[B, N]`

**Returns:**
- `Tensor`: Token embeddings `[B, N, D]`

**Example:**
```python
from crumb_llm import gather_linear

field = torch.randn(4, 256, 512)  # [B, F, D]
positions = torch.arange(128).expand(4, -1)
tokens = gather_linear(field, positions)
# tokens.shape: [4, 128, 512]
```

## Model Architecture

### `crumb_llm.WaveFieldConfig`

Configuration for CRUMB LLM model.

**Attributes:**
- `vocab_size` (int): Vocabulary size
- `dim` (int): Model dimension
- `n_layers` (int): Number of layers
- `n_heads` (int): Number of attention heads
- `field_size` (int): Wave field size (power of 2)
- `block_size` (int): Maximum sequence length
- `dropout` (float): Dropout probability
- `bias` (bool): Use bias in linear layers
- `norm_eps` (float): Layer norm epsilon

**Example:**
```python
from crumb_llm import WaveFieldConfig

config = WaveFieldConfig(
    vocab_size=256,
    dim=512,
    n_layers=8,
    n_heads=8,
    field_size=1024,
    block_size=512,
    dropout=0.1,
)
```

**Presets:**
```python
# Tiny (230K params)
config = WaveFieldConfig.tiny()

# Small (25M params)
config = WaveFieldConfig.small()

# Medium (350M params)
config = WaveFieldConfig.medium()

# Large (1.5B params)
config = WaveFieldConfig.large()
```

### `crumb_llm.WaveFieldLM`

Main CRUMB LLM model.

**Constructor:**
```python
WaveFieldLM(config: WaveFieldConfig)
```

**Methods:**

#### `forward(idx, targets=None)`

Forward pass through the model.

**Parameters:**
- `idx` (Tensor): Input token indices `[B, N]`
- `targets` (Tensor, optional): Target indices for loss `[B, N]`

**Returns:**
- `dict`: Dictionary containing:
  - `logits` (Tensor): Output logits `[B, N, vocab_size]`
  - `loss` (Tensor, optional): Cross-entropy loss if targets provided
  - `field_states` (list, optional): Field states per layer

**Example:**
```python
from crumb_llm import WaveFieldLM, WaveFieldConfig

config = WaveFieldConfig.small()
model = WaveFieldLM(config)

# Forward pass
idx = torch.randint(0, 256, (4, 128))
outputs = model(idx)
logits = outputs['logits']  # [4, 128, 256]

# With loss
targets = torch.randint(0, 256, (4, 128))
outputs = model(idx, targets=targets)
loss = outputs['loss']
```

#### `generate(idx, max_new_tokens, temperature=1.0, top_k=None)`

Generate new tokens autoregressively.

**Parameters:**
- `idx` (Tensor): Context tokens `[B, N]`
- `max_new_tokens` (int): Number of tokens to generate
- `temperature` (float): Sampling temperature
- `top_k` (int, optional): Top-k sampling

**Returns:**
- `Tensor`: Generated tokens `[B, N + max_new_tokens]`

**Example:**
```python
# Generate 100 tokens
context = torch.randint(0, 256, (1, 10))
generated = model.generate(context, max_new_tokens=100, temperature=0.8)
```

### `crumb_llm.WaveFieldBlock`

Single wave field transformer block.

**Constructor:**
```python
WaveFieldBlock(config: WaveFieldConfig)
```

**Methods:**

#### `forward(x)`

**Parameters:**
- `x` (Tensor): Input tokens `[B, N, D]`

**Returns:**
- `Tensor`: Output tokens `[B, N, D]`

### `crumb_llm.WaveFieldHead`

Single wave field attention head.

**Constructor:**
```python
WaveFieldHead(dim, field_size, head_dim)
```

**Parameters:**
- `dim` (int): Model dimension
- `field_size` (int): Wave field size
- `head_dim` (int): Dimension per head

## Training

### `crumb_llm.trainer.WaveFieldTrainer`

Comprehensive training loop with all features.

**Constructor:**
```python
WaveFieldTrainer(
    model,
    train_loader,
    eval_loader=None,
    config=None,
)
```

**Parameters:**
- `model` (WaveFieldLM): Model to train
- `train_loader` (DataLoader): Training data
- `eval_loader` (DataLoader, optional): Evaluation data
- `config` (TrainerConfig, optional): Training configuration

**Methods:**

#### `train()`

Run full training loop.

**Returns:**
- `dict`: Training metrics

**Example:**
```python
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig
from crumb_llm import WaveFieldLM, WaveFieldConfig

# Create model
model_config = WaveFieldConfig.small()
model = WaveFieldLM(model_config)

# Configure training
trainer_config = TrainerConfig(
    max_steps=50000,
    batch_size=32,
    learning_rate=3e-4,
    eval_every=500,
    save_every=1000,
)

# Train
trainer = WaveFieldTrainer(model, train_loader, eval_loader, trainer_config)
metrics = trainer.train()
```

### `crumb_llm.trainer.TrainerConfig`

Training configuration.

**Attributes:**
- `max_steps` (int): Maximum training steps
- `batch_size` (int): Batch size
- `learning_rate` (float): Learning rate
- `weight_decay` (float): Weight decay
- `gradient_accumulation_steps` (int): Gradient accumulation
- `mixed_precision` (bool): Use mixed precision
- `gradient_checkpointing` (bool): Use gradient checkpointing
- `eval_every` (int): Evaluate every N steps
- `save_every` (int): Save checkpoint every N steps
- `checkpoint_dir` (str): Checkpoint directory

### `crumb_llm.optimizer.configure_optimizer`

Configure optimizer with parameter grouping.

**Parameters:**
- `model` (nn.Module): Model to optimize
- `config` (OptimizerConfig): Optimizer configuration

**Returns:**
- `torch.optim.Optimizer`: Configured optimizer

**Example:**
```python
from crumb_llm.optimizer import configure_optimizer, OptimizerConfig

config = OptimizerConfig(
    learning_rate=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
)

optimizer = configure_optimizer(model, config)
```

### `crumb_llm.loss.WaveFieldLoss`

Loss function with auxiliary losses.

**Constructor:**
```python
WaveFieldLoss(config: LossConfig)
```

**Methods:**

#### `forward(logits, targets, field_states=None, head_spectra=None)`

**Parameters:**
- `logits` (Tensor): Model outputs `[B, N, V]`
- `targets` (Tensor): Target indices `[B, N]`
- `field_states` (list, optional): Field states for smoothness loss
- `head_spectra` (list, optional): Head spectra for diversity loss

**Returns:**
- `dict`: Dictionary with `loss`, `ce_loss`, `aux_losses`

## Inference

### `crumb_llm.generate.Generator`

Text generation with multiple sampling strategies.

**Constructor:**
```python
Generator(model, tokenizer, device='cuda')
```

**Methods:**

#### `generate(prompt, config)`

Generate text from prompt.

**Parameters:**
- `prompt` (str): Input prompt
- `config` (GenerationConfig): Generation configuration

**Returns:**
- `str`: Generated text

**Example:**
```python
from crumb_llm.generate import Generator, GenerationConfig
from crumb_llm.sample import load_checkpoint

model, tokenizer = load_checkpoint('model.pt')
generator = Generator(model, tokenizer)

config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
)

text = generator.generate("BEGIN CRUMB", config)
print(text)
```

### `crumb_llm.generate.GenerationConfig`

Generation configuration.

**Attributes:**
- `max_new_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature (0.0 = greedy)
- `top_k` (int, optional): Top-k sampling
- `top_p` (float, optional): Nucleus sampling
- `repetition_penalty` (float): Repetition penalty
- `stop_tokens` (list): Stop token IDs
- `use_cache` (bool): Use field-state caching

### `crumb_llm.inference.InferenceEngine`

Optimized inference engine.

**Constructor:**
```python
InferenceEngine(model, tokenizer, config=None)
```

**Class Methods:**

#### `from_checkpoint(path, config=None)`

Load model from checkpoint.

**Parameters:**
- `path` (str): Checkpoint path
- `config` (InferenceConfig, optional): Inference configuration

**Returns:**
- `InferenceEngine`: Configured engine

**Example:**
```python
from crumb_llm.inference import InferenceEngine, InferenceConfig

config = InferenceConfig(
    quantization='fp16',
    use_compile=True,
    compile_mode='reduce-overhead',
)

engine = InferenceEngine.from_checkpoint('model.pt', config)
text = engine.generate("Hello", max_tokens=100)
```

### `crumb_llm.cache.FieldStateCache`

Field-state caching for efficient generation.

**Constructor:**
```python
FieldStateCache(n_layers, field_size, dim, batch_size=1)
```

**Class Methods:**

#### `init(model, batch_size=1)`

Initialize cache from model.

**Example:**
```python
from crumb_llm.cache import FieldStateCache, generate_cached

cache = FieldStateCache.init(model, batch_size=1)

# Generate with caching (10-100× faster)
output_ids = generate_cached(
    model,
    input_ids,
    max_new_tokens=100,
    cache=cache,
)
```

### `crumb_llm.quantization.quantize_model`

Quantize model for faster inference.

**Parameters:**
- `model` (nn.Module): Model to quantize
- `config` (QuantizationConfig): Quantization configuration
- `calibration_data` (DataLoader, optional): For static quantization

**Returns:**
- `nn.Module`: Quantized model

**Example:**
```python
from crumb_llm.quantization import quantize_model, QuantizationConfig

# Dynamic INT8
config = QuantizationConfig(mode='dynamic_int8')
quantized = quantize_model(model, config)

# FP16
config = QuantizationConfig(mode='fp16')
quantized = quantize_model(model, config)
```

### `crumb_llm.streaming.StreamingGenerator`

Token-by-token streaming generation.

**Constructor:**
```python
StreamingGenerator(model, tokenizer, device='cuda')
```

**Methods:**

#### `stream(prompt, gen_config, stream_config)`

Stream tokens as they're generated.

**Parameters:**
- `prompt` (str): Input prompt
- `gen_config` (GenerationConfig): Generation config
- `stream_config` (StreamingConfig): Streaming config

**Yields:**
- `StreamToken`: Token with text, logprobs, metadata

**Example:**
```python
from crumb_llm.streaming import StreamingGenerator, StreamingConfig

generator = StreamingGenerator(model, tokenizer)
stream_config = StreamingConfig(buffer_size=1)

for token in generator.stream(prompt, gen_config, stream_config):
    print(token.text, end='', flush=True)
```

## Data Loading

### `crumb_llm.data_enhanced.TextDataset`

Dataset for plain text.

**Constructor:**
```python
TextDataset(text, tokenizer, block_size, random_crop=False)
```

**Parameters:**
- `text` (str): Training text
- `tokenizer`: Tokenizer instance
- `block_size` (int): Sequence length
- `random_crop` (bool): Random vs sliding window

**Example:**
```python
from crumb_llm.data_enhanced import TextDataset
from crumb_llm.tokenizer import ByteTokenizer

tokenizer = ByteTokenizer()
with open('data.txt') as f:
    text = f.read()

dataset = TextDataset(text, tokenizer, block_size=512, random_crop=True)
```

### `crumb_llm.data_enhanced.CRUMBDataset`

Dataset for CRUMB-structured documents.

**Constructor:**
```python
CRUMBDataset(
    crumb_files,
    tokenizer,
    block_size,
    preserve_structure=True,
)
```

**Parameters:**
- `crumb_files` (list): List of CRUMB file paths
- `tokenizer`: Tokenizer instance
- `block_size` (int): Sequence length
- `preserve_structure` (bool): Preserve section boundaries

**Example:**
```python
from crumb_llm.data_enhanced import CRUMBDataset
import glob

crumb_files = glob.glob('examples/*.crumb')
dataset = CRUMBDataset(
    crumb_files,
    tokenizer,
    block_size=512,
    preserve_structure=True,
)
```

### `crumb_llm.data_enhanced.create_dataloader`

Create dataloader with dynamic batching.

**Parameters:**
- `dataset`: Dataset instance
- `config` (DataConfig): Data configuration
- `is_train` (bool): Training mode

**Returns:**
- `DataLoader`: Configured dataloader

**Example:**
```python
from crumb_llm.data_enhanced import create_dataloader, DataConfig

config = DataConfig(
    batch_size=32,
    dynamic_batching=True,
    max_tokens_per_batch=16384,
    num_workers=4,
)

train_loader = create_dataloader(train_dataset, config, is_train=True)
```

## Utilities

### `crumb_llm.checkpoint.CheckpointManager`

Manage model checkpoints.

**Constructor:**
```python
CheckpointManager(
    checkpoint_dir,
    keep_best_n=3,
    keep_last_n=2,
    metric_name='eval_loss',
    metric_mode='min',
)
```

**Methods:**

#### `save_checkpoint(model, optimizer, metadata, is_best=False)`

Save checkpoint.

**Parameters:**
- `model` (nn.Module): Model to save
- `optimizer` (Optimizer): Optimizer state
- `metadata` (CheckpointMetadata): Checkpoint metadata
- `is_best` (bool): Mark as best checkpoint

#### `load_checkpoint(path, model, optimizer=None)`

Load checkpoint.

**Parameters:**
- `path` (str): Checkpoint path
- `model` (nn.Module): Model to load into
- `optimizer` (Optimizer, optional): Optimizer to load into

**Returns:**
- `CheckpointMetadata`: Checkpoint metadata

**Example:**
```python
from crumb_llm.checkpoint import CheckpointManager, CheckpointMetadata

manager = CheckpointManager('./checkpoints')

# Save
metadata = CheckpointMetadata(step=1000, epoch=1, loss=2.5)
manager.save_checkpoint(model, optimizer, metadata, is_best=True)

# Load
metadata = manager.load_checkpoint('./checkpoints/best.pt', model, optimizer)
```

### `crumb_llm.sample.load_checkpoint`

Load model and tokenizer from checkpoint.

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint

**Returns:**
- `tuple`: (model, tokenizer)

**Example:**
```python
from crumb_llm.sample import load_checkpoint

model, tokenizer = load_checkpoint('checkpoints/model.pt')
```

### `crumb_llm.hub.save_for_hub`

Save model for HuggingFace Hub.

**Parameters:**
- `model` (WaveFieldLM): Model to save
- `tokenizer`: Tokenizer
- `save_dir` (str): Output directory
- `model_name` (str): Model name
- `metadata` (dict, optional): Additional metadata

**Example:**
```python
from crumb_llm.hub import save_for_hub

save_for_hub(
    model,
    tokenizer,
    save_dir='./hub_model',
    model_name='crumb-llm-small',
    metadata={'trained_on': 'CRUMB corpus'},
)
```

## CLI Commands

### Training

```bash
# Train model
python -m crumb_llm.train \
    --config small \
    --data training_data.txt \
    --steps 50000 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --output-dir ./checkpoints

# Options:
#   --config: tiny, small, medium, large
#   --data: Training data path
#   --steps: Training steps
#   --batch-size: Batch size
#   --learning-rate: Learning rate
#   --grad-accum: Gradient accumulation steps
#   --mixed-precision: Enable mixed precision
#   --gradient-checkpointing: Enable gradient checkpointing
#   --eval-every: Evaluate every N steps
#   --save-every: Save every N steps
```

### Generation

```bash
# Interactive generation
python -m crumb_llm.generate \
    --model checkpoints/model.pt \
    --interactive

# Single generation
python -m crumb_llm.generate \
    --model checkpoints/model.pt \
    --prompt "BEGIN CRUMB" \
    --max-new-tokens 100 \
    --temperature 0.8

# Options:
#   --model: Model checkpoint path
#   --prompt: Input prompt
#   --max-new-tokens: Tokens to generate
#   --temperature: Sampling temperature
#   --top-k: Top-k sampling
#   --top-p: Nucleus sampling
#   --repetition-penalty: Repetition penalty
```

### Serving

```bash
# Start HTTP API server
python -m crumb_llm.serve \
    --model checkpoints/model.pt \
    --port 8000 \
    --quantization fp16 \
    --compile

# Options:
#   --model: Model checkpoint path
#   --port: Server port
#   --host: Server host
#   --quantization: none, fp16, int8
#   --compile: Enable torch.compile()
#   --workers: Number of workers
```

### Benchmarking

```bash
# Run full benchmark suite
python -m benchmarks.benchmark_suite \
    --model checkpoints/model.pt \
    --baselines gpt2,llama \
    --output results/

# Quick smoke test
python -m benchmarks.benchmark_suite \
    --model checkpoints/model.pt \
    --quick

# Individual benchmarks
python -m benchmarks.perplexity_bench --model MODEL
python -m benchmarks.speed_bench --model MODEL
python -m benchmarks.memory_bench --model MODEL
```

### Quantization

```bash
# Quantize model
python -m crumb_llm.quantization \
    --checkpoint model.pt \
    --mode fp16 \
    --output quantized.pt

# Analyze tradeoffs
python -m crumb_llm.quantization \
    --checkpoint model.pt \
    --analyze \
    --test-data test.txt

# Options:
#   --mode: dynamic_int8, static_int8, fp16, bf16
#   --output: Output path
#   --analyze: Show quality/speed tradeoffs
```

## Type Hints

All public APIs include type hints for better IDE support:

```python
from crumb_llm import WaveFieldLM, WaveFieldConfig
from torch import Tensor
from typing import Optional, Dict, List

def forward(
    self,
    idx: Tensor,
    targets: Optional[Tensor] = None
) -> Dict[str, Tensor]:
    ...
```

## Error Handling

Common exceptions:

```python
# Import without PyTorch
try:
    import crumb_llm
except ImportError as e:
    # "crumb_llm requires PyTorch. Install with: pip install 'crumb-format[llm]'"
    pass

# Invalid configuration
try:
    config = WaveFieldConfig(field_size=1000)  # Not power of 2
except ValueError as e:
    # "field_size must be a power of 2"
    pass

# Checkpoint not found
try:
    model, tokenizer = load_checkpoint('missing.pt')
except FileNotFoundError as e:
    # "Checkpoint not found: missing.pt"
    pass
```

## See Also

- [Getting Started](GETTING_STARTED.md) - Installation and first steps
- [Architecture](ARCHITECTURE.md) - Technical deep dive
- [Training Guide](TRAINING_GUIDE.md) - Advanced training
- [Inference Guide](INFERENCE_GUIDE.md) - Deployment and optimization
- [Examples](EXAMPLES.md) - Practical code examples

---

**Version:** 0.3.0  
**Last Updated:** 2026-05-28
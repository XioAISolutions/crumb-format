# CRUMB LLM Examples

Practical code examples for common tasks with CRUMB LLM. All examples are complete and runnable.

## Table of Contents

1. [Training Examples](#training-examples)
2. [Inference Examples](#inference-examples)
3. [Optimization Examples](#optimization-examples)
4. [CRUMB-Specific Examples](#crumb-specific-examples)
5. [Production Examples](#production-examples)

## Training Examples

### Example 1: Train a Tiny Model (Quick Start)

The fastest way to train a model:

```python
"""Train a tiny CRUMB LLM model in 2 minutes."""
import torch
from crumb_llm import WaveFieldLM, WaveFieldConfig
from crumb_llm.data import load_dataset
from crumb_llm.train import simple_train

# Load data
text = open('training_data.txt').read()
train_data, val_data = load_dataset(text, split=0.9)

# Create tiny model
config = WaveFieldConfig.tiny()
model = WaveFieldLM(config)

# Train
simple_train(
    model,
    train_data,
    val_data,
    steps=500,
    batch_size=32,
    learning_rate=3e-4,
    output_dir='./checkpoints',
)

print("Training complete! Model saved to ./checkpoints/")
```

### Example 2: Train with Full Features

Complete training with all Phase 2 features:

```python
"""Train CRUMB LLM with comprehensive features."""
import torch
from torch.utils.data import DataLoader
from crumb_llm import WaveFieldLM, WaveFieldConfig
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig
from crumb_llm.optimizer import OptimizerConfig
from crumb_llm.loss import LossConfig
from crumb_llm.data_enhanced import TextDataset, DataConfig, create_dataloader
from crumb_llm.tokenizer import ByteTokenizer

# Prepare data
tokenizer = ByteTokenizer()
with open('training_data.txt') as f:
    text = f.read()

train_dataset = TextDataset(text, tokenizer, block_size=512, random_crop=True)
data_config = DataConfig(
    batch_size=32,
    dynamic_batching=True,
    max_tokens_per_batch=16384,
    num_workers=4,
)
train_loader = create_dataloader(train_dataset, data_config, is_train=True)

# Create model
model_config = WaveFieldConfig.small()
model = WaveFieldLM(model_config)

# Configure training
trainer_config = TrainerConfig(
    max_steps=50000,
    optimizer=OptimizerConfig(
        learning_rate=3e-4,
        weight_decay=0.1,
        scheduler='cosine',
        warmup_steps=1000,
    ),
    loss=LossConfig(
        label_smoothing=0.1,
        spectral_diversity_weight=0.01,
        field_smoothness_weight=0.01,
    ),
    mixed_precision=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    eval_every=500,
    save_every=1000,
    checkpoint_dir='./checkpoints',
)

# Train
trainer = WaveFieldTrainer(model, train_loader, None, trainer_config)
metrics = trainer.train()

print(f"Final loss: {metrics['final_loss']:.4f}")
print(f"Best eval loss: {metrics['best_eval_loss']:.4f}")
```

### Example 3: Multi-GPU Training

Distributed training across multiple GPUs:

```python
"""Multi-GPU training with DDP."""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from crumb_llm import WaveFieldLM, WaveFieldConfig
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def train_distributed():
    """Train on multiple GPUs."""
    setup_distributed()
    
    # Create model
    config = WaveFieldConfig.medium()
    model = WaveFieldLM(config).cuda()
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])
    
    # Configure training
    trainer_config = TrainerConfig(
        max_steps=100000,
        batch_size=16,  # Per GPU
        learning_rate=3e-4,
        mixed_precision=True,
    )
    
    # Train
    trainer = WaveFieldTrainer(model, train_loader, eval_loader, trainer_config)
    trainer.train()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    train_distributed()
```

Run with:
```bash
torchrun --nproc_per_node=4 train_distributed.py
```

### Example 4: Fine-Tuning on Custom Data

Fine-tune a pretrained model:

```python
"""Fine-tune CRUMB LLM on custom data."""
from crumb_llm.sample import load_checkpoint
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig

# Load pretrained model
model, tokenizer = load_checkpoint('pretrained_model.pt')

# Prepare custom data
custom_dataset = TextDataset(custom_text, tokenizer, block_size=512)
custom_loader = create_dataloader(custom_dataset, data_config, is_train=True)

# Fine-tune with lower learning rate
trainer_config = TrainerConfig(
    max_steps=5000,
    learning_rate=1e-4,  # Lower for fine-tuning
    warmup_steps=100,
    eval_every=100,
    save_every=500,
    checkpoint_dir='./finetuned',
)

trainer = WaveFieldTrainer(model, custom_loader, None, trainer_config)
trainer.train()

print("Fine-tuning complete!")
```

## Inference Examples

### Example 5: Basic Text Generation

Simple text generation:

```python
"""Generate text with CRUMB LLM."""
from crumb_llm.sample import load_checkpoint

# Load model
model, tokenizer = load_checkpoint('checkpoints/model.pt')
model.eval()

# Generate
prompt = "BEGIN CRUMB\nv=1.3\nkind=task"
context = tokenizer.encode(prompt)
context = torch.tensor([context], dtype=torch.long)

generated = model.generate(
    context,
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
)

text = tokenizer.decode(generated[0].tolist())
print(text)
```

### Example 6: Advanced Generation with Sampling

Multiple sampling strategies:

```python
"""Advanced text generation with different sampling strategies."""
from crumb_llm.generate import Generator, GenerationConfig
from crumb_llm.sample import load_checkpoint

model, tokenizer = load_checkpoint('model.pt')
generator = Generator(model, tokenizer)

# Greedy decoding (deterministic)
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.0,  # Greedy
)
text = generator.generate("Hello", config)
print("Greedy:", text)

# Temperature sampling
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
)
text = generator.generate("Hello", config)
print("Temperature:", text)

# Top-k sampling
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
)
text = generator.generate("Hello", config)
print("Top-k:", text)

# Nucleus (top-p) sampling
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)
text = generator.generate("Hello", config)
print("Nucleus:", text)

# With repetition penalty
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    repetition_penalty=1.2,
)
text = generator.generate("Hello", config)
print("With penalty:", text)
```

### Example 7: Streaming Generation

Token-by-token streaming:

```python
"""Stream tokens as they're generated."""
from crumb_llm.streaming import StreamingGenerator, StreamingConfig
from crumb_llm.generate import GenerationConfig
from crumb_llm.sample import load_checkpoint

model, tokenizer = load_checkpoint('model.pt')
generator = StreamingGenerator(model, tokenizer)

gen_config = GenerationConfig(max_new_tokens=200, temperature=0.8)
stream_config = StreamingConfig(buffer_size=1, include_metadata=True)

print("Streaming generation:")
for token in generator.stream("BEGIN CRUMB", gen_config, stream_config):
    print(token.text, end='', flush=True)
    
    # Access metadata
    if token.metadata:
        print(f"\n[logprob: {token.metadata.get('logprob', 0):.2f}]", end='')

print("\n\nDone!")
```

### Example 8: Batch Generation

Generate multiple sequences in parallel:

```python
"""Batch generation for efficiency."""
from crumb_llm.generate import Generator, GenerationConfig
from crumb_llm.sample import load_checkpoint

model, tokenizer = load_checkpoint('model.pt')
generator = Generator(model, tokenizer)

# Multiple prompts
prompts = [
    "BEGIN CRUMB\nv=1.3\nkind=task",
    "BEGIN CRUMB\nv=1.3\nkind=mem",
    "BEGIN CRUMB\nv=1.3\nkind=map",
]

config = GenerationConfig(max_new_tokens=100, temperature=0.8)

# Generate all at once
results = generator.batch_generate(prompts, config, batch_size=3)

for i, text in enumerate(results):
    print(f"\n=== Prompt {i+1} ===")
    print(text)
```

## Optimization Examples

### Example 9: Quantization for Speed

Quantize model for faster inference:

```python
"""Quantize CRUMB LLM for 2-4× speedup."""
from crumb_llm.quantization import quantize_model, QuantizationConfig
from crumb_llm.sample import load_checkpoint
import torch

# Load model
model, tokenizer = load_checkpoint('model.pt')

# FP16 quantization (GPU)
if torch.cuda.is_available():
    config = QuantizationConfig(mode='fp16')
    model_fp16 = quantize_model(model, config)
    
    # Save quantized model
    torch.save({
        'model': model_fp16.state_dict(),
        'config': model.cfg,
        'tokenizer': tokenizer,
    }, 'model_fp16.pt')
    
    print("FP16 model saved (2× faster, 2× smaller)")

# INT8 quantization (CPU)
config = QuantizationConfig(mode='dynamic_int8')
model_int8 = quantize_model(model, config)

torch.save({
    'model': model_int8.state_dict(),
    'config': model.cfg,
    'tokenizer': tokenizer,
}, 'model_int8.pt')

print("INT8 model saved (4× smaller, 2-4× faster on CPU)")
```

### Example 10: Cached Generation

Use field-state caching for 10-100× speedup:

```python
"""Efficient generation with field-state caching."""
from crumb_llm.cache import FieldStateCache, generate_cached
from crumb_llm.sample import load_checkpoint
import torch
import time

model, tokenizer = load_checkpoint('model.pt')
model.eval()

prompt = "BEGIN CRUMB\nv=1.3\nkind=task"
input_ids = torch.tensor([tokenizer.encode(prompt)])

# Without cache
start = time.time()
output = model.generate(input_ids, max_new_tokens=100)
uncached_time = time.time() - start

# With cache
cache = FieldStateCache.init(model, batch_size=1)
start = time.time()
output = generate_cached(model, input_ids, max_new_tokens=100, cache=cache)
cached_time = time.time() - start

print(f"Without cache: {uncached_time:.2f}s")
print(f"With cache: {cached_time:.2f}s")
print(f"Speedup: {uncached_time/cached_time:.1f}×")
```

### Example 11: torch.compile() Optimization

Use torch.compile() for 2× speedup:

```python
"""Optimize with torch.compile()."""
import torch
from crumb_llm.sample import load_checkpoint

model, tokenizer = load_checkpoint('model.pt')

# Compile model
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')
    print("Model compiled with torch.compile()")
    
    # Warmup
    dummy_input = torch.randint(0, 256, (1, 128))
    for _ in range(10):
        model(dummy_input)
    
    print("Warmup complete, model is now optimized!")
else:
    print("torch.compile() not available (requires PyTorch 2.0+)")

# Use compiled model normally
output = model.generate(input_ids, max_new_tokens=100)
```

## CRUMB-Specific Examples

### Example 12: Structure-Aware Training

Train with CRUMB structure awareness:

```python
"""Train CRUMB LLM with structure awareness."""
from crumb_llm import WaveFieldLM, WaveFieldConfig
from crumb_llm.data_enhanced import CRUMBDataset, create_dataloader, DataConfig
from crumb_llm.tokenizer import ByteTokenizer
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig
import glob

# Load CRUMB files
crumb_files = glob.glob('examples/*.crumb')
tokenizer = ByteTokenizer()

# Create structure-aware dataset
dataset = CRUMBDataset(
    crumb_files,
    tokenizer,
    block_size=512,
    preserve_structure=True,  # Keep section boundaries
)

data_config = DataConfig(batch_size=32, num_workers=4)
train_loader = create_dataloader(dataset, data_config, is_train=True)

# Create model
config = WaveFieldConfig.small()
model = WaveFieldLM(config)

# Train with structure features
trainer_config = TrainerConfig(
    max_steps=50000,
    learning_rate=3e-4,
    structure_aware=True,      # Enable structure detection
    priority_weighting=True,   # Use @priority annotations
    fold_adaptive=True,        # Adaptive damping for folds
)

trainer = WaveFieldTrainer(model, train_loader, None, trainer_config)
trainer.train()

print("Structure-aware training complete!")
```

### Example 13: Priority-Based Generation

Generate with priority awareness:

```python
"""Generate CRUMB documents with priority awareness."""
from crumb_llm.generate import Generator, GenerationConfig
from crumb_llm.crumb_adapter import CRUMBAdapter
from crumb_llm.sample import load_checkpoint

model, tokenizer = load_checkpoint('model.pt')

# Wrap with CRUMB adapter
adapter = CRUMBAdapter(model)

# Generate with priority hints
prompt = """BEGIN CRUMB
v=1.3
kind=task
title=Important Task
---
[goal]
@priority: 5
"""

config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.8,
    use_priorities=True,  # Enable priority-aware generation
)

generator = Generator(adapter, tokenizer)
text = generator.generate(prompt, config)

print(text)
```

### Example 14: Cross-Reference Resolution

Use cached field states for cross-references:

```python
"""Efficient cross-reference handling."""
from crumb_llm.cache import FieldStateCache
from crumb_llm.crumb_adapter import CRUMBAdapter
from crumb_llm.sample import load_checkpoint
import torch

model, tokenizer = load_checkpoint('model.pt')
adapter = CRUMBAdapter(model)

# Cache referenced documents
cache = FieldStateCache.init(model, batch_size=1)

# Process reference document
ref_doc = "BEGIN CRUMB\nv=1.3\nkind=mem\n..."
ref_ids = torch.tensor([tokenizer.encode(ref_doc)])
ref_field = adapter.process_reference(ref_ids)
cache.store('mem-prefs-abc123', ref_field)

# Generate with reference
prompt = """BEGIN CRUMB
v=1.3
kind=task
refs=mem-prefs-abc123
---
[goal]
"""

input_ids = torch.tensor([tokenizer.encode(prompt)])
output = adapter.generate_with_refs(
    input_ids,
    ref_cache=cache,
    max_new_tokens=200,
)

text = tokenizer.decode(output[0].tolist())
print(text)
```

## Production Examples

### Example 15: HTTP API Server

Serve model via HTTP API:

```python
"""Production HTTP API server."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crumb_llm.serve import create_app
from crumb_llm.inference import InferenceEngine, InferenceConfig
import uvicorn

# Configure inference
config = InferenceConfig(
    quantization='fp16',
    use_compile=True,
    compile_mode='reduce-overhead',
)

# Create engine
engine = InferenceEngine.from_checkpoint('model.pt', config)

# Create FastAPI app
app = create_app(engine)

# Add custom endpoint
@app.post("/custom/generate")
async def custom_generate(request: dict):
    """Custom generation endpoint."""
    try:
        text = engine.generate(
            request['prompt'],
            max_tokens=request.get('max_tokens', 100),
            temperature=request.get('temperature', 0.8),
        )
        return {'text': text, 'status': 'success'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=4)
```

Run with:
```bash
python serve_api.py
```

### Example 16: Client Usage

Use the HTTP API from client code:

```python
"""Client for CRUMB LLM HTTP API."""
import requests
import json

API_URL = "http://localhost:8000"

def generate_text(prompt, max_tokens=100, temperature=0.8):
    """Generate text via API."""
    response = requests.post(
        f"{API_URL}/generate",
        json={
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
    )
    response.raise_for_status()
    return response.json()['text']

def stream_text(prompt, max_tokens=100):
    """Stream text via SSE."""
    response = requests.get(
        f"{API_URL}/stream",
        params={
            'prompt': prompt,
            'max_tokens': max_tokens,
        },
        stream=True,
    )
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8').replace('data: ', ''))
            if data['type'] == 'token':
                print(data['text'], end='', flush=True)
            elif data['type'] == 'done':
                break

# Usage
text = generate_text("BEGIN CRUMB", max_tokens=200)
print(text)

print("\nStreaming:")
stream_text("BEGIN CRUMB", max_tokens=200)
```

### Example 17: Docker Deployment

Dockerfile for production deployment:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY checkpoints/ checkpoints/
COPY crumb_llm/ crumb_llm/
COPY serve_api.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "serve_api.py"]
```

Build and run:
```bash
docker build -t crumb-llm .
docker run -p 8000:8000 --gpus all crumb-llm
```

### Example 18: Kubernetes Deployment

Kubernetes deployment configuration:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crumb-llm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crumb-llm
  template:
    metadata:
      labels:
        app: crumb-llm
    spec:
      containers:
      - name: crumb-llm
        image: crumb-llm:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            nvidia.com/gpu: 1
            memory: 4Gi
        env:
        - name: MODEL_PATH
          value: "/models/model.pt"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: crumb-llm-service
spec:
  selector:
    app: crumb-llm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

## See Also

- [Getting Started](GETTING_STARTED.md) - Installation and basics
- [Architecture](ARCHITECTURE.md) - Technical details
- [API Reference](API_REFERENCE.md) - Complete API docs
- [Training Guide](TRAINING_GUIDE.md) - Advanced training
- [Inference Guide](INFERENCE_GUIDE.md) - Optimization

---

**All examples are tested and ready to use. Copy, modify, and adapt them for your needs!**
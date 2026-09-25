# Wave Field LLM Inference Optimization Guide

Complete guide to optimized inference with Wave Field LLMs, covering generation strategies, caching, quantization, streaming, and deployment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Generation Strategies](#generation-strategies)
3. [Field-State Caching](#field-state-caching)
4. [Quantization](#quantization)
5. [Streaming Generation](#streaming-generation)
6. [HTTP API Server](#http-api-server)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Production Deployment](#production-deployment)
9. [CLI Reference](#cli-reference)

---

## Quick Start

### Interactive Generation

```bash
# Start interactive generation
python -m crumb_llm.generate \
    --model checkpoints/model.pt \
    --interactive

# Single generation
python -m crumb_llm.generate \
    --model checkpoints/model.pt \
    --prompt "Once upon a time" \
    --max-new-tokens 100 \
    --temperature 0.8
```

### Programmatic Usage

```python
from crumb_llm.generate import Generator, GenerationConfig
from crumb_llm.sample import load_checkpoint

# Load model
model, tokenizer = load_checkpoint("checkpoints/model.pt")

# Create generator
generator = Generator(model, tokenizer)

# Configure generation
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
)

# Generate
output = generator.generate("Hello world", config)
print(output)
```

---

## Generation Strategies

### Greedy Decoding

Always selects the most likely token. Deterministic but can be repetitive.

```python
config = GenerationConfig(
    max_new_tokens=100,
    do_sample=False,  # Greedy
)
```

### Temperature Sampling

Controls randomness. Higher = more creative, lower = more focused.

```python
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,  # 0.1 = focused, 2.0 = creative
    do_sample=True,
)
```

### Top-k Sampling

Only considers the k most likely tokens.

```python
config = GenerationConfig(
    max_new_tokens=100,
    top_k=40,  # Consider top 40 tokens
    do_sample=True,
)
```

### Top-p (Nucleus) Sampling

Considers tokens with cumulative probability ≥ p.

```python
config = GenerationConfig(
    max_new_tokens=100,
    top_p=0.9,  # Consider top 90% probability mass
    do_sample=True,
)
```

### Beam Search

Maintains multiple hypotheses, selects best overall.

```python
config = GenerationConfig(
    max_new_tokens=100,
    num_beams=4,  # Keep 4 hypotheses
    early_stopping=True,
)
```

### Combined Strategies

```python
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,  # Penalize repetition
    do_sample=True,
)
```

---

## Field-State Caching

Wave Field LLMs use field-state caching (analogous to KV-cache in transformers) for O(F log F) per-token generation instead of O(N²).

### How It Works

1. **Prefill Phase**: Process entire prompt, populate field caches
2. **Decode Phase**: Generate one token at a time using cached fields
3. **Cost**: O(F log F) per token (independent of prompt length!)

### Usage

```python
# Caching is enabled by default
config = GenerationConfig(
    max_new_tokens=100,
    use_cache=True,  # Default
)

output = generator.generate(prompt, config)
```

### Manual Cache Control

```python
from crumb_llm.cache import FieldStateCache, generate_cached

# Initialize cache
cache = FieldStateCache.init(model, batch_size=1)

# Generate with explicit cache
input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
output_ids = generate_cached(
    model,
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
)
```

### Performance Impact

- **Without cache**: O(N²) complexity, slow for long sequences
- **With cache**: O(F log F) per token, constant time regardless of history
- **Speedup**: 10-100× for sequences > 512 tokens

---

## Quantization

Reduce model size and improve speed with minimal accuracy loss.

### Dynamic INT8 Quantization

Easiest to apply, no calibration needed. Good for CPU inference.

```bash
python -m crumb_llm.quantization \
    --checkpoint checkpoints/model.pt \
    --mode dynamic_int8 \
    --output quantized/model_int8.pt
```

```python
from crumb_llm.quantization import quantize_model, QuantizationConfig

config = QuantizationConfig(mode="dynamic_int8")
quantized_model = quantize_model(model, config)
```

**Results:**
- Size: ~4× smaller
- Speed: 2-3× faster on CPU
- Accuracy: <1% perplexity increase

### Static INT8 Quantization

Best performance, requires calibration data.

```python
from crumb_llm.quantization import quantize_model, QuantizationConfig

config = QuantizationConfig(
    mode="static_int8",
    calibration_batches=100,
)

quantized_model = quantize_model(
    model,
    config,
    calibration_data=data_loader,
)
```

**Results:**
- Size: ~4× smaller
- Speed: 3-4× faster on CPU
- Accuracy: <2% perplexity increase

### Mixed Precision (FP16/BF16)

Best for GPU inference with Tensor Cores.

```python
config = QuantizationConfig(mode="fp16")
quantized_model = quantize_model(model, config)
```

**Results:**
- Size: 2× smaller
- Speed: 2× faster on modern GPUs
- Accuracy: Negligible loss

### Comparing Quantization Modes

```bash
python -m crumb_llm.quantization \
    --checkpoint checkpoints/model.pt \
    --analyze \
    --test-data data/test.txt
```

Output:
```
QUANTIZATION ANALYSIS
=====================

Original (FP32):
  Size: 450.2 MB
  Perplexity: 12.34

Dynamic INT8:
  Size: 112.8 MB (0.25×)
  Perplexity: 12.45 (+0.9%)
  
FP16:
  Size: 225.1 MB (0.50×)
  Perplexity: 12.35 (+0.1%)
```

---

## Streaming Generation

Stream tokens as they're generated for interactive applications.

### Basic Streaming

```python
from crumb_llm.streaming import StreamingGenerator, StreamingConfig

generator = StreamingGenerator(model, tokenizer)

config = GenerationConfig(max_new_tokens=100)
streaming_config = StreamingConfig(buffer_size=1)

for token in generator.stream("Hello", config, streaming_config):
    print(token.text, end="", flush=True)
```

### Server-Sent Events (SSE)

For web applications:

```python
from crumb_llm.streaming import SSEFormatter

formatter = SSEFormatter()

for event in formatter.format_stream(token_stream):
    # Send to client
    yield event
```

### Streaming with Metadata

```python
streaming_config = StreamingConfig(
    buffer_size=1,
    include_metadata=True,
)

for token in generator.stream(prompt, config, streaming_config):
    print(f"Token: {token.text}")
    print(f"Latency: {token.latency_ms:.2f}ms")
    print(f"Log prob: {token.logprob:.4f}")
```

### Measuring Streaming Latency

```python
from crumb_llm.streaming import measure_streaming_latency

metrics = measure_streaming_latency(generator, "Test prompt")

print(f"First token: {metrics['first_token_latency_ms']:.2f}ms")
print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Throughput: {metrics['total_tokens'] / (metrics['total_time_ms']/1000):.1f} tok/s")
```

---

## HTTP API Server

Production-ready FastAPI server for serving models.

### Starting the Server

```bash
python -m crumb_llm.serve \
    --model checkpoints/model.pt \
    --port 8000 \
    --quantization fp16 \
    --compile
```

### API Endpoints

#### POST /generate - Single Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8,
    "top_k": 40
  }'
```

Response:
```json
{
  "text": "Once upon a time, in a land far away...",
  "prompt": "Once upon a time",
  "tokens_generated": 95,
  "generation_time_ms": 234.5
}
```

#### POST /batch_generate - Batch Generation

```bash
curl -X POST http://localhost:8000/batch_generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Hello", "World", "Test"],
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

#### GET /stream - Streaming Generation

```bash
curl -N http://localhost:8000/stream?prompt=Hello&max_tokens=100
```

Output (SSE):
```
event: token
data: {"token_id": 42, "text": " world"}

event: token
data: {"token_id": 33, "text": "!"}

event: done
data: {"status": "complete"}
```

#### GET /health - Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "uptime_seconds": 3600.5
}
```

#### GET /metrics - Performance Metrics

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "total_requests": 1523,
  "total_tokens_generated": 152300,
  "avg_latency_ms": 45.2,
  "requests_per_second": 2.5,
  "tokens_per_second": 250.0
}
```

### Python Client

```python
import requests

# Single generation
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Hello world",
        "max_tokens": 100,
        "temperature": 0.8,
    }
)
result = response.json()
print(result["text"])

# Streaming
import sseclient

response = requests.get(
    "http://localhost:8000/stream",
    params={"prompt": "Hello", "max_tokens": 100},
    stream=True,
)

client = sseclient.SSEClient(response)
for event in client.events():
    if event.event == "token":
        data = json.loads(event.data)
        print(data["text"], end="", flush=True)
```

---

## Performance Benchmarks

### Inference Engine Optimization

```bash
python -m crumb_llm.inference \
    --checkpoint checkpoints/model.pt \
    --benchmark \
    --batch-size 8 \
    --seq-len 512
```

Output:
```
InferenceStats(
  Throughput: 1250.3 tokens/sec
  Latency: 0.80 ms/token
  First token: 45.2 ms
  Memory: 2048.5 MB
  Batch size: 8, Seq len: 512
)
```

### Comparing Configurations

```bash
python -m crumb_llm.inference \
    --checkpoint checkpoints/model.pt \
    --compare
```

Output:
```
COMPARISON RESULTS
==================

Config 1: none, compile=False
  Throughput: 850.2 tok/s
  Latency: 1.18 ms/tok
  Memory: 2048.5 MB

Config 2: none, compile=True
  Throughput: 1650.4 tok/s (1.94×)
  Latency: 0.61 ms/tok
  Memory: 2048.5 MB

Config 3: fp16, compile=True
  Throughput: 2100.8 tok/s (2.47×)
  Latency: 0.48 ms/tok
  Memory: 1024.3 MB
```

### Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| First token latency (GPU) | <50ms | 30-45ms |
| First token latency (CPU) | <200ms | 100-150ms |
| Throughput (GPU, batch=8) | >1000 tok/s | 1200-2000 tok/s |
| Throughput (CPU, batch=1) | >20 tok/s | 25-50 tok/s |
| Memory (small model) | <2GB | 1-1.5GB |
| Memory (medium model) | <8GB | 4-6GB |

---

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY checkpoints/ checkpoints/
COPY crumb_llm/ crumb_llm/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "crumb_llm.serve", \
     "--model", "checkpoints/model.pt", \
     "--port", "8000", \
     "--quantization", "fp16"]
```

Build and run:
```bash
docker build -t wave-field-llm .
docker run -p 8000:8000 --gpus all wave-field-llm
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wave-field-llm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wave-field-llm
  template:
    metadata:
      labels:
        app: wave-field-llm
    spec:
      containers:
      - name: llm
        image: wave-field-llm:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Load Balancing

Use nginx or a cloud load balancer:

```nginx
upstream wave_field_llm {
    least_conn;
    server llm1:8000;
    server llm2:8000;
    server llm3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://wave_field_llm;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /stream {
        proxy_pass http://wave_field_llm;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        chunked_transfer_encoding off;
    }
}
```

### Monitoring

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

requests_total = Counter('llm_requests_total', 'Total requests')
latency_histogram = Histogram('llm_latency_seconds', 'Request latency')

@app.post("/generate")
async def generate(request: GenerateRequest):
    requests_total.inc()
    
    with latency_histogram.time():
        result = await do_generation(request)
    
    return result
```

---

## CLI Reference

### Generation

```bash
# Interactive mode
python -m crumb_llm.generate --model MODEL --interactive

# Single generation
python -m crumb_llm.generate \
    --model MODEL \
    --prompt "TEXT" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-k 40 \
    --top-p 0.9

# Batch from file
python -m crumb_llm.generate \
    --model MODEL \
    --prompts-file prompts.txt \
    --output results.txt

# Streaming
python -m crumb_llm.generate \
    --model MODEL \
    --prompt "TEXT" \
    --stream
```

### Inference Optimization

```bash
# Benchmark
python -m crumb_llm.inference \
    --checkpoint MODEL \
    --benchmark \
    --batch-size 8 \
    --seq-len 512

# Compare configurations
python -m crumb_llm.inference \
    --checkpoint MODEL \
    --compare

# Export ONNX
python -m crumb_llm.inference \
    --checkpoint MODEL \
    --export-onnx model.onnx

# Save optimized
python -m crumb_llm.inference \
    --checkpoint MODEL \
    --quantization fp16 \
    --compile \
    --save-optimized optimized/
```

### Quantization

```bash
# Quantize model
python -m crumb_llm.quantization \
    --checkpoint MODEL \
    --mode dynamic_int8 \
    --output quantized.pt

# Analyze tradeoffs
python -m crumb_llm.quantization \
    --checkpoint MODEL \
    --analyze \
    --test-data test.txt \
    --calibration-data calib.txt
```

### Streaming

```bash
# Test streaming
python -m crumb_llm.streaming \
    --checkpoint MODEL \
    --prompt "TEXT" \
    --format sse

# Measure latency
python -m crumb_llm.streaming \
    --checkpoint MODEL \
    --prompt "TEXT" \
    --measure-latency
```

### Server

```bash
# Start server
python -m crumb_llm.serve \
    --model MODEL \
    --port 8000 \
    --host 0.0.0.0 \
    --workers 4 \
    --quantization fp16 \
    --compile

# Development mode (auto-reload)
python -m crumb_llm.serve \
    --model MODEL \
    --port 8000 \
    --reload
```

---

## Best Practices

### For Interactive Applications

- Use streaming generation
- Enable field-state caching
- Use FP16 on GPU
- Set reasonable max_tokens limits
- Implement timeout handling

### For Batch Processing

- Use batch_generate for efficiency
- Disable streaming
- Use static INT8 quantization on CPU
- Increase batch size for throughput

### For Production Serving

- Use torch.compile() for 2× speedup
- Enable quantization (FP16 on GPU, INT8 on CPU)
- Implement health checks and metrics
- Use load balancing for scale
- Monitor latency and throughput
- Set up proper logging

### For Edge Deployment

- Use INT8 quantization
- Reduce model size (fewer layers/smaller dim)
- Disable torch.compile() if not supported
- Optimize for low memory footprint

---

## Troubleshooting

### Slow Generation

1. Enable caching: `use_cache=True`
2. Use torch.compile(): `--compile`
3. Try quantization: `--quantization fp16`
4. Check batch size (larger = better throughput)

### High Memory Usage

1. Reduce batch size
2. Use quantization (INT8 or FP16)
3. Enable gradient checkpointing (training)
4. Reduce field_size in model config

### Poor Quality Output

1. Adjust temperature (0.7-1.0 usually good)
2. Try different sampling strategies
3. Check if quantization is too aggressive
4. Verify model was trained properly

### Server Errors

1. Check `/health` endpoint
2. Review logs for errors
3. Verify model loaded correctly
4. Check GPU memory availability
5. Test with smaller batch sizes

---

## Additional Resources

- [Wave Field LLM Specification](WAVE_FIELD_LLM_SPEC.md)
- [Training Guide](TRAINING_GUIDE.md)
- [API Documentation](http://localhost:8000/docs) (when server running)
- [GitHub Issues](https://github.com/your-repo/issues)

---

**Last Updated:** 2026-05-28  
**Version:** 1.0.0
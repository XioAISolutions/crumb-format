# Wave Field LLM Model Zoo

Complete infrastructure for training, evaluating, documenting, and distributing pretrained Wave Field LLM models.

## Overview

The Model Zoo provides:

- **7 Pretrained Model Configurations** - From 230K to 1.5B parameters
- **Training Pipeline** - Complete orchestration with distributed training
- **Evaluation Framework** - Comprehensive benchmarks and metrics
- **Model Registry** - Central registry with versioning and metadata
- **Distribution System** - Multi-cloud upload/download with CDN support
- **Model Cards** - Automated documentation generation
- **Quality Assurance** - Validation and testing framework

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Load a Pretrained Model

```python
from model_zoo import load_model

# Load from registry
model = load_model("wavefield-small")

# Load with quantization
model = load_model("wavefield-small", quantization="fp16")

# Load from HuggingFace
model = load_model("wavefield-llm/wavefield-small")
```

### Train a Model

```bash
# Train small model on 8 GPUs
./model_zoo/scripts/train_model.sh small 8

# Train with custom config
python -m model_zoo.train_pipeline \
  --config model_zoo/configs/small.yaml \
  --output-dir models/my-model \
  --num-gpus 8
```

### Evaluate a Model

```bash
# Evaluate on standard benchmarks
./model_zoo/scripts/evaluate_model.sh \
  models/wavefield-small \
  results/eval.json \
  mmlu hellaswag arc_challenge
```

## Available Models

| Model | Parameters | Context | Training Tokens | Use Case |
|-------|-----------|---------|-----------------|----------|
| **wavefield-tiny** | 230K | 2K | 10B | Experimentation |
| **wavefield-small** | 25M | 4K | 100B | General purpose |
| **wavefield-medium** | 350M | 8K | 300B | Production |
| **wavefield-large** | 1.5B | 16K | 1T | State-of-the-art |
| **wavefield-small-code** | 25M | 8K | 50B | Code generation |
| **wavefield-medium-crumb** | 350M | 16K | 100B | CRUMB format |
| **wavefield-large-instruct** | 1.5B | 32K | 80B | Instruction following |

## Architecture

```
model_zoo/
├── configs/              # Model configurations
│   ├── tiny.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   ├── large.yaml
│   ├── small-code.yaml
│   ├── medium-crumb.yaml
│   └── large-instruct.yaml
├── train_pipeline.py     # Training orchestration
├── loader.py             # Model loading interface
├── evaluation/           # Evaluation framework
│   ├── evaluator.py
│   ├── benchmarks.py
│   ├── crumb_eval.py
│   ├── human_eval.py
│   ├── safety_eval.py
│   └── performance_eval.py
├── registry/             # Model registry
│   ├── registry.py
│   ├── metadata.py
│   ├── database.py
│   └── api.py
├── distribution/         # Distribution system
│   ├── uploader.py
│   ├── downloader.py
│   ├── cdn.py
│   └── mirror.py
├── model_cards/          # Model documentation
│   ├── generator.py
│   ├── template.md
│   └── README.md
├── scripts/              # Training scripts
│   ├── train_model.sh
│   ├── evaluate_model.sh
│   └── README.md
├── qa/                   # Quality assurance
│   └── validator.py
└── docs/                 # Documentation
    └── ...
```

## Training Pipeline

### Features

- **Multi-stage Training** - Pretraining → Fine-tuning → Instruction tuning
- **Distributed Training** - DDP, FSDP, pipeline parallelism
- **Checkpoint Management** - Automatic saving, resumption, rotation
- **Logging** - Weights & Biases, TensorBoard integration
- **Evaluation** - Automatic evaluation during training
- **Early Stopping** - Based on validation metrics
- **Resource Monitoring** - GPU, memory, disk usage

### Example

```python
from model_zoo import TrainingPipeline
from model_zoo.train_pipeline import TrainingConfig

# Load configuration
config = TrainingConfig.from_yaml("model_zoo/configs/small.yaml")

# Create pipeline
pipeline = TrainingPipeline(
    config=config,
    output_dir="models/wavefield-small",
    num_gpus=8,
)

# Train
pipeline.train()
```

## Evaluation Framework

### Supported Benchmarks

- **MMLU** - Massive Multitask Language Understanding
- **HellaSwag** - Commonsense reasoning
- **ARC** - Science questions (Easy & Challenge)
- **Winogrande** - Pronoun resolution
- **TruthfulQA** - Truthfulness
- **HumanEval** - Code generation
- **MBPP** - Python programming
- **GSM8K** - Math reasoning
- **CRUMB** - CRUMB format understanding

### Example

```python
from model_zoo import ModelEvaluator

evaluator = ModelEvaluator(
    model_path="models/wavefield-small",
    device="cuda",
    batch_size=8,
)

results = evaluator.evaluate(
    benchmarks=['mmlu', 'hellaswag', 'humaneval'],
    include_safety=True,
    include_performance=True,
)

evaluator.save_results(results, "results/eval.json")
evaluator.print_summary(results)
```

## Model Registry

### Features

- **Versioning** - Track multiple versions of each model
- **Metadata** - Architecture, training, performance metrics
- **Search** - Find models by name, tags, or description
- **Comparison** - Compare different versions
- **REST API** - HTTP API for programmatic access

### Example

```python
from model_zoo import ModelRegistry

registry = ModelRegistry()

# Register a model
registry.register(
    name="wavefield-small",
    version="1.0.0",
    checkpoint_path="models/small/checkpoint-best.pt",
    config_path="model_zoo/configs/small.yaml",
    eval_results_path="results/eval.json",
)

# Get model info
metadata = registry.get("wavefield-small", version="1.0.0")

# List all models
models = registry.list()

# Search models
results = registry.search("code")
```

## Distribution System

### Features

- **Multi-cloud Upload** - AWS S3, Google Cloud Storage, Azure Blob
- **HuggingFace Hub** - Direct integration
- **Resume Support** - Resume interrupted downloads
- **Checksum Verification** - SHA-256 validation
- **CDN Integration** - CloudFront, Cloud CDN
- **Mirror Management** - Multiple mirrors per model

### Example

```python
from model_zoo.distribution import ModelUploader, ModelDownloader

# Upload model
uploader = ModelUploader()
results = uploader.upload(
    model_path="models/wavefield-small",
    destinations=['s3', 'huggingface'],
    model_name="wavefield-small",
    version="1.0.0",
    public=True,
)

# Download model
downloader = ModelDownloader()
model_path = downloader.download(
    url="s3://wavefield-llm-models/wavefield-small-v1.0.0",
    checksum="abc123...",
)
```

## Model Cards

Automatically generate comprehensive model cards:

```bash
python -m model_zoo.model_cards.generator \
  --config model_zoo/configs/small.yaml \
  --training-log logs/training.json \
  --eval-results results/eval.json \
  --output model_zoo/model_cards/wavefield-small.md
```

Model cards include:
- Architecture and specifications
- Training details
- Performance benchmarks
- Usage examples
- Limitations and ethical considerations
- Citation information

## Quality Assurance

Validate models before release:

```python
from model_zoo.qa import ModelValidator

validator = ModelValidator("models/wavefield-small/checkpoint-best.pt")
results = validator.validate()

if results['passed']:
    print("✓ Model passed all validation checks")
else:
    print("✗ Model failed validation")
    for check, result in results['checks'].items():
        if not result['passed']:
            print(f"  - {check}: {result['message']}")
```

## Documentation

- [Training Guide](docs/TRAINING_GUIDE.md) - How to train models
- [Evaluation Guide](docs/EVALUATION_GUIDE.md) - How to evaluate models
- [Distribution Guide](docs/DISTRIBUTION_GUIDE.md) - How to distribute models
- [API Reference](docs/API_REFERENCE.md) - API documentation
- [FAQ](docs/FAQ.md) - Frequently asked questions

## Requirements

### Hardware

- **Tiny/Small**: 8GB+ GPU, 32GB RAM
- **Medium**: 24GB+ GPU, 128GB RAM
- **Large**: 80GB+ GPU, 512GB RAM

### Software

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- See `requirements.txt` for full list

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

## Citation

```bibtex
@misc{wavefield-llm-model-zoo,
  title={Wave Field LLM Model Zoo},
  author={Wave Field LLM Team},
  year={2024},
  url={https://github.com/wavefield-llm/model-zoo}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/wavefield-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/wavefield-llm/discussions)
- **Email**: support@wavefield-llm.org

## Roadmap

- [ ] Additional model sizes (XL, XXL)
- [ ] More specialized variants (math, reasoning, multilingual)
- [ ] Improved quantization (GPTQ, AWQ)
- [ ] Model merging and ensembling
- [ ] Continuous pretraining support
- [ ] Advanced RLHF pipeline

---

**Wave Field LLM Model Zoo** - Production-ready infrastructure for LLM model management.
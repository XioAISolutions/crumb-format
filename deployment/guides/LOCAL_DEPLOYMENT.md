# Wave Field LLM Local Deployment Guide

Quick guide for deploying Wave Field LLM locally for development and testing.

## Quick Start

```bash
cd deployment/automation
./quick-start.sh
```

That's it! The service will be available at `http://localhost:8000`

## Detailed Setup

### Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)
- Python 3.9+
- 8GB+ RAM
- 20GB+ disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/wavefield-llm.git
cd wavefield-llm
```

### Step 2: Download Model

```bash
python -m model_zoo.loader download wavefield-small
```

### Step 3: Configure Environment

```bash
cp deployment/docker/.env.example .env.local
# Edit .env.local with your settings
```

### Step 4: Start Services

```bash
cd deployment/docker
docker-compose up -d
```

### Step 5: Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50}'
```

## Configuration Options

### Using GPU

```bash
./quick-start.sh --gpu
```

### Custom Port

```bash
./quick-start.sh --port 8080
```

### Different Model

```bash
./quick-start.sh --model wavefield-medium
```

## Development Workflow

### Hot Reload

```bash
# Mount source code for development
docker-compose -f docker-compose.dev.yml up
```

### View Logs

```bash
docker-compose logs -f wavefield-llm
```

### Stop Services

```bash
docker-compose down
```

### Clean Restart

```bash
./quick-start.sh --clean
```

## Troubleshooting

### Port Already in Use

```bash
./quick-start.sh --port 8080
```

### Out of Memory

Reduce batch size in `.env.local`:
```
BATCH_SIZE=4
MAX_WORKERS=2
```

### Model Download Fails

```bash
# Manual download
python -m model_zoo.loader download wavefield-small --force
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Next Steps

- See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for production deployment
- See [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md) for Kubernetes setup
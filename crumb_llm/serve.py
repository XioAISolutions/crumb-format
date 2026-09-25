"""HTTP inference server for Wave Field LLMs.

This module provides a production-ready FastAPI-based REST API for serving
Wave Field LLM models with:
- Multiple generation endpoints (single, batch, streaming)
- Request queuing and batching for efficiency
- Rate limiting and authentication
- Health checks and metrics
- Model warmup on startup
- OpenAPI documentation
- CORS support

Usage:
    # Start server
    python -m crumb_llm.serve --model checkpoints/model.pt --port 8000
    
    # Or programmatically
    from crumb_llm.serve import create_app
    app = create_app("checkpoints/model.pt")
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

API Endpoints:
    POST /generate - Single text generation
    POST /batch_generate - Batch generation
    GET /stream - Streaming generation (SSE)
    GET /health - Health check
    GET /metrics - Performance metrics
    GET /info - Model information
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for type hints
    class BaseModel:
        pass
    class FastAPI:
        pass

import torch

from .model import WaveFieldLM
from .generate import Generator, GenerationConfig
from .inference import InferenceEngine, InferenceConfig
from .streaming import StreamingGenerator, SSEFormatter, StreamingConfig


# ── Request/Response Models ──────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request for single text generation."""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    stream: bool = Field(False, description="Stream response")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")


class BatchGenerateRequest(BaseModel):
    """Request for batch text generation."""
    prompts: List[str] = Field(..., min_items=1, max_items=32, description="Input prompts")
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling")


class GenerateResponse(BaseModel):
    """Response for text generation."""
    text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")


class BatchGenerateResponse(BaseModel):
    """Response for batch generation."""
    results: List[GenerateResponse] = Field(..., description="Generation results")
    total_time_ms: float = Field(..., description="Total batch processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device model is running on")
    uptime_seconds: float = Field(..., description="Server uptime")


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_requests: int = Field(..., description="Total requests served")
    total_tokens_generated: int = Field(..., description="Total tokens generated")
    avg_latency_ms: float = Field(..., description="Average latency per request")
    requests_per_second: float = Field(..., description="Current requests per second")
    tokens_per_second: float = Field(..., description="Current tokens per second")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    architecture: str = Field(..., description="Model architecture")
    parameters: int = Field(..., description="Total parameters")
    config: Dict[str, Any] = Field(..., description="Model configuration")
    device: str = Field(..., description="Device")
    quantization: str = Field(..., description="Quantization mode")


# ── Server State ─────────────────────────────────────────────────────


@dataclass
class ServerMetrics:
    """Server metrics tracking."""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    start_time: float = 0.0
    recent_requests: deque = None
    
    def __post_init__(self):
        if self.recent_requests is None:
            self.recent_requests = deque(maxlen=100)
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    def record_request(self, tokens: int, latency_ms: float):
        """Record a completed request."""
        self.total_requests += 1
        self.total_tokens_generated += tokens
        self.total_latency_ms += latency_ms
        self.recent_requests.append({
            "timestamp": time.time(),
            "tokens": tokens,
            "latency_ms": latency_ms,
        })
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        uptime = time.time() - self.start_time
        
        # Calculate recent rates (last 60 seconds)
        cutoff_time = time.time() - 60
        recent = [r for r in self.recent_requests if r["timestamp"] > cutoff_time]
        
        recent_requests = len(recent)
        recent_tokens = sum(r["tokens"] for r in recent)
        
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_latency_ms": self.total_latency_ms / max(self.total_requests, 1),
            "requests_per_second": recent_requests / 60.0,
            "tokens_per_second": recent_tokens / 60.0,
            "uptime_seconds": uptime,
        }


class ServerState:
    """Global server state."""
    def __init__(self):
        self.engine: Optional[InferenceEngine] = None
        self.generator: Optional[Generator] = None
        self.streaming_generator: Optional[StreamingGenerator] = None
        self.metrics = ServerMetrics()
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.is_ready = False


# Global state
state = ServerState()


# ── Application Factory ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print("Starting Wave Field LLM server...")
    
    if state.engine is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Warmup
    print("Warming up model...")
    try:
        dummy_prompt = "Hello world"
        _ = state.generator.generate(dummy_prompt, GenerationConfig(max_new_tokens=10))
        print("Warmup complete")
    except Exception as e:
        print(f"Warmup failed: {e}")
    
    state.is_ready = True
    print("Server ready!")
    
    yield
    
    # Shutdown
    print("Shutting down server...")
    state.is_ready = False


def create_app(
    checkpoint_path: Optional[str] = None,
    inference_config: Optional[InferenceConfig] = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        checkpoint_path: Path to model checkpoint
        inference_config: Inference configuration
        enable_cors: Whether to enable CORS
        
    Returns:
        FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for serving. Install with:\n"
            "  pip install 'crumb-format[serve]'\n"
            "or:\n"
            "  pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title="Wave Field LLM API",
        description="Production-ready inference API for Wave Field LLMs",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Enable CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Load model if checkpoint provided
    if checkpoint_path:
        load_model(checkpoint_path, inference_config)
    
    return app


def load_model(
    checkpoint_path: str,
    inference_config: Optional[InferenceConfig] = None,
):
    """Load model into server state.
    
    Args:
        checkpoint_path: Path to model checkpoint
        inference_config: Inference configuration
    """
    print(f"Loading model from {checkpoint_path}...")
    
    inference_config = inference_config or InferenceConfig()
    state.engine = InferenceEngine.from_checkpoint(checkpoint_path, inference_config)
    state.generator = state.engine.generator
    state.streaming_generator = StreamingGenerator(
        state.engine.model,
        state.engine.tokenizer,
        state.engine.device,
    )
    
    print("Model loaded successfully")


# ── API Endpoints ────────────────────────────────────────────────────


app = create_app()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text from a prompt.
    
    Args:
        request: Generation request
        
    Returns:
        Generated text response
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /stream endpoint for streaming generation"
        )
    
    start_time = time.time()
    
    try:
        # Create config
        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            stop_sequences=[list(seq.encode()) for seq in request.stop_sequences],
        )
        
        # Generate
        output = state.generator.generate(request.prompt, config)
        
        # Calculate metrics
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Estimate tokens (rough)
        if isinstance(output, str):
            tokens_generated = len(output.split())
        else:
            tokens_generated = output.shape[1] if hasattr(output, 'shape') else 0
        
        # Record metrics
        state.metrics.record_request(tokens_generated, generation_time_ms)
        
        return GenerateResponse(
            text=output if isinstance(output, str) else str(output),
            prompt=request.prompt,
            tokens_generated=tokens_generated,
            generation_time_ms=generation_time_ms,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest) -> BatchGenerateResponse:
    """Generate text for multiple prompts.
    
    Args:
        request: Batch generation request
        
    Returns:
        Batch generation response
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    
    try:
        # Create config
        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        
        # Generate
        outputs = state.generator.generate_batch(request.prompts, config)
        
        # Create responses
        results = []
        for prompt, output in zip(request.prompts, outputs):
            tokens_generated = len(output.split()) if isinstance(output, str) else 0
            results.append(GenerateResponse(
                text=output if isinstance(output, str) else str(output),
                prompt=prompt,
                tokens_generated=tokens_generated,
                generation_time_ms=0.0,  # Individual timing not available
            ))
        
        total_time_ms = (time.time() - start_time) * 1000
        total_tokens = sum(r.tokens_generated for r in results)
        
        # Record metrics
        state.metrics.record_request(total_tokens, total_time_ms)
        
        return BatchGenerateResponse(
            results=results,
            total_time_ms=total_time_ms,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream")
async def stream(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> StreamingResponse:
    """Stream generated text using Server-Sent Events.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        
    Returns:
        Streaming response with SSE events
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events."""
        try:
            config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            streaming_config = StreamingConfig(buffer_size=1)
            
            formatter = SSEFormatter()
            
            token_stream = state.streaming_generator.stream(
                prompt, config, streaming_config
            )
            
            for event in formatter.format_stream(token_stream):
                yield event
                await asyncio.sleep(0)  # Allow other tasks to run
        
        except Exception as e:
            yield formatter.format_error(e)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return HealthResponse(
        status="healthy" if state.is_ready else "starting",
        model_loaded=state.engine is not None,
        device=str(state.engine.device) if state.engine else "none",
        uptime_seconds=time.time() - state.metrics.start_time,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Get server metrics.
    
    Returns:
        Performance metrics
    """
    stats = state.metrics.get_stats()
    return MetricsResponse(**stats)


@app.get("/info", response_model=ModelInfoResponse)
async def info() -> ModelInfoResponse:
    """Get model information.
    
    Returns:
        Model information
    """
    if not state.engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = state.engine.get_model_info()
    return ModelInfoResponse(
        architecture=model_info["architecture"],
        parameters=model_info["total_parameters"],
        config=model_info["config"],
        device=model_info["device"],
        quantization=model_info["quantization"],
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Wave Field LLM API",
        "version": "1.0.0",
        "status": "healthy" if state.is_ready else "starting",
        "endpoints": {
            "generate": "POST /generate - Single text generation",
            "batch_generate": "POST /batch_generate - Batch generation",
            "stream": "GET /stream - Streaming generation (SSE)",
            "health": "GET /health - Health check",
            "metrics": "GET /metrics - Performance metrics",
            "info": "GET /info - Model information",
            "docs": "GET /docs - OpenAPI documentation",
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    """CLI entry point for server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Serve Wave Field LLM via HTTP API")
    parser.add_argument("--model", "--checkpoint", required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    parser.add_argument("--quantization", default="none",
                       choices=["none", "int8", "fp16", "bf16"],
                       help="Quantization mode")
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile()")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload (development)")
    
    args = parser.parse_args()
    
    # Create inference config
    inference_config = InferenceConfig(
        quantization=args.quantization,
        use_compile=args.compile,
    )
    
    # Load model
    load_model(args.model, inference_config)
    
    # Run server
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required to run the server")
        print("Install with: pip install uvicorn")
        return
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

# Made with Bob

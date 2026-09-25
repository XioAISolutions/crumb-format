#!/usr/bin/env python3
"""
CRUMB LLM HTTP API Server Example

Production-ready HTTP API server for serving CRUMB LLM models.

Features:
- REST API endpoints
- Streaming generation (SSE)
- Batch generation
- Health checks and metrics
- Automatic optimization (quantization, compilation)

Usage:
    python examples/serve_api.py --model model.pt --port 8000
    
Then access:
    http://localhost:8000/docs - API documentation
    http://localhost:8000/health - Health check
    http://localhost:8000/metrics - Performance metrics
"""

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import time

from crumb_llm.inference import InferenceEngine, InferenceConfig
from crumb_llm.generate import GenerationConfig

# Request/Response models
class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    stop_tokens: Optional[List[str]] = None

class GenerateResponse(BaseModel):
    """Response for text generation."""
    text: str
    prompt: str
    tokens_generated: int
    time_ms: float
    tokens_per_second: float

class BatchGenerateRequest(BaseModel):
    """Request for batch generation."""
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    quantization: str

class MetricsResponse(BaseModel):
    """Metrics response."""
    total_requests: int
    total_tokens_generated: int
    average_tokens_per_second: float
    uptime_seconds: float

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CRUMB LLM HTTP API Server')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    
    # Optimization
    parser.add_argument('--quantization', type=str, default='none',
                       choices=['none', 'fp16', 'int8'],
                       help='Quantization mode')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile()')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='Compilation mode')
    
    return parser.parse_args()

def create_app(engine: InferenceEngine) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="CRUMB LLM API",
        description="HTTP API for CRUMB LLM text generation",
        version="1.0.0",
    )
    
    # Metrics
    metrics = {
        'total_requests': 0,
        'total_tokens': 0,
        'start_time': time.time(),
    }
    
    @app.get("/", tags=["Info"])
    async def root():
        """Root endpoint."""
        return {
            "name": "CRUMB LLM API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(engine.device),
            quantization=engine.config.quantization,
        )
    
    @app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
    async def get_metrics():
        """Get performance metrics."""
        uptime = time.time() - metrics['start_time']
        avg_tps = metrics['total_tokens'] / uptime if uptime > 0 else 0
        
        return MetricsResponse(
            total_requests=metrics['total_requests'],
            total_tokens_generated=metrics['total_tokens'],
            average_tokens_per_second=avg_tps,
            uptime_seconds=uptime,
        )
    
    @app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
    async def generate(request: GenerateRequest):
        """Generate text from prompt."""
        try:
            start_time = time.time()
            
            # Generate
            text = engine.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            )
            
            # Calculate metrics
            elapsed = (time.time() - start_time) * 1000  # ms
            tokens_generated = len(text.split())  # Approximate
            tps = tokens_generated / (elapsed / 1000) if elapsed > 0 else 0
            
            # Update metrics
            metrics['total_requests'] += 1
            metrics['total_tokens'] += tokens_generated
            
            return GenerateResponse(
                text=text,
                prompt=request.prompt,
                tokens_generated=tokens_generated,
                time_ms=elapsed,
                tokens_per_second=tps,
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch_generate", tags=["Generation"])
    async def batch_generate(request: BatchGenerateRequest):
        """Generate text for multiple prompts."""
        try:
            start_time = time.time()
            
            # Generate for each prompt
            results = []
            for prompt in request.prompts:
                text = engine.generate(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
                results.append({
                    'prompt': prompt,
                    'text': text,
                })
            
            elapsed = (time.time() - start_time) * 1000
            
            # Update metrics
            metrics['total_requests'] += len(request.prompts)
            
            return {
                'results': results,
                'time_ms': elapsed,
                'prompts_processed': len(request.prompts),
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stream", tags=["Generation"])
    async def stream_generate(
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
    ):
        """Stream generated tokens (SSE)."""
        async def generate_stream():
            try:
                # Note: This is a simplified streaming implementation
                # For production, use crumb_llm.streaming.StreamingGenerator
                
                text = engine.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                # Stream tokens
                for i, char in enumerate(text):
                    event = {
                        'type': 'token',
                        'text': char,
                        'index': i,
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                
                # Done
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                error = {'type': 'error', 'message': str(e)}
                yield f"data: {json.dumps(error)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )
    
    return app

def main():
    """Main server function."""
    args = parse_args()
    
    print("🌊 CRUMB LLM HTTP API Server")
    print("=" * 70)
    
    # Configure inference
    print(f"\n⚙️  Configuring inference...")
    print(f"   Quantization: {args.quantization}")
    print(f"   Compilation: {args.compile}")
    if args.compile:
        print(f"   Compile mode: {args.compile_mode}")
    
    config = InferenceConfig(
        quantization=args.quantization if args.quantization != 'none' else None,
        use_compile=args.compile,
        compile_mode=args.compile_mode if args.compile else None,
    )
    
    # Load model
    print(f"\n📦 Loading model from {args.model}...")
    engine = InferenceEngine.from_checkpoint(args.model, config)
    print("✅ Model loaded and optimized")
    
    # Create app
    app = create_app(engine)
    
    # Start server
    print(f"\n🚀 Starting server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print(f"\n📖 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"💚 Health Check: http://{args.host}:{args.port}/health")
    print(f"📊 Metrics: http://{args.host}:{args.port}/metrics")
    print("\n" + "=" * 70)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )

if __name__ == '__main__':
    main()

# Made with Bob

"""Streaming generation utilities for Wave Field LLMs.

This module provides utilities for streaming token generation with:
- Token-by-token streaming
- Server-sent events (SSE) support
- WebSocket support
- Buffering strategies for smooth output
- Latency optimization
- Backpressure handling

Streaming is essential for interactive applications where users want to see
output as it's generated rather than waiting for the complete response.

Usage:
    from crumb_llm.streaming import StreamingGenerator, SSEFormatter
    
    # Basic streaming
    generator = StreamingGenerator(model, tokenizer)
    for token in generator.stream("Hello world"):
        print(token, end="", flush=True)
    
    # SSE streaming for web
    formatter = SSEFormatter()
    for event in formatter.format_stream(generator.stream("Hello")):
        # Send event to client
        yield event
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, AsyncIterator, Optional, Dict, Any, Callable, List

import torch
from torch import Tensor

from .model import WaveFieldLM
from .generate import Generator, GenerationConfig
from .cache import FieldStateCache, forward_cached_block


@dataclass
class StreamingConfig:
    """Configuration for streaming generation.
    
    Attributes:
        buffer_size: Number of tokens to buffer before yielding
        min_latency_ms: Target minimum latency between tokens
        max_latency_ms: Maximum acceptable latency between tokens
        yield_partial_words: Yield tokens even if they don't complete words
        include_metadata: Include timing and token metadata in stream
        stop_on_eos: Stop streaming when EOS token is generated
        timeout_seconds: Timeout for stream (None = no timeout)
    """
    buffer_size: int = 1
    min_latency_ms: float = 0.0
    max_latency_ms: float = 1000.0
    yield_partial_words: bool = True
    include_metadata: bool = False
    stop_on_eos: bool = True
    timeout_seconds: Optional[float] = None


@dataclass
class StreamToken:
    """A single token in the stream with metadata.
    
    Attributes:
        token_id: Token ID
        text: Decoded text (if tokenizer available)
        logprob: Log probability of token
        timestamp: Generation timestamp
        latency_ms: Time since last token
        is_eos: Whether this is an EOS token
    """
    token_id: int
    text: Optional[str] = None
    logprob: Optional[float] = None
    timestamp: Optional[float] = None
    latency_ms: Optional[float] = None
    is_eos: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "token_id": self.token_id,
            "text": self.text,
            "logprob": self.logprob,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "is_eos": self.is_eos,
        }


class TokenBuffer:
    """Buffer for smoothing token output.
    
    Accumulates tokens and yields them in batches to reduce jitter.
    """
    
    def __init__(self, size: int = 1):
        """Initialize buffer.
        
        Args:
            size: Buffer size (number of tokens to accumulate)
        """
        self.size = size
        self.buffer: deque[StreamToken] = deque(maxlen=size)
    
    def add(self, token: StreamToken) -> Optional[List[StreamToken]]:
        """Add token to buffer.
        
        Args:
            token: Token to add
            
        Returns:
            List of tokens to yield (if buffer is full), or None
        """
        self.buffer.append(token)
        
        if len(self.buffer) >= self.size or token.is_eos:
            # Flush buffer
            tokens = list(self.buffer)
            self.buffer.clear()
            return tokens
        
        return None
    
    def flush(self) -> List[StreamToken]:
        """Flush remaining tokens in buffer.
        
        Returns:
            All remaining tokens
        """
        tokens = list(self.buffer)
        self.buffer.clear()
        return tokens


class StreamingGenerator:
    """Generator that streams tokens as they're produced.
    
    Wraps the standard Generator to provide streaming capabilities.
    """
    
    def __init__(
        self,
        model: WaveFieldLM,
        tokenizer: Optional[object] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize streaming generator.
        
        Args:
            model: Wave Field LM model
            tokenizer: Tokenizer (optional)
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
    
    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        streaming_config: Optional[StreamingConfig] = None,
    ) -> Iterator[StreamToken]:
        """Stream tokens from generation.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            streaming_config: Streaming configuration
            
        Yields:
            StreamToken objects
        """
        config = config or GenerationConfig()
        streaming_config = streaming_config or StreamingConfig()
        
        # Encode prompt
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for streaming")
        
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        # Initialize cache
        cache = FieldStateCache.init(self.model, batch_size=1, device=self.device)
        
        # Prefill phase
        x = self.model.embed(input_ids)
        for layer_idx, block in enumerate(self.model.blocks):
            x = block(x)
        
        x = self.model.norm_out(x)
        if self.model.lm_head is None:
            logits = x @ self.model.embed.weight.t()
        else:
            logits = self.model.lm_head(x)
        
        last_logits = logits[:, -1, :]
        cache.seq_pos = input_ids.shape[1]
        
        # Buffer for smoothing
        buffer = TokenBuffer(streaming_config.buffer_size)
        
        # Timing
        start_time = time.time()
        last_token_time = start_time
        
        # Generation loop
        for step in range(config.max_new_tokens):
            # Check timeout
            if streaming_config.timeout_seconds is not None:
                if time.time() - start_time > streaming_config.timeout_seconds:
                    break
            
            # Sample next token
            next_token_id = self._sample_token(last_logits, config)
            
            # Get log probability
            logprob = None
            if streaming_config.include_metadata:
                probs = torch.softmax(last_logits / max(config.temperature, 1e-6), dim=-1)
                logprob = torch.log(probs[0, next_token_id]).item()
            
            # Decode token
            token_text = self.tokenizer.decode([next_token_id])
            
            # Create stream token
            current_time = time.time()
            stream_token = StreamToken(
                token_id=next_token_id,
                text=token_text,
                logprob=logprob,
                timestamp=current_time if streaming_config.include_metadata else None,
                latency_ms=(current_time - last_token_time) * 1000 if streaming_config.include_metadata else None,
                is_eos=(config.eos_token_id is not None and next_token_id == config.eos_token_id),
            )
            last_token_time = current_time
            
            # Add to buffer
            tokens_to_yield = buffer.add(stream_token)
            if tokens_to_yield:
                for token in tokens_to_yield:
                    yield token
            
            # Check stopping conditions
            if stream_token.is_eos and streaming_config.stop_on_eos:
                break
            
            # Decode step with cache
            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            x_tok = self.model.embed(next_token_tensor)
            total_tokens = cache.seq_pos + 1
            field_size = self.model.cfg.field_size
            
            for layer_idx, block in enumerate(self.model.blocks):
                x_tok, cache.layers[layer_idx] = forward_cached_block(
                    block, x_tok, cache.layers[layer_idx],
                    token_pos=cache.seq_pos % field_size,
                    total_tokens=min(total_tokens, field_size),
                )
            
            x_tok = self.model.norm_out(x_tok)
            if self.model.lm_head is None:
                last_logits = (x_tok @ self.model.embed.weight.t()).squeeze(1)
            else:
                last_logits = self.model.lm_head(x_tok).squeeze(1)
            
            cache.seq_pos += 1
        
        # Flush remaining tokens
        remaining = buffer.flush()
        for token in remaining:
            yield token
    
    def _sample_token(self, logits: Tensor, config: GenerationConfig) -> int:
        """Sample next token from logits.
        
        Args:
            logits: [1, vocab_size] logits
            config: Generation configuration
            
        Returns:
            Token ID
        """
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / max(config.temperature, 1e-6)
        
        if not config.do_sample:
            return logits.argmax(dim=-1).item()
        
        # Top-k filtering
        if config.top_k is not None and config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Top-p filtering
        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.item()


class SSEFormatter:
    """Format streaming tokens as Server-Sent Events.
    
    SSE is a standard for streaming data from server to client over HTTP.
    """
    
    def __init__(self, event_type: str = "token"):
        """Initialize SSE formatter.
        
        Args:
            event_type: Event type name
        """
        self.event_type = event_type
    
    def format_token(self, token: StreamToken) -> str:
        """Format a single token as SSE event.
        
        Args:
            token: Token to format
            
        Returns:
            SSE-formatted string
        """
        data = token.to_dict()
        return f"event: {self.event_type}\ndata: {json.dumps(data)}\n\n"
    
    def format_stream(
        self,
        token_stream: Iterator[StreamToken],
    ) -> Iterator[str]:
        """Format a stream of tokens as SSE events.
        
        Args:
            token_stream: Iterator of tokens
            
        Yields:
            SSE-formatted strings
        """
        for token in token_stream:
            yield self.format_token(token)
        
        # Send completion event
        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
    
    def format_error(self, error: Exception) -> str:
        """Format an error as SSE event.
        
        Args:
            error: Exception to format
            
        Returns:
            SSE-formatted error string
        """
        data = {
            "error": str(error),
            "type": type(error).__name__,
        }
        return f"event: error\ndata: {json.dumps(data)}\n\n"


class WebSocketStreamer:
    """Stream tokens over WebSocket connection.
    
    Provides async streaming for WebSocket-based applications.
    """
    
    def __init__(self, generator: StreamingGenerator):
        """Initialize WebSocket streamer.
        
        Args:
            generator: Streaming generator
        """
        self.generator = generator
    
    async def stream_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        streaming_config: Optional[StreamingConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Async stream tokens.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            streaming_config: Streaming configuration
            
        Yields:
            Token dictionaries
        """
        # Run synchronous generator in executor
        loop = asyncio.get_event_loop()
        
        def sync_generator():
            return self.generator.stream(prompt, config, streaming_config)
        
        # Get the generator
        token_iter = await loop.run_in_executor(None, sync_generator)
        
        # Yield tokens asynchronously
        for token in token_iter:
            yield token.to_dict()
            # Allow other tasks to run
            await asyncio.sleep(0)


class ChunkedStreamer:
    """Stream tokens in chunks for better network efficiency.
    
    Accumulates multiple tokens before sending to reduce overhead.
    """
    
    def __init__(
        self,
        generator: StreamingGenerator,
        chunk_size: int = 5,
        chunk_timeout_ms: float = 100.0,
    ):
        """Initialize chunked streamer.
        
        Args:
            generator: Streaming generator
            chunk_size: Number of tokens per chunk
            chunk_timeout_ms: Max time to wait for chunk to fill
        """
        self.generator = generator
        self.chunk_size = chunk_size
        self.chunk_timeout_ms = chunk_timeout_ms
    
    def stream_chunks(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        streaming_config: Optional[StreamingConfig] = None,
    ) -> Iterator[List[StreamToken]]:
        """Stream tokens in chunks.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            streaming_config: Streaming configuration
            
        Yields:
            Lists of tokens (chunks)
        """
        chunk = []
        chunk_start_time = time.time()
        
        for token in self.generator.stream(prompt, config, streaming_config):
            chunk.append(token)
            
            # Yield chunk if full or timeout
            current_time = time.time()
            elapsed_ms = (current_time - chunk_start_time) * 1000
            
            if len(chunk) >= self.chunk_size or elapsed_ms >= self.chunk_timeout_ms or token.is_eos:
                yield chunk
                chunk = []
                chunk_start_time = current_time
        
        # Yield remaining tokens
        if chunk:
            yield chunk


def stream_to_string(
    token_stream: Iterator[StreamToken],
    include_metadata: bool = False,
) -> str:
    """Collect streaming tokens into a single string.
    
    Args:
        token_stream: Iterator of tokens
        include_metadata: Whether to include metadata in output
        
    Returns:
        Complete generated text
    """
    if include_metadata:
        tokens = list(token_stream)
        text = "".join(t.text or "" for t in tokens)
        metadata = {
            "total_tokens": len(tokens),
            "total_time_ms": tokens[-1].timestamp - tokens[0].timestamp if tokens else 0,
            "avg_latency_ms": sum(t.latency_ms or 0 for t in tokens) / len(tokens) if tokens else 0,
        }
        return json.dumps({"text": text, "metadata": metadata})
    else:
        return "".join(t.text or "" for t in token_stream)


def measure_streaming_latency(
    generator: StreamingGenerator,
    prompt: str,
    config: Optional[GenerationConfig] = None,
) -> Dict[str, float]:
    """Measure streaming latency metrics.
    
    Args:
        generator: Streaming generator
        prompt: Test prompt
        config: Generation configuration
        
    Returns:
        Dictionary with latency metrics
    """
    streaming_config = StreamingConfig(include_metadata=True)
    
    tokens = list(generator.stream(prompt, config, streaming_config))
    
    if not tokens:
        return {}
    
    latencies = [t.latency_ms for t in tokens if t.latency_ms is not None]
    
    return {
        "first_token_latency_ms": tokens[0].latency_ms or 0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "min_latency_ms": min(latencies) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "total_tokens": len(tokens),
        "total_time_ms": (tokens[-1].timestamp - tokens[0].timestamp) * 1000 if tokens[0].timestamp else 0,
    }


def main():
    """CLI for testing streaming."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test streaming generation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--prompt", default="Once upon a time", help="Test prompt")
    parser.add_argument("--format", default="text", choices=["text", "sse", "json"],
                       help="Output format")
    parser.add_argument("--buffer-size", type=int, default=1, help="Buffer size")
    parser.add_argument("--measure-latency", action="store_true",
                       help="Measure and report latency")
    
    args = parser.parse_args()
    
    # Load model
    from .sample import load_checkpoint
    model, tokenizer = load_checkpoint(args.checkpoint)
    
    # Create generator
    generator = StreamingGenerator(model, tokenizer)
    
    # Configure streaming
    streaming_config = StreamingConfig(
        buffer_size=args.buffer_size,
        include_metadata=args.measure_latency,
    )
    
    config = GenerationConfig(max_new_tokens=100)
    
    if args.measure_latency:
        # Measure latency
        metrics = measure_streaming_latency(generator, args.prompt, config)
        print("Streaming Latency Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
    
    elif args.format == "text":
        # Plain text streaming
        print("", end="", flush=True)
        for token in generator.stream(args.prompt, config, streaming_config):
            print(token.text or "", end="", flush=True)
        print()
    
    elif args.format == "sse":
        # SSE format
        formatter = SSEFormatter()
        for event in formatter.format_stream(
            generator.stream(args.prompt, config, streaming_config)
        ):
            print(event, end="")
    
    elif args.format == "json":
        # JSON format
        tokens = list(generator.stream(args.prompt, config, streaming_config))
        output = {
            "tokens": [t.to_dict() for t in tokens],
            "text": "".join(t.text or "" for t in tokens),
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

# Made with Bob

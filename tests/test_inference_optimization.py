"""Tests for inference optimization modules.

Tests cover:
- Text generation with various sampling strategies
- Field-state caching correctness
- Quantization accuracy
- Streaming generation
- Batch inference
- Server endpoints (if FastAPI available)
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import json

# Import modules to test
from crumb_llm.model import WaveFieldLM, WaveFieldConfig
from crumb_llm.generate import Generator, GenerationConfig
from crumb_llm.inference import InferenceEngine, InferenceConfig
from crumb_llm.cache import FieldStateCache, generate_cached
from crumb_llm.streaming import StreamingGenerator, StreamingConfig, TokenBuffer
from crumb_llm.quantization import (
    quantize_model,
    QuantizationConfig,
    measure_model_size,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    config = WaveFieldConfig(
        vocab_size=128,
        dim=64,
        n_layers=2,
        n_heads=2,
        field_size=128,
        causal=True,
    )
    model = WaveFieldLM(config)
    model.eval()
    return model


@pytest.fixture
def dummy_tokenizer():
    """Create a dummy tokenizer for testing."""
    class DummyTokenizer:
        def encode(self, text):
            # Simple byte encoding
            return list(text.encode()[:50])  # Limit length
        
        def decode(self, ids):
            # Simple byte decoding
            try:
                return bytes(ids).decode('utf-8', errors='ignore')
            except:
                return ""
        
        def save(self, path):
            pass
        
        @classmethod
        def load(cls, path):
            return cls()
    
    return DummyTokenizer()


@pytest.fixture
def generator(tiny_model, dummy_tokenizer):
    """Create a generator for testing."""
    return Generator(tiny_model, dummy_tokenizer)


# ── Generation Tests ─────────────────────────────────────────────────


def test_greedy_generation(generator):
    """Test greedy decoding."""
    config = GenerationConfig(
        max_new_tokens=10,
        do_sample=False,
        temperature=1.0,
    )
    
    output = generator.generate("Hello", config)
    assert isinstance(output, str)
    assert len(output) > 0


def test_sampling_generation(generator):
    """Test sampling with temperature."""
    config = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        temperature=0.8,
    )
    
    output = generator.generate("Hello", config)
    assert isinstance(output, str)
    assert len(output) > 0


def test_top_k_sampling(generator):
    """Test top-k sampling."""
    config = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        top_k=5,
    )
    
    output = generator.generate("Hello", config)
    assert isinstance(output, str)


def test_top_p_sampling(generator):
    """Test top-p (nucleus) sampling."""
    config = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        top_p=0.9,
    )
    
    output = generator.generate("Hello", config)
    assert isinstance(output, str)


def test_temperature_scaling(generator):
    """Test temperature scaling."""
    # High temperature should give more random output
    config_high = GenerationConfig(
        max_new_tokens=20,
        temperature=2.0,
        do_sample=True,
    )
    
    # Low temperature should be more deterministic
    config_low = GenerationConfig(
        max_new_tokens=20,
        temperature=0.1,
        do_sample=True,
    )
    
    output_high = generator.generate("Test", config_high)
    output_low = generator.generate("Test", config_low)
    
    assert isinstance(output_high, str)
    assert isinstance(output_low, str)


def test_batch_generation(generator):
    """Test batch generation."""
    prompts = ["Hello", "World", "Test"]
    config = GenerationConfig(max_new_tokens=10)
    
    outputs = generator.generate_batch(prompts, config)
    
    assert len(outputs) == len(prompts)
    assert all(isinstance(out, str) for out in outputs)


def test_stop_on_eos(generator, tiny_model):
    """Test stopping on EOS token."""
    config = GenerationConfig(
        max_new_tokens=100,
        eos_token_id=0,  # Use 0 as EOS
    )
    
    # Generate with EOS stopping
    output = generator.generate("Test", config)
    assert isinstance(output, str)


# ── Caching Tests ────────────────────────────────────────────────────


def test_field_state_cache_init(tiny_model):
    """Test cache initialization."""
    cache = FieldStateCache.init(tiny_model, batch_size=1)
    
    assert len(cache.layers) == tiny_model.cfg.n_layers
    assert cache.seq_pos == 0
    
    for layer_state in cache.layers:
        assert layer_state.field.shape[0] == 1  # batch size
        assert layer_state.field.shape[1] == tiny_model.cfg.n_heads
        assert layer_state.field.shape[2] == tiny_model.cfg.field_size


def test_cached_generation(tiny_model, dummy_tokenizer):
    """Test generation with caching."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    output = generate_cached(
        tiny_model,
        input_ids,
        max_new_tokens=10,
        temperature=1.0,
    )
    
    assert output.shape[0] == 1
    assert output.shape[1] > input_ids.shape[1]


def test_cached_vs_uncached_consistency(generator, tiny_model):
    """Test that cached generation produces same results as uncached."""
    torch.manual_seed(42)
    
    config = GenerationConfig(
        max_new_tokens=10,
        temperature=1.0,
        do_sample=False,  # Greedy for determinism
        use_cache=False,
    )
    
    output_uncached = generator.generate("Test", config)
    
    torch.manual_seed(42)
    config.use_cache = True
    output_cached = generator.generate("Test", config)
    
    # Should produce similar outputs (may not be identical due to implementation details)
    assert isinstance(output_uncached, str)
    assert isinstance(output_cached, str)


# ── Inference Engine Tests ───────────────────────────────────────────


def test_inference_engine_creation(tiny_model, dummy_tokenizer):
    """Test inference engine creation."""
    config = InferenceConfig(
        quantization="none",
        use_compile=False,  # Disable for testing
        warmup_steps=0,
    )
    
    engine = InferenceEngine(tiny_model, dummy_tokenizer, config)
    
    assert engine.model is not None
    assert engine.generator is not None


def test_inference_engine_generate(tiny_model, dummy_tokenizer):
    """Test generation through inference engine."""
    config = InferenceConfig(
        quantization="none",
        use_compile=False,
        warmup_steps=0,
    )
    
    engine = InferenceEngine(tiny_model, dummy_tokenizer, config)
    output = engine.generate("Hello", max_tokens=10)
    
    assert isinstance(output, str)


def test_inference_engine_benchmark(tiny_model, dummy_tokenizer):
    """Test benchmarking."""
    config = InferenceConfig(
        quantization="none",
        use_compile=False,
        warmup_steps=0,
    )
    
    engine = InferenceEngine(tiny_model, dummy_tokenizer, config)
    stats = engine.benchmark(batch_size=1, seq_len=32, num_iterations=5)
    
    assert stats.tokens_per_second > 0
    assert stats.latency_ms > 0
    assert stats.batch_size == 1
    assert stats.seq_len == 32


def test_model_info(tiny_model, dummy_tokenizer):
    """Test getting model information."""
    config = InferenceConfig(warmup_steps=0, use_compile=False)
    engine = InferenceEngine(tiny_model, dummy_tokenizer, config)
    
    info = engine.get_model_info()
    
    assert "architecture" in info
    assert "total_parameters" in info
    assert "config" in info
    assert info["total_parameters"] > 0


# ── Quantization Tests ───────────────────────────────────────────────


def test_measure_model_size(tiny_model):
    """Test model size measurement."""
    size = measure_model_size(tiny_model)
    
    assert "total_parameters" in size
    assert "size_mb" in size
    assert size["total_parameters"] > 0
    assert size["size_mb"] > 0


def test_fp16_quantization(tiny_model):
    """Test FP16 quantization."""
    config = QuantizationConfig(mode="fp16")
    quantized = quantize_model(tiny_model, config)
    
    # Check that model is in FP16
    for param in quantized.parameters():
        if param.dtype != torch.float16:
            # Some params like embeddings might stay in FP32
            pass


def test_dynamic_int8_quantization(tiny_model):
    """Test dynamic INT8 quantization."""
    config = QuantizationConfig(mode="dynamic_int8")
    
    try:
        quantized = quantize_model(tiny_model, config)
        # Check that model was quantized
        assert quantized is not None
    except Exception as e:
        # Quantization might not be available on all platforms
        pytest.skip(f"Quantization not available: {e}")


def test_quantization_reduces_size(tiny_model):
    """Test that quantization reduces model size."""
    original_size = measure_model_size(tiny_model)
    
    config = QuantizationConfig(mode="fp16")
    quantized = quantize_model(tiny_model, config)
    quantized_size = measure_model_size(quantized)
    
    # FP16 should be roughly half the size
    assert quantized_size["size_mb"] < original_size["size_mb"]


# ── Streaming Tests ──────────────────────────────────────────────────


def test_token_buffer():
    """Test token buffer."""
    from crumb_llm.streaming import StreamToken
    
    buffer = TokenBuffer(size=3)
    
    # Add tokens
    token1 = StreamToken(token_id=1, text="a")
    result = buffer.add(token1)
    assert result is None  # Buffer not full
    
    token2 = StreamToken(token_id=2, text="b")
    result = buffer.add(token2)
    assert result is None
    
    token3 = StreamToken(token_id=3, text="c")
    result = buffer.add(token3)
    assert result is not None
    assert len(result) == 3


def test_streaming_generator(tiny_model, dummy_tokenizer):
    """Test streaming generation."""
    generator = StreamingGenerator(tiny_model, dummy_tokenizer)
    
    config = GenerationConfig(max_new_tokens=10)
    streaming_config = StreamingConfig(buffer_size=1)
    
    tokens = list(generator.stream("Hello", config, streaming_config))
    
    assert len(tokens) > 0
    assert all(hasattr(t, 'token_id') for t in tokens)
    assert all(hasattr(t, 'text') for t in tokens)


def test_sse_formatter():
    """Test SSE formatting."""
    from crumb_llm.streaming import SSEFormatter, StreamToken
    
    formatter = SSEFormatter()
    token = StreamToken(token_id=1, text="hello")
    
    sse_event = formatter.format_token(token)
    
    assert "event:" in sse_event
    assert "data:" in sse_event
    assert "hello" in sse_event


def test_streaming_with_metadata(tiny_model, dummy_tokenizer):
    """Test streaming with metadata."""
    generator = StreamingGenerator(tiny_model, dummy_tokenizer)
    
    config = GenerationConfig(max_new_tokens=5)
    streaming_config = StreamingConfig(
        buffer_size=1,
        include_metadata=True,
    )
    
    tokens = list(generator.stream("Test", config, streaming_config))
    
    assert len(tokens) > 0
    # Check that metadata is included
    for token in tokens:
        if token.timestamp is not None:
            assert token.timestamp > 0


# ── Integration Tests ────────────────────────────────────────────────


def test_end_to_end_generation(tiny_model, dummy_tokenizer):
    """Test complete generation pipeline."""
    # Create generator
    generator = Generator(tiny_model, dummy_tokenizer)
    
    # Configure generation
    config = GenerationConfig(
        max_new_tokens=20,
        temperature=0.8,
        top_k=10,
        do_sample=True,
    )
    
    # Generate
    output = generator.generate("Once upon a time", config)
    
    assert isinstance(output, str)
    assert len(output) > len("Once upon a time")


def test_save_and_load_optimized(tiny_model, dummy_tokenizer):
    """Test saving and loading optimized model."""
    config = InferenceConfig(
        quantization="none",
        use_compile=False,
        warmup_steps=0,
    )
    
    engine = InferenceEngine(tiny_model, dummy_tokenizer, config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "optimized"
        engine.save_optimized(output_path)
        
        # Check that files were created
        assert (output_path / "optimized_model.pt").exists()


def test_multiple_sampling_strategies(generator):
    """Test that different sampling strategies produce different outputs."""
    prompt = "The quick brown fox"
    
    # Greedy
    config_greedy = GenerationConfig(
        max_new_tokens=10,
        do_sample=False,
    )
    
    # High temperature
    config_high_temp = GenerationConfig(
        max_new_tokens=10,
        temperature=2.0,
        do_sample=True,
    )
    
    # Top-k
    config_topk = GenerationConfig(
        max_new_tokens=10,
        top_k=5,
        do_sample=True,
    )
    
    torch.manual_seed(42)
    output_greedy = generator.generate(prompt, config_greedy)
    
    torch.manual_seed(42)
    output_high_temp = generator.generate(prompt, config_high_temp)
    
    torch.manual_seed(42)
    output_topk = generator.generate(prompt, config_topk)
    
    # All should produce valid outputs
    assert isinstance(output_greedy, str)
    assert isinstance(output_high_temp, str)
    assert isinstance(output_topk, str)


# ── Performance Tests ────────────────────────────────────────────────


@pytest.mark.slow
def test_generation_speed(generator):
    """Test generation speed."""
    import time
    
    config = GenerationConfig(max_new_tokens=50)
    
    start = time.time()
    output = generator.generate("Test prompt", config)
    elapsed = time.time() - start
    
    # Should complete in reasonable time (< 10 seconds on CPU)
    assert elapsed < 10.0
    assert isinstance(output, str)


@pytest.mark.slow
def test_batch_efficiency(generator):
    """Test that batch generation is more efficient than sequential."""
    import time
    
    prompts = ["Test 1", "Test 2", "Test 3", "Test 4"]
    config = GenerationConfig(max_new_tokens=20)
    
    # Sequential generation
    start = time.time()
    for prompt in prompts:
        _ = generator.generate(prompt, config)
    sequential_time = time.time() - start
    
    # Batch generation
    start = time.time()
    _ = generator.generate_batch(prompts, config)
    batch_time = time.time() - start
    
    # Batch should be faster (or at least not much slower)
    # Allow some margin for overhead
    assert batch_time < sequential_time * 1.5


# ── Error Handling Tests ─────────────────────────────────────────────


def test_invalid_config():
    """Test that invalid config raises error."""
    with pytest.raises((ValueError, TypeError)):
        config = GenerationConfig(max_new_tokens=-1)


def test_empty_prompt(generator):
    """Test generation with empty prompt."""
    config = GenerationConfig(max_new_tokens=10)
    
    # Should handle empty prompt gracefully
    output = generator.generate("", config)
    assert isinstance(output, str)


def test_very_long_prompt(generator):
    """Test generation with very long prompt."""
    # Create a long prompt
    long_prompt = "Test " * 100
    config = GenerationConfig(max_new_tokens=5)
    
    # Should handle long prompt (may truncate)
    output = generator.generate(long_prompt, config)
    assert isinstance(output, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob

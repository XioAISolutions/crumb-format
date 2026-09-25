"""Comprehensive benchmarking suite for Wave Field LLM.

This package provides tools for benchmarking the Wave Field LLM against
transformer baselines across multiple dimensions:

- Perplexity on various datasets
- Speed and throughput
- Memory usage
- Generation quality
- Ablation studies
- CRUMB-specific features

Usage:
    # Run full benchmark suite
    python -m benchmarks.benchmark_suite --model checkpoints/model.pt --output results/
    
    # Run specific benchmarks
    python -m benchmarks.perplexity_bench --model checkpoints/model.pt
    python -m benchmarks.speed_bench --model checkpoints/model.pt
    python -m benchmarks.memory_bench --model checkmarks/model.pt
"""

from .utils import (
    BenchmarkResult,
    BenchmarkConfig,
    set_seed,
    format_number,
    save_results,
    load_results,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkConfig",
    "set_seed",
    "format_number",
    "save_results",
    "load_results",
]

# Made with Bob

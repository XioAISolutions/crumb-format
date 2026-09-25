"""Memory usage benchmarking for Wave Field LLM.

Measures:
- Peak memory usage (training and inference)
- Memory per token
- Cache size (field-state vs KV-cache)
- Gradient memory

Tests memory scaling with sequence length and demonstrates O(F) vs O(N) scaling.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crumb_llm.model import WaveFieldLM, WaveFieldConfig
from benchmarks.utils import (
    BenchmarkResult,
    BenchmarkConfig,
    set_seed,
    save_results,
    format_memory,
    format_number,
    create_markdown_table,
    compute_memory_reduction,
)
from benchmarks.baseline_models import create_baseline_model


def reset_peak_memory() -> None:
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def get_peak_memory() -> float:
    """Get peak memory usage in bytes.
    
    Returns:
        Peak memory in bytes
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0.0


def get_current_memory() -> float:
    """Get current memory usage in bytes.
    
    Returns:
        Current memory in bytes
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0.0


@torch.no_grad()
def benchmark_inference_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: str
) -> Tuple[float, Dict[str, float]]:
    """Benchmark inference memory usage.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to use
        
    Returns:
        Tuple of (peak_memory_bytes, metrics_dict)
    """
    if device != "cuda":
        return 0.0, {"peak_mb": 0.0, "per_token_kb": 0.0}
    
    model.eval()
    model = model.to(device)
    vocab_size = model.cfg.vocab_size if hasattr(model, 'cfg') else 256
    
    # Reset memory stats
    reset_peak_memory()
    
    # Measure baseline (model parameters)
    baseline_memory = get_current_memory()
    
    # Run inference
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    _ = model(x)
    
    torch.cuda.synchronize()
    
    # Get peak memory
    peak_memory = get_peak_memory()
    activation_memory = peak_memory - baseline_memory
    
    # Compute metrics
    peak_mb = peak_memory / (1024 ** 2)
    activation_mb = activation_memory / (1024 ** 2)
    per_token_kb = activation_memory / (batch_size * seq_len) / 1024
    
    metrics = {
        "peak_mb": peak_mb,
        "activation_mb": activation_mb,
        "per_token_kb": per_token_kb,
        "baseline_mb": baseline_memory / (1024 ** 2),
    }
    
    return peak_memory, metrics


def benchmark_training_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: str
) -> Tuple[float, Dict[str, float]]:
    """Benchmark training memory usage (forward + backward).
    
    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to use
        
    Returns:
        Tuple of (peak_memory_bytes, metrics_dict)
    """
    if device != "cuda":
        return 0.0, {"peak_mb": 0.0, "per_token_kb": 0.0}
    
    model.train()
    model = model.to(device)
    vocab_size = model.cfg.vocab_size if hasattr(model, 'cfg') else 256
    
    # Reset memory stats
    reset_peak_memory()
    
    # Measure baseline
    baseline_memory = get_current_memory()
    
    # Run forward + backward
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    outputs = model(x, targets=x)
    loss = outputs["loss"]
    loss.backward()
    
    torch.cuda.synchronize()
    
    # Get peak memory
    peak_memory = get_peak_memory()
    training_memory = peak_memory - baseline_memory
    
    # Compute metrics
    peak_mb = peak_memory / (1024 ** 2)
    training_mb = training_memory / (1024 ** 2)
    per_token_kb = training_memory / (batch_size * seq_len) / 1024
    
    metrics = {
        "peak_mb": peak_mb,
        "training_mb": training_mb,
        "per_token_kb": per_token_kb,
        "baseline_mb": baseline_memory / (1024 ** 2),
    }
    
    # Clean up
    model.zero_grad()
    
    return peak_memory, metrics


def estimate_cache_size(
    model: nn.Module,
    seq_len: int,
    batch_size: int = 1
) -> Dict[str, float]:
    """Estimate cache size for the model.
    
    Args:
        model: Model to analyze
        seq_len: Sequence length
        batch_size: Batch size
        
    Returns:
        Dictionary with cache size estimates
    """
    if hasattr(model, 'cfg'):
        cfg = model.cfg
        
        if hasattr(cfg, 'field_size'):
            # Wave Field LLM - O(F) cache
            field_size = cfg.field_size
            n_heads = cfg.n_heads
            dim = cfg.dim
            d_head = dim // n_heads
            
            # Field state cache per layer
            field_cache_per_layer = batch_size * n_heads * field_size * d_head * 4  # 4 bytes per float32
            total_field_cache = field_cache_per_layer * cfg.n_layers
            
            return {
                "type": "field_state",
                "cache_mb": total_field_cache / (1024 ** 2),
                "per_layer_mb": field_cache_per_layer / (1024 ** 2),
                "scaling": "O(F)",
                "field_size": field_size,
            }
        else:
            # Transformer - O(N) KV cache
            n_heads = cfg.n_heads
            dim = cfg.dim
            d_head = dim // n_heads
            
            # KV cache per layer (2 for K and V)
            kv_cache_per_layer = 2 * batch_size * n_heads * seq_len * d_head * 4
            total_kv_cache = kv_cache_per_layer * cfg.n_layers
            
            return {
                "type": "kv_cache",
                "cache_mb": total_kv_cache / (1024 ** 2),
                "per_layer_mb": kv_cache_per_layer / (1024 ** 2),
                "scaling": "O(N)",
                "seq_len": seq_len,
            }
    
    return {"type": "unknown", "cache_mb": 0.0}


def benchmark_memory_sweep(
    model: nn.Module,
    model_name: str,
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Benchmark memory across different configurations.
    
    Args:
        model: Model to benchmark
        model_name: Model name for results
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Memory Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    if config.device != "cuda":
        print("Warning: Memory benchmarking requires CUDA")
        return results
    
    # Benchmark 1: Inference memory across sequence lengths
    print("\n1. Inference Memory (varying sequence length)")
    print("-" * 60)
    
    for seq_len in config.sequence_lengths:
        print(f"\nSequence length: {seq_len}")
        
        try:
            peak_memory, metrics = benchmark_inference_memory(
                model,
                config.batch_size,
                seq_len,
                config.device
            )
            
            print(f"  Peak memory: {format_memory(peak_memory)}")
            print(f"  Activation memory: {format_memory(metrics['activation_mb'] * 1024**2)}")
            print(f"  Per token: {metrics['per_token_kb']:.2f} KB")
            
            result = BenchmarkResult(
                name=f"inference_memory_seq{seq_len}",
                model_name=model_name,
                metric="peak_memory",
                value=peak_memory,
                unit="bytes",
                metadata={
                    "seq_len": seq_len,
                    "batch_size": config.batch_size,
                    **metrics
                }
            )
            results.append(result)
            
        except RuntimeError as e:
            print(f"  Error: {e}")
            continue
    
    # Benchmark 2: Training memory
    print("\n2. Training Memory (forward + backward)")
    print("-" * 60)
    
    for seq_len in config.sequence_lengths[:3]:  # Test first 3 lengths
        print(f"\nSequence length: {seq_len}")
        
        try:
            peak_memory, metrics = benchmark_training_memory(
                model,
                config.batch_size,
                seq_len,
                config.device
            )
            
            print(f"  Peak memory: {format_memory(peak_memory)}")
            print(f"  Training memory: {format_memory(metrics['training_mb'] * 1024**2)}")
            print(f"  Per token: {metrics['per_token_kb']:.2f} KB")
            
            result = BenchmarkResult(
                name=f"training_memory_seq{seq_len}",
                model_name=model_name,
                metric="peak_memory",
                value=peak_memory,
                unit="bytes",
                metadata={
                    "seq_len": seq_len,
                    "batch_size": config.batch_size,
                    "mode": "training",
                    **metrics
                }
            )
            results.append(result)
            
        except RuntimeError as e:
            print(f"  Error: {e}")
            continue
    
    # Benchmark 3: Cache size analysis
    print("\n3. Cache Size Analysis")
    print("-" * 60)
    
    for seq_len in config.sequence_lengths:
        cache_info = estimate_cache_size(model, seq_len, config.batch_size)
        
        print(f"\nSequence length: {seq_len}")
        print(f"  Cache type: {cache_info['type']}")
        print(f"  Cache size: {cache_info['cache_mb']:.2f} MB")
        print(f"  Scaling: {cache_info.get('scaling', 'N/A')}")
        
        result = BenchmarkResult(
            name=f"cache_size_seq{seq_len}",
            model_name=model_name,
            metric="cache_size",
            value=cache_info['cache_mb'] * 1024**2,
            unit="bytes",
            metadata={
                "seq_len": seq_len,
                **cache_info
            }
        )
        results.append(result)
    
    return results


def create_memory_comparison_table(results: List[BenchmarkResult]) -> str:
    """Create markdown comparison table for memory results.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Markdown table string
    """
    # Group by inference memory
    inference_results = [r for r in results if "inference_memory" in r.name]
    
    if not inference_results:
        return "No inference memory results available."
    
    # Get unique models and sequence lengths
    models = sorted(set(r.model_name for r in inference_results))
    seq_lens = sorted(set(r.metadata["seq_len"] for r in inference_results))
    
    # Create table
    headers = ["Seq Length"] + models + ["Reduction"]
    rows = []
    
    for seq_len in seq_lens:
        row = [str(seq_len)]
        
        # Get baseline (first model)
        baseline_result = next(
            (r for r in inference_results 
             if r.metadata["seq_len"] == seq_len and r.model_name == models[0]),
            None
        )
        baseline_value = baseline_result.value if baseline_result else 0
        
        for model in models:
            result = next(
                (r for r in inference_results 
                 if r.metadata["seq_len"] == seq_len and r.model_name == model),
                None
            )
            if result:
                row.append(format_memory(result.value))
            else:
                row.append("N/A")
        
        # Compute memory reduction (wavefield vs baseline)
        if len(models) > 1 and baseline_value > 0:
            wavefield_result = next(
                (r for r in inference_results 
                 if r.metadata["seq_len"] == seq_len and "wavefield" in r.model_name.lower()),
                None
            )
            if wavefield_result:
                reduction = compute_memory_reduction(baseline_value, wavefield_result.value)
                row.append(f"{reduction:.1f}%")
            else:
                row.append("N/A")
        else:
            row.append("N/A")
        
        rows.append(row)
    
    return create_markdown_table(headers, rows, alignments=["left"] + ["right"] * len(headers[1:]))


def main():
    parser = argparse.ArgumentParser(description="Memory benchmarking")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="wavefield",
                       choices=["wavefield", "gpt2", "llama", "mistral"],
                       help="Model type")
    parser.add_argument("--baselines", type=str, default="",
                       help="Comma-separated list of baseline models")
    parser.add_argument("--sequence-lengths", type=str, default="512,1024,2048,4096,8192",
                       help="Comma-separated list of sequence lengths")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--output", type=str, default="memory_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: Memory benchmarking requires CUDA")
        return
    
    # Set seed
    set_seed(args.seed)
    
    # Parse sequence lengths
    seq_lens = [int(x.strip()) for x in args.sequence_lengths.split(",")]
    
    # Create config
    config = BenchmarkConfig(
        model_path=args.model,
        model_type=args.model_type,
        seed=args.seed,
        device="cuda",
        batch_size=args.batch_size,
        sequence_lengths=seq_lens,
        output_dir=args.output,
    )
    
    # Create models
    models = {}
    
    # Main model
    if args.model and Path(args.model).exists():
        checkpoint = torch.load(args.model, map_location="cpu")
        model_cfg = WaveFieldConfig(**checkpoint.get("config", {}))
        model = WaveFieldLM(model_cfg)
        model.load_state_dict(checkpoint["model"])
        models[args.model_type] = model
    else:
        model_cfg = WaveFieldConfig(dim=128, n_layers=4, n_heads=4)
        models[args.model_type] = WaveFieldLM(model_cfg)
    
    # Baseline models
    if args.baselines:
        baseline_types = [b.strip() for b in args.baselines.split(",")]
        for baseline_type in baseline_types:
            if baseline_type:
                baseline = create_baseline_model(
                    baseline_type,
                    vocab_size=256,
                    dim=128,
                    n_layers=4,
                    n_heads=4
                )
                models[baseline_type] = baseline
    
    # Run benchmarks
    print(f"\n{'='*60}")
    print(f"Memory Benchmarking")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: CUDA")
    
    all_results = []
    for model_name, model in models.items():
        results = benchmark_memory_sweep(model, model_name, config)
        all_results.extend(results)
    
    # Save results
    output_dir = Path(args.output)
    save_results(all_results, output_dir, "memory_results")
    
    # Create comparison table
    table = create_memory_comparison_table(all_results)
    print(f"\n{'='*60}")
    print("Memory Comparison")
    print(f"{'='*60}\n")
    print(table)
    
    # Save table
    with open(output_dir / "memory_comparison.md", "w") as f:
        f.write("# Memory Comparison\n\n")
        f.write(table)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

# Made with Bob

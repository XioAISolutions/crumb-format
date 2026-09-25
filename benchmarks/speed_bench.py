"""Speed and throughput benchmarking for Wave Field LLM.

Measures:
- Tokens per second (training and inference)
- First token latency
- Time to generate N tokens
- Batch processing throughput

Tests across different sequence lengths, batch sizes, and hardware configurations.
"""

from __future__ import annotations

import argparse
import time
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
    format_time,
    format_number,
    create_markdown_table,
    compute_speedup,
)
from benchmarks.baseline_models import create_baseline_model


def warmup_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: str,
    num_warmup: int = 10
) -> None:
    """Warmup model to stabilize timings.
    
    Args:
        model: Model to warmup
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to use
        num_warmup: Number of warmup iterations
    """
    model.eval()
    vocab_size = model.cfg.vocab_size if hasattr(model, 'cfg') else 256
    
    with torch.no_grad():
        for _ in range(num_warmup):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            _ = model(x)
            
            if device == "cuda":
                torch.cuda.synchronize()


@torch.no_grad()
def benchmark_inference_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: str,
    num_iterations: int = 100,
    num_warmup: int = 10
) -> Tuple[float, Dict[str, float]]:
    """Benchmark inference throughput.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to use
        num_iterations: Number of iterations
        num_warmup: Number of warmup iterations
        
    Returns:
        Tuple of (tokens_per_second, metrics_dict)
    """
    model.eval()
    model = model.to(device)
    vocab_size = model.cfg.vocab_size if hasattr(model, 'cfg') else 256
    
    # Warmup
    warmup_model(model, batch_size, seq_len, device, num_warmup)
    
    # Benchmark
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        _ = model(x)
        
        if device == "cuda":
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    # Compute metrics
    total_tokens = batch_size * seq_len * num_iterations
    tokens_per_second = total_tokens / elapsed
    time_per_token = elapsed / total_tokens
    time_per_batch = elapsed / num_iterations
    
    metrics = {
        "tokens_per_second": tokens_per_second,
        "time_per_token_ms": time_per_token * 1000,
        "time_per_batch_ms": time_per_batch * 1000,
        "total_time_s": elapsed,
        "total_tokens": total_tokens,
    }
    
    return tokens_per_second, metrics


@torch.no_grad()
def benchmark_first_token_latency(
    model: nn.Module,
    seq_len: int,
    device: str,
    num_iterations: int = 100,
    num_warmup: int = 10
) -> Tuple[float, Dict[str, float]]:
    """Benchmark first token latency.
    
    Args:
        model: Model to benchmark
        seq_len: Sequence length
        device: Device to use
        num_iterations: Number of iterations
        num_warmup: Number of warmup iterations
        
    Returns:
        Tuple of (latency_ms, metrics_dict)
    """
    model.eval()
    model = model.to(device)
    vocab_size = model.cfg.vocab_size if hasattr(model, 'cfg') else 256
    
    # Warmup
    warmup_model(model, 1, seq_len, device, num_warmup)
    
    # Benchmark
    latencies = []
    
    for _ in range(num_iterations):
        x = torch.randint(0, vocab_size, (1, seq_len), device=device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        _ = model(x)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Compute statistics
    latencies_tensor = torch.tensor(latencies)
    mean_latency = latencies_tensor.mean().item()
    std_latency = latencies_tensor.std().item()
    min_latency = latencies_tensor.min().item()
    p50_latency = latencies_tensor.median().item()
    p95_latency = latencies_tensor.quantile(0.95).item()
    p99_latency = latencies_tensor.quantile(0.99).item()
    
    metrics = {
        "mean_ms": mean_latency,
        "std_ms": std_latency,
        "min_ms": min_latency,
        "p50_ms": p50_latency,
        "p95_ms": p95_latency,
        "p99_ms": p99_latency,
    }
    
    return mean_latency, metrics


def benchmark_generation_speed(
    model: nn.Module,
    prompt_len: int,
    num_tokens: int,
    device: str,
    num_iterations: int = 10
) -> Tuple[float, Dict[str, float]]:
    """Benchmark autoregressive generation speed.
    
    Args:
        model: Model to benchmark
        prompt_len: Prompt length
        num_tokens: Number of tokens to generate
        device: Device to use
        num_iterations: Number of iterations
        
    Returns:
        Tuple of (tokens_per_second, metrics_dict)
    """
    model.eval()
    model = model.to(device)
    vocab_size = model.cfg.vocab_size if hasattr(model, 'cfg') else 256
    
    times = []
    
    for _ in range(num_iterations):
        # Create prompt
        prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # Generate tokens
        generated = prompt
        for _ in range(num_tokens):
            with torch.no_grad():
                outputs = model(generated)
                logits = outputs["logits"]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Compute metrics
    mean_time = sum(times) / len(times)
    tokens_per_second = num_tokens / mean_time
    time_per_token = mean_time / num_tokens
    
    metrics = {
        "tokens_per_second": tokens_per_second,
        "time_per_token_ms": time_per_token * 1000,
        "total_time_s": mean_time,
        "num_tokens": num_tokens,
    }
    
    return tokens_per_second, metrics


def benchmark_speed_sweep(
    model: nn.Module,
    model_name: str,
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Benchmark speed across different configurations.
    
    Args:
        model: Model to benchmark
        model_name: Model name for results
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Speed Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Benchmark 1: Inference throughput across sequence lengths
    print("\n1. Inference Throughput (varying sequence length)")
    print("-" * 60)
    
    for seq_len in config.sequence_lengths:
        print(f"\nSequence length: {seq_len}")
        
        try:
            tokens_per_sec, metrics = benchmark_inference_throughput(
                model,
                config.batch_size,
                seq_len,
                config.device,
                config.num_iterations,
                config.num_warmup
            )
            
            print(f"  Throughput: {format_number(tokens_per_sec)} tokens/s")
            print(f"  Time per token: {metrics['time_per_token_ms']:.3f} ms")
            print(f"  Time per batch: {metrics['time_per_batch_ms']:.2f} ms")
            
            result = BenchmarkResult(
                name=f"throughput_seq{seq_len}",
                model_name=model_name,
                metric="tokens_per_second",
                value=tokens_per_sec,
                unit="tokens/s",
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
    
    # Benchmark 2: First token latency
    print("\n2. First Token Latency")
    print("-" * 60)
    
    for seq_len in config.sequence_lengths[:3]:  # Test first 3 lengths
        print(f"\nSequence length: {seq_len}")
        
        try:
            latency, metrics = benchmark_first_token_latency(
                model,
                seq_len,
                config.device,
                config.num_iterations,
                config.num_warmup
            )
            
            print(f"  Mean: {metrics['mean_ms']:.2f} ms")
            print(f"  P50: {metrics['p50_ms']:.2f} ms")
            print(f"  P95: {metrics['p95_ms']:.2f} ms")
            print(f"  P99: {metrics['p99_ms']:.2f} ms")
            
            result = BenchmarkResult(
                name=f"latency_seq{seq_len}",
                model_name=model_name,
                metric="first_token_latency",
                value=latency,
                unit="ms",
                metadata={
                    "seq_len": seq_len,
                    **metrics
                }
            )
            results.append(result)
            
        except RuntimeError as e:
            print(f"  Error: {e}")
            continue
    
    # Benchmark 3: Generation speed
    print("\n3. Autoregressive Generation Speed")
    print("-" * 60)
    
    prompt_len = 128
    num_tokens = 100
    
    print(f"\nPrompt length: {prompt_len}, Generate: {num_tokens} tokens")
    
    try:
        tokens_per_sec, metrics = benchmark_generation_speed(
            model,
            prompt_len,
            num_tokens,
            config.device,
            num_iterations=10
        )
        
        print(f"  Throughput: {format_number(tokens_per_sec)} tokens/s")
        print(f"  Time per token: {metrics['time_per_token_ms']:.2f} ms")
        print(f"  Total time: {metrics['total_time_s']:.2f} s")
        
        result = BenchmarkResult(
            name="generation_speed",
            model_name=model_name,
            metric="tokens_per_second",
            value=tokens_per_sec,
            unit="tokens/s",
            metadata={
                "prompt_len": prompt_len,
                "num_tokens": num_tokens,
                **metrics
            }
        )
        results.append(result)
        
    except RuntimeError as e:
        print(f"  Error: {e}")
    
    return results


def create_speed_comparison_table(results: List[BenchmarkResult]) -> str:
    """Create markdown comparison table for speed results.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Markdown table string
    """
    # Group by benchmark type
    throughput_results = [r for r in results if "throughput" in r.name]
    
    if not throughput_results:
        return "No throughput results available."
    
    # Get unique models and sequence lengths
    models = sorted(set(r.model_name for r in throughput_results))
    seq_lens = sorted(set(r.metadata["seq_len"] for r in throughput_results))
    
    # Create table
    headers = ["Seq Length"] + models + ["Speedup"]
    rows = []
    
    for seq_len in seq_lens:
        row = [str(seq_len)]
        
        # Get baseline (first model)
        baseline_result = next(
            (r for r in throughput_results 
             if r.metadata["seq_len"] == seq_len and r.model_name == models[0]),
            None
        )
        baseline_value = baseline_result.value if baseline_result else 0
        
        for model in models:
            result = next(
                (r for r in throughput_results 
                 if r.metadata["seq_len"] == seq_len and r.model_name == model),
                None
            )
            if result:
                row.append(f"{format_number(result.value)} tok/s")
            else:
                row.append("N/A")
        
        # Compute speedup (wavefield vs baseline)
        if len(models) > 1 and baseline_value > 0:
            wavefield_result = next(
                (r for r in throughput_results 
                 if r.metadata["seq_len"] == seq_len and "wavefield" in r.model_name.lower()),
                None
            )
            if wavefield_result:
                speedup = compute_speedup(baseline_value, wavefield_result.value)
                row.append(f"{speedup:.2f}×")
            else:
                row.append("N/A")
        else:
            row.append("N/A")
        
        rows.append(row)
    
    return create_markdown_table(headers, rows, alignments=["left"] + ["right"] * len(headers[1:]))


def main():
    parser = argparse.ArgumentParser(description="Speed benchmarking")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="wavefield",
                       choices=["wavefield", "gpt2", "llama", "mistral"],
                       help="Model type")
    parser.add_argument("--baselines", type=str, default="",
                       help="Comma-separated list of baseline models")
    parser.add_argument("--sequence-lengths", type=str, default="512,1024,2048,4096",
                       help="Comma-separated list of sequence lengths")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of iterations")
    parser.add_argument("--num-warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default="speed_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Parse sequence lengths
    seq_lens = [int(x.strip()) for x in args.sequence_lengths.split(",")]
    
    # Create config
    config = BenchmarkConfig(
        model_path=args.model,
        model_type=args.model_type,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        sequence_lengths=seq_lens,
        num_iterations=args.num_iterations,
        num_warmup=args.num_warmup,
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
    print(f"Speed Benchmarking")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    
    all_results = []
    for model_name, model in models.items():
        results = benchmark_speed_sweep(model, model_name, config)
        all_results.extend(results)
    
    # Save results
    output_dir = Path(args.output)
    save_results(all_results, output_dir, "speed_results")
    
    # Create comparison table
    table = create_speed_comparison_table(all_results)
    print(f"\n{'='*60}")
    print("Speed Comparison")
    print(f"{'='*60}\n")
    print(table)
    
    # Save table
    with open(output_dir / "speed_comparison.md", "w") as f:
        f.write("# Speed Comparison\n\n")
        f.write(table)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

# Made with Bob

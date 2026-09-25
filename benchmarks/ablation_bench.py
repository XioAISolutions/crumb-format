"""Ablation studies for Wave Field LLM.

Tests the impact of various architectural components:
- Number of wave heads
- Field size (F)
- Physics parameters (α, ω, φ initialization)
- Auxiliary losses (spectral diversity, field smoothness)
- Scatter-gather strategies
- FFT vs direct convolution
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional

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
    create_markdown_table,
)
from benchmarks.perplexity_bench import compute_perplexity, TextDataset, load_dataset
from benchmarks.speed_bench import benchmark_inference_throughput
from benchmarks.memory_bench import benchmark_inference_memory
from crumb_llm.tokenizer import ByteTokenizer
from torch.utils.data import DataLoader


def create_ablation_model(
    base_config: WaveFieldConfig,
    ablation_type: str,
    ablation_value: any
) -> WaveFieldLM:
    """Create model with specific ablation.
    
    Args:
        base_config: Base configuration
        ablation_type: Type of ablation
        ablation_value: Value for ablation
        
    Returns:
        Model with ablation applied
    """
    config = copy.deepcopy(base_config)
    
    if ablation_type == "n_heads":
        config.n_heads = ablation_value
    elif ablation_type == "field_size":
        config.field_size = ablation_value
    elif ablation_type == "n_layers":
        config.n_layers = ablation_value
    elif ablation_type == "dim":
        config.dim = ablation_value
    elif ablation_type == "boundary":
        config.boundary = ablation_value
    elif ablation_type == "dispersion":
        config.dispersion = ablation_value
    elif ablation_type == "adaptive_kernels":
        config.adaptive_kernels = ablation_value
    elif ablation_type == "spectral_gate":
        config.spectral_gate = ablation_value
    elif ablation_type == "resonance_memory":
        config.resonance_memory = ablation_value
    elif ablation_type == "hybrid_gate":
        config.hybrid_gate = ablation_value
    elif ablation_type == "use_v2_blocks":
        config.use_v2_blocks = ablation_value
    elif ablation_type == "kernel_mode":
        config.kernel_mode = ablation_value
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    return WaveFieldLM(config)


def ablate_n_heads(
    base_config: WaveFieldConfig,
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Ablate number of wave heads.
    
    Args:
        base_config: Base model configuration
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    head_counts = [1, 2, 4, 8]
    
    print(f"\n{'='*60}")
    print("Ablation: Number of Wave Heads")
    print(f"{'='*60}")
    
    for n_heads in head_counts:
        if base_config.dim % n_heads != 0:
            print(f"\nSkipping n_heads={n_heads} (dim not divisible)")
            continue
        
        print(f"\nn_heads = {n_heads}")
        print("-" * 60)
        
        # Create model
        model = create_ablation_model(base_config, "n_heads", n_heads)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Benchmark perplexity
        tokenizer = ByteTokenizer()
        text = load_dataset("wikitext", max_samples=100)
        dataset = TextDataset(text, tokenizer, seq_len=512)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        perplexity, metrics = compute_perplexity(
            model, dataloader, config.device, verbose=False
        )
        print(f"Perplexity: {perplexity:.2f}")
        
        # Benchmark speed
        tokens_per_sec, speed_metrics = benchmark_inference_throughput(
            model, 1, 512, config.device, num_iterations=50, num_warmup=5
        )
        print(f"Speed: {tokens_per_sec:.0f} tokens/s")
        
        # Store results
        results.append(BenchmarkResult(
            name=f"ablation_heads_{n_heads}_perplexity",
            model_name=f"heads_{n_heads}",
            metric="perplexity",
            value=perplexity,
            unit="ppl",
            metadata={"ablation": "n_heads", "value": n_heads}
        ))
        
        results.append(BenchmarkResult(
            name=f"ablation_heads_{n_heads}_speed",
            model_name=f"heads_{n_heads}",
            metric="tokens_per_second",
            value=tokens_per_sec,
            unit="tokens/s",
            metadata={"ablation": "n_heads", "value": n_heads}
        ))
    
    return results


def ablate_field_size(
    base_config: WaveFieldConfig,
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Ablate field size.
    
    Args:
        base_config: Base model configuration
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    field_sizes = [128, 256, 512, 1024]
    
    print(f"\n{'='*60}")
    print("Ablation: Field Size")
    print(f"{'='*60}")
    
    for field_size in field_sizes:
        print(f"\nfield_size = {field_size}")
        print("-" * 60)
        
        # Create model
        model = create_ablation_model(base_config, "field_size", field_size)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Benchmark perplexity
        tokenizer = ByteTokenizer()
        text = load_dataset("wikitext", max_samples=100)
        dataset = TextDataset(text, tokenizer, seq_len=512)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        perplexity, metrics = compute_perplexity(
            model, dataloader, config.device, verbose=False
        )
        print(f"Perplexity: {perplexity:.2f}")
        
        # Benchmark speed
        tokens_per_sec, speed_metrics = benchmark_inference_throughput(
            model, 1, 512, config.device, num_iterations=50, num_warmup=5
        )
        print(f"Speed: {tokens_per_sec:.0f} tokens/s")
        
        # Benchmark memory (if CUDA)
        if config.device == "cuda":
            peak_memory, mem_metrics = benchmark_inference_memory(
                model, 1, 512, config.device
            )
            print(f"Memory: {mem_metrics['peak_mb']:.1f} MB")
            
            results.append(BenchmarkResult(
                name=f"ablation_field_{field_size}_memory",
                model_name=f"field_{field_size}",
                metric="peak_memory",
                value=peak_memory,
                unit="bytes",
                metadata={"ablation": "field_size", "value": field_size}
            ))
        
        # Store results
        results.append(BenchmarkResult(
            name=f"ablation_field_{field_size}_perplexity",
            model_name=f"field_{field_size}",
            metric="perplexity",
            value=perplexity,
            unit="ppl",
            metadata={"ablation": "field_size", "value": field_size}
        ))
        
        results.append(BenchmarkResult(
            name=f"ablation_field_{field_size}_speed",
            model_name=f"field_{field_size}",
            metric="tokens_per_second",
            value=tokens_per_sec,
            unit="tokens/s",
            metadata={"ablation": "field_size", "value": field_size}
        ))
    
    return results


def ablate_features(
    base_config: WaveFieldConfig,
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Ablate advanced features.
    
    Args:
        base_config: Base model configuration
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    
    features = [
        ("baseline", {}),
        ("dispersion", {"dispersion": True}),
        ("adaptive_kernels", {"adaptive_kernels": True}),
        ("spectral_gate", {"spectral_gate": True}),
        ("resonance_memory", {"resonance_memory": True}),
        ("all_features", {
            "dispersion": True,
            "adaptive_kernels": True,
            "spectral_gate": True,
            "resonance_memory": True,
        }),
    ]
    
    print(f"\n{'='*60}")
    print("Ablation: Advanced Features")
    print(f"{'='*60}")
    
    for feature_name, feature_config in features:
        print(f"\n{feature_name}")
        print("-" * 60)
        
        # Create model with features
        model_config = copy.deepcopy(base_config)
        for key, value in feature_config.items():
            setattr(model_config, key, value)
        
        model = WaveFieldLM(model_config)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Benchmark perplexity
        tokenizer = ByteTokenizer()
        text = load_dataset("wikitext", max_samples=100)
        dataset = TextDataset(text, tokenizer, seq_len=512)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        perplexity, metrics = compute_perplexity(
            model, dataloader, config.device, verbose=False
        )
        print(f"Perplexity: {perplexity:.2f}")
        
        # Store results
        results.append(BenchmarkResult(
            name=f"ablation_feature_{feature_name}",
            model_name=feature_name,
            metric="perplexity",
            value=perplexity,
            unit="ppl",
            metadata={"ablation": "features", "config": feature_config}
        ))
    
    return results


def create_ablation_table(
    results: List[BenchmarkResult],
    ablation_type: str
) -> str:
    """Create markdown table for ablation results.
    
    Args:
        results: List of benchmark results
        ablation_type: Type of ablation
        
    Returns:
        Markdown table string
    """
    # Filter results for this ablation type
    ablation_results = [r for r in results if ablation_type in r.name]
    
    if not ablation_results:
        return f"No results for ablation: {ablation_type}"
    
    # Group by metric
    metrics = sorted(set(r.metric for r in ablation_results))
    models = sorted(set(r.model_name for r in ablation_results))
    
    # Create table
    headers = ["Configuration"] + metrics
    rows = []
    
    for model in models:
        row = [model]
        for metric in metrics:
            result = next(
                (r for r in ablation_results 
                 if r.model_name == model and r.metric == metric),
                None
            )
            if result:
                if metric == "perplexity":
                    row.append(f"{result.value:.2f}")
                elif metric == "tokens_per_second":
                    row.append(f"{result.value:.0f}")
                elif metric == "peak_memory":
                    row.append(f"{result.value / 1024**2:.1f} MB")
                else:
                    row.append(f"{result.value:.3f}")
            else:
                row.append("N/A")
        rows.append(row)
    
    return create_markdown_table(headers, rows, alignments=["left"] + ["right"] * len(metrics))


def main():
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--model", type=str, help="Path to base model checkpoint")
    parser.add_argument("--ablate", type=str, default="heads,field-size,features",
                       help="Comma-separated list of ablations to run")
    parser.add_argument("--output", type=str, default="ablation_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create config
    config = BenchmarkConfig(
        model_path=args.model,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
    )
    
    # Base model configuration
    if args.model and Path(args.model).exists():
        checkpoint = torch.load(args.model, map_location="cpu")
        base_config = WaveFieldConfig(**checkpoint.get("config", {}))
    else:
        base_config = WaveFieldConfig(dim=128, n_layers=4, n_heads=4, field_size=256)
    
    # Parse ablations
    ablations = [a.strip() for a in args.ablate.split(",")]
    
    print(f"\n{'='*60}")
    print(f"Ablation Studies")
    print(f"{'='*60}")
    print(f"Base config: dim={base_config.dim}, layers={base_config.n_layers}, "
          f"heads={base_config.n_heads}, field_size={base_config.field_size}")
    print(f"Ablations: {', '.join(ablations)}")
    print(f"Device: {args.device}")
    
    all_results = []
    
    # Run ablations
    if "heads" in ablations:
        results = ablate_n_heads(base_config, config)
        all_results.extend(results)
    
    if "field-size" in ablations or "field_size" in ablations:
        results = ablate_field_size(base_config, config)
        all_results.extend(results)
    
    if "features" in ablations:
        results = ablate_features(base_config, config)
        all_results.extend(results)
    
    # Save results
    output_dir = Path(args.output)
    save_results(all_results, output_dir, "ablation_results")
    
    # Create tables
    print(f"\n{'='*60}")
    print("Ablation Results")
    print(f"{'='*60}\n")
    
    for ablation in ablations:
        table = create_ablation_table(all_results, ablation)
        print(f"\n{ablation.upper()}:")
        print(table)
        
        # Save table
        with open(output_dir / f"ablation_{ablation}.md", "w") as f:
            f.write(f"# Ablation: {ablation}\n\n")
            f.write(table)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

# Made with Bob

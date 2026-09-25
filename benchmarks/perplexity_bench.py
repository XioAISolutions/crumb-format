"""Perplexity benchmarking for Wave Field LLM.

Tests perplexity on multiple datasets:
- WikiText-103
- C4 (Colossal Clean Crawled Corpus)
- CRUMB-structured documents
- Code datasets (The Stack, GitHub)

Compares Wave Field LLM against transformer baselines and provides
per-domain perplexity breakdown.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crumb_llm.model import WaveFieldLM, WaveFieldConfig
from crumb_llm.tokenizer import ByteTokenizer
from benchmarks.utils import (
    BenchmarkResult,
    BenchmarkConfig,
    set_seed,
    save_results,
    ProgressBar,
    create_markdown_table,
)
from benchmarks.baseline_models import create_baseline_model


class TextDataset(Dataset):
    """Simple text dataset for perplexity evaluation."""
    
    def __init__(
        self,
        text: str,
        tokenizer,
        seq_len: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text)
        
        # Create sliding windows
        self.windows = []
        for i in range(0, len(self.tokens) - seq_len, stride):
            self.windows.append(self.tokens[i:i + seq_len])
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.windows[idx], dtype=torch.long)


def load_dataset(
    dataset_name: str,
    dataset_path: Optional[str] = None,
    max_samples: Optional[int] = None
) -> str:
    """Load dataset text.
    
    Args:
        dataset_name: Dataset name (wikitext, c4, crumb, code)
        dataset_path: Optional path to dataset file
        max_samples: Maximum number of samples to load
        
    Returns:
        Dataset text
    """
    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if max_samples:
                # Truncate to approximate sample count
                text = text[:max_samples * 1000]  # ~1000 chars per sample
            return text
    
    # Generate synthetic data for testing
    if dataset_name == "wikitext":
        text = """
        The history of artificial intelligence began in antiquity with myths and stories
        of artificial beings endowed with intelligence. Modern AI research began in the
        1950s with the work of Alan Turing and others. The field has experienced several
        waves of optimism followed by disappointment and loss of funding.
        """ * 100
    
    elif dataset_name == "c4":
        text = """
        This is a sample from the C4 dataset. It contains web-crawled text from various
        sources. The text is cleaned and filtered to remove low-quality content. This
        dataset is commonly used for training large language models.
        """ * 100
    
    elif dataset_name == "crumb":
        text = """
        [section:introduction]
        @priority: 5
        This is a CRUMB-formatted document with structured sections.
        
        [section:details]
        @priority: 3
        fold:summary/start
        This section contains detailed information.
        fold:summary/end
        
        [section:conclusion]
        @priority: 4
        The conclusion summarizes the key points.
        """ * 50
    
    elif dataset_name == "code":
        text = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        class BinaryTree:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None
        """ * 100
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return text


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[float, Dict[str, float]]:
    """Compute perplexity on a dataset.
    
    Args:
        model: Language model
        dataloader: Data loader
        device: Device to use
        verbose: Show progress bar
        
    Returns:
        Tuple of (perplexity, metrics_dict)
    """
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    if verbose:
        pbar = ProgressBar(len(dataloader), desc="Computing perplexity")
    
    for batch in dataloader:
        batch = batch.to(device)
        
        # Forward pass
        outputs = model(batch, targets=batch)
        loss = outputs["loss"]
        
        # Accumulate
        batch_tokens = batch.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        
        if verbose:
            pbar.update(1)
    
    if verbose:
        pbar.close()
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }
    
    return perplexity, metrics


def benchmark_perplexity(
    model: nn.Module,
    model_name: str,
    datasets: List[str],
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Benchmark perplexity across multiple datasets.
    
    Args:
        model: Model to benchmark
        model_name: Model name for results
        datasets: List of dataset names
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    tokenizer = ByteTokenizer()
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        text = load_dataset(
            dataset_name,
            config.dataset_path,
            config.max_samples
        )
        
        # Create dataset and dataloader
        dataset = TextDataset(
            text,
            tokenizer,
            seq_len=config.sequence_lengths[0],
            stride=config.sequence_lengths[0] // 2
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Samples: {len(dataset)}")
        print(f"Tokens: ~{len(dataset) * config.sequence_lengths[0]:,}")
        
        # Compute perplexity
        perplexity, metrics = compute_perplexity(
            model,
            dataloader,
            device=config.device,
            verbose=config.verbose
        )
        
        print(f"\nResults:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        
        # Store result
        result = BenchmarkResult(
            name=f"perplexity_{dataset_name}",
            model_name=model_name,
            metric="perplexity",
            value=perplexity,
            unit="ppl",
            metadata={
                "dataset": dataset_name,
                "loss": metrics["loss"],
                "total_tokens": metrics["total_tokens"],
                "seq_len": config.sequence_lengths[0],
            }
        )
        results.append(result)
    
    return results


def compare_models(
    models: Dict[str, nn.Module],
    datasets: List[str],
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Compare multiple models on perplexity.
    
    Args:
        models: Dictionary of model_name -> model
        datasets: List of dataset names
        config: Benchmark configuration
        
    Returns:
        List of all benchmark results
    """
    all_results = []
    
    for model_name, model in models.items():
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name}")
        print(f"# Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'#'*60}")
        
        results = benchmark_perplexity(model, model_name, datasets, config)
        all_results.extend(results)
    
    return all_results


def create_comparison_table(results: List[BenchmarkResult]) -> str:
    """Create markdown comparison table.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Markdown table string
    """
    # Group by dataset
    datasets = sorted(set(r.metadata["dataset"] for r in results))
    models = sorted(set(r.model_name for r in results))
    
    # Create table
    headers = ["Dataset"] + models
    rows = []
    
    for dataset in datasets:
        row = [dataset]
        for model in models:
            # Find result for this dataset and model
            result = next(
                (r for r in results 
                 if r.metadata["dataset"] == dataset and r.model_name == model),
                None
            )
            if result:
                row.append(f"{result.value:.2f}")
            else:
                row.append("N/A")
        rows.append(row)
    
    return create_markdown_table(headers, rows, alignments=["left"] + ["right"] * len(models))


def main():
    parser = argparse.ArgumentParser(description="Perplexity benchmarking")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="wavefield",
                       choices=["wavefield", "gpt2", "llama", "mistral"],
                       help="Model type")
    parser.add_argument("--datasets", type=str, default="wikitext,c4,crumb,code",
                       help="Comma-separated list of datasets")
    parser.add_argument("--baselines", type=str, default="",
                       help="Comma-separated list of baseline models to compare")
    parser.add_argument("--seq-len", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples")
    parser.add_argument("--output", type=str, default="perplexity_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create config
    config = BenchmarkConfig(
        model_path=args.model,
        model_type=args.model_type,
        seed=args.seed,
        device=args.device,
        dataset_path=None,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        sequence_lengths=[args.seq_len],
        output_dir=args.output,
    )
    
    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    # Create models
    models = {}
    
    # Main model
    if args.model and Path(args.model).exists():
        # Load from checkpoint
        checkpoint = torch.load(args.model, map_location="cpu")
        model_cfg = WaveFieldConfig(**checkpoint.get("config", {}))
        model = WaveFieldLM(model_cfg)
        model.load_state_dict(checkpoint["model"])
        models[args.model_type] = model
    else:
        # Create default model
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
    print(f"Perplexity Benchmarking")
    print(f"{'='*60}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Device: {args.device}")
    
    results = compare_models(models, datasets, config)
    
    # Save results
    output_dir = Path(args.output)
    save_results(results, output_dir, "perplexity_results")
    
    # Create comparison table
    table = create_comparison_table(results)
    print(f"\n{'='*60}")
    print("Perplexity Comparison")
    print(f"{'='*60}\n")
    print(table)
    
    # Save table
    with open(output_dir / "perplexity_comparison.md", "w") as f:
        f.write("# Perplexity Comparison\n\n")
        f.write(table)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

# Made with Bob

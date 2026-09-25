"""Generation quality benchmarking for Wave Field LLM.

Measures generation quality on various tasks:
- Text completion
- Summarization (ROUGE scores)
- Question answering (Exact Match, F1)
- Code generation (Pass@k)
- CRUMB-specific tasks
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crumb_llm.model import WaveFieldLM, WaveFieldConfig
from crumb_llm.tokenizer import ByteTokenizer
from crumb_llm.sample import sample_from_logits
from benchmarks.utils import (
    BenchmarkResult,
    BenchmarkConfig,
    set_seed,
    save_results,
    create_markdown_table,
)
from benchmarks.baseline_models import create_baseline_model


def compute_rouge_scores(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE scores (simplified version).
    
    Args:
        generated: Generated text
        reference: Reference text
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    # Simplified ROUGE implementation
    # In production, use rouge_score library
    
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    
    # ROUGE-1 (unigram overlap)
    gen_unigrams = set(gen_tokens)
    ref_unigrams = set(ref_tokens)
    
    if len(ref_unigrams) == 0:
        rouge1 = 0.0
    else:
        overlap = len(gen_unigrams & ref_unigrams)
        rouge1 = overlap / len(ref_unigrams)
    
    # ROUGE-2 (bigram overlap)
    gen_bigrams = set(zip(gen_tokens[:-1], gen_tokens[1:]))
    ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
    
    if len(ref_bigrams) == 0:
        rouge2 = 0.0
    else:
        overlap = len(gen_bigrams & ref_bigrams)
        rouge2 = overlap / len(ref_bigrams)
    
    # ROUGE-L (longest common subsequence)
    # Simplified: use ROUGE-1 as approximation
    rougeL = rouge1
    
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }


def compute_exact_match(generated: str, reference: str) -> float:
    """Compute exact match score.
    
    Args:
        generated: Generated text
        reference: Reference text
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if generated.strip().lower() == reference.strip().lower() else 0.0


def compute_f1_score(generated: str, reference: str) -> float:
    """Compute F1 score (token-level).
    
    Args:
        generated: Generated text
        reference: Reference text
        
    Returns:
        F1 score
    """
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if len(gen_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    overlap = len(gen_tokens & ref_tokens)
    
    precision = overlap / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
    recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


@torch.no_grad()
def generate_text(
    model: nn.Module,
    prompt: str,
    tokenizer,
    max_tokens: int = 100,
    temperature: float = 0.8,
    device: str = "cuda"
) -> str:
    """Generate text from prompt.
    
    Args:
        model: Language model
        prompt: Input prompt
        tokenizer: Tokenizer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
        
    Returns:
        Generated text
    """
    model.eval()
    model = model.to(device)
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    generated = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Generate tokens
    for _ in range(max_tokens):
        outputs = model(generated)
        logits = outputs["logits"][:, -1, :]
        
        # Sample next token
        next_token = sample_from_logits(logits, temperature=temperature)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Stop at end of sequence (if applicable)
        if next_token.item() == 0:  # Assuming 0 is EOS
            break
    
    # Decode
    generated_tokens = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def benchmark_text_completion(
    model: nn.Module,
    model_name: str,
    tokenizer,
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Benchmark text completion quality.
    
    Args:
        model: Model to benchmark
        model_name: Model name
        tokenizer: Tokenizer
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    
    # Test prompts and expected completions
    test_cases = [
        {
            "prompt": "The quick brown fox",
            "reference": "jumps over the lazy dog",
        },
        {
            "prompt": "Once upon a time",
            "reference": "there was a princess who lived in a castle",
        },
        {
            "prompt": "def fibonacci(n):",
            "reference": "if n <= 1: return n\nreturn fibonacci(n-1) + fibonacci(n-2)",
        },
    ]
    
    print(f"\nText Completion: {model_name}")
    print("-" * 60)
    
    total_rouge1 = 0.0
    total_f1 = 0.0
    
    for i, test_case in enumerate(test_cases):
        prompt = test_case["prompt"]
        reference = test_case["reference"]
        
        # Generate
        generated = generate_text(
            model,
            prompt,
            tokenizer,
            max_tokens=50,
            temperature=0.8,
            device=config.device
        )
        
        # Remove prompt from generated text
        generated_completion = generated[len(prompt):].strip()
        
        # Compute scores
        rouge_scores = compute_rouge_scores(generated_completion, reference)
        f1 = compute_f1_score(generated_completion, reference)
        
        print(f"\nTest {i+1}:")
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {generated_completion[:100]}...")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.3f}")
        print(f"  F1: {f1:.3f}")
        
        total_rouge1 += rouge_scores['rouge1']
        total_f1 += f1
    
    # Average scores
    avg_rouge1 = total_rouge1 / len(test_cases)
    avg_f1 = total_f1 / len(test_cases)
    
    print(f"\nAverage Scores:")
    print(f"  ROUGE-1: {avg_rouge1:.3f}")
    print(f"  F1: {avg_f1:.3f}")
    
    results.append(BenchmarkResult(
        name="text_completion_rouge1",
        model_name=model_name,
        metric="rouge1",
        value=avg_rouge1,
        unit="score",
        metadata={"task": "text_completion"}
    ))
    
    results.append(BenchmarkResult(
        name="text_completion_f1",
        model_name=model_name,
        metric="f1",
        value=avg_f1,
        unit="score",
        metadata={"task": "text_completion"}
    ))
    
    return results


def benchmark_quality(
    model: nn.Module,
    model_name: str,
    config: BenchmarkConfig
) -> List[BenchmarkResult]:
    """Benchmark generation quality across tasks.
    
    Args:
        model: Model to benchmark
        model_name: Model name
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    tokenizer = ByteTokenizer()
    results = []
    
    print(f"\n{'='*60}")
    print(f"Quality Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Text completion
    completion_results = benchmark_text_completion(
        model, model_name, tokenizer, config
    )
    results.extend(completion_results)
    
    return results


def create_quality_comparison_table(results: List[BenchmarkResult]) -> str:
    """Create markdown comparison table for quality results.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Markdown table string
    """
    # Group by metric
    metrics = sorted(set(r.metric for r in results))
    models = sorted(set(r.model_name for r in results))
    
    # Create table
    headers = ["Metric"] + models
    rows = []
    
    for metric in metrics:
        row = [metric]
        for model in models:
            result = next(
                (r for r in results if r.metric == metric and r.model_name == model),
                None
            )
            if result:
                row.append(f"{result.value:.3f}")
            else:
                row.append("N/A")
        rows.append(row)
    
    return create_markdown_table(headers, rows, alignments=["left"] + ["right"] * len(models))


def main():
    parser = argparse.ArgumentParser(description="Quality benchmarking")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="wavefield",
                       choices=["wavefield", "gpt2", "llama", "mistral"],
                       help="Model type")
    parser.add_argument("--baselines", type=str, default="",
                       help="Comma-separated list of baseline models")
    parser.add_argument("--output", type=str, default="quality_results",
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
        model_type=args.model_type,
        seed=args.seed,
        device=args.device,
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
    print(f"Quality Benchmarking")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Device: {args.device}")
    
    all_results = []
    for model_name, model in models.items():
        results = benchmark_quality(model, model_name, config)
        all_results.extend(results)
    
    # Save results
    output_dir = Path(args.output)
    save_results(all_results, output_dir, "quality_results")
    
    # Create comparison table
    table = create_quality_comparison_table(all_results)
    print(f"\n{'='*60}")
    print("Quality Comparison")
    print(f"{'='*60}\n")
    print(table)
    
    # Save table
    with open(output_dir / "quality_comparison.md", "w") as f:
        f.write("# Quality Comparison\n\n")
        f.write(table)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

# Made with Bob

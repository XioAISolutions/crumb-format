#!/usr/bin/env python3
"""
Model Zoo Evaluation Framework

Main evaluation orchestrator that runs comprehensive benchmarks
on Wave Field LLM models.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .benchmarks import BenchmarkRunner
from .crumb_eval import CRUMBEvaluator
from .human_eval import HumanEvaluator
from .safety_eval import SafetyEvaluator
from .performance_eval import PerformanceEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation orchestrator."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        """Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model()
        
        # Initialize evaluators
        self.benchmark_runner = BenchmarkRunner(self.model, device, batch_size)
        self.crumb_evaluator = CRUMBEvaluator(self.model, device)
        self.human_evaluator = HumanEvaluator(self.model, device)
        self.safety_evaluator = SafetyEvaluator(self.model, device)
        self.performance_evaluator = PerformanceEvaluator(self.model, device)
    
    def _load_model(self):
        """Load model from checkpoint or HuggingFace."""
        # Try loading from local checkpoint
        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Model loading logic here
            logger.info("Loaded model from local checkpoint")
            return checkpoint  # Placeholder
        else:
            # Try loading from HuggingFace
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(self.model_path)
                model = model.to(self.device)
                logger.info("Loaded model from HuggingFace Hub")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def evaluate(
        self,
        benchmarks: Optional[List[str]] = None,
        include_crumb: bool = False,
        include_human: bool = False,
        include_safety: bool = True,
        include_performance: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation.
        
        Args:
            benchmarks: List of benchmark names to run. If None, runs all.
            include_crumb: Whether to run CRUMB-specific evaluation
            include_human: Whether to run human evaluation
            include_safety: Whether to run safety evaluation
            include_performance: Whether to run performance profiling
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        start_time = time.time()
        
        results = {
            'model_path': self.model_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'batch_size': self.batch_size,
        }
        
        # Run standard benchmarks
        if benchmarks is None:
            benchmarks = [
                'mmlu', 'hellaswag', 'arc_easy', 'arc_challenge',
                'winogrande', 'truthfulqa', 'humaneval', 'mbpp'
            ]
        
        logger.info(f"Running benchmarks: {benchmarks}")
        benchmark_results = self.benchmark_runner.run_benchmarks(benchmarks)
        results['benchmarks'] = benchmark_results
        
        # Run CRUMB evaluation
        if include_crumb:
            logger.info("Running CRUMB evaluation")
            crumb_results = self.crumb_evaluator.evaluate()
            results['crumb'] = crumb_results
        
        # Run human evaluation
        if include_human:
            logger.info("Running human evaluation")
            human_results = self.human_evaluator.evaluate()
            results['human_eval'] = human_results
        
        # Run safety evaluation
        if include_safety:
            logger.info("Running safety evaluation")
            safety_results = self.safety_evaluator.evaluate()
            results['safety'] = safety_results
        
        # Run performance evaluation
        if include_performance:
            logger.info("Running performance evaluation")
            performance_results = self.performance_evaluator.evaluate()
            results['performance'] = performance_results
        
        # Calculate summary metrics
        results['summary'] = self._calculate_summary(results)
        
        elapsed_time = time.time() - start_time
        results['evaluation_time'] = elapsed_time
        logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _calculate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics from evaluation results."""
        summary = {}
        
        # Average benchmark scores
        if 'benchmarks' in results:
            benchmark_scores = []
            for benchmark, result in results['benchmarks'].items():
                if isinstance(result, dict) and 'score' in result:
                    benchmark_scores.append(result['score'])
            
            if benchmark_scores:
                summary['avg_benchmark_score'] = sum(benchmark_scores) / len(benchmark_scores)
        
        # CRUMB metrics
        if 'crumb' in results:
            summary['crumb_validation_rate'] = results['crumb'].get('validation_rate', 0)
            summary['crumb_generation_quality'] = results['crumb'].get('generation_quality', 0)
        
        # Safety metrics
        if 'safety' in results:
            summary['safety_score'] = results['safety'].get('overall_score', 0)
            summary['toxicity_rate'] = results['safety'].get('toxicity_rate', 0)
        
        # Performance metrics
        if 'performance' in results:
            summary['inference_speed'] = results['performance'].get('tokens_per_second', 0)
            summary['memory_usage_gb'] = results['performance'].get('memory_usage_gb', 0)
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console.
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        summary = results.get('summary', {})
        
        print(f"\nModel: {results['model_path']}")
        print(f"Evaluation Time: {results.get('evaluation_time', 0):.2f}s")
        
        if 'avg_benchmark_score' in summary:
            print(f"\nAverage Benchmark Score: {summary['avg_benchmark_score']:.2%}")
        
        if 'benchmarks' in results:
            print("\nBenchmark Results:")
            for benchmark, result in results['benchmarks'].items():
                if isinstance(result, dict) and 'score' in result:
                    print(f"  {benchmark}: {result['score']:.2%}")
        
        if 'crumb' in results:
            print("\nCRUMB Evaluation:")
            print(f"  Validation Rate: {summary.get('crumb_validation_rate', 0):.2%}")
            print(f"  Generation Quality: {summary.get('crumb_generation_quality', 0):.2%}")
        
        if 'safety' in results:
            print("\nSafety Evaluation:")
            print(f"  Overall Safety Score: {summary.get('safety_score', 0):.2%}")
            print(f"  Toxicity Rate: {summary.get('toxicity_rate', 0):.2%}")
        
        if 'performance' in results:
            print("\nPerformance:")
            print(f"  Inference Speed: {summary.get('inference_speed', 0):.1f} tokens/s")
            print(f"  Memory Usage: {summary.get('memory_usage_gb', 0):.2f} GB")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Wave Field LLM models")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help="Benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--include-crumb",
        action="store_true",
        help="Include CRUMB-specific evaluation",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="Include human evaluation",
    )
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Skip safety evaluation",
    )
    parser.add_argument(
        "--no-performance",
        action="store_true",
        help="Skip performance evaluation",
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        benchmarks=args.benchmarks,
        include_crumb=args.include_crumb,
        include_human=args.include_human,
        include_safety=not args.no_safety,
        include_performance=not args.no_performance,
    )
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()

# Made with Bob

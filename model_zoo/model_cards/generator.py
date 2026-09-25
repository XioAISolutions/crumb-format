#!/usr/bin/env python3
"""
Model Card Generator

Automatically generates comprehensive model cards from training logs,
configuration files, and evaluation results.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """Generate model cards from training artifacts."""
    
    def __init__(self, template_path: Optional[str] = None):
        """Initialize the generator.
        
        Args:
            template_path: Path to custom template file. If None, uses default.
        """
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = Path(__file__).parent / "template.md"
        
        with open(self.template_path, 'r') as f:
            self.template = f.read()
    
    def generate(
        self,
        config_path: str,
        training_log_path: str,
        eval_results_path: str,
        output_path: str,
        **kwargs
    ) -> str:
        """Generate a model card.
        
        Args:
            config_path: Path to model configuration YAML
            training_log_path: Path to training log JSON
            eval_results_path: Path to evaluation results JSON
            output_path: Path to save generated model card
            **kwargs: Additional metadata to include
        
        Returns:
            Generated model card content
        """
        logger.info("Generating model card")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load training log
        with open(training_log_path, 'r') as f:
            training_log = json.load(f)
        
        # Load evaluation results
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        # Extract metadata
        metadata = self._extract_metadata(config, training_log, eval_results, **kwargs)
        
        # Fill template
        model_card = self._fill_template(metadata)
        
        # Save model card
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(model_card)
        
        logger.info(f"Model card saved to {output_path}")
        return model_card
    
    def _extract_metadata(
        self,
        config: Dict[str, Any],
        training_log: Dict[str, Any],
        eval_results: Dict[str, Any],
        **kwargs
    ) -> Dict[str, str]:
        """Extract metadata from training artifacts."""
        model_config = config['model']
        training_config = config['training']
        
        # Calculate number of parameters
        arch = model_config['architecture']
        num_params = self._estimate_parameters(arch)
        
        # Format dataset mixture
        dataset_mix = config['dataset']['mix']
        dataset_table = self._format_dataset_table(dataset_mix)
        
        # Format benchmark results
        benchmark_table = self._format_benchmark_table(eval_results.get('benchmarks', {}))
        
        # Extract training duration
        training_duration = training_log.get('training_duration', 'N/A')
        
        # Build metadata dictionary
        metadata = {
            'model_name': model_config['name'],
            'version': model_config['version'],
            'description': model_config.get('description', ''),
            'num_parameters': f"{num_params:,}",
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'license': kwargs.get('license', 'Apache 2.0'),
            
            # Architecture
            'dim': arch['dim'],
            'n_layers': arch['n_layers'],
            'n_heads': arch['n_heads'],
            'field_size': arch['field_size'],
            'vocab_size': arch['vocab_size'],
            'max_seq_len': arch['max_seq_len'],
            'wave_dim': model_config['wave_field']['wave_dim'],
            'num_frequencies': model_config['wave_field']['num_frequencies'],
            
            # Training
            'total_tokens': training_config['total_tokens'],
            'max_steps': training_config['max_steps'],
            'effective_batch_size': training_config['batch_size'] * training_config['gradient_accumulation'],
            'learning_rate': training_config['learning_rate'],
            'mixed_precision': training_config['mixed_precision'],
            'training_duration': training_duration,
            'hardware_used': kwargs.get('hardware_used', 'N/A'),
            
            # Dataset
            'dataset_mix_table': dataset_table,
            'total_training_data': f"{training_config['total_tokens']:,} tokens",
            
            # Performance
            'benchmark_results_table': benchmark_table,
            'val_perplexity': f"{eval_results.get('val_perplexity', 'N/A'):.2f}",
            'test_perplexity': f"{eval_results.get('test_perplexity', 'N/A'):.2f}",
            'inference_speed': eval_results.get('inference_speed', 'N/A'),
            'memory_usage': eval_results.get('memory_usage', 'N/A'),
            
            # Usage
            'intended_use_cases': self._format_list(kwargs.get('intended_uses', [])),
            'out_of_scope_uses': self._format_list(kwargs.get('out_of_scope', [])),
            'limitations': self._format_list(kwargs.get('limitations', [])),
            'bias_considerations': kwargs.get('bias_considerations', 'See evaluation section for bias analysis.'),
            'ethical_considerations': kwargs.get('ethical_considerations', 'This model should be used responsibly.'),
            
            # Training details
            'training_procedure': self._format_training_procedure(training_config),
            'batch_size': training_config['batch_size'],
            'gradient_accumulation': training_config['gradient_accumulation'],
            'warmup_steps': training_config['warmup_steps'],
            'weight_decay': training_config['weight_decay'],
            'grad_clip': training_config['grad_clip'],
            'lr_decay_style': training_config['lr_decay_style'],
            
            # Infrastructure
            'num_gpus': kwargs.get('num_gpus', 'N/A'),
            'gpu_type': kwargs.get('gpu_type', 'N/A'),
            'hardware_type': kwargs.get('hardware_type', 'N/A'),
            'hours_used': kwargs.get('hours_used', 'N/A'),
            'cloud_provider': kwargs.get('cloud_provider', 'N/A'),
            'carbon_emissions': kwargs.get('carbon_emissions', 'N/A'),
            'compute_region': kwargs.get('compute_region', 'N/A'),
            
            # Evaluation
            'evaluation_datasets': self._format_list(eval_results.get('datasets', [])),
            'evaluation_metrics': self._format_list(eval_results.get('metrics', [])),
            'detailed_evaluation_results': self._format_detailed_results(eval_results),
            
            # Citation
            'citation_key': model_config['name'].replace('-', '_'),
            'authors': kwargs.get('authors', 'Wave Field LLM Team'),
            'year': datetime.now().year,
            'publisher': kwargs.get('publisher', 'Wave Field LLM'),
            'model_url': kwargs.get('model_url', 'https://github.com/wavefield-llm'),
            
            # Metadata
            'model_card_authors': kwargs.get('card_authors', 'Wave Field LLM Team'),
            'contact_info': kwargs.get('contact', 'contact@wavefield-llm.org'),
            'changelog_entry': kwargs.get('changelog', 'Initial release'),
            'repository_url': kwargs.get('repo_url', 'https://github.com/wavefield-llm'),
            'paper_url': kwargs.get('paper_url', 'N/A'),
            'demo_url': kwargs.get('demo_url', 'N/A'),
            'docs_url': kwargs.get('docs_url', 'https://wavefield-llm.org/docs'),
            'acknowledgments': kwargs.get('acknowledgments', 'Thanks to the open-source community.'),
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        return metadata
    
    def _estimate_parameters(self, arch: Dict[str, Any]) -> int:
        """Estimate number of parameters from architecture."""
        dim = arch['dim']
        n_layers = arch['n_layers']
        n_heads = arch['n_heads']
        vocab_size = arch['vocab_size']
        
        # Embedding parameters
        embedding_params = vocab_size * dim
        
        # Transformer layer parameters (approximate)
        # Attention: 4 * dim^2 (Q, K, V, O projections)
        # FFN: 8 * dim^2 (assuming 4x expansion)
        # Layer norm: 2 * dim
        layer_params = (4 * dim * dim) + (8 * dim * dim) + (2 * dim)
        transformer_params = n_layers * layer_params
        
        # Output layer
        output_params = dim * vocab_size
        
        total = embedding_params + transformer_params + output_params
        return total
    
    def _format_dataset_table(self, dataset_mix: Dict[str, float]) -> str:
        """Format dataset mixture as markdown table."""
        table = "| Dataset | Weight |\n|---------|--------|\n"
        for dataset, weight in sorted(dataset_mix.items(), key=lambda x: x[1], reverse=True):
            table += f"| {dataset} | {weight:.1%} |\n"
        return table
    
    def _format_benchmark_table(self, benchmarks: Dict[str, Any]) -> str:
        """Format benchmark results as markdown table."""
        if not benchmarks:
            return "| Benchmark | Score |\n|-----------|-------|\n| N/A | N/A |\n"
        
        table = "| Benchmark | Score |\n|-----------|-------|\n"
        for benchmark, score in sorted(benchmarks.items()):
            if isinstance(score, float):
                table += f"| {benchmark} | {score:.2f} |\n"
            else:
                table += f"| {benchmark} | {score} |\n"
        return table
    
    def _format_list(self, items: List[str]) -> str:
        """Format list as markdown bullet points."""
        if not items:
            return "- Not specified"
        return "\n".join(f"- {item}" for item in items)
    
    def _format_training_procedure(self, training_config: Dict[str, Any]) -> str:
        """Format training procedure description."""
        procedure = f"""
The model was trained using the following procedure:

1. **Data Preparation**: Mixed dataset with {len(training_config)} sources
2. **Optimization**: AdamW optimizer with learning rate {training_config['learning_rate']}
3. **Learning Rate Schedule**: {training_config['lr_decay_style']} decay with {training_config['warmup_steps']} warmup steps
4. **Regularization**: Weight decay {training_config['weight_decay']}, gradient clipping {training_config['grad_clip']}
5. **Mixed Precision**: {training_config['mixed_precision']} training for efficiency
6. **Checkpointing**: Saved every {training_config['save_interval']} steps
7. **Evaluation**: Validated every {training_config['eval_interval']} steps
"""
        return procedure.strip()
    
    def _format_detailed_results(self, eval_results: Dict[str, Any]) -> str:
        """Format detailed evaluation results."""
        if not eval_results:
            return "Detailed results not available."
        
        results = "### Detailed Results\n\n"
        
        for category, metrics in eval_results.items():
            if isinstance(metrics, dict):
                results += f"#### {category.replace('_', ' ').title()}\n\n"
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        results += f"- **{metric}**: {value:.4f}\n"
                    else:
                        results += f"- **{metric}**: {value}\n"
                results += "\n"
        
        return results
    
    def _fill_template(self, metadata: Dict[str, str]) -> str:
        """Fill template with metadata."""
        model_card = self.template
        
        for key, value in metadata.items():
            placeholder = f"{{{key}}}"
            model_card = model_card.replace(placeholder, str(value))
        
        return model_card


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate model card")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--training-log", required=True, help="Path to training log JSON")
    parser.add_argument("--eval-results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--output", required=True, help="Output path for model card")
    parser.add_argument("--template", help="Custom template path")
    parser.add_argument("--license", default="Apache 2.0", help="Model license")
    parser.add_argument("--authors", default="Wave Field LLM Team", help="Model authors")
    
    args = parser.parse_args()
    
    generator = ModelCardGenerator(template_path=args.template)
    
    generator.generate(
        config_path=args.config,
        training_log_path=args.training_log,
        eval_results_path=args.eval_results,
        output_path=args.output,
        license=args.license,
        authors=args.authors,
    )
    
    logger.info("Model card generation complete")


if __name__ == "__main__":
    main()

# Made with Bob

"""
Model Registry

Central registry for managing Wave Field LLM models.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import RegistryDatabase
from .metadata import ModelMetadata, VersionInfo


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central model registry."""
    
    def __init__(self, db_path: str = "model_registry.db"):
        """Initialize registry.
        
        Args:
            db_path: Path to database file
        """
        self.db = RegistryDatabase(db_path)
    
    def register(
        self,
        name: str,
        version: str,
        checkpoint_path: str,
        config_path: str,
        eval_results_path: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Register a new model.
        
        Args:
            name: Model name
            version: Model version
            checkpoint_path: Path to model checkpoint
            config_path: Path to model config
            eval_results_path: Path to evaluation results
            **kwargs: Additional metadata
        
        Returns:
            True if successful
        """
        logger.info(f"Registering model {name} version {version}")
        
        # Calculate checkpoint hash
        checkpoint_sha256 = self._calculate_sha256(checkpoint_path)
        checkpoint_size = Path(checkpoint_path).stat().st_size // (1024 * 1024)  # MB
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load eval results if provided
        benchmarks = {}
        perplexity = 0.0
        inference_speed = 0.0
        
        if eval_results_path:
            import json
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
                benchmarks = eval_results.get('benchmarks', {})
                perplexity = eval_results.get('summary', {}).get('val_perplexity', 0.0)
                inference_speed = eval_results.get('summary', {}).get('inference_speed', 0.0)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=config['model'].get('description', ''),
            architecture='wavefield-llm',
            num_parameters=self._estimate_parameters(config['model']['architecture']),
            hidden_dim=config['model']['architecture']['dim'],
            num_layers=config['model']['architecture']['n_layers'],
            num_heads=config['model']['architecture']['n_heads'],
            training_tokens=config['training']['total_tokens'],
            training_steps=config['training']['max_steps'],
            training_duration=kwargs.get('training_duration', ''),
            benchmarks=benchmarks,
            perplexity=perplexity,
            inference_speed=inference_speed,
            checkpoint_url=kwargs.get('checkpoint_url', ''),
            checkpoint_size_mb=checkpoint_size,
            checkpoint_sha256=checkpoint_sha256,
            license=kwargs.get('license', 'Apache 2.0'),
            tags=kwargs.get('tags', []),
            base_model=config['training'].get('resume_from'),
            specialization=config['model'].get('specialization'),
        )
        
        # Register in database
        success = self.db.register_model(metadata)
        
        if success:
            logger.info(f"Successfully registered {name} v{version}")
        else:
            logger.error(f"Failed to register {name} v{version}")
        
        return success
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get model metadata.
        
        Args:
            name: Model name
            version: Model version (latest if None)
        
        Returns:
            Model metadata or None
        """
        return self.db.get_model(name, version)
    
    def list(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all models.
        
        Args:
            tags: Filter by tags
        
        Returns:
            List of model summaries
        """
        return self.db.list_models(tags)
    
    def list_versions(self, name: str) -> List[VersionInfo]:
        """List all versions of a model.
        
        Args:
            name: Model name
        
        Returns:
            List of version info
        """
        return self.db.list_versions(name)
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search models.
        
        Args:
            query: Search query
        
        Returns:
            List of matching models
        """
        return self.db.search_models(query)
    
    def get_download_url(self, name: str, version: Optional[str] = None) -> Optional[str]:
        """Get download URL for a model.
        
        Args:
            name: Model name
            version: Model version (latest if None)
        
        Returns:
            Download URL or None
        """
        metadata = self.get(name, version)
        if metadata:
            return metadata.checkpoint_url
        return None
    
    def verify_checksum(self, checkpoint_path: str, expected_sha256: str) -> bool:
        """Verify checkpoint checksum.
        
        Args:
            checkpoint_path: Path to checkpoint file
            expected_sha256: Expected SHA-256 hash
        
        Returns:
            True if checksum matches
        """
        actual_sha256 = self._calculate_sha256(checkpoint_path)
        return actual_sha256 == expected_sha256
    
    def compare_versions(
        self,
        name: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """Compare two versions of a model.
        
        Args:
            name: Model name
            version1: First version
            version2: Second version
        
        Returns:
            Comparison results
        """
        meta1 = self.get(name, version1)
        meta2 = self.get(name, version2)
        
        if not meta1 or not meta2:
            return {'error': 'Version not found'}
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'parameter_diff': meta2.num_parameters - meta1.num_parameters,
            'perplexity_diff': meta2.perplexity - meta1.perplexity,
            'speed_diff': meta2.inference_speed - meta1.inference_speed,
            'benchmark_diffs': {},
        }
        
        # Compare benchmarks
        for benchmark in set(meta1.benchmarks.keys()) | set(meta2.benchmarks.keys()):
            score1 = meta1.benchmarks.get(benchmark, 0)
            score2 = meta2.benchmarks.get(benchmark, 0)
            comparison['benchmark_diffs'][benchmark] = score2 - score1
        
        return comparison
    
    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to file
        
        Returns:
            SHA-256 hash as hex string
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _estimate_parameters(self, arch: Dict[str, Any]) -> int:
        """Estimate number of parameters.
        
        Args:
            arch: Architecture config
        
        Returns:
            Estimated parameter count
        """
        dim = arch['dim']
        n_layers = arch['n_layers']
        vocab_size = arch['vocab_size']
        
        # Rough estimate
        embedding_params = vocab_size * dim
        layer_params = (4 * dim * dim + 8 * dim * dim + 2 * dim)
        transformer_params = n_layers * layer_params
        output_params = dim * vocab_size
        
        return embedding_params + transformer_params + output_params
    
    def close(self):
        """Close registry."""
        self.db.close()

# Made with Bob

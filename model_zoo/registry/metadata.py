"""
Model Registry Metadata Schemas

Defines metadata structures for model registry.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    
    # Basic information
    name: str
    version: str
    description: str
    
    # Architecture
    architecture: str = "wavefield-llm"
    num_parameters: int = 0
    hidden_dim: int = 0
    num_layers: int = 0
    num_heads: int = 0
    
    # Training
    training_tokens: int = 0
    training_steps: int = 0
    training_duration: str = ""
    
    # Performance
    benchmarks: Dict[str, float] = field(default_factory=dict)
    perplexity: float = 0.0
    inference_speed: float = 0.0
    
    # Files
    checkpoint_url: str = ""
    checkpoint_size_mb: int = 0
    checkpoint_sha256: str = ""
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    license: str = "Apache 2.0"
    tags: List[str] = field(default_factory=list)
    
    # Additional info
    base_model: Optional[str] = None
    specialization: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'architecture': self.architecture,
            'num_parameters': self.num_parameters,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'training_tokens': self.training_tokens,
            'training_steps': self.training_steps,
            'training_duration': self.training_duration,
            'benchmarks': self.benchmarks,
            'perplexity': self.perplexity,
            'inference_speed': self.inference_speed,
            'checkpoint_url': self.checkpoint_url,
            'checkpoint_size_mb': self.checkpoint_size_mb,
            'checkpoint_sha256': self.checkpoint_sha256,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'license': self.license,
            'tags': self.tags,
            'base_model': self.base_model,
            'specialization': self.specialization,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VersionInfo:
    """Version information for a model."""
    
    version: str
    created_at: str
    changelog: str
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version': self.version,
            'created_at': self.created_at,
            'changelog': self.changelog,
            'deprecated': self.deprecated,
        }

# Made with Bob

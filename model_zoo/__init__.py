"""
Wave Field LLM Model Zoo

Complete infrastructure for training, evaluating, documenting, and distributing
pretrained Wave Field LLM models.
"""

__version__ = "1.0.0"

from .loader import load_model, list_available_models, get_model_info
from .registry import ModelRegistry
from .train_pipeline import TrainingPipeline
from .evaluation import ModelEvaluator
from .distribution import ModelUploader, ModelDownloader

__all__ = [
    # Model loading
    'load_model',
    'list_available_models',
    'get_model_info',
    
    # Registry
    'ModelRegistry',
    
    # Training
    'TrainingPipeline',
    
    # Evaluation
    'ModelEvaluator',
    
    # Distribution
    'ModelUploader',
    'ModelDownloader',
]

# Made with Bob

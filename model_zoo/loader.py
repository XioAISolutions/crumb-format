#!/usr/bin/env python3
"""
Model Loader

Unified interface for loading Wave Field LLM models from various sources.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .registry import ModelRegistry
from .distribution import ModelDownloader


logger = logging.getLogger(__name__)


def load_model(
    model_name_or_path: str,
    version: Optional[str] = None,
    device: str = "cuda",
    quantization: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs
):
    """Load a Wave Field LLM model.
    
    This is the main entry point for loading models. It supports:
    - Loading from local path
    - Loading from HuggingFace Hub
    - Loading from model registry
    - Loading from S3/GCS URLs
    - Automatic download and caching
    - Quantization on load
    
    Args:
        model_name_or_path: Model name, path, or URL
        version: Model version (for registry models)
        device: Device to load model on
        quantization: Quantization mode (fp16, int8, int4)
        cache_dir: Cache directory for downloads
        **kwargs: Additional arguments
    
    Returns:
        Loaded model
    
    Examples:
        >>> # Load from HuggingFace
        >>> model = load_model("wavefield-llm/wavefield-small")
        
        >>> # Load specific version from registry
        >>> model = load_model("wavefield-small", version="1.0.0")
        
        >>> # Load with quantization
        >>> model = load_model("wavefield-small", quantization="fp16")
        
        >>> # Load from S3
        >>> model = load_model("s3://bucket/model.pt")
        
        >>> # Load from local path
        >>> model = load_model("./models/my_model.pt")
    """
    logger.info(f"Loading model: {model_name_or_path}")
    
    # Determine source type
    if Path(model_name_or_path).exists():
        # Local path
        model_path = model_name_or_path
        logger.info(f"Loading from local path: {model_path}")
    
    elif model_name_or_path.startswith(('s3://', 'gs://', 'http://', 'https://')):
        # URL - download first
        logger.info(f"Downloading from URL: {model_name_or_path}")
        downloader = ModelDownloader(cache_dir=cache_dir or "~/.cache/wavefield-llm")
        model_path = downloader.download(model_name_or_path)
    
    elif '/' in model_name_or_path:
        # HuggingFace Hub format (org/model)
        logger.info(f"Loading from HuggingFace Hub: {model_name_or_path}")
        model_path = _load_from_huggingface(model_name_or_path, cache_dir)
    
    else:
        # Model registry
        logger.info(f"Loading from model registry: {model_name_or_path}")
        model_path = _load_from_registry(model_name_or_path, version, cache_dir)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}
    else:
        state_dict = checkpoint
        config = {}
    
    # Create model
    from crumb_llm.model import WaveFieldLLM
    
    model = WaveFieldLLM(
        vocab_size=config.get('vocab_size', 50257),
        dim=config.get('dim', 512),
        n_layers=config.get('n_layers', 8),
        n_heads=config.get('n_heads', 8),
        **kwargs
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    # Apply quantization if requested
    if quantization:
        logger.info(f"Applying {quantization} quantization")
        model = _apply_quantization(model, quantization)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def _load_from_huggingface(model_id: str, cache_dir: Optional[str]) -> str:
    """Load model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID (org/model)
        cache_dir: Cache directory
    
    Returns:
        Path to downloaded model
    """
    try:
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id=model_id,
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
        )
        
        return model_path
        
    except ImportError:
        raise ImportError("huggingface_hub is required to load from HuggingFace Hub. Install with: pip install huggingface_hub")
    except Exception as e:
        raise RuntimeError(f"Failed to load from HuggingFace Hub: {e}")


def _load_from_registry(
    model_name: str,
    version: Optional[str],
    cache_dir: Optional[str]
) -> str:
    """Load model from registry.
    
    Args:
        model_name: Model name
        version: Model version
        cache_dir: Cache directory
    
    Returns:
        Path to model
    """
    registry = ModelRegistry()
    
    # Get model metadata
    metadata = registry.get(model_name, version)
    
    if not metadata:
        raise ValueError(f"Model {model_name} version {version or 'latest'} not found in registry")
    
    # Check cache first
    downloader = ModelDownloader(cache_dir=cache_dir or "~/.cache/wavefield-llm")
    cached_path = downloader.get_cached_path(model_name, metadata.version)
    
    if cached_path:
        logger.info(f"Using cached model: {cached_path}")
        return cached_path
    
    # Download from URL
    if not metadata.checkpoint_url:
        raise ValueError(f"No download URL available for {model_name} v{metadata.version}")
    
    model_path = downloader.download(
        url=metadata.checkpoint_url,
        checksum=metadata.checkpoint_sha256,
    )
    
    return model_path


def _apply_quantization(model, quantization: str):
    """Apply quantization to model.
    
    Args:
        model: Model to quantize
        quantization: Quantization mode (fp16, int8, int4)
    
    Returns:
        Quantized model
    """
    if quantization == "fp16":
        model = model.half()
    
    elif quantization == "int8":
        # Placeholder - in production, use torch.quantization
        logger.warning("INT8 quantization not fully implemented")
        # model = torch.quantization.quantize_dynamic(
        #     model, {torch.nn.Linear}, dtype=torch.qint8
        # )
    
    elif quantization == "int4":
        # Placeholder - in production, use bitsandbytes or similar
        logger.warning("INT4 quantization not fully implemented")
    
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    
    return model


def list_available_models() -> Dict[str, Any]:
    """List all available models in the registry.
    
    Returns:
        Dictionary of available models
    """
    registry = ModelRegistry()
    models = registry.list()
    
    return {
        'models': models,
        'count': len(models),
    }


def get_model_info(model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Get information about a model.
    
    Args:
        model_name: Model name
        version: Model version (latest if None)
    
    Returns:
        Model information
    """
    registry = ModelRegistry()
    metadata = registry.get(model_name, version)
    
    if not metadata:
        raise ValueError(f"Model {model_name} not found")
    
    return metadata.to_dict()

# Made with Bob

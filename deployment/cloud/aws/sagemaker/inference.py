"""
Wave Field LLM - SageMaker Custom Inference Handler
Handles model loading, preprocessing, inference, and postprocessing
"""

import json
import logging
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model and tokenizer from the model directory.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Dictionary containing model and tokenizer
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    model.eval()
    
    logger.info("Model loaded successfully")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device
    }


def input_fn(request_body: str, content_type: str = "application/json") -> Dict[str, Any]:
    """
    Deserialize and prepare the input data.
    
    Args:
        request_body: The request body
        content_type: The content type of the request
        
    Returns:
        Parsed input data
    """
    if content_type == "application/json":
        input_data = json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
    return input_data


def predict_fn(input_data: Dict[str, Any], model_dict: Dict[str, Any]) -> List[str]:
    """
    Perform inference on the input data.
    
    Args:
        input_data: Preprocessed input data
        model_dict: Dictionary containing model and tokenizer
        
    Returns:
        Model predictions
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    
    # Extract inputs and parameters
    inputs = input_data.get("inputs", "")
    if isinstance(inputs, str):
        inputs = [inputs]
    
    parameters = input_data.get("parameters", {})
    
    # Default generation parameters
    max_length = parameters.get("max_length", 512)
    max_new_tokens = parameters.get("max_new_tokens", None)
    temperature = parameters.get("temperature", 0.7)
    top_k = parameters.get("top_k", 50)
    top_p = parameters.get("top_p", 0.95)
    do_sample = parameters.get("do_sample", True)
    num_return_sequences = parameters.get("num_return_sequences", 1)
    repetition_penalty = parameters.get("repetition_penalty", 1.0)
    length_penalty = parameters.get("length_penalty", 1.0)
    
    # Tokenize inputs
    encoded_inputs = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate
    with torch.no_grad():
        if max_new_tokens:
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                **encoded_inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    # Decode outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_texts


def output_fn(predictions: List[str], accept: str = "application/json") -> str:
    """
    Serialize the predictions.
    
    Args:
        predictions: Model predictions
        accept: The accept content type
        
    Returns:
        Serialized predictions
    """
    if accept == "application/json":
        return json.dumps({
            "generated_texts": predictions,
            "num_sequences": len(predictions)
        })
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Batch inference support
def transform_fn(model_dict: Dict[str, Any], request_body: bytes, content_type: str, accept: str) -> bytes:
    """
    Complete inference pipeline for batch processing.
    
    Args:
        model_dict: Dictionary containing model and tokenizer
        request_body: The request body
        content_type: The content type of the request
        accept: The accept content type
        
    Returns:
        Serialized predictions
    """
    # Parse input
    input_data = input_fn(request_body.decode('utf-8'), content_type)
    
    # Predict
    predictions = predict_fn(input_data, model_dict)
    
    # Serialize output
    output = output_fn(predictions, accept)
    
    return output.encode('utf-8')


# Health check
def ping() -> bool:
    """
    Health check endpoint.
    
    Returns:
        True if service is healthy
    """
    return True

# Made with Bob

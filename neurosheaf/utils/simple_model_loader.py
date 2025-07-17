"""Simple model loader utility for neurosheaf.

This module provides a clean, straightforward interface for loading PyTorch models
with state_dict format. It requires specifying the model class, making it reliable
and transparent.

Key features:
- Only supports state_dict format (.pth files)
- Requires model class as parameter
- Handles device placement automatically
- Saves/loads metadata (epoch, loss, etc.)
- Simple interface with clear documentation
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Dict, Any, Type, Optional
from datetime import datetime

from .device import detect_optimal_device
from .logging import setup_logger
from .exceptions import ValidationError

logger = setup_logger("neurosheaf.simple_model_loader")


def load_model(
    model_class: Type[nn.Module],
    weights_path: Union[str, Path],
    device: Optional[str] = None,
    **model_kwargs
) -> nn.Module:
    """Load a PyTorch model from state_dict weights.
    
    This function creates a new instance of the model class and loads
    the saved state_dict weights into it.
    
    Args:
        model_class: The model class to instantiate (e.g., MyModel, ResNet)
        weights_path: Path to the saved state_dict file (.pth)
        device: Device to load model on ('cpu', 'cuda', 'mps', or None for auto)
        **model_kwargs: Additional keyword arguments for model constructor
        
    Returns:
        nn.Module: The loaded model in evaluation mode
        
    Raises:
        ValidationError: If the file doesn't exist or loading fails
        
    Examples:
        >>> # Basic usage
        >>> model = load_model(MyModel, "model.pth")
        
        >>> # With model constructor arguments
        >>> model = load_model(MyModel, "model.pth", input_size=128, num_classes=10)
        
        >>> # Specify device
        >>> model = load_model(MyModel, "model.pth", device="cpu")
    """
    weights_path = Path(weights_path)
    
    # Validate file exists
    if not weights_path.exists():
        raise ValidationError(f"Model file not found: {weights_path}")
    
    # Validate file extension
    if weights_path.suffix != '.pth':
        raise ValidationError(f"Expected .pth file, got: {weights_path.suffix}")
    
    # Detect device
    if device is None:
        device = detect_optimal_device()
    else:
        device = torch.device(device)
    
    logger.info(f"Loading model from {weights_path}")
    logger.info(f"Model class: {model_class.__name__}")
    logger.info(f"Device: {device}")
    
    try:
        # Create model instance
        model = model_class(**model_kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        
        # Extract state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # Log metadata if available
                if 'epoch' in checkpoint:
                    logger.info(f"Loaded from epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    logger.info(f"Training loss: {checkpoint['loss']:.4f}")
                if 'accuracy' in checkpoint:
                    logger.info(f"Training accuracy: {checkpoint['accuracy']:.4f}")
                    
            else:
                # Assume the dict is the state_dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state_dict into model
        model.load_state_dict(state_dict)
        
        # Move to device and set eval mode
        model = model.to(device)
        model.eval()
        
        # Log success
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Successfully loaded model with {param_count:,} parameters")
        
        return model
        
    except Exception as e:
        raise ValidationError(f"Failed to load model: {e}")


def save_model(
    model: nn.Module,
    weights_path: Union[str, Path],
    **metadata
) -> Path:
    """Save a PyTorch model's state_dict with metadata.
    
    This function saves the model's state_dict along with optional metadata
    in a format that can be loaded with load_model().
    
    Args:
        model: PyTorch model to save
        weights_path: Path to save the model (.pth extension will be added if missing)
        **metadata: Additional metadata to save (epoch, loss, accuracy, etc.)
        
    Returns:
        Path: Path to the saved model file
        
    Examples:
        >>> # Basic save
        >>> save_model(model, "my_model.pth")
        
        >>> # Save with training metadata
        >>> save_model(model, "my_model.pth", epoch=100, loss=0.123, accuracy=0.95)
        
        >>> # Save with custom metadata
        >>> save_model(model, "my_model.pth", optimizer="Adam", lr=0.001, notes="Best model")
    """
    weights_path = Path(weights_path)
    
    # Ensure .pth extension
    if weights_path.suffix != '.pth':
        weights_path = weights_path.with_suffix('.pth')
    
    # Create parent directory if it doesn't exist
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'model_class_name': model.__class__.__name__,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'save_timestamp': datetime.now().isoformat(),
        **metadata
    }
    
    logger.info(f"Saving model to {weights_path}")
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Parameters: {checkpoint_data['model_parameters']:,}")
    
    if metadata:
        logger.info(f"Metadata: {metadata}")
    
    try:
        torch.save(checkpoint_data, weights_path)
        logger.info("Model saved successfully")
        return weights_path
        
    except Exception as e:
        raise ValidationError(f"Failed to save model: {e}")


def list_model_info(weights_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a saved model file.
    
    This function inspects a saved model file and returns information
    about its contents, including metadata and architecture details.
    
    Args:
        weights_path: Path to the saved model file
        
    Returns:
        Dict containing model information
        
    Examples:
        >>> info = list_model_info("my_model.pth")
        >>> print(f"Model class: {info['model_class_name']}")
        >>> print(f"Parameters: {info['model_parameters']:,}")
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise ValidationError(f"Model file not found: {weights_path}")
    
    logger.info(f"Inspecting model file: {weights_path}")
    
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        # Extract basic info
        info = {
            'file_path': str(weights_path),
            'file_size_mb': weights_path.stat().st_size / (1024 * 1024),
        }
        
        if isinstance(checkpoint, dict):
            # Extract metadata
            for key, value in checkpoint.items():
                if key == 'model_state_dict':
                    # Analyze state_dict
                    state_dict = value
                    info['num_layers'] = len(state_dict)
                    info['layer_names'] = list(state_dict.keys())
                    info['layer_shapes'] = {name: list(tensor.shape) for name, tensor in state_dict.items()}
                    
                    # Calculate parameters if not already present
                    if 'model_parameters' not in checkpoint:
                        info['model_parameters'] = sum(tensor.numel() for tensor in state_dict.values())
                    
                else:
                    # Store other metadata
                    info[key] = value
        else:
            # Direct state_dict
            info['num_layers'] = len(checkpoint)
            info['layer_names'] = list(checkpoint.keys())
            info['layer_shapes'] = {name: list(tensor.shape) for name, tensor in checkpoint.items()}
            info['model_parameters'] = sum(tensor.numel() for tensor in checkpoint.values())
        
        # Log summary
        logger.info(f"Model information:")
        logger.info(f"  File size: {info['file_size_mb']:.2f} MB")
        logger.info(f"  Parameters: {info.get('model_parameters', 'Unknown'):,}")
        logger.info(f"  Layers: {info.get('num_layers', 'Unknown')}")
        
        if 'model_class_name' in info:
            logger.info(f"  Model class: {info['model_class_name']}")
        if 'epoch' in info:
            logger.info(f"  Epoch: {info['epoch']}")
        if 'accuracy' in info:
            logger.info(f"  Accuracy: {info['accuracy']:.4f}")
        
        return info
        
    except Exception as e:
        raise ValidationError(f"Failed to read model file: {e}")


def validate_model_compatibility(
    model_class: Type[nn.Module],
    weights_path: Union[str, Path],
    **model_kwargs
) -> bool:
    """Check if a model class is compatible with saved weights.
    
    This function attempts to load the weights into a model instance
    to verify compatibility without fully loading the model.
    
    Args:
        model_class: The model class to test
        weights_path: Path to the saved weights
        **model_kwargs: Model constructor arguments
        
    Returns:
        bool: True if compatible, False otherwise
        
    Examples:
        >>> # Check if model class matches saved weights
        >>> is_compatible = validate_model_compatibility(MyModel, "model.pth")
        >>> if is_compatible:
        >>>     model = load_model(MyModel, "model.pth")
    """
    try:
        model = load_model(model_class, weights_path, device='cpu', **model_kwargs)
        logger.info(f"Model class {model_class.__name__} is compatible with {weights_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Model class {model_class.__name__} is not compatible with {weights_path}: {e}")
        return False


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of a loaded model.
    
    Args:
        model: PyTorch model to summarize
        
    Returns:
        Dict containing model summary information
    """
    summary = {
        'model_class': model.__class__.__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
        'device': str(next(model.parameters()).device),
        'training_mode': model.training,
    }
    
    return summary
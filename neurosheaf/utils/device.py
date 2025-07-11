"""Device detection and management utilities.

This module provides centralized device detection logic to avoid duplication
across the codebase and ensure consistent device handling.
"""

import platform
import torch
from typing import Optional, Union

from .logging import setup_logger

logger = setup_logger(__name__)


def detect_optimal_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Detect optimal device for computation across all platforms.
    
    This function centralizes device detection logic to ensure consistency
    across all CKA and sheaf modules.
    
    Args:
        device: Optional device specification. If None, auto-detects optimal device.
        
    Returns:
        torch.device: The optimal device for computation
        
    Device Selection Priority:
        1. Explicit device specification (if provided)
        2. MPS (Mac Metal Performance Shaders) if available on Darwin
        3. CUDA if available on other platforms
        4. CPU as fallback
    """
    if device is not None:
        return torch.device(device)
    
    # Mac-specific device detection
    if platform.system() == "Darwin":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.debug("Using MPS device for Mac acceleration")
            return torch.device("mps")
        else:
            logger.debug("MPS not available, using CPU on Mac")
            return torch.device("cpu")
    
    # Other platforms
    if torch.cuda.is_available():
        logger.debug("Using CUDA device")
        return torch.device("cuda")
    
    logger.debug("Using CPU device (fallback)")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Get comprehensive device information for logging and debugging.
    
    Returns:
        Dictionary with device information including:
        - platform: Operating system
        - is_mac: Whether running on macOS
        - is_apple_silicon: Whether running on Apple Silicon
        - cuda_available: Whether CUDA is available
        - mps_available: Whether MPS is available
        - optimal_device: Auto-detected optimal device
    """
    is_mac = platform.system() == "Darwin"
    is_apple_silicon = platform.processor() == "arm"
    
    device_info = {
        'platform': platform.system(),
        'is_mac': is_mac,
        'is_apple_silicon': is_apple_silicon,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if is_mac else False,
        'optimal_device': str(detect_optimal_device())
    }
    
    return device_info


def should_use_cpu_fallback(device: torch.device, operation: str = "svd") -> bool:
    """Determine if CPU fallback should be used for specific operations.
    
    Some operations have known numerical stability issues on certain devices.
    This function centralizes the logic for when to use CPU fallback.
    
    Args:
        device: The current device
        operation: The operation being performed (e.g., 'svd', 'qr', 'eigh')
        
    Returns:
        bool: Whether to use CPU fallback for the operation
        
    Known Issues:
        - SVD on MPS has documented numerical stability issues (PyTorch #78099)
        - QR decomposition on MPS can have similar issues
        - Eigendecomposition on MPS may require CPU fallback
    """
    if device.type == 'mps':
        # MPS has known issues with several linear algebra operations
        cpu_fallback_operations = {'svd', 'qr', 'eigh', 'eigvals', 'cond', 'pinv'}
        return operation.lower() in cpu_fallback_operations
    
    return False


def safe_to_device(tensor: torch.Tensor, device: torch.device, operation: str = "general") -> torch.Tensor:
    """Safely move tensor to device with automatic CPU fallback for problematic operations.
    
    Args:
        tensor: Tensor to move
        device: Target device  
        operation: Operation that will be performed (for fallback logic)
        
    Returns:
        Tensor on appropriate device (may be CPU if fallback is needed)
    """
    if should_use_cpu_fallback(device, operation):
        logger.debug(f"Using CPU fallback for {operation} operation due to device limitations")
        return tensor.cpu()
    
    return tensor.to(device)


def log_device_warning_once(device: torch.device, operation: str, warning_key: str = None) -> None:
    """Log device-specific warnings only once to avoid spam.
    
    Args:
        device: Device being used
        operation: Operation being performed
        warning_key: Optional key for the warning (defaults to f"{device.type}_{operation}")
    """
    if warning_key is None:
        warning_key = f"{device.type}_{operation}"
    
    # Use a simple attribute-based mechanism to track warnings
    warning_attr = f"_warned_{warning_key}"
    
    if not hasattr(log_device_warning_once, warning_attr):
        if device.type == 'mps' and operation in ['svd', 'qr', 'eigh']:
            logger.warning(
                f"Using CPU fallback for {operation} on MPS device due to known numerical "
                f"stability issues. This is a documented PyTorch limitation (GitHub #78099)."
            )
        
        # Mark this warning as shown
        setattr(log_device_warning_once, warning_attr, True)


def clear_device_cache(device: torch.device) -> None:
    """Clear device cache to free memory.
    
    Args:
        device: Device to clear cache for
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    # CPU doesn't have a cache to clear
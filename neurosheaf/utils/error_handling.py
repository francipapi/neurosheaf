"""Enhanced error handling utilities for Neurosheaf.

This module provides centralized error handling patterns, retry mechanisms,
and safe operation wrappers to improve robustness across the codebase.
"""

import functools
import time
from typing import Any, Callable, Optional, Type, Union, Tuple
import warnings

import torch

from .device import should_use_cpu_fallback, log_device_warning_once
from .exceptions import ComputationError, ValidationError
from .logging import setup_logger

logger = setup_logger(__name__)


def safe_torch_operation(
    operation: str,
    fallback_device: str = "cpu",
    max_retries: int = 2,
    retry_delay: float = 0.1
) -> Callable:
    """Decorator for safe torch operations with automatic device fallback.
    
    This decorator handles common PyTorch operations that may fail on certain
    devices (especially MPS) and provides automatic CPU fallback.
    
    Args:
        operation: Name of the operation (for logging)
        fallback_device: Device to fallback to (default: "cpu")
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        Decorator function
        
    Example:
        @safe_torch_operation("svd")
        def compute_svd(tensor):
            return torch.linalg.svd(tensor)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract device from first tensor argument
            device = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    device = arg.device
                    break
            
            if device is None:
                # No tensor found, proceed normally
                return func(*args, **kwargs)
            
            # Check if we should use CPU fallback for this operation
            if should_use_cpu_fallback(device, operation):
                log_device_warning_once(device, operation)
                
                # Move tensors to CPU, execute, then move result back
                cpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        cpu_args.append(arg.cpu())
                    else:
                        cpu_args.append(arg)
                
                cpu_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        cpu_kwargs[key] = value.cpu()
                    else:
                        cpu_kwargs[key] = value
                
                # Execute on CPU
                result = func(*cpu_args, **cpu_kwargs)
                
                # Move result back to original device
                if isinstance(result, torch.Tensor):
                    return result.to(device)
                elif isinstance(result, (tuple, list)):
                    return type(result)(
                        item.to(device) if isinstance(item, torch.Tensor) else item
                        for item in result
                    )
                else:
                    return result
            
            # Try normal execution with retries
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    last_exception = e
                    
                    # Check if this is a known MPS issue
                    if device.type == 'mps' and any(keyword in str(e).lower() for keyword in 
                                                   ['mps', 'metal', 'not implemented', 'unsupported']):
                        logger.warning(f"{operation} failed on MPS (attempt {attempt + 1}): {e}")
                        
                        if attempt < max_retries:
                            logger.info(f"Retrying {operation} on CPU...")
                            # Force CPU fallback for remaining attempts
                            cpu_args = []
                            for arg in args:
                                if isinstance(arg, torch.Tensor):
                                    cpu_args.append(arg.cpu())
                                else:
                                    cpu_args.append(arg)
                            
                            cpu_kwargs = {}
                            for key, value in kwargs.items():
                                if isinstance(value, torch.Tensor):
                                    cpu_kwargs[key] = value.cpu()
                                else:
                                    cpu_kwargs[key] = value
                            
                            try:
                                result = func(*cpu_args, **cpu_kwargs)
                                # Move result back to original device
                                if isinstance(result, torch.Tensor):
                                    return result.to(device)
                                elif isinstance(result, (tuple, list)):
                                    return type(result)(
                                        item.to(device) if isinstance(item, torch.Tensor) else item
                                        for item in result
                                    )
                                else:
                                    return result
                            except Exception as cpu_e:
                                logger.warning(f"CPU fallback also failed: {cpu_e}")
                                if attempt < max_retries:
                                    time.sleep(retry_delay)
                                continue
                        break
                    else:
                        logger.warning(f"{operation} failed (attempt {attempt + 1}): {e}")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                        else:
                            break
                except Exception as e:
                    last_exception = e
                    logger.error(f"Unexpected error in {operation}: {e}")
                    break
            
            # If we get here, all retries failed
            raise ComputationError(f"{operation} failed after {max_retries + 1} attempts: {last_exception}")
        
        return wrapper
    return decorator


def safe_file_operation(
    operation: str,
    max_retries: int = 3,
    retry_delay: float = 0.1,
    fallback_value: Any = None
) -> Callable:
    """Decorator for safe file operations with retry logic.
    
    Args:
        operation: Name of the operation (for logging)
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        fallback_value: Value to return if all retries fail (if None, raises exception)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError, PermissionError) as e:
                    last_exception = e
                    logger.warning(f"{operation} failed (attempt {attempt + 1}): {e}")
                    
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    else:
                        break
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error in {operation}: {e}")
                    last_exception = e
                    break
            
            # All retries failed
            if fallback_value is not None:
                logger.warning(f"{operation} failed after {max_retries + 1} attempts, using fallback value")
                return fallback_value
            else:
                raise ComputationError(f"{operation} failed after {max_retries + 1} attempts: {last_exception}")
        
        return wrapper
    return decorator


def validate_tensor_properties(
    tensor: torch.Tensor,
    name: str = "tensor",
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_dim: Optional[int] = None,
    max_dim: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    positive_semidefinite: bool = False
) -> None:
    """Validate tensor properties and raise informative errors.
    
    Args:
        tensor: Tensor to validate
        name: Name of the tensor (for error messages)
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        min_dim: Minimum number of dimensions
        max_dim: Maximum number of dimensions
        expected_shape: Expected exact shape
        positive_semidefinite: Whether to check if matrix is positive semidefinite
        
    Raises:
        ValidationError: If validation fails
    """
    # Check for NaN/Inf values
    if not allow_nan and torch.isnan(tensor).any():
        raise ValidationError(f"{name} contains NaN values")
    
    if not allow_inf and torch.isinf(tensor).any():
        raise ValidationError(f"{name} contains infinite values")
    
    # Check dimensions
    if min_dim is not None and tensor.ndim < min_dim:
        raise ValidationError(f"{name} has {tensor.ndim} dimensions, expected at least {min_dim}")
    
    if max_dim is not None and tensor.ndim > max_dim:
        raise ValidationError(f"{name} has {tensor.ndim} dimensions, expected at most {max_dim}")
    
    # Check exact shape
    if expected_shape is not None and tensor.shape != expected_shape:
        raise ValidationError(f"{name} has shape {tensor.shape}, expected {expected_shape}")
    
    # Check positive semidefinite property
    if positive_semidefinite:
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValidationError(f"{name} must be square matrix for PSD check, got shape {tensor.shape}")
        
        # Check eigenvalues (use CPU fallback if needed)
        try:
            eigenvals = torch.linalg.eigvals(tensor).real
            min_eigenval = torch.min(eigenvals).item()
            
            if min_eigenval < -1e-10:  # Allow small numerical errors
                raise ValidationError(f"{name} is not positive semidefinite (min eigenvalue: {min_eigenval:.2e})")
                
        except Exception as e:
            logger.warning(f"Could not verify PSD property for {name}: {e}")


def handle_numerical_instability(
    func: Callable,
    *args,
    epsilon: float = 1e-8,
    regularization: float = 0.0,
    max_condition_number: float = 1e12,
    **kwargs
) -> Any:
    """Handle numerical instability in matrix operations.
    
    Args:
        func: Function to call
        *args: Arguments to pass to function
        epsilon: Small value for numerical stability
        regularization: Regularization to add to diagonal
        max_condition_number: Maximum allowed condition number
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Result of function call with stability measures applied
        
    Raises:
        ComputationError: If numerical instability cannot be resolved
    """
    # Try normal execution first
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "singular" in str(e).lower() or "not invertible" in str(e).lower():
            logger.warning(f"Numerical instability detected: {e}")
            
            # Apply regularization to matrix arguments
            regularized_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.ndim == 2 and arg.shape[0] == arg.shape[1]:
                    # Add regularization to diagonal of square matrices
                    n = arg.shape[0]
                    reg_matrix = arg + regularization * torch.eye(n, device=arg.device, dtype=arg.dtype)
                    regularized_args.append(reg_matrix)
                    logger.info(f"Applied regularization {regularization} to matrix")
                else:
                    regularized_args.append(arg)
            
            # Retry with regularized matrices
            try:
                return func(*regularized_args, **kwargs)
            except RuntimeError as reg_e:
                logger.error(f"Regularization failed to resolve instability: {reg_e}")
                raise ComputationError(f"Numerical instability could not be resolved: {reg_e}")
        else:
            # Re-raise non-numerical errors
            raise


def with_error_context(context_info: dict) -> Callable:
    """Decorator to add context information to errors.
    
    Args:
        context_info: Dictionary with context information to add to errors
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add context to the error message
                context_str = ", ".join(f"{k}={v}" for k, v in context_info.items())
                enhanced_message = f"{str(e)} (Context: {context_str})"
                
                # Preserve the original exception type but enhance the message
                if isinstance(e, (ComputationError, ValidationError)):
                    raise type(e)(enhanced_message)
                else:
                    raise ComputationError(enhanced_message) from e
        
        return wrapper
    return decorator


# Commonly used safe operations
safe_svd = safe_torch_operation("svd")
safe_qr = safe_torch_operation("qr") 
safe_eigh = safe_torch_operation("eigh")
safe_eigvals = safe_torch_operation("eigvals")
safe_cond = safe_torch_operation("cond")
safe_pinv = safe_torch_operation("pinv")
safe_solve_triangular = safe_torch_operation("solve_triangular")


# File operation decorators
safe_json_load = safe_file_operation("json_load", fallback_value={})
safe_json_save = safe_file_operation("json_save")
safe_file_read = safe_file_operation("file_read", fallback_value="")
safe_file_write = safe_file_operation("file_write")
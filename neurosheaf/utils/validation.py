"""Validation utilities for activation tensors and CKA computation.

This module provides input validation functions to ensure data integrity
and proper formatting for CKA computations.
"""

import warnings
from typing import Tuple, Optional, Union, Dict

import numpy as np
import torch

from .exceptions import ValidationError


def validate_activations(
    X: torch.Tensor,
    Y: torch.Tensor,
    min_samples: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate activation tensors for CKA computation.
    
    Args:
        X: First activation matrix [n_samples, n_features_x]
        Y: Second activation matrix [n_samples, n_features_y]
        min_samples: Minimum number of samples required
        
    Returns:
        Tuple of validated (X, Y) tensors
        
    Raises:
        ValidationError: If validation fails
    """
    # Type checking
    if not isinstance(X, torch.Tensor):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        else:
            raise ValidationError(f"X must be torch.Tensor or numpy.ndarray, got {type(X)}")
    
    if not isinstance(Y, torch.Tensor):
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        else:
            raise ValidationError(f"Y must be torch.Tensor or numpy.ndarray, got {type(Y)}")
    
    # Dimension checking
    if X.dim() != 2:
        raise ValidationError(f"X must be 2D tensor, got {X.dim()}D")
    
    if Y.dim() != 2:
        raise ValidationError(f"Y must be 2D tensor, got {Y.dim()}D")
    
    # Sample dimension must match
    if X.shape[0] != Y.shape[0]:
        raise ValidationError(
            f"Sample dimensions must match: X has {X.shape[0]} samples, "
            f"Y has {Y.shape[0]} samples"
        )
    
    # Minimum samples check
    n_samples = X.shape[0]
    if n_samples < min_samples:
        raise ValidationError(
            f"Need at least {min_samples} samples for CKA computation, got {n_samples}"
        )
    
    # Check for NaN or Inf
    if torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)):
        raise ValidationError("X contains NaN or Inf values")
    
    if torch.any(torch.isnan(Y)) or torch.any(torch.isinf(Y)):
        raise ValidationError("Y contains NaN or Inf values")
    
    # Ensure float type for numerical stability
    if X.dtype not in [torch.float32, torch.float64]:
        X = X.float()
    
    if Y.dtype not in [torch.float32, torch.float64]:
        Y = Y.float()
    
    return X, Y


def validate_no_preprocessing(
    activations: Dict[str, torch.Tensor],
    warn_threshold: float = 1e-6
) -> None:
    """Warn if activations appear to be pre-centered.
    
    For debiased CKA, activations should NOT be centered to avoid
    double-centering issues.
    
    Args:
        activations: Dictionary mapping layer names to activation tensors
        warn_threshold: Threshold for mean norm to trigger warning
    """
    for name, act in activations.items():
        if not isinstance(act, torch.Tensor):
            continue
            
        # Check if mean is close to zero (indicating pre-centering)
        mean_norm = torch.norm(act.mean(dim=0))
        
        if mean_norm < warn_threshold:
            warnings.warn(
                f"Layer '{name}' appears to be centered (mean≈0). "
                f"For debiased CKA, use raw activations to avoid double-centering. "
                f"See Murphy et al. (2024) and updated-debiased-cka-v3.md.",
                UserWarning
            )


def validate_gram_matrix(K: torch.Tensor, name: str = "K") -> None:
    """Validate properties of a Gram matrix.
    
    Args:
        K: Gram matrix to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(K, torch.Tensor):
        raise ValidationError(f"{name} must be torch.Tensor")
    
    if K.dim() != 2:
        raise ValidationError(f"{name} must be 2D tensor, got {K.dim()}D")
    
    if K.shape[0] != K.shape[1]:
        raise ValidationError(
            f"{name} must be square matrix, got shape {K.shape}"
        )
    
    # Check symmetry (within numerical tolerance)
    if not torch.allclose(K, K.T, atol=1e-6):
        raise ValidationError(f"{name} must be symmetric")
    
    # Check for NaN or Inf
    if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
        raise ValidationError(f"{name} contains NaN or Inf values")
    
    # Gram matrices should be positive semi-definite
    # (all eigenvalues >= 0)
    try:
        eigenvalues = torch.linalg.eigvalsh(K)
        min_eigenvalue = eigenvalues.min().item()
        if min_eigenvalue < -1e-6:  # Allow small numerical errors
            warnings.warn(
                f"{name} has negative eigenvalue {min_eigenvalue:.2e}, "
                f"may not be positive semi-definite"
            )
    except Exception:
        # Skip eigenvalue check if it fails
        pass


def validate_sample_indices(
    indices: torch.Tensor,
    n_total: int,
    expected_size: Optional[int] = None
) -> torch.Tensor:
    """Validate sample indices for subsampling.
    
    Args:
        indices: Sample indices
        n_total: Total number of samples
        expected_size: Expected number of indices (optional)
        
    Returns:
        Validated indices tensor
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(indices, torch.Tensor):
        if isinstance(indices, (list, np.ndarray)):
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            raise ValidationError(
                f"Indices must be torch.Tensor, list, or numpy.ndarray, got {type(indices)}"
            )
    
    if indices.dtype not in [torch.int32, torch.int64]:
        indices = indices.long()
    
    # Check bounds
    if torch.any(indices < 0) or torch.any(indices >= n_total):
        raise ValidationError(
            f"Sample indices must be in range [0, {n_total-1}]"
        )
    
    # Check for duplicates
    if len(torch.unique(indices)) != len(indices):
        raise ValidationError("Sample indices contain duplicates")
    
    # Check expected size
    if expected_size is not None and len(indices) != expected_size:
        raise ValidationError(
            f"Expected {expected_size} indices, got {len(indices)}"
        )
    
    return indices


def validate_memory_limit(memory_limit_mb: float) -> None:
    """Validate memory limit is reasonable.
    
    Args:
        memory_limit_mb: Memory limit in megabytes
        
    Raises:
        ValidationError: If memory limit is invalid
    """
    if memory_limit_mb <= 0:
        raise ValidationError(f"Memory limit must be positive, got {memory_limit_mb}")
    
    if memory_limit_mb < 10:
        warnings.warn(
            f"Memory limit {memory_limit_mb}MB is very small, "
            f"may not be sufficient for CKA computation"
        )
    
    if memory_limit_mb > 100000:  # 100GB
        warnings.warn(
            f"Memory limit {memory_limit_mb}MB is very large, "
            f"consider if this is intentional"
        )
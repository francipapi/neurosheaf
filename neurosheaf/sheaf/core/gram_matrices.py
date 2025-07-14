"""Gram matrix operations for neural network analysis.

This module provides utilities for computing and validating Gram matrices
from neural network activations. Gram matrices K = X @ X.T capture the
inner product structure between samples and form the mathematical
foundation for sheaf construction.

All functions operate on raw (uncentered) activations following the
debiased CKA approach that avoids double-centering issues.
"""

from typing import Dict, Any, Optional

import torch


def compute_gram_matrix(activation: torch.Tensor, validate: bool = True) -> torch.Tensor:
    """Compute Gram matrix from activation tensor.
    
    Computes K = X @ X.T where X is the activation matrix. Uses raw activations
    without centering to avoid numerical issues in debiased CKA computation.
    
    Args:
        activation: Activation tensor (n_samples, n_features) or (n_samples, ...)
        validate: Whether to validate matrix properties
        
    Returns:
        Gram matrix K of shape (n_samples, n_samples)
        
    Raises:
        ValueError: If activation tensor has invalid shape
    """
    if activation.ndim < 2:
        raise ValueError(f"Activation must be at least 2D, got shape {activation.shape}")
    
    # Flatten to 2D if needed (samples × features)
    if activation.ndim > 2:
        n_samples = activation.shape[0]
        activation = activation.view(n_samples, -1)
    
    # Compute Gram matrix K = X @ X.T (raw activations, no centering)
    gram_matrix = activation @ activation.T
    
    if validate:
        _validate_gram_matrix(gram_matrix)
    
    return gram_matrix


def compute_gram_matrices_from_activations(
    activations: Dict[str, torch.Tensor], 
    validate: bool = True
) -> Dict[str, torch.Tensor]:
    """Compute Gram matrices for a dictionary of activations.
    
    Args:
        activations: Dictionary mapping layer names to activation tensors
        validate: Whether to validate each Gram matrix
        
    Returns:
        Dictionary mapping layer names to Gram matrices
    """
    gram_matrices = {}
    
    for layer_name, activation in activations.items():
        try:
            gram_matrices[layer_name] = compute_gram_matrix(activation, validate=validate)
        except Exception as e:
            # Log warning but continue with other layers
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to compute Gram matrix for {layer_name}: {e}")
            continue
    
    return gram_matrices


def validate_gram_matrix_properties(gram_matrix: torch.Tensor) -> Dict[str, Any]:
    """Validate mathematical properties of a Gram matrix.
    
    Args:
        gram_matrix: Gram matrix to validate
        
    Returns:
        Dictionary with validation results and diagnostics
    """
    n = gram_matrix.shape[0]
    
    # Check basic properties
    is_square = gram_matrix.shape[0] == gram_matrix.shape[1]
    is_symmetric = torch.allclose(gram_matrix, gram_matrix.T, atol=1e-6)
    
    # Check positive semidefinite property via eigenvalues
    try:
        eigenvals = torch.linalg.eigvals(gram_matrix).real
        min_eigenval = torch.min(eigenvals).item()
        max_eigenval = torch.max(eigenvals).item()
        is_psd = min_eigenval >= -1e-5  # Allow small numerical errors for ResNet-scale models
        
        # Compute rank and condition number
        pos_eigenvals = eigenvals[eigenvals > 1e-12]
        effective_rank = len(pos_eigenvals)
        condition_number = max_eigenval / (torch.min(pos_eigenvals).item() + 1e-12)
        
    except Exception as e:
        # Fallback if eigenvalue computation fails
        min_eigenval = float('nan')
        max_eigenval = float('nan')
        is_psd = False
        effective_rank = 0
        condition_number = float('inf')
    
    # Check for numerical issues
    has_nan = torch.isnan(gram_matrix).any().item()
    has_inf = torch.isinf(gram_matrix).any().item()
    frobenius_norm = torch.norm(gram_matrix, p='fro').item()
    
    return {
        'is_square': is_square,
        'is_symmetric': is_symmetric, 
        'is_positive_semidefinite': is_psd,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'shape': gram_matrix.shape,
        'frobenius_norm': frobenius_norm,
        'eigenvalue_range': (min_eigenval, max_eigenval),
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'rank_deficient': effective_rank < n,
    }


def _validate_gram_matrix(gram_matrix: torch.Tensor, strict: bool = True) -> None:
    """Internal validation function for Gram matrices.
    
    Args:
        gram_matrix: Gram matrix to validate
        strict: Whether to raise errors for violations
        
    Raises:
        ValueError: If gram matrix has invalid properties (when strict=True)
    """
    validation = validate_gram_matrix_properties(gram_matrix)
    
    errors = []
    
    if not validation['is_square']:
        errors.append(f"Gram matrix must be square, got shape {validation['shape']}")
    
    if validation['has_nan']:
        errors.append("Gram matrix contains NaN values")
        
    if validation['has_inf']:
        errors.append("Gram matrix contains infinite values")
    
    if not validation['is_symmetric']:
        errors.append("Gram matrix is not symmetric")
        
    if not validation['is_positive_semidefinite']:
        errors.append(f"Gram matrix is not positive semidefinite (min eigenvalue: {validation['eigenvalue_range'][0]:.2e})")
    
    if errors and strict:
        raise ValueError("Gram matrix validation failed: " + "; ".join(errors))
    elif errors:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Gram matrix validation warnings: " + "; ".join(errors))
"""Core mathematical operations for sheaf construction.

This module contains pure mathematical functions and classes for:
- Whitening transformations for exact metric compatibility
- Procrustes analysis for optimal restriction maps
- Gram matrix operations and validation
- Sheaf property validation

All functions in this module are focused on mathematical correctness
and operate in whitened coordinate spaces for optimal properties.
"""

from .whitening import WhiteningProcessor
from .procrustes import scaled_procrustes_whitened
from .validation import validate_sheaf_properties, validate_restriction_orthogonality
from .gram_matrices import (
    compute_gram_matrix, 
    compute_gram_matrices_from_activations,
    validate_gram_matrix_properties
)

__all__ = [
    "WhiteningProcessor",
    "scaled_procrustes_whitened", 
    "validate_sheaf_properties",
    "validate_restriction_orthogonality",
    "compute_gram_matrix",
    "compute_gram_matrices_from_activations", 
    "validate_gram_matrix_properties",
]
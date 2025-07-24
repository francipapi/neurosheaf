"""Sheaf construction module for neural network analysis.

This module provides a clean, mathematically sound implementation of cellular
sheaves for neural network analysis using whitened coordinates for optimal
numerical properties.

Key Features:
- FX-based automatic poset extraction from PyTorch models
- Whitened coordinate transformations for exact metric compatibility  
- Scaled Procrustes restriction maps between layers
- Clean sheaf data structures with validation
- Efficient sparse Laplacian assembly

All mathematical operations occur in whitened coordinate space for
optimal conditioning and exact orthogonality properties.
"""

# Core mathematical operations
from .core import (
    WhiteningProcessor,
    scaled_procrustes_whitened, 
    validate_sheaf_properties,
    validate_restriction_orthogonality,
    compute_gram_matrix,
    compute_gram_matrices_from_activations,
    validate_gram_matrix_properties,
    # Gromov-Wasserstein components (Phase 1)
    GWConfig,
    GromovWassersteinComputer,
    GWResult,
    CostMatrixCache,
)

# Data structures
from .data_structures import Sheaf, SheafValidationResult, WhiteningInfo, GWCouplingInfo

# Extraction utilities
from .extraction import (
    FXPosetExtractor,
    FXActivationExtractor,
    extract_activations_fx,
    FXToModuleNameMapper,
    create_unified_activation_dict
)

# Assembly pipeline
from .assembly import (
    SheafBuilder,
    RestrictionManager,
    compute_restrictions_for_sheaf,
    SheafLaplacianBuilder,
    build_sheaf_laplacian,
    LaplacianMetadata
)

__all__ = [
    # Core mathematical operations
    "WhiteningProcessor",
    "scaled_procrustes_whitened",
    "validate_sheaf_properties", 
    "validate_restriction_orthogonality",
    "compute_gram_matrix",
    "compute_gram_matrices_from_activations",
    "validate_gram_matrix_properties",
    
    # Gromov-Wasserstein components (Phase 1)
    "GWConfig",
    "GromovWassersteinComputer",
    "GWResult", 
    "CostMatrixCache",
    
    # Data structures
    "Sheaf",
    "SheafValidationResult", 
    "WhiteningInfo",
    "GWCouplingInfo",
    
    # Extraction utilities
    "FXPosetExtractor",
    "FXActivationExtractor",
    "extract_activations_fx",
    "FXToModuleNameMapper",
    "create_unified_activation_dict",
    
    # Assembly pipeline
    "SheafBuilder",
    "RestrictionManager",
    "compute_restrictions_for_sheaf",
    "SheafLaplacianBuilder", 
    "build_sheaf_laplacian",
    "LaplacianMetadata",
]
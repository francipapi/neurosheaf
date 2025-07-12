"""Sheaf construction module for neural network analysis.

This module implements:
- FX-based automatic poset extraction from PyTorch models
- Scaled Procrustes restriction maps between layers
- Sheaf data structure with mathematical validation
- Sparse Laplacian assembly for memory efficiency

Phase 3 Week 5: FX-based poset extraction implementation.
Phase 3 Week 6: Restriction maps and sheaf construction.
Phase 3 Week 7: Sparse Laplacian assembly and optimization.
"""

from .poset import FXPosetExtractor
from .restriction import ProcrustesMaps, WhiteningProcessor, validate_sheaf_properties
from .construction import Sheaf, SheafBuilder, create_sheaf_from_cka_analysis
from .laplacian import SheafLaplacianBuilder, LaplacianMetadata
from .name_mapper import FXToModuleNameMapper, create_unified_activation_dict
from .enhanced_extraction import EnhancedActivationExtractor

__all__ = [
    "FXPosetExtractor",
    "ProcrustesMaps", 
    "WhiteningProcessor",
    "validate_sheaf_properties",
    "Sheaf",
    "SheafBuilder",
    "create_sheaf_from_cka_analysis",
    "SheafLaplacianBuilder",
    "LaplacianMetadata", 
    "FXToModuleNameMapper",
    "create_unified_activation_dict",
    "EnhancedActivationExtractor",
]
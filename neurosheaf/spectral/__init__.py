"""Spectral analysis module.

This module contains persistent spectral analysis with subspace tracking,
static Laplacian with edge masking, and multi-parameter persistence.

Phase 3 Week 7: Static Laplacian implementation for filtration.
Phase 4: Persistent spectral analysis and subspace tracking.
"""

from .static_laplacian import StaticMaskedLaplacian, MaskingMetadata, create_static_masked_laplacian

# Placeholder for Phase 4 implementation  
# from .persistent import PersistentSpectralAnalyzer
# from .tracker import EigenSubspaceTracker

__all__ = [
    "StaticMaskedLaplacian",
    "MaskingMetadata", 
    "create_static_masked_laplacian",
]
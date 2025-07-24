"""Sheaf assembly modules for building cellular sheaves.

This module provides utilities for assembling cellular sheaves from
neural network data using multiple mathematical approaches:

- Procrustes-based methods: Uses whitened coordinate spaces for optimal properties
- Gromov-Wasserstein methods: Uses optimal transport for metric-preserving construction

Both approaches maintain mathematical rigor while supporting different use cases
and network architectures.
"""

from .builder import SheafBuilder
from .restrictions import RestrictionManager, compute_restrictions_for_sheaf
from .laplacian import SheafLaplacianBuilder, build_sheaf_laplacian, LaplacianMetadata
from .gw_builder import GWRestrictionManager, GWRestrictionError
from .gw_laplacian import GWLaplacianBuilder, GWLaplacianError as GWLapError, GWLaplacianMetadata

__all__ = [
    "SheafBuilder",
    "RestrictionManager", 
    "compute_restrictions_for_sheaf",
    "SheafLaplacianBuilder",
    "build_sheaf_laplacian",
    "LaplacianMetadata",
    # Gromov-Wasserstein components (Phase 2)
    "GWRestrictionManager",
    "GWRestrictionError",
    # Gromov-Wasserstein Laplacian components (Phase 3)
    "GWLaplacianBuilder",
    "GWLapError",
    "GWLaplacianMetadata",
]
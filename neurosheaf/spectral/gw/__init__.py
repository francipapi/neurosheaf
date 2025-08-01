# neurosheaf/spectral/gw/__init__.py
"""
Gromov-Wasserstein specific subspace tracking module.

This subpackage provides specialized eigenvalue tracking for Gromov-Wasserstein
based sheaf construction using Persistent Eigenvector Similarity (PES) methodology.

Key Components:
- GWSubspaceTracker: Main tracker using PES methodology
- PESComputer: Core mathematical engine for PES computation
- SheafInclusionMapper: Transport-based inclusion mappings
- GWEigenspaceEmbedder: SVD-based eigenspace embedding
- GWBirthDeathDetector: GW-aware birth-death event detection

Mathematical Foundation:
Based on "Disentangling the Spectral Properties of the Hodge Laplacian" 
and persistent sheaf Laplacian theory, with adaptations for increasing
complexity filtrations in Gromov-Wasserstein optimal transport context.
"""

from .gw_subspace_tracker import GWSubspaceTracker
from .pes_computation import PESComputer
from .sheaf_inclusion_mapper import SheafInclusionMapper
from .gw_eigenspace_embedder import GWEigenspaceEmbedder
from .gw_birth_death_detector import GWBirthDeathDetector

__all__ = [
    'GWSubspaceTracker',
    'PESComputer', 
    'SheafInclusionMapper',
    'GWEigenspaceEmbedder',
    'GWBirthDeathDetector'
]

__version__ = '1.0.0'
__author__ = 'Neurosheaf Development Team'
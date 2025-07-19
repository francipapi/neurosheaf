"""Real-complex conversion utilities for directed sheaf computation.

This module provides utilities for converting between complex mathematical 
representations and real computational representations as required for 
directed sheaf analysis.

Mathematical Foundation:
- Complex-to-real embedding: Z = X + iY → [[X, -Y], [Y, X]]
- Spectral property preservation: eigenvalues as conjugate pairs
- Hermitian-to-symmetric mapping: Hermitian matrices → symmetric matrices
- Positive definiteness preservation

Key Components:
- ComplexToRealEmbedding: Convert complex matrices to real representation
- RealToComplexReconstruction: Reconstruct complex results from real computation
- ConversionValidator: Validate conversion accuracy and properties
- PerformanceOptimizer: Optimize conversion operations for large matrices

Usage:
    from neurosheaf.directed_sheaf.conversion import (
        ComplexToRealEmbedding, 
        RealToComplexReconstruction
    )
    
    embedder = ComplexToRealEmbedding()
    reconstructor = RealToComplexReconstruction()
    
    # Convert complex matrix to real representation
    real_matrix = embedder.embed_matrix(complex_matrix)
    
    # Reconstruct complex results from real computation
    complex_eigenvalues = reconstructor.reconstruct_eigenvalues(real_eigenvalues)
"""

from .real_embedding import ComplexToRealEmbedding
from .complex_reconstruction import RealToComplexReconstruction
from .utilities import ConversionValidator, PerformanceOptimizer

__all__ = [
    'ComplexToRealEmbedding',
    'RealToComplexReconstruction', 
    'ConversionValidator',
    'PerformanceOptimizer'
]
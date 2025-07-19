"""Directed sheaf assembly module for Laplacian construction.

This module provides tools for constructing Hermitian sheaf Laplacians from
directed sheaf data, including complex-to-real conversion and validation.

Key Components:
- DirectedSheafLaplacianBuilder: Main Laplacian construction class
- Block-structured Hermitian Laplacian assembly
- Real embedding for efficient computation
- Sparse matrix optimization

Mathematical Foundation:
- Hermitian Laplacian: L^{F̃} = δ̃* δ̃
- Block structure with complex directional encoding
- Real representation: Z = X + iY → [[X, -Y], [Y, X]]
- Positive semi-definite property preservation

The assembly follows the mathematical formulation in:
docs/DirectedSheaf_mathematicalFormulation.md Section 3.2

Usage:
    from neurosheaf.directed_sheaf.assembly import DirectedSheafLaplacianBuilder
    
    builder = DirectedSheafLaplacianBuilder()
    real_laplacian = builder.build_real_embedded_laplacian(directed_sheaf)
    
    # Validate Hermitian properties
    hermitian_laplacian = builder.build_complex_laplacian(directed_sheaf)
    validation = builder.validate_hermitian_properties(hermitian_laplacian)
"""

# Main assembly classes
from .laplacian import DirectedSheafLaplacianBuilder, LaplacianMetadata
from .builder import DirectedSheafBuilder

# Version information
__version__ = "1.0.0"

# Module exports
__all__ = [
    "DirectedSheafLaplacianBuilder",
    "DirectedSheafBuilder",
    "LaplacianMetadata",
    "__version__",
]

# Assembly constants
ASSEMBLY_DEFAULTS = {
    'hermitian_tolerance': 1e-12,
    'positive_semidefinite_tolerance': 1e-12,
    'sparse_threshold': 0.1,  # Convert to sparse if density < 10%
    'use_sparse_operations': True,
    'validate_hermitian': True,
    'validate_positive_semidefinite': True
}

# Block structure constants
BLOCK_STRUCTURE_CONSTANTS = {
    'diagonal_identity_scaling': 1.0,
    'off_diagonal_sign': -1.0,
    'complex_conjugate_handling': 'automatic',
    'block_ordering': 'lexicographic'
}

def get_assembly_info() -> dict:
    """Get information about the assembly module.
    
    Returns:
        Dictionary containing assembly module metadata
    """
    return {
        'name': 'assembly',
        'version': __version__,
        'description': 'Directed sheaf Laplacian assembly utilities',
        'mathematical_foundation': 'Hermitian Laplacian construction with real embedding',
        'key_features': [
            'Hermitian Laplacian construction',
            'Block-structured assembly',
            'Complex-to-real embedding',
            'Sparse matrix optimization',
            'Mathematical validation'
        ],
        'assembly_defaults': ASSEMBLY_DEFAULTS,
        'block_structure_constants': BLOCK_STRUCTURE_CONSTANTS
    }

def validate_assembly_parameters(params: dict) -> bool:
    """Validate assembly parameters.
    
    Args:
        params: Dictionary of assembly parameters
        
    Returns:
        True if parameters are valid, False otherwise
    """
    required_keys = ['hermitian_tolerance', 'positive_semidefinite_tolerance']
    
    for key in required_keys:
        if key not in params:
            return False
        if not isinstance(params[key], (int, float)):
            return False
        if params[key] < 0:
            return False
    
    return True
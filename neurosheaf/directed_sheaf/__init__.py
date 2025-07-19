"""Directed sheaf module for neural network analysis.

This module provides support for directed cellular sheaves with complex-valued
stalks and Hermitian Laplacians, enabling asymmetric network analysis while
maintaining full compatibility with the existing neurosheaf pipeline.

Key Features:
- Complex-valued stalks for directional encoding
- Hermitian Laplacian construction with real embedding
- Directional encoding through complex phases T^{(q)}
- Backward compatibility with existing pipeline
- Mathematical correctness with comprehensive validation

Mathematical Foundation:
- Stalks: F(v) = C^{r_v} (complex vector spaces)
- Directional Encoding: T^{(q)} = exp(i 2π q (A - A^T))
- Directed Restriction Maps: Complex-valued with phase encoding
- Hermitian Laplacian: L^{F} = δ* δ (positive semi-definite)

The module follows the mathematical formulation in:
docs/DirectedSheaf_mathematicalFormulation.md

Usage:
    from neurosheaf.directed_sheaf import DirectedSheaf, DirectedSheafValidationResult
    
    # Create directed sheaf from existing real sheaf
    directed_sheaf = DirectedSheaf(
        poset=base_sheaf.poset,
        complex_stalks=extended_stalks,
        directed_restrictions=complex_restrictions,
        directionality_parameter=0.25
    )
    
    # Validate structure
    validation = directed_sheaf.validate_complex_structure()
    
    # Convert to real representation for computation
    real_stalks, real_restrictions = directed_sheaf.to_real_representation()
"""

# Core data structures
from .data_structures import (
    DirectedSheaf,
    DirectedSheafValidationResult,
    DirectedWhiteningInfo
)

# Assembly and pipeline classes
from .assembly import DirectedSheafBuilder, DirectedSheafLaplacianBuilder
from .compatibility import DirectedSheafAdapter

# Version information
__version__ = "1.0.0"

# Module metadata
__all__ = [
    # Core data structures
    "DirectedSheaf",
    "DirectedSheafValidationResult", 
    "DirectedWhiteningInfo",
    
    # Assembly and pipeline
    "DirectedSheafBuilder",
    "DirectedSheafLaplacianBuilder",
    "DirectedSheafAdapter",
    
    # Version info
    "__version__",
]

# Module-level constants
DEFAULT_DIRECTIONALITY_PARAMETER = 0.25
COMPLEX_TOLERANCE = 1e-12
HERMITIAN_TOLERANCE = 1e-12
REAL_EMBEDDING_FACTOR = 2  # Complex dimension to real dimension multiplier

# Validation constants
VALIDATION_DEFAULTS = {
    'hermitian_tolerance': HERMITIAN_TOLERANCE,
    'complex_tolerance': COMPLEX_TOLERANCE,
    'check_positive_semidefinite': True,
    'check_real_spectrum': True,
    'check_complex_structure': True,
    'check_directional_encoding': True
}

# Mathematical constants
MATHEMATICAL_CONSTANTS = {
    'default_q': DEFAULT_DIRECTIONALITY_PARAMETER,
    'pi': 3.141592653589793,
    'two_pi': 6.283185307179586,
    'complex_i': 1j
}

def get_module_info() -> dict:
    """Get information about the directed sheaf module.
    
    Returns:
        Dictionary containing module metadata
    """
    return {
        'name': 'directed_sheaf',
        'version': __version__,
        'description': 'Directed cellular sheaves for neural network analysis',
        'mathematical_foundation': 'Complex-valued stalks with Hermitian Laplacians',
        'key_features': [
            'Complex-valued stalks',
            'Directional encoding via T^{(q)}',
            'Hermitian Laplacian construction',
            'Real embedding for computation',
            'Backward compatibility'
        ],
        'constants': {
            'default_directionality_parameter': DEFAULT_DIRECTIONALITY_PARAMETER,
            'complex_tolerance': COMPLEX_TOLERANCE,
            'hermitian_tolerance': HERMITIAN_TOLERANCE,
            'real_embedding_factor': REAL_EMBEDDING_FACTOR
        },
        'validation_defaults': VALIDATION_DEFAULTS,
        'mathematical_constants': MATHEMATICAL_CONSTANTS
    }

def validate_directionality_parameter(q: float) -> bool:
    """Validate the directionality parameter q.
    
    Args:
        q: Directionality parameter
        
    Returns:
        True if q is valid, False otherwise
    """
    return isinstance(q, (int, float)) and 0.0 <= q <= 1.0

def create_default_directed_sheaf() -> DirectedSheaf:
    """Create a default directed sheaf for testing purposes.
    
    Returns:
        Empty DirectedSheaf with default parameters
    """
    return DirectedSheaf(
        directionality_parameter=DEFAULT_DIRECTIONALITY_PARAMETER,
        metadata={
            'construction_method': 'default_creation',
            'directed_sheaf': True,
            'validation_passed': False
        }
    )
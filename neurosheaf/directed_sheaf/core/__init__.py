"""Core mathematical operations for directed sheaf construction.

This module provides the mathematical operations required for directed cellular
sheaf construction, including complex extension, directional encoding, and
directed procrustes computation.

Mathematical Foundation:
- Complex Extension: R^{r_v} → C^{r_v} = R^{r_v} ⊗_R C
- Directional Encoding: T^{(q)} = exp(i 2π q (A - A^T))
- Directed Restrictions: Complex-valued maps with phase encoding

Key Features:
- Preserves whitened coordinate structure
- Maintains exact mathematical properties
- Integrates with existing procrustes infrastructure
- Provides comprehensive validation

Usage:
    from neurosheaf.directed_sheaf.core import (
        ComplexStalkExtender,
        DirectionalEncodingComputer,
        DirectedProcrustesComputer
    )
"""

from .complex_extension import ComplexStalkExtender
from .directional_encoding import DirectionalEncodingComputer
from .directed_procrustes import DirectedProcrustesComputer

__all__ = [
    "ComplexStalkExtender",
    "DirectionalEncodingComputer", 
    "DirectedProcrustesComputer",
]
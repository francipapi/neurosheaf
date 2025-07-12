"""Spectral analysis module.

This module contains persistent spectral analysis with subspace tracking,
static Laplacian with edge masking, and multi-parameter persistence.

Phase 3 Week 7: Static Laplacian implementation for filtration.
Phase 4: Persistent spectral analysis and subspace tracking.
"""

# Unified static Laplacian implementation (consolidates old separate classes)
from .static_laplacian_unified import (
    UnifiedStaticLaplacian,
    UnifiedMaskingMetadata as MaskingMetadata,
    create_unified_static_laplacian
)

# Main interface - unified implementation with backward compatibility alias
StaticLaplacianWithMasking = UnifiedStaticLaplacian

# Backward compatibility imports (with deprecation warnings)
from ._deprecated import StaticMaskedLaplacian, create_static_masked_laplacian

# Phase 4 Week 8: Subspace similarity tracker with optimal assignment
from .tracker import SubspaceTracker

# Phase 4 Week 9: Persistent spectral analysis
from .persistent import PersistentSpectralAnalyzer

# Phase 4 Week 10: Multi-parameter persistence
from .multi_parameter import (
    MultiParameterFiltration,
    ParameterPoint,
    ParameterCorrelationAnalyzer,
    MultiParameterPersistenceComputer,
    MultiParameterSpectralAnalyzer
)

__all__ = [
    # Unified implementation (primary interface)
    "UnifiedStaticLaplacian",
    "StaticLaplacianWithMasking",  # Alias for UnifiedStaticLaplacian
    "MaskingMetadata",
    "create_unified_static_laplacian",
    
    # Backward compatibility (deprecated)
    "StaticMaskedLaplacian",
    "create_static_masked_laplacian", 
    
    # Core functionality
    "SubspaceTracker",
    "PersistentSpectralAnalyzer",
    
    # Multi-parameter persistence
    "MultiParameterFiltration",
    "ParameterPoint",
    "ParameterCorrelationAnalyzer",
    "MultiParameterPersistenceComputer",
    "MultiParameterSpectralAnalyzer",
]
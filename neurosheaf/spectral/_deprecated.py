"""
Deprecated classes and functions from the spectral module.

This module provides backward compatibility for code that imports the old
separate static Laplacian classes. It issues deprecation warnings and
redirects to the unified implementation.
"""

import warnings
from .static_laplacian_unified import UnifiedStaticLaplacian


def _issue_deprecation_warning(old_name: str, new_name: str):
    """Issue a deprecation warning for old class usage."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version. "
        f"Use {new_name} instead for improved functionality and mathematical correctness. "
        f"The unified implementation provides better performance and correct Laplacian masking.",
        DeprecationWarning,
        stacklevel=3
    )


class StaticMaskedLaplacian(UnifiedStaticLaplacian):
    """Deprecated: Use UnifiedStaticLaplacian instead.
    
    This class provided lower-level Laplacian masking but used mathematically
    incorrect entry zeroing. The unified implementation uses proper block
    reconstruction for mathematical correctness.
    """
    
    def __init__(self, *args, **kwargs):
        _issue_deprecation_warning("StaticMaskedLaplacian", "UnifiedStaticLaplacian")
        # Convert old-style constructor arguments to new format
        super().__init__(**kwargs)


class StaticLaplacianWithMaskingOld(UnifiedStaticLaplacian):
    """Deprecated: Use UnifiedStaticLaplacian instead.
    
    The original StaticLaplacianWithMasking has been replaced by UnifiedStaticLaplacian
    which consolidates functionality and provides better mathematical correctness.
    """
    
    def __init__(self, *args, **kwargs):
        _issue_deprecation_warning("StaticLaplacianWithMasking (old)", "UnifiedStaticLaplacian")
        super().__init__(**kwargs)


# For imports that try to use the old functions
def create_static_masked_laplacian(*args, **kwargs):
    """Deprecated: Use create_unified_static_laplacian instead."""
    _issue_deprecation_warning("create_static_masked_laplacian", "create_unified_static_laplacian")
    from .static_laplacian_unified import create_unified_static_laplacian
    return create_unified_static_laplacian(*args, **kwargs)
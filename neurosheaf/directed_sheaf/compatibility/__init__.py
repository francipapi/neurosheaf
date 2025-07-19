"""Compatibility module for directed sheaf integration.

This module provides compatibility layers for integrating directed sheaves
with the existing neurosheaf pipeline. It handles format conversions,
adapter patterns, and ensures seamless integration without breaking changes.

Key Components:
- DirectedSheafAdapter: Main adapter for pipeline integration
- Format converters for spectral analysis and visualization
- Backward compatibility utilities
- Performance optimization helpers

The compatibility module ensures that directed sheaves can be used
with existing pipeline components while maintaining mathematical correctness
and performance characteristics.

Usage:
    from neurosheaf.directed_sheaf.compatibility import DirectedSheafAdapter
    
    adapter = DirectedSheafAdapter()
    real_laplacian, metadata = adapter.adapt_for_spectral_analysis(directed_sheaf)
    
    # Use with existing spectral analysis pipeline
    spectral_results = analyzer.analyze_from_laplacian(real_laplacian, metadata)
"""

# Main compatibility classes
from .adapter import DirectedSheafAdapter

# Version information
__version__ = "1.0.0"

# Module exports
__all__ = [
    "DirectedSheafAdapter",
    "__version__",
]

# Compatibility constants
COMPATIBILITY_DEFAULTS = {
    'preserve_metadata': True,
    'validate_conversions': True,
    'optimize_sparse_operations': True,
    'maintain_precision': True,
    'enable_caching': True
}

# Format conversion constants
FORMAT_CONVERSION_CONSTANTS = {
    'real_embedding_factor': 2,  # Complex to real dimension multiplier
    'spectral_tolerance': 1e-12,  # Tolerance for spectral properties
    'metadata_prefix': 'directed_',  # Prefix for directed-specific metadata
    'compatibility_version': '1.0.0'
}

def get_compatibility_info() -> dict:
    """Get information about the compatibility module.
    
    Returns:
        Dictionary containing compatibility module metadata
    """
    return {
        'name': 'compatibility',
        'version': __version__,
        'description': 'Directed sheaf compatibility layer for pipeline integration',
        'key_features': [
            'Directed sheaf to real Laplacian conversion',
            'Spectral analysis adapter',
            'Visualization format converter',
            'Metadata preservation',
            'Performance optimization'
        ],
        'compatibility_defaults': COMPATIBILITY_DEFAULTS,
        'format_constants': FORMAT_CONVERSION_CONSTANTS,
        'supported_formats': {
            'spectral_analysis': ['csr_matrix', 'dense_tensor'],
            'visualization': ['networkx_graph', 'plotly_data'],
            'persistence': ['persistence_diagram', 'barcode_data'],
            'eigenvalue_tracking': ['real_eigenvalues', 'subspace_data']
        }
    }

def validate_compatibility_parameters(params: dict) -> bool:
    """Validate compatibility adapter parameters.
    
    Args:
        params: Dictionary of compatibility parameters
        
    Returns:
        True if parameters are valid, False otherwise
    """
    required_keys = ['preserve_metadata', 'validate_conversions']
    
    for key in required_keys:
        if key not in params:
            return False
        if not isinstance(params[key], bool):
            return False
    
    return True

def create_default_adapter() -> DirectedSheafAdapter:
    """Create a default DirectedSheafAdapter instance.
    
    Returns:
        DirectedSheafAdapter with default configuration
    """
    return DirectedSheafAdapter(
        preserve_metadata=True,
        validate_conversions=True,
        optimize_sparse_operations=True
    )
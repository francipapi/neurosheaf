"""Interactive visualization module for neurosheaf analysis.

This module provides comprehensive interactive visualizations for neural sheaf
analysis using Plotly for enhanced interactivity and mathematical correctness.

Key Features:
- Interactive poset visualization with data-flow layout
- Persistence diagrams and barcodes with lifetime color-coding  
- Multi-scale eigenvalue evolution with logarithmic scaling
- Unified visualization factory for consistent interfaces

Enhanced Features:
- Intelligent node classification and styling
- Advanced edge rendering with restriction map analysis
- Architecture-aware layout algorithms
- Multi-modal information panels
- Comprehensive design system

Mathematical Focus:
- Proper handling of whitened coordinate space
- Support for rectangular restriction maps
- Integration with persistent spectral analysis pipeline
"""

# Original components
from .poset import PosetVisualizer
from .persistence import PersistenceVisualizer
from .spectral import SpectralVisualizer
from .factory import VisualizationFactory

# Enhanced components
from .enhanced_poset import EnhancedPosetVisualizer
from .enhanced_spectral import EnhancedSpectralVisualizer
from .enhanced_factory import EnhancedVisualizationFactory

# Enhanced sub-components
from .enhanced import (
    EnhancedNodeClassifier,
    IntelligentEdgeRenderer,
    ArchitectureAwareLayout,
    InfoPanelManager,
    DesignSystem,
    NodeType,
    NodeVisualProperties,
    EdgeVisualProperties
)

__all__ = [
    # Original components
    "PosetVisualizer",
    "PersistenceVisualizer", 
    "SpectralVisualizer",
    "VisualizationFactory",
    
    # Enhanced components
    "EnhancedPosetVisualizer",
    "EnhancedSpectralVisualizer", 
    "EnhancedVisualizationFactory",
    
    # Enhanced sub-components
    "EnhancedNodeClassifier",
    "IntelligentEdgeRenderer",
    "ArchitectureAwareLayout",
    "InfoPanelManager",
    "DesignSystem",
    "NodeType",
    "NodeVisualProperties",
    "EdgeVisualProperties"
]
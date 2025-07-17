"""Enhanced visualization components for neurosheaf.

This module contains advanced visualization features including:
- Intelligent node classification
- Advanced edge rendering
- Architecture-aware layouts
- Rich information panels
- Enhanced interaction patterns
"""

from .node_classifier import EnhancedNodeClassifier, NodeType, NodeVisualProperties
from .edge_renderer import IntelligentEdgeRenderer, EdgeVisualProperties
from .layout_algorithms import ArchitectureAwareLayout
from .info_panels import InfoPanelManager
from .design_system import DesignSystem

__all__ = [
    'EnhancedNodeClassifier',
    'NodeType', 
    'NodeVisualProperties',
    'IntelligentEdgeRenderer',
    'EdgeVisualProperties',
    'ArchitectureAwareLayout',
    'InfoPanelManager',
    'DesignSystem'
]
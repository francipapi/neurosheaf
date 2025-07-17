"""Enhanced node classification system for neural network components.

This module provides intelligent node type detection and visual property
assignment based on comprehensive analysis of neural network layers.
"""

import re
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from ...utils.logging import setup_logger

logger = setup_logger(__name__)


class NodeType(Enum):
    """Enumeration of neural network node types."""
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    CONV3D = "conv3d"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    DROPOUT = "dropout"
    POOLING = "pooling"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    RECURRENT = "recurrent"
    INPUT = "input"
    OUTPUT = "output"
    RESHAPE = "reshape"
    CONCAT = "concat"
    ADD = "add"
    UNKNOWN = "unknown"


@dataclass
class NodeVisualProperties:
    """Visual properties for a node."""
    shape: str  # circle, square, diamond, hexagon, rectangle, ellipse
    primary_color: str
    secondary_color: str
    border_style: str  # solid, dashed, dotted, double
    border_width: int
    icon_symbol: Optional[str]
    size_factor: float
    opacity: float
    label_position: str  # top, bottom, center, left, right


class EnhancedNodeClassifier:
    """Advanced node classification for neural network components."""
    
    def __init__(self):
        """Initialize the enhanced node classifier."""
        self._init_patterns()
        self._init_visual_mapping()
        self._init_icon_mapping()
        
    def _init_patterns(self):
        """Initialize regex patterns for node classification."""
        self.patterns = {
            NodeType.LINEAR: [
                r'linear|dense|fc|fully_connected',
                r'layers\.\d+\.weight',  # Sequential layer pattern
                r'classifier\.\d+',
                r'fc\d+',
                r'^\d+$.*linear',  # Fallback for Sequential indices pointing to linear layers
            ],
            NodeType.CONV2D: [
                r'conv2d|convolution2d',
                r'conv\d+',
                r'features\.\d+\.conv',
            ],
            NodeType.CONV1D: [
                r'conv1d|convolution1d',
                r'temporal_conv',
            ],
            NodeType.CONV3D: [
                r'conv3d|convolution3d',
                r'spatial_conv',
            ],
            NodeType.ACTIVATION: [
                r'relu|sigmoid|tanh|gelu|swish|silu|leaky_relu',
                r'activation|act',
                r'nonlinearity',
                r'^\d+$.*relu|^\d+$.*activation',  # Fallback for Sequential indices
            ],
            NodeType.NORMALIZATION: [
                r'batch_norm|batchnorm|bn\d*',
                r'layer_norm|layernorm|ln',
                r'group_norm|groupnorm|gn',
                r'instance_norm|instancenorm',
            ],
            NodeType.DROPOUT: [
                r'dropout|drop\d+',
                r'spatial_dropout',
                r'alpha_dropout',
            ],
            NodeType.POOLING: [
                r'pool|pooling',
                r'maxpool|avgpool|adaptivepool',
                r'global_pool',
            ],
            NodeType.ATTENTION: [
                r'attention|attn',
                r'multi_head|multihead',
                r'self_attention',
                r'cross_attention',
            ],
            NodeType.EMBEDDING: [
                r'embedding|embed',
                r'word_embeddings',
                r'position_embeddings',
            ],
            NodeType.RECURRENT: [
                r'lstm|gru|rnn',
                r'recurrent',
                r'recursive',
            ],
            NodeType.INPUT: [
                r'input|placeholder',
                r'x|data',
            ],
            NodeType.OUTPUT: [
                r'output|logits|predictions',
                r'head|classifier',
            ],
            NodeType.RESHAPE: [
                r'reshape|view|flatten',
                r'permute|transpose',
            ],
            NodeType.CONCAT: [
                r'concat|cat|concatenate',
                r'merge|join',
            ],
            NodeType.ADD: [
                r'add|sum|residual',
                r'skip|shortcut',
            ],
        }
        
    def _init_visual_mapping(self):
        """Initialize visual property mappings for each node type."""
        self.visual_mapping = {
            NodeType.LINEAR: NodeVisualProperties(
                shape="circle",
                primary_color="#1E88E5",  # Blue
                secondary_color="#90CAF9",
                border_style="solid",
                border_width=3,
                icon_symbol="⚡",
                size_factor=1.0,
                opacity=0.9,
                label_position="bottom"
            ),
            NodeType.CONV2D: NodeVisualProperties(
                shape="hexagon",
                primary_color="#43A047",  # Green
                secondary_color="#A5D6A7",
                border_style="solid",
                border_width=3,
                icon_symbol="⊞",
                size_factor=1.2,
                opacity=0.9,
                label_position="bottom"
            ),
            NodeType.CONV1D: NodeVisualProperties(
                shape="hexagon",
                primary_color="#00ACC1",  # Cyan
                secondary_color="#80DEEA",
                border_style="solid",
                border_width=3,
                icon_symbol="|||",
                size_factor=1.1,
                opacity=0.9,
                label_position="bottom"
            ),
            NodeType.ACTIVATION: NodeVisualProperties(
                shape="diamond",
                primary_color="#FB8C00",  # Orange
                secondary_color="#FFCC80",
                border_style="solid",
                border_width=2,
                icon_symbol="⚡",
                size_factor=0.8,
                opacity=0.85,
                label_position="right"
            ),
            NodeType.NORMALIZATION: NodeVisualProperties(
                shape="rectangle",
                primary_color="#8E24AA",  # Purple
                secondary_color="#CE93D8",
                border_style="solid",
                border_width=2,
                icon_symbol="📊",
                size_factor=0.9,
                opacity=0.85,
                label_position="bottom"
            ),
            NodeType.DROPOUT: NodeVisualProperties(
                shape="circle",
                primary_color="#E53935",  # Red
                secondary_color="#FFCDD2",
                border_style="dashed",
                border_width=2,
                icon_symbol="○",
                size_factor=0.7,
                opacity=0.7,
                label_position="right"
            ),
            NodeType.POOLING: NodeVisualProperties(
                shape="ellipse",
                primary_color="#00897B",  # Teal
                secondary_color="#80CBC4",
                border_style="solid",
                border_width=2,
                icon_symbol="▼",
                size_factor=0.9,
                opacity=0.85,
                label_position="bottom"
            ),
            NodeType.ATTENTION: NodeVisualProperties(
                shape="star",
                primary_color="#FFB300",  # Amber
                secondary_color="#FFE082",
                border_style="double",
                border_width=3,
                icon_symbol="👁",
                size_factor=1.3,
                opacity=0.95,
                label_position="top"
            ),
            NodeType.INPUT: NodeVisualProperties(
                shape="ellipse",
                primary_color="#4CAF50",  # Light Green
                secondary_color="#C8E6C9",
                border_style="solid",
                border_width=4,
                icon_symbol="▶",
                size_factor=1.1,
                opacity=1.0,
                label_position="bottom"
            ),
            NodeType.OUTPUT: NodeVisualProperties(
                shape="ellipse",
                primary_color="#F44336",  # Deep Red
                secondary_color="#FFCDD2",
                border_style="solid",
                border_width=4,
                icon_symbol="■",
                size_factor=1.1,
                opacity=1.0,
                label_position="bottom"
            ),
            NodeType.UNKNOWN: NodeVisualProperties(
                shape="circle",
                primary_color="#757575",  # Gray
                secondary_color="#E0E0E0",
                border_style="dotted",
                border_width=2,
                icon_symbol="?",
                size_factor=0.8,
                opacity=0.7,
                label_position="bottom"
            ),
        }
        
        # Add remaining types with default properties
        for node_type in NodeType:
            if node_type not in self.visual_mapping:
                self.visual_mapping[node_type] = self.visual_mapping[NodeType.UNKNOWN]
                
    def _init_icon_mapping(self):
        """Initialize icon mappings for different operations."""
        self.operation_icons = {
            'add': '➕',
            'multiply': '✖️',
            'concatenate': '🔗',
            'split': '✂️',
            'squeeze': '🗜',
            'unsqueeze': '🎈',
            'transpose': '🔄',
            'permute': '🔀',
            'matmul': '⊗',
            'softmax': '🌡',
            'layer_norm': '📊',
            'batch_norm': '📈',
            'gelu': '〰️',
            'relu': '📐',
            'sigmoid': 'σ',
            'tanh': '〜',
        }
        
    def classify_node(self, 
                     node_name: str, 
                     node_attrs: Dict[str, Any],
                     model_context: Optional[Dict[str, Any]] = None) -> NodeType:
        """
        Enhanced classification based on multiple factors.
        
        Args:
            node_name: Name of the node
            node_attrs: Node attributes from the graph
            model_context: Optional context about the model architecture
            
        Returns:
            NodeType classification
        """
        # First check explicit operation type
        op_type = node_attrs.get('op', '').lower()
        target = str(node_attrs.get('target', '')).lower()
        
        # Check for special operations
        if op_type == 'placeholder':
            return NodeType.INPUT
        elif op_type == 'output':
            return NodeType.OUTPUT
        elif op_type == 'call_module':
            # For call_module operations, try to get the actual module type
            if model_context and 'traced_model' in model_context:
                try:
                    traced_model = model_context['traced_model']
                    actual_module = traced_model.get_submodule(node_attrs.get('target', ''))
                    module_type = type(actual_module)
                    logger.debug(f"Node '{node_name}' (target: {node_attrs.get('target')}) is module type: {module_type}")
                    return self._classify_from_module_type(module_type)
                except Exception as e:
                    logger.debug(f"Could not get module type for node '{node_name}': {e}")
                    
            # Fallback: try to infer from target (for Sequential models)
            if target.isdigit():
                # For Sequential models, we can make educated guesses based on common patterns
                if model_context and 'module_types' in model_context:
                    module_type = model_context['module_types'].get(target) or model_context['module_types'].get(node_name)
                    if module_type:
                        return self._classify_from_module_type(module_type)
                        
        elif op_type == 'call_function':
            # Analyze the function being called
            if 'add' in target or 'sum' in target:
                return NodeType.ADD
            elif 'cat' in target or 'concat' in target:
                return NodeType.CONCAT
            elif 'reshape' in target or 'view' in target or 'flatten' in target:
                return NodeType.RESHAPE
                
        # Pattern-based classification
        combined_text = f"{node_name} {target} {op_type}".lower()
        
        for node_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    logger.debug(f"Node '{node_name}' classified as {node_type} by pattern '{pattern}'")
                    return node_type
                    
        # Try to infer from module type if available
        if model_context and 'module_types' in model_context:
            module_type = model_context['module_types'].get(node_name)
            if module_type:
                return self._classify_from_module_type(module_type)
                
        # Default to unknown
        logger.debug(f"Node '{node_name}' could not be classified, defaulting to UNKNOWN")
        return NodeType.UNKNOWN
        
    def _classify_from_module_type(self, module_type: type) -> NodeType:
        """Classify based on PyTorch module type."""
        type_mapping = {
            nn.Linear: NodeType.LINEAR,
            nn.Conv1d: NodeType.CONV1D,
            nn.Conv2d: NodeType.CONV2D,
            nn.Conv3d: NodeType.CONV3D,
            nn.ReLU: NodeType.ACTIVATION,
            nn.GELU: NodeType.ACTIVATION,
            nn.Sigmoid: NodeType.ACTIVATION,
            nn.Tanh: NodeType.ACTIVATION,
            nn.BatchNorm1d: NodeType.NORMALIZATION,
            nn.BatchNorm2d: NodeType.NORMALIZATION,
            nn.BatchNorm3d: NodeType.NORMALIZATION,
            nn.LayerNorm: NodeType.NORMALIZATION,
            nn.Dropout: NodeType.DROPOUT,
            nn.MaxPool1d: NodeType.POOLING,
            nn.MaxPool2d: NodeType.POOLING,
            nn.AvgPool1d: NodeType.POOLING,
            nn.AvgPool2d: NodeType.POOLING,
            nn.AdaptiveMaxPool1d: NodeType.POOLING,
            nn.AdaptiveMaxPool2d: NodeType.POOLING,
            nn.AdaptiveAvgPool1d: NodeType.POOLING,
            nn.AdaptiveAvgPool2d: NodeType.POOLING,
            nn.MultiheadAttention: NodeType.ATTENTION,
            nn.LSTM: NodeType.RECURRENT,
            nn.GRU: NodeType.RECURRENT,
            nn.RNN: NodeType.RECURRENT,
            nn.Embedding: NodeType.EMBEDDING,
        }
        
        for base_type, node_type in type_mapping.items():
            if issubclass(module_type, base_type):
                return node_type
                
        return NodeType.UNKNOWN
        
    def get_node_visual_properties(self, 
                                  node_type: NodeType,
                                  node_attrs: Dict[str, Any] = None) -> NodeVisualProperties:
        """
        Get comprehensive visual properties for a node type.
        
        Args:
            node_type: The classified node type
            node_attrs: Optional node attributes for customization
            
        Returns:
            NodeVisualProperties with all visual settings
        """
        base_props = self.visual_mapping[node_type]
        
        # Customize based on additional attributes
        if node_attrs:
            # Adjust size based on parameter count
            param_count = node_attrs.get('param_count', 0)
            if param_count > 0:
                # Logarithmic scaling for parameter count
                import math
                size_adjustment = 1 + math.log10(max(1, param_count)) * 0.1
                base_props.size_factor *= size_adjustment
                
            # Adjust opacity based on importance scores
            importance = node_attrs.get('importance_score', 1.0)
            base_props.opacity = min(1.0, base_props.opacity * importance)
            
        return base_props
        
    def get_operation_icon(self, operation: str) -> str:
        """Get icon for a specific operation."""
        return self.operation_icons.get(operation.lower(), '')
        
    def generate_node_label(self, 
                           node_name: str,
                           node_type: NodeType,
                           node_attrs: Dict[str, Any]) -> str:
        """
        Generate an informative label for a node.
        
        Args:
            node_name: Original node name
            node_type: Classified node type
            node_attrs: Node attributes
            
        Returns:
            Formatted label string
        """
        # Shorten long names
        if len(node_name) > 20:
            # Try to extract meaningful part
            parts = node_name.split('.')
            if len(parts) > 1:
                label = f"...{parts[-1]}"
            else:
                label = f"{node_name[:17]}..."
        else:
            label = node_name
            
        # Add type indicator if not obvious
        if node_type != NodeType.UNKNOWN and node_type.value not in label.lower():
            label = f"{label}\n[{node_type.value}]"
            
        # Add parameter count if significant
        param_count = node_attrs.get('param_count', 0)
        if param_count > 1000:
            if param_count >= 1_000_000:
                param_str = f"{param_count/1_000_000:.1f}M"
            elif param_count >= 1_000:
                param_str = f"{param_count/1_000:.1f}K"
            else:
                param_str = str(param_count)
            label = f"{label}\n({param_str} params)"
            
        return label
        
    def get_node_statistics(self, 
                           node_attrs: Dict[str, Any],
                           stalks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a node.
        
        Args:
            node_attrs: Node attributes
            stalks: Optional stalk tensors for the node
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'param_count': node_attrs.get('param_count', 0),
            'flops': node_attrs.get('flops', 0),
            'memory_usage': node_attrs.get('memory_usage', 0),
            'activation_size': node_attrs.get('activation_size', 0),
        }
        
        if stalks:
            # Add stalk-based statistics
            stalk_tensor = stalks.get(node_attrs.get('name'))
            if stalk_tensor is not None:
                stats['stalk_dimension'] = stalk_tensor.shape[0]
                stats['stalk_rank'] = torch.linalg.matrix_rank(stalk_tensor).item()
                stats['stalk_condition'] = torch.linalg.cond(stalk_tensor).item()
                
        return stats
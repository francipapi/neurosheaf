"""Architecture-aware layout algorithms for neural network visualization.

This module provides intelligent layout algorithms that understand 
neural network architecture patterns and arrange nodes accordingly.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from .node_classifier import NodeType, EnhancedNodeClassifier
from ...utils.logging import setup_logger

logger = setup_logger(__name__)


class ArchitectureAwareLayout:
    """Layout algorithms that understand neural network patterns."""
    
    def __init__(self, node_classifier: Optional[EnhancedNodeClassifier] = None):
        """
        Initialize the layout system.
        
        Args:
            node_classifier: Optional node classifier for architecture analysis
        """
        self.node_classifier = node_classifier or EnhancedNodeClassifier()
        
    def create_layout(self,
                     poset: nx.DiGraph,
                     sheaf_metadata: Dict[str, Any],
                     layout_type: str = 'auto') -> Dict[str, Tuple[float, float]]:
        """
        Create an optimized layout for the network.
        
        Args:
            poset: The network graph
            sheaf_metadata: Metadata about the sheaf
            layout_type: Type of layout ('auto', 'hierarchical', 'functional', 'flow')
            
        Returns:
            Dictionary mapping node names to (x, y) positions
        """
        # Analyze the architecture
        architecture_info = self._analyze_architecture(poset, sheaf_metadata)
        
        # Choose layout based on architecture if auto
        if layout_type == 'auto':
            layout_type = self._recommend_layout(architecture_info)
            
        # Apply the chosen layout
        if layout_type == 'hierarchical':
            return self._hierarchical_layout(poset, architecture_info)
        elif layout_type == 'functional':
            return self._functional_layout(poset, architecture_info)
        elif layout_type == 'flow':
            return self._flow_layout(poset, architecture_info)
        else:
            logger.warning(f"Unknown layout type {layout_type}, using hierarchical")
            return self._hierarchical_layout(poset, architecture_info)
            
    def _analyze_architecture(self,
                             poset: nx.DiGraph,
                             sheaf_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the network architecture to determine layout strategy.
        
        Args:
            poset: The network graph
            sheaf_metadata: Metadata about the sheaf
            
        Returns:
            Dictionary with architecture analysis
        """
        analysis = {
            'node_types': {},
            'functional_groups': defaultdict(list),
            'layers': [],
            'branches': [],
            'architecture_type': 'unknown'
        }
        
        # Classify all nodes
        for node in poset.nodes():
            node_attrs = poset.nodes[node]
            node_type = self.node_classifier.classify_node(node, node_attrs)
            analysis['node_types'][node] = node_type
            analysis['functional_groups'][node_type].append(node)
            
        # Detect architecture pattern
        analysis['architecture_type'] = self._detect_architecture_pattern(analysis['node_types'])
        
        # Find topological layers
        analysis['layers'] = self._find_topological_layers(poset)
        
        # Detect parallel branches
        analysis['branches'] = self._detect_parallel_branches(poset)
        
        return analysis
        
    def _detect_architecture_pattern(self, node_types: Dict[str, NodeType]) -> str:
        """Detect the overall architecture pattern."""
        type_counts = defaultdict(int)
        for node_type in node_types.values():
            type_counts[node_type] += 1
            
        # Simple heuristics for architecture detection
        if type_counts[NodeType.CONV2D] > 2 and type_counts[NodeType.POOLING] > 1:
            return 'cnn'
        elif type_counts[NodeType.RECURRENT] > 0:
            return 'rnn'
        elif type_counts[NodeType.ATTENTION] > 0:
            return 'transformer'
        elif type_counts[NodeType.LINEAR] > 2:
            return 'mlp'
        else:
            return 'mixed'
            
    def _find_topological_layers(self, poset: nx.DiGraph) -> List[List[str]]:
        """Find topological layers in the network."""
        layers = []
        
        # Use topological generations
        for generation in nx.topological_generations(poset):
            layers.append(sorted(list(generation)))
            
        return layers
        
    def _detect_parallel_branches(self, poset: nx.DiGraph) -> List[List[str]]:
        """Detect parallel branches in the network."""
        branches = []
        
        # Find nodes with multiple successors (branch points)
        for node in poset.nodes():
            successors = list(poset.successors(node))
            if len(successors) > 1:
                # Each successor starts a potential branch
                for successor in successors:
                    branch = self._trace_branch(poset, successor)
                    if len(branch) > 1:
                        branches.append(branch)
                        
        return branches
        
    def _trace_branch(self, poset: nx.DiGraph, start_node: str) -> List[str]:
        """Trace a branch from a starting node."""
        branch = [start_node]
        current = start_node
        
        while True:
            successors = list(poset.successors(current))
            if len(successors) == 1:
                current = successors[0]
                branch.append(current)
            else:
                break
                
        return branch
        
    def _recommend_layout(self, architecture_info: Dict[str, Any]) -> str:
        """Recommend the best layout based on architecture analysis."""
        arch_type = architecture_info['architecture_type']
        
        if arch_type in ['cnn', 'transformer']:
            return 'hierarchical'
        elif arch_type == 'rnn':
            return 'flow'
        elif len(architecture_info['branches']) > 2:
            return 'functional'
        else:
            return 'hierarchical'
            
    def _hierarchical_layout(self,
                           poset: nx.DiGraph,
                           architecture_info: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Create a hierarchical layout with clear layer separation.
        
        Args:
            poset: The network graph
            architecture_info: Architecture analysis results
            
        Returns:
            Dictionary of node positions
        """
        pos = {}
        layers = architecture_info['layers']
        
        # Calculate spacing
        layer_spacing = 2.0
        node_spacing = 1.5
        
        for layer_idx, layer in enumerate(layers):
            x = layer_idx * layer_spacing
            
            # Group nodes by type within layer for better organization
            type_groups = defaultdict(list)
            for node in layer:
                node_type = architecture_info['node_types'][node]
                type_groups[node_type].append(node)
                
            # Position nodes within layer
            y_offset = 0
            for node_type, nodes in type_groups.items():
                n_nodes = len(nodes)
                
                # Center the group
                start_y = y_offset - (n_nodes - 1) * node_spacing / 2
                
                for i, node in enumerate(sorted(nodes)):
                    y = start_y + i * node_spacing
                    pos[node] = (x, y)
                    
                y_offset += n_nodes * node_spacing + 0.5  # Group spacing
                
        return pos
        
    def _functional_layout(self,
                          poset: nx.DiGraph,
                          architecture_info: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Create a layout that groups nodes by function.
        
        Args:
            poset: The network graph
            architecture_info: Architecture analysis results
            
        Returns:
            Dictionary of node positions
        """
        pos = {}
        
        # Group nodes by functional blocks
        functional_blocks = self._identify_functional_blocks(poset, architecture_info)
        
        # Layout functional blocks
        block_positions = self._layout_functional_blocks(functional_blocks)
        
        # Position nodes within each block
        for block_name, block_nodes in functional_blocks.items():
            block_center = block_positions[block_name]
            block_pos = self._layout_nodes_in_block(block_nodes, block_center)
            pos.update(block_pos)
            
        return pos
        
    def _identify_functional_blocks(self,
                                   poset: nx.DiGraph,
                                   architecture_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify functional blocks in the network."""
        blocks = {}
        
        # Start with type-based grouping
        for node_type, nodes in architecture_info['functional_groups'].items():
            if len(nodes) > 1:
                blocks[f"{node_type.value}_block"] = nodes
                
        # Identify conv-bn-relu blocks
        conv_bn_relu_blocks = self._find_conv_bn_relu_blocks(poset, architecture_info)
        blocks.update(conv_bn_relu_blocks)
        
        return blocks
        
    def _find_conv_bn_relu_blocks(self,
                                 poset: nx.DiGraph,
                                 architecture_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Find conv-bn-relu functional blocks."""
        blocks = {}
        block_id = 0
        
        for node in poset.nodes():
            if architecture_info['node_types'][node] == NodeType.CONV2D:
                block_nodes = [node]
                current = node
                
                # Look for bn and relu following conv
                while True:
                    successors = list(poset.successors(current))
                    if len(successors) == 1:
                        successor = successors[0]
                        successor_type = architecture_info['node_types'][successor]
                        
                        if successor_type in [NodeType.NORMALIZATION, NodeType.ACTIVATION]:
                            block_nodes.append(successor)
                            current = successor
                        else:
                            break
                    else:
                        break
                        
                if len(block_nodes) > 1:
                    blocks[f"conv_block_{block_id}"] = block_nodes
                    block_id += 1
                    
        return blocks
        
    def _layout_functional_blocks(self, functional_blocks: Dict[str, List[str]]) -> Dict[str, Tuple[float, float]]:
        """Layout functional blocks in 2D space."""
        positions = {}
        
        # Simple grid layout for blocks
        n_blocks = len(functional_blocks)
        cols = int(np.ceil(np.sqrt(n_blocks)))
        rows = int(np.ceil(n_blocks / cols))
        
        block_spacing = 5.0
        
        for i, block_name in enumerate(functional_blocks.keys()):
            row = i // cols
            col = i % cols
            
            x = col * block_spacing
            y = row * block_spacing
            
            positions[block_name] = (x, y)
            
        return positions
        
    def _layout_nodes_in_block(self,
                              nodes: List[str],
                              center: Tuple[float, float]) -> Dict[str, Tuple[float, float]]:
        """Layout nodes within a functional block."""
        pos = {}
        cx, cy = center
        
        n_nodes = len(nodes)
        if n_nodes == 1:
            pos[nodes[0]] = center
        else:
            # Arrange in a small line
            spacing = 0.5
            start_x = cx - (n_nodes - 1) * spacing / 2
            
            for i, node in enumerate(nodes):
                x = start_x + i * spacing
                pos[node] = (x, cy)
                
        return pos
        
    def _flow_layout(self,
                    poset: nx.DiGraph,
                    architecture_info: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Create a flow-based layout emphasizing data flow.
        
        Args:
            poset: The network graph
            architecture_info: Architecture analysis results
            
        Returns:
            Dictionary of node positions
        """
        pos = {}
        
        # Find the main flow path
        main_path = self._find_main_flow_path(poset)
        
        # Position nodes along the main path
        path_spacing = 2.0
        
        for i, node in enumerate(main_path):
            pos[node] = (i * path_spacing, 0)
            
        # Position remaining nodes relative to main path
        remaining_nodes = set(poset.nodes()) - set(main_path)
        
        for node in remaining_nodes:
            # Find closest node in main path
            closest_path_node = self._find_closest_in_path(poset, node, main_path)
            closest_pos = pos[closest_path_node]
            
            # Position relative to closest path node
            offset_y = 1.0 if hash(node) % 2 == 0 else -1.0
            pos[node] = (closest_pos[0], closest_pos[1] + offset_y)
            
        return pos
        
    def _find_main_flow_path(self, poset: nx.DiGraph) -> List[str]:
        """Find the main flow path through the network."""
        # Find input and output nodes
        input_nodes = [n for n in poset.nodes() if poset.in_degree(n) == 0]
        output_nodes = [n for n in poset.nodes() if poset.out_degree(n) == 0]
        
        if not input_nodes or not output_nodes:
            # Fallback to topological sort
            return list(nx.topological_sort(poset))
            
        # Find shortest path from input to output
        try:
            path = nx.shortest_path(poset, input_nodes[0], output_nodes[0])
            return path
        except nx.NetworkXNoPath:
            # Fallback to topological sort
            return list(nx.topological_sort(poset))
            
    def _find_closest_in_path(self,
                             poset: nx.DiGraph,
                             node: str,
                             path: List[str]) -> str:
        """Find the closest node in the main path."""
        # Simple heuristic: find path node with shortest graph distance
        min_distance = float('inf')
        closest_node = path[0]
        
        for path_node in path:
            try:
                distance = nx.shortest_path_length(poset, path_node, node)
                if distance < min_distance:
                    min_distance = distance
                    closest_node = path_node
            except nx.NetworkXNoPath:
                try:
                    distance = nx.shortest_path_length(poset, node, path_node)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = path_node
                except nx.NetworkXNoPath:
                    continue
                    
        return closest_node
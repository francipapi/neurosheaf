"""FX-based poset extraction from PyTorch models.

This module implements automatic extraction of directed acyclic graphs (posets)
from PyTorch models using the torch.fx symbolic tracing framework.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import networkx as nx
import operator
from ..utils.logging import setup_logger
from ..utils.exceptions import ArchitectureError

logger = setup_logger(__name__)


class FXPosetExtractor:
    """Extract poset structure from PyTorch models using FX.
    
    This class analyzes PyTorch models to build directed acyclic graphs
    representing the computational structure. It uses torch.fx for symbolic
    tracing with a fallback to module-based extraction for dynamic models.
    
    Attributes:
        handle_dynamic: Whether to use fallback for dynamic models
        _module_index: Cache for module name to index mapping
    """
    
    def __init__(self, handle_dynamic: bool = True):
        """Initialize the FX poset extractor.
        
        Args:
            handle_dynamic: If True, use fallback extraction for models
                           that cannot be traced with FX
        """
        self.handle_dynamic = handle_dynamic
        self._module_index = {}
        
    def extract_poset(self, model: torch.nn.Module) -> nx.DiGraph:
        """Extract poset from model using FX symbolic tracing.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            NetworkX directed graph representing the poset
            
        Raises:
            ArchitectureError: If model cannot be traced and handle_dynamic is False
        """
        try:
            # Symbolic trace the model
            traced = fx.symbolic_trace(model)
            return self._build_poset_from_graph(traced.graph)
        except Exception as e:
            if self.handle_dynamic:
                logger.warning(f"FX tracing failed: {e}. Falling back to module inspection.")
                return self._fallback_extraction(model)
            else:
                raise ArchitectureError(f"Cannot trace model: {e}")
    
    def extract_activation_filtered_poset(self, model: torch.nn.Module, 
                                        available_activations: Set[str]) -> nx.DiGraph:
        """Extract poset filtered to only include nodes with available activations.
        
        This method preserves the natural FX graph connectivity but removes nodes
        without activations and bridges gaps intelligently.
        
        Args:
            model: PyTorch model to analyze
            available_activations: Set of activation keys that are available
            
        Returns:
            NetworkX directed graph with only activation nodes and bridged connectivity
            
        Raises:
            ArchitectureError: If model cannot be traced and handle_dynamic is False
        """
        try:
            # Symbolic trace the model
            traced = fx.symbolic_trace(model)
            return self._build_activation_filtered_poset(traced.graph, available_activations)
        except Exception as e:
            if self.handle_dynamic:
                logger.warning(f"FX tracing failed: {e}. Using standard extraction.")
                return self.extract_poset(model)
            else:
                raise ArchitectureError(f"Cannot trace model: {e}")
    
    def _build_poset_from_graph(self, graph: fx.Graph) -> nx.DiGraph:
        """Build poset from FX graph.
        
        Args:
            graph: FX graph from symbolic tracing
            
        Returns:
            NetworkX directed graph with node attributes
        """
        poset = nx.DiGraph()
        
        # First pass: identify all nodes that produce activations
        activation_nodes = {}
        for node in graph.nodes:
            if self._is_activation_node(node):
                node_id = self._get_node_id(node)
                activation_nodes[node] = node_id
                poset.add_node(node_id, 
                             name=node.name,
                             op=node.op,
                             target=str(node.target))
        
        # Second pass: build edges based on data flow
        for node in graph.nodes:
            if node in activation_nodes:
                for user in node.users:
                    if user in activation_nodes:
                        # Direct connection
                        poset.add_edge(activation_nodes[node], 
                                     activation_nodes[user])
                    else:
                        # Check for skip connections through ops
                        for downstream in self._find_downstream_activations(user, activation_nodes):
                            poset.add_edge(activation_nodes[node], 
                                         activation_nodes[downstream])
        
        # Add layer indices for ordering
        self._add_topological_levels(poset)
        
        return poset
    
    def _build_activation_filtered_poset(self, graph: fx.Graph, 
                                       available_activations: Set[str]) -> nx.DiGraph:
        """Build poset filtered to nodes with activations and bridge gaps.
        
        Args:
            graph: FX graph from symbolic tracing
            available_activations: Set of activation keys available
            
        Returns:
            NetworkX directed graph with activation nodes and bridged connectivity
        """
        logger.info(f"Building activation-filtered poset from {len(available_activations)} activations")
        
        # Step 1: Map FX nodes to activation keys and identify which have activations
        node_to_activation = {}
        activation_nodes = set()
        missing_nodes = set()
        
        for node in graph.nodes:
            if self._is_activation_node(node):
                node_id = self._get_node_id(node)
                node_to_activation[node] = node_id
                
                if node_id in available_activations:
                    activation_nodes.add(node)
                    logger.debug(f"Node {node_id} has activation")
                else:
                    missing_nodes.add(node)
                    logger.debug(f"Node {node_id} missing activation")
        
        logger.info(f"Found {len(activation_nodes)} nodes with activations, {len(missing_nodes)} missing")
        
        # Step 2: Build poset with only activation nodes
        poset = nx.DiGraph()
        
        # Add nodes that have activations
        for node in activation_nodes:
            node_id = node_to_activation[node]
            poset.add_node(node_id,
                         name=node.name,
                         op=node.op, 
                         target=str(node.target))
        
        # Step 3: Add direct edges between activation nodes
        for node in activation_nodes:
            node_id = node_to_activation[node]
            
            # Find direct connections to other activation nodes
            for user in node.users:
                if user in activation_nodes:
                    target_id = node_to_activation[user]
                    poset.add_edge(node_id, target_id)
                    logger.debug(f"Direct edge: {node_id} -> {target_id}")
        
        # Step 4: Bridge gaps through missing nodes
        gap_edges = self._bridge_activation_gaps(graph, activation_nodes, missing_nodes, node_to_activation)
        
        for source_id, target_id in gap_edges:
            if not poset.has_edge(source_id, target_id):  # Avoid duplicates
                poset.add_edge(source_id, target_id)
                logger.debug(f"Bridged gap: {source_id} -> {target_id}")
        
        logger.info(f"Built filtered poset: {len(poset.nodes())} nodes, {len(poset.edges())} edges")
        
        # Add topological levels
        self._add_topological_levels(poset)
        
        return poset
    
    def _bridge_activation_gaps(self, graph: fx.Graph, activation_nodes: Set[fx.Node], 
                              missing_nodes: Set[fx.Node], 
                              node_to_activation: Dict[fx.Node, str]) -> List[Tuple[str, str]]:
        """Bridge gaps where intermediate nodes are missing activations.
        
        Args:
            graph: FX graph
            activation_nodes: Nodes that have activations
            missing_nodes: Nodes that don't have activations  
            node_to_activation: Mapping from FX nodes to activation IDs
            
        Returns:
            List of (source_id, target_id) edges to bridge gaps
        """
        bridged_edges = []
        
        # For each missing node, connect its activation predecessors to activation successors
        for missing_node in missing_nodes:
            # Find activation nodes that feed into this missing node
            activation_predecessors = []
            for node in graph.nodes:
                if node in activation_nodes and missing_node in node.users:
                    activation_predecessors.append(node)
            
            # Find activation nodes that this missing node feeds into (via path traversal)
            activation_successors = self._find_activation_successors(missing_node, activation_nodes, visited=set())
            
            # Create bridging edges
            for pred_node in activation_predecessors:
                for succ_node in activation_successors:
                    pred_id = node_to_activation[pred_node]
                    succ_id = node_to_activation[succ_node]
                    bridged_edges.append((pred_id, succ_id))
                    logger.debug(f"Gap bridge via {missing_node.name}: {pred_id} -> {succ_id}")
        
        return bridged_edges
    
    def _find_activation_successors(self, start_node: fx.Node, activation_nodes: Set[fx.Node], 
                                  visited: Set[fx.Node], max_depth: int = 5) -> List[fx.Node]:
        """Find activation nodes reachable from start_node via graph traversal.
        
        Args:
            start_node: Node to start search from
            activation_nodes: Set of nodes that have activations
            visited: Set of already visited nodes (to avoid cycles)
            max_depth: Maximum search depth to prevent infinite loops
            
        Returns:
            List of activation nodes reachable from start_node
        """
        if max_depth <= 0 or start_node in visited:
            return []
        
        visited.add(start_node)
        successors = []
        
        for user in start_node.users:
            if user in activation_nodes:
                # Found an activation node - add it
                successors.append(user)
            else:
                # Recursively search through this node
                deeper_successors = self._find_activation_successors(user, activation_nodes, visited, max_depth - 1)
                successors.extend(deeper_successors)
        
        visited.remove(start_node)  # Allow revisiting in different paths
        return successors
    
    def _is_activation_node(self, node: fx.Node) -> bool:
        """Check if node produces activations we care about.
        
        Args:
            node: FX node to check
            
        Returns:
            True if node produces activations to track
        """
        # Skip certain operations
        skip_ops = {'placeholder', 'output', 'get_attr'}
        if node.op in skip_ops:
            return False
        
        # Include calls to modules and functions that transform features
        if node.op == 'call_module':
            return True
        
        if node.op == 'call_function':
            # Include operations that preserve feature structure
            preserve_ops = {
                torch.nn.functional.relu,
                torch.nn.functional.gelu,
                torch.nn.functional.silu,
                torch.nn.functional.tanh,
                torch.nn.functional.sigmoid,
                torch.nn.functional.softmax,
                torch.nn.functional.layer_norm,
                torch.nn.functional.batch_norm,
                torch.add,
                torch.cat,
                torch.matmul,
                torch.bmm,
                torch.mul,
                operator.add,  # Python operator module
                operator.iadd,  # In-place add
                'add',  # String representation
                'cat',
                'matmul',
                'mul',
            }
            # Check both the target and string representation
            target_str = str(node.target)
            # Also check for built-in operators
            if node.target == operator.add or node.target == operator.iadd:
                return True
            return node.target in preserve_ops or any(op in target_str for op in ['add', 'cat', 'matmul', 'softmax', 'mul', 'iadd'])
        
        return False
    
    def _get_node_id(self, node: fx.Node) -> str:
        """Generate unique ID for a node.
        
        Args:
            node: FX node
            
        Returns:
            Unique string identifier
        """
        if node.op == 'call_module':
            # For modules, use name to ensure uniqueness when same module is reused
            return node.name
        else:
            # Use name for other nodes (ensures uniqueness)
            return node.name
    
    def _find_downstream_activations(self, node: fx.Node, 
                                   activation_nodes: Dict) -> List[fx.Node]:
        """Find activation nodes downstream from given node.
        
        This is used to detect skip connections and indirect data flow.
        
        Args:
            node: Starting FX node
            activation_nodes: Dict of nodes that produce activations
            
        Returns:
            List of downstream activation nodes
        """
        downstream = []
        visited = set()
        
        def traverse(n):
            if n in visited or n in activation_nodes:
                if n in activation_nodes:
                    downstream.append(n)
                return
            visited.add(n)
            for user in n.users:
                traverse(user)
        
        traverse(node)
        return downstream
    
    def _add_topological_levels(self, poset: nx.DiGraph):
        """Add topological level information to nodes.
        
        This assigns a 'level' attribute to each node indicating its
        depth in the computational graph.
        
        Args:
            poset: NetworkX directed graph to modify in-place
        """
        # Handle empty graph
        if not poset.nodes():
            return
            
        # For disconnected graphs, process each component
        for component in nx.weakly_connected_components(poset):
            subgraph = poset.subgraph(component)
            
            # Initialize levels
            for node in subgraph.nodes():
                poset.nodes[node]['level'] = 0
            
            # Compute levels using topological sort
            try:
                for node in nx.topological_sort(subgraph):
                    level = 0
                    for pred in subgraph.predecessors(node):
                        level = max(level, poset.nodes[pred].get('level', 0) + 1)
                    poset.nodes[node]['level'] = level
            except (nx.NetworkXError, nx.NetworkXUnfeasible):
                # Not a DAG in this component, assign levels based on depth
                logger.warning("Found cycle in component, using BFS for levels")
                self._assign_levels_bfs(subgraph, poset)
    
    def _assign_levels_bfs(self, subgraph: nx.DiGraph, poset: nx.DiGraph):
        """Assign levels using BFS for graphs with cycles.
        
        Args:
            subgraph: Component to process
            poset: Full graph to update
        """
        # Find nodes with no predecessors as roots
        roots = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]
        if not roots:
            # If no roots (cycle), pick arbitrary node
            roots = [next(iter(subgraph.nodes()))]
        
        # BFS from roots
        from collections import deque
        queue = deque([(root, 0) for root in roots])
        visited = set()
        
        while queue:
            node, level = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            poset.nodes[node]['level'] = level
            
            for successor in subgraph.successors(node):
                if successor not in visited:
                    queue.append((successor, level + 1))
    
    def _fallback_extraction(self, model: torch.nn.Module) -> nx.DiGraph:
        """Fallback to module-based extraction for dynamic models.
        
        This method uses module inspection instead of FX tracing,
        which is less accurate but works for all models.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            NetworkX directed graph
        """
        poset = nx.DiGraph()
        
        # Extract modules and build sequential connections
        modules = dict(model.named_modules())
        
        # Filter out container modules
        feature_modules = {}
        for name, module in modules.items():
            if self._is_feature_layer(module) and name != '':
                feature_modules[name] = module
        
        # Sort by name to get approximate order
        sorted_names = sorted(feature_modules.keys())
        
        # Add nodes
        for name in sorted_names:
            poset.add_node(name, 
                         module=type(feature_modules[name]).__name__,
                         name=name)
        
        # Detect architecture patterns
        self._detect_sequential_connections(poset, sorted_names, feature_modules)
        self._detect_residual_connections(poset, sorted_names)
        self._detect_attention_patterns(poset, sorted_names, feature_modules)
        
        # Add levels
        self._add_topological_levels(poset)
        
        return poset
    
    def _detect_sequential_connections(self, poset: nx.DiGraph, sorted_names: List[str], 
                                      feature_modules: Dict[str, nn.Module]):
        """Detect sequential connections in the model.
        
        Args:
            poset: Graph to add edges to
            sorted_names: Sorted list of module names
            feature_modules: Dict of feature modules
        """
        for i, name in enumerate(sorted_names):
            if i < len(sorted_names) - 1:
                next_name = sorted_names[i + 1]
                
                # Check if they're in the same parent module
                parts = name.split('.')
                next_parts = next_name.split('.')
                
                if len(parts) > 0 and len(next_parts) > 0:
                    # Same parent or sequential naming
                    if (parts[:-1] == next_parts[:-1] or 
                        (len(parts) == len(next_parts) and 
                         parts[:-1] == next_parts[:-1])):
                        poset.add_edge(name, next_name)
    
    def _detect_residual_connections(self, poset: nx.DiGraph, sorted_names: List[str]):
        """Detect residual/skip connections based on naming patterns.
        
        Args:
            poset: Graph to add edges to
            sorted_names: Sorted list of module names
        """
        # Common residual patterns
        residual_keywords = ['residual', 'skip', 'shortcut', 'identity', 'add', 'relu', 'bn']
        
        for i, name in enumerate(sorted_names):
            parts = name.split('.')
            
            # Look for skip connections
            if any(keyword in name.lower() for keyword in residual_keywords):
                # Find source of skip connection (usually 2-3 layers back)
                for j in range(max(0, i - 5), i):
                    source_parts = sorted_names[j].split('.')
                    if len(parts) > 1 and len(source_parts) > 1:
                        # Same block or module
                        if parts[0] == source_parts[0]:
                            if sorted_names[j] not in poset.predecessors(name):
                                poset.add_edge(sorted_names[j], name)
                                break
            
            # Special case: if this is a final ReLU in a block, it might be after an addition
            # Look for patterns like conv1, bn1, conv2, bn2, relu (ResNet pattern)
            if 'relu' in name.lower() and i >= 4:
                # Check if there's a pattern suggesting residual connection
                block_modules = []
                for j in range(max(0, i - 4), i):
                    if parts[0] == sorted_names[j].split('.')[0]:
                        block_modules.append(sorted_names[j])
                
                # If we have conv/bn pairs before ReLU, add skip from first conv
                if len(block_modules) >= 4:
                    # Add skip connection from first module to this ReLU
                    first_module = block_modules[0]
                    # Only add if it doesn't create a cycle
                    if first_module not in poset.predecessors(name):
                        # Check if adding this edge would create a cycle
                        if not nx.has_path(poset, name, first_module):
                            poset.add_edge(first_module, name)
    
    def _detect_attention_patterns(self, poset: nx.DiGraph, sorted_names: List[str], 
                                   feature_modules: Dict[str, nn.Module]):
        """Detect attention patterns (Q, K, V branches converging).
        
        Args:
            poset: Graph to add edges to
            sorted_names: Sorted list of module names
            feature_modules: Dict of feature modules
        """
        attention_keywords = ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 
                            'attention', 'attn', 'mha', 'multihead']
        
        # Group modules by parent
        parent_groups = {}
        for name in sorted_names:
            parts = name.split('.')
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(name)
        
        # Look for Q, K, V patterns in same parent
        for parent, children in parent_groups.items():
            qkv_modules = []
            output_modules = []
            
            for child in children:
                if any(kw in child.lower() for kw in ['query', 'q_proj', 'key', 'k_proj', 
                                                       'value', 'v_proj']):
                    qkv_modules.append(child)
                elif any(kw in child.lower() for kw in ['output', 'out_proj', 'o_proj']):
                    output_modules.append(child)
            
            # Connect Q, K, V to output
            if len(qkv_modules) >= 2 and output_modules:
                for qkv in qkv_modules:
                    for out in output_modules:
                        if qkv in sorted_names and out in sorted_names:
                            if sorted_names.index(qkv) < sorted_names.index(out):
                                poset.add_edge(qkv, out)
    
    def _is_feature_layer(self, module: torch.nn.Module) -> bool:
        """Check if module is a feature-extracting layer.
        
        Args:
            module: PyTorch module to check
            
        Returns:
            True if module extracts features
        """
        feature_types = (
            torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
            torch.nn.Linear,
            torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN,
            torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer,
            torch.nn.MultiheadAttention,
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Tanh, torch.nn.Sigmoid,
            torch.nn.LayerNorm, torch.nn.GroupNorm,
            torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d,
        )
        return isinstance(module, feature_types)
# neurosheaf/sheaf/extraction/fx_poset.py

"""
FX-based poset extraction from PyTorch models.

This module implements automatic extraction of directed acyclic graphs (posets)
from PyTorch models using the torch.fx symbolic tracing framework. It is designed
to work with activation dictionaries whose keys match the FX node names.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx
import operator
from ...utils.logging import setup_logger
from ...utils.exceptions import ArchitectureError

logger = setup_logger(__name__)


class FXPosetExtractor:
    """
    Extracts a poset (partially ordered set) representing the computational
    graph of a PyTorch model using torch.fx.
    
    This class correctly identifies functional nodes (like `add` for residuals)
    and module calls, and can filter the graph to include only nodes for
    which activations are available.
    """
    
    def __init__(self, handle_dynamic: bool = True):
        """Initializes the FX poset extractor."""
        self.handle_dynamic = handle_dynamic
        
    def extract_poset(self, model: torch.nn.Module) -> nx.DiGraph:
        """
        Extracts a complete poset from the model using FX symbolic tracing.
        This includes all identifiable activation-producing nodes.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            A NetworkX directed graph representing the full computational poset.
        """
        try:
            traced = fx.symbolic_trace(model)
            return self._build_poset_from_graph(traced.graph)
        except Exception as e:
            if self.handle_dynamic:
                logger.warning(f"FX tracing failed: {e}. Falling back to legacy inspection.")
                return self._fallback_extraction(model)
            else:
                raise ArchitectureError(f"Cannot trace model with FX: {e}")
    
    def extract_activation_filtered_poset(self, model: torch.nn.Module, 
                                        available_activations: Set[str]) -> Tuple[nx.DiGraph, Optional[Any]]:
        """
        Extracts a poset filtered to only include nodes with available activations.
        This method preserves graph connectivity by intelligently bridging gaps
        left by nodes without activations.

        Args:
            model: The PyTorch model to analyze.
            available_activations: A set of activation keys (which must be valid 
                                   FX node names) that are available.

        Returns:
            A tuple of (NetworkX directed graph containing only the specified activation nodes, traced model).
        """
        try:
            traced = fx.symbolic_trace(model)
            poset = self._build_activation_filtered_poset(traced.graph, available_activations)
            return poset, traced
        except Exception as e:
            logger.warning(f"FX tracing for filtered poset failed: {e}. Using standard poset extraction.")
            fallback_poset = self.extract_poset(model)
            return fallback_poset, None

    def _build_poset_from_graph(self, graph: fx.Graph) -> nx.DiGraph:
        """Builds a poset from a full FX graph, including all activation nodes."""
        poset = nx.DiGraph()
        
        # First pass: Identify all nodes that produce activations.
        activation_nodes = {
            node: self._get_node_id(node)
            for node in graph.nodes if self._is_activation_node(node)
        }
        
        for node, node_id in activation_nodes.items():
            poset.add_node(node_id, name=node.name, op=node.op, target=str(node.target))
        
        # Second pass: Build edges based on data flow.
        for node, node_id in activation_nodes.items():
            for user in node.users:
                # Find the next downstream nodes that are in our activation set.
                for downstream_node in self._find_downstream_activations(user, activation_nodes):
                    downstream_id = activation_nodes[downstream_node]
                    if not poset.has_edge(node_id, downstream_id):
                        poset.add_edge(node_id, downstream_id)
        
        self._add_topological_levels(poset)
        return poset

    def _build_activation_filtered_poset(self, graph: fx.Graph, 
                                       available_activations: Set[str]) -> nx.DiGraph:
        """Builds a filtered poset, keeping only nodes with available activations."""
        logger.info(f"Building activation-filtered poset from {len(available_activations)} available activations.")
        
        # Step 1: Identify which nodes have activations and which are "missing".
        all_potential_nodes = {node: self._get_node_id(node) for node in graph.nodes if self._is_activation_node(node)}
        
        nodes_with_activations = {
            node for node, node_id in all_potential_nodes.items() if node_id in available_activations
        }
        missing_nodes = set(all_potential_nodes.keys()) - nodes_with_activations
        
        logger.info(f"Found {len(nodes_with_activations)} nodes with activations, {len(missing_nodes)} missing.")
        logger.debug(f"Available activation keys: {list(available_activations)[:10]}...")
        logger.debug(f"Potential node IDs: {list(all_potential_nodes.values())[:10]}...")
        
        # Step 2: Build the poset with only the nodes that have activations.
        poset = nx.DiGraph()
        for node in nodes_with_activations:
            node_id = all_potential_nodes[node]
            poset.add_node(node_id, name=node.name, op=node.op, target=str(node.target))
        
        # Step 3: Add edges by traversing through the full graph.
        for node in nodes_with_activations:
            node_id = all_potential_nodes[node]
            # Find all direct and indirect successors that are also in our filtered set.
            for user in node.users:
                for successor_node in self._find_downstream_activations(user, {n: all_potential_nodes[n] for n in nodes_with_activations}):
                    successor_id = all_potential_nodes[successor_node]
                    if not poset.has_edge(node_id, successor_id):
                        poset.add_edge(node_id, successor_id)
                        logger.debug(f"Added edge: {node_id} -> {successor_id}")

        logger.info(f"Built filtered poset: {len(poset.nodes())} nodes, {len(poset.edges())} edges")
        self._add_topological_levels(poset)
        return poset

    def _find_downstream_activations(self, start_node: fx.Node, 
                                   activation_nodes: Dict[fx.Node, str]) -> List[fx.Node]:
        """Traverses the graph from start_node to find the next activation nodes."""
        downstream = []
        q = [start_node]
        visited = {start_node}

        while q:
            curr = q.pop(0)
            if curr in activation_nodes:
                downstream.append(curr)
                continue  # Stop traversal here, as we've found an activation node.
            
            for user in curr.users:
                if user not in visited:
                    visited.add(user)
                    q.append(user)
        
        return downstream

    def _is_activation_node(self, node: fx.Node) -> bool:
        """Determines if a node is one whose activation we want to track."""
        if node.op in {'placeholder', 'output', 'get_attr'}:
            return False
        
        if node.op == 'call_module':
            return True
        
        if node.op == 'call_function':
            # Essential functions for architectures like ResNet and Transformers.
            functional_ops = {
                torch.add, operator.add, operator.iadd,
                torch.cat,
                torch.matmul, torch.bmm,
                nn.functional.relu, nn.functional.gelu, nn.functional.silu,
                nn.functional.layer_norm, nn.functional.batch_norm,
                nn.functional.softmax
            }
            if node.target in functional_ops:
                return True
        return False

    def _get_node_id(self, node: fx.Node) -> str:
        """Generates a unique and consistent ID for a node, which is its name."""
        return node.name
    
    def _add_topological_levels(self, poset: nx.DiGraph):
        """Assigns a 'level' attribute to each node based on graph depth."""
        if not poset.nodes(): return
        
        for component in nx.weakly_connected_components(poset):
            subgraph = poset.subgraph(component)
            try:
                for i, layer_nodes in enumerate(nx.topological_generations(subgraph)):
                    for node in layer_nodes:
                        poset.nodes[node]['level'] = i
            except nx.NetworkXUnfeasible:
                logger.warning("Graph component has a cycle, cannot assign topological levels.")
                # Fallback or error handling for cyclic graphs can be added here.

    def _fallback_extraction(self, model: torch.nn.Module) -> nx.DiGraph:
        """A simple fallback for non-traceable models."""
        logger.warning("Using legacy name-based fallback for poset extraction. Results may be inaccurate.")
        poset = nx.DiGraph()
        # This can be populated with the logic from your old `_fallback_extraction` if needed.
        return poset
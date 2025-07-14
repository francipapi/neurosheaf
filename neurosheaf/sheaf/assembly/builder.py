"""Sheaf builder for constructing cellular sheaves from neural networks.

This module provides the main SheafBuilder class that orchestrates
the complete sheaf construction pipeline using whitened coordinates
for optimal mathematical properties.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import networkx as nx

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)

from ..data_structures import Sheaf, WhiteningInfo
from ..core import (
    WhiteningProcessor, 
    scaled_procrustes_whitened,
    compute_gram_matrices_from_activations,
    validate_sheaf_properties
)
from ..extraction import FXPosetExtractor, create_unified_activation_dict, extract_activations_fx
from .restrictions import RestrictionManager


class BuilderError(Exception):
    """Exception raised during sheaf construction."""
    pass

class SheafBuilder:
    """
    Orchestrates the sheaf construction process with a unified, FX-first approach.
    
    1. Extracts activations with keys matching FX node names.
    2. Extracts a poset, filtered by the available activation keys.
    3. Computes Gram matrices from activations.
    4. Computes restriction maps in whitened coordinates.
    5. Assembles and validates the final Sheaf object.
    """
    
    def __init__(self):
        """Initializes the sheaf builder."""
        self.poset_extractor = FXPosetExtractor()
        self.restriction_manager = RestrictionManager()
        self.whitening_processor = WhiteningProcessor()

    def build_from_activations(self, 
                                model: nn.Module, 
                                input_tensor: torch.Tensor,
                                validate: bool = True) -> Sheaf:
            """
            Builds a sheaf from a model and an example input tensor.

            Args:
                model: The PyTorch model to analyze.
                input_tensor: An example input tensor to run the forward pass.
                validate: Whether to validate the final sheaf's properties.

            Returns:
                A constructed Sheaf object with whitened stalks and restrictions.
            """
            logger.info("Starting sheaf construction from model and input tensor.")
            
            try:
                # 1. Extract activations using the robust FX-based method.
                # The keys of this dictionary are now the ground truth.
                activations = extract_activations_fx(model, input_tensor)
                
                # 2. Extract poset filtered by the keys of our new activation dict.
                available_activations = set(activations.keys())
                poset = self.poset_extractor.extract_activation_filtered_poset(model, available_activations)
                
                # 3. Compute Gram matrices from the extracted activations.
                # We must filter the activations to only those nodes present in the final poset.
                poset_nodes = set(poset.nodes())
                filtered_activations = {k: v for k, v in activations.items() if k in poset_nodes}
                gram_matrices = compute_gram_matrices_from_activations(filtered_activations)
                
                # 4. Compute restriction maps in whitened space.
                # This step internally handles the whitening process.
                restrictions = self.restriction_manager.compute_all_restrictions(gram_matrices, poset, validate=validate)
                
                # 5. Define stalks for the sheaf. For whitened sheaves, stalks are identity matrices.
                # The actual data is in the whitening maps and restriction maps.
                stalks = {}
                for node_id, K in gram_matrices.items():
                    # We need the rank of the Gram matrix to define the dimension of the stalk.
                    rank = torch.linalg.matrix_rank(K.float()).item()
                    stalks[node_id] = torch.eye(int(rank))
                
                # 6. Create the final Sheaf object.
                sheaf = Sheaf(
                    poset=poset,
                    stalks=stalks,
                    restrictions=restrictions,
                    metadata={
                        'construction_method': 'fx_unified_whitened',
                        'nodes': len(poset.nodes()),
                        'edges': len(poset.edges()),
                        'whitened': True
                    }
                )
                
                # 7. Validate the sheaf's mathematical properties.
                if validate:
                    validation_results = validate_sheaf_properties(sheaf.restrictions, sheaf.poset)
                    sheaf.metadata['validation'] = validation_results
                    sheaf.metadata['is_valid'] = validation_results['valid_sheaf']
                
                logger.info("Sheaf construction complete.")
                return sheaf

            except Exception as e:
                logger.error(f"Sheaf construction failed: {e}", exc_info=True)
                raise RuntimeError(f"Sheaf building failed due to: {e}")

    def build_from_graph(self, 
                        graph: nx.Graph, 
                        stalk_dimensions: Dict[str, int],
                        restrictions: Dict[Tuple[str, str], torch.Tensor],
                        validate: bool = True) -> Sheaf:
        """
        Builds a sheaf directly from graph structure and restriction maps.
        
        This method allows manual construction of sheaves for testing and
        research purposes, bypassing the neural network activation extraction.
        
        Args:
            graph: NetworkX graph representing the sheaf structure
            stalk_dimensions: Dictionary mapping node names to stalk dimensions
            restrictions: Dictionary mapping edges to restriction map tensors
            validate: Whether to validate the resulting sheaf's mathematical properties
            
        Returns:
            A constructed Sheaf object
            
        Raises:
            BuilderError: If construction fails
        """
        logger.info(f"Building sheaf from graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        try:
            # Convert to directed graph if needed
            if isinstance(graph, nx.Graph):
                poset = nx.DiGraph()
                poset.add_nodes_from(graph.nodes())
                poset.add_edges_from(graph.edges())
            else:
                poset = graph.copy()
            
            # Validate input consistency
            self._validate_graph_input(poset, stalk_dimensions, restrictions)
            
            # Create stalks as identity matrices of specified dimensions
            stalks = {}
            for node, dimension in stalk_dimensions.items():
                if dimension <= 0:
                    raise BuilderError(f"Stalk dimension for node {node} must be positive, got {dimension}")
                stalks[node] = torch.eye(dimension)
            
            # Validate restriction map dimensions
            self._validate_restriction_dimensions(poset, stalk_dimensions, restrictions)
            
            # Create the Sheaf object
            sheaf = Sheaf(
                poset=poset,
                stalks=stalks,
                restrictions=restrictions,
                metadata={
                    'construction_method': 'graph_based',
                    'nodes': len(poset.nodes()),
                    'edges': len(poset.edges()),
                    'whitened': False,  # Not whitened since built from raw specifications
                    'manual_construction': True
                }
            )
            
            # Validate sheaf properties if requested
            if validate:
                validation_results = self._validate_graph_sheaf_properties(sheaf)
                sheaf.metadata['validation'] = validation_results
                sheaf.metadata['is_valid'] = validation_results.get('basic_properties_valid', False)
                
                if not validation_results.get('basic_properties_valid', False):
                    logger.warning("Sheaf validation failed. Check restriction map compatibility.")
            
            logger.info(f"Graph-based sheaf construction complete. Valid: {sheaf.metadata.get('is_valid', False)}")
            return sheaf
            
        except Exception as e:
            logger.error(f"Graph-based sheaf construction failed: {e}", exc_info=True)
            raise BuilderError(f"Graph-based sheaf building failed: {e}")
    
    def _validate_graph_input(self, poset: nx.DiGraph, 
                            stalk_dimensions: Dict[str, int],
                            restrictions: Dict[Tuple[str, str], torch.Tensor]) -> None:
        """Validate input for graph-based sheaf construction."""
        # Check that all nodes have stalk dimensions
        for node in poset.nodes():
            if node not in stalk_dimensions:
                raise BuilderError(f"Missing stalk dimension for node {node}")
        
        # Check that all edges have restriction maps
        for edge in poset.edges():
            if edge not in restrictions:
                raise BuilderError(f"Missing restriction map for edge {edge}")
        
        # Check for extra restrictions not in graph
        for edge in restrictions.keys():
            if edge not in poset.edges():
                logger.warning(f"Restriction map for edge {edge} not in graph. Ignoring.")
    
    def _validate_restriction_dimensions(self, poset: nx.DiGraph,
                                       stalk_dimensions: Dict[str, int],
                                       restrictions: Dict[Tuple[str, str], torch.Tensor]) -> None:
        """Validate that restriction maps have correct dimensions."""
        for edge, restriction in restrictions.items():
            if edge not in poset.edges():
                continue
                
            u, v = edge
            expected_source_dim = stalk_dimensions[u]
            expected_target_dim = stalk_dimensions[v]
            
            actual_target_dim, actual_source_dim = restriction.shape
            
            if actual_source_dim != expected_source_dim:
                raise BuilderError(
                    f"Restriction map {edge} has wrong source dimension. "
                    f"Expected {expected_source_dim}, got {actual_source_dim}"
                )
            
            if actual_target_dim != expected_target_dim:
                raise BuilderError(
                    f"Restriction map {edge} has wrong target dimension. "
                    f"Expected {expected_target_dim}, got {actual_target_dim}"
                )
    
    def _validate_graph_sheaf_properties(self, sheaf: Sheaf) -> Dict[str, any]:
        """Validate basic mathematical properties of graph-constructed sheaf."""
        results = {
            'basic_properties_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check that poset is acyclic
            if not nx.is_directed_acyclic_graph(sheaf.poset):
                results['basic_properties_valid'] = False
                results['errors'].append("Poset is not acyclic")
            
            # Check restriction map dimensions
            for edge, restriction in sheaf.restrictions.items():
                if edge not in sheaf.poset.edges():
                    results['warnings'].append(f"Restriction for edge {edge} not in poset")
                    continue
                
                u, v = edge
                if u not in sheaf.stalks or v not in sheaf.stalks:
                    results['basic_properties_valid'] = False
                    results['errors'].append(f"Missing stalks for edge {edge}")
                    continue
                
                expected_source_dim = sheaf.stalks[u].shape[0]
                expected_target_dim = sheaf.stalks[v].shape[0]
                actual_target_dim, actual_source_dim = restriction.shape
                
                if actual_source_dim != expected_source_dim or actual_target_dim != expected_target_dim:
                    results['basic_properties_valid'] = False
                    results['errors'].append(
                        f"Dimension mismatch for edge {edge}: "
                        f"restriction is {restriction.shape}, "
                        f"expected ({expected_target_dim}, {expected_source_dim})"
                    )
            
            # Check for isolated nodes
            isolated_nodes = list(nx.isolates(sheaf.poset))
            if isolated_nodes:
                results['warnings'].append(f"Isolated nodes: {isolated_nodes}")
            
            # Additional mathematical checks could be added here
            # (e.g., transitivity for longer paths, orthogonality tests, etc.)
            
        except Exception as e:
            results['basic_properties_valid'] = False
            results['errors'].append(f"Validation failed with exception: {e}")
        
        return results

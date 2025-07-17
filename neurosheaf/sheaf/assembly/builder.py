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
    compute_gram_matrices_with_regularization,
    validate_sheaf_properties,
    AdaptiveTikhonovRegularizer
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
                                validate: bool = True,
                                use_gram_regularization: bool = False,
                                regularization_config: Optional[Dict[str, Any]] = None) -> Sheaf:
            """
            Builds a sheaf from a model and an example input tensor.

            Args:
                model: The PyTorch model to analyze.
                input_tensor: An example input tensor to run the forward pass.
                validate: Whether to validate the final sheaf's properties.
                use_gram_regularization: Whether to apply Tikhonov regularization to Gram matrices.
                regularization_config: Configuration for Tikhonov regularization (if None, uses defaults).

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
                poset, traced_model = self.poset_extractor.extract_activation_filtered_poset(model, available_activations)
                
                # 3. Compute Gram matrices from the extracted activations.
                # We must filter the activations to only those nodes present in the final poset.
                poset_nodes = set(poset.nodes())
                filtered_activations = {k: v for k, v in activations.items() if k in poset_nodes}
                
                # Apply Tikhonov regularization if requested
                if use_gram_regularization:
                    # Create regularizer from config or use defaults
                    if regularization_config is not None:
                        from ..core import create_regularizer_from_config
                        regularizer = create_regularizer_from_config(regularization_config)
                    else:
                        # Use adaptive strategy by default
                        regularizer = AdaptiveTikhonovRegularizer(strategy='adaptive')
                    
                    # Infer batch size from input tensor
                    batch_size = input_tensor.shape[0]
                    
                    # Compute regularized Gram matrices
                    gram_matrices, regularization_info = compute_gram_matrices_with_regularization(
                        filtered_activations, 
                        regularizer=regularizer,
                        batch_size=batch_size,
                        validate=validate
                    )
                    
                    logger.info(f"Applied Tikhonov regularization to {len(gram_matrices)} layers")
                    
                    # Log regularization details for problematic cases
                    for layer_name, reg_info in regularization_info.items():
                        if reg_info.get('regularized', False):
                            condition_before = reg_info.get('condition_number', 'N/A')
                            condition_after = reg_info.get('post_condition_number', 'N/A')
                            lambda_val = reg_info.get('regularization_strength', 'N/A')
                            logger.info(f"Layer {layer_name}: λ={lambda_val:.2e}, "
                                      f"condition {condition_before:.2e} → {condition_after:.2e}")
                else:
                    # Standard Gram matrix computation without regularization
                    gram_matrices = compute_gram_matrices_from_activations(filtered_activations)
                    regularization_info = {}
                
                # 4. First, compute consistent ranks using torch.linalg.matrix_rank
                # Then whiten all Gram matrices using those ranks for consistency
                whitening_info = {}
                whitened_grams = {}
                
                logger.info("Computing consistent ranks and whitening transformations for all nodes")
                
                # Step 4a: Compute ranks consistently using torch.linalg.matrix_rank
                node_ranks = {}
                for node_id, K in gram_matrices.items():
                    rank = torch.linalg.matrix_rank(K.float()).item()
                    node_ranks[node_id] = rank
                    logger.debug(f"Node {node_id}: computed rank={rank}")
                
                # Step 4b: Whiten matrices with consistent rank truncation
                for node_id, K in gram_matrices.items():
                    target_rank = node_ranks[node_id]
                    
                    # Use double precision for better numerical stability
                    K_double = K.double()
                    
                    # Perform eigendecomposition in double precision
                    eigenvals, eigenvecs = torch.linalg.eigh(K_double)
                    eigenvals = eigenvals.real
                    eigenvecs = eigenvecs.real
                    
                    # Sort in descending order
                    indices = torch.argsort(eigenvals, descending=True)
                    eigenvals = eigenvals[indices]
                    eigenvecs = eigenvecs[:, indices]
                    
                    # Truncate to target rank (keep only top eigenvalues)
                    eigenvals_trunc = eigenvals[:target_rank]
                    eigenvecs_trunc = eigenvecs[:, :target_rank]
                    
                    # Create whitening matrix: W = Lambda^{-1/2} @ V^T  
                    # where eigenvals_trunc are positive by construction
                    sqrt_inv_eigenvals = torch.sqrt(1.0 / (eigenvals_trunc + 1e-15))  # Better precision
                    W = torch.diag(sqrt_inv_eigenvals) @ eigenvecs_trunc.T  # (rank × 500)
                    
                    # Convert back to float32 for consistency with rest of pipeline
                    W = W.float()
                    
                    # Verify whitening: K_whitened should be identity
                    K_whitened = W @ K @ W.T
                    
                    whitening_info[node_id] = {
                        'whitening_matrix': W,
                        'eigenvalues': eigenvals_trunc,
                        'eigenvectors': eigenvecs_trunc,
                        'rank': target_rank
                    }
                    whitened_grams[node_id] = K_whitened
                    logger.debug(f"Node {node_id}: whitened with rank={target_rank}, W.shape={W.shape}")
                
                # 5. Define stalks based on whitened ranks (consistent with restrictions)
                # For whitened sheaves, stalks are identity matrices of the whitened rank
                stalks = {}
                for node_id, info in whitening_info.items():
                    rank = info['rank']
                    stalks[node_id] = torch.eye(int(rank), dtype=torch.float32)
                    logger.debug(f"Created stalk for {node_id}: {stalks[node_id].shape}")
                
                # 6. Compute restriction maps using the same whitening information
                # This ensures dimensional consistency between stalks and restrictions
                restrictions = self.restriction_manager.compute_all_restrictions_with_whitening(
                    gram_matrices, 
                    whitening_info,
                    poset, 
                    validate=validate,
                    regularization_info=regularization_info if use_gram_regularization else None
                )
                
                # 7. Create module type mapping for visualization
                module_types = {}
                if traced_model is not None:
                    try:
                        for node_id in poset.nodes():
                            node_attrs = poset.nodes[node_id]
                            if node_attrs.get('op') == 'call_module':
                                target = node_attrs.get('target', '')
                                try:
                                    module = traced_model.get_submodule(target)
                                    module_types[node_id] = type(module)
                                    module_types[target] = type(module)  # Also store by target
                                except:
                                    pass
                    except Exception as e:
                        logger.debug(f"Could not extract module types: {e}")
                
                # 7. Create the final Sheaf object.
                sheaf = Sheaf(
                    poset=poset,
                    stalks=stalks,
                    restrictions=restrictions,
                    metadata={
                        'construction_method': 'fx_unified_whitened',
                        'nodes': len(poset.nodes()),
                        'edges': len(poset.edges()),
                        'whitened': True,
                        'gram_regularized': use_gram_regularization,
                        'regularization_info': regularization_info if use_gram_regularization else None,
                        'batch_size': input_tensor.shape[0],
                        'whitening_info': whitening_info,
                        'stalk_ranks': {node_id: info['rank'] for node_id, info in whitening_info.items()},
                        'traced_model': traced_model,
                        'module_types': module_types
                    }
                )
                
                # 8. Validate the sheaf's mathematical properties.
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

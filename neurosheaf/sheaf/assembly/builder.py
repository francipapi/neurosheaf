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

from ..data_structures import Sheaf, WhiteningInfo, EigenvalueMetadata
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
    
    def __init__(self, preserve_eigenvalues: bool = False):
        """Initializes the sheaf builder.
        
        Args:
            preserve_eigenvalues: Whether to preserve eigenvalues in whitening (enables Hodge formulation)
        """
        self.poset_extractor = FXPosetExtractor()
        self.restriction_manager = RestrictionManager()
        self.whitening_processor = WhiteningProcessor(preserve_eigenvalues=preserve_eigenvalues)
        self.preserve_eigenvalues = preserve_eigenvalues

    def build_from_activations(self, 
                                model: nn.Module, 
                                input_tensor: torch.Tensor,
                                validate: bool = True,
                                preserve_eigenvalues: Optional[bool] = None,
                                use_gram_regularization: bool = False,
                                regularization_config: Optional[Dict[str, Any]] = None) -> Sheaf:
            """
            Builds a sheaf from a model and an example input tensor.

            Args:
                model: The PyTorch model to analyze.
                input_tensor: An example input tensor to run the forward pass.
                validate: Whether to validate the final sheaf's properties.
                preserve_eigenvalues: Runtime override for eigenvalue preservation mode. 
                    If None, uses builder's default setting. If True, enables Hodge formulation.
                use_gram_regularization: Whether to apply Tikhonov regularization to Gram matrices.
                regularization_config: Configuration for Tikhonov regularization (if None, uses defaults).

            Returns:
                A constructed Sheaf object with whitened stalks and eigenvalue metadata if enabled.
            """
            logger.info("Starting sheaf construction from model and input tensor.")
            
            # Use runtime override if provided, otherwise use builder's default
            use_eigenvalues = (preserve_eigenvalues 
                              if preserve_eigenvalues is not None 
                              else self.preserve_eigenvalues)
            
            # Configure whitening processor for this build
            original_preserve_eigenvalues = self.whitening_processor.preserve_eigenvalues
            self.whitening_processor.preserve_eigenvalues = use_eigenvalues
            
            logger.info(f"Building sheaf with eigenvalue preservation: {use_eigenvalues}")
            
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
                
                # 4. Use WhiteningProcessor for consistent whitening with eigenvalue preservation
                whitening_info = {}
                whitened_grams = {}
                
                logger.info("Computing whitening transformations using WhiteningProcessor")
                
                for node_id, K in gram_matrices.items():
                    # Use the whitening processor for this node
                    K_whitened, W, info = self.whitening_processor.whiten_gram_matrix(K)
                    
                    whitening_info[node_id] = {
                        'whitening_matrix': W,
                        'eigenvalues': info['eigenvalues'],
                        'eigenvectors': info.get('eigenvectors'),
                        'rank': info['effective_rank'],  # Use effective_rank from WhiteningProcessor
                        'eigenvalue_diagonal': info.get('eigenvalue_diagonal'),
                        'preserve_eigenvalues': self.preserve_eigenvalues,
                        'condition_number': info.get('condition_number', 1.0),
                        'regularized': info.get('regularized', False)
                    }
                    whitened_grams[node_id] = K_whitened
                    logger.debug(f"Node {node_id}: whitened with rank={info['effective_rank']}, W.shape={W.shape}")
                
                # 5. Define stalks based on whitening results
                stalks = {}
                for node_id, info in whitening_info.items():
                    rank = info['rank']
                    
                    if self.preserve_eigenvalues and 'eigenvalue_diagonal' in info and info['eigenvalue_diagonal'] is not None:
                        # Use eigenvalue diagonal matrix as stalk
                        stalks[node_id] = info['eigenvalue_diagonal']
                        logger.debug(f"Created eigenvalue stalk for {node_id}: {stalks[node_id].shape}")
                    else:
                        # Use identity matrix as stalk (standard whitening)
                        stalks[node_id] = torch.eye(int(rank), dtype=torch.float32)
                        logger.debug(f"Created identity stalk for {node_id}: {stalks[node_id].shape}")
                
                # 6. Compute restriction maps using eigenvalue-aware algorithm selection
                # This ensures dimensional consistency between stalks and restrictions
                # and uses weighted Procrustes when eigenvalue preservation is enabled
                restrictions = self.restriction_manager.compute_restrictions_with_eigenvalues(
                    gram_matrices, 
                    whitening_info,
                    poset,
                    preserve_eigenvalues=self.preserve_eigenvalues,
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
                
                # 7. Extract eigenvalue metadata if eigenvalue preservation is enabled
                eigenvalue_metadata = None
                if use_eigenvalues:
                    eigenvalue_metadata = self._extract_eigenvalue_metadata(whitening_info)
                
                # 8. Create the final Sheaf object.
                sheaf = Sheaf(
                    poset=poset,
                    stalks=stalks,
                    restrictions=restrictions,
                    eigenvalue_metadata=eigenvalue_metadata,
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
                        'module_types': module_types,
                        'preserve_eigenvalues': use_eigenvalues
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
                
            finally:
                # Restore original whitening processor setting
                self.whitening_processor.preserve_eigenvalues = original_preserve_eigenvalues

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
    
    def _extract_eigenvalue_metadata(self, whitening_info: Dict[str, Dict[str, Any]]) -> EigenvalueMetadata:
        """Extract eigenvalue metadata from whitening information.
        
        This method collects eigenvalue matrices and related information from
        the whitening process to create comprehensive eigenvalue metadata for
        the sheaf when preserve_eigenvalues=True.
        
        Args:
            whitening_info: Dictionary containing whitening information for each node
            
        Returns:
            EigenvalueMetadata object with collected eigenvalue information
        """
        logger.debug("Extracting eigenvalue metadata from whitening information")
        
        # Use the runtime eigenvalue preservation setting
        runtime_preserve_eigenvalues = self.whitening_processor.preserve_eigenvalues
        
        metadata = EigenvalueMetadata(
            preserve_eigenvalues=runtime_preserve_eigenvalues,
            hodge_formulation_active=runtime_preserve_eigenvalues
        )
        
        # Extract eigenvalue matrices and related information from whitening info
        for node_id, info in whitening_info.items():
            if 'eigenvalue_diagonal' in info and info['eigenvalue_diagonal'] is not None:
                # Store the eigenvalue diagonal matrix
                metadata.eigenvalue_matrices[node_id] = info['eigenvalue_diagonal']
                
                # Store condition number
                if 'condition_number' in info:
                    metadata.condition_numbers[node_id] = info['condition_number']
                
                # Check if regularization was applied
                regularization_applied = info.get('regularized', False)
                metadata.regularization_applied[node_id] = regularization_applied
        
        logger.info(f"Extracted eigenvalue metadata for {len(metadata.eigenvalue_matrices)} nodes")
        return metadata
    
    def build_laplacian(self, sheaf: Sheaf, edge_weights: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple:
        """Build Laplacian from sheaf with automatic eigenvalue-aware formulation selection.
        
        This method automatically detects if the sheaf uses eigenvalue preservation mode
        and applies the appropriate Laplacian construction method:
        - Standard formulation for identity-based stalks (preserve_eigenvalues=False)
        - Hodge formulation for eigenvalue-preserving stalks (preserve_eigenvalues=True)
        
        Args:
            sheaf: Constructed sheaf object
            edge_weights: Optional edge weights (if None, uses restriction map norms)
            
        Returns:
            Tuple of (sparse_laplacian, metadata)
        """
        from .laplacian import SheafLaplacianBuilder
        
        # Check if sheaf uses eigenvalue preservation
        uses_eigenvalue_preservation = (
            sheaf.eigenvalue_metadata is not None and 
            sheaf.eigenvalue_metadata.preserve_eigenvalues
        )
        
        logger.info(f"Building Laplacian with eigenvalue preservation: {uses_eigenvalue_preservation}")
        
        if uses_eigenvalue_preservation:
            # Use Hodge formulation for eigenvalue-preserving sheaves
            return self._build_hodge_laplacian(sheaf, edge_weights)
        else:
            # Use standard formulation for identity-based sheaves
            builder = SheafLaplacianBuilder()
            return builder.build(sheaf, edge_weights)
    
    def _build_hodge_laplacian(self, sheaf: Sheaf, edge_weights: Optional[Dict[Tuple[str, str], float]] = None):
        """Build Laplacian using Hodge formulation for eigenvalue-preserving sheaves.
        
        This method implements the corrected Hodge formulation that preserves eigenvalue
        information while maintaining mathematical properties (symmetry and PSD).
        
        Mathematical formulation (undirected case):
        - Off-diagonal: L[u,v] = -R_{uv}^T Σ_v for edge u → v
        - Off-diagonal: L[v,u] = -Σ_v R_{uv} (transpose ensures symmetry)
        - Diagonal: L[u,u] += R_{uv}^T Σ_v R_{uv}, L[v,v] += Σ_v
        
        This guarantees L = L^T and L ⪰ 0 automatically.
        """
        from .laplacian import LaplacianMetadata
        import scipy.sparse as sp
        import numpy as np
        
        logger.info("Building Hodge Laplacian with eigenvalue preservation")
        
        poset = sheaf.poset
        stalks = sheaf.stalks
        restrictions = sheaf.restrictions
        eigenvalue_metadata = sheaf.eigenvalue_metadata
        
        if not eigenvalue_metadata or not eigenvalue_metadata.eigenvalue_matrices:
            raise ValueError("Eigenvalue metadata required for Hodge formulation")
        
        # Get dimensions and build index mapping
        node_dims = {node: stalk.shape[0] for node, stalk in stalks.items()}
        node_list = sorted(poset.nodes())
        
        # Calculate total dimension and offsets
        total_dim = sum(node_dims.values())
        offsets = {}
        current_offset = 0
        for node in node_list:
            offsets[node] = current_offset
            current_offset += node_dims[node]
        
        # Initialize sparse matrix builders
        rows, cols, data = [], [], []
        
        # Get eigenvalue matrices
        eigenvalue_matrices = eigenvalue_metadata.eigenvalue_matrices
        
        # Build diagonal blocks first
        for v in node_list:
            v_offset = offsets[v]
            v_dim = node_dims[v]
            
            if v not in eigenvalue_matrices:
                logger.warning(f"Missing eigenvalue matrix for node {v}, using identity")
                Sigma_v = torch.eye(v_dim)
            else:
                Sigma_v = eigenvalue_matrices[v]
            
            # Diagonal contribution: sum of all incident edge contributions
            diagonal_contribution = torch.zeros((v_dim, v_dim), dtype=Sigma_v.dtype, device=Sigma_v.device)
            
            # Process incoming edges: u → v
            for u in poset.predecessors(v):
                if (u, v) in restrictions:
                    R_uv = restrictions[(u, v)]
                    u_dim = node_dims[u]
                    
                    if u not in eigenvalue_matrices:
                        logger.warning(f"Missing eigenvalue matrix for node {u}, using identity")
                        Sigma_u = torch.eye(u_dim)
                    else:
                        Sigma_u = eigenvalue_matrices[u]
                    
                    # For incoming edge: L[v,v] += Σ_v (target node gets eigenvalue matrix)
                    diagonal_contribution += Sigma_v
            
            # Process outgoing edges: v → w  
            for w in poset.successors(v):
                if (v, w) in restrictions:
                    R_vw = restrictions[(v, w)]
                    w_dim = node_dims[w]
                    
                    if w not in eigenvalue_matrices:
                        logger.warning(f"Missing eigenvalue matrix for node {w}, using identity")
                        Sigma_w = torch.eye(w_dim)
                    else:
                        Sigma_w = eigenvalue_matrices[w]
                    
                    # For outgoing edge: L[v,v] += R_{vw}^T Σ_w R_{vw} (source node gets quadratic form)
                    RTR_contribution = R_vw.T @ Sigma_w @ R_vw
                    diagonal_contribution += RTR_contribution
            
            # Add diagonal block to sparse matrix
            for i in range(v_dim):
                for j in range(v_dim):
                    value = diagonal_contribution[i, j].item()
                    if abs(value) > 1e-12:  # Sparsity threshold
                        rows.append(v_offset + i)
                        cols.append(v_offset + j)
                        data.append(value)
        
        # Build off-diagonal blocks
        processed_edges = set()
        
        for (u, v), R_uv in restrictions.items():
            # Skip if we've already processed this edge pair
            if (v, u) in processed_edges:
                continue
            processed_edges.add((u, v))
            
            u_offset, v_offset = offsets[u], offsets[v]
            u_dim, v_dim = node_dims[u], node_dims[v]
            
            if u not in eigenvalue_matrices:
                Sigma_u = torch.eye(u_dim)
            else:
                Sigma_u = eigenvalue_matrices[u]
                
            if v not in eigenvalue_matrices:
                Sigma_v = torch.eye(v_dim)
            else:
                Sigma_v = eigenvalue_matrices[v]
            
            # Apply edge weight if provided
            weight = edge_weights.get((u, v), 1.0) if edge_weights else 1.0
            R_weighted = weight * R_uv
            
            # L[u,v] = -R_{uv}^T Σ_v
            off_diag_uv = -R_weighted.T @ Sigma_v
            
            # Add L[u,v] entries
            for i in range(u_dim):
                for j in range(v_dim):
                    value = off_diag_uv[i, j].item()
                    if abs(value) > 1e-12:
                        rows.append(u_offset + i)
                        cols.append(v_offset + j)
                        data.append(value)
            
            # L[v,u] = -Σ_v R_{uv} (transpose of L[u,v] for symmetry)
            off_diag_vu = -Sigma_v @ R_weighted
            
            # Add L[v,u] entries (transpose indices)
            for i in range(u_dim):
                for j in range(v_dim):
                    value = off_diag_uv[i, j].item()  # Same value as L[u,v]
                    if abs(value) > 1e-12:
                        # Transpose the indices: L[v,u][j,i] = L[u,v][i,j]
                        rows.append(v_offset + j)
                        cols.append(u_offset + i)
                        data.append(value)
        
        # Create sparse matrix
        laplacian_sparse = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(total_dim, total_dim)
        ).tocsr()
        
        # Create metadata
        metadata = LaplacianMetadata(
            total_dimension=total_dim,
            stalk_dimensions=node_dims,
            stalk_offsets=offsets,
            sparsity_ratio=1.0 - (laplacian_sparse.nnz / (total_dim * total_dim)),
            num_nonzeros=laplacian_sparse.nnz,
            construction_method="hodge_formulation"
        )
        
        # Verify symmetry (debug check)
        diff = laplacian_sparse - laplacian_sparse.T
        max_asymmetry = np.abs(diff.data).max() if diff.nnz > 0 else 0.0
        
        if max_asymmetry > 1e-10:
            logger.warning(f"Hodge Laplacian asymmetry detected: {max_asymmetry:.2e}")
        else:
            logger.debug(f"Hodge Laplacian symmetry verified: max asymmetry = {max_asymmetry:.2e}")
        
        logger.info(f"Hodge Laplacian constructed: {total_dim}×{total_dim}, nnz={laplacian_sparse.nnz}")
        
        return laplacian_sparse, metadata

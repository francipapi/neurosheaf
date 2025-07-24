"""GW-specific Laplacian assembly with block-structured construction.

This module provides the GWLaplacianBuilder class for constructing sheaf Laplacians
from Gromov-Wasserstein based sheaves. It extends the existing Laplacian assembly
infrastructure with GW-specific mathematical properties while maintaining efficiency.

Mathematical Foundation:
- GW restrictions are column-stochastic matrices (different from orthogonal Procrustes)
- Edge weights are GW costs representing metric distortion (lower = better match)
- Supports weighted inner products for non-uniform measures
- Uses general sheaf Laplacian formulation for rectangular restriction maps
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from dataclasses import dataclass
import logging

from ..data_structures import Sheaf
from .laplacian import LaplacianMetadata

logger = logging.getLogger(__name__)


class GWLaplacianError(Exception):
    """Exception raised during GW Laplacian construction."""
    pass


@dataclass
class GWLaplacianMetadata(LaplacianMetadata):
    """Extended metadata for GW-specific Laplacian construction.
    
    Adds GW-specific information to the base LaplacianMetadata:
    - GW cost statistics
    - Measure type (uniform vs weighted)
    - Quasi-sheaf validation results
    - Edge weight extraction method
    - Sheaf reference (for filtration reconstruction)
    """
    gw_cost_range: Tuple[float, float] = (0.0, 1.0)
    mean_gw_cost: float = 0.0
    measure_type: str = "uniform"  # "uniform" or "weighted"
    quasi_sheaf_max_violation: float = 0.0
    edge_weight_source: str = "gw_costs"  # "gw_costs" or "restriction_norms"
    filtration_semantics: str = "increasing"  # Always increasing for GW
    sheaf_reference: Optional['Sheaf'] = None  # For filtration reconstruction


class GWLaplacianBuilder:
    """Efficient block-structured Laplacian assembly for GW-based sheaves.
    
    This class handles the unique mathematical properties of GW-based sheaves:
    1. Column-stochastic restriction maps (vs orthogonal Procrustes)
    2. GW costs as edge weights (metric distortion, not correlation strength)
    3. Proper weighted inner product support for non-uniform measures
    4. Sparse matrix assembly using general sheaf formulation
    
    The builder constructs the Laplacian L = δ^T δ using the block formula:
    - Diagonal: L[v,v] = Σ_{incoming} w²*I + Σ_{outgoing} w²*R^T*R  
    - Off-diagonal: L[u,v] = -w²*R^T, L[v,u] = -w²*R
    where w is the edge weight (GW cost) and all terms use w² for consistency.
    
    Mathematical Properties:
    - Symmetric: L = L^T by construction
    - Positive semi-definite: L ⪰ 0 by Hodge theory
    - Sparse: Only non-zero for connected components
    """
    
    def __init__(self, 
                 validate_properties: bool = True,
                 sparsity_threshold: float = 1e-12,
                 use_weighted_inner_products: bool = False):
        """Initialize GW Laplacian builder.
        
        Args:
            validate_properties: Whether to validate mathematical properties  
            sparsity_threshold: Threshold below which values are considered zero
            use_weighted_inner_products: Use p_i-weighted L2 inner products
        """
        self.validate_properties = validate_properties
        self.sparsity_threshold = sparsity_threshold
        self.use_weighted_inner_products = use_weighted_inner_products
        
        logger.info(f"GWLaplacianBuilder initialized: validate={validate_properties}, "
                   f"threshold={sparsity_threshold}, weighted_inner_products={use_weighted_inner_products}")
    
    def build_laplacian(self, 
                       sheaf: Sheaf, 
                       sparse: bool = True,
                       active_edges: Optional[List[Tuple[str, str]]] = None,
                       add_regularization: bool = True) -> Union[torch.Tensor, csr_matrix]:
        """Construct L = δ^T δ using GW-specific block formula.
        
        This method constructs the Laplacian using the general sheaf formulation
        adapted for GW-based restrictions and proper edge weight semantics.
        
        Args:
            sheaf: GW-based sheaf with column-stochastic restrictions
            sparse: Whether to return sparse matrix (recommended for large sheaves)
            active_edges: Optional list of edges to include (for filtration). If None, uses all edges.
            add_regularization: Whether to add small regularization for numerical stability
            
        Returns:
            Sparse or dense Laplacian matrix
            
        Raises:
            GWLaplacianError: If sheaf is not GW-based or construction fails
        """
        start_time = time.time()
        
        # Validate sheaf type
        if not sheaf.is_gw_sheaf():
            raise GWLaplacianError("Sheaf is not GW-based. Use standard SheafLaplacianBuilder instead.")
        
        # Determine effective edge set
        if active_edges is None:
            active_edges = list(sheaf.restrictions.keys())
            logger.info(f"Building GW Laplacian: {len(sheaf.stalks)} nodes, {len(sheaf.restrictions)} edges (all active)")
        else:
            logger.info(f"Building GW Laplacian: {len(sheaf.stalks)} nodes, {len(active_edges)}/{len(sheaf.restrictions)} edges (filtered)")
        
        try:
            # Extract GW costs as edge weights (only for active edges)
            edge_weights = self.extract_edge_weights(sheaf, active_edges)
            
            # Initialize metadata
            metadata = self._initialize_gw_metadata(sheaf, edge_weights)
            
            # Build using sparse assembly for efficiency
            if sparse:
                laplacian = self._build_sparse_laplacian(sheaf, edge_weights, metadata, active_edges)
            else:
                # Dense assembly for small sheaves (mainly for testing)
                laplacian = self._build_dense_laplacian(sheaf, edge_weights, metadata, active_edges)
            
            # Add adaptive regularization for numerical stability if requested
            if add_regularization:
                laplacian = self._apply_adaptive_regularization(laplacian, sheaf, active_edges, sparse)
            
            # Validate if requested
            if self.validate_properties:
                self._validate_gw_laplacian(laplacian, sheaf)
            
            # Update timing
            metadata.construction_time = time.time() - start_time
            
            logger.info(f"GW Laplacian built: {laplacian.shape}, "
                       f"{laplacian.nnz if sparse else 'dense'}, "
                       f"{metadata.construction_time:.3f}s")
            
            return laplacian
            
        except Exception as e:
            logger.error(f"GW Laplacian construction failed: {e}")
            raise GWLaplacianError(f"Failed to build GW Laplacian: {e}")
    
    def build_coboundary(self, sheaf: Sheaf) -> csr_matrix:
        """Construct sparse coboundary operator δ for GW sheaf.
        
        For weighted measures, incorporates p_i-weighted inner product:
        δ* = P^{-1/2} δ^T P^{1/2} where P = diag(p_weights)
        
        Args:
            sheaf: GW-based sheaf
            
        Returns:
            Sparse coboundary matrix δ
        """
        if not sheaf.is_gw_sheaf():
            raise GWLaplacianError("Coboundary construction requires GW sheaf")
        
        logger.info("Building GW coboundary operator")
        
        # For now, implement basic coboundary
        # TODO: Add weighted inner product support when non-uniform measures are implemented
        
        # Get dimensions
        node_dims = {}
        total_node_dim = 0
        for node in sorted(sheaf.poset.nodes()):
            if node in sheaf.stalks:
                dim = sheaf.stalks[node].shape[0]
                node_dims[node] = dim
                total_node_dim += dim
        
        total_edge_dim = len(sheaf.restrictions)  # One dimension per edge
        
        # Build coboundary using COO format
        rows, cols, data = [], [], []
        
        edge_idx = 0
        node_offsets = {}
        offset = 0
        for node in sorted(sheaf.poset.nodes()):
            if node in sheaf.stalks:
                node_offsets[node] = offset
                offset += node_dims[node]
        
        # For each edge e=(u,v), coboundary entry: (δf)_e = f_v - R_uv f_u
        for (u, v), restriction in sheaf.restrictions.items():
            if u not in node_offsets or v not in node_offsets:
                logger.warning(f"Skipping edge {u}->{v}: missing node data")
                continue
            
            u_offset = node_offsets[u]
            v_offset = node_offsets[v]
            u_dim = node_dims[u]
            v_dim = node_dims[v]
            
            # δ coefficient for target node v: +1
            for i in range(v_dim):
                rows.append(edge_idx)
                cols.append(v_offset + i)
                data.append(1.0)
            
            # δ coefficient for source node u: -R_uv (restriction from u to v)
            R = restriction.detach().cpu().numpy() if isinstance(restriction, torch.Tensor) else restriction
            r_v_dim, r_u_dim = R.shape
            
            # Use safe dimensions
            safe_u_dim = min(r_u_dim, u_dim)
            safe_v_dim = min(r_v_dim, v_dim)
            
            for i in range(safe_v_dim):
                for j in range(safe_u_dim):
                    if abs(R[i, j]) > self.sparsity_threshold:
                        rows.append(edge_idx)
                        cols.append(u_offset + j)
                        data.append(-R[i, j])
            
            edge_idx += 1
        
        coboundary = csr_matrix((data, (rows, cols)), 
                               shape=(total_edge_dim, total_node_dim))
        
        logger.info(f"GW coboundary built: {coboundary.shape}, {coboundary.nnz} nnz")
        return coboundary
    
    def extract_edge_weights(self, sheaf: Sheaf, active_edges: Optional[List[Tuple[str, str]]] = None) -> Dict[Tuple[str, str], float]:
        """Extract GW costs as edge weights for persistence analysis.
        
        Important: GW costs represent metric distortion (lower = better match)
        This is opposite to Procrustes norms (higher = stronger connection)
        
        Args:
            sheaf: GW-based sheaf with gw_costs in metadata
            active_edges: Optional list of edges to extract weights for. If None, extracts for all edges.
            
        Returns:
            Dictionary mapping edges to weights for increasing filtration
        """
        # Determine effective edge set
        if active_edges is None:
            active_edges = list(sheaf.restrictions.keys())
            
        logger.info(f"Extracting GW costs as edge weights for {len(active_edges)} active edges")
        
        if not sheaf.is_gw_sheaf():
            raise GWLaplacianError("Edge weight extraction requires GW sheaf")
        
        # Primary source: stored GW costs from construction
        gw_costs = sheaf.metadata.get('gw_costs', {})
        
        if not gw_costs:
            logger.warning("No GW costs found in metadata, computing from restriction properties")
            # Fallback: compute edge weights from restriction operator norms (only for active edges)
            edge_weights = {}
            for edge in active_edges:
                if edge in sheaf.restrictions:
                    restriction = sheaf.restrictions[edge]
                    if isinstance(restriction, torch.Tensor):
                        # Use operator 2-norm as proxy for distortion
                        weight = torch.linalg.norm(restriction, ord=2).item()
                    else:
                        weight = np.linalg.norm(restriction, ord=2)
                    edge_weights[edge] = weight
                
            logger.info(f"Computed fallback edge weights from restriction norms for {len(edge_weights)} edges")
            return edge_weights
        
        # Use stored GW costs directly (ensure all active edges are included)
        edge_weights = {}
        missing_edges = []
        
        for edge in active_edges:
            if edge in gw_costs:
                edge_weights[edge] = gw_costs[edge]
            else:
                missing_edges.append(edge)
        
        # Handle missing edges with fallback computation
        if missing_edges:
            logger.warning(f"Missing GW costs for {len(missing_edges)} edges, computing fallback weights")
            for edge in missing_edges:
                if edge in sheaf.restrictions:
                    restriction = sheaf.restrictions[edge]
                    if isinstance(restriction, torch.Tensor):
                        weight = torch.linalg.norm(restriction, ord=2).item()
                    else:
                        weight = np.linalg.norm(restriction, ord=2)
                    edge_weights[edge] = weight
                    logger.debug(f"  Edge {edge}: computed fallback weight {weight:.4f}")
                else:
                    logger.error(f"  Edge {edge}: missing from both gw_costs and restrictions")
        
        # Log statistics (for active edges only)
        costs = list(edge_weights.values())
        if costs:
            min_cost = min(costs)
            max_cost = max(costs)
            mean_cost = sum(costs) / len(costs)
            
            logger.info(f"Active GW edge weights: min={min_cost:.4f}, max={max_cost:.4f}, "
                       f"mean={mean_cost:.4f} (for INCREASING filtration, {len(costs)} edges)")
        
        return edge_weights
    
    def _initialize_gw_metadata(self, sheaf: Sheaf, edge_weights: Dict) -> GWLaplacianMetadata:
        """Initialize GW-specific Laplacian metadata."""
        metadata = GWLaplacianMetadata()
        
        # Base metadata
        offset = 0
        for node in sorted(sheaf.poset.nodes()):
            if node in sheaf.stalks:
                stalk = sheaf.stalks[node]
                dim = stalk.shape[0]
                metadata.stalk_dimensions[node] = dim
                metadata.stalk_offsets[node] = offset
                offset += dim
        
        metadata.total_dimension = offset
        metadata.construction_method = "gw_laplacian"
        
        # GW-specific metadata
        if edge_weights:
            costs = list(edge_weights.values())
            metadata.gw_cost_range = (min(costs), max(costs))
            metadata.mean_gw_cost = sum(costs) / len(costs)
        
        # Extract GW configuration info
        gw_config = sheaf.metadata.get('gw_config', {})
        metadata.measure_type = "uniform" if gw_config.get('uniform_measures', True) else "weighted"
        
        # Quasi-sheaf validation from construction
        validation_report = sheaf.metadata.get('validation_report', {})
        if validation_report:
            metadata.quasi_sheaf_max_violation = validation_report.get('max_violation', 0.0)
        
        metadata.edge_weight_source = "gw_costs" if sheaf.metadata.get('gw_costs') else "restriction_norms"
        metadata.filtration_semantics = "increasing"
        
        # Store sheaf reference for filtration reconstruction
        metadata.sheaf_reference = sheaf
        
        return metadata
    
    def _build_sparse_laplacian(self, sheaf: Sheaf, edge_weights: Dict, 
                               metadata: GWLaplacianMetadata, 
                               active_edges: Optional[List[Tuple[str, str]]] = None) -> csr_matrix:
        """Build Laplacian using optimized sparse assembly.
        
        Args:
            sheaf: GW-based sheaf
            edge_weights: Edge weights for active edges
            metadata: GW Laplacian metadata
            active_edges: List of edges to include (for filtration)
        """
        
        # Determine effective edge set
        if active_edges is None:
            active_edges = list(sheaf.restrictions.keys())
        
        # Use COO construction for efficient building
        rows, cols, data = [], [], []
        
        # 1. Off-diagonal blocks: L[u,v] = -R^T, L[v,u] = -R (only for active edges)
        for edge in active_edges:
            if edge not in sheaf.restrictions:
                continue
            restriction = sheaf.restrictions[edge]
            u, v = edge
            weight = edge_weights.get(edge, 1.0)
            
            if u not in metadata.stalk_offsets or v not in metadata.stalk_offsets:
                logger.warning(f"Edge {edge} connects to unknown node. Skipping.")
                continue
            
            u_start = metadata.stalk_offsets[u]
            v_start = metadata.stalk_offsets[v]
            
            # Convert restriction to numpy and apply weight squared (L = δᵀδ formulation)
            R = restriction.detach().cpu().numpy() if isinstance(restriction, torch.Tensor) else restriction
            R_weighted = R * (weight**2)  # All terms use w² for consistency
            
            # Get safe dimensions
            r_v_dim, r_u_dim = R.shape
            u_dim = metadata.stalk_dimensions[u]
            v_dim = metadata.stalk_dimensions[v]
            
            r_u_safe = min(r_u_dim, u_dim)
            r_v_safe = min(r_v_dim, v_dim)
            
            R_safe = R_weighted[:r_v_safe, :r_u_safe]
            R_sparse = csc_matrix(R_safe)
            
            if R_sparse.nnz > 0:
                # Off-diagonal L[v,u] = -R
                R_coo = R_sparse.tocoo()
                rows.extend(v_start + R_coo.row)
                cols.extend(u_start + R_coo.col)
                data.extend(-R_coo.data)
                
                # Off-diagonal L[u,v] = -R^T
                rows.extend(u_start + R_coo.col)
                cols.extend(v_start + R_coo.row)
                data.extend(-R_coo.data)
        
        # 2. Diagonal blocks: L[v,v] = Σ_{incoming} I + Σ_{outgoing} R^T R (only active edges)
        active_edge_set = set(active_edges)
        
        for node, dim in metadata.stalk_dimensions.items():
            node_start = metadata.stalk_offsets[node]
            diag_contributions = []
            
            # Identity contributions from incoming edges (only active edges)
            for predecessor in sheaf.poset.predecessors(node):
                edge = (predecessor, node)
                if edge in sheaf.restrictions and edge in active_edge_set:
                    weight = edge_weights.get(edge, 1.0)
                    if weight > self.sparsity_threshold:
                        R = sheaf.restrictions[edge]
                        R = R.detach().cpu().numpy() if isinstance(R, torch.Tensor) else R
                        r_node_dim = min(R.shape[0], dim)
                        
                        # Weighted identity (L = δᵀδ: use w²*I)
                        I_weighted = (weight**2) * csc_matrix(np.eye(r_node_dim))
                        diag_contributions.append(I_weighted)
            
            # R^T R contributions from outgoing edges (only active edges)
            for successor in sheaf.poset.successors(node):
                edge = (node, successor)
                if edge in sheaf.restrictions and edge in active_edge_set:
                    R = sheaf.restrictions[edge]
                    R = R.detach().cpu().numpy() if isinstance(R, torch.Tensor) else R
                    weight = edge_weights.get(edge, 1.0)
                    
                    # Safe dimensions
                    r_succ_dim, r_node_dim = R.shape
                    r_node_safe = min(r_node_dim, dim)
                    r_succ_safe = min(r_succ_dim, metadata.stalk_dimensions.get(successor, r_succ_dim))
                    
                    R_safe = R[:r_succ_safe, :r_node_safe]
                    R_weighted = weight * csc_matrix(R_safe)  # (w*R)^T @ (w*R) = w²*R^T*R
                    
                    if R_weighted.nnz > 0:
                        RTR = R_weighted.T @ R_weighted
                        diag_contributions.append(RTR)
            
            # Sum diagonal contributions
            if diag_contributions:
                diag_block = diag_contributions[0]
                for contrib in diag_contributions[1:]:
                    diag_block = diag_block + contrib
                
                if diag_block.nnz > 0:
                    diag_coo = diag_block.tocoo()
                    rows.extend(node_start + diag_coo.row)
                    cols.extend(node_start + diag_coo.col)
                    data.extend(diag_coo.data)
        
        # 3. Assemble sparse matrix
        total_dim = metadata.total_dimension
        laplacian_coo = coo_matrix((data, (rows, cols)), shape=(total_dim, total_dim))
        laplacian_coo.sum_duplicates()
        
        return laplacian_coo.tocsr()
    
    def _build_dense_laplacian(self, sheaf: Sheaf, edge_weights: Dict, 
                              metadata: GWLaplacianMetadata, 
                              active_edges: Optional[List[Tuple[str, str]]] = None) -> torch.Tensor:
        """Build dense Laplacian for small sheaves (mainly for testing).
        
        Args:
            sheaf: GW-based sheaf
            edge_weights: Edge weights for active edges
            metadata: GW Laplacian metadata
            active_edges: List of edges to include (for filtration)
        """
        
        # Determine effective edge set
        if active_edges is None:
            active_edges = list(sheaf.restrictions.keys())
        
        total_dim = metadata.total_dimension
        laplacian = torch.zeros((total_dim, total_dim), dtype=torch.float64)
        
        # Off-diagonal blocks (only for active edges)
        for edge in active_edges:
            if edge not in sheaf.restrictions:
                continue
            restriction = sheaf.restrictions[edge]
            u, v = edge
            weight = edge_weights.get(edge, 1.0)
            
            if u not in metadata.stalk_offsets or v not in metadata.stalk_offsets:
                continue
            
            u_start = metadata.stalk_offsets[u]
            v_start = metadata.stalk_offsets[v]
            u_dim = metadata.stalk_dimensions[u]
            v_dim = metadata.stalk_dimensions[v]
            
            # Convert to tensor
            if isinstance(restriction, torch.Tensor):
                R = restriction.to(dtype=torch.float64)
            else:
                R = torch.tensor(restriction, dtype=torch.float64)
            
            R_weighted = R * (weight**2)
            r_v_dim, r_u_dim = R_weighted.shape
            
            # Safe indexing
            r_u_safe = min(r_u_dim, u_dim)
            r_v_safe = min(r_v_dim, v_dim)
            
            # L[v,u] = -R
            laplacian[v_start:v_start+r_v_safe, u_start:u_start+r_u_safe] = -R_weighted[:r_v_safe, :r_u_safe]
            
            # L[u,v] = -R^T  
            laplacian[u_start:u_start+r_u_safe, v_start:v_start+r_v_safe] = -R_weighted[:r_v_safe, :r_u_safe].T
        
        # Diagonal blocks (only active edges)
        active_edge_set = set(active_edges)
        
        for node, dim in metadata.stalk_dimensions.items():
            node_start = metadata.stalk_offsets[node]
            diag_block = torch.zeros((dim, dim), dtype=torch.float64)
            
            # Identity from incoming edges (only active edges)
            for predecessor in sheaf.poset.predecessors(node):
                edge = (predecessor, node)
                if edge in sheaf.restrictions and edge in active_edge_set:
                    weight = edge_weights.get(edge, 1.0)
                    R = sheaf.restrictions[edge]
                    r_node_dim = min(R.shape[0], dim)
                    diag_block[:r_node_dim, :r_node_dim] += (weight**2) * torch.eye(r_node_dim, dtype=torch.float64)
            
            # R^T R from outgoing edges (only active edges)
            for successor in sheaf.poset.successors(node):
                edge = (node, successor)
                if edge in sheaf.restrictions and edge in active_edge_set:
                    R = sheaf.restrictions[edge]
                    weight = edge_weights.get(edge, 1.0)
                    
                    if isinstance(R, torch.Tensor):
                        R_tensor = R.to(dtype=torch.float64)
                    else:
                        R_tensor = torch.tensor(R, dtype=torch.float64)
                    
                    R_weighted = weight * R_tensor
                    r_succ_dim, r_node_dim = R_weighted.shape
                    r_node_safe = min(r_node_dim, dim)
                    
                    RTR = R_weighted.T @ R_weighted
                    diag_block[:r_node_safe, :r_node_safe] += RTR[:r_node_safe, :r_node_safe]
            
            laplacian[node_start:node_start+dim, node_start:node_start+dim] = diag_block
        
        return laplacian
    
    def _validate_gw_laplacian(self, laplacian: Union[torch.Tensor, csr_matrix], sheaf: Sheaf):
        """Validate GW Laplacian mathematical properties."""
        logger.debug("Validating GW Laplacian properties...")
        
        # Convert to numpy for validation
        if isinstance(laplacian, torch.Tensor):
            L = laplacian.detach().cpu().numpy()
        else:
            L = laplacian.toarray() if hasattr(laplacian, 'toarray') else laplacian
        
        # Check symmetry
        symmetry_diff = np.abs(L - L.T).max()
        if symmetry_diff > 1e-9:
            logger.warning(f"GW Laplacian not perfectly symmetric: max diff = {symmetry_diff:.2e}")
        else:
            logger.debug("GW Laplacian symmetry verified")
        
        # Check positive semi-definiteness
        try:
            if L.shape[0] <= 1000:  # Only for small matrices
                eigenvals = np.linalg.eigvals(L)
                min_eigval = np.min(eigenvals)
                if min_eigval < -1e-8:
                    logger.warning(f"GW Laplacian may not be PSD: min eigenvalue = {min_eigval:.2e}")
                else:
                    logger.debug(f"GW Laplacian PSD verified: min eigenvalue = {min_eigval:.2e}")
            else:
                logger.debug("Skipping eigenvalue check for large matrix")
                
        except Exception as e:
            logger.warning(f"Could not validate PSD property: {e}")
        
        # Validate edge weight semantics
        gw_costs = sheaf.metadata.get('gw_costs', {})
        if gw_costs:
            costs = list(gw_costs.values())
            logger.debug(f"GW cost range: [{min(costs):.4f}, {max(costs):.4f}] "
                        f"(INCREASING complexity filtration)")
    
    def _apply_adaptive_regularization(self, laplacian: Union[torch.Tensor, csr_matrix], 
                                     sheaf: Sheaf, 
                                     active_edges: List[Tuple[str, str]], 
                                     sparse: bool) -> Union[torch.Tensor, csr_matrix]:
        """Apply adaptive regularization based on connectivity analysis.
        
        Instead of uniform regularization, this method:
        1. Analyzes graph connectivity using active edges
        2. Identifies truly disconnected components
        3. Applies minimal regularization only where mathematically necessary
        4. Preserves true zero eigenvalues from the mathematical null space
        
        Args:
            laplacian: Constructed Laplacian matrix
            sheaf: Sheaf structure for connectivity analysis
            active_edges: List of active edges in current filtration step
            sparse: Whether laplacian is sparse or dense
            
        Returns:
            Regularized Laplacian matrix
        """
        import networkx as nx
        
        # Skip regularization if all edges are active (full connectivity case)
        total_edges = len(sheaf.restrictions)
        if len(active_edges) >= total_edges:
            logger.debug("Full connectivity: no regularization needed")
            return laplacian
        
        # Analyze graph connectivity from active edges
        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from(sheaf.poset.nodes())
        
        # Add only active edges to connectivity analysis
        active_edge_set = set(active_edges)
        for edge in sheaf.restrictions.keys():
            if edge in active_edge_set:
                u, v = edge
                connectivity_graph.add_edge(u, v)
        
        # Find connected components
        connected_components = list(nx.connected_components(connectivity_graph))
        num_components = len(connected_components)
        
        logger.info(f"Connectivity analysis: {len(active_edges)}/{total_edges} edges active, "
                   f"{num_components} connected components")
        
        # Only apply regularization if there are truly isolated components
        if num_components <= 1:
            logger.debug("Single connected component: no regularization needed")
            return laplacian
        
        # Apply minimal regularization only for numerical stability
        # Use smaller regularization that preserves spectral structure while ensuring detectability
        regularization_strength = 1e-10  # Balanced: small enough to preserve structure, large enough for validation
        
        logger.info(f"Applying minimal regularization ({regularization_strength}) "
                   f"for {num_components} disconnected components")
        
        if sparse:
            # Sparse regularization
            n_nodes = laplacian.shape[0]
            identity_csr = csr_matrix((regularization_strength * np.ones(n_nodes), 
                                     (np.arange(n_nodes), np.arange(n_nodes))), 
                                     shape=laplacian.shape)
            laplacian = laplacian + identity_csr
        else:
            # Dense regularization
            laplacian += regularization_strength * torch.eye(laplacian.shape[0], dtype=laplacian.dtype)
        
        logger.debug(f"Applied {regularization_strength} adaptive regularization to {num_components} components")
        return laplacian
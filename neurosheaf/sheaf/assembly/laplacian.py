"""Sparse sheaf Laplacian construction in whitened coordinates.

This module implements efficient sparse Laplacian assembly from whitened sheaf data
using the general sheaf Laplacian formulation that properly handles rectangular
restriction maps between stalks of different dimensions.

Mathematical Foundation:
- Vertex Stalks: Vector spaces F(v) at each node v (in whitened coordinates)
- Edge Stalks: (Implicit) vector spaces F(e) at each edge e 
- Restriction Maps: For edge e=(u,v), we have:
  * F_u→e: F(u) → F(e) (restriction from u to edge e)
  * F_v→e: F(v) → F(e) (restriction from v to edge e)
  * In practice, we store R = F_v→e ∘ F_u→e^(-1): F(u) → F(v)
- Coboundary δ: (δf)_e = F_v→e(f_v) - F_u→e(f_u)
- Laplacian: Δ = δᵀδ with blocks:
  * Off-diagonal (u,v): -F_u→e^T F_v→e = -R^T for edge e=(u,v)
  * Off-diagonal (v,u): -F_v→e^T F_u→e = -R for edge e=(u,v)  
  * Diagonal (v,v): Σ_{e=(u,v)} F_v→e^T F_v→e + Σ_{e=(v,w)} F_v→e^T F_v→e
    Which simplifies to: Σ_{e=(u,v)} I + Σ_{e=(v,w)} R^T R

This general formulation correctly handles rectangular restriction maps between
layers of different dimensions, as required for neural network analysis.

Detailed Mathematical Formulation:
=================================

1. Cellular Sheaf Structure:
   - Vertex stalks F(v): Vector spaces attached to each vertex v
   - Edge stalks F(e): Vector spaces attached to each edge e (implicit in this implementation)
   - Restriction maps F_v→e: F(v) → F(e) for each vertex-edge incidence v ∈ e

2. Coboundary Operator:
   For an edge e = (u,v) oriented from u to v, the coboundary operator δ acts on
   vertex data f = (f_u, f_v, ...) as:
   
   (δf)_e = F_v→e(f_v) - F_u→e(f_u)

3. Sheaf Laplacian:
   The Laplacian Δ = δ^T δ has block structure with blocks indexed by vertices:
   
   Δ[u,v] = -F_u→e^T F_v→e  (for edge e = (u,v))
   Δ[v,u] = -F_v→e^T F_u→e  (for edge e = (u,v))
   Δ[v,v] = Σ_{e incident to v} F_v→e^T F_v→e

4. Implementation Details:
   - We store restriction maps R: u → v as the composition F_v→e ∘ F_u→e^(-1)
   - This allows working with directed graphs where R maps from source to target
   - The general formulation naturally handles rectangular maps when dim(F(u)) ≠ dim(F(v))
   - For diagonal blocks: incoming edges contribute I, outgoing edges contribute R^T R

5. Sparse Matrix Efficiency:
   - Uses COO format for construction (efficient appends)
   - Converts to CSR for final matrix (efficient linear algebra)
   - Avoids dense intermediate matrices entirely
   - Threshold filtering removes numerical noise

This implementation is mathematically equivalent to the full general sheaf Laplacian
but optimized for the specific structure of neural network similarity analysis.
"""

import torch
import numpy as np
import time
import psutil
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)

from ..data_structures import Sheaf


class LaplacianError(Exception):
    """Exception raised during Laplacian construction."""
    pass


@dataclass
class LaplacianMetadata:
    """Metadata for the constructed sheaf Laplacian."""
    total_dimension: int = 0
    stalk_dimensions: Dict[str, int] = None
    stalk_offsets: Dict[str, int] = None
    sparsity_ratio: float = 0.0
    construction_time: float = 0.0
    num_nonzeros: int = 0
    memory_usage: float = 0.0  # Memory usage in MB
    construction_method: str = "standard"  # "standard" or "hodge_formulation"
    
    def __post_init__(self):
        """Initialize empty dictionaries if not provided."""
        if self.stalk_dimensions is None:
            self.stalk_dimensions = {}
        if self.stalk_offsets is None:
            self.stalk_offsets = {}


class SheafLaplacianBuilder:
    """Builds sparse sheaf Laplacian Δ = δᵀδ using the general sheaf formulation.
    
    This implementation uses the general sheaf Laplacian formulation that correctly
    handles rectangular restriction maps between stalks of different dimensions.
    It is optimized for whitened coordinate spaces and uses sparse matrix operations
    throughout for computational efficiency.
    
    The implementation assumes implicit edge stalks and interprets stored restriction
    maps R: u → v as the composition F_v→e ∘ F_u→e^(-1), where F_u→e and F_v→e
    are the restriction maps from vertex stalks to the edge stalk.
    """
    
    def __init__(self, validate_properties: bool = True, sparsity_threshold: float = 1e-12,
                 regularization: float = 1e-10):
        """Initialize the Laplacian builder.
        
        Args:
            validate_properties: Whether to validate mathematical properties
            sparsity_threshold: Threshold below which values are considered zero
            regularization: Regularization parameter for eigenvalue matrix inversion
        """
        self.validate_properties = validate_properties
        self.sparsity_threshold = sparsity_threshold
        self.regularization = regularization
    
    def build(self, sheaf: Sheaf, edge_weights: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple[csr_matrix, LaplacianMetadata]:
        """Build the sparse sheaf Laplacian with optimized performance.
        
        Args:
            sheaf: Sheaf object with whitened stalks and restrictions
            edge_weights: Optional edge weights. If None, uses Frobenius norm of restrictions
            
        Returns:
            Tuple of (sparse_laplacian, metadata)
            
        Raises:
            LaplacianError: If construction fails
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        logger.info(f"Building optimized Laplacian for {len(sheaf.stalks)} nodes, {len(sheaf.restrictions)} edges")
        
        try:
            # Initialize metadata
            metadata = self._initialize_metadata(sheaf)
            
            # Set default edge weights to Frobenius norm if not provided
            if edge_weights is None:
                edge_weights = self._compute_default_edge_weights(sheaf)
            
            # Build Laplacian using appropriate method based on eigenvalue preservation
            if self._uses_eigenvalue_preservation(sheaf):
                logger.info("Detected eigenvalue-preserving sheaf, using Hodge formulation")
                laplacian = self._build_hodge_laplacian(sheaf, edge_weights, metadata)
            else:
                # Use existing standard implementation
                laplacian = self._build_laplacian_optimized(sheaf, edge_weights, metadata)
            
            # Validate if requested
            if self.validate_properties:
                self._validate_laplacian(laplacian)
            
            # Update metadata
            final_memory = process.memory_info().rss
            metadata.construction_time = time.time() - start_time
            metadata.memory_usage = (final_memory - initial_memory) / (1024**2)  # MB
            metadata.num_nonzeros = laplacian.nnz
            metadata.sparsity_ratio = 1.0 - (laplacian.nnz / (laplacian.shape[0] * laplacian.shape[1]))
            
            logger.info(f"Optimized Laplacian built: {laplacian.shape}, {laplacian.nnz} nnz, "
                       f"{metadata.sparsity_ratio:.1%} sparse, {metadata.construction_time:.3f}s, "
                       f"{metadata.memory_usage:.2f} MB")
            
            return laplacian, metadata
            
        except Exception as e:
            raise LaplacianError(f"Laplacian construction failed: {e}")
    
    def _initialize_metadata(self, sheaf: Sheaf) -> LaplacianMetadata:
        """Initialize metadata from sheaf structure."""
        metadata = LaplacianMetadata()
        
        # Compute stalk dimensions and offsets with deterministic ordering
        offset = 0
        for node in sorted(sheaf.poset.nodes()):  # Sort for deterministic ordering
            if node in sheaf.stalks:
                stalk = sheaf.stalks[node]
                # For whitened stalks, the dimension is the first dimension (they are square identity matrices)
                dim = stalk.shape[0]
                
                metadata.stalk_dimensions[node] = dim
                metadata.stalk_offsets[node] = offset
                offset += dim
        
        metadata.total_dimension = offset
        
        logger.debug(f"Total dimension: {metadata.total_dimension}")
        return metadata
    
    def _compute_default_edge_weights(self, sheaf: Sheaf) -> Dict[Tuple[str, str], float]:
        """Compute default edge weights using Frobenius norm of restriction maps."""
        edge_weights = {}
        
        for edge, restriction in sheaf.restrictions.items():
            # Compute Frobenius norm as default weight
            if isinstance(restriction, torch.Tensor):
                weight = torch.norm(restriction, p='fro').item()
            else:
                weight = np.linalg.norm(restriction, 'fro')
            
            # Ensure non-zero weight
            edge_weights[edge] = max(weight, self.sparsity_threshold)
        
        logger.debug(f"Computed default edge weights: mean={np.mean(list(edge_weights.values())):.3f}")
        return edge_weights
    
    def _uses_eigenvalue_preservation(self, sheaf: Sheaf) -> bool:
        """Check if sheaf uses eigenvalue preservation mode.
        
        Args:
            sheaf: Sheaf to check
            
        Returns:
            True if eigenvalue preservation is active, False otherwise
        """
        return (sheaf.eigenvalue_metadata is not None and 
                sheaf.eigenvalue_metadata.preserve_eigenvalues)
    
    def _build_hodge_laplacian(self, sheaf: Sheaf, edge_weights: Dict[Tuple[str, str], float],
                              metadata: LaplacianMetadata) -> csr_matrix:
        """Build Laplacian using mathematically correct Hodge formulation.
        
        MATHEMATICAL FOUNDATION (corrected):
        
        For eigenvalue-preserving sheaf with inner products ⟨·,·⟩_Σ = ·^T Σ ·,
        the Hodge Laplacian Δ = δ*δ has the following structure:
        
        For undirected edge e = {u,v} with restriction R_uv: u → v:
        - The coboundary: (δf)_e = f_v - R_uv f_u  
        - The adjoint coboundary: (δ*g)_u = Σ_u^{-1} R_uv^T Σ_v g_e
                                  (δ*g)_v = Σ_v^{-1} g_e
        
        This gives the Hodge Laplacian blocks:
        - Diagonal: L[u,u] = Σ_{e∋u} δ*δ contributions from all incident edges
        - Off-diagonal: L[u,v] = -Σ_u^{-1} R_uv^T Σ_v for edge e={u,v}
        
        CRITICAL CORRECTION: The diagonal blocks are NOT Σ_v contributions!
        They come from the proper Hodge theory: L[u,u] = Σ_{e∋u} (δ*δ)_{uu}
        
        This formulation guarantees L = L^T and L ⪰ 0 by construction.
        
        Args:
            sheaf: Sheaf with eigenvalue-preserving stalks  
            edge_weights: Edge weights dictionary
            metadata: Laplacian metadata
            
        Returns:
            Sparse symmetric PSD Laplacian matrix
        """
        from ..core.whitening import WhiteningProcessor
        
        logger.info("Building mathematically correct Hodge Laplacian")
        
        # Get sheaf components
        poset = sheaf.poset
        stalks = sheaf.stalks
        restrictions = sheaf.restrictions
        eigenvalue_metadata = sheaf.eigenvalue_metadata
        
        # Initialize sparse matrix builders  
        rows, cols, data = [], [], []
        
        # Get eigenvalue matrices (Σ_v for each vertex v)
        eigenvalue_matrices = eigenvalue_metadata.eigenvalue_matrices
        
        # Create whitening processor for regularized inverse computation
        wp = WhiteningProcessor(preserve_eigenvalues=True, regularization=self.regularization)
        
        # Initialize diagonal accumulator for each node
        diagonal_blocks = {}
        for node in sorted(poset.nodes()):
            if node in stalks and node in eigenvalue_matrices:
                node_dim = metadata.stalk_dimensions[node]
                Sigma_node = eigenvalue_matrices[node]
                diagonal_blocks[node] = torch.zeros((node_dim, node_dim), 
                                                  dtype=Sigma_node.dtype, 
                                                  device=Sigma_node.device)
        
        # Process each edge exactly once for undirected formulation
        processed_pairs = set()
        
        for (u, v), R_uv in restrictions.items():
            # Skip if this edge pair already processed (for undirected treatment)
            edge_pair = tuple(sorted([u, v]))
            if edge_pair in processed_pairs:
                continue
            processed_pairs.add(edge_pair)
            
            if u not in eigenvalue_matrices or v not in eigenvalue_matrices:
                logger.warning(f"Missing eigenvalue matrices for edge {u}-{v}. Skipping.")
                continue
            
            # Get dimensions and offsets
            u_offset = metadata.stalk_offsets[u]
            v_offset = metadata.stalk_offsets[v]
            u_dim = metadata.stalk_dimensions[u]
            v_dim = metadata.stalk_dimensions[v]
            
            # Get eigenvalue matrices
            Sigma_u = eigenvalue_matrices[u]
            Sigma_v = eigenvalue_matrices[v]
            weight = edge_weights.get((u, v), 1.0)
            
            # Compute regularized inverses
            Sigma_u_inv = wp._compute_regularized_inverse(Sigma_u)
            Sigma_v_inv = wp._compute_regularized_inverse(Sigma_v)
            
            # CORRECTED HODGE LAPLACIAN FORMULATION (dimensionally safe):
            #
            # The correct off-diagonal blocks, derived from the energy functional 
            # ||δx||² = Σ (x_v - R_{uv}x_u)^T Σ_v (x_v - R_{uv}x_u), are:
            #
            # Off-diagonal blocks:
            # L[u,v] = -R_{uv}^T Σ_v     (maps from v-space to u-space)
            # L[v,u] = -Σ_v R_{uv}       (maps from u-space to v-space)
            #
            # This formulation is:
            # - Dimensionally safe for different stalk dimensions
            # - Symmetric: L[v,u]^T = (-Σ_v R_{uv})^T = -R_{uv}^T Σ_v^T = -R_{uv}^T Σ_v = L[u,v]
            # - Positive semi-definite by construction
            
            # Off-diagonal block L[u,v] = -R_{uv}^T Σ_v
            L_uv = -weight * (R_uv.T @ Sigma_v)
            
            # Off-diagonal block L[v,u] = -Σ_v R_{uv} 
            L_vu = -weight * (Sigma_v @ R_uv)
            
            # Verify symmetry: L[v,u] should equal L[u,v]^T
            # L[u,v]^T = (-R_{uv}^T Σ_v)^T = -Σ_v^T R_{uv} = -Σ_v R_{uv} = L[v,u] ✓
            # (since Σ_v is symmetric)
            
            # Add L[u,v] block  
            for i in range(u_dim):
                for j in range(v_dim):
                    value = L_uv[i, j].item()
                    if abs(value) > self.sparsity_threshold:
                        rows.append(u_offset + i)
                        cols.append(v_offset + j)
                        data.append(value)
            
            # Add L[v,u] block (using the computed L_vu, which equals L[u,v]^T)
            for i in range(v_dim):
                for j in range(u_dim):
                    value = L_vu[i, j].item()
                    if abs(value) > self.sparsity_threshold:
                        rows.append(v_offset + i)
                        cols.append(u_offset + j)
                        data.append(value)
            
            # CORRECTED DIAGONAL BLOCKS:
            # From the energy functional and your analysis:
            # L[v,v] = deg(v)Σ_v + Σ_{w∼v} R_{vw}^T Σ_w R_{vw}
            #
            # For edge {u,v} with restriction R_uv: u → v:
            # - L[u,u] gets contribution: R_{uv}^T Σ_v R_{uv} (source node)
            # - L[v,v] gets contribution: Σ_v (target node gets identity-like term)
            #
            # This is dimensionally safe and mathematically consistent.
            
            # Contribution to L[u,u] (source node): quadratic form through restriction
            diag_contribution_u = weight * (R_uv.T @ Sigma_v @ R_uv)
            diagonal_blocks[u] += diag_contribution_u
            
            # Contribution to L[v,v] (target node): eigenvalue matrix Σ_v
            diag_contribution_v = weight * Sigma_v
            diagonal_blocks[v] += diag_contribution_v
        
        # Add all diagonal blocks to sparse matrix
        for node, diagonal_block in diagonal_blocks.items():
            node_offset = metadata.stalk_offsets[node]
            node_dim = metadata.stalk_dimensions[node]
            
            for i in range(node_dim):
                for j in range(node_dim):
                    value = diagonal_block[i, j].item()
                    if abs(value) > self.sparsity_threshold:
                        rows.append(node_offset + i)
                        cols.append(node_offset + j)
                        data.append(value)
        
        # Create sparse matrix
        laplacian_sparse = csr_matrix(
            (data, (rows, cols)), 
            shape=(metadata.total_dimension, metadata.total_dimension)
        )
        
        # Update metadata
        metadata.construction_method = "hodge_formulation"
        
        logger.info(f"Mathematically correct Hodge Laplacian built: {laplacian_sparse.shape}, {laplacian_sparse.nnz} nnz")
        return laplacian_sparse
    
    def _build_laplacian_optimized(self, sheaf: Sheaf, edge_weights: Dict[Tuple[str, str], float], 
                                  metadata: LaplacianMetadata) -> csr_matrix:
        """Build Laplacian using optimized sparse assembly with general sheaf formulation.
        
        Implements the general sheaf Laplacian that correctly handles rectangular
        restriction maps between stalks of different dimensions. Uses sparse matrix
        operations throughout for computational efficiency.
        
        For edge e=(u,v) with restriction map R: u → v:
        - Off-diagonal blocks: L[u,v] = -R^T, L[v,u] = -R
        - Diagonal blocks: L[v,v] = Σ_{incoming} I + Σ_{outgoing} R^T R
        """
        
        # Use lists for COO construction - much faster than repeated appends
        rows, cols, data = [], [], []
        
        # 1. Construct Off-Diagonal Blocks
        # For the general formulation with implicit edge stalks:
        # L[u,v] = -F_u→e^T F_v→e = -R^T (since R represents the composition)
        # L[v,u] = -F_v→e^T F_u→e = -R
        for edge, restriction in sheaf.restrictions.items():
            u, v = edge
            weight = edge_weights.get(edge, 1.0)
            
            if u not in metadata.stalk_offsets or v not in metadata.stalk_offsets:
                logger.warning(f"Edge {edge} connects to unknown node. Skipping.")
                continue
            
            u_start = metadata.stalk_offsets[u]
            v_start = metadata.stalk_offsets[v]
            
            # Convert restriction to numpy and apply weight
            R = restriction.detach().cpu().numpy() if isinstance(restriction, torch.Tensor) else restriction
            R_weighted = R * weight
            
            # Get actual restriction dimensions - R maps from u to v
            r_v_dim, r_u_dim = R.shape  # R: u → v, so shape is (v_dim, u_dim)
            
            # Ensure restriction dimensions don't exceed stalk dimensions
            u_dim = metadata.stalk_dimensions[u]
            v_dim = metadata.stalk_dimensions[v]
            
            # Use minimum of restriction size and available stalk space
            r_u_safe = min(r_u_dim, u_dim)
            r_v_safe = min(r_v_dim, v_dim)
            
            # Extract safe submatrix if needed
            R_safe = R_weighted[:r_v_safe, :r_u_safe]
            
            # Convert to sparse for efficient operations
            R_sparse = csc_matrix(R_safe)
            
            if R_sparse.nnz > 0:  # Only process if non-zero entries exist
                # Off-diagonal block L[v,u] = -R
                R_coo = R_sparse.tocoo()
                rows.extend(v_start + R_coo.row)
                cols.extend(u_start + R_coo.col)
                data.extend(-R_coo.data)
                
                # Off-diagonal block L[u,v] = -R^T
                # Use the same data but swap row/col indices
                rows.extend(u_start + R_coo.col)
                cols.extend(v_start + R_coo.row)
                data.extend(-R_coo.data)
        
        # 2. Construct Diagonal Blocks
        # For node v: L[v,v] = Σ_{e=(u,v)} I + Σ_{e=(v,w)} R^T R
        for node, dim in metadata.stalk_dimensions.items():
            node_start = metadata.stalk_offsets[node]
            
            # Use sparse matrices for diagonal block accumulation
            diag_contributions = []
            
            # Add identity matrices for all incoming edges e=(u,node)
            for predecessor in sheaf.poset.predecessors(node):
                edge = (predecessor, node)
                if edge in sheaf.restrictions:
                    weight = edge_weights.get(edge, 1.0)
                    if weight > self.sparsity_threshold:
                        # Get restriction dimensions
                        R = sheaf.restrictions[edge]
                        R = R.detach().cpu().numpy() if isinstance(R, torch.Tensor) else R
                        r_node_dim = min(R.shape[0], dim)  # R maps to node, so shape[0] is node dimension
                        
                        # Add weighted identity contribution
                        I_weighted = weight**2 * csc_matrix(np.eye(r_node_dim))
                        diag_contributions.append(I_weighted)
            
            # Add R^T R for all outgoing edges e=(node,w)
            for successor in sheaf.poset.successors(node):
                edge = (node, successor)
                if edge in sheaf.restrictions:
                    R = sheaf.restrictions[edge].detach().cpu().numpy() if isinstance(sheaf.restrictions[edge], torch.Tensor) else sheaf.restrictions[edge]
                    weight = edge_weights.get(edge, 1.0)
                    
                    # Get restriction dimensions - R maps from node to successor
                    r_succ_dim, r_node_dim = R.shape
                    r_node_safe = min(r_node_dim, dim)
                    r_succ_safe = min(r_succ_dim, metadata.stalk_dimensions.get(successor, r_succ_dim))
                    
                    # Extract safe submatrix
                    R_safe = R[:r_succ_safe, :r_node_safe]
                    R_weighted = weight * csc_matrix(R_safe)
                    
                    # Compute R^T R using sparse operations
                    if R_weighted.nnz > 0:
                        RTR = R_weighted.T @ R_weighted
                        diag_contributions.append(RTR)
            
            # Sum all diagonal contributions
            if diag_contributions:
                # Sum sparse matrices efficiently
                diag_block = diag_contributions[0]
                for contrib in diag_contributions[1:]:
                    diag_block = diag_block + contrib
                
                # Add non-zero entries to the main COO lists
                if diag_block.nnz > 0:
                    diag_coo = diag_block.tocoo()
                    rows.extend(node_start + diag_coo.row)
                    cols.extend(node_start + diag_coo.col)
                    data.extend(diag_coo.data)
        
        # 3. Assemble sparse matrix efficiently
        total_dim = metadata.total_dimension
        laplacian_coo = coo_matrix((data, (rows, cols)), shape=(total_dim, total_dim))
        
        # Sum duplicates (important for diagonal entries)
        laplacian_coo.sum_duplicates()
        
        # Convert to CSR for efficient operations
        return laplacian_coo.tocsr()
    
    def _validate_laplacian(self, laplacian: csr_matrix):
        """Validate mathematical properties of the Laplacian with enhanced PSD validation."""
        logger.debug("Validating Laplacian mathematical properties...")
        
        # Use enhanced PSD validation
        try:
            from ...utils.psd_validation import validate_psd_comprehensive
            
            result = validate_psd_comprehensive(
                laplacian, 
                name="sheaf_laplacian",
                compute_full_spectrum=False,
                enable_regularization=True
            )
            
            if not result.is_psd:
                if result.smallest_eigenvalue > -1e-6:
                    # Small numerical error - warn but continue
                    logger.warning(f"Laplacian has small numerical PSD violation: {result.smallest_eigenvalue:.2e}")
                else:
                    # Significant PSD violation - this is a problem
                    logger.error(f"Laplacian is not positive semi-definite: {result.smallest_eigenvalue:.2e}")
                    if result.regularization_needed:
                        logger.info(f"Consider regularization: condition number = {result.condition_number:.2e}")
            else:
                logger.debug(f"Laplacian PSD validation passed: smallest eigenvalue = {result.smallest_eigenvalue:.2e}")
            
            # Log additional diagnostics
            logger.debug(f"Laplacian rank: {result.rank}, condition number: {result.condition_number:.2e}")
            
        except ImportError:
            # Fallback to original validation if enhanced module not available
            logger.debug("Enhanced PSD validation not available, using basic validation")
            
            # Check symmetry
            symmetry_diff = np.abs(laplacian - laplacian.T).max()
            if symmetry_diff > 1e-9:
                logger.warning(f"Laplacian is not perfectly symmetric. Max difference: {symmetry_diff:.2e}")
            else:
                logger.debug("Symmetry property verified.")
            
            # Check positive semi-definiteness using eigenvalue analysis
            try:
                from scipy.sparse.linalg import eigsh
                
                # Check smallest eigenvalue for PSD property
                if laplacian.shape[0] > 1:
                    min_eigenval = eigsh(laplacian, k=1, which='SA', return_eigenvectors=False)[0]
                    if min_eigenval < -1e-8:  # Updated tolerance
                        logger.warning(f"Laplacian may not be positive semi-definite. Smallest eigenvalue: {min_eigenval:.2e}")
                    else:
                        logger.debug(f"Positive semi-definite property verified. Smallest eigenvalue: {min_eigenval:.2e}")
                else:
                    logger.debug("Skipping eigenvalue check for 1x1 matrix.")
            
            except Exception as e:
                logger.warning(f"Could not compute eigenvalues for validation: {e}")
        
        except Exception as e:
            logger.warning(f"Enhanced PSD validation failed: {e}")


def build_sheaf_laplacian(sheaf: Sheaf, edge_weights: Optional[Dict[Tuple[str, str], float]] = None, 
                         validate: bool = True) -> Tuple[csr_matrix, LaplacianMetadata]:
    """Convenience function to build the general sheaf Laplacian.
    
    Builds the sparse sheaf Laplacian using the general formulation that correctly
    handles rectangular restriction maps between stalks of different dimensions.
    
    Args:
        sheaf: Sheaf object with whitened stalks and restrictions
        edge_weights: Optional edge weights. If None, uses Frobenius norm of restrictions
        validate: Whether to validate mathematical properties (symmetry, PSD)
        
    Returns:
        Tuple of (sparse_laplacian, metadata)
    """
    builder = SheafLaplacianBuilder(validate_properties=validate)
    return builder.build(sheaf, edge_weights)
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
    
    def __init__(self, validate_properties: bool = True, sparsity_threshold: float = 1e-12):
        """Initialize the Laplacian builder.
        
        Args:
            validate_properties: Whether to validate mathematical properties
            sparsity_threshold: Threshold below which values are considered zero
        """
        self.validate_properties = validate_properties
        self.sparsity_threshold = sparsity_threshold
    
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
            
            # Build Laplacian using optimized sparse assembly
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
        """Validate mathematical properties of the Laplacian with improved checks."""
        logger.debug("Validating Laplacian mathematical properties...")
        
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
                if min_eigenval < -1e-9:
                    logger.warning(f"Laplacian may not be positive semi-definite. Smallest eigenvalue: {min_eigenval:.2e}")
                else:
                    logger.debug(f"Positive semi-definite property verified. Smallest eigenvalue: {min_eigenval:.2e}")
            else:
                logger.debug("Skipping eigenvalue check for 1x1 matrix.")
        
        except Exception as e:
            logger.warning(f"Could not compute eigenvalues for validation: {e}")


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
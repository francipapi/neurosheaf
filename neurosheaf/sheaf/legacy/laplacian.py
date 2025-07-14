"""Sparse sheaf Laplacian construction for neural network analysis.

This module implements an efficient and mathematically sound sparse Laplacian 
assembly from whitened sheaf data, optimized for performance. The Laplacian,
defined as Δ = δᵀδ (where δ is the coboundary operator), encodes the 
topological structure of the underlying graph and the relationships between data 
stalks.

Mathematical Foundation:
- Stalks: Vector spaces associated with each node (ℝ^{d_v}).
- Restriction Maps: Linear maps R_e: Stalk(u) → Stalk(v) for each edge e=(u,v).
- Coboundary Operator (δ): For a 0-cochain f = {f_v ∈ Stalk(v)}, its action on an edge e=(u,v) is (δf)_e = f_v - R_e f_u.
- Sheaf Laplacian (Δ = δᵀδ): A block matrix acting on 0-cochains. Its blocks are:
  - Diagonal Block (v,v): Δ_vv = Σ_{e=(v,w)} (R_eᵀ R_e) + Σ_{e=(u,v)} I
  - Off-Diagonal Block (v,w) for edge e=(v,w): Δ_vw = -R_eᵀ
  - Off-Diagonal Block (w,v) for edge e=(v,w): Δ_wv = -R_e
"""

import torch
import numpy as np
import time
import psutil
from scipy.sparse import csr_matrix, coo_matrix
from typing import Dict, List, Tuple, Optional

# Assuming the existence of these utility modules from the original structure
# from ..utils.logging import setup_logger
# from ..utils.exceptions import ComputationError
# from .construction import Sheaf
import logging
logger = logging.getLogger(__name__)

# --- Mock objects for stand-alone execution ---
from dataclasses import dataclass
import networkx as nx

class ComputationError(Exception):
    pass

@dataclass
class Sheaf:
    stalks: Dict[str, torch.Tensor]
    restrictions: Dict[Tuple[str, str], torch.Tensor]
    poset: nx.DiGraph
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LaplacianMetadata:
    """Metadata for the constructed sheaf Laplacian."""
    total_dimension: int = 0
    stalk_dimensions: Dict[str, int] = None
    stalk_offsets: Dict[str, int] = None
    sparsity_ratio: float = 0.0
    condition_number: Optional[float] = None
    construction_time: float = 0.0
    memory_usage: float = 0.0
    
    def __post_init__(self):
        """Initialize empty dictionaries if not provided."""
        if self.stalk_dimensions is None:
            self.stalk_dimensions = {}
        if self.stalk_offsets is None:
            self.stalk_offsets = {}

class SheafLaplacianBuilder:
    """
    Builds a sparse sheaf Laplacian Δ = δᵀδ from sheaf data.

    This implementation has been optimized for correctness, performance, and clarity.
    It uses a single, efficient COO-based construction method that correctly 
    implements the mathematical formula for the sheaf Laplacian.
    """
    
    def __init__(self, enable_gpu: bool = True, validate_properties: bool = True):
        """
        Initialize the Laplacian builder.
        
        Args:
            enable_gpu: If True, enables GPU support for tensor conversion.
            validate_properties: If True, validates mathematical properties (symmetry, PSD).
        """
        self.enable_gpu = enable_gpu
        self.validate_properties = validate_properties
        
        if self.enable_gpu and not torch.cuda.is_available():
            logger.warning("GPU support enabled, but no CUDA device found. Falling back to CPU.")
            self.enable_gpu = False
    
    def build(self, sheaf: Sheaf, edge_weights: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple[csr_matrix, LaplacianMetadata]:
        """
        Builds the sparse sheaf Laplacian using an efficient pre-allocation strategy.
        
        Args:
            sheaf: A Sheaf object containing stalks and restriction maps.
            edge_weights: Optional weights for edges. Defaults to 1.0.
            
        Returns:
            A tuple containing the sparse Laplacian (csr_matrix) and its metadata.
            
        Raises:
            ComputationError: If the construction fails.
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        logger.info(f"Starting sheaf Laplacian construction for a graph with {len(sheaf.stalks)} nodes and {len(sheaf.restrictions)} edges.")
        
        try:
            metadata = self._initialize_metadata(sheaf)


            #Problematic should be initializes to frobenius norm
            if edge_weights is None:
                edge_weights = {edge: 1.0 for edge in sheaf.restrictions.keys()} 
            
            laplacian = self._build_laplacian_optimized(sheaf, edge_weights, metadata)
            
            if self.validate_properties:
                self._validate_laplacian_properties(laplacian, metadata)
            
            final_memory = process.memory_info().rss
            
            metadata.construction_time = time.time() - start_time
            metadata.memory_usage = (final_memory - initial_memory) / (1024**2) # MB
            metadata.sparsity_ratio = 1.0 - (laplacian.nnz / (laplacian.shape[0] * laplacian.shape[1]))
            
            logger.info(
                f"Laplacian construction successful. Shape: {laplacian.shape}, "
                f"NNZ: {laplacian.nnz} ({metadata.sparsity_ratio:.2%} sparse), "
                f"Time: {metadata.construction_time:.3f}s, "
                f"Memory: {metadata.memory_usage:.2f} MB"
            )
            
            return laplacian, metadata
            
        except Exception as e:
            logger.error(f"Laplacian construction failed: {e}", exc_info=True)
            raise ComputationError(f"Laplacian construction failed: {e}", operation="build_laplacian")

    def _initialize_metadata(self, sheaf: Sheaf) -> LaplacianMetadata:
        """Computes dimensions and offsets from the sheaf structure."""
        metadata = LaplacianMetadata()
        
        # Stalk dimensions from whitened data
        for node, stalk_data in sheaf.stalks.items():
            metadata.stalk_dimensions[node] = stalk_data.shape[0]
        
        # Stalk offsets for global matrix indexing
        offset = 0
        for node in sorted(sheaf.poset.nodes()): # Sort for deterministic ordering
            if node in metadata.stalk_dimensions:
                metadata.stalk_offsets[node] = offset
                offset += metadata.stalk_dimensions[node]
        
        metadata.total_dimension = offset
        logger.debug(f"Total Laplacian dimension computed: {metadata.total_dimension}")
        return metadata

    def _build_laplacian_optimized(
        self,
        sheaf: Sheaf,
        edge_weights: Dict[Tuple[str, str], float],
        metadata: LaplacianMetadata
    ) -> csr_matrix:
        """
        Builds Δ = δᵀδ using a single-pass COO assembly, which is both fast and memory-efficient.
        
        This method correctly constructs the matrix by separating the assembly of off-diagonal
        and diagonal blocks, adhering to the precise mathematical formula.
        """
        rows, cols, data = [], [], []

        # 1. Construct Off-Diagonal Blocks: Δ_wv = -R and Δ_vw = -R^T
        for edge, restriction in sheaf.restrictions.items():
            u, v = edge
            weight = edge_weights.get(edge, 1.0)

            if u not in metadata.stalk_offsets or v not in metadata.stalk_offsets:
                logger.warning(f"Edge {edge} connects to an unknown node. Skipping.")
                continue

            u_start, v_start = metadata.stalk_offsets[u], metadata.stalk_offsets[v]
            R = restriction.detach().cpu().numpy() * weight

            # Find non-zero entries in the restriction map R
            nz_rows, nz_cols = np.where(np.abs(R) > 1e-12)
            
            # Contribution to Δ_vu = -R (or Δ_wv if e=(v,w) in other conventions)
            rows.extend(v_start + nz_rows)
            cols.extend(u_start + nz_cols)
            data.extend(-R[nz_rows, nz_cols])

            # Contribution to Δ_uv = -R^T
            rows.extend(u_start + nz_cols)
            cols.extend(v_start + nz_rows)
            data.extend(-R[nz_rows, nz_cols])

        # 2. Construct Diagonal Blocks: Δ_vv = Σ(R_eᵀ R_e) + Σ(I)
        for node, dim in metadata.stalk_dimensions.items():
            node_start = metadata.stalk_offsets[node]
            
            # Initialize the dense diagonal block for this node
            diag_block = np.zeros((dim, dim))

            # Add Σ(R_eᵀ R_e) for all outgoing edges e=(node, w)
            for successor in sheaf.poset.successors(node):
                edge = (node, successor)
                if edge in sheaf.restrictions:
                    R = sheaf.restrictions[edge].detach().cpu().numpy()
                    weight = edge_weights.get(edge, 1.0)
                    if R.shape[1] == dim: # Check dimension consistency
                        diag_block += (weight**2) * (R.T @ R)
            
            # Add Σ(I) for all incoming edges e=(u, node) with non-zero weight
            for predecessor in sheaf.poset.predecessors(node):
                edge = (predecessor, node)
                if edge in sheaf.restrictions and edge_weights.get(edge, 1.0) > 1e-12:
                    diag_block += np.eye(dim)

            # Add the non-zero entries of the computed diagonal block
            nz_rows, nz_cols = np.where(np.abs(diag_block) > 1e-12)
            rows.extend(node_start + nz_rows)
            cols.extend(node_start + nz_cols)
            data.extend(diag_block[nz_rows, nz_cols])

        # 3. Assemble the sparse matrix
        n = metadata.total_dimension
        laplacian_coo = coo_matrix((data, (rows, cols)), shape=(n, n))
        
        # Sum duplicates and convert to CSR for efficient arithmetic
        return laplacian_coo.tocsr()

    def _validate_laplacian_properties(self, laplacian: csr_matrix, metadata: LaplacianMetadata):
        """Validates the mathematical properties of the Laplacian."""
        logger.debug("Validating Laplacian properties...")
        # Check for symmetry
        symmetry_diff = np.abs(laplacian - laplacian.T).max()
        if symmetry_diff > 1e-9:
            logger.warning(f"Laplacian is not perfectly symmetric. Max difference: {symmetry_diff:.2e}")
        else:
            logger.debug("Symmetry property verified.")

        # Check for positive semi-definiteness by examining the smallest eigenvalue
        try:
            from scipy.sparse.linalg import eigsh
            # We only need the smallest eigenvalue to check for PSD
            # k=1 asks for one eigenvalue, 'SA' asks for the Smallest Algebraic value.
            # Make sure matrix is large enough for eigsh
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
    
    def to_torch_sparse(self, laplacian: csr_matrix) -> torch.sparse.FloatTensor:
        """Converts a SciPy CSR matrix to a PyTorch sparse COO tensor."""
        laplacian_coo = laplacian.tocoo()
        indices = torch.from_numpy(np.vstack((laplacian_coo.row, laplacian_coo.col))).long()
        values = torch.from_numpy(laplacian_coo.data).float()
        shape = torch.Size(laplacian_coo.shape)
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
        
        if self.enable_gpu:
            sparse_tensor = sparse_tensor.to('cuda')
        
        return sparse_tensor.coalesce()
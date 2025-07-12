# neurosheaf/spectral/static_laplacian_masking.py
"""Static Laplacian with edge masking for persistent spectral analysis.

This module implements the edge masking approach for persistence computation,
where a full Laplacian is built once and Boolean masks are applied to create
filtered versions for different threshold values.

Key Features:
- Static Laplacian construction using existing SheafLaplacianBuilder
- Efficient Boolean edge masking for filtration
- Sparse eigenvalue computation using LOBPCG
- Integration with SubspaceTracker for eigenspace evolution
- Support for different filtration types (threshold, CKA-based)
"""

import torch
import torch.sparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import lobpcg
import time
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError
from ..sheaf.construction import Sheaf
from ..sheaf.laplacian import SheafLaplacianBuilder

logger = setup_logger(__name__)


class StaticLaplacianWithMasking:
    """Compute persistence using static Laplacian with edge masking.
    
    This class implements the static masking approach where:
    1. Full Laplacian is built once using all edges
    2. Edge information is extracted and cached
    3. Boolean masks are applied for different filtration values
    4. Eigenvalues/eigenvectors are computed for each masked Laplacian
    5. SubspaceTracker handles eigenspace evolution
    
    Attributes:
        laplacian_builder: Builder for constructing the static Laplacian
        eigenvalue_method: Method for eigenvalue computation ('lobpcg', 'dense')
        max_eigenvalues: Maximum number of eigenvalues to compute
        enable_gpu: Whether to enable GPU acceleration
    """
    
    def __init__(self,
                 laplacian_builder: Optional[SheafLaplacianBuilder] = None,
                 eigenvalue_method: str = 'lobpcg',
                 max_eigenvalues: int = 100,
                 enable_gpu: bool = True):
        """Initialize StaticLaplacianWithMasking.
        
        Args:
            laplacian_builder: Builder for Laplacian construction (default: auto-create)
            eigenvalue_method: Eigenvalue computation method ('lobpcg', 'dense')
            max_eigenvalues: Maximum number of eigenvalues to compute
            enable_gpu: Whether to enable GPU operations
        """
        self.laplacian_builder = laplacian_builder or SheafLaplacianBuilder(
            enable_gpu=enable_gpu,
            validate_properties=True
        )
        self.eigenvalue_method = eigenvalue_method
        self.max_eigenvalues = max_eigenvalues
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        
        # Cache for static Laplacian and edge information
        self._cached_laplacian = None
        self._cached_edge_info = None
        self._cached_metadata = None
        self._edge_contributions = None  # Cache for pre-computed edge contributions
        
        logger.info(f"StaticLaplacianWithMasking initialized: {eigenvalue_method} method, "
                   f"max_eigenvals={max_eigenvalues}, GPU={self.enable_gpu}")
        
    def compute_persistence(self,
                           sheaf: Sheaf,
                           filtration_params: List[float],
                           edge_threshold_func: Callable[[float, float], bool]) -> Dict:
        """Compute persistence using edge masking approach.
        
        Args:
            sheaf: Sheaf object with full edge set
            filtration_params: List of filtration parameter values
            edge_threshold_func: Function that returns True if edge should be kept
                                Signature: (edge_weight, filtration_param) -> bool
            
        Returns:
            Dictionary with persistence information:
            - eigenvalue_sequences: List of eigenvalue tensors
            - eigenvector_sequences: List of eigenvector tensors
            - tracking_info: Subspace tracking results
            - filtration_params: Parameter values used
            - edge_info: Edge information for analysis
        """
        logger.info(f"Computing persistence with {len(filtration_params)} filtration steps")
        start_time = time.time()
        
        try:
            # Build full Laplacian once (with caching)
            full_laplacian, metadata = self._get_cached_laplacian(sheaf)
            
            # Extract edge information (with caching)
            edge_info = self._get_cached_edge_info(sheaf, full_laplacian, metadata)
            
            # Compute eigenvalues/eigenvectors for each filtration step
            eigenvalue_sequences = []
            eigenvector_sequences = []
            
            for i, param in enumerate(filtration_params):
                # Create edge mask for this filtration parameter
                edge_mask = self._create_edge_mask(edge_info, param, edge_threshold_func)
                
                # Apply mask to Laplacian
                masked_laplacian = self._apply_edge_mask(
                    full_laplacian, edge_mask, edge_info, metadata
                )
                
                # Compute eigenvalues/eigenvectors
                eigenvals, eigenvecs = self._compute_eigenvalues(masked_laplacian)
                
                eigenvalue_sequences.append(eigenvals)
                eigenvector_sequences.append(eigenvecs)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i + 1}/{len(filtration_params)} filtration steps")
            
            # Track eigenspaces using Week 8's SubspaceTracker
            from .tracker import SubspaceTracker
            tracker = SubspaceTracker()
            
            tracking_info = tracker.track_eigenspaces(
                eigenvalue_sequences,
                eigenvector_sequences,
                filtration_params
            )
            
            computation_time = time.time() - start_time
            logger.info(f"Persistence computation completed in {computation_time:.2f}s")
            
            return {
                'eigenvalue_sequences': eigenvalue_sequences,
                'eigenvector_sequences': eigenvector_sequences,
                'tracking_info': tracking_info,
                'filtration_params': filtration_params,
                'edge_info': edge_info,
                'computation_time': computation_time
            }
            
        except Exception as e:
            raise ComputationError(f"Persistence computation failed: {e}", 
                                 operation="compute_persistence")
    
    def _get_cached_laplacian(self, sheaf: Sheaf) -> Tuple[csr_matrix, object]:
        """Get cached Laplacian or build it if not cached."""
        if self._cached_laplacian is None:
            logger.debug("Building static Laplacian (not cached)")
            self._cached_laplacian, self._cached_metadata = self.laplacian_builder.build(sheaf)
            logger.info(f"Static Laplacian built: {self._cached_laplacian.shape}, "
                       f"{self._cached_laplacian.nnz:,} non-zeros")
            
            # Pre-compute edge contributions for efficient filtering
            self._precompute_edge_contributions(sheaf, self._cached_metadata)
        else:
            logger.debug("Using cached static Laplacian")
        
        return self._cached_laplacian, self._cached_metadata
    
    def _get_cached_edge_info(self, sheaf: Sheaf, laplacian: csr_matrix, metadata: object) -> Dict:
        """Get cached edge information or extract it if not cached."""
        if self._cached_edge_info is None:
            logger.debug("Extracting edge information (not cached)")
            self._cached_edge_info = self._extract_edge_info(sheaf, laplacian, metadata)
            logger.debug(f"Edge information extracted for {len(self._cached_edge_info)} edges")
        else:
            logger.debug("Using cached edge information")
        
        return self._cached_edge_info
    
    def _extract_edge_info(self, sheaf: Sheaf, laplacian: csr_matrix, metadata: object) -> Dict:
        """Extract edge information from sheaf and Laplacian.
        
        Args:
            sheaf: Sheaf object with restrictions
            laplacian: Static Laplacian matrix
            metadata: Laplacian construction metadata
            
        Returns:
            Dictionary mapping edges to their information:
            - weight: Edge weight (Frobenius norm of restriction)
            - restriction: Original restriction matrix
            - source/target: Edge endpoints
            - positions: Matrix positions (if available in metadata)
        """
        edge_info = {}
        
        for edge in sheaf.restrictions.keys():
            source, target = edge
            restriction = sheaf.restrictions[edge]
            
            # Compute edge weight using Frobenius norm
            weight = torch.norm(restriction, 'fro').item()
            
            # Store edge information
            edge_info[edge] = {
                'restriction': restriction,
                'restriction_matrix': restriction.detach().cpu().numpy(),  # For fallback reconstruction
                'weight': weight,
                'source': source,
                'target': target
            }
            
            # Add matrix positions if available from metadata
            if hasattr(metadata, 'edge_positions') and edge in metadata.edge_positions:
                edge_info[edge]['positions'] = metadata.edge_positions[edge]
            else:
                logger.debug(f"No position information available for edge {edge}")
        
        # Log weight statistics
        weights = [info['weight'] for info in edge_info.values()]
        if weights:
            logger.info(f"Edge weights: min={min(weights):.4f}, max={max(weights):.4f}, "
                       f"mean={np.mean(weights):.4f}")
        
        return edge_info
    
    def _create_edge_mask(self,
                         edge_info: Dict,
                         filtration_param: float,
                         edge_threshold_func: Callable[[float, float], bool]) -> Dict[Tuple, bool]:
        """Create boolean mask for edges based on filtration parameter.
        
        Args:
            edge_info: Edge information dictionary
            filtration_param: Current filtration parameter value
            edge_threshold_func: Function to determine if edge should be kept
            
        Returns:
            Dictionary mapping edges to boolean (True = keep, False = mask out)
        """
        edge_mask = {}
        
        for edge, info in edge_info.items():
            # Apply threshold function to determine if edge should be kept
            keep_edge = edge_threshold_func(info['weight'], filtration_param)
            edge_mask[edge] = keep_edge
        
        active_edges = sum(edge_mask.values())
        logger.debug(f"Edge mask for param {filtration_param:.4f}: "
                    f"{active_edges}/{len(edge_mask)} edges active")
        
        return edge_mask
    
    def _apply_edge_mask(self,
                        laplacian: csr_matrix,
                        edge_mask: Dict[Tuple, bool],
                        edge_info: Dict,
                        metadata: object) -> csr_matrix:
        """Apply edge mask to Laplacian matrix using mathematically correct reconstruction.
        
        MATHEMATICAL APPROACH: Instead of zeroing positions (incorrect), this method
        reconstructs the Laplacian using only active edges, following the formula:
        - Diagonal blocks: Δ_vv = Σ_{active edges e=(v,w)} (R_e^T R_e) + Σ_{active edges e=(u,v)} I
        - Off-diagonal blocks: Δ_vw = -R_e^T, Δ_wv = -R_e for active edge e=(v,w)
        
        Args:
            laplacian: Original static Laplacian (used only for dimensions)
            edge_mask: Boolean mask for edges
            edge_info: Edge information with pre-computed contributions
            metadata: Laplacian construction metadata
            
        Returns:
            Mathematically correct masked Laplacian with filtered edges
        """
        # Get active edges (those with mask = True)
        active_edges = [edge for edge, keep in edge_mask.items() if keep]
        
        logger.debug(f"Reconstructing Laplacian with {len(active_edges)} active edges "
                    f"out of {len(edge_mask)} total edges")
        
        # Use the cached pre-computed contributions if available
        if hasattr(self, '_edge_contributions') and self._edge_contributions:
            return self._build_filtered_laplacian_from_cache(active_edges, laplacian.shape)
        else:
            # Fallback: rebuild from sheaf (less efficient but correct)
            logger.warning("No cached edge contributions found, using fallback reconstruction")
            return self._build_filtered_laplacian_fallback(active_edges, edge_info, metadata, laplacian.shape)
    
    def _apply_edge_mask_sparse(self,
                              laplacian: csr_matrix,
                              edge_mask: Dict[Tuple, bool],
                              edge_info: Dict,
                              metadata: object) -> torch.sparse.FloatTensor:
        """Apply edge mask to Laplacian and return torch sparse tensor.
        
        This method is used by multi-parameter persistence computation.
        
        Args:
            laplacian: Original static Laplacian (scipy sparse)
            edge_mask: Boolean mask for edges
            edge_info: Edge information
            metadata: Laplacian construction metadata
            
        Returns:
            Masked Laplacian as torch sparse tensor
        """
        # Apply the regular edge mask first
        masked_laplacian_scipy = self._apply_edge_mask(laplacian, edge_mask, edge_info, metadata)
        
        # Convert to torch sparse tensor
        coo = masked_laplacian_scipy.tocoo()
        indices = torch.from_numpy(np.stack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data).float()
        shape = coo.shape
        
        masked_laplacian_torch = torch.sparse_coo_tensor(
            indices, values, shape, dtype=torch.float32
        ).coalesce()
        
        return masked_laplacian_torch
    
    def _estimate_edge_positions(self, edge: Tuple[str, str], metadata: object) -> List[Tuple[int, int]]:
        """Estimate matrix positions for an edge (fallback when positions not cached).
        
        Args:
            edge: Edge tuple (source, target)
            metadata: Laplacian construction metadata
            
        Returns:
            List of (row, col) positions estimated for this edge
        """
        positions = []
        
        # This is a simplified fallback - in practice, would need more sophisticated
        # position estimation based on the Laplacian construction method
        source, target = edge
        
        if hasattr(metadata, 'stalk_offsets') and hasattr(metadata, 'stalk_dimensions'):
            if source in metadata.stalk_offsets and target in metadata.stalk_offsets:
                source_start = metadata.stalk_offsets[source]
                target_start = metadata.stalk_offsets[target]
                source_dim = metadata.stalk_dimensions[source]
                target_dim = metadata.stalk_dimensions[target]
                
                # Add positions for off-diagonal blocks (simplified approximation)
                for i in range(target_dim):
                    for j in range(source_dim):
                        positions.append((target_start + i, source_start + j))
                        positions.append((source_start + j, target_start + i))
        
        return positions
    
    def _compute_eigenvalues(self, laplacian: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of sparse Laplacian.
        
        Args:
            laplacian: Sparse Laplacian matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) as torch tensors
        """
        try:
            if self.eigenvalue_method == 'lobpcg' and laplacian.shape[0] > 1:
                return self._compute_eigenvalues_lobpcg(laplacian)
            else:
                return self._compute_eigenvalues_dense(laplacian)
                
        except Exception as e:
            logger.warning(f"Eigenvalue computation failed with {self.eigenvalue_method}, "
                          f"trying dense fallback: {e}")
            return self._compute_eigenvalues_dense(laplacian)
    
    def _compute_eigenvalues_lobpcg(self, laplacian: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues using LOBPCG (efficient for sparse matrices)."""
        n = laplacian.shape[0]
        k = min(self.max_eigenvalues, n - 1)
        
        if k <= 0:
            # Handle degenerate case
            eigenvals = torch.zeros(1)
            eigenvecs = torch.ones(n, 1) / np.sqrt(n)
            return eigenvals, eigenvecs
        
        # Initial guess for eigenvectors
        X = np.random.randn(n, k)
        X = X / np.linalg.norm(X, axis=0)  # Normalize columns
        
        try:
            # Compute smallest eigenvalues using LOBPCG
            eigenvals, eigenvecs = lobpcg(laplacian, X, largest=False, tol=1e-8, maxiter=1000)
            
            # Handle case where LOBPCG returns fewer eigenvalues than requested
            if eigenvals.ndim == 0:
                eigenvals = np.array([eigenvals])
                eigenvecs = eigenvecs.reshape(-1, 1)
            
            # Convert to torch tensors
            eigenvals_torch = torch.from_numpy(eigenvals).float()
            eigenvecs_torch = torch.from_numpy(eigenvecs).float()
            
            # Ensure non-negative eigenvalues (Laplacian is PSD)
            eigenvals_torch = torch.clamp(eigenvals_torch, min=0.0)
            
            # Sort by eigenvalue
            sorted_idx = torch.argsort(eigenvals_torch)
            eigenvals_torch = eigenvals_torch[sorted_idx]
            eigenvecs_torch = eigenvecs_torch[:, sorted_idx]
            
            logger.debug(f"LOBPCG computed {len(eigenvals_torch)} eigenvalues, "
                        f"min={eigenvals_torch[0]:.2e}, max={eigenvals_torch[-1]:.2e}")
            
            return eigenvals_torch, eigenvecs_torch
            
        except Exception as e:
            raise ComputationError(f"LOBPCG eigenvalue computation failed: {e}",
                                 operation="_compute_eigenvalues_lobpcg")
    
    def _compute_eigenvalues_dense(self, laplacian: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues using dense decomposition (fallback method)."""
        # Convert to dense matrix
        laplacian_dense = laplacian.toarray()
        
        # Use torch for eigenvalue decomposition
        laplacian_torch = torch.from_numpy(laplacian_dense).float()
        
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(laplacian_torch)
            
            # Ensure non-negative eigenvalues
            eigenvals = torch.clamp(eigenvals, min=0.0)
            
            # Return only the smallest eigenvalues
            k = min(self.max_eigenvalues, len(eigenvals))
            eigenvals = eigenvals[:k]
            eigenvecs = eigenvecs[:, :k]
            
            logger.debug(f"Dense method computed {len(eigenvals)} eigenvalues, "
                        f"min={eigenvals[0]:.2e}, max={eigenvals[-1]:.2e}")
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            raise ComputationError(f"Dense eigenvalue computation failed: {e}",
                                 operation="_compute_eigenvalues_dense")
    
    def _torch_sparse_to_scipy(self, tensor: torch.sparse.Tensor) -> csr_matrix:
        """Convert torch sparse tensor to scipy sparse matrix."""
        coo_tensor = tensor.coalesce()
        indices = coo_tensor.indices().cpu().numpy()
        values = coo_tensor.values().cpu().numpy()
        shape = coo_tensor.shape
        
        coo_matrix_scipy = coo_matrix(
            (values, (indices[0], indices[1])),
            shape=shape
        )
        
        return coo_matrix_scipy.tocsr()
    
    def _build_filtered_laplacian_from_cache(self, active_edges: List[Tuple[str, str]], matrix_shape: Tuple[int, int]) -> csr_matrix:
        """Build filtered Laplacian using pre-computed edge contributions.
        
        This is the optimized path that uses cached block contributions for each edge.
        
        Args:
            active_edges: List of edges to include in the filtered Laplacian
            matrix_shape: Shape of the output matrix
            
        Returns:
            Filtered Laplacian matrix constructed from active edges only
        """
        from scipy.sparse import coo_matrix
        
        # Pre-allocate arrays for COO matrix construction
        total_nnz = self._estimate_filtered_nnz(active_edges)
        rows = np.zeros(total_nnz, dtype=np.int32)
        cols = np.zeros(total_nnz, dtype=np.int32) 
        data = np.zeros(total_nnz, dtype=np.float64)
        
        nnz_idx = 0
        
        # Accumulate diagonal blocks from active edges
        diagonal_blocks = {}  # node -> accumulated diagonal contribution
        
        # Add contributions from each active edge
        for edge in active_edges:
            if edge in self._edge_contributions:
                contrib = self._edge_contributions[edge]
                
                # Add off-diagonal contributions
                if 'off_diag_positions' in contrib and 'off_diag_values' in contrib:
                    off_positions = contrib['off_diag_positions']
                    off_values = contrib['off_diag_values']
                    
                    end_idx = nnz_idx + len(off_positions)
                    rows[nnz_idx:end_idx] = [pos[0] for pos in off_positions]
                    cols[nnz_idx:end_idx] = [pos[1] for pos in off_positions]
                    data[nnz_idx:end_idx] = off_values
                    nnz_idx = end_idx
                
                # Accumulate diagonal contributions
                if 'diag_contributions' in contrib:
                    for node, diag_block in contrib['diag_contributions'].items():
                        if node not in diagonal_blocks:
                            diagonal_blocks[node] = diag_block.copy()
                        else:
                            diagonal_blocks[node] += diag_block
        
        # Add accumulated diagonal blocks
        for node, diag_block in diagonal_blocks.items():
            if hasattr(self, '_cached_metadata') and self._cached_metadata:
                node_start = self._cached_metadata.stalk_offsets.get(node, 0)
                nz_rows, nz_cols = np.where(np.abs(diag_block) > 1e-12)
                
                block_nnz = len(nz_rows)
                if nnz_idx + block_nnz <= total_nnz:
                    end_idx = nnz_idx + block_nnz
                    rows[nnz_idx:end_idx] = node_start + nz_rows
                    cols[nnz_idx:end_idx] = node_start + nz_cols
                    data[nnz_idx:end_idx] = diag_block[nz_rows, nz_cols]
                    nnz_idx = end_idx
        
        # Trim arrays to actual size and create sparse matrix
        rows = rows[:nnz_idx]
        cols = cols[:nnz_idx]
        data = data[:nnz_idx]
        
        filtered_laplacian = coo_matrix((data, (rows, cols)), shape=matrix_shape)
        return filtered_laplacian.tocsr()
    
    def _build_filtered_laplacian_fallback(self, active_edges: List[Tuple[str, str]], 
                                         edge_info: Dict, metadata: object, 
                                         matrix_shape: Tuple[int, int]) -> csr_matrix:
        """Fallback method to build filtered Laplacian when cached contributions unavailable.
        
        This reconstructs the Laplacian from scratch using only active edges.
        Less efficient than the cached approach but mathematically correct.
        
        Args:
            active_edges: List of edges to include
            edge_info: Edge information dictionary
            metadata: Laplacian construction metadata
            matrix_shape: Shape of output matrix
            
        Returns:
            Filtered Laplacian matrix
        """
        from scipy.sparse import coo_matrix
        
        logger.debug(f"Using fallback reconstruction for {len(active_edges)} active edges")
        
        rows, cols, data = [], [], []
        
        # Build node information from metadata
        if hasattr(metadata, 'stalk_offsets') and hasattr(metadata, 'stalk_dimensions'):
            stalk_offsets = metadata.stalk_offsets
            stalk_dimensions = metadata.stalk_dimensions
        else:
            logger.error("Missing stalk information in metadata for fallback reconstruction")
            # Return zero matrix as emergency fallback
            return coo_matrix(matrix_shape).tocsr()
        
        # 1. Add off-diagonal blocks for active edges
        for edge in active_edges:
            if edge in edge_info and 'restriction_matrix' in edge_info[edge]:
                u, v = edge
                R = edge_info[edge]['restriction_matrix']
                
                if u in stalk_offsets and v in stalk_offsets:
                    u_start = stalk_offsets[u]
                    v_start = stalk_offsets[v]
                    
                    # Find non-zero entries in R
                    nz_rows, nz_cols = np.where(np.abs(R) > 1e-12)
                    
                    # Add Δ_vu = -R^T
                    rows.extend(v_start + nz_rows)
                    cols.extend(u_start + nz_cols)
                    data.extend(-R[nz_rows, nz_cols])
                    
                    # Add Δ_uv = -R
                    rows.extend(u_start + nz_cols) 
                    cols.extend(v_start + nz_rows)
                    data.extend(-R[nz_rows, nz_cols])
        
        # 2. Build diagonal blocks
        diagonal_contributions = {}
        for node in stalk_dimensions:
            diagonal_contributions[node] = np.zeros((stalk_dimensions[node], stalk_dimensions[node]))
        
        # Add R^T R for outgoing edges and I for incoming edges
        for edge in active_edges:
            if edge in edge_info and 'restriction_matrix' in edge_info[edge]:
                u, v = edge
                R = edge_info[edge]['restriction_matrix']
                
                # Add R^T R to source diagonal block
                if u in diagonal_contributions and R.shape[1] == stalk_dimensions[u]:
                    diagonal_contributions[u] += R.T @ R
                
                # Add I to target diagonal block
                if v in diagonal_contributions:
                    diagonal_contributions[v] += np.eye(stalk_dimensions[v])
        
        # Add diagonal blocks to sparse matrix data
        for node, diag_block in diagonal_contributions.items():
            if node in stalk_offsets:
                node_start = stalk_offsets[node]
                nz_rows, nz_cols = np.where(np.abs(diag_block) > 1e-12)
                
                rows.extend(node_start + nz_rows)
                cols.extend(node_start + nz_cols)
                data.extend(diag_block[nz_rows, nz_cols])
        
        # Create sparse matrix
        filtered_laplacian = coo_matrix((data, (rows, cols)), shape=matrix_shape)
        return filtered_laplacian.tocsr()
    
    def _estimate_filtered_nnz(self, active_edges: List[Tuple[str, str]]) -> int:
        """Estimate number of non-zeros in filtered Laplacian."""
        # Conservative overestimate: assume each edge contributes ~10 non-zeros on average
        # This will be refined based on actual cached contribution data
        base_estimate = len(active_edges) * 10
        
        # Add estimate for diagonal blocks
        if hasattr(self, '_cached_metadata') and self._cached_metadata:
            total_dim = sum(self._cached_metadata.stalk_dimensions.values())
            diagonal_estimate = total_dim  # Rough estimate
            return base_estimate + diagonal_estimate
        
        return max(base_estimate, 1000)  # Minimum reasonable size
    
    def _precompute_edge_contributions(self, sheaf, metadata):
        """Pre-compute and cache matrix contributions for each edge.
        
        This method calculates the exact matrix blocks that each edge contributes
        to the Laplacian, enabling fast filtered Laplacian construction.
        
        Args:
            sheaf: Sheaf object with restriction maps
            metadata: Laplacian construction metadata
        """
        logger.debug("Pre-computing edge contributions for fast filtering")
        
        self._edge_contributions = {}
        
        if not hasattr(metadata, 'stalk_offsets') or not hasattr(metadata, 'stalk_dimensions'):
            logger.warning("Cannot pre-compute contributions: missing stalk information")
            return
        
        for edge, restriction in sheaf.restrictions.items():
            u, v = edge
            
            if u not in metadata.stalk_offsets or v not in metadata.stalk_offsets:
                continue
                
            u_start = metadata.stalk_offsets[u]
            v_start = metadata.stalk_offsets[v] 
            R = restriction.detach().cpu().numpy()
            
            # Pre-compute off-diagonal contributions
            nz_rows, nz_cols = np.where(np.abs(R) > 1e-12)
            off_positions = []
            off_values = []
            
            # Δ_vu = -R^T
            for i, j in zip(nz_rows, nz_cols):
                off_positions.append((v_start + i, u_start + j))
                off_values.append(-R[i, j])
                
            # Δ_uv = -R  
            for i, j in zip(nz_rows, nz_cols):
                off_positions.append((u_start + j, v_start + i))
                off_values.append(-R[i, j])
            
            # Pre-compute diagonal contributions
            diag_contributions = {}
            
            # R^T R for source node
            if R.shape[1] == metadata.stalk_dimensions[u]:
                diag_contributions[u] = R.T @ R
            
            # I for target node  
            diag_contributions[v] = np.eye(metadata.stalk_dimensions[v])
            
            self._edge_contributions[edge] = {
                'off_diag_positions': off_positions,
                'off_diag_values': off_values,
                'diag_contributions': diag_contributions
            }
        
        logger.debug(f"Pre-computed contributions for {len(self._edge_contributions)} edges")
    
    def clear_cache(self):
        """Clear cached Laplacian and edge information to free memory."""
        self._cached_laplacian = None
        self._cached_edge_info = None
        self._cached_metadata = None
        self._edge_contributions = None
        logger.info("Cleared StaticLaplacianWithMasking cache")
    
    def get_cache_info(self) -> Dict[str, bool]:
        """Get information about cached data."""
        return {
            'laplacian_cached': self._cached_laplacian is not None,
            'edge_info_cached': self._cached_edge_info is not None,
            'metadata_cached': self._cached_metadata is not None
        }
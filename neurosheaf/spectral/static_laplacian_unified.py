#!/usr/bin/env python3
"""
Unified static Laplacian with edge masking for persistent spectral analysis.

This module consolidates the functionality from both static_laplacian.py and 
static_laplacian_masking.py into a single, mathematically correct implementation.

Key Features:
- Mathematically correct Laplacian masking via block reconstruction
- Efficient caching of Laplacian construction and edge contributions  
- GPU-compatible sparse tensor operations
- Complete persistence computation pipeline
- Eigenvalue computation with LOBPCG and dense fallbacks
- Comprehensive validation and diagnostics

Mathematical Approach:
Instead of zeroing matrix positions (incorrect), this implementation reconstructs
the Laplacian using only active edges, following the correct formula:
- Diagonal blocks: Δ_vv = Σ_{active edges e=(v,w)} (R_e^T R_e) + Σ_{active edges e=(u,v)} I
- Off-diagonal blocks: Δ_vw = -R_e^T, Δ_wv = -R_e for active edge e=(v,w)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import lobpcg
import time
from dataclasses import dataclass

from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError
from ..sheaf.data_structures import Sheaf
from ..sheaf.assembly.laplacian import SheafLaplacianBuilder, LaplacianMetadata

logger = setup_logger(__name__)


@dataclass
class UnifiedMaskingMetadata:
    """Comprehensive metadata for unified static Laplacian masking.
    
    Attributes:
        edge_weights: Dictionary mapping edges to their weights
        weight_range: (min_weight, max_weight) across all edges
        threshold_count: Number of thresholds processed
        masking_times: List of times taken for each masking operation
        active_edges: Dictionary mapping thresholds to number of active edges
        laplacian_ranks: Dictionary mapping thresholds to matrix rank (if computed)
        cache_info: Information about cached data
        validation_results: Results from mathematical property validation
    """
    edge_weights: Dict[Tuple[str, str], float] = None
    weight_range: Tuple[float, float] = (0.0, 1.0)
    threshold_count: int = 0
    masking_times: List[float] = None
    active_edges: Dict[float, int] = None
    laplacian_ranks: Dict[float, int] = None
    cache_info: Dict[str, bool] = None
    validation_results: Dict[str, Dict] = None
    
    def __post_init__(self):
        """Initialize empty collections if not provided."""
        if self.edge_weights is None:
            self.edge_weights = {}
        if self.masking_times is None:
            self.masking_times = []
        if self.active_edges is None:
            self.active_edges = {}
        if self.laplacian_ranks is None:
            self.laplacian_ranks = {}
        if self.cache_info is None:
            self.cache_info = {}
        if self.validation_results is None:
            self.validation_results = {}


class UnifiedStaticLaplacian:
    """Unified static Laplacian with mathematically correct edge masking.
    
    This class consolidates the functionality from both StaticMaskedLaplacian and
    StaticLaplacianWithMasking, providing a single interface for:
    
    1. Correct Laplacian masking via block reconstruction (not zeroing entries)
    2. Efficient caching and pre-computation of edge contributions
    3. GPU-compatible operations with CPU fallbacks
    4. Complete persistence computation pipeline
    5. Comprehensive eigenvalue computation methods
    6. Mathematical property validation
    
    Design Philosophy:
    - Mathematical correctness over computational shortcuts
    - Efficient caching to amortize construction costs
    - Robust fallbacks for different computational environments
    - Clear separation of concerns between masking and eigenvalue computation
    
    Attributes:
        laplacian_builder: Builder for constructing the static Laplacian
        eigenvalue_method: Method for eigenvalue computation ('lobpcg', 'dense', 'auto')
        max_eigenvalues: Maximum number of eigenvalues to compute
        enable_gpu: Whether to enable GPU acceleration
        enable_caching: Whether to cache intermediate results
        validate_properties: Whether to validate mathematical properties
    """
    
    def __init__(self,
                 laplacian_builder: Optional[SheafLaplacianBuilder] = None,
                 eigenvalue_method: str = 'auto',
                 max_eigenvalues: int = 100,
                 enable_gpu: bool = True,
                 enable_caching: bool = True,
                 validate_properties: bool = False,
                 use_double_precision: bool = False):
        """Initialize UnifiedStaticLaplacian.
        
        Args:
            laplacian_builder: Builder for Laplacian construction (auto-created if None)
            eigenvalue_method: Eigenvalue computation method ('lobpcg', 'dense', 'auto')
            max_eigenvalues: Maximum number of eigenvalues to compute
            enable_gpu: Whether to enable GPU operations
            enable_caching: Whether to cache intermediate computations
            validate_properties: Whether to validate mathematical properties
            use_double_precision: Whether to use double precision for eigenvalue computations
        """
        self.laplacian_builder = laplacian_builder or SheafLaplacianBuilder(
            validate_properties=validate_properties
        )
        self.eigenvalue_method = eigenvalue_method
        self.max_eigenvalues = max_eigenvalues
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.enable_caching = enable_caching
        self.validate_properties = validate_properties
        self.use_double_precision = use_double_precision
        
        # Caching infrastructure
        self._cached_laplacian = None
        self._cached_metadata = None
        self._cached_edge_info = None
        self._edge_contributions = None
        self._torch_laplacian = None
        
        # Metadata tracking
        self.masking_metadata = UnifiedMaskingMetadata()
        
        logger.info(f"UnifiedStaticLaplacian initialized: {eigenvalue_method} eigenvalues, "
                   f"max_k={max_eigenvalues}, GPU={self.enable_gpu}, caching={enable_caching}, "
                   f"precision={'double' if use_double_precision else 'single'}")
    
    @classmethod
    def create_adaptive(cls, batch_size: int, **kwargs):
        """Create UnifiedStaticLaplacian with precision adapted to batch size.
        
        Args:
            batch_size: Size of the batch being processed
            **kwargs: Additional arguments for constructor
            
        Returns:
            UnifiedStaticLaplacian instance with appropriate precision settings
        """
        # Use double precision for large batch sizes where numerical issues are common
        use_double_precision = batch_size >= 64
        
        if use_double_precision:
            logger.info(f"Using double precision for spectral analysis with batch size {batch_size}")
        
        return cls(use_double_precision=use_double_precision, **kwargs)
    
    def compute_persistence(self,
                           sheaf: Sheaf,
                           filtration_params: List[float],
                           edge_threshold_func: Callable[[float, float], bool]) -> Dict:
        """Compute persistence using mathematically correct edge masking.
        
        Args:
            sheaf: Sheaf object with restrictions and stalks
            filtration_params: List of filtration parameter values
            edge_threshold_func: Function that returns True if edge should be kept
                                Signature: (edge_weight, filtration_param) -> bool
            
        Returns:
            Dictionary with persistence computation results:
            - eigenvalue_sequences: List of eigenvalue tensors
            - eigenvector_sequences: List of eigenvector tensors
            - filtration_params: Parameter values used
            - edge_info: Edge information and weights
            - masking_metadata: Metadata about masking operations
            - computation_time: Total computation time
        """
        logger.info(f"Computing persistence with {len(filtration_params)} filtration steps")
        start_time = time.time()
        
        try:
            # Build or retrieve cached Laplacian
            static_laplacian, construction_metadata = self._get_or_build_laplacian(sheaf)
            
            # Extract and cache edge information
            edge_info = self._get_or_extract_edge_info(sheaf, static_laplacian, construction_metadata)
            
            # Pre-compute edge contributions for efficient filtering
            if self.enable_caching:
                self._ensure_edge_contributions(sheaf, construction_metadata)
            
            # Compute eigenvalues/eigenvectors for each filtration step
            eigenvalue_sequences = []
            eigenvector_sequences = []
            
            for i, param in enumerate(filtration_params):
                # Create edge mask for this filtration parameter
                edge_mask = self._create_edge_mask(edge_info, param, edge_threshold_func)
                
                # Apply mathematically correct masking (block reconstruction)
                masked_laplacian = self._apply_correct_masking(
                    static_laplacian, edge_mask, edge_info, construction_metadata
                )
                
                # Compute eigenvalues/eigenvectors
                eigenvals, eigenvecs = self._compute_eigenvalues(masked_laplacian)
                
                eigenvalue_sequences.append(eigenvals)
                eigenvector_sequences.append(eigenvecs)
                
                # Update metadata
                self._update_masking_statistics(param, edge_mask, edge_info)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i + 1}/{len(filtration_params)} filtration steps")
            
            # Track eigenspaces using SubspaceTracker for full persistence analysis
            from .tracker import SubspaceTracker
            tracker = SubspaceTracker()
            
            # MATHEMATICAL CLARIFICATION: Filtration semantics for threshold filtration
            # With threshold function (weight >= param) and increasing parameters:
            # - Higher parameters → fewer edges kept → decreasing complexity
            # - This is a DECREASING COMPLEXITY filtration 
            # - However, the tracker uses 'increasing' semantics for birth < death ordering
            # - The parameter name 'increasing' refers to parameter ordering, not complexity
            filtration_direction = 'increasing'
            
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
                'masking_metadata': self.masking_metadata,
                'computation_time': computation_time,
                'method': 'unified_correct_masking'
            }
            
        except Exception as e:
            raise ComputationError(f"Unified persistence computation failed: {e}",
                                 operation="compute_persistence")
    
    def _get_or_build_laplacian(self, sheaf: Sheaf) -> Tuple[csr_matrix, LaplacianMetadata]:
        """Get cached Laplacian or build it if not cached."""
        if not self.enable_caching or self._cached_laplacian is None:
            logger.debug("Building static Laplacian")
            self._cached_laplacian, self._cached_metadata = self.laplacian_builder.build(sheaf)
            logger.info(f"Static Laplacian built: {self._cached_laplacian.shape}, "
                       f"{self._cached_laplacian.nnz:,} non-zeros")
        else:
            logger.debug("Using cached static Laplacian")
        
        return self._cached_laplacian, self._cached_metadata
    
    def _get_or_extract_edge_info(self, sheaf: Sheaf, laplacian: csr_matrix, 
                                 metadata: LaplacianMetadata) -> Dict:
        """Get cached edge information or extract it if not cached."""
        if not self.enable_caching or self._cached_edge_info is None:
            logger.debug("Extracting edge information")
            self._cached_edge_info = self._extract_edge_info(sheaf, laplacian, metadata)
            logger.debug(f"Edge information extracted for {len(self._cached_edge_info)} edges")
        else:
            logger.debug("Using cached edge information")
        
        return self._cached_edge_info
    
    def _extract_edge_info(self, sheaf: Sheaf, laplacian: csr_matrix, 
                          metadata: LaplacianMetadata) -> Dict:
        """Extract comprehensive edge information from sheaf and Laplacian."""
        edge_info = {}
        
        for edge in sheaf.restrictions.keys():
            source, target = edge
            restriction = sheaf.restrictions[edge]
            
            # Compute edge weight using Frobenius norm
            weight = torch.norm(restriction, 'fro').item()
            
            edge_info[edge] = {
                'restriction': restriction,
                'restriction_matrix': restriction.detach().cpu().numpy(),
                'weight': weight,
                'source': source,
                'target': target
            }
            
            # Add matrix positions if available from metadata
            if hasattr(metadata, 'edge_positions') and edge in metadata.edge_positions:
                edge_info[edge]['positions'] = metadata.edge_positions[edge]
        
        # Log weight statistics
        weights = [info['weight'] for info in edge_info.values()]
        if weights:
            self.masking_metadata.weight_range = (min(weights), max(weights))
            self.masking_metadata.edge_weights = {edge: info['weight'] 
                                                for edge, info in edge_info.items()}
            logger.info(f"Edge weights: min={min(weights):.4f}, max={max(weights):.4f}, "
                       f"mean={np.mean(weights):.4f}")
        
        return edge_info
    
    def _create_edge_mask(self, edge_info: Dict, filtration_param: float,
                         edge_threshold_func: Callable[[float, float], bool]) -> Dict[Tuple, bool]:
        """Create boolean mask for edges based on filtration parameter."""
        edge_mask = {}
        
        for edge, info in edge_info.items():
            keep_edge = edge_threshold_func(info['weight'], filtration_param)
            edge_mask[edge] = keep_edge
        
        active_edges = sum(edge_mask.values())
        logger.debug(f"Edge mask for param {filtration_param:.4f}: "
                    f"{active_edges}/{len(edge_mask)} edges active")
        
        return edge_mask
    
    def _apply_correct_masking(self, laplacian: csr_matrix, edge_mask: Dict[Tuple, bool],
                              edge_info: Dict, metadata: LaplacianMetadata) -> csr_matrix:
        """Apply mathematically correct masking via block reconstruction.
        
        This is the key mathematical correction: instead of zeroing matrix positions
        (which destroys Laplacian structure), we reconstruct the Laplacian using
        only the active edges according to the correct general sheaf formulation.
        
        Uses dynamic weight application strategy:
        - Cache stores unweighted structural components (R, R^T R, I)
        - Weights are applied dynamically at filtration time
        - Maximizes cache reusability across different filtration parameters
        """
        active_edges = [edge for edge, keep in edge_mask.items() if keep]
        
        logger.debug(f"Applying correct masking with {len(active_edges)} active edges")
        
        # Use cached edge contributions if available for efficiency
        if self.enable_caching and self._edge_contributions:
            # Store edge_info for dynamic weight application
            self._current_edge_info = edge_info
            filtered_laplacian = self._build_laplacian_from_cache(active_edges, laplacian.shape)
            # Clear temporary reference
            self._current_edge_info = None
            return filtered_laplacian
        else:
            # Fallback: rebuild from scratch (less efficient but always correct)
            return self._build_laplacian_from_scratch(active_edges, edge_info, metadata, laplacian.shape)
    
    def _ensure_edge_contributions(self, sheaf: Sheaf, metadata: LaplacianMetadata):
        """Ensure edge contributions are pre-computed and cached."""
        if self._edge_contributions is None:
            logger.debug("Pre-computing edge contributions for efficient masking")
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
                
                # Pre-compute UNWEIGHTED off-diagonal contributions  
                # Store unweighted values; weight will be applied dynamically
                nz_rows, nz_cols = np.where(np.abs(R) > 1e-12)
                off_positions = []
                off_values = []
                
                # Δ_vu = -R^T (unweighted, weight applied at filtration time)
                for i, j in zip(nz_rows, nz_cols):
                    off_positions.append((v_start + i, u_start + j))
                    off_values.append(-R[i, j])  # Store unweighted value
                
                # Δ_uv = -R (unweighted, weight applied at filtration time)
                for i, j in zip(nz_rows, nz_cols):
                    off_positions.append((u_start + j, v_start + i))
                    off_values.append(-R[i, j])  # Store unweighted value
                
                # Pre-compute UNWEIGHTED diagonal contributions for general sheaf Laplacian
                # Weights will be applied dynamically at filtration time for maximum cache reusability
                diag_contributions = {}
                
                # For target v: store unweighted identity (weight² applied at filtration time)
                r_v_dim = min(R.shape[0], metadata.stalk_dimensions[v])
                diag_contributions[v] = np.eye(r_v_dim)
                
                # For source u: store unweighted R^T R (weight² applied at filtration time)
                r_u_dim = min(R.shape[1], metadata.stalk_dimensions[u])
                r_v_dim = min(R.shape[0], metadata.stalk_dimensions[v])
                R_safe = R[:r_v_dim, :r_u_dim]
                diag_contributions[u] = R_safe.T @ R_safe
                
                self._edge_contributions[edge] = {
                    'off_diag_positions': off_positions,
                    'off_diag_values': off_values,
                    'diag_contributions': diag_contributions
                }
            
            logger.debug(f"Pre-computed contributions for {len(self._edge_contributions)} edges")
    
    def _get_edge_weight_from_info(self, edge: Tuple[str, str]) -> float:
        """Get edge weight from cached edge information for dynamic application."""
        if hasattr(self, '_current_edge_info') and edge in self._current_edge_info:
            return self._current_edge_info[edge].get('weight', 1.0)
        elif hasattr(self, 'masking_metadata') and edge in self.masking_metadata.edge_weights:
            return self.masking_metadata.edge_weights[edge]
        else:
            logger.warning(f"No weight found for edge {edge}, using default 1.0")
            return 1.0
    
    def _build_laplacian_from_cache(self, active_edges: List[Tuple[str, str]], 
                                   matrix_shape: Tuple[int, int]) -> csr_matrix:
        """Build filtered Laplacian using cached edge contributions (efficient path)."""
        from scipy.sparse import coo_matrix
        
        # Estimate required space
        total_nnz = self._estimate_nnz_from_cache(active_edges)
        rows = np.zeros(total_nnz, dtype=np.int32)
        cols = np.zeros(total_nnz, dtype=np.int32)
        data = np.zeros(total_nnz, dtype=np.float64)
        
        nnz_idx = 0
        diagonal_blocks = {}
        
        # Add contributions from each active edge with dynamic weight application
        for edge in active_edges:
            if edge in self._edge_contributions:
                contrib = self._edge_contributions[edge]
                
                # Get edge weight for dynamic application
                edge_weight = self._get_edge_weight_from_info(edge)
                
                # Add off-diagonal contributions (apply weight dynamically)
                if 'off_diag_positions' in contrib and 'off_diag_values' in contrib:
                    off_positions = contrib['off_diag_positions']
                    off_values = contrib['off_diag_values']
                    
                    end_idx = nnz_idx + len(off_positions)
                    if end_idx <= total_nnz:
                        rows[nnz_idx:end_idx] = [pos[0] for pos in off_positions]
                        cols[nnz_idx:end_idx] = [pos[1] for pos in off_positions]
                        # Apply weight to off-diagonal values: -weight * R
                        data[nnz_idx:end_idx] = [val * edge_weight for val in off_values]
                        nnz_idx = end_idx
                
                # Accumulate diagonal contributions (apply weight² dynamically)
                if 'diag_contributions' in contrib:
                    weight_squared = edge_weight ** 2
                    for node, unweighted_diag_block in contrib['diag_contributions'].items():
                        weighted_diag_block = weight_squared * unweighted_diag_block
                        if node not in diagonal_blocks:
                            diagonal_blocks[node] = weighted_diag_block.copy()
                        else:
                            diagonal_blocks[node] += weighted_diag_block
        
        # Add accumulated diagonal blocks
        for node, diag_block in diagonal_blocks.items():
            if hasattr(self._cached_metadata, 'stalk_offsets'):
                node_start = self._cached_metadata.stalk_offsets.get(node, 0)
                nz_rows, nz_cols = np.where(np.abs(diag_block) > 1e-12)
                
                block_nnz = len(nz_rows)
                if nnz_idx + block_nnz <= total_nnz:
                    end_idx = nnz_idx + block_nnz
                    rows[nnz_idx:end_idx] = node_start + nz_rows
                    cols[nnz_idx:end_idx] = node_start + nz_cols
                    data[nnz_idx:end_idx] = diag_block[nz_rows, nz_cols]
                    nnz_idx = end_idx
        
        # Create sparse matrix
        rows = rows[:nnz_idx]
        cols = cols[:nnz_idx]
        data = data[:nnz_idx]
        
        filtered_laplacian = coo_matrix((data, (rows, cols)), shape=matrix_shape)
        return filtered_laplacian.tocsr()
    
    def _build_laplacian_from_scratch(self, active_edges: List[Tuple[str, str]], 
                                     edge_info: Dict, metadata: LaplacianMetadata,
                                     matrix_shape: Tuple[int, int]) -> csr_matrix:
        """Build filtered Laplacian from scratch (fallback method)."""
        from scipy.sparse import coo_matrix
        
        logger.debug(f"Building Laplacian from scratch for {len(active_edges)} active edges")
        
        rows, cols, data = [], [], []
        
        # Build node information from metadata
        if hasattr(metadata, 'stalk_offsets') and hasattr(metadata, 'stalk_dimensions'):
            stalk_offsets = metadata.stalk_offsets
            stalk_dimensions = metadata.stalk_dimensions
        else:
            logger.error("Missing stalk information in metadata")
            return coo_matrix(matrix_shape).tocsr()
        
        # Add off-diagonal blocks for active edges
        for edge in active_edges:
            if edge in edge_info and 'restriction_matrix' in edge_info[edge]:
                u, v = edge
                R = edge_info[edge]['restriction_matrix']
                
                if u in stalk_offsets and v in stalk_offsets:
                    u_start = stalk_offsets[u]
                    v_start = stalk_offsets[v]
                    
                    nz_rows, nz_cols = np.where(np.abs(R) > 1e-12)
                    
                    # Add Δ_vu = -R^T
                    rows.extend(v_start + nz_rows)
                    cols.extend(u_start + nz_cols)
                    data.extend(-R[nz_rows, nz_cols])
                    
                    # Add Δ_uv = -R
                    rows.extend(u_start + nz_cols)
                    cols.extend(v_start + nz_rows)
                    data.extend(-R[nz_rows, nz_cols])
        
        # Build diagonal blocks
        diagonal_contributions = {}
        for node in stalk_dimensions:
            diagonal_contributions[node] = np.zeros((stalk_dimensions[node], stalk_dimensions[node]))
        
        # Add contributions using GENERAL sheaf Laplacian formulation (not connection Laplacian)
        # For edge e=(u,v): L[v,v] += I (incoming), L[u,u] += R^T R (outgoing)
        for edge in active_edges:
            if edge in edge_info and 'restriction_matrix' in edge_info[edge]:
                u, v = edge
                R = edge_info[edge]['restriction_matrix']
                weight = edge_info[edge].get('weight', 1.0)
                
                # For target v: add weighted identity for incoming edge
                if v in diagonal_contributions:
                    r_v_dim = min(R.shape[0], stalk_dimensions[v])
                    weighted_identity = (weight**2) * np.eye(r_v_dim)
                    diagonal_contributions[v] += weighted_identity
                
                # For source u: add R^T R for outgoing edge
                if u in diagonal_contributions:
                    r_u_dim = min(R.shape[1], stalk_dimensions[u])
                    r_v_dim = min(R.shape[0], stalk_dimensions[v])
                    R_safe = R[:r_v_dim, :r_u_dim]
                    R_weighted = weight * R_safe
                    diagonal_contributions[u] += R_weighted.T @ R_weighted
        
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
    
    def _estimate_nnz_from_cache(self, active_edges: List[Tuple[str, str]]) -> int:
        """Estimate number of non-zeros from cached contributions."""
        if not self._edge_contributions:
            return len(active_edges) * 20  # Conservative estimate
        
        total_nnz = 0
        for edge in active_edges:
            if edge in self._edge_contributions:
                contrib = self._edge_contributions[edge]
                total_nnz += len(contrib.get('off_diag_positions', []))
                # Add estimate for diagonal blocks
                total_nnz += sum(diag.size for diag in contrib.get('diag_contributions', {}).values())
        
        return max(total_nnz, 1000)  # Minimum reasonable estimate
    
    def _compute_eigenvalues(self, laplacian: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors with automatic method selection."""
        if self.eigenvalue_method == 'auto':
            # Automatic method selection based on matrix size
            if laplacian.shape[0] > 1000:
                method = 'lobpcg'
            else:
                method = 'dense'
        else:
            method = self.eigenvalue_method
        
        try:
            if method == 'lobpcg':
                return self._compute_eigenvalues_lobpcg(laplacian)
            else:
                return self._compute_eigenvalues_dense(laplacian)
        except Exception as e:
            logger.warning(f"Eigenvalue computation failed with {method}, trying dense fallback: {e}")
            return self._compute_eigenvalues_dense(laplacian)
    
    def _compute_eigenvalues_lobpcg(self, laplacian: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues using LOBPCG (efficient for sparse matrices)."""
        n = laplacian.shape[0]
        # Adaptive eigenvalue count: use matrix size or max_eigenvalues, whichever is smaller
        # For small matrices, compute all available eigenvalues
        if n <= 50:
            k = n - 1  # Compute all eigenvalues for small matrices
        else:
            k = min(self.max_eigenvalues, n - 1)  # Use limit for large matrices
        
        if k <= 0:
            eigenvals = torch.zeros(1)
            eigenvecs = torch.ones(n, 1) / np.sqrt(n)
            return eigenvals, eigenvecs
        
        # Initial guess
        X = np.random.randn(n, k)
        X = X / np.linalg.norm(X, axis=0)
        
        try:
            eigenvals, eigenvecs = lobpcg(laplacian, X, largest=False, tol=1e-8, maxiter=1000)
            
            if eigenvals.ndim == 0:
                eigenvals = np.array([eigenvals])
                eigenvecs = eigenvecs.reshape(-1, 1)
            
            # Convert to torch and ensure non-negative (PSD property)
            eigenvals_torch = torch.from_numpy(eigenvals).float()
            eigenvecs_torch = torch.from_numpy(eigenvecs).float()
            eigenvals_torch = torch.clamp(eigenvals_torch, min=0.0)
            
            # Sort by eigenvalue
            sorted_idx = torch.argsort(eigenvals_torch)
            eigenvals_torch = eigenvals_torch[sorted_idx]
            eigenvecs_torch = eigenvecs_torch[:, sorted_idx]
            
            return eigenvals_torch, eigenvecs_torch
            
        except Exception as e:
            raise ComputationError(f"LOBPCG eigenvalue computation failed: {e}")
    
    def _compute_eigenvalues_dense(self, laplacian: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues using dense decomposition (fallback method)."""
        laplacian_dense = laplacian.toarray()
        
        # Use appropriate precision for eigenvalue computation
        if self.use_double_precision:
            laplacian_torch = torch.from_numpy(laplacian_dense.astype(np.float64)).double()
        else:
            laplacian_torch = torch.from_numpy(laplacian_dense).float()
        
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(laplacian_torch)
            eigenvals = torch.clamp(eigenvals, min=0.0)
            
            # Return appropriate number of smallest eigenvalues
            n = len(eigenvals)
            if n <= 50:
                k = n  # Return all eigenvalues for small matrices
            else:
                k = min(self.max_eigenvalues, n)  # Use limit for large matrices
            eigenvals = eigenvals[:k]
            eigenvecs = eigenvecs[:, :k]
            
            # Convert back to consistent precision if needed
            if self.use_double_precision:
                eigenvals = eigenvals.double()
                eigenvecs = eigenvecs.double()
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            raise ComputationError(f"Dense eigenvalue computation failed: {e}")
    
    def _update_masking_statistics(self, filtration_param: float, edge_mask: Dict[Tuple, bool], 
                                  edge_info: Dict):
        """Update masking metadata with current operation statistics."""
        active_edges = sum(edge_mask.values())
        self.masking_metadata.active_edges[filtration_param] = active_edges
        self.masking_metadata.threshold_count += 1
    
    def clear_cache(self):
        """Clear all cached data to free memory."""
        self._cached_laplacian = None
        self._cached_metadata = None
        self._cached_edge_info = None
        self._edge_contributions = None
        self._torch_laplacian = None
        self.masking_metadata = UnifiedMaskingMetadata()
        logger.info("Cleared UnifiedStaticLaplacian cache")
    
    def get_cache_info(self) -> Dict[str, bool]:
        """Get information about cached data."""
        return {
            'laplacian_cached': self._cached_laplacian is not None,
            'edge_info_cached': self._cached_edge_info is not None,
            'metadata_cached': self._cached_metadata is not None,
            'edge_contributions_cached': self._edge_contributions is not None,
            'torch_laplacian_cached': self._torch_laplacian is not None
        }
    
    def validate_masking_correctness(self, sheaf: Sheaf, test_threshold: float = 1.0) -> Dict:
        """Validate that masking preserves mathematical properties."""
        if not self.validate_properties:
            return {'validation_skipped': True}
        
        try:
            # Test with a specific threshold
            edge_info = self._get_or_extract_edge_info(sheaf, *self._get_or_build_laplacian(sheaf))
            edge_mask = self._create_edge_mask(edge_info, test_threshold, lambda w, t: w >= t)
            
            static_laplacian, metadata = self._get_or_build_laplacian(sheaf)
            masked_laplacian = self._apply_correct_masking(static_laplacian, edge_mask, edge_info, metadata)
            
            # Check mathematical properties
            validation_results = {
                'threshold': test_threshold,
                'shape': masked_laplacian.shape,
                'nnz': masked_laplacian.nnz,
                'symmetric': None,
                'positive_semidefinite': None
            }
            
            # Use enhanced PSD validation
            try:
                from ..utils.psd_validation import validate_psd_comprehensive
                
                psd_result = validate_psd_comprehensive(
                    masked_laplacian,
                    name=f"masked_laplacian_t{test_threshold}",
                    compute_full_spectrum=False,
                    enable_regularization=True
                )
                
                validation_results['symmetric'] = psd_result.diagnostics.get('symmetry_error', 0.0) < 1e-10
                validation_results['positive_semidefinite'] = psd_result.is_psd
                validation_results['smallest_eigenvalue'] = psd_result.smallest_eigenvalue
                validation_results['condition_number'] = psd_result.condition_number
                validation_results['rank'] = psd_result.rank
                validation_results['regularization_needed'] = psd_result.regularization_needed
                
                if not psd_result.is_psd:
                    logger.warning(f"Masked Laplacian PSD validation failed at threshold {test_threshold}: {psd_result.smallest_eigenvalue:.2e}")
                
            except ImportError:
                # Fallback to original validation
                logger.debug("Enhanced PSD validation not available, using basic validation")
                
                # Check symmetry
                symmetry_error = (masked_laplacian - masked_laplacian.T).max()
                validation_results['symmetric'] = symmetry_error < 1e-10
                
                # Check smallest eigenvalue (PSD test)
                try:
                    from scipy.sparse.linalg import eigsh
                    if masked_laplacian.shape[0] > 1:
                        min_eigenval = eigsh(masked_laplacian, k=1, which='SA', return_eigenvectors=False)[0]
                        validation_results['positive_semidefinite'] = min_eigenval >= -1e-8  # Updated tolerance
                        validation_results['smallest_eigenvalue'] = min_eigenval
                    else:
                        validation_results['positive_semidefinite'] = True
                        validation_results['smallest_eigenvalue'] = 0.0
                except:
                    validation_results['positive_semidefinite'] = None
                    validation_results['smallest_eigenvalue'] = None
            
            except Exception as e:
                logger.warning(f"Enhanced PSD validation failed: {e}")
                validation_results['validation_error'] = str(e)
            
            self.masking_metadata.validation_results[test_threshold] = validation_results
            return validation_results
            
        except Exception as e:
            return {'validation_error': str(e)}


def create_unified_static_laplacian(sheaf: Sheaf, **kwargs) -> UnifiedStaticLaplacian:
    """Convenience function to create UnifiedStaticLaplacian from sheaf.
    
    Args:
        sheaf: Sheaf object with restrictions and stalks
        **kwargs: Additional arguments for UnifiedStaticLaplacian constructor
        
    Returns:
        UnifiedStaticLaplacian ready for persistence analysis
    """
    unified_laplacian = UnifiedStaticLaplacian(**kwargs)
    
    # Pre-build and validate if requested
    if kwargs.get('validate_properties', False):
        validation_result = unified_laplacian.validate_masking_correctness(sheaf)
        logger.info(f"Masking validation: {validation_result}")
    
    return unified_laplacian


# Backward compatibility aliases
StaticLaplacianWithMasking = UnifiedStaticLaplacian  # Main alias for existing code

# Deprecation warnings for old classes (to be implemented when migrating)
def _create_deprecation_warning(old_class_name: str, new_class_name: str):
    """Helper to create deprecation warnings."""
    import warnings
    warnings.warn(
        f"{old_class_name} is deprecated and will be removed in a future version. "
        f"Use {new_class_name} instead for mathematically correct Laplacian masking.",
        DeprecationWarning,
        stacklevel=3
    )
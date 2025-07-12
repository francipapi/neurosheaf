"""Static Laplacian with efficient edge masking for filtration.

This module implements the static masking approach where the full Laplacian
is built once and Boolean masks are applied for different threshold values.
This avoids expensive matrix reconstruction at each filtration level.

Key Features:
- Static Laplacian construction from whitened sheaf data
- Efficient Boolean masking for edge-weight filtration  
- GPU-compatible sparse tensor operations
- Edge position caching for masking integrity
- Memory-efficient threshold sweeping
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError
from ..sheaf.laplacian import SheafLaplacianBuilder, LaplacianMetadata

logger = setup_logger(__name__)


@dataclass 
class MaskingMetadata:
    """Metadata for edge masking operations.
    
    Attributes:
        edge_weights: Dictionary mapping edges to their weights
        weight_range: (min_weight, max_weight) across all edges
        threshold_count: Number of thresholds processed
        masking_times: List of times taken for each masking operation
        active_edges: Dictionary mapping thresholds to number of active edges
        laplacian_ranks: Dictionary mapping thresholds to matrix rank (if computed)
    """
    edge_weights: Dict[Tuple[str, str], float] = None
    weight_range: Tuple[float, float] = (0.0, 1.0)
    threshold_count: int = 0
    masking_times: List[float] = None
    active_edges: Dict[float, int] = None
    laplacian_ranks: Dict[float, int] = None
    
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


class StaticMaskedLaplacian:
    """Static Laplacian with efficient edge masking for filtration.
    
    Key innovation: Build Laplacian once, apply masks for different thresholds.
    This avoids expensive matrix reconstruction at each filtration level.
    
    The masking works by:
    1. Building full Laplacian Δ with all edges included
    2. Caching edge positions for each restriction map
    3. For threshold τ: zero out entries from edges with weight ≤ τ
    4. Adjust diagonal to maintain Laplacian property
    
    Attributes:
        L_static: Full static Laplacian (scipy.sparse.csr_matrix)
        L_torch: GPU-compatible torch sparse tensor
        edge_cache: Mapping from edges to matrix positions
        metadata: Construction and masking metadata
        enable_gpu: Whether GPU operations are enabled
    """
    
    def __init__(self, static_laplacian: csr_matrix, metadata: LaplacianMetadata,
                 masking_metadata: MaskingMetadata, enable_gpu: bool = True):
        """Initialize with pre-built static Laplacian.
        
        Args:
            static_laplacian: Full Laplacian with all edges included
            metadata: Metadata from Laplacian construction
            masking_metadata: Edge weight and masking information
            enable_gpu: Whether to enable GPU sparse tensor operations
        """
        self.L_static = static_laplacian
        self.construction_metadata = metadata
        self.masking_metadata = masking_metadata
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        
        # Convert to torch sparse tensor for GPU operations
        if self.enable_gpu:
            self.L_torch = self._csr_to_torch_sparse(static_laplacian)
            logger.info(f"Static Laplacian loaded on GPU: {static_laplacian.shape}")
        else:
            self.L_torch = None
            logger.info(f"Static Laplacian in CPU mode: {static_laplacian.shape}")
        
        # Validate edge cache integrity
        self._validate_edge_cache()
        
        # Initialize masking statistics
        self._update_weight_statistics()
    
    def _csr_to_torch_sparse(self, L_csr: csr_matrix) -> torch.sparse.FloatTensor:
        """Convert scipy CSR to torch sparse COO tensor."""
        L_coo = L_csr.tocoo()
        
        indices = torch.stack([
            torch.from_numpy(L_coo.row).long(),
            torch.from_numpy(L_coo.col).long()
        ])
        values = torch.from_numpy(L_coo.data).float()
        
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, L_coo.shape, dtype=torch.float32
        )
        
        if self.enable_gpu:
            sparse_tensor = sparse_tensor.cuda()
        
        return sparse_tensor.coalesce()
    
    def _validate_edge_cache(self):
        """Validate that edge cache covers all matrix entries correctly."""
        if not hasattr(self.construction_metadata, 'edge_positions'):
            logger.warning("No edge position cache available - masking may be inefficient")
            return
        
        total_cached_positions = sum(len(positions) for positions in 
                                   self.construction_metadata.edge_positions.values())
        
        logger.debug(f"Edge cache contains {len(self.construction_metadata.edge_positions)} edges "
                    f"with {total_cached_positions} matrix positions")
        
        # Verify no overlapping positions (each entry should belong to exactly one edge)
        all_positions = set()
        overlaps = 0
        
        for edge, positions in self.construction_metadata.edge_positions.items():
            for pos in positions:
                if pos in all_positions:
                    overlaps += 1
                all_positions.add(pos)
        
        if overlaps > 0:
            logger.warning(f"Found {overlaps} overlapping positions in edge cache - "
                          "masking may not preserve Laplacian structure")
        else:
            logger.debug("Edge cache validation passed - no overlapping positions")
    
    def _update_weight_statistics(self):
        """Update weight range and distribution statistics."""
        if not self.masking_metadata.edge_weights:
            logger.warning("No edge weights available for masking")
            return
        
        weights = list(self.masking_metadata.edge_weights.values())
        self.masking_metadata.weight_range = (min(weights), max(weights))
        
        logger.info(f"Edge weight range: [{self.masking_metadata.weight_range[0]:.4f}, "
                   f"{self.masking_metadata.weight_range[1]:.4f}]")
    
    def apply_threshold_mask(self, threshold: float, return_torch: bool = False) -> Union[csr_matrix, torch.sparse.FloatTensor]:
        """Apply threshold mask to create filtered Laplacian.
        
        For threshold τ, keeps only edges with weight > τ and adjusts
        the Laplacian structure accordingly.
        
        Args:
            threshold: Weight threshold (edges with weight ≤ threshold are removed)
            return_torch: Whether to return torch.sparse tensor (GPU-compatible)
            
        Returns:
            Filtered Laplacian as scipy.sparse.csr_matrix or torch.sparse.FloatTensor
            
        Raises:
            ComputationError: If masking operation fails
        """
        start_time = time.time()
        
        try:
            if return_torch and self.L_torch is not None:
                # GPU-based masking
                masked_laplacian = self._apply_mask_torch(threshold)
            else:
                # CPU-based masking
                masked_laplacian = self._apply_mask_scipy(threshold)
            
            # Update masking statistics
            masking_time = time.time() - start_time
            self.masking_metadata.masking_times.append(masking_time)
            self.masking_metadata.threshold_count += 1
            
            # Count active edges
            active_edges = sum(1 for weight in self.masking_metadata.edge_weights.values() 
                             if weight > threshold)
            self.masking_metadata.active_edges[threshold] = active_edges
            
            logger.debug(f"Applied threshold {threshold:.4f}: {active_edges} active edges, "
                        f"{masking_time:.4f}s")
            
            return masked_laplacian
            
        except Exception as e:
            raise ComputationError(f"Threshold masking failed at τ={threshold}: {e}",
                                  operation="apply_threshold_mask")
    
    def _apply_mask_scipy(self, threshold: float) -> csr_matrix:
        """Apply threshold mask using scipy sparse operations."""
        # Start with a copy of the static Laplacian
        L_masked = self.L_static.copy()
        
        # Identify edges to remove (weight ≤ threshold)
        edges_to_remove = [edge for edge, weight in self.masking_metadata.edge_weights.items()
                          if weight <= threshold]
        
        # Zero out matrix entries for removed edges
        for edge in edges_to_remove:
            if edge in self.construction_metadata.edge_positions:
                positions = self.construction_metadata.edge_positions[edge]
                for row, col in positions:
                    L_masked[row, col] = 0.0
        
        # Eliminate zeros and recompute structure
        L_masked.eliminate_zeros()
        
        return L_masked
    
    def _apply_mask_torch(self, threshold: float) -> torch.sparse.FloatTensor:
        """Apply threshold mask using torch sparse operations."""
        # Get the COO representation
        L_coo = self.L_torch.coalesce()
        indices = L_coo.indices()
        values = L_coo.values()
        
        # Create mask for entries to keep
        mask = torch.ones_like(values, dtype=torch.bool)
        
        # Mark entries from removed edges
        edges_to_remove = [edge for edge, weight in self.masking_metadata.edge_weights.items()
                          if weight <= threshold]
        
        for edge in edges_to_remove:
            if edge in self.construction_metadata.edge_positions:
                for row, col in self.construction_metadata.edge_positions[edge]:
                    # Find corresponding entries in sparse tensor
                    entry_mask = (indices[0] == row) & (indices[1] == col)
                    mask[entry_mask] = False
        
        # Apply mask
        filtered_indices = indices[:, mask]
        filtered_values = values[mask]
        
        # Create new sparse tensor
        masked_tensor = torch.sparse_coo_tensor(
            filtered_indices, filtered_values, L_coo.shape,
            dtype=torch.float32, device=L_coo.device
        )
        
        return masked_tensor.coalesce()
    
    def compute_filtration_sequence(self, thresholds: List[float], 
                                   return_torch: bool = False) -> List[Union[csr_matrix, torch.sparse.FloatTensor]]:
        """Compute Laplacian sequence for multiple threshold values.
        
        Args:
            thresholds: List of threshold values in ascending order
            return_torch: Whether to return torch.sparse tensors
            
        Returns:
            List of filtered Laplacians corresponding to each threshold
        """
        logger.info(f"Computing filtration sequence for {len(thresholds)} thresholds")
        
        laplacian_sequence = []
        
        for i, threshold in enumerate(thresholds):
            masked_laplacian = self.apply_threshold_mask(threshold, return_torch=return_torch)
            laplacian_sequence.append(masked_laplacian)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(thresholds)} thresholds")
        
        avg_masking_time = np.mean(self.masking_metadata.masking_times[-len(thresholds):])
        logger.info(f"Filtration sequence complete: avg {avg_masking_time:.4f}s per threshold")
        
        return laplacian_sequence
    
    def get_weight_distribution(self, num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Get histogram of edge weights for threshold selection.
        
        Args:
            num_bins: Number of histogram bins
            
        Returns:
            Tuple of (bin_centers, counts) for edge weight distribution
        """
        weights = list(self.masking_metadata.edge_weights.values())
        
        if not weights:
            return np.array([]), np.array([])
        
        counts, bin_edges = np.histogram(weights, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, counts
    
    def suggest_thresholds(self, num_thresholds: int = 50, 
                          strategy: str = 'uniform') -> List[float]:
        """Suggest threshold values for filtration.
        
        Args:
            num_thresholds: Number of threshold values to generate
            strategy: Threshold selection strategy ('uniform', 'quantile', 'adaptive')
            
        Returns:
            List of suggested threshold values
        """
        weights = list(self.masking_metadata.edge_weights.values())
        
        if not weights:
            return []
        
        min_weight, max_weight = self.masking_metadata.weight_range
        
        if strategy == 'uniform':
            # Uniform spacing between min and max weights
            thresholds = np.linspace(min_weight * 0.9, max_weight * 1.1, num_thresholds)
        
        elif strategy == 'quantile':
            # Quantile-based thresholds
            quantiles = np.linspace(0, 1, num_thresholds)
            thresholds = np.quantile(weights, quantiles)
        
        elif strategy == 'adaptive':
            # Adaptive spacing based on weight distribution
            bin_centers, counts = self.get_weight_distribution(num_bins=num_thresholds//2)
            if len(bin_centers) > 0:
                # More thresholds in regions with more edges
                cumulative_counts = np.cumsum(counts)
                normalized_cumulative = cumulative_counts / cumulative_counts[-1]
                quantiles = np.linspace(0, 1, num_thresholds)
                thresholds = np.interp(quantiles, normalized_cumulative, bin_centers)
            else:
                # Fallback to uniform
                thresholds = np.linspace(min_weight * 0.9, max_weight * 1.1, num_thresholds)
        
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")
        
        # Ensure thresholds are sorted and unique
        thresholds = np.unique(np.sort(thresholds))
        
        logger.info(f"Generated {len(thresholds)} thresholds using '{strategy}' strategy")
        return thresholds.tolist()
    
    def validate_masking_integrity(self, threshold: float) -> Dict[str, Any]:
        """Validate that masking preserves Laplacian mathematical properties.
        
        Args:
            threshold: Threshold value to test
            
        Returns:
            Dictionary with validation results
        """
        try:
            masked_laplacian = self.apply_threshold_mask(threshold, return_torch=False)
            
            validation_results = {
                'threshold': threshold,
                'shape': masked_laplacian.shape,
                'nnz': masked_laplacian.nnz,
                'symmetric': None,
                'positive_semidefinite': None,
                'diagonal_sum': None
            }
            
            # Check symmetry
            symmetry_error = (masked_laplacian - masked_laplacian.T).max()
            validation_results['symmetric'] = symmetry_error < 1e-10
            
            # Check diagonal sum (should be sum of off-diagonal elements)
            diagonal_sum = masked_laplacian.diagonal().sum()
            off_diagonal_sum = (masked_laplacian.sum() - diagonal_sum) / 2  # Account for symmetry
            validation_results['diagonal_sum'] = abs(diagonal_sum + off_diagonal_sum)
            
            # Check smallest eigenvalue (positive semi-definite test)
            try:
                from scipy.sparse.linalg import eigsh
                if masked_laplacian.shape[0] > 1:
                    min_eigenval = eigsh(masked_laplacian, k=1, which='SA', return_eigenvectors=False)[0]
                    validation_results['positive_semidefinite'] = min_eigenval >= -1e-10
                else:
                    validation_results['positive_semidefinite'] = True
            except:
                validation_results['positive_semidefinite'] = None
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Masking validation failed: {e}")
            return {'threshold': threshold, 'error': str(e)}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in GB.
        
        Returns:
            Dictionary with memory usage breakdown
        """
        memory_stats = {}
        
        # Static Laplacian memory
        static_memory = self.L_static.data.nbytes + self.L_static.indices.nbytes + self.L_static.indptr.nbytes
        memory_stats['static_laplacian_gb'] = static_memory / (1024**3)
        
        # GPU tensor memory
        if self.L_torch is not None:
            torch_memory = self.L_torch.values().element_size() * self.L_torch.values().numel()
            torch_memory += self.L_torch.indices().element_size() * self.L_torch.indices().numel()
            memory_stats['torch_tensor_gb'] = torch_memory / (1024**3)
        else:
            memory_stats['torch_tensor_gb'] = 0.0
        
        # Edge cache memory  
        cache_memory = 0
        if hasattr(self.construction_metadata, 'edge_positions'):
            for positions in self.construction_metadata.edge_positions.values():
                cache_memory += len(positions) * 16  # Rough estimate: 2 ints per position
        memory_stats['edge_cache_gb'] = cache_memory / (1024**3)
        
        memory_stats['total_gb'] = sum(memory_stats.values())
        
        return memory_stats


def create_static_masked_laplacian(sheaf, enable_gpu: bool = True) -> StaticMaskedLaplacian:
    """Convenience function to create StaticMaskedLaplacian from sheaf.
    
    Args:
        sheaf: Sheaf object with whitened stalks and restrictions
        enable_gpu: Whether to enable GPU operations
        
    Returns:
        StaticMaskedLaplacian ready for filtration analysis
    """
    # Build static Laplacian
    builder = SheafLaplacianBuilder(enable_gpu=enable_gpu, validate_properties=True)
    static_laplacian, construction_metadata = builder.build(sheaf)
    
    # Extract edge weights for masking
    edge_weights = {}
    for edge, restriction in sheaf.restrictions.items():
        # Use Frobenius norm as default weight
        edge_weights[edge] = torch.norm(restriction, p='fro').item()
    
    # Create masking metadata
    masking_metadata = MaskingMetadata(edge_weights=edge_weights)
    
    # Create StaticMaskedLaplacian
    return StaticMaskedLaplacian(
        static_laplacian=static_laplacian,
        metadata=construction_metadata,
        masking_metadata=masking_metadata,
        enable_gpu=enable_gpu
    )
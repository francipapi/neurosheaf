# neurosheaf/spectral/gw/pes_computation.py
"""
Core computation engine for Persistent Eigenvector Similarity (PES).

Implements the mathematical framework from "Disentangling the Spectral Properties 
of the Hodge Laplacian" with GW-specific adaptations for transport-based weighting
and optimal eigenvector matching using the Hungarian algorithm.

Mathematical Foundation:
PES(v_i, v_j) = |ι(v_i)^T v_j| / (||v_i||_2 · ||v_j||_2)

With optional transport weighting:
PES_transport(v_i, v_j) = PES(v_i, v_j) * exp(-α * transport_cost)
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from scipy.optimize import linear_sum_assignment
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError

logger = setup_logger(__name__)


class PESComputer:
    """
    Core computation engine for Persistent Eigenvector Similarity (PES).
    
    Implements the mathematical framework with GW-specific adaptations including
    transport-based weighting and optimal eigenvector matching.
    """
    
    def __init__(self, 
                 threshold: float = 0.8,
                 transport_weighting_alpha: float = 1.0,
                 numerical_tolerance: float = 1e-12):
        """
        Initialize PES computer.
        
        Args:
            threshold: Minimum PES similarity for accepting matches
            transport_weighting_alpha: Exponential weighting parameter for transport costs
            numerical_tolerance: Tolerance for numerical computations
        """
        self.threshold = threshold
        self.transport_weighting_alpha = transport_weighting_alpha
        self.numerical_tolerance = numerical_tolerance
        
        logger.info(f"PESComputer initialized: threshold={threshold}, "
                   f"transport_alpha={transport_weighting_alpha}")
    
    def compute_pes_matrix(self, 
                          prev_eigenvecs: torch.Tensor,
                          curr_eigenvecs: torch.Tensor,
                          transport_weighting: bool = True,
                          transport_costs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute PES similarity matrix between consecutive filtration steps.
        
        Mathematical formula:
        PES(v_i, v_j) = |ι(v_i)^T v_j| / (||v_i||_2 · ||v_j||_2)
        
        With optional transport weighting:
        PES_transport(v_i, v_j) = PES(v_i, v_j) * exp(-α * transport_cost)
        
        Args:
            prev_eigenvecs: Eigenvectors from previous filtration step [dim x n_prev]
            curr_eigenvecs: Eigenvectors from current filtration step [dim x n_curr]
            transport_weighting: Whether to apply transport-based weighting
            transport_costs: Transport cost matrix [n_prev x n_curr] (optional)
            
        Returns:
            PES similarity matrix [n_prev x n_curr]
            
        Raises:
            ComputationError: If eigenvector matrices have incompatible dimensions
        """
        if prev_eigenvecs.shape[0] != curr_eigenvecs.shape[0]:
            raise ComputationError(
                f"Eigenvector dimension mismatch: {prev_eigenvecs.shape[0]} vs {curr_eigenvecs.shape[0]}",
                operation="compute_pes_matrix"
            )
        
        n_prev, n_curr = prev_eigenvecs.shape[1], curr_eigenvecs.shape[1]
        pes_matrix = torch.zeros(n_prev, n_curr, dtype=prev_eigenvecs.dtype)
        
        logger.debug(f"Computing PES matrix: {n_prev} x {n_curr} eigenvectors")
        
        # Precompute norms for efficiency
        prev_norms = torch.norm(prev_eigenvecs, p=2, dim=0)  # [n_prev]
        curr_norms = torch.norm(curr_eigenvecs, p=2, dim=0)  # [n_curr]
        
        # Batch computation using matrix multiplication
        # Compute all dot products at once: [n_prev x n_curr]
        dot_products = torch.abs(torch.mm(prev_eigenvecs.T, curr_eigenvecs))
        
        # Compute normalized similarities using broadcasting
        # Handle numerical issues with small norms
        valid_prev = prev_norms > self.numerical_tolerance
        valid_curr = curr_norms > self.numerical_tolerance
        
        for i in range(n_prev):
            for j in range(n_curr):
                if valid_prev[i] and valid_curr[j]:
                    pes_similarity = dot_products[i, j] / (prev_norms[i] * curr_norms[j])
                else:
                    pes_similarity = 0.0
                
                # Apply transport weighting if requested and available
                if transport_weighting and transport_costs is not None:
                    transport_weight = self._compute_transport_weight(i, j, transport_costs)
                    pes_similarity *= transport_weight
                
                pes_matrix[i, j] = pes_similarity
        
        # Log statistics
        non_zero_similarities = torch.sum(pes_matrix > self.numerical_tolerance).item()
        above_threshold = torch.sum(pes_matrix > self.threshold).item()
        
        logger.debug(f"PES matrix computed: {non_zero_similarities} non-zero similarities, "
                    f"{above_threshold} above threshold {self.threshold}")
        
        return pes_matrix
    
    def optimal_eigenvector_matching(self, 
                                   pes_matrix: torch.Tensor,
                                   use_threshold: bool = True) -> List[Tuple[int, int, float]]:
        """
        Find optimal eigenvector matching using Hungarian algorithm.
        
        Maximizes total PES similarity across all matches while ensuring
        one-to-one correspondence between eigenvectors.
        
        Args:
            pes_matrix: PES similarity matrix [n_prev x n_curr]
            use_threshold: Whether to filter matches by similarity threshold
            
        Returns:
            List of matches as (prev_idx, curr_idx, similarity) tuples,
            sorted by similarity in descending order
            
        Raises:
            ComputationError: If Hungarian algorithm fails
        """
        try:
            # Convert similarities to costs (Hungarian minimizes cost)
            cost_matrix = 1.0 - pes_matrix.detach().cpu().numpy()
            
            # Handle rectangular matrices by padding with high cost
            n_prev, n_curr = cost_matrix.shape
            if n_prev != n_curr:
                max_dim = max(n_prev, n_curr)
                padded_cost = np.full((max_dim, max_dim), 2.0)  # High cost > 1
                padded_cost[:n_prev, :n_curr] = cost_matrix
                cost_matrix = padded_cost
            
            # Solve optimal assignment problem
            prev_indices, curr_indices = linear_sum_assignment(cost_matrix)
            
            # Extract valid matches
            matches = []
            for prev_idx, curr_idx in zip(prev_indices, curr_indices):
                # Only consider matches within original matrix dimensions
                if prev_idx < n_prev and curr_idx < n_curr:
                    similarity = pes_matrix[prev_idx, curr_idx].item()
                    
                    # Apply threshold filter if requested
                    if not use_threshold or similarity > self.threshold:
                        matches.append((prev_idx, curr_idx, similarity))
            
            # Sort by similarity for consistency and debugging
            matches.sort(key=lambda x: x[2], reverse=True)
            
            logger.debug(f"Optimal matching found {len(matches)} matches "
                        f"{'above threshold' if use_threshold else 'total'}")
            
            return matches
            
        except Exception as e:
            raise ComputationError(
                f"Hungarian algorithm failed: {e}",
                operation="optimal_eigenvector_matching"
            )
    
    def greedy_eigenvector_matching(self, 
                                  pes_matrix: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        Fallback greedy matching when Hungarian algorithm is not available.
        
        Greedily selects highest similarity matches while avoiding duplicates.
        Less optimal than Hungarian algorithm but computationally simpler.
        
        Args:
            pes_matrix: PES similarity matrix [n_prev x n_curr]
            
        Returns:
            List of matches as (prev_idx, curr_idx, similarity) tuples
        """
        n_prev, n_curr = pes_matrix.shape
        matches = []
        used_prev = set()
        used_curr = set()
        
        # Create list of all (similarity, prev_idx, curr_idx) tuples
        all_similarities = []
        for i in range(n_prev):
            for j in range(n_curr):
                similarity = pes_matrix[i, j].item()
                if similarity > self.threshold:
                    all_similarities.append((similarity, i, j))
        
        # Sort by similarity in descending order
        all_similarities.sort(reverse=True)
        
        # Greedily select matches
        for similarity, prev_idx, curr_idx in all_similarities:
            if prev_idx not in used_prev and curr_idx not in used_curr:
                matches.append((prev_idx, curr_idx, similarity))
                used_prev.add(prev_idx)
                used_curr.add(curr_idx)
        
        logger.debug(f"Greedy matching found {len(matches)} matches")
        return matches
    
    def _compute_transport_weight(self, 
                                i: int, 
                                j: int, 
                                transport_costs: torch.Tensor) -> float:
        """
        Compute transport-based weighting for PES similarity.
        
        Uses exponential weighting: weight = exp(-α * transport_cost)
        Lower transport cost → higher weight → higher effective similarity
        
        Args:
            i: Previous eigenvector index
            j: Current eigenvector index
            transport_costs: Transport cost matrix
            
        Returns:
            Transport weight in [0, 1]
        """
        if transport_costs.numel() == 0:
            return 1.0
        
        # Handle index bounds gracefully
        max_i = min(i, transport_costs.shape[0] - 1)
        max_j = min(j, transport_costs.shape[1] - 1)
        
        cost = transport_costs[max_i, max_j]
        
        # Exponential weighting with clipping for numerical stability
        weight = torch.exp(-self.transport_weighting_alpha * cost)
        return float(torch.clamp(weight, min=0.0, max=1.0))
    
    def compute_pes_statistics(self, 
                              pes_matrix: torch.Tensor) -> Dict[str, Union[float, int]]:
        """
        Compute statistics about PES similarity matrix.
        
        Args:
            pes_matrix: PES similarity matrix
            
        Returns:
            Dictionary with similarity statistics
        """
        pes_flat = pes_matrix.flatten()
        
        stats = {
            'mean_similarity': torch.mean(pes_flat).item(),
            'std_similarity': torch.std(pes_flat).item(),
            'max_similarity': torch.max(pes_flat).item(),
            'min_similarity': torch.min(pes_flat).item(),
            'median_similarity': torch.median(pes_flat).item(),
            'above_threshold_count': torch.sum(pes_flat > self.threshold).item(),
            'above_threshold_fraction': (torch.sum(pes_flat > self.threshold) / pes_flat.numel()).item(),
            'non_zero_count': torch.sum(pes_flat > self.numerical_tolerance).item(),
            'sparsity': 1.0 - (torch.sum(pes_flat > self.numerical_tolerance) / pes_flat.numel()).item()
        }
        
        return stats
    
    def validate_pes_matrix(self, pes_matrix: torch.Tensor) -> bool:
        """
        Validate PES similarity matrix for mathematical correctness.
        
        Checks:
        1. All values in [0, 1] range
        2. No NaN or infinite values
        3. Matrix dimensions are positive
        
        Args:
            pes_matrix: PES similarity matrix to validate
            
        Returns:
            True if matrix is valid, False otherwise
        """
        try:
            # Check for NaN or infinite values
            if torch.any(torch.isnan(pes_matrix)) or torch.any(torch.isinf(pes_matrix)):
                logger.error("PES matrix contains NaN or infinite values")
                return False
            
            # Check value range [0, 1]
            if torch.any(pes_matrix < 0) or torch.any(pes_matrix > 1):
                logger.error("PES matrix contains values outside [0, 1] range")
                return False
            
            # Check dimensions
            if pes_matrix.shape[0] == 0 or pes_matrix.shape[1] == 0:
                logger.error("PES matrix has zero dimension")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"PES matrix validation failed: {e}")
            return False
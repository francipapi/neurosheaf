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
from scipy.linalg import cholesky, LinAlgError
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError
from ..dtype_policy import DEFAULT_SPECTRAL_POLICY, to_spectral_dtype

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
                 numerical_tolerance: float = 1e-12,
                 metric: Optional[str] = None):
        """
        Initialize PES computer.
        
        Args:
            threshold: Minimum PES similarity for accepting matches
            transport_weighting_alpha: Exponential weighting parameter for transport costs
            numerical_tolerance: Tolerance for numerical computations
            metric: Inner product metric ('M' for M-orthogonal, None for Euclidean)
        """
        self.threshold = threshold
        self.transport_weighting_alpha = transport_weighting_alpha
        self.numerical_tolerance = numerical_tolerance
        self.metric = metric
        
        logger.info(f"PESComputer initialized: threshold={threshold}, "
                   f"transport_alpha={transport_weighting_alpha}, metric={metric}")
    
    def compute_pes_matrix(self, 
                          prev_eigenvecs: torch.Tensor,
                          curr_eigenvecs: torch.Tensor,
                          transport_weighting: bool = True,
                          transport_costs: Optional[torch.Tensor] = None,
                          mass_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute PES similarity matrix between consecutive filtration steps.
        
        Mathematical formula:
        - Euclidean: PES(v_i, v_j) = |v_i^T v_j| / (||v_i||_2 · ||v_j||_2)
        - M-orthogonal: PES(v_i, v_j) = |v_i^T M v_j| / (||v_i||_M · ||v_j||_M)
        
        With optional transport weighting:
        PES_transport(v_i, v_j) = PES(v_i, v_j) * exp(-α * transport_cost)
        
        Args:
            prev_eigenvecs: Eigenvectors from previous filtration step [dim x n_prev]
            curr_eigenvecs: Eigenvectors from current filtration step [dim x n_curr]
            transport_weighting: Whether to apply transport-based weighting
            transport_costs: Transport cost matrix [n_prev x n_curr] (optional)
            mass_matrix: Mass matrix M for M-orthogonal computation (optional)
            
        Returns:
            PES similarity matrix [n_prev x n_curr]
            
        Raises:
            ComputationError: If eigenvector matrices have incompatible dimensions
        """
        logger.debug(f"PES DEBUG: compute_pes_matrix called with prev_eigenvecs type: {type(prev_eigenvecs)}, curr_eigenvecs type: {type(curr_eigenvecs)}")
        
        if isinstance(prev_eigenvecs, tuple):
            logger.debug(f"PES DEBUG: prev_eigenvecs is tuple: {[type(x) for x in prev_eigenvecs]}")
            return DEFAULT_SPECTRAL_POLICY.create_zeros(1, 1)
        if isinstance(curr_eigenvecs, tuple):
            logger.debug(f"PES DEBUG: curr_eigenvecs is tuple: {[type(x) for x in curr_eigenvecs]}")
            return DEFAULT_SPECTRAL_POLICY.create_zeros(1, 1)
        
        if prev_eigenvecs.shape[0] != curr_eigenvecs.shape[0]:
            raise ComputationError(
                f"Eigenvector dimension mismatch: {prev_eigenvecs.shape[0]} vs {curr_eigenvecs.shape[0]}",
                operation="compute_pes_matrix"
            )
            
        # Determine whether to use M-orthogonal computation
        use_m_orthogonal = (self.metric == 'M' and mass_matrix is not None)
        
        if use_m_orthogonal:
            return self._compute_pes_matrix_m_orthogonal(
                prev_eigenvecs, curr_eigenvecs, mass_matrix,
                transport_weighting, transport_costs
            )
        else:
            return self._compute_pes_matrix_euclidean(
                prev_eigenvecs, curr_eigenvecs,
                transport_weighting, transport_costs
            )
    
    def _compute_pes_matrix_euclidean(self,
                                    prev_eigenvecs: torch.Tensor,
                                    curr_eigenvecs: torch.Tensor,
                                    transport_weighting: bool,
                                    transport_costs: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute PES matrix using standard Euclidean inner product.
        
        Args:
            prev_eigenvecs: Previous eigenvectors [dim x n_prev]
            curr_eigenvecs: Current eigenvectors [dim x n_curr]
            transport_weighting: Whether to apply transport weighting
            transport_costs: Transport cost matrix (optional)
            
        Returns:
            PES similarity matrix [n_prev x n_curr]
        """
        n_prev, n_curr = prev_eigenvecs.shape[1], curr_eigenvecs.shape[1]
        pes_matrix = DEFAULT_SPECTRAL_POLICY.create_zeros(n_prev, n_curr, device=prev_eigenvecs.device)
        
        logger.debug(f"Computing Euclidean PES matrix: {n_prev} x {n_curr} eigenvectors")
        
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
        
        logger.debug(f"Euclidean PES matrix computed: {non_zero_similarities} non-zero similarities, "
                    f"{above_threshold} above threshold {self.threshold}")
        
        return pes_matrix
    
    def _compute_pes_matrix_m_orthogonal(self,
                                       prev_eigenvecs: torch.Tensor,
                                       curr_eigenvecs: torch.Tensor,
                                       mass_matrix: torch.Tensor,
                                       transport_weighting: bool,
                                       transport_costs: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute PES matrix using M-orthogonal inner product.
        
        For eigenvectors from generalized eigenvalue problems Av = λMv,
        eigenvectors are M-orthonormal. This method uses the proper
        M-inner product: <v, w>_M = v^T M w.
        
        Implementation uses Cholesky decomposition: M = C^T C, then
        transforms to Euclidean space before computing similarities.
        
        Args:
            prev_eigenvecs: Previous eigenvectors [dim x n_prev]
            curr_eigenvecs: Current eigenvectors [dim x n_curr] 
            mass_matrix: Mass matrix M [dim x dim]
            transport_weighting: Whether to apply transport weighting
            transport_costs: Transport cost matrix (optional)
            
        Returns:
            PES similarity matrix [n_prev x n_curr]
            
        Raises:
            ComputationError: If Cholesky decomposition fails
        """
        n_prev, n_curr = prev_eigenvecs.shape[1], curr_eigenvecs.shape[1]
        
        logger.debug(f"Computing M-orthogonal PES matrix: {n_prev} x {n_curr} eigenvectors")
        
        try:
            # Convert mass matrix to numpy for Cholesky decomposition
            M_numpy = mass_matrix.detach().cpu().numpy().astype(DEFAULT_SPECTRAL_POLICY.numpy_dtype)
            
            # Add regularization for numerical stability
            regularization = 1e-12 * np.trace(M_numpy) / M_numpy.shape[0]
            M_regularized = M_numpy + regularization * np.eye(M_numpy.shape[0])
            
            # Cholesky decomposition: M = C^T C
            C = cholesky(M_regularized, lower=False)  # Upper triangular
            C_tensor = DEFAULT_SPECTRAL_POLICY.from_numpy(C, device=prev_eigenvecs.device)
            
            # Transform eigenvectors to Euclidean space: v_euclidean = C @ v_M
            prev_euclidean = torch.mm(C_tensor, prev_eigenvecs)  # [dim x n_prev]
            curr_euclidean = torch.mm(C_tensor, curr_eigenvecs)  # [dim x n_curr]
            
            # Compute PES in Euclidean space (which corresponds to M-inner product in original space)
            return self._compute_pes_matrix_euclidean(
                prev_euclidean, curr_euclidean,
                transport_weighting, transport_costs
            )
            
        except LinAlgError as e:
            logger.error(f"Cholesky decomposition failed for mass matrix: {e}")
            logger.warning("Falling back to Euclidean PES computation")
            return self._compute_pes_matrix_euclidean(
                prev_eigenvecs, curr_eigenvecs,
                transport_weighting, transport_costs
            )
        except Exception as e:
            raise ComputationError(
                f"M-orthogonal PES computation failed: {e}",
                operation="_compute_pes_matrix_m_orthogonal"
            )
    
    def optimal_eigenvector_matching(self, 
                                   pes_matrix: torch.Tensor,
                                   use_threshold: bool = True,
                                   adaptive_threshold: bool = True,
                                   clustered_eigenvals: Optional[List[List[int]]] = None) -> List[Tuple[int, int, float]]:
        """
        Find optimal eigenvector matching using Hungarian algorithm with sign/permutation resolution.
        
        🔧 LARGE PRINCIPAL ANGLE FIX: Enhanced for clustered eigenvalues
        Maximizes total PES similarity across all matches while ensuring
        one-to-one correspondence between eigenvectors. Uses adaptive thresholding
        and handles sign/permutation resolution within eigenvalue clusters.
        
        Args:
            pes_matrix: PES similarity matrix [n_prev x n_curr]
            use_threshold: Whether to filter matches by similarity threshold
            adaptive_threshold: Whether to adapt threshold based on matrix statistics
            clustered_eigenvals: Optional list of eigenvalue clusters [[indices], ...]
            
        Returns:
            List of matches as (prev_idx, curr_idx, similarity) tuples,
            sorted by similarity in descending order
            
        Raises:
            ComputationError: If Hungarian algorithm fails
        """
        try:
            # 🔧 SIGN/PERMUTATION RESOLUTION: Handle clustered eigenvalues with absolute overlaps
            if clustered_eigenvals is not None:
                enhanced_pes_matrix = self._resolve_sign_permutation_in_clusters(
                    pes_matrix, clustered_eigenvals
                )
            else:
                enhanced_pes_matrix = pes_matrix
            
            # Convert similarities to costs (Hungarian minimizes cost)
            cost_matrix = 1.0 - enhanced_pes_matrix.detach().cpu().numpy()
            
            # Handle rectangular matrices by padding with high cost
            n_prev, n_curr = cost_matrix.shape
            if n_prev != n_curr:
                max_dim = max(n_prev, n_curr)
                padded_cost = np.full((max_dim, max_dim), 2.0)  # High cost > 1
                padded_cost[:n_prev, :n_curr] = cost_matrix
                cost_matrix = padded_cost
            
            # Solve optimal assignment problem
            prev_indices, curr_indices = linear_sum_assignment(cost_matrix)
            
            # Determine adaptive threshold if requested
            effective_threshold = self.threshold
            if use_threshold and adaptive_threshold:
                effective_threshold = self._compute_adaptive_threshold(enhanced_pes_matrix)
                if effective_threshold != self.threshold:
                    logger.debug(f"Adaptive threshold: {self.threshold:.3f} → {effective_threshold:.3f}")
            
            # Extract valid matches
            matches = []
            for prev_idx, curr_idx in zip(prev_indices, curr_indices):
                # Only consider matches within original matrix dimensions
                if prev_idx < n_prev and curr_idx < n_curr:
                    similarity = enhanced_pes_matrix[prev_idx, curr_idx].item()
                    
                    # Apply threshold filter if requested
                    if not use_threshold or similarity > effective_threshold:
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
    
    def _compute_adaptive_threshold(self, pes_matrix: torch.Tensor) -> float:
        """
        Compute adaptive threshold based on PES matrix statistics.
        
        Analyzes the distribution of similarity values to determine an appropriate
        threshold that balances precision (avoiding false matches) with recall
        (not missing true matches) for eigenvalue tracking.
        
        Args:
            pes_matrix: PES similarity matrix [n_prev x n_curr]
            
        Returns:
            Adaptive threshold value
        """
        try:
            # Flatten matrix and get basic statistics
            similarities = pes_matrix.flatten()
            non_zero_similarities = similarities[similarities > self.numerical_tolerance]
            
            if len(non_zero_similarities) == 0:
                logger.warning("No non-zero similarities found, using default threshold")
                return self.threshold
            
            # Compute statistical measures
            mean_sim = torch.mean(non_zero_similarities).item()
            std_sim = torch.std(non_zero_similarities).item()
            max_sim = torch.max(non_zero_similarities).item()
            median_sim = torch.median(non_zero_similarities).item()
            
            # Count values above different threshold candidates
            above_default = torch.sum(non_zero_similarities > self.threshold).item()
            above_median = torch.sum(non_zero_similarities > median_sim).item()
            above_mean = torch.sum(non_zero_similarities > mean_sim).item()
            
            # Adaptive strategy based on similarity distribution characteristics
            adaptive_threshold = self.threshold
            
            # Case 1: Very low similarities overall (poor embedding quality)
            # Lower threshold to avoid losing all matches
            if max_sim < 0.5:
                adaptive_threshold = max(0.1, median_sim * 0.8)
                logger.debug(f"Low max similarity {max_sim:.3f}, using median-based threshold")
                
            # Case 2: High variance in similarities (mixed embedding quality)
            # Use threshold based on mean + std to focus on clearly good matches
            elif std_sim > 0.2 and mean_sim > 0.3:
                adaptive_threshold = min(self.threshold, mean_sim + 0.5 * std_sim)
                logger.debug(f"High variance {std_sim:.3f}, using mean+0.5*std threshold")
                
            # Case 3: Very few matches with default threshold
            # Lower threshold to get more matches for tracking continuity
            elif above_default < min(pes_matrix.shape) * 0.3:  # Less than 30% of expected matches
                adaptive_threshold = max(0.2, median_sim)
                logger.debug(f"Too few matches ({above_default}), using median threshold")
                
            # Case 4: Too many high-quality matches (might indicate noise)
            # Raise threshold to be more selective
            elif above_default > min(pes_matrix.shape) * 1.5:  # More than 150% of expected
                adaptive_threshold = min(0.95, mean_sim + std_sim)
                logger.debug(f"Too many matches ({above_default}), using mean+std threshold")
                
            # Case 5: Normal case - use default but bounded by statistics
            else:
                # Keep default threshold but ensure it's reasonable given the data
                adaptive_threshold = max(0.1, min(self.threshold, median_sim * 0.9))
            
            # Ensure adaptive threshold is within reasonable bounds
            adaptive_threshold = max(0.05, min(0.95, adaptive_threshold))
            
            # Log detailed statistics for debugging
            logger.debug(f"PES matrix stats: mean={mean_sim:.3f}, std={std_sim:.3f}, "
                        f"median={median_sim:.3f}, max={max_sim:.3f}")
            logger.debug(f"Matches above thresholds: default({self.threshold:.2f})={above_default}, "
                        f"median({median_sim:.2f})={above_median}, mean({mean_sim:.2f})={above_mean}")
            
            return adaptive_threshold
            
        except Exception as e:
            logger.warning(f"Adaptive threshold computation failed: {e}")
            return self.threshold  # Fallback to default
    
    def _resolve_sign_permutation_in_clusters(self, 
                                            pes_matrix: torch.Tensor,
                                            clustered_eigenvals: List[List[int]]) -> torch.Tensor:
        """
        🔧 SIGN/PERMUTATION RESOLUTION: Enhance PES matrix for clustered eigenvalues.
        
        Within eigenvalue clusters (multiplicities), eigenvectors can have arbitrary
        signs and permutations. This method uses absolute overlaps and optimal
        assignment within clusters to resolve these ambiguities.
        
        Args:
            pes_matrix: Original PES matrix [n_prev x n_curr]
            clustered_eigenvals: List of eigenvalue clusters [[indices], ...]
            
        Returns:
            Enhanced PES matrix with sign/permutation resolution
        """
        try:
            enhanced_matrix = pes_matrix.clone()
            
            # Process each cluster pair
            for prev_cluster_indices in clustered_eigenvals:
                for curr_cluster_indices in clustered_eigenvals:
                    
                    # Extract cluster submatrix
                    prev_indices = torch.tensor(prev_cluster_indices, dtype=torch.long)
                    curr_indices = torch.tensor(curr_cluster_indices, dtype=torch.long)
                    
                    if len(prev_indices) == 0 or len(curr_indices) == 0:
                        continue
                    
                    # Get cluster submatrix using advanced indexing
                    cluster_pes = pes_matrix[prev_indices][:, curr_indices]
                    
                    # Apply absolute value for sign-invariant matching within cluster
                    abs_cluster_pes = torch.abs(cluster_pes)
                    
                    # Use Hungarian algorithm within this cluster to find optimal permutation
                    cost_submatrix = 1.0 - abs_cluster_pes.detach().cpu().numpy()
                    
                    # Handle non-square cluster submatrices
                    if cost_submatrix.shape[0] != cost_submatrix.shape[1]:
                        # Pad to square matrix
                        max_dim = max(cost_submatrix.shape)
                        padded_cost = np.full((max_dim, max_dim), 2.0)
                        padded_cost[:cost_submatrix.shape[0], :cost_submatrix.shape[1]] = cost_submatrix
                        cost_submatrix = padded_cost
                    
                    try:
                        # Solve assignment within cluster
                        cluster_prev_indices, cluster_curr_indices = linear_sum_assignment(cost_submatrix)
                        
                        # Update enhanced matrix with optimal absolute values
                        for cp_idx, cc_idx in zip(cluster_prev_indices, cluster_curr_indices):
                            if (cp_idx < len(prev_indices) and cc_idx < len(curr_indices)):
                                prev_global_idx = prev_indices[cp_idx]
                                curr_global_idx = curr_indices[cc_idx]
                                
                                # Use absolute value of original similarity
                                original_val = pes_matrix[prev_global_idx, curr_global_idx]
                                enhanced_matrix[prev_global_idx, curr_global_idx] = torch.abs(original_val)
                    
                    except Exception as e:
                        logger.debug(f"Hungarian assignment failed for cluster, using absolute values: {e}")
                        # Fallback: just use absolute values without optimal assignment
                        for i, prev_idx in enumerate(prev_indices):
                            for j, curr_idx in enumerate(curr_indices):
                                enhanced_matrix[prev_idx, curr_idx] = torch.abs(pes_matrix[prev_idx, curr_idx])
            
            logger.debug(f"Applied sign/permutation resolution to {len(clustered_eigenvals)} clusters")
            return enhanced_matrix
            
        except Exception as e:
            logger.warning(f"Sign/permutation resolution failed: {e}, using original matrix")
            return pes_matrix
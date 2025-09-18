"""Integrated Sliced Wasserstein (ISW) distance for eigenvalue evolution.

This module provides ISW distance computation for comparing eigenvalue evolution
sequences from the neurosheaf persistent spectral analysis pipeline.

ISW distance is permutation-invariant to eigenvalue ordering and captures the
temporal dynamics of spectral properties across filtration parameters.

Key features:
- Quantile-based comparison (permutation invariant)
- Time-weighted integration with optional exponential decay
- Robust preprocessing with log scaling and normalization
- Compatible with save_eigenvalue_evolution output format
- Efficient pairwise distance matrix computation

References:
- 1-D Wasserstein = L^p distance between quantile functions
- Integrated over time with optional weighting w(t) ∝ exp(-alpha * t)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union, Any
from pathlib import Path
import warnings
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError, ComputationError
from ..io.eigenvalue_io import load_eigenvalue_evolution

# Optional scipy import for more robust Wasserstein computation
try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Using fallback Wasserstein implementation.")

logger = setup_logger(__name__)


class EigenvalueISW:
    """Integrated Sliced Wasserstein distance for eigenvalue evolution comparison.
    
    This class provides methods to compute ISW distances between eigenvalue evolution
    sequences, enabling comparison of spectral dynamics across different neural networks
    or analysis parameters.
    
    The ISW distance treats eigenvalues at each time step as a 1D distribution,
    computes Wasserstein distance between quantile functions, and integrates over time.
    This approach is permutation-invariant and robust to eigenvalue crossing events.
    
    Attributes:
        p: L^p norm parameter (1 or 2, default: 1)
        n_quantiles: Number of quantiles for distribution comparison (default: 199)
        tail_trim: Fraction to trim from distribution tails (default: 0.02)
        time_weight_alpha: Exponential decay parameter for time weighting (default: 0.0)
        log_scale: Whether to apply log1p transformation (default: False)
        eps: Small constant for numerical stability (default: 1e-3)
        normalize: Normalization method ('p95', 'max', or None, default: 'p95')
    """
    
    def __init__(self,
                 p: int = 1,
                 n_quantiles: int = 199,
                 tail_trim: float = 0.02,
                 time_weight_alpha: float = 0.0,
                 log_scale: bool = False,
                 eps: float = 1e-3,
                 normalize: Optional[str] = 'p95',
                 use_extrapolation: bool = True):
        """Initialize EigenvalueISW.
        
        Args:
            p: L^p norm parameter (1 or 2)
            n_quantiles: Number of quantiles for comparison
            tail_trim: Fraction to trim from tails (0.0-0.5)
            time_weight_alpha: Time weighting parameter (0.0 = uniform)
            log_scale: Apply log1p transformation
            eps: Floor value for numerical stability
            normalize: Normalization method ('p95', 'max', None)
            use_extrapolation: If True, use edge values for extrapolation; 
                             if False, use NaN outside observed time range
        """
        if p not in [1, 2]:
            raise ValueError("p must be 1 or 2")
        if not (0.0 <= tail_trim < 0.5):
            raise ValueError("tail_trim must be in [0.0, 0.5)")
        if normalize and normalize not in ['p95', 'p99', 'max']:
            raise ValueError("normalize must be 'p95', 'p99', 'max', or None")
        
        self.p = p
        self.n_quantiles = n_quantiles
        self.tail_trim = tail_trim
        self.time_weight_alpha = time_weight_alpha
        self.log_scale = log_scale
        self.eps = eps
        self.normalize = normalize
        self.use_extrapolation = use_extrapolation
        
        # Create quantile grid
        lo = self.tail_trim
        hi = 1.0 - self.tail_trim
        self.quantile_grid = np.linspace(lo, hi, self.n_quantiles)
        
        logger.debug(f"Initialized EigenvalueISW with p={p}, n_quantiles={n_quantiles}, "
                    f"tail_trim={tail_trim}, alpha={time_weight_alpha}")
    
    def _preprocess_eigenvalues(self, 
                               eigenvalue_matrix: np.ndarray) -> np.ndarray:
        """Apply preprocessing to eigenvalue matrix.
        
        Includes type conversion, negativity clipping, flooring, and log transformation.
        
        Args:
            eigenvalue_matrix: Array of shape (T, n_eigs)
            
        Returns:
            Preprocessed eigenvalue matrix in float64
        """
        # Force float64 for numerical stability
        L = np.asarray(eigenvalue_matrix, dtype=np.float64)
        
        # Clip numerical negative eigenvalues (Laplacian eigenvalues should be >= 0)
        L[L < 0] = 0.0
        
        # Apply floor for numerical stability
        if self.eps > 0:
            L[L < self.eps] = self.eps
        
        # Apply log transformation
        if self.log_scale:
            L = np.log1p(L)
        
        return L
    
    def _normalize_eigenvalues(self, 
                              eigenvalue_matrices: List[np.ndarray]) -> List[np.ndarray]:
        """Apply global normalization across all eigenvalue matrices.
        
        Args:
            eigenvalue_matrices: List of eigenvalue matrices
            
        Returns:
            List of normalized eigenvalue matrices
        """
        if self.normalize is None:
            return eigenvalue_matrices
        
        # Compute global normalization factor
        all_vals = np.concatenate([L.ravel() for L in eigenvalue_matrices])
        all_vals = all_vals[np.isfinite(all_vals)]  # Remove NaN/inf
        
        if len(all_vals) == 0:
            logger.warning("No finite values for normalization")
            return eigenvalue_matrices
        
        if self.normalize == 'p95':
            scale = np.percentile(all_vals, 95.0)
        elif self.normalize == 'p99':
            scale = np.percentile(all_vals, 99.0)
        elif self.normalize == 'max':
            scale = np.max(all_vals)
        else:
            scale = 1.0
        
        scale = float(scale) if scale > 0 else 1.0
        
        logger.debug(f"Normalizing with scale factor: {scale:.6e}")
        return [L / scale for L in eigenvalue_matrices]
    
    def _resample_to_common_grid(self,
                                eigenvalue_matrix: np.ndarray,
                                time_vector: np.ndarray,
                                common_time: np.ndarray) -> np.ndarray:
        """Resample eigenvalue matrix to common time grid.
        
        Args:
            eigenvalue_matrix: Array of shape (T, n_eigs)
            time_vector: Time vector of length T
            common_time: Common time grid of length K
            
        Returns:
            Resampled eigenvalue matrix of shape (K, n_eigs)
        """
        T, n_eigs = eigenvalue_matrix.shape
        K = len(common_time)
        
        # Handle edge cases
        if T == 0 or n_eigs == 0:
            return np.full((K, n_eigs), np.nan)
        
        if T == 1:
            # Single time point - replicate
            return np.tile(eigenvalue_matrix[0:1, :], (K, 1))
        
        # Interpolate each eigenvalue sequence
        resampled = np.empty((K, n_eigs))
        
        for j in range(n_eigs):
            # Handle NaN values by masking
            valid_mask = np.isfinite(eigenvalue_matrix[:, j])
            if not np.any(valid_mask):
                # All NaN - fill with NaN
                resampled[:, j] = np.nan
                continue
            
            valid_times = time_vector[valid_mask]
            valid_eigs = eigenvalue_matrix[valid_mask, j]
            
            if len(valid_times) == 1:
                # Single valid point - replicate
                resampled[:, j] = valid_eigs[0]
            else:
                # Ensure sorted times for np.interp (required for monotonic x)
                order = np.argsort(valid_times)
                valid_times = valid_times[order]
                valid_eigs = valid_eigs[order]
                
                # Interpolate with optional extrapolation
                if self.use_extrapolation:
                    resampled[:, j] = np.interp(
                        common_time, valid_times, valid_eigs,
                        left=valid_eigs[0], right=valid_eigs[-1]
                    )
                else:
                    resampled[:, j] = np.interp(
                        common_time, valid_times, valid_eigs,
                        left=np.nan, right=np.nan
                    )
        
        return resampled
    
    def _compute_quantiles(self, 
                          eigenvalue_matrix: np.ndarray) -> np.ndarray:
        """Compute quantiles across eigenvalues at each time step.
        
        Vectorized implementation using np.nanquantile for efficiency.
        Compatible with different NumPy versions.
        
        Args:
            eigenvalue_matrix: Array of shape (T, n_eigs)
            
        Returns:
            Quantile matrix of shape (T, n_quantiles)
        """
        # Vectorized quantile computation with NumPy version compatibility
        # result shape: (n_quantiles, T) -> transpose to get (T, n_quantiles)
        try:
            # NumPy >= 1.22 uses 'method' parameter
            quantiles = np.nanquantile(
                eigenvalue_matrix, 
                self.quantile_grid, 
                axis=1, 
                method='linear'
            )
        except TypeError:
            # NumPy < 1.22 uses 'interpolation' parameter
            quantiles = np.nanquantile(
                eigenvalue_matrix, 
                self.quantile_grid, 
                axis=1, 
                interpolation='linear'
            )
        
        return quantiles.T
    
    def _compute_time_weights(self, time_vector: np.ndarray) -> np.ndarray:
        """Compute time weights for integration.
        
        Args:
            time_vector: Time vector of length T
            
        Returns:
            Normalized weight vector of length T
        """
        if self.time_weight_alpha <= 0:
            # Uniform weighting
            weights = np.ones_like(time_vector)
        else:
            # Exponential decay weighting
            t_norm = (time_vector - time_vector.min()) / (
                time_vector.max() - time_vector.min() + 1e-12
            )
            weights = np.exp(-self.time_weight_alpha * t_norm)
        
        # Normalize weights to integrate to 1
        if len(time_vector) > 1:
            weights = weights / np.trapz(weights, x=time_vector)
        else:
            weights = weights / weights.sum()
        
        return weights
    
    def _wasserstein_1d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute exact 1D Wasserstein distance between two distributions.
        
        Uses scipy when available, otherwise implements exact W₁ as area between
        empirical CDFs on union grid (handles unequal sample sizes correctly).
        
        Args:
            x: First distribution samples
            y: Second distribution samples
            
        Returns:
            Wasserstein-1 distance
        """
        if SCIPY_AVAILABLE:
            return float(wasserstein_distance(x, y))
        
        # Exact fallback implementation for unequal sample sizes
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Remove non-finite values
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        
        # Handle edge cases
        if x.size == 0 and y.size == 0:
            return 0.0
        if x.size == 0:
            return float(np.mean(np.abs(y - np.median(y))))
        if y.size == 0:
            return float(np.mean(np.abs(x - np.median(x))))
        
        # Sort distributions
        xs = np.sort(x)
        ys = np.sort(y)
        
        # Create union grid for computing CDFs
        grid = np.unique(np.concatenate([xs, ys]))
        
        # Compute empirical CDFs on the shared grid
        Fx = np.searchsorted(xs, grid, side="right") / xs.size
        Fy = np.searchsorted(ys, grid, side="right") / ys.size
        
        # W₁ = integral of |F_x - F_y| 
        return float(np.trapz(np.abs(Fx - Fy), x=grid))
    
    def isw_distance(self,
                    eigenvalue_matrix1: np.ndarray,
                    time_vector1: np.ndarray,
                    eigenvalue_matrix2: np.ndarray,
                    time_vector2: np.ndarray,
                    common_time: Optional[np.ndarray] = None,
                    scale: Optional[float] = None) -> float:
        """Compute ISW distance between two eigenvalue evolution sequences.
        
        Args:
            eigenvalue_matrix1: First eigenvalue matrix (T1, n_eigs1)
            time_vector1: First time vector (T1,)
            eigenvalue_matrix2: Second eigenvalue matrix (T2, n_eigs2)
            time_vector2: Second time vector (T2,)
            common_time: Optional common time grid. If None, computed automatically
            scale: Optional normalization scale. If None, computed from these two matrices
            
        Returns:
            ISW distance (scalar)
            
        Note:
            When common_time and scale are provided, the distance is directly comparable
            to results from pairwise_isw_matrix using the same parameters.
        """
        logger.debug(f"Computing ISW distance between matrices of shapes "
                    f"{eigenvalue_matrix1.shape} and {eigenvalue_matrix2.shape}")
        
        # Use provided common_time or compute automatically
        if common_time is None:
            # Determine common time domain (intersection)
            t_min = max(time_vector1.min(), time_vector2.min())
            t_max = min(time_vector1.max(), time_vector2.max())
            
            if t_max <= t_min:
                raise ValueError("No overlapping time domain between sequences")
            
            # Create common time grid
            common_time = np.linspace(t_min, t_max, max(len(time_vector1), len(time_vector2)))
        
        # Preprocess eigenvalues
        L1 = self._preprocess_eigenvalues(eigenvalue_matrix1)
        L2 = self._preprocess_eigenvalues(eigenvalue_matrix2)
        
        # Apply normalization (global or provided scale)
        if scale is None:
            L1, L2 = self._normalize_eigenvalues([L1, L2])
        else:
            if self.normalize is not None:
                L1 = L1 / scale
                L2 = L2 / scale
        
        # Resample to common time grid
        L1_resampled = self._resample_to_common_grid(L1, time_vector1, common_time)
        L2_resampled = self._resample_to_common_grid(L2, time_vector2, common_time)
        
        # Compute quantiles at each time step
        Q1 = self._compute_quantiles(L1_resampled)  # (K, n_quantiles)
        Q2 = self._compute_quantiles(L2_resampled)  # (K, n_quantiles)
        
        # Compute L^p distance between quantiles at each time
        diff = np.abs(Q1 - Q2)
        
        if self.p == 1:
            per_time_distances = np.nanmean(diff, axis=1)
        elif self.p == 2:
            per_time_distances = np.sqrt(np.nanmean(diff ** 2, axis=1))
        else:
            per_time_distances = np.nanmean(diff ** self.p, axis=1) ** (1.0 / self.p)
        
        # Handle NaN time points properly - don't bias to zero
        valid_mask = np.isfinite(per_time_distances)
        
        if not np.any(valid_mask):
            logger.warning("No valid time points for ISW distance computation")
            return 0.0
        
        # Compute time weights and renormalize on valid times only
        weights = self._compute_time_weights(common_time)
        
        valid_times = common_time[valid_mask]
        valid_weights = weights[valid_mask]
        valid_distances = per_time_distances[valid_mask]
        
        # Renormalize weights to integrate to 1 over valid domain
        if len(valid_times) > 1:
            valid_weights = valid_weights / np.trapz(valid_weights, x=valid_times)
            distance = float(np.trapz(valid_distances * valid_weights, x=valid_times))
        else:
            distance = float(valid_distances[0])
        
        logger.debug(f"Computed ISW distance: {distance:.6e}")
        return distance
    
    def load_eigenvalue_data(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, str]:
        """Load eigenvalue evolution data from file.
        
        Args:
            filepath: Path to eigenvalue evolution file
            
        Returns:
            Tuple of (eigenvalue_matrix, time_vector, name)
        """
        filepath = Path(filepath)
        
        try:
            eigenvalue_matrix, time_vector, metadata = load_eigenvalue_evolution(filepath)
            name = filepath.stem
            
            logger.debug(f"Loaded {name}: matrix shape {eigenvalue_matrix.shape}, "
                        f"time vector length {len(time_vector)}")
            
            return eigenvalue_matrix, time_vector, name
            
        except Exception as e:
            raise ComputationError(f"Failed to load eigenvalue data from {filepath}: {e}") from e
    
    def pairwise_isw_matrix(self, 
                           filepaths: List[Union[str, Path]], 
                           K: int = 200) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise ISW distance matrix with global preprocessing.
        
        This implementation ensures distances are comparable across all pairs by:
        1. Using a single global time grid for all resampling
        2. Computing a single global normalization scale across all data
        3. Precomputing quantiles once for efficiency
        4. Properly handling NaN values with weight renormalization
        
        Args:
            filepaths: List of paths to eigenvalue evolution files
            K: Number of time points in common grid (default: 200)
            
        Returns:
            Tuple of (distance_matrix, names)
            - distance_matrix: Symmetric matrix of shape (N, N)
            - names: List of file names corresponding to matrix indices
        """
        logger.info(f"Computing pairwise ISW matrix for {len(filepaths)} files")
        
        # 1) Load all data and ensure monotone time ordering
        raw_data = []
        names = []
        
        for filepath in filepaths:
            eigenvalue_matrix, time_vector, name = self.load_eigenvalue_data(filepath)
            
            # Ensure monotone time ordering
            order = np.argsort(time_vector)
            sorted_eigs = eigenvalue_matrix[order, :]
            sorted_time = time_vector[order]
            
            raw_data.append((sorted_eigs, sorted_time))
            names.append(name)
        
        # 2) Compute global time intersection and create fixed grid
        t_min = max(t[0] for _, t in raw_data)
        t_max = min(t[-1] for _, t in raw_data)
        
        if not (t_max > t_min):
            raise ValidationError("No overlapping time domain across inputs.")
        
        common_time = np.linspace(t_min, t_max, K)
        logger.debug(f"Using global time grid: [{t_min:.3f}, {t_max:.3f}] with {K} points")
        
        # 3) Preprocess all eigenvalue matrices
        preprocessed = []
        for eigenvalue_matrix, _ in raw_data:
            prep_matrix = self._preprocess_eigenvalues(eigenvalue_matrix)
            preprocessed.append(prep_matrix)
        
        # 4) Compute single global normalization scale
        if self.normalize is not None:
            all_vals = np.concatenate([P.ravel() for P in preprocessed])
            all_vals = all_vals[np.isfinite(all_vals)]
            
            if all_vals.size == 0:
                raise ComputationError("No finite values for normalization.")
            
            if self.normalize == 'p95':
                scale = float(np.percentile(all_vals, 95.0))
            elif self.normalize == 'p99':
                scale = float(np.percentile(all_vals, 99.0))
            else:  # 'max'
                scale = float(np.max(all_vals))
            
            if scale <= 0:
                scale = 1.0
            
            logger.debug(f"Global normalization scale: {scale:.6e}")
            normalized = [P / scale for P in preprocessed]
        else:
            normalized = preprocessed
        
        # 5) Resample all to the same grid and compute quantiles once
        resampled_matrices = []
        quantile_matrices = []
        
        for (norm_matrix, (_, time_vector)) in zip(normalized, raw_data):
            resampled = self._resample_to_common_grid(norm_matrix, time_vector, common_time)
            quantiles = self._compute_quantiles(resampled)  # (K, n_quantiles)
            
            resampled_matrices.append(resampled)
            quantile_matrices.append(quantiles)
        
        # 6) Compute time weights once
        time_weights = self._compute_time_weights(common_time)
        
        # 7) Helper function for ISW distance from precomputed quantiles
        def _isw_from_quantiles(Q1: np.ndarray, Q2: np.ndarray) -> float:
            """Compute ISW distance from precomputed quantiles with proper NaN handling."""
            # Compute L^p distance between quantiles at each time
            diff = np.abs(Q1 - Q2)
            
            if self.p == 1:
                per_time_distances = np.nanmean(diff, axis=1)
            elif self.p == 2:
                per_time_distances = np.sqrt(np.nanmean(diff ** 2, axis=1))
            else:
                per_time_distances = np.nanmean(diff ** self.p, axis=1) ** (1.0 / self.p)
            
            # Handle NaN time points properly - don't bias to zero
            valid_mask = np.isfinite(per_time_distances)
            
            if not np.any(valid_mask):
                return 0.0
            
            # Renormalize weights on valid times only
            valid_times = common_time[valid_mask]
            valid_weights = time_weights[valid_mask]
            valid_distances = per_time_distances[valid_mask]
            
            # Renormalize weights to integrate to 1 over valid domain
            if len(valid_times) > 1:
                valid_weights = valid_weights / np.trapz(valid_weights, x=valid_times)
                distance = float(np.trapz(valid_distances * valid_weights, x=valid_times))
            else:
                distance = float(valid_distances[0])
            
            return distance
        
        # 8) Compute pairwise distances using precomputed quantiles
        N = len(quantile_matrices)
        distance_matrix = np.zeros((N, N))
        
        for i in range(N):
            distance_matrix[i, i] = 0.0
            
            for j in range(i + 1, N):
                try:
                    distance = _isw_from_quantiles(quantile_matrices[i], quantile_matrices[j])
                    distance_matrix[i, j] = distance_matrix[j, i] = distance
                    
                    logger.debug(f"Distance {names[i]} <-> {names[j]}: {distance:.6e}")
                    
                except Exception as e:
                    logger.warning(f"Failed to compute distance between {names[i]} and {names[j]}: {e}")
                    distance_matrix[i, j] = distance_matrix[j, i] = np.nan
        
        logger.info(f"Completed pairwise ISW matrix computation with global preprocessing")
        return distance_matrix, names
    
    def endpoint_wasserstein_matrix(self,
                                   filepaths: List[Union[str, Path]]) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise Wasserstein distance at final time point with consistent preprocessing.
        
        Uses the same preprocessing pipeline as the main ISW computation to ensure
        consistent scaling and transformations.
        
        Args:
            filepaths: List of paths to eigenvalue evolution files
            
        Returns:
            Tuple of (distance_matrix, names)
        """
        logger.info(f"Computing endpoint Wasserstein matrix for {len(filepaths)} files")
        
        # Load and preprocess first (consistent with pairwise_isw_matrix)
        loaded_data = []
        names = []
        
        for filepath in filepaths:
            eigenvalue_matrix, time_vector, name = self.load_eigenvalue_data(filepath)
            loaded_data.append((eigenvalue_matrix, time_vector, name))
            names.append(name)
        
        # Apply preprocessing to all datasets
        preprocessed_matrices = []
        for eigenvalue_matrix, _, _ in loaded_data:
            prep_matrix = self._preprocess_eigenvalues(eigenvalue_matrix)
            preprocessed_matrices.append(prep_matrix)
        
        # Apply global normalization scale if specified (consistent with pairwise method)
        if self.normalize is not None:
            all_vals = np.concatenate([P.ravel() for P in preprocessed_matrices])
            all_vals = all_vals[np.isfinite(all_vals)]
            
            if all_vals.size == 0:
                logger.warning("No finite values for endpoint normalization")
                scale = 1.0
            else:
                if self.normalize == 'p95':
                    scale = float(np.percentile(all_vals, 95.0))
                elif self.normalize == 'p99':
                    scale = float(np.percentile(all_vals, 99.0))
                else:  # 'max'
                    scale = float(np.max(all_vals))
                
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
            
            logger.debug(f"Endpoint normalization scale: {scale:.6e}")
            normalized_matrices = [P / scale for P in preprocessed_matrices]
        else:
            normalized_matrices = preprocessed_matrices
        
        # Extract final eigenvalues from preprocessed data
        final_eigenvalues = []
        for norm_matrix in normalized_matrices:
            final_eigs = norm_matrix[-1, :]
            final_eigs = final_eigs[np.isfinite(final_eigs)]  # Remove NaN
            final_eigenvalues.append(final_eigs)
        
        # Compute pairwise Wasserstein distances
        N = len(final_eigenvalues)
        distance_matrix = np.zeros((N, N))
        
        for i in range(N):
            distance_matrix[i, i] = 0.0
            
            for j in range(i + 1, N):
                try:
                    distance = self._wasserstein_1d(final_eigenvalues[i], final_eigenvalues[j])
                    distance_matrix[i, j] = distance_matrix[j, i] = distance
                    
                except Exception as e:
                    logger.warning(f"Failed to compute endpoint distance between {names[i]} and {names[j]}: {e}")
                    distance_matrix[i, j] = distance_matrix[j, i] = np.nan
        
        logger.info(f"Completed endpoint Wasserstein matrix computation with consistent preprocessing")
        return distance_matrix, names


def create_isw_comparator(**kwargs) -> EigenvalueISW:
    """Factory function to create an EigenvalueISW instance.
    
    Args:
        **kwargs: Arguments passed to EigenvalueISW constructor
        
    Returns:
        Configured EigenvalueISW instance
    """
    return EigenvalueISW(**kwargs)


def quick_isw_comparison(filepath1: Union[str, Path], 
                        filepath2: Union[str, Path],
                        **kwargs) -> float:
    """Quick ISW distance computation between two eigenvalue evolution files.
    
    Args:
        filepath1: First eigenvalue evolution file
        filepath2: Second eigenvalue evolution file
        **kwargs: Arguments passed to EigenvalueISW constructor
        
    Returns:
        ISW distance
        
    Note:
        Distances from this function are not guaranteed to be on the same scale 
        as those from pairwise_isw_matrix unless you pass the same common_time 
        and scale parameters. For comparable distances across multiple files, 
        use pairwise_isw_matrix instead.
    """
    isw = create_isw_comparator(**kwargs)
    
    eig1, t1, _ = isw.load_eigenvalue_data(filepath1)
    eig2, t2, _ = isw.load_eigenvalue_data(filepath2)
    
    return isw.isw_distance(eig1, t1, eig2, t2)
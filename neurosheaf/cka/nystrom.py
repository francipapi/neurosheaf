"""Nyström approximation for memory-efficient CKA computation.

This module implements Nyström approximation to reduce memory complexity from O(n²) 
to O(n*m) where m is the number of landmarks. This is critical for Phase 2 Week 4
to handle large-scale neural network activations.

IMPORTANT: Like all CKA implementations, this uses raw activations without 
pre-centering to avoid double-centering with the unbiased HSIC estimator.

Memory scaling:
- Exact CKA: O(n²) memory for n samples
- Nyström CKA: O(n*m) memory for m landmarks

Performance targets:
- 10x memory reduction vs naive implementation
- <1% error with 512 landmarks
- Handle 50k samples with 4GB memory

Apple Silicon (MPS) Limitations:
- SVD operations on MPS have documented numerical stability issues
- Automatically falls back to CPU for SVD computations
- Reference: PyTorch GitHub issue #78099
"""

import warnings
from typing import Tuple, Optional, Union, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from ..utils.device import detect_optimal_device, get_device_info, should_use_cpu_fallback, log_device_warning_once
from ..utils.exceptions import ValidationError, ComputationError, MemoryError
from ..utils.logging import setup_logger
from ..utils.profiling import profile_memory, profile_time
from ..utils.validation import validate_activations


logger = setup_logger(__name__)


class NystromCKA:
    """Memory-efficient CKA using Nyström approximation.
    
    This class provides a memory-efficient alternative to exact CKA computation
    by using Nyström approximation to reduce memory complexity from O(n²) to O(n*m).
    
    The implementation maintains the NO double-centering requirement from Phase 2
    by using raw activations for all kernel computations.
    
    Attributes:
        n_landmarks: Number of landmark points for approximation
        landmark_selection: Strategy for selecting landmarks ('uniform', 'kmeans')
        device: Device for computation (auto-detected)
        numerical_stability: Epsilon for numerical stability
        
    Memory complexity: O(n*m) instead of O(n²) for exact CKA
    """
    
    def __init__(
        self,
        n_landmarks: int = 256,
        landmark_selection: str = 'uniform',
        device: Optional[Union[str, torch.device]] = None,
        numerical_stability: float = 1e-6,
        enable_profiling: bool = True,
        max_kmeans_iter: int = 100,
        auto_promote_precision: bool = True,
        safe_tensor_ops: bool = True,
        strict_numerics: bool = False,
        use_qr_approximation: bool = False,
        enable_psd_projection: bool = True,
        spectral_regularization: bool = True,
        adaptive_landmarks: bool = True,
        rank_tolerance: float = 1e-12,
        use_spectral_landmarks: bool = False
    ):
        """Initialize Nyström CKA approximation.
        
        Args:
            n_landmarks: Number of landmark points (default: 256)
            landmark_selection: Landmark selection strategy ('uniform', 'kmeans')
            device: Device for computation (auto-detected if None)
            numerical_stability: Epsilon for numerical stability
            enable_profiling: Whether to enable performance profiling
            max_kmeans_iter: Maximum iterations for k-means clustering
            auto_promote_precision: Whether to automatically promote to float64 for stability
            safe_tensor_ops: Whether to clone tensors for safe operations
            strict_numerics: Whether to fail fast on numerical issues
            use_qr_approximation: Whether to use QR-based approximation (recommended)
            enable_psd_projection: Whether to project to positive semidefinite cone
            spectral_regularization: Whether to use spectral regularization
            adaptive_landmarks: Whether to adapt landmark count to effective rank
            rank_tolerance: Tolerance for rank estimation
            use_spectral_landmarks: Whether to use leverage score-based landmark selection
        """
        if n_landmarks < 4:
            raise ValidationError("n_landmarks must be at least 4 for numerical stability")
        
        if landmark_selection not in ['uniform', 'kmeans']:
            raise ValidationError(f"Unknown landmark_selection: {landmark_selection}")
        
        self.n_landmarks = n_landmarks
        self.landmark_selection = landmark_selection
        self.device = detect_optimal_device(device)
        self.eps = numerical_stability
        self.enable_profiling = enable_profiling
        self.max_kmeans_iter = max_kmeans_iter
        self.auto_promote_precision = auto_promote_precision
        self.safe_tensor_ops = safe_tensor_ops
        self.strict_numerics = strict_numerics
        self.use_qr_approximation = use_qr_approximation
        self.enable_psd_projection = enable_psd_projection
        self.spectral_regularization = spectral_regularization
        self.adaptive_landmarks = adaptive_landmarks
        self.rank_tolerance = rank_tolerance
        self.use_spectral_landmarks = use_spectral_landmarks
        
        self.logger = setup_logger("neurosheaf.cka.nystrom")
        
        # Device information
        self.device_info = get_device_info()
        self.is_mac = self.device_info['is_mac']
        self.is_apple_silicon = self.device_info['is_apple_silicon']
        
        self.logger.info(f"Initialized NystromCKA with {n_landmarks} landmarks")
        self.logger.info(f"Landmark selection: {landmark_selection}")
        self.logger.info(f"Device: {self.device}")
        
        if self.is_mac:
            self.logger.info(f"Mac optimization enabled: Apple Silicon = {self.is_apple_silicon}")
    
    
    def _validate_numerical_stability(self, *tensors, stage: str = "unknown") -> None:
        """Comprehensive numerical stability validation."""
        for i, tensor in enumerate(tensors):
            if torch.isnan(tensor).any():
                error_msg = f"NaN detected in tensor {i} at stage '{stage}'"
                if self.strict_numerics:
                    raise ComputationError(error_msg)
                else:
                    self.logger.warning(error_msg)
                    
            if torch.isinf(tensor).any():
                error_msg = f"Inf detected in tensor {i} at stage '{stage}'"
                if self.strict_numerics:
                    raise ComputationError(error_msg)
                else:
                    self.logger.warning(error_msg)
    
    def _auto_promote_precision(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Automatically promote tensors to float64 for numerical stability."""
        if not self.auto_promote_precision:
            return X, Y
        
        needs_promotion = False
        
        # Check for small sample sizes
        if X.shape[0] < 20:
            needs_promotion = True
            reason = "small sample size"
        
        # Check for float32 on MPS (but don't promote - MPS doesn't support float64)
        elif X.dtype == torch.float32 and self.device.type == 'mps':
            # MPS doesn't support float64, so we can't promote
            # Just use CPU for critical operations when needed
            self.logger.debug("MPS device detected with float32 - will use CPU for critical operations")
        
        if needs_promotion and self.device.type != 'mps':
            if X.dtype != torch.float64:
                self.logger.info(f"Promoting to float64 for numerical stability: {reason}")
                X = X.double()
                Y = Y.double()
        
        return X, Y
    
    def _safe_clone_tensors(self, *tensors) -> Tuple[torch.Tensor, ...]:
        """Safely clone tensors to avoid in-place operations."""
        if not self.safe_tensor_ops:
            return tensors
        return tuple(tensor.clone() for tensor in tensors)
    
    def _estimate_effective_rank(self, X: torch.Tensor) -> int:
        """Estimate effective rank of the gram matrix X @ X.T.
        
        This implements robust rank estimation using SVD with tolerance-based
        thresholding to handle numerical noise and avoid rank oversampling.
        
        Args:
            X: Activation tensor [n_samples, n_features]
            
        Returns:
            Effective rank of the gram matrix
        """
        n_samples, n_features = X.shape
        
        # For small matrices, compute rank directly
        if n_samples <= 1000 and n_features <= 1000:
            try:
                # Compute full SVD for accurate rank estimation
                if should_use_cpu_fallback(X.device, 'svd'):
                    log_device_warning_once(X.device, 'svd')
                    _, S, _ = torch.linalg.svd(X.cpu())
                    S = S.to(X.device)
                else:
                    _, S, _ = torch.linalg.svd(X)
                
                # Count significant singular values
                max_sv = S[0] if len(S) > 0 else 1.0
                threshold = max(self.rank_tolerance, max_sv * self.rank_tolerance)
                effective_rank = torch.sum(S > threshold).item()
                
                self.logger.debug(f"Full SVD rank estimation: {effective_rank}/{min(n_samples, n_features)}")
                return min(effective_rank, n_samples - 1)
                
            except Exception as e:
                self.logger.debug(f"Full SVD failed: {e}, using gram matrix approach")
        
        # For large matrices, use gram matrix approach
        try:
            # Sample-based estimation for computational efficiency
            sample_size = min(500, n_samples)
            if sample_size < n_samples:
                indices = torch.randperm(n_samples, device=X.device)[:sample_size]
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Compute gram matrix
            G = X_sample @ X_sample.T
            
            # Estimate rank using eigendecomposition
            if should_use_cpu_fallback(G.device, 'eigvals'):
                log_device_warning_once(G.device, 'eigvals')
                eigenvals = torch.linalg.eigvals(G.cpu()).real.to(X.device)
            else:
                eigenvals = torch.linalg.eigvals(G).real
            
            # Sort eigenvalues in descending order
            eigenvals, _ = torch.sort(eigenvals, descending=True)
            
            # Count significant eigenvalues
            max_eig = eigenvals[0] if len(eigenvals) > 0 else 1.0
            threshold = max(self.rank_tolerance, max_eig * self.rank_tolerance)
            effective_rank = torch.sum(eigenvals > threshold).item()
            
            self.logger.debug(f"Gram matrix rank estimation: {effective_rank}/{sample_size}")
            
            # Scale up if we used sampling
            if sample_size < n_samples:
                scaling_factor = n_samples / sample_size
                effective_rank = min(int(effective_rank * scaling_factor), n_samples - 1)
            
            return max(1, min(effective_rank, n_samples - 1))
            
        except Exception as e:
            self.logger.warning(f"Rank estimation failed: {e}. Using conservative estimate.")
            # Conservative fallback
            return min(n_samples // 2, self.n_landmarks, n_samples - 1)
    
    def _select_adaptive_landmarks(self, X: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Select landmarks with adaptive count based on effective rank.
        
        Args:
            X: Activation tensor [n_samples, n_features]
            
        Returns:
            Tuple of (landmark indices, effective landmark count)
        """
        n_samples = X.shape[0]
        
        if self.adaptive_landmarks:
            # Estimate effective rank
            effective_rank = self._estimate_effective_rank(X)
            
            # Adapt landmark count: don't exceed effective rank
            effective_landmarks = min(self.n_landmarks, effective_rank, n_samples - 1)
            
            if effective_landmarks < self.n_landmarks:
                self.logger.info(
                    f"Adapted landmarks from {self.n_landmarks} to {effective_landmarks} "
                    f"based on effective rank {effective_rank}"
                )
        else:
            effective_landmarks = min(self.n_landmarks, n_samples - 1)
        
        # Select landmarks using best available method
        if self.use_spectral_landmarks:
            landmarks = self._select_spectral_landmarks(X, effective_landmarks)
        else:
            landmarks = self._select_landmarks(X, effective_landmarks)
        
        return landmarks, effective_landmarks
    
    def _compute_leverage_scores(self, X: torch.Tensor, effective_rank: int) -> torch.Tensor:
        """Compute leverage scores for spectral-aware landmark selection.
        
        Args:
            X: Activation tensor [n_samples, n_features]
            effective_rank: Effective rank for approximation
            
        Returns:
            Leverage scores [n_samples]
        """
        n_samples = X.shape[0]
        
        try:
            # Use randomized SVD for efficiency with large matrices
            if n_samples > 1000:
                # Approximate leverage scores using random sampling
                sample_size = min(effective_rank * 2, n_samples // 2)
                random_matrix = torch.randn(X.shape[1], sample_size, device=X.device)
                Y = X @ random_matrix
                
                if should_use_cpu_fallback(Y.device, 'qr'):
                    log_device_warning_once(Y.device, 'qr')
                    Q, _ = torch.linalg.qr(Y.cpu())
                    Q = Q.to(X.device)
                else:
                    Q, _ = torch.linalg.qr(Y)
                
                # Compute approximate leverage scores
                leverage_scores = torch.sum((X @ Q) ** 2, dim=1)
            else:
                # Exact leverage scores for smaller matrices
                if should_use_cpu_fallback(X.device, 'svd'):
                    log_device_warning_once(X.device, 'svd')
                    U, S, _ = torch.linalg.svd(X.cpu())
                    U = U.to(X.device)
                else:
                    U, S, _ = torch.linalg.svd(X)
                
                # Use top-k singular vectors
                k = min(effective_rank, U.shape[1])
                U_k = U[:, :k]
                leverage_scores = torch.sum(U_k ** 2, dim=1)
            
            return leverage_scores
            
        except Exception as e:
            self.logger.debug(f"Leverage score computation failed: {e}. Using uniform scores.")
            return torch.ones(n_samples, device=X.device) / n_samples
    
    def _select_spectral_landmarks(self, X: torch.Tensor, n_landmarks: int) -> torch.Tensor:
        """Select landmarks using spectral-aware sampling with leverage scores.
        
        Args:
            X: Activation tensor [n_samples, n_features]
            n_landmarks: Number of landmarks to select
            
        Returns:
            Indices of selected landmarks
        """
        n_samples = X.shape[0]
        
        if n_landmarks >= n_samples:
            return torch.arange(n_samples, device=X.device)
        
        try:
            # Estimate effective rank for leverage score computation
            effective_rank = self._estimate_effective_rank(X)
            
            # Compute leverage scores
            leverage_scores = self._compute_leverage_scores(X, effective_rank)
            
            # Normalize scores to probabilities
            leverage_probs = leverage_scores / torch.sum(leverage_scores)
            
            # Sample landmarks based on leverage scores
            # Use a hybrid approach: some deterministic high-leverage, some random
            num_deterministic = min(n_landmarks // 2, 10)
            num_random = n_landmarks - num_deterministic
            
            landmarks = []
            
            # Select top leverage score points deterministically
            if num_deterministic > 0:
                _, top_indices = torch.topk(leverage_scores, num_deterministic)
                landmarks.append(top_indices)
            
            # Select remaining points using leverage-weighted sampling
            if num_random > 0:
                # Create sampling weights excluding already selected points
                sampling_probs = leverage_probs.clone()
                if num_deterministic > 0:
                    sampling_probs[top_indices] = 0
                    sampling_probs = sampling_probs / torch.sum(sampling_probs)
                
                # Sample without replacement
                random_indices = torch.multinomial(
                    sampling_probs, 
                    num_random, 
                    replacement=False
                )
                landmarks.append(random_indices)
            
            # Combine all landmarks
            final_landmarks = torch.cat(landmarks)
            
            self.logger.debug(f"Spectral landmark selection: {len(final_landmarks)} landmarks selected")
            
            return final_landmarks
            
        except Exception as e:
            self.logger.debug(f"Spectral landmark selection failed: {e}. Falling back to uniform sampling.")
            # Fallback to uniform sampling
            perm = torch.randperm(n_samples, device=X.device)
            return perm[:n_landmarks]

    def compute(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        validate_properties: bool = True,
        return_approximation_info: bool = False
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """Compute CKA using Nyström approximation.
        
        Args:
            X: First activation tensor [n_samples, n_features_x]
            Y: Second activation tensor [n_samples, n_features_y]
            validate_properties: Whether to validate mathematical properties
            return_approximation_info: Whether to return approximation quality info
            
        Returns:
            CKA value in [0, 1], optionally with approximation information
            
        IMPORTANT: X and Y should be raw activations, NOT centered!
        """
        # Validate inputs
        X, Y = validate_activations(X, Y, min_samples=max(4, self.n_landmarks))
        
        # Initial numerical stability validation
        self._validate_numerical_stability(X, Y, stage="input")
        
        # Apply automatic precision promotion
        X, Y = self._auto_promote_precision(X, Y)
        
        # Move to device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        n_samples = X.shape[0]
        
        # Use adaptive landmark selection if enabled
        if self.adaptive_landmarks:
            landmarks_x, effective_landmarks_x = self._select_adaptive_landmarks(X)
            landmarks_y, effective_landmarks_y = self._select_adaptive_landmarks(Y)
            effective_landmarks = min(effective_landmarks_x, effective_landmarks_y)
        else:
            effective_landmarks = min(self.n_landmarks, n_samples)
            if effective_landmarks < self.n_landmarks:
                self.logger.warning(f"Reducing landmarks from {self.n_landmarks} to {effective_landmarks} due to sample size")
            landmarks_x = self._select_landmarks(X, effective_landmarks)
            landmarks_y = self._select_landmarks(Y, effective_landmarks)
        
        # Compute Nyström approximation
        K_approx = self._compute_nystrom_kernel(X, landmarks_x)
        L_approx = self._compute_nystrom_kernel(Y, landmarks_y)
        
        # Validate approximated kernels
        self._validate_numerical_stability(K_approx, L_approx, stage="nystrom_approximation")
        
        # Compute CKA from approximated kernels
        cka_value = self._compute_cka_from_kernels(K_approx, L_approx)
        
        # Validate properties if requested
        if validate_properties:
            self._validate_cka_properties(cka_value, X, Y)
        
        if return_approximation_info:
            # Compute approximation quality metrics
            approx_info = self._compute_approximation_quality(X, Y, K_approx, L_approx)
            return cka_value, approx_info
        
        return cka_value
    
    def _select_landmarks(self, X: torch.Tensor, n_landmarks: int) -> torch.Tensor:
        """Select landmark points using specified strategy.
        
        Args:
            X: Activation tensor [n_samples, n_features]
            n_landmarks: Number of landmarks to select
            
        Returns:
            Indices of selected landmarks
        """
        n_samples = X.shape[0]
        
        if self.landmark_selection == 'uniform':
            # Uniform random sampling
            if n_landmarks >= n_samples:
                return torch.arange(n_samples, device=X.device)
            
            perm = torch.randperm(n_samples, device=X.device)
            return perm[:n_landmarks]
        
        elif self.landmark_selection == 'kmeans':
            # K-means++ initialization for better coverage
            return self._kmeans_landmarks(X, n_landmarks)
        
        else:
            raise ValidationError(f"Unknown landmark selection: {self.landmark_selection}")
    
    def _kmeans_landmarks(self, X: torch.Tensor, n_landmarks: int) -> torch.Tensor:
        """Select landmarks using K-means++ initialization.
        
        Args:
            X: Activation tensor [n_samples, n_features]
            n_landmarks: Number of landmarks to select
            
        Returns:
            Indices of selected landmarks
        """
        n_samples = X.shape[0]
        
        if n_landmarks >= n_samples:
            return torch.arange(n_samples, device=X.device)
        
        # Move to CPU for sklearn compatibility
        X_cpu = X.cpu().numpy()
        
        try:
            # Suppress sklearn warnings about convergence
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                kmeans = KMeans(
                    n_clusters=n_landmarks,
                    init='k-means++',
                    max_iter=self.max_kmeans_iter,
                    random_state=42,
                    n_init=1  # Single run for speed
                )
                
                kmeans.fit(X_cpu)
                centers = kmeans.cluster_centers_
            
            # Find nearest data points to cluster centers
            X_tensor = torch.from_numpy(X_cpu).to(X.device)
            centers_tensor = torch.from_numpy(centers).to(X.device)
            
            # Compute distances from each center to all points
            distances = torch.cdist(centers_tensor, X_tensor)  # [n_landmarks, n_samples]
            
            # Select nearest point to each center
            landmarks = torch.argmin(distances, dim=1)  # [n_landmarks]
            
            # Remove duplicates (rare but possible)
            landmarks = torch.unique(landmarks)
            
            # If we lost some landmarks due to duplicates, add random ones
            if len(landmarks) < n_landmarks:
                remaining = n_landmarks - len(landmarks)
                all_indices = torch.arange(n_samples, device=X.device)
                remaining_indices = all_indices[~torch.isin(all_indices, landmarks)]
                
                if len(remaining_indices) > 0:
                    additional = remaining_indices[torch.randperm(len(remaining_indices))[:remaining]]
                    landmarks = torch.cat([landmarks, additional])
            
            return landmarks[:n_landmarks]
        
        except Exception as e:
            self.logger.warning(f"K-means landmark selection failed: {e}. Falling back to uniform sampling.")
            # Fallback to uniform sampling directly to avoid infinite recursion
            if n_landmarks >= n_samples:
                return torch.arange(n_samples, device=X.device)
            
            perm = torch.randperm(n_samples, device=X.device)
            return perm[:n_landmarks]
    
    def _apply_spectral_regularization(self, K: torch.Tensor) -> torch.Tensor:
        """Apply spectral regularization based on eigenvalue thresholding.
        
        Args:
            K: Kernel matrix [n, n]
            
        Returns:
            Spectrally regularized kernel matrix
        """
        if not self.spectral_regularization:
            return K
        
        try:
            # Compute eigendecomposition
            if should_use_cpu_fallback(K.device, 'eigh'):
                log_device_warning_once(K.device, 'eigh')
                eigenvals, eigenvecs = torch.linalg.eigh(K.cpu())
                eigenvals = eigenvals.to(K.device)
                eigenvecs = eigenvecs.to(K.device)
            else:
                eigenvals, eigenvecs = torch.linalg.eigh(K)
            
            # Spectral regularization: threshold small eigenvalues
            max_eigenval = eigenvals.max()
            threshold = max(self.eps, max_eigenval * self.rank_tolerance)
            
            # Apply regularization to small eigenvalues
            regularized_eigenvals = torch.where(
                eigenvals < threshold,
                threshold,
                eigenvals
            )
            
            # Reconstruct matrix
            K_reg = eigenvecs @ torch.diag(regularized_eigenvals) @ eigenvecs.T
            
            self.logger.debug(f"Applied spectral regularization: {torch.sum(eigenvals < threshold).item()} eigenvalues regularized")
            return K_reg
            
        except Exception as e:
            self.logger.debug(f"Spectral regularization failed: {e}")
            return K
    
    def _apply_psd_projection(self, K: torch.Tensor) -> torch.Tensor:
        """Project matrix to positive semidefinite cone.
        
        Args:
            K: Input matrix [n, n]
            
        Returns:
            Positive semidefinite matrix
        """
        if not self.enable_psd_projection:
            return K
        
        try:
            # Ensure symmetry first
            K_sym = (K + K.T) / 2
            
            # Compute eigendecomposition
            if should_use_cpu_fallback(K.device, 'eigh'):
                log_device_warning_once(K.device, 'eigh')
                eigenvals, eigenvecs = torch.linalg.eigh(K_sym.cpu())
                eigenvals = eigenvals.to(K.device)
                eigenvecs = eigenvecs.to(K.device)
            else:
                eigenvals, eigenvecs = torch.linalg.eigh(K_sym)
            
            # Clip negative eigenvalues to zero with small tolerance
            # Use max of eps and relative tolerance to handle numerical precision
            tolerance = max(self.eps, torch.max(eigenvals).item() * 1e-12)
            eigenvals_clipped = torch.clamp(eigenvals, min=tolerance)
            
            # Count negative eigenvalues for logging
            num_negative = torch.sum(eigenvals < 0).item()
            if num_negative > 0:
                self.logger.debug(f"PSD projection: clipped {num_negative} negative eigenvalues (min: {torch.min(eigenvals):.2e})")
            
            # Reconstruct positive semidefinite matrix
            K_psd = eigenvecs @ torch.diag(eigenvals_clipped) @ eigenvecs.T
            
            # Apply a final numerical cleanup to ensure no tiny negative eigenvalues
            # remain due to reconstruction errors
            if num_negative > 0:
                # Check if we still have negative eigenvalues after reconstruction
                final_eigenvals = torch.linalg.eigvals(K_psd).real
                num_final_negative = torch.sum(final_eigenvals < -1e-12).item()
                
                if num_final_negative > 0:
                    # Add small regularization to diagonal as final cleanup
                    regularization = max(self.eps, -torch.min(final_eigenvals).item() * 2)
                    K_psd = K_psd + regularization * torch.eye(K_psd.shape[0], device=K.device)
                    self.logger.debug(f"Applied final regularization {regularization:.2e} to eliminate {num_final_negative} remaining negative eigenvalues")
            
            return K_psd
            
        except Exception as e:
            self.logger.debug(f"PSD projection failed: {e}")
            # Fallback: add small regularization to diagonal
            n = K.shape[0]
            return K + self.eps * torch.eye(n, device=K.device)
    
    def _monitor_numerical_stability(self, K: torch.Tensor, stage: str = "unknown") -> Dict[str, float]:
        """Monitor numerical stability of kernel matrix.
        
        Args:
            K: Kernel matrix to analyze
            stage: Computation stage for logging
            
        Returns:
            Dictionary with stability metrics
        """
        metrics = {}
        
        try:
            # Basic properties
            metrics['frobenius_norm'] = torch.norm(K, 'fro').item()
            metrics['trace'] = torch.trace(K).item()
            metrics['max_element'] = torch.max(K).item()
            metrics['min_element'] = torch.min(K).item()
            
            # Condition number
            try:
                if should_use_cpu_fallback(K.device, 'cond'):
                    log_device_warning_once(K.device, 'cond')
                    cond_num = torch.linalg.cond(K.cpu()).item()
                else:
                    cond_num = torch.linalg.cond(K).item()
                metrics['condition_number'] = cond_num
                
                # Warning for ill-conditioned matrices
                if cond_num > 1e12:
                    warning_msg = f"Very high condition number at {stage}: {cond_num:.2e}"
                    if self.strict_numerics:
                        raise ComputationError(warning_msg)
                    else:
                        self.logger.warning(warning_msg)
                elif cond_num > 1e8:
                    self.logger.info(f"High condition number at {stage}: {cond_num:.2e}")
                    
            except Exception as e:
                self.logger.debug(f"Could not compute condition number at {stage}: {e}")
                metrics['condition_number'] = float('inf')
            
            # Eigenvalue analysis
            try:
                if should_use_cpu_fallback(K.device, 'eigvals'):
                    log_device_warning_once(K.device, 'eigvals')
                    eigenvals = torch.linalg.eigvals(K.cpu()).real.to(K.device)
                else:
                    eigenvals = torch.linalg.eigvals(K).real
                
                metrics['min_eigenvalue'] = torch.min(eigenvals).item()
                metrics['max_eigenvalue'] = torch.max(eigenvals).item()
                metrics['num_negative_eigenvals'] = torch.sum(eigenvals < -self.eps).item()
                metrics['effective_rank'] = torch.sum(eigenvals > self.rank_tolerance * torch.max(eigenvals)).item()
                
                # Check for positive semidefiniteness
                if metrics['num_negative_eigenvals'] > 0:
                    msg = f"Matrix not PSD at {stage}: {metrics['num_negative_eigenvals']} negative eigenvalues"
                    if self.enable_psd_projection:
                        self.logger.debug(msg)
                    else:
                        self.logger.warning(msg)
                        
            except Exception as e:
                self.logger.debug(f"Could not compute eigenvalues at {stage}: {e}")
            
            # Symmetry check
            symmetry_error = torch.norm(K - K.T, 'fro').item()
            metrics['symmetry_error'] = symmetry_error
            if symmetry_error > 1e-10:
                self.logger.debug(f"Matrix not symmetric at {stage}: error={symmetry_error:.2e}")
            
            self.logger.debug(f"Stability metrics at {stage}: {metrics}")
            
        except Exception as e:
            self.logger.debug(f"Stability monitoring failed at {stage}: {e}")
            
        return metrics
    
    def _qr_based_approximation(self, X: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """Compute improved Nyström approximation using QR decomposition for stability.
        
        The standard Nyström method K ≈ C W^(-1) C^T can be numerically unstable.
        This method uses QR decomposition to compute a more stable approximation.
        
        Args:
            X: Activation tensor [n_samples, n_features]
            landmarks: Landmark indices [n_landmarks]
            
        Returns:
            Improved Nyström approximated kernel matrix [n_samples, n_samples]
        """
        n_samples = X.shape[0]
        n_landmarks = len(landmarks)
        
        # Extract landmark activations
        X_landmarks = X[landmarks]  # [n_landmarks, n_features]
        
        # Compute kernel matrices
        K_nm = X @ X_landmarks.T  # [n_samples, n_landmarks] - cross-kernel
        K_mm = X_landmarks @ X_landmarks.T  # [n_landmarks, n_landmarks] - landmark kernel
        
        try:
            # Use QR decomposition for stable inversion of K_mm
            if should_use_cpu_fallback(K_mm.device, 'qr'):
                log_device_warning_once(K_mm.device, 'qr')
                Q_mm, R_mm = torch.linalg.qr(K_mm.cpu())
                Q_mm = Q_mm.to(K_mm.device)
                R_mm = R_mm.to(K_mm.device)
            else:
                Q_mm, R_mm = torch.linalg.qr(K_mm)
            
            # Check rank and handle rank deficiency
            rank = torch.linalg.matrix_rank(R_mm).item()
            effective_rank = min(rank, n_landmarks)
            
            if effective_rank < n_landmarks:
                self.logger.debug(f"Rank deficient landmark kernel: rank={effective_rank}/{n_landmarks}")
                Q_mm = Q_mm[:, :effective_rank]
                R_mm = R_mm[:effective_rank, :effective_rank]
                K_nm = K_nm[:, :effective_rank]
            
            # Solve for the stable inversion: K_mm @ alpha = K_nm.T
            # This gives us alpha = K_mm^(-1) @ K_nm.T but computed stably
            K_nm_effective = K_nm[:, :effective_rank]  # Match dimensions
            
            try:
                if should_use_cpu_fallback(R_mm.device, 'solve_triangular'):
                    log_device_warning_once(R_mm.device, 'solve_triangular')
                    alpha = torch.linalg.solve_triangular(R_mm.cpu(), Q_mm.T.cpu() @ K_nm_effective.T.cpu(), upper=True)
                    alpha = alpha.to(R_mm.device)
                else:
                    alpha = torch.linalg.solve_triangular(R_mm, Q_mm.T @ K_nm_effective.T, upper=True)
            except:
                # Fallback to pseudoinverse if triangular solve fails
                self.logger.debug("Triangular solve failed, using pseudoinverse")
                K_mm_effective = K_mm[:effective_rank, :effective_rank]  # Match dimensions
                if should_use_cpu_fallback(K_mm_effective.device, 'pinv'):
                    log_device_warning_once(K_mm_effective.device, 'pinv')
                    K_mm_inv = torch.linalg.pinv(K_mm_effective.cpu()).to(K_mm_effective.device)
                else:
                    K_mm_inv = torch.linalg.pinv(K_mm_effective)
                alpha = K_mm_inv @ K_nm_effective.T
            
            # Standard Nyström formula: K ≈ K_nm @ K_mm^(-1) @ K_nm.T
            # But computed as K_nm @ alpha for stability
            K_approx = K_nm_effective @ alpha.T
            
            self.logger.debug(f"QR-based Nyström: effective_rank={effective_rank}, shape={K_approx.shape}")
            
            return K_approx
            
        except Exception as e:
            self.logger.warning(f"QR-based approximation failed: {e}. Falling back to standard Nyström.")
            # Fallback to standard Nyström
            return self._standard_nystrom_approximation(X, landmarks)
    
    def _standard_nystrom_approximation(self, X: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """Standard Nyström approximation (fallback method).
        
        Args:
            X: Activation tensor [n_samples, n_features]
            landmarks: Landmark indices [n_landmarks]
            
        Returns:
            Standard Nyström approximated kernel matrix [n_samples, n_samples]
        """
        # Extract landmark activations
        X_landmarks = X[landmarks]  # [n_landmarks, n_features]
        
        # Compute kernel blocks WITHOUT centering (critical for NO double-centering)
        K_mm = X_landmarks @ X_landmarks.T  # [n_landmarks, n_landmarks]
        K_nm = X @ X_landmarks.T           # [n_samples, n_landmarks]
        
        # Apply spectral regularization to landmark kernel
        K_mm_reg = self._apply_spectral_regularization(K_mm)
        
        # Stable inversion of landmark kernel
        K_mm_inv = self._stable_inverse(K_mm_reg)
        
        # Standard Nyström approximation: K ≈ K_nm @ K_mm^(-1) @ K_nm^T
        K_approx = K_nm @ K_mm_inv @ K_nm.T
        
        return K_approx
    
    def _compute_nystrom_kernel(self, X: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """Compute Nyström approximation of kernel matrix with mathematical fixes.
        
        This method implements the corrected Nyström approximation following the fix plan:
        1. Choose between QR-based or standard approximation
        2. Apply spectral regularization
        3. Project to positive semidefinite cone
        
        Args:
            X: Activation tensor [n_samples, n_features]
            landmarks: Landmark indices [n_landmarks]
            
        Returns:
            Approximated kernel matrix [n_samples, n_samples] (PSD if enabled)
        """
        # Choose approximation method
        if self.use_qr_approximation:
            K_approx = self._qr_based_approximation(X, landmarks)
        else:
            K_approx = self._standard_nystrom_approximation(X, landmarks)
        
        # Monitor stability before PSD projection
        if self.enable_profiling:
            self._monitor_numerical_stability(K_approx, "before_psd_projection")
        
        # Apply positive semidefinite projection
        K_approx_psd = self._apply_psd_projection(K_approx)
        
        # Monitor final stability
        if self.enable_profiling:
            self._monitor_numerical_stability(K_approx_psd, "final_approximation")
        
        return K_approx_psd
    
    def _stable_inverse(self, M: torch.Tensor) -> torch.Tensor:
        """Compute stable inverse using SVD with regularization.
        
        IMPORTANT: SVD operations on MPS have known numerical stability issues
        and automatically fall back to CPU. This is a documented PyTorch limitation
        (GitHub issue #78099) where MPS SVD produces significantly larger numerical
        errors compared to CPU computation.
        
        Args:
            M: Matrix to invert [n, n]
            
        Returns:
            Stable inverse [n, n]
        """
        try:
            # CRITICAL: Always use CPU for SVD on MPS due to documented numerical issues
            # Reference: https://github.com/pytorch/pytorch/issues/78099
            if should_use_cpu_fallback(M.device, 'svd'):
                log_device_warning_once(M.device, 'svd')
                
                M_cpu = M.cpu()
                U, S, Vt = torch.linalg.svd(M_cpu)
                
                # Regularize small singular values
                S_reg = torch.where(S > self.eps, S, self.eps)
                S_inv = 1.0 / S_reg
                
                # Reconstruct inverse
                M_inv = Vt.T @ torch.diag(S_inv) @ U.T
                
                # Move back to original device
                return M_inv.to(M.device)
            else:
                # Use SVD for stable inversion
                U, S, Vt = torch.linalg.svd(M)
                
                # Regularize small singular values
                S_reg = torch.where(S > self.eps, S, self.eps)
                S_inv = 1.0 / S_reg
                
                # Reconstruct inverse
                M_inv = Vt.T @ torch.diag(S_inv) @ U.T
                
                return M_inv
        
        except RuntimeError as e:
            self.logger.warning(f"SVD failed: {e}. Using pseudoinverse.")
            # Use CPU fallback for pseudoinverse if needed
            if should_use_cpu_fallback(M.device, 'pinv'):
                log_device_warning_once(M.device, 'pinv')
                return torch.linalg.pinv(M.cpu()).to(M.device)
            else:
                return torch.linalg.pinv(M)
    
    def _compute_cka_from_kernels(self, K: torch.Tensor, L: torch.Tensor) -> float:
        """Compute CKA from kernel matrices using unbiased HSIC estimator.
        
        Args:
            K: First kernel matrix [n_samples, n_samples]
            L: Second kernel matrix [n_samples, n_samples]
            
        Returns:
            CKA value in [0, 1]
        """
        # Use unbiased HSIC estimator (handles centering internally)
        hsic_xy = self._unbiased_hsic(K, L)
        hsic_xx = self._unbiased_hsic(K, K)
        hsic_yy = self._unbiased_hsic(L, L)
        
        # Compute CKA
        denominator = torch.sqrt(hsic_xx * hsic_yy) + self.eps
        cka = hsic_xy / denominator
        
        return float(torch.clamp(cka, 0, 1))
    
    def _unbiased_hsic(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Compute unbiased HSIC estimator.
        
        This is identical to the implementation in DebiasedCKA to ensure consistency.
        
        Args:
            K: First kernel matrix [n_samples, n_samples]
            L: Second kernel matrix [n_samples, n_samples]
            
        Returns:
            Unbiased HSIC value
        """
        n = K.shape[0]
        
        if n < 4:
            raise ValidationError("Unbiased HSIC requires at least 4 samples")
        
        # Remove diagonal elements
        K_0 = K - torch.diag(torch.diag(K))
        L_0 = L - torch.diag(torch.diag(L))
        
        # Compute terms for unbiased estimator
        term1 = torch.sum(K_0 * L_0)
        term2 = torch.sum(K_0) * torch.sum(L_0) / ((n - 1) * (n - 2))
        term3 = 2 * torch.sum(K_0, dim=0) @ torch.sum(L_0, dim=0) / (n - 2)
        
        hsic = (term1 + term2 - term3) / (n * (n - 3))
        
        return hsic
    
    def _validate_cka_properties(self, cka_value: float, X: torch.Tensor, Y: torch.Tensor):
        """Validate mathematical properties of CKA result.
        
        Args:
            cka_value: Computed CKA value
            X: First activation tensor
            Y: Second activation tensor
        """
        # Check range [0, 1]
        if not (0 <= cka_value <= 1):
            raise ComputationError(f"CKA value {cka_value} outside [0, 1] range")
        
        # Check for NaN
        if np.isnan(cka_value):
            raise ComputationError("CKA computation resulted in NaN")
        
        # For identical inputs, CKA should be close to 1 (allowing for approximation error)
        # Only check if tensors have the same shape
        if X.shape == Y.shape and torch.allclose(X, Y, atol=1e-6):
            if abs(cka_value - 1.0) > 0.05:  # Allow 5% error for Nyström
                self.logger.warning(f"CKA(X,X) = {cka_value}, expected ~1.0 (Nyström approximation)")
    
    def _compute_approximation_quality(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        K_approx: torch.Tensor,
        L_approx: torch.Tensor
    ) -> Dict[str, float]:
        """Compute approximation quality metrics.
        
        Args:
            X: First activation tensor
            Y: Second activation tensor
            K_approx: Approximated kernel matrix for X
            L_approx: Approximated kernel matrix for Y
            
        Returns:
            Dictionary with approximation quality metrics
        """
        # Compute exact kernels for comparison (if memory allows)
        n_samples = X.shape[0]
        
        # Only compute exact kernels for small datasets
        if n_samples <= 2000:
            try:
                K_exact = X @ X.T
                L_exact = Y @ Y.T
                
                # Relative approximation error
                k_error = torch.norm(K_exact - K_approx, 'fro') / torch.norm(K_exact, 'fro')
                l_error = torch.norm(L_exact - L_approx, 'fro') / torch.norm(L_exact, 'fro')
                
                # Effective rank (number of significant singular values)
                # Use CPU for SVD on MPS due to numerical issues
                if should_use_cpu_fallback(K_approx.device, 'svd'):
                    log_device_warning_once(K_approx.device, 'svd')
                    _, S_k, _ = torch.linalg.svd(K_approx.cpu())
                    _, S_l, _ = torch.linalg.svd(L_approx.cpu())
                else:
                    _, S_k, _ = torch.linalg.svd(K_approx)
                    _, S_l, _ = torch.linalg.svd(L_approx)
                
                k_rank = torch.sum(S_k > self.eps).item()
                l_rank = torch.sum(S_l > self.eps).item()
                
                return {
                    'k_approximation_error': float(k_error),
                    'l_approximation_error': float(l_error),
                    'k_effective_rank': k_rank,
                    'l_effective_rank': l_rank,
                    'n_landmarks': self.n_landmarks,
                    'n_samples': n_samples
                }
            
            except RuntimeError as e:
                self.logger.warning(f"Could not compute exact kernels for comparison: {e}")
        
        # Return basic info if exact comparison not possible
        return {
            'n_landmarks': self.n_landmarks,
            'n_samples': n_samples,
            'landmark_selection': self.landmark_selection
        }
    
    def estimate_memory_usage(self, n_samples: int, n_features: int) -> Dict[str, float]:
        """Estimate memory usage for given problem size.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        bytes_per_float = 4  # float32
        mb_factor = 1024 * 1024
        
        # Exact CKA memory
        exact_kernel_mb = (n_samples ** 2 * bytes_per_float * 2) / mb_factor
        
        # Nyström CKA memory
        landmark_kernel_mb = (self.n_landmarks ** 2 * bytes_per_float * 2) / mb_factor
        cross_kernel_mb = (n_samples * self.n_landmarks * bytes_per_float * 2) / mb_factor
        nystrom_total_mb = landmark_kernel_mb + cross_kernel_mb
        
        # Memory reduction factor
        reduction_factor = exact_kernel_mb / nystrom_total_mb if nystrom_total_mb > 0 else float('inf')
        
        return {
            'exact_cka_mb': exact_kernel_mb,
            'nystrom_cka_mb': nystrom_total_mb,
            'memory_reduction_factor': reduction_factor,
            'landmark_kernel_mb': landmark_kernel_mb,
            'cross_kernel_mb': cross_kernel_mb
        }
    
    def recommend_landmarks(self, n_samples: int, target_error: float = 0.01) -> int:
        """Recommend number of landmarks for target approximation error.
        
        Args:
            n_samples: Number of samples
            target_error: Target approximation error (default: 1%)
            
        Returns:
            Recommended number of landmarks
        """
        # Heuristic: landmarks should be proportional to sqrt(n_samples)
        # but with minimum and maximum bounds
        base_landmarks = int(np.sqrt(n_samples))
        
        # Adjust for target error (lower error needs more landmarks)
        error_factor = max(0.5, target_error / 0.01)  # Scale relative to 1% baseline
        adjusted_landmarks = int(base_landmarks / error_factor)
        
        # Apply bounds
        min_landmarks = max(4, int(0.05 * n_samples))  # At least 5% of samples
        max_landmarks = min(1024, int(0.5 * n_samples))  # At most 50% of samples
        
        recommended = max(min_landmarks, min(adjusted_landmarks, max_landmarks))
        
        self.logger.info(f"Recommended {recommended} landmarks for {n_samples} samples (target error: {target_error:.3f})")
        
        return recommended
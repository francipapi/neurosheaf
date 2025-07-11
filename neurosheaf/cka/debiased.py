"""Debiased CKA computation without double-centering.

This module implements the mathematically correct CKA computation that avoids
double-centering bias. This is the CRITICAL requirement for Phase 2.

IMPORTANT: The unbiased HSIC estimator used in debiased CKA already performs
centering internally. Therefore:
1. DO NOT center features before computing Gram matrices
2. Pass raw activations to all CKA functions
3. The Gram matrices K = X @ X.T should use raw X

This fix ensures accurate similarity measurements and prevents
artificially suppressed CKA values.

References:
- Murphy et al. (2024): Debiased CKA theory showing unbiased HSIC centers internally
- updated-debiased-cka-v3.md: Fixed implementation details
"""

import warnings
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn

from ..utils.config import Config
from ..utils.device import detect_optimal_device, get_device_info, should_use_cpu_fallback, log_device_warning_once
from ..utils.exceptions import ValidationError, ComputationError, MemoryError
from ..utils.logging import setup_logger
from ..utils.profiling import profile_memory, profile_time
from ..utils.validation import validate_activations, validate_no_preprocessing


class DebiasedCKA:
    """Debiased CKA computation without double-centering.
    
    This class implements the mathematically correct CKA computation that avoids
    the double-centering bias. The implementation is optimized for Mac hardware
    including Apple Silicon MPS support.
    
    Mathematical Properties:
        - CKA(X, X) = 1 (self-similarity)
        - CKA(X, Y) = CKA(Y, X) (symmetry)  
        - 0 ≤ CKA(X, Y) ≤ 1 (bounded)
    
    CRITICAL: This implementation uses raw activations X @ X.T without
    pre-centering. This is essential for mathematical correctness.
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        use_unbiased: bool = True,
        enable_profiling: bool = True,
        numerical_stability: float = Config.numerical.DEFAULT_EPSILON,
        regularization: float = 0.0,
        auto_promote_precision: bool = True,
        safe_tensor_ops: bool = True,
        strict_numerics: bool = False,
        adaptive_epsilon: bool = True,
        enable_gradients: bool = False
    ):
        """Initialize the debiased CKA computer.
        
        Args:
            device: Device to use for computation (auto-detected if None)
            use_unbiased: Whether to use unbiased HSIC estimator (recommended)
            enable_profiling: Whether to enable performance profiling
            numerical_stability: Epsilon for numerical stability
            regularization: Regularization parameter for ill-conditioned matrices (0 = no regularization)
            auto_promote_precision: Whether to automatically promote to float64 for stability
            safe_tensor_ops: Whether to clone tensors for safe operations
            strict_numerics: Whether to fail fast on numerical issues
            adaptive_epsilon: Whether to adapt epsilon based on condition numbers
            enable_gradients: Whether to enable gradient computation through CKA
        """
        self.logger = setup_logger("neurosheaf.cka.debiased")
        self.device = detect_optimal_device(device)
        self.use_unbiased = use_unbiased
        self.enable_profiling = enable_profiling
        self.eps = numerical_stability
        self.regularization = regularization
        self.auto_promote_precision = auto_promote_precision
        self.safe_tensor_ops = safe_tensor_ops
        self.strict_numerics = strict_numerics
        self.adaptive_epsilon = adaptive_epsilon
        self.enable_gradients = enable_gradients
        
        # Device information
        self.device_info = get_device_info()
        self.is_mac = self.device_info['is_mac']
        self.is_apple_silicon = self.device_info['is_apple_silicon']
        
        self.logger.info(f"Initialized DebiasedCKA on {self.device}")
        self.logger.info(f"Using {'unbiased' if use_unbiased else 'biased'} HSIC estimator")
        if self.is_mac:
            self.logger.info(f"Mac optimization enabled: Apple Silicon = {self.is_apple_silicon}")
    
    
    def _validate_numerical_stability(self, *tensors, stage: str = "unknown") -> None:
        """Comprehensive numerical stability validation.
        
        Args:
            tensors: Tensors to validate
            stage: Computation stage for error reporting
            
        Raises:
            ComputationError: If numerical issues detected
        """
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
        """Automatically promote tensors to float64 for numerical stability.
        
        Args:
            X: First activation tensor
            Y: Second activation tensor
            
        Returns:
            Tuple of potentially promoted tensors
        """
        if not self.auto_promote_precision:
            return X, Y
        
        # Check if promotion is needed
        needs_promotion = False
        
        # Check for small sample sizes
        if X.shape[0] < Config.cka.SMALL_SAMPLE_SIZE_THRESHOLD:
            needs_promotion = True
            reason = "small sample size"
        
        # Check for float32 on MPS (known numerical issues)
        elif X.dtype == torch.float32 and self.device.type == 'mps':
            needs_promotion = True
            reason = "float32 on MPS device"
        
        # Check condition numbers if possible
        elif X.shape[0] <= 1000:  # Only for reasonably sized matrices
            try:
                # Quick condition number check
                sample_gram = (X[:min(100, X.shape[0])] @ X[:min(100, X.shape[0])].T)
                if should_use_cpu_fallback(sample_gram.device, 'cond'):
                    cond_est = torch.linalg.cond(sample_gram.cpu()).item()
                else:
                    cond_est = torch.linalg.cond(sample_gram).item()
                
                if cond_est > Config.cka.CONDITION_NUMBER_PROMOTION_THRESHOLD:
                    needs_promotion = True
                    reason = f"high condition number ({cond_est:.2e})"
            except:
                pass  # If condition number check fails, continue without promotion
        
        if needs_promotion:
            if X.dtype != torch.float64:
                self.logger.info(f"Promoting to float64 for numerical stability: {reason}")
                X = X.double()
                Y = Y.double()
        
        return X, Y
    
    def _safe_clone_tensors(self, *tensors) -> Tuple[torch.Tensor, ...]:
        """Safely clone tensors to avoid in-place operations.
        
        Args:
            tensors: Tensors to clone
            
        Returns:
            Tuple of cloned tensors
        """
        if not self.safe_tensor_ops:
            return tensors
        
        return tuple(tensor.clone() for tensor in tensors)
    
    def _compute_adaptive_epsilon(self, K: torch.Tensor, L: torch.Tensor) -> float:
        """Compute adaptive epsilon based on condition numbers.
        
        Args:
            K: First kernel matrix
            L: Second kernel matrix
            
        Returns:
            Adaptive epsilon value
        """
        if not self.adaptive_epsilon:
            return self.eps
        
        try:
            # Compute condition numbers with device fallback
            if should_use_cpu_fallback(K.device, 'cond'):
                log_device_warning_once(K.device, 'cond')
                cond_k = torch.linalg.cond(K.cpu()).item()
                cond_l = torch.linalg.cond(L.cpu()).item()
            else:
                cond_k = torch.linalg.cond(K).item()
                cond_l = torch.linalg.cond(L).item()
            
            # Adapt epsilon based on condition numbers
            max_cond = max(cond_k, cond_l)
            adaptive_eps = max(self.eps, 1e-15 * max_cond)
            
            if adaptive_eps > self.eps:
                self.logger.debug(f"Adaptive epsilon: {adaptive_eps:.2e} (condition numbers: K={cond_k:.2e}, L={cond_l:.2e})")
            
            return adaptive_eps
        
        except Exception as e:
            self.logger.debug(f"Could not compute adaptive epsilon: {e}")
            return self.eps
    
    def _prepare_tensors_for_gradients(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare tensors for gradient computation if enabled.
        
        Args:
            X: First activation tensor
            Y: Second activation tensor
            
        Returns:
            Tuple of tensors prepared for gradient computation
        """
        if not self.enable_gradients:
            return X.detach(), Y.detach()
        
        # Ensure tensors require gradients
        if not X.requires_grad:
            X = X.requires_grad_(True)
        if not Y.requires_grad:
            Y = Y.requires_grad_(True)
        
        return X, Y

    def compute(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        validate_properties: bool = True
    ) -> float:
        """Alias for compute_cka for backward compatibility."""
        return self.compute_cka(X, Y, validate_properties)
    
    @profile_memory()
    @profile_time()
    def compute_cka(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        validate_properties: bool = True
    ) -> float:
        """Compute debiased CKA between two activation matrices.
        
        CRITICAL: This implementation uses raw activations without pre-centering.
        The unbiased HSIC estimator handles centering internally, so pre-centering
        would cause double-centering and suppress CKA values.
        
        Args:
            X: First activation matrix [n_samples, n_features_x]
            Y: Second activation matrix [n_samples, n_features_y]
            validate_properties: Whether to validate CKA mathematical properties
            
        Returns:
            float: CKA similarity value in [0, 1]
            
        Raises:
            ValidationError: If input validation fails
            ComputationError: If CKA computation fails
            MemoryError: If memory requirements exceed limits
        """
        self.logger.debug(f"Computing CKA for shapes X={X.shape}, Y={Y.shape}")
        
        # Validate inputs
        X, Y = validate_activations(X, Y, min_samples=4 if self.use_unbiased else 2)
        
        # Initial numerical stability validation
        self._validate_numerical_stability(X, Y, stage="input")
        
        # Apply automatic precision promotion
        X, Y = self._auto_promote_precision(X, Y)
        
        # Prepare tensors for gradients if needed
        X, Y = self._prepare_tensors_for_gradients(X, Y)
        
        # Handle device placement
        n_samples = X.shape[0]
        if X.dtype == torch.float64 and self.device.type == 'mps':
            # Float64 on MPS may have issues, use CPU
            X = X.cpu()
            Y = Y.cpu()
            compute_device = torch.device('cpu')
            self.logger.debug("Using CPU for float64 computation due to MPS limitations")
        else:
            # Normal device handling
            X = X.to(self.device)
            Y = Y.to(self.device)
            compute_device = self.device
        
        # Check memory requirements
        self._check_memory_requirements(X, Y)
        
        try:
            # Compute Gram matrices from RAW activations (NO centering!)
            # Use gradient-safe computation if enabled
            if self.enable_gradients:
                K = torch.mm(X, X.T)  # Raw gram matrix
                L = torch.mm(Y, Y.T)  # Raw gram matrix
            else:
                K = X @ X.T  # Raw gram matrix
                L = Y @ Y.T  # Raw gram matrix
            
            # Validate kernel matrices
            self._validate_numerical_stability(K, L, stage="kernel_computation")
            
            # Clone tensors for safety if needed (but not for gradients)
            if self.safe_tensor_ops and not self.enable_gradients:
                K, L = self._safe_clone_tensors(K, L)
            
            # Apply regularization if needed
            if self.regularization > 0:
                n = K.shape[0]
                K_trace = torch.trace(K) / n
                L_trace = torch.trace(L) / n
                K = K + self.regularization * torch.eye(n, device=K.device) * K_trace
                L = L + self.regularization * torch.eye(n, device=L.device) * L_trace
                self.logger.debug(f"Applied regularization with alpha={self.regularization}")
            
            # Check condition numbers for numerical stability
            if self.enable_profiling:
                try:
                    if should_use_cpu_fallback(K.device, 'cond'):
                        log_device_warning_once(K.device, 'cond')
                        K_cond = torch.linalg.cond(K.cpu()).item()
                        L_cond = torch.linalg.cond(L.cpu()).item()
                    else:
                        K_cond = torch.linalg.cond(K).item()
                        L_cond = torch.linalg.cond(L).item()
                    
                    if K_cond > Config.cka.CONDITION_WARNING_THRESHOLD or L_cond > Config.cka.CONDITION_WARNING_THRESHOLD:
                        warning_msg = f"High condition numbers detected: K={K_cond:.2e}, L={L_cond:.2e}. Results may be numerically unstable."
                        if self.strict_numerics:
                            raise ComputationError(warning_msg)
                        else:
                            self.logger.warning(warning_msg)
                except Exception as e:
                    # If condition number computation fails, log but continue
                    self.logger.debug(f"Could not compute condition numbers: {e}")
            
            # Compute adaptive epsilon
            adaptive_eps = self._compute_adaptive_epsilon(K, L)
            
            # Compute CKA using appropriate HSIC estimator
            if self.use_unbiased:
                cka_value = self._compute_unbiased_cka(K, L, adaptive_eps)
            else:
                cka_value = self._compute_biased_cka(K, L, adaptive_eps)
            
            # If we used CPU for stability, move result back if needed
            if compute_device != self.device and isinstance(cka_value, torch.Tensor):
                cka_value = float(cka_value)
            
            # Validate mathematical properties
            if validate_properties:
                self._validate_cka_properties(cka_value, X, Y)
            
            self.logger.debug(f"CKA value: {cka_value:.6f}")
            return cka_value
            
        except Exception as e:
            raise ComputationError(f"CKA computation failed: {str(e)}")
    
    def _compute_unbiased_cka(self, K: torch.Tensor, L: torch.Tensor, eps: Optional[float] = None) -> float:
        """Compute debiased CKA from Gram matrices using unbiased HSIC.
        
        This implements the unbiased HSIC estimator from Murphy et al. (2024).
        The formula already includes centering, so K and L should be
        computed from raw (uncentered) features.
        
        Args:
            K: Gram matrix from raw activations X @ X.T
            L: Gram matrix from raw activations Y @ Y.T
            eps: Optional epsilon value (uses adaptive if None)
            
        Returns:
            float: CKA value in [0, 1]
        """
        if eps is None:
            eps = self.eps
        
        n = K.shape[0]
        
        # Compute HSIC values with numerical stability checks
        hsic_xy = self._unbiased_hsic(K, L)
        hsic_xx = self._unbiased_hsic(K, K)
        hsic_yy = self._unbiased_hsic(L, L)
        
        # Validate HSIC values
        self._validate_numerical_stability(hsic_xy, hsic_xx, hsic_yy, stage="hsic_computation")
        
        # Handle negative HSIC values (can occur due to numerical errors)
        if hsic_xx < 0 or hsic_yy < 0:
            warning_msg = f"Negative HSIC values detected: hsic_xx={hsic_xx:.6e}, hsic_yy={hsic_yy:.6e}. Setting to epsilon for stability."
            if self.strict_numerics:
                raise ComputationError(warning_msg)
            else:
                self.logger.warning(warning_msg)
            hsic_xx = max(hsic_xx, eps)
            hsic_yy = max(hsic_yy, eps)
        
        # Compute CKA with stable division
        denominator = torch.sqrt(hsic_xx * hsic_yy)
        cka = hsic_xy / torch.clamp(denominator, min=eps)
        result = float(torch.clamp(cka, 0, 1))
        
        # Final validation
        self._validate_numerical_stability(torch.tensor(result), stage="final_result")
        
        return result
    
    def _unbiased_hsic(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Compute unbiased HSIC estimator.
        
        This estimator handles centering internally, so K and L must be
        computed from raw activations to avoid double-centering.
        
        Args:
            K: Gram matrix from raw activations
            L: Gram matrix from raw activations
            
        Returns:
            torch.Tensor: Unbiased HSIC value
        """
        n = K.shape[0]
        
        if n < Config.cka.MIN_SAMPLES_UNBIASED:
            raise ValidationError(
                f"Unbiased HSIC requires at least {Config.cka.MIN_SAMPLES_UNBIASED} samples, got {n}"
            )
        
        # Remove diagonal
        K_0 = K - torch.diag(torch.diag(K))
        L_0 = L - torch.diag(torch.diag(L))
        
        # Compute terms for unbiased estimator
        term1 = torch.sum(K_0 * L_0)
        term2 = torch.sum(K_0) * torch.sum(L_0) / ((n - 1) * (n - 2))
        term3 = 2 * torch.sum(K_0, dim=0) @ torch.sum(L_0, dim=0) / (n - 2)
        
        hsic = (term1 + term2 - term3) / (n * (n - 3))
        
        return hsic
    
    def _compute_biased_cka(self, K: torch.Tensor, L: torch.Tensor, eps: Optional[float] = None) -> float:
        """Compute CKA using biased HSIC estimator for comparison.
        
        Args:
            K: Gram matrix from raw activations
            L: Gram matrix from raw activations
            eps: Optional epsilon value (uses adaptive if None)
            
        Returns:
            float: CKA value in [0, 1]
        """
        if eps is None:
            eps = self.eps
        
        n = K.shape[0]
        
        # Centering matrix H = I - (1/n) * ones
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        
        # Center the kernels (safe cloning if enabled, but not for gradients)
        if self.safe_tensor_ops and not self.enable_gradients:
            K_c = H @ K.clone() @ H
            L_c = H @ L.clone() @ H
        else:
            K_c = H @ K @ H
            L_c = H @ L @ H
        
        # Validate centered kernels
        self._validate_numerical_stability(K_c, L_c, stage="kernel_centering")
        
        # Compute HSIC values
        hsic_xy = torch.sum(K_c * L_c) / ((n - 1) ** 2)
        hsic_xx = torch.sum(K_c * K_c) / ((n - 1) ** 2)
        hsic_yy = torch.sum(L_c * L_c) / ((n - 1) ** 2)
        
        # Validate HSIC values
        self._validate_numerical_stability(hsic_xy, hsic_xx, hsic_yy, stage="biased_hsic_computation")
        
        # Handle negative HSIC values
        if hsic_xx < 0 or hsic_yy < 0:
            warning_msg = f"Negative HSIC values in biased estimator: hsic_xx={hsic_xx:.6e}, hsic_yy={hsic_yy:.6e}"
            if self.strict_numerics:
                raise ComputationError(warning_msg)
            else:
                self.logger.warning(warning_msg)
            hsic_xx = max(hsic_xx, eps)
            hsic_yy = max(hsic_yy, eps)
        
        # Compute CKA with stable division
        denominator = torch.sqrt(hsic_xx * hsic_yy)
        cka = hsic_xy / torch.clamp(denominator, min=eps)
        result = float(torch.clamp(cka, 0, 1))
        
        # Final validation
        self._validate_numerical_stability(torch.tensor(result), stage="biased_final_result")
        
        return result
    
    def compute_cka_matrix(
        self,
        activations: Dict[str, torch.Tensor],
        symmetric: bool = True,
        warn_preprocessing: bool = True
    ) -> torch.Tensor:
        """Compute CKA similarity matrix between multiple activation sets.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
                        MUST be raw activations (not centered!)
            symmetric: Whether to compute only upper triangle (assumes symmetry)
            warn_preprocessing: Whether to warn if activations appear centered
            
        Returns:
            torch.Tensor: CKA similarity matrix [n_layers, n_layers]
        """
        if not activations:
            raise ValidationError("Activations dictionary cannot be empty")
        
        layer_names = list(activations.keys())
        n_layers = len(layer_names)
        
        if n_layers < 2:
            raise ValidationError("Need at least 2 layers to compute CKA matrix")
        
        # Warn if activations appear to be pre-centered
        if warn_preprocessing:
            validate_no_preprocessing(activations)
        
        # Initialize similarity matrix
        cka_matrix = torch.zeros(n_layers, n_layers, device=self.device)
        
        # Compute pairwise CKA similarities
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                if symmetric and j < i:
                    # Use symmetry to avoid redundant computation
                    cka_matrix[i, j] = cka_matrix[j, i]
                else:
                    cka_value = self.compute_cka(
                        activations[layer_i],
                        activations[layer_j],
                        validate_properties=False  # Skip validation for efficiency
                    )
                    cka_matrix[i, j] = cka_value
        
        return cka_matrix
    
    
    def _check_memory_requirements(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Check if memory requirements are within limits.
        
        Args:
            X: First activation matrix
            Y: Second activation matrix
            
        Raises:
            MemoryError: If memory requirements exceed limits
        """
        n_samples = X.shape[0]
        
        # Estimate memory usage for Gram matrices (n x n each)
        gram_memory_bytes = 2 * n_samples * n_samples * 4  # 2 matrices, 4 bytes per float32
        gram_memory_gb = gram_memory_bytes / (1024**3)
        
        # Check against system memory
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if gram_memory_gb > available_memory_gb * 0.8:  # Use max 80% of available memory
            raise MemoryError(
                f"CKA computation requires {gram_memory_gb:.2f}GB but only "
                f"{available_memory_gb:.2f}GB available",
                memory_required_gb=gram_memory_gb,
                memory_available_gb=available_memory_gb
            )
    
    def _validate_cka_properties(
        self,
        cka_value: float,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> None:
        """Validate CKA mathematical properties.
        
        Args:
            cka_value: Computed CKA value
            X: First activation matrix
            Y: Second activation matrix
            
        Raises:
            ComputationError: If mathematical properties are violated
        """
        # Property 1: 0 ≤ CKA ≤ 1
        if not (0 <= cka_value <= 1 + self.eps):
            raise ComputationError(
                f"CKA value {cka_value:.6f} outside valid range [0, 1]"
            )
        
        # Property 2: CKA(X, X) = 1 (self-similarity)
        if torch.equal(X, Y):
            if abs(cka_value - 1.0) > self.eps:
                raise ComputationError(
                    f"CKA(X, X) = {cka_value:.6f}, expected 1.0"
                )
    
    def benchmark_memory_usage(
        self,
        sizes: List[int],
        feature_dim: int = 512
    ) -> Dict[str, List[float]]:
        """Benchmark memory usage for different input sizes.
        
        Args:
            sizes: List of sample sizes to test
            feature_dim: Feature dimension for synthetic data
            
        Returns:
            Dictionary with memory usage statistics
        """
        results = {
            'sizes': sizes,
            'memory_gb': [],
            'computation_time': []
        }
        
        for n in sizes:
            self.logger.info(f"Benchmarking size {n}...")
            
            # Generate synthetic data
            X = torch.randn(n, feature_dim, device=self.device)
            Y = torch.randn(n, feature_dim, device=self.device)
            
            # Measure memory and time
            import time
            import psutil
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**3)
            
            start_time = time.time()
            cka_value = self.compute_cka(X, Y, validate_properties=False)
            end_time = time.time()
            
            peak_memory = process.memory_info().rss / (1024**3)
            
            results['memory_gb'].append(peak_memory - initial_memory)
            results['computation_time'].append(end_time - start_time)
            
            self.logger.info(f"Size {n}: Memory={peak_memory-initial_memory:.2f}GB, Time={end_time-start_time:.2f}s")
        
        return results
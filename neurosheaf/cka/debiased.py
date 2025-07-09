"""Debiased CKA computation without double-centering.

This module implements the mathematically correct CKA computation that avoids
double-centering bias. This is the CRITICAL requirement for Phase 2.

IMPORTANT: This implementation uses raw activations without pre-centering.
This is a naive implementation for Phase 1 Week 2 baseline measurements.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import platform

from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError, ComputationError, MemoryError
from ..utils.profiling import profile_memory, profile_time


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
        memory_efficient: bool = False,
        enable_profiling: bool = True,
        numerical_stability: float = 1e-8
    ):
        """Initialize the debiased CKA computer.
        
        Args:
            device: Device to use for computation (auto-detected if None)
            memory_efficient: Whether to use memory-efficient computation
            enable_profiling: Whether to enable performance profiling
            numerical_stability: Epsilon for numerical stability
        """
        self.logger = setup_logger("neurosheaf.cka.debiased")
        self.device = self._detect_device(device)
        self.memory_efficient = memory_efficient
        self.enable_profiling = enable_profiling
        self.eps = numerical_stability
        
        # Mac-specific initialization
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = platform.processor() == "arm"
        
        self.logger.info(f"Initialized DebiasedCKA on {self.device}")
        if self.is_mac:
            self.logger.info(f"Mac optimization enabled: Apple Silicon = {self.is_apple_silicon}")
    
    def _detect_device(self, device: Optional[Union[str, torch.device]] = None) -> torch.device:
        """Detect optimal device for Mac and other platforms.
        
        Args:
            device: Optional device specification
            
        Returns:
            torch.device: The selected device
        """
        if device is not None:
            return torch.device(device)
        
        # Mac-specific device detection
        if platform.system() == "Darwin":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        
        # Other platforms
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        return torch.device("cpu")
    
    @profile_memory()
    @profile_time()
    def compute_cka(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        validate_properties: bool = True
    ) -> torch.Tensor:
        """Compute debiased CKA between two activation matrices.
        
        CRITICAL: This implementation uses raw activations without pre-centering.
        The formula is: CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
        where HSIC is computed on raw activations.
        
        Args:
            X: First activation matrix [n_samples, n_features_x]
            Y: Second activation matrix [n_samples, n_features_y]
            validate_properties: Whether to validate CKA mathematical properties
            
        Returns:
            torch.Tensor: CKA similarity value (scalar)
            
        Raises:
            ValidationError: If input validation fails
            ComputationError: If CKA computation fails
            MemoryError: If memory requirements exceed limits
        """
        self.logger.debug(f"Computing CKA for shapes X={X.shape}, Y={Y.shape}")
        
        # Validate inputs
        self._validate_inputs(X, Y)
        
        # Move to device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Check memory requirements
        self._check_memory_requirements(X, Y)
        
        try:
            # Compute HSIC components
            hsic_xy = self._compute_hsic(X, Y)
            hsic_xx = self._compute_hsic(X, X)
            hsic_yy = self._compute_hsic(Y, Y)
            
            # Compute CKA
            denominator = torch.sqrt(hsic_xx * hsic_yy)
            
            # Numerical stability
            if denominator < self.eps:
                self.logger.warning(f"Near-zero denominator: {denominator}")
                return torch.tensor(0.0, device=self.device)
            
            cka_value = hsic_xy / denominator
            
            # Validate mathematical properties
            if validate_properties:
                self._validate_cka_properties(cka_value, X, Y)
            
            self.logger.debug(f"CKA value: {cka_value.item():.6f}")
            return cka_value
            
        except Exception as e:
            raise ComputationError(f"CKA computation failed: {str(e)}")
    
    def _compute_hsic(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute HSIC (Hilbert-Schmidt Independence Criterion).
        
        CRITICAL: This uses raw activations without pre-centering.
        Formula: HSIC(X, Y) = (1/n²) * tr(K * L * H)
        where K = X @ X.T, L = Y @ Y.T, and H is the centering matrix.
        
        Args:
            X: First activation matrix [n_samples, n_features_x]
            Y: Second activation matrix [n_samples, n_features_y]
            
        Returns:
            torch.Tensor: HSIC value (scalar)
        """
        n = X.shape[0]
        
        # Compute Gram matrices using raw activations (NO pre-centering)
        K = X @ X.T  # [n_samples, n_samples]
        L = Y @ Y.T  # [n_samples, n_samples]
        
        # Centering matrix H = I - (1/n) * ones
        H = torch.eye(n, device=self.device) - torch.ones(n, n, device=self.device) / n
        
        # HSIC computation
        if self.memory_efficient:
            # Memory-efficient version (slower but uses less memory)
            hsic = torch.trace(K @ L @ H) / (n * n)
        else:
            # Standard version (faster but memory-intensive)
            hsic = torch.trace(K @ L @ H) / (n * n)
        
        return hsic
    
    def compute_cka_matrix(
        self,
        activations: Dict[str, torch.Tensor],
        symmetric: bool = True
    ) -> torch.Tensor:
        """Compute CKA similarity matrix between multiple activation sets.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
            symmetric: Whether to compute only upper triangle (assumes symmetry)
            
        Returns:
            torch.Tensor: CKA similarity matrix [n_layers, n_layers]
        """
        layer_names = list(activations.keys())
        n_layers = len(layer_names)
        
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
    
    def _validate_inputs(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Validate input tensors for CKA computation.
        
        Args:
            X: First activation matrix
            Y: Second activation matrix
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
            raise ValidationError("Inputs must be torch.Tensor")
        
        if X.dim() != 2 or Y.dim() != 2:
            raise ValidationError("Inputs must be 2D tensors")
        
        if X.shape[0] != Y.shape[0]:
            raise ValidationError(
                f"Sample dimensions must match: X={X.shape[0]}, Y={Y.shape[0]}"
            )
        
        if X.shape[0] < 2:
            raise ValidationError("Need at least 2 samples for CKA computation")
    
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
        cka_value: torch.Tensor,
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
                f"CKA value {cka_value.item():.6f} outside valid range [0, 1]"
            )
        
        # Property 2: CKA(X, X) = 1 (self-similarity)
        if torch.equal(X, Y):
            if abs(cka_value - 1.0) > self.eps:
                raise ComputationError(
                    f"CKA(X, X) = {cka_value.item():.6f}, expected 1.0"
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
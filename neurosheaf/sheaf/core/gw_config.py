"""Configuration for Gromov-Wasserstein sheaf construction.

This module provides configuration classes and utilities for GW-based
sheaf construction, including entropic regularization parameters,
convergence criteria, and performance optimization settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import numpy as np


@dataclass
class GWConfig:
    """Configuration for GW-based sheaf construction.
    
    This configuration controls the entropic GW optimization process,
    numerical stability parameters, and performance settings.
    
    Mathematical Parameters:
    - epsilon: Entropic regularization strength (higher = smoother couplings)
    - max_iter: Maximum iterations for entropic GW solver
    - tolerance: Convergence threshold for solver termination
    
    Quality Control:
    - quasi_sheaf_tolerance: Maximum allowed functoriality violation ε
    - validate_couplings: Runtime validation of marginal constraints
    
    Performance:  
    - use_gpu: GPU acceleration via POT backend
    - cache_cost_matrices: Memory vs computation tradeoff
    
    Measures and Inner Products:
    - uniform_measures: Use uniform distributions vs importance sampling
    - weighted_inner_product: Use p_i-weighted L2 inner products
    """
    
    # Core GW optimization parameters
    epsilon: float = 0.1                     # Entropic regularization strength
    max_iter: int = 1000                     # Maximum GW iterations
    tolerance: float = 1e-9                  # Convergence tolerance
    
    # Sheaf quality control
    quasi_sheaf_tolerance: float = 0.1       # ε-sheaf validation threshold
    
    # Performance optimization
    use_gpu: bool = True                     # GPU acceleration
    cache_cost_matrices: bool = True         # Cache expensive cost matrices
    cache_hash_method: str = 'sha1'          # Hash method: 'sha1' (fast), 'id' (session-only)
    
    # Runtime validation (can disable for performance)
    validate_couplings: bool = True          # Validate marginal constraints
    validate_costs: bool = True              # Validate cost matrix properties
    
    # Measure and inner product options
    uniform_measures: bool = True            # Use uniform p_i vs importance sampling
    weighted_inner_product: bool = False     # Use p_i-weighted L2 inner products
    
    # Numerical stability  
    cost_matrix_eps: float = 1e-12           # Numerical threshold for cost matrices
    coupling_eps: float = 1e-10              # Threshold for coupling validation
    measure_eps: float = 1e-6                # Floor value for variance-based measures
    
    # Memory management
    max_cache_size_gb: float = 2.0           # Maximum cache size in GB
    
    # Adaptive epsilon parameters
    adaptive_epsilon: bool = False             # Enable adaptive scaling (default False for backward compatibility)
    base_epsilon: float = 0.1                  # Base epsilon for reference size (matches current default)
    reference_n: float = 50.0                  # Reference sample size (current working size)
    epsilon_scaling_method: str = 'sqrt'       # Scaling method: 'sqrt' (recommended by theory)
    epsilon_min: float = 0.01                  # Minimum allowed epsilon
    epsilon_max: float = 0.5                   # Maximum allowed epsilon
    
    # Unit vs Sample alignment
    align_units: bool = True                   # If True, align units/neurons; if False, align samples (deprecated)
    
    # Numerical precision control
    computation_dtype: str = 'float64'         # Primary dtype for GW computations ('float32' or 'float64')
    
    # Quality control and fallback behavior
    strict_quality_mode: bool = False          # If True, fail fast when POT unavailable or low quality
    exclude_fallback_edges: bool = True        # If True, exclude fallback edges from Laplacian by default  
    min_coupling_quality: float = 0.1          # Minimum quality score threshold [0,1]
    
    # GW Restriction validation parameters
    validate_restrictions: bool = True          # Enable restriction map validation
    stochastic_tolerance: float = 1e-6         # Tolerance for row-stochasticity check
    correction_threshold: float = 1e-3         # Auto-correction threshold for small violations
    strict_validation_threshold: float = 0.1   # Threshold for strict mode violations
    strict_validation_mode: bool = False       # Raise errors vs warnings for large violations
    auto_correct_restrictions: bool = True     # Automatically correct small violations
    
    # Normalized Laplacian option
    use_normalized_laplacian: bool = False      # Use normalized Hodge Laplacian (L x = λ M x) vs standard Laplacian
    
    def __post_init__(self):
        """Automatically validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
            
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
            
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
            
        if self.quasi_sheaf_tolerance < 0:
            raise ValueError(f"quasi_sheaf_tolerance must be non-negative, got {self.quasi_sheaf_tolerance}")
            
        if self.max_cache_size_gb <= 0:
            raise ValueError(f"max_cache_size_gb must be positive, got {self.max_cache_size_gb}")
            
        if self.cost_matrix_eps <= 0:
            raise ValueError(f"cost_matrix_eps must be positive, got {self.cost_matrix_eps}")
            
        if self.cache_hash_method not in {'sha1', 'id'}:
            raise ValueError(f"cache_hash_method must be 'sha1' or 'id', got '{self.cache_hash_method}'")
            
        if self.coupling_eps <= 0:
            raise ValueError(f"coupling_eps must be positive, got {self.coupling_eps}")
            
        if self.measure_eps <= 0:
            raise ValueError(f"measure_eps must be positive, got {self.measure_eps}")
            
        # Validate computation dtype
        if self.computation_dtype not in {'float32', 'float64'}:
            raise ValueError(f"computation_dtype must be 'float32' or 'float64', got '{self.computation_dtype}'")
            
        # Validate quality control parameters
        if not 0.0 <= self.min_coupling_quality <= 1.0:
            raise ValueError(f"min_coupling_quality must be in [0,1], got {self.min_coupling_quality}")
            
        # Validate adaptive epsilon parameters
        if self.adaptive_epsilon:
            if self.base_epsilon <= 0:
                raise ValueError(f"base_epsilon must be positive, got {self.base_epsilon}")
                
            if self.reference_n <= 0:
                raise ValueError(f"reference_n must be positive, got {self.reference_n}")
                
            if self.epsilon_min <= 0:
                raise ValueError(f"epsilon_min must be positive, got {self.epsilon_min}")
                
            if self.epsilon_max <= 0:
                raise ValueError(f"epsilon_max must be positive, got {self.epsilon_max}")
                
            if self.epsilon_min > self.epsilon_max:
                raise ValueError(f"epsilon_min ({self.epsilon_min}) must be <= epsilon_max ({self.epsilon_max})")
                
            if self.epsilon_scaling_method not in ['sqrt']:
                raise ValueError(f"epsilon_scaling_method must be 'sqrt', got {self.epsilon_scaling_method}")
        
        # Validate restriction validation parameters
        if self.stochastic_tolerance <= 0:
            raise ValueError(f"stochastic_tolerance must be positive, got {self.stochastic_tolerance}")
            
        if self.correction_threshold <= 0:
            raise ValueError(f"correction_threshold must be positive, got {self.correction_threshold}")
            
        if self.strict_validation_threshold <= 0:
            raise ValueError(f"strict_validation_threshold must be positive, got {self.strict_validation_threshold}")
            
        if self.correction_threshold > self.strict_validation_threshold:
            raise ValueError(f"correction_threshold ({self.correction_threshold}) must be <= "
                           f"strict_validation_threshold ({self.strict_validation_threshold})")
        
        # Warn about deprecated sample-based alignment
        if not self.align_units:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Sample-based alignment (align_units=False) is deprecated. "
                         "Unit-based alignment is mathematically correct for neural network analysis.")
    
    def get_torch_dtype(self) -> torch.dtype:
        """Get the corresponding PyTorch dtype for computation_dtype.
        
        Returns:
            torch.float32 or torch.float64 based on computation_dtype
        """
        if self.computation_dtype == 'float32':
            return torch.float32
        elif self.computation_dtype == 'float64':
            return torch.float64
        else:
            raise ValueError(f"Unsupported computation_dtype: {self.computation_dtype}")
    
    def get_numpy_dtype(self) -> np.dtype:
        """Get the corresponding NumPy dtype for computation_dtype.
        
        Returns:
            np.float32 or np.float64 based on computation_dtype
        """
        if self.computation_dtype == 'float32':
            return np.float32
        elif self.computation_dtype == 'float64':
            return np.float64
        else:
            raise ValueError(f"Unsupported computation_dtype: {self.computation_dtype}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'epsilon': self.epsilon,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'quasi_sheaf_tolerance': self.quasi_sheaf_tolerance,
            'use_gpu': self.use_gpu,
            'cache_cost_matrices': self.cache_cost_matrices,
            'cache_hash_method': self.cache_hash_method,
            'validate_couplings': self.validate_couplings,
            'validate_costs': self.validate_costs,
            'uniform_measures': self.uniform_measures,
            'weighted_inner_product': self.weighted_inner_product,
            'cost_matrix_eps': self.cost_matrix_eps,
            'coupling_eps': self.coupling_eps,
            'measure_eps': self.measure_eps,
            'max_cache_size_gb': self.max_cache_size_gb,
            'adaptive_epsilon': self.adaptive_epsilon,
            'base_epsilon': self.base_epsilon,
            'reference_n': self.reference_n,
            'epsilon_scaling_method': self.epsilon_scaling_method,
            'epsilon_min': self.epsilon_min,
            'epsilon_max': self.epsilon_max,
            'align_units': self.align_units,
            'computation_dtype': self.computation_dtype,
            'use_normalized_laplacian': self.use_normalized_laplacian,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GWConfig':
        """Create configuration from dictionary."""
        # Filter out unknown keys for forward compatibility
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    @classmethod
    def default_fast(cls) -> 'GWConfig':
        """Create configuration optimized for speed over accuracy."""
        return cls(
            epsilon=0.05,        # Higher regularization = faster convergence
            max_iter=500,        # Fewer iterations
            tolerance=1e-6,      # Looser convergence
            validate_couplings=False,  # Skip runtime validation
            validate_costs=False,
            use_normalized_laplacian=False,  # Default to standard Laplacian for compatibility
        )
    
    @classmethod 
    def default_accurate(cls) -> 'GWConfig':
        """Create configuration optimized for accuracy over speed."""
        return cls(
            epsilon=0.01,        # Lower regularization = more accurate
            max_iter=2000,       # More iterations
            tolerance=1e-12,     # Tight convergence  
            validate_couplings=True,   # Full validation
            validate_costs=True,
            use_normalized_laplacian=False,  # Default to standard Laplacian for compatibility
        )
    
    @classmethod
    def default_debugging(cls) -> 'GWConfig':
        """Create configuration for debugging with full validation."""
        return cls(
            epsilon=0.1,
            max_iter=1000,
            tolerance=1e-9,
            validate_couplings=True,
            validate_costs=True,
            cache_cost_matrices=False,  # Disable caching for debugging
            use_gpu=False,              # Use CPU for better error messages
            use_normalized_laplacian=False,  # Default to standard Laplacian for compatibility
        )
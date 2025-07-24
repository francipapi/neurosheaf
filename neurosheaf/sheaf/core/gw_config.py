"""Configuration for Gromov-Wasserstein sheaf construction.

This module provides configuration classes and utilities for GW-based
sheaf construction, including entropic regularization parameters,
convergence criteria, and performance optimization settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


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
    
    # Runtime validation (can disable for performance)
    validate_couplings: bool = True          # Validate marginal constraints
    validate_costs: bool = True              # Validate cost matrix properties
    
    # Measure and inner product options
    uniform_measures: bool = True            # Use uniform p_i vs importance sampling
    weighted_inner_product: bool = False     # Use p_i-weighted L2 inner products
    
    # Numerical stability  
    cost_matrix_eps: float = 1e-12           # Numerical threshold for cost matrices
    coupling_eps: float = 1e-10              # Threshold for coupling validation
    
    # Memory management
    max_cache_size_gb: float = 2.0           # Maximum cache size in GB
    
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
            
        if self.coupling_eps <= 0:
            raise ValueError(f"coupling_eps must be positive, got {self.coupling_eps}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'epsilon': self.epsilon,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'quasi_sheaf_tolerance': self.quasi_sheaf_tolerance,
            'use_gpu': self.use_gpu,
            'cache_cost_matrices': self.cache_cost_matrices,
            'validate_couplings': self.validate_couplings,
            'validate_costs': self.validate_costs,
            'uniform_measures': self.uniform_measures,
            'weighted_inner_product': self.weighted_inner_product,
            'cost_matrix_eps': self.cost_matrix_eps,
            'coupling_eps': self.coupling_eps,
            'max_cache_size_gb': self.max_cache_size_gb,
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
        )
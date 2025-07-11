"""Configuration constants for Neurosheaf.

This module centralizes all configuration constants used throughout the
Neurosheaf framework to improve maintainability and consistency.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class NumericalConstants:
    """Numerical stability and tolerance constants."""
    
    # Numerical stability epsilons
    DEFAULT_EPSILON: float = 1e-8
    STRICT_EPSILON: float = 1e-12
    LOOSE_EPSILON: float = 1e-6
    
    # Eigenvalue thresholds
    MIN_EIGENVALUE: float = 1e-8
    RANK_TOLERANCE: float = 1e-12
    CONDITION_NUMBER_THRESHOLD: float = 1e12
    
    # Regularization values
    DEFAULT_REGULARIZATION: float = 1e-10
    STRONG_REGULARIZATION: float = 1e-6
    
    # Convergence thresholds
    CONVERGENCE_TOLERANCE: float = 1e-6
    MAX_ITERATIONS: int = 1000


@dataclass(frozen=True)
class MemoryConstants:
    """Memory management constants."""
    
    # Memory thresholds (in MB)
    DEFAULT_MEMORY_THRESHOLD_MB: float = 1000.0
    STRICT_MEMORY_THRESHOLD_MB: float = 500.0
    LOOSE_MEMORY_THRESHOLD_MB: float = 3000.0
    
    # Memory targets (in GB)
    BASELINE_TARGET_GB: float = 20.0
    OPTIMIZED_TARGET_GB: float = 3.0
    TARGET_REDUCTION_FACTOR: float = 7.0
    
    # Memory limits
    DEFAULT_MEMORY_LIMIT_GB: float = 8.0
    STRICT_MEMORY_LIMIT_GB: float = 4.0
    
    # Batch size limits
    MAX_BATCH_SIZE: int = 10000
    DEFAULT_BATCH_SIZE: int = 1000
    MIN_BATCH_SIZE: int = 4


@dataclass(frozen=True)
class CKAConstants:
    """CKA computation constants."""
    
    # Sample size requirements
    MIN_SAMPLES_BIASED: int = 2
    MIN_SAMPLES_UNBIASED: int = 4
    MIN_SAMPLES_ROBUST: int = 10
    
    # Validation thresholds
    CKA_MIN_VALUE: float = 0.0
    CKA_MAX_VALUE: float = 1.0
    SELF_SIMILARITY_TOLERANCE: float = 0.05
    
    # Numerical parameters
    HSIC_EPSILON: float = 1e-8
    CONDITION_WARNING_THRESHOLD: float = 1e10
    
    # Precision promotion thresholds
    SMALL_SAMPLE_SIZE_THRESHOLD: int = 20
    CONDITION_NUMBER_PROMOTION_THRESHOLD: float = 1e10


@dataclass(frozen=True)
class NystromConstants:
    """Nyström approximation constants."""
    
    # Landmark selection
    DEFAULT_LANDMARKS: int = 256
    MIN_LANDMARKS: int = 4
    MAX_LANDMARKS: int = 1024
    
    # Approximation quality
    TARGET_APPROXIMATION_ERROR: float = 0.01
    MAX_APPROXIMATION_ERROR: float = 0.05
    
    # K-means parameters
    MAX_KMEANS_ITERATIONS: int = 100
    KMEANS_RANDOM_STATE: int = 42
    
    # Memory reduction targets
    TARGET_MEMORY_REDUCTION_FACTOR: float = 10.0
    MIN_MEMORY_REDUCTION_FACTOR: float = 5.0


@dataclass(frozen=True)
class SheafConstants:
    """Sheaf construction constants."""
    
    # Restriction map quality thresholds
    EXCELLENT_RESIDUAL_THRESHOLD: float = 0.01
    GOOD_RESIDUAL_THRESHOLD: float = 0.05
    ACCEPTABLE_RESIDUAL_THRESHOLD: float = 0.10
    
    # Whitening parameters
    WHITENING_MIN_EIGENVALUE: float = 1e-8
    WHITENING_REGULARIZATION: float = 1e-10
    
    # Transitivity validation
    TRANSITIVITY_TOLERANCE: float = 1e-6
    MAX_TRANSITIVITY_ERROR: float = 1e-3
    
    # Sparsity thresholds
    TARGET_SPARSITY_PERCENTAGE: float = 60.0
    MIN_SPARSITY_PERCENTAGE: float = 30.0


@dataclass(frozen=True)
class PerformanceConstants:
    """Performance and timing constants."""
    
    # Time thresholds (in seconds)
    DEFAULT_TIME_THRESHOLD: float = 60.0
    FAST_OPERATION_THRESHOLD: float = 5.0
    SLOW_OPERATION_THRESHOLD: float = 300.0
    
    # Timeout values
    DEFAULT_TIMEOUT: float = 120.0
    LONG_TIMEOUT: float = 600.0
    
    # Retry parameters
    DEFAULT_MAX_RETRIES: int = 3
    RETRY_DELAY: float = 0.1
    
    # Profiling parameters
    WARMUP_RUNS: int = 2
    BENCHMARK_RUNS: int = 10


@dataclass(frozen=True)
class ValidationConstants:
    """Validation and testing constants."""
    
    # Test tolerances
    NUMERICAL_TEST_TOLERANCE: float = 1e-6
    LOOSE_TEST_TOLERANCE: float = 1e-4
    STRICT_TEST_TOLERANCE: float = 1e-8
    
    # Quality thresholds
    GOOD_QUALITY_THRESHOLD: float = 0.8
    ACCEPTABLE_QUALITY_THRESHOLD: float = 0.6
    
    # Coverage requirements
    MIN_TEST_COVERAGE: float = 0.95
    TARGET_TEST_COVERAGE: float = 0.98


# Global configuration instance
class Config:
    """Global configuration object containing all constants."""
    
    numerical = NumericalConstants()
    memory = MemoryConstants()
    cka = CKAConstants()
    nystrom = NystromConstants()
    sheaf = SheafConstants()
    performance = PerformanceConstants()
    validation = ValidationConstants()
    
    @classmethod
    def get_all_constants(cls) -> Dict[str, Any]:
        """Get all constants as a flat dictionary.
        
        Returns:
            Dictionary with all configuration constants
        """
        constants = {}
        
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, (NumericalConstants, MemoryConstants, CKAConstants, 
                               NystromConstants, SheafConstants, PerformanceConstants,
                               ValidationConstants)):
                category = attr_name
                for field_name in attr.__dataclass_fields__:
                    field_value = getattr(attr, field_name)
                    constants[f"{category}.{field_name}"] = field_value
        
        return constants
    
    @classmethod
    def override_constants(cls, overrides: Dict[str, Any]) -> None:
        """Override specific constants (for testing purposes).
        
        Note: This will replace the dataclass instances, losing immutability.
        Only use for testing.
        
        Args:
            overrides: Dictionary with keys like "numerical.DEFAULT_EPSILON"
        """
        # This is intentionally not implemented for production use
        # Constants should remain immutable in production
        raise NotImplementedError("Constant overrides are not supported in production")


# Convenience access to commonly used constants
DEFAULT_EPSILON = Config.numerical.DEFAULT_EPSILON
DEFAULT_MEMORY_THRESHOLD_MB = Config.memory.DEFAULT_MEMORY_THRESHOLD_MB
MIN_SAMPLES_UNBIASED = Config.cka.MIN_SAMPLES_UNBIASED
DEFAULT_LANDMARKS = Config.nystrom.DEFAULT_LANDMARKS
GOOD_RESIDUAL_THRESHOLD = Config.sheaf.GOOD_RESIDUAL_THRESHOLD
DEFAULT_TIME_THRESHOLD = Config.performance.DEFAULT_TIME_THRESHOLD
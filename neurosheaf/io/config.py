"""Production configuration for Global Section (H⁰) tracking.

This module provides comprehensive configuration management for numerically
robust H⁰ persistence tracking with transport-informed evolution.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class H0Config:
    """Production configuration for H⁰ persistence tracking.
    
    This configuration ensures numerical stability at machine precision
    with comprehensive safeguards and error recovery mechanisms.
    
    Mathematical Parameters:
    - c_in: Zero entry threshold multiplier (τ_in = c_in × √ε × ||δ̃||₂)
    - c_keep: RRQR keep threshold multiplier (τ_keep = c_keep × √ε × ||Y||₂)  
    - gap: Hysteresis gap factor (τ_out = gap × τ_in)
    
    Stability Controls:
    - confirm_steps: Two-step confirmation window to prevent noise flickering
    - mass_floor_factor: Floor for tiny masses to prevent division by zero
    - max_cholesky_retries: Automatic regularization retry attempts
    """
    
    # Numerical precision (MANDATORY for stability at λ ≈ 10⁻¹²)
    dtype: str = "float64"
    seed: int = 123
    deterministic: bool = True
    
    # Core threshold parameters (from production plan)
    # UPDATED: Adjusted for normalized Laplacian with eigenvalues in [0, 2]
    c_in: float = 100.0              # Zero entry threshold multiplier
    gap: float = 2.0                 # Hysteresis gap (τ_out = gap × τ_in) - reduced for stability
    c_keep: float = 200.0            # RRQR keep threshold multiplier
    margin: int = 12                 # Extra singular values to compute
    
    # NEW: Normalized Laplacian specific parameters
    use_normalized_thresholds: bool = False  # Auto-detect normalized Laplacian
    c_in_normalized: float = 20.0   # Adjusted multiplier for normalized case
    gap_normalized: float = 2.0     # Tighter gap for normalized eigenvalues
    
    # Stability controls (critical for transport noise)
    confirm_steps: int = 2           # Two-step confirmation window
    mass_floor_factor: float = 1e-12 # Floor for tiny masses (×median)
    max_cholesky_retries: int = 3    # Cholesky regularization retries
    
    # Performance tuning
    dense_threshold: int = 5000      # Switch to sparse methods above this size
    spectral_norm_iterations: int = 8 # Power iteration steps for ||δ̃||₂
    cache_cholesky: bool = True      # Cache factorizations between steps
    
    # Error recovery parameters
    transport_noise_threshold: float = 0.1    # Detect excessive transport noise
    transport_stabilization_alpha: float = 0.05  # Convex blend for noise mitigation
    cholesky_regularization_start: float = 1e-12  # Initial regularization
    cholesky_regularization_factor: float = 10.0  # Multiplicative increase
    
    # Observability and debugging
    log_level: int = logging.INFO
    save_diagnostics: bool = True
    validate_certificates: bool = True
    log_thresholds: bool = True
    
    # Algorithm selection thresholds
    sparse_density_threshold: float = 0.1    # Switch to sparse for density < 0.1
    randomized_svd_threshold: int = 10000    # Use randomized SVD for n > threshold
    max_svd_compute: int = 2000              # Maximum number of singular values to compute for large nullity cases
    
    # SVD numerical rank detection
    svd_zero_tol_scale: float = 10.0        # Multiplier for SVD numerical rank threshold (τ = scale × max(m,n) × ε × σ_max)
    
    # RRQR threshold behavior control
    use_neural_network_threshold: bool = False  # Whether to use aggressive neural network threshold for RRQR
    
    @property
    def machine_eps(self) -> float:
        """Machine epsilon for the configured data type."""
        if self.dtype == "float64":
            return np.finfo(np.float64).eps
        elif self.dtype == "float32":
            return np.finfo(np.float32).eps
        else:
            return np.finfo(float).eps
    
    @property
    def sqrt_eps(self) -> float:
        """Square root of machine epsilon for threshold scaling."""
        return np.sqrt(self.machine_eps)
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.dtype not in ["float64", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be 'float64' or 'float32'")
        
        if self.c_in <= 0:
            raise ValueError(f"c_in must be positive, got {self.c_in}")
        
        if self.c_keep <= 0:
            raise ValueError(f"c_keep must be positive, got {self.c_keep}")
            
        if self.gap <= 1.0:
            raise ValueError(f"gap must be > 1.0 for hysteresis, got {self.gap}")
        
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
            
        if self.confirm_steps < 1:
            raise ValueError(f"confirm_steps must be >= 1, got {self.confirm_steps}")
        
        if not (0.0 < self.mass_floor_factor <= 1e-6):
            raise ValueError(f"mass_floor_factor must be in (0, 1e-6], got {self.mass_floor_factor}")
            
        if not (100 <= self.max_svd_compute <= 10000):
            raise ValueError(f"max_svd_compute must be between 100 and 10000, got {self.max_svd_compute}")
    
    def create_step_thresholds(self, spectral_norm: float, is_normalized: bool = None) -> dict:
        """Create step-specific thresholds based on spectral norm.
        
        Adaptive thresholds for normalized vs unnormalized Laplacians:
        - Normalized Laplacian: eigenvalues in [0, 2], use tighter thresholds
        - Unnormalized Laplacian: wider spectrum, use standard thresholds
        
        Args:
            spectral_norm: ||δ̃||₂ for current step
            is_normalized: Whether using normalized Laplacian (auto-detect if None)
            
        Returns:
            Dictionary with tau_in, tau_out, and related thresholds
        """
        # Auto-detect normalized Laplacian from spectral norm
        if is_normalized is None:
            is_normalized = self.use_normalized_thresholds or (spectral_norm <= 2.0)
        
        # Use appropriate thresholds based on normalization
        if is_normalized:
            c_in_used = self.c_in_normalized
            gap_used = self.gap_normalized
        else:
            c_in_used = self.c_in
            gap_used = self.gap
        
        tau_in = c_in_used * self.sqrt_eps * spectral_norm
        tau_out = gap_used * tau_in
        
        return {
            'tau_in': tau_in,
            'tau_out': tau_out,
            'spectral_norm': spectral_norm,
            'c_in_used': c_in_used,
            'gap_used': gap_used,
            'sqrt_eps': self.sqrt_eps,
            'is_normalized': is_normalized
        }
    
    @classmethod
    def for_neural_networks(cls, 
                           sensitivity: str = "high", 
                           **kwargs) -> 'H0Config':
        """Create H0 configuration optimized for neural network persistence analysis.
        
        Neural networks rarely have exact rank deficiencies, so we need much more
        relaxed thresholds to detect meaningful global sections.
        
        Args:
            sensitivity: Detection sensitivity level
                - "low": Very strict thresholds, only detects very clear rank deficiencies
                - "medium": Balanced thresholds for typical neural networks  
                - "high": Relaxed thresholds, more sensitive to small singular values
                - "maximum": Very relaxed, detects even tiny apparent rank deficiencies
            **kwargs: Additional configuration overrides
            
        Returns:
            H0Config optimized for neural networks
        """
        sensitivity_configs = {
            "low": {
                "c_in": 500.0,
                "gap": 2.5,
                "c_keep": 1000.0,
                "margin": 20,
                "use_neural_network_threshold": True
            },
            "medium": {
                "c_in": 1000.0,
                "gap": 2.0,
                "c_keep": 2000.0,
                "margin": 25,
                "use_neural_network_threshold": True
            },
            "high": {
                "c_in": 5000.0,
                "gap": 2.0,
                "c_keep": 10000.0,
                "margin": 30,
                "use_neural_network_threshold": True
            },
            "maximum": {
                "c_in": 20000.0,
                "gap": 1.5,
                "c_keep": 50000.0,
                "margin": 50,
                "use_neural_network_threshold": True
            }
        }
        
        if sensitivity not in sensitivity_configs:
            raise ValueError(f"Invalid sensitivity '{sensitivity}'. Must be one of: {list(sensitivity_configs.keys())}")
        
        config_params = sensitivity_configs[sensitivity]
        config_params.update(kwargs)  # Allow overrides
        
        return cls(**config_params)
    
    @classmethod
    def for_normalized_laplacian(cls, **kwargs) -> 'H0Config':
        """Create H0 configuration optimized for normalized Laplacian.
        
        Normalized Laplacians have eigenvalues bounded in [0, 2], requiring
        different threshold parameters than unnormalized Laplacians.
        
        Args:
            **kwargs: Additional configuration overrides
            
        Returns:
            H0Config optimized for normalized Laplacian
        """
        config_params = {
            "use_normalized_thresholds": True,
            "c_in": 20.0,              # Much smaller multiplier for bounded spectrum
            "c_in_normalized": 20.0,   # Explicit normalized setting
            "gap": 2.0,                # Tighter gap for stability
            "gap_normalized": 2.0,     # Explicit normalized setting
            "c_keep": 50.0,            # Adjusted for normalized case
            "margin": 15,              # Slightly larger margin for safety
            "confirm_steps": 2,        # Keep confirmation requirement
            "mass_floor_factor": 1e-12
        }
        config_params.update(kwargs)  # Allow overrides
        
        return cls(**config_params)
    
    def create_rrqr_threshold(self, Y_norm: float) -> float:
        """Create RRQR keep threshold based on Y matrix norm.
        
        Two threshold strategies:
        1. Standard (default): Mathematically principled threshold for sheaf analysis
        2. Neural network: Aggressive threshold for neural network death detection
        
        Args:
            Y_norm: ||Y||₂ where Y = F @ Q_prev
            
        Returns:
            RRQR keep threshold τ_keep
        """
        # Standard mathematical threshold: τ_keep = c_keep × √ε × ||Y||₂
        standard_threshold = self.c_keep * self.sqrt_eps * Y_norm
        
        if self.use_neural_network_threshold:
            # NEURAL NETWORK DEATH DETECTION FIX:
            # For neural networks, transport-induced generators have singular values 
            # in range [0.01, 0.1] but standard threshold ~ 1e-06. We need much more 
            # aggressive thresholds to detect transport-induced deaths.
            neural_network_threshold = 1.1 * Y_norm  # 110% of the transported generator norm
            
            # Use the larger threshold (more aggressive detection)
            return max(standard_threshold, neural_network_threshold)
        else:
            # Default: Use standard mathematical threshold for sheaf analysis
            return standard_threshold


# Default production configuration
DEFAULT_H0_CONFIG = H0Config()

# Validate default configuration on import
DEFAULT_H0_CONFIG.validate()


# Alternative configurations for different use cases

@dataclass(frozen=True)
class StrictH0Config(H0Config):
    """Stricter configuration for high-precision requirements."""
    c_in: float = 200.0              # More conservative zero detection
    c_keep: float = 400.0            # More conservative RRQR keeping  
    confirm_steps: int = 3           # Longer confirmation window
    mass_floor_factor: float = 1e-14 # Stricter mass floor
    validate_certificates: bool = True # Always validate


@dataclass(frozen=True)  
class FastH0Config(H0Config):
    """Faster configuration with reduced precision for development."""
    c_in: float = 50.0               # More aggressive zero detection
    c_keep: float = 100.0            # More aggressive RRQR keeping
    margin: int = 6                  # Fewer extra singular values
    spectral_norm_iterations: int = 4 # Fewer power iteration steps
    validate_certificates: bool = False # Skip validation for speed


@dataclass(frozen=True)
class DebugH0Config(H0Config):
    """Debug configuration with extensive logging and validation."""
    log_level: int = logging.DEBUG
    save_diagnostics: bool = True
    validate_certificates: bool = True
    log_thresholds: bool = True
    deterministic: bool = True       # Ensure reproducibility for debugging
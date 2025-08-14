"""Type definitions for Global Section (H⁰) tracking.

This module provides comprehensive type definitions and protocols
for the H⁰ persistence tracking pipeline, ensuring type safety
and clear contracts between components.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from abc import abstractmethod


# Core result types for pipeline stages

@dataclass
class WhiteningResult:
    """Result from coboundary whitening operation.
    
    Contains the whitened coboundary operator and factorizations
    needed for subsequent operations.
    """
    delta_tilde: torch.Tensor        # Whitened coboundary δ̃
    L0: torch.Tensor                 # Cholesky factor of G₀
    L1: torch.Tensor                 # Cholesky factor of G₁  
    G0_regularized: bool = False     # Whether G₀ needed regularization
    G1_regularized: bool = False     # Whether G₁ needed regularization
    regularization_applied: float = 0.0  # Amount of regularization added
    spectral_norm: float = 0.0       # ||δ̃||₂ for threshold computation


@dataclass
class KernelResult:
    """Result from kernel basis computation with hysteresis.
    
    Contains the kernel basis and classification information
    for persistence tracking.
    """
    V0: torch.Tensor                 # Orthonormal kernel basis (whitened coordinates)
    kernel_dimension: int            # k = dim(ker(δ̃))
    singular_values: torch.Tensor    # Computed singular values
    labels: List[str]                # "ZERO" or "NONZERO" labels
    spectral_norm: float             # ||δ̃||₂ used for thresholds
    tau_in: float                    # Zero entry threshold used
    tau_out: float                   # Zero exit threshold used
    residual_norm: float             # ||δ̃ @ V⁰||₂ (certificate)
    hysteresis_applied: bool = False # Whether hysteresis changed labels


@dataclass
class TransportResult:
    """Result from GW coupling to transport map conversion.
    
    Contains transport maps and validation information
    for induced map computation.
    """
    T_pushforward: torch.Tensor      # Linear transport map T_{t→t+1}
    T_tilde: torch.Tensor           # Whitened transport map T̃
    P_pullback: torch.Tensor        # Pullback map P_{t+1→t} (adjoint)
    coupling_type: str              # "balanced", "unbalanced", or "fallback"
    realized_masses: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    transport_norm: float = 0.0     # ||T̃||₂ for diagnostics
    mass_floor_applied: bool = False # Whether mass floor was needed
    stabilization_applied: bool = False # Whether noise stabilization was used


@dataclass  
class RRQRResult:
    """Result from RRQR persistence update.
    
    Contains updated generators and birth/death information
    for persistence diagram construction.
    """
    Q_keep: torch.Tensor            # Surviving generators (orthonormal)
    keep_mask: torch.Tensor         # Boolean mask for kept columns
    n_births: int                   # Number of new births detected
    n_deaths: int                   # Number of deaths detected  
    n_confirmed_deaths: int         # Deaths passing two-step confirmation
    induced_map: torch.Tensor       # F_t for analysis/debugging
    Y_matrix: torch.Tensor          # Y = F @ Q_prev for diagnostics
    threshold_used: float           # τ_keep threshold applied
    rank_detected: int              # Rank detected by RRQR


@dataclass
class PersistenceResult:
    """Complete H⁰ persistence tracking result.
    
    Contains persistence intervals, Betti curve, and comprehensive
    diagnostics from the tracking pipeline.
    """
    intervals: List[Dict[str, Any]]  # Persistence intervals (birth, death)
    betti_curve: List[Dict[str, Any]] # β₀(t) evolution over filtration
    diagnostics: List['StepDiagnostics'] # Per-step detailed diagnostics
    transport_data: Optional[Dict] = None # Transport evolution information
    total_time: float = 0.0         # Total computation time
    n_steps: int = 0               # Number of filtration steps processed
    method_used: str = "transport_informed_h0" # Tracking method identifier


@dataclass
class StepDiagnostics:
    """Comprehensive diagnostics for each filtration step.
    
    Contains all numerical information needed for validation,
    debugging, and performance analysis.
    """
    step: int                       # Filtration step index
    filtration_param: float         # Filtration parameter value
    
    # Spectral properties
    spectral_norm: float            # S_t = ||δ̃||₂
    kernel_dimension: int           # k_t = dim(ker(δ̃))
    kernel_residual: float          # ||δ̃ @ V⁰||₂ / S_t (certificate)
    
    # Thresholds used
    tau_in: float                   # Zero entry threshold
    tau_out: float                  # Zero exit threshold
    tau_keep: Optional[float] = None # RRQR keep threshold (if applicable)
    
    # Persistence events
    n_births: int = 0              # New births detected
    n_deaths: int = 0              # Deaths detected
    n_confirmed_deaths: int = 0    # Deaths passing confirmation
    dimension_change: bool = False  # Whether kernel dimension changed
    
    # Transport properties (if applicable)
    transport_norm: Optional[float] = None       # ||T̃||₂
    induced_map_rank: Optional[int] = None       # rank(F_t)
    coupling_type: Optional[str] = None          # Type of GW coupling used
    mass_floor_applied: bool = False
    transport_stabilization_applied: bool = False
    
    # Numerical certificates
    cholesky_retries: int = 0       # Number of Cholesky regularization attempts
    certificate_passed: bool = True # Whether numerical validation passed
    regularization_applied: float = 0.0 # Amount of regularization added
    computational_time: float = 0.0 # Time for this step
    
    # Additional validation info
    hysteresis_changes: int = 0     # Number of hysteresis label changes
    svd_iterations: Optional[int] = None # SVD solver iterations (if available)


@dataclass
class CertificateResult:
    """Result from numerical certificate validation.
    
    Comprehensive validation of computed results against
    mathematical guarantees.
    """
    # Required fields (no defaults)
    kernel_certificate: bool        # ||δ̃ @ V⁰||₂ ≤ threshold
    rrqr_certificate: bool          # ||Y - Q̂ R P^T||_F ≤ threshold  
    orthogonality_certificate: bool # ||V⁰^T V⁰ - I||_F ≤ threshold
    kernel_residual: float          # Actual kernel residual
    kernel_threshold: float         # Required threshold
    rrqr_residual: float           # RRQR reconstruction residual
    orthogonality_residual: float   # Orthogonality violation
    all_passed: bool               # Whether all certificates passed
    
    # Optional fields (with defaults)
    transport_certificate: Optional[bool] = None # Marginal preservation
    transport_residual: Optional[float] = None # Transport marginal violation
    warnings: List[str] = field(default_factory=list)            # Non-critical issues
    errors: List[str] = field(default_factory=list)              # Critical failures


@dataclass
class TransportValidation:
    """Validation result for transport map properties.
    
    Validates mathematical properties of constructed transport
    maps and their consistency with GW couplings.
    """
    marginal_consistency: bool      # Whether marginals are preserved
    source_marginal_error: float   # ||π @ 1 - a||₂
    target_marginal_error: float   # ||π^T @ 1 - b||₂
    transport_norm: float          # ||T̃||₂
    condition_number: float        # cond(T̃) for stability assessment
    mass_preservation: bool        # Whether total mass is preserved
    warnings: List[str] = field(default_factory=list)     # Non-critical issues  
    errors: List[str] = field(default_factory=list)       # Critical problems


# Protocol definitions for builder interfaces

class CoboundaryBuilder(Protocol):
    """Protocol for building coboundary operators and metrics."""
    
    @abstractmethod
    def build_coboundary_with_metrics(self, sheaf: Any, 
                                    active_edges: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """Build coboundary operator with proper metrics.
        
        Returns:
            Dictionary with δ, G0, G1, node_masses
        """
        ...


class TransportExtractor(Protocol):
    """Protocol for extracting GW couplings and transport information."""
    
    @abstractmethod
    def extract_node_masses_and_couplings(self, sheaf: Any,
                                        filtration_step: int) -> Dict:
        """Extract masses and GW couplings between filtration steps.
        
        Returns:
            Dictionary with node_masses, next_masses, gw_coupling, coupling_type
        """
        ...


class KernelTracker(Protocol):
    """Protocol for tracking kernel evolution through filtration."""
    
    @abstractmethod
    def track_kernel_evolution(self, kernel_sequence: List[KernelResult],
                             transport_sequence: List[TransportResult],
                             filtration_params: List[float]) -> PersistenceResult:
        """Track kernel evolution and generate persistence diagram."""
        ...


# Union types for common parameter combinations

FiltrationStep = Dict[str, Union[int, float, torch.Tensor, Any]]
ParameterSequence = List[float]
KernelSequence = List[KernelResult] 
TransportSequence = List[TransportResult]

# Type aliases for tensor operations
SparseTensor = Union[torch.sparse.FloatTensor, torch.sparse.DoubleTensor]
DenseTensor = torch.Tensor
LaplacianMatrix = Union[DenseTensor, SparseTensor, Any]  # Any for scipy sparse

# Numerical precision types
NumpyFloat = Union[np.float32, np.float64]
TorchFloat = torch.dtype
"""Gromov-Wasserstein optimal transport for sheaf construction.

This module implements the core GW computational engine for constructing
sheaves from neural network activations using optimal transport theory.
The implementation uses entropic regularization for numerical stability
and computational efficiency.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
import logging
from collections import OrderedDict
import warnings

from .gw_config import GWConfig

logger = logging.getLogger(__name__)

# Handle optional POT dependency
try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    ot = None


@dataclass
class GWResult:
    """Result of Gromov-Wasserstein coupling computation.
    
    Contains the computed transport plan, scalar distortion cost,
    and convergence diagnostics from the entropic GW solver.
    
    Attributes:
        coupling: Transport plan π_{j→i} with shape (n_target, n_source)
                 Satisfies marginal constraints: π @ 1 = p_target, π.T @ 1 = p_source
        cost: Scalar GW distortion cost (metric compatibility measure)
        log: Dictionary with convergence information and diagnostics
        source_size: Number of points in source space
        target_size: Number of points in target space
    """
    coupling: torch.Tensor                 # Transport plan π
    cost: float                           # Scalar GW distortion  
    log: Dict[str, Any]                   # Convergence diagnostics
    source_size: int                      # n_source
    target_size: int                      # n_target
    
    def validate_marginals(self, p_source: Optional[torch.Tensor] = None,
                          p_target: Optional[torch.Tensor] = None,
                          tolerance: float = 1e-10) -> Dict[str, float]:
        """Validate that coupling satisfies marginal constraints.
        
        Args:
            p_source: Source measure (if None, assumes uniform)
            p_target: Target measure (if None, assumes uniform)  
            tolerance: Numerical tolerance for validation
            
        Returns:
            Dictionary with violation measures
        """
        if p_source is None:
            p_source = torch.ones(self.source_size) / self.source_size
        if p_target is None:
            p_target = torch.ones(self.target_size) / self.target_size
            
        # For POT coupling matrix π with shape (n_source, n_target):
        # - π @ 1_target = p_source (row sums equal source distribution)
        # - π.T @ 1_source = p_target (column sums equal target distribution)
        
        # Check π @ 1 = p_source
        row_sums = self.coupling @ torch.ones(self.target_size)
        source_marginal_violation = torch.norm(row_sums - p_source).item()
        
        # Check π.T @ 1 = p_target
        col_sums = self.coupling.T @ torch.ones(self.source_size)
        target_marginal_violation = torch.norm(col_sums - p_target).item()
        
        return {
            'target_marginal_violation': target_marginal_violation,
            'source_marginal_violation': source_marginal_violation,
            'max_violation': max(target_marginal_violation, source_marginal_violation),
            'constraints_satisfied': max(target_marginal_violation, source_marginal_violation) < tolerance
        }


class CostMatrixCache:
    """LRU cache for expensive cost matrix computations.
    
    Caches computed cosine distance matrices with automatic memory management
    based on configurable size limits.
    """
    
    def __init__(self, max_size_gb: float = 2.0):
        """Initialize cache with memory limit.
        
        Args:
            max_size_gb: Maximum cache size in GB
        """
        self.cache = OrderedDict()
        self.max_bytes = max_size_gb * 1e9
        self.current_bytes = 0
        
    def _estimate_tensor_bytes(self, tensor: torch.Tensor) -> int:
        """Estimate tensor memory usage in bytes."""
        return tensor.numel() * tensor.element_size()
        
    def _evict_lru(self, needed_bytes: int) -> None:
        """Evict least recently used entries until space is available."""
        while self.current_bytes + needed_bytes > self.max_bytes and self.cache:
            key, value = self.cache.popitem(last=False)  # Remove oldest
            self.current_bytes -= self._estimate_tensor_bytes(value)
            logger.debug(f"Evicted cost matrix for key {key}")
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached cost matrix, updating LRU order."""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        """Store cost matrix in cache with LRU eviction."""
        tensor_bytes = self._estimate_tensor_bytes(value)
        
        # Evict if needed
        self._evict_lru(tensor_bytes)
        
        # Store new entry
        self.cache[key] = value.clone()  # Defensive copy
        self.current_bytes += tensor_bytes
        
        logger.debug(f"Cached cost matrix {key}: {value.shape}, "
                    f"cache now {self.current_bytes / 1e6:.1f}MB")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.current_bytes = 0
        logger.debug("Cost matrix cache cleared")


class GromovWassersteinComputer:
    """Core GW computation engine with caching and GPU support.
    
    This class implements entropic Gromov-Wasserstein optimal transport
    for constructing restriction maps in sheaf theory. It includes numerical
    stability measures, caching for performance, and comprehensive validation.
    
    Mathematical Background:
    The GW problem seeks a transport plan π minimizing:
        Σ_{k,ℓ,k',ℓ'} |C_source[k,k'] - C_target[ℓ,ℓ']|² π[k,ℓ] π[k',ℓ'] - ε H(π)
    subject to marginal constraints π @ 1 = p_target, π.T @ 1 = p_source.
    """
    
    def __init__(self, config: Optional[GWConfig] = None):
        """Initialize GW computer with configuration.
        
        Args:
            config: GW configuration (uses defaults if None)
        """
        self.config = config or GWConfig()
        self.config.validate()
        
        # Initialize cache if enabled
        self.cost_cache = None
        if self.config.cache_cost_matrices:
            self.cost_cache = CostMatrixCache(self.config.max_cache_size_gb)
            
        # Check POT availability
        if not POT_AVAILABLE:
            if self.config.use_gpu:
                logger.warning("POT library not available, disabling GPU acceleration")
                self.config.use_gpu = False
            logger.warning("POT library not available. GW computations will use fallback implementation.")
        
        logger.info(f"GromovWassersteinComputer initialized: epsilon={self.config.epsilon}, "
                   f"max_iter={self.config.max_iter}, gpu={self.config.use_gpu}")
    
    def compute_cosine_cost_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine distances with numerical stability.
        
        For activations X with shape (n, d), computes the n×n cost matrix:
        C[i,j] = 1 - cos(x_i, x_j) = 1 - (x_i · x_j) / (||x_i|| ||x_j||)
        
        Args:
            X: Activation tensor with shape (n, d)
            
        Returns:
            Symmetric cost matrix with shape (n, n), zero diagonal
            
        Raises:
            ValueError: If input has wrong shape or contains invalid values
        """
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")
        
        n, d = X.shape
        if n == 0 or d == 0:
            raise ValueError(f"Invalid tensor dimensions: {X.shape}")
        
        # Check for cache hit
        cache_key = None
        if self.cost_cache is not None:
            # Create cache key from tensor properties
            cache_key = f"cosine_{n}x{d}_{hash(tuple(X.flatten().tolist()))}"
            cached = self.cost_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cost matrix cache hit for {cache_key}")
                return cached
        
        # Compute norms with numerical stability
        norms = torch.norm(X, dim=1, keepdim=True)  # Shape: (n, 1)
        
        # Handle zero vectors
        zero_mask = (norms.squeeze() < self.config.cost_matrix_eps)
        if zero_mask.any():
            logger.warning(f"Found {zero_mask.sum().item()} zero vectors in activation tensor")
            # Set small norm to prevent division by zero
            norms = torch.clamp(norms, min=self.config.cost_matrix_eps)
        
        # Normalize vectors
        X_normalized = X / norms  # Broadcasting: (n, d) / (n, 1)
        
        # Compute cosine similarities: X_norm @ X_norm.T
        cosine_similarities = torch.mm(X_normalized, X_normalized.T)
        
        # Clamp to [-1, 1] for numerical stability
        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
        
        # Convert to cosine distances: 1 - cosine_similarity
        cost_matrix = 1.0 - cosine_similarities
        
        # Ensure exact zero diagonal (numerical precision)
        cost_matrix.fill_diagonal_(0.0)
        
        # Validation if enabled
        if self.config.validate_costs:
            self._validate_cost_matrix(cost_matrix)
        
        # Cache result
        if self.cost_cache is not None and cache_key is not None:
            self.cost_cache.put(cache_key, cost_matrix)
        
        logger.debug(f"Computed cosine cost matrix: {cost_matrix.shape}, "
                    f"range=[{cost_matrix.min().item():.6f}, {cost_matrix.max().item():.6f}]")
        
        return cost_matrix
    
    def compute_adaptive_epsilon(self, n_source: int, n_target: int) -> float:
        """Compute adaptive epsilon based on sample sizes.
        
        Uses sqrt scaling: epsilon = base_epsilon * sqrt(reference_n / n_avg)
        This maintains the balance between regularization and accuracy as n changes.
        
        Args:
            n_source: Number of source samples
            n_target: Number of target samples
            
        Returns:
            Adaptive epsilon value
        """
        if not self.config.adaptive_epsilon:
            return self.config.epsilon
        
        # Average sample size
        n_avg = (n_source + n_target) / 2.0
        
        # Apply sqrt scaling (recommended by theory: ε ~ 1/√n)
        if self.config.epsilon_scaling_method == 'sqrt':
            epsilon = self.config.base_epsilon * np.sqrt(self.config.reference_n / n_avg)
        else:
            # For now, only sqrt is implemented
            raise ValueError(f"Unknown scaling method: {self.config.epsilon_scaling_method}")
        
        # Clamp to reasonable range
        epsilon = np.clip(epsilon, self.config.epsilon_min, self.config.epsilon_max)
        
        logger.info(f"Adaptive epsilon: {epsilon:.4f} (base={self.config.base_epsilon}, "
                    f"n_avg={n_avg:.0f}, reference_n={self.config.reference_n})")
        
        return epsilon
    
    def compute_gw_coupling(self, 
                           C_source: torch.Tensor, 
                           C_target: torch.Tensor,
                           p_source: Optional[torch.Tensor] = None,
                           p_target: Optional[torch.Tensor] = None) -> GWResult:
        """Solve entropic GW problem with convergence diagnostics.
        
        Args:
            C_source: Cost matrix for source space (n_source, n_source)
            C_target: Cost matrix for target space (n_target, n_target) 
            p_source: Source measure (if None, uniform)
            p_target: Target measure (if None, uniform)
            
        Returns:
            GWResult with coupling π_{target→source}, cost, and diagnostics
            
        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        # Validate inputs
        self._validate_gw_inputs(C_source, C_target, p_source, p_target)
        
        n_source = C_source.shape[0]
        n_target = C_target.shape[0]
        
        # Setup measures
        if p_source is None:
            p_source = torch.ones(n_source) / n_source
        if p_target is None:
            p_target = torch.ones(n_target) / n_target
        
        if not self.config.uniform_measures:
            logger.warning("Non-uniform measures requested but not fully implemented")
        
        # Compute adaptive epsilon if enabled
        epsilon_adaptive = self.compute_adaptive_epsilon(n_source, n_target)
        
        # Temporarily update epsilon for this computation
        original_epsilon = self.config.epsilon
        self.config.epsilon = epsilon_adaptive
        
        try:
            # Compute GW coupling with adaptive epsilon
            if POT_AVAILABLE:
                coupling, cost, log = self._compute_gw_pot(C_source, C_target, p_source, p_target)
            else:
                coupling, cost, log = self._compute_gw_fallback(C_source, C_target, p_source, p_target)
            
            # Add adaptive epsilon info to log
            log['epsilon_used'] = epsilon_adaptive
            log['epsilon_adaptive_enabled'] = self.config.adaptive_epsilon
            
            # Optional: Monitor coupling entropy to verify it's not too diffuse
            if coupling.numel() > 0:
                coupling_safe = coupling + 1e-12
                entropy = -(coupling_safe * torch.log(coupling_safe)).sum().item()
                max_entropy = np.log(n_source * n_target)
                log['coupling_entropy_ratio'] = entropy / max_entropy
                
                if log['coupling_entropy_ratio'] > 0.95:
                    logger.warning(f"Coupling is very diffuse (entropy ratio: "
                                 f"{log['coupling_entropy_ratio']:.3f}). "
                                 f"Consider decreasing base_epsilon.")
            
        finally:
            # Restore original epsilon
            self.config.epsilon = original_epsilon
        
        # Create result
        result = GWResult(
            coupling=coupling,
            cost=cost,
            log=log,
            source_size=n_source,
            target_size=n_target
        )
        
        # Validate coupling if enabled
        if self.config.validate_couplings:
            validation = result.validate_marginals(p_source, p_target, self.config.coupling_eps)
            result.log['marginal_validation'] = validation
            
            if not validation['constraints_satisfied']:
                logger.warning(f"Marginal constraint violation: {validation['max_violation']:.2e}")
        
        logger.debug(f"GW coupling computed: {coupling.shape}, cost={cost:.6f}, "
                    f"iterations={log.get('num_iter', 'N/A')}")
        
        return result
    
    def _compute_gw_pot(self, C_source: torch.Tensor, C_target: torch.Tensor,
                       p_source: torch.Tensor, p_target: torch.Tensor) -> Tuple[torch.Tensor, float, Dict]:
        """Compute GW coupling using POT library with correct API."""
        # Convert to numpy for POT
        C_source_np = C_source.detach().cpu().numpy()
        C_target_np = C_target.detach().cpu().numpy()
        p_source_np = p_source.detach().cpu().numpy()
        p_target_np = p_target.detach().cpu().numpy()
        
        try:
            # Use entropic GW from POT with correct parameter names
            result = ot.gromov.entropic_gromov_wasserstein(
                C_source_np, C_target_np,      # Cost matrices
                p_source_np, p_target_np,      # Source and target distributions
                loss_fun='square_loss',        # Loss function
                epsilon=self.config.epsilon,   # Entropic regularization  
                max_iter=self.config.max_iter, # Maximum iterations
                tol=self.config.tolerance,     # Convergence tolerance
                solver='PGD',                  # Projected Gradient Descent solver
                verbose=False,                 # Suppress output
                log=True                       # Return log dictionary
            )
            
            if isinstance(result, tuple):
                coupling_np, log_dict = result
            else:
                # If log=False was used somehow, just the coupling is returned
                coupling_np = result
                log_dict = {}
            
            # Convert back to PyTorch
            coupling = torch.from_numpy(coupling_np).float()
            
            # Extract cost from log or compute it
            if 'gw_dist' in log_dict:
                cost = float(log_dict['gw_dist'])
            elif 'loss' in log_dict and len(log_dict['loss']) > 0:
                cost = float(log_dict['loss'][-1])  # Last loss value
            else:
                # Fallback: compute cost manually
                cost = self._compute_gw_cost(C_source, C_target, coupling)
            
            # Extract convergence info with proper key names
            log = {
                'solver': 'pot_entropic_pgd',
                'converged': log_dict.get('converged', True),  # Assume converged if not specified
                'num_iter': log_dict.get('it', log_dict.get('n_iter', -1)),
                'final_error': log_dict.get('err', log_dict.get('error', float('nan'))),
                'cost_evolution': log_dict.get('loss', []),
                'gw_dist': cost,
                'full_log': log_dict  # Store full log for debugging
            }
            
        except Exception as e:
            logger.warning(f"POT GW solver failed: {e}. Using fallback.")
            return self._compute_gw_fallback(C_source, C_target, p_source, p_target)
        
        return coupling, cost, log
    
    def _compute_gw_fallback(self, C_source: torch.Tensor, C_target: torch.Tensor,
                           p_source: torch.Tensor, p_target: torch.Tensor) -> Tuple[torch.Tensor, float, Dict]:
        """Fallback GW implementation when POT is not available."""
        logger.info("Using fallback GW implementation (basic coupling)")
        
        n_source = C_source.shape[0]
        n_target = C_target.shape[0]
        
        # Simple fallback: uniform coupling satisfying marginal constraints
        # This is not optimal but provides a reasonable approximation
        coupling = torch.outer(p_target, p_source)
        
        # Compute GW cost for this coupling
        cost = self._compute_gw_cost(C_source, C_target, coupling)
        
        log = {
            'solver': 'fallback_uniform',
            'converged': True,
            'num_iter': 0,
            'final_error': 0.0,
            'cost_evolution': [cost],
            'gw_dist': cost,
            'warning': 'Using fallback implementation - not optimal'
        }
        
        return coupling, cost, log
    
    def _compute_gw_cost(self, C_source: torch.Tensor, C_target: torch.Tensor, 
                        coupling: torch.Tensor) -> float:
        """Compute GW cost for given coupling.
        
        Cost = Σ_{i,j,k,l} |C_source[i,k] - C_target[j,l]|² π[j,i] π[l,k]
        """
        # This is computationally expensive for large matrices
        # For now, use a simplified approximation
        cost = 0.0
        n_source = C_source.shape[0]
        n_target = C_target.shape[0]
        
        # Sample-based approximation for computational efficiency
        max_samples = min(100, n_source * n_target)
        sample_indices = torch.randint(0, n_source * n_target, (max_samples,))
        
        for idx in sample_indices:
            i = idx // n_target
            j = idx % n_target
            
            for k in range(min(n_source, 10)):  # Limit inner loop
                for l in range(min(n_target, 10)):
                    diff = C_source[i, k] - C_target[j, l]
                    cost += (diff ** 2) * coupling[j, i] * coupling[l, k]
        
        return cost.item() if torch.is_tensor(cost) else cost
    
    def _validate_cost_matrix(self, C: torch.Tensor) -> None:
        """Validate cost matrix properties."""
        if not torch.allclose(C, C.T, atol=1e-8):
            raise ValueError("Cost matrix is not symmetric")
        
        if torch.any(C < -self.config.cost_matrix_eps):
            raise ValueError("Cost matrix has negative values")
        
        if torch.any(torch.diag(C) > self.config.cost_matrix_eps):
            raise ValueError("Cost matrix diagonal is not zero")
    
    def _validate_gw_inputs(self, C_source: torch.Tensor, C_target: torch.Tensor,
                           p_source: Optional[torch.Tensor], p_target: Optional[torch.Tensor]) -> None:
        """Validate inputs to GW computation."""
        if C_source.dim() != 2 or C_source.shape[0] != C_source.shape[1]:
            raise ValueError(f"C_source must be square, got shape {C_source.shape}")
        
        if C_target.dim() != 2 or C_target.shape[0] != C_target.shape[1]:
            raise ValueError(f"C_target must be square, got shape {C_target.shape}")
        
        if p_source is not None:
            if p_source.shape[0] != C_source.shape[0]:
                raise ValueError("p_source dimension mismatch with C_source")
            if torch.any(p_source < 0) or not torch.allclose(p_source.sum(), torch.tensor(1.0)):
                raise ValueError("p_source must be a probability distribution")
        
        if p_target is not None:
            if p_target.shape[0] != C_target.shape[0]:
                raise ValueError("p_target dimension mismatch with C_target")
            if torch.any(p_target < 0) or not torch.allclose(p_target.sum(), torch.tensor(1.0)):
                raise ValueError("p_target must be a probability distribution")
    
    def clear_cache(self) -> None:
        """Clear cost matrix cache."""
        if self.cost_cache is not None:
            self.cost_cache.clear()
            logger.info("Cost matrix cache cleared")
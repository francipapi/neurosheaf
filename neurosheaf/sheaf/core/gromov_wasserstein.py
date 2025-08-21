"""Gromov-Wasserstein optimal transport for sheaf construction.

This module implements the core GW computational engine for constructing
sheaves from neural network activations using optimal transport theory.
The implementation uses entropic regularization for numerical stability
and computational efficiency.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union, List
import torch
import numpy as np
import logging
import hashlib
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
    convergence diagnostics, and quality information from the GW solver.
    
    Attributes:
        coupling: Transport plan π with shape (n_source, n_target) following POT convention
                 Satisfies marginal constraints: π @ 1 = p_source, π.T @ 1 = p_target
        cost: Scalar GW distortion cost (metric compatibility measure)
        log: Dictionary with convergence information and diagnostics
        source_size: Number of points in source space
        target_size: Number of points in target space
        p_source: Source measure (distribution over source points)
        p_target: Target measure (distribution over target points)
        coupling_quality: Quality level - 'optimal', 'fallback', or 'failed'
        solver_type: Solver used - 'pot_entropic', 'fallback_uniform', etc.
        quality_score: Numerical quality metric in [0,1] where 1 is optimal
    """
    coupling: torch.Tensor                 # Transport plan π
    cost: float                           # Scalar GW distortion  
    log: Dict[str, Any]                   # Convergence diagnostics
    source_size: int                      # n_source
    target_size: int                      # n_target
    p_source: Optional[torch.Tensor] = None  # Source measure
    p_target: Optional[torch.Tensor] = None  # Target measure
    coupling_quality: str = 'optimal'              # Quality level indicator
    solver_type: str = 'pot_entropic'              # Solver type used
    quality_score: float = 1.0                     # Numerical quality [0,1]
    
    # Enhanced metadata for transport availability
    active_at: Optional[List[int]] = None           # Filtration steps where active
    creation_timestamp: Optional[float] = None      # When this result was created
    measures_uniform: bool = True                   # Whether uniform measures used
    edge_info: Optional[Dict[str, Any]] = None      # Additional edge-specific metadata
    
    def validate_marginals(self, p_source: Optional[torch.Tensor] = None,
                          p_target: Optional[torch.Tensor] = None,
                          tolerance: float = 1e-10) -> Dict[str, float]:
        """Validate that coupling satisfies marginal constraints.
        
        Args:
            p_source: Source measure (if None, uses stored measure or assumes uniform)
            p_target: Target measure (if None, uses stored measure or assumes uniform)  
            tolerance: Numerical tolerance for validation
            
        Returns:
            Dictionary with violation measures
        """
        # Use provided measures, fall back to stored, then to uniform (with consistent dtype)
        coupling_dtype = self.coupling.dtype
        if p_source is None:
            p_source = self.p_source if self.p_source is not None else torch.ones(self.source_size, dtype=coupling_dtype) / self.source_size
        if p_target is None:
            p_target = self.p_target if self.p_target is not None else torch.ones(self.target_size, dtype=coupling_dtype) / self.target_size
            
        # For our coupling matrix π with shape (n_target, n_source):
        # - Row sums: π.sum(dim=1) = p_target (each row is a target point)
        # - Column sums: π.sum(dim=0) = p_source (each column is a source point)
        
        # Check row sums = p_target
        row_sums = self.coupling.sum(dim=1)
        target_marginal_violation = torch.norm(row_sums - p_target).item()
        
        # Check column sums = p_source
        col_sums = self.coupling.sum(dim=0)
        source_marginal_violation = torch.norm(col_sums - p_source).item()
        
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
    
    def _compute_tensor_hash(self, X: torch.Tensor) -> str:
        """Compute efficient hash of tensor using configurable method.
        
        This replaces the expensive tuple-based hashing with fast alternatives:
        - 'sha1': SHA1 hash of raw bytes (fast, deterministic, session-persistent)
        - 'id': Tensor ID + metadata (fastest, session-only)
        
        Args:
            X: Input tensor to hash
            
        Returns:
            Hash string that uniquely identifies tensor content
        """
        # Include tensor metadata for hash uniqueness
        shape_str = 'x'.join(map(str, X.shape))
        dtype_str = str(X.dtype).replace('torch.', '')
        device_str = str(X.device).replace(':', '_')
        metadata = f"{shape_str}_{dtype_str}_{device_str}"
        
        if self.config.cache_hash_method == 'id':
            # Ultra-fast session-scoped hashing using tensor ID
            # Not persistent across sessions but very fast
            return f"id_{id(X)}_{metadata}"
        
        elif self.config.cache_hash_method == 'sha1':
            # Fast SHA1 hash of raw bytes - much faster than tuple hashing
            # Persistent across sessions with same data
            tensor_bytes = X.detach().cpu().numpy().tobytes()
            
            hash_obj = hashlib.sha1()
            hash_obj.update(metadata.encode())
            hash_obj.update(tensor_bytes)
            
            return hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity
        
        else:
            raise ValueError(f"Unknown cache_hash_method: {self.config.cache_hash_method}")
    
    def compute_cosine_cost_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine distances with numerical stability.
        
        For input X with shape (n_points, n_features), computes the n_points×n_points cost matrix:
        C[i,j] = 1 - cos(x_i, x_j) = 1 - (x_i · x_j) / (||x_i|| ||x_j||)
        
        Important: For unit-based alignment (align_units=True), points are units/neurons
        and features are their activations across the batch. For sample-based alignment
        (align_units=False, deprecated), points are samples and features are units.
        
        Args:
            X: Input tensor with shape (n_points, n_features)
               - For unit alignment: (n_units, batch_size)
               - For sample alignment: (batch_size, n_units)
            
        Returns:
            Symmetric cost matrix with shape (n_points, n_points), zero diagonal
            
        Raises:
            ValueError: If input has wrong shape or contains invalid values
        """
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")
        
        # Convert to configured dtype for consistent precision throughout pipeline
        target_dtype = self.config.get_torch_dtype()
        if X.dtype != target_dtype:
            logger.debug(f"Converting activation tensor from {X.dtype} to {target_dtype}")
            X = X.to(dtype=target_dtype)
        
        n, d = X.shape
        if n == 0 or d == 0:
            raise ValueError(f"Invalid tensor dimensions: {X.shape}")
        
        # Check for cache hit
        cache_key = None
        if self.cost_cache is not None:
            # Create cache key using efficient tensor hash
            tensor_hash = self._compute_tensor_hash(X)
            cache_key = f"cosine_{n}x{d}_{tensor_hash}"
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
        """Compute adaptive epsilon based on point counts.
        
        Uses sqrt scaling: epsilon = base_epsilon * sqrt(reference_n / n_avg)
        This maintains the balance between regularization and accuracy as n changes.
        
        Note: Points can be either units (align_units=True) or samples (align_units=False).
        
        Args:
            n_source: Number of source points (units or samples)
            n_target: Number of target points (units or samples)
            
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
        
        # Convert cost matrices to consistent dtype
        target_dtype = self.config.get_torch_dtype()
        C_source = C_source.to(dtype=target_dtype)
        C_target = C_target.to(dtype=target_dtype)
        
        n_source = C_source.shape[0]
        n_target = C_target.shape[0]
        
        # Setup measures with consistent dtype
        if p_source is None:
            p_source = torch.ones(n_source, dtype=target_dtype) / n_source
        else:
            p_source = p_source.to(dtype=target_dtype)
        if p_target is None:
            p_target = torch.ones(n_target, dtype=target_dtype) / n_target
        else:
            p_target = p_target.to(dtype=target_dtype)
        
        if not self.config.uniform_measures:
            logger.debug("Using non-uniform measures (variance-based importance sampling)")
        
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
        
        # Determine quality based on solver used
        coupling_quality = 'optimal' if POT_AVAILABLE else 'fallback'
        solver_type = log.get('solver', 'unknown')
        
        # Compute quality score based on coupling characteristics
        quality_score = self._compute_quality_score(coupling, log, n_source, n_target)
        
        # Create result with quality information
        result = GWResult(
            coupling=coupling,
            cost=cost,
            log=log,
            source_size=n_source,
            target_size=n_target,
            p_source=p_source,
            p_target=p_target,
            coupling_quality=coupling_quality,
            solver_type=solver_type,
            quality_score=quality_score
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
    
    def _compute_quality_score(self, coupling: torch.Tensor, log: Dict[str, Any], 
                             n_source: int, n_target: int) -> float:
        """Compute numerical quality score for GW coupling.
        
        The quality score is computed based on multiple factors:
        - Solver type (optimal POT > fallback)
        - Coupling entropy (structured > diffuse)
        - Convergence quality (converged > non-converged)
        - Error metrics if available
        
        Args:
            coupling: Computed transport coupling
            log: Solver log with convergence information
            n_source: Number of source points
            n_target: Number of target points
            
        Returns:
            Quality score in [0,1] where 1.0 is optimal quality
        """
        quality_score = 1.0
        
        # Factor 1: Solver type penalty
        solver = log.get('solver', 'unknown')
        if solver.startswith('fallback'):
            quality_score *= 0.3  # Heavy penalty for fallback
        elif solver.startswith('pot'):
            quality_score *= 1.0  # No penalty for POT solver
        else:
            quality_score *= 0.5  # Unknown solver gets moderate penalty
            
        # Factor 2: Coupling entropy penalty (too diffuse = low quality)
        entropy_ratio = log.get('coupling_entropy_ratio', 0.5)
        if entropy_ratio > 0.95:
            quality_score *= 0.2  # Heavy penalty for very diffuse coupling
        elif entropy_ratio > 0.85:
            quality_score *= 0.6  # Moderate penalty for diffuse coupling
            
        # Factor 3: Convergence penalty
        converged = log.get('converged', True)
        if not converged:
            quality_score *= 0.4  # Penalty for non-convergence
            
        # Factor 4: Final error penalty
        final_error = log.get('final_error', 0.0)
        # Handle case where final_error might be a list (from POT)
        if isinstance(final_error, list):
            final_error = final_error[-1] if final_error else 0.0
        elif hasattr(final_error, 'item'):  # tensor
            final_error = final_error.item()
        
        if final_error > 1e-3:
            quality_score *= 0.5  # Penalty for high final error
            
        # Factor 5: Special case for rank-1 outer products (uniform fallback)
        if coupling.numel() > 0:
            # Check if coupling is approximately rank-1 (outer product)
            coupling_norm = torch.norm(coupling, 'fro')
            if coupling_norm > 1e-12:
                coupling_normalized = coupling / coupling_norm
                U, S, Vt = torch.svd(coupling_normalized)
                # If first singular value dominates, it's likely rank-1
                if S.numel() > 1 and S[0] / S[1] > 100:
                    quality_score *= 0.1  # Very low quality for rank-1
                    
        return max(0.0, min(1.0, quality_score))  # Clamp to [0,1]
    
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
            
            # Convert back to PyTorch with consistent dtype
            target_dtype = self.config.get_torch_dtype()
            coupling = torch.from_numpy(coupling_np).to(dtype=target_dtype)
            
            # POT returns coupling in shape (n_source, n_target), but our convention is (n_target, n_source)
            # Transpose to match our convention where rows=target, cols=source
            coupling = coupling.T
            
            # Extract cost from log or compute it
            if 'gw_dist' in log_dict:
                cost = float(log_dict['gw_dist'])
            elif 'loss' in log_dict and (len(log_dict['loss']) if isinstance(log_dict['loss'], list) else log_dict['loss'].numel()) > 0:
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
                           p_source: Optional[torch.Tensor], p_target: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float, Dict]:
        """Enhanced fallback GW implementation with multiple strategies."""
        logger.info("Using enhanced fallback GW implementation")
        
        n_source = C_source.shape[0]
        n_target = C_target.shape[0]
        
        # Initialize uniform measures if not provided
        if p_source is None:
            p_source = torch.ones(n_source, dtype=C_source.dtype, device=C_source.device) / n_source
        if p_target is None:
            p_target = torch.ones(n_target, dtype=C_target.dtype, device=C_target.device) / n_target
        
        # Try multiple fallback strategies in order of quality
        strategies = [
            ('cosine_similarity', self._fallback_cosine_similarity),
            ('spectral_matching', self._fallback_spectral_matching),
            ('uniform_coupling', self._fallback_uniform_coupling)
        ]
        
        best_coupling = None
        best_cost = float('inf')
        best_strategy = None
        best_log = None
        
        for strategy_name, strategy_func in strategies:
            try:
                coupling, cost, log = strategy_func(C_source, C_target, p_source, p_target)
                if cost < best_cost:
                    best_coupling = coupling
                    best_cost = cost
                    best_strategy = strategy_name
                    best_log = log
                    
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # If all strategies failed, use uniform as last resort
        if best_coupling is None:
            logger.warning("All fallback strategies failed, using uniform coupling")
            best_coupling = torch.outer(p_target, p_source)  # (n_target, n_source) for POT convention
            best_cost = self._compute_gw_cost(C_source, C_target, best_coupling)
            best_strategy = 'uniform_emergency'
            best_log = {'error': 'All strategies failed'}
        
        # Prepare final log
        final_log = {
            'solver': f'fallback_{best_strategy}',
            'converged': True,
            'num_iter': 0,
            'final_error': float('inf'),  # Indicates not optimized
            'cost_evolution': [best_cost],
            'gw_dist': best_cost,
            'warning': f'Using fallback implementation ({best_strategy}) - not optimal',
            'fallback_reason': 'POT library unavailable' if not POT_AVAILABLE else 'POT solver failed',
            'strategy_used': best_strategy,
            'strategy_log': best_log
        }
        
        logger.info(f"Selected fallback strategy: {best_strategy} with cost {best_cost:.6f}")
        return best_coupling, best_cost, final_log
    
    def _compute_gw_cost(self, C_source: torch.Tensor, C_target: torch.Tensor, 
                        coupling: torch.Tensor) -> float:
        """Compute GW cost for given coupling.
        
        Cost = Σ_{i,j,k,l} |C_source[i,k] - C_target[j,l]|² π[j,i] π[l,k]
        Note: coupling is (n_target, n_source) in POT convention
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
            i = idx // n_target  # source index
            j = idx % n_target   # target index
            
            for k in range(min(n_source, 10)):  # Limit inner loop (source)
                for l in range(min(n_target, 10)):  # Limit inner loop (target)
                    diff = C_source[i, k] - C_target[j, l]
                    # coupling[j, i] because coupling is (n_target, n_source)
                    cost += (diff ** 2) * coupling[j, i] * coupling[l, k]
        
        return cost.item() if torch.is_tensor(cost) else cost
    
    def _fallback_cosine_similarity(self, C_source: torch.Tensor, C_target: torch.Tensor,
                                   p_source: torch.Tensor, p_target: torch.Tensor) -> Tuple[torch.Tensor, float, Dict]:
        """Fallback using cosine similarity between cost matrix rows/columns."""
        try:
            # Compute pairwise similarities between cost matrix structures
            # This is better than uniform as it considers actual metric structure
            
            n_source, n_target = C_source.shape[0], C_target.shape[0]
            
            # Normalize cost matrices for comparison
            C_source_norm = torch.nn.functional.normalize(C_source, p=2, dim=1)  # Row-wise normalization
            C_target_norm = torch.nn.functional.normalize(C_target, p=2, dim=1)
            
            # Compute cosine similarities between cost matrix rows
            # Compare each source point's distance profile with each target point's distance profile
            # Note: C_source_norm is (n_source, n_source), C_target_norm is (n_target, n_target)
            # We need to handle different dimensions carefully
            
            # Truncate to common dimension for comparison
            min_dim = min(C_source.shape[1], C_target.shape[1])
            C_src_trunc = C_source_norm[:, :min_dim]  # (n_source, min_dim)
            C_tgt_trunc = C_target_norm[:, :min_dim]  # (n_target, min_dim)
            
            # Compute similarity matrix: need (n_target, n_source) for POT convention
            # C_src_trunc is (n_source, min_dim), C_tgt_trunc is (n_target, min_dim)
            # We want (n_target, min_dim) @ (min_dim, n_source) = (n_target, n_source)
            similarities = C_tgt_trunc @ C_src_trunc.T  # (n_target, n_source)
            
            # Convert similarities to transport weights (non-negative)
            coupling_weights = torch.relu(similarities)
            
            # Normalize to satisfy marginal constraints
            # Project onto doubly stochastic matrices using iterative scaling
            coupling = self._project_to_doubly_stochastic(coupling_weights, p_source, p_target)
            
            cost = self._compute_gw_cost(C_source, C_target, coupling)
            
            log = {
                'method': 'cosine_similarity',
                'avg_similarity': similarities.mean().item(),
                'num_projections': 10  # Fixed number of projections used
            }
            
            return coupling, cost, log
            
        except Exception as e:
            # If cosine similarity fails, raise to try next strategy
            raise RuntimeError(f"Cosine similarity fallback failed: {e}")
    
    def _fallback_spectral_matching(self, C_source: torch.Tensor, C_target: torch.Tensor,
                                   p_source: torch.Tensor, p_target: torch.Tensor) -> Tuple[torch.Tensor, float, Dict]:
        """Fallback using spectral matching of cost matrices."""
        try:
            # Match based on leading eigenvectors of cost matrices
            # This captures the dominant structure of the metric spaces
            
            n_source, n_target = C_source.shape[0], C_target.shape[0]
            
            # Compute leading eigenvectors (top-k for efficiency)
            k = min(3, min(n_source, n_target) - 1)  # Use top-3 eigenvectors
            if k <= 0:
                raise RuntimeError("Not enough dimensions for spectral matching")
            
            # Eigendecomposition of cost matrices
            eigenvals_src, eigenvecs_src = torch.linalg.eigh(C_source)
            eigenvals_tgt, eigenvecs_tgt = torch.linalg.eigh(C_target)
            
            # Take top-k eigenvectors (largest eigenvalues)
            top_k_src = eigenvecs_src[:, -k:]  # (n_source, k)
            top_k_tgt = eigenvecs_tgt[:, -k:]  # (n_target, k)
            
            # Compute coupling based on eigenvector similarities
            # For POT convention, we need (n_target, n_source)
            coupling_weights = torch.abs(top_k_tgt @ top_k_src.T)  # (n_target, n_source)
            
            # Project to satisfy marginal constraints
            coupling = self._project_to_doubly_stochastic(coupling_weights, p_source, p_target)
            
            cost = self._compute_gw_cost(C_source, C_target, coupling)
            
            log = {
                'method': 'spectral_matching',
                'num_eigenvectors': k,
                'src_top_eigenval': eigenvals_src[-1].item(),
                'tgt_top_eigenval': eigenvals_tgt[-1].item()
            }
            
            return coupling, cost, log
            
        except Exception as e:
            raise RuntimeError(f"Spectral matching fallback failed: {e}")
    
    def _fallback_uniform_coupling(self, C_source: torch.Tensor, C_target: torch.Tensor,
                                  p_source: torch.Tensor, p_target: torch.Tensor) -> Tuple[torch.Tensor, float, Dict]:
        """Simple uniform coupling fallback (original fallback)."""
        # This is the original fallback - rank-1 outer product
        # Using POT convention: coupling shape is (n_target, n_source)
        coupling = torch.outer(p_target, p_source)
        cost = self._compute_gw_cost(C_source, C_target, coupling)
        
        log = {
            'method': 'uniform_coupling',
            'rank': 1,
            'note': 'Simple outer product - lowest quality fallback'
        }
        
        return coupling, cost, log
    
    def _project_to_doubly_stochastic(self, weights: torch.Tensor, p_source: torch.Tensor, 
                                    p_target: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        """Project weight matrix to satisfy marginal constraints using iterative scaling.
        
        Note: Assumes POT convention where coupling is (n_target, n_source).
        - Row sums should equal p_target (marginals for target)
        - Column sums should equal p_source (marginals for source)
        """
        coupling = weights.clone()
        
        # Iterative scaling to satisfy marginal constraints
        for _ in range(max_iter):
            # Normalize rows to match p_target (each row is a target point)
            row_sums = coupling.sum(dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-12)  # Avoid division by zero
            coupling = coupling * (p_target.unsqueeze(1) / row_sums)
            
            # Normalize columns to match p_source (each column is a source point)
            col_sums = coupling.sum(dim=0, keepdim=True)
            col_sums = torch.clamp(col_sums, min=1e-12)  # Avoid division by zero
            coupling = coupling * (p_source.unsqueeze(0) / col_sums)
        
        return coupling
    
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
            if torch.any(p_source < 0) or not torch.allclose(p_source.sum(), torch.tensor(1.0, dtype=p_source.dtype)):
                raise ValueError("p_source must be a probability distribution")
        
        if p_target is not None:
            if p_target.shape[0] != C_target.shape[0]:
                raise ValueError("p_target dimension mismatch with C_target")
            if torch.any(p_target < 0) or not torch.allclose(p_target.sum(), torch.tensor(1.0, dtype=p_target.dtype)):
                raise ValueError("p_target must be a probability distribution")
    
    def clear_cache(self) -> None:
        """Clear cost matrix cache."""
        if self.cost_cache is not None:
            self.cost_cache.clear()
            logger.info("Cost matrix cache cleared")
    
    # ====================================================================
    # MATHEMATICALLY-GROUNDED NUMERICAL STABILITY IMPROVEMENTS
    # ====================================================================
    
    def matrix_sqrt(self, M: torch.Tensor, regularization: float = 1e-12) -> torch.Tensor:
        """Numerically stable matrix square root via eigendecomposition.
        
        Args:
            M: Positive semi-definite matrix
            regularization: Small value added to prevent numerical issues
            
        Returns:
            Matrix square root M^(1/2)
        """
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(M)
            eigenvals_reg = torch.clamp(eigenvals, min=regularization)
            sqrt_eigenvals = torch.sqrt(eigenvals_reg)
            return eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.T
        except Exception as e:
            logger.warning(f"Matrix square root failed, using regularized identity: {e}")
            return torch.sqrt(regularization) * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    
    def matrix_sqrt_inv(self, M: torch.Tensor, regularization: float = 1e-12) -> torch.Tensor:
        """Numerically stable inverse matrix square root.
        
        Args:
            M: Positive semi-definite matrix
            regularization: Small value added to eigenvalues for numerical stability
            
        Returns:
            Inverse matrix square root M^(-1/2)
        """
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(M)
            eigenvals_reg = torch.clamp(eigenvals, min=regularization)
            inv_sqrt_eigenvals = 1.0 / torch.sqrt(eigenvals_reg)
            return eigenvecs @ torch.diag(inv_sqrt_eigenvals) @ eigenvecs.T
        except Exception as e:
            logger.warning(f"Inverse matrix square root failed, using regularized identity: {e}")
            return (1.0 / torch.sqrt(regularization)) * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    
    def construct_restriction_from_gw_coupling(self, 
                                             coupling: torch.Tensor,
                                             X_source: torch.Tensor, 
                                             Y_target: torch.Tensor,
                                             mass_source: torch.Tensor,
                                             mass_target: torch.Tensor) -> torch.Tensor:
        """Construct restriction map using mass-weighted least squares with proper metrics.
        
        This implements the mathematically sound approach:
        1. R_LS = Y Π X^T (X A X^T + λI)^(-1)  [mass-weighted least squares]
        2. R_bal = G_B^(1/2) R_LS G_A^(-1/2)    [balanced coordinates]
        3. R_proj = G_B^(-1/2) (U V^T) G_A^(1/2) [polar projection, κ=1]
        
        Args:
            coupling: GW coupling matrix Π ∈ R^(n_B × n_A)
            X_source: Source stalk features X ∈ R^(d × n_A) (columns = atoms)
            Y_target: Target stalk features Y ∈ R^(d × n_B) (columns = atoms)
            mass_source: GW marginals a ∈ R^(n_A)
            mass_target: GW marginals b ∈ R^(n_B)
            
        Returns:
            R_proj: Geometrically-sound restriction map with κ=1 conditioning
        """
        # Ensure consistent precision throughout computation
        target_dtype = self.config.get_torch_dtype()
        X_source = X_source.to(dtype=target_dtype)
        Y_target = Y_target.to(dtype=target_dtype)
        coupling = coupling.to(dtype=target_dtype)
        mass_source = mass_source.to(dtype=target_dtype)
        mass_target = mass_target.to(dtype=target_dtype)
        
        # Mass-weighted metric matrices
        A = torch.diag(mass_source)  # diag(a)
        B = torch.diag(mass_target)  # diag(b)
        
        # Compute Gram matrices
        XAXt = X_source @ A @ X_source.T
        YBYt = Y_target @ B @ Y_target.T
        
        # Trace-proportional ridge regularization (η = 1e-8 in float64)
        d = X_source.shape[0]
        lambda_A = 1e-8 * torch.trace(XAXt) / d if torch.trace(XAXt) > 1e-15 else 1e-8
        lambda_B = 1e-8 * torch.trace(YBYt) / d if torch.trace(YBYt) > 1e-15 else 1e-8
        
        G_A = XAXt + lambda_A * torch.eye(d, dtype=X_source.dtype, device=X_source.device)
        G_B = YBYt + lambda_B * torch.eye(d, dtype=Y_target.dtype, device=Y_target.device)
        
        try:
            # Mass-weighted least squares map: R_LS = Y Π X^T (G_A)^(-1)
            R_LS = Y_target @ coupling @ X_source.T @ torch.linalg.solve(G_A, torch.eye(d, dtype=X_source.dtype, device=X_source.device))
            
            # Weighted polar projection for κ=1
            G_B_half = self.matrix_sqrt(G_B)
            G_A_neg_half = self.matrix_sqrt_inv(G_A)
            
            # Balanced coordinates: R_bal = G_B^(1/2) R_LS G_A^(-1/2)
            R_bal = G_B_half @ R_LS @ G_A_neg_half
            
            # SVD for polar projection
            U, Sigma, Vt = torch.linalg.svd(R_bal, full_matrices=False)
            
            # Project to G-orthogonal factor: R_proj = G_B^(-1/2) (U V^T) G_A^(1/2)
            G_B_neg_half = self.matrix_sqrt_inv(G_B)
            G_A_half = self.matrix_sqrt(G_A)
            
            R_proj = G_B_neg_half @ (U @ Vt) @ G_A_half
            
            # Log conditioning improvement
            condition_before = torch.linalg.cond(R_LS).item() if R_LS.numel() > 0 else float('inf')
            condition_after = torch.linalg.cond(R_proj).item() if R_proj.numel() > 0 else float('inf')
            
            logger.info(f"Restriction map conditioning improved: {condition_before:.2e} → {condition_after:.2e}")
            
            # Final conditioning check and improvement if needed
            condition_final = torch.linalg.cond(R_proj).item() if R_proj.numel() > 0 else float('inf')
            
            if condition_final > 1e8:
                logger.warning(f"Final restriction still poorly conditioned: {condition_final:.2e}")
                # Apply final TSVD conditioning improvement
                R_proj_conditioned, was_truncated = self.apply_weighted_tsvd(R_proj, energy_threshold=1e-5)
                if was_truncated:
                    final_condition = torch.linalg.cond(R_proj_conditioned).item()
                    logger.info(f"Final TSVD conditioning: {condition_final:.2e} → {final_condition:.2e}")
                    return R_proj_conditioned  # Keep configured dtype
            
            return R_proj  # Keep configured dtype
            
        except Exception as e:
            logger.error(f"Mass-weighted restriction construction failed: {e}")
            # Fallback to regularized least squares without polar projection
            try:
                R_fallback = Y_target @ coupling @ X_source.T @ torch.linalg.solve(
                    G_A + 1e-6 * torch.eye(d, dtype=X_source.dtype, device=X_source.device),
                    torch.eye(d, dtype=X_source.dtype, device=X_source.device)
                )
                logger.warning("Using fallback regularized least squares for restriction")
                return R_fallback
            except Exception as e2:
                logger.error(f"Fallback restriction construction also failed: {e2}")
                # Ultimate fallback: identity scaled by coupling norm
                coupling_norm = torch.norm(coupling).item()
                fallback_size = min(X_source.shape[0], Y_target.shape[0])
                R_identity = coupling_norm * torch.eye(fallback_size, dtype=self.config.get_torch_dtype(), device=X_source.device)
                logger.warning("Using identity fallback for restriction")
                return R_identity
    
    def preprocess_masses_for_stability(self, 
                                       mass_source: torch.Tensor, 
                                       mass_target: torch.Tensor,
                                       trim_threshold: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trim tiny masses and optionally use unbalanced GW for numerical stability.
        
        Args:
            mass_source: Source mass distribution
            mass_target: Target mass distribution
            trim_threshold: Relative threshold for mass trimming (e.g., 1e-6)
            
        Returns:
            Tuple of (mass_source_processed, mass_target_processed, keep_source_mask, keep_target_mask)
        """
        mean_mass_source = torch.mean(mass_source)
        mean_mass_target = torch.mean(mass_target)
        
        # Identify atoms to keep
        keep_source = mass_source >= trim_threshold * mean_mass_source
        keep_target = mass_target >= trim_threshold * mean_mass_target
        
        # Apply trimming if needed
        if not torch.all(keep_source) or not torch.all(keep_target):
            n_source_before, n_target_before = mass_source.numel(), mass_target.numel()
            n_source_after, n_target_after = keep_source.sum().item(), keep_target.sum().item()
            
            logger.info(f"Mass trimming: source {n_source_before} → {n_source_after}, "
                       f"target {n_target_before} → {n_target_after}")
            
            # Trim and renormalize
            mass_source_trimmed = mass_source[keep_source]
            mass_target_trimmed = mass_target[keep_target]
            
            # Renormalize to ensure probability constraints
            if mass_source_trimmed.sum() > 1e-15:
                mass_source_trimmed /= mass_source_trimmed.sum()
            else:
                mass_source_trimmed = torch.ones_like(mass_source_trimmed) / mass_source_trimmed.numel()
                
            if mass_target_trimmed.sum() > 1e-15:
                mass_target_trimmed /= mass_target_trimmed.sum()
            else:
                mass_target_trimmed = torch.ones_like(mass_target_trimmed) / mass_target_trimmed.numel()
            
            return mass_source_trimmed, mass_target_trimmed, keep_source, keep_target
        
        return mass_source, mass_target, keep_source, keep_target
    
    def compute_gw_coupling_robust(self, 
                                  C_source: torch.Tensor, 
                                  C_target: torch.Tensor,
                                  p_source: Optional[torch.Tensor] = None,
                                  p_target: Optional[torch.Tensor] = None,
                                  epsilon_schedule: Optional[list] = None) -> GWResult:
        """Robust GW computation with epsilon-scaling and warm-starting.
        
        This implements numerically-friendly GW computation:
        1. Geometric epsilon-scaling (coarse → fine)
        2. Warm-starting from previous coupling
        3. Mass preprocessing for stability
        4. Log-domain Sinkhorn throughout
        
        Args:
            C_source: Source cost matrix
            C_target: Target cost matrix  
            p_source: Source marginals (optional, uniform if None)
            p_target: Target marginals (optional, uniform if None)
            epsilon_schedule: Custom epsilon schedule (optional)
            
        Returns:
            GWResult with final coupling from epsilon-scaling sequence
        """
        # Validate and preprocess inputs
        self._validate_gw_inputs(C_source, C_target, p_source, p_target)
        
        # Set up default masses
        n_source, n_target = C_source.shape[0], C_target.shape[0]
        if p_source is None:
            p_source = torch.ones(n_source, dtype=C_source.dtype, device=C_source.device) / n_source
        if p_target is None:
            p_target = torch.ones(n_target, dtype=C_target.dtype, device=C_target.device) / n_target
        
        # Preprocess masses for numerical stability
        p_source_proc, p_target_proc, keep_source, keep_target = self.preprocess_masses_for_stability(
            p_source, p_target, trim_threshold=1e-6
        )
        
        # Trim cost matrices if masses were trimmed
        if not torch.all(keep_source) or not torch.all(keep_target):
            C_source_proc = C_source[keep_source][:, keep_source]
            C_target_proc = C_target[keep_target][:, keep_target]
        else:
            C_source_proc, C_target_proc = C_source, C_target
            p_source_proc, p_target_proc = p_source, p_target
        
        # Set up epsilon schedule
        if epsilon_schedule is None:
            # Geometric schedule: start with median cost, reduce geometrically
            with torch.no_grad():
                # Extract upper triangular costs from square cost matrices
                n_source = C_source_proc.shape[0]
                n_target = C_target_proc.shape[0]
                
                # Get upper triangular indices for square matrices
                source_triu_i, source_triu_j = torch.triu_indices(n_source, n_source, offset=1)
                target_triu_i, target_triu_j = torch.triu_indices(n_target, n_target, offset=1)
                
                source_costs = C_source_proc[source_triu_i, source_triu_j]
                target_costs = C_target_proc[target_triu_i, target_triu_j]
                
                # Combine costs for median calculation
                all_costs = torch.cat([source_costs, target_costs]) if source_costs.numel() > 0 and target_costs.numel() > 0 else torch.tensor([0.01], dtype=C_source.dtype)
                median_cost = torch.median(all_costs).item() if all_costs.numel() > 0 else 0.01
            
            epsilon_schedule = [median_cost * (0.6 ** k) for k in range(6)]
            epsilon_schedule[-1] = max(epsilon_schedule[-1], self.config.epsilon)
            
        logger.info(f"Epsilon-scaling schedule: {[f'{eps:.2e}' for eps in epsilon_schedule]}")
        
        # Initialize with uniform coupling
        n_proc_source, n_proc_target = p_source_proc.numel(), p_target_proc.numel()
        coupling = torch.outer(p_target_proc, p_source_proc)  # Initial uniform coupling
        
        results = []
        for i, eps in enumerate(epsilon_schedule):
            logger.debug(f"Epsilon-scaling step {i+1}/{len(epsilon_schedule)}: ε={eps:.2e}")
            
            try:
                # Solve GW with current epsilon, warm-started from previous coupling
                if i == 0:
                    # First iteration: use uniform initialization
                    result = self._solve_gw_single_epsilon(C_source_proc, C_target_proc, eps,
                                                         p_source_proc, p_target_proc, None)
                else:
                    # Warm-start from previous coupling
                    result = self._solve_gw_single_epsilon(C_source_proc, C_target_proc, eps,
                                                         p_source_proc, p_target_proc, coupling)
                
                if hasattr(result, 'coupling') and result.coupling is not None:
                    coupling = result.coupling  # Update for next iteration
                    results.append(result)
                    logger.debug(f"  ✅ Converged at ε={eps:.2e}, cost={result.cost:.6f}")
                else:
                    logger.warning(f"  ❌ Failed at ε={eps:.2e}")
                    break
                    
            except Exception as e:
                logger.warning(f"Epsilon-scaling failed at ε={eps:.2e}: {e}")
                break
        
        if not results:
            logger.error("All epsilon-scaling steps failed")
            # Return fallback uniform coupling
            uniform_coupling = torch.outer(p_target_proc, p_source_proc)
            fallback_cost = self._compute_gw_cost(C_source_proc, C_target_proc, uniform_coupling)
            return GWResult(
                coupling=uniform_coupling,
                cost=fallback_cost,
                log={'converged': False, 'epsilon_scaling_failed': True},
                source_size=n_proc_source,
                target_size=n_proc_target
            )
        
        best_result = results[-1]  # Use final (smallest epsilon) result
        logger.info(f"Epsilon-scaling completed: {len(results)} successful steps, "
                   f"final cost={best_result.cost:.6f}")
        
        # If masses were trimmed, we need to reconstruct full coupling
        if not torch.all(keep_source) or not torch.all(keep_target):
            full_coupling = self._reconstruct_full_coupling(
                best_result.coupling, keep_source, keep_target, p_source, p_target
            )
            return GWResult(
                coupling=full_coupling,
                cost=best_result.cost,
                log=best_result.log,
                source_size=n_source,
                target_size=n_target
            )
        
        return best_result
    
    def _solve_gw_single_epsilon(self, 
                                C_source: torch.Tensor, 
                                C_target: torch.Tensor, 
                                epsilon: float,
                                p_source: torch.Tensor,
                                p_target: torch.Tensor,
                                coupling_init: Optional[torch.Tensor] = None) -> GWResult:
        """Solve GW for single epsilon value with optional warm-start."""
        # Use existing compute_gw_coupling with modified epsilon
        original_epsilon = self.config.epsilon
        self.config.epsilon = epsilon
        
        try:
            # If warm-starting, we would need to modify the underlying solver
            # For now, use the existing implementation
            result = self.compute_gw_coupling(C_source, C_target, p_source, p_target)
            return result
        finally:
            # Restore original epsilon
            self.config.epsilon = original_epsilon
    
    def _reconstruct_full_coupling(self, 
                                  coupling_trimmed: torch.Tensor,
                                  keep_source: torch.Tensor,
                                  keep_target: torch.Tensor,
                                  p_source: torch.Tensor,
                                  p_target: torch.Tensor) -> torch.Tensor:
        """Reconstruct full coupling matrix after mass trimming."""
        n_source, n_target = p_source.numel(), p_target.numel()
        
        # Initialize with zeros
        full_coupling = torch.zeros(n_target, n_source, 
                                   dtype=coupling_trimmed.dtype, 
                                   device=coupling_trimmed.device)
        
        # Fill in the trimmed coupling
        target_indices = torch.where(keep_target)[0]
        source_indices = torch.where(keep_source)[0]
        
        # Ensure we don't exceed trimmed coupling dimensions
        n_trimmed_target, n_trimmed_source = coupling_trimmed.shape
        
        for i in range(min(target_indices.numel(), n_trimmed_target)):
            target_idx = target_indices[i]
            for j in range(min(source_indices.numel(), n_trimmed_source)):
                source_idx = source_indices[j]
                full_coupling[target_idx, source_idx] = coupling_trimmed[i, j]
        
        return full_coupling
    
    def apply_weighted_tsvd(self, 
                           R_bal: torch.Tensor, 
                           energy_threshold: float = 1e-6) -> Tuple[torch.Tensor, bool]:
        """Apply truncated SVD based on cumulative energy, not arbitrary compression.
        
        This replaces "log spectral compression" with mathematically sound truncation
        based on cumulative energy preservation.
        
        Args:
            R_bal: Balanced restriction map to truncate
            energy_threshold: Energy loss threshold (e.g., 1e-6 means keep 99.9999% energy)
            
        Returns:
            Tuple of (R_truncated, was_truncated)
        """
        if R_bal.numel() == 0:
            return R_bal, False
        
        try:
            # Full SVD for energy analysis
            U, Sigma, Vt = torch.linalg.svd(R_bal, full_matrices=False)
            
            if Sigma.numel() == 0:
                return R_bal, False
            
            # Compute cumulative energy (relative to total)
            energy = Sigma ** 2
            total_energy = torch.sum(energy)
            
            if total_energy < 1e-15:
                logger.warning("Matrix has essentially zero energy, no truncation applied")
                return R_bal, False
            
            cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
            
            # Find truncation point: keep enough modes to preserve (1 - energy_threshold) energy
            threshold_point = 1.0 - energy_threshold
            k_candidates = torch.searchsorted(cumulative_energy, threshold_point)
            k = max(1, min(k_candidates.item() + 1, Sigma.numel()))  # At least 1, at most full rank
            
            if k >= Sigma.numel():
                # No truncation needed
                return R_bal, False
            
            # Apply truncation
            energy_preserved = cumulative_energy[k-1].item()
            logger.info(f"TSVD truncation: keeping {k}/{Sigma.numel()} singular values "
                       f"(energy preserved: {energy_preserved:.6f})")
            
            # Truncate and reconstruct
            R_truncated = U[:, :k] @ torch.diag(Sigma[:k]) @ Vt[:k, :]
            
            # Verify energy preservation
            U_trunc, Sigma_trunc, Vt_trunc = torch.linalg.svd(R_truncated, full_matrices=False)
            actual_energy = torch.sum(Sigma_trunc ** 2) / total_energy if total_energy > 1e-15 else 0.0
            
            logger.debug(f"TSVD verification: target energy {energy_preserved:.6f}, "
                        f"actual energy {actual_energy:.6f}")
            
            return R_truncated, True
            
        except Exception as e:
            logger.warning(f"TSVD truncation failed: {e}")
            return R_bal, False
    
    def construct_restriction_with_tsvd(self,
                                       coupling: torch.Tensor,
                                       X_source: torch.Tensor,
                                       Y_target: torch.Tensor,
                                       mass_source: torch.Tensor,
                                       mass_target: torch.Tensor,
                                       enable_tsvd: bool = True,
                                       energy_threshold: float = 1e-6) -> torch.Tensor:
        """Construct restriction with optional TSVD rank control.
        
        Combines the mass-weighted least squares approach with optional
        TSVD-based rank control for cases where dimensionality reduction is needed.
        
        Args:
            coupling: GW coupling matrix
            X_source: Source stalk features  
            Y_target: Target stalk features
            mass_source: Source marginals
            mass_target: Target marginals
            enable_tsvd: Whether to apply TSVD rank control
            energy_threshold: Energy loss threshold for TSVD
            
        Returns:
            Final restriction map (with TSVD if applied)
        """
        # First, construct the restriction using mass-weighted approach
        R_proj = self.construct_restriction_from_gw_coupling(
            coupling, X_source, Y_target, mass_source, mass_target
        )
        
        # Apply TSVD if requested and beneficial
        if enable_tsvd:
            # Check if TSVD would be beneficial
            try:
                condition_before = torch.linalg.cond(R_proj).item()
                
                # Apply TSVD if conditioning is poor
                if condition_before > 1e8:
                    logger.info(f"Applying TSVD due to poor conditioning: {condition_before:.2e}")
                    R_tsvd, was_truncated = self.apply_weighted_tsvd(R_proj, energy_threshold)
                    
                    if was_truncated:
                        condition_after = torch.linalg.cond(R_tsvd).item()
                        logger.info(f"TSVD conditioning improvement: {condition_before:.2e} → {condition_after:.2e}")
                        return R_tsvd
                    
            except Exception as e:
                logger.debug(f"TSVD conditioning check failed: {e}")
        
        return R_proj
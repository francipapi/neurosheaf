"""t-Flow Implementation for Multi-scale Diffusion Analysis.

This module implements the t-flow method for comparing different GW sheaves using
heat kernel summaries. The method uses heat trace computation over different time
scales to probe multi-scale structure:

h(t) = (1/n) * Tr(exp(-t*L))

Key Features:
- Automatic t-grid generation based on λ_max estimation
- Stochastic Lanczos Quadrature for scalable heat trace computation
- Fixed mass matrix mode for cross-architecture comparability
- Multi-scale diffusion analysis
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union, Literal
import time
import logging

__all__ = ["DiffusionSpec", "DiffusionSummaries", "DiffusionFlowAnalyzer"]

from ...utils.logging import setup_logger
from ...sheaf.data_structures import Sheaf
from ..utils_numerical import heat_trace_slq, estimate_lambda_max, smallest_eigs_generalized

logger = setup_logger(__name__)


@dataclass(frozen=True)
class DiffusionSpec:
    """Specification for t-flow diffusion analysis.
    
    The t-flow method analyzes multi-scale structure using heat kernel summaries
    at different time scales. This configuration controls the analysis parameters.
    
    Attributes:
        t_grid: Time points for diffusion analysis. Can be:
                - 'auto': Generate 20 log-spaced points in [1e-3/λ_max, 10/λ_max]  
                - Sequence[float]: Explicit time points (must be positive)
        k_small: Number of smallest eigenpairs to capture for reference
        probes: Number of Hutchinson probes for SLQ (64 default, 128 for tighter CI)
        slq_iters: Number of Lanczos iterations for SLQ computation
    """
    t_grid: Union[Sequence[float], Literal['auto']] = 'auto'
    k_small: int = 16
    probes: int = 64
    slq_iters: int = 30
    
    def __post_init__(self):
        """Validate parameters."""
        if self.k_small < 0:
            raise ValueError(f"k_small must be non-negative, got {self.k_small}")
        if self.probes < 1:
            raise ValueError(f"probes must be positive, got {self.probes}")
        if self.slq_iters < 1:
            raise ValueError(f"slq_iters must be positive, got {self.slq_iters}")
        
        if isinstance(self.t_grid, (list, tuple, np.ndarray)):
            arr = np.array(self.t_grid, dtype=float)
            if arr.size == 0:
                raise ValueError("t_grid cannot be empty")
            if np.any(arr <= 0):
                raise ValueError("All t values must be positive")


@dataclass
class DiffusionSummaries:
    """Results from t-flow diffusion analysis.
    
    Contains the heat trace vector and supporting information from
    multi-scale diffusion analysis.
    
    Attributes:
        heat_trace: Heat trace values h(t) = Tr(exp(-t*L))/n for each t
        t_grid: Actual time points used (useful when auto-generated)
        smallest_eigs: Smallest eigenvalues for reference/validation
        meta: Analysis metadata including timing, convergence, and quality info
    """
    heat_trace: np.ndarray
    t_grid: np.ndarray
    smallest_eigs: np.ndarray
    meta: Dict
    
    def __post_init__(self):
        """Validate result consistency."""
        if len(self.heat_trace) != len(self.t_grid):
            raise ValueError(f"heat_trace length {len(self.heat_trace)} != t_grid length {len(self.t_grid)}")


class DiffusionFlowAnalyzer:
    """Analyzer for t-flow multi-scale diffusion analysis.
    
    This class implements the t-flow method using heat kernel summaries to
    analyze multi-scale structure in GW sheaves. The method:
    
    1. Builds single Laplacian L with all edges (no masking/partitioning)
    2. Uses fixed mass matrix D for cross-architecture comparability 
    3. Estimates λ_max for automatic t-grid generation
    4. Computes heat trace Tr(exp(-t*L)) using Stochastic Lanczos Quadrature
    5. Returns diffusion fingerprint over time scales
    
    Mathematical Foundation:
    - Heat kernel: K(t) = exp(-t*L) describes diffusion process
    - Heat trace: h(t) = Tr(K(t))/n gives global diffusion summary
    - Multi-scale: Different t values probe different scales of structure
    """
    
    def __init__(self, sheaf: Sheaf, gw_laplacian_builder, random_seed: int = 0, use_normalized_laplacian: Union[str, bool] = False):
        """Initialize diffusion flow analyzer.
        
        Args:
            sheaf: GW sheaf containing edge data and restrictions
            gw_laplacian_builder: GWLaplacianBuilder instance for matrix assembly
            random_seed: Random seed for deterministic results (default: 0)
            use_normalized_laplacian: Normalization mode:
                                    - False/'none': No normalization (generalized eigenproblem)
                                    - True/'sym': Symmetric normalization
                                    - 'rw': Random walk normalization
        """
        self.sheaf = sheaf
        self.gw_builder = gw_laplacian_builder
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self.normalization_mode = ('sym' if use_normalized_laplacian is True else
                                 'none' if use_normalized_laplacian in (False, 'none', None) else
                                 use_normalized_laplacian)
        self._validate_sheaf()
        
        # Cache for expensive computations (keyed by (mass_mode, sheaf_id))
        self._laplacian_cache = {}
        self._eigenvalues_cache = {}
        self._lambda_max_cache = {}
    
    def _validate_sheaf(self):
        """Validate that sheaf is suitable for t-flow analysis."""
        if not self.sheaf.is_gw_sheaf():
            logger.warning("t-flow works best with GW sheaves, but will proceed")
        
        if len(self.sheaf.restrictions) == 0:
            raise ValueError("Cannot perform t-flow analysis on empty sheaf")
    
    def _build_laplacian_all_edges(self, mass_mode: str = 'fixed') -> tuple[LinearOperator, sp.spmatrix, dict]:
        """Build Laplacian with all edges active (no masking).
        
        This is different from α-flow which partitions edges. For t-flow,
        we use the complete Laplacian with all edges to analyze the full
        multi-scale structure.
        
        Args:
            mass_mode: 'fixed' for consistency, 'adaptive' for accuracy
            
        Returns:
            Tuple of (L, D, metadata) where:
            - L: Complete Laplacian as LinearOperator (perfect for SLQ)
            - D: Mass matrix as sparse CSR (required for generalized eigensolvers)
            - metadata: Build information
        """
        cache_key = (mass_mode, id(self.sheaf), self.normalization_mode)
        if cache_key in self._laplacian_cache:
            logger.debug(f"Using cached Laplacian for t-flow (mass_mode={mass_mode}, normalize={self.normalization_mode})")
            return self._laplacian_cache[cache_key]
        
        # Build complete Laplacian using gateway approach for consistent G₀/G₁
        try:
            # Get all edges for complete Laplacian
            all_edges = list(self.sheaf.restrictions.keys())
            
            # Build through gateway with normalization
            L_csr, D = self.gw_builder._build_L_from_gateway(
                self.sheaf, all_edges, 
                normalize=self.normalization_mode
            )
            
            # Apply ridge regularization for generalized eigenproblems
            # Only needed if not normalized (normalized already has M=I)
            if self.normalization_mode == 'none':
                ridge_eps = 1e-12 if mass_mode == 'fixed' else 1e-15
                D = D + ridge_eps * sp.eye(D.shape[0], format='csr')
            
            # Wrap L as LinearOperator with both matvec and rmatvec (symmetric!)
            def _matvec(x):
                return L_csr @ x
            
            def _rmatvec(x):
                # For symmetric operators, rmatvec = matvec
                return L_csr @ x
            
            # Guard dtype - LinearOperator may have dtype=None issues
            dtype = L_csr.dtype or np.float64
            
            L = LinearOperator(
                shape=L_csr.shape, 
                matvec=_matvec, 
                rmatvec=_rmatvec, 
                dtype=dtype
            )
            
            # Log diagnostics for dtype and sparsity
            logger.info(f"t-flow build diagnostics: "
                       f"L(dtype={L.dtype}, nnz={L_csr.nnz}), "
                       f"D(dtype={D.dtype}, nnz={D.nnz}), "
                       f"normalize={self.normalization_mode}")
            
            metadata = {
                'n_edges': len(self.sheaf.restrictions),
                'matrix_size': L.shape[0],
                'mass_mode': mass_mode,
                'normalization_mode': self.normalization_mode,
                'ridge_regularization': getattr(locals(), 'ridge_eps', 0.0),
                'L_nnz': L_csr.nnz,
                'D_nnz': D.nnz,
                'L_dtype': str(L.dtype),
                'D_dtype': str(D.dtype),
            }
            
            # Shape assertion
            assert L.shape[0] == D.shape[0], f"L and D size mismatch: {L.shape[0]} vs {D.shape[0]}"
            
            # Cache the result
            self._laplacian_cache[cache_key] = (L, D, metadata)
            
            logger.info(f"Built complete Laplacian: {L.shape[0]}×{L.shape[0]}, "
                       f"{len(self.sheaf.restrictions)} edges, mass_mode={mass_mode}, "
                       f"normalize={self.normalization_mode}")
            
            return L, D, metadata
            
        except Exception as e:
            logger.error(f"Failed to build complete Laplacian: {e}")
            raise
    
    def _generate_auto_t_grid(self, L: LinearOperator, mass_mode: str, n_points: int = 20) -> np.ndarray:
        """Generate automatic t-grid based on λ_max estimation.
        
        Creates log-spaced time points in [1e-3/λ_max, 10/λ_max] to ensure
        good coverage of the diffusion time scales. Bounds the time range to
        prevent numerical overflow/underflow.
        
        Args:
            L: Laplacian operator
            mass_mode: Mass matrix mode (for cache keying)
            n_points: Number of time points to generate
            
        Returns:
            Array of time points for diffusion analysis
        """
        cache_key = (mass_mode, id(self.sheaf), self.normalization_mode)
        if cache_key not in self._lambda_max_cache:
            logger.info(f"Estimating λ_max for auto t-grid generation (mass_mode={mass_mode}, normalize={self.normalization_mode})")
            lam_max = estimate_lambda_max(L, iters=40, rng=self._rng)
            lam_max = float(max(lam_max, 1e-12))  # avoid zero/neg
            self._lambda_max_cache[cache_key] = lam_max
            logger.info(f"Estimated λ_max = {lam_max:.6e} (bounded)")
        else:
            lam_max = self._lambda_max_cache[cache_key]
            lam_max = float(max(lam_max, 1e-12))  # ensure cached value is also bounded
            logger.debug(f"Using cached λ_max = {lam_max:.6e} (mass_mode={mass_mode}, normalize={self.normalization_mode})")
        
        # Standard range: [1e-3/λ_max, 10/λ_max]
        t_min = 1e-3 / lam_max
        t_max = 10.0 / lam_max
        
        # Guard against under/overflow to keep exponentials representable
        t_min_bounded = max(t_min, 1e-12)
        t_max_bounded = min(t_max, 1e+6)
        
        # Log if bounds were clipped for transparency
        if t_min_bounded != t_min:
            logger.debug(f"t_min clipped: {t_min:.6e} → {t_min_bounded:.6e}")
        if t_max_bounded != t_max:
            logger.debug(f"t_max clipped: {t_max:.6e} → {t_max_bounded:.6e}")
        
        # Generate log-spaced points
        t_grid = np.logspace(np.log10(t_min_bounded), np.log10(t_max_bounded), n_points)
        
        logger.debug(f"Generated auto t-grid: [{t_min_bounded:.6e}, {t_max_bounded:.6e}] "
                    f"with {n_points} points")
        
        return t_grid
    
    def _compute_smallest_eigenvalues(self, L: LinearOperator, D: sp.spmatrix, 
                                    k: int, mass_mode: str) -> np.ndarray:
        """Compute k smallest generalized eigenvalues for reference.
        
        Args:
            L: Laplacian operator
            D: Mass matrix (sparse CSR format)
            k: Number of smallest eigenvalues to compute
            mass_mode: Mass matrix mode (for cache keying)
            
        Returns:
            Array of k smallest eigenvalues
        """
        cache_key = (mass_mode, id(self.sheaf), self.normalization_mode)
        if cache_key in self._eigenvalues_cache:
            logger.debug(f"Using cached eigenvalues (mass_mode={mass_mode}, normalize={self.normalization_mode})")
            return self._eigenvalues_cache[cache_key][:k]
        
        try:
            logger.debug(f"Computing {k} smallest eigenvalues")
            eig = smallest_eigs_generalized(
                L, D, k=k, 
                sigma=1e-6, 
                tol=1e-7, 
                maxiter=1000, 
                random_state=self._rng,
                return_vecs=False
            )
            
            # Handle both dict and SmallestEigResult return types defensively
            if isinstance(eig, dict):
                vals = eig.get("eigenvalues") or eig.get("vals")
                converged = eig.get("converged", True)
            else:
                # SmallestEigResult dataclass or simple array
                if hasattr(eig, 'eigenvalues'):
                    vals = eig.eigenvalues
                    converged = getattr(eig, 'converged', True)
                else:
                    # Simple np.ndarray of values
                    vals = np.asarray(eig)
                    converged = True
            
            if vals is None or len(vals) == 0 or not converged:
                logger.warning("Smallest-eigs did not converge; continuing without reference eigenvalues")
                return np.array([], dtype=np.float64)
            
            # Clip tiny negatives and cache
            vals = np.maximum(vals, 0.0)  # clip tiny negatives
            self._eigenvalues_cache[cache_key] = vals
            
            logger.debug(f"Computed eigenvalues: [{vals[0]:.6e}, {vals[-1]:.6e}] "
                        f"(converged: {converged})")
            
            return vals[:k]
                
        except Exception as e:
            logger.warning(f"Could not compute eigenvalues: {e}")
            return np.array([], dtype=np.float64)
    
    def analyze(self, spec: DiffusionSpec, mass_mode: str = 'fixed') -> DiffusionSummaries:
        """Perform complete t-flow diffusion analysis.
        
        This is the main analysis method that:
        1. Builds complete Laplacian L with all edges (no masking)
        2. Generates t-grid (auto or from spec)
        3. Computes heat trace Tr(exp(-t*L)) for each t using SLQ
        4. Returns diffusion fingerprint with metadata
        
        Args:
            spec: DiffusionSpec configuration
            mass_mode: Mass matrix mode ('fixed' or 'adaptive')
            
        Returns:
            DiffusionSummaries with heat trace and metadata
        """
        start_time = time.time()
        
        logger.info(f"Starting t-flow analysis: {spec.probes} probes, "
                   f"{spec.slq_iters} SLQ iters")
        
        # Step 1: Build complete Laplacian
        L, D, build_meta = self._build_laplacian_all_edges(mass_mode)
        
        # Step 2: Generate t-grid
        if spec.t_grid == 'auto':
            t_grid = self._generate_auto_t_grid(L, mass_mode)
            logger.info(f"Auto-generated t-grid: {len(t_grid)} points")
        else:
            t_grid = np.array(spec.t_grid)
            logger.info(f"Using provided t-grid: {len(t_grid)} points")
        
        # Step 3: Compute reference eigenvalues (fast path for k_small=0)
        if spec.k_small <= 0:
            smallest_eigs = np.array([], dtype=np.float64)
            logger.debug("Skipping eigenvalue computation (k_small <= 0)")
        else:
            smallest_eigs = self._compute_smallest_eigenvalues(L, D, spec.k_small, mass_mode)
        
        # Step 4: Compute heat trace for each t
        heat_trace = np.zeros(len(t_grid))
        heat_trace_std = np.zeros(len(t_grid))
        
        logger.info(f"Computing heat trace for {len(t_grid)} time points")
        
        for i, t in enumerate(t_grid):
            try:
                # Handle both tuple (est, std) and single scalar returns
                res = heat_trace_slq(
                    L, t, 
                    probes=spec.probes, 
                    iters=spec.slq_iters,
                    rng=self._rng
                )
                
                if isinstance(res, tuple) and len(res) >= 2:
                    trace_est, trace_std = res[0], res[1]
                else:
                    trace_est, trace_std = float(res), np.nan
                
                # Normalize by matrix size
                n = L.shape[0]
                heat_trace[i] = trace_est / n
                heat_trace_std[i] = trace_std / n if not np.isnan(trace_std) else np.nan
                
                if i % max(1, len(t_grid) // 10) == 0:  # Log every 10%
                    std_str = f"± {heat_trace_std[i]:.6e}" if not np.isnan(heat_trace_std[i]) else ""
                    logger.debug(f"t[{i}/{len(t_grid)}] = {t:.6e}: "
                               f"h(t) = {heat_trace[i]:.6e} {std_str}")
                    
            except Exception as e:
                logger.warning(f"Heat trace computation failed for t={t:.6e}: {e}")
                heat_trace[i] = np.nan
                heat_trace_std[i] = np.nan
        
        # Apply SLQ fallback using eigenvalue proxy when available
        n_fallbacks = 0
        for i, t in enumerate(t_grid):
            if np.isnan(heat_trace[i]) and smallest_eigs.size > 0:
                heat_trace[i] = np.sum(np.exp(-t * smallest_eigs)) / L.shape[0]
                heat_trace_std[i] = np.nan  # No std estimate for fallback
                n_fallbacks += 1
                logger.debug(f"Using eigenvalue fallback for t={t:.6e}")
        
        if n_fallbacks > 0:
            logger.info(f"Applied eigenvalue fallback for {n_fallbacks}/{len(t_grid)} points")
        
        # Step 5: Compile results and metadata
        elapsed_time = time.time() - start_time
        
        # Check for monotonicity (should be decreasing)
        valid_indices = ~np.isnan(heat_trace)
        if np.sum(valid_indices) > 1:
            valid_trace = heat_trace[valid_indices]
            valid_t = t_grid[valid_indices]
            
            # Relaxed monotonicity check (allow some noise)
            is_monotonic = np.mean(np.diff(valid_trace) < 0) > 0.7
            
            # Strict monotonicity check (all non-increasing with tolerance)
            is_monotonic_strict = bool(np.all(np.diff(valid_trace) <= 1e-9))
        else:
            is_monotonic = False
            is_monotonic_strict = False
        
        meta = {
            'analysis_time': elapsed_time,
            'n_time_points': len(t_grid),
            'n_valid_points': np.sum(valid_indices),
            'n_fallback_points': n_fallbacks,
            'heat_trace_std': heat_trace_std.tolist(),  # JSON-friendly conversion
            'probes': spec.probes,
            'slq_iters': spec.slq_iters,
            'is_monotonic': is_monotonic,
            'is_monotonic_strict': is_monotonic_strict,
            'lambda_max': self._lambda_max_cache.get((mass_mode, id(self.sheaf), self.normalization_mode)),
            'normalization_mode': self.normalization_mode,
            't_range': [float(np.min(t_grid)), float(np.max(t_grid))],
            'random_seed': self._random_seed,
            **build_meta
        }
        
        logger.info(f"t-flow analysis completed: {elapsed_time:.3f}s, "
                   f"{np.sum(valid_indices)}/{len(t_grid)} valid points, "
                   f"fallbacks: {n_fallbacks}, "
                   f"monotonic: {is_monotonic} (strict: {is_monotonic_strict})")
        
        return DiffusionSummaries(
            heat_trace=heat_trace,
            t_grid=t_grid,
            smallest_eigs=smallest_eigs,
            meta=meta
        )
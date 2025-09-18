"""Numerical utilities and safeguards for Global Section tracking.

This module provides production-ready numerical operations with comprehensive
error handling, certificates, and deterministic execution for H⁰ persistence
tracking at machine precision.
"""

import torch
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import expm
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import warnings
from dataclasses import dataclass

from ..io.config import H0Config, DEFAULT_H0_CONFIG
from ..io.types import CertificateResult, TransportValidation
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class SmallestEigResult:
    """Result from smallest eigenvalue computation with metadata."""
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray] = None
    converged: bool = True
    num_iterations: int = 0
    residual_norms: Optional[np.ndarray] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def setup_deterministic_execution(cfg: H0Config) -> None:
    """Configure deterministic execution for reproducible results.
    
    Sets all random seeds, enables deterministic algorithms where available,
    and configures warning filters for nondeterministic operations.
    
    Args:
        cfg: H0 configuration with seed and deterministic settings
    """
    if cfg.deterministic:
        # Set PyTorch seeds
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
        
        # Set NumPy seed
        np.random.seed(cfg.seed)
        
        # Enable deterministic algorithms (with warnings for unavailable ones)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Configure cuDNN for deterministic execution
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Deterministic execution enabled with seed {cfg.seed}")


def robust_cholesky_with_fallback(G: torch.Tensor, 
                                cfg: H0Config) -> Tuple[torch.Tensor, bool, float]:
    """Cholesky decomposition with automatic regularization on failure.
    
    Implements the robust Cholesky strategy from the production plan:
    1. Try standard Cholesky
    2. If fails, add diagonal regularization: G + ε×I  
    3. Retry with progressively larger ε
    4. If still fails, fall back to diagonal approximation
    
    Args:
        G: Symmetric positive semi-definite matrix to factorize
        cfg: Configuration with retry parameters
        
    Returns:
        L: Lower triangular Cholesky factor
        regularized: Whether regularization was applied
        regularization_amount: Amount of regularization added
    """
    if G.dtype != getattr(torch, cfg.dtype):
        G = G.to(dtype=getattr(torch, cfg.dtype))
    
    regularization = 0.0
    
    for attempt in range(cfg.max_cholesky_retries):
        try:
            if attempt > 0:
                # Progressive regularization
                regularization = cfg.cholesky_regularization_start * (
                    cfg.cholesky_regularization_factor ** attempt
                )
                G_reg = G + regularization * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
                logger.debug(f"Cholesky attempt {attempt + 1}: regularization = {regularization:.2e}")
            else:
                G_reg = G
            
            L = torch.linalg.cholesky(G_reg)
            
            if attempt > 0:
                logger.info(f"Cholesky succeeded with regularization {regularization:.2e}")
            
            return L, attempt > 0, regularization
            
        except torch.linalg.LinAlgError as e:
            if attempt == cfg.max_cholesky_retries - 1:
                logger.warning(f"Cholesky failed after {cfg.max_cholesky_retries} attempts: {e}")
                break
            continue
    
    # Ultimate fallback: diagonal approximation
    logger.warning("Using diagonal fallback for Cholesky decomposition")
    diagonal = torch.diag(G)
    
    # Ensure positive diagonal
    diagonal = torch.clamp(diagonal, min=cfg.mass_floor_factor * torch.median(diagonal))
    L_fallback = torch.diag(torch.sqrt(diagonal))
    
    return L_fallback, True, float('inf')  # Mark as heavily regularized


def apply_mass_floor(masses: torch.Tensor, cfg: H0Config) -> Tuple[torch.Tensor, bool]:
    """Apply mass floor to prevent division by zero.
    
    Implements the mass floor strategy: masses ← max(masses, ε × median(masses))
    where ε = cfg.mass_floor_factor.
    
    Args:
        masses: Tensor of probability masses
        cfg: Configuration with mass floor factor
        
    Returns:
        masses_safe: Masses with floor applied
        floor_applied: Whether any masses were floored
    """
    if masses.numel() == 0:
        return masses, False
    
    # Compute adaptive floor based on median
    masses_median = torch.median(masses)
    floor_value = cfg.mass_floor_factor * masses_median
    
    # Apply floor
    masses_safe = torch.clamp(masses, min=floor_value)
    
    # Check if floor was actually applied
    floor_applied = not torch.allclose(masses, masses_safe)
    
    if floor_applied:
        n_floored = (masses < floor_value).sum().item()
        logger.debug(f"Applied mass floor {floor_value:.2e} to {n_floored}/{masses.numel()} masses")
    
    return masses_safe, floor_applied


def spectral_norm_estimate(A: torch.Tensor, 
                          n_iterations: int = None,
                          cfg: H0Config = None) -> float:
    """Estimate spectral norm ||A||₂ using power iteration.
    
    Efficient estimation of the largest singular value for threshold computation.
    Uses power iteration with configurable number of steps.
    
    Args:
        A: Matrix to compute spectral norm of
        n_iterations: Number of power iteration steps (defaults to config)
        cfg: Configuration (used for default iterations)
        
    Returns:
        Estimated spectral norm ||A||₂
    """
    cfg = cfg or DEFAULT_H0_CONFIG
    n_iterations = n_iterations or cfg.spectral_norm_iterations
    
    if A.numel() == 0:
        return 0.0
    
    m, n = A.shape
    
    # For small matrices, use exact computation
    if max(m, n) <= 100:
        return torch.linalg.norm(A, ord=2).item()
    
    # Power iteration for large matrices
    device = A.device
    dtype = A.dtype
    
    # Initialize with random vector
    v = torch.randn(n, device=device, dtype=dtype)
    v = v / torch.linalg.norm(v)
    
    for _ in range(n_iterations):
        # v ← A^T A v
        Av = A @ v
        AtAv = A.T @ Av
        
        # Normalize
        v_norm = torch.linalg.norm(AtAv)
        if v_norm > 0:
            v = AtAv / v_norm
        else:
            break
    
    # Final multiplication to get singular value
    Av = A @ v
    sigma_max = torch.linalg.norm(Av).item()
    
    return sigma_max


def compute_numerical_rank_qr(A: torch.Tensor, 
                              cfg: H0Config = None) -> Tuple[int, float]:
    """Compute numerical rank via column-pivoted QR decomposition.
    
    Performs QR on A^T for efficiency and uses diagonal threshold to determine rank.
    This is more reliable than incomplete SVD for wide matrices.
    
    Args:
        A: Matrix to compute rank of (can be wide or tall)
        cfg: Configuration for numerical parameters
        
    Returns:
        rank: Numerical rank of the matrix
        threshold: Threshold used for rank determination
    """
    cfg = cfg or DEFAULT_H0_CONFIG
    
    if A.numel() == 0:
        return 0, 0.0
    
    m, n = A.shape
    
    # Get machine epsilon for the dtype
    if A.dtype == torch.float32:
        eps = np.finfo(np.float32).eps
    elif A.dtype == torch.float64:
        eps = np.finfo(np.float64).eps
    else:
        eps = 1e-15
    
    # Estimate spectral norm for threshold computation
    spectral_norm = spectral_norm_estimate(A, cfg=cfg)
    
    # Compute threshold based on matrix dimensions and spectral norm
    # Standard numerical rank threshold: max(m,n) * eps * ||A||_2
    svd_tol_scale = cfg.svd_zero_tol_scale if hasattr(cfg, 'svd_zero_tol_scale') else 10.0
    threshold = max(m, n) * eps * svd_tol_scale * spectral_norm
    
    try:
        # For wide matrices (m < n), QR on A^T is more efficient
        if m < n:
            # QR decomposition of A^T
            At = A.T
            At_np = At.detach().cpu().numpy()
            
            # Column-pivoted QR decomposition
            Q_np, R_np, P_indices = scipy.linalg.qr(At_np, mode='economic', pivoting=True)
            R = torch.from_numpy(R_np).to(device=A.device, dtype=A.dtype)
        else:
            # For tall matrices, QR on A directly
            A_np = A.detach().cpu().numpy()
            Q_np, R_np, P_indices = scipy.linalg.qr(A_np, mode='economic', pivoting=True)
            R = torch.from_numpy(R_np).to(device=A.device, dtype=A.dtype)
        
        # Count diagonal elements above threshold
        diag_R = torch.abs(torch.diag(R))
        rank = (diag_R > threshold).sum().item()
        
        logger.debug(f"QR rank computation: rank={rank}/{min(m,n)}, threshold={threshold:.2e}, "
                    f"spectral_norm={spectral_norm:.2e}")
        
        return rank, threshold
        
    except Exception as e:
        logger.warning(f"QR rank computation failed: {e}, using fallback")
        # Fallback: use SVD-based rank estimation
        try:
            S = torch.linalg.svdvals(A)
            rank = (S > threshold).sum().item()
            return rank, threshold
        except:
            # Ultimate fallback: assume full rank
            return min(m, n), threshold


def rank_revealing_qr_with_pivoting(Y: torch.Tensor,
                                   threshold: float,
                                   cfg: H0Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Column-pivoted QR decomposition for persistence updates.
    
    Implements RRQR with proper threshold-based rank detection:
    Y @ P = Q @ R where P is permutation, Q orthogonal, R upper triangular.
    
    Args:
        Y: Matrix to decompose (typically F @ Q_prev)
        threshold: Keep threshold τ_keep = c_keep × √ε × ||Y||₂
        cfg: Configuration for numerical parameters
        
    Returns:
        Q: Orthogonal factor  
        R: Upper triangular factor
        P: Permutation matrix
        keep_mask: Boolean mask for kept columns
    """
    if Y.numel() == 0:
        return Y, Y, torch.eye(Y.shape[1]), torch.zeros(Y.shape[1], dtype=torch.bool)
    
    # Use scipy for column-pivoted QR (PyTorch doesn't have pivoting)
    Y_np = Y.detach().cpu().numpy()
    
    try:
        # Column-pivoted QR
        Q_np, R_np, P_indices = scipy.linalg.qr(Y_np, mode='economic', pivoting=True)
        
        # Convert back to PyTorch
        Q = torch.from_numpy(Q_np).to(device=Y.device, dtype=Y.dtype)
        R = torch.from_numpy(R_np).to(device=Y.device, dtype=Y.dtype)
        
        # Create permutation matrix
        n_cols = Y.shape[1]
        P = torch.zeros(n_cols, n_cols, device=Y.device, dtype=Y.dtype)
        for i, j in enumerate(P_indices):
            P[j, i] = 1.0
        
        # Apply threshold to diagonal elements
        diag_R = torch.abs(torch.diag(R))
        keep_mask = diag_R >= threshold
        
        logger.debug(f"RRQR: kept {keep_mask.sum().item()}/{n_cols} columns "
                    f"with threshold {threshold:.2e}")
        
        return Q, R, P, keep_mask
        
    except Exception as e:
        logger.error(f"RRQR failed: {e}, using standard QR")
        
        # Fallback to standard QR without pivoting
        Q, R = torch.linalg.qr(Y, mode='reduced')
        P = torch.eye(Y.shape[1], device=Y.device, dtype=Y.dtype)
        
        # Apply threshold without pivoting information
        diag_R = torch.abs(torch.diag(R))
        keep_mask = diag_R >= threshold
        
        return Q, R, P, keep_mask


def compute_small_svd(A: torch.Tensor, 
                     r: int,
                     method: str = "auto",
                     cfg: H0Config = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute r smallest singular values and vectors efficiently.
    
    Automatically selects between dense and sparse methods based on problem size.
    
    Args:
        A: Matrix to decompose  
        r: Number of smallest singular values to compute
        method: "dense", "sparse", or "auto"
        cfg: Configuration for size thresholds
        
    Returns:
        U: Left singular vectors (last r columns)
        S: Singular values (smallest r)  
        Vt: Right singular vectors (last r rows)
    """
    cfg = cfg or DEFAULT_H0_CONFIG
    
    if A.numel() == 0:
        m, n = A.shape
        return torch.zeros(m, 0), torch.zeros(0), torch.zeros(0, n)
    
    m, n = A.shape
    total_size = m * n
    
    # Automatic method selection
    if method == "auto":
        if total_size <= cfg.dense_threshold ** 2:
            method = "dense"
        else:
            method = "sparse"
    
    if method == "dense":
        # Dense SVD: compute full decomposition then extract r smallest values  
        try:
            # Use full_matrices=False for memory efficiency
            U, S, Vt = torch.linalg.svd(A, full_matrices=False)
            
            # Sort in ascending order (smallest first)
            sorted_indices = torch.argsort(S)
            S_sorted = S[sorted_indices]
            U_sorted = U[:, sorted_indices]
            Vt_sorted = Vt[sorted_indices, :]
            
            m, n = A.shape
            
            # Compute numerical rank with tolerance
            if S.numel() > 0:
                # Get machine epsilon for the dtype
                if A.dtype == torch.float32:
                    eps = np.finfo(np.float32).eps
                elif A.dtype == torch.float64:
                    eps = np.finfo(np.float64).eps
                else:
                    eps = 1e-15
                
                # Compute threshold for numerical rank
                svd_tol_scale = cfg.svd_zero_tol_scale if hasattr(cfg, 'svd_zero_tol_scale') else 10.0
                tau = max(m, n) * eps * svd_tol_scale * S[-1].item()  # S[-1] is largest after sorting
                
                # Determine numerical rank
                numerical_rank = (S > tau).sum().item()
                
                logger.debug(f"SVD numerical rank: {numerical_rank} out of {min(m, n)} possible")
                logger.debug(f"Using threshold {tau:.2e} (scale={svd_tol_scale}, eps={eps:.2e}, σ_max={S[-1].item():.2e})")
            else:
                numerical_rank = 0
                tau = 0.0
            
            # Handle rank deficiency (works for both square and rectangular matrices)
            if numerical_rank < n:  # Rank deficient
                n_zeros = n - numerical_rank
                logger.debug(f"Matrix is rank-deficient: numerical rank {numerical_rank} < dimension {n}")
                
                # Mark small singular values as zeros
                S_sorted[S_sorted <= tau] = 0.0
                
                # If we need to add explicit null vectors (when len(S) < n for rectangular case)
                if len(S) < n:
                    # Compute null space vectors
                    Q_null, _ = torch.linalg.qr(A.T, mode='complete')
                    null_vectors = Q_null[:, numerical_rank:].T
                    
                    # Prepend zeros and null space vectors
                    n_explicit_zeros = n - len(S)
                    S_full = torch.cat([torch.zeros(n_explicit_zeros, dtype=A.dtype, device=A.device), S_sorted])
                    U_full = torch.cat([torch.zeros(m, n_explicit_zeros, dtype=A.dtype, device=A.device), U_sorted], dim=1)
                    Vt_full = torch.cat([null_vectors[:n_explicit_zeros], Vt_sorted], dim=0)
                    
                    S_sorted, U_sorted, Vt_sorted = S_full, U_full, Vt_full
            
            # Select r smallest values (but ensure we have enough)
            r_actual = min(r, len(S_sorted))
            
            # Check if we need more values than what SVD provided (for wide matrices)
            if r > len(S_sorted) and m < n:
                # We need additional zero singular values for the null space
                n_additional = min(r - len(S_sorted), n - len(S_sorted))
                if n_additional > 0:
                    logger.debug(f"Wide matrix: adding {n_additional} additional null space vectors")
                    
                    # Compute complete null space if needed
                    At = A.T
                    Q_complete, _ = torch.linalg.qr(At, mode='complete')
                    null_vectors = Q_complete[:, len(S_sorted):len(S_sorted)+n_additional].T
                    
                    # Extend with zeros and null vectors
                    S_sorted = torch.cat([S_sorted, torch.zeros(n_additional, dtype=A.dtype, device=A.device)])
                    Vt_sorted = torch.cat([Vt_sorted, null_vectors], dim=0)
                    U_sorted = torch.cat([U_sorted, torch.zeros(m, n_additional, dtype=A.dtype, device=A.device)], dim=1)
                    
                    r_actual = len(S_sorted)
            
            S_small = S_sorted[:r_actual]
            U_small = U_sorted[:, :r_actual]
            Vt_small = Vt_sorted[:r_actual, :]
            
            logger.debug(f"Dense SVD: returning {r_actual} smallest singular values in ascending order")
            if len(S_small) > 0:
                logger.debug(f"Smallest values: {S_small[:min(5, len(S_small))].numpy()}")
            
            return U_small, S_small, Vt_small
            
        except Exception as e:
            logger.warning(f"Dense SVD failed: {e}, trying sparse method")
            method = "sparse"
    
    if method == "sparse":
        # Sparse SVD using scipy  
        try:
            from scipy.sparse.linalg import svds
            
            A_np = A.detach().cpu().numpy()
            
            # For wide matrices where r >= min(m,n), we need full SVD
            if r >= min(m, n):
                logger.debug(f"Requested {r} singular values >= min dimension {min(m,n)}, using dense SVD")
                return compute_small_svd(A, r, method="dense", cfg=cfg)
            
            r_actual = min(r, min(m, n) - 1)  # svds requires k < min(m,n)
            
            if r_actual <= 0:
                # All singular values are needed - use dense
                return compute_small_svd(A, r, method="dense", cfg=cfg)
            
            # Compute smallest singular values
            U_np, S_np, Vt_np = svds(A_np, k=r_actual, which='SM')
            
            # Sort in ascending order (smallest first)
            sorted_indices = np.argsort(S_np)
            S_np = S_np[sorted_indices]
            U_np = U_np[:, sorted_indices]
            Vt_np = Vt_np[sorted_indices, :]
            
            # Convert back to PyTorch
            U = torch.from_numpy(U_np).to(device=A.device, dtype=A.dtype)
            S = torch.from_numpy(S_np).to(device=A.device, dtype=A.dtype)
            Vt = torch.from_numpy(Vt_np).to(device=A.device, dtype=A.dtype)
            
            # For wide matrices, check if we're missing zero singular values
            if m < n and r_actual < r:
                # We have structural zeros that svds couldn't compute
                n_structural_zeros = min(n - m, r - r_actual)
                if n_structural_zeros > 0:
                    logger.debug(f"Adding {n_structural_zeros} structural zero singular values for wide matrix")
                    
                    # Prepend zeros to singular values
                    S = torch.cat([torch.zeros(n_structural_zeros, dtype=S.dtype, device=S.device), S])
                    
                    # Need to compute null space vectors for the zero singular values
                    # Use QR decomposition of A^T to get null space
                    At = A.T
                    Q, R = torch.linalg.qr(At, mode='complete')
                    null_vectors = Q[:, m:m+n_structural_zeros].T  # Null space vectors
                    
                    # Prepend null vectors to Vt
                    Vt = torch.cat([null_vectors, Vt], dim=0)
                    
                    # Prepend zero columns to U
                    U = torch.cat([torch.zeros(m, n_structural_zeros, dtype=U.dtype, device=U.device), U], dim=1)
            
            return U, S, Vt
            
        except ImportError:
            logger.warning("scipy not available for sparse SVD, using dense fallback")
            return compute_small_svd(A, r, method="dense", cfg=cfg)
        except Exception as e:
            logger.error(f"Sparse SVD failed: {e}, using dense fallback")  
            return compute_small_svd(A, r, method="dense", cfg=cfg)


def compute_numerical_certificates(delta_tilde: torch.Tensor,
                                  V0: torch.Tensor,
                                  Y: Optional[torch.Tensor] = None,
                                  Q_hat: Optional[torch.Tensor] = None,
                                  R: Optional[torch.Tensor] = None,
                                  P: Optional[torch.Tensor] = None,
                                  cfg: H0Config = DEFAULT_H0_CONFIG) -> CertificateResult:
    """Comprehensive numerical validation of computed results.
    
    Implements all certificates from the production plan:
    1. Kernel residual: ||δ̃ @ V⁰||₂ / ||δ̃||₂ ≤ 10√ε
    2. RRQR consistency: ||Y - Q̂ @ R @ P^T||_F / ||Y||_F ≤ √ε  
    3. Orthogonality: ||V⁰^T @ V⁰ - I||_F ≤ 10√ε
    
    Args:
        delta_tilde: Whitened coboundary operator δ̃
        V0: Kernel basis V⁰
        Y: Matrix for RRQR validation (optional)
        Q_hat, R, P: RRQR factors (optional)
        cfg: Configuration for thresholds
        
    Returns:
        Comprehensive certificate validation result
    """
    result = CertificateResult(
        kernel_certificate=False,
        rrqr_certificate=True,  # Default true if not tested
        orthogonality_certificate=False,
        kernel_residual=float('inf'),
        kernel_threshold=0.0,
        rrqr_residual=0.0,
        orthogonality_residual=float('inf'),
        all_passed=False,
        warnings=[],
        errors=[]
    )
    
    try:
        # Certificate 1: Kernel residual
        if V0.numel() > 0 and delta_tilde.numel() > 0:
            residual = torch.linalg.norm(delta_tilde @ V0, ord='fro')
            spectral_norm = spectral_norm_estimate(delta_tilde, cfg=cfg)
            threshold = 10.0 * cfg.sqrt_eps * spectral_norm
            
            result.kernel_residual = residual.item()
            result.kernel_threshold = threshold
            result.kernel_certificate = residual <= threshold
            
            if not result.kernel_certificate:
                result.warnings.append(f"Kernel residual {residual:.2e} exceeds threshold {threshold:.2e}")
        
        # Certificate 2: RRQR consistency  
        if all(x is not None for x in [Y, Q_hat, R, P]):
            Y_reconstructed = Q_hat @ R @ P.T
            reconstruction_error = torch.linalg.norm(Y - Y_reconstructed, ord='fro')
            Y_norm = torch.linalg.norm(Y, ord='fro')
            
            if Y_norm > 0:
                relative_error = reconstruction_error / Y_norm
                threshold_rrqr = cfg.sqrt_eps
                
                result.rrqr_residual = relative_error.item()
                result.rrqr_certificate = relative_error <= threshold_rrqr
                
                if not result.rrqr_certificate:
                    result.errors.append(f"RRQR reconstruction error {relative_error:.2e} exceeds {threshold_rrqr:.2e}")
        
        # Certificate 3: Orthogonality
        if V0.numel() > 0 and V0.shape[1] > 0:
            VtV = V0.T @ V0
            I = torch.eye(V0.shape[1], device=V0.device, dtype=V0.dtype)
            orthogonality_error = torch.linalg.norm(VtV - I, ord='fro')
            threshold_ortho = 10.0 * cfg.sqrt_eps
            
            result.orthogonality_residual = orthogonality_error.item()
            result.orthogonality_certificate = orthogonality_error <= threshold_ortho
            
            if not result.orthogonality_certificate:
                result.warnings.append(f"Orthogonality error {orthogonality_error:.2e} exceeds {threshold_ortho:.2e}")
        else:
            # Empty kernel basis is trivially orthogonal
            result.orthogonality_certificate = True
            result.orthogonality_residual = 0.0
        
        # Overall assessment
        result.all_passed = (result.kernel_certificate and 
                            result.rrqr_certificate and 
                            result.orthogonality_certificate)
        
    except Exception as e:
        result.errors.append(f"Certificate computation failed: {e}")
        result.all_passed = False
        logger.error(f"Certificate validation error: {e}")
    
    return result


def hutchinson_trace_power(L: LinearOperator, 
                          power: int, 
                          probes: int = 64,
                          rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """Estimate Tr(L^power) via Rademacher probes.
    
    Uses the Hutchinson trace estimator with Rademacher random vectors (±1 entries)
    to compute an unbiased estimate of Tr(L^k) for arbitrary powers k.
    
    The estimator works as: Tr(L^k) ≈ (1/m) * Σᵢ vᵢᵀ L^k vᵢ
    where vᵢ are Rademacher random vectors and m is the number of probes.
    
    Args:
        L: Linear operator to compute trace of
        power: Power k to compute Tr(L^k)
        probes: Number of random probe vectors (default 64, use 128 for tighter CI)
        rng: Random number generator for reproducibility
        
    Returns:
        mean_estimate: Mean estimate of Tr(L^power)
        empirical_std: Empirical standard deviation for variance control
        
    Mathematical Properties:
        - Unbiased: E[estimate] = Tr(L^k)
        - Variance: Var[estimate] ≈ O(1/probes)
        - Works for any LinearOperator (dense or sparse)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if L.shape[0] != L.shape[1]:
        raise ValueError(f"Linear operator must be square, got shape {L.shape}")
    
    n = L.shape[0]
    estimates = np.zeros(probes)
    
    for i in range(probes):
        # Generate Rademacher probe vector (±1 entries)
        v = 2 * rng.integers(0, 2, size=n) - 1
        v = v.astype(np.float64)
        
        # Compute L^k v by repeatedly applying L
        Lkv = v.copy()
        for _ in range(power):
            Lkv = L @ Lkv
            
        # Compute vᵀ L^k v
        estimates[i] = np.dot(v, Lkv)
    
    mean_estimate = np.mean(estimates)
    empirical_std = np.std(estimates, ddof=1) if probes > 1 else 0.0
    
    logger.debug(f"Hutchinson Tr(L^{power}): {mean_estimate:.6e} ± {empirical_std:.6e} "
                f"(probes={probes})")
    
    return mean_estimate, empirical_std


def smallest_eigs_generalized(L: LinearOperator, 
                             D: LinearOperator, 
                             k: int = 16, 
                             sigma: float = 1e-6, 
                             maxiter: int = 1000,
                             random_state: Optional[np.random.Generator] = None,
                             return_vecs: bool = False) -> SmallestEigResult:
    """Robust smallest eigenpairs via shift-invert with automatic regularization.
    
    Solves the generalized eigenvalue problem L x = λ D x for the k smallest
    eigenvalues using shift-invert mode with automatic regularization for
    numerical stability.
    
    Uses eigsh(L, k, M=D, which='LM', sigma=σ) with σ>0 to find smallest 
    eigenvalues. Internally factorizes A = L - σD using robust sparse LU,
    not Cholesky, to handle potentially indefinite shifted matrices.
    
    Args:
        L: System matrix (LinearOperator or sparse matrix)
        D: Mass matrix (LinearOperator or sparse matrix) 
        k: Number of smallest eigenvalues to compute
        sigma: Shift parameter (σ > 0 for smallest eigenvalues)
        maxiter: Maximum number of iterations
        random_state: Random number generator for reproducibility
        return_vecs: Whether to return eigenvectors
        
    Returns:
        SmallestEigResult with eigenvalues, optional eigenvectors, and metadata
        
    Numerical Properties:
        - Uses robust sparse LU factorization (SuperLU/UMFPACK)
        - Automatically adds εI regularization if factorization fails
        - Clips tiny negative eigenvalues: λ_clipped = max(λ, -1e-12)
        - Sorted in ascending order (smallest first)
    """
    try:
        # Set up random state if provided
        if random_state is not None:
            # eigsh doesn't directly accept np.random.Generator, so we use the legacy interface
            np.random.seed(random_state.integers(0, 2**31))
        
        # Ensure k doesn't exceed matrix dimensions
        n = L.shape[0]
        k_actual = min(k, n - 1)
        
        if k_actual <= 0:
            # Handle trivial case
            result = SmallestEigResult(
                eigenvalues=np.array([]),
                eigenvectors=np.zeros((n, 0)) if return_vecs else None,
                converged=True,
                num_iterations=0
            )
            return result
        
        warnings_list = []
        
        try:
            # Use shift-invert mode to find smallest eigenvalues
            # sigma > 0 transforms smallest eigenvalues to largest of (L - σD)^{-1}
            if return_vecs:
                eigenvalues, eigenvectors, info = eigsh(
                    L, k=k_actual, M=D, which='LM', sigma=sigma,
                    maxiter=maxiter, return_eigenvectors=True, 
                    mode='normal'  # Use normal mode for better stability
                )
            else:
                eigenvalues = eigsh(
                    L, k=k_actual, M=D, which='LM', sigma=sigma,
                    maxiter=maxiter, return_eigenvectors=False,
                    mode='normal'
                )
                eigenvectors = None
            
            # Check convergence
            converged = True  # eigsh raises exception if not converged
            num_iterations = maxiter  # eigsh doesn't return actual iterations
            
        except Exception as e:
            # If shift-invert fails, try adding regularization
            logger.warning(f"Standard shift-invert failed: {e}, trying with regularization")
            
            # Add small regularization to improve conditioning
            eps = 1e-12
            if hasattr(L, 'shape') and hasattr(D, 'shape'):
                n = L.shape[0]
                # Create regularized system: (L + εI) x = λ D x
                if hasattr(L, 'todense'):  # Sparse matrix
                    L_reg = L + eps * sp.identity(n, format='csr')
                else:  # LinearOperator - create regularized version
                    I = sp.identity(n, format='csr')
                    L_reg = LinearOperator(
                        shape=L.shape,
                        matvec=lambda x: L @ x + eps * x,
                        dtype=L.dtype
                    )
                
                try:
                    if return_vecs:
                        eigenvalues, eigenvectors, info = eigsh(
                            L_reg, k=k_actual, M=D, which='LM', sigma=sigma,
                            maxiter=maxiter, return_eigenvectors=True,
                            mode='normal'
                        )
                    else:
                        eigenvalues = eigsh(
                            L_reg, k=k_actual, M=D, which='LM', sigma=sigma,
                            maxiter=maxiter, return_eigenvectors=False,
                            mode='normal'
                        )
                        eigenvectors = None
                    
                    warnings_list.append(f"Used ε={eps} regularization for numerical stability")
                    converged = True
                    num_iterations = maxiter
                    
                except Exception as e2:
                    logger.error(f"Regularized shift-invert also failed: {e2}")
                    # Return empty result with error information
                    return SmallestEigResult(
                        eigenvalues=np.array([]),
                        eigenvectors=np.zeros((n, 0)) if return_vecs else None,
                        converged=False,
                        num_iterations=maxiter,
                        warnings=[f"Both standard and regularized solvers failed: {e2}"]
                    )
            else:
                raise e
        
        # Sort eigenvalues in ascending order (smallest first)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        if eigenvectors is not None:
            eigenvectors = eigenvectors[:, sort_idx]
        
        # Clip tiny negative eigenvalues (numerical errors)
        negative_mask = eigenvalues < 0
        if np.any(negative_mask):
            n_negative = np.sum(negative_mask)
            min_negative = np.min(eigenvalues[negative_mask])
            
            # Clip to small positive value if they're tiny
            if min_negative > -1e-12:
                eigenvalues[negative_mask] = -1e-12
                warnings_list.append(
                    f"Clipped {n_negative} tiny negative eigenvalues (min: {min_negative:.2e})"
                )
            else:
                warnings_list.append(
                    f"Found {n_negative} significant negative eigenvalues (min: {min_negative:.2e})"
                )
        
        # Log results
        if len(eigenvalues) > 0:
            logger.debug(f"Shift-invert solver: computed {len(eigenvalues)} eigenvalues, "
                        f"range [{eigenvalues[0]:.6e}, {eigenvalues[-1]:.6e}], "
                        f"converged={converged}")
        
        return SmallestEigResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            converged=converged,
            num_iterations=num_iterations,
            warnings=warnings_list
        )
        
    except Exception as e:
        logger.error(f"Shift-invert eigenvalue solver failed: {e}")
        n = L.shape[0] if hasattr(L, 'shape') else 0
        return SmallestEigResult(
            eigenvalues=np.array([]),
            eigenvectors=np.zeros((n, 0)) if return_vecs else None,
            converged=False,
            num_iterations=0,
            warnings=[f"Solver completely failed: {e}"]
        )


def heat_trace_slq(L: LinearOperator, 
                   t: float, 
                   probes: int = 64, 
                   iters: int = 30, 
                   rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """SLQ estimate of Tr(exp(-tL)) - numerically stable and unbiased.
    
    Uses Stochastic Lanczos Quadrature to estimate the heat trace Tr(exp(-tL))
    for any positive time t. The method is based on Lanczos approximation of
    the matrix exponential applied to random probe vectors.
    
    The algorithm works by:
    1. Generate m random probe vectors
    2. For each probe, run Lanczos iteration to get T_k tridiagonal matrix
    3. Compute e^(-t*T_k) * e_1 where e_1 = [1,0,...,0]
    4. Average over all probes to get final estimate
    
    Args:
        L: Linear operator (should be symmetric positive semidefinite)
        t: Time parameter (t > 0)
        probes: Number of random probe vectors (default 64, use 128 for tighter CI)
        iters: Number of Lanczos iterations (default 30)
        rng: Random number generator for reproducibility
        
    Returns:
        mean_estimate: Mean estimate of Tr(exp(-tL))
        empirical_std: Empirical standard deviation for variance control
        
    Mathematical Properties:
        - Unbiased: E[estimate] = Tr(exp(-tL))
        - Variance: Var[estimate] ~ O(1/probes)
        - Numerically stable for all t > 0
        - Works with any symmetric LinearOperator
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if L.shape[0] != L.shape[1]:
        raise ValueError(f"Linear operator must be square, got shape {L.shape}")
    
    if t <= 0:
        raise ValueError(f"Time parameter must be positive, got t={t}")
    
    n = L.shape[0]
    estimates = np.zeros(probes)
    
    for probe_idx in range(probes):
        # Generate random probe vector (Gaussian)
        v = rng.standard_normal(n)
        v = v / np.linalg.norm(v)  # Normalize
        
        # Run Lanczos iteration
        alpha = np.zeros(iters)  # Diagonal elements of tridiagonal matrix
        beta = np.zeros(iters-1)  # Off-diagonal elements
        
        # Storage for Lanczos vectors
        q_prev = np.zeros(n)
        q_curr = v.copy()
        
        for j in range(iters):
            # Apply linear operator
            w = L @ q_curr
            
            # Compute diagonal element
            alpha[j] = np.dot(q_curr, w)
            
            # Orthogonalize against previous vector
            if j > 0:
                w = w - beta[j-1] * q_prev
            
            # Orthogonalize against current vector
            w = w - alpha[j] * q_curr
            
            # Compute off-diagonal element and prepare for next iteration
            if j < iters - 1:
                beta[j] = np.linalg.norm(w)
                
                # Check for breakdown
                if beta[j] < 1e-14:
                    # Lanczos breakdown - truncate
                    alpha = alpha[:j+1]
                    beta = beta[:j]
                    break
                
                # Prepare for next iteration
                q_prev = q_curr.copy()
                q_curr = w / beta[j]
        
        # Build tridiagonal matrix T_k
        k = len(alpha)
        if k == 1:
            T = np.array([[alpha[0]]])
        else:
            T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        
        # Compute exp(-t*T) using dense matrix exponential
        # This is efficient since T is small (k x k with k ~= iters)
        try:
            exp_neg_tT = expm(-t * T)
            
            # Extract trace contribution: ||v||^2 * e_1^T exp(-t*T) e_1
            # Since v was normalized, ||v||^2 = 1
            trace_contrib = exp_neg_tT[0, 0] * (np.linalg.norm(v)**2)
            estimates[probe_idx] = trace_contrib
            
        except Exception as e:
            logger.warning(f"Matrix exponential failed for probe {probe_idx}: {e}")
            # Use fallback: exp(-t*alpha[0]) if only one Lanczos step
            if k >= 1:
                estimates[probe_idx] = np.exp(-t * alpha[0])
            else:
                estimates[probe_idx] = 0.0
    
    # Average over probes to get final estimate
    mean_estimate = n * np.mean(estimates)  # Factor of n from trace normalization
    empirical_std = n * np.std(estimates, ddof=1) if probes > 1 else 0.0
    
    logger.debug(f"Heat trace SLQ at t={t:.6e}: {mean_estimate:.6e} ± {empirical_std:.6e} "
                f"(probes={probes}, lanczos_iters={iters})")
    
    return mean_estimate, empirical_std


def estimate_lambda_max(L: LinearOperator, 
                       iters: int = 40,
                       rng: Optional[np.random.Generator] = None) -> float:
    """Power iteration for crude λ_max estimation.
    
    Uses the power method to estimate the largest eigenvalue of a symmetric
    positive semidefinite linear operator. This is used primarily for automatic
    t-grid generation in diffusion flow analysis.
    
    The power method iteratively applies L to a random vector and normalizes:
    v_{k+1} = L v_k / ||L v_k||
    
    The largest eigenvalue is approximated by the Rayleigh quotient:
    λ_max ≈ v_k^T L v_k / ||v_k||^2
    
    Args:
        L: Linear operator (should be symmetric positive semidefinite)
        iters: Number of power iterations (default 40)
        rng: Random number generator for reproducibility
        
    Returns:
        Estimated largest eigenvalue λ_max
        
    Mathematical Properties:
        - Converges to largest eigenvalue if it's simple
        - Convergence rate: O((λ_2/λ_1)^k) where λ_1 > λ_2
        - Works for any symmetric LinearOperator
        - Cheap approximation suitable for t-grid generation
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if L.shape[0] != L.shape[1]:
        raise ValueError(f"Linear operator must be square, got shape {L.shape}")
    
    n = L.shape[0]
    
    if n == 0:
        return 0.0
    
    # Initialize with random vector
    v = rng.standard_normal(n)
    v_norm = np.linalg.norm(v)
    
    if v_norm == 0:
        # Fallback to ones vector
        v = np.ones(n)
        v_norm = np.sqrt(n)
    
    v = v / v_norm
    
    lambda_max_est = 0.0
    
    for i in range(iters):
        # Apply linear operator
        Lv = L @ v
        
        # Compute Rayleigh quotient as eigenvalue estimate
        lambda_max_est = np.dot(v, Lv)
        
        # Normalize for next iteration
        Lv_norm = np.linalg.norm(Lv)
        
        if Lv_norm < 1e-14:
            # Near-zero vector, likely converged to zero eigenspace
            logger.debug(f"Power iteration converged to zero eigenspace at iteration {i}")
            break
        
        v = Lv / Lv_norm
    
    # Ensure non-negative (for positive semidefinite matrices)
    lambda_max_est = max(lambda_max_est, 0.0)
    
    logger.debug(f"Power method λ_max estimate: {lambda_max_est:.6e} ({iters} iterations)")
    
    return lambda_max_est


def validate_transport_properties(T_tilde: torch.Tensor,
                                 Pi: torch.Tensor,
                                 masses_t: torch.Tensor,
                                 masses_tp1: torch.Tensor,
                                 cfg: H0Config = DEFAULT_H0_CONFIG) -> TransportValidation:
    """Validate transport map properties and marginal consistency.
    
    Checks mathematical properties of transport maps constructed from
    GW couplings to ensure they preserve the required structure.
    
    Args:
        T_tilde: Whitened transport map T̃  
        Pi: Original GW coupling matrix π
        masses_t: Source masses a_t
        masses_tp1: Target masses b_{t+1}
        cfg: Configuration
        
    Returns:
        Comprehensive transport validation result
    """
    result = TransportValidation(
        marginal_consistency=False,
        source_marginal_error=float('inf'),
        target_marginal_error=float('inf'),
        transport_norm=0.0,
        condition_number=float('inf'),
        mass_preservation=False,
        warnings=[],
        errors=[]
    )
    
    try:
        # Check marginal consistency of original coupling
        if Pi.numel() > 0:
            # Row sums should equal target masses (or realized marginals)
            row_sums = Pi.sum(dim=1)
            target_error = torch.linalg.norm(row_sums - masses_t).item()
            
            # Column sums should equal source masses (or realized marginals)  
            col_sums = Pi.sum(dim=0)
            source_error = torch.linalg.norm(col_sums - masses_tp1).item()
            
            result.source_marginal_error = source_error
            result.target_marginal_error = target_error
            
            # Tolerance based on problem scale
            tolerance = cfg.sqrt_eps * max(torch.sum(masses_t).item(), torch.sum(masses_tp1).item())
            result.marginal_consistency = (source_error <= tolerance and target_error <= tolerance)
            
            if not result.marginal_consistency:
                result.warnings.append(f"Marginal errors: source={source_error:.2e}, target={target_error:.2e}")
        
        # Transport map properties
        if T_tilde.numel() > 0:
            result.transport_norm = torch.linalg.norm(T_tilde, ord=2).item()
            
            try:
                result.condition_number = torch.linalg.cond(T_tilde).item()
            except:
                result.condition_number = float('inf')
                result.warnings.append("Could not compute transport condition number")
            
            # Check for excessive condition number
            if result.condition_number > 1.0 / cfg.transport_noise_threshold:
                result.warnings.append(f"Transport map is ill-conditioned: cond={result.condition_number:.2e}")
        
        # Mass preservation check
        total_mass_t = torch.sum(masses_t).item()
        total_mass_tp1 = torch.sum(masses_tp1).item()
        mass_difference = abs(total_mass_t - total_mass_tp1)
        mass_tolerance = cfg.sqrt_eps * max(total_mass_t, total_mass_tp1)
        
        result.mass_preservation = mass_difference <= mass_tolerance
        
        if not result.mass_preservation:
            result.warnings.append(f"Mass not preserved: {total_mass_t:.6f} → {total_mass_tp1:.6f}")
    
    except Exception as e:
        result.errors.append(f"Transport validation failed: {e}")
        logger.error(f"Transport validation error: {e}")
    
    return result
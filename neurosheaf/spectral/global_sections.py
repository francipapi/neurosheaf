"""Global Section (H⁰) tracking with whitening and kernel extraction.

This module implements the core mathematical operations for Global Section
persistence tracking, including numerically stable whitening of coboundary
operators and hysteretic kernel basis computation in σ-space.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..io.config import H0Config, DEFAULT_H0_CONFIG
from ..io.types import WhiteningResult, KernelResult, CertificateResult
from .utils_numerical import (
    robust_cholesky_with_fallback,
    apply_mass_floor,
    spectral_norm_estimate,
    compute_small_svd,
    compute_numerical_certificates,
    setup_deterministic_execution,
    compute_numerical_rank_qr
)
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class GlobalSectionProcessor:
    """Core processor for Global Section (H⁰) tracking operations.
    
    This class implements the mathematical core of transport-informed
    H⁰ persistence tracking with numerical stability at machine precision.
    
    Key Features:
    - Robust whitening using Cholesky decomposition with fallback
    - Hysteretic σ-space kernel classification  
    - Numerical certificates for all operations
    - Comprehensive error recovery mechanisms
    """
    
    def __init__(self, cfg: H0Config = None):
        """Initialize Global Section processor.
        
        Args:
            cfg: Configuration for numerical parameters and thresholds
        """
        self.cfg = cfg or DEFAULT_H0_CONFIG
        self.cfg.validate()
        
        # Setup deterministic execution if requested
        if self.cfg.deterministic:
            setup_deterministic_execution(self.cfg)
        
        # Cache for expensive computations
        self._cholesky_cache = {} if self.cfg.cache_cholesky else None
        self._last_spectral_norm = None
        
        logger.info(f"GlobalSectionProcessor initialized: dtype={self.cfg.dtype}, "
                   f"c_in={self.cfg.c_in}, c_keep={self.cfg.c_keep}, "
                   f"deterministic={self.cfg.deterministic}")
    
    def whiten_coboundary_robust(self, 
                                delta: torch.Tensor,
                                G0: torch.Tensor, 
                                G1: torch.Tensor,
                                step_id: Optional[str] = None) -> WhiteningResult:
        """Whiten coboundary operator using numerically stable methods.
        
        Implements δ̃ = G₁^(1/2) @ δ @ G₀^(-1/2) using Cholesky decomposition
        with triangular solves to avoid explicit matrix inverses.
        
        Mathematical Foundation:
        - G₀ = L₀ L₀^T (Cholesky of 0-cochain metric)
        - G₁ = L₁ L₁^T (Cholesky of 1-cochain metric)  
        - δ̃ = L₁ @ δ @ solve_triangular(L₀, I)
        
        Args:
            delta: Coboundary operator δ
            G0: Metric on 0-cochains (stalk masses or full SPD matrix)
            G1: Metric on 1-cochains (edge weights or full SPD matrix)
            step_id: Optional identifier for caching
            
        Returns:
            WhiteningResult with δ̃, factors, and diagnostics
        """
        start_time = time.time()
        
        # Ensure correct dtype
        target_dtype = getattr(torch, self.cfg.dtype)
        delta = delta.to(dtype=target_dtype)
        G0 = G0.to(dtype=target_dtype)  
        G1 = G1.to(dtype=target_dtype)
        
        logger.debug(f"Whitening coboundary: δ shape {delta.shape}, "
                    f"G0 shape {G0.shape}, G1 shape {G1.shape}")
        
        # Check cache for Cholesky factors using content-based hashing
        cache_key = None
        if step_id and self._cholesky_cache:
            # Use step_id and matrix content hash for robust caching
            G0_hash = torch.sum(G0).item() + G0.shape[0] * G0.shape[1]  # Simple content hash
            G1_hash = torch.sum(G1).item() + G1.shape[0] * G1.shape[1]  # Simple content hash
            cache_key = f"{step_id}_{G0_hash:.6e}_{G1_hash:.6e}"
        if cache_key and cache_key in self._cholesky_cache:
            L0, L1, G0_reg, G1_reg, reg_amount = self._cholesky_cache[cache_key]
            logger.debug(f"Using cached Cholesky factors for {cache_key}")
        else:
            # Compute Cholesky factorizations with robust fallback
            # robust_cholesky_with_fallback returns (L, regularized_flag, regularization_amount)
            L0, G0_regularized_flag, reg_amount_0 = robust_cholesky_with_fallback(G0, self.cfg)
            L1, G1_regularized_flag, reg_amount_1 = robust_cholesky_with_fallback(G1, self.cfg)
            
            reg_amount = max(reg_amount_0, reg_amount_1)
            # Create regularized matrices if regularization was applied
            if G0_regularized_flag:
                G0_reg = G0 + reg_amount_0 * torch.eye(G0.shape[0], dtype=G0.dtype, device=G0.device)
            else:
                G0_reg = G0
            
            if G1_regularized_flag:
                G1_reg = G1 + reg_amount_1 * torch.eye(G1.shape[0], dtype=G1.dtype, device=G1.device)
            else:
                G1_reg = G1
            
            # Cache factors if enabled
            if cache_key:
                self._cholesky_cache[cache_key] = (L0, L1, G0_reg, G1_reg, reg_amount)
        
        try:
            # Compute whitened coboundary: δ̃ = L₁ @ δ @ L₀^(-1)
            # Use triangular solve instead of explicit inverse
            if G0.shape[0] == delta.shape[1]:
                # Standard case: solve δ @ L₀^(-1)
                delta_L0inv = torch.linalg.solve_triangular(
                    L0.T, delta.T, upper=True
                ).T
            else:
                # Handle dimension mismatch gracefully
                logger.warning(f"Dimension mismatch: G0 {G0.shape[0]} vs delta cols {delta.shape[1]}")
                # Use pseudoinverse as fallback
                L0_inv = torch.linalg.pinv(L0)
                delta_L0inv = delta @ L0_inv
            
            # Complete whitening: δ̃ = L₁ @ (δ @ L₀^(-1))
            delta_tilde = L1 @ delta_L0inv
            
            # Compute spectral norm for threshold computation
            spectral_norm = spectral_norm_estimate(delta_tilde, cfg=self.cfg)
            self._last_spectral_norm = spectral_norm
            
            # Create result
            result = WhiteningResult(
                delta_tilde=delta_tilde,
                L0=L0,
                L1=L1,
                G0_regularized=G0_reg,
                G1_regularized=G1_reg,
                regularization_applied=reg_amount,
                spectral_norm=spectral_norm
            )
            
            computation_time = time.time() - start_time
            logger.debug(f"Whitening completed in {computation_time:.3f}s, "
                        f"spectral norm: {spectral_norm:.2e}, "
                        f"regularization: {reg_amount:.2e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Coboundary whitening failed: {e}")
            raise RuntimeError(f"Failed to whiten coboundary operator: {e}")
    
    def kernel_basis_with_hysteresis(self,
                                   delta_tilde: torch.Tensor,
                                   S_prev: Optional[float] = None,
                                   labels_prev: Optional[List[str]] = None,
                                   prev_nullity: Optional[int] = None) -> KernelResult:
        """Compute orthonormal kernel basis using hysteretic σ-space classification.
        
        Implements the core algorithm from the production plan:
        1. Compute r = prev_nullity + cfg.margin smallest singular values
        2. Apply hysteretic labeling: σ ≤ τ_in → ZERO, σ ≥ τ_out → NONZERO  
        3. Extract kernel basis V⁰ from ZERO-labeled right singular vectors
        4. Verify residual: ||δ̃ @ V⁰||₂ ≤ 10√ε × ||δ̃||₂
        
        Args:
            delta_tilde: Whitened coboundary operator δ̃
            S_prev: Previous spectral norm for consistency
            labels_prev: Previous singular value labels for hysteresis
            prev_nullity: Previous kernel dimension for SVD sizing
            
        Returns:
            KernelResult with V⁰, dimensions, and validation certificates
        """
        start_time = time.time()
        
        if delta_tilde.numel() == 0:
            logger.info("🔄 Empty coboundary δ = 0: kernel is entire stalk space")
            return self._create_full_stalk_kernel_result(delta_tilde)
        
        # Determine spectral norm for threshold computation
        if S_prev is not None and abs(S_prev - (self._last_spectral_norm or 0)) < 0.1 * S_prev:
            # Reuse previous spectral norm if consistent
            spectral_norm = S_prev
        else:
            spectral_norm = spectral_norm_estimate(delta_tilde, cfg=self.cfg)
        
        # Create thresholds using configuration
        thresholds = self.cfg.create_step_thresholds(spectral_norm)
        tau_in = thresholds['tau_in']
        tau_out = thresholds['tau_out']
        
        logger.info(f"🔍 KERNEL COMPUTATION DIAGNOSTICS:")
        logger.info(f"   Spectral norm ||δ̃||₂ = {spectral_norm:.2e}")
        logger.info(f"   τ_in = {tau_in:.2e} (c_in={self.cfg.c_in:.1f} × √ε × ||δ̃||₂)")
        logger.info(f"   τ_out = {tau_out:.2e} (gap={self.cfg.gap:.1f} × τ_in)")
        logger.info(f"   sqrt_eps = {self.cfg.sqrt_eps:.2e}")
        
        # Compute numerical rank to determine nullity
        m, n = delta_tilde.shape
        rank, rank_threshold = compute_numerical_rank_qr(delta_tilde, self.cfg)
        nullity = n - rank
        
        # Extract current edge count from coboundary matrix dimensions
        current_edge_count = m  # Number of edges = number of rows in coboundary
        
        # Check for degeneracy condition: more edges should reduce nullity
        if prev_nullity is not None:
            expected_nullity_decrease = prev_nullity >= nullity
            more_edges = current_edge_count > getattr(self, '_prev_edge_count', 0)
            
            # Only warn if we have more edges but nullity didn't decrease as expected
            if more_edges and not expected_nullity_decrease:
                logger.warning(f"   ⚠️  DEGENERACY DETECTED: {current_edge_count} edges (prev: {getattr(self, '_prev_edge_count', 0)}) "
                              f"but nullity {nullity} ≥ previous {prev_nullity} (expected decrease)")
            
            # Cache edge count for next iteration
            self._prev_edge_count = current_edge_count
        else:
            # First iteration - just cache the edge count
            self._prev_edge_count = current_edge_count
        
        logger.info(f"   Matrix dimensions: {m}×{n} ({current_edge_count} edges)")
        logger.info(f"   Numerical rank: {rank} (threshold={rank_threshold:.2e})")
        logger.info(f"   Expected nullity: {nullity} (n - rank)")
        
        # Determine number of singular values to compute based on nullity
        if prev_nullity is not None:
            # Add margin to previous nullity for hysteresis
            r = min(prev_nullity + self.cfg.margin, n)
        else:
            # Use computed nullity plus margin
            r = min(nullity + self.cfg.margin, n)
        
        # For memory efficiency, cap at a reasonable maximum if nullity is huge
        max_svd_compute = min(1000, n)  # Configurable max for safety
        if r > max_svd_compute:
            logger.warning(f"   Large nullity {nullity}: limiting SVD computation to {max_svd_compute} values")
            r = max_svd_compute
        
        r = max(r, 1)  # Compute at least 1 singular value
        
        # Compute smallest singular values
        try:
            U, S, Vt = compute_small_svd(delta_tilde, r, cfg=self.cfg)
            
            if S.numel() == 0:
                logger.warning("SVD returned no singular values, using trivial kernel")
                return self._create_trivial_kernel_result()
            
            # Detailed singular value analysis
            logger.info(f"   Computing {r} smallest singular values for matrix {delta_tilde.shape}")
            logger.info(f"   Raw singular values: {S.numpy()}")
            logger.info(f"   Smallest σ = {S[0].item():.2e}, Largest σ = {S[-1].item():.2e}")
            
            # Compute σ_min/σ_max ratio with proper division by zero handling
            if S[-1].item() != 0:
                ratio = (S[0] / S[-1]).item()
                logger.info(f"   σ_min/σ_max ratio = {ratio:.2e}")
            else:
                logger.info(f"   σ_min/σ_max ratio = undefined (σ_max = 0)")
            
        except Exception as e:
            logger.error(f"SVD computation failed: {e}")
            return self._create_trivial_kernel_result()
        
        # Apply hysteretic labeling in σ-space
        labels = self._apply_hysteretic_labeling(
            S, tau_in, tau_out, labels_prev
        )
        
        # Detailed classification analysis
        zero_indices = [i for i, label in enumerate(labels) if label == "ZERO"]
        nonzero_indices = [i for i, label in enumerate(labels) if label == "NONZERO"]
        kernel_dimension = len(zero_indices)
        
        # Conditional detailed logging based on log level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"   🏷️  DETAILED SINGULAR VALUE CLASSIFICATION:")
            for i, (sigma, label) in enumerate(zip(S, labels)):
                comparison = "≤" if sigma <= tau_in else "≥" if sigma >= tau_out else "∈"
                logger.debug(f"      σ_{i} = {sigma:.2e} {comparison} threshold → {label}")
        else:
            # Concise summary for INFO level
            n_zero = len(zero_indices) 
            n_nonzero = len(nonzero_indices)
            if len(S) > 6:
                logger.info(f"   🏷️  SINGULAR VALUES: σ_0..σ_2 = [{S[0]:.2e}, {S[1]:.2e}, {S[2]:.2e}], "
                           f"σ_{len(S)-3}..σ_{len(S)-1} = [{S[-3]:.2e}, {S[-2]:.2e}, {S[-1]:.2e}]")
            elif len(S) > 3:
                logger.info(f"   🏷️  SINGULAR VALUES: [{', '.join(f'{s:.2e}' for s in S[:3])}, ..., {', '.join(f'{s:.2e}' for s in S[-3:])}]")
            else:
                logger.info(f"   🏷️  SINGULAR VALUES: [{', '.join(f'{s:.2e}' for s in S)}]")
        
        logger.info(f"   📊 CLASSIFICATION SUMMARY:")
        logger.info(f"      ZERO labels: {len(zero_indices)} → kernel dim = {kernel_dimension}")
        logger.info(f"      NONZERO labels: {len(nonzero_indices)}")
        
        if kernel_dimension == 0:
            logger.warning(f"   ⚠️  NO GLOBAL SECTIONS DETECTED!")
            logger.warning(f"      All {len(S)} singular values > τ_in = {tau_in:.2e}")
            logger.warning(f"      Smallest σ = {S[-1]:.2e} is {(S[-1]/tau_in):.1f}× larger than threshold")
        
        if kernel_dimension > 0:
            logger.info(f"   ✅ GLOBAL SECTIONS DETECTED! dim = {kernel_dimension}")
            
            # Extract right singular vectors corresponding to zero singular values
            V0 = Vt[zero_indices, :].T  # Transpose to get column vectors
            
            # Ensure orthonormality (should already be from SVD)
            if V0.shape[1] > 1:
                V0, _ = torch.linalg.qr(V0, mode='reduced')
        else:
            # No kernel - create empty basis
            V0 = torch.zeros(delta_tilde.shape[1], 0, 
                           dtype=delta_tilde.dtype, device=delta_tilde.device)
        
        # Compute residual certificate
        residual_norm = self._compute_kernel_residual(delta_tilde, V0, spectral_norm)
        
        # Check for hysteresis application
        hysteresis_applied = (labels_prev is not None and 
                            labels != self._classify_without_hysteresis(S, tau_in))
        
        result = KernelResult(
            V0=V0,
            kernel_dimension=kernel_dimension,
            singular_values=S,
            labels=labels,
            spectral_norm=spectral_norm,
            tau_in=tau_in,
            tau_out=tau_out,
            residual_norm=residual_norm,
            hysteresis_applied=hysteresis_applied
        )
        
        computation_time = time.time() - start_time
        logger.debug(f"Kernel computation completed in {computation_time:.3f}s: "
                    f"dim(ker)={kernel_dimension}, residual={residual_norm:.2e}")
        
        return result
    
    def validate_kernel_certificate(self, 
                                  delta_tilde: torch.Tensor,
                                  V0: torch.Tensor, 
                                  spectral_norm: float) -> CertificateResult:
        """Validate kernel basis with numerical certificate.
        
        Certificate: ||δ̃ @ V⁰||₂ / ||δ̃||₂ ≤ 10√ε
        
        Args:
            delta_tilde: Whitened coboundary operator
            V0: Kernel basis to validate
            spectral_norm: ||δ̃||₂ for normalization
            
        Returns:
            Certificate validation result
        """
        return compute_numerical_certificates(
            delta_tilde=delta_tilde,
            V0=V0,
            cfg=self.cfg
        )
    
    def _apply_hysteretic_labeling(self,
                                  singular_values: torch.Tensor,
                                  tau_in: float,
                                  tau_out: float, 
                                  labels_prev: Optional[List[str]] = None) -> List[str]:
        """Apply hysteretic thresholding to singular values.
        
        Hysteresis rules:
        - If σ ≤ τ_in: label as ZERO
        - If σ ≥ τ_out: label as NONZERO  
        - If τ_in < σ < τ_out: keep previous label (prevent flickering)
        
        Args:
            singular_values: Singular values to classify
            tau_in: Entry threshold (classify as zero)
            tau_out: Exit threshold (classify as nonzero)
            labels_prev: Previous labels for hysteresis
            
        Returns:
            List of "ZERO" or "NONZERO" labels
        """
        n_values = len(singular_values)
        labels = ["NONZERO"] * n_values  # Default to nonzero
        
        for i, sigma in enumerate(singular_values):
            sigma_val = sigma.item()
            
            if sigma_val <= tau_in:
                # Definitely zero
                labels[i] = "ZERO"
            elif sigma_val >= tau_out:
                # Definitely nonzero
                labels[i] = "NONZERO"
            else:
                # Hysteresis region - keep previous label if available
                if labels_prev and i < len(labels_prev):
                    labels[i] = labels_prev[i]
                else:
                    # No previous info - default to nonzero for safety
                    labels[i] = "NONZERO"
        
        return labels
    
    def _classify_without_hysteresis(self,
                                   singular_values: torch.Tensor,
                                   tau_in: float) -> List[str]:
        """Classify singular values without hysteresis for comparison."""
        return ["ZERO" if sigma.item() <= tau_in else "NONZERO" 
                for sigma in singular_values]
    
    def _compute_kernel_residual(self,
                               delta_tilde: torch.Tensor,
                               V0: torch.Tensor,
                               spectral_norm: float) -> float:
        """Compute kernel residual ||δ̃ @ V⁰||₂ / ||δ̃||₂."""
        if V0.numel() == 0 or spectral_norm == 0:
            return 0.0
        
        residual_unnormalized = torch.linalg.norm(delta_tilde @ V0, ord='fro')
        return (residual_unnormalized / spectral_norm).item()
    
    def _create_trivial_kernel_result(self) -> KernelResult:
        """Create result for trivial (empty) kernel case."""
        return KernelResult(
            V0=torch.zeros(1, 0),  # Empty basis  
            kernel_dimension=0,
            singular_values=torch.zeros(0),
            labels=[],
            spectral_norm=0.0,
            tau_in=0.0,
            tau_out=0.0,
            residual_norm=0.0,
            hysteresis_applied=False
        )
    
    def _create_full_stalk_kernel_result(self, delta_tilde: torch.Tensor) -> KernelResult:
        """Create result for empty coboundary case where kernel is entire stalk space.
        
        When δ = 0 (no constraints), every vector in the stalk space is a global section.
        This represents the disconnected/trivial sheaf state.
        
        Args:
            delta_tilde: Empty coboundary with shape (0, n)
            
        Returns:
            KernelResult with full stalk space as kernel
        """
        n = delta_tilde.shape[1]  # Total stalk dimension
        
        # Kernel basis is the identity matrix (entire stalk space)
        V0 = torch.eye(n, dtype=delta_tilde.dtype, device=delta_tilde.device)
        
        # All singular values are zero (no constraints)
        singular_values = torch.zeros(n, dtype=delta_tilde.dtype, device=delta_tilde.device)
        labels = ["ZERO"] * n
        
        logger.info(f"🔄 Full stalk kernel: dimension = {n} (entire stalk space)")
        
        return KernelResult(
            V0=V0,
            kernel_dimension=n,
            singular_values=singular_values,
            labels=labels,
            spectral_norm=0.0,  # Empty matrix has zero spectral norm
            tau_in=0.0,
            tau_out=0.0,
            residual_norm=0.0,  # ||0 @ I||₂ = 0
            hysteresis_applied=False
        )
    
    def clear_cache(self):
        """Clear Cholesky factor cache."""
        if self._cholesky_cache:
            self._cholesky_cache.clear()
            logger.debug("Cleared Cholesky factor cache")


# Convenience functions for common usage patterns

def whiten_delta_from_masses(delta: torch.Tensor,
                            node_masses: torch.Tensor,
                            edge_weights: torch.Tensor,
                            cfg: H0Config = None) -> WhiteningResult:
    """Convenience function for whitening with diagonal mass metrics.
    
    Creates G₀ = diag(node_masses) and G₁ = diag(edge_weights) then
    performs whitening using the most efficient diagonal operations.
    
    Args:
        delta: Coboundary operator
        node_masses: Probability masses for nodes (0-cochains)
        edge_weights: Weights for edges (1-cochains)
        cfg: Configuration
        
    Returns:
        WhiteningResult with optimized diagonal computation
    """
    cfg = cfg or DEFAULT_H0_CONFIG
    
    # Apply mass floors for numerical stability
    node_masses_safe, _ = apply_mass_floor(node_masses, cfg)
    edge_weights_safe, _ = apply_mass_floor(edge_weights, cfg)
    
    # For diagonal metrics, whitening simplifies to element-wise operations
    # δ̃ = diag(√w) @ δ @ diag(1/√a) = diag(√w / √a) applied to δ
    
    # But we need to maintain the full interface for consistency
    G0 = torch.diag(node_masses_safe)
    G1 = torch.diag(edge_weights_safe)
    
    processor = GlobalSectionProcessor(cfg)
    return processor.whiten_coboundary_robust(delta, G0, G1)


def extract_kernel_from_laplacian(laplacian: torch.Tensor,
                                 cfg: H0Config = None) -> KernelResult:
    """Extract kernel basis directly from Laplacian eigendecomposition.
    
    For comparison with coboundary-based approach. Uses Laplacian
    eigenvalues λ = σ² relationship.
    
    Args:
        laplacian: Sheaf Laplacian L = δ^T δ
        cfg: Configuration
        
    Returns:
        KernelResult from Laplacian eigendecomposition
    """
    cfg = cfg or DEFAULT_H0_CONFIG
    
    # Compute smallest eigenvalues/eigenvectors  
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
        
        # Convert λ to σ: σ = √λ (but be careful with negative eigenvals due to numerics)
        eigenvals_pos = torch.clamp(eigenvals, min=0)
        singular_values = torch.sqrt(eigenvals_pos)
        
        # Use same threshold logic but convert to λ-space
        spectral_norm = torch.sqrt(eigenvals[-1]).item()  # σ_max = √λ_max
        tau_in_sigma = cfg.c_in * cfg.sqrt_eps * spectral_norm
        tau_in_lambda = tau_in_sigma ** 2
        
        # Find kernel (eigenvals ≤ threshold)
        zero_mask = eigenvals <= tau_in_lambda
        kernel_dimension = zero_mask.sum().item()
        
        V0 = eigenvecs[:, zero_mask] if kernel_dimension > 0 else torch.zeros(laplacian.shape[0], 0)
        
        return KernelResult(
            V0=V0,
            kernel_dimension=kernel_dimension,
            singular_values=singular_values,
            labels=["ZERO" if zero_mask[i] else "NONZERO" for i in range(len(eigenvals))],
            spectral_norm=spectral_norm,
            tau_in=tau_in_sigma,
            tau_out=3.0 * tau_in_sigma,  # Default gap
            residual_norm=0.0,  # Exact for eigendecomposition
            hysteresis_applied=False
        )
        
    except Exception as e:
        logger.error(f"Laplacian eigendecomposition failed: {e}")
        raise RuntimeError(f"Failed to extract kernel from Laplacian: {e}")
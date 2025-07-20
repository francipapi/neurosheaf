"""Whitening transformation for exact metric compatibility.

This module implements the whitening transformation that enables exact
mathematical properties in sheaf construction:
- Spectral factorization of Gram matrices
- Whitening maps for coordinate transformation
- Identity inner products in whitened spaces
- Exact metric compatibility for restriction maps

The whitening approach is the mathematical foundation that allows
sheaf construction to achieve exact rather than approximate properties.
"""

from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)


class WhiteningProcessor:
    """Implements whitening transformation for exact metric compatibility.
    
    This class provides the whitening transformation described in the pipeline report:
    - Spectral factorization: K_v = U_v Σ_v U_v^T
    - Whitening map: W_v = Σ_v^(-1/2) U_v^T : ℝ^n → ℝ^r_v
    - Whitened stalk: Ṽ_v = (ℝ^r_v, ⟨·,·⟩_I) with identity inner product
    - Whitened restrictions: R̃_e = W_w R_e W_v^† achieve exact orthogonality
    
    This enables exact metric compatibility: R̃_e^T R̃_e = I in whitened coordinates.
    """
    
    def __init__(self, min_eigenvalue: float = 1e-8, regularization: float = 1e-10, 
                 use_double_precision: bool = False, preserve_eigenvalues: bool = False,
                 use_matrix_rank: bool = True):
        """Initialize whitening processor.
        
        Args:
            min_eigenvalue: Minimum eigenvalue threshold for numerical stability
            regularization: Small value added to eigenvalues for regularization
            use_double_precision: Whether to use double precision for numerical computations
            preserve_eigenvalues: Whether to preserve eigenvalues in diagonal form instead of identity
            use_matrix_rank: Whether to use torch.linalg.matrix_rank for rank computation (legacy behavior)
        """
        self.min_eigenvalue = min_eigenvalue
        self.regularization = regularization
        # Disable double precision on MPS devices (Apple Silicon) as they don't support float64
        if use_double_precision and torch.backends.mps.is_available():
            logger.warning("Disabling double precision on MPS device (Apple Silicon) - float64 not supported")
            use_double_precision = False
        self.use_double_precision = use_double_precision
        self.preserve_eigenvalues = preserve_eigenvalues
        self.use_matrix_rank = use_matrix_rank
    
    @classmethod
    def create_adaptive(cls, batch_size: int, min_eigenvalue: float = 1e-8, 
                       regularization: float = 1e-10):
        """Create WhiteningProcessor with precision adapted to batch size.
        
        Args:
            batch_size: Size of the batch being processed
            min_eigenvalue: Minimum eigenvalue threshold for numerical stability
            regularization: Small value added to eigenvalues for regularization
            
        Returns:
            WhiteningProcessor instance with appropriate precision settings
        """
        # Use double precision for large batch sizes where numerical issues are common
        use_double_precision = batch_size >= 64
        
        if use_double_precision:
            # More stringent thresholds for double precision
            min_eigenvalue = min(min_eigenvalue, 1e-12)
            regularization = min(regularization, 1e-15)
            logger.info(f"Using double precision for batch size {batch_size}")
        else:
            logger.info(f"Using single precision for batch size {batch_size}")
            
        return cls(min_eigenvalue=min_eigenvalue, 
                  regularization=regularization,
                  use_double_precision=use_double_precision)
    
    def compute_whitening_map(self, K: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute whitening map W = Σ^(-1/2) U^T from Gram matrix K.
        
        Args:
            K: Gram matrix (n x n), possibly rank-deficient
            
        Returns:
            Tuple of (W, info):
            - W: Whitening map (r x n) where r = effective rank
            - info: Dictionary with whitening metadata
        """
        # Use double precision for critical numerical operations
        if self.use_double_precision:
            K_compute = K.double()
        else:
            K_compute = K
            
        K_np = K_compute.detach().cpu().numpy()
        n = K_np.shape[0]
        
        # Compute eigendecomposition with appropriate precision
        if self.use_double_precision:
            eigenvals, eigenvecs = np.linalg.eigh(K_np.astype(np.float64))
        else:
            eigenvals, eigenvecs = np.linalg.eigh(K_np)
        
        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Determine effective rank based on configuration
        if self.use_matrix_rank:
            # Use torch.linalg.matrix_rank for legacy behavior
            # First convert to torch tensor if needed
            if isinstance(K, torch.Tensor):
                K_torch = K
            else:
                K_torch = torch.from_numpy(K_np)
                if hasattr(K, 'device'):
                    K_torch = K_torch.to(K.device)
            
            # Use float32 for matrix rank computation to match legacy behavior
            effective_rank = torch.linalg.matrix_rank(K_torch.float()).item()
            
            # Select top eigenvalues based on matrix rank
            pos_mask = np.arange(len(eigenvals)) < effective_rank
        else:
            # Original behavior: filter by minimum eigenvalue threshold
            pos_mask = eigenvals > self.min_eigenvalue
            effective_rank = np.sum(pos_mask)
        
        if effective_rank == 0:
            logger.warning("Matrix has no significant positive eigenvalues, using regularized identity")
            # Fallback to regularized identity with appropriate precision and device
            dtype = torch.float64 if self.use_double_precision else torch.float32
            W = torch.eye(n, dtype=dtype, device=K.device) * (1.0 / np.sqrt(self.regularization))
            # Convert back to original precision if needed
            if not self.use_double_precision:
                W = W.float()
            info = {
                'effective_rank': n,
                'condition_number': 1.0,
                'eigenvalue_range': (self.regularization, self.regularization),
                'whitening_quality': 0.0,
                'regularized': True,
                'precision_used': 'float64' if self.use_double_precision else 'float32'
            }
            return W, info
        
        # Extract positive eigenvalues and corresponding eigenvectors
        pos_eigenvals = eigenvals[pos_mask]
        pos_eigenvecs = eigenvecs[:, pos_mask]
        
        # Add regularization to prevent numerical issues
        regularized_eigenvals = pos_eigenvals + self.regularization
        
        # Compute whitening map: W = Σ^(-1/2) U^T
        inv_sqrt_eigenvals = 1.0 / np.sqrt(regularized_eigenvals)
        W_np = np.diag(inv_sqrt_eigenvals) @ pos_eigenvecs.T
        
        # Convert to torch with appropriate precision and device
        if self.use_double_precision:
            W = torch.from_numpy(W_np.astype(np.float64)).double()
        else:
            W = torch.from_numpy(W_np).float()
        
        # Move to same device as input tensor K
        W = W.to(K.device)
        
        # Compute metadata
        condition_number = np.max(regularized_eigenvals) / np.min(regularized_eigenvals)
        whitening_quality = np.sum(pos_eigenvals) / np.sum(eigenvals) if np.sum(eigenvals) > 0 else 0.0
        
        info = {
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'eigenvalue_range': (float(np.min(pos_eigenvals)), float(np.max(pos_eigenvals))),
            'whitening_quality': whitening_quality,
            'regularized': False,  # This refers to eigenvalue regularization, not input regularization
            'eigenvalues': pos_eigenvals,
            'total_eigenvalues': eigenvals,
            'precision_used': 'float64' if self.use_double_precision else 'float32',
            'input_regularized': False  # Will be set to True if input Gram matrix was regularized
        }
        
        return W, info
    
    def whiten_gram_matrix(self, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Transform Gram matrix to whitened coordinates with configurable inner product.
        
        Args:
            K: Original Gram matrix (n x n)
            
        Returns:
            Tuple of (K_whitened, W, info):
            - K_whitened: Identity matrix (default) or eigenvalue diagonal matrix (if preserve_eigenvalues=True)
            - W: Whitening map (r x n)
            - info: Whitening metadata
        """
        W, info = self.compute_whitening_map(K)
        r = W.shape[0]
        dtype = W.dtype
        
        if self.preserve_eigenvalues:
            # Return diagonal matrix of eigenvalues: K_whitened = diag(λ₁, λ₂, ..., λᵣ)
            eigenvals = info['eigenvalues'][:r]  # Only positive eigenvalues
            K_whitened = torch.diag(torch.from_numpy(eigenvals).to(dtype).to(K.device))
        else:
            # Current behavior: return identity matrix for standard whitening
            K_whitened = torch.eye(r, dtype=dtype, device=K.device)
        
        # Verify whitening quality: W K W^T should match K_whitened
        # Ensure K has same precision as W for computation
        K_compute = K.to(dtype=dtype) if K.dtype != dtype else K
        WKWt = W @ K_compute @ W.T
        whitening_error = torch.norm(WKWt - K_whitened, p='fro').item()
        
        # Store metadata
        info['whitening_error'] = whitening_error
        info['whitened_dimension'] = r
        info['preserve_eigenvalues'] = self.preserve_eigenvalues
        info['eigenvalue_diagonal'] = K_whitened
        
        return K_whitened, W, info
    
    def compute_whitened_restriction(self, R: torch.Tensor, W_source: torch.Tensor, 
                                   W_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute whitened restriction map R̃ = W_target R W_source^†.
        
        Args:
            R: Original restriction map (n_target x n_source)
            W_source: Source whitening map (r_source x n_source)
            W_target: Target whitening map (r_target x n_target)
            
        Returns:
            Tuple of (R_whitened, info):
            - R_whitened: Whitened restriction map (r_target x r_source)
            - info: Computation metadata
        """
        # Ensure all tensors have compatible precision
        target_dtype = W_target.dtype
        W_source_compute = W_source.to(dtype=target_dtype) if W_source.dtype != target_dtype else W_source
        R_compute = R.to(dtype=target_dtype) if R.dtype != target_dtype else R
        
        # Compute pseudo-inverse of source whitening map with appropriate precision
        if self.use_double_precision:
            W_source_pinv = torch.linalg.pinv(W_source_compute, atol=1e-15, rtol=1e-12)
        else:
            W_source_pinv = torch.linalg.pinv(W_source_compute)
        
        # Whitened restriction: R̃ = W_target R W_source^†
        R_whitened = W_target @ R_compute @ W_source_pinv
        
        # Check orthogonality: R̃^T R̃ should be close to identity
        RtR = R_whitened.T @ R_whitened
        r_source = W_source.shape[0]
        identity = torch.eye(r_source, dtype=target_dtype)
        
        orthogonality_error = torch.norm(RtR - identity, p='fro').item()
        
        # Adaptive orthogonality threshold based on precision
        ortho_threshold = 1e-14 if self.use_double_precision else 1e-10
        
        info = {
            'whitened_dimensions': (W_target.shape[0], W_source.shape[0]),
            'orthogonality_error': orthogonality_error,
            'exact_orthogonal': orthogonality_error < ortho_threshold,
            'precision_used': 'float64' if self.use_double_precision else 'float32',
            'orthogonality_threshold': ortho_threshold
        }
        
        return R_whitened, info
    
    def whiten_regularized_gram_matrix(self, 
                                     K: torch.Tensor, 
                                     regularization_info: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Transform regularized Gram matrix to whitened coordinates.
        
        This is a wrapper around whiten_gram_matrix that properly handles
        regularized inputs and includes regularization information in the output.
        
        Args:
            K: Regularized Gram matrix (n x n)
            regularization_info: Information about the regularization applied to K
            
        Returns:
            Tuple of (K_whitened, W, info):
            - K_whitened: Identity matrix in whitened space (r x r)
            - W: Whitening map (r x n)
            - info: Whitening metadata including regularization information
        """
        K_whitened, W, info = self.whiten_gram_matrix(K)
        
        # Add regularization information to the whitening info
        info['input_regularized'] = regularization_info.get('regularized', False)
        if info['input_regularized']:
            info['input_regularization_strength'] = regularization_info.get('regularization_strength', 0.0)
            info['input_condition_improvement'] = regularization_info.get('condition_improvement', 1.0)
            info['regularization_strategy'] = regularization_info.get('strategy', 'unknown')
        
        return K_whitened, W, info
    
    def compute_hodge_adjoint(self, R: torch.Tensor, 
                             Sigma_source: torch.Tensor, 
                             Sigma_target: torch.Tensor,
                             regularization: Optional[float] = None) -> torch.Tensor:
        """Compute Hodge adjoint R* = Σₛ⁻¹ R^T Σₜ for eigenvalue-preserving framework.
        
        This method implements the adjoint computation for non-identity inner products
        as required by the Hodge Laplacian formulation when eigenvalues are preserved.
        
        Args:
            R: Restriction map from source to target
            Sigma_source: Source eigenvalue diagonal matrix
            Sigma_target: Target eigenvalue diagonal matrix
            regularization: Override regularization parameter
            
        Returns:
            R*: Hodge adjoint of R with respect to eigenvalue-diagonal inner products
        """
        if not self.preserve_eigenvalues:
            logger.warning("compute_hodge_adjoint called without preserve_eigenvalues=True")
            # Fall back to standard transpose for identity inner products
            return R.T
        
        eps = regularization if regularization is not None else self.regularization
        
        # Compute regularized inverse of source eigenvalue matrix
        Sigma_source_inv = self._compute_regularized_inverse(Sigma_source, eps)
        
        # Hodge adjoint: R* = Σₛ⁻¹ R^T Σₜ
        return Sigma_source_inv @ R.T @ Sigma_target
    
    def _compute_regularized_inverse(self, Sigma: torch.Tensor, 
                                    regularization: Optional[float] = None) -> torch.Tensor:
        """Compute regularized inverse of diagonal eigenvalue matrix.
        
        This method provides numerically stable inversion of eigenvalue diagonal matrices
        by adding regularization to prevent division by very small eigenvalues.
        
        Args:
            Sigma: Diagonal eigenvalue matrix
            regularization: Regularization parameter (uses self.regularization if None)
            
        Returns:
            Regularized inverse of Sigma: (Σ + εI)⁻¹
        """
        eps = regularization if regularization is not None else self.regularization
        
        # For diagonal matrices, we can compute the inverse directly
        # This ensures exact symmetry for diagonal matrices
        if torch.allclose(Sigma, torch.diag(torch.diag(Sigma)), atol=1e-10):
            # Extract diagonal elements
            diag_elements = torch.diag(Sigma)
            # Add regularization and invert
            diag_inv = 1.0 / (diag_elements + eps)
            # Return as diagonal matrix
            return torch.diag(diag_inv)
        else:
            # Fallback to general matrix inversion with symmetrization
            Sigma_reg = Sigma + eps * torch.eye(
                Sigma.shape[0], dtype=Sigma.dtype, device=Sigma.device
            )
            
            # Compute inverse using solve for numerical stability
            identity = torch.eye(Sigma.shape[0], dtype=Sigma.dtype, device=Sigma.device)
            Sigma_inv = torch.linalg.solve(Sigma_reg, identity)
            
            # Ensure exact symmetry for numerical stability
            return 0.5 * (Sigma_inv + Sigma_inv.T)
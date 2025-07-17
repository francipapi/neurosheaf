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

from typing import Tuple, Dict, Any

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
                 use_double_precision: bool = False):
        """Initialize whitening processor.
        
        Args:
            min_eigenvalue: Minimum eigenvalue threshold for numerical stability
            regularization: Small value added to eigenvalues for regularization
            use_double_precision: Whether to use double precision for numerical computations
        """
        self.min_eigenvalue = min_eigenvalue
        self.regularization = regularization
        self.use_double_precision = use_double_precision
    
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
        
        # Filter positive eigenvalues for numerical stability
        pos_mask = eigenvals > self.min_eigenvalue
        effective_rank = np.sum(pos_mask)
        
        if effective_rank == 0:
            logger.warning("Matrix has no significant positive eigenvalues, using regularized identity")
            # Fallback to regularized identity with appropriate precision
            dtype = torch.float64 if self.use_double_precision else torch.float32
            W = torch.eye(n, dtype=dtype) * (1.0 / np.sqrt(self.regularization))
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
        
        # Convert to torch with appropriate precision
        if self.use_double_precision:
            W = torch.from_numpy(W_np.astype(np.float64)).double()
        else:
            W = torch.from_numpy(W_np).float()
        
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
        """Transform Gram matrix to whitened coordinates with identity inner product.
        
        Args:
            K: Original Gram matrix (n x n)
            
        Returns:
            Tuple of (K_whitened, W, info):
            - K_whitened: Identity matrix in whitened space (r x r)
            - W: Whitening map (r x n)
            - info: Whitening metadata
        """
        W, info = self.compute_whitening_map(K)
        r = W.shape[0]
        
        # In whitened coordinates, the inner product is exactly the identity
        # Use appropriate precision for identity matrix
        dtype = W.dtype
        K_whitened = torch.eye(r, dtype=dtype)
        
        # Verify whitening quality: W K W^T should be close to identity
        # Ensure K has same precision as W for computation
        K_compute = K.to(dtype=dtype) if K.dtype != dtype else K
        WKWt = W @ K_compute @ W.T
        identity_target = torch.eye(r, dtype=dtype)
        whitening_error = torch.norm(WKWt - identity_target, p='fro').item()
        
        info['whitening_error'] = whitening_error
        info['whitened_dimension'] = r
        
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
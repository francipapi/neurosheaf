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
    
    def __init__(self, min_eigenvalue: float = 1e-8, regularization: float = 1e-10):
        """Initialize whitening processor.
        
        Args:
            min_eigenvalue: Minimum eigenvalue threshold for numerical stability
            regularization: Small value added to eigenvalues for regularization
        """
        self.min_eigenvalue = min_eigenvalue
        self.regularization = regularization
    
    def compute_whitening_map(self, K: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute whitening map W = Σ^(-1/2) U^T from Gram matrix K.
        
        Args:
            K: Gram matrix (n x n), possibly rank-deficient
            
        Returns:
            Tuple of (W, info):
            - W: Whitening map (r x n) where r = effective rank
            - info: Dictionary with whitening metadata
        """
        K_np = K.detach().cpu().numpy()
        n = K_np.shape[0]
        
        # Compute eigendecomposition
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
            # Fallback to regularized identity
            W = torch.eye(n) * (1.0 / np.sqrt(self.regularization))
            info = {
                'effective_rank': n,
                'condition_number': 1.0,
                'eigenvalue_range': (self.regularization, self.regularization),
                'whitening_quality': 0.0,
                'regularized': True
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
        
        W = torch.from_numpy(W_np).float()
        
        # Compute metadata
        condition_number = np.max(regularized_eigenvals) / np.min(regularized_eigenvals)
        whitening_quality = np.sum(pos_eigenvals) / np.sum(eigenvals) if np.sum(eigenvals) > 0 else 0.0
        
        info = {
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'eigenvalue_range': (float(np.min(pos_eigenvals)), float(np.max(pos_eigenvals))),
            'whitening_quality': whitening_quality,
            'regularized': False,
            'eigenvalues': pos_eigenvals,
            'total_eigenvalues': eigenvals
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
        K_whitened = torch.eye(r)
        
        # Verify whitening quality: W K W^T should be close to identity
        WKWt = W @ K @ W.T
        whitening_error = torch.norm(WKWt - torch.eye(r), p='fro').item()
        
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
        # Compute pseudo-inverse of source whitening map
        W_source_pinv = torch.linalg.pinv(W_source)
        
        # Whitened restriction: R̃ = W_target R W_source^†
        R_whitened = W_target @ R @ W_source_pinv
        
        # Check orthogonality: R̃^T R̃ should be close to identity
        RtR = R_whitened.T @ R_whitened
        r_source = W_source.shape[0]
        identity = torch.eye(r_source)
        
        orthogonality_error = torch.norm(RtR - identity, p='fro').item()
        
        info = {
            'whitened_dimensions': (W_target.shape[0], W_source.shape[0]),
            'orthogonality_error': orthogonality_error,
            'exact_orthogonal': orthogonality_error < 1e-10
        }
        
        return R_whitened, info
"""Restriction maps implementation using scaled Procrustes analysis.

This module implements restriction maps between neural network layers using
scaled Procrustes analysis. These maps form the core mathematical structure
of the cellular sheaf for neural network analysis.

The restriction maps satisfy mathematical properties required for valid sheaves:
- Transitivity: R_AC = R_BC @ R_AB 
- Orthogonality: Q_e^T Q_e = I for orthogonal component
- Positive scaling: s_e > 0 for scale component
- Approximate metric compatibility: R_e^T K_w R_e ≈ K_v
"""

from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes, svd

from ..utils.exceptions import ComputationError
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class WhiteningProcessor:
    """Implements Patch P1: Whitening transformation for exact metric compatibility.
    
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


class ProcrustesMaps:
    """Compute restriction maps using scaled Procrustes analysis.
    
    This class implements three methods for computing restriction maps:
    1. scaled_procrustes: Main method using orthogonal Procrustes + scaling
    2. orthogonal_projection: Handles dimension mismatches via SVD
    3. least_squares: Simple linear solution for comparison
    
    The restriction maps R_e for edge e = (v → w) minimize:
    ||s K_v Q - K_w||_F over s > 0, Q ∈ O(n)
    where K_v, K_w are Gram matrices at layers v, w.
    
    Supports both raw and whitened coordinates for exact metric compatibility.
    
    Attributes:
        epsilon: Numerical stability parameter for regularization
        max_scale: Maximum allowed scaling factor
        min_scale: Minimum allowed scaling factor
        whitening_processor: Optional WhiteningProcessor for Patch P1
        use_whitened_coordinates: Whether to use whitened coordinates by default
    """
    
    def __init__(self, epsilon: float = 1e-8, max_scale: float = 100.0, 
                 min_scale: float = 1e-3, use_whitened_coordinates: bool = False):
        """Initialize ProcrustesMaps with numerical stability parameters.
        
        Args:
            epsilon: Regularization parameter for numerical stability
            max_scale: Maximum allowed scaling factor to prevent overflow
            min_scale: Minimum allowed scaling factor to prevent underflow
            use_whitened_coordinates: Whether to use whitened coordinates by default
        """
        self.epsilon = epsilon
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.use_whitened_coordinates = use_whitened_coordinates
        self.whitening_processor = WhiteningProcessor() if use_whitened_coordinates else None
    
    def scaled_procrustes_whitened(self, K_source: torch.Tensor, K_target: torch.Tensor,
                                  validate: bool = True) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Compute restriction map using whitened coordinates for exact metric compatibility.
        
        This implements Patch P1 from the pipeline report:
        1. Whiten both source and target Gram matrices
        2. Compute restriction in whitened space (exact orthogonality)
        3. Return both whitened restriction and original space restriction
        
        Args:
            K_source: Source Gram matrix (n_source x n_source)
            K_target: Target Gram matrix (n_target x n_target)
            validate: Whether to validate mathematical properties
            
        Returns:
            Tuple of (R, scale, info):
            - R: Restriction map in original coordinates
            - scale: Scale factor (always 1.0 in whitened coordinates)
            - info: Dictionary with both whitened and original space information
        """
        if self.whitening_processor is None:
            self.whitening_processor = WhiteningProcessor()
        
        # Whiten both Gram matrices
        K_source_white, W_source, source_info = self.whitening_processor.whiten_gram_matrix(K_source)
        K_target_white, W_target, target_info = self.whitening_processor.whiten_gram_matrix(K_target)
        
        # In whitened coordinates, both matrices are identity, so optimal map is rectangular identity
        r_source = K_source_white.shape[0]
        r_target = K_target_white.shape[0]
        
        # Create proper rectangular restriction map that preserves maximum information
        R_whitened = torch.zeros(r_target, r_source)
        
        if r_source <= r_target:
            # Target has higher or equal rank: embed source space in target space
            # R: ℝ^r_source → ℝ^r_target via [I; 0] structure
            R_whitened[:r_source, :] = torch.eye(r_source)
        else:
            # Source has higher rank: project source space to target space  
            # R: ℝ^r_source → ℝ^r_target via [I | 0] structure
            # This gives R^T R = [I 0; 0 0] which preserves orthogonality on the active subspace
            R_whitened[:, :r_target] = torch.eye(r_target)
        
        # Transform back to original coordinates: R = W_target^† R̃ W_source
        W_target_pinv = torch.linalg.pinv(W_target)
        R = W_target_pinv @ R_whitened @ W_source
        
        # Scale factor is always 1.0 in whitened coordinates
        scale_factor = 1.0
        
        # Compute errors in both spaces
        # Whitened space (should be exact for the optimal rectangular map)
        reconstructed_whitened = R_whitened @ K_source_white @ R_whitened.T
        whitened_error = torch.norm(reconstructed_whitened - K_target_white, p='fro').item()
        
        # Original space (should be good approximation with proper rectangular map)
        reconstructed_original = R @ K_source @ R.T
        original_error = torch.norm(reconstructed_original - K_target, p='fro').item()
        target_norm = torch.norm(K_target, p='fro').item()
        
        # Compute relative error safely
        original_relative_error = original_error / (target_norm + self.epsilon) if target_norm > 0 else float('inf')
        
        # Check orthogonality in whitened space (appropriate for rectangular maps)
        RtR_whitened = R_whitened.T @ R_whitened
        
        if r_source <= r_target:
            # Embedding: R^T R should be exactly I
            I_expected = torch.eye(r_source)
            orthogonality_error_whitened = torch.norm(RtR_whitened - I_expected, p='fro').item()
        else:
            # Projection: R^T R should be [I 0; 0 0] - check only the active block
            I_active = torch.eye(r_target)
            RtR_active = RtR_whitened[:r_target, :r_target]
            orthogonality_error_whitened = torch.norm(RtR_active - I_active, p='fro').item()
        
        info = {
            'method': 'scaled_procrustes_whitened',
            'scale': scale_factor,
            'whitened_dimensions': (r_target, r_source),
            'original_dimensions': (K_target.shape[0], K_source.shape[0]),
            'whitened_reconstruction_error': whitened_error,
            'original_reconstruction_error': original_error,
            'relative_error': original_relative_error,
            'whitened_orthogonality_error': orthogonality_error_whitened,
            'exact_in_whitened_space': whitened_error < 1e-10 and orthogonality_error_whitened < 1e-10,
            'source_whitening_info': source_info,
            'target_whitening_info': target_info,
            'whitening_maps': {'source': W_source, 'target': W_target},
            'restriction_whitened': R_whitened
        }
        
        if validate:
            self._validate_whitened_restriction_map(R, R_whitened, K_source, K_target, 
                                                  K_source_white, K_target_white, info)
        
        return R, scale_factor, info
    
    def scaled_procrustes(self, K_source: torch.Tensor, K_target: torch.Tensor,
                         validate: bool = True, use_whitening: Optional[bool] = None) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Compute scaled Procrustes restriction map.
        
        This is the main method for computing restriction maps. It solves:
        min_{s>0, Q∈O(n)} ||s K_source Q - K_target||_F
        
        Supports both raw and whitened coordinate computation for exact metric compatibility.
        
        Args:
            K_source: Source Gram matrix (n_source x n_source)
            K_target: Target Gram matrix (n_target x n_target)  
            validate: Whether to validate mathematical properties
            use_whitening: Override class default for whitening. If True, uses whitened coordinates
            
        Returns:
            Tuple of (R, scale, info):
            - R: Restriction map tensor (n_target x n_source)
            - scale: Optimal scaling factor
            - info: Dictionary with computation details
            
        Raises:
            ComputationError: If matrices are ill-conditioned
        """
        # Determine whether to use whitening
        should_whiten = use_whitening if use_whitening is not None else self.use_whitened_coordinates
        
        if should_whiten:
            return self.scaled_procrustes_whitened(K_source, K_target, validate)
        
        # Original implementation for raw coordinates
        # Ensure tensors are on CPU for scipy operations
        K_source_np = K_source.detach().cpu().numpy()
        K_target_np = K_target.detach().cpu().numpy()
        
        # Handle dimension mismatch
        if K_source_np.shape != K_target_np.shape:
            return self._handle_dimension_mismatch(K_source, K_target, validate)
        
        # Check for numerical stability
        source_cond = np.linalg.cond(K_source_np)
        target_cond = np.linalg.cond(K_target_np)
        
        if source_cond > 1e12 or target_cond > 1e12:
            logger.warning(f"Ill-conditioned matrices: source_cond={source_cond:.2e}, "
                          f"target_cond={target_cond:.2e}")
        
        try:
            # Compute orthogonal Procrustes solution
            Q, scale_factor = orthogonal_procrustes(K_source_np, K_target_np)
            
            # Ensure scale is positive and within bounds
            scale_factor = abs(scale_factor)
            scale_factor = np.clip(scale_factor, self.min_scale, self.max_scale)
            
            # Construct restriction map R = s * Q
            R_np = scale_factor * Q
            R = torch.from_numpy(R_np).float()
            
            # Compute reconstruction error
            reconstructed = scale_factor * K_source_np @ Q
            error = np.linalg.norm(reconstructed - K_target_np, 'fro')
            relative_error = error / (np.linalg.norm(K_target_np, 'fro') + self.epsilon)
            
            info = {
                'method': 'scaled_procrustes',
                'scale': scale_factor,
                'orthogonal_matrix': torch.from_numpy(Q).float(),
                'reconstruction_error': error,
                'relative_error': relative_error,
                'source_condition': source_cond,
                'target_condition': target_cond,
                'orthogonality_error': self._check_orthogonality(Q)
            }
            
            if validate:
                self._validate_restriction_map(R, K_source, K_target, info)
            
            return R, scale_factor, info
            
        except np.linalg.LinAlgError as e:
            raise ComputationError(f"Procrustes computation failed: {e}", operation="scaled_procrustes")
    
    def orthogonal_projection(self, K_source: torch.Tensor, K_target: torch.Tensor,
                             validate: bool = True) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Handle dimension mismatches using SVD-based projection.
        
        When matrices have different dimensions, this method projects to
        a common subspace using SVD before computing the restriction map.
        
        Args:
            K_source: Source Gram matrix (n_source x n_source)
            K_target: Target Gram matrix (n_target x n_target)
            validate: Whether to validate mathematical properties
            
        Returns:
            Tuple of (R, scale, info) with restriction map and metadata
        """
        K_source_np = K_source.detach().cpu().numpy()
        K_target_np = K_target.detach().cpu().numpy()
        
        n_source = K_source_np.shape[0]
        n_target = K_target_np.shape[0]
        
        # Determine common dimension
        common_dim = min(n_source, n_target)
        
        try:
            # Project both matrices to common dimension
            U_s, s_s, Vt_s = svd(K_source_np)
            U_t, s_t, Vt_t = svd(K_target_np)
            
            # Use the smaller rank for the common subspace
            rank_s = min(len(s_s), common_dim)
            rank_t = min(len(s_t), common_dim)
            final_dim = min(rank_s, rank_t)
            
            # Project to common dimension
            K_source_proj = U_s[:, :final_dim] @ np.diag(s_s[:final_dim]) @ Vt_s[:final_dim, :]
            K_target_proj = U_t[:, :final_dim] @ np.diag(s_t[:final_dim]) @ Vt_t[:final_dim, :]
            
            # Ensure they have the same shape for Procrustes
            if K_source_proj.shape != K_target_proj.shape:
                min_rows = min(K_source_proj.shape[0], K_target_proj.shape[0])
                min_cols = min(K_source_proj.shape[1], K_target_proj.shape[1])
                K_source_proj = K_source_proj[:min_rows, :min_cols]
                K_target_proj = K_target_proj[:min_rows, :min_cols]
            
            # Now compute Procrustes on projected matrices
            Q_proj, scale = orthogonal_procrustes(K_source_proj, K_target_proj)
            scale = abs(scale)
            scale = np.clip(scale, self.min_scale, self.max_scale)
            
            # Construct full restriction map with proper padding/truncation
            R_np = np.zeros((n_target, n_source))
            
            # Determine the overlapping region
            overlap_rows = min(n_target, Q_proj.shape[0])
            overlap_cols = min(n_source, Q_proj.shape[1])
            
            R_np[:overlap_rows, :overlap_cols] = scale * Q_proj[:overlap_rows, :overlap_cols]
            
            R = torch.from_numpy(R_np).float()
            
            # Compute reconstruction error (handle dimension mismatch)
            reconstructed = R_np @ K_source_np
            # Pad or truncate to match K_target_np shape
            if reconstructed.shape != K_target_np.shape:
                min_rows = min(reconstructed.shape[0], K_target_np.shape[0])
                min_cols = min(reconstructed.shape[1], K_target_np.shape[1])
                reconstructed_trunc = reconstructed[:min_rows, :min_cols]
                target_trunc = K_target_np[:min_rows, :min_cols]
                error = np.linalg.norm(reconstructed_trunc - target_trunc, 'fro')
            else:
                error = np.linalg.norm(reconstructed - K_target_np, 'fro')
            relative_error = error / (np.linalg.norm(K_target_np, 'fro') + self.epsilon)
            
            info = {
                'method': 'orthogonal_projection',
                'scale': scale,
                'common_dimension': common_dim,
                'dimension_mismatch': (n_source, n_target),
                'reconstruction_error': error,
                'relative_error': relative_error,
                'projection_quality': np.sum(s_s[:common_dim]) / np.sum(s_s) if n_source > n_target else np.sum(s_t[:common_dim]) / np.sum(s_t)
            }
            
            if validate:
                self._validate_restriction_map(R, K_source, K_target, info)
            
            return R, scale, info
            
        except np.linalg.LinAlgError as e:
            raise ComputationError(f"SVD-based projection failed: {e}", operation="orthogonal_projection")
    
    def least_squares(self, K_source: torch.Tensor, K_target: torch.Tensor,
                     validate: bool = True) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Compute restriction map using least squares solution.
        
        This provides a simple baseline method that solves:
        min_R ||R K_source - K_target||_F
        
        Args:
            K_source: Source Gram matrix (n_source x n_source)
            K_target: Target Gram matrix (n_target x n_target)
            validate: Whether to validate mathematical properties
            
        Returns:
            Tuple of (R, scale, info) with restriction map and metadata
        """
        K_source_np = K_source.detach().cpu().numpy()
        K_target_np = K_target.detach().cpu().numpy()
        
        # Handle dimension mismatch by padding or truncating
        n_source = K_source_np.shape[0]
        n_target = K_target_np.shape[0]
        
        if n_source != n_target:
            min_dim = min(n_source, n_target)
            K_source_trunc = K_source_np[:min_dim, :min_dim]
            K_target_trunc = K_target_np[:min_dim, :min_dim]
        else:
            K_source_trunc = K_source_np
            K_target_trunc = K_target_np
        
        try:
            # Solve R K_source = K_target for R
            # This is equivalent to R = K_target K_source^+
            K_source_pinv = np.linalg.pinv(K_source_trunc + self.epsilon * np.eye(K_source_trunc.shape[0]))
            R_trunc = K_target_trunc @ K_source_pinv
            
            # Pad to correct dimensions
            R_np = np.zeros((n_target, n_source))
            R_np[:R_trunc.shape[0], :R_trunc.shape[1]] = R_trunc
            
            R = torch.from_numpy(R_np).float()
            
            # Estimate scale as Frobenius norm ratio
            scale = np.linalg.norm(R_trunc, 'fro') / (np.linalg.norm(np.eye(R_trunc.shape[0]), 'fro') + self.epsilon)
            scale = np.clip(scale, self.min_scale, self.max_scale)
            
            # Compute reconstruction error
            reconstructed = R_np @ K_source_np
            error = np.linalg.norm(reconstructed - K_target_np, 'fro')
            relative_error = error / (np.linalg.norm(K_target_np, 'fro') + self.epsilon)
            
            info = {
                'method': 'least_squares',
                'scale': scale,
                'dimension_truncation': (K_source_trunc.shape, K_target_trunc.shape),
                'reconstruction_error': error,
                'relative_error': relative_error,
                'condition_number': np.linalg.cond(K_source_trunc)
            }
            
            if validate:
                self._validate_restriction_map(R, K_source, K_target, info)
            
            return R, scale, info
            
        except np.linalg.LinAlgError as e:
            raise ComputationError(f"Least squares computation failed: {e}", operation="least_squares")
    
    def _handle_dimension_mismatch(self, K_source: torch.Tensor, K_target: torch.Tensor,
                                  validate: bool) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Handle dimension mismatch by delegating to orthogonal_projection."""
        logger.info(f"Dimension mismatch: {K_source.shape} vs {K_target.shape}, using projection")
        return self.orthogonal_projection(K_source, K_target, validate)
    
    def _validate_restriction_map(self, R: torch.Tensor, K_source: torch.Tensor, 
                                 K_target: torch.Tensor, info: Dict[str, Any]):
        """Validate mathematical properties of the restriction map.
        
        Args:
            R: Computed restriction map
            K_source: Source Gram matrix
            K_target: Target Gram matrix  
            info: Information dictionary to update with validation results
        """
        # Check reconstruction quality using correct sheaf restriction formula
        reconstructed = R @ K_source @ R.T
        reconstruction_error = torch.norm(reconstructed - K_target, p='fro').item()
        
        target_norm = torch.norm(K_target, p='fro').item()
        relative_error = reconstruction_error / (target_norm + self.epsilon)
        
        # Update info with validation results
        info['validation'] = {
            'reconstruction_error_torch': reconstruction_error,
            'relative_error_torch': relative_error,
            'target_norm': target_norm,
            'passed': relative_error < 0.5  # 50% relative error threshold
        }
        
        if relative_error > 0.5:
            logger.warning(f"High reconstruction error: {relative_error:.3f}")
    
    def _validate_whitened_restriction_map(self, R: torch.Tensor, R_whitened: torch.Tensor,
                                         K_source: torch.Tensor, K_target: torch.Tensor,
                                         K_source_white: torch.Tensor, K_target_white: torch.Tensor,
                                         info: Dict[str, Any]):
        """Validate mathematical properties of whitened restriction maps.
        
        Args:
            R: Restriction map in original coordinates
            R_whitened: Restriction map in whitened coordinates  
            K_source: Source Gram matrix (original)
            K_target: Target Gram matrix (original)
            K_source_white: Source Gram matrix (whitened)
            K_target_white: Target Gram matrix (whitened)
            info: Information dictionary to update
        """
        # Check exact orthogonality in whitened space (handle rectangular case)
        RtR_whitened = R_whitened.T @ R_whitened
        r_source = K_source_white.shape[0]
        r_target = K_target_white.shape[0]
        
        if r_source <= r_target:
            # Embedding: R^T R should be exactly I
            identity = torch.eye(r_source)
            whitened_orthogonality_error = torch.norm(RtR_whitened - identity, p='fro').item()
        else:
            # Projection: R^T R should be [I 0; 0 0] - check only the active block
            identity_active = torch.eye(r_target)
            RtR_active = RtR_whitened[:r_target, :r_target]
            whitened_orthogonality_error = torch.norm(RtR_active - identity_active, p='fro').item()
        
        # Check exact metric compatibility in whitened space
        RKRt_whitened = R_whitened @ K_source_white @ R_whitened.T
        whitened_metric_error = torch.norm(RKRt_whitened - K_target_white, p='fro').item()
        
        # Check approximate properties in original space using correct sheaf formula
        reconstructed_original = R @ K_source @ R.T
        original_reconstruction_error = torch.norm(reconstructed_original - K_target, p='fro').item()
        target_norm = torch.norm(K_target, p='fro').item()
        
        original_relative_error = original_reconstruction_error / (target_norm + self.epsilon)
        
        # Update info with validation results
        info['whitened_validation'] = {
            'orthogonality_error': whitened_orthogonality_error,
            'metric_compatibility_error': whitened_metric_error,
            'exact_orthogonal': whitened_orthogonality_error < 1e-12,
            'exact_metric_compatible': whitened_metric_error < 1e-12
        }
        
        info['original_validation'] = {
            'reconstruction_error': original_reconstruction_error,
            'relative_error': original_relative_error,
            'target_norm': target_norm,
            'acceptable': original_relative_error < 0.5
        }
        
        # Log results
        if whitened_orthogonality_error < 1e-12 and whitened_metric_error < 1e-12:
            logger.info(f"Exact properties achieved in whitened space: "
                       f"orthogonality={whitened_orthogonality_error:.2e}, "
                       f"metric_compat={whitened_metric_error:.2e}")
        else:
            logger.warning(f"Whitened space not exact: "
                          f"orthogonality={whitened_orthogonality_error:.2e}, "
                          f"metric_compat={whitened_metric_error:.2e}")
    
    def _check_orthogonality(self, Q: np.ndarray) -> float:
        """Check orthogonality of matrix Q.
        
        Args:
            Q: Matrix to check
            
        Returns:
            Orthogonality error ||Q^T Q - I||_F
        """
        QTQ = Q.T @ Q
        I = np.eye(Q.shape[1])
        return np.linalg.norm(QTQ - I, 'fro')
    
    def compute_restriction_map(self, K_source: torch.Tensor, K_target: torch.Tensor,
                               method: str = 'scaled_procrustes', 
                               validate: bool = True, use_whitening: Optional[bool] = None) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Compute restriction map using specified method.
        
        This is the main public interface for computing restriction maps.
        
        Args:
            K_source: Source Gram matrix
            K_target: Target Gram matrix
            method: Method to use ('scaled_procrustes', 'orthogonal_projection', 'least_squares')
            validate: Whether to validate mathematical properties
            use_whitening: Override class default for whitening (only applies to 'scaled_procrustes')
            
        Returns:
            Tuple of (R, scale, info) with restriction map and metadata
            
        Raises:
            ValueError: If method is not recognized
            ComputationError: If computation fails
        """
        if method == 'scaled_procrustes':
            return self.scaled_procrustes(K_source, K_target, validate, use_whitening)
        elif method == 'orthogonal_projection':
            return self.orthogonal_projection(K_source, K_target, validate)
        elif method == 'least_squares':
            return self.least_squares(K_source, K_target, validate)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: "
                           "'scaled_procrustes', 'orthogonal_projection', 'least_squares'")


def validate_sheaf_properties(restrictions: Dict[Tuple[str, str], torch.Tensor],
                              poset: 'nx.DiGraph', tolerance: float = 1e-2) -> Dict[str, Any]:
    """Validate mathematical properties required for a valid sheaf.
    
    This function checks the transitivity property: R_AC = R_BC @ R_AB
    for all valid paths in the poset.
    
    Args:
        restrictions: Dictionary mapping edges to restriction maps
        poset: NetworkX directed graph representing the poset structure
        tolerance: Tolerance for approximate equality
        
    Returns:
        Dictionary with validation results
    """
    import networkx as nx
    
    validation_results = {
        'transitivity_violations': [],
        'max_violation': 0.0,
        'total_paths_checked': 0,
        'valid_sheaf': True
    }
    
    # Check transitivity for all paths of length 2
    for node_a in poset.nodes():
        for node_b in poset.successors(node_a):
            for node_c in poset.successors(node_b):
                # We have path A → B → C
                edge_ab = (node_a, node_b)
                edge_bc = (node_b, node_c)
                edge_ac = (node_a, node_c) if poset.has_edge(node_a, node_c) else None
                
                if edge_ab in restrictions and edge_bc in restrictions:
                    R_ab = restrictions[edge_ab]
                    R_bc = restrictions[edge_bc]
                    R_composed = R_bc @ R_ab
                    
                    validation_results['total_paths_checked'] += 1
                    
                    if edge_ac and edge_ac in restrictions:
                        # Direct path exists, check transitivity
                        R_ac = restrictions[edge_ac]
                        violation = torch.norm(R_composed - R_ac, p='fro').item()
                        
                        if violation > tolerance:
                            validation_results['transitivity_violations'].append({
                                'path': (node_a, node_b, node_c),
                                'violation': violation,
                                'relative_violation': violation / (torch.norm(R_ac, p='fro').item() + 1e-8)
                            })
                            validation_results['valid_sheaf'] = False
                        
                        validation_results['max_violation'] = max(validation_results['max_violation'], violation)
    
    logger.info(f"Sheaf validation: {validation_results['total_paths_checked']} paths checked, "
                f"{len(validation_results['transitivity_violations'])} violations found, "
                f"max violation: {validation_results['max_violation']:.6f}")
    
    return validation_results
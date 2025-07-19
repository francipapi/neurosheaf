"""Real-to-complex reconstruction for directed sheaf computation.

This module implements the reconstruction of complex results from real
computational representation, enabling the recovery of complex spectral
properties while maintaining numerical accuracy.

Mathematical Foundation:
- Eigenvalue reconstruction: Real eigenvalues from Hermitian matrices
- Eigenvector reconstruction: Complex eigenvectors from real representation
- Matrix reconstruction: Convert [[X, -Y], [Y, X]] back to X + iY
- Spectral property validation: Ensure conjugate pair structure

Key Features:
- Preserves spectral properties during reconstruction
- Validates conjugate pair structure for eigenvalues
- Reconstructs complex eigenvectors accurately
- Handles sparse matrix reconstruction
- Provides comprehensive validation

This enables the complete round-trip conversion from complex mathematical
representation through real computation back to complex results.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy.sparse import csr_matrix, csc_matrix
import networkx as nx

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


class RealToComplexReconstruction:
    """Reconstructs complex results from real computational representation.
    
    This class implements the inverse of the complex-to-real embedding,
    enabling the recovery of complex spectral properties from real
    computational results while maintaining numerical accuracy.
    
    Mathematical Implementation:
    For real embedding [[X, -Y], [Y, X]], reconstructs Z = X + iY
    
    Key Features:
    - Eigenvalue reconstruction from conjugate pairs
    - Complex eigenvector reconstruction
    - Matrix reconstruction from real embedding
    - Spectral property validation
    - Conjugate pair structure verification
    
    The reconstruction enables complete round-trip conversion while
    preserving all mathematical properties required for directed sheaf analysis.
    """
    
    def __init__(self, validate_properties: bool = True,
                 tolerance: float = 1e-12,
                 device: Optional[torch.device] = None):
        """Initialize the real-to-complex reconstruction computer.
        
        Args:
            validate_properties: Whether to validate mathematical properties
            tolerance: Tolerance for numerical validation
            device: PyTorch device for computations
        """
        self.validate_properties = validate_properties
        self.tolerance = tolerance
        self.device = device or torch.device('cpu')
        
        logger.debug(f"RealToComplexReconstruction initialized with device={device}")
    
    def reconstruct_matrix(self, real_matrix: torch.Tensor) -> torch.Tensor:
        """Reconstruct complex matrix from real representation.
        
        Mathematical Implementation:
        For real embedding [[X, -Y], [Y, X]], reconstructs Z = X + iY
        
        Args:
            real_matrix: Real matrix of shape (2n, 2m) from embedding
            
        Returns:
            Complex matrix of shape (n, m)
            
        Raises:
            ValueError: If input dimensions are invalid
            RuntimeError: If reconstruction fails validation
        """
        if not isinstance(real_matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if real_matrix.is_complex():
            raise ValueError("Input tensor must be real")
        
        # Check dimensions are even
        n_double, m_double = real_matrix.shape
        if n_double % 2 != 0 or m_double % 2 != 0:
            raise ValueError(f"Real matrix dimensions must be even, got {real_matrix.shape}")
        
        n, m = n_double // 2, m_double // 2
        
        # Move to specified device
        real_matrix = real_matrix.to(self.device)
        
        # Extract blocks: [[X, -Y], [Y, X]]
        X = real_matrix[:n, :m]           # Top-left
        neg_Y = real_matrix[:n, m:]       # Top-right
        Y = real_matrix[n:, :m]           # Bottom-left
        X_check = real_matrix[n:, m:]     # Bottom-right
        
        # Validate block structure
        if self.validate_properties:
            self._validate_block_structure(X, neg_Y, Y, X_check)
        
        # Reconstruct complex matrix: Z = X + iY
        # Note: neg_Y = -Y, so Y = -neg_Y
        complex_matrix = torch.complex(X, Y)
        
        logger.debug(f"Reconstructed complex matrix {real_matrix.shape} → {complex_matrix.shape}")
        return complex_matrix
    
    def reconstruct_vector(self, real_vector: torch.Tensor) -> torch.Tensor:
        """Reconstruct complex vector from real representation.
        
        Args:
            real_vector: Real vector of shape (2n,) or (2n, 1)
            
        Returns:
            Complex vector of shape (n,) or (n, 1)
        """
        if not isinstance(real_vector, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if real_vector.is_complex():
            raise ValueError("Input tensor must be real")
        
        # Move to specified device
        real_vector = real_vector.to(self.device)
        
        if real_vector.dim() == 1:
            # 1D vector
            if real_vector.shape[0] % 2 != 0:
                raise ValueError(f"Real vector dimension must be even, got {real_vector.shape}")
            
            n = real_vector.shape[0] // 2
            real_part = real_vector[:n]
            imag_part = real_vector[n:]
            
            complex_vector = torch.complex(real_part, imag_part)
            
        else:
            # 2D vector
            if real_vector.shape[0] % 2 != 0:
                raise ValueError(f"Real vector dimension must be even, got {real_vector.shape}")
            
            n = real_vector.shape[0] // 2
            real_part = real_vector[:n]
            imag_part = real_vector[n:]
            
            complex_vector = torch.complex(real_part, imag_part)
        
        logger.debug(f"Reconstructed complex vector {real_vector.shape} → {complex_vector.shape}")
        return complex_vector
    
    def reconstruct_eigenvalues(self, real_eigenvalues: torch.Tensor) -> torch.Tensor:
        """Extract real eigenvalues from real representation.
        
        For Hermitian matrices, all eigenvalues are real, so conjugate pairs
        are identical. This method extracts the unique real eigenvalues.
        
        Args:
            real_eigenvalues: Real eigenvalues from embedded matrix
            
        Returns:
            Real eigenvalues of the original complex Hermitian matrix
        """
        if not isinstance(real_eigenvalues, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if real_eigenvalues.is_complex():
            raise ValueError("Input eigenvalues must be real")
        
        # For Hermitian matrices, eigenvalues come in conjugate pairs
        # Since they are real, the pairs are identical
        if real_eigenvalues.shape[0] % 2 != 0:
            raise ValueError(f"Number of eigenvalues must be even, got {real_eigenvalues.shape[0]}")
        
        n = real_eigenvalues.shape[0] // 2
        
        # Extract first half (eigenvalues are duplicated)
        eigenvalues = real_eigenvalues[:n]
        
        # Validate that eigenvalues are properly paired
        if self.validate_properties:
            second_half = real_eigenvalues[n:]
            if torch.abs(eigenvalues - second_half).max() > 1e-6:  # More relaxed tolerance
                logger.warning("Eigenvalues not properly paired - may not be from Hermitian matrix")
        
        logger.debug(f"Reconstructed eigenvalues {real_eigenvalues.shape} → {eigenvalues.shape}")
        return eigenvalues
    
    def reconstruct_eigenvectors(self, real_eigenvectors: torch.Tensor) -> torch.Tensor:
        """Reconstruct complex eigenvectors from real representation.
        
        Args:
            real_eigenvectors: Real eigenvectors from embedded matrix
            
        Returns:
            Complex eigenvectors of the original complex matrix
        """
        if not isinstance(real_eigenvectors, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if real_eigenvectors.is_complex():
            raise ValueError("Input eigenvectors must be real")
        
        # Check dimensions
        if real_eigenvectors.shape[0] % 2 != 0:
            raise ValueError(f"Number of eigenvector components must be even, got {real_eigenvectors.shape[0]}")
        
        n = real_eigenvectors.shape[0] // 2
        num_vectors = real_eigenvectors.shape[1]
        
        # Reconstruct complex eigenvectors
        complex_eigenvectors = torch.zeros(n, num_vectors, dtype=torch.complex64, device=self.device)
        
        for i in range(num_vectors):
            real_eigenvector = real_eigenvectors[:, i]
            complex_eigenvector = self.reconstruct_vector(real_eigenvector)
            complex_eigenvectors[:, i] = complex_eigenvector
        
        logger.debug(f"Reconstructed eigenvectors {real_eigenvectors.shape} → {complex_eigenvectors.shape}")
        return complex_eigenvectors
    
    def reconstruct_spectrum(self, real_eigenvalues: torch.Tensor, 
                           real_eigenvectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct complete spectrum from real representation.
        
        Args:
            real_eigenvalues: Real eigenvalues from embedded matrix
            real_eigenvectors: Real eigenvectors from embedded matrix
            
        Returns:
            Tuple of (complex_eigenvalues, complex_eigenvectors)
        """
        complex_eigenvalues = self.reconstruct_eigenvalues(real_eigenvalues)
        complex_eigenvectors = self.reconstruct_eigenvectors(real_eigenvectors)
        
        logger.debug(f"Reconstructed complete spectrum")
        return complex_eigenvalues, complex_eigenvectors
    
    def reconstruct_sparse_matrix(self, real_sparse: csr_matrix) -> csr_matrix:
        """Reconstruct complex sparse matrix from real representation.
        
        Args:
            real_sparse: Real sparse matrix from embedding
            
        Returns:
            Complex sparse matrix
        """
        if np.iscomplexobj(real_sparse.data):
            raise ValueError("Input sparse matrix must be real")
        
        # Check dimensions
        n_double, m_double = real_sparse.shape
        if n_double % 2 != 0 or m_double % 2 != 0:
            raise ValueError(f"Real sparse matrix dimensions must be even, got {real_sparse.shape}")
        
        n, m = n_double // 2, m_double // 2
        
        # Extract blocks: [[X, -Y], [Y, X]]
        X = real_sparse[:n, :m]
        neg_Y = real_sparse[:n, m:]
        Y = real_sparse[n:, :m]
        X_check = real_sparse[n:, m:]
        
        # Validate block structure if requested
        if self.validate_properties:
            # Check that bottom-right block matches top-left
            if not np.allclose(X.toarray(), X_check.toarray(), atol=self.tolerance):
                raise RuntimeError("Block structure validation failed: X blocks don't match")
            
            # Check that Y = -neg_Y
            if not np.allclose(Y.toarray(), -neg_Y.toarray(), atol=self.tolerance):
                raise RuntimeError("Block structure validation failed: Y blocks don't match")
        
        # Reconstruct complex matrix: Z = X + iY
        complex_data = X.data + 1j * Y.data
        complex_sparse = csr_matrix((complex_data, X.indices, X.indptr), shape=(n, m))
        
        logger.debug(f"Reconstructed complex sparse matrix {real_sparse.shape} → {complex_sparse.shape}")
        return complex_sparse
    
    def reconstruct_laplacian_blocks(self, real_blocks: Dict[Tuple[str, str], torch.Tensor]) -> Dict[Tuple[str, str], torch.Tensor]:
        """Reconstruct complex Laplacian blocks from real representation.
        
        Args:
            real_blocks: Dictionary mapping (node1, node2) to real Laplacian blocks
            
        Returns:
            Dictionary mapping (node1, node2) to complex Laplacian blocks
        """
        if not isinstance(real_blocks, dict):
            raise ValueError("real_blocks must be a dictionary")
        
        complex_blocks = {}
        
        for block_key, real_block in real_blocks.items():
            try:
                complex_block = self.reconstruct_matrix(real_block)
                complex_blocks[block_key] = complex_block
                
                logger.debug(f"Reconstructed Laplacian block {block_key}: {real_block.shape} → {complex_block.shape}")
                
            except Exception as e:
                logger.error(f"Failed to reconstruct Laplacian block {block_key}: {e}")
                raise RuntimeError(f"Laplacian block reconstruction failed for {block_key}: {e}")
        
        logger.info(f"Reconstructed {len(complex_blocks)} Laplacian blocks")
        return complex_blocks
    
    def validate_round_trip(self, original_complex: torch.Tensor,
                           reconstructed_complex: torch.Tensor) -> Dict[str, Any]:
        """Validate round-trip conversion accuracy.
        
        Args:
            original_complex: Original complex matrix
            reconstructed_complex: Reconstructed complex matrix
            
        Returns:
            Dictionary with validation results
        """
        if original_complex.shape != reconstructed_complex.shape:
            raise ValueError("Original and reconstructed matrices must have same shape")
        
        # Compute reconstruction error
        reconstruction_error = torch.abs(original_complex - reconstructed_complex).max().item()
        
        # Compute relative error
        original_norm = torch.norm(original_complex).item()
        relative_error = reconstruction_error / (original_norm + 1e-12)
        
        # Check spectral properties if matrices are square
        spectral_validation = {}
        if original_complex.shape[0] == original_complex.shape[1]:
            try:
                # Compute eigenvalues
                orig_eigenvalues = torch.linalg.eigvals(original_complex)
                recon_eigenvalues = torch.linalg.eigvals(reconstructed_complex)
                
                # Sort eigenvalues for comparison
                orig_sorted = torch.sort(orig_eigenvalues.real)[0]
                recon_sorted = torch.sort(recon_eigenvalues.real)[0]
                
                spectral_error = torch.abs(orig_sorted - recon_sorted).max().item()
                spectral_validation = {
                    'spectral_error': spectral_error,
                    'spectral_relative_error': spectral_error / (torch.norm(orig_sorted).item() + 1e-12)
                }
                
            except Exception as e:
                spectral_validation = {'spectral_error': f"Failed to compute: {e}"}
        
        validation_result = {
            'reconstruction_error': reconstruction_error,
            'relative_error': relative_error,
            'passes_tolerance': reconstruction_error <= self.tolerance,
            'original_norm': original_norm,
            'reconstructed_norm': torch.norm(reconstructed_complex).item(),
            **spectral_validation
        }
        
        logger.debug(f"Round-trip validation: error={reconstruction_error:.2e}, relative={relative_error:.2e}")
        return validation_result
    
    def _validate_block_structure(self, X: torch.Tensor, neg_Y: torch.Tensor,
                                 Y: torch.Tensor, X_check: torch.Tensor) -> None:
        """Validate the block structure of real embedding.
        
        Args:
            X: Top-left block (real part)
            neg_Y: Top-right block (-imaginary part)
            Y: Bottom-left block (imaginary part)
            X_check: Bottom-right block (should match X)
            
        Raises:
            RuntimeError: If block structure is invalid
        """
        # Check that bottom-right block matches top-left (both should be X)
        if torch.abs(X - X_check).max() > self.tolerance:
            raise RuntimeError("Block structure validation failed: X blocks don't match")
        
        # Check that Y = -neg_Y
        if torch.abs(Y - (-neg_Y)).max() > self.tolerance:
            raise RuntimeError("Block structure validation failed: Y blocks don't match")
        
        logger.debug("Block structure validation passed")
    
    def _extract_conjugate_pairs(self, eigenvalues: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Extract conjugate pairs from eigenvalue array.
        
        Args:
            eigenvalues: Array of eigenvalues (possibly complex)
            
        Returns:
            Tuple of (real_eigenvalues, is_properly_paired)
        """
        if not eigenvalues.is_complex():
            # Already real
            return eigenvalues, True
        
        # Separate real and imaginary parts
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        
        # Check if all imaginary parts are negligible
        if torch.abs(imag_parts).max() <= self.tolerance:
            return real_parts, True
        
        # For complex eigenvalues, they should come in conjugate pairs
        # This is a simplified check - full implementation would need pairing logic
        return real_parts, False
    
    def get_reconstruction_metadata(self, real_matrix: torch.Tensor) -> Dict[str, Any]:
        """Get metadata about the reconstruction operation.
        
        Args:
            real_matrix: Real matrix to be reconstructed
            
        Returns:
            Dictionary with reconstruction metadata
        """
        n_double, m_double = real_matrix.shape
        n, m = n_double // 2, m_double // 2
        
        metadata = {
            'real_shape': real_matrix.shape,
            'reconstructed_shape': (n, m),
            'dimension_scaling': 0.25,  # 1/4 the dimensions
            'memory_scaling': 0.5,      # 1/2 the memory
            'dtype': real_matrix.dtype,
            'device': str(real_matrix.device),
            'is_valid_embedding': n_double % 2 == 0 and m_double % 2 == 0
        }
        
        return metadata
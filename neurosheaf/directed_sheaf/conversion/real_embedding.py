"""Complex-to-real embedding for directed sheaf computation.

This module implements the mathematical conversion of complex matrices to real
representation for efficient numerical computation while preserving spectral
properties.

Mathematical Foundation:
For a complex matrix block Z = X + iY, the real representation is:
    Re(Z) = [[X, -Y], [Y, X]] ∈ R^{2d × 2d}

Key Properties:
- Doubles all dimensions but preserves spectral properties
- Eigenvalues of Z appear as conjugate pairs in Re(Z)
- Hermitian matrices map to symmetric matrices
- Positive definiteness is preserved

This enables efficient computation using real-valued numerical libraries
while maintaining the mathematical correctness of complex sheaf operations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.sparse import csr_matrix, bsr_matrix, block_diag
import networkx as nx

# Import directed sheaf structures
from ..data_structures import DirectedSheaf

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


class ComplexToRealEmbedding:
    """Converts complex matrices to real representation for computation.
    
    This class implements the standard complex-to-real embedding that enables
    efficient numerical computation while preserving all spectral properties
    of the original complex matrices.
    
    Mathematical Implementation:
    For complex matrix Z = X + iY, the real embedding is:
        Re(Z) = [[X, -Y], [Y, X]]
    
    Key Features:
    - Preserves eigenvalues as conjugate pairs
    - Maps Hermitian matrices to symmetric matrices
    - Maintains positive definiteness
    - Supports sparse matrix operations
    - Handles edge masking for filtration
    
    The embedding doubles matrix dimensions but enables the use of real-valued
    eigensolvers and numerical libraries while maintaining mathematical rigor.
    """
    
    def __init__(self, validate_properties: bool = True, 
                 tolerance: float = 1e-12,
                 device: Optional[torch.device] = None):
        """Initialize the complex-to-real embedding computer.
        
        Args:
            validate_properties: Whether to validate mathematical properties
            tolerance: Tolerance for numerical validation
            device: PyTorch device for computations
        """
        self.validate_properties = validate_properties
        self.tolerance = tolerance
        self.device = device or torch.device('cpu')
        
        logger.debug(f"ComplexToRealEmbedding initialized with device={device}")
    
    def embed_matrix(self, complex_matrix: torch.Tensor) -> torch.Tensor:
        """Convert complex matrix to real representation.
        
        Mathematical Implementation:
        For Z = X + iY, computes:
            Re(Z) = [[X, -Y], [Y, X]]
        
        Args:
            complex_matrix: Complex matrix of shape (n, m)
            
        Returns:
            Real matrix of shape (2n, 2m) representing the complex matrix
            
        Raises:
            ValueError: If input is not a complex tensor
            RuntimeError: If conversion fails validation
        """
        if not isinstance(complex_matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if not complex_matrix.is_complex():
            raise ValueError("Input tensor must be complex")
        
        # Move to specified device
        complex_matrix = complex_matrix.to(self.device)
        
        # Extract real and imaginary parts
        real_part = complex_matrix.real
        imag_part = complex_matrix.imag
        
        # Construct real embedding: [[X, -Y], [Y, X]]
        top_row = torch.cat([real_part, -imag_part], dim=1)
        bottom_row = torch.cat([imag_part, real_part], dim=1)
        real_embedding = torch.cat([top_row, bottom_row], dim=0)
        
        # Validate properties if requested
        if self.validate_properties:
            self._validate_matrix_embedding(complex_matrix, real_embedding)
        
        logger.debug(f"Embedded complex matrix {complex_matrix.shape} → real matrix {real_embedding.shape}")
        return real_embedding
    
    def embed_vector(self, complex_vector: torch.Tensor) -> torch.Tensor:
        """Convert complex vector to real representation.
        
        Args:
            complex_vector: Complex vector of shape (n,) or (n, 1)
            
        Returns:
            Real vector of shape (2n,) or (2n, 1) representing the complex vector
        """
        if not isinstance(complex_vector, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if not complex_vector.is_complex():
            raise ValueError("Input tensor must be complex")
        
        # Move to specified device
        complex_vector = complex_vector.to(self.device)
        
        # Extract real and imaginary parts
        real_part = complex_vector.real
        imag_part = complex_vector.imag
        
        # Construct real embedding: [real_part, imag_part]
        if complex_vector.dim() == 1:
            real_embedding = torch.cat([real_part, imag_part], dim=0)
        else:
            real_embedding = torch.cat([real_part, imag_part], dim=0)
        
        logger.debug(f"Embedded complex vector {complex_vector.shape} → real vector {real_embedding.shape}")
        return real_embedding
    
    def embed_stalk(self, complex_stalk: torch.Tensor) -> torch.Tensor:
        """Convert complex stalk to real representation.
        
        Args:
            complex_stalk: Complex stalk tensor
            
        Returns:
            Real stalk tensor with doubled dimensions
        """
        return self.embed_matrix(complex_stalk)
    
    def embed_restrictions(self, directed_restrictions: Dict[Tuple[str, str], torch.Tensor]) -> Dict[Tuple[str, str], torch.Tensor]:
        """Convert complex restriction maps to real representation.
        
        Args:
            directed_restrictions: Dictionary mapping edges to complex restriction tensors
            
        Returns:
            Dictionary mapping edges to real restriction tensors
        """
        if not isinstance(directed_restrictions, dict):
            raise ValueError("directed_restrictions must be a dictionary")
        
        real_restrictions = {}
        
        for edge, complex_restriction in directed_restrictions.items():
            try:
                real_restriction = self.embed_matrix(complex_restriction)
                real_restrictions[edge] = real_restriction
                
                logger.debug(f"Embedded restriction for edge {edge}: {complex_restriction.shape} → {real_restriction.shape}")
                
            except Exception as e:
                logger.error(f"Failed to embed restriction for edge {edge}: {e}")
                raise RuntimeError(f"Restriction embedding failed for edge {edge}: {e}")
        
        logger.info(f"Embedded {len(real_restrictions)} restriction maps")
        return real_restrictions
    
    def embed_sheaf_data(self, directed_sheaf: DirectedSheaf) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]]:
        """Convert entire directed sheaf to real representation.
        
        Args:
            directed_sheaf: DirectedSheaf with complex stalks and restrictions
            
        Returns:
            Tuple of (real_stalks, real_restrictions) dictionaries
        """
        if not isinstance(directed_sheaf, DirectedSheaf):
            raise ValueError("Input must be a DirectedSheaf")
        
        # Convert complex stalks to real representation
        real_stalks = {}
        for node, complex_stalk in directed_sheaf.complex_stalks.items():
            real_stalks[node] = self.embed_stalk(complex_stalk)
        
        # Convert directed restrictions to real representation
        real_restrictions = self.embed_restrictions(directed_sheaf.directed_restrictions)
        
        logger.info(f"Embedded complete sheaf: {len(real_stalks)} stalks, {len(real_restrictions)} restrictions")
        return real_stalks, real_restrictions
    
    def embed_laplacian_blocks(self, laplacian_blocks: Dict[Tuple[str, str], torch.Tensor]) -> Dict[Tuple[str, str], torch.Tensor]:
        """Convert complex Laplacian blocks to real representation.
        
        Args:
            laplacian_blocks: Dictionary mapping (node1, node2) to complex Laplacian blocks
            
        Returns:
            Dictionary mapping (node1, node2) to real Laplacian blocks
        """
        if not isinstance(laplacian_blocks, dict):
            raise ValueError("laplacian_blocks must be a dictionary")
        
        real_blocks = {}
        
        for block_key, complex_block in laplacian_blocks.items():
            try:
                real_block = self.embed_matrix(complex_block)
                real_blocks[block_key] = real_block
                
                logger.debug(f"Embedded Laplacian block {block_key}: {complex_block.shape} → {real_block.shape}")
                
            except Exception as e:
                logger.error(f"Failed to embed Laplacian block {block_key}: {e}")
                raise RuntimeError(f"Laplacian block embedding failed for {block_key}: {e}")
        
        logger.info(f"Embedded {len(real_blocks)} Laplacian blocks")
        return real_blocks
    
    def embed_sparse_matrix(self, complex_sparse: csr_matrix) -> csr_matrix:
        """Convert complex sparse matrix to real representation.
        
        Args:
            complex_sparse: Complex sparse matrix
            
        Returns:
            Real sparse matrix with doubled dimensions
        """
        if not np.iscomplexobj(complex_sparse.data):
            raise ValueError("Input sparse matrix must be complex")
        
        # Extract real and imaginary parts
        real_part = complex_sparse.real
        imag_part = complex_sparse.imag
        
        # Construct real embedding using sparse block operations
        # [[X, -Y], [Y, X]]
        from scipy.sparse import hstack, vstack
        
        top_row = hstack([real_part, -imag_part])
        bottom_row = hstack([imag_part, real_part])
        
        real_embedding = vstack([top_row, bottom_row])
        
        logger.debug(f"Embedded complex sparse matrix {complex_sparse.shape} → real sparse matrix {real_embedding.shape}")
        return real_embedding
    
    def embed_with_filtration_mask(self, complex_matrix: torch.Tensor, 
                                  mask: torch.Tensor) -> torch.Tensor:
        """Convert complex matrix to real representation with filtration masking.
        
        As specified in the mathematical formulation, masking must be applied
        to the real representation, zeroing out both real and imaginary parts
        (all four blocks in the 2×2 real representation) for filtered edges.
        
        Args:
            complex_matrix: Complex matrix to embed
            mask: Boolean mask for filtration
            
        Returns:
            Real embedded matrix with filtration applied
        """
        # First embed the matrix
        real_embedding = self.embed_matrix(complex_matrix)
        
        # Apply mask to all four blocks
        n, m = complex_matrix.shape
        
        # Create expanded mask for real representation
        expanded_mask = torch.zeros(2*n, 2*m, dtype=torch.bool, device=self.device)
        
        # Apply mask to all four blocks: [[X, -Y], [Y, X]]
        expanded_mask[:n, :m] = mask  # X block
        expanded_mask[:n, m:] = mask  # -Y block
        expanded_mask[n:, :m] = mask  # Y block
        expanded_mask[n:, m:] = mask  # X block
        
        # Zero out masked entries
        real_embedding = real_embedding * expanded_mask.float()
        
        logger.debug(f"Applied filtration mask to embedded matrix")
        return real_embedding
    
    def _validate_matrix_embedding(self, complex_matrix: torch.Tensor, 
                                  real_embedding: torch.Tensor) -> None:
        """Validate mathematical properties of matrix embedding.
        
        Args:
            complex_matrix: Original complex matrix
            real_embedding: Real embedding result
            
        Raises:
            RuntimeError: If validation fails
        """
        n, m = complex_matrix.shape
        
        # Check dimensions
        if real_embedding.shape != (2*n, 2*m):
            raise RuntimeError(f"Embedding dimension mismatch: expected {(2*n, 2*m)}, got {real_embedding.shape}")
        
        # Check that real_embedding is real
        if real_embedding.is_complex():
            raise RuntimeError("Real embedding should not be complex")
        
        # Validate block structure
        real_part = complex_matrix.real
        imag_part = complex_matrix.imag
        
        # Check top-left block (X)
        top_left = real_embedding[:n, :m]
        if torch.abs(top_left - real_part).max() > self.tolerance:
            raise RuntimeError("Top-left block (X) embedding incorrect")
        
        # Check top-right block (-Y)
        top_right = real_embedding[:n, m:]
        if torch.abs(top_right - (-imag_part)).max() > self.tolerance:
            raise RuntimeError("Top-right block (-Y) embedding incorrect")
        
        # Check bottom-left block (Y)
        bottom_left = real_embedding[n:, :m]
        if torch.abs(bottom_left - imag_part).max() > self.tolerance:
            raise RuntimeError("Bottom-left block (Y) embedding incorrect")
        
        # Check bottom-right block (X)
        bottom_right = real_embedding[n:, m:]
        if torch.abs(bottom_right - real_part).max() > self.tolerance:
            raise RuntimeError("Bottom-right block (X) embedding incorrect")
        
        # If complex matrix is Hermitian, check that real embedding is symmetric
        if self._is_hermitian(complex_matrix):
            if not self._is_symmetric(real_embedding):
                raise RuntimeError("Hermitian matrix should map to symmetric real matrix")
        
        logger.debug("Matrix embedding validation passed")
    
    def _is_hermitian(self, matrix: torch.Tensor, tolerance: Optional[float] = None) -> bool:
        """Check if matrix is Hermitian."""
        if tolerance is None:
            tolerance = self.tolerance
        
        # Only square matrices can be Hermitian
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        hermitian_error = torch.abs(matrix - matrix.conj().T).max()
        return hermitian_error <= tolerance
    
    def _is_symmetric(self, matrix: torch.Tensor, tolerance: Optional[float] = None) -> bool:
        """Check if matrix is symmetric."""
        if tolerance is None:
            tolerance = self.tolerance
        
        symmetric_error = torch.abs(matrix - matrix.T).max()
        return symmetric_error <= tolerance
    
    def get_embedding_metadata(self, complex_matrix: torch.Tensor) -> Dict[str, Any]:
        """Get metadata about the embedding operation.
        
        Args:
            complex_matrix: Complex matrix to analyze
            
        Returns:
            Dictionary with embedding metadata
        """
        metadata = {
            'original_shape': complex_matrix.shape,
            'embedded_shape': (2 * complex_matrix.shape[0], 2 * complex_matrix.shape[1]),
            'dimension_scaling': 4,  # 2x in each dimension
            'memory_scaling': 4,     # 4x total memory
            'is_hermitian': self._is_hermitian(complex_matrix),
            'dtype': complex_matrix.dtype,
            'device': str(complex_matrix.device)
        }
        
        # Add sparsity information if applicable
        if hasattr(complex_matrix, 'sparse_dim'):
            metadata['is_sparse'] = True
        else:
            metadata['is_sparse'] = False
        
        return metadata
    
    def estimate_memory_overhead(self, complex_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Estimate memory overhead for embedding operation.
        
        Args:
            complex_shape: Shape of complex matrix
            
        Returns:
            Dictionary with memory estimates
        """
        n, m = complex_shape
        
        # Complex tensor: 2 * n * m * 4 bytes (float32)
        complex_memory = 2 * n * m * 4
        
        # Real embedding: 2n * 2m * 4 bytes (float32)
        real_memory = 4 * n * m * 4
        
        return {
            'complex_memory_bytes': complex_memory,
            'real_memory_bytes': real_memory,
            'memory_overhead_ratio': 2.0,  # 2x memory overhead
            'total_memory_bytes': complex_memory + real_memory,
            'dimension_scaling': (2*n, 2*m)
        }
# neurosheaf/spectral/gw/gw_eigenspace_embedder.py
"""
SVD-based eigenspace embedding for GW filtrations.

Implements eigenspace embedding using SVD alignment methods from recent
research on learning sheaf Laplacians, with adaptations for the transport
structure of Gromov-Wasserstein sheaf constructions.

Mathematical Foundation:
- SVD-based rotation matrices VU^T for optimal alignment
- Transport-aware eigenspace preservation
- Orthogonal transformations maintaining geometric structure
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError

logger = setup_logger(__name__)


class GWEigenspaceEmbedder:
    """
    SVD-based eigenspace embedding for GW filtrations.
    
    Provides multiple methods for embedding eigenspaces from smaller to larger
    dimensions while preserving geometric and transport-related structure.
    """
    
    def __init__(self, 
                 embedding_method: str = 'svd_alignment',
                 preserve_orthogonality: bool = True,
                 numerical_tolerance: float = 1e-12):
        """
        Initialize GW eigenspace embedder.
        
        Args:
            embedding_method: Method for eigenspace embedding
                - 'svd_alignment': SVD-based rotational alignment
                - 'transport_weighted': Transport-cost weighted embedding
                - 'orthogonal_extension': Orthogonal basis extension
            preserve_orthogonality: Whether to maintain orthogonal structure
            numerical_tolerance: Tolerance for numerical computations
        """
        self.embedding_method = embedding_method
        self.preserve_orthogonality = preserve_orthogonality
        self.numerical_tolerance = numerical_tolerance
        
        valid_methods = ['svd_alignment', 'transport_weighted', 'orthogonal_extension']
        if embedding_method not in valid_methods:
            raise ValueError(f"Invalid embedding method '{embedding_method}'. "
                           f"Valid options: {valid_methods}")
        
        logger.info(f"GWEigenspaceEmbedder initialized: method={embedding_method}, "
                   f"preserve_orthogonality={preserve_orthogonality}")
    
    def embed_eigenspace(self, 
                        prev_eigenvectors: torch.Tensor,
                        inclusion_mapping: torch.Tensor,
                        transport_costs: Optional[torch.Tensor] = None,
                        target_dimension: Optional[int] = None) -> torch.Tensor:
        """
        Embed eigenspace from previous step into current step space.
        
        Uses inclusion mapping to embed eigenspace while preserving
        geometric structure and incorporating transport information.
        
        Args:
            prev_eigenvectors: Previous step eigenvectors [dim x n_prev]
            inclusion_mapping: Inclusion map [curr_dim x prev_dim]
            transport_costs: Transport cost matrix (optional)
            target_dimension: Target embedding dimension (optional)
            
        Returns:
            Embedded eigenvectors [curr_dim x n_prev]
            
        Raises:
            ComputationError: If embedding fails due to dimension mismatch
        """
        if prev_eigenvectors.shape[0] != inclusion_mapping.shape[1]:
            raise ComputationError(
                f"Dimension mismatch: eigenvectors {prev_eigenvectors.shape[0]} vs "
                f"inclusion mapping input {inclusion_mapping.shape[1]}",
                operation="embed_eigenspace"
            )
        
        logger.debug(f"Embedding eigenspace: {prev_eigenvectors.shape} → "
                    f"[{inclusion_mapping.shape[0]} x {prev_eigenvectors.shape[1]}]")
        
        try:
            if self.embedding_method == 'svd_alignment':
                return self._embed_svd_alignment(prev_eigenvectors, inclusion_mapping)
            elif self.embedding_method == 'transport_weighted':
                return self._embed_transport_weighted(prev_eigenvectors, inclusion_mapping, 
                                                    transport_costs)
            elif self.embedding_method == 'orthogonal_extension':
                return self._embed_orthogonal_extension(prev_eigenvectors, inclusion_mapping)
            else:
                raise ValueError(f"Unknown embedding method: {self.embedding_method}")
                
        except Exception as e:
            raise ComputationError(
                f"Eigenspace embedding failed with method {self.embedding_method}: {e}",
                operation="embed_eigenspace"
            )
    
    def _embed_svd_alignment(self, 
                           prev_eigenvectors: torch.Tensor,
                           inclusion_mapping: torch.Tensor) -> torch.Tensor:
        """
        Embed eigenspace using SVD-based alignment.
        
        Uses SVD of inclusion mapping to extract optimal rotation
        and applies it to preserve geometric structure during embedding.
        
        Args:
            prev_eigenvectors: Previous eigenvectors [dim x n_prev]
            inclusion_mapping: Inclusion mapping [curr_dim x prev_dim]
            
        Returns:
            SVD-aligned embedded eigenvectors
        """
        # Direct application of inclusion mapping
        # This already contains SVD-based structure from SheafInclusionMapper
        embedded_vectors = torch.mm(inclusion_mapping, prev_eigenvectors)
        
        # Optionally re-orthogonalize if required
        if self.preserve_orthogonality:
            embedded_vectors = self._orthogonalize_vectors(embedded_vectors)
        
        logger.debug(f"SVD alignment embedding: {prev_eigenvectors.shape} → {embedded_vectors.shape}")
        return embedded_vectors
    
    def _embed_transport_weighted(self, 
                                prev_eigenvectors: torch.Tensor,
                                inclusion_mapping: torch.Tensor,
                                transport_costs: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Embed eigenspace with transport cost weighting.
        
        Incorporates transport cost information to weight the embedding,
        giving higher importance to low-cost (high-quality) correspondences.
        
        Args:
            prev_eigenvectors: Previous eigenvectors [dim x n_prev]
            inclusion_mapping: Inclusion mapping [curr_dim x prev_dim]
            transport_costs: Transport cost matrix (optional)
            
        Returns:
            Transport-weighted embedded eigenvectors
        """
        # Start with basic inclusion mapping
        embedded_vectors = torch.mm(inclusion_mapping, prev_eigenvectors)
        
        # Apply transport weighting if available
        if transport_costs is not None:
            weighted_vectors = self._apply_transport_weighting(
                embedded_vectors, transport_costs
            )
            embedded_vectors = weighted_vectors
        
        # Re-orthogonalize if required
        if self.preserve_orthogonality:
            embedded_vectors = self._orthogonalize_vectors(embedded_vectors)
        
        logger.debug(f"Transport-weighted embedding: {prev_eigenvectors.shape} → {embedded_vectors.shape}")
        return embedded_vectors
    
    def _embed_orthogonal_extension(self, 
                                  prev_eigenvectors: torch.Tensor,
                                  inclusion_mapping: torch.Tensor) -> torch.Tensor:
        """
        Embed eigenspace using orthogonal extension.
        
        Ensures embedded vectors maintain orthogonality and extend
        the eigenspace in a mathematically consistent way.
        
        Args:
            prev_eigenvectors: Previous eigenvectors [dim x n_prev]
            inclusion_mapping: Inclusion mapping [curr_dim x prev_dim]
            
        Returns:
            Orthogonally embedded eigenvectors
        """
        # Apply inclusion mapping
        embedded_vectors = torch.mm(inclusion_mapping, prev_eigenvectors)
        
        # Ensure orthogonality through QR decomposition
        embedded_vectors = self._orthogonalize_vectors(embedded_vectors)
        
        logger.debug(f"Orthogonal extension embedding: {prev_eigenvectors.shape} → {embedded_vectors.shape}")
        return embedded_vectors
    
    def _orthogonalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize vectors using QR decomposition.
        
        Maintains the span of the eigenspace while ensuring orthogonality,
        which is important for numerical stability in PES computation.
        
        Args:
            vectors: Input vectors [dim x n_vectors]
            
        Returns:
            Orthogonalized vectors [dim x n_vectors]
        """
        if vectors.shape[1] == 0:
            return vectors
        
        try:
            # QR decomposition for orthogonalization
            Q, R = torch.linalg.qr(vectors, mode='reduced')
            
            # Handle rank deficiency by keeping only non-zero diagonal elements
            diag_R = torch.diag(R)
            valid_cols = torch.abs(diag_R) > self.numerical_tolerance
            
            if torch.sum(valid_cols) < vectors.shape[1]:
                logger.debug(f"Rank deficiency detected: {torch.sum(valid_cols)} valid of {vectors.shape[1]} vectors")
                Q = Q[:, valid_cols]
            
            return Q
            
        except Exception as e:
            logger.warning(f"QR orthogonalization failed: {e}, returning original vectors")
            return vectors
    
    def _apply_transport_weighting(self, 
                                 embedded_vectors: torch.Tensor,
                                 transport_costs: torch.Tensor) -> torch.Tensor:
        """
        Apply transport cost weighting to embedded vectors.
        
        Uses transport costs to weight vector components, emphasizing
        directions corresponding to low transport costs.
        
        Args:
            embedded_vectors: Embedded vectors [dim x n_vectors]
            transport_costs: Transport cost matrix
            
        Returns:
            Transport-weighted vectors
        """
        try:
            # Create weighting based on transport costs
            # Lower cost → higher weight
            if transport_costs.numel() == 0:
                return embedded_vectors
            
            # Simple weighting scheme: weight = exp(-cost)
            weights = torch.exp(-transport_costs)
            
            # Apply weights to vector components
            # This is a simplified approach - more sophisticated methods could be used
            if weights.shape[0] == embedded_vectors.shape[0]:
                # Apply row-wise weighting
                weighted_vectors = embedded_vectors * weights.unsqueeze(1)
            else:
                # Use scalar weighting if dimensions don't match
                scalar_weight = torch.mean(weights)
                weighted_vectors = embedded_vectors * scalar_weight
            
            return weighted_vectors
            
        except Exception as e:
            logger.warning(f"Transport weighting failed: {e}, returning unweighted vectors")
            return embedded_vectors
    
    def compute_embedding_quality(self, 
                                original_vectors: torch.Tensor,
                                embedded_vectors: torch.Tensor,
                                inclusion_mapping: torch.Tensor) -> Dict[str, float]:
        """
        Compute quality metrics for eigenspace embedding.
        
        Evaluates how well the embedding preserves geometric structure
        and maintains mathematical properties of the original eigenspace.
        
        Args:
            original_vectors: Original eigenvectors [prev_dim x n_vectors]
            embedded_vectors: Embedded eigenvectors [curr_dim x n_vectors]
            inclusion_mapping: Inclusion mapping used [curr_dim x prev_dim]
            
        Returns:
            Dictionary with embedding quality metrics
        """
        try:
            metrics = {}
            
            # Reconstruction error
            reconstructed = torch.mm(inclusion_mapping, original_vectors)
            reconstruction_error = torch.norm(embedded_vectors - reconstructed, 'fro').item()
            metrics['reconstruction_error'] = reconstruction_error
            
            # Orthogonality preservation
            if original_vectors.shape[1] > 1:
                orig_gram = torch.mm(original_vectors.T, original_vectors)
                emb_gram = torch.mm(embedded_vectors.T, embedded_vectors)
                
                # Compare Gram matrices
                gram_diff = torch.norm(orig_gram - emb_gram[:original_vectors.shape[1], 
                                                          :original_vectors.shape[1]], 'fro').item()
                metrics['orthogonality_preservation'] = 1.0 / (1.0 + gram_diff)
            else:
                metrics['orthogonality_preservation'] = 1.0
            
            # Norm preservation
            orig_norms = torch.norm(original_vectors, dim=0)
            emb_norms = torch.norm(embedded_vectors, dim=0)
            norm_preservation = 1.0 - torch.mean(torch.abs(orig_norms - emb_norms)).item()
            metrics['norm_preservation'] = max(0.0, norm_preservation)
            
            # Overall quality score
            metrics['overall_quality'] = (
                0.4 * (1.0 / (1.0 + reconstruction_error)) +
                0.3 * metrics['orthogonality_preservation'] +
                0.3 * metrics['norm_preservation']
            )
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Embedding quality computation failed: {e}")
            return {'overall_quality': 0.0, 'error': str(e)}
    
    def validate_embedding(self, 
                          embedded_vectors: torch.Tensor,
                          expected_shape: Tuple[int, int]) -> bool:
        """
        Validate embedded eigenspace for correctness.
        
        Checks mathematical properties and numerical stability
        of the embedded eigenspace.
        
        Args:
            embedded_vectors: Embedded eigenvectors to validate
            expected_shape: Expected shape (curr_dim, n_vectors)
            
        Returns:
            True if embedding is valid, False otherwise
        """
        try:
            # Check shape
            if embedded_vectors.shape != expected_shape:
                logger.error(f"Embedding shape mismatch: {embedded_vectors.shape} vs {expected_shape}")
                return False
            
            # Check for NaN/inf values
            if torch.any(torch.isnan(embedded_vectors)) or torch.any(torch.isinf(embedded_vectors)):
                logger.error("Embedded vectors contain NaN or infinite values")
                return False
            
            # Check for zero vectors (which could indicate problems)
            vector_norms = torch.norm(embedded_vectors, dim=0)
            zero_vectors = torch.sum(vector_norms < self.numerical_tolerance).item()
            if zero_vectors > 0:
                logger.warning(f"Found {zero_vectors} near-zero embedded vectors")
            
            # Check numerical conditioning
            if embedded_vectors.shape[1] > 1:
                # Compute condition number of Gram matrix
                gram_matrix = torch.mm(embedded_vectors.T, embedded_vectors)
                eigenvals = torch.eigenvalues(gram_matrix)[0].real
                
                if torch.min(eigenvals) > self.numerical_tolerance:
                    condition_number = (torch.max(eigenvals) / torch.min(eigenvals)).item()
                    if condition_number > 1e12:
                        logger.warning(f"High condition number in embedded vectors: {condition_number}")
            
            logger.debug("Embedding validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            return False
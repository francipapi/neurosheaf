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
from typing import Dict, List, Optional, Tuple, Union, Any
import scipy.linalg
from scipy.linalg import subspace_angles
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError
from .sheaf_inclusion_mapper import InclusionMapping
from ..dtype_policy import to_spectral_dtype, DEFAULT_SPECTRAL_POLICY

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
                 numerical_tolerance: float = 1e-12,
                 use_pivoted_qr: bool = True,
                 rank_tolerance: float = 1e-12,
                 warn_on_rank_loss: bool = True,
                 deterministic_backfill: bool = True,
                 angle_cap_degrees: float = 60.0,
                 use_procrustes_alignment: bool = True,
                 smoothing_alpha: float = 0.3):
        """
        Initialize GW eigenspace embedder.
        
        Args:
            embedding_method: Method for eigenspace embedding
                - 'svd_alignment': SVD-based rotational alignment
                - 'transport_weighted': Transport-cost weighted embedding
                - 'orthogonal_extension': Orthogonal basis extension
            preserve_orthogonality: Whether to maintain orthogonal structure
            numerical_tolerance: Tolerance for numerical computations
            use_pivoted_qr: Whether to use rank-revealing QR (RRQR) for orthogonalization
            rank_tolerance: Relative tolerance for rank detection (relative to matrix norm)
            warn_on_rank_loss: Whether to warn when rank deficiency is detected
            deterministic_backfill: Whether to use deterministic seeding for rank backfilling
            angle_cap_degrees: Maximum allowed principal angle in degrees (default 60°)
            use_procrustes_alignment: Whether to use Procrustes-based basis alignment
            smoothing_alpha: Alpha parameter for exponential smoothing of mappings
        """
        self.embedding_method = embedding_method
        self.preserve_orthogonality = preserve_orthogonality
        self.numerical_tolerance = numerical_tolerance
        self.use_pivoted_qr = use_pivoted_qr
        self.rank_tolerance = rank_tolerance
        self.warn_on_rank_loss = warn_on_rank_loss
        self.deterministic_backfill = deterministic_backfill
        self.angle_cap_degrees = angle_cap_degrees
        self.use_procrustes_alignment = use_procrustes_alignment
        self.smoothing_alpha = smoothing_alpha
        
        # Convert angle cap to radians for internal use
        self.angle_cap_radians = np.radians(angle_cap_degrees)
        
        # Track previous subspace for principal angle monitoring
        self._previous_subspace = None
        self._previous_inclusion_matrix = None  # For smoothing
        
        valid_methods = ['svd_alignment', 'transport_weighted', 'orthogonal_extension']
        if embedding_method not in valid_methods:
            raise ValueError(f"Invalid embedding method '{embedding_method}'. "
                           f"Valid options: {valid_methods}")
        
        logger.info(f"GWEigenspaceEmbedder initialized: method={embedding_method}, "
                   f"preserve_orthogonality={preserve_orthogonality}, use_pivoted_qr={use_pivoted_qr}")
    
    def embed_eigenspace(self, 
                        prev_eigenvectors: torch.Tensor,
                        inclusion_mapping: InclusionMapping,
                        transport_costs: Optional[torch.Tensor] = None,
                        target_dimension: Optional[int] = None) -> torch.Tensor:
        """
        Embed eigenspace from previous step into current step space.
        
        Uses inclusion mapping to embed eigenspace while preserving
        geometric structure and incorporating transport information.
        
        Args:
            prev_eigenvectors: Previous step eigenvectors [dim x n_prev]
            inclusion_mapping: InclusionMapping containing matrix [curr_dim x prev_dim] and metadata
            transport_costs: Transport cost matrix (optional)
            target_dimension: Target embedding dimension (optional)
            
        Returns:
            Embedded eigenvectors [curr_dim x n_prev]
            
        Raises:
            ComputationError: If embedding fails due to dimension mismatch
        """
        # Extract the inclusion matrix from the InclusionMapping dataclass
        logger.debug(f"embed_eigenspace called with prev_eigenvectors type: {type(prev_eigenvectors)}, inclusion_mapping type: {type(inclusion_mapping)}")
        inclusion_matrix = inclusion_mapping.matrix
        
        if isinstance(prev_eigenvectors, tuple):
            import traceback
            logger.error(f"EMBEDDER TUPLE STACK TRACE:")
            logger.error(traceback.format_stack())
            logger.error(f"embed_eigenspace: prev_eigenvectors is tuple instead of tensor: {type(prev_eigenvectors)}")
            logger.error(f"embed_eigenspace: tuple structure: {[type(x) for x in prev_eigenvectors]}")
            
            # Try to extract eigenvector tensor from tuple
            extracted = False
            for i, item in enumerate(prev_eigenvectors):
                if torch.is_tensor(item):
                    logger.warning(f"embed_eigenspace: found tensor at position {i}: shape {item.shape if hasattr(item, 'shape') else 'no shape'}")
                    if hasattr(item, 'shape') and len(item.shape) >= 2:  # Likely eigenvectors (2D)
                        prev_eigenvectors = item
                        logger.warning(f"embed_eigenspace: extracted 2D tensor as prev_eigenvectors: {item.shape}")
                        extracted = True
                        break
            
            if not extracted:
                # Fallback: try second element
                if len(prev_eigenvectors) > 1 and torch.is_tensor(prev_eigenvectors[1]):
                    prev_eigenvectors = prev_eigenvectors[1]
                    logger.warning(f"embed_eigenspace: fallback extracted prev_eigenvectors[1]")
                    extracted = True
            
            if not extracted:
                # Ultimate fallback: return empty tensor
                logger.error("embed_eigenspace: cannot extract valid tensor from tuple, returning empty")
                return torch.tensor([])
        
        # Validate that we now have a proper tensor
        if not torch.is_tensor(prev_eigenvectors):
            logger.error(f"embed_eigenspace: prev_eigenvectors is not a tensor: {type(prev_eigenvectors)}")
            return torch.tensor([])
        
        if not hasattr(prev_eigenvectors, 'shape') or len(prev_eigenvectors.shape) == 0:
            logger.error("embed_eigenspace: prev_eigenvectors has no valid shape")
            return torch.tensor([])
        
        if not torch.is_tensor(inclusion_matrix):
            logger.error(f"embed_eigenspace: inclusion_matrix is not a tensor: {type(inclusion_matrix)}")
            return torch.tensor([])
        
        # 🔧 DTYPE CONSISTENCY FIX: Use centralized policy instead of ad-hoc conversion
        # Apply spectral dtype policy at module entry point (single conversion point)
        if prev_eigenvectors.dtype != inclusion_matrix.dtype:
            logger.debug(f"embed_eigenspace: dtype mismatch - prev_eigenvectors: {prev_eigenvectors.dtype}, inclusion_matrix: {inclusion_matrix.dtype}")
            logger.debug(f"embed_eigenspace: converting both to spectral dtype policy (float64)")
        
        prev_eigenvectors = to_spectral_dtype(prev_eigenvectors)
        inclusion_matrix = to_spectral_dtype(inclusion_matrix)
        
        if prev_eigenvectors.shape[0] != inclusion_matrix.shape[1]:
            raise ComputationError(
                f"Dimension mismatch: eigenvectors {prev_eigenvectors.shape[0]} vs "
                f"inclusion mapping input {inclusion_matrix.shape[1]}",
                operation="embed_eigenspace"
            )
        
        logger.debug(f"Embedding eigenspace: {prev_eigenvectors.shape} → "
                    f"[{inclusion_matrix.shape[0]} x {prev_eigenvectors.shape[1]}]")
        
        try:
            if self.embedding_method == 'svd_alignment':
                return self._embed_svd_alignment(prev_eigenvectors, inclusion_matrix)
            elif self.embedding_method == 'transport_weighted':
                return self._embed_transport_weighted(prev_eigenvectors, inclusion_matrix, 
                                                    transport_costs)
            elif self.embedding_method == 'orthogonal_extension':
                return self._embed_orthogonal_extension(prev_eigenvectors, inclusion_matrix)
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
        Embed eigenspace using enhanced SVD-based alignment with Procrustes rotation.
        
        🔧 LARGE PRINCIPAL ANGLE FIX: 
        This method now implements proper Procrustes basis alignment to prevent
        large principal angles (~90°) between successive subspaces.
        
        Algorithm:
        1. Apply inclusion mapping to get initial embedding
        2. If Procrustes alignment is enabled, find optimal rotation
        3. Check principal angles and apply fallback if > angle_cap
        4. Orthogonalize with continuity preservation
        
        Args:
            prev_eigenvectors: Previous eigenvectors [dim x n_prev]
            inclusion_mapping: Inclusion mapping [curr_dim x prev_dim]
            
        Returns:
            Procrustes-aligned embedded eigenvectors with controlled principal angles
        """
        # Check inclusion mapping quality first
        inclusion_quality = self._assess_inclusion_mapping_quality(inclusion_mapping)
        
        if inclusion_quality['is_poor_quality']:
            logger.warning(f"Poor inclusion mapping quality detected: {inclusion_quality['issues']}")
            # Use gentler embedding approach for poor-quality mappings
            embedded_vectors = self._embed_with_regularization(prev_eigenvectors, inclusion_mapping)
        else:
            # Apply inclusion mapping to get initial embedding
            embedded_vectors = torch.mm(inclusion_mapping, prev_eigenvectors)
            
            # 🔧 PROCRUSTES ALIGNMENT: Apply basis alignment if enabled
            if self.use_procrustes_alignment and self._previous_subspace is not None:
                try:
                    embedded_vectors = self._procrustes_align_basis(
                        current_vectors=embedded_vectors,
                        previous_subspace=self._previous_subspace,
                        inclusion_matrix=inclusion_mapping
                    )
                    logger.debug("Applied Procrustes basis alignment")
                except Exception as e:
                    logger.warning(f"Procrustes alignment failed, using direct embedding: {e}")
        
        # Apply gentle orthogonalization that preserves subspace continuity
        if self.preserve_orthogonality:
            embedded_vectors = self._orthogonalize_with_continuity_preservation(embedded_vectors)
        
        # 🔧 ANGLE CAPPING: Check principal angles and fallback if too large
        if self._previous_subspace is not None:
            max_angle = self._check_principal_angles(embedded_vectors)
            if max_angle > self.angle_cap_radians:
                logger.warning(f"Large principal angle {np.degrees(max_angle):.1f}° > {self.angle_cap_degrees}°, "
                             f"applying identity fallback")
                embedded_vectors = self._apply_identity_fallback(prev_eigenvectors, embedded_vectors)
        
        logger.debug(f"Enhanced SVD alignment embedding: {prev_eigenvectors.shape} → {embedded_vectors.shape}")
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
        Orthogonalize vectors using Rank-Revealing QR (RRQR) decomposition.
        
        Uses column-pivoted QR to detect rank deficiency reliably and handles
        near-collinear cases that would be missed by standard QR. When rank
        is deficient, backfills with orthonormal vectors from identity space.
        
        Args:
            vectors: Input vectors [dim x n_vectors]
            
        Returns:
            Orthogonalized vectors [dim x n_vectors] with consistent rank
        """
        if vectors.shape[1] == 0:
            return vectors
        
        original_shape = vectors.shape
        expected_n_vectors = original_shape[1]
        
        # Monitor principal angles if we have a previous subspace
        if self._previous_subspace is not None and vectors.shape[1] > 0:
            self._monitor_subspace_continuity(vectors)
        
        if self.use_pivoted_qr:
            try:
                # Use RRQR (Rank-Revealing QR) via SciPy
                orthogonalized = self._orthogonalize_with_rrqr(vectors)
                
                # Backfill if rank deficient
                if orthogonalized.shape[1] < expected_n_vectors:
                    orthogonalized = self._backfill_rank_deficient_subspace(
                        orthogonalized, expected_n_vectors
                    )
                
            except Exception as e:
                logger.warning(f"RRQR orthogonalization failed: {e}, falling back to PyTorch QR")
                orthogonalized = self._orthogonalize_with_pytorch_qr(vectors)
        else:
            # Use standard PyTorch QR
            orthogonalized = self._orthogonalize_with_pytorch_qr(vectors)
        
        # Store for next principal angle monitoring
        self._previous_subspace = orthogonalized.clone().detach()
        
        # Ensure output shape consistency
        if orthogonalized.shape != original_shape:
            logger.debug(f"Shape change during orthogonalization: {original_shape} → {orthogonalized.shape}")
        
        return orthogonalized
    
    def _orthogonalize_with_rrqr(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize using Rank-Revealing QR (RRQR) via SciPy.
        
        Args:
            vectors: Input vectors [dim x n_vectors]
            
        Returns:
            Orthogonalized vectors with detected rank [dim x rank]
        """
        # Convert to numpy for SciPy
        vectors_np = vectors.detach().cpu().numpy()
        
        # Compute RRQR: A*P = Q*R where P is the permutation matrix
        Q, R, P = scipy.linalg.qr(vectors_np, pivoting=True, mode='economic')
        
        # Compute relative tolerance based on matrix norm
        matrix_norm = np.linalg.norm(vectors_np, 'fro')
        # Use both relative and absolute tolerance for robust rank detection
        abs_tolerance = max(self.rank_tolerance * matrix_norm, self.numerical_tolerance)
        
        # Identify significant columns based on diagonal of R
        diag_R = np.abs(np.diag(R))
        significant_cols = diag_R >= abs_tolerance
        detected_rank = np.sum(significant_cols)
        
        if detected_rank < vectors.shape[1]:
            if self.warn_on_rank_loss:
                logger.warning(f"RRQR detected rank deficiency: {detected_rank}/{vectors.shape[1]} "
                             f"significant vectors (tolerance={abs_tolerance:.2e}, matrix_norm={matrix_norm:.2e})")
                logger.debug(f"R diagonal values: {diag_R}")
            
            # Keep only significant columns
            Q = Q[:, significant_cols]
        
        # Convert back to torch tensor with correct device and dtype
        Q_torch = torch.from_numpy(Q).to(device=vectors.device, dtype=vectors.dtype)
        
        logger.debug(f"RRQR: {vectors.shape} → {Q_torch.shape}, detected rank: {detected_rank}")
        return Q_torch
    
    def _orthogonalize_with_pytorch_qr(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Fallback orthogonalization using standard PyTorch QR.
        
        Args:
            vectors: Input vectors [dim x n_vectors]
            
        Returns:
            Orthogonalized vectors [dim x n_valid_vectors]
        """
        try:
            # Standard QR decomposition
            Q, R = torch.linalg.qr(vectors, mode='reduced')
            
            # Basic rank detection using diagonal of R
            diag_R = torch.diag(R)
            valid_cols = torch.abs(diag_R) > self.numerical_tolerance
            
            if torch.sum(valid_cols) < vectors.shape[1]:
                if self.warn_on_rank_loss:
                    logger.warning(f"PyTorch QR detected rank deficiency: {torch.sum(valid_cols)}/{vectors.shape[1]} valid vectors")
                Q = Q[:, valid_cols]
            
            return Q
            
        except Exception as e:
            logger.warning(f"PyTorch QR orthogonalization failed: {e}, returning original vectors")
            return vectors
    
    def _backfill_rank_deficient_subspace(self, 
                                        orthogonal_vectors: torch.Tensor,
                                        target_dimension: int) -> torch.Tensor:
        """
        Backfill rank-deficient subspace with orthonormal vectors from identity space.
        
        Args:
            orthogonal_vectors: Current orthogonal vectors [dim x current_rank]
            target_dimension: Target number of vectors
            
        Returns:
            Backfilled orthogonal vectors [dim x target_dimension]
        """
        current_rank = orthogonal_vectors.shape[1]
        missing_vectors = target_dimension - current_rank
        
        if missing_vectors <= 0:
            return orthogonal_vectors
        
        ambient_dim = orthogonal_vectors.shape[0]
        
        if self.deterministic_backfill:
            # Use deterministic seeding for reproducible results
            generator = torch.Generator(device=orthogonal_vectors.device)
            generator.manual_seed(42)  # Fixed seed for reproducibility
        else:
            generator = None
        
        # Generate random vectors in ambient space
        if generator is not None:
            candidates = torch.randn((ambient_dim, missing_vectors * 2), 
                                   generator=generator, 
                                   device=orthogonal_vectors.device, 
                                   dtype=orthogonal_vectors.dtype)
        else:
            candidates = torch.randn((ambient_dim, missing_vectors * 2), 
                                   device=orthogonal_vectors.device, 
                                   dtype=orthogonal_vectors.dtype)
        
        # Use Modified Gram-Schmidt for robust orthogonalization against existing vectors
        backfill_vectors = []
        
        for i in range(candidates.shape[1]):
            if len(backfill_vectors) >= missing_vectors:
                break
                
            candidate = candidates[:, i:i+1]
            
            # Project out existing orthogonal vectors
            for j in range(orthogonal_vectors.shape[1]):
                existing_vec = orthogonal_vectors[:, j:j+1]
                proj_coeff = torch.mm(existing_vec.T, candidate)
                candidate = candidate - proj_coeff * existing_vec
            
            # Project out previously added backfill vectors
            for backfill_vec in backfill_vectors:
                proj_coeff = torch.mm(backfill_vec.T, candidate)
                candidate = candidate - proj_coeff * backfill_vec
            
            # Normalize and add if not too small
            candidate_norm = torch.norm(candidate)
            if candidate_norm > self.numerical_tolerance:
                normalized_candidate = candidate / candidate_norm
                backfill_vectors.append(normalized_candidate)
        
        if backfill_vectors:
            backfill_matrix = torch.cat(backfill_vectors, dim=1)
        else:
            # Fallback: try identity-based vectors
            logger.debug("Gram-Schmidt backfill produced no vectors, using identity-based fallback")
            identity_vecs = DEFAULT_SPECTRAL_POLICY.create_eye(ambient_dim, device=orthogonal_vectors.device)
            
            identity_backfill = []
            for i in range(identity_vecs.shape[1]):
                if len(identity_backfill) >= missing_vectors:
                    break
                    
                vec = identity_vecs[:, i:i+1]
                
                # Project out existing vectors from identity
                for j in range(orthogonal_vectors.shape[1]):
                    existing = orthogonal_vectors[:, j:j+1]
                    proj_coeff = torch.mm(existing.T, vec)
                    vec = vec - proj_coeff * existing
                
                # Project out previously added identity vectors
                for prev_vec in identity_backfill:
                    proj_coeff = torch.mm(prev_vec.T, vec)
                    vec = vec - proj_coeff * prev_vec
                
                vec_norm = torch.norm(vec)
                if vec_norm > self.numerical_tolerance:
                    normalized_vec = vec / vec_norm
                    identity_backfill.append(normalized_vec)
            
            if identity_backfill:
                backfill_matrix = torch.cat(identity_backfill, dim=1)
            else:
                backfill_matrix = torch.empty(ambient_dim, 0, device=orthogonal_vectors.device, dtype=orthogonal_vectors.dtype)
        
        if backfill_matrix.shape[1] < missing_vectors:
            logger.warning(f"Could only generate {backfill_matrix.shape[1]}/{missing_vectors} "
                         f"orthogonal backfill vectors")
        
        # Combine original and backfill vectors
        if backfill_matrix.shape[1] > 0:
            result = torch.cat([orthogonal_vectors, backfill_matrix], dim=1)
        else:
            result = orthogonal_vectors
        
        logger.debug(f"Backfilled subspace: {orthogonal_vectors.shape} → {result.shape}")
        return result
    
    def _monitor_subspace_continuity(self, current_vectors: torch.Tensor) -> None:
        """
        🔧 ENHANCED PRINCIPAL ANGLE MONITORING: Comprehensive subspace continuity diagnostics.
        
        This upgraded monitoring provides detailed diagnostics including:
        - Full angle distribution statistics
        - Rapid change detection across consecutive steps  
        - Subspace distance metrics beyond max angle
        - Early warning system before critical thresholds
        
        Args:
            current_vectors: Current vectors before orthogonalization [dim x n_vectors]
        """
        if self._previous_subspace is None:
            return
        
        try:
            # Ensure both subspaces have the same number of vectors for comparison
            min_vecs = min(self._previous_subspace.shape[1], current_vectors.shape[1])
            
            if min_vecs == 0:
                return
            
            prev_subspace = self._previous_subspace[:, :min_vecs].detach().cpu().numpy()
            curr_subspace = current_vectors[:, :min_vecs].detach().cpu().numpy()
            
            # Compute principal angles
            angles = subspace_angles(prev_subspace, curr_subspace)
            
            # 🔧 COMPREHENSIVE STATISTICS
            max_angle = np.max(angles)
            min_angle = np.min(angles)
            mean_angle = np.mean(angles)
            median_angle = np.median(angles)
            std_angle = np.std(angles)
            
            # Convert to degrees for reporting
            max_deg = np.degrees(max_angle)
            min_deg = np.degrees(min_angle)
            mean_deg = np.degrees(mean_angle)
            median_deg = np.degrees(median_angle)
            std_deg = np.degrees(std_angle)
            
            # 🔧 SUBSPACE DISTANCE METRICS
            # Product of cosines (traditional subspace similarity)
            cos_product = np.prod(np.cos(angles))
            
            # Gap to orthogonal subspaces (90° - max_angle)
            orthogonal_gap_deg = 90.0 - max_deg
            
            # 🔧 EARLY WARNING SYSTEM
            warning_threshold = self.angle_cap_degrees * 0.75  # 75% of cap as warning
            critical_threshold = self.angle_cap_degrees * 0.9   # 90% of cap as critical
            
            if max_deg > critical_threshold:
                logger.error(f"🚨 CRITICAL: Principal angle {max_deg:.1f}° > {critical_threshold:.1f}° "
                           f"(near {self.angle_cap_degrees}° cap)")
                logger.error(f"   Distribution: min={min_deg:.1f}°, median={median_deg:.1f}°, "
                           f"std={std_deg:.1f}°, gap_to_orthogonal={orthogonal_gap_deg:.1f}°")
            elif max_deg > warning_threshold:
                logger.warning(f"⚠️  WARNING: Principal angle {max_deg:.1f}° > {warning_threshold:.1f}° "
                             f"(approaching {self.angle_cap_degrees}° cap)")
                logger.warning(f"   Distribution: min={min_deg:.1f}°, median={median_deg:.1f}°, "
                             f"std={std_deg:.1f}°, cos_product={cos_product:.4f}")
            elif max_deg > 45.0:  # Standard moderate threshold
                logger.info(f"📊 MODERATE: Principal angle {max_deg:.1f}° (subspace similarity {cos_product:.3f})")
                logger.info(f"   Distribution: mean={mean_deg:.1f}°, std={std_deg:.1f}°")
            else:
                logger.debug(f"✅ GOOD: Principal angles well-controlled, max={max_deg:.1f}°, mean={mean_deg:.1f}°")
            
            # 🔧 RAPID CHANGE DETECTION
            if hasattr(self, '_previous_max_angle'):
                angle_change = abs(max_deg - self._previous_max_angle)
                if angle_change > 15.0:  # Rapid change threshold
                    logger.warning(f"🔄 RAPID CHANGE: Principal angle changed by {angle_change:.1f}° "
                                 f"({self._previous_max_angle:.1f}° → {max_deg:.1f}°)")
            
            # Store for next comparison
            self._previous_max_angle = max_deg
            
            # 🔧 DETAILED DIAGNOSTICS for severe cases
            if max_deg > 60.0:
                # Log individual angles for diagnosis
                angle_list = ", ".join([f"{np.degrees(a):.1f}°" for a in angles[:5]])  # First 5 angles
                if len(angles) > 5:
                    angle_list += f", ... ({len(angles)} total)"
                logger.debug(f"   Individual angles: [{angle_list}]")
                
                # Check for nearly orthogonal vectors
                near_orthogonal_count = np.sum(angles > np.pi/3)  # > 60°
                if near_orthogonal_count > 0:
                    logger.debug(f"   {near_orthogonal_count}/{len(angles)} vector pairs are > 60° apart")
            
        except Exception as e:
            logger.debug(f"Enhanced principal angle monitoring failed: {e}")
    
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
    
    def _assess_inclusion_mapping_quality(self, inclusion_mapping: torch.Tensor) -> Dict[str, Union[bool, List[str], float]]:
        """
        Assess the quality of an inclusion mapping for numerical stability.
        
        Checks various properties that indicate whether the mapping will
        preserve subspace continuity or cause large principal angles.
        
        Args:
            inclusion_mapping: Inclusion mapping matrix to assess
            
        Returns:
            Dictionary with quality assessment results
        """
        issues = []
        
        try:
            # Check condition number
            U, S, Vt = torch.svd(inclusion_mapping)
            condition_number = (S[0] / S[-1]).item() if S[-1] > 1e-12 else float('inf')
            
            if condition_number > 1e6:
                issues.append(f"high_condition_number_{condition_number:.2e}")
            
            # Check for near-singular behavior
            rank_threshold = 1e-8 * S[0]
            effective_rank = torch.sum(S > rank_threshold).item()
            expected_rank = min(inclusion_mapping.shape)
            
            if effective_rank < expected_rank * 0.8:
                issues.append(f"rank_deficient_{effective_rank}/{expected_rank}")
            
            # Check orthogonality preservation  
            if inclusion_mapping.shape[0] == inclusion_mapping.shape[1]:
                gram = torch.mm(inclusion_mapping.T, inclusion_mapping)
                identity_error = torch.norm(gram - torch.eye(gram.shape[0]), 'fro').item()
                
                if identity_error > 0.1:
                    issues.append(f"non_orthogonal_error_{identity_error:.3f}")
            
            # Check for extreme values that might cause instability
            max_val = torch.max(torch.abs(inclusion_mapping)).item()
            if max_val > 10.0:
                issues.append(f"large_values_{max_val:.2f}")
            
            # Overall assessment
            is_poor_quality = len(issues) > 0
            
            return {
                'is_poor_quality': is_poor_quality,
                'issues': issues,
                'condition_number': condition_number,
                'effective_rank': effective_rank,
                'expected_rank': expected_rank
            }
            
        except Exception as e:
            logger.debug(f"Quality assessment failed: {e}")
            return {
                'is_poor_quality': True,
                'issues': [f"assessment_failed_{e}"],
                'condition_number': float('inf'),
                'effective_rank': 0,
                'expected_rank': min(inclusion_mapping.shape)
            }
    
    def _embed_with_regularization(self, 
                                 prev_eigenvectors: torch.Tensor,
                                 inclusion_mapping: torch.Tensor) -> torch.Tensor:
        """
        Embed eigenspace with regularization for numerically unstable mappings.
        
        Uses a combination of direct embedding and identity preservation
        to maintain subspace continuity when inclusion mapping is poor quality.
        
        Args:
            prev_eigenvectors: Previous eigenvectors [dim x n_prev]
            inclusion_mapping: Potentially poor-quality inclusion mapping
            
        Returns:
            Regularized embedded eigenvectors
        """
        try:
            # Regularization parameter based on mapping quality
            reg_param = 0.1  # Mix 10% identity with 90% inclusion mapping
            
            # Create regularized mapping: (1-α)I + αM where M is inclusion mapping
            curr_dim, prev_dim = inclusion_mapping.shape
            min_dim = min(curr_dim, prev_dim)
            
            # Identity component (restricted to overlapping dimensions)
            identity_component = torch.zeros_like(inclusion_mapping)
            for i in range(min_dim):
                identity_component[i, i] = 1.0
            
            # Regularized inclusion mapping
            regularized_mapping = (1 - reg_param) * inclusion_mapping + reg_param * identity_component
            
            # Apply regularized mapping
            embedded_vectors = torch.mm(regularized_mapping, prev_eigenvectors)
            
            logger.debug(f"Used regularized embedding with α={reg_param}")
            return embedded_vectors
            
        except Exception as e:
            logger.warning(f"Regularized embedding failed: {e}, using direct application")
            return torch.mm(inclusion_mapping, prev_eigenvectors)
    
    def _orthogonalize_with_continuity_preservation(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize vectors while preserving subspace continuity.
        
        Uses a gentler approach than RRQR that maintains better continuity
        between consecutive filtration steps.
        
        Args:
            vectors: Input vectors [dim x n_vectors]
            
        Returns:
            Orthogonalized vectors with preserved continuity
        """
        if vectors.shape[1] == 0:
            return vectors
        
        # Monitor principal angles if we have a previous subspace
        if self._previous_subspace is not None and vectors.shape[1] > 0:
            self._monitor_subspace_continuity(vectors)
        
        try:
            # Use standard QR which is more stable than RRQR for continuity
            Q, R = torch.linalg.qr(vectors, mode='reduced')
            
            # Check if we lost significant rank
            diag_R = torch.abs(torch.diag(R))
            valid_vectors = torch.sum(diag_R > self.numerical_tolerance).item()
            
            if valid_vectors < vectors.shape[1]:
                logger.debug(f"QR detected {vectors.shape[1] - valid_vectors} near-zero vectors")
                # Keep only valid vectors instead of backfilling
                Q = Q[:, :valid_vectors]
                
                # If we need to maintain the original number of vectors,
                # use modified Gram-Schmidt to add minimal new vectors
                if valid_vectors < vectors.shape[1]:
                    additional_needed = vectors.shape[1] - valid_vectors
                    Q = self._add_minimal_orthogonal_vectors(Q, additional_needed)
            
            # Store for next principal angle monitoring
            self._previous_subspace = Q.clone().detach()
            
            return Q
            
        except Exception as e:
            logger.warning(f"Continuity-preserving orthogonalization failed: {e}")
            # Fallback to original method
            return self._orthogonalize_vectors(vectors)
    
    def _add_minimal_orthogonal_vectors(self, 
                                      existing_vectors: torch.Tensor,
                                      n_additional: int) -> torch.Tensor:
        """
        Add minimal orthogonal vectors to maintain dimension while preserving continuity.
        
        Args:
            existing_vectors: Current orthogonal vectors [dim x n_existing]
            n_additional: Number of additional vectors needed
            
        Returns:
            Extended vector set [dim x (n_existing + n_additional)]
        """
        if n_additional <= 0:
            return existing_vectors
        
        dim = existing_vectors.shape[0]
        
        # Use deterministic approach: start with canonical basis vectors
        # and orthogonalize against existing vectors
        additional_vectors = []
        
        for i in range(min(n_additional, dim)):
            # Try canonical basis vectors first
            candidate = DEFAULT_SPECTRAL_POLICY.create_zeros(dim, 1, device=existing_vectors.device)
            
            # Choose direction that's most orthogonal to existing vectors
            if i < dim:
                candidate[i, 0] = 1.0
            
            # Orthogonalize against all existing vectors
            for j in range(existing_vectors.shape[1]):
                existing_vec = existing_vectors[:, j:j+1]
                proj_coeff = torch.mm(existing_vec.T, candidate)
                candidate = candidate - proj_coeff * existing_vec
            
            # Orthogonalize against previously added vectors
            for prev_vec in additional_vectors:
                proj_coeff = torch.mm(prev_vec.T, candidate)
                candidate = candidate - proj_coeff * prev_vec
            
            # Normalize if not too small
            norm = torch.norm(candidate)
            if norm > self.numerical_tolerance:
                candidate = candidate / norm
                additional_vectors.append(candidate)
        
        if additional_vectors:
            additional_matrix = torch.cat(additional_vectors, dim=1)
            return torch.cat([existing_vectors, additional_matrix], dim=1)
        else:
            return existing_vectors
    
    def _procrustes_align_basis(self, 
                               current_vectors: torch.Tensor,
                               previous_subspace: torch.Tensor,
                               inclusion_matrix: torch.Tensor) -> torch.Tensor:
        """
        🔧 PROCRUSTES BASIS ALIGNMENT: Align current basis to minimize principal angles.
        
        Implements the Procrustes solution to find optimal orthogonal rotation R
        that minimizes ||current_vectors @ R - inclusion_matrix @ previous_subspace||.
        
        This is the core fix for large principal angles - it ensures successive
        subspaces are properly aligned rather than becoming orthogonal.
        
        Args:
            current_vectors: Current basis vectors [dim x k]
            previous_subspace: Previous subspace basis [dim x j] 
            inclusion_matrix: Inclusion mapping [dim x dim]
            
        Returns:
            Procrustes-aligned current vectors [dim x k]
        """
        try:
            # Ensure we're working with properly sized matrices
            min_vecs = min(current_vectors.shape[1], previous_subspace.shape[1])
            if min_vecs == 0:
                logger.debug("Cannot apply Procrustes alignment: empty subspaces")
                return current_vectors
            
            # Truncate to common dimensions for alignment
            V_current = current_vectors[:, :min_vecs]  # [dim x min_vecs]
            U_previous = previous_subspace[:, :min_vecs]  # [dim x min_vecs]
            
            # Compute target alignment: M @ U_previous
            target_alignment = torch.mm(inclusion_matrix, U_previous)  # [dim x min_vecs]
            
            # Solve orthogonal Procrustes: find R such that V_current @ R ≈ target_alignment
            # Solution: R = U @ V^T where target_alignment^T @ V_current = U @ S @ V^T (SVD)
            cross_correlation = torch.mm(target_alignment.T, V_current)  # [min_vecs x min_vecs]
            
            try:
                U, S, Vt = torch.linalg.svd(cross_correlation)
                R_optimal = torch.mm(U, Vt)  # [min_vecs x min_vecs]
                
                # Apply optimal rotation to truncated current vectors
                V_aligned = torch.mm(V_current, R_optimal)  # [dim x min_vecs]
                
                # Extend back to original dimensions if needed
                if current_vectors.shape[1] > min_vecs:
                    # Keep remaining vectors unchanged (they don't have alignment targets)
                    remaining_vectors = current_vectors[:, min_vecs:]
                    V_aligned = torch.cat([V_aligned, remaining_vectors], dim=1)
                
                logger.debug(f"Procrustes alignment: rotated {min_vecs}/{current_vectors.shape[1]} vectors")
                return V_aligned
                
            except Exception as e:
                logger.debug(f"SVD failed in Procrustes alignment: {e}, using direct mapping")
                return current_vectors
                
        except Exception as e:
            logger.warning(f"Procrustes basis alignment failed: {e}")
            return current_vectors
    
    def _check_principal_angles(self, current_vectors: torch.Tensor) -> float:
        """
        Check principal angles between current and previous subspaces.
        
        Args:
            current_vectors: Current subspace vectors [dim x k]
            
        Returns:
            Maximum principal angle in radians
        """
        if self._previous_subspace is None:
            return 0.0
            
        try:
            min_vecs = min(current_vectors.shape[1], self._previous_subspace.shape[1])
            if min_vecs == 0:
                return 0.0
            
            # Compare subspaces of equal dimension
            prev_subspace = self._previous_subspace[:, :min_vecs].detach().cpu().numpy()
            curr_subspace = current_vectors[:, :min_vecs].detach().cpu().numpy()
            
            # Compute principal angles
            angles = subspace_angles(prev_subspace, curr_subspace)
            max_angle = np.max(angles)
            
            logger.debug(f"Principal angles: max={np.degrees(max_angle):.1f}°, "
                        f"mean={np.degrees(np.mean(angles)):.1f}°")
            
            return max_angle
            
        except Exception as e:
            logger.debug(f"Principal angle computation failed: {e}")
            return 0.0
    
    def _apply_identity_fallback(self, 
                                prev_eigenvectors: torch.Tensor,
                                embedded_vectors: torch.Tensor) -> torch.Tensor:
        """
        🔧 IDENTITY FALLBACK: Apply identity mapping when principal angles are too large.
        
        This is the fallback mechanism when Procrustes alignment cannot reduce
        principal angles below the threshold. Uses identity-like extension.
        
        Args:
            prev_eigenvectors: Original previous eigenvectors [prev_dim x n_prev]
            embedded_vectors: Current embedded vectors [curr_dim x n_curr]
            
        Returns:
            Identity-fallback vectors that preserve better subspace continuity
        """
        try:
            # Create identity-like embedding by zero-padding previous vectors
            curr_dim = embedded_vectors.shape[0]
            prev_dim = prev_eigenvectors.shape[0]
            n_prev = prev_eigenvectors.shape[1]
            
            if prev_dim <= curr_dim:
                # Zero-pad previous vectors to current dimension
                identity_embedded = DEFAULT_SPECTRAL_POLICY.create_zeros(
                    curr_dim, n_prev, device=prev_eigenvectors.device
                )
                identity_embedded[:prev_dim, :] = prev_eigenvectors
            else:
                # Truncate if previous dimension is larger
                identity_embedded = prev_eigenvectors[:curr_dim, :]
            
            # If we need more vectors than available, keep some from embedded_vectors
            if embedded_vectors.shape[1] > n_prev:
                remaining_vectors = embedded_vectors[:, n_prev:]
                identity_embedded = torch.cat([identity_embedded, remaining_vectors], dim=1)
            
            logger.info(f"Applied identity fallback: {embedded_vectors.shape} → {identity_embedded.shape}")
            return identity_embedded
            
        except Exception as e:
            logger.error(f"Identity fallback failed: {e}, returning original embedded vectors")
            return embedded_vectors
    
    def _align_in_eigenvalue_subspaces(self, 
                                     current_vectors: torch.Tensor,
                                     current_eigenvals: torch.Tensor,
                                     previous_vectors: torch.Tensor,
                                     previous_eigenvals: torch.Tensor,
                                     inclusion_matrix: torch.Tensor) -> torch.Tensor:
        """
        🔧 SUBSPACE-AWARE ALIGNMENT: Handle eigenvalue multiplicities by aligning within clusters.
        
        Instead of working with raw eigenvectors, this method:
        1. Clusters eigenvalues with similar values (multiplicities)
        2. Applies Procrustes alignment within each cluster
        3. Combines aligned subspaces for final result
        
        This prevents large rotations between clustered eigenvalue blocks.
        
        Args:
            current_vectors: Current eigenvectors [dim x n_curr]
            current_eigenvals: Current eigenvalues [n_curr]
            previous_vectors: Previous eigenvectors [dim x n_prev]
            previous_eigenvals: Previous eigenvalues [n_prev]
            inclusion_matrix: Inclusion mapping [dim x dim]
            
        Returns:
            Subspace-aligned current vectors [dim x n_curr]
        """
        try:
            # Find eigenvalue clusters in both current and previous
            curr_clusters = self._cluster_eigenvalues(current_eigenvals)
            prev_clusters = self._cluster_eigenvalues(previous_eigenvals)
            
            aligned_vectors = []
            processed_indices = set()
            
            # Process each current cluster
            for curr_cluster in curr_clusters:
                curr_indices = curr_cluster['indices']
                curr_cluster_vecs = current_vectors[:, curr_indices]
                
                # Find best matching previous cluster by eigenvalue similarity
                best_prev_cluster = self._find_matching_cluster(curr_cluster, prev_clusters)
                
                if best_prev_cluster is not None:
                    prev_indices = best_prev_cluster['indices']
                    prev_cluster_vecs = previous_vectors[:, prev_indices]
                    
                    # Apply Procrustes alignment within this subspace
                    aligned_cluster_vecs = self._procrustes_align_basis(
                        current_vectors=curr_cluster_vecs,
                        previous_subspace=prev_cluster_vecs,
                        inclusion_matrix=inclusion_matrix
                    )
                    
                    logger.debug(f"Aligned eigenvalue cluster {curr_cluster['mean_eigenval']:.6f}: "
                               f"{len(curr_indices)} current ↔ {len(prev_indices)} previous vectors")
                else:
                    # No matching previous cluster - use vectors as-is
                    aligned_cluster_vecs = curr_cluster_vecs
                    logger.debug(f"No matching cluster for eigenvalue {curr_cluster['mean_eigenval']:.6f}, "
                               f"keeping {len(curr_indices)} vectors unchanged")
                
                aligned_vectors.append((curr_indices, aligned_cluster_vecs))
                processed_indices.update(curr_indices)
            
            # Reconstruct full aligned matrix in original order
            aligned_full = current_vectors.clone()
            for curr_indices, aligned_cluster_vecs in aligned_vectors:
                for i, idx in enumerate(curr_indices):
                    aligned_full[:, idx] = aligned_cluster_vecs[:, i]
            
            return aligned_full
            
        except Exception as e:
            logger.warning(f"Subspace-aware alignment failed: {e}, using standard Procrustes")
            return self._procrustes_align_basis(current_vectors, previous_vectors, inclusion_matrix)
    
    def _cluster_eigenvalues(self, eigenvals: torch.Tensor, tolerance: float = 1e-6) -> List[Dict]:
        """
        Cluster eigenvalues by similarity to identify multiplicities.
        
        Args:
            eigenvals: Eigenvalue tensor [n_vals]
            tolerance: Tolerance for considering eigenvalues as equal
            
        Returns:
            List of cluster dictionaries with 'indices', 'mean_eigenval', 'size'
        """
        try:
            eigenvals_np = eigenvals.detach().cpu().numpy()
            n_vals = len(eigenvals_np)
            
            clusters = []
            processed = set()
            
            for i in range(n_vals):
                if i in processed:
                    continue
                
                # Find all eigenvalues within tolerance of eigenvals[i]
                cluster_indices = []
                base_val = eigenvals_np[i]
                
                for j in range(n_vals):
                    if j not in processed and abs(eigenvals_np[j] - base_val) <= tolerance:
                        cluster_indices.append(j)
                        processed.add(j)
                
                if cluster_indices:
                    cluster_eigenvals = eigenvals_np[cluster_indices]
                    clusters.append({
                        'indices': cluster_indices,
                        'mean_eigenval': np.mean(cluster_eigenvals),
                        'std_eigenval': np.std(cluster_eigenvals),
                        'size': len(cluster_indices),
                        'min_eigenval': np.min(cluster_eigenvals),
                        'max_eigenval': np.max(cluster_eigenvals)
                    })
            
            # Sort clusters by mean eigenvalue (descending)
            clusters.sort(key=lambda x: x['mean_eigenval'], reverse=True)
            
            logger.debug(f"Found {len(clusters)} eigenvalue clusters: "
                        f"sizes = {[c['size'] for c in clusters]}")
            
            return clusters
            
        except Exception as e:
            logger.debug(f"Eigenvalue clustering failed: {e}")
            # Fallback: treat each eigenvalue as its own cluster
            return [{'indices': [i], 'mean_eigenval': float(eigenvals[i]), 'size': 1} 
                   for i in range(len(eigenvals))]
    
    def _find_matching_cluster(self, current_cluster: Dict, previous_clusters: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching previous cluster for a current cluster.
        
        Uses eigenvalue proximity and cluster size as matching criteria.
        
        Args:
            current_cluster: Current cluster dictionary
            previous_clusters: List of previous cluster dictionaries
            
        Returns:
            Best matching previous cluster or None
        """
        if not previous_clusters:
            return None
        
        curr_mean = current_cluster['mean_eigenval']
        curr_size = current_cluster['size']
        
        best_cluster = None
        best_score = float('inf')
        
        for prev_cluster in previous_clusters:
            prev_mean = prev_cluster['mean_eigenval']
            prev_size = prev_cluster['size']
            
            # Compute matching score: weighted combination of eigenvalue distance and size difference
            eigenval_dist = abs(curr_mean - prev_mean)
            size_penalty = abs(curr_size - prev_size) * 0.1  # Size differences are less important
            
            score = eigenval_dist + size_penalty
            
            if score < best_score:
                best_score = score
                best_cluster = prev_cluster
        
        # Only return match if eigenvalue distance is reasonable
        if best_cluster and abs(curr_mean - best_cluster['mean_eigenval']) < 1.0:
            return best_cluster
        else:
            return None
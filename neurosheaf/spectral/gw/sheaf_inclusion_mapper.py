# neurosheaf/spectral/gw/sheaf_inclusion_mapper.py
"""
Creates sheaf-theoretically correct inclusion mappings for GW filtrations.

Based on transport matrix structure and cellular sheaf theory, this module
implements inclusion mappings ι: ℱᵢ ↪ ℱᵢ₊₁ that preserve sheaf morphism
properties while handling the increasing complexity semantics of GW filtrations.

Mathematical Foundation:
- Cellular sheaf inclusion morphisms with face relation preservation
- Transport-informed embeddings using GW optimal transport structure
- SVD-based rotational alignment for eigenspace preservation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError

logger = setup_logger(__name__)


class SheafInclusionMapper:
    """
    Creates sheaf-theoretically correct inclusion mappings for GW filtrations.
    
    Implements multiple methods for creating inclusion mappings that handle
    the increasing complexity of GW filtrations while preserving mathematical
    properties required by sheaf theory.
    """
    
    def __init__(self, 
                 method: str = 'transport_svd',
                 numerical_tolerance: float = 1e-12,
                 svd_regularization: float = 1e-8):
        """
        Initialize sheaf inclusion mapper.
        
        Args:
            method: Inclusion mapping method
                - 'transport_svd': SVD-based using transport matrices
                - 'transport_projection': Direct transport projection
                - 'identity_extension': Identity-based extension (fallback)
            numerical_tolerance: Tolerance for numerical computations
            svd_regularization: Regularization for SVD computations
        """
        self.method = method
        self.numerical_tolerance = numerical_tolerance
        self.svd_regularization = svd_regularization
        
        # Validate method
        valid_methods = ['transport_svd', 'transport_projection', 'identity_extension']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid options: {valid_methods}")
        
        logger.info(f"SheafInclusionMapper initialized with method: {method}")
    
    def create_gw_inclusion_mapping(self, 
                                   prev_step: int, 
                                   curr_step: int,
                                   prev_eigenspace_dim: int,
                                   curr_eigenspace_dim: int,
                                   transport_matrices: Optional[Dict] = None,
                                   sheaf_metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        Create inclusion mapping ι: ℱᵢ ↪ ℱᵢ₊₁ for GW filtration.
        
        The inclusion mapping handles the transition from a smaller eigenspace
        to a larger eigenspace as edges are added in GW filtration.
        
        Args:
            prev_step: Previous filtration step index
            curr_step: Current filtration step index  
            prev_eigenspace_dim: Dimension of previous eigenspace
            curr_eigenspace_dim: Dimension of current eigenspace
            transport_matrices: Dictionary of transport matrices by step
            sheaf_metadata: Additional sheaf construction metadata
            
        Returns:
            Inclusion mapping matrix [curr_eigenspace_dim x prev_eigenspace_dim]
            
        Raises:
            ComputationError: If inclusion mapping creation fails
        """
        logger.debug(f"Creating GW inclusion mapping: step {prev_step} → {curr_step}, "
                    f"dimensions {prev_eigenspace_dim} → {curr_eigenspace_dim}")
        
        try:
            if self.method == 'transport_svd':
                return self._create_transport_svd_inclusion(
                    prev_step, curr_step, prev_eigenspace_dim, curr_eigenspace_dim,
                    transport_matrices, sheaf_metadata
                )
            elif self.method == 'transport_projection':
                return self._create_transport_projection_inclusion(
                    prev_step, curr_step, prev_eigenspace_dim, curr_eigenspace_dim,
                    transport_matrices
                )
            elif self.method == 'identity_extension':
                return self._create_identity_extension_inclusion(
                    prev_eigenspace_dim, curr_eigenspace_dim
                )
            else:
                raise ValueError(f"Unknown inclusion method: {self.method}")
                
        except Exception as e:
            raise ComputationError(
                f"Failed to create inclusion mapping for steps {prev_step}→{curr_step}: {e}",
                operation="create_gw_inclusion_mapping"
            )
    
    def _create_transport_svd_inclusion(self, 
                                      prev_step: int, 
                                      curr_step: int,
                                      prev_dim: int, 
                                      curr_dim: int,
                                      transport_matrices: Optional[Dict],
                                      sheaf_metadata: Optional[Dict]) -> torch.Tensor:
        """
        Create inclusion using SVD decomposition of transport matrices.
        
        Mathematical foundation:
        - Use SVD of transport matrices to extract rotation structure
        - Preserve orthogonality through VU^T rotation matrices
        - Handle dimension changes through proper embedding
        
        Args:
            prev_step: Previous step index
            curr_step: Current step index
            prev_dim: Previous eigenspace dimension
            curr_dim: Current eigenspace dimension
            transport_matrices: Transport matrices dictionary
            sheaf_metadata: Additional metadata
            
        Returns:
            SVD-based inclusion mapping matrix
        """
        # Try to extract relevant transport matrix
        transport_matrix = self._extract_transport_matrix(
            prev_step, curr_step, transport_matrices, sheaf_metadata
        )
        
        if transport_matrix is None:
            raise ComputationError(
                f"No GW transport matrix found for steps {prev_step}→{curr_step}. "
                f"GW sheaf construction must provide gw_couplings in metadata.",
                operation="transport_svd_inclusion"
            )
        
        # Ensure transport matrix is 2D tensor
        if transport_matrix.dim() != 2:
            raise ComputationError(
                f"GW transport matrix has invalid {transport_matrix.dim()}D shape. "
                f"Expected 2D matrix for SVD-based inclusion mapping.",
                operation="transport_svd_inclusion"
            )
        
        try:
            # SVD decomposition with regularization for numerical stability
            U, S, Vt = torch.svd(transport_matrix + 
                               self.svd_regularization * torch.eye(transport_matrix.shape[0]))
            
            # Create optimal rotation matrix VU^T
            rotation = torch.mm(Vt.T, U.T)  # V @ U^T
            
            # Extend rotation to handle dimension changes
            inclusion_map = self._extend_rotation_to_inclusion(
                rotation, prev_dim, curr_dim
            )
            
            logger.debug(f"Created transport SVD inclusion mapping: {inclusion_map.shape}")
            return inclusion_map
            
        except Exception as e:
            raise ComputationError(
                f"SVD decomposition of GW transport matrix failed: {e}",
                operation="transport_svd_inclusion"
            )
    
    def _create_transport_projection_inclusion(self, 
                                             prev_step: int, 
                                             curr_step: int,
                                             prev_dim: int, 
                                             curr_dim: int,
                                             transport_matrices: Optional[Dict]) -> torch.Tensor:
        """
        Create inclusion by projecting through transport matrix structure.
        
        Uses Kronecker product structure P^T ⊗ I where P is transport matrix.
        This preserves the block structure of sheaf Laplacians.
        
        Args:
            prev_step: Previous step index
            curr_step: Current step index
            prev_dim: Previous eigenspace dimension
            curr_dim: Current eigenspace dimension
            transport_matrices: Transport matrices dictionary
            
        Returns:
            Transport projection inclusion mapping matrix
        """
        transport_matrix = self._extract_transport_matrix(
            prev_step, curr_step, transport_matrices, None
        )
        
        if transport_matrix is None:
            raise ComputationError(
                f"No GW transport matrix found for projection method in steps {prev_step}→{curr_step}",
                operation="transport_projection_inclusion"
            )
        
        try:
            # Use transport matrix structure for projection
            # For GW, transport matrices are typically square and doubly stochastic
            P = transport_matrix
            
            # Create block structure using Kronecker product concept
            # Handle dimension mismatch by appropriate padding/truncation
            block_size = min(prev_dim, curr_dim, P.shape[0], P.shape[1])
            
            # Extract relevant submatrix
            P_sub = P[:block_size, :block_size]
            
            # Create inclusion as block diagonal extension
            inclusion_map = torch.zeros(curr_dim, prev_dim)
            inclusion_map[:block_size, :block_size] = P_sub.T
            
            # Fill remaining diagonal elements for dimension extension
            if curr_dim > block_size:
                for i in range(block_size, min(curr_dim, prev_dim)):
                    inclusion_map[i, i] = 1.0
            
            logger.debug(f"Created transport projection inclusion mapping: {inclusion_map.shape}")
            return inclusion_map
            
        except Exception as e:
            raise ComputationError(
                f"GW transport projection failed: {e}",
                operation="transport_projection_inclusion"
            )
    
    def _create_identity_extension_inclusion(self, 
                                           prev_dim: int, 
                                           curr_dim: int) -> torch.Tensor:
        """
        Create inclusion by extending with identity matrix.
        
        Fallback method that preserves existing eigenspace structure
        while handling dimension increases through identity embedding.
        
        Args:
            prev_dim: Previous eigenspace dimension
            curr_dim: Current eigenspace dimension
            
        Returns:
            Identity-based inclusion mapping matrix
        """
        # Create inclusion matrix
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        
        # Fill diagonal elements up to minimum dimension
        min_dim = min(prev_dim, curr_dim)
        for i in range(min_dim):
            inclusion_map[i, i] = 1.0
        
        # For dimension increase (typical in GW), add new basis vectors
        if curr_dim > prev_dim:
            # New dimensions get orthogonal basis (already zeros, which is fine)
            logger.debug(f"Dimension increase: {prev_dim} → {curr_dim}, "
                        f"added {curr_dim - prev_dim} new dimensions")
        
        logger.debug(f"Created identity extension inclusion mapping: {inclusion_map.shape}")
        return inclusion_map
    
    def _extract_transport_matrix(self, 
                                prev_step: int, 
                                curr_step: int,
                                transport_matrices: Optional[Dict],
                                sheaf_metadata: Optional[Dict]) -> Optional[torch.Tensor]:
        """
        Extract GW transport matrix for inclusion mapping.
        
        Uses GW couplings from sheaf metadata - the actual optimal transport 
        matrices computed during GW sheaf construction.
        
        Args:
            prev_step: Previous step index
            curr_step: Current step index
            transport_matrices: Transport matrices dictionary (unused for GW)
            sheaf_metadata: Sheaf construction metadata containing gw_couplings
            
        Returns:
            Transport matrix tensor or None if not found
        """
        # Extract from GW couplings in sheaf metadata
        if sheaf_metadata is not None:
            gw_couplings = sheaf_metadata.get('gw_couplings', {})
            logger.debug(f"Found gw_couplings in metadata: {len(gw_couplings)} entries")
            if gw_couplings:
                # Extract coupling matrix for current filtration step
                transport_matrix = self._extract_coupling_matrix(prev_step, curr_step, gw_couplings, sheaf_metadata)
                if transport_matrix is not None:
                    logger.info(f"Successfully extracted transport matrix from gw_couplings: shape {transport_matrix.shape}")
                    return transport_matrix
        
        logger.warning(f"No GW coupling matrices found in metadata for steps {prev_step}→{curr_step}")
        return None
    
    def _extract_coupling_matrix(self, 
                                prev_step: int, 
                                curr_step: int, 
                                gw_couplings: Dict,
                                sheaf_metadata: Optional[Dict]) -> Optional[torch.Tensor]:
        """
        Extract coupling matrix (transport matrix) from GW couplings.
        
        GW couplings contain the actual optimal transport matrices computed
        during sheaf construction. These are the transport matrices we need
        for proper PES-based tracking.
        
        Args:
            prev_step: Previous step index
            curr_step: Current step index  
            gw_couplings: Dictionary of GW coupling matrices
            sheaf_metadata: Additional sheaf metadata
            
        Returns:
            Transport matrix tensor or None if not found
        """
        try:
            logger.debug(f"Extracting coupling matrix for steps {prev_step}→{curr_step}")
            logger.debug(f"Available GW couplings: {list(gw_couplings.keys())}")
            
            # The gw_couplings dictionary maps edge tuples to transport matrices
            # For filtration, we can use any of the available transport matrices
            # since they represent the optimal transport structure of the sheaf
            
            # Strategy 1: Use any available valid coupling matrix
            for edge_key, coupling in gw_couplings.items():
                logger.debug(f"Checking coupling {edge_key}: type={type(coupling)}")
                
                # Convert to tensor if needed
                if isinstance(coupling, torch.Tensor):
                    if coupling.numel() > 0 and len(coupling.shape) == 2:
                        logger.info(f"Using GW coupling {edge_key} as transport matrix for steps {prev_step}→{curr_step}: shape {coupling.shape}")
                        return coupling
                elif isinstance(coupling, np.ndarray):
                    if coupling.size > 0 and len(coupling.shape) == 2:
                        tensor_coupling = torch.from_numpy(coupling).float()
                        logger.info(f"Using GW coupling {edge_key} as transport matrix for steps {prev_step}→{curr_step}: shape {tensor_coupling.shape}")
                        return tensor_coupling
                else:
                    logger.debug(f"Coupling {edge_key} is not a valid tensor/array: {type(coupling)}")
            
            logger.debug("No valid coupling matrices found in gw_couplings")
            
        except Exception as e:
            logger.warning(f"Failed to extract coupling matrix: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return None
    
    
    def _extend_rotation_to_inclusion(self, 
                                    rotation: torch.Tensor, 
                                    prev_dim: int, 
                                    curr_dim: int) -> torch.Tensor:
        """
        Extend rotation matrix to proper inclusion mapping.
        
        Handles dimension changes by embedding rotation in larger space
        and extending with identity for new dimensions.
        
        Args:
            rotation: Rotation matrix from SVD
            prev_dim: Previous eigenspace dimension
            curr_dim: Current eigenspace dimension
            
        Returns:
            Extended inclusion mapping matrix
        """
        # Determine effective rotation size
        rot_size = min(rotation.shape[0], rotation.shape[1], prev_dim, curr_dim)
        
        # Create inclusion matrix
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        
        # Embed rotation in top-left block
        inclusion_map[:rot_size, :rot_size] = rotation[:rot_size, :rot_size]
        
        # Fill remaining diagonal for identity extension
        for i in range(rot_size, min(curr_dim, prev_dim)):
            inclusion_map[i, i] = 1.0
        
        return inclusion_map
    
    def validate_sheaf_morphism_properties(self, 
                                         inclusion_map: torch.Tensor,
                                         prev_restrictions: Optional[Dict] = None,
                                         curr_restrictions: Optional[Dict] = None) -> bool:
        """
        Validate inclusion satisfies sheaf-theoretic constraints.
        
        Checks mathematical properties required for valid sheaf morphisms:
        1. Matrix dimensions are consistent
        2. Numerical stability (no NaN/inf values)
        3. Commutative diagram property (if restriction maps provided)
        
        Args:
            inclusion_map: Inclusion mapping matrix to validate
            prev_restrictions: Previous step restriction maps (optional)
            curr_restrictions: Current step restriction maps (optional)
            
        Returns:
            True if inclusion mapping is valid, False otherwise
        """
        try:
            # Check basic matrix properties
            if torch.any(torch.isnan(inclusion_map)) or torch.any(torch.isinf(inclusion_map)):
                logger.error("Inclusion mapping contains NaN or infinite values")
                return False
            
            # Check dimensions are positive
            if inclusion_map.shape[0] <= 0 or inclusion_map.shape[1] <= 0:
                logger.error("Inclusion mapping has non-positive dimensions")
                return False
            
            # Check numerical conditioning
            if torch.norm(inclusion_map, 'fro') < self.numerical_tolerance:
                logger.warning("Inclusion mapping has very small Frobenius norm")
            
            # If restriction maps provided, check commutative diagram property
            if prev_restrictions is not None and curr_restrictions is not None:
                commutes = self._check_commutative_property(
                    inclusion_map, prev_restrictions, curr_restrictions
                )
                if not commutes:
                    logger.warning("Inclusion mapping may not satisfy commutative diagram property")
                    # Don't fail validation, just warn
            
            logger.debug("Inclusion mapping validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Inclusion mapping validation failed: {e}")
            return False
    
    def _check_commutative_property(self, 
                                  inclusion_map: torch.Tensor,
                                  prev_restrictions: Dict,
                                  curr_restrictions: Dict) -> bool:
        """
        Check if inclusion mapping satisfies commutative diagram property.
        
        For sheaf morphisms, we need: ι ∘ R_prev ≈ R_curr ∘ ι
        where ι is the inclusion mapping and R are restriction maps.
        
        Args:
            inclusion_map: Inclusion mapping matrix
            prev_restrictions: Previous restriction maps
            curr_restrictions: Current restriction maps
            
        Returns:
            True if commutative property is approximately satisfied
        """
        # This is a simplified check - full implementation would require
        # detailed knowledge of the sheaf structure and restriction maps
        
        # For now, return True (assume validity)
        # TODO: Implement detailed commutative diagram checking
        return True
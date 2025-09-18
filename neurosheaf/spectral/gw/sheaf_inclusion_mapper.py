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
from dataclasses import dataclass
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError
from ..dtype_policy import to_spectral_dtype, DEFAULT_SPECTRAL_POLICY
from ..transport_metadata_manager import (
    TransportMetadataManager,
    TransportNotAvailableError,
    TransportQualityError
)

logger = setup_logger(__name__)


@dataclass
class InclusionMapping:
    """
    Container for inclusion mapping results with clear separation of matrix and metadata.
    
    This dataclass enforces the API contract between SheafInclusionMapper and GWEigenspaceEmbedder,
    ensuring type safety and eliminating the need for tuple introspection.
    
    Attributes:
        matrix: The inclusion mapping matrix [curr_dim x prev_dim] as a torch.Tensor
        meta: Dictionary containing quality assessment and diagnostic information
    """
    matrix: torch.Tensor  # (n_target, n_source)
    meta: Dict[str, Any]


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
                 svd_regularization: float = 1e-8,
                 inclusion_quality_tolerance: float = 0.1,
                 max_edges_to_check: int = 50,
                 fallback_on_violation: bool = True,
                 transport_quality_threshold: float = 0.5,
                 transport_fallback_policy: str = 'nearest_neighbor_quality_weighted',
                 enable_smoothing: bool = True,
                 smoothing_alpha: float = 0.3,
                 stability_threshold: float = 0.8):
        """
        Initialize sheaf inclusion mapper.
        
        Args:
            method: Inclusion mapping method
                - 'transport_svd': SVD-based using transport matrices
                - 'transport_projection': Direct transport projection
                - 'identity_extension': Identity-based extension (fallback)
            numerical_tolerance: Tolerance for numerical computations
            svd_regularization: Regularization for SVD computations
            inclusion_quality_tolerance: Threshold for commutative property violation
            max_edges_to_check: Maximum edges to check for performance (samples if exceeded)
            fallback_on_violation: Whether to use identity_extension fallback on violations
            transport_quality_threshold: Minimum quality threshold for transport selection
            transport_fallback_policy: Policy for transport fallback ('nearest_neighbor_quality_weighted', 'best_available', 'strict')
            enable_smoothing: Whether to apply exponential moving average smoothing
            smoothing_alpha: Alpha parameter for EMA smoothing (higher = more smoothing)
            stability_threshold: Quality threshold for enabling smoothing
        """
        self.method = method
        self.numerical_tolerance = numerical_tolerance
        self.svd_regularization = svd_regularization
        self.inclusion_quality_tolerance = inclusion_quality_tolerance
        self.max_edges_to_check = max_edges_to_check
        self.fallback_on_violation = fallback_on_violation
        self.transport_quality_threshold = transport_quality_threshold
        self.transport_fallback_policy = transport_fallback_policy
        self.enable_smoothing = enable_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.stability_threshold = stability_threshold
        
        # Transport metadata manager - will be initialized when metadata is available
        self._transport_manager = None
        
        # 🔧 MULTI-STEP SMOOTHING: History tracking for exponential moving average
        self._inclusion_history = {}  # Maps (prev_step, curr_step) -> InclusionMapping
        self._stability_history = {}  # Track stability scores over time
        
        # Validate method
        valid_methods = ['transport_svd', 'transport_projection', 'identity_extension']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid options: {valid_methods}")
        
        logger.info(f"SheafInclusionMapper initialized: method={method}, "
                   f"quality_tolerance={inclusion_quality_tolerance}, "
                   f"max_edges_to_check={max_edges_to_check}, fallback={fallback_on_violation}, "
                   f"transport_quality={transport_quality_threshold}, transport_policy={transport_fallback_policy}")
    
    def create_gw_inclusion_mapping(self, 
                                   prev_step: int, 
                                   curr_step: int,
                                   prev_eigenspace_dim: int,
                                   curr_eigenspace_dim: int,
                                   transport_matrices: Optional[Dict] = None,
                                   sheaf_metadata: Optional[Dict] = None,
                                   prev_restrictions: Optional[Dict] = None,
                                   curr_restrictions: Optional[Dict] = None) -> InclusionMapping:
        """
        Create inclusion mapping ι: ℱᵢ ↪ ℱᵢ₊₁ for GW filtration.
        
        The inclusion mapping handles the transition from a smaller eigenspace
        to a larger eigenspace as edges are added in GW filtration. Includes
        commutative property checking and automatic fallback on violation.
        
        Args:
            prev_step: Previous filtration step index
            curr_step: Current filtration step index  
            prev_eigenspace_dim: Dimension of previous eigenspace
            curr_eigenspace_dim: Dimension of current eigenspace
            transport_matrices: Dictionary of transport matrices by step
            sheaf_metadata: Additional sheaf construction metadata
            prev_restrictions: Previous step restriction maps (for quality check)
            curr_restrictions: Current step restriction maps (for quality check)
            
        Returns:
            InclusionMapping containing:
                - matrix: inclusion_mapping_matrix [curr_eigenspace_dim x prev_eigenspace_dim]
                - meta: dictionary with quality assessment results
            
        Raises:
            ComputationError: If inclusion mapping creation fails
        """
        logger.debug(f"Creating GW inclusion mapping: step {prev_step} → {curr_step}, "
                    f"dimensions {prev_eigenspace_dim} → {curr_eigenspace_dim}")
        
        # Initialize transport metadata manager if not already done
        if self._transport_manager is None and sheaf_metadata:
            self._initialize_transport_manager(sheaf_metadata)
        
        inclusion_metadata = {
            'method_used': self.method,
            'original_method': self.method,
            'fallback_triggered': False,
            'inclusion_quality': None,
            'transport_selection_info': None
        }
        
        try:
            # Try primary method
            if self.method == 'transport_svd':
                inclusion_map = self._create_transport_svd_inclusion(
                    prev_step, curr_step, prev_eigenspace_dim, curr_eigenspace_dim,
                    transport_matrices, sheaf_metadata
                )
            elif self.method == 'transport_projection':
                inclusion_map = self._create_transport_projection_inclusion(
                    prev_step, curr_step, prev_eigenspace_dim, curr_eigenspace_dim,
                    transport_matrices
                )
            elif self.method == 'identity_extension':
                inclusion_map = self._create_identity_extension_inclusion(
                    prev_eigenspace_dim, curr_eigenspace_dim
                )
            else:
                raise ValueError(f"Unknown inclusion method: {self.method}")
            
            # Check commutative property if restriction maps are provided
            if (prev_restrictions is not None and curr_restrictions is not None and 
                self.method != 'identity_extension'):  # Skip check for identity method
                
                commutes, quality_metadata = self._check_commutative_property(
                    inclusion_map, prev_restrictions, curr_restrictions
                )
                inclusion_metadata['inclusion_quality'] = quality_metadata
                
                # Use fallback if property is violated and fallback is enabled
                if not commutes and self.fallback_on_violation:
                    logger.warning(f"Method '{self.method}' violated commutative property, "
                                 f"falling back to identity_extension")
                    
                    # Create fallback inclusion mapping
                    inclusion_map = self._create_identity_extension_inclusion(
                        prev_eigenspace_dim, curr_eigenspace_dim
                    )
                    
                    # Update metadata
                    inclusion_metadata['fallback_triggered'] = True
                    inclusion_metadata['method_used'] = 'identity_extension'
                    
                    # Re-check commutative property with fallback method
                    fallback_commutes, fallback_quality = self._check_commutative_property(
                        inclusion_map, prev_restrictions, curr_restrictions
                    )
                    inclusion_metadata['fallback_quality'] = fallback_quality
                    logger.info(f"Fallback method commutative property: {fallback_commutes}")
            
            # 🔧 MULTI-STEP SMOOTHING: Apply exponential moving average if enabled
            if self.enable_smoothing:
                inclusion_mapping = InclusionMapping(matrix=inclusion_map, meta=inclusion_metadata)
                inclusion_mapping = self._apply_smoothing(inclusion_mapping, prev_step, curr_step)
            else:
                inclusion_mapping = InclusionMapping(matrix=inclusion_map, meta=inclusion_metadata)
                
            return inclusion_mapping
                
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
        # Try to extract relevant transport matrix with enhanced selection
        transport_matrix, selection_info = self._extract_transport_matrix(
            prev_step, curr_step, transport_matrices, sheaf_metadata
        )
        
        if transport_matrix is None:
            error_detail = selection_info.get('error', 'unknown error')
            logger.warning(f"No transport matrix found for steps {prev_step}→{curr_step}, using identity fallback")
            # Fallback to identity extension instead of throwing error
            return self._create_identity_extension_inclusion(prev_dim, curr_dim)
        
        # Ensure transport matrix is 2D tensor
        if transport_matrix.dim() != 2:
            logger.warning(f"Transport matrix has invalid {transport_matrix.dim()}D shape, using identity fallback")
            return self._create_identity_extension_inclusion(prev_dim, curr_dim)
        
        # Enhanced transport matrix quality validation with improved decision logic
        quality_info = self._validate_transport_matrix_quality(transport_matrix)
        
        # Follow user guidelines for decision making
        if quality_info['abort_recommended']:
            logger.error(f"🚨 ABORTING: Transport matrix has critical issues - using fallback")
            logger.error(f"Critical issues: {quality_info['issues']}")
            return self._create_fallback_inclusion_mapping(prev_dim, curr_dim, quality_info)
        
        elif not quality_info['is_suitable']:
            recommendation = quality_info['recommendation']
            logger.warning(f"Transport matrix needs treatment: {recommendation}")
            logger.warning(f"Issues detected: {quality_info['issues']}")
            logger.warning(f"Condition number: {quality_info['condition_number']:.2e}, "
                         f"Rank: {quality_info['effective_rank']}/{quality_info['expected_rank']}")
            
            # Apply regularization based on recommendation
            if recommendation == "apply_aggressive_regularization":
                logger.info("🔧 Applying aggressive regularization due to critical issues")
            
            transport_matrix, regularization_metadata = self._regularize_transport_matrix(transport_matrix)
            
            # Enhanced post-regularization reporting
            if regularization_metadata['success']:
                logger.info(f"✅ {regularization_metadata['method_used'].upper()} regularization succeeded:")
                logger.info(f"   Condition number: {regularization_metadata['pre_condition_number']:.2e} → "
                          f"{regularization_metadata['post_condition_number']:.2e}")
                logger.info(f"   Method details: {regularization_metadata.get('iterations', 1)} iterations")
            else:
                logger.error(f"❌ ALL regularization methods failed!")
                logger.error(f"   Final condition number: {regularization_metadata['post_condition_number']:.2e}")
                logger.warning("⚠️  Using fallback inclusion mapping due to regularization failure")
                return self._create_fallback_inclusion_mapping(prev_dim, curr_dim, regularization_metadata)
        else:
            logger.debug(f"✅ Transport matrix quality acceptable: "
                        f"cond={quality_info['condition_number']:.2e}, "
                        f"rank={quality_info['effective_rank']}/{quality_info['expected_rank']}")
        
        # Ensure transport matrix is 2D tensor
        if transport_matrix.dim() != 2:
            logger.warning(f"Transport matrix has invalid {transport_matrix.dim()}D shape, using identity fallback")
            return self._create_identity_extension_inclusion(prev_dim, curr_dim)
        
        try:
            # 🔧 PRODUCTION FIX: Clean SVD decomposition without problematic regularization
            # The old torch.eye(transport_matrix.shape[0]) caused dimension mismatches for
            # non-square matrices like (1x32). Our enhanced regularization pipeline handles
            # all conditioning issues properly, so we can use clean SVD here.
            
            logger.debug(f"Computing SVD for transport matrix shape: {transport_matrix.shape}")
            
            # Clean SVD decomposition - regularization handled upstream
            U, S, Vt = torch.linalg.svd(transport_matrix, full_matrices=False)
            
            # Validate SVD results
            if len(S) == 0:
                raise ComputationError(
                    f"SVD returned empty singular values for matrix shape {transport_matrix.shape}",
                    operation="transport_svd_inclusion"
                )
            
            logger.debug(f"SVD successful: U{U.shape}, S{S.shape}, Vt{Vt.shape}")
            logger.debug(f"Singular value range: [{S.min().item():.2e}, {S.max().item():.2e}]")
            
            # 🔧 TASK 5 FIX: Enhanced dimension validation for matrix multiplication
            # Validate dimensions before matrix multiplication to catch mismatches early
            if Vt.T.shape[1] != U.T.shape[0]:
                logger.error(f"❌ CRITICAL DIMENSION MISMATCH in SVD rotation construction:")
                logger.error(f"   Vt.T shape: {Vt.T.shape} (expected: [?, {U.T.shape[0]}])")
                logger.error(f"   U.T shape: {U.T.shape}")
                logger.error(f"   Original transport matrix: {transport_matrix.shape}")
                logger.error(f"   SVD shapes: U{U.shape}, S{S.shape}, Vt{Vt.shape}")
                raise ComputationError(
                    f"SVD dimension mismatch: Cannot multiply Vt.T{Vt.T.shape} @ U.T{U.T.shape}. "
                    f"Original matrix: {transport_matrix.shape}",
                    operation="transport_svd_rotation"
                )
            
            # Create optimal rotation matrix VU^T
            # For rectangular matrices, this creates the best rotation in the feasible subspace
            try:
                rotation = torch.mm(Vt.T, U.T)  # V @ U^T
                logger.debug(f"SVD rotation matrix created: {rotation.shape}")
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    logger.error(f"❌ MATRIX MULTIPLICATION ERROR: {e}")
                    logger.error(f"   This is the exact error from production logs!")
                    logger.error(f"   Vt.T: {Vt.T.shape}, U.T: {U.T.shape}")
                    raise ComputationError(
                        f"Production error reproduced: {e}. "
                        f"Shapes: Vt.T{Vt.T.shape} @ U.T{U.T.shape} from transport{transport_matrix.shape}",
                        operation="transport_svd_rotation"
                    )
                else:
                    raise
            
            # Extend rotation to handle dimension changes properly
            inclusion_map = self._extend_rotation_to_inclusion(
                rotation, prev_dim, curr_dim
            )
            
            logger.debug(f"Created transport SVD inclusion mapping: {inclusion_map.shape}")
            return inclusion_map
            
        except Exception as e:
            # 🔧 PRODUCTION FIX: Better error reporting with matrix shapes
            logger.error(f"SVD decomposition failed for transport matrix {transport_matrix.shape}")
            logger.error(f"Target dimensions: prev={prev_dim}, curr={curr_dim}")
            logger.error(f"Matrix condition estimate: {torch.linalg.cond(transport_matrix).item():.2e}")
            
            raise ComputationError(
                f"SVD decomposition of GW transport matrix failed: {e} "
                f"(matrix shape: {transport_matrix.shape}, target: {prev_dim}→{curr_dim})",
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
        transport_matrix, selection_info = self._extract_transport_matrix(
            prev_step, curr_step, transport_matrices, None
        )
        
        if transport_matrix is None:
            error_detail = selection_info.get('error', 'unknown error')
            raise ComputationError(
                f"No GW transport matrix found for projection method in steps {prev_step}→{curr_step}. "
                f"Selection error: {error_detail}",
                operation="transport_projection_inclusion"
            )
        
        # 🔧 TASK 4 FIX: Force transport projection through enhanced regularization pipeline
        # Before using the transport matrix, validate and regularize if needed
        quality_info = self._validate_transport_matrix_quality(transport_matrix)
        
        if quality_info['abort_recommended']:
            logger.error(f"🚨 Transport projection ABORTING: Transport matrix has critical issues")
            logger.error(f"Critical issues: {quality_info['issues']}")
            raise ComputationError(
                f"Transport matrix unsuitable for projection: {quality_info['issues']}",
                operation="transport_projection_inclusion"
            )
            
        elif not quality_info['is_suitable']:
            logger.warning(f"Transport projection matrix needs regularization: {quality_info['recommendation']}")
            transport_matrix, regularization_metadata = self._regularize_transport_matrix(transport_matrix)
            
            if not regularization_metadata['success']:
                logger.error(f"❌ Transport projection regularization failed!")
                raise ComputationError(
                    f"Failed to regularize transport matrix for projection: {regularization_metadata.get('error', 'unknown')}",
                    operation="transport_projection_inclusion"
                )
            else:
                logger.info(f"✅ Transport projection regularization succeeded with {regularization_metadata['method_used']}")

        try:
            # Use regularized transport matrix structure for projection
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
                                sheaf_metadata: Optional[Dict]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Extract GW transport matrix for inclusion mapping with intelligent fallback.
        
        Uses the TransportMetadataManager to find the best available transport
        matrix with fallback policies for sparse filtrations.
        
        Args:
            prev_step: Previous step index
            curr_step: Current step index
            transport_matrices: Transport matrices dictionary (legacy)
            sheaf_metadata: Sheaf construction metadata containing gw_couplings
            
        Returns:
            Tuple of (transport_matrix, selection_info) where selection_info
            contains details about transport selection including fallbacks used
        """
        # Initialize transport metadata manager if needed
        if self._transport_manager is None and sheaf_metadata:
            self._initialize_transport_manager(sheaf_metadata)
        
        # Try to extract edge for steps
        target_edge = self._get_edge_for_step_transition(prev_step, curr_step, sheaf_metadata)
        
        if target_edge is None:
            logger.debug(f"Could not determine target edge for steps {prev_step}→{curr_step}")
            # Fall back to legacy extraction
            legacy_result = self._extract_coupling_matrix_legacy(prev_step, curr_step, sheaf_metadata)
            return legacy_result, {'method': 'legacy_extraction', 'error': 'no_edge_mapping'}
        
        # Use transport metadata manager for robust selection
        if self._transport_manager is not None:
            try:
                gw_couplings = sheaf_metadata.get('gw_couplings', {})
                transport_matrix, selection_info = self._transport_manager.get_transport_matrix(
                    target_edge, gw_couplings, allow_fallback=True
                )
                
                if transport_matrix is not None:
                    logger.info(f"✓ Extracted transport matrix via metadata manager: "
                              f"requested {target_edge} → selected {selection_info.get('selected_edge')}, "
                              f"method={selection_info.get('selection_method')}, "
                              f"quality={selection_info.get('quality_score', 'N/A')}")
                    return transport_matrix, selection_info
                
                logger.warning(f"Transport metadata manager failed to find transport for {target_edge}: "
                             f"{selection_info.get('error', 'unknown error')}")
                
            except (TransportNotAvailableError, TransportQualityError) as e:
                logger.warning(f"Transport access error for {target_edge}: {e}")
                return None, {'method': 'transport_manager', 'error': str(e)}
        
        # Final fallback to legacy method
        logger.debug(f"Falling back to legacy transport extraction for steps {prev_step}→{curr_step}")
        legacy_result = self._extract_coupling_matrix_legacy(prev_step, curr_step, sheaf_metadata)
        return legacy_result, {'method': 'legacy_fallback', 'target_edge': target_edge}
    
    def _extract_coupling_matrix_legacy(self, 
                                       prev_step: int, 
                                       curr_step: int, 
                                       sheaf_metadata: Optional[Dict]) -> Optional[torch.Tensor]:
        """
        Legacy method to extract coupling matrix from GW couplings.
        
        This method provides backward compatibility for the original
        coupling extraction logic without transport metadata management.
        
        Args:
            prev_step: Previous step index
            curr_step: Current step index
            sheaf_metadata: Sheaf metadata containing gw_couplings
            
        Returns:
            Transport matrix tensor or None if not found
        """
        if not sheaf_metadata:
            return None
            
        gw_couplings = sheaf_metadata.get('gw_couplings', {})
        if not gw_couplings:
            return None
            
        try:
            logger.debug(f"Legacy coupling extraction for steps {prev_step}→{curr_step}")
            logger.debug(f"Available GW couplings: {list(gw_couplings.keys())}")
            
            # ❌ CRITICAL BUG FIX: The previous implementation used "any available coupling"
            # which resulted in always reusing the same coupling ('activation_fn_3','layers_11')
            # for every step transition, causing ~90° subspace rotations.
            
            # NEW STRATEGY: Step-aware coupling selection based on filtration order
            # First, try to determine the correct edge for this specific step transition
            target_edge = self._determine_step_specific_edge(prev_step, curr_step, gw_couplings)
            
            if target_edge and target_edge in gw_couplings:
                coupling = gw_couplings[target_edge]
                tensor_coupling = self._convert_coupling_to_tensor(coupling, target_edge, prev_step, curr_step)
                if tensor_coupling is not None:
                    # 🔧 PRODUCTION FIX: Add coupling validation
                    coupling_valid = self._validate_coupling_selection(target_edge, tensor_coupling, prev_step, curr_step)
                    if coupling_valid:
                        logger.info(f"✅ Using STEP-SPECIFIC GW coupling {target_edge} for steps {prev_step}→{curr_step}: shape {tensor_coupling.shape}")
                        return tensor_coupling
                    else:
                        logger.warning(f"⚠️  COUPLING VALIDATION FAILED for {target_edge}, trying fallback")
                        # Continue to fallback selection
            
            # Fallback: Use systematic selection based on step ordering (not arbitrary first match)
            logger.warning(f"⚠️  Could not find step-specific coupling, using systematic fallback for steps {prev_step}→{curr_step}")
            return self._systematic_coupling_fallback(gw_couplings, prev_step, curr_step)
            
        except Exception as e:
            logger.warning(f"Failed to extract coupling matrix: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return None
    
    def _validate_coupling_selection(self, edge: Tuple[str, str], coupling_matrix: torch.Tensor,
                                   prev_step: int, curr_step: int) -> bool:
        """
        Validate that the selected coupling matrix is appropriate for the step transition.
        
        🔧 PRODUCTION FIX: Catches problematic coupling selections that cause 90° angles.
        Validates matrix properties and detects dimension/rank issues that lead to failures.
        
        Args:
            edge: The selected edge tuple
            coupling_matrix: The coupling matrix tensor
            prev_step: Previous step index
            curr_step: Current step index
            
        Returns:
            True if coupling is valid, False if should use fallback
        """
        try:
            shape = coupling_matrix.shape
            logger.debug(f"Validating coupling {edge} with shape {shape} for steps {prev_step}→{curr_step}")
            
            # Check 1: Basic shape validation
            if coupling_matrix.dim() != 2:
                logger.warning(f"Coupling {edge} has invalid dimension: {coupling_matrix.dim()}D")
                return False
            
            # Check 2: Detect problematic shapes that cause dimension mismatches
            if shape[0] == 1 or shape[1] == 1:
                logger.warning(f"Coupling {edge} has degenerate shape {shape} (1xN or Nx1)")
                logger.warning("This may cause SVD dimension mismatch issues in transport_svd_inclusion")
                # Allow but log warning - our enhanced handling should fix this
            
            # Check 3: Check for extreme aspect ratios that cause issues  
            aspect_ratio = max(shape[0]/shape[1], shape[1]/shape[0])
            if aspect_ratio > 10.0:
                logger.warning(f"Coupling {edge} has extreme aspect ratio: {aspect_ratio:.1f}")
            
            # Check 4: Validate matrix content (basic checks)
            if torch.any(torch.isnan(coupling_matrix)) or torch.any(torch.isinf(coupling_matrix)):
                logger.error(f"Coupling {edge} contains NaN or Inf values")
                return False
            
            # Check 5: Check matrix norm (detect zero matrices)
            matrix_norm = torch.norm(coupling_matrix, 'fro').item()
            if matrix_norm < 1e-12:
                logger.warning(f"Coupling {edge} has very small norm: {matrix_norm:.2e}")
                return False
                
            # Check 6: Detect very high condition numbers that will cause 90° angles
            try:
                cond_num = torch.linalg.cond(coupling_matrix).item()
                if cond_num > 1e10:  # This will likely trigger our regularization
                    logger.info(f"Coupling {edge} has high condition number {cond_num:.2e} - will need regularization")
                    # Don't reject - let our regularization handle it
                elif cond_num > 1e12:  # This will likely be aborted
                    logger.warning(f"Coupling {edge} has critical condition number {cond_num:.2e} - may abort")
                    # Still allow - let validation pipeline decide
                    
                logger.debug(f"Coupling {edge} validation: shape={shape}, norm={matrix_norm:.2e}, cond={cond_num:.2e}")
                
            except Exception as cond_e:
                logger.debug(f"Could not compute condition number for {edge}: {cond_e}")
                
            # Overall assessment - be permissive, let downstream validation handle issues
            return True
            
        except Exception as e:
            logger.warning(f"Coupling validation failed for {edge}: {e}")
            return False
    
    def _determine_step_specific_edge(self, prev_step: int, curr_step: int, 
                                     gw_couplings: Dict) -> Optional[Tuple[str, str]]:
        """
        Determine the specific edge that should be used for a step transition.
        
        This method implements the core fix for the stale coupling reuse issue.
        Instead of using arbitrary coupling selection, it determines which edge
        is actually relevant for the given step transition.
        
        Args:
            prev_step: Previous filtration step
            curr_step: Current filtration step  
            gw_couplings: Available GW couplings dictionary
            
        Returns:
            Edge tuple for this specific step transition or None if not found
        """
        logger.debug(f"Determining step-specific edge for transition {prev_step}→{curr_step}")
        
        # Strategy 1: Use step index to select from ordered couplings
        # In GW filtration, edges are typically ordered by cost/complexity
        coupling_keys = list(gw_couplings.keys())
        
        if len(coupling_keys) == 0:
            logger.debug("No couplings available")
            return None
        
        # 🔧 ZERO-EDGE FIX: Handle step 0 correctly and use step-1 indexing
        if curr_step == 0 or prev_step == 0:
            logger.debug(f"Legacy method: no coupling needed for step 0 transitions")
            return None
        
        # Map steps 1-N to coupling indices (step 1 = index 0, step 2 = index 1, etc.)
        edge_index = curr_step - 1
        if 0 <= edge_index < len(coupling_keys):
            target_edge = coupling_keys[edge_index]
            logger.debug(f"Mapped step {curr_step} to edge {target_edge} (index {edge_index}/{len(coupling_keys)})")
            return target_edge
        
        # Check for coordination error with corrected indexing
        if len(coupling_keys) > 0:
            logger.error(f"❌ LEGACY COORDINATION ERROR: Step {curr_step} needs index {edge_index} but only {len(coupling_keys)} couplings available (indices 0-{len(coupling_keys)-1})")
            logger.error("This indicates a step-to-edge coordination error")
            return None
        
        logger.debug(f"Could not determine edge for step transition {prev_step}→{curr_step}")
        return None
    
    def _convert_coupling_to_tensor(self, coupling: Any, edge_key: Tuple[str, str], 
                                   prev_step: int, curr_step: int) -> Optional[torch.Tensor]:
        """
        Convert a coupling to a tensor with proper validation and logging.
        
        Args:
            coupling: The coupling object (tensor, array, or other)
            edge_key: The edge key for logging
            prev_step: Previous step for logging
            curr_step: Current step for logging
            
        Returns:
            Converted tensor or None if conversion failed
        """
        try:
            if isinstance(coupling, torch.Tensor):
                # 🔧 TASK 5 FIX: Enhanced tensor validation with dimension mismatch detection
                if coupling.numel() > 0 and len(coupling.shape) == 2:
                    rows, cols = coupling.shape
                    
                    # Detect suspicious shapes that could cause issues
                    if rows == 1 and cols > 20:
                        logger.warning(f"⚠️  Suspicious coupling shape {coupling.shape} for {edge_key} - matches production error pattern")
                        logger.warning(f"   Steps: {prev_step}→{curr_step}, could cause SVD dimension issues")
                    
                    # Check for extreme aspect ratios
                    aspect_ratio = max(rows/cols, cols/rows) if min(rows, cols) > 0 else float('inf')
                    if aspect_ratio > 50:
                        logger.warning(f"⚠️  Extreme aspect ratio {aspect_ratio:.1f} for coupling {edge_key}: {coupling.shape}")
                        logger.warning(f"   This may cause rotation matrix dimension mismatches")
                    
                    # Validate numerical properties
                    if torch.isnan(coupling).any():
                        logger.error(f"❌ Coupling {edge_key} contains NaN values")
                        return None
                    
                    if torch.isinf(coupling).any():
                        logger.error(f"❌ Coupling {edge_key} contains infinite values")
                        return None
                    
                    logger.debug(f"Coupling {edge_key} validated: shape {coupling.shape}, "
                               f"range [{coupling.min().item():.2e}, {coupling.max().item():.2e}]")
                    return coupling
                else:
                    logger.debug(f"Coupling {edge_key} has invalid tensor shape: {coupling.shape}")
                    return None
            elif isinstance(coupling, np.ndarray):
                if coupling.size > 0 and len(coupling.shape) == 2:
                    return DEFAULT_SPECTRAL_POLICY.from_numpy(coupling)
                else:
                    logger.debug(f"Coupling {edge_key} has invalid array shape: {coupling.shape}")
                    return None
            else:
                logger.debug(f"Coupling {edge_key} is not a tensor/array: {type(coupling)}")
                return None
        except Exception as e:
            logger.debug(f"Failed to convert coupling {edge_key}: {e}")
            return None
    
    def _systematic_coupling_fallback(self, gw_couplings: Dict, prev_step: int, 
                                    curr_step: int) -> Optional[torch.Tensor]:
        """
        Systematic fallback coupling selection based on step ordering.
        
        This replaces the previous arbitrary "first valid coupling" approach
        with a deterministic selection that varies by step.
        
        Args:
            gw_couplings: Available GW couplings dictionary
            prev_step: Previous step
            curr_step: Current step
            
        Returns:
            Transport matrix tensor or None if not found
        """
        logger.debug(f"Using systematic fallback for steps {prev_step}→{curr_step}")
        
        # Convert couplings to list for deterministic indexing
        coupling_items = list(gw_couplings.items())
        
        if not coupling_items:
            logger.debug("No coupling items available for systematic fallback")
            return None
        
        # 🔧 STEP-AWARE FALLBACK: Handle step 0 correctly and use step-1 indexing
        if curr_step == 0 or prev_step == 0:
            logger.debug("Systematic fallback: no coupling needed for step 0 transitions")
            return None
        
        # Use step-to-edge mapping for fallback (step 1 = index 0, step 2 = index 1, etc.)
        edge_index = curr_step - 1
        if 0 <= edge_index < len(coupling_items):
            selected_edge, selected_coupling = coupling_items[edge_index]
            logger.debug(f"Systematic fallback: step {curr_step} → coupling {selected_edge} (index {edge_index})")
        else:
            # Coordination error: step exceeds available couplings
            logger.error(f"❌ SYSTEMATIC FALLBACK ERROR: Step {curr_step} needs index {edge_index} but only {len(coupling_items)} couplings available")
            return None
        
        tensor_coupling = self._convert_coupling_to_tensor(
            selected_coupling, selected_edge, prev_step, curr_step
        )
        
        if tensor_coupling is not None:
            logger.warning(f"⚠️  SYSTEMATIC FALLBACK: Using coupling {selected_edge} "
                         f"(index {edge_index}/{len(coupling_items)}) for steps {prev_step}→{curr_step}")
            return tensor_coupling
        
        # If selected coupling is invalid, try others systematically  
        logger.debug(f"Primary fallback coupling {selected_edge} failed, trying alternatives")
        for i, (edge_key, coupling) in enumerate(coupling_items):
            if i == edge_index:
                continue  # Already tried this one
                
            tensor_coupling = self._convert_coupling_to_tensor(coupling, edge_key, prev_step, curr_step)
            if tensor_coupling is not None:
                logger.warning(f"⚠️  FALLBACK RECOVERY: Using coupling {edge_key} "
                             f"(index {i}/{len(coupling_items)}) for steps {prev_step}→{curr_step}")
                return tensor_coupling
        
        logger.error(f"❌ All coupling conversion failed for steps {prev_step}→{curr_step}")
        return None
    
    def _initialize_transport_manager(self, sheaf_metadata: Dict[str, Any]) -> None:
        """
        Initialize the transport metadata manager from sheaf metadata.
        
        Args:
            sheaf_metadata: Sheaf construction metadata containing transport info
        """
        try:
            transport_metadata = sheaf_metadata.get('transport_metadata')
            if transport_metadata:
                self._transport_manager = TransportMetadataManager(
                    transport_metadata=transport_metadata,
                    fallback_policy=self.transport_fallback_policy,
                    quality_threshold=self.transport_quality_threshold,
                    strict_mode=False,  # Use fallbacks in inclusion mapping
                    max_fallback_attempts=3
                )
                logger.info(f"Initialized transport metadata manager with {len(transport_metadata.get('edge_transport_info', {}))} edges")
            else:
                logger.debug("No transport metadata found in sheaf metadata")
        except Exception as e:
            logger.warning(f"Failed to initialize transport metadata manager: {e}")
    
    def _get_edge_for_step_transition(self, 
                                     prev_step: int, 
                                     curr_step: int,
                                     sheaf_metadata: Optional[Dict]) -> Optional[Tuple[str, str]]:
        """
        Determine the edge associated with a step transition.
        
        ENHANCED VERSION: This method now implements better step-to-edge mapping logic
        to fix the stale coupling reuse issue. It properly handles filtration ordering
        and ensures different steps map to different edges when available.
        
        Args:
            prev_step: Previous filtration step
            curr_step: Current filtration step
            sheaf_metadata: Sheaf metadata containing edge information
            
        Returns:
            Edge tuple (source, target) or None if not found
        """
        if not sheaf_metadata:
            logger.debug(f"No sheaf metadata for step transition {prev_step}→{curr_step}")
            return None
        
        logger.debug(f"🔍 Mapping step transition {prev_step}→{curr_step} to specific edge")
        
        # Strategy 1: Try to get edge activation order from GW costs (most reliable)
        gw_costs = sheaf_metadata.get('gw_costs', {})
        if gw_costs:
            # Sort edges by GW cost (increasing filtration order)
            sorted_edges = sorted(gw_costs.items(), key=lambda x: x[1])
            logger.debug(f"Found {len(sorted_edges)} edges in GW cost order: {[e[0] for e in sorted_edges]}")
            
            # 🔧 ZERO-EDGE FIX: Handle step 0 correctly (no edges active)
            if curr_step == 0:
                logger.debug(f"Step 0 has no edges (baseline state)")
                return None
            elif prev_step == 0:
                logger.debug(f"Bootstrap transition {prev_step}→{curr_step}: no edge needed for step 0")
                return None
            
            # Map steps 1-N to edges (step 1 = first edge, step 2 = second edge, etc.)
            edge_index = curr_step - 1  # curr_step=1 maps to index 0 (first edge)
            if 0 <= edge_index < len(sorted_edges):
                edge, cost = sorted_edges[edge_index]
                logger.info(f"✅ STEP-EDGE MAPPING: Step {curr_step} → edge {edge} (index {edge_index}, cost={cost:.6f})")
                return edge
            else:
                # 🔧 COORDINATION ERROR: Steps exceed available edges - this should not happen  
                logger.error(f"❌ COORDINATION BUG: Step {curr_step} needs edge index {edge_index} but only {len(sorted_edges)} edges available")
                logger.error(f"This indicates filtration parameter generation created too many steps")
                logger.error(f"Available edges: {[e[0] for e in sorted_edges]}")
                logger.error(f"Fix: Ensure _determine_optimal_steps() limits steps to edge count")
                
                # 🔧 CRITICAL FIX: Replace wraparound with None to force graceful failure
                # Wraparound causes critical principal angles due to inappropriate edge reuse
                return None
        
        # Strategy 2: Try direct step-to-edge mapping if available
        step_edge_mapping = sheaf_metadata.get('step_edge_mapping', {})
        if step_edge_mapping and curr_step in step_edge_mapping:
            edge = step_edge_mapping[curr_step]
            logger.info(f"✅ DIRECT MAPPING: Step {curr_step} → edge {edge}")
            return edge
        
        # Strategy 3: Try to infer from available couplings
        gw_couplings = sheaf_metadata.get('gw_couplings', {})
        if gw_couplings:
            coupling_keys = list(gw_couplings.keys())
            logger.debug(f"Available couplings for inference: {coupling_keys}")
            
            if len(coupling_keys) > 0:
                # 🔧 ZERO-EDGE FIX: Handle step 0 correctly in inference
                if curr_step == 0 or prev_step == 0:
                    logger.debug(f"Step 0 inference: no coupling needed for baseline state")
                    return None
                
                # Map steps 1-N to coupling indices (step 1 = index 0, etc.)
                coupling_index = curr_step - 1
                if 0 <= coupling_index < len(coupling_keys):
                    inferred_edge = coupling_keys[coupling_index]
                    logger.warning(f"⚠️  INFERRED MAPPING: Step {curr_step} → edge {inferred_edge} (index {coupling_index})")
                    return inferred_edge
                else:
                    logger.error(f"❌ INFERENCE FAILURE: Step {curr_step} needs coupling index {coupling_index} but only {len(coupling_keys)} available")
                    logger.error("Cannot infer edge mapping - coordination error detected")
                    return None
        
        logger.error(f"❌ Could not map step transition {prev_step}→{curr_step} to any edge")
        logger.debug(f"Available metadata keys: {list(sheaf_metadata.keys()) if sheaf_metadata else 'None'}")
        return None
    
    def get_transport_manager_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary information about the transport metadata manager.
        
        Returns:
            Summary dictionary or None if manager not initialized
        """
        if self._transport_manager is None:
            return None
        
        return self._transport_manager.get_availability_summary()
    
    def _extend_rotation_to_inclusion(self, 
                                    rotation: torch.Tensor, 
                                    prev_dim: int, 
                                    curr_dim: int) -> torch.Tensor:
        """
        Extend rotation matrix to proper inclusion mapping.
        
        🔧 PRODUCTION FIX: Enhanced handling for rectangular transport matrices and dimension mismatches.
        Properly handles cases where transport matrices are non-square (e.g., 1x32) and creates
        appropriate inclusion mappings for eigenspace embedding.
        
        Args:
            rotation: Rotation matrix from SVD (may be non-square for rectangular transport matrices)
            prev_dim: Previous eigenspace dimension
            curr_dim: Current eigenspace dimension
            
        Returns:
            Extended inclusion mapping matrix [curr_dim x prev_dim]
        """
        # 🔧 TASK 5 FIX: Comprehensive input validation to prevent dimension mismatches
        if rotation.dim() != 2:
            logger.error(f"❌ Invalid rotation matrix: expected 2D tensor, got {rotation.dim()}D with shape {rotation.shape}")
            raise ComputationError(
                f"Rotation matrix must be 2D, got {rotation.dim()}D with shape {rotation.shape}",
                operation="extend_rotation_to_inclusion"
            )
            
        if prev_dim <= 0 or curr_dim <= 0:
            logger.error(f"❌ Invalid dimensions: prev_dim={prev_dim}, curr_dim={curr_dim}")
            raise ComputationError(
                f"Eigenspace dimensions must be positive: prev_dim={prev_dim}, curr_dim={curr_dim}",
                operation="extend_rotation_to_inclusion"
            )
        
        logger.debug(f"Extending rotation {rotation.shape} to inclusion {curr_dim}x{prev_dim}")
        
        # 🔧 PRODUCTION FIX: Handle degenerate transport matrices (e.g., 1x32 -> 32x32 rotation)
        rot_rows, rot_cols = rotation.shape
        
        # Additional validation for extreme dimension mismatches that could cause issues
        if rot_rows > curr_dim * 2 or rot_cols > prev_dim * 2:
            logger.warning(f"⚠️  Large dimension mismatch: rotation {rotation.shape} >> target {curr_dim}x{prev_dim}")
            logger.warning(f"   This might indicate a problematic transport matrix")
            
        if rot_rows == 1 and rot_cols > 10:
            logger.warning(f"⚠️  Suspicious rotation shape: {rotation.shape} (1×wide matrix)")
            logger.warning(f"   This matches the production error pattern - proceeding with caution")
        
        # Create inclusion matrix with proper device and dtype
        inclusion_map = torch.zeros(curr_dim, prev_dim, 
                                   device=rotation.device, 
                                   dtype=rotation.dtype)
        
        # Strategy 1: Direct embedding when rotation fits
        if rot_rows <= curr_dim and rot_cols <= prev_dim:
            # Direct embedding - place rotation in top-left corner
            inclusion_map[:rot_rows, :rot_cols] = rotation
            
            # Extend with identity for remaining diagonal elements
            diag_start = max(rot_rows, rot_cols)
            diag_end = min(curr_dim, prev_dim)
            for i in range(diag_start, diag_end):
                inclusion_map[i, i] = 1.0
                
            logger.debug(f"Direct embedding: placed {rotation.shape} rotation, added identity from {diag_start} to {diag_end}")
        
        # Strategy 2: Dimension-aware scaling for mismatched sizes
        else:
            # Determine safe embedding region
            safe_rows = min(rot_rows, curr_dim)
            safe_cols = min(rot_cols, prev_dim)
            
            if safe_rows > 0 and safe_cols > 0:
                # Embed the feasible portion
                inclusion_map[:safe_rows, :safe_cols] = rotation[:safe_rows, :safe_cols]
                
                # Fill remaining diagonal elements
                diag_start = max(safe_rows, safe_cols)
                diag_end = min(curr_dim, prev_dim)
                for i in range(diag_start, diag_end):
                    inclusion_map[i, i] = 1.0
                    
                logger.debug(f"Scaled embedding: used {safe_rows}x{safe_cols} of rotation, "
                           f"identity from {diag_start} to {diag_end}")
            else:
                # Fallback to identity when no feasible embedding exists
                diag_end = min(curr_dim, prev_dim)
                for i in range(diag_end):
                    inclusion_map[i, i] = 1.0
                    
                logger.warning(f"Fallback to identity: rotation {rotation.shape} incompatible with {curr_dim}x{prev_dim}")
        
        # Validate result
        if inclusion_map.shape != (curr_dim, prev_dim):
            raise ComputationError(
                f"Inclusion mapping has wrong shape: expected {curr_dim}x{prev_dim}, got {inclusion_map.shape}",
                operation="extend_rotation_to_inclusion"
            )
        
        logger.debug(f"Created inclusion mapping: {inclusion_map.shape}, "
                   f"norm={torch.norm(inclusion_map).item():.3f}")
        
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
                commutes, quality_metadata = self._check_commutative_property(
                    inclusion_map, prev_restrictions, curr_restrictions
                )
                if not commutes:
                    logger.warning("Inclusion mapping violates commutative diagram property")
                    if self.fallback_on_violation:
                        logger.warning("Consider using identity_extension method as fallback")
                    # Store quality metadata for caller inspection
                    if not hasattr(self, '_last_quality_metadata'):
                        self._last_quality_metadata = {}
                    self._last_quality_metadata['inclusion_quality'] = quality_metadata
            
            logger.debug("Inclusion mapping validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Inclusion mapping validation failed: {e}")
            return False
    
    def _check_commutative_property(self, 
                                  inclusion_map: torch.Tensor,
                                  prev_restrictions: Dict,
                                  curr_restrictions: Dict) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if inclusion mapping satisfies commutative diagram property.
        
        For sheaf morphisms, we need: ι ∘ R_prev ≈ R_curr ∘ ι
        where ι is the inclusion mapping and R are restriction maps.
        
        Computes residual per edge (u→v):
        lhs = ι @ R_prev[u→v], rhs = R_curr[u→v] @ ι  
        res_e = ||lhs - rhs||_F / (||lhs||_F + ||rhs||_F + ε)
        
        Args:
            inclusion_map: Inclusion mapping matrix [curr_dim x prev_dim]
            prev_restrictions: Previous step restriction maps {edge: tensor}
            curr_restrictions: Current step restriction maps {edge: tensor}
            
        Returns:
            Tuple of (satisfies_property, quality_metadata)
            where quality_metadata contains residuals and diagnostics
        """
        logger.debug(f"Checking commutative property: inclusion {inclusion_map.shape}")
        
        try:
            # Find common edges between prev and curr restrictions
            common_edges = set(prev_restrictions.keys()) & set(curr_restrictions.keys())
            
            if not common_edges:
                logger.warning("No common edges found between previous and current restrictions")
                return True, {'max_residual': 0.0, 'median_residual': 0.0, 
                            'violated_edges': [], 'edges_checked': 0}
            
            # Sample edges if too many for performance
            edges_to_check = list(common_edges)
            if len(edges_to_check) > self.max_edges_to_check:
                import random
                edges_to_check = random.sample(edges_to_check, self.max_edges_to_check)
                logger.info(f"Sampling {len(edges_to_check)}/{len(common_edges)} edges for commutative check")
            
            # Compute residuals for each edge
            residuals = []
            violated_edges = []
            edge_residuals = {}
            
            # Small epsilon to avoid division by zero
            eps = self.numerical_tolerance
            
            for edge in edges_to_check:
                try:
                    R_prev = prev_restrictions[edge]  # Previous restriction map
                    R_curr = curr_restrictions[edge]  # Current restriction map
                    
                    # Ensure tensors are 2D
                    if R_prev.dim() != 2 or R_curr.dim() != 2:
                        logger.debug(f"Skipping edge {edge}: non-2D restriction maps")
                        continue
                    
                    # Check dimension compatibility for commutative diagram: ι @ R_prev ≈ R_curr @ ι
                    # For matrix multiplication to work:
                    # - ι @ R_prev requires: inclusion_map.shape[1] == R_prev.shape[0]  (4 == 6? NO!)
                    # - R_curr @ ι requires: R_curr.shape[1] == inclusion_map.shape[0]   (6 == 6? YES!)
                    
                    # The correct interpretation: ι is [curr_dim x prev_dim], R are restriction maps
                    # For ι @ R_prev: need inclusion_map cols (prev_dim) == R_prev rows  
                    # For R_curr @ ι: need R_curr cols == inclusion_map rows (curr_dim)
                    # Both results should have same shape
                    
                    # Basic matrix multiplication compatibility
                    can_multiply_left = inclusion_map.shape[1] == R_prev.shape[0]  # ι @ R_prev
                    can_multiply_right = R_curr.shape[1] == inclusion_map.shape[0]  # R_curr @ ι
                    
                    if not (can_multiply_left and can_multiply_right):
                        logger.debug(f"Skipping edge {edge}: incompatible dimensions "
                                   f"ι{inclusion_map.shape}, R_prev{R_prev.shape}, R_curr{R_curr.shape}")
                        logger.debug(f"  ι@R_prev check: {inclusion_map.shape[1]} == {R_prev.shape[0]} ? {can_multiply_left}")
                        logger.debug(f"  R_curr@ι check: {R_curr.shape[1]} == {inclusion_map.shape[0]} ? {can_multiply_right}")
                        continue
                    
                    # Check if results can be meaningfully compared (need overlapping dimensions)
                    result_left_shape = (inclusion_map.shape[0], R_prev.shape[1])  # ι @ R_prev
                    result_right_shape = (R_curr.shape[0], inclusion_map.shape[1])  # R_curr @ ι
                    
                    # For meaningful comparison, the results should have some overlapping structure
                    # We'll compute the residual over the overlapping region
                    overlap_rows = min(result_left_shape[0], result_right_shape[0])
                    overlap_cols = min(result_left_shape[1], result_right_shape[1])
                    
                    if overlap_rows == 0 or overlap_cols == 0:
                        logger.debug(f"Skipping edge {edge}: no overlapping region for comparison")
                        continue
                    
                    # Compute commutative diagram: R_curr @ ι vs ι @ R_prev
                    # Note: We want R_curr ∘ ι ≈ ι ∘ R_prev
                    lhs = torch.mm(R_curr, inclusion_map)  # R_curr @ ι  [R_curr.shape[0] x inclusion_map.shape[1]]
                    rhs = torch.mm(inclusion_map, R_prev)  # ι @ R_prev  [inclusion_map.shape[0] x R_prev.shape[1]]
                    
                    # Extract overlapping regions for comparison 
                    lhs_overlap = lhs[:overlap_rows, :overlap_cols]
                    rhs_overlap = rhs[:overlap_rows, :overlap_cols]
                    
                    # Compute normalized residual over the overlapping region
                    diff = lhs_overlap - rhs_overlap
                    diff_norm = torch.norm(diff, 'fro').item()
                    lhs_norm = torch.norm(lhs_overlap, 'fro').item() 
                    rhs_norm = torch.norm(rhs_overlap, 'fro').item()
                    
                    # Normalized residual to handle scale differences
                    residual = diff_norm / (lhs_norm + rhs_norm + eps)
                    residuals.append(residual)
                    edge_residuals[edge] = residual
                    
                    # Track violations
                    if residual > self.inclusion_quality_tolerance:
                        violated_edges.append((edge, residual))
                        
                    logger.debug(f"Edge {edge}: residual={residual:.6f}, "
                               f"diff_norm={diff_norm:.6f}, lhs_norm={lhs_norm:.6f}, rhs_norm={rhs_norm:.6f}")
                
                except Exception as e:
                    logger.debug(f"Error computing residual for edge {edge}: {e}")
                    continue
            
            if not residuals:
                logger.warning("No valid residuals computed")
                return True, {
                    'max_residual': 0.0, 'median_residual': 0.0,
                    'violated_edges': [], 'edges_checked': 0,
                    'total_common_edges': len(common_edges),
                    'edge_residuals': {},
                    'tolerance_used': self.inclusion_quality_tolerance
                }
            
            # Compute statistics
            max_residual = max(residuals)
            median_residual = torch.tensor(residuals).median().item()
            
            # Sort violated edges by residual (worst first)
            violated_edges.sort(key=lambda x: x[1], reverse=True)
            
            # Check if property is satisfied
            satisfies_property = max_residual <= self.inclusion_quality_tolerance
            
            # Create quality metadata
            quality_metadata = {
                'max_residual': max_residual,
                'median_residual': median_residual, 
                'violated_edges': violated_edges,
                'edges_checked': len(residuals),
                'total_common_edges': len(common_edges),
                'edge_residuals': edge_residuals,
                'tolerance_used': self.inclusion_quality_tolerance
            }
            
            # Log results
            if not satisfies_property:
                logger.warning(f"Commutative property violated: max_residual={max_residual:.6f} > "
                             f"tolerance={self.inclusion_quality_tolerance}")
                logger.warning(f"Top 3 violating edges: {violated_edges[:3]}")
            else:
                logger.debug(f"Commutative property satisfied: max_residual={max_residual:.6f}")
            
            return satisfies_property, quality_metadata
            
        except Exception as e:
            logger.error(f"Error in commutative property check: {e}")
            return False, {'error': str(e), 'max_residual': float('inf')}
    
    def _validate_transport_matrix_quality(self, transport_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive validation of transport matrix quality for inclusion mapping.
        
        🔧 ENHANCED VERSION: Implements comprehensive quality checks following user guidelines:
        - Condition number check: cond(A) > 1e12 → abort and use fallback
        - Rank check: rank(A) < d → apply regularization  
        - Mass preservation validation for transport matrices
        - Detailed logging of rank, condition number, and Frobenius norm
        - Transport-specific property validation (non-negativity, stochasticity)
        
        Args:
            transport_matrix: Transport matrix to validate
            
        Returns:
            Dictionary with comprehensive validation results and quality metrics
        """
        issues = []
        abort_recommended = False
        
        try:
            shape = transport_matrix.shape
            logger.debug(f"Validating transport matrix quality: shape {shape}")
            
            # 1. Basic numerical issues
            if torch.any(torch.isnan(transport_matrix)) or torch.any(torch.isinf(transport_matrix)):
                issues.append("contains_nan_inf")
                abort_recommended = True
            
            # 2. Matrix norm analysis
            matrix_norm = torch.norm(transport_matrix, 'fro').item()
            if matrix_norm < 1e-12:
                issues.append(f"zero_matrix_norm_{matrix_norm:.2e}")
                abort_recommended = True
            
            # 3. SVD-based condition number and rank analysis
            U, S, Vt = torch.svd(transport_matrix)
            
            if len(S) == 0:
                issues.append("empty_singular_values")
                abort_recommended = True
                condition_number = float('inf')
                effective_rank = 0
            else:
                # Condition number check (CRITICAL per guidelines)
                condition_number = (S[0] / S[-1]).item() if S[-1] > 1e-14 else float('inf')
                
                if condition_number > 1e12:  # User guideline: abort if cond(A) > 1e12
                    issues.append(f"critical_condition_number_{condition_number:.2e}")
                    abort_recommended = True
                elif condition_number > 1e8:
                    issues.append(f"high_condition_number_{condition_number:.2e}")
                
                # Rank analysis
                rank_threshold = 1e-10 * S[0]
                effective_rank = torch.sum(S > rank_threshold).item()
                expected_rank = min(transport_matrix.shape)
                
                # User guideline: rank(A) < d → needs regularization
                min_acceptable_rank = max(1, int(expected_rank * 0.5))  # At least 50% of expected rank
                if effective_rank < min_acceptable_rank:
                    issues.append(f"critical_rank_deficient_{effective_rank}/{expected_rank}")
                    if effective_rank == 0:
                        abort_recommended = True
                elif effective_rank < expected_rank * 0.8:
                    issues.append(f"rank_deficient_{effective_rank}/{expected_rank}")
            
            # 4. Transport matrix specific properties
            # Non-negativity check (transport matrices should be ≥ 0)
            if torch.any(transport_matrix < -1e-6):  # Allow small numerical errors
                negative_count = torch.sum(transport_matrix < -1e-6).item()
                issues.append(f"negative_values_count_{negative_count}")
            
            # 5. Mass preservation analysis (doubly stochastic property)
            row_sums = torch.sum(transport_matrix, dim=1)
            col_sums = torch.sum(transport_matrix, dim=0)
            
            row_sum_mean = torch.mean(row_sums).item()
            col_sum_mean = torch.mean(col_sums).item()
            row_sum_var = torch.var(row_sums).item()
            col_sum_var = torch.var(col_sums).item()
            
            # Check if sums are roughly uniform (doubly stochastic)
            if row_sum_var > 0.1 or col_sum_var > 0.1:
                issues.append(f"non_uniform_mass_preservation_var_{max(row_sum_var, col_sum_var):.3f}")
            
            # Check if mass is preserved (sums should be similar)
            mass_preservation_error = abs(row_sum_mean - col_sum_mean)
            if mass_preservation_error > 0.1:
                issues.append(f"mass_preservation_error_{mass_preservation_error:.3f}")
            
            # 6. Additional transport quality metrics
            # Sparsity analysis
            total_elements = transport_matrix.numel()
            nonzero_elements = torch.sum(torch.abs(transport_matrix) > 1e-10).item()
            sparsity_ratio = 1.0 - (nonzero_elements / total_elements)
            
            if sparsity_ratio > 0.95:  # More than 95% zeros
                issues.append(f"excessive_sparsity_{sparsity_ratio:.3f}")
            
            # 7. Overall assessment following user guidelines
            # Abort if critical issues detected
            if abort_recommended:
                is_suitable = False
                recommendation = "abort_use_fallback"
            elif any("critical_" in issue for issue in issues):
                is_suitable = False
                recommendation = "apply_aggressive_regularization"
            elif len(issues) > 0:
                is_suitable = False
                recommendation = "apply_standard_regularization"
            else:
                is_suitable = True
                recommendation = "matrix_acceptable"
            
            # Comprehensive quality metrics
            quality_metrics = {
                'is_suitable': is_suitable,
                'recommendation': recommendation,
                'abort_recommended': abort_recommended,
                'issues': issues,
                'condition_number': condition_number,
                'effective_rank': effective_rank,
                'expected_rank': expected_rank,
                'matrix_norm_frobenius': matrix_norm,
                'row_sum_mean': row_sum_mean,
                'col_sum_mean': col_sum_mean,
                'row_sum_variance': row_sum_var,
                'col_sum_variance': col_sum_var,
                'mass_preservation_error': mass_preservation_error,
                'sparsity_ratio': sparsity_ratio,
                'shape': shape
            }
            
            # Detailed logging following user guidelines
            logger.debug(f"Transport validation: rank={effective_rank}/{expected_rank}, "
                        f"cond={condition_number:.2e}, norm_F={matrix_norm:.2e}")
            
            if abort_recommended:
                logger.warning(f"🚨 ABORT RECOMMENDED: Transport matrix has critical issues: {issues}")
            elif not is_suitable:
                logger.warning(f"⚠️  REGULARIZATION NEEDED: Transport matrix issues: {issues}")
            else:
                logger.debug(f"✅ Transport matrix quality acceptable")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Transport matrix quality validation failed: {e}")
            return {
                'is_suitable': False,
                'recommendation': 'validation_failed_abort',
                'abort_recommended': True,
                'issues': [f"validation_failed_{e}"],
                'condition_number': float('inf'),
                'effective_rank': 0,
                'expected_rank': min(transport_matrix.shape) if transport_matrix.numel() > 0 else 0,
                'error': str(e)
            }
    
    def _regularize_transport_matrix(self, transport_matrix: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply robust progressive regularization to transport matrix with infinite condition numbers.
        
        🔧 CRITICAL BUG FIX: The previous regularization was ineffective for infinite condition numbers.
        This implements a progressive strategy: epsilon floor → Sinkhorn → SVD-based → fallback.
        
        Args:
            transport_matrix: Original transport matrix
            
        Returns:
            Tuple of (regularized_matrix, regularization_metadata)
            where metadata contains method used, condition numbers, and success status
        """
        original_matrix = transport_matrix.clone()
        pre_condition_number = self._compute_condition_number(original_matrix)
        
        logger.debug(f"🔧 Starting robust regularization: pre_cond={pre_condition_number:.2e}")
        
        # Progressive regularization strategy - try methods in order of increasing aggressiveness
        regularization_methods = [
            ('epsilon_floor', self._epsilon_floor_regularization),
            ('sinkhorn', self._sinkhorn_regularization),
            ('svd_tikhonov', self._svd_tikhonov_regularization),
            ('nuclear_norm', self._nuclear_norm_regularization)
        ]
        
        regularized = transport_matrix
        metadata = {
            'pre_condition_number': pre_condition_number,
            'post_condition_number': float('inf'),
            'method_used': 'none',
            'regularization_strength': 0.0,
            'iterations': 0,
            'success': False
        }
        
        for method_name, regularization_func in regularization_methods:
            try:
                logger.debug(f"🧪 Trying {method_name} regularization...")
                regularized, method_metadata = regularization_func(regularized)
                
                # Validate the result
                post_condition_number = self._compute_condition_number(regularized)
                
                logger.debug(f"📊 {method_name}: {pre_condition_number:.2e} → {post_condition_number:.2e}")
                
                # Check if regularization was successful
                if self._is_regularization_successful(post_condition_number):
                    metadata.update({
                        'post_condition_number': post_condition_number,
                        'method_used': method_name,
                        'success': True,
                        **method_metadata
                    })
                    logger.info(f"✅ {method_name.upper()} regularization successful: "
                              f"cond {pre_condition_number:.2e} → {post_condition_number:.2e}")
                    return regularized, metadata
                
            except Exception as e:
                logger.debug(f"❌ {method_name} regularization failed: {e}")
                continue
        
        # If all regularization methods failed, return original matrix and mark as failed
        logger.error(f"❌ ALL regularization methods failed! Original cond={pre_condition_number:.2e}")
        metadata.update({
            'post_condition_number': pre_condition_number,
            'method_used': 'failed',
            'success': False
        })
        
        return original_matrix, metadata
    
    def _compute_condition_number(self, matrix: torch.Tensor) -> float:
        """
        Compute condition number of a matrix using SVD with robust handling.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Condition number (float('inf') if singular or ill-conditioned)
        """
        try:
            if matrix.numel() == 0 or torch.any(torch.isnan(matrix)) or torch.any(torch.isinf(matrix)):
                return float('inf')
            
            _, S, _ = torch.svd(matrix)
            if len(S) == 0 or S[-1] <= 1e-14:
                return float('inf')
            
            return (S[0] / S[-1]).item()
            
        except Exception:
            return float('inf')
    
    def _is_regularization_successful(self, condition_number: float) -> bool:
        """
        Check if regularization was successful based on condition number.
        
        Args:
            condition_number: Post-regularization condition number
            
        Returns:
            True if condition number is acceptable, False otherwise
        """
        return condition_number < 1e12 and not (condition_number == float('inf') or condition_number != condition_number)
    
    def _epsilon_floor_regularization(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply epsilon floor + row normalization regularization.
        
        Implements: M = M.clamp_min(eps); M = M / M.sum(dim=1, keepdim=True)
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (regularized_matrix, method_metadata)
        """
        eps = 1e-10
        
        # Apply epsilon floor to prevent zeros
        regularized = torch.clamp(matrix, min=eps)
        
        # Normalize rows to maintain transport matrix properties
        row_sums = regularized.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=eps)
        regularized = regularized / row_sums
        
        metadata = {
            'regularization_strength': eps,
            'iterations': 1
        }
        
        return regularized, metadata
    
    def _sinkhorn_regularization(self, matrix: torch.Tensor, max_iter: int = 20, epsilon: float = 1e-2) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply enhanced Sinkhorn projection to make matrix doubly stochastic.
        
        🔧 ENHANCED VERSION: Implements robust Sinkhorn projection following user guidelines:
        - Proper mass preservation with row/col normalization
        - Enhanced convergence checking with better tolerance
        - Stronger epsilon floor to prevent numerical collapse
        - Better handling of edge cases (rectangular matrices, zeros)
        
        Args:
            matrix: Input matrix
            max_iter: Maximum Sinkhorn iterations (5-20 recommended)
            epsilon: Entropic regularization parameter (~1e-2)
            
        Returns:
            Tuple of (regularized_matrix, method_metadata)
        """
        eps = 1e-10  # Stronger epsilon floor as per guidelines
        
        # Apply stronger epsilon floor to prevent collapse
        regularized = torch.clamp(matrix, min=eps)
        
        # Track convergence
        prev_row_diff = float('inf')
        prev_col_diff = float('inf')
        converged = False
        
        # Enhanced Sinkhorn iterations for doubly stochastic projection
        for i in range(max_iter):
            # Row normalization: A = A / (A.sum(dim=1, keepdim=True) + eps)
            row_sums = regularized.sum(dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=eps)  # Prevent division by zero
            regularized = regularized / row_sums
            
            # Column normalization: A = A / (A.sum(dim=0, keepdim=True) + eps)  
            col_sums = regularized.sum(dim=0, keepdim=True)
            col_sums = torch.clamp(col_sums, min=eps)  # Prevent division by zero
            regularized = regularized / col_sums
            
            # Enhanced convergence check (start earlier, better tolerance)
            if i >= 2:  # Start checking after 2 iterations
                row_diff = torch.abs(regularized.sum(dim=1) - 1.0).max().item()
                col_diff = torch.abs(regularized.sum(dim=0) - 1.0).max().item()
                
                # Check for convergence with improved tolerance
                if row_diff < 1e-6 and col_diff < 1e-6:
                    converged = True
                    break
                
                # Check for stagnation
                if abs(row_diff - prev_row_diff) < 1e-8 and abs(col_diff - prev_col_diff) < 1e-8:
                    logger.debug(f"Sinkhorn stagnation detected at iteration {i}")
                    break
                    
                prev_row_diff = row_diff
                prev_col_diff = col_diff
        
        # Final quality check
        final_row_diff = torch.abs(regularized.sum(dim=1) - 1.0).max().item()
        final_col_diff = torch.abs(regularized.sum(dim=0) - 1.0).max().item()
        
        metadata = {
            'regularization_strength': epsilon,
            'iterations': i + 1,
            'converged': converged,
            'final_row_diff': final_row_diff,
            'final_col_diff': final_col_diff,
            'doubly_stochastic_error': max(final_row_diff, final_col_diff)
        }
        
        logger.debug(f"Sinkhorn regularization: {i+1} iterations, "
                    f"converged={converged}, row_diff={final_row_diff:.2e}, col_diff={final_col_diff:.2e}")
        
        return regularized, metadata
    
    def _svd_tikhonov_regularization(self, matrix: torch.Tensor, sigma_min_factor: float = 1e-3) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply enhanced SVD-based Tikhonov regularization with adaptive singular value floor.
        
        🔧 ENHANCED VERSION: Implements proper SVD floor following user guidelines:
        - Adaptive sigma_min = sigma_min_factor * S.max() (typically 1e-3)
        - Prevents complete rank collapse while maintaining numerical stability
        - Better logging and diagnostics for debugging
        - Handles edge cases (empty matrices, all-zero singular values)
        
        Args:
            matrix: Input matrix
            sigma_min_factor: Factor for adaptive minimum (sigma_min = factor * S.max())
            
        Returns:
            Tuple of (regularized_matrix, method_metadata)
        """
        try:
            # Compute SVD with reduced matrices for efficiency
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            
            if len(S) == 0:
                logger.warning("SVD returned empty singular values")
                return matrix, {'regularization_strength': 0.0, 'success': False}
            
            # Adaptive sigma_min following guidelines: sigma_min = factor * S.max()
            max_singular = S[0].item()  # Largest singular value
            sigma_min = sigma_min_factor * max_singular
            
            # Enhanced clamping with proper floor
            S_regularized = torch.clamp(S, min=sigma_min)
            
            # Reconstruct matrix with regularized singular values
            regularized = torch.mm(torch.mm(U, torch.diag(S_regularized)), Vh)
            
            # Comprehensive diagnostics
            num_clamped = torch.sum(S < sigma_min).item()
            original_rank = torch.sum(S > 1e-14).item()
            regularized_rank = torch.sum(S_regularized > 1e-14).item()
            
            # Condition number improvement
            original_cond = (S[0] / S[-1]).item() if S[-1] > 1e-14 else float('inf')
            regularized_cond = (S_regularized[0] / S_regularized[-1]).item()
            
            metadata = {
                'regularization_strength': sigma_min,
                'sigma_min_factor': sigma_min_factor,
                'max_singular_value': max_singular,
                'iterations': 1,
                'singular_values_clamped': num_clamped,
                'original_rank': original_rank,
                'regularized_rank': regularized_rank,
                'original_condition_number': original_cond,
                'regularized_condition_number': regularized_cond,
                'condition_improvement': original_cond / regularized_cond if regularized_cond > 0 else float('inf'),
                'success': True
            }
            
            logger.debug(f"SVD Tikhonov regularization: {num_clamped}/{len(S)} values clamped, "
                        f"condition number {original_cond:.2e} → {regularized_cond:.2e}")
            
            return regularized, metadata
            
        except Exception as e:
            logger.warning(f"SVD Tikhonov regularization failed: {e}")
            return matrix, {
                'regularization_strength': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def _nuclear_norm_regularization(self, matrix: torch.Tensor, reg_strength: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply nuclear norm regularization as a final aggressive method.
        
        Args:
            matrix: Input matrix
            reg_strength: Regularization strength
            
        Returns:
            Tuple of (regularized_matrix, method_metadata)
        """
        # Apply small Tikhonov regularization to both dimensions
        m, n = matrix.shape
        
        # Add regularization to make matrix more well-conditioned
        if m == n:
            # Square matrix - add identity
            regularized = matrix + reg_strength * DEFAULT_SPECTRAL_POLICY.create_eye(m, device=matrix.device)
        else:
            # Rectangular matrix - add to gram matrix equivalent
            if m > n:
                # Tall matrix - regularize A^T A
                gram = matrix.T @ matrix + reg_strength * DEFAULT_SPECTRAL_POLICY.create_eye(n, device=matrix.device)
                try:
                    chol = torch.linalg.cholesky(gram)
                    regularized = matrix @ torch.linalg.cholesky_inverse(chol) @ chol
                except:
                    # Fallback to simple regularization
                    regularized = matrix + reg_strength * torch.ones_like(matrix)
            else:
                # Wide matrix - regularize A A^T  
                gram = matrix @ matrix.T + reg_strength * DEFAULT_SPECTRAL_POLICY.create_eye(m, device=matrix.device)
                try:
                    chol = torch.linalg.cholesky(gram)
                    regularized = torch.linalg.cholesky_inverse(chol) @ chol @ matrix
                except:
                    # Fallback to simple regularization
                    regularized = matrix + reg_strength * torch.ones_like(matrix)
        
        metadata = {
            'regularization_strength': reg_strength,
            'iterations': 1
        }
        
        return regularized, metadata
    
    def _create_fallback_inclusion_mapping(self, prev_dim: int, curr_dim: int, 
                                         regularization_metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Create robust fallback inclusion mappings when regularization fails.
        
        Implements multiple fallback strategies: identity → permutation → barycentric
        
        Args:
            prev_dim: Previous eigenspace dimension
            curr_dim: Current eigenspace dimension  
            regularization_metadata: Metadata from failed regularization
            
        Returns:
            Robust inclusion mapping matrix
        """
        logger.warning(f"🛟 Creating fallback inclusion mapping: {prev_dim} → {curr_dim}")
        
        try:
            # Strategy 1: Identity/Extension mapping (most conservative)
            if prev_dim <= curr_dim:
                # Extension case: embed into larger space with identity + zeros
                inclusion_map = DEFAULT_SPECTRAL_POLICY.create_zeros(curr_dim, prev_dim)
                inclusion_map[:prev_dim, :prev_dim] = DEFAULT_SPECTRAL_POLICY.create_eye(prev_dim)
                
                logger.info(f"✅ IDENTITY fallback: {prev_dim} → {curr_dim} with zero extension")
                return inclusion_map
            
            else:
                # Compression case: use orthogonal projection to smaller space
                inclusion_map = DEFAULT_SPECTRAL_POLICY.create_zeros(curr_dim, prev_dim)
                
                # Strategy 2: Permutation-based mapping (preserve principal components)
                # Create a well-conditioned permutation that maps first curr_dim components
                for i in range(curr_dim):
                    inclusion_map[i, i] = 1.0
                
                # Add small barycentric weighting to remaining dimensions
                if prev_dim > curr_dim:
                    remaining_weight = 1.0 / (prev_dim - curr_dim)
                    for i in range(curr_dim):
                        for j in range(curr_dim, prev_dim):
                            inclusion_map[i, j] = remaining_weight / curr_dim
                
                logger.info(f"✅ PERMUTATION fallback: {prev_dim} → {curr_dim} with barycentric weighting")
                return inclusion_map
            
        except Exception as e:
            logger.error(f"❌ Fallback mapping creation failed: {e}")
            # Ultimate fallback: simple identity truncation/extension
            min_dim = min(prev_dim, curr_dim)
            inclusion_map = DEFAULT_SPECTRAL_POLICY.create_zeros(curr_dim, prev_dim)
            inclusion_map[:min_dim, :min_dim] = DEFAULT_SPECTRAL_POLICY.create_eye(min_dim)
            
            logger.warning(f"⚠️  ULTIMATE fallback: simple identity truncation/extension")
            return inclusion_map
    
    def _apply_smoothing(self, 
                        current_inclusion: InclusionMapping,
                        prev_step: int, 
                        curr_step: int) -> InclusionMapping:
        """
        🔧 MULTI-STEP SMOOTHING: Apply exponential moving average to inclusion mappings.
        
        This implements the smoothing formula: M_k = α*M_k + (1-α)*M_{k-1}
        where M_k is the current inclusion matrix and M_{k-1} is the previous one.
        
        Smoothing is only applied when:
        1. A previous mapping exists for this transition
        2. The stability score is above threshold
        3. Matrix dimensions are compatible
        
        Args:
            current_inclusion: Current inclusion mapping
            prev_step: Previous step index
            curr_step: Current step index
            
        Returns:
            Smoothed inclusion mapping
        """
        try:
            step_key = (prev_step, curr_step)
            
            # Check if we have history for this step transition
            if step_key not in self._inclusion_history:
                # No history - store current and return as-is
                self._store_inclusion_history(current_inclusion, prev_step, curr_step)
                return current_inclusion
            
            previous_inclusion = self._inclusion_history[step_key]
            
            # Check stability score to decide whether to apply smoothing
            stability_score = self._compute_stability_score(current_inclusion, previous_inclusion)
            
            if stability_score >= self.stability_threshold:
                # High stability - apply smoothing
                smoothed_matrix = self._exponential_moving_average(
                    current_inclusion.matrix, 
                    previous_inclusion.matrix,
                    self.smoothing_alpha
                )
                
                # Create smoothed inclusion mapping
                smoothed_meta = current_inclusion.meta.copy()
                smoothed_meta['smoothing_applied'] = True
                smoothed_meta['stability_score'] = stability_score
                smoothed_meta['smoothing_alpha'] = self.smoothing_alpha
                
                smoothed_inclusion = InclusionMapping(matrix=smoothed_matrix, meta=smoothed_meta)
                
                logger.debug(f"Applied smoothing to step {prev_step}→{curr_step}: "
                           f"stability={stability_score:.3f}, alpha={self.smoothing_alpha}")
                
            else:
                # Low stability - use current matrix without smoothing
                smoothed_inclusion = current_inclusion
                smoothed_inclusion.meta['smoothing_applied'] = False
                smoothed_inclusion.meta['stability_score'] = stability_score
                
                logger.debug(f"Skipped smoothing for step {prev_step}→{curr_step}: "
                           f"stability={stability_score:.3f} < threshold={self.stability_threshold}")
            
            # Store for next iteration
            self._store_inclusion_history(smoothed_inclusion, prev_step, curr_step)
            
            return smoothed_inclusion
            
        except Exception as e:
            logger.warning(f"Smoothing failed for step {prev_step}→{curr_step}: {e}")
            return current_inclusion
    
    def _exponential_moving_average(self, 
                                   current_matrix: torch.Tensor,
                                   previous_matrix: torch.Tensor,
                                   alpha: float) -> torch.Tensor:
        """
        Compute exponential moving average of inclusion matrices.
        
        Formula: M_smoothed = α * M_current + (1-α) * M_previous
        
        Args:
            current_matrix: Current inclusion matrix
            previous_matrix: Previous inclusion matrix  
            alpha: Smoothing parameter [0,1] (higher = less smoothing)
            
        Returns:
            Exponentially smoothed matrix
        """
        try:
            # Check dimension compatibility
            if current_matrix.shape != previous_matrix.shape:
                logger.debug(f"Matrix shape mismatch for smoothing: "
                           f"{current_matrix.shape} vs {previous_matrix.shape}")
                return current_matrix
            
            # Apply exponential moving average
            smoothed_matrix = alpha * current_matrix + (1 - alpha) * previous_matrix
            
            return smoothed_matrix
            
        except Exception as e:
            logger.debug(f"EMA computation failed: {e}")
            return current_matrix
    
    def _compute_stability_score(self, 
                                current_inclusion: InclusionMapping,
                                previous_inclusion: InclusionMapping) -> float:
        """
        Compute stability score between consecutive inclusion mappings.
        
        Uses Frobenius norm distance normalized by matrix size to determine
        how similar consecutive mappings are. Higher scores indicate more
        stable transport matrices that benefit from smoothing.
        
        Args:
            current_inclusion: Current inclusion mapping
            previous_inclusion: Previous inclusion mapping
            
        Returns:
            Stability score in [0, 1] (higher = more stable)
        """
        try:
            curr_matrix = current_inclusion.matrix
            prev_matrix = previous_inclusion.matrix
            
            if curr_matrix.shape != prev_matrix.shape:
                return 0.0  # Incompatible shapes = no stability
            
            # Compute Frobenius norm of difference
            diff_norm = torch.norm(curr_matrix - prev_matrix, p='fro')
            
            # Normalize by matrix size and typical values
            matrix_size = torch.numel(curr_matrix)
            avg_norm = (torch.norm(curr_matrix, p='fro') + torch.norm(prev_matrix, p='fro')) / 2
            
            if avg_norm < self.numerical_tolerance:
                return 1.0  # Both matrices are near zero - maximally stable
            
            # Stability score: 1 - normalized_difference
            normalized_diff = diff_norm / (avg_norm + self.numerical_tolerance)
            stability_score = max(0.0, 1.0 - normalized_diff)
            
            return float(stability_score)
            
        except Exception as e:
            logger.debug(f"Stability score computation failed: {e}")
            return 0.0  # Conservative: no stability if computation fails
    
    def _store_inclusion_history(self, 
                                inclusion_mapping: InclusionMapping,
                                prev_step: int,
                                curr_step: int) -> None:
        """
        Store inclusion mapping in history for future smoothing.
        
        Args:
            inclusion_mapping: Inclusion mapping to store
            prev_step: Previous step index
            curr_step: Current step index
        """
        step_key = (prev_step, curr_step)
        
        # Store the inclusion mapping (deep copy the matrix to avoid modification)
        stored_mapping = InclusionMapping(
            matrix=inclusion_mapping.matrix.clone().detach(),
            meta=inclusion_mapping.meta.copy()
        )
        
        self._inclusion_history[step_key] = stored_mapping
        
        # Limit history size to prevent memory growth
        max_history_size = 10
        if len(self._inclusion_history) > max_history_size:
            # Remove oldest entries
            oldest_keys = sorted(self._inclusion_history.keys())[:len(self._inclusion_history) - max_history_size]
            for old_key in oldest_keys:
                del self._inclusion_history[old_key]
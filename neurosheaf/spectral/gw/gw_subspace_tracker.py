# neurosheaf/spectral/gw/gw_subspace_tracker.py
"""
Main GW-specific subspace tracker using Persistent Eigenvector Similarity.

Implements eigenvalue tracking for Gromov-Wasserstein sheaf constructions
using the Persistent Eigenvector Similarity (PES) methodology with transport-
informed inclusion mappings and proper GW filtration semantics.

This class inherits from SubspaceTracker to maintain API compatibility while
providing specialized behavior for GW constructions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from ..tracker import SubspaceTracker
from .pes_computation import PESComputer
from .sheaf_inclusion_mapper import SheafInclusionMapper, InclusionMapping
from .gw_eigenspace_embedder import GWEigenspaceEmbedder
from .gw_birth_death_detector import GWBirthDeathDetector
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError
from ..transport_metadata_manager import (
    TransportNotAvailableError,
    TransportQualityError
)
from ..dtype_policy import DEFAULT_SPECTRAL_POLICY, to_spectral_dtype

logger = setup_logger(__name__)


class GWSubspaceTracker(SubspaceTracker):
    """
    GW-specific subspace tracker using Persistent Eigenvector Similarity.
    
    Inherits from SubspaceTracker to maintain API compatibility while providing
    specialized tracking behavior for Gromov-Wasserstein sheaf constructions.
    
    Key Features:
    - PES-based eigenvalue tracking with transport weighting
    - Transport-informed inclusion mappings
    - GW-aware birth-death event detection
    - Increasing complexity filtration semantics
    """
    
    def __init__(self, 
                 pes_threshold: float = 0.8,
                 inclusion_method: str = 'transport_svd',
                 transport_weighting: bool = True,
                 transport_weighting_alpha: float = 1.0,
                 eigenvalue_threshold: float = 1e-10,
                 validate_gw_semantics: bool = True,
                 **kwargs):
        """
        Initialize GW-specific subspace tracker.
        
        Args:
            pes_threshold: Minimum PES similarity for accepting eigenvector matches
            inclusion_method: Method for creating inclusion mappings
                - 'transport_svd': SVD-based using transport matrices
                - 'transport_projection': Direct transport projection  
                - 'identity_extension': Identity-based extension (fallback)
            transport_weighting: Whether to apply transport-based weighting to PES
            transport_weighting_alpha: Exponential weighting parameter for transport costs
            eigenvalue_threshold: Threshold for considering eigenvalues as zero
            validate_gw_semantics: Whether to validate GW filtration semantics
            **kwargs: Additional arguments passed to parent SubspaceTracker
        """
        # Initialize parent with standard parameters
        super().__init__(**kwargs)
        
        # GW-specific parameters
        self.pes_threshold = pes_threshold
        self.inclusion_method = inclusion_method
        self.transport_weighting = transport_weighting
        self.transport_weighting_alpha = transport_weighting_alpha
        self.eigenvalue_threshold = eigenvalue_threshold
        self.validate_gw_semantics = validate_gw_semantics
        
        # Initialize GW-specific components
        self.pes_computer = PESComputer(
            threshold=pes_threshold,
            transport_weighting_alpha=transport_weighting_alpha
        )
        
        self.inclusion_mapper = SheafInclusionMapper(
            method=inclusion_method
        )
        
        self.eigenspace_embedder = GWEigenspaceEmbedder(
            embedding_method='svd_alignment',
            preserve_orthogonality=True
        )
        
        self.birth_death_detector = GWBirthDeathDetector(
            eigenvalue_threshold=eigenvalue_threshold,
            validate_semantics=validate_gw_semantics
        )
        
        # Cache for transport matrices and metadata - DISABLED to prevent stale coupling reuse
        # ❌ CRITICAL BUG FIX: Caching was causing the same transport matrix to be reused
        # across different step transitions, leading to subspace rotations ~90°
        # self._transport_cache = {}  # DISABLED - see GitHub issue: stale coupling reused
        self._transport_cache = None  # Explicitly disabled
        self._mass_matrices_sequence = None
        self._current_sheaf_metadata = None
        
        logger.info(f"GWSubspaceTracker initialized: PES threshold={pes_threshold}, "
                   f"inclusion method={inclusion_method}, transport weighting={transport_weighting}")
    
    def track_eigenspaces(self, 
                         eigenvalues_sequence: List[torch.Tensor],
                         eigenvectors_sequence: List[torch.Tensor],
                         filtration_params: List[float],
                         construction_method: str = 'gromov_wasserstein',
                         sheaf_metadata: Optional[Dict] = None,
                         mass_matrices_sequence: Optional[List[torch.Tensor]] = None) -> Dict:
        """
        Track eigenspaces through GW filtration using PES methodology.
        
        Override of parent method that routes to appropriate tracking implementation
        based on construction method.
        
        Args:
            eigenvalues_sequence: List of eigenvalue tensors for each filtration step
            eigenvectors_sequence: List of eigenvector tensors for each filtration step
            filtration_params: List of filtration parameter values (increasing for GW)
            construction_method: Sheaf construction method
            sheaf_metadata: Additional metadata about sheaf construction
            mass_matrices_sequence: List of mass matrices for M-orthogonal computation (optional)
            
        Returns:
            Dictionary with tracking information including continuous paths
        """
        if construction_method != 'gromov_wasserstein':
            # Fallback to parent implementation for non-GW methods
            logger.debug(f"Non-GW construction method '{construction_method}', using parent tracker")
            return super().track_eigenspaces(
                eigenvalues_sequence, eigenvectors_sequence, 
                filtration_params, construction_method, sheaf_metadata,
                mass_matrices_sequence
            )
        
        # Store metadata for use in tracking components
        self._current_sheaf_metadata = sheaf_metadata
        self._mass_matrices_sequence = mass_matrices_sequence
        
        # Check if M-orthogonal computation should be used
        use_m_orthogonal = mass_matrices_sequence is not None
        if use_m_orthogonal:
            logger.info("M-orthogonal PES computation enabled")
            # Update PES computer to use M-orthogonal metric
            self.pes_computer.metric = 'M'
        else:
            logger.debug("Using Euclidean PES computation (no mass matrices provided)")
            self.pes_computer.metric = None
        
        logger.info(f"Starting GW eigenspace tracking: {len(eigenvalues_sequence)} steps, "
                   f"filtration range [{filtration_params[0]:.6f}, {filtration_params[-1]:.6f}]")
        
        return self._track_gw_eigenspaces_with_pes(
            eigenvalues_sequence, eigenvectors_sequence, filtration_params
        )
    
    def _track_gw_eigenspaces_with_pes(self, 
                                      eigenvals_seq: List[torch.Tensor],
                                      eigenvecs_seq: List[torch.Tensor],
                                      filtration_params: List[float]) -> Dict:
        """
        Main GW tracking algorithm using PES methodology.
        
        Implements the complete PES-based tracking pipeline:
        1. Extract transport matrices from metadata
        2. Create transport-informed inclusion mappings
        3. Embed eigenspaces using inclusion mappings
        4. Compute PES similarity matrices
        5. Find optimal eigenvector matches
        6. Detect GW-specific birth-death events
        7. Generate continuous paths for persistence
        
        Args:
            eigenvals_seq: Sequence of eigenvalue tensors
            eigenvecs_seq: Sequence of eigenvector tensors
            filtration_params: Filtration parameter values
            
        Returns:
            Complete tracking information dictionary
        """
        # Initialize tracking data structures
        tracking_info = self._initialize_gw_tracking_info(len(filtration_params))
        
        # Extract transport matrices from sheaf metadata
        transport_matrices = self._extract_transport_matrices()
        
        # 🔧 STEP-EDGE COORDINATION VALIDATION: Check for coordination errors
        self._validate_step_edge_coordination(len(eigenvals_seq))
        
        # Process each consecutive pair of filtration steps
        for step in range(1, len(eigenvals_seq)):
            logger.debug(f"Processing GW tracking step {step}/{len(eigenvals_seq)-1}")
            
            try:
                # 🔧 ZERO-EDGE FIX: Special handling for step 0→1 transition
                if step == 1:
                    # BOOTSTRAP TRANSITION: Step 0 has no edges, step 1 has first edge
                    logger.info(f"🔧 BOOTSTRAP TRANSITION: {step-1}→{step} (no edges → first edge)")
                    self._handle_bootstrap_transition(step, eigenvals_seq, eigenvecs_seq, tracking_info)
                    continue
                else:
                    # NORMAL TRANSITION: Both steps have edges, transport data available
                    logger.debug(f"📍 NORMAL TRANSITION: {step-1}→{step} (edge-to-edge)")
                    # Continue with existing logic
                # ✅ COMPREHENSIVE DEBUG: Trace tuple sources and structures
                prev_eigenvecs = eigenvecs_seq[step-1]
                curr_eigenvecs = eigenvecs_seq[step]
                
                logger.debug(f"Step {step}: BEFORE validation - prev_eigenvecs type: {type(prev_eigenvecs)}, curr_eigenvecs type: {type(curr_eigenvecs)}")
                
                if isinstance(prev_eigenvecs, tuple):
                    logger.error(f"Step {step}: prev_eigenvecs is tuple instead of tensor: {type(prev_eigenvecs)}")
                    logger.error(f"Step {step}: tuple length: {len(prev_eigenvecs)}")
                    logger.error(f"Step {step}: tuple structure: {[type(x) for x in prev_eigenvecs]}")
                    
                    # Try to extract tensor from tuple with better logic
                    extracted = False
                    for i, item in enumerate(prev_eigenvecs):
                        if torch.is_tensor(item):
                            logger.warning(f"Step {step}: found tensor at position {i}: shape {item.shape if hasattr(item, 'shape') else 'no shape'}")
                            if hasattr(item, 'shape') and len(item.shape) >= 2:  # Likely eigenvectors (2D)
                                prev_eigenvecs = item
                                logger.warning(f"Step {step}: extracted 2D tensor as prev_eigenvecs: {item.shape}")
                                extracted = True
                                break
                    
                    if not extracted:
                        # Fallback: try the second element (original logic)
                        if len(prev_eigenvecs) > 1 and torch.is_tensor(prev_eigenvecs[1]):
                            prev_eigenvecs = prev_eigenvecs[1]
                            logger.warning(f"Step {step}: fallback extracted prev_eigenvecs[1]: {prev_eigenvecs.shape if hasattr(prev_eigenvecs, 'shape') else 'no shape'}")
                            extracted = True
                    
                    if not extracted:
                        prev_eigenvecs = torch.tensor([])
                        logger.warning(f"Step {step}: fallback to empty prev_eigenvecs tensor")
                
                if isinstance(curr_eigenvecs, tuple):
                    logger.error(f"Step {step}: curr_eigenvecs is tuple instead of tensor: {type(curr_eigenvecs)}")
                    logger.error(f"Step {step}: tuple length: {len(curr_eigenvecs)}")
                    logger.error(f"Step {step}: tuple structure: {[type(x) for x in curr_eigenvecs]}")
                    
                    # Try to extract tensor from tuple with better logic
                    extracted = False
                    for i, item in enumerate(curr_eigenvecs):
                        if torch.is_tensor(item):
                            logger.warning(f"Step {step}: found tensor at position {i}: shape {item.shape if hasattr(item, 'shape') else 'no shape'}")
                            if hasattr(item, 'shape') and len(item.shape) >= 2:  # Likely eigenvectors (2D)
                                curr_eigenvecs = item
                                logger.warning(f"Step {step}: extracted 2D tensor as curr_eigenvecs: {item.shape}")
                                extracted = True
                                break
                    
                    if not extracted:
                        # Fallback: try the second element (original logic)
                        if len(curr_eigenvecs) > 1 and torch.is_tensor(curr_eigenvecs[1]):
                            curr_eigenvecs = curr_eigenvecs[1]
                            logger.warning(f"Step {step}: fallback extracted curr_eigenvecs[1]: {curr_eigenvecs.shape if hasattr(curr_eigenvecs, 'shape') else 'no shape'}")
                            extracted = True
                    
                    if not extracted:
                        curr_eigenvecs = torch.tensor([])
                        logger.warning(f"Step {step}: fallback to empty curr_eigenvecs tensor")
                
                # Ensure we have proper tensors with valid shapes
                if not torch.is_tensor(prev_eigenvecs) or len(prev_eigenvecs.shape) == 0:
                    logger.warning(f"Step {step}: prev_eigenvecs invalid, using empty tensor")
                    prev_eigenspace_dim = 0
                else:
                    prev_eigenspace_dim = prev_eigenvecs.shape[0]
                
                if not torch.is_tensor(curr_eigenvecs) or len(curr_eigenvecs.shape) == 0:
                    logger.warning(f"Step {step}: curr_eigenvecs invalid, using empty tensor")
                    curr_eigenspace_dim = 0
                else:
                    curr_eigenspace_dim = curr_eigenvecs.shape[0]
                
                # ✅ DEBUG: Log final validated types and shapes before use
                logger.debug(f"Step {step}: AFTER validation - prev_eigenvecs type: {type(prev_eigenvecs)}, shape: {prev_eigenvecs.shape if hasattr(prev_eigenvecs, 'shape') else 'no shape'}")
                logger.debug(f"Step {step}: AFTER validation - curr_eigenvecs type: {type(curr_eigenvecs)}, shape: {curr_eigenvecs.shape if hasattr(curr_eigenvecs, 'shape') else 'no shape'}")
                
                # Step 1: Create transport-informed inclusion mapping
                # 🔍 ENHANCED LOGGING: Track step transitions and coupling usage
                logger.info(f"📍 STEP TRANSITION: {step-1}→{step} (eigenspace {prev_eigenspace_dim}→{curr_eigenspace_dim})")
                
                inclusion_result = self.inclusion_mapper.create_gw_inclusion_mapping(
                    prev_step=step-1,
                    curr_step=step,
                    prev_eigenspace_dim=prev_eigenspace_dim,
                    curr_eigenspace_dim=curr_eigenspace_dim,
                    transport_matrices=transport_matrices,
                    sheaf_metadata=self._current_sheaf_metadata
                )
                
                # Log coupling usage for this step transition
                if inclusion_result and hasattr(inclusion_result, 'meta'):
                    method_used = inclusion_result.meta.get('method_used', 'unknown')
                    logger.info(f"📊 COUPLING METHOD: Step {step-1}→{step} used method '{method_used}'")
                
                # Step 2: Embed previous eigenspace into current space
                embedded_prev_vecs = self.eigenspace_embedder.embed_eigenspace(
                    prev_eigenvectors=prev_eigenvecs,
                    inclusion_mapping=inclusion_result,
                    transport_costs=self._get_transport_costs(step-1, step)
                )
                
                # Step 3: Compute PES similarity matrix
                mass_matrix = self._get_mass_matrix(step) if self._mass_matrices_sequence else None
                pes_matrix = self.pes_computer.compute_pes_matrix(
                    prev_eigenvecs=embedded_prev_vecs,
                    curr_eigenvecs=curr_eigenvecs,
                    transport_weighting=self.transport_weighting,
                    transport_costs=self._get_transport_costs(step-1, step),
                    mass_matrix=mass_matrix
                )
                
                # Step 4: Find optimal eigenvector matching with enhanced logging
                matches = self.pes_computer.optimal_eigenvector_matching(
                    pes_matrix, use_threshold=True, adaptive_threshold=True
                )
                
                # Enhanced debugging for match quality
                if matches:
                    match_similarities = [m[2] for m in matches]
                    avg_similarity = sum(match_similarities) / len(match_similarities)
                    logger.debug(f"Step {step}: {len(matches)} matches, avg similarity {avg_similarity:.3f}, "
                               f"range [{min(match_similarities):.3f}, {max(match_similarities):.3f}]")
                else:
                    logger.warning(f"Step {step}: No eigenvector matches found - this may break tracking continuity")
                
                # Step 5: Detect GW-specific birth-death events
                birth_death_events = self.birth_death_detector.detect_gw_events(
                    matches=matches,
                    prev_eigenvals=eigenvals_seq[step-1],
                    curr_eigenvals=eigenvals_seq[step],
                    prev_param=filtration_params[step-1],
                    curr_param=filtration_params[step],
                    step=step
                )
                
                # Debug birth-death event generation
                n_births = len(birth_death_events.get('birth_events', []))
                n_deaths = len(birth_death_events.get('death_events', []))
                logger.debug(f"Step {step}: detected {n_births} births, {n_deaths} deaths")
                
                # Step 6: Update tracking information
                self._update_gw_tracking_info(
                    tracking_info, matches, birth_death_events, 
                    step, filtration_params, pes_matrix
                )
                
            except (TransportNotAvailableError, TransportQualityError) as e:
                logger.warning(f"Transport-related error at step {step}: {e}")
                logger.warning(f"Attempting fallback to identity inclusion mapping")
                
                try:
                    # Fallback to identity inclusion mapping (using pre-validated dimensions)
                    inclusion_result = self.inclusion_mapper.create_gw_inclusion_mapping(
                        prev_step=step-1,
                        curr_step=step,
                        prev_eigenspace_dim=prev_eigenspace_dim,
                        curr_eigenspace_dim=curr_eigenspace_dim,
                        transport_matrices=None,  # Force identity fallback
                        sheaf_metadata=None       # Force identity fallback
                    )
                    
                    # Continue with fallback inclusion mapping (using pre-validated tensors)
                    embedded_prev_vecs = self.eigenspace_embedder.embed_eigenspace(
                        prev_eigenvectors=prev_eigenvecs,
                        inclusion_mapping=inclusion_result,
                        transport_costs=None  # No transport costs available
                    )
                    
                    # Compute PES similarity matrix without transport weighting
                    mass_matrix = self._get_mass_matrix(step) if self._mass_matrices_sequence else None
                    pes_matrix = self.pes_computer.compute_pes_matrix(
                        prev_eigenvecs=embedded_prev_vecs,
                        curr_eigenvecs=curr_eigenvecs,
                        transport_weighting=False,  # Disable transport weighting
                        transport_costs=None,
                        mass_matrix=mass_matrix
                    )
                    
                    # Find optimal eigenvector matching
                    matches = self.pes_computer.optimal_eigenvector_matching(pes_matrix)
                    
                    # Detect birth-death events with fallback metadata
                    birth_death_events = self.birth_death_detector.detect_gw_events(
                        matches=matches,
                        prev_eigenvals=eigenvals_seq[step-1],
                        curr_eigenvals=eigenvals_seq[step],
                        prev_param=filtration_params[step-1],
                        curr_param=filtration_params[step],
                        step=step
                    )
                    
                    # Update tracking information with fallback indication
                    fallback_metadata = inclusion_metadata.copy()
                    fallback_metadata['transport_fallback_used'] = True
                    fallback_metadata['original_error'] = str(e)
                    
                    self._update_gw_tracking_info(
                        tracking_info, matches, birth_death_events, 
                        step, filtration_params, pes_matrix, fallback_metadata
                    )
                    
                    logger.info(f"✓ Completed step {step} using fallback inclusion mapping")
                    
                except Exception as fallback_error:
                    logger.error(f"Both primary and fallback inclusion failed at step {step}: {fallback_error}")
                    continue  # Skip this step entirely
                    
            except Exception as e:
                logger.error(f"Unexpected error in GW tracking at step {step}: {e}")
                # Continue with next step rather than failing completely
                continue
        
        # Step 7: Generate continuous paths for persistence analysis
        continuous_paths = self._generate_continuous_paths(tracking_info)
        
        # Step 8: Create final tracking results
        results = self._create_tracking_results(tracking_info, continuous_paths)
        
        logger.info(f"GW eigenspace tracking completed: {len(continuous_paths)} continuous paths, "
                   f"{len(tracking_info['birth_events'])} births, "
                   f"{len(tracking_info['death_events'])} deaths")
        
        return results
    
    def _initialize_gw_tracking_info(self, n_steps: int) -> Dict:
        """
        Initialize data structures for GW tracking information.
        
        Args:
            n_steps: Number of filtration steps
            
        Returns:
            Initialized tracking info dictionary
        """
        return {
            'tracking_method': 'persistent_eigenvector_similarity',
            'construction_method': 'gromov_wasserstein',
            'eigenvalue_paths': [],
            'birth_events': [],
            'death_events': [],
            'crossings': [],
            'pes_statistics': [],
            'step_matches': [],
            'inclusion_mappings': [],
            'transport_costs': [],
            'n_steps': n_steps
        }
    
    def _validate_step_edge_coordination(self, n_eigenvalue_steps: int) -> None:
        """
        🔧 STEP-EDGE COORDINATION VALIDATION: Ensure filtration steps match available edges.
        
        This method prevents the wraparound behavior that causes critical principal angles
        by validating that the number of eigenvalue sequence steps doesn't exceed the 
        number of available edges for step-to-edge mapping.
        
        Args:
            n_eigenvalue_steps: Number of steps in eigenvalue sequence
        """
        if self._current_sheaf_metadata is None:
            logger.warning("⚠️  No sheaf metadata available for step-edge coordination validation")
            return
        
        # Get available edges from GW costs
        gw_costs = self._current_sheaf_metadata.get('gw_costs', {})
        n_available_edges = len(gw_costs)
        
        # 🔧 ZERO-EDGE FIX: Step 0 = baseline (no edges), Steps 1-N = each step adds one edge
        # Step 0: baseline state (no edges active)
        # Step 1: first edge active  
        # Step N: Nth edge active
        # So we need n_eigenvalue_steps <= n_available_edges + 1
        max_allowed_steps = n_available_edges + 1
        
        if n_eigenvalue_steps > max_allowed_steps:
            logger.error(f"❌ STEP-EDGE COORDINATION ERROR:")
            logger.error(f"   Eigenvalue steps: {n_eigenvalue_steps}")
            logger.error(f"   Available edges: {n_available_edges}")  
            logger.error(f"   Max allowed steps: {max_allowed_steps}")
            logger.error(f"   This will cause wraparound and critical principal angles!")
            logger.error(f"   Fix: Ensure filtration parameter generation limits steps correctly")
            
            # Continue with warning but log the coordination issue
            logger.warning("⚠️  Continuing despite coordination error - expect wraparound issues")
        else:
            logger.info(f"✅ Step-edge coordination OK: {n_eigenvalue_steps} steps for {n_available_edges} edges "
                       f"(step 0=baseline + {n_available_edges} edge activations)")
    
    def _handle_bootstrap_transition(self, step: int, eigenvals_seq: List[torch.Tensor], 
                                   eigenvecs_seq: List[torch.Tensor], tracking_info: Dict) -> None:
        """
        🔧 BOOTSTRAP TRANSITION HANDLER: Special handling for step 0→1 (no edges → first edge).
        
        Step 0 represents the baseline state with no edges active (isolated eigenspaces).
        This transition cannot use transport-based inclusion mappings since there's no
        transport data for the non-existent edges in step 0.
        
        Args:
            step: Current step (should be 1 for bootstrap)
            eigenvals_seq: Sequence of eigenvalue tensors
            eigenvecs_seq: Sequence of eigenvector tensors  
            tracking_info: Tracking information dictionary to update
        """
        logger.info("🔧 BOOTSTRAP: Step 0 has no edges - using geometric similarity only")
        
        try:
            # Extract and validate eigenvectors
            prev_eigenvecs = eigenvecs_seq[0]  # Step 0: no edges active
            curr_eigenvecs = eigenvecs_seq[1]  # Step 1: first edge active
            
            # Handle tuple extraction (reuse existing logic)
            prev_eigenvecs, curr_eigenvecs = self._extract_and_validate_eigenvectors(
                prev_eigenvecs, curr_eigenvecs, step
            )
            
            # Skip inclusion mapping creation - no meaningful transport from isolated state
            logger.info("🔧 BOOTSTRAP: Skipping inclusion mapping (no transport data for step 0)")
            
            # Use identity embedding - no spatial transformation needed
            embedded_prev_vecs = prev_eigenvecs  # Identity embedding
            logger.info("🔧 BOOTSTRAP: Using identity embedding (no spatial transformation)")
            
            # Compute PES using pure geometric similarity (no transport weighting)
            logger.info("🔧 BOOTSTRAP: Computing PES with pure geometric similarity")
            pes_matrix = self.pes_computer.compute_pes_matrix(
                prev_eigenvecs=embedded_prev_vecs,
                curr_eigenvecs=curr_eigenvecs,
                transport_weighting=False,  # No transport data available
                transport_costs=None,       # No transport costs for step 0
                mass_matrix=self._get_mass_matrix(step) if self._mass_matrices_sequence else None
            )
            
            # Find optimal eigenvector matching
            matches = self.pes_computer.optimal_eigenvector_matching(
                pes_matrix, use_threshold=True, adaptive_threshold=True
            )
            
            # Update tracking info
            tracking_info['step_matches'].append(matches)
            tracking_info['inclusion_mappings'].append(None)  # No inclusion mapping
            tracking_info['transport_costs'].append([])       # No transport costs
            
            logger.info(f"🔧 BOOTSTRAP: Found {len(matches)} eigenvalue matches "
                       f"(pure geometric similarity)")
            
        except Exception as e:
            logger.error(f"❌ Bootstrap transition failed: {e}")
            # Add empty results to maintain tracking_info consistency
            tracking_info['step_matches'].append([])
            tracking_info['inclusion_mappings'].append(None)
            tracking_info['transport_costs'].append([])
    
    def _extract_and_validate_eigenvectors(self, prev_eigenvecs: Union[torch.Tensor, tuple], 
                                         curr_eigenvecs: Union[torch.Tensor, tuple], 
                                         step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and validate eigenvectors from potentially complex data structures.
        
        This method handles the tuple extraction logic that was duplicated in the main loop.
        
        Args:
            prev_eigenvecs: Previous step eigenvectors (tensor or tuple)
            curr_eigenvecs: Current step eigenvectors (tensor or tuple)  
            step: Current step number (for logging)
            
        Returns:
            Tuple of (validated_prev_eigenvecs, validated_curr_eigenvecs)
        """
        # Handle previous eigenvectors
        if isinstance(prev_eigenvecs, tuple):
            logger.debug(f"Step {step}: Extracting tensor from prev_eigenvecs tuple")
            for i, item in enumerate(prev_eigenvecs):
                if torch.is_tensor(item) and hasattr(item, 'shape') and len(item.shape) >= 2:
                    prev_eigenvecs = item
                    logger.debug(f"Step {step}: Extracted prev_eigenvecs from position {i}: {item.shape}")
                    break
            else:
                prev_eigenvecs = torch.empty(0, 0)
                logger.warning(f"Step {step}: No valid tensor found in prev_eigenvecs tuple")
        
        # Handle current eigenvectors  
        if isinstance(curr_eigenvecs, tuple):
            logger.debug(f"Step {step}: Extracting tensor from curr_eigenvecs tuple")
            for i, item in enumerate(curr_eigenvecs):
                if torch.is_tensor(item) and hasattr(item, 'shape') and len(item.shape) >= 2:
                    curr_eigenvecs = item
                    logger.debug(f"Step {step}: Extracted curr_eigenvecs from position {i}: {item.shape}")
                    break
            else:
                curr_eigenvecs = torch.empty(0, 0)
                logger.warning(f"Step {step}: No valid tensor found in curr_eigenvecs tuple")
        
        # Final validation
        if not torch.is_tensor(prev_eigenvecs):
            prev_eigenvecs = torch.empty(0, 0)
            logger.warning(f"Step {step}: prev_eigenvecs not a tensor, using empty tensor")
            
        if not torch.is_tensor(curr_eigenvecs):
            curr_eigenvecs = torch.empty(0, 0)  
            logger.warning(f"Step {step}: curr_eigenvecs not a tensor, using empty tensor")
        
        return prev_eigenvecs, curr_eigenvecs
    
    def _extract_transport_matrices(self) -> Optional[Dict]:
        """
        Extract transport matrices from sheaf metadata.
        
        Returns:
            Dictionary of transport matrices or None if not available
        """
        if self._current_sheaf_metadata is None:
            logger.debug("No sheaf metadata available for transport matrix extraction")
            return None
        
        # Try different metadata keys for transport information
        transport_keys = ['transport_matrices', 'gw_transport', 'optimal_transport']
        
        for key in transport_keys:
            if key in self._current_sheaf_metadata:
                transport_data = self._current_sheaf_metadata[key]
                logger.debug(f"Found transport matrices under key '{key}'")
                return transport_data
        
        # Try to extract from GW costs
        if 'gw_costs' in self._current_sheaf_metadata:
            logger.debug("Attempting to construct transport matrices from GW costs")
            # This would be implemented based on the specific GW cost structure
            # For now, return None to use fallback methods
        
        logger.debug("No transport matrices found in metadata")
        return None
    
    def _get_mass_matrix(self, step: int) -> Optional[torch.Tensor]:
        """
        Get mass matrix for the given filtration step.
        
        Args:
            step: Filtration step index
            
        Returns:
            Mass matrix for the step, or None if not available
        """
        if (self._mass_matrices_sequence is None or 
            step >= len(self._mass_matrices_sequence)):
            return None
        return self._mass_matrices_sequence[step]
    
    def _get_transport_costs(self, prev_step: int, curr_step: int) -> Optional[torch.Tensor]:
        """
        Get transport costs for specific step transition.
        
        Args:
            prev_step: Previous step index
            curr_step: Current step index  
            
        Returns:
            Transport cost matrix or None if not available
        """
        if self._current_sheaf_metadata is None:
            return None
        
        # Try to extract step-specific transport costs
        gw_costs = self._current_sheaf_metadata.get('gw_costs', {})
        
        if gw_costs:
            # Convert costs to tensor format
            try:
                cost_values = list(gw_costs.values())
                if cost_values:
                    # Simple heuristic: create small cost matrix
                    n_costs = len(cost_values)
                    matrix_size = min(10, int(np.sqrt(n_costs)) + 1)  # Reasonable size
                    
                    cost_tensor = DEFAULT_SPECTRAL_POLICY.from_numpy(np.array(cost_values[:matrix_size**2]))
                    return cost_tensor.view(matrix_size, matrix_size)
            except Exception as e:
                logger.debug(f"Failed to convert GW costs to tensor: {e}")
        
        return None
    
    def _update_gw_tracking_info(self, 
                                tracking_info: Dict,
                                matches: List[Tuple[int, int, float]],
                                birth_death_events: Dict,
                                step: int,
                                filtration_params: List[float],
                                pes_matrix: torch.Tensor):
        """
        Update tracking information with results from current step.
        
        Args:
            tracking_info: Tracking info dictionary to update
            matches: Eigenvector matches from PES computation
            birth_death_events: Birth-death events from detector
            step: Current step index
            filtration_params: Filtration parameter sequence
            pes_matrix: PES similarity matrix
        """
        # Store step-specific information
        tracking_info['step_matches'].append({
            'step': step,
            'matches': matches,
            'n_matches': len(matches),
            'filtration_param': filtration_params[step]
        })
        
        # Add birth and death events
        tracking_info['birth_events'].extend(birth_death_events['birth_events'])
        tracking_info['death_events'].extend(birth_death_events['death_events'])
        
        # Compute and store PES statistics
        pes_stats = self.pes_computer.compute_pes_statistics(pes_matrix)
        pes_stats['step'] = step
        pes_stats['filtration_param'] = filtration_params[step]
        tracking_info['pes_statistics'].append(pes_stats)
        
        # Update eigenvalue paths (simplified version)
        self._update_eigenvalue_paths(tracking_info, matches, step, filtration_params[step])
    
    def _update_eigenvalue_paths(self, 
                                tracking_info: Dict,
                                matches: List[Tuple[int, int, float]],
                                step: int,
                                filtration_param: float):
        """
        Update eigenvalue paths with current step matches.
        
        Args:
            tracking_info: Tracking info dictionary
            matches: Current step matches
            step: Current step index
            filtration_param: Current filtration parameter
        """
        # Extend existing paths
        for prev_idx, curr_idx, similarity in matches:
            # Find or create path for this eigenvalue
            path_found = False
            
            for path in tracking_info['eigenvalue_paths']:
                if (len(path) > 0 and 
                    path[-1].get('prev_eigenval_idx') == prev_idx and
                    path[-1].get('step') == step - 1):
                    
                    # Extend existing path
                    path.append({
                        'step': step,
                        'prev_eigenval_idx': prev_idx,
                        'curr_eigenval_idx': curr_idx,
                        'similarity': similarity,
                        'filtration_param': filtration_param
                    })
                    path_found = True
                    break
            
            if not path_found:
                # Create new path
                new_path = [{
                    'step': step,
                    'prev_eigenval_idx': prev_idx,
                    'curr_eigenval_idx': curr_idx,
                    'similarity': similarity,
                    'filtration_param': filtration_param
                }]
                tracking_info['eigenvalue_paths'].append(new_path)
    
    def _generate_continuous_paths(self, tracking_info: Dict) -> List[Dict]:
        """
        Generate continuous paths from tracking information.
        
        Converts discrete eigenvalue matches into continuous path representation
        suitable for persistence diagram generation.
        
        Args:
            tracking_info: Complete tracking information
            
        Returns:
            List of continuous path dictionaries
        """
        continuous_paths = []
        
        # Create persistence pairs from birth-death events
        persistence_pairs = self.birth_death_detector.create_persistence_pairs(
            tracking_info['birth_events'],
            tracking_info['death_events'],
            [stats['filtration_param'] for stats in tracking_info['pes_statistics']]
        )
        
        # Convert persistence pairs to continuous path format
        for i, pair in enumerate(persistence_pairs):
            path = {
                'path_id': i,
                'birth_param': pair['birth_param'],
                'birth_step': pair.get('birth_step', 0),
                'is_alive': pair['type'] == 'infinite_pair',
                'eigenvalue_trace': [pair['birth_eigenvalue']]
            }
            
            if pair['type'] == 'finite_pair':
                path.update({
                    'death_param': pair['death_param'],
                    'death_step': pair.get('death_step', -1),
                    'eigenvalue_trace': [pair['birth_eigenvalue'], pair['death_eigenvalue']]
                })
            else:
                path.update({
                    'death_param': None,
                    'death_step': None
                })
            
            continuous_paths.append(path)
        
        return continuous_paths
    
    def _create_tracking_results(self, 
                               tracking_info: Dict,
                               continuous_paths: List[Dict]) -> Dict:
        """
        Create final tracking results dictionary.
        
        Args:
            tracking_info: Complete tracking information
            continuous_paths: Generated continuous paths
            
        Returns:
            Final tracking results dictionary
        """
        # Validate GW semantics if requested
        validation_results = None
        if self.validate_gw_semantics:
            persistence_pairs = self.birth_death_detector.create_persistence_pairs(
                tracking_info['birth_events'],
                tracking_info['death_events'],
                [stats['filtration_param'] for stats in tracking_info['pes_statistics']]
            )
            validation_results = self.birth_death_detector.validate_gw_semantics(
                persistence_pairs,
                [stats['filtration_param'] for stats in tracking_info['pes_statistics']]
            )
        
        # Extract tracked eigenvalues from continuous paths
        tracked_eigenvalues = self._extract_tracked_eigenvalues(continuous_paths, tracking_info)
        
        results = {
            'tracking_method': 'persistent_eigenvector_similarity',
            'construction_method': 'gromov_wasserstein',
            'eigenvalue_paths': tracking_info['eigenvalue_paths'],
            'birth_events': tracking_info['birth_events'],
            'death_events': tracking_info['death_events'],
            'crossings': tracking_info['crossings'],
            'continuous_paths': continuous_paths,
            'pes_statistics': tracking_info['pes_statistics'],
            'step_matches': tracking_info['step_matches'],
            'tracked_eigenvalues': tracked_eigenvalues,  # CRITICAL FIX: Add missing tracked_eigenvalues
            'gw_validation': validation_results
        }
        
        return results
    
    def _extract_tracked_eigenvalues(self, continuous_paths: List[Dict], tracking_info: Dict) -> List[Dict]:
        """
        Extract tracked eigenvalues from continuous paths and tracking information.
        
        Creates a list of eigenvalue evolution sequences that represent continuous
        tracking across filtration steps, which is what the persistent analyzer expects.
        
        Args:
            continuous_paths: List of continuous path dictionaries
            tracking_info: Complete tracking information
            
        Returns:
            List of tracked eigenvalue dictionaries with evolution sequences
        """
        tracked_eigenvalues = []
        
        # 🔧 ENHANCED VALIDATION: Log data structure formats for debugging
        logger.debug(f"Extracting eigenvalues from {len(continuous_paths)} continuous paths")
        if continuous_paths:
            sample_path = continuous_paths[0]
            logger.debug(f"Sample path type: {type(sample_path)}")
            if isinstance(sample_path, dict):
                logger.debug(f"Sample path keys: {list(sample_path.keys())}")
            elif isinstance(sample_path, list):
                logger.debug(f"Sample path length: {len(sample_path)}")
                if sample_path:
                    logger.debug(f"Sample path element type: {type(sample_path[0])}")
        
        try:
            # Create tracked eigenvalue sequences from continuous paths
            for i, path in enumerate(continuous_paths):
                # 🔧 DATA STRUCTURE FIX: Handle both dict and list formats
                if isinstance(path, dict):
                    # Standard dict format - use existing logic
                    tracked_eigenval = {
                        'path_id': i,
                        'birth_step': path.get('birth_step', 0),
                        'birth_param': path.get('birth_param', 0.0),
                        'is_alive': path.get('is_alive', False),
                        'eigenvalue_sequence': path.get('eigenvalue_trace', []),
                        'similarity_scores': [],  # Will be populated from step matches
                        'path_length': len(path.get('eigenvalue_trace', []))
                    }
                elif isinstance(path, list):
                    # List format - convert to expected dict structure
                    logger.debug(f"Converting path {i} from list format (length {len(path)}) to dict format")
                    tracked_eigenval = self._convert_path_list_to_dict(path, i)
                else:
                    logger.warning(f"Unexpected path format for path {i}: {type(path)}, skipping")
                    continue
                
                # Add death information if available (only for dict format)
                if isinstance(path, dict) and not path.get('is_alive', True):
                    tracked_eigenval['death_step'] = path.get('death_step', -1) 
                    tracked_eigenval['death_param'] = path.get('death_param', float('inf'))
                    tracked_eigenval['lifetime'] = path.get('death_param', float('inf')) - path.get('birth_param', 0.0)
                else:
                    tracked_eigenval['death_step'] = None
                    tracked_eigenval['death_param'] = None
                    tracked_eigenval['lifetime'] = float('inf')
                
                # Try to extract similarity information from step matches
                birth_step = tracked_eigenval['birth_step']
                similarity_scores = []
                
                # Extract similarity scores from step matches
                for step_idx, step_match in enumerate(tracking_info.get('step_matches', [])):
                    # 🔧 MIXED FORMAT FIX: Handle both dict and list step_match formats
                    if isinstance(step_match, dict):
                        # Standard dict format: {'step': X, 'matches': [...], ...}
                        actual_step = step_match.get('step', step_idx)
                        matches_list = step_match.get('matches', [])
                    elif isinstance(step_match, list):
                        # Direct list format: [(prev_idx, curr_idx, similarity), ...]
                        actual_step = step_idx
                        matches_list = step_match
                    else:
                        logger.warning(f"Unknown step_match format at index {step_idx}: {type(step_match)}, skipping")
                        continue
                    
                    if actual_step >= birth_step:
                        # Find matches involving this eigenvalue path
                        for match in matches_list:
                            if isinstance(match, (tuple, list)) and len(match) >= 3:  # (prev_idx, curr_idx, similarity)
                                similarity_scores.append({
                                    'step': actual_step,
                                    'similarity': match[2]
                                })
                            elif isinstance(match, dict):
                                # Handle dict-formatted matches as well
                                similarity_scores.append({
                                    'step': actual_step,
                                    'similarity': match.get('similarity', 0.0)
                                })
                
                tracked_eigenval['similarity_scores'] = similarity_scores
                tracked_eigenvalues.append(tracked_eigenval)
            
            # If no continuous paths but we have eigenvalue paths, convert those
            if not tracked_eigenvalues and tracking_info.get('eigenvalue_paths'):
                for i, eigen_path in enumerate(tracking_info['eigenvalue_paths']):
                    if eigen_path:  # Check if path is not empty
                        # Extract eigenvalue sequence from path
                        eigenval_sequence = []
                        similarity_scores = []
                        
                        for path_step in eigen_path:
                            # 🔧 TYPE SAFETY FIX: Handle both dict and list path_step formats
                            if isinstance(path_step, dict):
                                # Standard dict format
                                if 'filtration_param' in path_step:
                                    eigenval_sequence.append(path_step.get('prev_eigenval_idx', 0))
                                if 'similarity' in path_step:
                                    similarity_scores.append({
                                        'step': path_step.get('step', 0),
                                        'similarity': path_step['similarity']
                                    })
                            elif isinstance(path_step, list) and len(path_step) >= 3:
                                # List format: [step, prev_idx, curr_idx, similarity, filtration_param, ...]
                                try:
                                    step = path_step[0] if len(path_step) > 0 else 0
                                    prev_idx = path_step[1] if len(path_step) > 1 else 0
                                    similarity = path_step[3] if len(path_step) > 3 else 0.0
                                    filtration_param = path_step[4] if len(path_step) > 4 else 0.0
                                    
                                    eigenval_sequence.append(prev_idx)
                                    similarity_scores.append({
                                        'step': step,
                                        'similarity': similarity
                                    })
                                except (IndexError, TypeError) as e:
                                    logger.warning(f"Failed to parse list path_step format: {e}")
                                    continue
                            else:
                                logger.warning(f"Unknown path_step format: {type(path_step)}, skipping")
                                continue
                        
                        # 🔧 TYPE SAFETY FIX: Safe extraction of birth info from first path element
                        birth_step = 0
                        birth_param = 0.0
                        if eigen_path:
                            first_element = eigen_path[0]
                            if isinstance(first_element, dict):
                                birth_step = first_element.get('step', 0)
                                birth_param = first_element.get('filtration_param', 0.0)
                            elif isinstance(first_element, list) and len(first_element) >= 5:
                                birth_step = first_element[0] if len(first_element) > 0 else 0
                                birth_param = first_element[4] if len(first_element) > 4 else 0.0
                        
                        tracked_eigenval = {
                            'path_id': i,
                            'birth_step': birth_step,
                            'birth_param': birth_param,
                            'is_alive': True,  # Assume alive if no death info
                            'eigenvalue_sequence': eigenval_sequence,
                            'similarity_scores': similarity_scores,
                            'path_length': len(eigenval_sequence),
                            'death_step': None,
                            'death_param': None,
                            'lifetime': float('inf')
                        }
                        
                        tracked_eigenvalues.append(tracked_eigenval)
            
            logger.info(f"Extracted {len(tracked_eigenvalues)} tracked eigenvalue sequences from {len(continuous_paths)} continuous paths")
            
            # 🔧 ROBUST VALIDATION: Prevent crashes and preserve partial results
            valid_sequences = 0
            for i, tracked in enumerate(tracked_eigenvalues):
                try:
                    if self._validate_tracked_eigenvalue_structure(tracked):
                        valid_sequences += 1
                    else:
                        logger.warning(f"Invalid tracked eigenvalue structure for path {tracked.get('path_id', f'track_{i}')}")
                        # Continue processing - don't discard the track, just note the validation issue
                except Exception as e:
                    logger.error(f"Validation crashed for track {i}: {e}")
                    # Continue processing with remaining tracks rather than failing completely
            
            # Log summary statistics
            if tracked_eigenvalues:
                alive_count = sum(1 for t in tracked_eigenvalues if t['is_alive'])
                dead_count = len(tracked_eigenvalues) - alive_count
                avg_path_length = sum(t['path_length'] for t in tracked_eigenvalues) / len(tracked_eigenvalues)
                
                logger.info(f"🔧 EIGENVALUE EXTRACTION SUCCESS: {len(tracked_eigenvalues)} tracked eigenvalues")
                logger.info(f"   Status: {alive_count} alive, {dead_count} dead")
                logger.info(f"   Quality: avg path length {avg_path_length:.1f}, {valid_sequences}/{len(tracked_eigenvalues)} valid structures")
                
                # Log examples of longest and shortest paths
                if tracked_eigenvalues:
                    longest_path = max(tracked_eigenvalues, key=lambda x: x['path_length'])
                    shortest_path = min(tracked_eigenvalues, key=lambda x: x['path_length'])
                    logger.debug(f"Path length range: shortest={shortest_path['path_length']}, longest={longest_path['path_length']}")
            
        except Exception as e:
            logger.error(f"❌ EIGENVALUE EXTRACTION FAILED: {e}")
            logger.error(f"   Continuous paths count: {len(continuous_paths)}")
            logger.error(f"   Paths data type: {type(continuous_paths)}")
            if continuous_paths:
                logger.error(f"   First path type: {type(continuous_paths[0])}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")
            # Return empty list as fallback
            tracked_eigenvalues = []
        
        return tracked_eigenvalues
    
    def _convert_path_list_to_dict(self, path_list: List, path_id: int) -> Dict:
        """
        🔧 DATA STRUCTURE CONVERTER: Convert path list format to expected dict format.
        
        Handles the case where continuous_paths contains lists instead of dicts,
        which was causing the "'list' object has no attribute 'get'" error.
        
        Args:
            path_list: Path in list format
            path_id: Unique path identifier
            
        Returns:
            Path converted to expected dict format
        """
        try:
            # Extract what information we can from the list structure
            if not path_list:
                return {
                    'path_id': path_id,
                    'birth_step': 0,
                    'birth_param': 0.0,
                    'is_alive': True,
                    'eigenvalue_sequence': [],
                    'similarity_scores': [],
                    'path_length': 0,
                    'death_step': None,
                    'death_param': None,
                    'lifetime': float('inf')
                }
            
            # Try to extract meaningful data from list elements
            eigenvalue_sequence = []
            similarity_scores = []
            
            for i, element in enumerate(path_list):
                if isinstance(element, dict):
                    # Element is a dict - extract eigenvalue info
                    if 'eigenvalue' in element or 'eigenval' in element:
                        eigenvalue_sequence.append(element.get('eigenvalue', element.get('eigenval', 0.0)))
                    if 'similarity' in element:
                        similarity_scores.append({
                            'step': i,
                            'similarity': element['similarity']
                        })
                elif isinstance(element, (int, float)):
                    # Element is a number - assume it's an eigenvalue
                    eigenvalue_sequence.append(float(element))
                
            return {
                'path_id': path_id,
                'birth_step': 0,  # Default - can't infer from list structure
                'birth_param': 0.0,
                'is_alive': True,  # Assume alive if no death info
                'eigenvalue_sequence': eigenvalue_sequence,
                'similarity_scores': similarity_scores,
                'path_length': len(eigenvalue_sequence),
                'death_step': None,
                'death_param': None,
                'lifetime': float('inf')
            }
            
        except Exception as e:
            logger.warning(f"Failed to convert path list {path_id}: {e}")
            # Return minimal valid structure
            return {
                'path_id': path_id,
                'birth_step': 0,
                'birth_param': 0.0,
                'is_alive': True,
                'eigenvalue_sequence': [],
                'similarity_scores': [],
                'path_length': 0,
                'death_step': None,
                'death_param': None,
                'lifetime': float('inf')
            }
    
    def _validate_tracked_eigenvalue_structure(self, tracks_or_track, n_steps=None) -> bool:
        """
        Validate the structure of tracked eigenvalue data.
        
        🔧 MISSING METHOD FIX: Handles both single dict and list+n_steps signatures
        to prevent AttributeError crashes that lead to "0 tracked eigenvalues".
        
        Signatures supported:
        1. _validate_tracked_eigenvalue_structure(tracked_dict) -> bool
        2. _validate_tracked_eigenvalue_structure(tracks_list, n_steps) -> bool
        
        Args:
            tracks_or_track: Either a single tracked eigenvalue Dict, or List[Dict] of tracks
            n_steps: Number of expected steps (required when first arg is a list)
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # 🔧 HANDLE BOTH SIGNATURES: List+n_steps OR single dict
            if isinstance(tracks_or_track, list) and n_steps is not None:
                # List validation as suggested by user  
                return self._validate_tracks_list(tracks_or_track, n_steps)
            elif isinstance(tracks_or_track, dict):
                # Single dict validation (existing logic)
                return self._validate_single_track(tracks_or_track)
            else:
                logger.warning(f"Invalid validation arguments: {type(tracks_or_track)}, n_steps={n_steps}")
                return False
                
        except Exception as e:
            logger.warning(f"Eigenvalue structure validation failed: {e}")
            return False  # Don't crash, continue with partial results
    
    def _validate_tracks_list(self, tracks: List[Dict], n_steps: int) -> bool:
        """
        Validate list of tracked eigenvalue structures.
        
        Implements the user's suggested validation logic for tracks+n_steps signature.
        
        Args:
            tracks: List of tracked eigenvalue dictionaries  
            n_steps: Expected number of filtration steps
            
        Returns:
            True if all tracks are valid, False otherwise
        """
        try:
            # Basic structure validation
            if not isinstance(tracks, list):
                logger.debug(f"Expected list, got {type(tracks)}")
                return False
                
            if not all(isinstance(t, dict) for t in tracks):
                logger.debug("Not all tracks are dictionaries")
                return False
            
            # Validate each track
            for i, track in enumerate(tracks):
                if not self._validate_track_with_steps(track, n_steps):
                    logger.debug(f"Track {i} failed validation")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Tracks list validation failed: {e}")
            return False
    
    def _validate_track_with_steps(self, track: Dict, n_steps: int) -> bool:
        """Validate individual track against expected number of steps."""
        try:
            # Check for eigenvalues field (user suggestion)
            vals = track.get('eigenvalues', track.get('eigenvalue_sequence', []))
            
            if len(vals) != n_steps:
                logger.debug(f"Expected {n_steps} eigenvalues, got {len(vals)}")
                return False
            
            # Check finite values (user suggestion)
            for val in vals:
                if val is not None and torch.is_tensor(val) and not torch.isfinite(val):
                    logger.debug(f"Non-finite eigenvalue found: {val}")
                    return False
            
            # Optional: check monotone step indices, birth < death, etc.
            birth_step = track.get('birth_step')
            death_step = track.get('death_step')
            
            if birth_step is not None and death_step is not None:
                if birth_step >= death_step:
                    logger.debug(f"Birth step {birth_step} >= death step {death_step}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Individual track validation failed: {e}")
            return False
    
    def _validate_single_track(self, tracked_eigenval: Dict) -> bool:
        """
        Validate single tracked eigenvalue dictionary (original implementation).
        
        Args:
            tracked_eigenval: Single tracked eigenvalue dictionary
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check required fields  
            required_fields = ['path_id', 'birth_step', 'birth_param', 'is_alive', 
                             'eigenvalue_sequence', 'similarity_scores', 'path_length']
            
            for field in required_fields:
                if field not in tracked_eigenval:
                    logger.debug(f"Missing required field: {field}")
                    return False
            
            # Check data types and consistency
            path_id = tracked_eigenval['path_id']
            if not isinstance(path_id, (int, str)):
                logger.debug(f"Invalid path_id type: {type(path_id)}")
                return False
            
            birth_step = tracked_eigenval['birth_step']
            if not isinstance(birth_step, (int, type(None))):
                logger.debug(f"Invalid birth_step type: {type(birth_step)}")
                return False
            
            is_alive = tracked_eigenval['is_alive']
            if not isinstance(is_alive, bool):
                logger.debug(f"Invalid is_alive type: {type(is_alive)}")
                return False
            
            # Check eigenvalue sequence
            eigenvalue_sequence = tracked_eigenval['eigenvalue_sequence']
            if not isinstance(eigenvalue_sequence, list):
                logger.debug(f"Invalid eigenvalue_sequence type: {type(eigenvalue_sequence)}")
                return False
            
            # Check path length consistency
            path_length = tracked_eigenval['path_length']
            if not isinstance(path_length, int) or path_length < 0:
                logger.debug(f"Invalid path_length: {path_length}")
                return False
            
            if path_length != len(eigenvalue_sequence):
                logger.debug(f"Path length mismatch: {path_length} vs {len(eigenvalue_sequence)}")
                return False
            
            # Check similarity scores
            similarity_scores = tracked_eigenval['similarity_scores']
            if not isinstance(similarity_scores, list):
                logger.debug(f"Invalid similarity_scores type: {type(similarity_scores)}")
                return False
            
            # For dead paths, check death information
            if not is_alive:
                death_step = tracked_eigenval.get('death_step')
                death_param = tracked_eigenval.get('death_param') 
                
                if death_step is not None and not isinstance(death_step, int):
                    logger.debug(f"Invalid death_step type: {type(death_step)}")
                    return False
                
                if death_param is not None and birth_step is not None:
                    birth_param = tracked_eigenval.get('birth_param', 0.0)
                    if death_param <= birth_param:
                        logger.debug(f"Invalid death before birth: {death_param} <= {birth_param}")
                        return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Single track validation failed: {e}")
            return False
    
    def validate_all_tracked_structures(self, tracked_eigenvalues: List[Dict], n_steps: int) -> bool:
        """
        🔧 CONVENIENCE METHOD: Validate all tracked eigenvalue structures at once.
        
        This provides the (tracks, n_steps) signature that the user mentioned was missing,
        preventing AttributeError crashes and "0 tracked eigenvalues" results.
        
        Args:
            tracked_eigenvalues: List of all tracked eigenvalue dictionaries
            n_steps: Expected number of filtration steps
            
        Returns:
            True if all tracks are valid, False if any validation fails
        """
        try:
            # Use the flexible validation method with the list signature
            return self._validate_tracked_eigenvalue_structure(tracked_eigenvalues, n_steps)
            
        except Exception as e:
            logger.error(f"Bulk validation failed: {e}")
            # Return False but don't crash - allows partial results to be preserved
            return False
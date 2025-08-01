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
from .sheaf_inclusion_mapper import SheafInclusionMapper
from .gw_eigenspace_embedder import GWEigenspaceEmbedder
from .gw_birth_death_detector import GWBirthDeathDetector
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError

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
        
        # Cache for transport matrices and metadata
        self._transport_cache = {}
        self._current_sheaf_metadata = None
        
        logger.info(f"GWSubspaceTracker initialized: PES threshold={pes_threshold}, "
                   f"inclusion method={inclusion_method}, transport weighting={transport_weighting}")
    
    def track_eigenspaces(self, 
                         eigenvalues_sequence: List[torch.Tensor],
                         eigenvectors_sequence: List[torch.Tensor],
                         filtration_params: List[float],
                         construction_method: str = 'gromov_wasserstein',
                         sheaf_metadata: Optional[Dict] = None) -> Dict:
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
            
        Returns:
            Dictionary with tracking information including continuous paths
        """
        if construction_method != 'gromov_wasserstein':
            # Fallback to parent implementation for non-GW methods
            logger.debug(f"Non-GW construction method '{construction_method}', using parent tracker")
            return super().track_eigenspaces(
                eigenvalues_sequence, eigenvectors_sequence, 
                filtration_params, construction_method
            )
        
        # Store metadata for use in tracking components
        self._current_sheaf_metadata = sheaf_metadata
        
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
        
        # Process each consecutive pair of filtration steps
        for step in range(1, len(eigenvals_seq)):
            logger.debug(f"Processing GW tracking step {step}/{len(eigenvals_seq)-1}")
            
            try:
                # Step 1: Create transport-informed inclusion mapping
                inclusion_map = self.inclusion_mapper.create_gw_inclusion_mapping(
                    prev_step=step-1,
                    curr_step=step,
                    prev_eigenspace_dim=eigenvecs_seq[step-1].shape[0],
                    curr_eigenspace_dim=eigenvecs_seq[step].shape[0],
                    transport_matrices=transport_matrices,
                    sheaf_metadata=self._current_sheaf_metadata
                )
                
                # Step 2: Embed previous eigenspace into current space
                embedded_prev_vecs = self.eigenspace_embedder.embed_eigenspace(
                    prev_eigenvectors=eigenvecs_seq[step-1],
                    inclusion_mapping=inclusion_map,
                    transport_costs=self._get_transport_costs(step-1, step)
                )
                
                # Step 3: Compute PES similarity matrix
                pes_matrix = self.pes_computer.compute_pes_matrix(
                    prev_eigenvecs=embedded_prev_vecs,
                    curr_eigenvecs=eigenvecs_seq[step],
                    transport_weighting=self.transport_weighting,
                    transport_costs=self._get_transport_costs(step-1, step)
                )
                
                # Step 4: Find optimal eigenvector matching
                matches = self.pes_computer.optimal_eigenvector_matching(pes_matrix)
                
                # Step 5: Detect GW-specific birth-death events
                birth_death_events = self.birth_death_detector.detect_gw_events(
                    matches=matches,
                    prev_eigenvals=eigenvals_seq[step-1],
                    curr_eigenvals=eigenvals_seq[step],
                    prev_param=filtration_params[step-1],
                    curr_param=filtration_params[step],
                    step=step
                )
                
                # Step 6: Update tracking information
                self._update_gw_tracking_info(
                    tracking_info, matches, birth_death_events, 
                    step, filtration_params, pes_matrix
                )
                
            except Exception as e:
                logger.error(f"GW tracking failed at step {step}: {e}")
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
                    
                    cost_tensor = torch.tensor(cost_values[:matrix_size**2], dtype=torch.float32)
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
            'gw_validation': validation_results
        }
        
        return results
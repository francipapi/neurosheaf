# neurosheaf/spectral/tracker.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.linalg import subspace_angles
import networkx as nx
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError

logger = setup_logger(__name__)

class SubspaceTracker:
    """Track eigenspace evolution through filtration using subspace similarity.
    
    Extended to handle different construction methods with appropriate
    filtration semantics and parameter interpretation.
    """
    
    def __init__(self, 
                 gap_eps: float = 1e-6,
                 cos_tau: float = 0.80,
                 max_groups: int = 100):
        """Initialize SubspaceTracker.
        
        Args:
            gap_eps: Threshold for eigenvalue grouping (handles degeneracies)
            cos_tau: Cosine similarity threshold for eigenspace matching
            max_groups: Maximum number of eigenvalue groups to track
        """
        self.gap_eps = gap_eps
        self.cos_tau = cos_tau
        self.max_groups = max_groups
        
        # Method-specific handlers for different construction approaches
        self._method_handlers = {
            'gromov_wasserstein': self._track_gw_subspaces,
            'standard': self._track_standard_subspaces,
            'scaled_procrustes': self._track_standard_subspaces,
            'whitened_procrustes': self._track_standard_subspaces,
            'fx_unified_whitened': self._track_standard_subspaces
        }
        
        logger.info(f"SubspaceTracker initialized with gap_eps={gap_eps}, cos_tau={cos_tau}")
        
    def track_eigenspaces(self,
                         eigenvalues_sequence: List[torch.Tensor],
                         eigenvectors_sequence: List[torch.Tensor],
                         filtration_params: List[float],
                         construction_method: str = 'standard',
                         sheaf_metadata: Optional[Dict] = None) -> Dict:
        """Track eigenspaces through filtration parameter changes.
        
        Route to appropriate tracking method based on construction method.
        Both GW and standard methods use increasing parameter sequences,
        but with different threshold semantics and interpretations.
        
        Args:
            eigenvalues_sequence: List of eigenvalue tensors for each filtration
            eigenvectors_sequence: List of eigenvector tensors for each filtration
            filtration_params: List of filtration parameter values
            construction_method: Sheaf construction method
            sheaf_metadata: Additional metadata about sheaf construction (for GW methods)
            
        Returns:
            Dictionary with tracking information including paths, birth/death events
        """
        if len(eigenvalues_sequence) != len(eigenvectors_sequence):
            raise ValueError("Eigenvalues and eigenvectors sequences must have same length")
        
        if len(eigenvalues_sequence) != len(filtration_params):
            raise ValueError("Sequences and filtration parameters must have same length")
        
        # Route to appropriate method handler
        handler = self._method_handlers.get(construction_method, self._track_standard_subspaces)
        logger.info(f"Routing to {construction_method} tracking handler")
        
        return handler(eigenvalues_sequence, eigenvectors_sequence, filtration_params)
    
    def _group_eigenvalues(self,
                          eigenvalues: torch.Tensor,
                          eigenvectors: torch.Tensor) -> List[Dict]:
        """Group eigenvalues by proximity to handle degeneracies.
        
        Args:
            eigenvalues: Tensor of eigenvalues
            eigenvectors: Tensor of eigenvectors (columns are eigenvectors)
            
        Returns:
            List of eigenvalue groups with subspace information
        """
        # Sort eigenvalues for consistent grouping
        sorted_idx = torch.argsort(eigenvalues)
        sorted_vals = eigenvalues[sorted_idx]
        sorted_vecs = eigenvectors[:, sorted_idx]
        
        groups = []
        current_group = {
            'eigenvalues': [sorted_vals[0].item()],
            'indices': [0],
            'mean_eigenvalue': sorted_vals[0],
            'subspace': sorted_vecs[:, [0]]
        }
        
        # Group eigenvalues by proximity
        for i in range(1, len(sorted_vals)):
            if abs(sorted_vals[i] - sorted_vals[i-1]) < self.gap_eps:
                # Add to current group
                current_group['eigenvalues'].append(sorted_vals[i].item())
                current_group['indices'].append(i)
                current_group['subspace'] = torch.cat([current_group['subspace'], 
                                                     sorted_vecs[:, [i]]], dim=1)
                current_group['mean_eigenvalue'] = torch.mean(torch.tensor(current_group['eigenvalues']))
            else:
                # Start new group
                groups.append(current_group)
                current_group = {
                    'eigenvalues': [sorted_vals[i].item()],
                    'indices': [i],
                    'mean_eigenvalue': sorted_vals[i],
                    'subspace': sorted_vecs[:, [i]]
                }
        
        # Add final group
        groups.append(current_group)
        
        return groups[:self.max_groups]  # Limit number of groups
    
    def _match_eigenspaces(self, 
                          prev_groups: List[Dict],
                          curr_groups: List[Dict]) -> List[Tuple[int, int, float]]:
        """Match eigenspaces between consecutive steps using principal angles.
        
        Args:
            prev_groups: Eigenvalue groups from previous step
            curr_groups: Eigenvalue groups from current step
            
        Returns:
            List of matches as (prev_idx, curr_idx, similarity) tuples
        """
        matches = []
        
        # Compute all pairwise similarities
        similarity_matrix = torch.zeros(len(prev_groups), len(curr_groups))
        
        for i, prev_group in enumerate(prev_groups):
            for j, curr_group in enumerate(curr_groups):
                similarity = self._compute_subspace_similarity(
                    prev_group['subspace'],
                    curr_group['subspace']
                )
                similarity_matrix[i, j] = similarity
        
        # Find optimal matches using Hungarian algorithm
        matches = self._optimal_assignment_matching(similarity_matrix, prev_groups, curr_groups)
        
        logger.debug(f"Matched {len(matches)} eigenspaces out of {len(prev_groups)} previous groups")
        return matches
    
    def _optimal_assignment_matching(self, 
                                   similarity_matrix: torch.Tensor,
                                   prev_groups: List[Dict],
                                   curr_groups: List[Dict]) -> List[Tuple[int, int, float]]:
        """Find optimal eigenspace matching using Hungarian algorithm.
        
        Args:
            similarity_matrix: Matrix of subspace similarities [n_prev x n_curr]
            prev_groups: Previous step eigenvalue groups
            curr_groups: Current step eigenvalue groups
            
        Returns:
            List of optimal matches as (prev_idx, curr_idx, similarity) tuples
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            logger.warning("scipy not available, falling back to greedy matching")
            return self._greedy_matching_fallback(similarity_matrix, prev_groups, curr_groups)
        
        # Convert to numpy for scipy
        sim_matrix = similarity_matrix.detach().cpu().numpy()
        
        # Hungarian algorithm minimizes cost, so we need to convert similarities to costs
        # Cost = 1 - similarity (so high similarity = low cost)
        cost_matrix = 1.0 - sim_matrix
        
        # Handle rectangular matrices by padding with high cost
        n_prev, n_curr = cost_matrix.shape
        max_dim = max(n_prev, n_curr)
        
        if n_prev != n_curr:
            # Create square matrix with high cost padding
            square_cost_matrix = np.full((max_dim, max_dim), 2.0)  # Cost > 1 means similarity < 0
            square_cost_matrix[:n_prev, :n_curr] = cost_matrix
            cost_matrix = square_cost_matrix
        
        # Solve optimal assignment
        prev_indices, curr_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid matches (within original dimensions and above threshold)
        matches = []
        for prev_idx, curr_idx in zip(prev_indices, curr_indices):
            # Only consider matches within original matrix dimensions
            if prev_idx < n_prev and curr_idx < n_curr:
                similarity = sim_matrix[prev_idx, curr_idx]
                
                # Only accept matches above similarity threshold
                if similarity > self.cos_tau:
                    matches.append((prev_idx, curr_idx, float(similarity)))
        
        # Sort matches by similarity for consistency
        matches.sort(key=lambda x: x[2], reverse=True)
        
        logger.debug(f"Optimal assignment found {len(matches)} matches above threshold {self.cos_tau}")
        return matches
    
    def _greedy_matching_fallback(self,
                                similarity_matrix: torch.Tensor,
                                prev_groups: List[Dict],
                                curr_groups: List[Dict]) -> List[Tuple[int, int, float]]:
        """Fallback greedy matching when optimal assignment is not available.
        
        This is the original greedy algorithm, kept as a fallback.
        """
        matches = []
        used_curr = set()
        
        # Sort by similarity (greedy matching)
        flat_similarities = []
        for i in range(len(prev_groups)):
            for j in range(len(curr_groups)):
                flat_similarities.append((similarity_matrix[i, j], i, j))
        
        flat_similarities.sort(reverse=True)
        
        for similarity, i, j in flat_similarities:
            if i not in [m[0] for m in matches] and j not in used_curr:
                if similarity > self.cos_tau:
                    matches.append((i, j, similarity.item()))
                    used_curr.add(j)
        
        logger.debug(f"Greedy fallback found {len(matches)} matches")
        return matches
    
    def _compute_subspace_similarity(self,
                                   subspace1: torch.Tensor,
                                   subspace2: torch.Tensor) -> float:
        """Compute similarity between subspaces using principal angles.
        
        Args:
            subspace1: First subspace (columns are basis vectors)
            subspace2: Second subspace (columns are basis vectors)
            
        Returns:
            Subspace similarity as product of cosines of principal angles
        """
        # Handle dimensional mismatches
        if subspace1.shape[0] != subspace2.shape[0]:
            logger.warning(f"Subspace dimension mismatch: {subspace1.shape[0]} vs {subspace2.shape[0]}")
            return 0.0
        
        # Convert to numpy for scipy
        Q1 = subspace1.detach().cpu().numpy()
        Q2 = subspace2.detach().cpu().numpy()
        
        # Compute principal angles
        try:
            angles = subspace_angles(Q1, Q2)
            # Similarity is product of cosines
            similarity = np.prod(np.cos(angles))
            return float(max(0.0, min(similarity, 1.0)))  # Ensure bounds [0, 1]
        except Exception as e:
            logger.warning(f"Subspace angle computation failed: {e}")
            # Fallback: use Frobenius norm of projection
            try:
                # Project Q1 onto Q2's space
                Q1_proj = Q2 @ (Q2.T @ Q1)
                # Compute relative similarity
                q1_norm = np.linalg.norm(Q1, 'fro')
                if q1_norm == 0:
                    return 0.0
                similarity = np.linalg.norm(Q1_proj, 'fro') / q1_norm
                return float(max(0.0, min(similarity, 1.0)))  # Ensure bounds [0, 1]
            except Exception as e2:
                logger.warning(f"Fallback similarity computation also failed: {e2}")
                return 0.0
    
    def _update_tracking_info(self,
                            tracking_info: Dict,
                            matching: List[Tuple[int, int, float]],
                            filtration_params: List[float],
                            step_idx: int,
                            num_prev_groups: int = None,
                            num_curr_groups: int = None):
        """Update tracking information with current matching.
        
        Args:
            tracking_info: Dictionary to update with tracking information
            matching: List of (prev_idx, curr_idx, similarity) matches
            filtration_params: Filtration parameters for [prev_step, curr_step]
            step_idx: Current step index
            num_prev_groups: Number of previous groups (for birth/death detection)
            num_curr_groups: Number of current groups (for birth/death detection)
        """
        # Record eigenvalue paths
        for prev_idx, curr_idx, similarity in matching:
            # Ensure paths list is long enough
            while len(tracking_info['eigenvalue_paths']) <= prev_idx:
                tracking_info['eigenvalue_paths'].append([])
            
            tracking_info['eigenvalue_paths'][prev_idx].append({
                'step': step_idx,
                'current_group': curr_idx,
                'similarity': similarity,
                'filtration_param': filtration_params[1]
            })
        
        # Detect birth/death events if group counts are provided
        if num_prev_groups is not None and num_curr_groups is not None:
            matched_prev = set(m[0] for m in matching)
            matched_curr = set(m[1] for m in matching)
            
            # Birth events: current groups that weren't matched
            for curr_idx in range(num_curr_groups):
                if curr_idx not in matched_curr:
                    tracking_info['birth_events'].append({
                        'step': step_idx,
                        'group': curr_idx,
                        'filtration_param': filtration_params[1]
                    })
            
            # Death events: previous groups that weren't matched
            for prev_idx in range(num_prev_groups):
                if prev_idx not in matched_prev:
                    tracking_info['death_events'].append({
                        'step': step_idx - 1,
                        'group': prev_idx,
                        'filtration_param': filtration_params[0]
                    })
        
        # Detect crossings based on eigenvalue ordering changes
        # This is a simplified version - could be enhanced
        if len(matching) > 1:
            # Check if order has changed
            prev_order = [m[0] for m in matching]
            curr_order = [m[1] for m in matching]
            
            if prev_order != sorted(prev_order) or curr_order != sorted(curr_order):
                tracking_info['crossings'].append({
                    'step': step_idx,
                    'filtration_param': filtration_params[1],
                    'matching': matching
                })
    
    def _update_paths_from_matching(self, active_paths: List[Dict], matching: List[Tuple],
                                  curr_groups: List[Dict], step_idx: int, 
                                  filtration_param: float, tracking_info: Dict):
        """Update active paths based on eigenspace matching results.
        
        Args:
            active_paths: List of currently active paths
            matching: List of (prev_group_idx, curr_group_idx, similarity) tuples
            curr_groups: Current eigenvalue groups
            step_idx: Current step index
            filtration_param: Current filtration parameter value
            tracking_info: Tracking information dictionary
        """
        matched_prev = set()
        matched_curr = set()
        
        # Continue existing paths that have matches
        for prev_idx, curr_idx, similarity in matching:
            matched_prev.add(prev_idx)
            matched_curr.add(curr_idx)
            
            # Find the active path corresponding to prev_idx
            for path in active_paths:
                if (path['is_alive'] and 
                    len(path['group_trace']) > 0 and 
                    path['group_trace'][-1] == prev_idx):
                    
                    # Continue this path
                    path['eigenvalue_trace'].append(curr_groups[curr_idx]['mean_eigenvalue'].item())
                    path['step_trace'].append(step_idx)
                    path['param_trace'].append(filtration_param)
                    path['group_trace'].append(curr_idx)
                    break
        
        # Kill paths that have no match (death events)
        for path in active_paths:
            if (path['is_alive'] and 
                len(path['group_trace']) > 0 and 
                path['group_trace'][-1] not in matched_prev):
                
                # This path dies in the transition to the current step
                path['is_alive'] = False
                path['death_step'] = step_idx - 1  # Last step where it was alive
                # Death parameter is the current filtration parameter where the feature disappears
                path['death_param'] = filtration_param
                
                # Record death event
                tracking_info['death_events'].append({
                    'step': step_idx - 1,
                    'group': path['group_trace'][-1] if path['group_trace'] else -1,
                    'filtration_param': path['death_param'],
                    'path_id': path['path_id']
                })
        
        # Create new paths for unmatched current groups (birth events)
        for curr_idx in range(len(curr_groups)):
            if curr_idx not in matched_curr:
                # New path is born
                new_path = {
                    'path_id': len(active_paths),
                    'birth_step': step_idx,
                    'birth_param': filtration_param,
                    'death_step': None,
                    'death_param': None,
                    'eigenvalue_trace': [curr_groups[curr_idx]['mean_eigenvalue'].item()],
                    'step_trace': [step_idx],
                    'param_trace': [filtration_param],
                    'group_trace': [curr_idx],
                    'is_alive': True
                }
                active_paths.append(new_path)
                
                # Record birth event
                tracking_info['birth_events'].append({
                    'step': step_idx,
                    'group': curr_idx,
                    'filtration_param': filtration_param,
                    'path_id': new_path['path_id']
                })
    
    def _finalize_paths(self, active_paths: List[Dict], final_step: int, final_param: float):
        """Finalize path tracking - mark surviving paths as infinite.
        
        Args:
            active_paths: List of all paths
            final_step: Final step index
            final_param: Final filtration parameter value
        """
        for path in active_paths:
            if path['is_alive']:
                # This path survives to the end - infinite bar
                path['death_step'] = None
                path['death_param'] = None  # Infinite
                
        logger.debug(f"Finalized {len(active_paths)} paths: "
                    f"{sum(1 for p in active_paths if p['death_param'] is not None)} finite, "
                    f"{sum(1 for p in active_paths if p['death_param'] is None)} infinite")
    
    def _track_gw_subspaces(self,
                           eigenvalues_sequence: List[torch.Tensor],
                           eigenvectors_sequence: List[torch.Tensor],
                           filtration_params: List[float]) -> Dict:
        """Track subspaces for GW-based filtration with zero-crossing detection.
        
        GW-specific birth-death semantics:
        - Birth = zero eigenvalue becomes non-zero (spectral feature emerges)
        - Death = non-zero eigenvalue becomes zero (spectral feature collapses)
        - Parameters increase → complexity increases → more non-zero eigenvalues
        
        This is fundamentally different from standard filtration where we track
        eigenspace appearance/disappearance rather than magnitude transitions.
        """
        logger.info("Tracking eigenvalue magnitude transitions for GW filtration (zero-crossing detection)")
        
        # Use GW-specific tracker with zero-crossing detection
        return self._track_gw_magnitude_transitions(eigenvalues_sequence, eigenvectors_sequence, filtration_params)
    
    def _track_standard_subspaces(self,
                                 eigenvalues_sequence: List[torch.Tensor],
                                 eigenvectors_sequence: List[torch.Tensor],
                                 filtration_params: List[float]) -> Dict:
        """Track subspaces for standard (Procrustes) filtration.
        
        Standard considerations:
        - Parameters increase, complexity decreases (standard interpretation)
        - Birth/death semantics: birth < death with increasing parameters
        - Early parameters → dense graphs → fewer large eigenvalues
        - Later parameters → sparse graphs → more small eigenvalues
        """
        logger.info("Tracking subspaces for standard filtration (decreasing complexity)")
        
        # This is the existing logic - just renamed for clarity
        n_steps = len(eigenvalues_sequence)
        tracking_info = {
            'eigenvalue_paths': [],
            'birth_events': [],
            'death_events': [],
            'crossings': [],
            'persistent_pairs': [],
            'continuous_paths': []
        }
        
        logger.info(f"Starting eigenspace tracking through {n_steps} filtration steps")
        
        # Initialize active paths from first step
        prev_groups = self._group_eigenvalues(
            eigenvalues_sequence[0], 
            eigenvectors_sequence[0]
        )
        
        # Initialize paths: each group in the first step starts a path
        active_paths = []
        for group_idx, group in enumerate(prev_groups):
            path = {
                'path_id': len(active_paths),
                'birth_step': 0,
                'birth_param': filtration_params[0],
                'death_step': None,
                'death_param': None,
                'eigenvalue_trace': [group['mean_eigenvalue'].item()],
                'step_trace': [0],
                'param_trace': [filtration_params[0]],
                'group_trace': [group_idx],
                'is_alive': True
            }
            active_paths.append(path)
        
        # Track through sequence with proper path continuity
        for i in range(1, n_steps):
            curr_eigenvals = eigenvalues_sequence[i]
            curr_eigenvecs = eigenvectors_sequence[i]
            curr_groups = self._group_eigenvalues(curr_eigenvals, curr_eigenvecs)
            
            # Match groups between steps using subspace similarity
            matching = self._match_eigenspaces(prev_groups, curr_groups)
            
            # Update active paths based on matching
            self._update_paths_from_matching(
                active_paths, matching, curr_groups, 
                i, filtration_params[i], tracking_info
            )
            
            prev_groups = curr_groups
        
        # Finalize remaining active paths
        self._finalize_paths(active_paths, n_steps - 1, filtration_params[-1])
        
        # Extract persistence pairs for diagram construction
        finite_pairs = []
        infinite_pairs = []
        
        for path in active_paths:
            if path['death_param'] is not None:
                birth = path['birth_param']
                death = path['death_param']
                lifetime = death - birth
                
                # Validate mathematical correctness: birth <= death for increasing parameters
                if birth > death:
                    logger.warning(f"Invalid persistence pair: birth={birth:.6f} > death={death:.6f}, "
                                 f"path_id={path['path_id']}. Skipping.")
                    continue
                    
                finite_pairs.append({
                    'birth_param': birth,
                    'death_param': death,
                    'lifetime': lifetime,
                    'path_id': path['path_id']
                })
            else:
                infinite_pairs.append({
                    'birth_param': path['birth_param'],
                    'path_id': path['path_id']
                })
        
        tracking_info['finite_pairs'] = finite_pairs
        tracking_info['infinite_pairs'] = infinite_pairs
        tracking_info['continuous_paths'] = active_paths
        
        logger.info(f"Path tracking completed. Found {len(finite_pairs)} finite paths, "
                   f"{len(infinite_pairs)} infinite paths")
        
        return tracking_info
    
    def _track_gw_magnitude_transitions(self,
                                       eigenvalues_sequence: List[torch.Tensor],
                                       eigenvectors_sequence: List[torch.Tensor],
                                       filtration_params: List[float]) -> Dict:
        """Track eigenvalue magnitude transitions for GW filtration.
        
        GW filtration semantics: increasing parameters = increasing complexity
        - Birth = zero eigenvalue → non-zero (spectral feature emerges)
        - Death = non-zero eigenvalue → zero (spectral feature collapses) 
        - Tracks magnitude crossings rather than subspace similarity
        
        Args:
            eigenvalues_sequence: Eigenvalues for each filtration step
            eigenvectors_sequence: Eigenvectors for each filtration step  
            filtration_params: Increasing filtration parameters
            
        Returns:
            Dictionary with continuous paths and birth-death events
        """
        n_steps = len(eigenvalues_sequence)
        zero_threshold = max(self.gap_eps * 10, 1e-8)  # Configurable zero detection
        
        logger.info(f"GW magnitude tracking: {n_steps} steps, zero_threshold={zero_threshold:.2e}")
        
        # Initialize tracking data
        tracking_info = {
            'eigenvalue_paths': [],
            'birth_events': [],
            'death_events': [],
            'crossings': [],
            'persistent_pairs': [],
            'continuous_paths': []
        }
        
        # Track eigenvalue paths through magnitude evolution
        active_paths = []
        
        # Initialize paths from first step
        first_eigenvals = eigenvalues_sequence[0]
        first_groups = self._group_eigenvalues_gw(first_eigenvals, eigenvectors_sequence[0], zero_threshold)
        
        for group_idx, group in enumerate(first_groups):
            if group['is_positive']:  # Only track positive eigenvalues initially
                path = {
                    'path_id': len(active_paths),
                    'birth_step': 0,
                    'birth_param': filtration_params[0],
                    'death_step': None,
                    'death_param': None,
                    'eigenvalue_trace': [group['mean_eigenvalue'].item()],
                    'step_trace': [0],
                    'param_trace': [filtration_params[0]],
                    'group_trace': [group_idx],
                    'is_alive': True,
                    'zero_crossing_events': []
                }
                active_paths.append(path)
        
        # Track through remaining steps
        for step_idx in range(1, n_steps):
            curr_eigenvals = eigenvalues_sequence[step_idx]
            curr_eigenvecs = eigenvectors_sequence[step_idx]
            prev_eigenvals = eigenvalues_sequence[step_idx - 1]
            curr_param = filtration_params[step_idx]
            
            # Detect zero crossings between consecutive steps
            birth_indices, death_indices = self._detect_zero_crossings(
                prev_eigenvals, curr_eigenvals, zero_threshold
            )
            
            # Group current eigenvalues
            curr_groups = self._group_eigenvalues_gw(curr_eigenvals, curr_eigenvecs, zero_threshold)
            
            # Handle birth events (zero → positive)
            for birth_idx in birth_indices:
                new_path = {
                    'path_id': len(active_paths),
                    'birth_step': step_idx,
                    'birth_param': curr_param,
                    'death_step': None,
                    'death_param': None,
                    'eigenvalue_trace': [curr_eigenvals[birth_idx].item()],
                    'step_trace': [step_idx],
                    'param_trace': [curr_param],
                    'group_trace': [birth_idx],
                    'is_alive': True,
                    'zero_crossing_events': [{'type': 'birth', 'step': step_idx, 'param': curr_param}]
                }
                active_paths.append(new_path)
                
                # Record birth event
                tracking_info['birth_events'].append({
                    'step': step_idx,
                    'group': birth_idx,
                    'filtration_param': curr_param,
                    'path_id': new_path['path_id'],
                    'eigenvalue': curr_eigenvals[birth_idx].item()
                })
            
            # Handle death events (positive → zero)
            for death_idx in death_indices:
                # Find corresponding active path
                for path in active_paths:
                    if (path['is_alive'] and len(path['group_trace']) > 0):
                        # Use eigenvalue ordering for robust matching
                        if self._eigenvalue_corresponds_to_path(path, death_idx, prev_eigenvals):
                            path['is_alive'] = False
                            path['death_step'] = step_idx - 1
                            path['death_param'] = curr_param
                            path['zero_crossing_events'].append({
                                'type': 'death', 'step': step_idx, 'param': curr_param
                            })
                            
                            # Record death event
                            tracking_info['death_events'].append({
                                'step': step_idx - 1,
                                'group': death_idx,
                                'filtration_param': curr_param,
                                'path_id': path['path_id'],
                                'eigenvalue': prev_eigenvals[death_idx].item()
                            })
                            break
            
            # Update continuing paths (positive → positive)
            self._update_gw_continuing_paths(active_paths, curr_groups, step_idx, curr_param, zero_threshold)
        
        # Finalize paths
        self._finalize_paths(active_paths, n_steps - 1, filtration_params[-1])
        tracking_info['continuous_paths'] = active_paths
        
        # Generate persistence pairs
        finite_pairs = []
        infinite_pairs = []
        
        for path in active_paths:
            if path['death_param'] is not None:
                finite_pairs.append({
                    'birth_param': path['birth_param'],
                    'death_param': path['death_param'], 
                    'lifetime': path['death_param'] - path['birth_param'],
                    'path_id': path['path_id']
                })
            else:
                infinite_pairs.append({
                    'birth_param': path['birth_param'],
                    'path_id': path['path_id']
                })
        
        tracking_info['finite_pairs'] = finite_pairs
        tracking_info['infinite_pairs'] = infinite_pairs
        
        logger.info(f"GW magnitude tracking completed: {len(finite_pairs)} finite, "
                   f"{len(infinite_pairs)} infinite paths")
        
        return tracking_info
    
    def _group_eigenvalues_gw(self, 
                             eigenvalues: torch.Tensor,
                             eigenvectors: torch.Tensor,
                             zero_threshold: float) -> List[Dict]:
        """Group eigenvalues for GW filtration with zero/positive separation.
        
        Args:
            eigenvalues: Tensor of eigenvalues
            eigenvectors: Tensor of eigenvectors
            zero_threshold: Threshold for zero detection
            
        Returns:
            List of groups with zero/positive classification
        """
        # Sort eigenvalues for consistent ordering
        sorted_idx = torch.argsort(eigenvalues)
        sorted_vals = eigenvalues[sorted_idx]
        sorted_vecs = eigenvectors[:, sorted_idx]
        
        groups = []
        
        # First, create zero group (all eigenvalues ≤ threshold)
        zero_indices = []
        zero_eigenvals = []
        for i, eigenval in enumerate(sorted_vals):
            if eigenval <= zero_threshold:
                zero_indices.append(i)
                zero_eigenvals.append(eigenval.item())
        
        if zero_indices:
            zero_group = {
                'eigenvalues': zero_eigenvals,
                'indices': zero_indices,
                'mean_eigenvalue': torch.mean(sorted_vals[zero_indices]),
                'subspace': sorted_vecs[:, zero_indices],
                'is_positive': False
            }
            groups.append(zero_group)
        
        # Then, group positive eigenvalues by proximity
        pos_start_idx = len(zero_indices)
        if pos_start_idx < len(sorted_vals):
            current_group = {
                'eigenvalues': [sorted_vals[pos_start_idx].item()],
                'indices': [pos_start_idx],
                'mean_eigenvalue': sorted_vals[pos_start_idx],
                'subspace': sorted_vecs[:, [pos_start_idx]],
                'is_positive': True
            }
            
            # Group remaining positive eigenvalues by proximity
            for i in range(pos_start_idx + 1, len(sorted_vals)):
                if abs(sorted_vals[i] - sorted_vals[i-1]) < self.gap_eps:
                    # Add to current positive group
                    current_group['eigenvalues'].append(sorted_vals[i].item())
                    current_group['indices'].append(i)
                    current_group['subspace'] = torch.cat([current_group['subspace'],
                                                         sorted_vecs[:, [i]]], dim=1)
                    current_group['mean_eigenvalue'] = torch.mean(torch.tensor(current_group['eigenvalues']))
                else:
                    # Start new positive group
                    groups.append(current_group)
                    current_group = {
                        'eigenvalues': [sorted_vals[i].item()],
                        'indices': [i],
                        'mean_eigenvalue': sorted_vals[i],
                        'subspace': sorted_vecs[:, [i]],
                        'is_positive': True
                    }
            
            # Add final positive group
            groups.append(current_group)
        
        return groups[:self.max_groups]
    
    def _detect_zero_crossings(self,
                              prev_eigenvals: torch.Tensor,
                              curr_eigenvals: torch.Tensor,
                              zero_threshold: float) -> Tuple[List[int], List[int]]:
        """Detect eigenvalue zero crossings between consecutive steps.
        
        Args:
            prev_eigenvals: Previous step eigenvalues
            curr_eigenvals: Current step eigenvalues  
            zero_threshold: Threshold for zero detection
            
        Returns:
            Tuple of (birth_indices, death_indices)
            birth_indices: Indices where eigenvalues crossed zero → positive
            death_indices: Indices where eigenvalues crossed positive → zero
        """
        # Sort both for consistent correspondence
        prev_sorted, prev_indices = torch.sort(prev_eigenvals)
        curr_sorted, curr_indices = torch.sort(curr_eigenvals)
        
        birth_indices = []
        death_indices = []
        
        # Handle dimension mismatch gracefully
        min_len = min(len(prev_sorted), len(curr_sorted))
        
        for i in range(min_len):
            prev_val = prev_sorted[i]
            curr_val = curr_sorted[i]
            
            # Birth: zero → positive
            if prev_val <= zero_threshold and curr_val > zero_threshold:
                birth_indices.append(curr_indices[i].item())
            
            # Death: positive → zero
            elif prev_val > zero_threshold and curr_val <= zero_threshold:
                death_indices.append(prev_indices[i].item())
        
        return birth_indices, death_indices
    
    def _eigenvalue_corresponds_to_path(self,
                                       path: Dict,
                                       eigenval_idx: int,
                                       eigenvals: torch.Tensor) -> bool:
        """Check if eigenvalue corresponds to an active path.
        
        Uses eigenvalue magnitude and ordering for robust correspondence.
        
        Args:
            path: Active path dictionary
            eigenval_idx: Index of eigenvalue to check
            eigenvals: Tensor of eigenvalues
            
        Returns:
            True if eigenvalue corresponds to path
        """
        if not path['eigenvalue_trace']:
            return False
        
        # Simple correspondence based on eigenvalue ordering and magnitude similarity
        last_eigenval = path['eigenvalue_trace'][-1]
        current_eigenval = eigenvals[eigenval_idx].item()
        
        # Allow for some numerical variation but require similar magnitude
        magnitude_ratio = abs(current_eigenval / max(last_eigenval, 1e-10))
        
        return 0.1 <= magnitude_ratio <= 10.0  # Reasonable range for eigenvalue evolution
    
    def _update_gw_continuing_paths(self,
                                   active_paths: List[Dict],
                                   curr_groups: List[Dict],
                                   step_idx: int,
                                   curr_param: float,
                                   zero_threshold: float):
        """Update paths that continue (positive → positive transitions).
        
        Args:
            active_paths: List of currently active paths
            curr_groups: Current eigenvalue groups
            step_idx: Current step index
            curr_param: Current filtration parameter
            zero_threshold: Zero detection threshold
        """
        # Find positive groups only
        positive_groups = [g for g in curr_groups if g.get('is_positive', True)]
        
        if not positive_groups:
            return
        
        # Simple correspondence: match paths to positive groups by ordering
        active_positive_paths = [p for p in active_paths if p['is_alive']]
        
        for i, path in enumerate(active_positive_paths):
            if i < len(positive_groups):
                group = positive_groups[i]
                
                # Continue path with current group
                path['eigenvalue_trace'].append(group['mean_eigenvalue'].item())
                path['step_trace'].append(step_idx)
                path['param_trace'].append(curr_param)
                path['group_trace'].append(i)
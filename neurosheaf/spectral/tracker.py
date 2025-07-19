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
    """Track eigenspace evolution through filtration using subspace similarity."""
    
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
        logger.info(f"SubspaceTracker initialized with gap_eps={gap_eps}, cos_tau={cos_tau}")
        
    def track_eigenspaces(self,
                         eigenvalues_sequence: List[torch.Tensor],
                         eigenvectors_sequence: List[torch.Tensor],
                         filtration_params: List[float]) -> Dict:
        """Track eigenspaces through filtration parameter changes.
        
        Args:
            eigenvalues_sequence: List of eigenvalue tensors for each filtration
            eigenvectors_sequence: List of eigenvector tensors for each filtration
            filtration_params: List of filtration parameter values
            
        Returns:
            Dictionary with tracking information including paths, birth/death events
        """
        if len(eigenvalues_sequence) != len(eigenvectors_sequence):
            raise ValueError("Eigenvalues and eigenvectors sequences must have same length")
        
        if len(eigenvalues_sequence) != len(filtration_params):
            raise ValueError("Sequences and filtration parameters must have same length")
        
        n_steps = len(eigenvalues_sequence)
        tracking_info = {
            'eigenvalue_paths': [],
            'birth_events': [],
            'death_events': [],
            'crossings': [],
            'persistent_pairs': [],
            'continuous_paths': []  # New: properly tracked continuous paths
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
        
        # Finalize paths - mark surviving paths as infinite
        self._finalize_paths(active_paths, n_steps - 1, filtration_params[-1])
        
        # Store all paths in tracking info
        tracking_info['continuous_paths'] = active_paths
        
        # Generate persistence pairs from continuous paths
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
        
        logger.info(f"Path tracking completed. Found {len(finite_pairs)} finite paths, "
                   f"{len(infinite_pairs)} infinite paths")
        
        return tracking_info
    
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
        # Sort eigenvalues
        sorted_idx = torch.argsort(eigenvalues)
        sorted_vals = eigenvalues[sorted_idx]
        sorted_vecs = eigenvectors[:, sorted_idx]
        
        groups = []
        current_group = {
            'eigenvalues': [sorted_vals[0]],
            'eigenvectors': [sorted_vecs[:, 0:1]],
            'indices': [sorted_idx[0]],
            'mean_eigenvalue': sorted_vals[0]
        }
        
        for i in range(1, len(sorted_vals)):
            gap = sorted_vals[i] - sorted_vals[i-1]
            
            if gap < self.gap_eps:
                # Add to current group
                current_group['eigenvalues'].append(sorted_vals[i])
                current_group['eigenvectors'].append(sorted_vecs[:, i:i+1])
                current_group['indices'].append(sorted_idx[i])
                current_group['mean_eigenvalue'] = torch.mean(
                    torch.stack(current_group['eigenvalues'])
                )
            else:
                # Finalize current group
                current_group['subspace'] = torch.cat(
                    current_group['eigenvectors'], dim=1
                )
                groups.append(current_group)
                
                # Start new group
                current_group = {
                    'eigenvalues': [sorted_vals[i]],
                    'eigenvectors': [sorted_vecs[:, i:i+1]],
                    'indices': [sorted_idx[i]],
                    'mean_eigenvalue': sorted_vals[i]
                }
        
        # Don't forget the last group
        current_group['subspace'] = torch.cat(
            current_group['eigenvectors'], dim=1
        )
        groups.append(current_group)
        
        logger.debug(f"Grouped {len(eigenvalues)} eigenvalues into {len(groups)} groups")
        return groups
    
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
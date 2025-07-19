"""Persistence diagram distance metrics.

This module provides implementations of Wasserstein and Bottleneck distances
for persistence diagrams, along with helper functions for preprocessing and
pairwise distance computation.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import scipy.optimize
from scipy.spatial.distance import cdist
import warnings


def preprocess_diagram(
    diagram: np.ndarray,
    remove_diagonal: bool = False,
    max_persistence: Optional[float] = None
) -> np.ndarray:
    """Preprocess a persistence diagram for distance computation.
    
    Args:
        diagram: Persistence diagram as array of shape (n, 2) with columns [birth, death]
        remove_diagonal: Whether to remove points on the diagonal (birth == death)
        max_persistence: Maximum persistence value to consider (filters out infinite points)
    
    Returns:
        Preprocessed diagram
    """
    if diagram.size == 0:
        return np.empty((0, 2))
    
    # Ensure 2D array
    diagram = np.atleast_2d(diagram)
    
    # Remove diagonal points if requested
    if remove_diagonal:
        persistence = diagram[:, 1] - diagram[:, 0]
        diagram = diagram[persistence > 1e-10]
    
    # Filter by max persistence if specified
    if max_persistence is not None:
        persistence = diagram[:, 1] - diagram[:, 0]
        finite_mask = np.isfinite(diagram).all(axis=1)
        persistence_mask = persistence <= max_persistence
        diagram = diagram[finite_mask & persistence_mask]
    
    return diagram


def add_diagonal_points(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    n_diagonal: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Add projections to diagonal for optimal transport.
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        n_diagonal: Number of diagonal points to add (default: max of diagram sizes)
    
    Returns:
        Extended diagrams with diagonal projections
    """
    n1, n2 = len(diagram1), len(diagram2)
    
    if n_diagonal is None:
        n_diagonal = max(n1, n2)
    
    # Compute projections to diagonal
    proj1 = np.array([])
    proj2 = np.array([])
    
    if n1 > 0:
        proj1 = 0.5 * (diagram1[:, 0] + diagram1[:, 1])
        diag1 = np.column_stack([proj1, proj1])
    else:
        diag1 = np.empty((0, 2))
    
    if n2 > 0:
        proj2 = 0.5 * (diagram2[:, 0] + diagram2[:, 1])
        diag2 = np.column_stack([proj2, proj2])
    else:
        diag2 = np.empty((0, 2))
    
    # Add diagonal points
    if n_diagonal > 0:
        # Sample additional diagonal points if needed
        if n1 > 0 and n2 > 0:
            all_projs = np.concatenate([proj1, proj2])
        elif n1 > 0:
            all_projs = proj1
        elif n2 > 0:
            all_projs = proj2
        else:
            all_projs = np.array([0.0])
        
        if len(all_projs) > 0:
            diag_range = [all_projs.min(), all_projs.max()]
        else:
            diag_range = [0.0, 1.0]
            
        extra_diag = np.linspace(diag_range[0], diag_range[1], n_diagonal)
        extra_diag = np.column_stack([extra_diag, extra_diag])
        
        # Combine original points with diagonal projections
        parts1 = [p for p in [diagram1, diag2, extra_diag] if len(p) > 0]
        parts2 = [p for p in [diagram2, diag1, extra_diag] if len(p) > 0]
        
        diagram1_ext = np.vstack(parts1) if parts1 else np.empty((0, 2))
        diagram2_ext = np.vstack(parts2) if parts2 else np.empty((0, 2))
    else:
        # No extra diagonal points
        if n1 > 0 and len(diag2) > 0:
            diagram1_ext = np.vstack([diagram1, diag2])
        elif n1 > 0:
            diagram1_ext = diagram1
        else:
            diagram1_ext = diag2
            
        if n2 > 0 and len(diag1) > 0:
            diagram2_ext = np.vstack([diagram2, diag1])
        elif n2 > 0:
            diagram2_ext = diagram2
        else:
            diagram2_ext = diag1
    
    return diagram1_ext, diagram2_ext


def _compute_distance_matrix(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    p: float = 2.0,
    delta: float = 0.01
) -> np.ndarray:
    """Compute pairwise distance matrix between points in two diagrams.
    
    Args:
        diagram1: First diagram
        diagram2: Second diagram
        p: Power for distance computation
        delta: Small value to handle infinite coordinates
    
    Returns:
        Distance matrix
    """
    # Handle empty diagrams
    if len(diagram1) == 0 or len(diagram2) == 0:
        return np.empty((len(diagram1), len(diagram2)))
    
    # Handle infinite coordinates
    diagram1_finite = np.copy(diagram1)
    diagram2_finite = np.copy(diagram2)
    
    # Find finite values to determine scale
    finite_vals1 = diagram1[np.isfinite(diagram1)]
    finite_vals2 = diagram2[np.isfinite(diagram2)]
    
    if len(finite_vals1) > 0 and len(finite_vals2) > 0:
        max_finite = max(np.abs(finite_vals1).max(), np.abs(finite_vals2).max())
    elif len(finite_vals1) > 0:
        max_finite = np.abs(finite_vals1).max()
    elif len(finite_vals2) > 0:
        max_finite = np.abs(finite_vals2).max()
    else:
        max_finite = 1.0
        
    large_value = max_finite * 1000
    
    # Replace infinities with large finite values
    diagram1_finite[np.isinf(diagram1)] = large_value
    diagram2_finite[np.isinf(diagram2)] = large_value
    
    # Replace NaN with 0 (should not happen but just in case)
    diagram1_finite[np.isnan(diagram1_finite)] = 0
    diagram2_finite[np.isnan(diagram2_finite)] = 0
    
    if p == np.inf:
        # L-infinity norm
        dist_matrix = cdist(diagram1_finite, diagram2_finite, metric='chebyshev')
    else:
        # L-p norm
        dist_matrix = cdist(diagram1_finite, diagram2_finite, metric='minkowski', p=p)
    
    return dist_matrix


def wasserstein_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    p: float = 2.0,
    q: float = 2.0,
    delta: float = 0.01
) -> float:
    """Compute the Wasserstein distance between two persistence diagrams.
    
    The q-Wasserstein distance is defined as:
    W_q(D1, D2) = inf (sum ||x - φ(x)||^q)^(1/q)
    where the infimum is over all bijections φ: D1 ∪ Δ → D2 ∪ Δ
    
    Args:
        diagram1: First persistence diagram (n1 x 2 array)
        diagram2: Second persistence diagram (n2 x 2 array)
        p: Norm to use for point distances (default: 2)
        q: Power for Wasserstein distance (default: 2)
        delta: Small value for numerical stability
    
    Returns:
        The q-Wasserstein distance
    """
    # Preprocess diagrams
    diagram1 = preprocess_diagram(diagram1)
    diagram2 = preprocess_diagram(diagram2)
    
    # Handle empty diagrams
    if len(diagram1) == 0 and len(diagram2) == 0:
        return 0.0
    elif len(diagram1) == 0:
        # Distance is sum of persistences in diagram2
        persistence = diagram2[:, 1] - diagram2[:, 0]
        return float(np.sum(persistence ** q) ** (1.0 / q))
    elif len(diagram2) == 0:
        # Distance is sum of persistences in diagram1
        persistence = diagram1[:, 1] - diagram1[:, 0]
        return float(np.sum(persistence ** q) ** (1.0 / q))
    
    # Add diagonal points for optimal transport
    diagram1_ext, diagram2_ext = add_diagonal_points(diagram1, diagram2)
    
    # Compute distance matrix
    dist_matrix = _compute_distance_matrix(diagram1_ext, diagram2_ext, p=p, delta=delta)
    
    # Raise distances to power q
    cost_matrix = dist_matrix ** q
    
    # Solve optimal transport problem using linear_sum_assignment
    # This minimizes the sum of costs
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    # Compute Wasserstein distance
    total_cost = cost_matrix[row_ind, col_ind].sum()
    wasserstein_dist = float(total_cost ** (1.0 / q))
    
    return wasserstein_dist


def bottleneck_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    delta: float = 0.01
) -> float:
    """Compute the bottleneck distance between two persistence diagrams.
    
    The bottleneck distance is:
    d_B(D1, D2) = inf max ||x - φ(x)||_∞
    where the infimum is over all bijections φ: D1 ∪ Δ → D2 ∪ Δ
    
    Args:
        diagram1: First persistence diagram (n1 x 2 array)
        diagram2: Second persistence diagram (n2 x 2 array)
        delta: Small value for numerical stability
    
    Returns:
        The bottleneck distance
    """
    # Preprocess diagrams
    diagram1 = preprocess_diagram(diagram1)
    diagram2 = preprocess_diagram(diagram2)
    
    # Handle empty diagrams
    if len(diagram1) == 0 and len(diagram2) == 0:
        return 0.0
    elif len(diagram1) == 0:
        # Distance is maximum persistence in diagram2
        persistence = diagram2[:, 1] - diagram2[:, 0]
        return float(np.max(persistence))
    elif len(diagram2) == 0:
        # Distance is maximum persistence in diagram1
        persistence = diagram1[:, 1] - diagram1[:, 0]
        return float(np.max(persistence))
    
    # Add diagonal points for optimal transport
    diagram1_ext, diagram2_ext = add_diagonal_points(diagram1, diagram2)
    
    # Compute distance matrix with L-infinity norm
    dist_matrix = _compute_distance_matrix(diagram1_ext, diagram2_ext, p=np.inf, delta=delta)
    
    # For bottleneck distance, we need to find the minimum over all matchings
    # of the maximum distance in the matching
    # We use linear assignment but minimize the maximum (bottleneck)
    # This is an approximation; exact bottleneck requires a different algorithm
    
    # Try multiple approaches to approximate bottleneck distance
    candidates = []
    
    # Approach 1: Use assignment on original distances
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix)
    max_dist = np.max(dist_matrix[row_ind, col_ind])
    candidates.append(max_dist)
    
    # Approach 2: Binary search on threshold
    # Find the minimum threshold such that there exists a perfect matching
    # with all edges having weight <= threshold
    low, high = 0.0, np.max(dist_matrix)
    tolerance = 1e-6
    
    while high - low > tolerance:
        mid = (low + high) / 2
        # Create bipartite graph with edges having weight <= mid
        feasible_edges = dist_matrix <= mid
        # Check if perfect matching exists (simplified check)
        if np.all(np.any(feasible_edges, axis=1)) and np.all(np.any(feasible_edges, axis=0)):
            high = mid
        else:
            low = mid
    
    candidates.append(high)
    
    # Return the minimum of our approximations
    return float(min(candidates))


def extract_persistence_diagram_array(
    diagrams_dict: dict,
    include_infinite: bool = False,
    inf_replacement: Optional[float] = None
) -> np.ndarray:
    """Extract persistence diagram as numpy array from neurosheaf analyzer results.
    
    Args:
        diagrams_dict: Dictionary from PersistentSpectralAnalyzer with 'birth_death_pairs' key
        include_infinite: Whether to include infinite bars (default: False)
        inf_replacement: Value to replace np.inf with (default: None, keeps np.inf)
    
    Returns:
        Numpy array of shape (n, 2) with birth/death pairs
    """
    pairs = diagrams_dict.get('birth_death_pairs', [])
    diagram_list = []
    
    # Add finite pairs
    for pair in pairs:
        diagram_list.append([pair['birth'], pair['death']])
    
    # Optionally add infinite bars
    if include_infinite and 'infinite_bars' in diagrams_dict:
        for bar in diagrams_dict['infinite_bars']:
            death_value = np.inf if inf_replacement is None else inf_replacement
            diagram_list.append([bar['birth'], death_value])
    
    if len(diagram_list) == 0:
        return np.empty((0, 2))
    
    return np.array(diagram_list)


def sliced_wasserstein_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    n_slices: int = 50,
    p: float = 2.0,
    seed: Optional[int] = None
) -> float:
    """Compute the sliced Wasserstein distance between persistence diagrams.
    
    This is an approximation that projects diagrams onto random lines and
    computes 1D Wasserstein distances, then averages.
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        n_slices: Number of random projections
        p: Power for Wasserstein distance
        seed: Random seed for reproducibility
    
    Returns:
        Approximated Wasserstein distance
    """
    # Preprocess diagrams
    diagram1 = preprocess_diagram(diagram1)
    diagram2 = preprocess_diagram(diagram2)
    
    # Handle empty diagrams
    if len(diagram1) == 0 and len(diagram2) == 0:
        return 0.0
    
    # Convert to birth-persistence coordinates
    if len(diagram1) > 0:
        persist1 = np.column_stack([diagram1[:, 0], diagram1[:, 1] - diagram1[:, 0]])
    else:
        persist1 = np.empty((0, 2))
    
    if len(diagram2) > 0:
        persist2 = np.column_stack([diagram2[:, 0], diagram2[:, 1] - diagram2[:, 0]])
    else:
        persist2 = np.empty((0, 2))
    
    # Generate random directions
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n_slices)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    distances = []
    for direction in directions:
        # Project onto direction
        if len(persist1) > 0:
            proj1 = persist1 @ direction
        else:
            proj1 = np.array([])
        
        if len(persist2) > 0:
            proj2 = persist2 @ direction
        else:
            proj2 = np.array([])
        
        # Compute 1D Wasserstein distance
        if len(proj1) == 0 and len(proj2) == 0:
            dist = 0.0
        elif len(proj1) == 0:
            dist = np.sum(np.abs(proj2) ** p) ** (1.0 / p)
        elif len(proj2) == 0:
            dist = np.sum(np.abs(proj1) ** p) ** (1.0 / p)
        else:
            # Sort projections
            proj1_sorted = np.sort(proj1)
            proj2_sorted = np.sort(proj2)
            
            # Pad shorter array
            n_max = max(len(proj1_sorted), len(proj2_sorted))
            if len(proj1_sorted) < n_max:
                proj1_sorted = np.pad(proj1_sorted, (0, n_max - len(proj1_sorted)), constant_values=0)
            if len(proj2_sorted) < n_max:
                proj2_sorted = np.pad(proj2_sorted, (0, n_max - len(proj2_sorted)), constant_values=0)
            
            # 1D Wasserstein is just sorted differences
            dist = np.sum(np.abs(proj1_sorted - proj2_sorted) ** p) ** (1.0 / p)
        
        distances.append(dist)
    
    return float(np.mean(distances))


def compute_pairwise_distances(
    diagrams: List[np.ndarray],
    metric: str = 'wasserstein',
    **kwargs
) -> np.ndarray:
    """Compute pairwise distances between multiple persistence diagrams.
    
    Args:
        diagrams: List of persistence diagrams
        metric: Distance metric ('wasserstein', 'bottleneck', or 'sliced_wasserstein')
        **kwargs: Additional arguments for the distance function
    
    Returns:
        Symmetric distance matrix of shape (n_diagrams, n_diagrams)
    """
    n_diagrams = len(diagrams)
    distances = np.zeros((n_diagrams, n_diagrams))
    
    # Select distance function
    if metric == 'wasserstein':
        dist_func = wasserstein_distance
    elif metric == 'bottleneck':
        dist_func = bottleneck_distance
    elif metric == 'sliced_wasserstein':
        dist_func = sliced_wasserstein_distance
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Compute upper triangle
    for i in range(n_diagrams):
        for j in range(i + 1, n_diagrams):
            distances[i, j] = dist_func(diagrams[i], diagrams[j], **kwargs)
            distances[j, i] = distances[i, j]  # Symmetric
    
    return distances


def persistence_fisher_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    sigma: float = 1.0,
    n_samples: int = 100
) -> float:
    """Compute Fisher information distance between persistence diagrams.
    
    This uses a kernel density estimate approach to compute the Fisher
    information metric between persistence diagrams.
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        sigma: Bandwidth for kernel density estimation
        n_samples: Number of samples for integration
    
    Returns:
        Fisher information distance
    """
    # This is a placeholder for more advanced distance metrics
    # For now, return Wasserstein distance with specific parameters
    warnings.warn(
        "Fisher distance not fully implemented, using Wasserstein distance as approximation",
        UserWarning
    )
    return wasserstein_distance(diagram1, diagram2, p=2.0, q=2.0)
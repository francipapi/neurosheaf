# neurosheaf/spectral/multi_parameter.py
"""Multi-parameter persistence for neural sheaf spectral analysis.

This module implements multi-parameter persistent spectral analysis, extending
single-parameter persistence to handle multiple filtration parameters simultaneously.
This provides richer topological information about neural network similarity structures.

Key Features:
- Multi-dimensional parameter space discretization
- Multi-parameter persistence computation
- Parameter correlation analysis
- Integration with existing spectral analysis components

References:
- Carlsson & Zomorodian (2009): "The theory of multidimensional persistence"
- Lesnick (2015): "The theory of the interleaving distance on multidimensional persistence modules"
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from itertools import product
import time
from dataclasses import dataclass
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError
from ..sheaf.construction import Sheaf

logger = setup_logger(__name__)


@dataclass
class ParameterPoint:
    """Represents a point in multi-parameter space."""
    coordinates: Tuple[float, ...]
    index: int
    
    def __post_init__(self):
        self.coordinates = tuple(self.coordinates)  # Ensure immutable
    
    def __hash__(self):
        return hash(self.coordinates)
    
    def __eq__(self, other):
        return isinstance(other, ParameterPoint) and self.coordinates == other.coordinates
    
    def __lt__(self, other):
        """Component-wise comparison for partial ordering."""
        if not isinstance(other, ParameterPoint):
            return NotImplemented
        return all(a <= b for a, b in zip(self.coordinates, other.coordinates))
    
    def __le__(self, other):
        return self < other or self == other
    
    def dominates(self, other) -> bool:
        """Check if this point dominates another (all coordinates >=)."""
        return all(a >= b for a, b in zip(self.coordinates, other.coordinates))


class MultiParameterFiltration:
    """Manages multi-parameter filtration spaces and computations.
    
    This class handles the discretization of multi-dimensional parameter spaces
    and provides utilities for multi-parameter persistence computation.
    
    Attributes:
        parameter_names: Names of the filtration parameters
        parameter_ranges: Ranges for each parameter
        grid_sizes: Number of grid points for each parameter
        total_points: Total number of parameter points
    """
    
    def __init__(self,
                 parameter_names: List[str],
                 parameter_ranges: List[Tuple[float, float]],
                 grid_sizes: List[int]):
        """Initialize multi-parameter filtration.
        
        Args:
            parameter_names: Names of filtration parameters (e.g., ['cka_threshold', 'edge_weight'])
            parameter_ranges: Min/max range for each parameter
            grid_sizes: Number of discretization points for each parameter
        """
        if len(parameter_names) != len(parameter_ranges) != len(grid_sizes):
            raise ValueError("All parameter specifications must have same length")
        
        if len(parameter_names) < 2:
            raise ValueError("Multi-parameter filtration requires at least 2 parameters")
        
        self.parameter_names = parameter_names
        self.parameter_ranges = parameter_ranges
        self.grid_sizes = grid_sizes
        self.dimension = len(parameter_names)
        
        # Generate parameter grid
        self._generate_parameter_grid()
        
        logger.info(f"MultiParameterFiltration initialized: {self.dimension}D space, "
                   f"{self.total_points} total points")
    
    def _generate_parameter_grid(self):
        """Generate discretized parameter grid."""
        # Create individual parameter arrays
        parameter_arrays = []
        for i, (name, (min_val, max_val), n_points) in enumerate(
            zip(self.parameter_names, self.parameter_ranges, self.grid_sizes)
        ):
            if n_points < 2:
                raise ValueError(f"Grid size must be at least 2 for parameter {name}")
            
            param_array = np.linspace(min_val, max_val, n_points)
            parameter_arrays.append(param_array)
        
        # Generate all combinations
        self.parameter_grid = list(product(*parameter_arrays))
        self.total_points = len(self.parameter_grid)
        
        # Create ParameterPoint objects
        self.parameter_points = [
            ParameterPoint(coordinates=coords, index=i)
            for i, coords in enumerate(self.parameter_grid)
        ]
        
        # Create lookup dictionaries
        self.coords_to_index = {
            coords: i for i, coords in enumerate(self.parameter_grid)
        }
        
        logger.debug(f"Generated {self.total_points} parameter points")
    
    def get_parameter_point(self, index: int) -> ParameterPoint:
        """Get parameter point by index."""
        if not 0 <= index < self.total_points:
            raise IndexError(f"Parameter index {index} out of range [0, {self.total_points})")
        return self.parameter_points[index]
    
    def find_index(self, coordinates: Tuple[float, ...]) -> Optional[int]:
        """Find index of parameter point with given coordinates."""
        return self.coords_to_index.get(coordinates)
    
    def get_neighbors(self, point: ParameterPoint, 
                     neighborhood_type: str = 'face') -> List[ParameterPoint]:
        """Get neighboring parameter points.
        
        Args:
            point: Parameter point
            neighborhood_type: Type of neighborhood ('face', 'vertex', 'moore')
        
        Returns:
            List of neighboring parameter points
        """
        if neighborhood_type == 'face':
            # Face neighbors: differ in exactly one coordinate by one grid step
            return self._get_face_neighbors(point)
        elif neighborhood_type == 'vertex':
            # Vertex neighbors: adjacent in grid connectivity
            return self._get_vertex_neighbors(point)
        elif neighborhood_type == 'moore':
            # Moore neighbors: all adjacent including diagonals
            return self._get_moore_neighbors(point)
        else:
            raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")
    
    def _get_face_neighbors(self, point: ParameterPoint) -> List[ParameterPoint]:
        """Get face neighbors (1-step in each coordinate direction)."""
        neighbors = []
        base_coords = point.coordinates
        
        # Get grid indices for current point
        grid_indices = self._coords_to_grid_indices(base_coords)
        
        for dim in range(self.dimension):
            for delta in [-1, 1]:
                new_indices = list(grid_indices)
                new_indices[dim] += delta
                
                # Check bounds
                if 0 <= new_indices[dim] < self.grid_sizes[dim]:
                    new_coords = self._grid_indices_to_coords(new_indices)
                    neighbor_index = self.find_index(new_coords)
                    if neighbor_index is not None:
                        neighbors.append(self.parameter_points[neighbor_index])
        
        return neighbors
    
    def _get_vertex_neighbors(self, point: ParameterPoint) -> List[ParameterPoint]:
        """Get vertex neighbors (connected in discrete grid)."""
        # For now, same as face neighbors
        return self._get_face_neighbors(point)
    
    def _get_moore_neighbors(self, point: ParameterPoint) -> List[ParameterPoint]:
        """Get Moore neighbors (all adjacent including diagonals)."""
        neighbors = []
        base_coords = point.coordinates
        grid_indices = self._coords_to_grid_indices(base_coords)
        
        # Generate all combinations of {-1, 0, 1} for each dimension
        deltas = product([-1, 0, 1], repeat=self.dimension)
        
        for delta in deltas:
            if all(d == 0 for d in delta):  # Skip the point itself
                continue
            
            new_indices = [gi + d for gi, d in zip(grid_indices, delta)]
            
            # Check bounds
            if all(0 <= ni < gs for ni, gs in zip(new_indices, self.grid_sizes)):
                new_coords = self._grid_indices_to_coords(new_indices)
                neighbor_index = self.find_index(new_coords)
                if neighbor_index is not None:
                    neighbors.append(self.parameter_points[neighbor_index])
        
        return neighbors
    
    def _coords_to_grid_indices(self, coords: Tuple[float, ...]) -> List[int]:
        """Convert coordinates to grid indices."""
        indices = []
        for i, (coord, (min_val, max_val), n_points) in enumerate(
            zip(coords, self.parameter_ranges, self.grid_sizes)
        ):
            # Find closest grid index
            param_array = np.linspace(min_val, max_val, n_points)
            idx = np.argmin(np.abs(param_array - coord))
            indices.append(idx)
        return indices
    
    def _grid_indices_to_coords(self, indices: List[int]) -> Tuple[float, ...]:
        """Convert grid indices to coordinates."""
        coords = []
        for i, (idx, (min_val, max_val), n_points) in enumerate(
            zip(indices, self.parameter_ranges, self.grid_sizes)
        ):
            param_array = np.linspace(min_val, max_val, n_points)
            coords.append(param_array[idx])
        return tuple(coords)
    
    def get_parameter_subspace(self, 
                              fixed_params: Dict[str, float]) -> 'MultiParameterFiltration':
        """Get subspace by fixing some parameters.
        
        Args:
            fixed_params: Dictionary of parameter_name -> value for fixed parameters
        
        Returns:
            New MultiParameterFiltration for the subspace
        """
        if not fixed_params:
            return self
        
        # Identify free parameters
        free_param_indices = []
        free_param_names = []
        free_param_ranges = []
        free_grid_sizes = []
        
        for i, name in enumerate(self.parameter_names):
            if name not in fixed_params:
                free_param_indices.append(i)
                free_param_names.append(name)
                free_param_ranges.append(self.parameter_ranges[i])
                free_grid_sizes.append(self.grid_sizes[i])
        
        if len(free_param_names) < 2:
            raise ValueError("Subspace must have at least 2 free parameters")
        
        return MultiParameterFiltration(
            parameter_names=free_param_names,
            parameter_ranges=free_param_ranges,
            grid_sizes=free_grid_sizes
        )
    
    def filter_points(self, condition: Callable[[ParameterPoint], bool]) -> List[ParameterPoint]:
        """Filter parameter points by condition.
        
        Args:
            condition: Function that takes ParameterPoint and returns bool
        
        Returns:
            List of parameter points satisfying condition
        """
        return [point for point in self.parameter_points if condition(point)]
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for each parameter."""
        return dict(zip(self.parameter_names, self.parameter_ranges))
    
    def compute_parameter_density(self) -> Dict[str, float]:
        """Compute density (points per unit) for each parameter."""
        densities = {}
        for name, (min_val, max_val), n_points in zip(
            self.parameter_names, self.parameter_ranges, self.grid_sizes
        ):
            range_size = max_val - min_val
            density = (n_points - 1) / range_size if range_size > 0 else float('inf')
            densities[name] = density
        return densities
    
    def interpolate_parameter_point(self, coordinates: Tuple[float, ...]) -> ParameterPoint:
        """Interpolate to nearest grid point.
        
        Args:
            coordinates: Continuous coordinates in parameter space
        
        Returns:
            Nearest parameter point on grid
        """
        if len(coordinates) != self.dimension:
            raise ValueError(f"Coordinates must have {self.dimension} dimensions")
        
        # Find nearest grid point
        grid_indices = self._coords_to_grid_indices(coordinates)
        grid_coords = self._grid_indices_to_coords(grid_indices)
        
        point_index = self.find_index(grid_coords)
        if point_index is None:
            raise ComputationError(f"Failed to find grid point for coordinates {coordinates}")
        
        return self.parameter_points[point_index]
    
    def __len__(self) -> int:
        """Return total number of parameter points."""
        return self.total_points
    
    def __iter__(self):
        """Iterate over parameter points."""
        return iter(self.parameter_points)
    
    def __getitem__(self, index: int) -> ParameterPoint:
        """Get parameter point by index."""
        return self.get_parameter_point(index)
    
    def __repr__(self) -> str:
        return (f"MultiParameterFiltration({self.dimension}D, "
                f"params={self.parameter_names}, "
                f"sizes={self.grid_sizes}, "
                f"points={self.total_points})")


class ParameterCorrelationAnalyzer:
    """Analyzes correlations and relationships between filtration parameters.
    
    This class provides tools for understanding how different filtration parameters
    relate to each other and their combined effects on persistence.
    """
    
    def __init__(self):
        """Initialize parameter correlation analyzer."""
        self.correlations = {}
        self.parameter_stats = {}
        
        logger.info("ParameterCorrelationAnalyzer initialized")
    
    def compute_parameter_correlations(self, 
                                     parameter_values: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Compute pairwise correlations between parameters.
        
        Args:
            parameter_values: Dictionary mapping parameter names to value lists
        
        Returns:
            Dictionary of pairwise correlation coefficients
        """
        import scipy.stats
        
        param_names = list(parameter_values.keys())
        correlations = {}
        
        for i, param1 in enumerate(param_names):
            correlations[param1] = {}
            for j, param2 in enumerate(param_names):
                if i == j:
                    correlations[param1][param2] = 1.0
                elif param2 in correlations and param1 in correlations[param2]:
                    # Use previously computed correlation (symmetric)
                    correlations[param1][param2] = correlations[param2][param1]
                else:
                    # Compute correlation
                    values1 = np.array(parameter_values[param1])
                    values2 = np.array(parameter_values[param2])
                    
                    if len(values1) != len(values2):
                        raise ValueError(f"Parameter value lists must have same length")
                    
                    if len(values1) < 2:
                        correlation = 0.0
                    else:
                        correlation, _ = scipy.stats.pearsonr(values1, values2)
                        if np.isnan(correlation):
                            correlation = 0.0
                    
                    correlations[param1][param2] = correlation
        
        self.correlations = correlations
        logger.info(f"Computed correlations for {len(param_names)} parameters")
        
        return correlations
    
    def compute_parameter_statistics(self, 
                                   parameter_values: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Compute statistical measures for each parameter.
        
        Args:
            parameter_values: Dictionary mapping parameter names to value lists
        
        Returns:
            Dictionary of parameter statistics
        """
        stats = {}
        
        for param_name, values in parameter_values.items():
            values_array = np.array(values)
            
            param_stats = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'median': np.median(values_array),
                'range': np.max(values_array) - np.min(values_array),
                'q25': np.percentile(values_array, 25),
                'q75': np.percentile(values_array, 75),
                'iqr': np.percentile(values_array, 75) - np.percentile(values_array, 25)
            }
            
            stats[param_name] = param_stats
        
        self.parameter_stats = stats
        logger.info(f"Computed statistics for {len(parameter_values)} parameters")
        
        return stats
    
    def identify_dominant_parameters(self, 
                                   parameter_effects: Dict[str, List[float]],
                                   threshold: float = 0.1) -> List[str]:
        """Identify parameters with strongest effects on persistence.
        
        Args:
            parameter_effects: Dictionary mapping parameter names to effect measures
            threshold: Minimum effect size to consider dominant
        
        Returns:
            List of dominant parameter names, sorted by effect size
        """
        # Compute effect magnitudes
        effect_magnitudes = {}
        for param_name, effects in parameter_effects.items():
            if len(effects) > 0:
                # Use coefficient of variation as effect measure
                mean_effect = np.mean(np.abs(effects))
                std_effect = np.std(effects)
                cv = std_effect / mean_effect if mean_effect > 0 else 0
                effect_magnitudes[param_name] = cv
            else:
                effect_magnitudes[param_name] = 0.0
        
        # Filter and sort by effect size
        dominant_params = [
            param for param, magnitude in effect_magnitudes.items()
            if magnitude >= threshold
        ]
        
        dominant_params.sort(key=lambda p: effect_magnitudes[p], reverse=True)
        
        logger.info(f"Identified {len(dominant_params)} dominant parameters")
        
        return dominant_params
    
    def compute_parameter_interactions(self, 
                                     parameter_values: Dict[str, List[float]],
                                     interaction_measure: str = 'mutual_info') -> Dict[Tuple[str, str], float]:
        """Compute interaction strengths between parameter pairs.
        
        Args:
            parameter_values: Dictionary mapping parameter names to value lists
            interaction_measure: Type of interaction measure ('mutual_info', 'chi2')
        
        Returns:
            Dictionary mapping parameter pairs to interaction strengths
        """
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import StandardScaler
        
        param_names = list(parameter_values.keys())
        interactions = {}
        
        # Prepare data
        param_arrays = {name: np.array(values) for name, values in parameter_values.items()}
        n_samples = len(list(param_arrays.values())[0])
        
        # Compute pairwise interactions
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names[i+1:], i+1):
                if interaction_measure == 'mutual_info':
                    # Use mutual information
                    X1 = param_arrays[param1].reshape(-1, 1)
                    y2 = param_arrays[param2]
                    
                    # Normalize data
                    scaler = StandardScaler()
                    X1_scaled = scaler.fit_transform(X1)
                    
                    mi = mutual_info_regression(X1_scaled, y2, random_state=42)
                    interaction_strength = mi[0]
                
                elif interaction_measure == 'chi2':
                    # Use chi-squared test (for categorical data)
                    # Convert continuous to categorical bins
                    bins1 = np.digitize(param_arrays[param1], np.linspace(
                        param_arrays[param1].min(), param_arrays[param1].max(), 5
                    ))
                    bins2 = np.digitize(param_arrays[param2], np.linspace(
                        param_arrays[param2].min(), param_arrays[param2].max(), 5
                    ))
                    
                    from scipy.stats import chi2_contingency
                    contingency = np.histogram2d(bins1, bins2, bins=5)[0]
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    interaction_strength = chi2 / (n_samples * min(contingency.shape))
                
                else:
                    raise ValueError(f"Unknown interaction measure: {interaction_measure}")
                
                interactions[(param1, param2)] = interaction_strength
        
        logger.info(f"Computed {len(interactions)} parameter interactions")
        
        return interactions
    
    def generate_correlation_report(self) -> Dict[str, Any]:
        """Generate comprehensive correlation analysis report.
        
        Returns:
            Dictionary containing complete correlation analysis
        """
        if not self.correlations or not self.parameter_stats:
            logger.warning("No correlation data available for report")
            return {}
        
        report = {
            'correlations': self.correlations,
            'parameter_statistics': self.parameter_stats,
            'summary': {
                'n_parameters': len(self.parameter_stats),
                'correlation_matrix_size': len(self.correlations),
                'analysis_timestamp': time.time()
            }
        }
        
        # Find strongest correlations
        strong_correlations = []
        for param1, corr_dict in self.correlations.items():
            for param2, corr_val in corr_dict.items():
                if param1 != param2 and abs(corr_val) > 0.5:
                    strong_correlations.append((param1, param2, corr_val))
        
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        report['strong_correlations'] = strong_correlations[:10]  # Top 10
        
        logger.info(f"Generated correlation report with {len(strong_correlations)} strong correlations")
        
        return report


class MultiParameterPersistenceComputer:
    """Computes persistence for multi-parameter filtrations.
    
    This class extends single-parameter persistence computation to handle
    multi-dimensional parameter spaces, tracking birth and death events
    across multiple filtration parameters simultaneously.
    """
    
    def __init__(self, 
                 max_eigenvalues: int = 100,
                 eigenvalue_method: str = 'lobpcg',
                 enable_caching: bool = True):
        """Initialize multi-parameter persistence computer.
        
        Args:
            max_eigenvalues: Maximum number of eigenvalues to compute
            eigenvalue_method: Method for eigenvalue computation ('lobpcg', 'dense')
            enable_caching: Whether to cache intermediate computations
        """
        self.max_eigenvalues = max_eigenvalues
        self.eigenvalue_method = eigenvalue_method
        self.enable_caching = enable_caching
        
        # Cache for expensive computations
        self._eigenvalue_cache = {} if enable_caching else None
        self._laplacian_cache = {} if enable_caching else None
        
        logger.info(f"MultiParameterPersistenceComputer initialized: "
                   f"max_eigenvals={max_eigenvalues}, method={eigenvalue_method}")
    
    def compute_multi_parameter_persistence(self,
                                          sheaf: Sheaf,
                                          filtration: MultiParameterFiltration,
                                          threshold_functions: Dict[str, Callable[[float, float], bool]],
                                          progress_callback: Optional[Callable] = None) -> Dict:
        """Compute persistence for multi-parameter filtration.
        
        Args:
            sheaf: Sheaf object to analyze
            filtration: Multi-parameter filtration specification
            threshold_functions: Dictionary mapping parameter names to threshold functions
            progress_callback: Optional callback for progress reporting
        
        Returns:
            Dictionary containing multi-parameter persistence results
        """
        logger.info(f"Computing multi-parameter persistence for {len(filtration)} parameter points")
        start_time = time.time()
        
        # Validate threshold functions
        for param_name in filtration.parameter_names:
            if param_name not in threshold_functions:
                raise ValueError(f"Missing threshold function for parameter {param_name}")
        
        # Import required components
        from .static_laplacian_masking import StaticLaplacianWithMasking
        from .tracker import SubspaceTracker
        
        # Initialize components
        static_analyzer = StaticLaplacianWithMasking(
            max_eigenvalues=self.max_eigenvalues,
            eigenvalue_method=self.eigenvalue_method
        )
        subspace_tracker = SubspaceTracker()
        
        # Pre-compute base Laplacian (cached)
        base_laplacian, base_metadata = self._get_cached_base_laplacian(sheaf, static_analyzer)
        edge_info = static_analyzer._extract_edge_info(sheaf, base_laplacian, base_metadata)
        
        # Initialize results storage
        results = {
            'eigenvalue_tensor': {},  # ParameterPoint -> eigenvalues
            'eigenvector_tensor': {},  # ParameterPoint -> eigenvectors
            'persistence_events': [],  # Multi-parameter birth/death events
            'parameter_space': filtration,
            'computation_metadata': {
                'total_points': len(filtration),
                'computed_points': 0,
                'failed_points': 0,
                'computation_time': 0.0
            }
        }
        
        # Compute eigenvalues for each parameter point
        for i, param_point in enumerate(filtration):
            try:
                # Create combined edge mask for this parameter point
                combined_mask = self._create_multi_parameter_mask(
                    edge_info, param_point, threshold_functions, filtration.parameter_names
                )
                
                # Apply mask and compute eigenvalues
                masked_laplacian = static_analyzer._apply_edge_mask_sparse(
                    base_laplacian, combined_mask, edge_info, base_metadata
                )
                
                eigenvals, eigenvecs = self._compute_eigenvalues_cached(
                    masked_laplacian, param_point
                )
                
                # Store results
                results['eigenvalue_tensor'][param_point] = eigenvals
                results['eigenvector_tensor'][param_point] = eigenvecs
                results['computation_metadata']['computed_points'] += 1
                
                # Progress reporting
                if progress_callback and (i + 1) % max(1, len(filtration) // 20) == 0:
                    progress = (i + 1) / len(filtration)
                    progress_callback(progress, f"Computed {i + 1}/{len(filtration)} points")
                
            except Exception as e:
                logger.warning(f"Failed to compute eigenvalues for parameter point {i}: {e}")
                results['computation_metadata']['failed_points'] += 1
                continue
        
        # Compute multi-parameter persistence events
        logger.info("Computing multi-parameter persistence events...")
        persistence_events = self._compute_persistence_events(
            results['eigenvalue_tensor'],
            results['eigenvector_tensor'],
            filtration,
            subspace_tracker
        )
        results['persistence_events'] = persistence_events
        
        # Finalize metadata
        computation_time = time.time() - start_time
        results['computation_metadata']['computation_time'] = computation_time
        
        logger.info(f"Multi-parameter persistence computation completed in {computation_time:.2f}s. "
                   f"Computed {results['computation_metadata']['computed_points']} points, "
                   f"failed {results['computation_metadata']['failed_points']} points")
        
        return results
    
    def _get_cached_base_laplacian(self, sheaf: Sheaf, static_analyzer):
        """Get base Laplacian with caching."""
        sheaf_hash = self._hash_sheaf(sheaf)
        
        if self.enable_caching and sheaf_hash in self._laplacian_cache:
            logger.debug("Using cached base Laplacian")
            return self._laplacian_cache[sheaf_hash]
        
        # Compute base Laplacian
        laplacian, metadata = static_analyzer._get_cached_laplacian(sheaf)
        
        if self.enable_caching:
            self._laplacian_cache[sheaf_hash] = (laplacian, metadata)
        
        return laplacian, metadata
    
    def _hash_sheaf(self, sheaf: Sheaf) -> str:
        """Create hash for sheaf caching."""
        # Simple hash based on sheaf structure
        node_hash = hash(tuple(sorted(sheaf.stalks.keys())))
        edge_hash = hash(tuple(sorted(sheaf.restrictions.keys())))
        return f"{node_hash}_{edge_hash}"
    
    def _create_multi_parameter_mask(self,
                                   edge_info: Dict,
                                   param_point: ParameterPoint,
                                   threshold_functions: Dict[str, Callable],
                                   parameter_names: List[str]) -> Dict:
        """Create combined edge mask for multi-parameter point."""
        combined_mask = {}
        
        for edge, info in edge_info.items():
            edge_weight = info['weight']
            
            # Apply all threshold functions
            include_edge = True
            for i, param_name in enumerate(parameter_names):
                param_value = param_point.coordinates[i]
                threshold_func = threshold_functions[param_name]
                
                if not threshold_func(edge_weight, param_value):
                    include_edge = False
                    break
            
            combined_mask[edge] = include_edge
        
        return combined_mask
    
    def _compute_eigenvalues_cached(self, 
                                  laplacian: torch.sparse.FloatTensor,
                                  param_point: ParameterPoint) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues with caching."""
        if not self.enable_caching:
            return self._compute_eigenvalues_direct(laplacian)
        
        # Create cache key
        laplacian_shape = laplacian.shape
        nnz = laplacian._nnz()
        cache_key = f"{param_point.index}_{laplacian_shape}_{nnz}"
        
        if cache_key in self._eigenvalue_cache:
            logger.debug(f"Using cached eigenvalues for parameter point {param_point.index}")
            return self._eigenvalue_cache[cache_key]
        
        # Compute eigenvalues
        eigenvals, eigenvecs = self._compute_eigenvalues_direct(laplacian)
        
        # Cache results
        self._eigenvalue_cache[cache_key] = (eigenvals, eigenvecs)
        
        return eigenvals, eigenvecs
    
    def _compute_eigenvalues_direct(self, 
                                  laplacian: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Direct eigenvalue computation."""
        if self.eigenvalue_method == 'lobpcg':
            return self._compute_eigenvalues_lobpcg(laplacian)
        elif self.eigenvalue_method == 'dense':
            return self._compute_eigenvalues_dense(laplacian)
        else:
            raise ValueError(f"Unknown eigenvalue method: {self.eigenvalue_method}")
    
    def _compute_eigenvalues_lobpcg(self, 
                                  laplacian: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues using LOBPCG."""
        from scipy.sparse.linalg import lobpcg
        import scipy.sparse as sp
        
        # Convert to scipy format
        coo = laplacian.coalesce()
        indices = coo.indices().cpu().numpy()
        values = coo.values().cpu().numpy()
        shape = coo.shape
        
        L_scipy = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
        
        # Initial guess
        n = L_scipy.shape[0]
        k = min(self.max_eigenvalues, n - 1)
        if k <= 0:
            # Handle trivial case
            return torch.zeros(0), torch.zeros(n, 0)
        
        X = np.random.randn(n, k)
        
        try:
            # Compute smallest eigenvalues
            eigenvals, eigenvecs = lobpcg(L_scipy, X, largest=False, tol=1e-8, maxiter=1000)
            
            # Convert back to torch
            eigenvals = torch.from_numpy(eigenvals).float()
            eigenvecs = torch.from_numpy(eigenvecs).float()
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            logger.warning(f"LOBPCG failed, falling back to dense: {e}")
            return self._compute_eigenvalues_dense(laplacian)
    
    def _compute_eigenvalues_dense(self, 
                                 laplacian: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues using dense solver."""
        # Convert to dense
        L_dense = laplacian.to_dense()
        
        # Compute eigenvalues
        eigenvals, eigenvecs = torch.linalg.eigh(L_dense)
        
        # Return smallest eigenvalues
        k = min(self.max_eigenvalues, len(eigenvals))
        return eigenvals[:k], eigenvecs[:, :k]
    
    def _compute_persistence_events(self,
                                  eigenvalue_tensor: Dict[ParameterPoint, torch.Tensor],
                                  eigenvector_tensor: Dict[ParameterPoint, torch.Tensor],
                                  filtration: MultiParameterFiltration,
                                  subspace_tracker: 'SubspaceTracker') -> List[Dict]:
        """Compute multi-parameter persistence events."""
        events = []
        
        # For multi-parameter persistence, we need to track changes along
        # paths in parameter space. For now, implement a simplified version
        # that tracks along coordinate axes.
        
        for dim in range(filtration.dimension):
            axis_events = self._compute_axis_persistence_events(
                eigenvalue_tensor, eigenvector_tensor, filtration, dim, subspace_tracker
            )
            events.extend(axis_events)
        
        logger.info(f"Computed {len(events)} multi-parameter persistence events")
        return events
    
    def _compute_axis_persistence_events(self,
                                       eigenvalue_tensor: Dict,
                                       eigenvector_tensor: Dict,
                                       filtration: MultiParameterFiltration,
                                       axis_dim: int,
                                       subspace_tracker: 'SubspaceTracker') -> List[Dict]:
        """Compute persistence events along a coordinate axis."""
        events = []
        
        # Get parameter values along this axis
        axis_param_name = filtration.parameter_names[axis_dim]
        
        # Group points by other coordinates (fix other dimensions)
        coordinate_groups = {}
        for point in filtration.parameter_points:
            if point not in eigenvalue_tensor:
                continue
            
            # Create key from all coordinates except axis_dim
            other_coords = tuple(
                coord for i, coord in enumerate(point.coordinates) if i != axis_dim
            )
            
            if other_coords not in coordinate_groups:
                coordinate_groups[other_coords] = []
            coordinate_groups[other_coords].append(point)
        
        # Process each group (1D slice through parameter space)
        for other_coords, points in coordinate_groups.items():
            if len(points) < 2:
                continue
            
            # Sort points by axis coordinate
            points.sort(key=lambda p: p.coordinates[axis_dim])
            
            # Create sequences for tracking
            eigenval_sequence = [eigenvalue_tensor[p] for p in points]
            eigenvec_sequence = [eigenvector_tensor[p] for p in points]
            axis_values = [p.coordinates[axis_dim] for p in points]
            
            try:
                # Track eigenspaces along this axis
                tracking_info = subspace_tracker.track_eigenspaces(
                    eigenval_sequence, eigenvec_sequence, axis_values
                )
                
                # Convert single-parameter events to multi-parameter events
                for birth_event in tracking_info['birth_events']:
                    event = {
                        'type': 'birth',
                        'parameter_dimension': axis_dim,
                        'parameter_name': axis_param_name,
                        'parameter_point': points[birth_event['step']],
                        'fixed_coordinates': other_coords,
                        'group': birth_event['group'],
                        'filtration_value': birth_event['filtration_param']
                    }
                    events.append(event)
                
                for death_event in tracking_info['death_events']:
                    event = {
                        'type': 'death',
                        'parameter_dimension': axis_dim,
                        'parameter_name': axis_param_name,
                        'parameter_point': points[death_event['step']],
                        'fixed_coordinates': other_coords,
                        'group': death_event['group'],
                        'filtration_value': death_event['filtration_param']
                    }
                    events.append(event)
                    
            except Exception as e:
                logger.warning(f"Failed to track eigenspaces for coordinate group {other_coords}: {e}")
                continue
        
        return events
    
    def extract_multi_parameter_features(self, persistence_result: Dict) -> Dict:
        """Extract features from multi-parameter persistence results.
        
        Args:
            persistence_result: Results from compute_multi_parameter_persistence
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        eigenvalue_tensor = persistence_result['eigenvalue_tensor']
        persistence_events = persistence_result['persistence_events']
        filtration = persistence_result['parameter_space']
        
        # Basic statistics
        features['n_parameter_points'] = len(eigenvalue_tensor)
        features['n_persistence_events'] = len(persistence_events)
        features['parameter_space_dimension'] = filtration.dimension
        features['parameter_names'] = filtration.parameter_names
        
        # Event statistics by type
        birth_events = [e for e in persistence_events if e['type'] == 'birth']
        death_events = [e for e in persistence_events if e['type'] == 'death']
        
        features['n_birth_events'] = len(birth_events)
        features['n_death_events'] = len(death_events)
        
        # Event statistics by parameter dimension
        events_by_dim = {}
        for dim in range(filtration.dimension):
            dim_events = [e for e in persistence_events if e['parameter_dimension'] == dim]
            events_by_dim[filtration.parameter_names[dim]] = {
                'n_events': len(dim_events),
                'n_births': len([e for e in dim_events if e['type'] == 'birth']),
                'n_deaths': len([e for e in dim_events if e['type'] == 'death'])
            }
        
        features['events_by_parameter'] = events_by_dim
        
        # Eigenvalue statistics across parameter space
        all_eigenvals = []
        for eigenvals in eigenvalue_tensor.values():
            all_eigenvals.extend(eigenvals.tolist())
        
        if all_eigenvals:
            all_eigenvals = np.array(all_eigenvals)
            features['eigenvalue_statistics'] = {
                'mean': np.mean(all_eigenvals),
                'std': np.std(all_eigenvals),
                'min': np.min(all_eigenvals),
                'max': np.max(all_eigenvals),
                'n_zero': np.sum(all_eigenvals < 1e-8),
                'n_small': np.sum(all_eigenvals < 1e-6)
            }
        else:
            features['eigenvalue_statistics'] = {}
        
        # Parameter space coverage
        computed_points = len(eigenvalue_tensor)
        total_points = len(filtration)
        features['parameter_space_coverage'] = computed_points / total_points if total_points > 0 else 0.0
        
        # Computation metadata
        if 'computation_metadata' in persistence_result:
            features['computation_metadata'] = persistence_result['computation_metadata']
        
        logger.info(f"Extracted {len(features)} multi-parameter features")
        
        return features
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.enable_caching:
            self._eigenvalue_cache.clear()
            self._laplacian_cache.clear()
            logger.info("Cleared multi-parameter persistence cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        if not self.enable_caching:
            return {'caching_enabled': False}
        
        return {
            'caching_enabled': True,
            'eigenvalue_cache_size': len(self._eigenvalue_cache),
            'laplacian_cache_size': len(self._laplacian_cache),
            'total_cached_items': len(self._eigenvalue_cache) + len(self._laplacian_cache)
        }


class MultiParameterSpectralAnalyzer:
    """High-level interface for multi-parameter persistent spectral analysis.
    
    This class provides the main user interface for performing multi-parameter
    persistent spectral analysis of neural sheaves, integrating all components
    into a unified workflow.
    
    Attributes:
        persistence_computer: MultiParameterPersistenceComputer instance
        correlation_analyzer: ParameterCorrelationAnalyzer instance
        default_grid_sizes: Default grid sizes for parameter discretization
    """
    
    def __init__(self,
                 persistence_computer: Optional[MultiParameterPersistenceComputer] = None,
                 correlation_analyzer: Optional[ParameterCorrelationAnalyzer] = None,
                 default_grid_sizes: Optional[List[int]] = None):
        """Initialize multi-parameter spectral analyzer.
        
        Args:
            persistence_computer: MultiParameterPersistenceComputer instance
            correlation_analyzer: ParameterCorrelationAnalyzer instance  
            default_grid_sizes: Default grid sizes for parameter discretization
        """
        self.persistence_computer = persistence_computer or MultiParameterPersistenceComputer()
        self.correlation_analyzer = correlation_analyzer or ParameterCorrelationAnalyzer()
        self.default_grid_sizes = default_grid_sizes or [10, 10, 8]  # 2D, 3D, higher-D defaults
        
        logger.info("MultiParameterSpectralAnalyzer initialized")
    
    def analyze_multi_parameter(self,
                              sheaf: Sheaf,
                              parameter_specifications: Dict[str, Dict],
                              grid_sizes: Optional[List[int]] = None,
                              enable_correlation_analysis: bool = True,
                              progress_callback: Optional[Callable] = None) -> Dict:
        """Perform complete multi-parameter persistent spectral analysis.
        
        Args:
            sheaf: Sheaf object to analyze
            parameter_specifications: Dictionary specifying parameters:
                {
                    'param_name': {
                        'range': (min_val, max_val),
                        'threshold_func': callable
                    }
                }
            grid_sizes: Grid sizes for each parameter (auto-determined if None)
            enable_correlation_analysis: Whether to perform parameter correlation analysis
            progress_callback: Optional progress reporting callback
        
        Returns:
            Complete multi-parameter analysis results
        """
        logger.info(f"Starting multi-parameter analysis with {len(parameter_specifications)} parameters")
        start_time = time.time()
        
        # Validate parameter specifications
        self._validate_parameter_specifications(parameter_specifications)
        
        # Extract parameter information
        parameter_names = list(parameter_specifications.keys())
        parameter_ranges = [spec['range'] for spec in parameter_specifications.values()]
        threshold_functions = {
            name: spec['threshold_func'] 
            for name, spec in parameter_specifications.items()
        }
        
        # Determine grid sizes
        if grid_sizes is None:
            grid_sizes = self._auto_determine_grid_sizes(len(parameter_names))
        
        if len(grid_sizes) != len(parameter_names):
            raise ValueError(f"Grid sizes length ({len(grid_sizes)}) must match "
                           f"number of parameters ({len(parameter_names)})")
        
        # Create multi-parameter filtration
        filtration = MultiParameterFiltration(
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges,
            grid_sizes=grid_sizes
        )
        
        logger.info(f"Created {filtration.dimension}D parameter space with {len(filtration)} points")
        
        # Compute multi-parameter persistence
        persistence_result = self.persistence_computer.compute_multi_parameter_persistence(
            sheaf=sheaf,
            filtration=filtration,
            threshold_functions=threshold_functions,
            progress_callback=progress_callback
        )
        
        # Extract features
        features = self.persistence_computer.extract_multi_parameter_features(persistence_result)
        
        # Perform correlation analysis if requested
        correlation_results = {}
        if enable_correlation_analysis:
            try:
                correlation_results = self._perform_correlation_analysis(
                    persistence_result, parameter_specifications
                )
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
                correlation_results = {'error': str(e)}
        
        # Create comprehensive results
        analysis_time = time.time() - start_time
        results = {
            'persistence_result': persistence_result,
            'features': features,
            'correlation_analysis': correlation_results,
            'filtration_specification': {
                'parameter_names': parameter_names,
                'parameter_ranges': parameter_ranges,
                'grid_sizes': grid_sizes,
                'total_parameter_points': len(filtration)
            },
            'analysis_metadata': {
                'analysis_time': analysis_time,
                'sheaf_nodes': len(sheaf.stalks),
                'sheaf_edges': len(sheaf.restrictions),
                'parameter_space_dimension': filtration.dimension,
                'analysis_timestamp': time.time()
            }
        }
        
        logger.info(f"Multi-parameter analysis completed in {analysis_time:.2f}s")
        
        return results
    
    def analyze_2d_parameter_space(self,
                                 sheaf: Sheaf,
                                 param1_spec: Dict,
                                 param2_spec: Dict,
                                 grid_size: Tuple[int, int] = (10, 10),
                                 enable_visualization_data: bool = True) -> Dict:
        """Convenient method for 2D parameter space analysis.
        
        Args:
            sheaf: Sheaf object to analyze
            param1_spec: Specification for first parameter:
                {'name': str, 'range': (min, max), 'threshold_func': callable}
            param2_spec: Specification for second parameter:
                {'name': str, 'range': (min, max), 'threshold_func': callable}
            grid_size: Grid size for (param1, param2)
            enable_visualization_data: Whether to prepare data for visualization
        
        Returns:
            Analysis results optimized for 2D visualization
        """
        # Create parameter specifications
        parameter_specifications = {
            param1_spec['name']: {
                'range': param1_spec['range'],
                'threshold_func': param1_spec['threshold_func']
            },
            param2_spec['name']: {
                'range': param2_spec['range'],
                'threshold_func': param2_spec['threshold_func']
            }
        }
        
        # Perform analysis
        results = self.analyze_multi_parameter(
            sheaf=sheaf,
            parameter_specifications=parameter_specifications,
            grid_sizes=list(grid_size)
        )
        
        # Add 2D-specific processing
        if enable_visualization_data:
            results['visualization_data'] = self._prepare_2d_visualization_data(results)
        
        return results
    
    def analyze_parameter_sensitivity(self,
                                    sheaf: Sheaf,
                                    base_parameter_specs: Dict[str, Dict],
                                    sensitivity_parameter: str,
                                    sensitivity_range: Tuple[float, float],
                                    n_sensitivity_points: int = 20) -> Dict:
        """Analyze sensitivity of persistence to one parameter while fixing others.
        
        Args:
            sheaf: Sheaf object to analyze
            base_parameter_specs: Base parameter specifications
            sensitivity_parameter: Name of parameter to vary for sensitivity analysis
            sensitivity_range: Range for sensitivity parameter
            n_sensitivity_points: Number of points for sensitivity analysis
        
        Returns:
            Sensitivity analysis results
        """
        logger.info(f"Performing sensitivity analysis for parameter '{sensitivity_parameter}'")
        
        if sensitivity_parameter not in base_parameter_specs:
            raise ValueError(f"Sensitivity parameter '{sensitivity_parameter}' not in base specifications")
        
        # Create sensitivity parameter values
        sensitivity_values = np.linspace(
            sensitivity_range[0], sensitivity_range[1], n_sensitivity_points
        )
        
        sensitivity_results = []
        
        for i, sens_value in enumerate(sensitivity_values):
            logger.debug(f"Sensitivity analysis step {i+1}/{n_sensitivity_points}: "
                        f"{sensitivity_parameter} = {sens_value:.4f}")
            
            # Create modified parameter specifications
            current_specs = base_parameter_specs.copy()
            current_specs[sensitivity_parameter]['range'] = (sens_value, sens_value)
            
            # Create single-point "grid" for the sensitivity parameter
            grid_sizes = [1 if name == sensitivity_parameter else 3 
                         for name in current_specs.keys()]
            
            try:
                # Perform analysis for this sensitivity point
                result = self.analyze_multi_parameter(
                    sheaf=sheaf,
                    parameter_specifications=current_specs,
                    grid_sizes=grid_sizes,
                    enable_correlation_analysis=False  # Skip for efficiency
                )
                
                sensitivity_results.append({
                    'parameter_value': sens_value,
                    'features': result['features'],
                    'n_persistence_events': result['features']['n_persistence_events'],
                    'parameter_space_coverage': result['features']['parameter_space_coverage']
                })
                
            except Exception as e:
                logger.warning(f"Sensitivity analysis failed for {sensitivity_parameter}={sens_value}: {e}")
                sensitivity_results.append({
                    'parameter_value': sens_value,
                    'error': str(e)
                })
        
        # Compute sensitivity metrics
        sensitivity_metrics = self._compute_sensitivity_metrics(sensitivity_results, sensitivity_parameter)
        
        return {
            'sensitivity_parameter': sensitivity_parameter,
            'sensitivity_range': sensitivity_range,
            'sensitivity_values': sensitivity_values.tolist(),
            'sensitivity_results': sensitivity_results,
            'sensitivity_metrics': sensitivity_metrics
        }
    
    def compare_multi_parameter_analyses(self,
                                       sheaves: List[Sheaf],
                                       parameter_specifications: Dict[str, Dict],
                                       comparison_metrics: Optional[List[str]] = None) -> Dict:
        """Compare multi-parameter analyses across multiple sheaves.
        
        Args:
            sheaves: List of sheaf objects to analyze and compare
            parameter_specifications: Common parameter specifications for all sheaves
            comparison_metrics: Specific metrics to compare (default: all available)
        
        Returns:
            Comparative analysis results
        """
        logger.info(f"Comparing multi-parameter analyses across {len(sheaves)} sheaves")
        
        if comparison_metrics is None:
            comparison_metrics = [
                'n_persistence_events', 'n_birth_events', 'n_death_events',
                'parameter_space_coverage', 'eigenvalue_statistics'
            ]
        
        # Analyze each sheaf
        individual_results = []
        for i, sheaf in enumerate(sheaves):
            logger.debug(f"Analyzing sheaf {i+1}/{len(sheaves)}")
            
            try:
                result = self.analyze_multi_parameter(
                    sheaf=sheaf,
                    parameter_specifications=parameter_specifications,
                    enable_correlation_analysis=False  # Skip for efficiency
                )
                individual_results.append(result)
                
            except Exception as e:
                logger.warning(f"Analysis failed for sheaf {i}: {e}")
                individual_results.append({'error': str(e)})
        
        # Compute comparative metrics
        comparative_metrics = self._compute_comparative_metrics(
            individual_results, comparison_metrics
        )
        
        return {
            'individual_results': individual_results,
            'comparative_metrics': comparative_metrics,
            'n_sheaves': len(sheaves),
            'parameter_specifications': parameter_specifications
        }
    
    def _validate_parameter_specifications(self, parameter_specifications: Dict[str, Dict]):
        """Validate parameter specifications format."""
        for name, spec in parameter_specifications.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Parameter specification for '{name}' must be a dictionary")
            
            if 'range' not in spec:
                raise ValueError(f"Parameter '{name}' missing 'range' specification")
            
            if 'threshold_func' not in spec:
                raise ValueError(f"Parameter '{name}' missing 'threshold_func' specification")
            
            range_val = spec['range']
            if not isinstance(range_val, (tuple, list)) or len(range_val) != 2:
                raise ValueError(f"Parameter '{name}' range must be tuple/list of 2 values")
            
            if range_val[0] >= range_val[1]:
                raise ValueError(f"Parameter '{name}' range must have min < max")
            
            if not callable(spec['threshold_func']):
                raise ValueError(f"Parameter '{name}' threshold_func must be callable")
    
    def _auto_determine_grid_sizes(self, n_parameters: int) -> List[int]:
        """Automatically determine appropriate grid sizes."""
        if n_parameters <= len(self.default_grid_sizes):
            return self.default_grid_sizes[:n_parameters]
        else:
            # For higher dimensions, use smaller grid sizes to control computation
            base_sizes = self.default_grid_sizes + [6] * (n_parameters - len(self.default_grid_sizes))
            return base_sizes[:n_parameters]
    
    def _perform_correlation_analysis(self,
                                    persistence_result: Dict,
                                    parameter_specifications: Dict[str, Dict]) -> Dict:
        """Perform correlation analysis on persistence results."""
        # Extract parameter values and persistence features
        parameter_values = {}
        filtration = persistence_result['parameter_space']
        
        for param_name in filtration.parameter_names:
            param_index = filtration.parameter_names.index(param_name)
            values = [point.coordinates[param_index] for point in filtration.parameter_points]
            parameter_values[param_name] = values
        
        # Add persistence features as "parameters" for correlation analysis
        events = persistence_result['persistence_events']
        birth_events_by_param = {}
        death_events_by_param = {}
        
        for param_name in filtration.parameter_names:
            param_births = [e for e in events if e['parameter_name'] == param_name and e['type'] == 'birth']
            param_deaths = [e for e in events if e['parameter_name'] == param_name and e['type'] == 'death']
            
            birth_events_by_param[f"{param_name}_births"] = [len(param_births)] * len(filtration)
            death_events_by_param[f"{param_name}_deaths"] = [len(param_deaths)] * len(filtration)
        
        # Combine all values for correlation analysis
        all_values = {**parameter_values, **birth_events_by_param, **death_events_by_param}
        
        # Compute correlations
        correlations = self.correlation_analyzer.compute_parameter_correlations(all_values)
        statistics = self.correlation_analyzer.compute_parameter_statistics(all_values)
        
        return {
            'correlations': correlations,
            'statistics': statistics,
            'parameter_effects': birth_events_by_param
        }
    
    def _prepare_2d_visualization_data(self, results: Dict) -> Dict:
        """Prepare data structures optimized for 2D visualization."""
        persistence_result = results['persistence_result']
        filtration = persistence_result['parameter_space']
        
        if filtration.dimension != 2:
            logger.warning("2D visualization data requested for non-2D parameter space")
            return {}
        
        # Create 2D grids for visualization
        param1_name, param2_name = filtration.parameter_names
        grid_size1, grid_size2 = filtration.grid_sizes
        
        # Initialize grids
        eigenvalue_grids = {}
        persistence_event_grid = np.zeros((grid_size1, grid_size2))
        
        # Fill grids with data
        for point in filtration.parameter_points:
            if point in persistence_result['eigenvalue_tensor']:
                i, j = self._point_to_grid_indices(point, filtration)
                
                # Store eigenvalue information
                eigenvals = persistence_result['eigenvalue_tensor'][point]
                eigenvalue_grids[f'grid_{i}_{j}'] = eigenvals.tolist()
                
                # Count persistence events at this point
                point_events = [
                    e for e in persistence_result['persistence_events']
                    if e.get('parameter_point') == point
                ]
                persistence_event_grid[i, j] = len(point_events)
        
        return {
            'parameter_names': [param1_name, param2_name],
            'grid_sizes': [grid_size1, grid_size2],
            'parameter_ranges': filtration.parameter_ranges,
            'eigenvalue_grids': eigenvalue_grids,
            'persistence_event_grid': persistence_event_grid.tolist(),
            'total_computed_points': len(persistence_result['eigenvalue_tensor'])
        }
    
    def _point_to_grid_indices(self, point: ParameterPoint, filtration: MultiParameterFiltration) -> Tuple[int, int]:
        """Convert parameter point to 2D grid indices."""
        coord1, coord2 = point.coordinates
        range1, range2 = filtration.parameter_ranges
        size1, size2 = filtration.grid_sizes
        
        # Map coordinates to grid indices
        i = int((coord1 - range1[0]) / (range1[1] - range1[0]) * (size1 - 1))
        j = int((coord2 - range2[0]) / (range2[1] - range2[0]) * (size2 - 1))
        
        # Clamp to valid range
        i = max(0, min(size1 - 1, i))
        j = max(0, min(size2 - 1, j))
        
        return i, j
    
    def _compute_sensitivity_metrics(self, sensitivity_results: List[Dict], parameter_name: str) -> Dict:
        """Compute sensitivity metrics from sensitivity analysis results."""
        # Extract valid results (no errors)
        valid_results = [r for r in sensitivity_results if 'error' not in r]
        
        if len(valid_results) < 2:
            return {'error': 'Insufficient valid results for sensitivity analysis'}
        
        # Extract time series
        param_values = [r['parameter_value'] for r in valid_results]
        n_events = [r['n_persistence_events'] for r in valid_results]
        
        # Compute sensitivity measures
        sensitivity_metrics = {
            'parameter_name': parameter_name,
            'n_valid_points': len(valid_results),
            'parameter_range': (min(param_values), max(param_values)),
            'event_count_range': (min(n_events), max(n_events)),
            'event_count_variance': np.var(n_events),
            'parameter_sensitivity': np.std(n_events) / np.mean(n_events) if np.mean(n_events) > 0 else 0,
        }
        
        # Compute correlation between parameter and event count
        if len(param_values) > 1:
            correlation = np.corrcoef(param_values, n_events)[0, 1]
            sensitivity_metrics['parameter_event_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        return sensitivity_metrics
    
    def _compute_comparative_metrics(self, individual_results: List[Dict], comparison_metrics: List[str]) -> Dict:
        """Compute comparative metrics across multiple analyses."""
        valid_results = [r for r in individual_results if 'error' not in r]
        
        if len(valid_results) < 2:
            return {'error': 'Insufficient valid results for comparison'}
        
        comparative_metrics = {}
        
        for metric_name in comparison_metrics:
            values = []
            for result in valid_results:
                try:
                    if metric_name in result['features']:
                        value = result['features'][metric_name]
                        if isinstance(value, dict):
                            # For nested metrics, use a summary statistic
                            if 'mean' in value:
                                values.append(value['mean'])
                        else:
                            values.append(value)
                except Exception:
                    continue
            
            if values:
                comparative_metrics[metric_name] = {
                    'values': values,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                }
        
        return comparative_metrics
    
    def clear_cache(self):
        """Clear all cached data."""
        self.persistence_computer.clear_cache()
        logger.info("Cleared MultiParameterSpectralAnalyzer cache")
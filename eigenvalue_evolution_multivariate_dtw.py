#!/usr/bin/env python3
"""
Eigenvalue Evolution Multivariate DTW Comparison Script

This script compares eigenvalue evolution across filtration between neural networks
using multivariate Dynamic Time Warping (DTW) from tslearn with parallel processing.

Key features:
1. Multivariate DTW for simultaneous comparison of multiple eigenvalue trajectories
2. Log transformation for better handling of eigenvalue scales
3. Linear interpolation for common grid resampling
4. Eigenvalue collapse and rise handling with adaptive thresholding
5. Parallel processing for efficiency
6. Hungarian matching for permutation-invariant set-to-set distance

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv
    python eigenvalue_evolution_multivariate_dtw.py
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import time
import concurrent.futures as futures

# DTW imports
try:
    from tslearn.metrics import dtw, dtw_path
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    warnings.warn("tslearn not available. This script requires tslearn for multivariate DTW.")

# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Neurosheaf imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.sheaf.core.gw_config import GWConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MultivariateDTWConfig:
    """Configuration for multivariate DTW comparison pipeline.
    
    Optimized for comparing eigenvalue evolution patterns between neural networks
    using tslearn's multivariate DTW with log-scale transformation and adaptive
    preprocessing for optimal trained vs random model discrimination.
    """
    # Data generation
    data_batch_size: int = 100
    random_seed: int = 456  # Try different seed again
    
    # Eigenvalue selection and processing - SPECTRAL GAP FOCUS
    top_n_eigenvalues: int =50
    interpolation_points: int = 200  # Lower resolution to emphasize major patterns
    
    # Log transformation parameters - NO LOG TRANSFORMATION
    use_log_transform: bool = True  # NO log to preserve raw scale differences
    log_floor_epsilon: float = 1e-15  # Not used
    adaptive_threshold_percentile: float = 0.5  # Not used
    adaptive_threshold_multiplier: float = 1e-2  # Not used
    
    # DTW parameters - VERY LOOSE CONSTRAINTS
    dtw_global_constraint: str = "sakoe_chiba"  
    dtw_constraint_ratio: float = 0  # Very loose constraint
    
    # Normalization and preprocessing - PRESERVE ALL RAW DIFFERENCES
    use_sequence_normalization: bool = False  # NO normalization 
    normalization_method: str = "none"  # No normalization at all
    handle_nan_inf: bool = True  # Handle NaN/Inf values robustly
    
    # Advanced DTW optimization parameters
    enable_early_stopping: bool = True  # Stop DTW computation early if distance is very large
    early_stopping_threshold: float = 100.0  # Threshold for early stopping
    use_adaptive_constraints: bool = True  # Adapt constraint radius based on sequence similarity
    similarity_threshold_for_constraint: float = 0.8  # Threshold for constraint adaptation
    
    # Hungarian matching for set distances
    use_trimmed_mean: bool = True
    trim_percentage: float = 0.05
    
    # Parallel processing - OPTIMIZED
    num_workers_models: int = 5  # Optimal for I/O bound model loading
    num_workers_dtw: int = 10    # Balanced for CPU-bound DTW computation
    batch_size_dtw: int = 500    # Process DTW computations in smaller batches
    
    # Visualization and output
    save_visualizations: bool = True
    output_dir: str = "multivariate_dtw_results"
    
    # Logging control - OPTIMIZED for less verbose output
    verbose_mode: bool = False  # Reduce detailed printing during computation
    progress_update_freq: int = 10  # Update progress every N operations (reduce frequency)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Note: tslearn availability is checked at runtime in main() rather than config creation
        
        if self.top_n_eigenvalues <= 0:
            raise ValueError("top_n_eigenvalues must be positive")
        
        if self.interpolation_points <= 1:
            raise ValueError("interpolation_points must be greater than 1")
        
        if not 0 <= self.dtw_constraint_ratio <= 1:
            raise ValueError("dtw_constraint_ratio must be between 0 and 1")
        
        if not 0 <= self.trim_percentage < 0.5:
            raise ValueError("trim_percentage must be between 0 and 0.5")
        
        if self.normalization_method not in ["zscore", "minmax", "robust", "none"]:
            raise ValueError("normalization_method must be 'zscore', 'minmax', 'robust', or 'none'")
        
        if not 0 < self.early_stopping_threshold < 1000:
            raise ValueError("early_stopping_threshold must be between 0 and 1000")
        
        if not 0 < self.similarity_threshold_for_constraint < 1:
            raise ValueError("similarity_threshold_for_constraint must be between 0 and 1")


# ============================================================================
# Model Classes
# ============================================================================

class MLPModel(nn.Module):
    """MLP model architecture matching saved configurations."""
    def __init__(
        self,
        input_dim: int = 3,
        num_hidden_layers: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 1,
        activation_fn_name: str = 'relu',
        output_activation_fn_name: str = 'sigmoid',
        dropout_rate: float = 0.0012
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(self.output_activation_fn)
        
        self.layers = nn.Sequential(*layers_list)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'softmax': nn.Softmax(dim=-1),
            'none': nn.Identity()
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name.lower()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ActualCustomModel(nn.Module):
    """Custom model with Conv1D layers."""
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(3, 32),                                    # layers.0
            nn.ReLU(),                                           # layers.1
            nn.Linear(32, 32),                                   # layers.2
            nn.ReLU(),                                           # layers.3
            nn.Dropout(0.0),                                     # layers.4
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.5
            nn.ReLU(),                                           # layers.6
            nn.Dropout(0.0),                                     # layers.7
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.8
            nn.ReLU(),                                           # layers.9
            nn.Dropout(0.0),                                     # layers.10
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.11
            nn.ReLU(),                                           # layers.12
            nn.Dropout(0.0),                                     # layers.13
            nn.Linear(32, 1),                                    # layers.14
            nn.Sigmoid()                                         # layers.15
        )
    
    def forward(self, x):
        x = self.layers[1](self.layers[0](x))
        x = self.layers[4](self.layers[3](self.layers[2](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[7](self.layers[6](self.layers[5](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[10](self.layers[9](self.layers[8](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[13](self.layers[12](self.layers[11](x)))
        x = x.view(x.size(0), -1)
        x = self.layers[15](self.layers[14](x))
        return x


# ============================================================================
# Utility Functions
# ============================================================================

def _rebuild_model(class_name: str) -> nn.Module:
    """Rebuild model from class name for parallel workers."""
    if class_name == 'MLPModel':
        return MLPModel()
    if class_name == 'ActualCustomModel':
        return ActualCustomModel()
    raise ValueError(f"Unknown model class: {class_name}")


# ============================================================================
# Eigenvalue Processing Functions
# ============================================================================

def track_eigenvalues_hungarian(eigenvalue_sequences: List[np.ndarray]) -> np.ndarray:
    """Track eigenvalues through filtration using Hungarian algorithm.
    
    Args:
        eigenvalue_sequences: List of eigenvalue arrays at each filtration step
        
    Returns:
        tracked_curves: Array of shape (n_eigenvalues, n_steps)
    """
    n_steps = len(eigenvalue_sequences)
    
    # Handle varying number of eigenvalues across steps
    max_eigenvalues = max(len(seq) for seq in eigenvalue_sequences)
    
    # Pad sequences to have same length
    padded_sequences = []
    for seq in eigenvalue_sequences:
        if len(seq) < max_eigenvalues:
            padded = np.zeros(max_eigenvalues)
            padded[:len(seq)] = seq
            padded_sequences.append(padded)
        else:
            padded_sequences.append(seq[:max_eigenvalues])
    
    n_eigenvalues = max_eigenvalues
    
    # Initialize tracked curves
    tracked_curves = np.zeros((n_eigenvalues, n_steps))
    tracked_curves[:, 0] = padded_sequences[0]
    
    # Track through consecutive steps
    for t in range(1, n_steps):
        prev_values = tracked_curves[:, t-1]
        curr_values = padded_sequences[t]
        
        # Build cost matrix based on value difference
        cost_matrix = np.abs(prev_values[:, np.newaxis] - curr_values[np.newaxis, :])
        
        # Add penalty for large changes to encourage smooth tracking
        if t > 1:
            prev_diff = tracked_curves[:, t-1] - tracked_curves[:, t-2]
            for i in range(n_eigenvalues):
                for j in range(n_eigenvalues):
                    expected = prev_values[i] + prev_diff[i]
                    cost_matrix[i, j] += 0.1 * np.abs(curr_values[j] - expected)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Apply assignment
        for i, j in zip(row_indices, col_indices):
            tracked_curves[i, t] = curr_values[j]
    
    return tracked_curves


def interpolate_curves_to_grid(
    curves: np.ndarray,
    original_grid: np.ndarray,
    n_points: int = 100,
    additional_points: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate curves to a common grid using linear interpolation.
    
    Args:
        curves: Array of shape (n_curves, n_original_points)
        original_grid: Original filtration parameter values
        n_points: Number of points in uniform grid (used if additional_points is None)
        additional_points: Optional array of additional points to include in the grid
        
    Returns:
        interpolated: Array of shape (n_curves, n_grid_points)
        grid_points: Array of actual grid points used (normalized to [0, 1])
    """
    n_curves = curves.shape[0]
    
    # Convert to numpy array if needed
    if not isinstance(original_grid, np.ndarray):
        original_grid = np.array(original_grid)
    
    # Normalize original grid to [0, 1]
    grid_min, grid_max = original_grid.min(), original_grid.max()
    if grid_max - grid_min < 1e-10:
        # Handle constant grid case
        norm_original = np.linspace(0, 1, len(original_grid))
    else:
        norm_original = (original_grid - grid_min) / (grid_max - grid_min)
    
    # Create the interpolation grid
    if additional_points is not None:
        # Normalize additional points to same [0, 1] range
        if grid_max - grid_min > 1e-10:
            norm_additional = (additional_points - grid_min) / (grid_max - grid_min)
        else:
            norm_additional = np.linspace(0, 1, len(additional_points))
        
        # Combine uniform grid with additional points
        uniform_grid = np.linspace(0, 1, n_points)
        all_points = np.unique(np.concatenate([uniform_grid, norm_additional]))
        # Sort and remove duplicates
        grid_points = np.sort(all_points)
    else:
        # Use uniform grid only
        grid_points = np.linspace(0, 1, n_points)
    
    # Interpolate each curve using linear interpolation
    interpolated = np.zeros((n_curves, len(grid_points)))
    for i in range(n_curves):
        f = interp1d(norm_original, curves[i], kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        interpolated[i] = f(grid_points)
    
    return interpolated, grid_points


def interpolate_curves_with_union_grid(
    curves1: np.ndarray,
    grid1: np.ndarray,
    curves2: np.ndarray,
    grid2: np.ndarray,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate two sets of curves to a common grid that includes union of original grids.
    
    Args:
        curves1: First set of curves, shape (n_curves1, n_points1)
        grid1: Filtration parameters for first set
        curves2: Second set of curves, shape (n_curves2, n_points2)
        grid2: Filtration parameters for second set
        n_points: Base number of points for uniform grid
        
    Returns:
        interpolated_curves1: First set interpolated to common grid
        interpolated_curves2: Second set interpolated to common grid
        common_grid: The common grid used (normalized to [0, 1])
    """
    # Get the range of both grids
    all_points = np.concatenate([grid1, grid2])
    global_min = all_points.min()
    global_max = all_points.max()
    
    if global_max - global_min < 1e-10:
        # Handle constant grid case
        common_grid = np.linspace(0, 1, n_points)
        interpolated1, _ = interpolate_curves_to_grid(curves1, grid1, n_points)
        interpolated2, _ = interpolate_curves_to_grid(curves2, grid2, n_points)
        return interpolated1, interpolated2, common_grid
    
    # Normalize both grids to [0, 1] using the global range
    norm_grid1 = (grid1 - global_min) / (global_max - global_min)
    norm_grid2 = (grid2 - global_min) / (global_max - global_min)
    
    # Create union of normalized grids
    uniform_grid = np.linspace(0, 1, n_points)
    union_grid = np.unique(np.concatenate([uniform_grid, norm_grid1, norm_grid2]))
    common_grid = np.sort(union_grid)
    
    # Interpolate both curve sets to the common grid
    interpolated1 = np.zeros((curves1.shape[0], len(common_grid)))
    for i in range(curves1.shape[0]):
        f = interp1d(norm_grid1, curves1[i], kind='linear',
                    bounds_error=False, fill_value='extrapolate')
        interpolated1[i] = f(common_grid)
    
    interpolated2 = np.zeros((curves2.shape[0], len(common_grid)))
    for i in range(curves2.shape[0]):
        f = interp1d(norm_grid2, curves2[i], kind='linear',
                    bounds_error=False, fill_value='extrapolate')
        interpolated2[i] = f(common_grid)
    
    return interpolated1, interpolated2, common_grid


def extract_and_process_eigenvalues(
    model: nn.Module,
    data: torch.Tensor,
    config: MultivariateDTWConfig,
    model_name: str = "model"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and process eigenvalue evolution from a model.
    
    Args:
        model: Neural network model
        data: Input data tensor
        config: Configuration parameters
        model_name: Name for logging purposes
        
    Returns:
        processed_curves: Array of shape (n_eigenvalues, n_time_points)
        filtration_params: Array of filtration parameter values
    """
    print(f"📊 Extracting eigenvalue evolution for {model_name}...")
    
    # Configure Gromov-Wasserstein
    gw_config = GWConfig(
        epsilon=0.05,
        max_iter=100,
        tolerance=1e-8,
        quasi_sheaf_tolerance=0.08,
        adaptive_epsilon=True,
        base_epsilon=0.05,
        reference_n=50,
        epsilon_scaling_method='sqrt',
        epsilon_min=0.01,
        epsilon_max=0.2
    )
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu')
    
    # Analyze model
    result = analyzer.analyze(
        model, data,
        method='gromov_wasserstein',
        gw_config=gw_config,
        exclude_final_single_output=True
    )
    
    # Extract spectral analysis
    spectral_analyzer = PersistentSpectralAnalyzer(
        default_n_steps=50,
        default_filtration_type='threshold'
    )
    spectral = spectral_analyzer.analyze(
        result['sheaf'], 
        filtration_type='threshold',
        n_steps=100
    )
    
    eigenvalue_sequences = spectral['persistence_result']['eigenvalue_sequences']
    filtration_params = spectral['filtration_params']
    
    # Filter to top-N eigenvalues
    if config.top_n_eigenvalues is not None:
        filtered_sequences = []
        for step_eigenvals in eigenvalue_sequences:
            if len(step_eigenvals) > config.top_n_eigenvalues:
                sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
                filtered_sequences.append(sorted_eigenvals[:config.top_n_eigenvalues].numpy())
            else:
                padded = np.zeros(config.top_n_eigenvalues)
                padded[:len(step_eigenvals)] = step_eigenvals.numpy()
                filtered_sequences.append(padded)
        eigenvalue_sequences = filtered_sequences
    
    # Track eigenvalues through Hungarian algorithm
    print(f"  🔗 Tracking eigenvalues through filtration...")
    tracked = track_eigenvalues_hungarian(eigenvalue_sequences)
    
    # Interpolate to common grid
    print(f"  📈 Interpolating to {config.interpolation_points} point grid...")
    interpolated, interpolated_grid = interpolate_curves_to_grid(tracked, filtration_params, config.interpolation_points)
    
    print(f"  ✓ Processed eigenvalue evolution: {interpolated.shape}")
    
    # Since curves are interpolated to uniform grid, we don't need to pass original grid
    # All models will have the same interpolation_points, so they're already aligned
    return interpolated, None


# ============================================================================
# Log Transformation and Preprocessing
# ============================================================================

def compute_adaptive_threshold(data: np.ndarray, config: MultivariateDTWConfig) -> float:
    """Compute adaptive threshold for log transformation based on data distribution.
    
    Args:
        data: Input eigenvalue data array
        config: Configuration parameters
        
    Returns:
        adaptive_threshold: Computed adaptive threshold
    """
    positive_data = data[data > 0]
    if len(positive_data) == 0:
        return config.log_floor_epsilon
    
    # Use percentile-based adaptive thresholding
    percentile_val = np.percentile(positive_data, config.adaptive_threshold_percentile)
    adaptive_threshold = max(
        config.log_floor_epsilon,
        percentile_val * config.adaptive_threshold_multiplier
    )
    
    return adaptive_threshold


def apply_log_transformation(curves: np.ndarray, config: MultivariateDTWConfig) -> np.ndarray:
    """Apply log10 transformation to eigenvalue curves with adaptive thresholding.
    
    Args:
        curves: Array of shape (n_curves, n_time_points)
        config: Configuration parameters
        
    Returns:
        log_curves: Log10-transformed curves
    """
    if not config.use_log_transform:
        return curves.copy()
    
    if config.verbose_mode:
        print("  🔢 Applying log10 transformation with adaptive thresholding...")
    
    # Compute adaptive threshold based on all data
    adaptive_threshold = compute_adaptive_threshold(curves, config)
    if config.verbose_mode:
        print(f"    Adaptive threshold: {adaptive_threshold:.2e}")
    
    # Apply log10 transformation with thresholding
    curves_thresholded = np.maximum(curves, adaptive_threshold)
    log_curves = np.log10(curves_thresholded)
    
    # Handle any remaining non-finite values
    if config.handle_nan_inf:
        log_curves = np.nan_to_num(
            log_curves, 
            nan=np.log10(adaptive_threshold),
            posinf=10.0,
            neginf=np.log10(adaptive_threshold)
        )
    
    if config.verbose_mode:
        print(f"    Log transform range: [{np.min(log_curves):.2f}, {np.max(log_curves):.2f}]")
    
    return log_curves


def normalize_sequences(curves: np.ndarray, config: MultivariateDTWConfig) -> np.ndarray:
    """Normalize sequences according to configuration.
    
    Args:
        curves: Array of shape (n_curves, n_time_points)
        config: Configuration parameters
        
    Returns:
        normalized_curves: Normalized curves
    """
    if not config.use_sequence_normalization or config.normalization_method == "none":
        return curves.copy()
    
    if config.verbose_mode:
        print(f"  ⚖️ Applying {config.normalization_method} normalization...")
    
    if config.normalization_method == "zscore":
        # Z-score normalization: (x - mean) / std
        mean = np.mean(curves)
        std = np.std(curves)
        if std > 1e-12:
            normalized = (curves - mean) / std
        else:
            normalized = curves - mean
    
    elif config.normalization_method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        data_min = np.min(curves)
        data_max = np.max(curves)
        if data_max - data_min > 1e-12:
            normalized = (curves - data_min) / (data_max - data_min)
        else:
            normalized = curves - data_min
    
    elif config.normalization_method == "robust":
        # Robust normalization using median and MAD (more resilient to outliers)
        median = np.median(curves)
        mad = np.median(np.abs(curves - median))
        if mad > 1e-12:
            normalized = (curves - median) / (mad * 1.4826)  # 1.4826 makes MAD comparable to std
        else:
            normalized = curves - median
    
    else:
        normalized = curves.copy()
    
    if config.verbose_mode:
        print(f"    Normalized range: [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
    
    return normalized


def preprocess_eigenvalue_curves(
    curves1: np.ndarray, 
    curves2: np.ndarray, 
    config: MultivariateDTWConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess eigenvalue curves for DTW comparison.
    
    Args:
        curves1: First set of curves, shape (n_curves, n_time_points)
        curves2: Second set of curves, shape (n_curves, n_time_points)  
        config: Configuration parameters
        
    Returns:
        processed_curves1: Preprocessed first curves
        processed_curves2: Preprocessed second curves
    """
    if config.verbose_mode:
        print("🔧 Preprocessing eigenvalue curves...")
    
    # Combine curves for consistent preprocessing
    all_curves = np.vstack([curves1, curves2])
    
    # Apply log transformation
    log_curves = apply_log_transformation(all_curves, config)
    
    # Apply normalization
    normalized_curves = normalize_sequences(log_curves, config)
    
    # Split back into original sets
    n1 = curves1.shape[0]
    processed_curves1 = normalized_curves[:n1]
    processed_curves2 = normalized_curves[n1:]
    
    if config.verbose_mode:
        print(f"  ✓ Preprocessing complete: {processed_curves1.shape}, {processed_curves2.shape}")
    
    return processed_curves1, processed_curves2


def prepare_multivariate_sequences(
    curves1: np.ndarray,
    curves2: np.ndarray,
    config: MultivariateDTWConfig,
    grid1: Optional[np.ndarray] = None,
    grid2: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare multivariate sequences for DTW comparison.
    
    Transform from (n_curves, n_time_points) to (n_time_points, n_features)
    format expected by tslearn.
    
    Args:
        curves1: First set of curves, shape (n_curves, n_time_points)
        curves2: Second set of curves, shape (n_curves, n_time_points)
        config: Configuration parameters
        grid1: Optional filtration parameters for first set (not used for individual curves)
        grid2: Optional filtration parameters for second set (not used for individual curves)
        
    Returns:
        seq1: First multivariate sequence, shape (n_time_points, n_features)
        seq2: Second multivariate sequence, shape (n_time_points, n_features)
    """
    if config.verbose_mode:
        print("🔄 Preparing multivariate sequences for DTW...")
    
    # Note: Grid-based interpolation should be done at the set level, not for individual curves
    # If you reach here with grids, they've already been processed at a higher level
    
    # Preprocess curves
    proc_curves1, proc_curves2 = preprocess_eigenvalue_curves(curves1, curves2, config)
    
    # Transpose to get (n_time_points, n_features) format
    seq1 = proc_curves1.T  # Shape: (n_time_points, n_eigenvalues)
    seq2 = proc_curves2.T  # Shape: (n_time_points, n_eigenvalues)
    
    if config.verbose_mode:
        print(f"  ✓ Multivariate sequences prepared: {seq1.shape}, {seq2.shape}")
    
    return seq1, seq2


# ============================================================================
# Multivariate DTW Implementation
# ============================================================================

def compute_multivariate_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    config: MultivariateDTWConfig
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute multivariate DTW distance and alignment path using tslearn.
    
    Args:
        seq1: First multivariate sequence, shape (n_time_points, n_features)
        seq2: Second multivariate sequence, shape (n_time_points, n_features)
        config: Configuration parameters
        
    Returns:
        distance: DTW distance
        alignment_path: List of (i, j) alignment tuples
    """
    if not TSLEARN_AVAILABLE:
        raise RuntimeError("tslearn is required but not available")
    
    try:
        # Validate input shapes
        if seq1.ndim != 2 or seq2.ndim != 2:
            raise ValueError(f"Sequences must be 2D, got shapes {seq1.shape}, {seq2.shape}")
        
        if seq1.shape[1] != seq2.shape[1]:
            raise ValueError(f"Sequences must have same number of features: {seq1.shape[1]} vs {seq2.shape[1]}")
        
        # Ensure float dtype for tslearn compatibility
        seq1 = seq1.astype(np.float64)
        seq2 = seq2.astype(np.float64)
        
        # Configure DTW parameters with adaptive constraints
        dtw_kwargs = {}
        
        if config.dtw_global_constraint == "sakoe_chiba" and config.dtw_constraint_ratio > 0:
            # Adaptive constraint radius based on sequence similarity (if enabled)
            base_radius = max(1, int(max(len(seq1), len(seq2)) * config.dtw_constraint_ratio))
            
            if config.use_adaptive_constraints:
                # Quick similarity check using simple correlation
                try:
                    # Simple correlation between sequence means for quick similarity estimate
                    mean1 = np.mean(seq1, axis=1)
                    mean2 = np.mean(seq2, axis=1)
                    if len(mean1) == len(mean2):
                        correlation = np.corrcoef(mean1, mean2)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                        
                        # Adapt radius based on similarity
                        if abs(correlation) > config.similarity_threshold_for_constraint:
                            # High similarity - use tighter constraint
                            adapted_radius = max(1, int(base_radius * 0.7))
                        else:
                            # Low similarity - use more flexible constraint
                            adapted_radius = int(base_radius * 1.3)
                    else:
                        adapted_radius = base_radius
                except:
                    adapted_radius = base_radius
            else:
                adapted_radius = base_radius
            
            dtw_kwargs['global_constraint'] = 'sakoe_chiba'
            dtw_kwargs['sakoe_chiba_radius'] = adapted_radius
        
        # Compute DTW distance and alignment path
        # tslearn.dtw_path returns (path, distance) - path first, then distance
        alignment_path, distance = dtw_path(seq1, seq2, **dtw_kwargs)
        
        # Validate outputs
        if not isinstance(distance, (int, float)) or not np.isfinite(distance):
            raise ValueError(f"Invalid DTW distance returned: {distance}")
        
        if not hasattr(alignment_path, '__iter__'):
            raise ValueError(f"Invalid alignment path returned: {type(alignment_path)}")
        
        # Early stopping check for very large distances
        if config.enable_early_stopping and distance > config.early_stopping_threshold:
            # Return simplified result for clearly dissimilar sequences
            alignment = [(0, 0), (len(seq1)-1, len(seq2)-1)]  # Simple diagonal
            return float(config.early_stopping_threshold), alignment
        
        # Convert alignment path to list of tuples
        alignment = [(int(i), int(j)) for i, j in alignment_path]
        
        return float(distance), alignment
    
    except Exception as e:
        print(f"    Warning: tslearn DTW failed ({e}), using fallback Euclidean distance")
        
        # Fallback to simple Euclidean distance
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        # Pad shorter sequence
        if len(seq1) < max_len:
            padded_seq1 = np.pad(seq1, ((0, max_len - len(seq1)), (0, 0)), 
                               mode='edge')
            padded_seq2 = seq2
        elif len(seq2) < max_len:
            padded_seq1 = seq1
            padded_seq2 = np.pad(seq2, ((0, max_len - len(seq2)), (0, 0)), 
                               mode='edge')
        else:
            padded_seq1, padded_seq2 = seq1, seq2
        
        # Compute Euclidean distance
        distance = np.linalg.norm(padded_seq1 - padded_seq2)
        
        # Create diagonal alignment
        alignment = [(i, i) for i in range(min_len)]
        
        return float(distance), alignment


def compute_normalized_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    distance: float,
    alignment: List[Tuple[int, int]],
    config: MultivariateDTWConfig
) -> float:
    """Normalize DTW distance by alignment path length.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        distance: Raw DTW distance
        alignment: DTW alignment path
        config: Configuration parameters
        
    Returns:
        normalized_distance: Normalized DTW distance
    """
    if len(alignment) == 0:
        # Fallback to simple normalization
        return distance / max(len(seq1), len(seq2))
    
    # Normalize by alignment path length (more robust than sequence length)
    normalized_distance = distance / len(alignment)
    
    return max(0.0, normalized_distance)  # Ensure non-negative


def compute_pairwise_dtw_distances(
    curves1: np.ndarray,
    curves2: np.ndarray,
    config: MultivariateDTWConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute pairwise DTW distances between two sets of eigenvalue curves.
    
    Note: Curves should already be interpolated to common grid if needed.
    
    Args:
        curves1: First set of curves, shape (n_curves1, n_time_points)
        curves2: Second set of curves, shape (n_curves2, n_time_points)
        config: Configuration parameters
        
    Returns:
        distance_matrix: Pairwise distance matrix, shape (n_curves1, n_curves2)
        metadata: Dictionary with computation metadata
    """
    if config.verbose_mode:
        print("🔄 Computing pairwise multivariate DTW distances...")
    
    n_curves1, n_curves2 = curves1.shape[0], curves2.shape[0]
    distance_matrix = np.zeros((n_curves1, n_curves2))
    alignment_info = {}
    
    total_pairs = n_curves1 * n_curves2
    processed_pairs = 0
    
    for i in range(n_curves1):
        if config.verbose_mode and (i % config.progress_update_freq == 0 or i == n_curves1 - 1):
            print(f"  Processing row {i+1}/{n_curves1}...")
        
        for j in range(n_curves2):
            # Prepare multivariate sequences for this pair
            curve1 = curves1[i:i+1]  # Keep as 2D array
            curve2 = curves2[j:j+1]  # Keep as 2D array
            
            # Don't pass grids for individual curve comparisons
            seq1, seq2 = prepare_multivariate_sequences(curve1, curve2, config)
            
            # Compute DTW distance
            distance, alignment = compute_multivariate_dtw_distance(seq1, seq2, config)
            
            # Normalize distance
            normalized_distance = compute_normalized_dtw_distance(
                seq1, seq2, distance, alignment, config
            )
            
            distance_matrix[i, j] = normalized_distance
            alignment_info[(i, j)] = {
                'raw_distance': distance,
                'normalized_distance': normalized_distance,
                'alignment_length': len(alignment),
                'sequence_lengths': (len(seq1), len(seq2))
            }
            
            processed_pairs += 1
    
    print(f"  ✓ Computed {processed_pairs} pairwise DTW distances")
    
    metadata = {
        'total_pairs': processed_pairs,
        'distance_stats': {
            'mean': np.mean(distance_matrix),
            'std': np.std(distance_matrix),
            'min': np.min(distance_matrix),
            'max': np.max(distance_matrix)
        },
        'alignment_info': alignment_info
    }
    
    return distance_matrix, metadata


# ============================================================================
# Parallel Worker Functions
# ============================================================================

def _analyze_model_worker(args: Tuple[str, str, str, int, Tuple[int, ...], Dict[str, Any]]) -> Dict[str, Any]:
    """Worker function to analyze a single model in parallel.
    
    Args:
        args: Tuple containing (model_key, model_class_name, weights_path, 
              data_seed, data_shape, config_dict)
              
    Returns:
        Dictionary with model key, processed curves, and metadata
    """
    model_key, model_class_name, weights_path, data_seed, data_shape, config_dict = args
    
    # Recreate config from dict (needed for multiprocessing)
    config = MultivariateDTWConfig(**config_dict)
    
    # Set seeds for reproducibility
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    
    print(f"  Worker: Analyzing {model_key}...")
    
    try:
        # Rebuild model and load weights
        model = _rebuild_model(model_class_name)
        model = load_model(lambda: model, weights_path, device='cpu')
        
        # Generate data
        batch_size, input_dim = data_shape
        data = 8 * torch.randn(batch_size, input_dim)
        
        # Extract and process eigenvalues
        processed_curves, filtration_params = extract_and_process_eigenvalues(
            model, data, config, model_key
        )
        
        print(f"  Worker: Completed {model_key}")
        
        # Return serializable data
        return {
            'key': model_key,
            'curves': processed_curves.tolist(),  # Convert to list for serialization
            'shape': processed_curves.shape,
            'filtration_params': filtration_params.tolist() if filtration_params is not None else None,
            'success': True
        }
        
    except Exception as e:
        print(f"  Worker: Failed to analyze {model_key}: {e}")
        return {
            'key': model_key,
            'curves': None,
            'shape': None,
            'filtration_params': None,
            'success': False,
            'error': str(e)
        }


def _compute_dtw_distance_worker(args: Tuple[int, int, np.ndarray, np.ndarray, Dict[str, Any]]) -> Tuple[int, int, float]:
    """Worker function to compute DTW distance between two curve sets.
    
    Args:
        args: Tuple containing (i, j, curves1, curves2, config_dict)
        
    Returns:
        Tuple of (i, j, distance)
    """
    i, j, curves1, curves2, config_dict = args
    
    # Recreate config
    config = MultivariateDTWConfig(**config_dict)
    
    try:
        # Prepare multivariate sequences (no grids for individual curves)
        seq1, seq2 = prepare_multivariate_sequences(
            curves1.reshape(1, -1), curves2.reshape(1, -1), config
        )
        
        # Compute DTW distance
        distance, alignment = compute_multivariate_dtw_distance(seq1, seq2, config)
        
        # Normalize distance
        normalized_distance = compute_normalized_dtw_distance(
            seq1, seq2, distance, alignment, config
        )
        
        return (i, j, normalized_distance)
        
    except Exception as e:
        print(f"    Warning: DTW computation failed for pair ({i}, {j}): {e}")
        # Return a large distance as fallback
        return (i, j, 1000.0)


def analyze_models_parallel(
    model_paths: Dict[str, Tuple[str, str]],
    config: MultivariateDTWConfig
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Analyze multiple models in parallel to extract eigenvalue evolution.
    
    Args:
        model_paths: Dictionary mapping model keys to (class_name, path) tuples
        config: Configuration parameters
        
    Returns:
        Dictionary mapping model keys to tuple of (curves, filtration_params)
    """
    print("📊 Phase 1: Analyzing models in parallel...")
    
    # Prepare arguments for parallel workers
    model_args = []
    for key, (class_name, path) in model_paths.items():
        model_args.append((
            key, class_name, path,
            config.random_seed,
            (config.data_batch_size, 3),  # data shape
            config.__dict__
        ))
    
    start_time = time.time()
    analyzed_models = {}
    failed_models = []
    
    # Run parallel model analysis
    with futures.ProcessPoolExecutor(max_workers=config.num_workers_models) as pool:
        results = list(pool.map(_analyze_model_worker, model_args))
    
    # Process results
    for result in results:
        if result['success']:
            # Convert back to numpy arrays
            curves = np.array(result['curves'])
            filtration_params = np.array(result['filtration_params']) if result['filtration_params'] is not None else None
            analyzed_models[result['key']] = (curves, filtration_params)
            print(f"  ✓ {result['key']}: {curves.shape}")
        else:
            failed_models.append((result['key'], result['error']))
            print(f"  ✗ {result['key']}: {result['error']}")
    
    phase1_time = time.time() - start_time
    
    if failed_models:
        print(f"  Warning: {len(failed_models)} models failed to analyze")
        for key, error in failed_models:
            print(f"    - {key}: {error}")
    
    print(f"✅ Phase 1 completed in {phase1_time:.1f}s")
    print(f"   Successfully analyzed: {len(analyzed_models)}/{len(model_paths)} models")
    
    return analyzed_models


def compute_dtw_distances_parallel(
    curves1: np.ndarray,
    curves2: np.ndarray,
    config: MultivariateDTWConfig
) -> np.ndarray:
    """Compute pairwise DTW distances in parallel.
    
    Note: Curves should already be interpolated to common grid if needed.
    
    Args:
        curves1: First set of curves, shape (n_curves1, n_time_points)
        curves2: Second set of curves, shape (n_curves2, n_time_points)
        config: Configuration parameters
        
    Returns:
        distance_matrix: Pairwise distance matrix, shape (n_curves1, n_curves2)
    """
    print("🔄 Computing DTW distances in parallel...")
    
    n_curves1, n_curves2 = curves1.shape[0], curves2.shape[0]
    
    # Prepare arguments for parallel workers
    worker_args = []
    for i in range(n_curves1):
        for j in range(n_curves2):
            # Don't pass grids - curves should already be on common grid if needed
            worker_args.append((i, j, curves1[i], curves2[j], config.__dict__))
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n_curves1, n_curves2))
    
    # Process in batches to manage memory (use optimized batch size)
    batch_size = config.batch_size_dtw
    n_batches = (len(worker_args) + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(worker_args))
        batch_args = worker_args[start_idx:end_idx]
        
        if batch_idx % config.progress_update_freq == 0:
            print(f"  Processing batch {batch_idx+1}/{n_batches}...")
        
        # Compute distances for this batch
        with futures.ProcessPoolExecutor(max_workers=config.num_workers_dtw) as pool:
            batch_results = list(pool.map(_compute_dtw_distance_worker, batch_args))
        
        # Fill distance matrix
        for i, j, distance in batch_results:
            distance_matrix[i, j] = distance
    
    computation_time = time.time() - start_time
    print(f"  ✓ Computed {len(worker_args)} pairwise distances in {computation_time:.1f}s")
    print(f"  Distance statistics: mean={np.mean(distance_matrix):.4f}, "
          f"std={np.std(distance_matrix):.4f}, range=[{np.min(distance_matrix):.4f}, {np.max(distance_matrix):.4f}]")
    
    return distance_matrix


# ============================================================================
# Hungarian Matching and Set Distance
# ============================================================================

def compute_set_distance_hungarian(
    curves1: np.ndarray,
    curves2: np.ndarray,
    config: MultivariateDTWConfig,
    grid1: Optional[np.ndarray] = None,
    grid2: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute set-to-set distance using Hungarian matching algorithm.
    
    Args:
        curves1: First set of curves, shape (n_curves1, n_time_points)
        curves2: Second set of curves, shape (n_curves2, n_time_points)
        config: Configuration parameters
        grid1: Optional filtration parameters for first set (for reference only)
        grid2: Optional filtration parameters for second set (for reference only)
        
    Returns:
        Dictionary with distance metrics and matching information
    """
    print("🎯 Computing set-to-set distance using Hungarian matching...")
    
    n_curves1, n_curves2 = curves1.shape[0], curves2.shape[0]
    n_curves = min(n_curves1, n_curves2)
    
    # Curves are already interpolated to uniform grids during extraction
    # All models use the same config.interpolation_points, so they should be aligned
    print(f"  📊 Curves on uniform grids: {curves1.shape[1]} and {curves2.shape[1]} points")
    
    if curves1.shape[1] != curves2.shape[1]:
        print(f"  ⚠️ Warning: Different grid sizes, DTW will handle alignment")
    
    # Compute pairwise distance matrix
    if config.num_workers_dtw > 1:
        # Use parallel computation for large matrices
        distance_matrix = compute_dtw_distances_parallel(curves1, curves2, config)
    else:
        # Use sequential computation
        distance_matrix, _ = compute_pairwise_dtw_distances(curves1, curves2, config)
    
    print("🔍 Solving Hungarian assignment problem...")
    
    # If matrices have different sizes, truncate to square matrix
    if n_curves1 != n_curves2:
        print(f"  Note: Truncating {distance_matrix.shape} matrix to {n_curves}x{n_curves} for Hungarian algorithm")
        square_matrix = distance_matrix[:n_curves, :n_curves]
    else:
        square_matrix = distance_matrix
    
    # Solve Hungarian assignment
    row_indices, col_indices = linear_sum_assignment(square_matrix)
    
    # Extract matched costs
    matched_costs = square_matrix[row_indices, col_indices]
    
    # Compute statistics
    if config.use_trimmed_mean and len(matched_costs) > 4:
        # Trimmed mean (remove top/bottom percentiles)
        trim_n = max(1, int(len(matched_costs) * config.trim_percentage))
        sorted_costs = np.sort(matched_costs)
        trimmed_costs = sorted_costs[trim_n:-trim_n] if trim_n < len(sorted_costs)//2 else sorted_costs
        mean_distance = np.mean(trimmed_costs)
        print(f"  Using trimmed mean: removed {trim_n} values from each end")
    else:
        mean_distance = np.mean(matched_costs)
    
    # Find best and worst matches
    best_idx = np.argmin(matched_costs)
    worst_idx = np.argmax(matched_costs)
    
    best_match = (row_indices[best_idx], col_indices[best_idx], matched_costs[best_idx])
    worst_match = (row_indices[worst_idx], col_indices[worst_idx], matched_costs[worst_idx])
    
    print(f"  ✓ Hungarian matching complete: {len(matched_costs)} pairs matched")
    print(f"    Mean distance: {mean_distance:.4f}")
    print(f"    Best match: curves {best_match[0]} ↔ {best_match[1]} (distance: {best_match[2]:.4f})")
    print(f"    Worst match: curves {worst_match[0]} ↔ {worst_match[1]} (distance: {worst_match[2]:.4f})")
    
    return {
        'mean_distance': float(mean_distance),
        'median_distance': float(np.median(matched_costs)),
        'std_distance': float(np.std(matched_costs)),
        'min_distance': float(np.min(matched_costs)),
        'max_distance': float(np.max(matched_costs)),
        'matched_costs': matched_costs.tolist(),
        'matching': list(zip(row_indices.tolist(), col_indices.tolist())),
        'best_match': (int(best_match[0]), int(best_match[1]), float(best_match[2])),
        'worst_match': (int(worst_match[0]), int(worst_match[1]), float(worst_match[2])),
        'cost_matrix': distance_matrix.tolist(),
        'matrix_shape': distance_matrix.shape,
        'n_matched_pairs': len(matched_costs),
        'trimmed_mean_used': config.use_trimmed_mean
    }


def compare_model_pairs(
    analyzed_models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_pairs: List[Tuple[str, str]],
    config: MultivariateDTWConfig
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple pairs of models using DTW and Hungarian matching.
    
    Args:
        analyzed_models: Dictionary mapping model keys to (curves, filtration_params) tuples
        model_pairs: List of (model1_key, model2_key) tuples to compare
        config: Configuration parameters
        
    Returns:
        Dictionary mapping pair names to comparison results
    """
    print("📐 Phase 2: Computing pairwise model comparisons...")
    
    all_results = {}
    start_time = time.time()
    
    for i, (model1_key, model2_key) in enumerate(model_pairs):
        print(f"\n🔄 Comparison {i+1}/{len(model_pairs)}: {model1_key} vs {model2_key}")
        
        if model1_key not in analyzed_models:
            print(f"  ✗ Model {model1_key} not found in analyzed models")
            continue
        
        if model2_key not in analyzed_models:
            print(f"  ✗ Model {model2_key} not found in analyzed models")
            continue
        
        # Get eigenvalue curves and filtration parameters
        curves1, grid1 = analyzed_models[model1_key]
        curves2, grid2 = analyzed_models[model2_key]
        
        print(f"  Comparing curves: {curves1.shape} vs {curves2.shape}")
        
        # Compute set-to-set distance (curves already on uniform grids)
        comparison_start = time.time()
        result = compute_set_distance_hungarian(curves1, curves2, config, grid1, grid2)
        comparison_time = time.time() - comparison_start
        
        # Add metadata
        result.update({
            'model1_name': model1_key,
            'model2_name': model2_key,
            'model1_shape': curves1.shape,
            'model2_shape': curves2.shape,
            'comparison_time': comparison_time,
            'config': {
                'top_n_eigenvalues': config.top_n_eigenvalues,
                'interpolation_points': config.interpolation_points,
                'use_log_transform': config.use_log_transform,
                'normalization_method': config.normalization_method,
                'dtw_constraint_ratio': config.dtw_constraint_ratio
            }
        })
        
        # Store result
        pair_name = f"{model1_key}_vs_{model2_key}"
        all_results[pair_name] = result
        
        print(f"  ✅ Completed in {comparison_time:.1f}s")
        print(f"     Mean distance: {result['mean_distance']:.4f}")
        print(f"     Distance range: [{result['min_distance']:.4f}, {result['max_distance']:.4f}]")
    
    total_time = time.time() - start_time
    print(f"\n✅ Phase 2 completed in {total_time:.1f}s")
    print(f"   Successfully compared: {len(all_results)}/{len(model_pairs)} pairs")
    
    return all_results


def analyze_separation_metrics(comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze separation metrics between different model categories.
    
    Args:
        comparison_results: Dictionary of comparison results from compare_model_pairs
        
    Returns:
        Dictionary with separation analysis results
    """
    print(f"\n{'='*60}")
    print("📊 SEPARATION ANALYSIS")
    print(f"{'='*60}")
    
    # Categorize comparisons
    intra_trained = []      # trained vs trained
    inter_group = []        # trained vs random
    intra_random = []       # random vs random
    
    for pair_name, result in comparison_results.items():
        distance = result['mean_distance']
        model1, model2 = pair_name.split('_vs_')
        
        # Classify pair type
        is_trained1 = 'trained' in model1 and 'rand' not in model1
        is_trained2 = 'trained' in model2 and 'rand' not in model2
        is_random1 = 'rand' in model1
        is_random2 = 'rand' in model2
        
        if is_trained1 and is_trained2:
            intra_trained.append(distance)
        elif (is_trained1 and is_random2) or (is_random1 and is_trained2):
            inter_group.append(distance)
        elif is_random1 and is_random2:
            intra_random.append(distance)
    
    # Display results
    print("\n📈 Distance Statistics by Category:")
    print("-" * 40)
    
    stats = {}
    
    if intra_trained:
        mean_intra_trained = np.mean(intra_trained)
        std_intra_trained = np.std(intra_trained)
        stats['intra_trained'] = {'mean': mean_intra_trained, 'std': std_intra_trained, 'values': intra_trained}
        print(f"🟢 Intra-Trained (trained vs trained):")
        print(f"   Mean: {mean_intra_trained:.4f}")
        print(f"   Std:  {std_intra_trained:.4f}")
        print(f"   Values: {[f'{d:.4f}' for d in intra_trained]}")
    
    if inter_group:
        mean_inter_group = np.mean(inter_group)
        std_inter_group = np.std(inter_group)
        stats['inter_group'] = {'mean': mean_inter_group, 'std': std_inter_group, 'values': inter_group}
        print(f"\n🔴 Inter-Group (trained vs random):")
        print(f"   Mean: {mean_inter_group:.4f}")
        print(f"   Std:  {std_inter_group:.4f}")
        print(f"   Values: {[f'{d:.4f}' for d in inter_group]}")
    
    if intra_random:
        mean_intra_random = np.mean(intra_random)
        std_intra_random = np.std(intra_random)
        stats['intra_random'] = {'mean': mean_intra_random, 'std': std_intra_random, 'values': intra_random}
        print(f"\n🟡 Intra-Random (random vs random):")
        print(f"   Mean: {mean_intra_random:.4f}")
        print(f"   Std:  {std_intra_random:.4f}")
        print(f"   Values: {[f'{d:.4f}' for d in intra_random]}")
    
    # Compute separation ratio
    separation_info = {}
    if intra_trained and inter_group:
        separation_ratio = mean_inter_group / mean_intra_trained
        separation_info = {
            'separation_ratio': separation_ratio,
            'intra_trained_mean': mean_intra_trained,
            'inter_group_mean': mean_inter_group
        }
        
        print(f"\n🏆 SEPARATION RATIO: {separation_ratio:.2f}x")
        
        if separation_ratio > 10:
            quality = "EXCELLENT: Strong separation between trained and random models"
            print(f"   ✅ {quality}")
        elif separation_ratio > 5:
            quality = "GOOD: Clear separation between trained and random models"
            print(f"   ✅ {quality}")
        elif separation_ratio > 2:
            quality = "MODERATE: Some separation between trained and random models"
            print(f"   ⚠️  {quality}")
        else:
            quality = "POOR: Insufficient separation"
            print(f"   ❌ {quality}")
        
        separation_info['quality_assessment'] = quality
    
    return {
        'statistics': stats,
        'separation_info': separation_info,
        'n_comparisons': len(comparison_results)
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def create_comparison_visualizations(
    comparison_results: Dict[str, Dict[str, Any]],
    config: MultivariateDTWConfig
) -> None:
    """Create visualization plots for comparison results.
    
    Args:
        comparison_results: Dictionary of comparison results
        config: Configuration parameters
    """
    if not config.save_visualizations:
        return
    
    print("📊 Creating visualization plots...")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    for pair_name, result in comparison_results.items():
        print(f"  Creating plots for {pair_name}...")
        
        try:
            # Create distance distribution histogram
            create_distance_histogram(result, config, pair_name)
            
            # Create cost matrix heatmap
            create_cost_matrix_heatmap(result, config, pair_name)
            
            # Create matching visualization
            create_matching_visualization(result, config, pair_name)
            
        except Exception as e:
            print(f"    Warning: Failed to create plots for {pair_name}: {e}")
    
    # Create summary comparison plot
    try:
        create_summary_comparison_plot(comparison_results, config)
    except Exception as e:
        print(f"    Warning: Failed to create summary plot: {e}")
    
    print(f"  ✓ Visualizations saved to {config.output_dir}/")


def create_distance_histogram(
    result: Dict[str, Any],
    config: MultivariateDTWConfig,
    pair_name: str
) -> None:
    """Create histogram of matched pair distances."""
    matched_costs = result['matched_costs']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=matched_costs,
        nbinsx=min(30, len(matched_costs) // 2) if len(matched_costs) > 2 else 10,
        name='Matched Pair Distances',
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add vertical lines for statistics
    fig.add_vline(
        x=result['mean_distance'], 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f"Mean: {result['mean_distance']:.4f}"
    )
    fig.add_vline(
        x=result['median_distance'], 
        line_dash="dash", 
        line_color="green", 
        annotation_text=f"Median: {result['median_distance']:.4f}"
    )
    
    model1_name = result['model1_name']
    model2_name = result['model2_name']
    
    fig.update_layout(
        title=f"Distribution of Matched Pair DTW Distances<br>{model1_name} vs {model2_name}",
        xaxis_title="DTW Distance",
        yaxis_title="Count",
        showlegend=True,
        template="plotly_white"
    )
    
    filename = os.path.join(config.output_dir, f"{pair_name}_distance_histogram.html")
    fig.write_html(filename)


def create_cost_matrix_heatmap(
    result: Dict[str, Any],
    config: MultivariateDTWConfig,
    pair_name: str
) -> None:
    """Create heatmap of the cost matrix with Hungarian matching overlay."""
    cost_matrix = np.array(result['cost_matrix'])
    matching = result['matching']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cost_matrix,
        colorscale='Viridis',
        colorbar=dict(title="DTW Distance"),
        hoverongaps=False
    ))
    
    # Add matching overlay (circles for matched pairs)
    for i, j in matching:
        fig.add_shape(
            type="circle",
            x0=j-0.4, y0=i-0.4, 
            x1=j+0.4, y1=i+0.4,
            line=dict(color="red", width=3),
            fillcolor="rgba(255,0,0,0.2)"
        )
    
    model1_name = result['model1_name']
    model2_name = result['model2_name']
    
    fig.update_layout(
        title=f"Cost Matrix with Hungarian Matching<br>{model1_name} vs {model2_name}",
        xaxis_title=f"{model2_name} (curves)",
        yaxis_title=f"{model1_name} (curves)",
        width=600,
        height=600,
        template="plotly_white"
    )
    
    filename = os.path.join(config.output_dir, f"{pair_name}_cost_matrix.html")
    fig.write_html(filename)


def create_matching_visualization(
    result: Dict[str, Any],
    config: MultivariateDTWConfig,
    pair_name: str
) -> None:
    """Create visualization showing best and worst matched pairs."""
    best_match = result['best_match']
    worst_match = result['worst_match']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Best Match: Pair ({best_match[0]}, {best_match[1]}) - Distance: {best_match[2]:.4f}",
            f"Worst Match: Pair ({worst_match[0]}, {worst_match[1]}) - Distance: {worst_match[2]:.4f}"
        ],
        vertical_spacing=0.1
    )
    
    # Note: We don't have the actual curve data here, so we'll show summary statistics
    # In a full implementation, you might want to store curve data in the results
    
    # Create placeholder data for demonstration
    x_points = list(range(config.interpolation_points))
    
    # Best match subplot
    fig.add_trace(
        go.Scatter(
            x=x_points, 
            y=[best_match[2]] * len(x_points),
            mode='lines',
            name=f"Best Match Distance",
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )
    
    # Worst match subplot  
    fig.add_trace(
        go.Scatter(
            x=x_points, 
            y=[worst_match[2]] * len(x_points),
            mode='lines',
            name=f"Worst Match Distance",
            line=dict(color='red', dash='dash')
        ),
        row=2, col=1
    )
    
    model1_name = result['model1_name']
    model2_name = result['model2_name']
    
    fig.update_layout(
        title=f"Best and Worst DTW Matches<br>{model1_name} vs {model2_name}",
        height=600,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Time Steps")
    fig.update_yaxes(title_text="DTW Distance")
    
    filename = os.path.join(config.output_dir, f"{pair_name}_matched_pairs.html")
    fig.write_html(filename)


def create_summary_comparison_plot(
    comparison_results: Dict[str, Dict[str, Any]],
    config: MultivariateDTWConfig
) -> None:
    """Create summary comparison plot showing all pairwise distances."""
    
    # Extract data for plotting
    pair_names = []
    mean_distances = []
    std_distances = []
    categories = []
    
    for pair_name, result in comparison_results.items():
        pair_names.append(pair_name.replace('_vs_', ' vs '))
        mean_distances.append(result['mean_distance'])
        std_distances.append(result['std_distance'])
        
        # Categorize pair
        model1, model2 = pair_name.split('_vs_')
        is_trained1 = 'trained' in model1 and 'rand' not in model1
        is_trained2 = 'trained' in model2 and 'rand' not in model2
        is_random1 = 'rand' in model1
        is_random2 = 'rand' in model2
        
        if is_trained1 and is_trained2:
            categories.append('Trained vs Trained')
            color = 'green'
        elif (is_trained1 and is_random2) or (is_random1 and is_trained2):
            categories.append('Trained vs Random')
            color = 'red'
        elif is_random1 and is_random2:
            categories.append('Random vs Random')
            color = 'orange'
        else:
            categories.append('Other')
            color = 'gray'
    
    # Create bar plot
    fig = go.Figure()
    
    # Group by category for better visualization
    category_colors = {
        'Trained vs Trained': 'green',
        'Trained vs Random': 'red', 
        'Random vs Random': 'orange',
        'Other': 'gray'
    }
    
    for category in set(categories):
        indices = [i for i, c in enumerate(categories) if c == category]
        fig.add_trace(go.Bar(
            x=[pair_names[i] for i in indices],
            y=[mean_distances[i] for i in indices],
            error_y=dict(array=[std_distances[i] for i in indices]),
            name=category,
            marker_color=category_colors[category],
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Summary of DTW Distance Comparisons",
        xaxis_title="Model Pairs",
        yaxis_title="Mean DTW Distance",
        xaxis_tickangle=-45,
        height=600,
        template="plotly_white",
        showlegend=True,
        barmode='group'
    )
    
    filename = os.path.join(config.output_dir, "summary_comparison.html")
    fig.write_html(filename)


def print_results_table(comparison_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted table of comparison results."""
    print(f"\n📋 Complete Comparison Results:")
    print("-" * 80)
    print(f"{'Comparison':<35} {'Mean Dist':<12} {'Median':<12} {'Std':<12} {'Range':<20}")
    print("-" * 80)
    
    for pair_name, result in comparison_results.items():
        range_str = f"[{result['min_distance']:.3f}, {result['max_distance']:.3f}]"
        print(f"{pair_name:<35} {result['mean_distance']:<12.4f} "
              f"{result['median_distance']:<12.4f} {result['std_distance']:<12.4f} {range_str:<20}")


# ============================================================================
# Main Execution Pipeline
# ============================================================================

def main():
    """Main execution function for multivariate DTW eigenvalue evolution comparison."""
    
    print("🚀 Multivariate DTW Eigenvalue Evolution Comparison")
    print("=" * 70)
    print("Using tslearn for multivariate DTW with log transformation and parallelization")
    print("=" * 70)
    
    # Initialize configuration
    config = MultivariateDTWConfig()
    
    print(f"\n⚙️  Configuration Summary:")
    print(f"  • Top eigenvalues: {config.top_n_eigenvalues}")
    print(f"  • Interpolation points: {config.interpolation_points}")
    print(f"  • Log transformation: {config.use_log_transform}")
    print(f"  • Normalization: {config.normalization_method}")
    print(f"  • DTW constraint: {config.dtw_global_constraint} (ratio: {config.dtw_constraint_ratio})")
    print(f"  • Parallel workers: {config.num_workers_models} (models), {config.num_workers_dtw} (DTW)")
    print(f"  • Output directory: {config.output_dir}")
    
    # Validate tslearn availability
    if not TSLEARN_AVAILABLE:
        print("\n❌ Error: tslearn is not available!")
        print("   Please install tslearn: pip install tslearn")
        return None
    
    print("\n✅ tslearn is available for multivariate DTW")
    
    # Set global seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Define model paths
    print("\n📥 Model Configuration:")
    model_paths = {
        'mlp_trained': ("MLPModel", "models/torch_mlp_acc_1.0000_epoch_200.pth"),
        'mlp_trained_98': ("MLPModel", "models/torch_mlp_acc_0.9857_epoch_100.pth"),
        'custom_trained': ("ActualCustomModel", "models/torch_custom_acc_1.0000_epoch_200.pth"),
        'rand_mlp': ("MLPModel", "models/random_mlp_net_000_default_seed_42.pth"),
        'rand_custom': ("ActualCustomModel", "models/random_custom_net_000_default_seed_42.pth")
    }
    
    for key, (class_name, path) in model_paths.items():
        print(f"  • {key}: {class_name} from {path}")
    
    # Start timing
    total_start_time = time.time()
    
    # Phase 1: Analyze models in parallel
    print(f"\n{'='*50}")
    print("PHASE 1: MODEL ANALYSIS")
    print(f"{'='*50}")
    
    analyzed_models = analyze_models_parallel(model_paths, config)
    
    if len(analyzed_models) < 2:
        print("\n❌ Error: Need at least 2 successfully analyzed models for comparison")
        return None
    
    # Phase 2: Compare model pairs
    print(f"\n{'='*50}")
    print("PHASE 2: MODEL COMPARISONS")
    print(f"{'='*50}")
    
    # Define comparison pairs
    model_pairs = [
        ('mlp_trained', 'custom_trained'),      # Different architectures, both trained
        ('mlp_trained', 'mlp_trained_98'),      # Same architecture, different training
        ('mlp_trained', 'rand_mlp'),            # Trained vs random
        ('custom_trained', 'rand_custom'),      # Trained vs random
        ('rand_mlp', 'rand_custom'),            # Random vs random
    ]
    
    # Filter pairs to only include available models
    available_pairs = []
    for model1, model2 in model_pairs:
        if model1 in analyzed_models and model2 in analyzed_models:
            available_pairs.append((model1, model2))
        else:
            print(f"  Skipping {model1} vs {model2} (model not available)")
    
    print(f"\n🔄 Computing {len(available_pairs)} model pair comparisons...")
    
    comparison_results = compare_model_pairs(analyzed_models, available_pairs, config)
    
    if not comparison_results:
        print("\n❌ Error: No successful model comparisons")
        return None
    
    # Phase 3: Analyze separation metrics
    print(f"\n{'='*50}")
    print("PHASE 3: SEPARATION ANALYSIS")
    print(f"{'='*50}")
    
    separation_analysis = analyze_separation_metrics(comparison_results)
    
    # Phase 4: Create visualizations
    print(f"\n{'='*50}")
    print("PHASE 4: VISUALIZATION")
    print(f"{'='*50}")
    
    create_comparison_visualizations(comparison_results, config)
    
    # Print summary results
    print_results_table(comparison_results)
    
    # Calculate total execution time
    total_time = time.time() - total_start_time
    
    # Print final summary
    print(f"\n{'='*70}")
    print("🎉 ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds")
    print(f"📊 Successfully analyzed: {len(analyzed_models)} models")
    print(f"🔄 Successfully compared: {len(comparison_results)} model pairs")
    print(f"📁 Results saved to: {config.output_dir}/")
    
    # Show separation quality if available
    if 'separation_info' in separation_analysis and separation_analysis['separation_info']:
        sep_info = separation_analysis['separation_info']
        if 'separation_ratio' in sep_info:
            print(f"🏆 Separation ratio: {sep_info['separation_ratio']:.2f}x")
            print(f"   {sep_info['quality_assessment']}")
    
    print(f"\n✨ Multivariate DTW analysis using tslearn completed successfully!")
    
    return {
        'analyzed_models': analyzed_models,
        'comparison_results': comparison_results,
        'separation_analysis': separation_analysis,
        'config': config,
        'execution_time': total_time
    }


if __name__ == "__main__":
    # Run the main analysis
    results = main()
    
    if results:
        print(f"\n🔍 Analysis Summary:")
        print(f"  • Models analyzed: {len(results['analyzed_models'])}")
        print(f"  • Comparisons completed: {len(results['comparison_results'])}")
        print(f"  • Total execution time: {results['execution_time']:.1f}s")
        
        # Optional: Save results to file for further analysis
        output_file = os.path.join(results['config'].output_dir, "analysis_results.json")
        try:
            import json
            with open(output_file, 'w') as f:
                # Create serializable results
                serializable_results = {
                    'comparison_results': results['comparison_results'],
                    'separation_analysis': results['separation_analysis'],
                    'execution_time': results['execution_time'],
                    'config': {
                        'top_n_eigenvalues': results['config'].top_n_eigenvalues,
                        'interpolation_points': results['config'].interpolation_points,
                        'use_log_transform': results['config'].use_log_transform,
                        'normalization_method': results['config'].normalization_method,
                        'dtw_constraint_ratio': results['config'].dtw_constraint_ratio
                    }
                }
                json.dump(serializable_results, f, indent=2)
            print(f"  • Detailed results saved to: {output_file}")
        except Exception as e:
            print(f"  • Warning: Could not save results to JSON: {e}")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")
        exit(1)
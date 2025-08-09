#!/usr/bin/env python3
"""
Eigenvalue Evolution SRVF Comparison Script

This script compares eigenvalue evolution across filtration between neural networks
using a novel approach based on SRVF (Square Root Velocity Function) and Hungarian matching.

The pipeline implements:
1. Eigenvalue trajectory extraction from neural networks
2. Feature augmentation (log-scale, alive indicators)
3. SRVF-based elastic distance computation for speed-invariant comparison
4. Hungarian matching for permutation-invariant set-to-set distance
5. Comprehensive diagnostics and visualization

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv
    python eigenvalue_evolution_srvf_comparison.py
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
import fdasrsf
from dataclasses import dataclass
import time

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
class SRVFConfig:
    """Configuration for SRVF comparison pipeline."""
    # Data generation - INCREASED for better sheaf quality
    data_batch_size: int = 100  # Increased from 50
    random_seed: int = 42
    
    # Eigenvalue selection
    top_n_eigenvalues: int = 100  # Use all 100 eigenvalues as per requirements
    
    # Interpolation
    interpolation_points: int = 101  # Common grid size
    
    # Feature augmentation - MORE SENSITIVE collapse detection
    log_floor_epsilon: float = 1e-9
    collapse_threshold: float = 1e-14  # Decreased from 1e-12 for more sensitivity
    alive_ramp_samples: int = 2
    collapse_smoothing_sigma: float = 1.0  # Temporal smoothing for collapse channel
    
    # Channel weights - EMPHASIZE COLLAPSE TIMING
    value_channel_weight: float = 0.4  # Decreased from 0.7
    alive_channel_weight: float = 0.6  # Increased from 0.3 for better separation
    
    # SRVF parameters
    srvf_warp_penalty: float = 0.1  # Medium penalty
    use_rotation: bool = False
    use_scale: bool = False
    
    # Hungarian matching
    use_trimmed_mean: bool = True
    trim_percentage: float = 0.05
    
    # Visualization
    save_visualizations: bool = True
    output_dir: str = "srvf_comparison_results"


# ============================================================================
# Model Classes (from simple_dtw_comparison.py)
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
# Eigenvalue Evolution Extraction
# ============================================================================

def extract_eigenvalue_evolution(
    model: nn.Module,
    data: torch.Tensor,
    config: SRVFConfig,
    model_name: str = "model"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Extract eigenvalue evolution from a model using neurosheaf pipeline.
    
    Returns:
        eigenvalue_sequences: List of arrays, one per filtration step
        filtration_params: Array of filtration parameter values
    """
    print(f"\n📊 Extracting eigenvalue evolution for {model_name}...")
    
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
    
    # Filter to top-N eigenvalues if configured
    if config.top_n_eigenvalues is not None:
        filtered_sequences = []
        for step_eigenvals in eigenvalue_sequences:
            if len(step_eigenvals) > config.top_n_eigenvalues:
                sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
                filtered_sequences.append(sorted_eigenvals[:config.top_n_eigenvalues].numpy())
            else:
                # Pad with zeros if fewer eigenvalues
                padded = np.zeros(config.top_n_eigenvalues)
                padded[:len(step_eigenvals)] = step_eigenvals.numpy()
                filtered_sequences.append(padded)
        eigenvalue_sequences = filtered_sequences
    
    print(f"  ✓ Extracted {len(eigenvalue_sequences)} filtration steps")
    print(f"  ✓ Each step has {len(eigenvalue_sequences[0])} eigenvalues")
    
    return eigenvalue_sequences, filtration_params


# ============================================================================
# Eigenvalue Tracking and Interpolation
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
    # Find maximum number of eigenvalues
    max_eigenvalues = max(len(seq) for seq in eigenvalue_sequences)
    
    # Pad sequences to have same length
    padded_sequences = []
    for seq in eigenvalue_sequences:
        if len(seq) < max_eigenvalues:
            # Pad with zeros
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
        
        # Add small penalty for large changes to encourage smooth tracking
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
    original_grid,
    n_points: int = 101
) -> np.ndarray:
    """Interpolate curves to a common uniform grid.
    
    Args:
        curves: Array of shape (n_curves, n_original_points)
        original_grid: Original filtration parameter values (list or array)
        n_points: Number of points in uniform grid
        
    Returns:
        interpolated: Array of shape (n_curves, n_points)
    """
    n_curves = curves.shape[0]
    
    # Convert to numpy array if needed
    if not isinstance(original_grid, np.ndarray):
        original_grid = np.array(original_grid)
    
    # Create uniform grid on [0, 1]
    uniform_grid = np.linspace(0, 1, n_points)
    
    # Normalize original grid to [0, 1]
    norm_original = (original_grid - original_grid.min()) / (original_grid.max() - original_grid.min())
    
    # Interpolate each curve
    interpolated = np.zeros((n_curves, n_points))
    for i in range(n_curves):
        # Use linear interpolation with endpoint clamping to avoid oscillations/extrapolation artifacts
        f = interp1d(
            norm_original,
            curves[i],
            kind='linear',
            bounds_error=False,
            fill_value=(curves[i][0], curves[i][-1])
        )
        interpolated[i] = f(uniform_grid)
    
    return interpolated


# ============================================================================
# Feature Augmentation
# ============================================================================

def augment_eigenvalue_curves(
    curves: np.ndarray,
    config: SRVFConfig
) -> np.ndarray:
    """Augment eigenvalue curves with additional features.
    
    Args:
        curves: Array of shape (n_curves, n_points)
        config: Configuration parameters
        
    Returns:
        augmented: Array of shape (n_curves, n_points, 2) without per-model standardization/weighting
    """
    n_curves, n_points = curves.shape
    
    # Initialize augmented array (2 channels: log-value, alive indicator)
    augmented = np.zeros((n_curves, n_points, 2))
    
    for i in range(n_curves):
        curve = curves[i]
        
        # Skip all-zero curves (padded eigenvalues)
        if np.all(curve == 0):
            # For all-zero curves, use constant small values
            augmented[i, :, 0] = np.log(config.log_floor_epsilon)
            augmented[i, :, 1] = 0  # Not alive
            continue
        
        # Channel 1: Log-transformed values with floor
        log_values = np.log(np.maximum(curve, config.log_floor_epsilon))
        
        # Handle any remaining NaN/Inf
        log_values = np.nan_to_num(log_values, 
                                  nan=np.log(config.log_floor_epsilon),
                                  posinf=10.0,
                                  neginf=np.log(config.log_floor_epsilon))
        augmented[i, :, 0] = log_values
        
        # Channel 2: Alive indicator with smooth ramping
        alive = np.ones(n_points)
        
        # Find where curve drops below threshold
        below_threshold = curve < config.collapse_threshold
        if np.any(below_threshold):
            # Find first index where it goes below threshold
            collapse_idx = np.argmax(below_threshold)
            
            # Create smooth ramp to 0
            ramp_start = max(0, collapse_idx - config.alive_ramp_samples)
            ramp_end = collapse_idx
            
            if ramp_start < ramp_end:
                ramp_values = np.linspace(1, 0, ramp_end - ramp_start)
                alive[ramp_start:ramp_end] = ramp_values
            
            # Everything after ramp is 0
            alive[ramp_end:] = 0
        
        # Temporal smoothing on collapse channel to tolerate small timing shifts
        try:
            from scipy.ndimage import gaussian_filter1d
            alive = gaussian_filter1d(alive, sigma=config.collapse_smoothing_sigma, mode='nearest')
        except Exception:
            # If scipy not available, leave as-is
            pass
        
        augmented[i, :, 1] = alive
    
    return augmented


# ============================================================================
# SRVF Distance Computation
# ============================================================================

def compute_srvf_distance(
    curve1: np.ndarray,
    curve2: np.ndarray,
    config: SRVFConfig
) -> Tuple[float, Optional[np.ndarray]]:
    """Compute SRVF elastic distance between two curves.
    
    Args:
        curve1, curve2: Arrays of shape (n_points, n_channels)
        config: Configuration parameters
        
    Returns:
        distance: Elastic distance between curves
        warp: Optimal warping function (if available)
    """
    try:
        # Validate input data
        if not np.all(np.isfinite(curve1)) or not np.all(np.isfinite(curve2)):
            warnings.warn("Non-finite values in curves, using Fréchet fallback")
            return compute_frechet_distance(curve1, curve2), None
        
        # Check for constant curves (no variation)
        std1 = np.std(curve1, axis=0)
        std2 = np.std(curve2, axis=0)
        
        if np.all(std1 < 1e-10) or np.all(std2 < 1e-10):
            # At least one curve is essentially constant
            # Use simple L2 distance between mean positions
            mean1 = np.mean(curve1, axis=0)
            mean2 = np.mean(curve2, axis=0)
            distance = np.linalg.norm(mean1 - mean2)
            # Add small warping regularization via collapse (alive) channel mismatch
            alive_penalty = float(np.mean(np.abs(curve1[:, 1] - curve2[:, 1]))) if curve1.shape[1] > 1 else 0.0
            distance += config.srvf_warp_penalty * alive_penalty
            return float(distance), None
        
        # Add small noise to avoid numerical issues in SRVF
        curve1_safe = curve1 + np.random.randn(*curve1.shape) * 1e-8
        curve2_safe = curve2 + np.random.randn(*curve2.shape) * 1e-8
        
        # Transpose for fdasrsf format: (n_channels, n_points)
        beta1 = curve1_safe.T
        beta2 = curve2_safe.T
        
        # Ensure curves have enough variation for SRVF
        # Scale curves to have similar magnitudes
        scale1 = np.max(np.abs(beta1))
        scale2 = np.max(np.abs(beta2))
        
        if scale1 > 1e-10:
            beta1 = beta1 / scale1
        if scale2 > 1e-10:
            beta2 = beta2 / scale2
        
        # Compute elastic distance
        from fdasrsf import curve_functions as cf
        
        # Use elastic_distance_curve for multivariate curves
        dist_result = cf.elastic_distance_curve(
            beta1, beta2,
            closed=0,  # Open curves
            rotation=config.use_rotation,
            scale=config.use_scale,
            method='DP'  # Dynamic programming
        )
        
        # Extract distance (first element is shape distance)
        distance = dist_result[0] if isinstance(dist_result, tuple) else dist_result
        
        # Check for valid distance
        if not np.isfinite(distance):
            warnings.warn("SRVF returned non-finite distance, using Fréchet fallback")
            base = compute_frechet_distance(curve1, curve2)
            # Regularize with alive channel mismatch
            alive_penalty = float(np.mean(np.abs(curve1[:, 1] - curve2[:, 1]))) if curve1.shape[1] > 1 else 0.0
            return base + config.srvf_warp_penalty * alive_penalty, None
        
        # Try to get warping function for diagnostics
        warp = None
        try:
            # Find optimal alignment
            aligned_result = cf.find_rotation_and_seed_coord(
                beta1, beta2,
                closed=0,
                rotation=config.use_rotation,
                method='DP'
            )
            if len(aligned_result) > 2:
                warp = aligned_result[2]  # Warping function
        except:
            pass
        
        # Add warping regularization via alive channel mismatch (penalize over-warping that ignores collapse timing)
        alive_penalty = float(np.mean(np.abs(curve1[:, 1] - curve2[:, 1]))) if curve1.shape[1] > 1 else 0.0
        distance = float(distance) + config.srvf_warp_penalty * alive_penalty
        return distance, warp
        
    except Exception as e:
        warnings.warn(f"SRVF computation failed: {e}, using Fréchet fallback")
        base = compute_frechet_distance(curve1, curve2)
        alive_penalty = float(np.mean(np.abs(curve1[:, 1] - curve2[:, 1]))) if curve1.shape[1] > 1 else 0.0
        return base + config.srvf_warp_penalty * alive_penalty, None


def compute_frechet_distance(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """Compute discrete Fréchet distance between two curves.
    
    Implementation of discrete Fréchet distance using dynamic programming.
    """
    n = len(curve1)
    m = len(curve2)
    
    # Initialize DP table
    dp = np.full((n, m), np.inf)
    
    # Base case
    dp[0, 0] = np.linalg.norm(curve1[0] - curve2[0])
    
    # Fill first row
    for j in range(1, m):
        dp[0, j] = max(dp[0, j-1], np.linalg.norm(curve1[0] - curve2[j]))
    
    # Fill first column
    for i in range(1, n):
        dp[i, 0] = max(dp[i-1, 0], np.linalg.norm(curve1[i] - curve2[0]))
    
    # Fill rest of table
    for i in range(1, n):
        for j in range(1, m):
            cost = np.linalg.norm(curve1[i] - curve2[j])
            dp[i, j] = max(cost, min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]))
    
    return dp[n-1, m-1]


# ============================================================================
# Hungarian Matching for Set Distance
# ============================================================================

def compute_set_distance(
    curves1: np.ndarray,
    curves2: np.ndarray,
    config: SRVFConfig
) -> Dict[str, Any]:
    """Compute set-to-set distance using Hungarian matching.
    
    Args:
        curves1, curves2: Arrays of shape (n_curves, n_points, n_channels)
        config: Configuration parameters
        
    Returns:
        Dictionary with distance metrics and matching information
    """
    n_curves = curves1.shape[0]
    
    print("\n🔄 Computing pairwise curve distances...")
    
    # Build cost matrix
    cost_matrix = np.zeros((n_curves, n_curves))
    warp_functions = {}
    
    for i in range(n_curves):
        if i % 10 == 0:
            print(f"  Processing curve {i+1}/{n_curves}...")
        
        for j in range(n_curves):
            distance, warp = compute_srvf_distance(
                curves1[i], curves2[j], config
            )
            cost_matrix[i, j] = distance
            
            if warp is not None:
                warp_functions[(i, j)] = warp
    
    print("✓ Cost matrix computed")
    
    # Solve Hungarian assignment
    print("🎯 Solving optimal matching...")
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract matched costs
    matched_costs = cost_matrix[row_indices, col_indices]
    
    # Compute statistics
    if config.use_trimmed_mean:
        # Trimmed mean (remove top/bottom percentiles)
        trim_n = int(n_curves * config.trim_percentage)
        if trim_n > 0:
            sorted_costs = np.sort(matched_costs)
            trimmed_costs = sorted_costs[trim_n:-trim_n] if trim_n < len(sorted_costs)//2 else sorted_costs
            mean_distance = np.mean(trimmed_costs)
        else:
            mean_distance = np.mean(matched_costs)
    else:
        mean_distance = np.mean(matched_costs)
    
    # Find best and worst matches
    best_idx = np.argmin(matched_costs)
    worst_idx = np.argmax(matched_costs)
    
    best_match = (row_indices[best_idx], col_indices[best_idx], matched_costs[best_idx])
    worst_match = (row_indices[worst_idx], col_indices[worst_idx], matched_costs[worst_idx])
    
    return {
        'mean_distance': mean_distance,
        'median_distance': np.median(matched_costs),
        'std_distance': np.std(matched_costs),
        'min_distance': np.min(matched_costs),
        'max_distance': np.max(matched_costs),
        'matched_costs': matched_costs,
        'matching': list(zip(row_indices, col_indices)),
        'best_match': best_match,
        'worst_match': worst_match,
        'cost_matrix': cost_matrix,
        'warp_functions': warp_functions
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_matched_curves(
    curves1: np.ndarray,
    curves2: np.ndarray,
    result: Dict[str, Any],
    config: SRVFConfig,
    name1: str = "Model 1",
    name2: str = "Model 2"
):
    """Create visualizations of matched curves and analysis results."""
    
    if not config.save_visualizations:
        return
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Distance distribution histogram
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=result['matched_costs'],
        nbinsx=30,
        name='Matched Pair Distances',
        marker_color='blue'
    ))
    fig_dist.add_vline(x=result['mean_distance'], line_dash="dash", 
                      line_color="red", annotation_text="Mean")
    fig_dist.add_vline(x=result['median_distance'], line_dash="dash", 
                      line_color="green", annotation_text="Median")
    fig_dist.update_layout(
        title=f"Distribution of Matched Pair Distances<br>{name1} vs {name2}",
        xaxis_title="SRVF Distance",
        yaxis_title="Count",
        showlegend=True
    )
    fig_dist.write_html(os.path.join(config.output_dir, "distance_distribution.html"))
    
    # 2. Best and worst matched pairs
    fig_matches = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Best Match - Value Channel', 'Best Match - Alive Channel',
                       'Worst Match - Value Channel', 'Worst Match - Alive Channel']
    )
    
    # Best match
    best_i, best_j, best_dist = result['best_match']
    x = np.linspace(0, 1, curves1.shape[1])
    
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves1[best_i, :, 0], name=f'{name1} #{best_i}', line=dict(color='blue')),
        row=1, col=1
    )
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves2[best_j, :, 0], name=f'{name2} #{best_j}', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves1[best_i, :, 1], name=f'{name1} #{best_i}', line=dict(color='blue'), showlegend=False),
        row=1, col=2
    )
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves2[best_j, :, 1], name=f'{name2} #{best_j}', line=dict(color='red', dash='dash'), showlegend=False),
        row=1, col=2
    )
    
    # Worst match
    worst_i, worst_j, worst_dist = result['worst_match']
    
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves1[worst_i, :, 0], name=f'{name1} #{worst_i}', line=dict(color='blue'), showlegend=False),
        row=2, col=1
    )
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves2[worst_j, :, 0], name=f'{name2} #{worst_j}', line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1
    )
    
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves1[worst_i, :, 1], name=f'{name1} #{worst_i}', line=dict(color='blue'), showlegend=False),
        row=2, col=2
    )
    fig_matches.add_trace(
        go.Scatter(x=x, y=curves2[worst_j, :, 1], name=f'{name2} #{worst_j}', line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=2
    )
    
    fig_matches.update_layout(
        title=f"Best (dist={best_dist:.3f}) and Worst (dist={worst_dist:.3f}) Matched Pairs",
        height=800,
        showlegend=True
    )
    fig_matches.write_html(os.path.join(config.output_dir, "matched_pairs.html"))
    
    # 3. Cost matrix heatmap
    fig_cost = go.Figure(data=go.Heatmap(
        z=result['cost_matrix'],
        x=[f"M2-{i}" for i in range(result['cost_matrix'].shape[1])],
        y=[f"M1-{i}" for i in range(result['cost_matrix'].shape[0])],
        colorscale='Viridis',
        colorbar=dict(title="Distance")
    ))
    
    # Add matching line
    for i, j in result['matching']:
        fig_cost.add_shape(
            type="circle",
            x0=j-0.4, y0=i-0.4, x1=j+0.4, y1=i+0.4,
            line=dict(color="red", width=2)
        )
    
    fig_cost.update_layout(
        title=f"Cost Matrix with Hungarian Matching<br>{name1} vs {name2}",
        xaxis_title=name2,
        yaxis_title=name1,
        width=800,
        height=800
    )
    fig_cost.write_html(os.path.join(config.output_dir, "cost_matrix.html"))
    
    print(f"✓ Visualizations saved to {config.output_dir}/")


# ============================================================================
# Main Comparison Pipeline
# ============================================================================

def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    name1: str,
    name2: str,
    config: SRVFConfig
) -> Dict[str, Any]:
    """Complete pipeline for comparing two models."""
    
    print(f"\n{'='*60}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Generate data
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    data = 8 * torch.randn(config.data_batch_size, 3)
    
    # Extract eigenvalue evolution
    eigen_seq1, filt_params1 = extract_eigenvalue_evolution(model1, data, config, name1)
    eigen_seq2, filt_params2 = extract_eigenvalue_evolution(model2, data, config, name2)
    
    # Track eigenvalues to create continuous curves
    print("\n📈 Tracking eigenvalues through filtration...")
    tracked1 = track_eigenvalues_hungarian(eigen_seq1)
    tracked2 = track_eigenvalues_hungarian(eigen_seq2)
    print(f"  ✓ Tracked curves shape: {tracked1.shape}")
    
    # Interpolate to common grid
    print("\n🔄 Interpolating to common grid...")
    interp1 = interpolate_curves_to_grid(tracked1, filt_params1, config.interpolation_points)
    interp2 = interpolate_curves_to_grid(tracked2, filt_params2, config.interpolation_points)
    print(f"  ✓ Interpolated shape: {interp1.shape}")
    
    # Augment with features (no per-model scaling/weighting here)
    print("\n🎨 Augmenting curves with features...")
    augmented1 = augment_eigenvalue_curves(interp1, config)
    augmented2 = augment_eigenvalue_curves(interp2, config)
    print(f"  ✓ Augmented shape: {augmented1.shape}")

    # Apply common scaling across both models to avoid inconsistent standardization
    print("\n⚖️  Applying common channel scaling across both models...")
    stacked = np.concatenate([augmented1, augmented2], axis=0)  # (n1+n2, n_points, n_channels)
    n_channels = stacked.shape[2]
    means = np.zeros(n_channels)
    stds = np.ones(n_channels)
    for ch in range(n_channels):
        data_ch = stacked[:, :, ch].flatten()
        finite = data_ch[np.isfinite(data_ch)]
        if len(finite) > 0:
            means[ch] = np.mean(finite)
            std = np.std(finite)
            stds[ch] = std if std > 1e-10 else 1.0
    # Standardize both models with the same stats
    for ch in range(n_channels):
        augmented1[:, :, ch] = (augmented1[:, :, ch] - means[ch]) / stds[ch]
        augmented2[:, :, ch] = (augmented2[:, :, ch] - means[ch]) / stds[ch]
    # Apply channel weights AFTER common scaling to preserve emphasis
    augmented1[:, :, 0] *= config.value_channel_weight
    augmented1[:, :, 1] *= config.alive_channel_weight
    augmented2[:, :, 0] *= config.value_channel_weight
    augmented2[:, :, 1] *= config.alive_channel_weight
    
    # Compute set distance
    result = compute_set_distance(augmented1, augmented2, config)
    
    # Add model names to result
    result['model1_name'] = name1
    result['model2_name'] = name2
    
    # Create visualizations
    visualize_matched_curves(augmented1, augmented2, result, config, name1, name2)
    
    return result


def analyze_collapse_positions(
    curves: np.ndarray,
    config: SRVFConfig
) -> np.ndarray:
    """Analyze where eigenvalues collapse to zero."""
    n_curves = curves.shape[0]
    collapse_positions = []
    
    for i in range(n_curves):
        curve = curves[i]
        below_threshold = curve < config.collapse_threshold
        if np.any(below_threshold):
            collapse_idx = np.argmax(below_threshold)
            collapse_positions.append(collapse_idx / len(curve))  # Normalized position
        else:
            collapse_positions.append(1.0)  # Never collapses
    
    return np.array(collapse_positions)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    
    print("🚀 SRVF-based Eigenvalue Evolution Comparison")
    print("=" * 60)
    
    # Initialize configuration
    config = SRVFConfig()
    
    # Load models
    print("\n📥 Loading models...")
    model_paths = {
        'mlp_trained': "models/torch_mlp_acc_1.0000_epoch_200.pth",
        'mlp_trained_98': "models/torch_mlp_acc_0.9857_epoch_100.pth",
        'custom_trained': "models/torch_custom_acc_1.0000_epoch_200.pth",
        'rand_mlp': "models/random_mlp_net_000_default_seed_42.pth",
        'rand_custom': "models/random_custom_net_000_default_seed_42.pth"
    }
    
    models = {}
    for key, path in model_paths.items():
        if 'mlp' in key:
            models[key] = load_model(MLPModel, path, device='cpu')
        else:
            models[key] = load_model(ActualCustomModel, path, device='cpu')
    
    print(f"✓ Loaded {len(models)} models")
    
    # Perform comparisons
    all_results = {}
    
    # Key comparisons
    comparisons = [
        #('mlp_trained', 'custom_trained'),      # Different architectures, both trained
        #('mlp_trained', 'mlp_trained_98'),      # Same architecture, different training
        ('mlp_trained', 'rand_mlp'),            # Trained vs random
        #('custom_trained', 'rand_custom'),      # Trained vs random
        #('rand_mlp', 'rand_custom'),            # Random vs random
    ]
    
    for model1_key, model2_key in comparisons:
        result = compare_models(
            models[model1_key],
            models[model2_key],
            model1_key,
            model2_key,
            config
        )
        all_results[f"{model1_key}_vs_{model2_key}"] = result
    
    # Compute separation metrics
    print(f"\n{'='*60}")
    print("📊 SEPARATION ANALYSIS")
    print(f"{'='*60}")
    
    # Categorize distances
    intra_trained = []
    inter_group = []
    intra_random = []
    
    for key, result in all_results.items():
        distance = result['mean_distance']
        model1, model2 = key.split('_vs_')
        
        if 'trained' in model1 and 'trained' in model2 and 'rand' not in model1 and 'rand' not in model2:
            intra_trained.append(distance)
        elif ('trained' in model1 and 'rand' not in model1 and 'rand' in model2) or \
             ('rand' in model1 and 'trained' in model2 and 'rand' not in model2):
            inter_group.append(distance)
        elif 'rand' in model1 and 'rand' in model2:
            intra_random.append(distance)
    
    # Display results
    print("\n📈 Distance Statistics by Category:")
    print("-" * 40)
    
    if intra_trained:
        print(f"🟢 Intra-Trained (trained vs trained):")
        print(f"   Mean: {np.mean(intra_trained):.4f}")
        print(f"   Std:  {np.std(intra_trained):.4f}")
        print(f"   Values: {[f'{d:.4f}' for d in intra_trained]}")
    
    if inter_group:
        print(f"\n🔴 Inter-Group (trained vs random):")
        print(f"   Mean: {np.mean(inter_group):.4f}")
        print(f"   Std:  {np.std(inter_group):.4f}")
        print(f"   Values: {[f'{d:.4f}' for d in inter_group]}")
    
    if intra_random:
        print(f"\n🟡 Intra-Random (random vs random):")
        print(f"   Mean: {np.mean(intra_random):.4f}")
        print(f"   Std:  {np.std(intra_random):.4f}")
        print(f"   Values: {[f'{d:.4f}' for d in intra_random]}")
    
    # Compute separation ratio
    if intra_trained and inter_group:
        separation_ratio = np.mean(inter_group) / np.mean(intra_trained)
        print(f"\n🏆 SEPARATION RATIO: {separation_ratio:.2f}x")
        
        if separation_ratio > 10:
            print("   ✅ EXCELLENT: Strong separation between trained and random models")
        elif separation_ratio > 5:
            print("   ✅ GOOD: Clear separation between trained and random models")
        elif separation_ratio > 2:
            print("   ⚠️  MODERATE: Some separation between trained and random models")
        else:
            print("   ❌ POOR: Insufficient separation")
    
    # Display complete comparison table
    print(f"\n📋 Complete Comparison Results:")
    print("-" * 60)
    print(f"{'Comparison':<35} {'Mean Dist':<12} {'Median':<12} {'Std':<12}")
    print("-" * 60)
    
    for key, result in all_results.items():
        print(f"{key:<35} {result['mean_distance']:<12.4f} {result['median_distance']:<12.4f} {result['std_distance']:<12.4f}")
    
    print(f"\n✅ Analysis complete! Results saved to {config.output_dir}/")
    
    return all_results


if __name__ == "__main__":
    start_time = time.time()
    results = main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed:.1f} seconds")
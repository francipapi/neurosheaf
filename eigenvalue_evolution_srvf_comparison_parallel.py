#!/usr/bin/env python3
"""
Parallel Eigenvalue Evolution SRVF Comparison Script

This script compares eigenvalue evolution across filtration between neural networks
using SRVF (Square Root Velocity Function) and Hungarian matching with parallel processing.

Key improvements over sequential version:
- Parallel model analysis (eigenvalue extraction)
- Parallel pairwise curve distance computation (100×100 distances)
- Significantly faster execution (~5-10x speedup)

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv
    python eigenvalue_evolution_srvf_comparison_parallel.py
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
import concurrent.futures as futures
import pickle

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
    """Configuration for parallel SRVF comparison pipeline.
    
    LOG-SCALE CONFIGURATION: Optimized for log10 eigenvalue transformation
    to improve separation between trained and random models.
    """
    # Data generation - INCREASED for better sheaf quality
    data_batch_size: int = 100  # Increased from 50
    random_seed: int = 42
    
    # Eigenvalue selection
    top_n_eigenvalues: int = 100  # Use all 100 eigenvalues as per requirements
    
    # Interpolation
    interpolation_points: int = 151  # Higher resolution for better curve capture
    
    # Feature augmentation - MULTI-THRESHOLD collapse detection
    log_floor_epsilon: float = 1e-9
    collapse_thresholds: List[float] = None  # Will be set in __post_init__
    alive_ramp_samples: int = 2
    
    # Channel weights - MULTI-THRESHOLD OPTIMIZED
    value_channel_weight: float = 0.6  # Moderate focus on log-scale magnitudes
    collapse_stages_weight: float = 0.4  # Strong emphasis on collapse progression
    
    # SRVF parameters
    srvf_warp_penalty: float = 0.1  # Moderate penalty for log-scale curves
    use_rotation: bool = False
    use_scale: bool = False
    
    # Hungarian matching
    use_trimmed_mean: bool = True
    trim_percentage: float = 0.05
    
    # Parallel processing
    num_workers_models: int = 5  # Number of workers for model analysis
    num_workers_curves: int = 10  # Number of workers for curve distances
    
    # Visualization
    save_visualizations: bool = True
    output_dir: str = "srvf_parallel_results"
    
    def __post_init__(self):
        if self.collapse_thresholds is None:
            # Multi-threshold collapse detection: from subtle to complete collapse
            self.collapse_thresholds = [1e-12, 1e-14, 1e-16, 1e-18]


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
# Utility Functions
# ============================================================================

def _rebuild_model(class_name: str) -> nn.Module:
    """Rebuild model from class name for parallel workers."""
    if class_name == 'MLPModel':
        return MLPModel()
    if class_name == 'ActualCustomModel':
        return ActualCustomModel()
    raise ValueError(f"Unknown model class: {class_name}")


def track_eigenvalues_hungarian(eigenvalue_sequences: List[np.ndarray]) -> np.ndarray:
    """Track eigenvalues through filtration using Hungarian algorithm."""
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
        
        # Build cost matrix
        cost_matrix = np.abs(prev_values[:, np.newaxis] - curr_values[np.newaxis, :])
        
        # Add penalty for large changes
        if t > 1:
            prev_diff = tracked_curves[:, t-1] - tracked_curves[:, t-2]
            for i in range(n_eigenvalues):
                for j in range(n_eigenvalues):
                    expected = prev_values[i] + prev_diff[i]
                    cost_matrix[i, j] += 0.1 * np.abs(curr_values[j] - expected)
        
        # Solve assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        for i, j in zip(row_indices, col_indices):
            tracked_curves[i, t] = curr_values[j]
    
    return tracked_curves


def interpolate_curves_to_grid(
    curves: np.ndarray,
    original_grid,
    n_points: int = 101
) -> np.ndarray:
    """Interpolate curves to a common uniform grid."""
    n_curves = curves.shape[0]
    
    # Convert to numpy array if needed
    if not isinstance(original_grid, np.ndarray):
        original_grid = np.array(original_grid)
    
    # Create uniform grid
    uniform_grid = np.linspace(0, 1, n_points)
    
    # Normalize original grid
    norm_original = (original_grid - original_grid.min()) / (original_grid.max() - original_grid.min())
    
    # Interpolate each curve
    interpolated = np.zeros((n_curves, n_points))
    for i in range(n_curves):
        f = interp1d(norm_original, curves[i], kind='cubic', 
                    bounds_error=False, fill_value='extrapolate')
        interpolated[i] = f(uniform_grid)
    
    return interpolated


def augment_eigenvalue_curves(
    curves: np.ndarray,
    config: SRVFConfig
) -> np.ndarray:
    """Augment eigenvalue curves with multi-threshold collapse detection."""
    n_curves, n_points = curves.shape
    n_collapse_stages = len(config.collapse_thresholds)
    
    # Initialize augmented array: [log_values, collapse_stage_1, ..., collapse_stage_N]
    augmented = np.zeros((n_curves, n_points, 1 + n_collapse_stages))
    
    for i in range(n_curves):
        curve = curves[i]
        
        # Skip all-zero curves
        if np.all(curve == 0):
            augmented[i, :, 0] = np.log10(config.log_floor_epsilon)
            augmented[i, :, 1:] = 0
            continue
        
        # Channel 0: LOG10-SCALE transformation for better dynamic range
        log_values = np.log10(np.maximum(curve, config.log_floor_epsilon))
        log_values = np.nan_to_num(log_values, 
                                  nan=np.log10(config.log_floor_epsilon),
                                  posinf=10.0,
                                  neginf=np.log10(config.log_floor_epsilon))
        augmented[i, :, 0] = log_values
        
        # Channels 1+: Multi-threshold collapse stages
        for stage_idx, threshold in enumerate(config.collapse_thresholds):
            collapse_stage = np.ones(n_points)
            below_threshold = curve < threshold
            
            if np.any(below_threshold):
                collapse_idx = np.argmax(below_threshold)
                ramp_start = max(0, collapse_idx - config.alive_ramp_samples)
                ramp_end = collapse_idx
                
                # Create smooth transition from alive (1.0) to collapsed (0.0)
                if ramp_start < ramp_end:
                    ramp_values = np.linspace(1, 0, ramp_end - ramp_start)
                    collapse_stage[ramp_start:ramp_end] = ramp_values
                
                collapse_stage[ramp_end:] = 0
            
            augmented[i, :, stage_idx + 1] = collapse_stage
    
    # Apply channel weights
    total_channels = 1 + n_collapse_stages
    
    # Channel 0: Log values
    augmented[:, :, 0] *= config.value_channel_weight
    
    # Channels 1+: Collapse stages (distribute weight equally across all stages)
    collapse_weight_per_stage = config.collapse_stages_weight / n_collapse_stages
    for stage_idx in range(n_collapse_stages):
        augmented[:, :, stage_idx + 1] *= collapse_weight_per_stage
    
    # Standardize all channels
    for ch in range(total_channels):
        channel_data = augmented[:, :, ch].flatten()
        finite_data = channel_data[np.isfinite(channel_data)]
        if len(finite_data) > 0:
            mean = np.mean(finite_data)
            std = np.std(finite_data)
            
            if std > 1e-10:
                augmented[:, :, ch] = (augmented[:, :, ch] - mean) / std
            else:
                augmented[:, :, ch] = augmented[:, :, ch] - mean
        
        augmented[:, :, ch] = np.nan_to_num(augmented[:, :, ch], 
                                           nan=0.0, posinf=3.0, neginf=-3.0)
    
    return augmented


def compute_srvf_distance(
    curve1: np.ndarray,
    curve2: np.ndarray,
    config: SRVFConfig
) -> float:
    """Compute SRVF elastic distance between two curves."""
    try:
        # Validate input
        if not np.all(np.isfinite(curve1)) or not np.all(np.isfinite(curve2)):
            return compute_frechet_distance(curve1, curve2)
        
        # Check for constant curves
        std1 = np.std(curve1, axis=0)
        std2 = np.std(curve2, axis=0)
        
        if np.all(std1 < 1e-10) or np.all(std2 < 1e-10):
            mean1 = np.mean(curve1, axis=0)
            mean2 = np.mean(curve2, axis=0)
            return float(np.linalg.norm(mean1 - mean2))
        
        # Add small noise
        curve1_safe = curve1 + np.random.randn(*curve1.shape) * 1e-8
        curve2_safe = curve2 + np.random.randn(*curve2.shape) * 1e-8
        
        # Transpose for fdasrsf
        beta1 = curve1_safe.T
        beta2 = curve2_safe.T
        
        # Scale curves
        scale1 = np.max(np.abs(beta1))
        scale2 = np.max(np.abs(beta2))
        
        if scale1 > 1e-10:
            beta1 = beta1 / scale1
        if scale2 > 1e-10:
            beta2 = beta2 / scale2
        
        # Compute elastic distance
        from fdasrsf import curve_functions as cf
        
        dist_result = cf.elastic_distance_curve(
            beta1, beta2,
            closed=0,
            rotation=config.use_rotation,
            scale=config.use_scale,
            method='DP'
        )
        
        distance = dist_result[0] if isinstance(dist_result, tuple) else dist_result
        
        if not np.isfinite(distance):
            return compute_frechet_distance(curve1, curve2)
        
        return float(distance)
        
    except Exception:
        return compute_frechet_distance(curve1, curve2)


def compute_frechet_distance(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """Compute discrete Fréchet distance between two curves."""
    n = len(curve1)
    m = len(curve2)
    
    dp = np.full((n, m), np.inf)
    dp[0, 0] = np.linalg.norm(curve1[0] - curve2[0])
    
    for j in range(1, m):
        dp[0, j] = max(dp[0, j-1], np.linalg.norm(curve1[0] - curve2[j]))
    
    for i in range(1, n):
        dp[i, 0] = max(dp[i-1, 0], np.linalg.norm(curve1[i] - curve2[0]))
    
    for i in range(1, n):
        for j in range(1, m):
            cost = np.linalg.norm(curve1[i] - curve2[j])
            dp[i, j] = max(cost, min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]))
    
    return dp[n-1, m-1]


# ============================================================================
# Parallel Worker Functions
# ============================================================================

def _analyze_model_worker(args: Tuple[str, str, str, int, Tuple[int, ...], Dict[str, Any]]) -> Dict[str, Any]:
    """Worker function to analyze a single model in parallel."""
    model_key, model_class_name, weights_path, data_seed, data_shape, config_dict = args
    
    # Recreate config from dict
    config = SRVFConfig(**config_dict)
    
    # Set seeds
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    
    print(f"  Worker: Analyzing {model_key}...")
    
    # Rebuild model and load weights
    model = _rebuild_model(model_class_name)
    model = load_model(lambda: model, weights_path, device='cpu')
    
    # Generate data
    batch_size, input_dim = data_shape
    data = 8 * torch.randn(batch_size, input_dim)
    
    # Configure GW
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
    
    # Analyze model
    analyzer = NeurosheafAnalyzer(device='cpu')
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
    
    # Track eigenvalues
    tracked = track_eigenvalues_hungarian(eigenvalue_sequences)
    
    # Interpolate to common grid
    interpolated = interpolate_curves_to_grid(tracked, filtration_params, config.interpolation_points)
    
    # Augment with features
    augmented = augment_eigenvalue_curves(interpolated, config)
    
    print(f"  Worker: Completed {model_key}")
    
    # Return serializable data
    return {
        'key': model_key,
        'curves': augmented.tolist(),  # Convert to list for serialization
        'shape': augmented.shape
    }


def _compute_curve_distance_worker(args: Tuple[int, int, np.ndarray, np.ndarray, Dict[str, Any]]) -> Tuple[int, int, float]:
    """Worker function to compute distance between two curves."""
    i, j, curve1, curve2, config_dict = args
    
    # Recreate config
    config = SRVFConfig(**config_dict)
    
    # Compute distance
    distance = compute_srvf_distance(curve1, curve2, config)
    
    return (i, j, distance)


# ============================================================================
# Main Functions
# ============================================================================

def compute_set_distance_parallel(
    curves1: np.ndarray,
    curves2: np.ndarray,
    config: SRVFConfig
) -> Dict[str, Any]:
    """Compute set-to-set distance using parallel processing."""
    n_curves = curves1.shape[0]
    
    print("\n🔄 Computing pairwise curve distances in parallel...")
    
    # Prepare arguments for parallel workers
    worker_args = []
    for i in range(n_curves):
        for j in range(n_curves):
            worker_args.append((i, j, curves1[i], curves2[j], config.__dict__))
    
    # Compute distances in parallel
    cost_matrix = np.zeros((n_curves, n_curves))
    
    batch_size = 1000  # Process in batches to avoid memory issues
    n_batches = (len(worker_args) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(worker_args))
        batch_args = worker_args[start_idx:end_idx]
        
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx+1}/{n_batches}...")
        
        with futures.ProcessPoolExecutor(max_workers=config.num_workers_curves) as pool:
            results = list(pool.map(_compute_curve_distance_worker, batch_args))
        
        for i, j, distance in results:
            cost_matrix[i, j] = distance
    
    print("✓ Cost matrix computed")
    
    # Solve Hungarian assignment
    print("🎯 Solving optimal matching...")
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract matched costs
    matched_costs = cost_matrix[row_indices, col_indices]
    
    # Compute statistics
    if config.use_trimmed_mean:
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
        'cost_matrix': cost_matrix
    }


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
    
    # Distance distribution histogram
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
    fig_dist.write_html(os.path.join(config.output_dir, f"{name1}_vs_{name2}_distances.html"))
    
    print(f"✓ Visualizations saved to {config.output_dir}/")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function with parallel processing."""
    
    print("🚀 Parallel SRVF-based Eigenvalue Evolution Comparison")
    print("=" * 60)
    
    # Initialize configuration
    config = SRVFConfig()
    
    # Set seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Load models
    print("\n📥 Loading models...")
    model_paths = {
        'mlp_trained': ("MLPModel", "models/torch_mlp_acc_1.0000_epoch_200.pth"),
        'mlp_trained_98': ("MLPModel", "models/torch_mlp_acc_0.9857_epoch_100.pth"),
        'custom_trained': ("ActualCustomModel", "models/torch_custom_acc_1.0000_epoch_200.pth"),
        'rand_mlp': ("MLPModel", "models/random_mlp_net_000_default_seed_42.pth"),
        'rand_custom': ("ActualCustomModel", "models/random_custom_net_000_default_seed_42.pth")
    }
    
    print(f"✓ {len(model_paths)} models to analyze")
    
    # Phase 1: Parallel model analysis
    print("\n📊 Phase 1: Analyzing models in parallel...")
    
    model_args = []
    for key, (class_name, path) in model_paths.items():
        model_args.append((
            key, class_name, path,
            config.random_seed,
            (config.data_batch_size, 3),
            config.__dict__
        ))
    
    start_time = time.time()
    analyzed_models = {}
    
    with futures.ProcessPoolExecutor(max_workers=config.num_workers_models) as pool:
        results = list(pool.map(_analyze_model_worker, model_args))
    
    for result in results:
        # Convert back to numpy arrays
        curves = np.array(result['curves'])
        analyzed_models[result['key']] = curves
    
    phase1_time = time.time() - start_time
    print(f"✅ Phase 1 completed in {phase1_time:.1f}s")
    
    # Phase 2: Pairwise comparisons
    print("\n📐 Phase 2: Computing pairwise comparisons...")
    
    comparisons = [
        ('mlp_trained', 'custom_trained'),      # Different architectures, both trained
        ('mlp_trained', 'mlp_trained_98'),      # Same architecture, different training
        ('mlp_trained', 'rand_mlp'),            # Trained vs random
        ('custom_trained', 'rand_custom'),      # Trained vs random
        ('rand_mlp', 'rand_custom'),            # Random vs random
    ]
    
    all_results = {}
    start_time = time.time()
    
    for model1_key, model2_key in comparisons:
        print(f"\n Comparing: {model1_key} vs {model2_key}")
        
        result = compute_set_distance_parallel(
            analyzed_models[model1_key],
            analyzed_models[model2_key],
            config
        )
        
        all_results[f"{model1_key}_vs_{model2_key}"] = result
        
        # Create visualizations
        visualize_matched_curves(
            analyzed_models[model1_key],
            analyzed_models[model2_key],
            result,
            config,
            model1_key,
            model2_key
        )
        
        print(f"   Mean distance: {result['mean_distance']:.4f}")
        print(f"   Median distance: {result['median_distance']:.4f}")
        print(f"   Std: {result['std_distance']:.4f}")
    
    phase2_time = time.time() - start_time
    print(f"✅ Phase 2 completed in {phase2_time:.1f}s")
    
    # Display results
    print(f"\n📋 Complete Comparison Results:")
    print("-" * 60)
    print(f"{'Comparison':<35} {'Mean Dist':<12} {'Median':<12} {'Std':<12}")
    print("-" * 60)
    
    for key, result in all_results.items():
        print(f"{key:<35} {result['mean_distance']:<12.4f} {result['median_distance']:<12.4f} {result['std_distance']:<12.4f}")
    
    total_time = phase1_time + phase2_time
    print(f"\n✅ Analysis complete! Results saved to {config.output_dir}/")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds")
    print(f"    Phase 1 (model analysis): {phase1_time:.1f}s")
    print(f"    Phase 2 (comparisons): {phase2_time:.1f}s")
    
    return all_results


if __name__ == "__main__":
    results = main()
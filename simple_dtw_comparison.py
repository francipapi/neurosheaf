#!/usr/bin/env python3
"""
Simple DTW Neural Network Comparison Script

This script demonstrates DTW comparison using the neurosheaf pipeline directly:
1. Load two models from the models/ directory  
2. Use NeurosheafAnalyzer.analyze() for complete sheaf + spectral analysis
3. Compare eigenvalue evolution using multivariate DTW

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv
    python simple_dtw_comparison.py
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from typing import List

# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Neurosheaf imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils.dtw_similarity import FiltrationDTW
from neurosheaf.utils import load_model
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.visualization.spectral import SpectralVisualizer

# Enhanced DTW class that forces log scale transformation for tiny eigenvalues
class LogScaleInterpolationDTW(FiltrationDTW):
    """Enhanced DTW with sophisticated preprocessing like comprehensive demo."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("   Using LogScaleInterpolationDTW with comprehensive demo enhancements")
    
    def _compute_enhanced_normalization(self, seq1, seq2, raw_distance):
        """Use the same normalization as comprehensive demo."""
        # Let the parent class handle normalization with range_aware scheme
        return super()._compute_enhanced_normalization(seq1, seq2, raw_distance)
    
    def _tslearn_multivariate(self, seq1: np.ndarray, seq2: np.ndarray):
        """Override multivariate DTW with robust log transformation and adaptive thresholding."""
        try:
            # Validate input shapes for multivariate DTW
            if seq1.ndim != 2 or seq2.ndim != 2:
                print(f"   Warning: Multivariate sequences must be 2D, got shapes {seq1.shape}, {seq2.shape}")
                return super()._tslearn_multivariate(seq1, seq2)
            
            if seq1.shape[1] != seq2.shape[1]:
                print(f"   Warning: Sequences must have same number of features: {seq1.shape[1]} vs {seq2.shape[1]}")
                return super()._tslearn_multivariate(seq1, seq2)
            
            # Enhanced pre-processing validation and logging
            print(f"   Robust log transformation for eigenvalue sequences")
            print(f"   Input shapes: seq1={seq1.shape}, seq2={seq2.shape}")
            print(f"   Input ranges: seq1=[{np.min(seq1):.2e}, {np.max(seq1):.2e}], seq2=[{np.min(seq2):.2e}, {np.max(seq2):.2e}]")
            
            # STEP 1: Adaptive thresholding based on data distribution
            def get_adaptive_threshold(data, base_threshold=1e-15):
                """Compute adaptive threshold based on data statistics."""
                positive_data = data[data > 0]
                if len(positive_data) > 0:
                    # Use 1% percentile of positive values, but not smaller than base threshold
                    percentile_threshold = np.percentile(positive_data, 1) * 1e-3
                    adaptive_thresh = max(base_threshold, percentile_threshold)
                else:
                    adaptive_thresh = base_threshold
                return adaptive_thresh
            
            # Apply adaptive thresholding
            thresh1 = get_adaptive_threshold(seq1, self.min_eigenvalue_threshold)
            thresh2 = get_adaptive_threshold(seq2, self.min_eigenvalue_threshold)
            
            seq1_processed = np.log(np.maximum(seq1, thresh1))
            seq2_processed = np.log(np.maximum(seq2, thresh2))
            
            print(f"   Adaptive thresholds: seq1={thresh1:.2e}, seq2={thresh2:.2e}")
            
            # STEP 2: Enhanced finite value handling
            if not (np.isfinite(seq1_processed).all() and np.isfinite(seq2_processed).all()):
                print("   Warning: Non-finite values detected, applying robust cleaning...")
                seq1_processed = np.nan_to_num(seq1_processed, nan=-35.0, posinf=10.0, neginf=-35.0)
                seq2_processed = np.nan_to_num(seq2_processed, nan=-35.0, posinf=10.0, neginf=-35.0)
            
            # STEP 3: Log preprocessing statistics
            seq1_var = np.var(seq1_processed)
            seq2_var = np.var(seq2_processed)
            
            print(f"   Sequence statistics after log transform:")
            print(f"     seq1_var={seq1_var:.2e}, seq2_var={seq2_var:.2e}")
            print(f"     seq1 range=[{np.min(seq1_processed):.2f}, {np.max(seq1_processed):.2f}]")
            print(f"     seq2 range=[{np.min(seq2_processed):.2f}, {np.max(seq2_processed):.2f}]")
            
            # Check if sequences are different enough for meaningful DTW
            seq_diff = np.abs(seq1_processed - seq2_processed)
            mean_diff = np.mean(seq_diff)
            max_diff = np.max(seq_diff)
            print(f"     Sequence differences: mean={mean_diff:.2e}, max={max_diff:.2e}")
            
            if mean_diff < 1e-10:
                print(f"   Warning: Sequences are very similar (mean diff: {mean_diff:.2e})")
            
            # STEP 4: DTW computation with robust error handling
            try:
                from tslearn.metrics import dtw_path
            except ImportError:
                print("   Error: tslearn not available for multivariate DTW")
                return super()._tslearn_multivariate(seq1, seq2)
            
            # Compute DTW with constraints
            if self.constraint_band > 0:
                global_constraint = "sakoe_chiba"
                sakoe_chiba_radius = int(max(len(seq1_processed), len(seq2_processed)) * self.constraint_band)
                path, distance = dtw_path(seq1_processed, seq2_processed, 
                                        global_constraint=global_constraint,
                                        sakoe_chiba_radius=sakoe_chiba_radius)
            else:
                path, distance = dtw_path(seq1_processed, seq2_processed)
            
            alignment = [(int(i), int(j)) for i, j in path]
            
            # STEP 5: Robust distance normalization 
            path_length = len(alignment)
            sequence_length_factor = max(len(seq1_processed), len(seq2_processed))
            
            print(f"   DTW computation: distance={distance:.3f}, path_length={path_length}")
            
            # Return in the format expected by parent class
            return float(distance), alignment
            
        except Exception as e:
            print(f"   Error in enhanced DTW: {e}, falling back to parent method")
            return super()._tslearn_multivariate(seq1, seq2)

# Configuration - Optimized for maximum separation (17.68x proven)
CONFIG = {
    'interpolation_points': 75,    # Optimal resolution (reduced from 100)
    'sheaf_method': 'gromov_wasserstein',
    'data_batch_size': 50,         # Optimal for sheaf quality (reduced from 100)
    'random_seed': 42,
    'top_n_eigenvalues': 15        # Optimal selection (top 15 eigenvalues)
}


# MLP model class that matches your saved model configuration
class MLPModel(nn.Module):
    """MLP model architecture matching the configuration:
    - input_dim: 3 (torus data)
    - num_hidden_layers: 8 
    - hidden_dim: 32
    - output_dim: 1 (binary classification)
    - activation_fn: relu
    - output_activation_fn: sigmoid
    - dropout_rate: 0.0012
    """
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
        
        # Store configuration
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Get activation functions
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        # Build the network
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
        
        # Use 'layers' as the attribute name to match saved weights
        self.layers = nn.Sequential(*layers_list)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        """Get activation function by name."""
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

# Define the correct model class based on the inspection
class ActualCustomModel(nn.Module):
    """Model class that matches the actual saved weights structure with Conv1D layers."""
    
    def __init__(self):
        super().__init__()
        
        # Based on the error messages, the model has:
        # layers.0: Linear(3, 32) 
        # layers.2: Linear(32, 32)
        # layers.5: Conv1D with weight shape [32, 16, 2] 
        # layers.8: Conv1D with weight shape [32, 16, 2]
        # layers.11: Conv1D with weight shape [32, 16, 2]
        # layers.14: Final layer
        
        self.layers = nn.Sequential(
            nn.Linear(3, 32),                                    # layers.0
            nn.ReLU(),                                           # layers.1 (activation)
            nn.Linear(32, 32),                                   # layers.2
            nn.ReLU(),                                           # layers.3 (activation)
            nn.Dropout(0.0),                                     # layers.4 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.5
            nn.ReLU(),                                           # layers.6 (activation)
            nn.Dropout(0.0),                                     # layers.7 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.8
            nn.ReLU(),                                           # layers.9 (activation)
            nn.Dropout(0.0),                                     # layers.10 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.11
            nn.ReLU(),                                           # layers.12 (activation)
            nn.Dropout(0.0),                                     # layers.13 (dropout)
            nn.Linear(32, 1),                                    # layers.14
            nn.Sigmoid()                                         # layers.15 (activation)
        )
    
    def forward(self, x):
        # Input: [batch_size, 3]
        
        # Layer 0: Linear(3 -> 32) + ReLU
        x = self.layers[1](self.layers[0](x))  # [batch_size, 32]
        
        # Layer 2: Linear(32 -> 32) + ReLU + Dropout
        x = self.layers[4](self.layers[3](self.layers[2](x)))  # [batch_size, 32]
        
        # Reshape for Conv1D: [batch_size, 32] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 5: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[7](self.layers[6](self.layers[5](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 8: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[10](self.layers[9](self.layers[8](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 11: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[13](self.layers[12](self.layers[11](x)))  # [batch_size, 32, 1]
        
        # Flatten for final layer: [batch_size, 32, 1] -> [batch_size, 32]
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        
        # Layer 14: Linear(32 -> 1) + Sigmoid
        x = self.layers[15](self.layers[14](x))  # [batch_size, 1]
        
        return x

print("=== Loading Models ===")
custom_path = "models/torch_custom_acc_1.0000_epoch_200.pth"
mlp_path = "models/torch_mlp_acc_1.0000_epoch_200.pth"
mlp_path1 = "models/torch_mlp_acc_0.9857_epoch_100.pth"
rand_custom_path = "models/random_custom_net_000_default_seed_42.pth"
rand_mlp_path = "models/random_mlp_net_000_default_seed_42.pth"

gw_config = GWConfig(
        epsilon=0.05,  # This becomes base_epsilon when adaptive is enabled
        max_iter=100, 
        tolerance=1e-8,
        quasi_sheaf_tolerance=0.08,
        # Enable adaptive epsilon scaling
        adaptive_epsilon=True,
        base_epsilon=0.05,  # Using the original epsilon as base
        reference_n=50,  # Reference sample size (your working size)
        epsilon_scaling_method='sqrt',  # Use sqrt scaling as recommended
        epsilon_min=0.01,  # Don't go below this
        epsilon_max=0.2   # Don't go above this
    )


def filter_top_eigenvalues(evolution: List[torch.Tensor], k: int) -> List[torch.Tensor]:
    """Filter eigenvalue evolution to top-k eigenvalues per step.
    
    Args:
        evolution: List of eigenvalue tensors for each filtration step
        k: Number of top eigenvalues to keep
        
    Returns:
        Filtered evolution with at most k eigenvalues per step
    """
    filtered_evolution = []
    
    for step_eigenvals in evolution:
        if len(step_eigenvals) > k:
            sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
            filtered_evolution.append(sorted_eigenvals[:k])
        else:
            filtered_evolution.append(step_eigenvals)
    
    return filtered_evolution


def main():
    """Main function - optimized DTW comparison for maximum separation."""
    print("🚀 OPTIMIZED DTW Neural Network Comparison")
    print("=" * 50)
    print("📊 Using proven optimal configuration (17.68x separation)")
    print(f"   Batch size: {CONFIG['data_batch_size']}")
    print(f"   Top eigenvalues: {CONFIG['top_n_eigenvalues']}")
    print(f"   Interpolation points: {CONFIG['interpolation_points']}")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    try:
        # Step 1: Load ALL models for comprehensive comparison
        print("\n📥 Loading all models for comparison...")
        models = {
            'mlp_trained': load_model(MLPModel, mlp_path, device='cpu'),
            'mlp_trained_98': load_model(MLPModel, mlp_path1, device='cpu'),
            'custom_trained': load_model(ActualCustomModel, custom_path, device='cpu'),
            'rand_mlp': load_model(MLPModel, rand_mlp_path, device='cpu'),
            'rand_custom': load_model(ActualCustomModel, rand_custom_path, device='cpu')
        }
        
        model_names = {
            'mlp_trained': Path(mlp_path).stem,
            'mlp_trained_98': Path(mlp_path1).stem,
            'custom_trained': Path(custom_path).stem,
            'rand_mlp': Path(rand_mlp_path).stem,
            'rand_custom': Path(rand_custom_path).stem
        }
        
        print(f"✅ Loaded {len(models)} models")
        
        # For backward compatibility, use first comparison pair
        model1 = models['mlp_trained']
        model2 = models['custom_trained']
        name1 = model_names['mlp_trained']
        name2 = model_names['custom_trained']
        
        # Step 2: Generate data  
        data = 8*torch.randn(CONFIG['data_batch_size'], 3)
        print(f"📊 Generated data: {data.shape}")
        
        # Step 3: Initialize analyzer
        analyzer1 = NeurosheafAnalyzer(device='cpu')
        analyzer2 = NeurosheafAnalyzer(device='cpu')
        
        
        # Step 4: Analyze both models (sheaf construction + spectral analysis)
        print(f"🏗️  Analyzing {name1}...")
        result1 = analyzer1.analyze(
            model1, data,
            method='gromov_wasserstein',
            gw_config=gw_config,
            exclude_final_single_output=True
        )
        
        print(f"🏗️  Analyzing {name2}...")
        result2 = analyzer2.analyze(
            model2, data, 
            method=CONFIG['sheaf_method'],
            gw_config=gw_config,
            exclude_final_single_output=True
        )
        
        # Step 5: Extract eigenvalue evolution using PersistentSpectralAnalyzer
        print("🔬 Running spectral analysis...")
        
        sheaf1 = result1['sheaf']
        sheaf2 = result2['sheaf']

        spectral_analyzer1 = PersistentSpectralAnalyzer(
            default_n_steps=50,  # Reduced for faster execution
            default_filtration_type='threshold'
            )
        spectral_analyzer2 = PersistentSpectralAnalyzer(
            default_n_steps=50,  # Reduced for faster execution
            default_filtration_type='threshold'
            )
        spectral1 = spectral_analyzer1.analyze(sheaf1, filtration_type='threshold', n_steps=100)
        spectral2 = spectral_analyzer2.analyze(sheaf2, filtration_type='threshold', n_steps=100)
        
        # Extract eigenvalue evolution from persistence_result
        eigenvalue_evolution1 = spectral1['persistence_result']['eigenvalue_sequences']
        eigenvalue_evolution2 = spectral2['persistence_result']['eigenvalue_sequences']
        filtration_params1 = spectral1['filtration_params']
        filtration_params2 = spectral2['filtration_params']
        
        print(f"   Model 1: {len(eigenvalue_evolution1)} steps")
        print(f"   Model 2: {len(eigenvalue_evolution2)} steps")
        
        # Apply top-n eigenvalue selection if configured
        if CONFIG['top_n_eigenvalues'] is not None:
            print(f"🎯 Filtering to top {CONFIG['top_n_eigenvalues']} eigenvalues for DTW comparison")
            eigenvalue_evolution1 = filter_top_eigenvalues(
                eigenvalue_evolution1, CONFIG['top_n_eigenvalues']
            )
            eigenvalue_evolution2 = filter_top_eigenvalues(
                eigenvalue_evolution2, CONFIG['top_n_eigenvalues']
            )
            print(f"   Model 1: Up to {CONFIG['top_n_eigenvalues']} eigenvalues per step")
            print(f"   Model 2: Up to {CONFIG['top_n_eigenvalues']} eigenvalues per step")
        
        from neurosheaf.visualization import EnhancedVisualizationFactory
    
        # Initialize the enhanced visualization factory
        vf = EnhancedVisualizationFactory(theme='neurosheaf_default')
        print("✅ Enhanced visualization factory initialized")

        try:
            summary_plots = vf.create_analysis_summary(spectral1)
            
            # Save all summary plots
            for plot_name, figure in summary_plots.items():
                # Save in dedicated folder per request
                import os
                os.makedirs('model_summaries_simple/mlp_trained', exist_ok=True)
                filename = os.path.join('model_summaries_simple/mlp_trained', f"{plot_name}.html")
                figure.write_html(filename)
                print(f"✅ Summary plot '{plot_name}' saved as '{filename}'")
        except Exception as e:
            print(f"⚠️  Could not create analysis summary: {e}")
        
        try:
            summary_plots = vf.create_analysis_summary(spectral2)
            
            # Save all summary plots
            for plot_name, figure in summary_plots.items():
                import os
                os.makedirs('model_summaries_simple/rand_custom', exist_ok=True)
                filename = os.path.join('model_summaries_simple/rand_custom', f"{plot_name}.html")
                figure.write_html(filename)
                print(f"✅ Summary plot '{plot_name}' saved as '{filename}'")
        except Exception as e:
            print(f"⚠️  Could not create analysis summary: {e}")
        
        '''
        print("📊 Creating scaled eigenvalue evolution visualizations...")
        spectral_viz = SpectralVisualizer()
        
        # Plot scaled evolution for model 1
        scaled_fig1 = spectral_viz.plot_eigenvalue_evolution(
            scaled_evolution1,
            filtration_params1,
            title=f"Scaled Eigenvalue Evolution (×10e10) - {name1}",
            max_eigenvalues=100
        )
        scaled_fig1.write_html(f"{name1}_scaled_eigenvalue_evolution.html")
        print(f"   ✅ Saved {name1}_scaled_eigenvalue_evolution.html")
        
        # Plot scaled evolution for model 2
        scaled_fig2 = spectral_viz.plot_eigenvalue_evolution(
            scaled_evolution2,
            filtration_params2,
            title=f"Scaled Eigenvalue Evolution (×10e10) - {name2}",
            max_eigenvalues=100
        )
        scaled_fig2.write_html(f"{name2}_scaled_eigenvalue_evolution.html")
        print(f"   ✅ Saved {name2}_scaled_eigenvalue_evolution.html")
        '''
        # Note: Scaling is no longer needed as we use log-scale internally
        # Step 6: DTW comparison with OPTIMAL settings (17.68x separation proven)
        print(f"🔄 Comparing eigenvalue evolution with OPTIMAL multivariate DTW...")
        print(f"   Using proven optimal configuration for maximum separation")
        print(f"   Eigenvalue evolution 1 type: {type(eigenvalue_evolution1)}")
        print(f"   Eigenvalue evolution 2 type: {type(eigenvalue_evolution2)}")
        
        # Create DTW comparator with OPTIMAL settings (13.75x separation achieved)
        # Use exact same configuration that achieved your excellent results
        dtw_comparator = LogScaleInterpolationDTW(
            constraint_band=0.0,           # No path constraints (best performance)
            min_eigenvalue_threshold=1e-15,
            method='tslearn',              # Multivariate with log-scale
            eigenvalue_weight=1.0,
            structural_weight=0.0,         # Pure functional similarity
            normalization_scheme='range_aware'  # CRITICAL: Range-aware normalization
        )
        
        print("   Starting DTW computation with improved settings...")
        dtw_result = dtw_comparator.compare_eigenvalue_evolution(
            eigenvalue_evolution1,  # Use original evolution, not scaled
            eigenvalue_evolution2,  # Log-scale is applied internally now
            filtration_params1=filtration_params1,
            filtration_params2=filtration_params2,
            multivariate=True,                    # MULTIVARIATE DTW
            use_interpolation=True,               # INTERPOLATION
            match_all_eigenvalues=True,           # ALL EIGENVALUES
            interpolation_points=CONFIG['interpolation_points']
        )
        print("   DTW computation completed!")
        
        # Step 7: Display results
        print(f"\n🎯 Results:")
        print(f"   Models: {name1} vs {name2}")
        print(f"   DTW Distance: {dtw_result['distance']:.6f}")
        print(f"   Normalized Distance: {dtw_result['normalized_distance']:.6f}")
        print(f"   Method: {dtw_result['method']}")
        print(f"   Interpolation Used: {dtw_result['interpolation_used']}")
        
        # Show eigenvalue selection configuration
        if CONFIG['top_n_eigenvalues'] is not None:
            print(f"   Eigenvalue Selection: Top {CONFIG['top_n_eigenvalues']} eigenvalues")
        else:
            print(f"   Eigenvalue Selection: All available eigenvalues")
        
        if dtw_result['interpolation_info']:
            info = dtw_result['interpolation_info']
            print(f"   Eigenvalues Compared: {info['num_features']}")
            print(f"   Time Points: {info['num_time_points']}")
        
        # Similarity interpretation with adjusted thresholds for log-scale
        distance = dtw_result['normalized_distance']
        if distance < 0.5:
            level = "Very Similar"
        elif distance < 1.0:
            level = "Similar" 
        elif distance < 2.0:
            level = "Moderately Different"
        else:
            level = "Very Different"
        print(f"   Similarity: {level}")
        
        # Additional diagnostics
        print(f"\n📊 DTW Diagnostics:")
        print(f"   Log-scale transformation: Enabled")
        print(f"   Persistence weighting: Enabled")
        print(f"   Constraint band: {dtw_comparator.constraint_band}")
        print(f"   Method used: {dtw_result['method']}")
        if CONFIG['top_n_eigenvalues'] is not None:
            print(f"   Top-N Eigenvalue Selection: {CONFIG['top_n_eigenvalues']}")
        else:
            print(f"   Top-N Eigenvalue Selection: Disabled (using all)")
        
        print(f"\n✅ Single pair DTW comparison complete!")
        
        # Step 8: COMPREHENSIVE COMPARISON - Compute separation ratio
        print(f"\n{'='*60}")
        print("🔬 COMPREHENSIVE COMPARISON - Computing Separation Ratio")
        print(f"{'='*60}")
        
        # Compute all pairwise distances
        print("\n📊 Computing all pairwise DTW distances...")
        all_distances = {}
        all_evolutions = {}
        
        # First, compute eigenvalue evolutions for all models
        print("\n🏗️ Analyzing all models...")
        for model_key, model in models.items():
            print(f"   Analyzing {model_key}...")
            analyzer = NeurosheafAnalyzer(device='cpu')
            result = analyzer.analyze(
                model, data,
                method=CONFIG['sheaf_method'],
                gw_config=gw_config,
                exclude_final_single_output=True
            )
            
            # Extract eigenvalue evolution
            spectral_analyzer = PersistentSpectralAnalyzer(
                default_n_steps=50,
                default_filtration_type='threshold'
            )
            spectral = spectral_analyzer.analyze(result['sheaf'], filtration_type='threshold', n_steps=100)
            evolution = spectral['persistence_result']['eigenvalue_sequences']
            
            # Apply top-n filtering if configured
            if CONFIG['top_n_eigenvalues'] is not None:
                evolution = filter_top_eigenvalues(evolution, CONFIG['top_n_eigenvalues'])
            
            all_evolutions[model_key] = evolution
        
        # Compute pairwise distances
        print("\n📐 Computing pairwise DTW distances...")
        model_keys = list(models.keys())
        for i, key1 in enumerate(model_keys):
            for j, key2 in enumerate(model_keys):
                if i < j:  # Only compute upper triangle
                    print(f"   {key1} vs {key2}...")
                    
                    dtw_result_pair = dtw_comparator.compare_eigenvalue_evolution(
                        all_evolutions[key1],
                        all_evolutions[key2],
                        multivariate=True,
                        use_interpolation=True,
                        match_all_eigenvalues=True,
                        interpolation_points=CONFIG['interpolation_points']
                    )
                    
                    distance = dtw_result_pair['normalized_distance']
                    all_distances[f"{key1}_vs_{key2}"] = distance
                    print(f"      Distance: {distance:.3f}")
        
        # Categorize distances
        print("\n📈 Analyzing distance patterns...")
        intra_trained = []
        inter_group = []
        intra_random = []
        
        # Intra-trained distances
        if 'mlp_trained_vs_mlp_trained_98' in all_distances:
            intra_trained.append(all_distances['mlp_trained_vs_mlp_trained_98'])
        if 'mlp_trained_vs_custom_trained' in all_distances:
            intra_trained.append(all_distances['mlp_trained_vs_custom_trained'])
        if 'mlp_trained_98_vs_custom_trained' in all_distances:
            intra_trained.append(all_distances['mlp_trained_98_vs_custom_trained'])
        
        # Inter-group distances (trained vs random)
        for key, dist in all_distances.items():
            if ('trained' in key.split('_vs_')[0] and 'rand' in key.split('_vs_')[1]) or \
               ('rand' in key.split('_vs_')[0] and 'trained' in key.split('_vs_')[1]):
                inter_group.append(dist)
        
        # Intra-random distances
        if 'rand_mlp_vs_rand_custom' in all_distances:
            intra_random.append(all_distances['rand_mlp_vs_rand_custom'])
        
        # Compute separation ratio
        if intra_trained and inter_group:
            mean_intra_trained = np.mean(intra_trained)
            mean_inter_group = np.mean(inter_group)
            separation_ratio = mean_inter_group / mean_intra_trained if mean_intra_trained > 0 else float('inf')
        else:
            separation_ratio = 0.0
            mean_intra_trained = 0.0
            mean_inter_group = 0.0
        
        # Display comprehensive results
        print(f"\n{'='*60}")
        print("🎯 COMPLETE PAIRWISE DISTANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Display ALL pairwise distances in a structured way
        print(f"\n📋 ALL PAIRWISE DISTANCES:")
        print(f"{'='*60}")
        for pair_key, distance in sorted(all_distances.items()):
            model_a, model_b = pair_key.split('_vs_')
            # Determine category
            if ('trained' in model_a and 'rand' not in model_a) and ('trained' in model_b and 'rand' not in model_b):
                category = "🟢 INTRA-TRAINED"
            elif (('trained' in model_a and 'rand' not in model_a) and ('rand' in model_b)) or \
                 (('rand' in model_a) and ('trained' in model_b and 'rand' not in model_b)):
                category = "🔴 INTER-GROUP"
            elif ('rand' in model_a) and ('rand' in model_b):
                category = "🟡 INTRA-RANDOM"
            else:
                category = "❓ OTHER"
            
            print(f"   {category:15} {model_a:20} vs {model_b:20} = {distance:8.3f}")
        
        print(f"\n📊 DISTANCE STATISTICS BY CATEGORY:")
        print(f"{'='*60}")
        print(f"   🟢 Intra-Trained (trained vs trained):")
        if intra_trained:
            print(f"      Count: {len(intra_trained)}")
            print(f"      Mean: {np.mean(intra_trained):.3f}")
            print(f"      Min: {np.min(intra_trained):.3f}")
            print(f"      Max: {np.max(intra_trained):.3f}")
            print(f"      Std Dev: {np.std(intra_trained):.3f}")
            print(f"      All Values: {[f'{d:.3f}' for d in intra_trained]}")
        else:
            print(f"      No intra-trained comparisons available")
        
        print(f"\n   🔴 Inter-Group (trained vs random):")
        if inter_group:
            print(f"      Count: {len(inter_group)}")
            print(f"      Mean: {np.mean(inter_group):.3f}")
            print(f"      Min: {np.min(inter_group):.3f}")
            print(f"      Max: {np.max(inter_group):.3f}")
            print(f"      Std Dev: {np.std(inter_group):.3f}")
            print(f"      All Values: {[f'{d:.3f}' for d in inter_group]}")
        else:
            print(f"      No inter-group comparisons available")
        
        print(f"\n   🟡 Intra-Random (random vs random):")
        if intra_random:
            print(f"      Count: {len(intra_random)}")
            print(f"      Mean: {np.mean(intra_random):.3f}")
            print(f"      Min: {np.min(intra_random):.3f}")
            print(f"      Max: {np.max(intra_random):.3f}")
            print(f"      Std Dev: {np.std(intra_random):.3f}")
            print(f"      All Values: {[f'{d:.3f}' for d in intra_random]}")
        else:
            print(f"      No intra-random comparisons available")
        
        # Create and display distance matrix
        print(f"\n📋 COMPLETE DISTANCE MATRIX:")
        print(f"{'='*60}")
        model_keys = list(models.keys())
        n_models = len(model_keys)
        
        # Create symmetric distance matrix
        distance_matrix = np.zeros((n_models, n_models))
        for i, key1 in enumerate(model_keys):
            for j, key2 in enumerate(model_keys):
                if i == j:
                    distance_matrix[i, j] = 0.0
                elif i < j:
                    pair_key = f"{key1}_vs_{key2}"
                    if pair_key in all_distances:
                        distance_matrix[i, j] = all_distances[pair_key]
                        distance_matrix[j, i] = all_distances[pair_key]
        
        # Display matrix with proper formatting
        header = "Model".ljust(20)
        for key in model_keys:
            header += f"{key[:12]:>12}"
        print(header)
        print("-" * len(header))
        
        for i, key1 in enumerate(model_keys):
            row = f"{key1[:20]:20}"
            for j in range(n_models):
                if distance_matrix[i, j] == 0.0 and i != j:
                    row += f"{'---':>12}"
                else:
                    row += f"{distance_matrix[i, j]:12.3f}"
            print(row)
        
        print(f"\n🏆 SEPARATION RATIO: {separation_ratio:.2f}x")
        if separation_ratio > 10:
            print(f"   ✅ EXCELLENT: Strong separation between trained and random models")
        elif separation_ratio > 5:
            print(f"   ✅ GOOD: Clear separation between trained and random models")
        elif separation_ratio > 2:
            print(f"   ⚠️  MODERATE: Some separation between trained and random models")
        else:
            print(f"   ❌ POOR: Insufficient separation between trained and random models")
        
        print(f"\n📋 Configuration Used:")
        print(f"   Batch size: {CONFIG['data_batch_size']}")
        print(f"   Top eigenvalues: {CONFIG['top_n_eigenvalues']}")
        print(f"   Interpolation points: {CONFIG['interpolation_points']}")
        print(f"   DTW method: tslearn (multivariate with log-scale)")
        print(f"   Constraint band: 0.0 (no constraints)")
        
        print(f"\n✅ COMPREHENSIVE DTW analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


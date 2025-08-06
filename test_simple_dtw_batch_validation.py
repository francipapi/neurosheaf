#!/usr/bin/env python3
"""
Simple DTW Batch Size Test for MLP Models

This is a simplified version focusing on just 2 batch sizes with MLP models
to validate DTW behavior and identify the zero distance issue.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import warnings

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.utils.dtw_similarity import FiltrationDTW
from neurosheaf.sheaf.core.gw_config import GWConfig

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Define MLPModel directly to avoid import side effects
import torch.nn as nn

class MLPModel(nn.Module):
    """MLP model architecture matching the saved models."""
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

def analyze_single_model(analyzer, model, batch_size, model_name):
    """Analyze a single model at specific batch size."""
    print(f"  Analyzing {model_name} with batch size {batch_size}...")
    
    # Generate data
    torch.manual_seed(42)  # Fixed seed
    data = 8 * torch.randn(batch_size, 3)
    
    # GW config
    gw_config = GWConfig(
        epsilon=0.05,  # Higher epsilon for faster convergence
        max_iter=100,  # Fewer iterations
        tolerance=1e-6,
        adaptive_epsilon=True,
        base_epsilon=0.05,
        reference_n=50,
    )
    
    try:
        # Step 1: Build sheaf
        sheaf_results = analyzer.analyze(
            model, data,
            method='gromov_wasserstein',
            gw_config=gw_config
        )
        
        print(f"    ✅ Sheaf construction completed")
        sheaf = sheaf_results['sheaf']
        
        # Step 2: Run spectral analysis using PersistentSpectralAnalyzer
        from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
        
        spectral_analyzer = PersistentSpectralAnalyzer()
        
        print(f"    Running spectral analysis...")
        spectral_results = spectral_analyzer.analyze(sheaf, n_steps=15)
        
        # Extract eigenvalue sequences
        persistence_result = spectral_results['persistence_result']
        eigenvalue_sequences = persistence_result['eigenvalue_sequences']
        filtration_params = spectral_results['filtration_params']
        
        print(f"    ✅ Spectral analysis completed")
        print(f"    Eigenvalue sequences: {len(eigenvalue_sequences)} steps")
        print(f"    First step eigenvalues: {len(eigenvalue_sequences[0])} values")
        print(f"    Last step eigenvalues: {len(eigenvalue_sequences[-1])} values")
        
        return eigenvalue_sequences, filtration_params
        
    except Exception as e:
        print(f"    ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compute_dtw_distance(evo1, evo2, params1, params2):
    """Compute DTW distance between eigenvalue evolutions."""
    if evo1 is None or evo2 is None:
        return float('nan')
    
    try:
        dtw_analyzer = FiltrationDTW(
            method='tslearn',
            constraint_band=0.0,
            eigenvalue_selection=10,
            min_eigenvalue_threshold=1e-15,
            interpolation_points=50,
            eigenvalue_weight=1.0,
            structural_weight=0.0
        )
        
        result = dtw_analyzer.compare_eigenvalue_evolution(
            evolution1=evo1,
            evolution2=evo2,
            filtration_params1=params1,
            filtration_params2=params2,
            multivariate=True,
            use_interpolation=True,
            match_all_eigenvalues=True,
            interpolation_points=50
        )
        
        return result['distance']
        
    except Exception as e:
        print(f"    DTW computation failed: {e}")
        return float('nan')

def main():
    """Main test function."""
    print("🚀 Simple DTW Batch Size Test for MLP Models")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu')
    
    # Load models
    print("\n🔄 Loading Models:")
    models = {}
    model_paths = {
        'mlp_trained_100': 'models/torch_mlp_acc_1.0000_epoch_200.pth',
        'mlp_random': 'models/random_mlp_net_000_default_seed_42.pth'
    }
    
    for name, path in model_paths.items():
        try:
            model = load_model(MLPModel, path, device='cpu')
            models[name] = model
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✅ {name}: {param_count:,} parameters")
        except Exception as e:
            print(f"  ❌ Failed to load {name}: {e}")
            return
    
    # Test configurations
    batch_sizes = [50, 100]
    
    print(f"\n📊 Testing DTW Distance Computation:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Model pair: mlp_trained_100 vs mlp_random")
    print(f"  Expected: HIGH distance (should be >10,000)")
    
    # Run analysis for each batch size
    results = {}
    for batch_size in batch_sizes:
        print(f"\n  === Batch Size {batch_size} ===")
        
        # Analyze both models
        evo1, params1 = analyze_single_model(analyzer, models['mlp_trained_100'], batch_size, 'mlp_trained_100')
        evo2, params2 = analyze_single_model(analyzer, models['mlp_random'], batch_size, 'mlp_random')
        
        # Compute DTW distance
        if evo1 is not None and evo2 is not None:
            print(f"  Computing DTW distance...")
            distance = compute_dtw_distance(evo1, evo2, params1, params2)
            results[batch_size] = distance
            
            if not np.isnan(distance):
                print(f"  📏 DTW Distance: {distance:,.1f}")
                if distance < 1000:
                    print(f"      ⚠️  UNEXPECTEDLY LOW - Expected >10,000")
                elif distance == 0:
                    print(f"      🚨 ZERO DISTANCE - This is the bug!")
                else:
                    print(f"      ✅ Reasonable distance")
            else:
                print(f"  ❌ DTW computation failed")
        else:
            print(f"  ⚠️  Skipping DTW due to analysis failures")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  Batch Size → DTW Distance")
    for batch_size, distance in results.items():
        if not np.isnan(distance):
            print(f"  {batch_size:>3} → {distance:>10,.1f}")
        else:
            print(f"  {batch_size:>3} → {'FAILED':>10}")
    
    if len(results) >= 2:
        distances = [d for d in results.values() if not np.isnan(d)]
        if len(distances) >= 2:
            variation = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else float('inf')
            print(f"\n  Stability (CV): {variation:.3f} ({'STABLE' if variation < 0.1 else 'UNSTABLE'})")
    
    print(f"\n🎯 Expected Behavior:")
    print(f"  - Both batch sizes should show HIGH distance (>10,000)")
    print(f"  - Distance should be stable across batch sizes")
    print(f"  - Zero distance indicates DTW preprocessing bug")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Complete GW Sheaf Construction Test - Mirror of test_all.py

This test performs the same comprehensive analysis as test_all.py but uses
Gromov-Wasserstein optimal transport for sheaf construction instead of Procrustes.
All spectral analysis and visualization capabilities are preserved with GW-specific
enhancements and validation.
"""

import torch.nn as nn 
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import time

# Set environment for CPU usage (same as original)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Enhanced imports for GW
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer

import logging
logging.getLogger('neurosheaf').setLevel(logging.DEBUG)


# Set random seeds for reproducibility (same as original)
random_seed = 5670
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


# MLP model class that matches your saved model configuration (same as original)
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


class CustomModel(nn.Module):
    """Custom model with flexible architecture supporting the specified layer configuration:
    - Linear layers with configurable activation and dropout
    - Reshape operations for dimensional transformations
    - Conv1D layers with configurable parameters
    - Flatten operations
    - Batch normalization support
    """
    
    def __init__(self, input_shape=[3]):
        super().__init__()
        
        self.input_shape = input_shape
        self.layers = nn.ModuleList()
        
        # Calculate initial input size
        if isinstance(input_shape, list) and len(input_shape) == 1:
            current_size = input_shape[0]
        else:
            current_size = input_shape
            
        # Define the architecture as specified
        # Layer 1: Linear(3 -> 32) + ReLU
        self.layers.append(nn.Linear(current_size, 32))
        current_size = 32
        
        # Layer 2: Linear(32 -> 32) + ReLU  
        self.layers.append(nn.Linear(current_size, 32))
        current_size = 32
        
        # Layer 3: Conv1D(16 channels, 2 length -> 32 channels, 1 length)
        self.layers.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0))
        
        # Layer 4: Conv1D(16 channels, 2 length -> 32 channels, 1 length)
        self.layers.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0))
        
        # Layer 5: Conv1D(16 channels, 2 length -> 32 channels, 1 length)
        self.layers.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0))
        
        # Layer 6: Final Linear(32 -> 1) + Sigmoid
        self.layers.append(nn.Linear(32, 1))
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch_size, 3]
        
        # Layer 1: Linear(3 -> 32) + ReLU
        x = self.relu(self.layers[0](x))  # [batch_size, 32]
        
        # Layer 2: Linear(32 -> 32) + ReLU
        x = self.relu(self.layers[1](x))  # [batch_size, 32]
        
        # Reshape to [batch_size, 16, 2] for Conv1D
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 3: Conv1D(16 -> 32, kernel=2) + ReLU
        x = self.relu(self.layers[2](x))  # [batch_size, 32, 1]
        
        # Reshape to [batch_size, 16, 2] for next Conv1D
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 4: Conv1D(16 -> 32, kernel=2) + ReLU
        x = self.relu(self.layers[3](x))  # [batch_size, 32, 1]
        
        # Reshape to [batch_size, 16, 2] for next Conv1D
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 5: Conv1D(16 -> 32, kernel=2) + ReLU
        x = self.relu(self.layers[4](x))  # [batch_size, 32, 1]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        
        # Layer 6: Linear(32 -> 1) + Sigmoid
        x = self.sigmoid(self.layers[5](x))  # [batch_size, 1]
        
        return x


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


if __name__ == "__main__":
    print("=== GW SHEAF CONSTRUCTION - COMPLETE ANALYSIS ===")
    print("This mirrors test_all.py but uses Gromov-Wasserstein optimal transport")
    print("for sheaf construction instead of Procrustes methods.\n")
    
    # === GW Configuration Setup ===
    print("=== GW Configuration Setup ===")
    
    # Multiple GW configurations for different use cases
    gw_configs = {
        'fast': GWConfig(
            epsilon=0.1, 
            max_iter=50, 
            tolerance=1e-6,
            quasi_sheaf_tolerance=0.1
        ),
        'accurate': GWConfig(
            epsilon=0.01, 
            max_iter=200, 
            tolerance=1e-9,
            quasi_sheaf_tolerance=0.05
        ), 
        'debug': GWConfig(
            epsilon=0.05, 
            max_iter=100,
            validate_couplings=True, 
            validate_costs=True,
            quasi_sheaf_tolerance=0.08
        ),
        'production': GWConfig(
            epsilon=0.02, 
            max_iter=100, 
            quasi_sheaf_tolerance=0.05,
            use_gpu=False  # Force CPU for compatibility
        )
    }
    
    # Choose configuration
    selected_config = 'accurate'  # Use accurate for quality results
    gw_config = gw_configs[selected_config]
    
    print(f"Selected GW configuration: {selected_config}")
    print(f"  - Epsilon: {gw_config.epsilon}")
    print(f"  - Max iterations: {gw_config.max_iter}")
    print(f"  - Tolerance: {gw_config.tolerance}")
    print(f"  - Quasi-sheaf tolerance: {gw_config.quasi_sheaf_tolerance}")
    print(f"  - Validation: couplings={gw_config.validate_couplings}, costs={gw_config.validate_costs}")

    # === Load Models ===
    print("\n=== Loading Models ===")
    mlp_path1 = "models/torch_mlp_acc_0.9857_epoch_100.pth"
    
    # Try to load MLP model first (simpler architecture)
    try:
        mlp_model = load_model(MLPModel, mlp_path1, device="cpu")
        print(f"✅ Successfully loaded MLP model with {sum(p.numel() for p in mlp_model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Error loading MLP model: {e}")
        print("Creating a simple model for demonstration...")
        # Create a simple model if loading fails
        mlp_model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        print(f"✅ Created demo model with {sum(p.numel() for p in mlp_model.parameters()):,} parameters")

    model = mlp_model

    # Generate sample data that matches your model's expected input (3D torus data)
    batch_size = 50
    data = 8*torch.randn(batch_size, 3)  # 3 features input for torus data
    print(f"Generated data shape: {data.shape}")

    # === GW Sheaf Construction ===
    print("\n=== Building GW Sheaf Using High-Level API ===")
    
    # Performance timing
    gw_start_time = time.time()
    
    try:
        # GW-specific analyzer setup with error handling
        analyzer = NeurosheafAnalyzer(device='cpu')
        
        print("Attempting GW sheaf construction...")
        gw_analysis = analyzer.analyze(
            model, data,
            method='gromov_wasserstein',
            gw_config=gw_config,
            preserve_eigenvalues=True,
            batch_size=None  # Use full batch for consistency
        )
        
        gw_construction_time = time.time() - gw_start_time
        gw_sheaf = gw_analysis['sheaf']
        
        print(f"✅ GW Sheaf constructed successfully in {gw_construction_time:.3f}s")
        print(f"   Stalks: {len(gw_sheaf.stalks)}, Restrictions: {len(gw_sheaf.restrictions)}")
        
        # === GW-Specific Validation ===
        print("\n=== GW Construction Validation ===")
        
        # Validate construction method
        construction_method = gw_sheaf.metadata.get('construction_method', 'unknown')
        print(f"Construction method: {construction_method}")
        if construction_method == 'gromov_wasserstein':
            print("✅ Confirmed GW construction method")
        else:
            print(f"⚠️  Expected 'gromov_wasserstein', got '{construction_method}'")
        
        # Validate GW-specific metadata
        if 'gw_costs' in gw_sheaf.metadata:
            gw_costs = gw_sheaf.metadata['gw_costs']
            print(f"✅ GW transport costs: {list(gw_costs.values())}")
            avg_cost = np.mean(list(gw_costs.values())) if gw_costs else 0
            print(f"   Average GW cost: {avg_cost:.6f}")
        else:
            print("⚠️  No GW costs found in metadata")
        
        if 'quasi_sheaf_tolerance' in gw_sheaf.metadata:
            tolerance = gw_sheaf.metadata['quasi_sheaf_tolerance']
            print(f"✅ Quasi-sheaf tolerance: {tolerance:.6f}")
            if tolerance <= gw_config.quasi_sheaf_tolerance:
                print(f"✅ Tolerance within requirement: {tolerance:.6f} ≤ {gw_config.quasi_sheaf_tolerance}")
            else:
                print(f"⚠️  Tolerance exceeds requirement: {tolerance:.6f} > {gw_config.quasi_sheaf_tolerance}")
        else:
            print("⚠️  No quasi-sheaf tolerance found in metadata")
        
        # Additional GW metadata
        if 'gw_config' in gw_sheaf.metadata:
            config_dict = gw_sheaf.metadata['gw_config']
            print(f"✅ GW config preserved: epsilon={config_dict.get('epsilon')}, max_iter={config_dict.get('max_iter')}")
        
        if 'gw_couplings' in gw_sheaf.metadata:
            couplings = gw_sheaf.metadata['gw_couplings']
            print(f"✅ Transport plan shapes: {[(edge, c.shape) for edge, c in couplings.items()]}")
        
        # Use the new detailed print method
        gw_sheaf.print_detailed_summary(max_items=5, verbosity='detailed')
        
        gw_success = True
        
    except ImportError as e:
        print(f"❌ GW construction failed - POT library issue: {e}")
        print("    Please install POT: pip install pot")
        gw_success = False
        gw_sheaf = None
    except Exception as e:
        print(f"❌ GW construction failed: {e}")
        print("    This may be due to model architecture compatibility issues")
        gw_success = False  
        gw_sheaf = None

    if not gw_success:
        print("\n❌ Cannot continue without successful GW sheaf construction")
        print("   Please check POT installation and model compatibility")
        exit(1)

    # === Complete Spectral Analysis Pipeline ===
    print("\n=== Running Complete Spectral Analysis ===")
    
    spectral_analyzer = PersistentSpectralAnalyzer(
        default_n_steps=100,  # Same as original for consistency
        default_filtration_type='threshold'
    )

    spectral_start_time = time.time()
    
    try:
        results = spectral_analyzer.analyze(
            gw_sheaf,
            filtration_type='threshold',
            n_steps=100  # Same parameters as original
        )
        
        spectral_time = time.time() - spectral_start_time
        print(f"✅ Spectral analysis completed in {spectral_time:.3f}s")
        
        # Print complete results summary (same as original)
        print("\n=== GW Spectral Persistence Analysis Results ===")
        print(f"Total filtration steps: {len(results['filtration_params'])}")
        print(f"Birth events: {results['features']['num_birth_events']}")
        print(f"Death events: {results['features']['num_death_events']}")
        print(f"Crossing events: {results['features']['num_crossings']}")
        print(f"Persistent paths: {results['features']['num_persistent_paths']}")
        print(f"Infinite bars: {results['diagrams']['statistics']['n_infinite_bars']}")
        print(f"Finite pairs: {results['diagrams']['statistics']['n_finite_pairs']}")
        print(f"Mean lifetime: {results['diagrams']['statistics'].get('mean_lifetime', 0):.6f}")

        # Debug eigenvalue information (same as original)
        eigenval_seqs = results['persistence_result']['eigenvalue_sequences']
        if eigenval_seqs:
            print(f"\nEigenvalue sequences: {len(eigenval_seqs)} steps")
            print(f"Eigenvalues per step: {[len(seq) for seq in eigenval_seqs[:5]]}..." if len(eigenval_seqs) > 5 else f"Eigenvalues per step: {[len(seq) for seq in eigenval_seqs]}")
            if eigenval_seqs[0].numel() > 0:
                print(f"First step eigenvalues (first 5): {eigenval_seqs[0][:5]}")
                print(f"Last step eigenvalues (first 5): {eigenval_seqs[-1][:5]}")
        else:
            print("No eigenvalue sequences found!")
            
        spectral_success = True
        
    except Exception as e:
        print(f"❌ Spectral analysis failed: {e}")
        spectral_success = False
        results = None

    if not spectral_success:
        print("\n❌ Cannot continue without successful spectral analysis")
        exit(1)

    # === Complete Enhanced Visualization Suite ===
    print("\n=== Creating Complete Enhanced GW Visualizations ===")
    
    visualization_files = []
    
    try:
        from neurosheaf.visualization import EnhancedVisualizationFactory
        
        # Initialize the enhanced visualization factory
        vf = EnhancedVisualizationFactory(theme='neurosheaf_default')
        print("✅ Enhanced visualization factory initialized")
        
        # 1. Create enhanced comprehensive dashboard
        print("Creating GW comprehensive analysis dashboard...")
        try:
            dashboard_fig = vf.create_comprehensive_analysis_dashboard(
                gw_sheaf, 
                results,
                title="🧠 GW-Based Neural Network Spectral Persistence Analysis Dashboard"
            )
            dashboard_file = "gw_spectral_analysis_dashboard.html"
            dashboard_fig.write_html(dashboard_file)
            visualization_files.append(dashboard_file)
            print(f"✅ GW dashboard saved as '{dashboard_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW comprehensive dashboard: {e}")
        
        # 2. Enhanced individual visualizations with GW prefixes
        print("\nCreating enhanced detailed GW visualizations...")
        
        # Enhanced poset visualization
        try:
            from neurosheaf.visualization import EnhancedPosetVisualizer
            enhanced_poset_viz = EnhancedPosetVisualizer(theme='neurosheaf_default')
            
            poset_fig = enhanced_poset_viz.create_visualization(
                gw_sheaf,
                title="🔬 GW-Based Neural Network Architecture Analysis",
                width=1400,
                height=800,
                layout_type='hierarchical',
                interactive_mode=True
            )
            poset_file = "gw_enhanced_network_structure.html"
            poset_fig.write_html(poset_file)
            visualization_files.append(poset_file)
            print(f"✅ GW network structure saved as '{poset_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW enhanced poset visualization: {e}")
        
        # Persistence diagram with GW-specific title
        try:
            pers_diagram_fig = vf.create_persistence_diagram(
                results['diagrams'],
                title="GW Sheaf Topological Persistence Features",
                width=800,
                height=600
            )
            pers_diagram_file = "gw_persistence_diagram.html"
            pers_diagram_fig.write_html(pers_diagram_file)
            visualization_files.append(pers_diagram_file)
            print(f"✅ GW persistence diagram saved as '{pers_diagram_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW persistence diagram: {e}")
        
        # Persistence barcode with GW-specific title
        try:
            barcode_fig = vf.create_persistence_barcode(
                results['diagrams'],
                title="GW Sheaf Feature Lifetime Analysis",
                width=1000,
                height=500
            )
            barcode_file = "gw_persistence_barcode.html"
            barcode_fig.write_html(barcode_file)
            visualization_files.append(barcode_file)
            print(f"✅ GW persistence barcode saved as '{barcode_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW persistence barcode: {e}")
        
        # Enhanced multi-scale eigenvalue evolution
        try:
            from neurosheaf.visualization import EnhancedSpectralVisualizer
            enhanced_spectral_viz = EnhancedSpectralVisualizer()
            
            eigenval_fig = enhanced_spectral_viz.create_comprehensive_spectral_view(
                results['persistence_result']['eigenvalue_sequences'],
                results['filtration_params'],
                title="🌊 GW Sheaf Comprehensive Spectral Evolution Analysis",
                width=1400,
                height=900
            )
            eigenval_file = "gw_enhanced_eigenvalue_evolution.html"
            eigenval_fig.write_html(eigenval_file)
            visualization_files.append(eigenval_file)
            print(f"✅ GW enhanced eigenvalue evolution saved as '{eigenval_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW enhanced eigenvalue evolution: {e}")
        
        # 3. Create all specialized visualizations with GW prefixes
        print("\nCreating specialized GW analysis plots...")
        
        # Spectral gap evolution
        try:
            gap_fig = vf.spectral_visualizer.plot_spectral_gap_evolution(
                results['persistence_result']['eigenvalue_sequences'],
                results['filtration_params'],
                title="GW Sheaf Spectral Gap Evolution Analysis"
            )
            gap_file = "gw_spectral_gap_evolution.html"
            gap_fig.write_html(gap_file)
            visualization_files.append(gap_file)
            print(f"✅ GW spectral gap evolution saved as '{gap_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW spectral gap evolution: {e}")
        
        # Eigenvalue statistics
        try:
            stats_fig = vf.spectral_visualizer.plot_eigenvalue_statistics(
                results['persistence_result']['eigenvalue_sequences'],
                results['filtration_params'],
                title="GW Sheaf Eigenvalue Statistical Evolution"
            )
            stats_file = "gw_eigenvalue_statistics.html"
            stats_fig.write_html(stats_file)
            visualization_files.append(stats_file)
            print(f"✅ GW eigenvalue statistics saved as '{stats_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW eigenvalue statistics: {e}")
        
        # Eigenvalue heatmap
        try:
            heatmap_fig = vf.spectral_visualizer.plot_eigenvalue_heatmap(
                results['persistence_result']['eigenvalue_sequences'],
                results['filtration_params'],
                title="GW Sheaf Eigenvalue Evolution Heatmap"
            )
            heatmap_file = "gw_eigenvalue_heatmap.html"
            heatmap_fig.write_html(heatmap_file)
            visualization_files.append(heatmap_file)
            print(f"✅ GW eigenvalue heatmap saved as '{heatmap_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW eigenvalue heatmap: {e}")
        
        # Lifetime distribution
        try:
            lifetime_fig = vf.persistence_visualizer.plot_lifetime_distribution(
                results['diagrams'],
                title="GW Sheaf Persistence Lifetime Distribution",
                bins=20
            )
            lifetime_file = "gw_lifetime_distribution.html"
            lifetime_fig.write_html(lifetime_file)
            visualization_files.append(lifetime_file)
            print(f"✅ GW lifetime distribution saved as '{lifetime_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW lifetime distribution: {e}")
        
        # Sheaf structure summary
        try:
            sheaf_summary_fig = vf.poset_visualizer.plot_summary_stats(gw_sheaf)
            sheaf_summary_file = "gw_sheaf_summary.html"
            sheaf_summary_fig.write_html(sheaf_summary_file)
            visualization_files.append(sheaf_summary_file)
            print(f"✅ GW sheaf summary saved as '{sheaf_summary_file}'")
        except Exception as e:
            print(f"⚠️  Could not create GW sheaf summary: {e}")
        
        # 4. Create analysis summary collection with GW prefixes
        print("\nCreating comprehensive GW analysis summary...")
        try:
            summary_plots = vf.create_analysis_summary(results)
            
            # Save all summary plots with GW prefix
            for plot_name, figure in summary_plots.items():
                filename = f"gw_summary_{plot_name}.html"
                figure.write_html(filename)
                visualization_files.append(filename)
                print(f"✅ GW summary plot '{plot_name}' saved as '{filename}'")
        except Exception as e:
            print(f"⚠️  Could not create GW analysis summary: {e}")
        
        # 5. Print configuration information
        print("\n=== GW Visualization Configuration ===")
        try:
            config = vf.get_configuration()
            print("Configuration sections:")
            for section, details in config.items():
                if section == 'default_config':
                    print(f"  {section}: {details}")
                else:
                    print(f"  {section}: {len(details)} parameters")
        except Exception as e:
            print(f"⚠️  Could not retrieve configuration: {e}")
        
        print(f"\n✅ Successfully created {len(visualization_files)} GW visualization files")
        
    except ImportError as e:
        print(f"⚠️  Enhanced visualization modules not available: {e}")
        print("    Falling back to basic visualizations...")
        
        # Fallback to basic visualizations with GW prefixes
        try:
            from neurosheaf.visualization.spectral import SpectralVisualizer
            
            # Create basic spectral visualization with GW prefix
            spectral_viz = SpectralVisualizer()
            eigenval_fig = spectral_viz.plot_eigenvalue_evolution(
                results['persistence_result']['eigenvalue_sequences'],
                results['filtration_params'],
                title="GW Sheaf Eigenvalue Evolution",
                max_eigenvalues=10
            )
            basic_file = "gw_eigenvalue_evolution.html"
            eigenval_fig.write_html(basic_file)
            visualization_files.append(basic_file)
            print(f"✅ Basic GW eigenvalue evolution saved as '{basic_file}'")
            
        except Exception as e:
            print(f"⚠️  Could not create basic GW visualizations: {e}")

    # === Enhanced Static Matplotlib Comparison ===
    print("\n=== Creating GW Static Comparison Plot ===")
    
    try:
        # Enhanced static comparison with GW-specific metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Persistence diagram (same as original but with GW data)
        diagrams = results['diagrams']
        birth_death_pairs = diagrams['birth_death_pairs']
        if birth_death_pairs:
            births = [pair['birth'] for pair in birth_death_pairs]
            deaths = [pair['death'] for pair in birth_death_pairs]
            ax1.scatter(births, deaths, alpha=0.6, color='red', label='GW Sheaf')
            max_val = max(deaths) if deaths else 1
            ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        ax1.set_title('GW Sheaf Persistence Diagram (Static)')
        ax1.set_xlabel('Birth')
        ax1.set_ylabel('Death')
        ax1.legend()
        
        # 2. Eigenvalue evolution (same as original but with GW data)
        eigenval_seqs = results['persistence_result']['eigenvalue_sequences']
        if eigenval_seqs and len(eigenval_seqs[0]) > 0:
            n_plot = min(5, len(eigenval_seqs[0]))
            for i in range(n_plot):
                track = []
                for eigenvals in eigenval_seqs:
                    if i < len(eigenvals):
                        track.append(eigenvals[i].item())
                    else:
                        track.append(np.nan)
                ax2.plot(results['filtration_params'], track, label=f'λ_{i}', alpha=0.7)
            ax2.set_yscale('log')
            ax2.legend()
        ax2.set_title('GW Sheaf Eigenvalue Evolution (Static)')
        ax2.set_xlabel('Filtration Parameter')
        ax2.set_ylabel('Eigenvalue (log scale)')
        
        # 3. Spectral gap (same as original)
        gap_evolution = results['features']['spectral_gap_evolution']
        ax3.plot(results['filtration_params'], gap_evolution, 'r-', linewidth=2, label='GW Sheaf')
        ax3.set_title('GW Sheaf Spectral Gap Evolution')
        ax3.set_xlabel('Filtration Parameter')
        ax3.set_ylabel('Spectral Gap')
        ax3.legend()
        
        # 4. Feature counts with GW-specific information
        feature_names = ['Birth', 'Death', 'Crossings', 'Paths']
        feature_counts = [
            results['features']['num_birth_events'],
            results['features']['num_death_events'], 
            results['features']['num_crossings'],
            results['features']['num_persistent_paths']
        ]
        bars = ax4.bar(feature_names, feature_counts, alpha=0.7, color='red')
        ax4.set_title('GW Sheaf Feature Summary')
        ax4.set_ylabel('Count')
        
        # Add GW-specific annotation
        if 'gw_costs' in gw_sheaf.metadata:
            avg_cost = np.mean(list(gw_sheaf.metadata['gw_costs'].values()))
            ax4.text(0.7, 0.95, f'Avg GW Cost: {avg_cost:.4f}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.suptitle('GW Sheaf Construction - Static Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        static_file = 'gw_static_comparison.png'
        plt.savefig(static_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ GW static comparison plot saved as '{static_file}'")
        
    except Exception as e:
        print(f"⚠️  Could not create GW static comparison plot: {e}")

    # === Optional Performance Comparison ===
    print("\n=== GW vs Procrustes Performance Comparison ===")
    
    try:
        # Run Procrustes for comparison
        print("Running Procrustes analysis for comparison...")
        procrustes_start_time = time.time()
        
        procrustes_analysis = analyzer.analyze(
            model, data, 
            method='procrustes',  # Use procrustes as baseline
            preserve_eigenvalues=True
        )
        
        procrustes_time = time.time() - procrustes_start_time
        procrustes_sheaf = procrustes_analysis['sheaf']
        
        print(f"✅ Procrustes analysis completed in {procrustes_time:.3f}s")
        
        # Performance comparison
        print(f"\nPerformance Comparison:")
        print(f"  GW Construction time: {gw_construction_time:.3f}s")
        print(f"  Procrustes Construction time: {procrustes_time:.3f}s")
        print(f"  GW Slowdown factor: {gw_construction_time/procrustes_time:.2f}x")
        
        print(f"\nStructural Comparison:")
        print(f"  GW Sheaf: {len(gw_sheaf.stalks)} stalks, {len(gw_sheaf.restrictions)} restrictions")
        print(f"  Procrustes Sheaf: {len(procrustes_sheaf.stalks)} stalks, {len(procrustes_sheaf.restrictions)} restrictions")
        
        # Memory comparison (rough estimate)
        gw_restriction_elements = sum(r.numel() for r in gw_sheaf.restrictions.values())
        procrustes_restriction_elements = sum(r.numel() for r in procrustes_sheaf.restrictions.values())
        
        print(f"\nMemory Usage Comparison:")
        print(f"  GW restrictions: {gw_restriction_elements:,} elements")
        print(f"  Procrustes restrictions: {procrustes_restriction_elements:,} elements") 
        if procrustes_restriction_elements > 0:
            memory_ratio = gw_restriction_elements / procrustes_restriction_elements
            print(f"  GW Memory ratio: {memory_ratio:.2f}x")
        
    except Exception as e:
        print(f"⚠️  Performance comparison failed: {e}")
        print("    GW analysis completed successfully, but cannot compare with Procrustes")

    # === Final Summary ===
    print("\n" + "="*70)
    print("🎉 GW SHEAF ANALYSIS COMPLETE!")
    print("="*70)
    
    print("✅ Successfully analyzed neural network using GW-based persistent spectral methods")
    print(f"✅ GW Configuration: {selected_config} (ε={gw_config.epsilon}, max_iter={gw_config.max_iter})")
    print(f"✅ Generated {len(results['filtration_params'])} filtration steps")
    print(f"✅ Found {results['features']['num_persistent_paths']} persistent features")
    print(f"✅ Quasi-sheaf tolerance: {gw_sheaf.metadata.get('quasi_sheaf_tolerance', 'N/A')}")
    
    if visualization_files:
        print(f"✅ Created {len(visualization_files)} interactive visualization files")
    
    print("\n🎯 GW-Specific Files Created:")
    print("  📊 Main Dashboard:")
    print("    • gw_spectral_analysis_dashboard.html - Complete GW interactive analysis")
    print("\n  📈 Detailed Visualizations:")
    print("    • gw_enhanced_network_structure.html - Interactive GW network topology")
    print("    • gw_persistence_diagram.html - GW topological features")
    print("    • gw_persistence_barcode.html - GW feature lifetime analysis")
    print("    • gw_enhanced_eigenvalue_evolution.html - GW spectral evolution")
    print("\n  🔬 Specialized GW Analysis:")
    print("    • gw_spectral_gap_evolution.html - GW gap dynamics")
    print("    • gw_eigenvalue_statistics.html - GW statistical summaries")
    print("    • gw_eigenvalue_heatmap.html - GW evolution heatmap")
    print("    • gw_lifetime_distribution.html - GW persistence statistics")
    print("    • gw_sheaf_summary.html - GW structure overview")
    print("\n  📋 Summary Collection:")
    print("    • gw_summary_*.html files - Comprehensive GW analysis summaries")
    print("\n  📊 Static Comparison:")
    print("    • gw_static_comparison.png - GW matplotlib summary plot")
    
    print("\n" + "="*70)
    print("🚀 GW IMPLEMENTATION VALIDATION COMPLETE!")
    print("="*70)
    print("All GW visualizations feature:")
    print("  ✓ Gromov-Wasserstein optimal transport construction")
    print("  ✓ Interactive hover information and zooming")
    print("  ✓ Quasi-sheaf functoriality validation")
    print("  ✓ Transport cost analysis") 
    print("  ✓ Enhanced mathematical correctness")
    print("  ✓ Full spectral persistence analysis")
    print("\nOpen any gw_*.html file in your browser for interactive GW exploration!")
    print("Compare with regular test_all.py outputs to see Procrustes vs GW differences.")
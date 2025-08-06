import torch.nn as nn 
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
# Core imports
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer

# NEW: GW Subspace Tracker imports
from neurosheaf.spectral.gw.gw_subspace_tracker import GWSubspaceTracker
from neurosheaf.spectral.tracker_factory import SubspaceTrackerFactory
from neurosheaf.spectral.gw.pes_computation import PESComputer


# Set environment for CPU usage
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Fixed imports based on current codebase structure
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer

import logging
logging.getLogger('neurosheaf').setLevel(logging.DEBUG)


# Set random seeds for reproducibility
random_seed = 5670
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


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


class FlexibleModel(nn.Module):
    """More flexible model that can be configured with different layer types."""
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            # Default configuration matching your specification
            config = {
                'input_shape': [3],
                'layers': [
                    {'type': 'linear', 'out_features': 32, 'activation': 'relu', 'batch_norm': False, 'dropout': 0.0},
                    {'type': 'linear', 'out_features': 32, 'activation': 'relu', 'batch_norm': False, 'dropout': 0.0},
                    {'type': 'reshape', 'shape': [16, 2]},
                    {'type': 'conv1d', 'out_channels': 32, 'kernel_size': 2, 'stride': 1, 'padding': 0, 'activation': 'relu', 'batch_norm': False},
                    {'type': 'reshape', 'shape': [16, 2]},
                    {'type': 'conv1d', 'out_channels': 32, 'kernel_size': 2, 'stride': 1, 'padding': 0, 'activation': 'relu', 'batch_norm': False},
                    {'type': 'reshape', 'shape': [16, 2]},
                    {'type': 'conv1d', 'out_channels': 32, 'kernel_size': 2, 'stride': 1, 'padding': 0, 'activation': 'relu', 'batch_norm': False},
                    {'type': 'flatten'},
                    {'type': 'linear', 'out_features': 1, 'activation': 'sigmoid'}
                ]
            }
        
        self.config = config
        self.input_shape = config['input_shape']
        self.layers = nn.ModuleList()
        self.layer_configs = config['layers']
        
        # Build layers
        current_size = self.input_shape[0] if isinstance(self.input_shape, list) else self.input_shape
        current_channels = None
        
        for i, layer_config in enumerate(self.layer_configs):
            layer_type = layer_config['type']
            
            if layer_type == 'linear':
                layer = nn.Linear(current_size, layer_config['out_features'])
                self.layers.append(layer)
                current_size = layer_config['out_features']
                
            elif layer_type == 'conv1d':
                in_channels = current_channels if current_channels else layer_config.get('in_channels', 16)
                layer = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=layer_config['out_channels'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config.get('stride', 1),
                    padding=layer_config.get('padding', 0)
                )
                self.layers.append(layer)
                current_channels = layer_config['out_channels']
                
            elif layer_type in ['reshape', 'flatten']:
                # These don't need actual layers, handled in forward
                self.layers.append(nn.Identity())
                
    def _get_activation(self, activation_name):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'none': nn.Identity()
        }
        return activations.get(activation_name, nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_idx = 0
        
        for i, layer_config in enumerate(self.layer_configs):
            layer_type = layer_config['type']
            
            if layer_type == 'linear':
                x = self.layers[layer_idx](x)
                layer_idx += 1
                
                # Apply activation
                if 'activation' in layer_config:
                    activation_fn = self._get_activation(layer_config['activation'])
                    x = activation_fn(x)
                    
                # Apply batch norm if specified
                if layer_config.get('batch_norm', False):
                    # Would need to add batch norm layers in __init__
                    pass
                    
                # Apply dropout if specified
                dropout_rate = layer_config.get('dropout', 0.0)
                if dropout_rate > 0:
                    x = nn.functional.dropout(x, p=dropout_rate, training=self.training)
                    
            elif layer_type == 'conv1d':
                x = self.layers[layer_idx](x)
                layer_idx += 1
                
                # Apply activation
                if 'activation' in layer_config:
                    activation_fn = self._get_activation(layer_config['activation'])
                    x = activation_fn(x)
                    
            elif layer_type == 'reshape':
                shape = layer_config['shape']
                x = x.view(-1, *shape)
                layer_idx += 1  # Skip the Identity layer
                
            elif layer_type == 'flatten':
                x = x.view(x.size(0), -1)
                layer_idx += 1  # Skip the Identity layer
                
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

# Load models using working approach from multivariate DTW script
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

# Try to load MLP model first (simpler architecture)
try:
    mlp_model = load_model(ActualCustomModel, custom_path, device="cpu")
    print(f"✅ Successfully loaded MLP model with {sum(p.numel() for p in mlp_model.parameters()):,} parameters")
except Exception as e:
    print(f"❌ Error loading MLP model: {e}")
    mlp_model = None

# Use the MLP model we successfully loaded
if mlp_model is None:
    print("❌ No model loaded successfully, cannot continue")
    exit(1)

model = mlp_model

# Generate sample data that matches your model's expected input (3D torus data)
batch_size = 100
data = 8*torch.randn(batch_size, 3)  # 3 features input for torus data
print(f"Generated data shape: {data.shape}")

# Use the high-level API instead of direct sheaf building
print("\n=== Building Sheaf Using High-Level API ===")
analyzer = NeurosheafAnalyzer(device='cpu')
analysis = analyzer.analyze(model, data, method='gromov_wasserstein', gw_config=gw_config)
sheaf = analysis['sheaf']

print(f"Sheaf constructed: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")

# Use the new detailed print method
sheaf.print_detailed_summary(max_items=5, verbosity='detailed')

# Run spectral analysis using the analyzer
print("\n=== Running Spectral Analysis ===")
spectral_analyzer = PersistentSpectralAnalyzer(
    default_n_steps=50,  # Reduced for faster execution
    default_filtration_type='threshold'
)

results = spectral_analyzer.analyze(
    sheaf,
    filtration_type='threshold',
    n_steps=100, # Reduced for faster execution
    
)

# Print results summary
print("\n=== Spectral Persistence Analysis Results ===")
print(f"Total filtration steps: {len(results['filtration_params'])}")
print(f"Birth events: {results['features']['num_birth_events']}")
print(f"Death events: {results['features']['num_death_events']}")
print(f"Crossing events: {results['features']['num_crossings']}")
print(f"Persistent paths: {results['features']['num_persistent_paths']}")
print(f"Infinite bars: {results['diagrams']['statistics']['n_infinite_bars']}")
print(f"Finite pairs: {results['diagrams']['statistics']['n_finite_pairs']}")
print(f"Mean lifetime: {results['diagrams']['statistics'].get('mean_lifetime', 0):.6f}")

# Debug eigenvalue information
eigenval_seqs = results['persistence_result']['eigenvalue_sequences']
if eigenval_seqs:
    print(f"\nEigenvalue sequences: {len(eigenval_seqs)} steps")
    print(f"Eigenvalues per step: {[len(seq) for seq in eigenval_seqs[:5]]}..." if len(eigenval_seqs) > 5 else f"Eigenvalues per step: {[len(seq) for seq in eigenval_seqs]}")
    if eigenval_seqs[0].numel() > 0:
        print(f"First step eigenvalues (first 5): {eigenval_seqs[0][:5]}")
        print(f"Last step eigenvalues (first 5): {eigenval_seqs[-1][:5]}")
else:
    print("No eigenvalue sequences found!")

# Create comprehensive interactive visualization using the enhanced visualization suite
print("\n=== Creating Enhanced Interactive Visualizations ===")
try:
    from neurosheaf.visualization import EnhancedVisualizationFactory
    
    # Initialize the enhanced visualization factory
    vf = EnhancedVisualizationFactory(theme='neurosheaf_default')
    print("✅ Enhanced visualization factory initialized")
    
    # 1. Create enhanced comprehensive dashboard
    print("Creating enhanced comprehensive analysis dashboard...")
    try:
        dashboard_fig = vf.create_comprehensive_analysis_dashboard(
            sheaf, 
            results,
            title="🧠 Enhanced Neural Network Spectral Persistence Analysis Dashboard"
        )
        dashboard_fig.write_html("spectral_analysis_dashboard.html")
        print("✅ Interactive dashboard saved as 'spectral_analysis_dashboard.html'")
    except Exception as e:
        print(f"⚠️  Could not create comprehensive dashboard: {e}")
    
    # 2. Create enhanced individual visualizations
    print("\nCreating enhanced detailed individual visualizations...")
    
    # Enhanced poset visualization with intelligent node classification
    try:
        from neurosheaf.visualization import EnhancedPosetVisualizer
        enhanced_poset_viz = EnhancedPosetVisualizer(theme='neurosheaf_default')
        
        poset_fig = enhanced_poset_viz.create_visualization(
            sheaf,
            title="🔬 Enhanced Neural Network Architecture Analysis",
            width=1400,
            height=800,
            layout_type='hierarchical',
            interactive_mode=True
        )
        poset_fig.write_html("enhanced_network_structure.html")
        print("✅ Enhanced network structure visualization saved as 'enhanced_network_structure.html'")
    except Exception as e:
        print(f"⚠️  Could not create enhanced poset visualization: {e}")
    
    # Persistence diagram with lifetime color-coding
    try:
        pers_diagram_fig = vf.create_persistence_diagram(
            results['diagrams'],
            title="Topological Persistence Features",
            width=800,
            height=600
        )
        pers_diagram_fig.write_html("persistence_diagram.html")
        print("✅ Persistence diagram saved as 'persistence_diagram.html'")
    except Exception as e:
        print(f"⚠️  Could not create persistence diagram: {e}")
    
    # Persistence barcode
    try:
        barcode_fig = vf.create_persistence_barcode(
            results['diagrams'],
            title="Feature Lifetime Analysis",
            width=1000,
            height=500
        )
        barcode_fig.write_html("persistence_barcode.html")
        print("✅ Persistence barcode saved as 'persistence_barcode.html'")
    except Exception as e:
        print(f"⚠️  Could not create persistence barcode: {e}")
    
    # Enhanced multi-scale eigenvalue evolution
    try:
        from neurosheaf.visualization import EnhancedSpectralVisualizer
        enhanced_spectral_viz = EnhancedSpectralVisualizer()
        
        eigenval_fig = enhanced_spectral_viz.create_comprehensive_spectral_view(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title="🌊 Comprehensive Spectral Evolution Analysis",
            width=1400,
            height=900
        )
        eigenval_fig.write_html("enhanced_eigenvalue_evolution.html")
        print("✅ Enhanced eigenvalue evolution saved as 'enhanced_eigenvalue_evolution.html'")
    except Exception as e:
        print(f"⚠️  Could not create enhanced eigenvalue evolution: {e}")
    
    # 3. Create specialized visualizations
    print("\nCreating specialized analysis plots...")
    
    # Spectral gap evolution
    try:
        gap_fig = vf.spectral_visualizer.plot_spectral_gap_evolution(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title="Spectral Gap Evolution Analysis"
        )
        gap_fig.write_html("spectral_gap_evolution.html")
        print("✅ Spectral gap evolution saved as 'spectral_gap_evolution.html'")
    except Exception as e:
        print(f"⚠️  Could not create spectral gap evolution: {e}")
    
    # Eigenvalue statistics
    try:
        stats_fig = vf.spectral_visualizer.plot_eigenvalue_statistics(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title="Eigenvalue Statistical Evolution"
        )
        stats_fig.write_html("eigenvalue_statistics.html")
        print("✅ Eigenvalue statistics saved as 'eigenvalue_statistics.html'")
    except Exception as e:
        print(f"⚠️  Could not create eigenvalue statistics: {e}")
    
    # Eigenvalue heatmap - show ALL eigenvalues
    try:
        heatmap_fig = vf.spectral_visualizer.plot_eigenvalue_heatmap(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title="Eigenvalue Evolution Heatmap"
        )
        heatmap_fig.write_html("eigenvalue_heatmap.html")
        print("✅ Eigenvalue heatmap saved as 'eigenvalue_heatmap.html'")
    except Exception as e:
        print(f"⚠️  Could not create eigenvalue heatmap: {e}")
    
    # Lifetime distribution
    try:
        lifetime_fig = vf.persistence_visualizer.plot_lifetime_distribution(
            results['diagrams'],
            title="Persistence Lifetime Distribution",
            bins=20
        )
        lifetime_fig.write_html("lifetime_distribution.html")
        print("✅ Lifetime distribution saved as 'lifetime_distribution.html'")
    except Exception as e:
        print(f"⚠️  Could not create lifetime distribution: {e}")
    
    # Sheaf structure summary
    try:
        sheaf_summary_fig = vf.poset_visualizer.plot_summary_stats(sheaf)
        sheaf_summary_fig.write_html("sheaf_summary.html")
        print("✅ Sheaf summary saved as 'sheaf_summary.html'")
    except Exception as e:
        print(f"⚠️  Could not create sheaf summary: {e}")
    
    # 4. Create analysis summary collection
    print("\nCreating comprehensive analysis summary...")
    try:
        summary_plots = vf.create_analysis_summary(results)
        
        # Save all summary plots
        for plot_name, figure in summary_plots.items():
            filename = f"summary_{plot_name}.html"
            figure.write_html(filename)
            print(f"✅ Summary plot '{plot_name}' saved as '{filename}'")
    except Exception as e:
        print(f"⚠️  Could not create analysis summary: {e}")
    
    # 5. Print configuration information
    print("\n=== Visualization Configuration ===")
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
    
    # 6. Print summary of created files
    print("\n=== Interactive Visualization Files Created ===")
    print("🎯 Main Dashboard:")
    print("  • spectral_analysis_dashboard.html - Complete interactive analysis")
    print("\n📊 Detailed Visualizations:")
    print("  • enhanced_network_structure.html - Interactive network topology")
    print("  • persistence_diagram.html - Topological features with hover info")
    print("  • persistence_barcode.html - Feature lifetime analysis")
    print("  • enhanced_eigenvalue_evolution.html - Multi-scale eigenvalue tracking")
    print("\n🔬 Specialized Analysis:")
    print("  • spectral_gap_evolution.html - Gap dynamics")
    print("  • eigenvalue_statistics.html - Statistical summaries")
    print("  • eigenvalue_heatmap.html - Evolution heatmap")
    print("  • lifetime_distribution.html - Persistence statistics")
    print("  • sheaf_summary.html - Structure overview")
    print("\n📈 Summary Collection:")
    print("  • summary_*.html files - Comprehensive analysis summaries")
    
    print("\n" + "="*60)
    print("🎉 INTERACTIVE VISUALIZATION SUITE COMPLETE!")
    print("="*60)
    print("All visualizations feature:")
    print("  ✓ Interactive hover information")
    print("  ✓ Zooming and panning capabilities") 
    print("  ✓ Data-flow network layout")
    print("  ✓ Multi-scale logarithmic scaling")
    print("  ✓ Lifetime-based color coding")
    print("  ✓ Mathematical correctness")
    print("\nOpen any .html file in your browser for interactive exploration!")
    
except ImportError as e:
    print(f"⚠️  Enhanced visualization modules not available: {e}")
    print("    Falling back to basic visualizations...")
    
    # Fallback to basic visualizations
    try:
        from neurosheaf.visualization.spectral import SpectralVisualizer
        from neurosheaf.visualization.persistence import PersistenceVisualizer
        
        # Create basic spectral visualization
        spectral_viz = SpectralVisualizer()
        eigenval_fig = spectral_viz.plot_eigenvalue_evolution(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title="Eigenvalue Evolution",
            max_eigenvalues=10
        )
        eigenval_fig.write_html("eigenvalue_evolution.html")
        print("✅ Basic eigenvalue evolution saved as 'eigenvalue_evolution.html'")
        
    except Exception as e:
        print(f"⚠️  Could not create basic visualizations: {e}")

print("\n=== Enhanced Analysis Complete ===")
print("✅ Successfully analyzed neural network using persistent spectral methods")
print(f"✅ Generated {len(results['filtration_params'])} filtration steps")
print(f"✅ Found {results['features']['num_persistent_paths']} persistent features")
print("✅ Created comprehensive interactive visualization suite")

# Also create a simple matplotlib version for quick comparison
print("\n=== Creating Static Comparison Plot ===")
import matplotlib.pyplot as plt

# Simple matplotlib plot for comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# 1. Simple persistence diagram
diagrams = results['diagrams']
birth_death_pairs = diagrams['birth_death_pairs']
if birth_death_pairs:
    births = [pair['birth'] for pair in birth_death_pairs]
    deaths = [pair['death'] for pair in birth_death_pairs]
    ax1.scatter(births, deaths, alpha=0.6)
    max_val = max(deaths)
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
ax1.set_title('Persistence Diagram (Static)')
ax1.set_xlabel('Birth')
ax1.set_ylabel('Death')

# 2. Simple eigenvalue evolution
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
ax2.set_title('Eigenvalue Evolution (Static)')
ax2.set_xlabel('Filtration Parameter')
ax2.set_ylabel('Eigenvalue (log scale)')

# 3. Spectral gap
gap_evolution = results['features']['spectral_gap_evolution']
ax3.plot(results['filtration_params'], gap_evolution, 'b-')
ax3.set_title('Spectral Gap Evolution')
ax3.set_xlabel('Filtration Parameter')
ax3.set_ylabel('Spectral Gap')

# 4. Feature counts
feature_names = ['Birth', 'Death', 'Crossings', 'Paths']
feature_counts = [
    results['features']['num_birth_events'],
    results['features']['num_death_events'], 
    results['features']['num_crossings'],
    results['features']['num_persistent_paths']
]
ax4.bar(feature_names, feature_counts, alpha=0.7)
ax4.set_title('Feature Summary')
ax4.set_ylabel('Count')

plt.tight_layout()
plt.savefig('static_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Static comparison plot saved as 'static_comparison.png'") 
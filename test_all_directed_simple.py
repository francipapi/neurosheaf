import torch.nn as nn 
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set environment for CPU usage
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Fixed imports based on current codebase structure
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer

# Directed sheaf imports
from neurosheaf.directed_sheaf import DirectedSheafBuilder, DirectedSheafAdapter

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

# Try to load MLP model first (simpler architecture)
try:
    mlp_model = load_model(ActualCustomModel, rand_custom_path, device="cpu")
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
batch_size = 50
data = 8*torch.randn(batch_size, 3)  # 3 features input for torus data
print(f"Generated data shape: {data.shape}")

# Use the high-level API for directed sheaf analysis
print("\n=== Building Directed Sheaf Using High-Level API ===")
analyzer = NeurosheafAnalyzer(device='cpu')

# Perform directed analysis with q=0.25 and Tikhonov regularization
directionality_parameter = 0.25
print(f"Performing directed analysis with q = {directionality_parameter}")
print("Enabling Tikhonov regularization for improved numerical stability")

# Configure Tikhonov regularization for better conditioning
regularization_config = {
    'strategy': 'moderate',  # Use moderate regularization strategy
    'target_condition': 1e8,  # Relax condition number target for directed sheaves
    'min_regularization': 1e-10,
    'max_regularization': 1e-4
}

analysis = analyzer.analyze(
    model, 
    data, 
    directed=True, 
    directionality_parameter=directionality_parameter,
    use_gram_regularization= True,  # Disable Tikhonov regularization to match legacy behavior
    preserve_eigenvalues= True,
    regularization_config=regularization_config
)

directed_sheaf = analysis['directed_sheaf']
real_laplacian = analysis['real_laplacian']

print(f"Directed sheaf constructed: {len(directed_sheaf.complex_stalks)} complex stalks")
print(f"Directionality parameter: {directed_sheaf.directionality_parameter}")

# Show first 3 complex stalks
for i, (node, stalk) in enumerate(directed_sheaf.complex_stalks.items()):
    if i < 3:  # Show first 3
        print(f"  Complex stalk {node}: {stalk.shape}, dtype={stalk.dtype}")

# Validate directed sheaf properties
print("\n=== Directed Sheaf Validation ===")
print(f"Complex stalks: {len(directed_sheaf.complex_stalks)}")
print(f"Directed restrictions: {len(directed_sheaf.directed_restrictions)}")

# Check complex stalk properties
for i, (node_id, stalk) in enumerate(list(directed_sheaf.complex_stalks.items())[:3]):
    is_complex = stalk.dtype in [torch.complex64, torch.complex128]
    print(f"Node {node_id}: complex={is_complex}, shape={stalk.shape}")

# Check real embedding properties
laplacian_metadata = analysis['laplacian_metadata']
print(f"Complex dimension: {laplacian_metadata.complex_dimension}")
print(f"Real dimension: {laplacian_metadata.real_dimension}")
print(f"Real embedding ratio: {laplacian_metadata.real_dimension/max(1,laplacian_metadata.complex_dimension):.2f}")
print(f"Sparsity: {laplacian_metadata.sparsity:.3f}")

# Run spectral analysis using the analyzer
print("\n=== Running Spectral Analysis ===")
spectral_analyzer = PersistentSpectralAnalyzer(
    default_n_steps=50,  # Reduced for faster execution and fewer convergence issues
    default_filtration_type='threshold'
)

# Create compatibility sheaf for spectral analysis
adapter = DirectedSheafAdapter()
compatibility_sheaf = adapter.create_compatibility_sheaf(directed_sheaf)
print(f"Compatibility sheaf created with {len(compatibility_sheaf.stalks)} stalks")

results = spectral_analyzer.analyze(
    compatibility_sheaf,
    filtration_type='threshold',
    n_steps=50, # Reduced for faster execution and fewer convergence issues
)

# Print results summary
print("\n=== Directed Spectral Persistence Analysis Results ===")
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
    print("Creating enhanced comprehensive directed analysis dashboard...")
    try:
        dashboard_fig = vf.create_comprehensive_analysis_dashboard(
            compatibility_sheaf, 
            results,
            title=f"🧠 Enhanced Directed Neural Network Spectral Persistence Analysis Dashboard (q={directionality_parameter})"
        )
        dashboard_fig.write_html("directed_spectral_analysis_dashboard.html")
        print("✅ Interactive directed dashboard saved as 'directed_spectral_analysis_dashboard.html'")
    except Exception as e:
        print(f"⚠️  Could not create comprehensive dashboard: {e}")
    
    # 2. Create enhanced individual visualizations
    print("\nCreating enhanced detailed individual visualizations...")
    
    # Enhanced poset visualization with intelligent node classification
    try:
        from neurosheaf.visualization import EnhancedPosetVisualizer
        enhanced_poset_viz = EnhancedPosetVisualizer(theme='neurosheaf_default')
        
        poset_fig = enhanced_poset_viz.create_visualization(
            compatibility_sheaf,
            title=f"🔬 Enhanced Directed Neural Network Architecture Analysis (q={directionality_parameter})",
            width=1400,
            height=800,
            layout_type='hierarchical',
            interactive_mode=True
        )
        poset_fig.write_html("directed_enhanced_network_structure.html")
        print("✅ Enhanced directed network structure visualization saved as 'directed_enhanced_network_structure.html'")
    except Exception as e:
        print(f"⚠️  Could not create enhanced poset visualization: {e}")
    
    # Persistence diagram with lifetime color-coding
    try:
        pers_diagram_fig = vf.create_persistence_diagram(
            results['diagrams'],
            title=f"Directed Topological Persistence Features (q={directionality_parameter})",
            width=800,
            height=600
        )
        pers_diagram_fig.write_html("directed_persistence_diagram.html")
        print("✅ Directed persistence diagram saved as 'directed_persistence_diagram.html'")
    except Exception as e:
        print(f"⚠️  Could not create persistence diagram: {e}")
    
    # Persistence barcode
    try:
        barcode_fig = vf.create_persistence_barcode(
            results['diagrams'],
            title=f"Directed Feature Lifetime Analysis (q={directionality_parameter})",
            width=1000,
            height=500
        )
        barcode_fig.write_html("directed_persistence_barcode.html")
        print("✅ Directed persistence barcode saved as 'directed_persistence_barcode.html'")
    except Exception as e:
        print(f"⚠️  Could not create persistence barcode: {e}")
    
    # Enhanced multi-scale eigenvalue evolution
    try:
        from neurosheaf.visualization import EnhancedSpectralVisualizer
        enhanced_spectral_viz = EnhancedSpectralVisualizer()
        
        eigenval_fig = enhanced_spectral_viz.create_comprehensive_spectral_view(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title=f"🌊 Comprehensive Directed Spectral Evolution Analysis (q={directionality_parameter})",
            width=1400,
            height=900
        )
        eigenval_fig.write_html("directed_enhanced_eigenvalue_evolution.html")
        print("✅ Enhanced directed eigenvalue evolution saved as 'directed_enhanced_eigenvalue_evolution.html'")
    except Exception as e:
        print(f"⚠️  Could not create enhanced eigenvalue evolution: {e}")
    
    # 3. Create specialized visualizations
    print("\nCreating specialized directed analysis plots...")
    
    # Spectral gap evolution
    try:
        gap_fig = vf.spectral_visualizer.plot_spectral_gap_evolution(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title=f"Directed Spectral Gap Evolution Analysis (q={directionality_parameter})"
        )
        gap_fig.write_html("directed_spectral_gap_evolution.html")
        print("✅ Directed spectral gap evolution saved as 'directed_spectral_gap_evolution.html'")
    except Exception as e:
        print(f"⚠️  Could not create spectral gap evolution: {e}")
    
    # Eigenvalue statistics
    try:
        stats_fig = vf.spectral_visualizer.plot_eigenvalue_statistics(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title=f"Directed Eigenvalue Statistical Evolution (q={directionality_parameter})"
        )
        stats_fig.write_html("directed_eigenvalue_statistics.html")
        print("✅ Directed eigenvalue statistics saved as 'directed_eigenvalue_statistics.html'")
    except Exception as e:
        print(f"⚠️  Could not create eigenvalue statistics: {e}")
    
    # Eigenvalue heatmap - show ALL eigenvalues
    try:
        heatmap_fig = vf.spectral_visualizer.plot_eigenvalue_heatmap(
            results['persistence_result']['eigenvalue_sequences'],
            results['filtration_params'],
            title=f"Directed Eigenvalue Evolution Heatmap (q={directionality_parameter})"
        )
        heatmap_fig.write_html("directed_eigenvalue_heatmap.html")
        print("✅ Directed eigenvalue heatmap saved as 'directed_eigenvalue_heatmap.html'")
    except Exception as e:
        print(f"⚠️  Could not create eigenvalue heatmap: {e}")
    
    # Lifetime distribution
    try:
        lifetime_fig = vf.persistence_visualizer.plot_lifetime_distribution(
            results['diagrams'],
            title=f"Directed Persistence Lifetime Distribution (q={directionality_parameter})",
            bins=20
        )
        lifetime_fig.write_html("directed_lifetime_distribution.html")
        print("✅ Directed lifetime distribution saved as 'directed_lifetime_distribution.html'")
    except Exception as e:
        print(f"⚠️  Could not create lifetime distribution: {e}")
    
    # Sheaf structure summary
    try:
        sheaf_summary_fig = vf.poset_visualizer.plot_summary_stats(compatibility_sheaf)
        sheaf_summary_fig.write_html("directed_sheaf_summary.html")
        print("✅ Directed sheaf summary saved as 'directed_sheaf_summary.html'")
    except Exception as e:
        print(f"⚠️  Could not create sheaf summary: {e}")
    
    # 4. Create analysis summary collection
    print("\nCreating comprehensive directed analysis summary...")
    try:
        summary_plots = vf.create_analysis_summary(results)
        
        # Save all summary plots
        for plot_name, figure in summary_plots.items():
            filename = f"directed_summary_{plot_name}.html"
            figure.write_html(filename)
            print(f"✅ Directed summary plot '{plot_name}' saved as '{filename}'")
    except Exception as e:
        print(f"⚠️  Could not create analysis summary: {e}")
    
    # 5. Print configuration information
    print("\n=== Directed Visualization Configuration ===")
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
    print("\n=== Interactive Directed Visualization Files Created ===")
    print("🎯 Main Dashboard:")
    print("  • directed_spectral_analysis_dashboard.html - Complete interactive directed analysis")
    print("\n📊 Detailed Visualizations:")
    print("  • directed_enhanced_network_structure.html - Interactive directed network topology")
    print("  • directed_persistence_diagram.html - Directed topological features with hover info")
    print("  • directed_persistence_barcode.html - Directed feature lifetime analysis")
    print("  • directed_enhanced_eigenvalue_evolution.html - Multi-scale directed eigenvalue tracking")
    print("\n🔬 Specialized Analysis:")
    print("  • directed_spectral_gap_evolution.html - Directed gap dynamics")
    print("  • directed_eigenvalue_statistics.html - Directed statistical summaries")
    print("  • directed_eigenvalue_heatmap.html - Directed evolution heatmap")
    print("  • directed_lifetime_distribution.html - Directed persistence statistics")
    print("  • directed_sheaf_summary.html - Directed structure overview")
    print("\n📈 Summary Collection:")
    print("  • directed_summary_*.html files - Comprehensive directed analysis summaries")
    
    print("\n" + "="*60)
    print("🎉 DIRECTED INTERACTIVE VISUALIZATION SUITE COMPLETE!")
    print("="*60)
    print("All directed visualizations feature:")
    print("  ✓ Interactive hover information")
    print("  ✓ Zooming and panning capabilities") 
    print("  ✓ Data-flow network layout")
    print("  ✓ Multi-scale logarithmic scaling")
    print("  ✓ Lifetime-based color coding")
    print("  ✓ Mathematical correctness")
    print("  ✓ Directionality parameter annotation")
    print("  ✓ Complex-to-real embedding visualization")
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
            title=f"Directed Eigenvalue Evolution (q={directionality_parameter})",
            max_eigenvalues=10
        )
        eigenval_fig.write_html("directed_eigenvalue_evolution.html")
        print("✅ Basic directed eigenvalue evolution saved as 'directed_eigenvalue_evolution.html'")
        
    except Exception as e:
        print(f"⚠️  Could not create basic visualizations: {e}")

print("\n=== Enhanced Directed Analysis Complete ===")
print("✅ Successfully analyzed neural network using directed persistent spectral methods")
print(f"✅ Generated {len(results['filtration_params'])} filtration steps")
print(f"✅ Found {results['features']['num_persistent_paths']} persistent features")
print("✅ Created comprehensive interactive directed visualization suite")

# Also create a simple matplotlib version for quick comparison
print("\n=== Creating Static Directed Comparison Plot ===")
import matplotlib.pyplot as plt

# Simple matplotlib plot for comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# 1. Simple persistence diagram
diagrams = results['diagrams']
birth_death_pairs = diagrams['birth_death_pairs']
if birth_death_pairs:
    births = [pair['birth'] for pair in birth_death_pairs]
    deaths = [pair['death'] for pair in birth_death_pairs]
    ax1.scatter(births, deaths, alpha=0.6, color='red', label='Directed')
    max_val = max(deaths)
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
ax1.set_title(f'Directed Persistence Diagram (q={directionality_parameter})')
ax1.set_xlabel('Birth')
ax1.set_ylabel('Death')
ax1.legend()

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
ax2.set_title(f'Directed Eigenvalue Evolution (q={directionality_parameter})')
ax2.set_xlabel('Filtration Parameter')
ax2.set_ylabel('Eigenvalue (log scale)')

# 3. Spectral gap
gap_evolution = results['features']['spectral_gap_evolution']
ax3.plot(results['filtration_params'], gap_evolution, 'r-', label=f'Directed (q={directionality_parameter})')
ax3.set_title('Directed Spectral Gap Evolution')
ax3.set_xlabel('Filtration Parameter')
ax3.set_ylabel('Spectral Gap')
ax3.legend()

# 4. Feature counts
feature_names = ['Birth', 'Death', 'Crossings', 'Paths']
feature_counts = [
    results['features']['num_birth_events'],
    results['features']['num_death_events'], 
    results['features']['num_crossings'],
    results['features']['num_persistent_paths']
]
ax4.bar(feature_names, feature_counts, alpha=0.7, color='red')
ax4.set_title(f'Directed Feature Summary (q={directionality_parameter})')
ax4.set_ylabel('Count')

plt.tight_layout()
plt.savefig('directed_static_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Static directed comparison plot saved as 'directed_static_comparison.png'") 
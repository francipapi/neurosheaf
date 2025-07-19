import torch.nn as nn 
import torch
import matplotlib.pyplot as plt
import numpy as np
from neurosheaf.sheaf import SheafBuilder, build_sheaf_laplacian, Sheaf
from neurosheaf.spectral import PersistentSpectralAnalyzer
from neurosheaf.utils import load_model, wasserstein_distance, bottleneck_distance, extract_persistence_diagram_array

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

# First, let's inspect the saved model to understand its structure
from neurosheaf.utils import list_model_info

print("=== Inspecting Saved Model ===")
custom_path = "models/torch_custom_acc_1.0000_epoch_200.pth"
mlp_path = "models/torch_mlp_acc_1.0000_epoch_200.pth"
mlp_path1 = "models/torch_mlp_acc_0.9857_epoch_100.pth"
rand_custom_path = "models/random_custom_net_000_default_seed_42.pth"
rand_mlp_path = "models/random_mlp_net_000_default_seed_42.pth"

try:
    model_info = list_model_info(custom_path)
    print("Model structure:")
    for layer_name in model_info.get('layer_names', [])[:10]:  # Show first 10 layers
        print(f"  {layer_name}: {model_info['layer_shapes'][layer_name]}")
    
    # Based on the error, it seems like the model has this structure:
    # layers.0, layers.2, layers.5, layers.8, layers.11, layers.14
    # This suggests: Linear -> activation -> Linear -> activation -> Linear -> activation -> Linear
    
    print(f"\nTotal parameters: {model_info.get('model_parameters', 'Unknown'):,}")
    
except Exception as e:
    print(f"Error inspecting model: {e}")

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

print("\n=== Loading Custom Model ===")
try:
    model = load_model(MLPModel, mlp_path, device="cpu")
    model1 = load_model(MLPModel, rand_mlp_path, device="cpu")
    print("✅ Successfully loaded custom model")
    
    # Print model summary
    print(f"Model type: {type(model)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"❌ Error loading custom model: {e}")

# Generate sample data that matches your model's expected input (3D torus data)
batch_size = 200
data = torch.randn(batch_size, 3)  # 3 features input for torus data
print(f"Generated data shape: {data.shape}")

builder = SheafBuilder()
sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=True)

builder1 = SheafBuilder()
sheaf1 = builder.build_from_activations(model1, data, use_gram_regularization=True, validate=True)

print(f"Sheaf constructed: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
for i, (node, stalk) in enumerate(sheaf.stalks.items()):
    if i < 3:  # Show first 3
        print(f"  Stalk {node}: {stalk.shape}")

print(f"Sheaf constructed: {len(sheaf.stalks)} stalks, {len(sheaf1.restrictions)} restrictions")
for i, (node, stalk) in enumerate(sheaf1.stalks.items()):
    if i < 3:  # Show first 3
        print(f"  Stalk {node}: {stalk.shape}")


laplacian, metadata = build_sheaf_laplacian(sheaf, validate=True)
print(f"Laplacian built: {laplacian.shape}, {laplacian.nnz} non-zeros")

laplacian1, metadata1 = build_sheaf_laplacian(sheaf1, validate=True)
print(f"Laplacian built: {laplacian1.shape}, {laplacian1.nnz} non-zeros")

analyzer = PersistentSpectralAnalyzer(
            default_n_steps=100, 
            default_filtration_type='threshold'
        )

results = analyzer.analyze(
            sheaf,
            filtration_type='threshold',
            n_steps=100,
            param_range=(0.0, 10.0)
        )

analyzer1 = PersistentSpectralAnalyzer(
            default_n_steps=100, 
            default_filtration_type='threshold'
        )

results1 = analyzer1.analyze(
            sheaf1,
            filtration_type='threshold',
            n_steps=100,
            param_range=(0.0, 10.0)
        )

# Extract persistence diagrams as numpy arrays using the utility function
diagram1 = extract_persistence_diagram_array(results1['diagrams'])
diagram2 = extract_persistence_diagram_array(results['diagrams'])

wdist = wasserstein_distance(diagram1, diagram2)
bdist = bottleneck_distance(diagram1, diagram2)

print("Wassernstein Distance", wdist)
print("Bottleneck distance:", bdist)
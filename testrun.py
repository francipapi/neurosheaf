import torch.nn as nn 
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import psutil
import tracemalloc
import time
# Core imports
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer, AlphaFlowSpec, StaticBuildConfig
from neurosheaf.spectral.flows.alpha_flow import AlphaGroupingPolicy
from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.visualization import EnhancedVisualizationFactory

# NEW: GW Subspace Tracker imports
from neurosheaf.spectral.gw.gw_subspace_tracker import GWSubspaceTracker
from neurosheaf.spectral.tracker_factory import SubspaceTrackerFactory
from neurosheaf.spectral.gw.pes_computation import PESComputer

from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.spectral import UnifiedStaticLaplacian

import logging
logging.getLogger('neurosheaf').setLevel(logging.DEBUG)


# Set random seeds for reproducibility
random_seed = 30
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


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

class MLP4x256(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        dims = [784] + [hidden_dim] * num_layers
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.head(x)

print("=== Loading Models ===")
custom_path = "models/custom_trained_acc100_ep200.pth"
mlp_path = "models/mlp_trained_acc100_ep200.pth"
mlp_path1 = "models/mlp_trained_acc98_ep100.pth"
rand_custom_path = "models/custom_random_seed42.pth"
rand_mlp_path = "models/mlp_random_seed42.pth"
mlp4 = "models/mlp4layer_mnist_seed42.pth"
mlp4_rand = "models/mnist_mlp_random_001.pth"

model = load_model(MLP4x256, mlp4)

# Load actual MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
])

print("Loading MNIST dataset...")
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Get first batch_size test samples
batch_size = 100
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
data, labels = next(iter(test_loader))
#data = 10 * torch.randn(self.batch_size, 3)
print(f"Loaded MNIST data shape: {data.shape}")
print(f"Labels for first 10 samples: {labels[:10].tolist()}")

print("\n=== Building Sheaf Using High-Level API ===")
analyzer = NeurosheafAnalyzer(device='cpu')

print("\nRunning analysis WITH layer filtering (exclude final single-output layers):")
analysis = analyzer.analyze(
    model, data, 
    method='gromov_wasserstein',
    use_normalized_laplacian = False,
    exclude_final_single_output=False  # NEW: Enable layer filtering to reduce degeneracy
)
sheaf = analysis['sheaf']

print(f"\n=== Filtered Sheaf Details ===")
sheaf.print_detailed_summary(max_items=25, verbosity='detailed')

print("\n=== Running Spectral Analysis with NORMALIZED HODGE LAPLACIAN ===")

spectral_analyzer = PersistentSpectralAnalyzer(
    default_n_steps=30,
    default_filtration_type='threshold'
)
'''
# Create α-flow specification with default parameters
spec = AlphaFlowSpec(
    alpha_grid=(0.0, 0.1, 0.3, 1.0, 3.0),  # α values to analyze
    k_small=50,  # Number of smallest eigenvalues
    probes=256,  # Hutchinson probes for trace estimation
    moments=(1, 2, 3, 4, 5, 6),  # Include k=1 for trace computation
    grouping=AlphaGroupingPolicy(kind='quantile', param=0.5)  # Edge partitioning
)

# Create build configuration
config = StaticBuildConfig(
    mass_mode='fixed',  # Fixed mass matrix for consistency
    precision='double',  # Double precision
    random_state=42  # For reproducibility
)

# Run α-flow analysis with proper arguments
results = spectral_analyzer.analyze_alpha_flow(sheaf, spec, config, True)

# Import and use the pretty printer
from neurosheaf.utils import print_alpha_flow_results

print(print_alpha_flow_results(results))
'''
results = spectral_analyzer.analyze(
        sheaf,
        filtration_type='threshold',
        n_steps=30
    )

vf = EnhancedVisualizationFactory(theme='neurosheaf_default')

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

from neurosheaf.io import save_eigenvalue_evolution
save_eigenvalue_evolution(
      results['persistence_result']['eigenvalue_sequences'],
      results['filtration_params'],
      'eigenvalue_evolution.npz'
  )

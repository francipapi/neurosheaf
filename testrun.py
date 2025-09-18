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
import torch.nn.functional as F

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
    def __init__(self, num_layers: int = 4, hidden_dim: int = 16, num_classes: int = 10):
        super().__init__()
        # store config so get_model_info can read it
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        dims = [28*28] + [hidden_dim] * num_layers
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.head(x)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # (B, 1, 28, 28)
            nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2),   # -> (B, 4, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),   # -> (B, 8, 7, 7)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                               # -> (B, 8, 1, 1)
            nn.Flatten(),                                          # -> (B, 8)
            nn.Linear(8, 10)                                       # -> (B, 10) logits
        )

    def forward(self, x):
        return self.layers(x)
    
class Hybrid1DCNN(nn.Module):
    """
    Redesigned Hybrid 1D CNN with aggressive dimension reduction.
    Uses strided convolutions and early pooling to minimize activation sizes for neurosheaf pipeline.
    Target Laplacian size: ~2K×2K (95% reduction from original 37K×37K)
    """
    
    def __init__(self, input_dim: int = 104, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Stage 1: Initial feature extraction with immediate dimension reduction
        # Input: [B, 1, 104]
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        # Output: [B, 16, 52] - Halved spatial dimension immediately
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # After pool1: [B, 16, 26] - Total dims: 416
        
        # Stage 2: Feature refinement with further reduction
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False)
        # Output: [B, 32, 13] - Total dims: 416
        self.bn2 = nn.BatchNorm1d(32)
        
        # Stage 3: Deep features with aggressive spatial reduction
        self.conv3 = nn.Conv1d(32, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # Output: [B, 48, 13] - Total dims: 624
        self.bn3 = nn.BatchNorm1d(48)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # After pool2: [B, 48, 6] - Total dims: 288
        
        # Stage 4: Final conv with very small spatial dimension
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=1, padding=0, bias=False)
        # Output: [B, 64, 4] - Very small spatial dimension, Total dims: 256
        self.bn4 = nn.BatchNorm1d(64)
        
        # Global pooling to fixed size for consistent FC input
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # Output: [B, 64, 1] - Total dims: 64
        
        # Compact MLP Head (all under 100 units)
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(16)
        
        # Classification output
        self.head = nn.Linear(16, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch_size, features]
        # Reshape for 1D conv: [batch_size, channels=1, features]
        x = x.unsqueeze(1)  # [B, 1, 104]
        
        # Stage 1: Initial feature extraction with immediate reduction
        x = F.relu(self.bn1(self.conv1(x)))      # [B, 16, 52]
        x = self.pool1(x)                        # [B, 16, 26]
        
        # Stage 2: Feature refinement with further reduction
        x = F.relu(self.bn2(self.conv2(x)))      # [B, 32, 13]
        
        # Stage 3: Deep features with aggressive spatial reduction
        x = F.relu(self.bn3(self.conv3(x)))      # [B, 48, 13]
        x = self.pool2(x)                        # [B, 48, 6]
        
        # Stage 4: Final conv with very small spatial dimension
        x = F.relu(self.bn4(self.conv4(x)))      # [B, 64, 4]
        
        # Global pooling to fixed size
        x = self.global_avg_pool(x)              # [B, 64, 1]
        x = x.squeeze(-1)                        # [B, 64]
        
        # Compact MLP Head with regularization
        x = F.relu(self.bn_fc1(self.fc1(x)))     # [B, 32]
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))     # [B, 16]
        
        # Classification output (logits)
        return self.head(x)        

class DeepMLP(nn.Module):
    """Deep MLP for binary classification with emphasis on depth over width."""
    
    def __init__(self, input_dim: int = 104, num_layers: int = 12, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Build deep network: input -> hidden1 -> ... -> hiddenN -> output
        dims = [input_dim] + [hidden_dim] * num_layers
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            
        self.backbone = nn.Sequential(*layers)
        
        # Classification head: output logits for binary classification
        self.head = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)

class DigitsFullCNN(nn.Module):
    def __init__(self, hidden_dim: int = 32, dropout_prob: float = 0.0, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: (B, 1, 8, 8)

            # Conv backbone (unchanged)
            nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=1),  # -> (B, 10, 4, 4)
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1), # -> (B, 20, 2, 2)
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # -> (B, 20, 1, 1)

            # Small MLP head
            nn.Flatten(),                          # -> (B, 20)
            nn.Linear(20, hidden_dim),             # -> (B, hidden_dim)
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes)     # -> (B, num_classes) logits
        )

    def forward(self, x):
        return self.layers(x)

    def forward(self, x):
        return self.layers(x)

class DigitsMLP(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 16, num_classes: int = 10):
        super().__init__()
        # store config so get_model_info can read it
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Create sequential architecture: 64 -> hidden_dim -> ... -> hidden_dim -> 10
        layers = []
        
        # Input layer: 64 -> hidden_dim
        layers.extend([
            nn.Linear(64, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        # Output layer: hidden_dim -> num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        return self.layers(x)

class Pendigits1DCNN(nn.Module):
    def __init__(
        self,
        conv1_channels: int = 8,        # ↓ from 16
        conv2_channels: int = 16,       # ↓ from 32
        kernel_size: int = 3,
        dropout_prob: float = 0.10,
        mlp_hidden_dim: int = 24,       # ↓ from 32
        num_classes: int = 10,
    ):
        super().__init__()
        pad = kernel_size // 2

        # Store config for metadata
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_classes = num_classes

        # Input: (N, 16)  → reshape to (N, 1, 16)
        # Conv blocks use GroupNorm(1, C): stable for small batches and on MPS
        self.layers = nn.Sequential(
            nn.LayerNorm(16),                       # stabilize inputs (helps MPS)
            nn.Unflatten(1, (1, 16)),

            nn.Conv1d(1, conv1_channels, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(1, conv1_channels),
            nn.ReLU(),

            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(1, conv2_channels),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),               # → (N, C, 1)
            nn.Flatten(),                           # → (N, C)

            nn.Dropout(dropout_prob),
            nn.Linear(conv2_channels, mlp_hidden_dim, bias=True),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Ensure flattened input (N, 16)
        return self.layers(x)

class PendigitsMLP(nn.Module):
    def __init__(self, num_layers: int = 3, hidden_dim: int = 32, 
                 dropout_prob: float = 0.15, num_classes: int = 10):
        super().__init__()
        # Store config for metadata
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        # Create MLP architecture: 16 -> 64 -> [64 -> 64] x (num_layers-1) -> 10
        layers = []
        
        # Input layer: 16 -> hidden_dim
        layers.extend([
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        ])
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
        
        # Output layer: hidden_dim -> num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 16)
        return self.layers(x)

print("=== Loading Models ===")
custom_path = "models/custom_trained_acc100_ep200.pth"
mlp_path = "models/mlp_trained_v01_acc1.0000_ep20.pth"
mlp_path1 = "models/mlp_trained_acc98_ep100.pth"
rand_custom_path = "models/custom_random_seed42.pth"
rand_mlp_path = "models/mlp_random_seed42.pth"
mlp4 = "models/mlp4layer_mnist_seed52.pth"
mlp4_rand = "models/mnist_mlp_random_042.pth"
mnist_cnn = "models/tinycnn_mnist1.pth"
rand_mnist_cnn = "models/tinycnn_random_001.pth"
adult_cnn = "models/hybrid_cnn_adult.pth"
adult_mlp = "models/deep_mlp_adult_seed42.pth"
digit_mlp = "models/digits_mlp_seed49.pth"
digit_mlp_rand = "models/digits_mlp_random_001.pth"
digit_cnn = "models/slimcnn_digits_seed60.pth"
digit_cnn_random = "models/slimcnn_digits_random_001.pth"
pedigit_cnn = "models/pendigits_cnn_slim_seed65.pth"
pedigit_mlp = "models/pendigits_mlp_seed45.pth"

model = load_model(PendigitsMLP, pedigit_mlp)

batch_size = 1000

# Transform logic (commented out - only needed for MNIST)

'''
# Apply appropriate transform based on model type
if isinstance(model, TinyCNN):
    # CNN models need 2D images: [batch_size, 1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [1, 28, 28]
    ])
    print("Using 2D transform for CNN model")
else:
    # MLP models need flattened input: [batch_size, 784]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
    ])
    print("Using flattened transform for MLP model")

print("Loading MNIST dataset...")
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Get first batch_size test samples

test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
data, labels = next(iter(test_loader))
#data = 10 * torch.randn(self.batch_size, 3)
print(f"Loaded MNIST data shape: {data.shape}")
print(f"Labels for first 10 samples: {labels[:10].tolist()}")


data = 12 * torch.rand((batch_size, 3))
# Generate probe data for Adult dataset (104 features, binary classification)
print("Generating probe data for Adult dataset...")
data = torch.randn(batch_size, 104)  # 104 features for Adult dataset
labels = torch.randint(0, 2, (batch_size,))  # Binary labels (0 or 1)
'''
data = torch.rand((batch_size, 1, 16), dtype=torch.float32)

print("\n=== Building Sheaf Using High-Level API ===")
analyzer = NeurosheafAnalyzer(device='cpu')

# Create improved GW configuration for better convergence
print("Creating improved GW configuration with increased iterations and float64 precision...")
improved_gw_config = GWConfig(
    max_iter=2000,  # Increased from default 1000 for better convergence
    computation_dtype='float64'  # Use float64 for better numerical precision
)
print(f"GW Config: max_iter={improved_gw_config.max_iter}, dtype={improved_gw_config.computation_dtype}")

print("\nRunning analysis WITH layer filtering (exclude final single-output layers):")
analysis = analyzer.analyze(
    model, data, 
    method='gromov_wasserstein',
    gw_config=improved_gw_config,  # Pass the improved configuration
    use_normalized_laplacian = False,
    exclude_final_single_output=False  # NEW: Enable layer filtering to reduce degeneracy
)
sheaf = analysis['sheaf']

print(f"\n=== Filtered Sheaf Details ===")
sheaf.print_detailed_summary(max_items=25, verbosity='detailed')

print(f"\n=== GW COST ANALYSIS ===")
print("="*60)

# Extract GW costs from sheaf metadata
gw_costs = sheaf.get_gw_costs()
if gw_costs:
    print(f"Total edges: {len(gw_costs)}")
    
    # Convert to sorted list for analysis
    cost_items = [(edge, cost) for edge, cost in gw_costs.items()]
    cost_items.sort(key=lambda x: x[1], reverse=True)  # Sort by cost (descending)
    
    costs_only = [cost for _, cost in cost_items]
    print(f"GW cost range: [{min(costs_only):.6f}, {max(costs_only):.6f}]")
    print(f"GW cost mean: {sum(costs_only)/len(costs_only):.6f}")
    print(f"GW cost std: {(sum((c - sum(costs_only)/len(costs_only))**2 for c in costs_only) / len(costs_only))**0.5:.6f}")
    
    # Identify outliers (costs > mean + 2*std)
    mean_cost = sum(costs_only) / len(costs_only)
    std_cost = (sum((c - mean_cost)**2 for c in costs_only) / len(costs_only))**0.5
    outlier_threshold = mean_cost + 2 * std_cost
    
    print(f"\n🔍 HIGHEST GW COSTS (Top 10):")
    print("-" * 60)
    for i, (edge, cost) in enumerate(cost_items[:10]):
        source, target = edge
        outlier_flag = "⚠️ OUTLIER" if cost > outlier_threshold else ""
        print(f"  {i+1:2d}. {source} → {target}")
        print(f"      GW Cost: {cost:.6f} {outlier_flag}")
    
    # Highlight the most problematic edge
    worst_edge, worst_cost = cost_items[0]
    print(f"\n🚨 MOST PROBLEMATIC EDGE: {worst_edge[0]} → {worst_edge[1]}")
    print(f"   GW Cost: {worst_cost:.6f}")
    print(f"   Cost ratio (worst/mean): {worst_cost/mean_cost:.2f}x")
    
    # Show layer type analysis
    print(f"\n📊 LAYER TYPE ANALYSIS:")
    print("-" * 40)
    layer_costs = {}
    for (source, target), cost in cost_items:
        source_type = source.split('.')[0] if '.' in source else source.split('_')[0]
        target_type = target.split('.')[0] if '.' in target else target.split('_')[0]
        edge_type = f"{source_type} → {target_type}"
        
        if edge_type not in layer_costs:
            layer_costs[edge_type] = []
        layer_costs[edge_type].append(cost)
    
    for edge_type, costs in sorted(layer_costs.items(), key=lambda x: max(x[1]), reverse=True):
        avg_cost = sum(costs) / len(costs)
        max_cost = max(costs)
        print(f"  {edge_type}: avg={avg_cost:.4f}, max={max_cost:.4f}, count={len(costs)}")

else:
    print("❌ No GW costs found - not a GW sheaf or costs not stored")

print("\n=== Running Spectral Analysis with NORMALIZED HODGE LAPLACIAN ===")

spectral_analyzer = PersistentSpectralAnalyzer(
    default_n_steps=50,
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
        n_steps=50
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

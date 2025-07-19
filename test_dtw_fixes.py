#!/usr/bin/env python3
"""
Test script to verify the DTW sensitivity fixes.

This script tests a few model pairs to check if the DTW distance fixes
are working and producing meaningful different values rather than converging
to the same ~10.46 value.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model

# Model architectures (copied from original script)
class MLPModel(nn.Module):
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
        
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        
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
        x = self.layers[1](self.layers[0](x))  # [batch_size, 32]
        x = self.layers[4](self.layers[3](self.layers[2](x)))  # [batch_size, 32]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        x = self.layers[7](self.layers[6](self.layers[5](x)))  # [batch_size, 32, 1]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        x = self.layers[10](self.layers[9](self.layers[8](x)))  # [batch_size, 32, 1]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        x = self.layers[13](self.layers[12](self.layers[11](x)))  # [batch_size, 32, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        x = self.layers[15](self.layers[14](x))  # [batch_size, 1]
        return x


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print("🔧 Environment Setup Complete")


def load_test_models():
    """Load a subset of models for testing the fixes."""
    models_dir = Path("models")
    
    test_models = [
        ("torch_mlp_acc_1.0000_epoch_200.pth", MLPModel, "Trained MLP (100% acc)"),
        ("random_mlp_net_000_default_seed_42.pth", MLPModel, "Random MLP"),
        ("torch_custom_acc_1.0000_epoch_200.pth", ActualCustomModel, "Trained Custom (100% acc)")
    ]
    
    loaded_models = {}
    model_descriptions = {}
    
    print(f"\n📦 Loading test models...")
    
    for model_file, model_class, description in test_models:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                model = load_model(model_class, model_path, device='cpu')
                loaded_models[model_file] = model
                model_descriptions[model_file] = description
                print(f"   ✅ {description}")
            except Exception as e:
                print(f"   ❌ Failed to load {model_file}: {e}")
        else:
            print(f"   ⚠️  Model not found: {model_file}")
    
    return loaded_models, model_descriptions


def test_dtw_sensitivity(models, descriptions):
    """Test DTW sensitivity with the fixed implementation."""
    print(f"\n🔬 Testing DTW Sensitivity with Fixed Implementation")
    print("=" * 60)
    
    # Generate test data
    data = torch.randn(100, 3)
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    model_names = list(models.keys())
    results = []
    
    # Test a few key comparisons
    test_pairs = [
        # Same architecture, different training states
        ("torch_mlp_acc_1.0000_epoch_200.pth", "random_mlp_net_000_default_seed_42.pth"),
        # Cross-architecture comparison  
        ("torch_mlp_acc_1.0000_epoch_200.pth", "torch_custom_acc_1.0000_epoch_200.pth"),
    ]
    
    for model1_name, model2_name in test_pairs:
        if model1_name in models and model2_name in models:
            print(f"\n🔍 Testing: {descriptions[model1_name]} vs {descriptions[model2_name]}")
            
            try:
                start_time = time.time()
                
                result = analyzer.compare_networks(
                    models[model1_name], 
                    models[model2_name], 
                    data, 
                    method='dtw',
                    eigenvalue_index=None,  # All eigenvalues
                    multivariate=True  # Multivariate DTW
                )
                
                elapsed = time.time() - start_time
                
                similarity_score = result['similarity_score']
                
                # Extract diagnostic information
                dtw_details = result.get('dtw_comparison', {})
                similarity_metrics = dtw_details.get('similarity_metrics', {})
                
                raw_dtw_distance = similarity_metrics.get('raw_dtw_distance', 'N/A')
                dtw_similarity = similarity_metrics.get('dtw_similarity', 'N/A')
                normalized_dtw = similarity_metrics.get('normalized_dtw_distance', 'N/A')
                
                results.append({
                    'pair': f"{model1_name} vs {model2_name}",
                    'similarity_score': similarity_score,
                    'raw_dtw_distance': raw_dtw_distance,
                    'dtw_similarity': dtw_similarity,
                    'normalized_dtw': normalized_dtw,
                    'time': elapsed
                })
                
                print(f"   ✅ Final Similarity Score: {similarity_score:.6f}")
                print(f"   📊 Raw DTW Distance: {raw_dtw_distance}")
                print(f"   📊 DTW Similarity Component: {dtw_similarity}")
                print(f"   📊 Normalized DTW Distance: {normalized_dtw}")
                print(f"   ⏱️  Time: {elapsed:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Comparison failed: {e}")
                results.append({
                    'pair': f"{model1_name} vs {model2_name}",
                    'error': str(e)
                })
    
    return results


def analyze_results(results):
    """Analyze the test results to check if the fixes are working."""
    print(f"\n📊 DTW Fix Analysis")
    print("=" * 60)
    
    successful_results = [r for r in results if 'error' not in r]
    
    if len(successful_results) < 2:
        print("❌ Not enough successful comparisons to analyze sensitivity")
        return
    
    similarities = [r['similarity_score'] for r in successful_results]
    
    print(f"📈 Similarity Scores:")
    for result in successful_results:
        print(f"   • {result['pair']}: {result['similarity_score']:.6f}")
    
    # Check for sensitivity
    similarity_range = max(similarities) - min(similarities)
    print(f"\n🎯 Sensitivity Analysis:")
    print(f"   • Range of similarities: {similarity_range:.6f}")
    print(f"   • Min similarity: {min(similarities):.6f}")
    print(f"   • Max similarity: {max(similarities):.6f}")
    
    if similarity_range > 0.1:
        print("   ✅ GOOD: DTW shows meaningful sensitivity to model differences")
    elif similarity_range > 0.01:
        print("   ⚠️  MODERATE: Some sensitivity, but could be better")
    else:
        print("   ❌ POOR: Still low sensitivity - may need further fixes")
    
    # Check if we're still getting the problematic ~10.46 values
    avg_similarity = np.mean(similarities)
    if 10.4 < avg_similarity < 10.5:
        print("   ❌ CRITICAL: Still getting ~10.46 values - fixes may not be working")
    else:
        print("   ✅ GOOD: No longer converging to problematic ~10.46 range")
    
    print(f"   📊 Average similarity: {avg_similarity:.6f}")


def main():
    """Main execution function."""
    print("🚀 Testing DTW Sensitivity Fixes")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    try:
        # Load test models
        models, descriptions = load_test_models()
        
        if len(models) < 2:
            print("❌ Need at least 2 models to test DTW sensitivity")
            return 1
        
        # Test DTW sensitivity
        results = test_dtw_sensitivity(models, descriptions)
        
        # Analyze results
        analyze_results(results)
        
        print(f"\n✅ DTW sensitivity test complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
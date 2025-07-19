#!/usr/bin/env python3
"""
Final MLP DTW Test

This script tests the corrected DTW implementation using the two MLP models
with the correct API configuration.
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
from neurosheaf.utils.dtw_similarity import FiltrationDTW


# Import model architecture from test_all.py
class MLPModel(nn.Module):
    """MLP model architecture matching the configuration."""
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


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🔧 Environment Setup Complete")


def load_mlp_models():
    """Load the two MLP models."""
    models_dir = Path("models")
    
    model1_path = models_dir / "torch_mlp_acc_0.9857_epoch_100.pth"
    model2_path = models_dir / "torch_mlp_acc_1.0000_epoch_200.pth"
    
    if not model1_path.exists():
        raise FileNotFoundError(f"Model 1 not found: {model1_path}")
    if not model2_path.exists():
        raise FileNotFoundError(f"Model 2 not found: {model2_path}")
    
    print(f"🔍 Loading MLP Models:")
    print(f"   • Model 1: {model1_path.name}")
    print(f"   • Model 2: {model2_path.name}")
    
    # Load both models
    model1 = load_model(MLPModel, model1_path, device='cpu')
    model2 = load_model(MLPModel, model2_path, device='cpu')
    
    print(f"   ✅ Both models loaded successfully")
    
    return model1, model2, model1_path.stem, model2_path.stem


def test_dtw_simple(model1, model2, model1_name, model2_name):
    """Test DTW with simple configuration."""
    print(f"\n🔬 Testing DTW (Simple Configuration)...")
    
    # Generate smaller test data for faster computation
    data = torch.randn(30, 3)  # Small dataset
    print(f"   Test data: {data.shape}")
    print(f"   Comparing {model1_name} vs {model2_name}")
    
    results = {}
    
    # Test 1: Single eigenvalue DTW with default analyzer
    print(f"\n   Test 1: Single Eigenvalue DTW")
    try:
        start_time = time.time()
        
        # Use default analyzer configuration
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=0,  # Just the largest eigenvalue
            multivariate=False
        )
        
        elapsed_time = time.time() - start_time
        
        similarity_score = result['similarity_score']
        results['single_eigenvalue'] = {
            'similarity_score': similarity_score,
            'time': elapsed_time,
            'result': result
        }
        
        print(f"      ✅ Similarity Score: {similarity_score:.4f}")
        print(f"      ⏱️  Time: {elapsed_time:.1f}s")
        
        # Print detailed DTW information
        if 'dtw_comparison' in result:
            dtw_info = result['dtw_comparison']
            print(f"      📊 DTW Details:")
            if isinstance(dtw_info, dict):
                for key, value in dtw_info.items():
                    if isinstance(value, (int, float, str, bool)):
                        print(f"         {key}: {value}")
        
        # Check if this shows high similarity as expected for same architecture
        if similarity_score > 0.8:
            print(f"      🎯 HIGH SIMILARITY: Models show strong functional similarity")
        elif similarity_score > 0.6:
            print(f"      ⚠️  MODERATE SIMILARITY: Some functional similarity detected")
        else:
            print(f"      ❌ LOW SIMILARITY: Unexpected low similarity for same architecture")
            
    except Exception as e:
        print(f"      ❌ Single eigenvalue DTW failed: {e}")
        results['single_eigenvalue'] = {'error': str(e)}
        import traceback
        traceback.print_exc()
    
    # Test 2: Second eigenvalue for comparison
    print(f"\n   Test 2: Second Eigenvalue DTW")
    try:
        start_time = time.time()
        
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=1,  # Second largest eigenvalue
            multivariate=False
        )
        
        elapsed_time = time.time() - start_time
        
        similarity_score = result['similarity_score']
        results['second_eigenvalue'] = {
            'similarity_score': similarity_score,
            'time': elapsed_time
        }
        
        print(f"      ✅ Similarity Score: {similarity_score:.4f}")
        print(f"      ⏱️  Time: {elapsed_time:.1f}s")
        
    except Exception as e:
        print(f"      ❌ Second eigenvalue DTW failed: {e}")
        results['second_eigenvalue'] = {'error': str(e)}
    
    # Test 3: Multivariate DTW (all eigenvalues)
    print(f"\n   Test 3: Multivariate DTW")
    try:
        start_time = time.time()
        
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=None,  # All eigenvalues
            multivariate=True
        )
        
        elapsed_time = time.time() - start_time
        
        similarity_score = result['similarity_score']
        results['multivariate'] = {
            'similarity_score': similarity_score,
            'time': elapsed_time
        }
        
        print(f"      ✅ Multivariate DTW: {similarity_score:.4f}")
        print(f"      ⏱️  Time: {elapsed_time:.1f}s")
        
        if similarity_score > 0.8:
            print(f"      🎯 HIGH MULTIVARIATE SIMILARITY: Strong functional alignment")
        
    except Exception as e:
        print(f"      ❌ Multivariate DTW failed: {e}")
        results['multivariate'] = {'error': str(e)}
        import traceback
        traceback.print_exc()
    
    return results


def test_comparison_methods(model1, model2, model1_name, model2_name):
    """Test other comparison methods for reference."""
    print(f"\n🔍 Testing Other Comparison Methods...")
    
    data = torch.randn(30, 3)
    results = {}
    
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Test euclidean distance
    try:
        result = analyzer.compare_networks(
            model1, model2, data, method='euclidean'
        )
        results['euclidean'] = result['similarity_score']
        print(f"   • Euclidean Distance: {result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   • Euclidean Distance: Failed ({e})")
        results['euclidean'] = None
    
    # Test cosine similarity  
    try:
        result = analyzer.compare_networks(
            model1, model2, data, method='cosine'
        )
        results['cosine'] = result['similarity_score']
        print(f"   • Cosine Similarity: {result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   • Cosine Similarity: Failed ({e})")
        results['cosine'] = None
    
    return results


def print_final_summary(dtw_results, comparison_results, model1_name, model2_name):
    """Print final summary of all results."""
    print(f"\n📊 FINAL DTW ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n🎯 **Models Tested** (Same Architecture - Expected High Similarity)")
    print(f"   • Model 1: {model1_name} (98.57% accuracy)")
    print(f"   • Model 2: {model2_name} (100.00% accuracy)")
    print(f"   • Architecture: MLPModel (8 hidden layers, 32 units)")
    print(f"   • Training: Same dataset, different epochs")
    
    print(f"\n🔬 **DTW Implementation Test Results:**")
    
    # DTW Results
    dtw_scores = []
    
    if 'single_eigenvalue' in dtw_results and 'error' not in dtw_results['single_eigenvalue']:
        score = dtw_results['single_eigenvalue']['similarity_score']
        time_taken = dtw_results['single_eigenvalue']['time']
        dtw_scores.append(score)
        print(f"   ✅ Single Eigenvalue DTW: {score:.4f} ({time_taken:.1f}s)")
    else:
        print(f"   ❌ Single Eigenvalue DTW: Failed")
    
    if 'second_eigenvalue' in dtw_results and 'error' not in dtw_results['second_eigenvalue']:
        score = dtw_results['second_eigenvalue']['similarity_score']
        time_taken = dtw_results['second_eigenvalue']['time']
        dtw_scores.append(score)
        print(f"   ✅ Second Eigenvalue DTW: {score:.4f} ({time_taken:.1f}s)")
    else:
        print(f"   ❌ Second Eigenvalue DTW: Failed")
    
    if 'multivariate' in dtw_results and 'error' not in dtw_results['multivariate']:
        score = dtw_results['multivariate']['similarity_score']
        time_taken = dtw_results['multivariate']['time']
        dtw_scores.append(score)
        print(f"   ✅ Multivariate DTW: {score:.4f} ({time_taken:.1f}s)")
    else:
        print(f"   ❌ Multivariate DTW: Failed")
    
    # Comparison with other methods
    print(f"\n🔍 **Reference Methods:**")
    if comparison_results.get('euclidean') is not None:
        print(f"   • Euclidean Distance: {comparison_results['euclidean']:.4f}")
    if comparison_results.get('cosine') is not None:
        print(f"   • Cosine Similarity: {comparison_results['cosine']:.4f}")
    
    # Overall assessment
    print(f"\n🎯 **FINAL ASSESSMENT:**")
    
    if dtw_scores:
        avg_dtw = np.mean(dtw_scores)
        std_dtw = np.std(dtw_scores)
        
        print(f"   📈 DTW Results Summary:")
        print(f"      • Average Similarity: {avg_dtw:.4f}")
        print(f"      • Standard Deviation: {std_dtw:.4f}")
        print(f"      • Range: [{min(dtw_scores):.4f}, {max(dtw_scores):.4f}]")
        
        if avg_dtw > 0.8:
            print(f"\n   ✅ SUCCESS: DTW IMPLEMENTATION WORKING CORRECTLY")
            print(f"      • High functional similarity detected ({avg_dtw:.4f})")
            print(f"      • Same-architecture models properly aligned")
            print(f"      • Both univariate and multivariate DTW functional")
            print(f"      • API fixes successful - all eigenvalue tracking works")
        elif avg_dtw > 0.6:
            print(f"\n   ⚠️  PARTIAL SUCCESS: DTW SHOWS MODERATE SIMILARITY")
            print(f"      • Moderate functional similarity ({avg_dtw:.4f})")
            print(f"      • DTW implementation working but may need parameter tuning")
        else:
            print(f"\n   ❌ UNEXPECTED: LOW SIMILARITY FOR SAME ARCHITECTURE")
            print(f"      • Lower than expected similarity ({avg_dtw:.4f})")
            print(f"      • May indicate issues with DTW parameters or data")
        
        if std_dtw < 0.1:
            print(f"      • CONSISTENT results across eigenvalue indices")
        else:
            print(f"      • VARIABLE results across eigenvalue indices")
    else:
        print(f"\n   ❌ CRITICAL FAILURE: NO SUCCESSFUL DTW COMPUTATIONS")
        print(f"      • DTW implementation has issues")
        print(f"      • API fixes may be incomplete")


def main():
    """Main execution function."""
    print("🚀 Final MLP DTW Implementation Test")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    try:
        # Load models
        model1, model2, model1_name, model2_name = load_mlp_models()
        
        # Test DTW
        dtw_results = test_dtw_simple(model1, model2, model1_name, model2_name)
        
        # Test comparison methods
        comparison_results = test_comparison_methods(model1, model2, model1_name, model2_name)
        
        # Print final summary
        print_final_summary(dtw_results, comparison_results, model1_name, model2_name)
        
        print(f"\n✅ DTW Implementation Test Complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
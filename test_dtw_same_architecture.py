#!/usr/bin/env python3
"""
Test DTW Implementation with Same Architecture Models

This script tests the corrected DTW implementation using two MLP models
with the same architecture trained on the same dataset. These should show
high functional similarity scores.

Focus: Testing both univariate and multivariate DTW for all eigenvalues.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.utils.dtw_similarity import FiltrationDTW, quick_dtw_comparison


# Import model architecture from test_all.py
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


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🔧 Environment Setup Complete")
    print(f"   PyTorch Version: {torch.__version__}")


def load_mlp_models() -> Dict[str, Any]:
    """Load the two specified MLP models."""
    models_dir = Path("models")
    
    model1_path = models_dir / "torch_mlp_acc_0.9857_epoch_100.pth"
    model2_path = models_dir / "torch_mlp_acc_1.0000_epoch_200.pth"
    
    if not model1_path.exists():
        raise FileNotFoundError(f"Model 1 not found: {model1_path}")
    if not model2_path.exists():
        raise FileNotFoundError(f"Model 2 not found: {model2_path}")
    
    print(f"🔍 Loading Same Architecture Models:")
    print(f"   • Model 1: {model1_path.name}")
    print(f"   • Model 2: {model2_path.name}")
    
    # Load both models
    model1 = load_model(MLPModel, model1_path, device='cpu')
    model2 = load_model(MLPModel, model2_path, device='cpu')
    
    print(f"   ✅ Both models loaded successfully")
    
    return {
        'model1': model1,
        'model2': model2,
        'model1_name': model1_path.stem,
        'model2_name': model2_path.stem
    }


def test_dtw_corrected_implementation(models: Dict[str, Any]) -> Dict[str, Any]:
    """Test the corrected DTW implementation."""
    print(f"\n🔬 Testing Corrected DTW Implementation...")
    
    # Generate test data
    data = torch.randn(100, 3)
    print(f"   Generated test data: {data.shape}")
    
    model1 = models['model1']
    model2 = models['model2']
    model1_name = models['model1_name']
    model2_name = models['model2_name']
    
    print(f"   Comparing {model1_name} vs {model2_name}")
    
    results = {}
    
    # Test 1: Pure Functional Similarity DTW (Corrected)
    print(f"\n   Test 1: Pure Functional DTW (Fixed Implementation)")
    try:
        start_time = time.time()
        
        # Create DTW comparator with corrected implementation
        pure_functional_dtw = FiltrationDTW(
            method='dtaidistance',
            constraint_band=0.1,
            eigenvalue_weight=1.0,     # Pure functional similarity
            structural_weight=0.0      # No structural penalty
        )
        
        # Initialize analyzer
        analyzer = NeurosheafAnalyzer(
            device='cpu', 
            enable_profiling=False
        )
        
        # Test univariate DTW for largest eigenvalue
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=0,  # Largest eigenvalue
            multivariate=False
        )
        
        elapsed_time = time.time() - start_time
        
        results['univariate_dtw'] = result
        print(f"      ✅ Univariate DTW (eigenvalue 0): {result['similarity_score']:.4f}")
        print(f"      ⏱️  Computation time: {elapsed_time:.2f}s")
        
        # Print detailed DTW information
        if 'dtw_comparison' in result:
            dtw_info = result['dtw_comparison']
            print(f"      📊 DTW Details:")
            if isinstance(dtw_info, dict) and 'distance' in dtw_info:
                print(f"         Distance: {dtw_info['distance']:.6f}")
                print(f"         Normalized Distance: {dtw_info.get('normalized_distance', 'N/A')}")
                print(f"         Method: {dtw_info.get('method', 'N/A')}")
        
    except Exception as e:
        print(f"      ❌ Univariate DTW failed: {e}")
        results['univariate_dtw'] = {'error': str(e)}
        import traceback
        traceback.print_exc()
    
    # Test 2: Multivariate DTW for all eigenvalues
    print(f"\n   Test 2: Multivariate DTW (All Eigenvalues)")
    try:
        start_time = time.time()
        
        # Use tslearn for multivariate DTW
        multivariate_dtw = FiltrationDTW(
            method='tslearn',
            constraint_band=0.1,
            eigenvalue_weight=1.0,
            structural_weight=0.0
        )
        
        analyzer = NeurosheafAnalyzer(
            device='cpu', 
            enable_profiling=False
        )
        
        # Test multivariate DTW
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=None,  # All eigenvalues
            multivariate=True       # Multivariate DTW
        )
        
        elapsed_time = time.time() - start_time
        
        results['multivariate_dtw'] = result
        print(f"      ✅ Multivariate DTW (all eigenvalues): {result['similarity_score']:.4f}")
        print(f"      ⏱️  Computation time: {elapsed_time:.2f}s")
        
        # Print detailed DTW information
        if 'dtw_comparison' in result:
            dtw_info = result['dtw_comparison']
            print(f"      📊 DTW Details:")
            if isinstance(dtw_info, dict) and 'distance' in dtw_info:
                print(f"         Distance: {dtw_info['distance']:.6f}")
                print(f"         Normalized Distance: {dtw_info.get('normalized_distance', 'N/A')}")
                print(f"         Method: {dtw_info.get('method', 'N/A')}")
                print(f"         Multivariate: {dtw_info.get('multivariate', 'N/A')}")
        
    except Exception as e:
        print(f"      ❌ Multivariate DTW failed: {e}")
        results['multivariate_dtw'] = {'error': str(e)}
        import traceback
        traceback.print_exc()
    
    # Test 3: Multiple eigenvalue indices individually
    print(f"\n   Test 3: Individual Eigenvalue DTW (Top 5)")
    eigenvalue_results = {}
    
    for eigenvalue_idx in range(5):  # Test top 5 eigenvalues
        try:
            start_time = time.time()
            
            result = analyzer.compare_networks(
                model1, model2, data, 
                method='dtw',
                eigenvalue_index=eigenvalue_idx,
                multivariate=False
            )
            
            elapsed_time = time.time() - start_time
            
            eigenvalue_results[f'eigenvalue_{eigenvalue_idx}'] = {
                'similarity_score': result['similarity_score'],
                'computation_time': elapsed_time
            }
            
            print(f"      ✅ Eigenvalue {eigenvalue_idx}: {result['similarity_score']:.4f} ({elapsed_time:.2f}s)")
            
        except Exception as e:
            print(f"      ❌ Eigenvalue {eigenvalue_idx} failed: {e}")
            eigenvalue_results[f'eigenvalue_{eigenvalue_idx}'] = {'error': str(e)}
    
    results['individual_eigenvalues'] = eigenvalue_results
    
    # Test 4: Compare with other methods
    print(f"\n   Test 4: Comparison with Other Methods")
    try:
        # Euclidean distance
        euclidean_result = analyzer.compare_networks(
            model1, model2, data, method='euclidean'
        )
        results['euclidean'] = euclidean_result
        print(f"      ✅ Euclidean Distance: {euclidean_result['similarity_score']:.4f}")
        
        # Cosine similarity
        cosine_result = analyzer.compare_networks(
            model1, model2, data, method='cosine'
        )
        results['cosine'] = cosine_result
        print(f"      ✅ Cosine Similarity: {cosine_result['similarity_score']:.4f}")
        
    except Exception as e:
        print(f"      ❌ Other methods failed: {e}")
        results['other_methods'] = {'error': str(e)}
    
    return results


def print_analysis_summary(results: Dict[str, Any], models: Dict[str, Any]):
    """Print comprehensive analysis summary."""
    print(f"\n📊 DTW Implementation Analysis Summary")
    print("=" * 60)
    
    print(f"\n🎯 **Test Models** (Same Architecture - Should Show High Similarity)")
    print(f"   • Model 1: {models['model1_name']}")
    print(f"   • Model 2: {models['model2_name']}")
    print(f"   • Architecture: MLPModel (8 hidden layers, 32 units)")
    print(f"   • Expected: High functional similarity (>0.8)")
    
    print(f"\n🔬 **DTW Test Results:**")
    
    # Univariate DTW
    if 'univariate_dtw' in results and 'error' not in results['univariate_dtw']:
        score = results['univariate_dtw']['similarity_score']
        print(f"   • Univariate DTW (eigenvalue 0): {score:.4f} {'✅' if score > 0.8 else '⚠️'}")
    else:
        print(f"   • Univariate DTW: ❌ Failed")
    
    # Multivariate DTW
    if 'multivariate_dtw' in results and 'error' not in results['multivariate_dtw']:
        score = results['multivariate_dtw']['similarity_score']
        print(f"   • Multivariate DTW (all eigenvalues): {score:.4f} {'✅' if score > 0.8 else '⚠️'}")
    else:
        print(f"   • Multivariate DTW: ❌ Failed")
    
    # Individual eigenvalues
    if 'individual_eigenvalues' in results:
        print(f"\n   **Individual Eigenvalue Analysis:**")
        eigenvalue_data = results['individual_eigenvalues']
        scores = []
        for key, result in eigenvalue_data.items():
            if 'error' not in result:
                score = result['similarity_score']
                time_taken = result['computation_time']
                eigenvalue_idx = key.replace('eigenvalue_', '')
                print(f"     - Eigenvalue {eigenvalue_idx}: {score:.4f} ({time_taken:.2f}s)")
                scores.append(score)
        
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"     - Average: {avg_score:.4f} ± {std_score:.4f}")
            print(f"     - Consistency: {'✅ Good' if std_score < 0.1 else '⚠️ Variable'}")
    
    # Comparison with other methods
    if 'euclidean' in results and 'error' not in results['euclidean']:
        euclidean_score = results['euclidean']['similarity_score']
        print(f"\n🔍 **Method Comparison:**")
        print(f"   • Euclidean Distance: {euclidean_score:.4f}")
        
        if 'cosine' in results and 'error' not in results['cosine']:
            cosine_score = results['cosine']['similarity_score']
            print(f"   • Cosine Similarity: {cosine_score:.4f}")
    
    # Overall assessment
    print(f"\n🎯 **Overall Assessment:**")
    
    dtw_scores = []
    if 'univariate_dtw' in results and 'error' not in results['univariate_dtw']:
        dtw_scores.append(results['univariate_dtw']['similarity_score'])
    if 'multivariate_dtw' in results and 'error' not in results['multivariate_dtw']:
        dtw_scores.append(results['multivariate_dtw']['similarity_score'])
    
    if dtw_scores:
        avg_dtw_score = np.mean(dtw_scores)
        if avg_dtw_score > 0.8:
            print(f"   ✅ HIGH FUNCTIONAL SIMILARITY: {avg_dtw_score:.4f}")
            print(f"      The corrected DTW implementation successfully detects")
            print(f"      functional similarity between same-architecture models.")
        elif avg_dtw_score > 0.6:
            print(f"   ⚠️  MODERATE FUNCTIONAL SIMILARITY: {avg_dtw_score:.4f}")
            print(f"      Some similarity detected, but lower than expected for same architecture.")
        else:
            print(f"   ❌ LOW FUNCTIONAL SIMILARITY: {avg_dtw_score:.4f}")
            print(f"      DTW implementation may still have issues.")
    else:
        print(f"   ❌ DTW IMPLEMENTATION FAILED")
        print(f"      No successful DTW computations completed.")


def main():
    """Main execution function."""
    print("🚀 DTW Implementation Test: Same Architecture Models")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    try:
        # Step 1: Load the two MLP models
        models = load_mlp_models()
        
        # Step 2: Test DTW implementation
        results = test_dtw_corrected_implementation(models)
        
        # Step 3: Print comprehensive analysis
        print_analysis_summary(results, models)
        
        # Step 4: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"dtw_same_architecture_test_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: str(v) for k, v in value.items()}
                else:
                    json_results[key] = str(value)
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"\n✅ DTW Implementation Test Complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
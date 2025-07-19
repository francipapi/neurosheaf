#!/usr/bin/env python3
"""
Focused MLP DTW Test

This script tests the corrected DTW implementation using the two MLP models
with reduced complexity to ensure it completes in reasonable time.
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


def test_dtw_with_reduced_complexity(model1, model2, model1_name, model2_name):
    """Test DTW with reduced analysis complexity for faster execution."""
    print(f"\n🔬 Testing DTW with Reduced Complexity...")
    
    # Generate smaller test data for faster computation
    data = torch.randn(50, 3)  # Reduced from 100 to 50
    print(f"   Test data: {data.shape}")
    print(f"   Comparing {model1_name} vs {model2_name}")
    
    results = {}
    
    # Configure analyzer with reduced complexity
    analyzer_config = {
        'device': 'cpu',
        'enable_profiling': False,
        'spectral_analyzer_config': {
            'filtration_steps': 10,  # Reduced from default 50 to 10
            'eigenvalue_threshold': 1e-6,
            'max_eigenvalues': 10  # Limit eigenvalues for faster computation
        }
    }
    
    # Test 1: Single eigenvalue DTW
    print(f"\n   Test 1: Single Eigenvalue DTW (Fastest)")
    try:
        start_time = time.time()
        
        analyzer = NeurosheafAnalyzer(**analyzer_config)
        
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
            'time': elapsed_time
        }
        
        print(f"      ✅ Similarity Score: {similarity_score:.4f}")
        print(f"      ⏱️  Time: {elapsed_time:.1f}s")
        
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
    
    # Test 2: Top 3 eigenvalues individually
    print(f"\n   Test 2: Top 3 Eigenvalues Individually")
    eigenvalue_results = {}
    
    for eigenvalue_idx in range(3):
        try:
            start_time = time.time()
            
            analyzer = NeurosheafAnalyzer(**analyzer_config)
            
            result = analyzer.compare_networks(
                model1, model2, data, 
                method='dtw',
                eigenvalue_index=eigenvalue_idx,
                multivariate=False
            )
            
            elapsed_time = time.time() - start_time
            
            similarity_score = result['similarity_score']
            eigenvalue_results[f'eigenvalue_{eigenvalue_idx}'] = {
                'similarity_score': similarity_score,
                'time': elapsed_time
            }
            
            print(f"      ✅ Eigenvalue {eigenvalue_idx}: {similarity_score:.4f} ({elapsed_time:.1f}s)")
            
        except Exception as e:
            print(f"      ❌ Eigenvalue {eigenvalue_idx} failed: {e}")
            eigenvalue_results[f'eigenvalue_{eigenvalue_idx}'] = {'error': str(e)}
    
    results['individual_eigenvalues'] = eigenvalue_results
    
    # Test 3: Multivariate DTW (if we have time)
    print(f"\n   Test 3: Multivariate DTW (All Eigenvalues)")
    try:
        start_time = time.time()
        
        # Use tslearn for multivariate
        multivariate_config = analyzer_config.copy()
        multivariate_config['spectral_analyzer_config']['dtw_method'] = 'tslearn'
        
        analyzer = NeurosheafAnalyzer(**multivariate_config)
        
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
        
        print(f"      ✅ Multivariate DTW: {similarity_score:.4f} ({elapsed_time:.1f}s)")
        
        if similarity_score > 0.8:
            print(f"      🎯 HIGH MULTIVARIATE SIMILARITY: Strong functional alignment")
        
    except Exception as e:
        print(f"      ❌ Multivariate DTW failed: {e}")
        results['multivariate'] = {'error': str(e)}
    
    return results


def print_final_analysis(results, model1_name, model2_name):
    """Print final analysis of DTW results."""
    print(f"\n📊 DTW Analysis Results")
    print("=" * 50)
    
    print(f"\n🎯 **Models Tested** (Same Architecture - Expected High Similarity)")
    print(f"   • Model 1: {model1_name}")
    print(f"   • Model 2: {model2_name}")
    print(f"   • Architecture: MLPModel (8 hidden layers, 32 units)")
    
    print(f"\n🔬 **DTW Test Results:**")
    
    # Single eigenvalue result
    if 'single_eigenvalue' in results and 'error' not in results['single_eigenvalue']:
        single_result = results['single_eigenvalue']
        score = single_result['similarity_score']
        time_taken = single_result['time']
        
        print(f"   • Single Eigenvalue DTW: {score:.4f} ({time_taken:.1f}s)")
        
        if score > 0.8:
            print(f"     ✅ EXCELLENT: Strong functional similarity detected")
        elif score > 0.6:
            print(f"     ⚠️  GOOD: Moderate functional similarity")
        else:
            print(f"     ❌ POOR: Lower than expected for same architecture")
    else:
        print(f"   • Single Eigenvalue DTW: ❌ Failed")
    
    # Individual eigenvalues
    if 'individual_eigenvalues' in results:
        print(f"\n   **Individual Eigenvalue Analysis:**")
        eigenvalue_data = results['individual_eigenvalues']
        scores = []
        
        for key, result in eigenvalue_data.items():
            if 'error' not in result:
                score = result['similarity_score']
                time_taken = result['time']
                eigenvalue_idx = key.replace('eigenvalue_', '')
                print(f"     - Eigenvalue {eigenvalue_idx}: {score:.4f} ({time_taken:.1f}s)")
                scores.append(score)
        
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"     - Average: {avg_score:.4f} ± {std_score:.4f}")
            
            if avg_score > 0.8:
                print(f"     ✅ CONSISTENT HIGH SIMILARITY across eigenvalues")
            elif std_score < 0.1:
                print(f"     ✅ CONSISTENT similarity pattern")
            else:
                print(f"     ⚠️  VARIABLE similarity across eigenvalues")
    
    # Multivariate result
    if 'multivariate' in results and 'error' not in results['multivariate']:
        multivariate_result = results['multivariate']
        score = multivariate_result['similarity_score']
        time_taken = multivariate_result['time']
        
        print(f"\n   • Multivariate DTW (all eigenvalues): {score:.4f} ({time_taken:.1f}s)")
        
        if score > 0.8:
            print(f"     ✅ EXCELLENT: Strong multivariate functional similarity")
    else:
        print(f"\n   • Multivariate DTW: ❌ Failed or skipped")
    
    # Overall assessment
    print(f"\n🎯 **Overall Assessment:**")
    
    all_scores = []
    if 'single_eigenvalue' in results and 'error' not in results['single_eigenvalue']:
        all_scores.append(results['single_eigenvalue']['similarity_score'])
    
    if 'individual_eigenvalues' in results:
        for result in results['individual_eigenvalues'].values():
            if 'error' not in result:
                all_scores.append(result['similarity_score'])
    
    if 'multivariate' in results and 'error' not in results['multivariate']:
        all_scores.append(results['multivariate']['similarity_score'])
    
    if all_scores:
        avg_similarity = np.mean(all_scores)
        
        if avg_similarity > 0.8:
            print(f"   ✅ SUCCESS: DTW correctly detects HIGH functional similarity ({avg_similarity:.4f})")
            print(f"      Same-architecture models show strong DTW alignment")
            print(f"      The corrected DTW implementation is working properly")
        elif avg_similarity > 0.6:
            print(f"   ⚠️  PARTIAL: DTW detects MODERATE functional similarity ({avg_similarity:.4f})")
            print(f"      Some similarity detected, analysis may need tuning")
        else:
            print(f"   ❌ UNEXPECTED: DTW shows LOW similarity ({avg_similarity:.4f})")
            print(f"      Lower than expected for identical architectures")
    else:
        print(f"   ❌ FAILED: No successful DTW computations")


def main():
    """Main execution function."""
    print("🚀 Focused MLP DTW Test")
    print("=" * 40)
    
    # Setup
    setup_environment()
    
    try:
        # Load models
        model1, model2, model1_name, model2_name = load_mlp_models()
        
        # Test DTW with reduced complexity
        results = test_dtw_with_reduced_complexity(model1, model2, model1_name, model2_name)
        
        # Print final analysis
        print_final_analysis(results, model1_name, model2_name)
        
        print(f"\n✅ Focused DTW Test Complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Cross-Architecture DTW Test

This script tests DTW functional similarity between different architectures:
- MLP (torch_mlp_acc_1.0000_epoch_200.pth)
- ActualCustomModel (torch_custom_acc_1.0000_epoch_200.pth)

Both models achieve 100% accuracy on the same dataset but have completely different architectures.
This tests DTW's ability to measure pure functional similarity across architectures.
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


# Import model architectures from test_all.py
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


class ActualCustomModel(nn.Module):
    """Model class that matches the actual saved weights structure with Conv1D layers."""
    
    def __init__(self):
        super().__init__()
        
        # Based on the actual saved model structure:
        # Linear layers followed by Conv1D layers with reshape operations
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


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🔧 Environment Setup Complete")


def load_cross_architecture_models():
    """Load the two different architecture models."""
    models_dir = Path("models")
    
    mlp_path = models_dir / "torch_mlp_acc_1.0000_epoch_200.pth"
    custom_path = models_dir / "torch_custom_acc_1.0000_epoch_200.pth"
    
    if not mlp_path.exists():
        raise FileNotFoundError(f"MLP model not found: {mlp_path}")
    if not custom_path.exists():
        raise FileNotFoundError(f"Custom model not found: {custom_path}")
    
    print(f"🔍 Loading Cross-Architecture Models:")
    print(f"   • MLP Model: {mlp_path.name}")
    print(f"   • Custom Model: {custom_path.name}")
    print(f"   • Both achieve 100% accuracy on same dataset")
    
    # Load both models
    mlp_model = load_model(MLPModel, mlp_path, device='cpu')
    custom_model = load_model(ActualCustomModel, custom_path, device='cpu')
    
    print(f"   ✅ Both models loaded successfully")
    
    return mlp_model, custom_model, mlp_path.stem, custom_path.stem


def analyze_architecture_differences(mlp_model, custom_model):
    """Analyze the architectural differences between the two models."""
    print(f"\n🏗️  Architecture Analysis:")
    
    # Count parameters
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    custom_params = sum(p.numel() for p in custom_model.parameters())
    
    print(f"   📊 Parameter Count:")
    print(f"      • MLP Model: {mlp_params:,} parameters")
    print(f"      • Custom Model: {custom_params:,} parameters")
    print(f"      • Difference: {abs(mlp_params - custom_params):,} parameters")
    
    print(f"\n   🏗️  Architecture Types:")
    print(f"      • MLP Model: Pure feedforward (Linear + ReLU + Dropout)")
    print(f"      • Custom Model: Hybrid (Linear + Conv1D + Reshape operations)")
    
    print(f"\n   🎯 Functional Similarity Expectation:")
    print(f"      • Both models achieve 100% accuracy on same task")
    print(f"      • Different architectures should show interesting DTW patterns")
    print(f"      • Pure functional similarity (no structural penalty) is key")


def test_cross_architecture_dtw(mlp_model, custom_model, mlp_name, custom_name):
    """Test DTW functional similarity across different architectures."""
    print(f"\n🔬 Testing Cross-Architecture DTW...")
    
    # Generate test data
    data = torch.randn(30, 3)  # Small dataset for reasonable compute time
    print(f"   Test data: {data.shape}")
    print(f"   Comparing {mlp_name} (MLP) vs {custom_name} (Custom)")
    
    results = {}
    
    # Test 1: Single eigenvalue DTW (functional similarity focus)
    print(f"\n   Test 1: Single Eigenvalue DTW (Pure Functional)")
    try:
        start_time = time.time()
        
        # Use pure functional similarity (no structural penalty)
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        result = analyzer.compare_networks(
            mlp_model, custom_model, data, 
            method='dtw',
            eigenvalue_index=0,  # Largest eigenvalue
            multivariate=False
        )
        
        elapsed_time = time.time() - start_time
        
        similarity_score = result['similarity_score']
        results['single_eigenvalue'] = {
            'similarity_score': similarity_score,
            'time': elapsed_time,
            'result': result
        }
        
        print(f"      ✅ Cross-Architecture Similarity: {similarity_score:.4f}")
        print(f"      ⏱️  Time: {elapsed_time:.1f}s")
        
        # Interpret cross-architecture similarity
        if similarity_score > 0.8:
            print(f"      🎯 HIGH FUNCTIONAL SIMILARITY: Despite different architectures!")
        elif similarity_score > 0.6:
            print(f"      ⚠️  MODERATE FUNCTIONAL SIMILARITY: Some shared behavior patterns")
        elif similarity_score > 0.4:
            print(f"      📊 LOW-MODERATE SIMILARITY: Limited functional overlap")
        else:
            print(f"      ❌ LOW FUNCTIONAL SIMILARITY: Very different behavioral patterns")
            
    except Exception as e:
        print(f"      ❌ Single eigenvalue DTW failed: {e}")
        results['single_eigenvalue'] = {'error': str(e)}
        import traceback
        traceback.print_exc()
    
    # Test 2: Multiple eigenvalue analysis
    print(f"\n   Test 2: Multiple Eigenvalue Analysis")
    eigenvalue_results = {}
    
    for eigenvalue_idx in range(3):  # Test top 3 eigenvalues
        try:
            start_time = time.time()
            
            analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
            
            result = analyzer.compare_networks(
                mlp_model, custom_model, data, 
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
    
    # Test 3: Multivariate DTW (if computational resources allow)
    print(f"\n   Test 3: Multivariate DTW (All Eigenvalues)")
    try:
        start_time = time.time()
        
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        result = analyzer.compare_networks(
            mlp_model, custom_model, data, 
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
        
        if similarity_score > 8.0:
            print(f"      📊 HIGH-DIMENSIONAL ALIGNMENT: Strong multivariate patterns")
        
    except Exception as e:
        print(f"      ❌ Multivariate DTW failed: {e}")
        results['multivariate'] = {'error': str(e)}
    
    return results


def test_reference_methods(mlp_model, custom_model, mlp_name, custom_name):
    """Test other comparison methods for reference."""
    print(f"\n🔍 Testing Reference Methods...")
    
    data = torch.randn(30, 3)
    results = {}
    
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Test euclidean distance
    try:
        result = analyzer.compare_networks(
            mlp_model, custom_model, data, method='euclidean'
        )
        results['euclidean'] = result['similarity_score']
        print(f"   • Euclidean Distance: {result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   • Euclidean Distance: Failed ({e})")
        results['euclidean'] = None
    
    # Test cosine similarity  
    try:
        result = analyzer.compare_networks(
            mlp_model, custom_model, data, method='cosine'
        )
        results['cosine'] = result['similarity_score']
        print(f"   • Cosine Similarity: {result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   • Cosine Similarity: Failed ({e})")
        results['cosine'] = None
    
    return results


def interpret_cross_architecture_results(dtw_results, reference_results, mlp_name, custom_name):
    """Interpret the cross-architecture DTW results."""
    print(f"\n📊 CROSS-ARCHITECTURE DTW ANALYSIS")
    print("=" * 70)
    
    print(f"\n🎯 **Models Compared** (Different Architectures - Functional Similarity Test)")
    print(f"   • MLP Model: {mlp_name}")
    print(f"     - Architecture: 8 hidden layers, pure feedforward")
    print(f"     - Components: Linear + ReLU + Dropout")
    print(f"   • Custom Model: {custom_name}")
    print(f"     - Architecture: Hybrid Linear + Conv1D + Reshape")
    print(f"     - Components: Linear → Conv1D sequence")
    print(f"   • Performance: Both achieve 100% accuracy on same dataset")
    
    print(f"\n🔬 **Cross-Architecture DTW Results:**")
    
    # DTW Results
    dtw_scores = []
    
    if 'single_eigenvalue' in dtw_results and 'error' not in dtw_results['single_eigenvalue']:
        score = dtw_results['single_eigenvalue']['similarity_score']
        time_taken = dtw_results['single_eigenvalue']['time']
        dtw_scores.append(score)
        print(f"   ✅ Single Eigenvalue DTW: {score:.4f} ({time_taken:.1f}s)")
    else:
        print(f"   ❌ Single Eigenvalue DTW: Failed")
    
    # Individual eigenvalues analysis
    if 'individual_eigenvalues' in dtw_results:
        print(f"\n   **Eigenvalue-by-Eigenvalue Analysis:**")
        eigenvalue_scores = []
        
        for key, result in dtw_results['individual_eigenvalues'].items():
            if 'error' not in result:
                score = result['similarity_score']
                time_taken = result['time']
                eigenvalue_idx = key.replace('eigenvalue_', '')
                eigenvalue_scores.append(score)
                print(f"      • Eigenvalue {eigenvalue_idx}: {score:.4f} ({time_taken:.1f}s)")
        
        if eigenvalue_scores:
            avg_eigenvalue = np.mean(eigenvalue_scores)
            std_eigenvalue = np.std(eigenvalue_scores)
            print(f"      • Average: {avg_eigenvalue:.4f} ± {std_eigenvalue:.4f}")
            dtw_scores.extend(eigenvalue_scores)
    
    # Multivariate results
    if 'multivariate' in dtw_results and 'error' not in dtw_results['multivariate']:
        score = dtw_results['multivariate']['similarity_score']
        time_taken = dtw_results['multivariate']['time']
        dtw_scores.append(score)
        print(f"\n   ✅ Multivariate DTW: {score:.4f} ({time_taken:.1f}s)")
    
    # Reference methods
    print(f"\n🔍 **Reference Methods:**")
    if reference_results.get('euclidean') is not None:
        print(f"   • Euclidean Distance: {reference_results['euclidean']:.4f}")
    if reference_results.get('cosine') is not None:
        print(f"   • Cosine Similarity: {reference_results['cosine']:.4f}")
    
    # Overall interpretation
    print(f"\n🎯 **CROSS-ARCHITECTURE INTERPRETATION:**")
    
    if dtw_scores:
        # Filter out extreme multivariate values for more meaningful interpretation
        filtered_scores = [s for s in dtw_scores if s <= 2.0]  # Focus on similarity scores
        
        if filtered_scores:
            avg_similarity = np.mean(filtered_scores)
            std_similarity = np.std(filtered_scores)
            
            print(f"   📈 DTW Functional Similarity Summary:")
            print(f"      • Average Similarity: {avg_similarity:.4f}")
            print(f"      • Standard Deviation: {std_similarity:.4f}")
            print(f"      • Range: [{min(filtered_scores):.4f}, {max(filtered_scores):.4f}]")
            
            print(f"\n   🧠 **Functional Similarity Interpretation:**")
            
            if avg_similarity > 0.8:
                print(f"   ✅ REMARKABLE: HIGH FUNCTIONAL SIMILARITY ACROSS ARCHITECTURES")
                print(f"      • Despite completely different architectures (MLP vs Conv1D)")
                print(f"      • Both models converged to similar functional behaviors")
                print(f"      • DTW successfully detected deep functional equivalence")
                print(f"      • Pure functional similarity measurement working perfectly")
                
            elif avg_similarity > 0.6:
                print(f"   ⚠️  SIGNIFICANT: MODERATE FUNCTIONAL SIMILARITY")
                print(f"      • Different architectures show meaningful functional overlap")
                print(f"      • Some shared computational patterns despite structural differences")
                print(f"      • DTW detects partial functional convergence")
                
            elif avg_similarity > 0.4:
                print(f"   📊 PARTIAL: LIMITED FUNCTIONAL SIMILARITY")
                print(f"      • Some functional overlap but significant differences")
                print(f"      • Architectural differences create distinct computational paths")
                print(f"      • DTW detects architectural influence on function")
                
            else:
                print(f"   ❌ DISTINCT: LOW FUNCTIONAL SIMILARITY")
                print(f"      • Architectures lead to very different functional behaviors")
                print(f"      • Despite same task performance, computational approaches differ")
                print(f"      • DTW correctly identifies functional divergence")
            
            print(f"\n   🔬 **Research Implications:**")
            print(f"      • DTW enables fair comparison across architectures")
            print(f"      • Pure functional similarity (weight 1.0, 0.0) eliminates structural bias")
            print(f"      • Eigenvalue evolution captures deep computational patterns")
            print(f"      • Method can identify functional convergence vs architectural influence")
            
            if std_similarity < 0.1:
                print(f"      • CONSISTENT functional patterns across eigenvalue dimensions")
            else:
                print(f"      • VARIABLE functional patterns across eigenvalue dimensions")
    else:
        print(f"\n   ❌ CRITICAL: NO SUCCESSFUL CROSS-ARCHITECTURE DTW ANALYSIS")
        print(f"      • Cannot assess functional similarity across architectures")
        print(f"      • Implementation may need architecture-specific adjustments")


def main():
    """Main execution function."""
    print("🚀 Cross-Architecture DTW Functional Similarity Test")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    try:
        # Load models
        mlp_model, custom_model, mlp_name, custom_name = load_cross_architecture_models()
        
        # Analyze architecture differences
        analyze_architecture_differences(mlp_model, custom_model)
        
        # Test cross-architecture DTW
        dtw_results = test_cross_architecture_dtw(mlp_model, custom_model, mlp_name, custom_name)
        
        # Test reference methods
        reference_results = test_reference_methods(mlp_model, custom_model, mlp_name, custom_name)
        
        # Interpret results
        interpret_cross_architecture_results(dtw_results, reference_results, mlp_name, custom_name)
        
        print(f"\n✅ Cross-Architecture DTW Test Complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
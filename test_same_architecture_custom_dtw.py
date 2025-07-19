#!/usr/bin/env python3
"""
Same Architecture Custom DTW Test

This script tests DTW functional similarity between two ActualCustomModel architectures:
- random_custom_net_000_default_seed_42.pth (random/untrained)
- torch_custom_acc_1.0000_epoch_200.pth (trained to 100% accuracy)

Both models have identical ActualCustomModel architecture but very different training states.
This tests how training quality affects functional similarity within the same architecture.
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


def load_same_architecture_models():
    """Load the two ActualCustomModel architectures with different training states."""
    models_dir = Path("models")
    
    random_path = models_dir / "random_custom_net_000_default_seed_42.pth"
    trained_path = models_dir / "torch_custom_acc_1.0000_epoch_200.pth"
    
    if not random_path.exists():
        raise FileNotFoundError(f"Random model not found: {random_path}")
    if not trained_path.exists():
        raise FileNotFoundError(f"Trained model not found: {trained_path}")
    
    print(f"🔍 Loading Same Architecture Models (ActualCustomModel):")
    print(f"   • Random Model: {random_path.name}")
    print(f"   • Trained Model: {trained_path.name}")
    print(f"   • Architecture: Both use ActualCustomModel (Linear + Conv1D)")
    
    # Load both models
    random_model = load_model(ActualCustomModel, random_path, device='cpu')
    trained_model = load_model(ActualCustomModel, trained_path, device='cpu')
    
    print(f"   ✅ Both models loaded successfully")
    
    return random_model, trained_model, random_path.stem, trained_path.stem


def analyze_training_differences(random_model, trained_model):
    """Analyze the training state differences between the two models."""
    print(f"\n🎯 Training State Analysis:")
    
    # Count parameters
    random_params = sum(p.numel() for p in random_model.parameters())
    trained_params = sum(p.numel() for p in trained_model.parameters())
    
    print(f"   📊 Parameter Count:")
    print(f"      • Random Model: {random_params:,} parameters")
    print(f"      • Trained Model: {trained_params:,} parameters")
    print(f"      • Architecture: Identical (ActualCustomModel)")
    
    print(f"\n   🎓 Training State:")
    print(f"      • Random Model: Untrained/random weights (seed 42)")
    print(f"      • Trained Model: 100% accuracy after 200 epochs")
    print(f"      • Task: Same dataset, same objective")
    
    print(f"\n   🎯 Expected DTW Results:")
    print(f"      • Should show LOW functional similarity")
    print(f"      • Random weights vs optimized weights = very different behavior")
    print(f"      • Same architecture but completely different function")
    print(f"      • This tests DTW's sensitivity to training quality")


def test_training_state_dtw(random_model, trained_model, random_name, trained_name):
    """Test DTW functional similarity between random and trained states."""
    print(f"\n🔬 Testing Training State DTW...")
    
    # Generate test data
    data = torch.randn(200, 3)  # Small dataset for reasonable compute time
    print(f"   Test data: {data.shape}")
    print(f"   Comparing {random_name} (Random) vs {trained_name} (Trained)")
    
    results = {}
    
    # Test 1: Single eigenvalue DTW (should show low similarity)
    print(f"\n   Test 1: Single Eigenvalue DTW (Random vs Trained)")
    try:
        start_time = time.time()
        
        # Use pure functional similarity (no structural penalty)
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        result = analyzer.compare_networks(
            random_model, trained_model, data, 
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
        
        print(f"      ✅ Random vs Trained Similarity: {similarity_score:.4f}")
        print(f"      ⏱️  Time: {elapsed_time:.1f}s")
        
        # Interpret training state similarity
        if similarity_score > 0.8:
            print(f"      🤔 UNEXPECTED HIGH SIMILARITY: Random and trained show similar behavior")
            print(f"         This could indicate:")
            print(f"         - The task is very simple")
            print(f"         - Random initialization was already good")
            print(f"         - Architecture constrains function strongly")
        elif similarity_score > 0.5:
            print(f"      ⚠️  MODERATE SIMILARITY: Some shared patterns")
            print(f"         - Architecture provides some functional bias")
            print(f"         - Training refined but didn't completely change behavior")
        elif similarity_score > 0.2:
            print(f"      📊 LOW SIMILARITY: Training significantly changed behavior")
            print(f"         - Expected result: training optimized the function")
            print(f"         - DTW correctly detects functional differences")
        else:
            print(f"      ✅ VERY LOW SIMILARITY: Training completely transformed function")
            print(f"         - Strong evidence of learning and optimization")
            print(f"         - Random vs trained are functionally very different")
            
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
                random_model, trained_model, data, 
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
    
    # Test 3: Multivariate DTW 
    print(f"\n   Test 3: Multivariate DTW (All Eigenvalues)")
    try:
        start_time = time.time()
        
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        result = analyzer.compare_networks(
            random_model, trained_model, data, 
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
        
    except Exception as e:
        print(f"      ❌ Multivariate DTW failed: {e}")
        results['multivariate'] = {'error': str(e)}
    
    return results


def test_reference_methods(random_model, trained_model, random_name, trained_name):
    """Test other comparison methods for reference."""
    print(f"\n🔍 Testing Reference Methods...")
    
    data = torch.randn(30, 3)
    results = {}
    
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Test euclidean distance
    try:
        result = analyzer.compare_networks(
            random_model, trained_model, data, method='euclidean'
        )
        results['euclidean'] = result['similarity_score']
        print(f"   • Euclidean Distance: {result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   • Euclidean Distance: Failed ({e})")
        results['euclidean'] = None
    
    # Test cosine similarity  
    try:
        result = analyzer.compare_networks(
            random_model, trained_model, data, method='cosine'
        )
        results['cosine'] = result['similarity_score']
        print(f"   • Cosine Similarity: {result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   • Cosine Similarity: Failed ({e})")
        results['cosine'] = None
    
    return results


def interpret_training_state_results(dtw_results, reference_results, random_name, trained_name):
    """Interpret the training state DTW results."""
    print(f"\n📊 TRAINING STATE DTW ANALYSIS")
    print("=" * 70)
    
    print(f"\n🎯 **Models Compared** (Same Architecture - Training Quality Test)")
    print(f"   • Random Model: {random_name}")
    print(f"     - State: Untrained/random weights (seed 42)")
    print(f"     - Performance: Unknown (likely poor)")
    print(f"   • Trained Model: {trained_name}")
    print(f"     - State: Trained to 100% accuracy (200 epochs)")
    print(f"     - Performance: Optimized for task")
    print(f"   • Architecture: Both use ActualCustomModel (identical structure)")
    
    print(f"\n🔬 **Training State DTW Results:**")
    
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
    print(f"\n🎯 **TRAINING STATE INTERPRETATION:**")
    
    if dtw_scores:
        # Filter out extreme multivariate values for more meaningful interpretation
        filtered_scores = [s for s in dtw_scores if s <= 2.0]  # Focus on similarity scores
        
        if filtered_scores:
            avg_similarity = np.mean(filtered_scores)
            std_similarity = np.std(filtered_scores)
            
            print(f"   📈 DTW Training State Summary:")
            print(f"      • Average Similarity: {avg_similarity:.4f}")
            print(f"      • Standard Deviation: {std_similarity:.4f}")
            print(f"      • Range: [{min(filtered_scores):.4f}, {max(filtered_scores):.4f}]")
            
            print(f"\n   🧠 **Training Impact Interpretation:**")
            
            if avg_similarity > 0.8:
                print(f"   🤔 SURPRISING: HIGH SIMILARITY DESPITE TRAINING DIFFERENCE")
                print(f"      • Random initialization was already near-optimal")
                print(f"      • Task may be very simple for this architecture")
                print(f"      • Architecture strongly constrains functional behavior")
                print(f"      • Training didn't significantly change eigenvalue evolution")
                
            elif avg_similarity > 0.5:
                print(f"   ⚠️  MODERATE: PARTIAL TRAINING IMPACT")
                print(f"      • Training changed behavior but didn't completely transform it")
                print(f"      • Architecture provides some functional constraints")
                print(f"      • Random weights had some useful patterns")
                
            elif avg_similarity > 0.2:
                print(f"   ✅ EXPECTED: LOW SIMILARITY SHOWS TRAINING IMPACT")
                print(f"      • Training significantly optimized functional behavior")
                print(f"      • Random vs trained weights produce different eigenvalue evolution")
                print(f"      • DTW correctly detects the impact of optimization")
                print(f"      • Clear evidence of learning and functional improvement")
                
            else:
                print(f"   ✅ STRONG: VERY LOW SIMILARITY SHOWS MAJOR TRAINING IMPACT")
                print(f"      • Training completely transformed functional behavior")
                print(f"      • Random weights produce completely different spectral patterns")
                print(f"      • Maximum evidence of learning and optimization")
                print(f"      • DTW successfully distinguishes trained vs untrained states")
            
            print(f"\n   🔬 **Research Implications:**")
            print(f"      • DTW can measure the functional impact of training")
            print(f"      • Eigenvalue evolution captures optimization effects")
            print(f"      • Method can distinguish random vs trained functional states")
            print(f"      • Pure functional similarity reveals training quality")
            
            if std_similarity < 0.1:
                print(f"      • CONSISTENT training impact across eigenvalue dimensions")
            else:
                print(f"      • VARIABLE training impact across eigenvalue dimensions")
    else:
        print(f"\n   ❌ CRITICAL: NO SUCCESSFUL TRAINING STATE DTW ANALYSIS")
        print(f"      • Cannot assess functional impact of training")


def compare_with_previous_tests(dtw_results):
    """Compare results with previous tests for context."""
    print(f"\n📋 **Comparison with Previous DTW Tests:**")
    
    # Extract average similarity for comparison
    dtw_scores = []
    if 'single_eigenvalue' in dtw_results and 'error' not in dtw_results['single_eigenvalue']:
        dtw_scores.append(dtw_results['single_eigenvalue']['similarity_score'])
    
    if 'individual_eigenvalues' in dtw_results:
        for result in dtw_results['individual_eigenvalues'].values():
            if 'error' not in result:
                dtw_scores.append(result['similarity_score'])
    
    if dtw_scores:
        current_avg = np.mean([s for s in dtw_scores if s <= 2.0])
        
        print(f"\n   📊 **DTW Similarity Comparison:**")
        print(f"      • Same Architecture MLPs: ~0.983 (very high)")
        print(f"      • Cross Architecture (MLP vs Custom): ~0.980 (very high)")
        print(f"      • Same Architecture (Random vs Trained): {current_avg:.3f}")
        print(f"")
        
        if current_avg > 0.8:
            print(f"   🔍 **Pattern Analysis:**")
            print(f"      All three tests show HIGH similarity!")
            print(f"      • Same architecture + same training → 0.983")
            print(f"      • Different architecture + same training → 0.980") 
            print(f"      • Same architecture + different training → {current_avg:.3f}")
            print(f"")
            print(f"   💡 **Key Insight:**")
            print(f"      Architecture seems to be more important than training state")
            print(f"      for eigenvalue evolution patterns in this task.")
            
        elif current_avg > 0.5:
            print(f"   🔍 **Pattern Analysis:**")
            print(f"      Training state has MODERATE impact on functional similarity")
            print(f"      • Architecture provides strong constraints (0.98 cross-arch similarity)")
            print(f"      • Training refines but doesn't transform behavior")
            
        else:
            print(f"   🔍 **Pattern Analysis:**")
            print(f"      Training state has MAJOR impact on functional similarity")
            print(f"      • Same/different architecture matters less than training quality")
            print(f"      • Random vs optimized weights create very different functions")


def main():
    """Main execution function."""
    print("🚀 Same Architecture Training State DTW Test")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    try:
        # Load models
        random_model, trained_model, random_name, trained_name = load_same_architecture_models()
        
        # Analyze training differences
        analyze_training_differences(random_model, trained_model)
        
        # Test training state DTW
        dtw_results = test_training_state_dtw(random_model, trained_model, random_name, trained_name)
        
        # Test reference methods
        reference_results = test_reference_methods(random_model, trained_model, random_name, trained_name)
        
        # Interpret results
        interpret_training_state_results(dtw_results, reference_results, random_name, trained_name)
        
        # Compare with previous tests
        compare_with_previous_tests(dtw_results)
        
        print(f"\n✅ Training State DTW Test Complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
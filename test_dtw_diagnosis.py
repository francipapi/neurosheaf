#!/usr/bin/env python3
"""
DTW Over-Similarity Diagnosis Script

This script investigates why DTW shows high similarity between trained and untrained networks.
We'll examine the eigenvalue evolution patterns, DTW computation, and normalization.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.utils.dtw_similarity import FiltrationDTW


class MLPModel(nn.Module):
    """Standard MLP model for testing."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)


def analyze_eigenvalue_patterns(evolution1, evolution2, title1="Model 1", title2="Model 2"):
    """Analyze and visualize eigenvalue evolution patterns."""
    print(f"\n🔍 Analyzing Eigenvalue Patterns: {title1} vs {title2}")
    
    # Extract largest eigenvalue sequences
    seq1 = [eigenvals[0].item() if len(eigenvals) > 0 else 0.0 for eigenvals in evolution1]
    seq2 = [eigenvals[0].item() if len(eigenvals) > 0 else 0.0 for eigenvals in evolution2]
    
    print(f"   {title1} eigenvalues: {len(seq1)} steps")
    print(f"   {title2} eigenvalues: {len(seq2)} steps")
    print(f"   {title1} range: [{min(seq1):.6f}, {max(seq1):.6f}]")
    print(f"   {title2} range: [{min(seq2):.6f}, {max(seq2):.6f}]")
    
    # Check if sequences are nearly identical
    seq1_arr = np.array(seq1)
    seq2_arr = np.array(seq2)
    
    if len(seq1_arr) == len(seq2_arr):
        mse = np.mean((seq1_arr - seq2_arr) ** 2)
        max_diff = np.max(np.abs(seq1_arr - seq2_arr))
        correlation = np.corrcoef(seq1_arr, seq2_arr)[0, 1]
        
        print(f"   MSE between sequences: {mse:.8f}")
        print(f"   Max absolute difference: {max_diff:.8f}")
        print(f"   Correlation: {correlation:.6f}")
        
        if mse < 1e-10:
            print(f"   ⚠️  WARNING: Sequences are nearly identical (MSE < 1e-10)")
        if correlation > 0.99:
            print(f"   ⚠️  WARNING: Very high correlation (> 0.99)")
    
    # Check for constant sequences
    if np.std(seq1_arr) < 1e-10:
        print(f"   ⚠️  WARNING: {title1} sequence is nearly constant (std < 1e-10)")
    if np.std(seq2_arr) < 1e-10:
        print(f"   ⚠️  WARNING: {title2} sequence is nearly constant (std < 1e-10)")
    
    return seq1_arr, seq2_arr


def test_dtw_computation_details(seq1, seq2, title="DTW Test"):
    """Test DTW computation with detailed analysis."""
    print(f"\n🧮 DTW Computation Analysis: {title}")
    
    # Manual DTW distance computation
    try:
        from dtaidistance import dtw
        raw_distance = dtw.distance(seq1.astype(np.float64), seq2.astype(np.float64))
        print(f"   Raw DTW distance: {raw_distance:.8f}")
        print(f"   Sequence 1 length: {len(seq1)}")
        print(f"   Sequence 2 length: {len(seq2)}")
        
        # Current normalization
        current_norm = raw_distance / max(len(seq1), len(seq2))
        print(f"   Current normalization (/ max_len): {current_norm:.8f}")
        
        # Alternative normalizations
        sum_norm = raw_distance / (len(seq1) + len(seq2))
        print(f"   Sum normalization (/ sum_len): {sum_norm:.8f}")
        
        path_norm = raw_distance / (len(seq1) * len(seq2))
        print(f"   Path normalization (/ product): {path_norm:.8f}")
        
        # Range-based normalization
        range1 = max(seq1) - min(seq1)
        range2 = max(seq2) - min(seq2)
        avg_range = (range1 + range2) / 2
        if avg_range > 0:
            range_norm = raw_distance / avg_range
            print(f"   Range normalization (/ avg_range): {range_norm:.8f}")
            print(f"   Sequence ranges: {range1:.8f}, {range2:.8f}")
        
        # Euclidean baseline for comparison
        if len(seq1) == len(seq2):
            euclidean_dist = np.sqrt(np.sum((seq1 - seq2) ** 2))
            print(f"   Euclidean distance: {euclidean_dist:.8f}")
            print(f"   DTW vs Euclidean ratio: {raw_distance / euclidean_dist:.3f}")
        
    except Exception as e:
        print(f"   Error in DTW computation: {e}")


def test_different_model_pairs():
    """Test DTW on different model pairs to understand over-similarity."""
    print("\n🚀 Testing DTW on Different Model Configurations")
    print("=" * 60)
    
    # Create models with different training states
    torch.manual_seed(42)
    random_model = MLPModel()
    
    torch.manual_seed(123)
    different_random_model = MLPModel()
    
    # Create a "trained" model by manually setting specific patterns
    trained_model = MLPModel()
    with torch.no_grad():
        for name, param in trained_model.named_parameters():
            if 'weight' in name:
                # Set to specific patterns that should create different eigenvalue evolution
                param.data = torch.randn_like(param) * 0.1 + 0.5
            elif 'bias' in name:
                param.data = torch.randn_like(param) * 0.05
    
    # Generate test data
    data = torch.randn(100, 3)
    
    # Create analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Test cases
    test_cases = [
        (random_model, random_model, "Same Model (Identity)", "Should be 0.0 distance"),
        (random_model, different_random_model, "Different Random Init", "Should show some difference"),
        (random_model, trained_model, "Random vs Manual", "Should show significant difference"),
    ]
    
    for model1, model2, name, expectation in test_cases:
        print(f"\n🧪 Test Case: {name}")
        print(f"   Expected: {expectation}")
        
        try:
            # Get eigenvalue evolutions directly
            analysis1 = analyzer.analyze(model1, data)
            analysis2 = analyzer.analyze(model2, data)
            
            sheaf1 = analysis1['sheaf']
            sheaf2 = analysis2['sheaf']
            
            # Get detailed eigenvalue evolution
            spectral_result1 = analyzer.spectral_analyzer.analyze(
                sheaf1, filtration_type='threshold', n_steps=20
            )
            spectral_result2 = analyzer.spectral_analyzer.analyze(
                sheaf2, filtration_type='threshold', n_steps=20
            )
            
            evolution1 = spectral_result1['persistence_result']['eigenvalue_sequences']
            evolution2 = spectral_result2['persistence_result']['eigenvalue_sequences']
            
            # Analyze patterns
            seq1_arr, seq2_arr = analyze_eigenvalue_patterns(
                evolution1, evolution2, f"{name} - Model1", f"{name} - Model2"
            )
            
            # Test DTW computation
            test_dtw_computation_details(seq1_arr, seq2_arr, name)
            
            # Compare with API result
            api_result = analyzer.compare_networks(
                model1, model2, data, method='dtw', n_steps=20
            )
            print(f"   API similarity score: {api_result['similarity_score']:.6f}")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()


def test_synthetic_sequences():
    """Test DTW on synthetic sequences with known properties."""
    print(f"\n🧬 Testing DTW on Synthetic Sequences")
    print("=" * 50)
    
    # Create synthetic eigenvalue-like sequences
    n_steps = 50
    
    # Test case 1: Identical sequences
    seq_identical = np.exp(-np.linspace(0, 5, n_steps))  # Exponential decay
    
    # Test case 2: Shifted sequence
    seq_shifted = np.roll(seq_identical, 5)
    
    # Test case 3: Scaled sequence
    seq_scaled = seq_identical * 0.5
    
    # Test case 4: Different pattern
    seq_different = np.sin(np.linspace(0, 4*np.pi, n_steps)) * 0.5 + 1.0
    
    # Test case 5: Noisy sequence
    np.random.seed(42)
    seq_noisy = seq_identical + np.random.normal(0, 0.1, n_steps)
    
    synthetic_tests = [
        (seq_identical, seq_identical, "Identical", "Should be 0.0"),
        (seq_identical, seq_shifted, "Time Shifted", "Should be low (DTW handles shifts)"),
        (seq_identical, seq_scaled, "Amplitude Scaled", "Should be moderate"),
        (seq_identical, seq_different, "Different Pattern", "Should be high"),
        (seq_identical, seq_noisy, "With Noise", "Should be low-moderate"),
    ]
    
    for seq1, seq2, name, expectation in synthetic_tests:
        print(f"\n🔬 Synthetic Test: {name}")
        print(f"   Expected: {expectation}")
        test_dtw_computation_details(seq1, seq2, name)


def main():
    """Main diagnostic function."""
    print("🔍 DTW Over-Similarity Investigation")
    print("=" * 80)
    print("This script investigates why DTW shows high similarity between")
    print("trained and untrained networks when we expect differences.")
    
    # Test 1: Real model pairs
    test_different_model_pairs()
    
    # Test 2: Synthetic sequences
    test_synthetic_sequences()
    
    print(f"\n📊 Summary of Potential Issues Found:")
    print("=" * 50)
    print("1. Check console output for warnings about:")
    print("   - Nearly identical eigenvalue sequences")
    print("   - Very high correlations (> 0.99)")
    print("   - Constant eigenvalue sequences")
    print("   - Poor normalization ratios")
    print("\n2. Look for patterns in the data that might explain over-similarity:")
    print("   - All models producing similar eigenvalue evolution")
    print("   - Current normalization being too aggressive")
    print("   - DTW being too permissive in alignment")
    
    print(f"\n💡 Next Steps Based on Findings:")
    print("   - If sequences are nearly identical → investigate sheaf construction")
    print("   - If normalization is poor → implement better normalization")
    print("   - If DTW is too permissive → add stricter constraints")
    print("   - If patterns differ but DTW doesn't detect → improve distance metric")


if __name__ == "__main__":
    main()
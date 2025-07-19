#!/usr/bin/env python3
"""
Test Improved DTW Implementation
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer


class SimpleModel(nn.Module):
    def __init__(self, weight_scale=1.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        # Scale weights to create different models
        with torch.no_grad():
            for param in self.parameters():
                param.data *= weight_scale
    
    def forward(self, x):
        return self.layers(x)


def test_improved_sensitivity():
    """Test if the improved DTW is more sensitive to model differences."""
    print("🧪 Testing Improved DTW Sensitivity")
    print("=" * 50)
    
    # Create models with different characteristics
    torch.manual_seed(42)
    model1 = SimpleModel(weight_scale=1.0)
    
    torch.manual_seed(123)
    model2 = SimpleModel(weight_scale=2.0)  # Different weights
    
    torch.manual_seed(456)
    model3 = SimpleModel(weight_scale=0.1)  # Very different weights
    
    data = torch.randn(30, 3)
    
    # Test with improved DTW settings
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    test_cases = [
        (model1, model1, "Same Model", "Should be ~1.0"),
        (model1, model2, "Different Scale (2x)", "Should be < 1.0"),
        (model1, model3, "Very Different (0.1x)", "Should be much < 1.0"),
    ]
    
    results = []
    
    for model_a, model_b, name, expected in test_cases:
        print(f"\n🔬 Test: {name}")
        print(f"   Expected: {expected}")
        
        try:
            # Test with reduced steps for faster execution
            result = analyzer.compare_networks(
                model_a, model_b, data,
                method='dtw',
                filtration_type='threshold',
                n_steps=15,  # Reduced for speed
                eigenvalue_index=0
            )
            
            similarity = result['similarity_score']
            results.append((name, similarity))
            
            print(f"   Similarity: {similarity:.4f}")
            
            if similarity >= 0.95:
                print(f"   Status: ✅ High similarity (expected for same/similar models)")
            elif similarity >= 0.7:
                print(f"   Status: ⚠️  Moderate similarity")
            elif similarity >= 0.5:
                print(f"   Status: 📊 Low similarity (good for different models)")
            else:
                print(f"   Status: 🎯 Very low similarity (excellent for very different models)")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append((name, -1.0))
    
    # Summary
    print(f"\n📊 Summary of Results:")
    print("=" * 40)
    
    for name, similarity in results:
        if similarity >= 0:
            print(f"   {name:25}: {similarity:.4f}")
        else:
            print(f"   {name:25}: Failed")
    
    # Check if we have proper gradation
    if len(results) >= 3 and all(r[1] >= 0 for r in results):
        same_sim = results[0][1]
        diff_sim = results[1][1]
        very_diff_sim = results[2][1]
        
        print(f"\n🎯 Sensitivity Analysis:")
        if same_sim > diff_sim > very_diff_sim:
            print("   ✅ EXCELLENT: DTW shows proper sensitivity gradation")
            print("      Same > Different > Very Different")
        elif same_sim > diff_sim:
            print("   ⚠️  PARTIAL: DTW shows some sensitivity")
            print("      Same > Different, but very different needs improvement")
        elif same_sim == diff_sim == very_diff_sim:
            print("   ❌ POOR: DTW shows no sensitivity (all similarities equal)")
        else:
            print("   🤔 UNEXPECTED: Unusual similarity pattern")
        
        print(f"   Sensitivity range: {very_diff_sim:.3f} to {same_sim:.3f}")
        print(f"   Dynamic range: {same_sim - very_diff_sim:.3f}")
    
    return results


def test_zero_sequence_handling():
    """Test how the improved DTW handles zero sequences."""
    print(f"\n🔍 Testing Zero Sequence Handling")
    print("=" * 40)
    
    # Create a very simple model that might produce near-zero eigenvalues
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 1)
            with torch.no_grad():
                self.layer.weight.data = torch.tensor([[0.01, 0.01]])
                self.layer.bias.data = torch.tensor([0.0])
        
        def forward(self, x):
            return self.layer(x)
    
    model_tiny = TinyModel()
    data_tiny = torch.randn(10, 2)
    
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    try:
        result = analyzer.compare_networks(
            model_tiny, model_tiny, data_tiny,
            method='dtw',
            n_steps=10
        )
        
        print(f"   Tiny model self-comparison: {result['similarity_score']:.6f}")
        print("   (Should be close to 1.0 even with potential zero eigenvalues)")
        
    except Exception as e:
        print(f"   Zero sequence test failed: {e}")


def main():
    print("🚀 Testing Improved DTW Implementation")
    print("=" * 60)
    
    # Test sensitivity improvements
    results = test_improved_sensitivity()
    
    # Test zero sequence handling
    test_zero_sequence_handling()
    
    print(f"\n🎉 Improved DTW Testing Complete!")
    print("\nKey improvements implemented:")
    print("1. ✅ Enhanced normalization schemes")
    print("2. ✅ Zero sequence detection and penalties")  
    print("3. ✅ Better filtration parameter range")
    print("4. ✅ Multiple sensitivity checks")


if __name__ == "__main__":
    main()
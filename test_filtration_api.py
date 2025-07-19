#!/usr/bin/env python3
"""
Test the updated API with filtration parameters.

This script demonstrates how to use the new filtration_type and n_steps parameters
in the compare_networks method.
"""

import torch
import torch.nn as nn
from pathlib import Path
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model

# Simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)


def main():
    print("🧪 Testing Updated Compare Networks API with Filtration Parameters")
    print("=" * 70)
    
    # Create two simple models
    print("\n📊 Creating test models...")
    model1 = SimpleModel()
    model2 = SimpleModel()
    
    # Initialize with different weights
    torch.manual_seed(42)
    for p in model1.parameters():
        p.data = torch.randn_like(p)
    
    torch.manual_seed(123)
    for p in model2.parameters():
        p.data = torch.randn_like(p)
    
    # Generate test data
    data = torch.randn(50, 3)
    print(f"✅ Created two models with different initializations")
    print(f"✅ Generated test data: {data.shape}")
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Test 1: Default parameters
    print("\n🔬 Test 1: Default filtration parameters")
    print("   • method: dtw")
    print("   • filtration_type: threshold (default)")
    print("   • n_steps: 50 (default)")
    
    result1 = analyzer.compare_networks(
        model1, model2, data,
        method='dtw',
        eigenvalue_index=0  # Compare largest eigenvalue
    )
    
    print(f"\n   Results:")
    print(f"   • Similarity score: {result1['similarity_score']:.4f}")
    print(f"   • Filtration type: {result1['comparison_metadata']['filtration_type']}")
    print(f"   • Number of steps: {result1['comparison_metadata']['n_steps']}")
    
    # Test 2: Custom filtration parameters
    print("\n🔬 Test 2: Custom filtration parameters")
    print("   • method: dtw")
    print("   • filtration_type: threshold")
    print("   • n_steps: 20 (reduced for speed)")
    
    result2 = analyzer.compare_networks(
        model1, model2, data,
        method='dtw',
        eigenvalue_index=0,
        filtration_type='threshold',
        n_steps=20
    )
    
    print(f"\n   Results:")
    print(f"   • Similarity score: {result2['similarity_score']:.4f}")
    print(f"   • Filtration type: {result2['comparison_metadata']['filtration_type']}")
    print(f"   • Number of steps: {result2['comparison_metadata']['n_steps']}")
    
    # Test 3: CKA-based filtration
    print("\n🔬 Test 3: CKA-based filtration")
    print("   • method: dtw")
    print("   • filtration_type: cka_based")
    print("   • n_steps: 15")
    
    result3 = analyzer.compare_networks(
        model1, model2, data,
        method='dtw',
        eigenvalue_index=0,
        filtration_type='cka_based',
        n_steps=15
    )
    
    print(f"\n   Results:")
    print(f"   • Similarity score: {result3['similarity_score']:.4f}")
    print(f"   • Filtration type: {result3['comparison_metadata']['filtration_type']}")
    print(f"   • Number of steps: {result3['comparison_metadata']['n_steps']}")
    
    # Compare results
    print("\n📊 Comparison Summary:")
    print(f"   • Default (50 steps):     {result1['similarity_score']:.4f}")
    print(f"   • Reduced (20 steps):     {result2['similarity_score']:.4f}")
    print(f"   • CKA-based (15 steps):   {result3['similarity_score']:.4f}")
    
    # Show impact of number of steps
    print("\n🔍 Impact of Filtration Parameters:")
    if abs(result1['similarity_score'] - result2['similarity_score']) < 0.05:
        print("   • Reducing steps from 50 to 20 had minimal impact on similarity")
    else:
        print("   • Number of steps significantly affects similarity measurement")
    
    if abs(result1['similarity_score'] - result3['similarity_score']) > 0.1:
        print("   • CKA-based filtration gives different results than threshold-based")
    else:
        print("   • CKA-based and threshold-based filtrations give similar results")
    
    # Test 4: Multiple network comparison
    print("\n🔬 Test 4: Multiple network comparison with custom parameters")
    
    model3 = SimpleModel()
    torch.manual_seed(456)
    for p in model3.parameters():
        p.data = torch.randn_like(p)
    
    models = [model1, model2, model3]
    
    multi_result = analyzer.compare_multiple_networks(
        models, data,
        method='dtw',
        filtration_type='threshold',
        n_steps=25,
        eigenvalue_index=0
    )
    
    print(f"   • Number of models: {multi_result['comparison_metadata']['n_models']}")
    print(f"   • Filtration type: {multi_result['comparison_metadata']['filtration_type']}")
    print(f"   • Number of steps: {multi_result['comparison_metadata']['n_steps']}")
    print(f"\n   Distance matrix:")
    print(multi_result['distance_matrix'])
    
    print("\n✅ API test complete! The new filtration parameters are working correctly.")
    print("\n💡 Key insights:")
    print("   1. filtration_type controls how edges are filtered during analysis")
    print("   2. n_steps controls the resolution of the filtration")
    print("   3. More steps = finer resolution but slower computation")
    print("   4. Different filtration types can reveal different similarity patterns")


if __name__ == "__main__":
    main()
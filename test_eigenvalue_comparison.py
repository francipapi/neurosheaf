#!/usr/bin/env python3
"""
Quick comparison test to verify preserve_eigenvalues flag produces different results.
"""
import torch
import torch.nn as nn 
import numpy as np
import os

# Set environment for CPU usage
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.directed_sheaf import DirectedSheafAdapter

# Simple test model matching test_all_directed_simple.py structure
class MLPModel(nn.Module):
    def __init__(self, input_dim=3, num_hidden_layers=8, hidden_dim=32, output_dim=1):
        super().__init__()
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(nn.ReLU())
        
        # Output layer
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        layers_list.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.layers(x)

def run_comparison():
    """Compare eigenvalue evolution with preserve_eigenvalues True vs False."""
    print("=== Eigenvalue Preservation Comparison Test ===")
    
    # Create model and data
    model = MLPModel()
    data = torch.randn(50, 3)  # Same as test_all_directed_simple.py
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu')
    spectral_analyzer = PersistentSpectralAnalyzer(default_n_steps=20)  # Reduced for speed
    adapter = DirectedSheafAdapter()
    
    results = {}
    
    for preserve_eigenvalues in [True, False]:
        print(f"\n--- Running with preserve_eigenvalues={preserve_eigenvalues} ---")
        
        # Build directed sheaf
        analysis = analyzer.analyze(
            model, 
            data, 
            directed=True, 
            directionality_parameter=0.25,
            preserve_eigenvalues=preserve_eigenvalues
        )
        
        directed_sheaf = analysis['directed_sheaf']
        
        # Check base sheaf eigenvalue metadata
        base_sheaf = directed_sheaf.base_sheaf
        print(f"Base sheaf eigenvalue_metadata.preserve_eigenvalues: {base_sheaf.eigenvalue_metadata.preserve_eigenvalues if base_sheaf.eigenvalue_metadata else None}")
        
        # Check base sheaf stalks
        identity_count = 0
        eigenvalue_count = 0
        for node_id, stalk in base_sheaf.stalks.items():
            if torch.allclose(stalk, torch.eye(stalk.shape[0]), atol=1e-6):
                identity_count += 1
            else:
                eigenvalue_count += 1
        
        print(f"Base sheaf stalks: {identity_count} identity, {eigenvalue_count} eigenvalue matrices")
        
        # Create compatibility sheaf for spectral analysis
        compatibility_sheaf = adapter.create_compatibility_sheaf(directed_sheaf)
        
        # Run spectral analysis
        spectral_results = spectral_analyzer.analyze(
            compatibility_sheaf,
            filtration_type='threshold',
            n_steps=20
        )
        
        # Extract first few eigenvalues from first and last steps
        eigenval_seqs = spectral_results['persistence_result']['eigenvalue_sequences']
        if eigenval_seqs and len(eigenval_seqs) > 0:
            first_step = eigenval_seqs[0][:5] if len(eigenval_seqs[0]) >= 5 else eigenval_seqs[0]
            last_step = eigenval_seqs[-1][:5] if len(eigenval_seqs[-1]) >= 5 else eigenval_seqs[-1]
            
            print(f"First 5 eigenvalues - First step: {first_step.tolist()}")
            print(f"First 5 eigenvalues - Last step: {last_step.tolist()}")
            
            results[preserve_eigenvalues] = {
                'first_step': first_step.tolist(),
                'last_step': last_step.tolist(),
                'identity_stalks': identity_count,
                'eigenvalue_stalks': eigenvalue_count
            }
        else:
            print("No eigenvalue sequences found!")
            results[preserve_eigenvalues] = None
    
    # Compare results
    print("\n=== COMPARISON RESULTS ===")
    
    if results[True] and results[False]:
        # Compare eigenvalue evolution
        first_diff = np.array(results[True]['first_step']) - np.array(results[False]['first_step'])
        last_diff = np.array(results[True]['last_step']) - np.array(results[False]['last_step'])
        
        print(f"First step eigenvalue differences: {first_diff}")
        print(f"Last step eigenvalue differences: {last_diff}")
        print(f"Max absolute difference (first): {np.max(np.abs(first_diff)):.6f}")
        print(f"Max absolute difference (last): {np.max(np.abs(last_diff)):.6f}")
        
        # Compare stalk types
        print(f"\nStalk comparison:")
        print(f"preserve_eigenvalues=True:  {results[True]['identity_stalks']} identity, {results[True]['eigenvalue_stalks']} eigenvalue")
        print(f"preserve_eigenvalues=False: {results[False]['identity_stalks']} identity, {results[False]['eigenvalue_stalks']} eigenvalue")
        
        # Check if results are different
        if np.max(np.abs(first_diff)) > 1e-6 or np.max(np.abs(last_diff)) > 1e-6:
            print("\n✅ SUCCESS: preserve_eigenvalues flag produces DIFFERENT eigenvalue evolution!")
        else:
            print("\n❌ PROBLEM: preserve_eigenvalues flag produces IDENTICAL eigenvalue evolution!")
        
        if results[True]['eigenvalue_stalks'] > 0 and results[False]['identity_stalks'] == len(model.layers):
            print("✅ SUCCESS: Stalk types are different as expected!")
        else:
            print("❌ PROBLEM: Stalk types are not as expected!")
    else:
        print("❌ Could not compare results - one or both analyses failed")

if __name__ == "__main__":
    run_comparison()
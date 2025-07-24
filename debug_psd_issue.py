#!/usr/bin/env python3
"""Debug script to understand PSD failure in GW Laplacian tests."""

import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

# Add the project to path
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder

def analyze_test_sheaf():
    """Recreate the failing test case and analyze the mathematical issue."""
    
    print("=== Debugging GW Laplacian PSD Failure ===\n")
    
    # Recreate the exact test sheaf from the failing test
    poset = nx.DiGraph()
    poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
    
    # Create identity stalks (typical for GW sheaves)
    stalks = {
        'layer1': torch.eye(5, dtype=torch.float64),
        'layer2': torch.eye(4, dtype=torch.float64),
        'layer3': torch.eye(3, dtype=torch.float64)
    }
    
    # Create column-stochastic restrictions (typical for GW)
    # layer1 -> layer2 (5 -> 4)
    torch.manual_seed(42)  # For reproducibility
    R_12 = torch.rand(4, 5, dtype=torch.float64)
    R_12 = R_12 / R_12.sum(dim=0, keepdim=True)  # Column-stochastic
    
    # layer2 -> layer3 (4 -> 3)
    R_23 = torch.rand(3, 4, dtype=torch.float64)
    R_23 = R_23 / R_23.sum(dim=0, keepdim=True)  # Column-stochastic
    
    restrictions = {
        ('layer1', 'layer2'): R_12,
        ('layer2', 'layer3'): R_23
    }
    
    # Create GW-specific metadata with exact test values
    gw_costs = {
        ('layer1', 'layer2'): 0.3,
        ('layer2', 'layer3'): 0.2
    }
    
    metadata = {
        'construction_method': 'gromov_wasserstein',
        'gw_costs': gw_costs,
        'gw_config': GWConfig().to_dict(),
        'whitened': False,
        'validation_passed': True
    }
    
    sheaf = Sheaf(
        poset=poset,
        stalks=stalks,
        restrictions=restrictions,
        metadata=metadata
    )
    
    print("Sheaf structure:")
    print(f"  Nodes: {list(poset.nodes())}")
    print(f"  Edges: {list(poset.edges())}")
    print(f"  Stalk dimensions: {[(n, s.shape[0]) for n, s in stalks.items()]}")
    print(f"  GW costs: {gw_costs}")
    
    print("\nRestriction properties:")
    for edge, R in restrictions.items():
        print(f"  {edge}: shape {R.shape}")
        print(f"    Column sums: {R.sum(dim=0).tolist()}")
        print(f"    Row sums: {R.sum(dim=1).tolist()}")
        print(f"    Operator 2-norm: {torch.linalg.norm(R, ord=2).item():.6f}")
        print(f"    Spectral radius: {torch.linalg.eigvals(R.T @ R).real.max().item():.6f}")
    
    # Build the Laplacian
    builder = GWLaplacianBuilder(validate_properties=True)
    laplacian = builder.build_laplacian(sheaf, sparse=False)
    L = laplacian.numpy()
    
    print(f"\nLaplacian properties:")
    print(f"  Shape: {L.shape}")
    print(f"  Symmetry error: {np.abs(L - L.T).max():.2e}")
    
    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(L)
    eigenvals = np.sort(eigenvals)
    print(f"  Eigenvalues:")
    for i, eigval in enumerate(eigenvals):
        print(f"    {i}: {eigval:.6f}")
    
    print(f"  Min eigenvalue: {eigenvals[0]:.6f}")
    print(f"  Spectral gap: {eigenvals[1]:.6f}")
    print(f"  PSD violation: {max(0, -eigenvals[0]):.6f}")
    
    # Analyze block structure
    print(f"\nAnalyzing block structure:")
    
    # Block indices: layer1 [0:5], layer2 [5:9], layer3 [9:12]
    layer1_idx = slice(0, 5)
    layer2_idx = slice(5, 9) 
    layer3_idx = slice(9, 12)
    
    # Extract blocks manually
    print("Diagonal blocks:")
    L11 = L[layer1_idx, layer1_idx]  # layer1 self-block
    L22 = L[layer2_idx, layer2_idx]  # layer2 self-block  
    L33 = L[layer3_idx, layer3_idx]  # layer3 self-block
    
    print(f"  L[layer1,layer1] eigenvalues: {np.linalg.eigvals(L11)}")
    print(f"  L[layer2,layer2] eigenvalues: {np.linalg.eigvals(L22)}")  
    print(f"  L[layer3,layer3] eigenvalues: {np.linalg.eigvals(L33)}")
    
    print("Off-diagonal blocks:")
    L12 = L[layer1_idx, layer2_idx]  # layer1 -> layer2
    L21 = L[layer2_idx, layer1_idx]  # layer2 -> layer1
    L23 = L[layer2_idx, layer3_idx]  # layer2 -> layer3
    L32 = L[layer3_idx, layer2_idx]  # layer3 -> layer2
    
    print(f"  L[layer1,layer2] vs -w*R12^T: {np.allclose(L12, -0.3 * R_12.T.numpy())}")
    print(f"  L[layer2,layer1] vs -w*R12: {np.allclose(L21, -0.3 * R_12.numpy())}")
    print(f"  L[layer2,layer3] vs -w*R23^T: {np.allclose(L23, -0.2 * R_23.T.numpy())}")
    print(f"  L[layer3,layer2] vs -w*R23: {np.allclose(L32, -0.2 * R_23.numpy())}")
    
    # Analyze the mathematical formula issue
    print(f"\nAnalyzing formula implementation:")
    
    # Current implementation uses: L[v,v] = Σ_{incoming} w²*I + Σ_{outgoing} w²*R^T*R
    # But this might be incorrect - let's check what should be there
    
    print("Expected contributions to diagonal blocks:")
    
    # Layer1 block: outgoing edge to layer2
    w12 = 0.3
    expected_L11 = w12**2 * (R_12.T @ R_12).numpy()
    print(f"  L11 expected (w²R^T*R): trace = {np.trace(expected_L11):.6f}")
    print(f"  L11 actual: trace = {np.trace(L11):.6f}")
    
    # Layer2 block: incoming from layer1, outgoing to layer3  
    w23 = 0.2
    expected_L22_incoming = w12**2 * np.eye(4)  # Identity from incoming edge
    expected_L22_outgoing = w23**2 * (R_23.T @ R_23).numpy()  # R^T*R from outgoing
    expected_L22_total = expected_L22_incoming + expected_L22_outgoing
    
    print(f"  L22 expected incoming (w²I): trace = {np.trace(expected_L22_incoming):.6f}")
    print(f"  L22 expected outgoing (w²R^T*R): trace = {np.trace(expected_L22_outgoing):.6f}")
    print(f"  L22 expected total: trace = {np.trace(expected_L22_total):.6f}")
    print(f"  L22 actual: trace = {np.trace(L22):.6f}")
    
    # Layer3 block: incoming from layer2
    expected_L33 = w23**2 * np.eye(3)
    print(f"  L33 expected (w²I): trace = {np.trace(expected_L33):.6f}")
    print(f"  L33 actual: trace = {np.trace(L33):.6f}")
    
    return L, eigenvals

if __name__ == "__main__":
    analyze_test_sheaf()
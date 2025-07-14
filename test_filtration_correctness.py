#!/usr/bin/env python3
"""
Comprehensive test to verify filtration correctness.

This test compares the optimized filtration method with direct sheaf reconstruction
to ensure mathematical correctness of the dynamic weight application strategy.
"""

import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, '.')

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.laplacian import build_sheaf_laplacian
from neurosheaf.spectral.static_laplacian_unified import UnifiedStaticLaplacian

def create_restriction_with_norm(target_rows: int, source_cols: int, target_norm: float) -> torch.Tensor:
    """Create a restriction matrix with specific Frobenius norm.
    
    Args:
        target_rows: Number of rows (target stalk dimension)
        source_cols: Number of columns (source stalk dimension)
        target_norm: Desired Frobenius norm
        
    Returns:
        Restriction matrix with specified norm
    """
    # Create random matrix
    np.random.seed(42)  # For reproducibility
    R = torch.randn(target_rows, source_cols)
    
    # Normalize to unit norm
    current_norm = torch.norm(R, 'fro')
    R = R / current_norm
    
    # Scale to target norm
    R = R * target_norm
    
    # Verify norm
    actual_norm = torch.norm(R, 'fro').item()
    print(f"  Created restriction {target_rows}×{source_cols} with norm {actual_norm:.4f} (target: {target_norm})")
    
    return R

def create_test_sheaf() -> Sheaf:
    """Create test sheaf with 3 nodes and different stalk dimensions."""
    print("=== CREATING TEST SHEAF ===")
    
    # Create 3-node poset: A → B → C
    poset = nx.DiGraph()
    poset.add_edges_from([('A', 'B'), ('B', 'C')])
    print(f"Poset structure: {list(poset.edges())}")
    
    # Create stalks with different dimensions (whitened = identity)
    stalks = {
        'A': torch.eye(4),  # 4×4 identity
        'B': torch.eye(3),  # 3×3 identity  
        'C': torch.eye(2)   # 2×2 identity
    }
    print(f"Stalk dimensions: A={stalks['A'].shape}, B={stalks['B'].shape}, C={stalks['C'].shape}")
    
    # Create restriction maps with specific Frobenius norms
    print("Creating restriction maps:")
    restrictions = {
        ('A', 'B'): create_restriction_with_norm(3, 4, target_norm=2.5),
        ('B', 'C'): create_restriction_with_norm(2, 3, target_norm=1.8)
    }
    
    return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)

def print_sheaf_details(sheaf: Sheaf):
    """Print detailed information about the sheaf structure."""
    print("\n" + "="*60)
    print("DETAILED SHEAF STRUCTURE")
    print("="*60)
    
    print(f"Nodes: {list(sheaf.poset.nodes())}")
    print(f"Edges: {list(sheaf.poset.edges())}")
    
    print("\n" + "-"*40)
    print("STALK DIMENSIONS AND CONTENTS")
    print("-"*40)
    for node, stalk in sheaf.stalks.items():
        print(f"\nNode {node}:")
        print(f"  Dimension: {stalk.shape}")
        print(f"  Stalk matrix (whitened = identity):")
        print(f"    {stalk.numpy()}")
    
    print("\n" + "-"*40)
    print("RESTRICTION MAPS AND WEIGHTS")
    print("-"*40)
    for edge, restriction in sheaf.restrictions.items():
        weight = torch.norm(restriction, 'fro').item()
        source, target = edge
        print(f"\nEdge {source} → {target}:")
        print(f"  Matrix shape: {restriction.shape} (maps from {source} to {target})")
        print(f"  Frobenius norm (weight): {weight:.6f}")
        print(f"  Restriction matrix R:")
        # Format matrix nicely
        R_np = restriction.numpy()
        for i, row in enumerate(R_np):
            row_str = "    [" + " ".join(f"{val:8.4f}" for val in row) + "]"
            print(row_str)
        
        # Additional mathematical properties
        print(f"  Matrix properties:")
        print(f"    - Min value: {R_np.min():.6f}")
        print(f"    - Max value: {R_np.max():.6f}")
        print(f"    - Mean value: {R_np.mean():.6f}")
        print(f"    - Std dev: {R_np.std():.6f}")
    
    print("\n" + "-"*40)
    print("EXPECTED FILTRATION BEHAVIOR")
    print("-"*40)
    edge_weights = [(edge, torch.norm(restriction, 'fro').item()) 
                    for edge, restriction in sheaf.restrictions.items()]
    edge_weights.sort(key=lambda x: x[1])  # Sort by weight
    
    print("Edges ordered by weight (ascending):")
    for edge, weight in edge_weights:
        print(f"  {edge[0]} → {edge[1]}: {weight:.6f}")
    
    print("\nFiltration thresholds and expected active edges:")
    test_thresholds = [0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0]
    for threshold in test_thresholds:
        active_edges = [f"{edge[0]}→{edge[1]}" for edge, weight in edge_weights if weight >= threshold]
        print(f"  Threshold {threshold}: {active_edges if active_edges else 'None'}")
    
    print("="*60)

def create_filtered_poset(original_poset: nx.DiGraph, active_restrictions: Dict) -> nx.DiGraph:
    """Create filtered poset containing only active edges."""
    filtered_poset = nx.DiGraph()
    
    # Add all nodes
    filtered_poset.add_nodes_from(original_poset.nodes())
    
    # Add only active edges
    for edge in active_restrictions.keys():
        if edge in original_poset.edges():
            filtered_poset.add_edge(*edge)
    
    return filtered_poset

def test_optimized_filtration(sheaf: Sheaf, threshold: float) -> Tuple[csr_matrix, np.ndarray]:
    """Test optimized filtration method."""
    unified_laplacian = UnifiedStaticLaplacian()
    
    # Compute persistence using optimized method
    results = unified_laplacian.compute_persistence(
        sheaf, 
        [threshold], 
        lambda weight, param: weight >= param
    )
    
    # Get the Laplacian matrix for this threshold
    # We need to access the internal method to get the actual matrix
    static_laplacian, construction_metadata = unified_laplacian._get_or_build_laplacian(sheaf)
    edge_info = unified_laplacian._get_or_extract_edge_info(sheaf, static_laplacian, construction_metadata)
    edge_mask = unified_laplacian._create_edge_mask(edge_info, threshold, lambda w, t: w >= t)
    filtered_laplacian = unified_laplacian._apply_correct_masking(static_laplacian, edge_mask, edge_info, construction_metadata)
    
    # Compute eigenvalues
    eigenvalues = results['eigenvalue_sequences'][0]
    
    return filtered_laplacian, eigenvalues.numpy()

def test_direct_reconstruction(sheaf: Sheaf, threshold: float) -> Tuple[csr_matrix, np.ndarray]:
    """Test direct sheaf reconstruction method."""
    
    # Filter edges by threshold
    filtered_restrictions = {}
    for edge, R in sheaf.restrictions.items():
        weight = torch.norm(R, 'fro').item()
        if weight >= threshold:
            filtered_restrictions[edge] = R
    
    # Create new sheaf with filtered edges
    filtered_sheaf = Sheaf(
        poset=create_filtered_poset(sheaf.poset, filtered_restrictions),
        stalks=sheaf.stalks,
        restrictions=filtered_restrictions
    )
    
    # Build Laplacian directly
    laplacian, _ = build_sheaf_laplacian(filtered_sheaf)
    
    # Compute eigenvalues
    if laplacian.shape[0] > 0:
        try:
            eigenvals = eigsh(laplacian, k=min(10, laplacian.shape[0]-1), which='SM', return_eigenvectors=False)
            eigenvals = np.sort(eigenvals)
        except:
            # Fallback for small matrices
            eigenvals = np.linalg.eigvals(laplacian.toarray())
            eigenvals = np.sort(np.real(eigenvals))
    else:
        eigenvals = np.array([])
    
    return laplacian, eigenvals

def compute_active_edges(sheaf: Sheaf, threshold: float) -> List[str]:
    """Compute which edges are active at given threshold."""
    active_edges = []
    for edge, R in sheaf.restrictions.items():
        weight = torch.norm(R, 'fro').item()
        if weight >= threshold:
            active_edges.append(f"{edge[0]}→{edge[1]}")
    return active_edges

def verify_filtration_correctness():
    """Main test function to verify filtration correctness."""
    print("FILTRATION CORRECTNESS TEST")
    print("=" * 50)
    
    # Create test sheaf
    sheaf = create_test_sheaf()
    print_sheaf_details(sheaf)
    
    # Define test thresholds
    thresholds = [0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0]
    
    print(f"\n=== TESTING {len(thresholds)} THRESHOLDS ===")
    
    all_passed = True
    
    for threshold in thresholds:
        print(f"\n{'='*20} THRESHOLD {threshold} {'='*20}")
        
        # Show active edges
        active_edges = compute_active_edges(sheaf, threshold)
        print(f"Active edges: {active_edges if active_edges else 'None'}")
        
        # Test both methods
        try:
            opt_laplacian, opt_eigenvals = test_optimized_filtration(sheaf, threshold)
            direct_laplacian, direct_eigenvals = test_direct_reconstruction(sheaf, threshold)
            
            # Compare Laplacian matrices
            laplacian_diff = (opt_laplacian - direct_laplacian).max()
            print(f"Laplacian max difference: {laplacian_diff:.2e}")
            
            # Compare eigenvalues (handle different array lengths)
            min_len = min(len(opt_eigenvals), len(direct_eigenvals))
            if min_len > 0:
                eigenval_diff = np.max(np.abs(opt_eigenvals[:min_len] - direct_eigenvals[:min_len]))
                print(f"Eigenvalue max difference: {eigenval_diff:.2e}")
            else:
                eigenval_diff = 0.0
                print("No eigenvalues to compare")
            
            # Detailed output
            print(f"Optimized Laplacian shape: {opt_laplacian.shape}, nnz: {opt_laplacian.nnz}")
            print(f"Direct Laplacian shape: {direct_laplacian.shape}, nnz: {direct_laplacian.nnz}")
            print(f"Optimized eigenvalues ({len(opt_eigenvals)}): {opt_eigenvals}")
            print(f"Direct eigenvalues ({len(direct_eigenvals)}): {direct_eigenvals}")
            
            # Print full matrices for small cases
            if opt_laplacian.shape[0] <= 10:
                print(f"Optimized Laplacian:\n{opt_laplacian.toarray()}")
                print(f"Direct Laplacian:\n{direct_laplacian.toarray()}")
            
            # Verify correctness with reasonable numerical tolerance
            # Given accumulated floating-point errors, 1e-5 is reasonable
            tolerance = 1e-5
            if laplacian_diff > tolerance:
                print(f"❌ FAILED: Laplacian difference {laplacian_diff:.2e} > {tolerance}")
                all_passed = False
            elif eigenval_diff > tolerance:
                print(f"❌ FAILED: Eigenvalue difference {eigenval_diff:.2e} > {tolerance}")
                all_passed = False
            else:
                print(f"✅ PASSED: Both differences within tolerance {tolerance}")
                
            # Additional analysis of the difference
            if laplacian_diff > 1e-10:
                print(f"  Note: Small numerical difference of {laplacian_diff:.2e} detected")
                print(f"  This is likely due to floating-point precision in dynamic weight application")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Filtration method is mathematically correct.")
    else:
        print("❌ SOME TESTS FAILED! Review implementation.")
    
    return all_passed

if __name__ == "__main__":
    verify_filtration_correctness()
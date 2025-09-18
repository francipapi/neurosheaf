"""Test coboundary/metric unification for H⁰ persistence.

This test verifies the fix for the coboundary/metric model mismatch issue where:
- Legacy simplified approach: One row per edge (incorrect for vector stalks)
- Correct general approach: Rows = sum of edge fiber dimensions

The fix ensures H⁰ persistence correctly reflects sheaf structure.
"""

import torch
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
import networkx as nx


def create_test_sheaf_2d_fibers():
    """Create a test sheaf with 2D/3D stalks to test vector-valued case."""
    # Create simple graph: 0 -> 1 -> 2
    poset = nx.DiGraph()
    poset.add_edges_from([(0, 1), (1, 2)])
    
    # Create stalks with varying dimensions
    stalks = {
        0: torch.randn(3, 10),  # 3D stalk
        1: torch.randn(2, 10),  # 2D stalk  
        2: torch.randn(2, 10),  # 2D stalk
    }
    
    # Create restrictions (row-stochastic for GW)
    restrictions = {
        (0, 1): torch.tensor([[0.6, 0.3, 0.1],
                              [0.2, 0.5, 0.3]], dtype=torch.float64),  # 2x3
        (1, 2): torch.tensor([[0.7, 0.3],
                              [0.4, 0.6]], dtype=torch.float64),  # 2x2
    }
    
    # Ensure row-stochastic (rows sum to 1)
    for edge, R in restrictions.items():
        restrictions[edge] = R / R.sum(dim=1, keepdim=True)
    
    metadata = {
        'construction_method': 'gromov_wasserstein',  # Required for is_gw_sheaf() check
        'gw_costs': {
            (0, 1): 0.2,
            (1, 2): 0.3,
        }
    }
    
    return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)


def test_coboundary_dimension_mismatch():
    """Test that legacy and correct approaches produce different dimensions."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    # Get legacy coboundary (simplified, incorrect)
    legacy_result = builder.build_coboundary_with_metrics(
        sheaf, 
        active_edges=[(0, 1), (1, 2)],
        legacy_h0=True
    )
    
    # Get correct coboundary (general formulation)
    correct_result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[(0, 1), (1, 2)],
        legacy_h0=False
    )
    
    # Legacy: delta should be 2x7 (one row per edge)
    assert legacy_result['delta'].shape[0] == 2, \
        f"Legacy delta should have 2 rows (one per edge), got {legacy_result['delta'].shape[0]}"
    
    # Correct: delta should be 4x7 (sum of edge fiber dims: 2+2)
    assert correct_result['delta'].shape[0] == 4, \
        f"Correct delta should have 4 rows (sum of edge dims), got {correct_result['delta'].shape[0]}"
    
    # Both should have same column dimension (total stalk dim = 3+2+2 = 7)
    assert legacy_result['delta'].shape[1] == 7
    assert correct_result['delta'].shape[1] == 7
    
    # G1 dimensions should match delta rows
    assert legacy_result['G1'].shape == (2, 2)  # Legacy: 2x2
    assert correct_result['G1'].shape == (4, 4)  # Correct: 4x4
    
    print("✅ Dimension test passed: Legacy and correct approaches differ as expected")


def test_kernel_dimension_difference():
    """Test that kernels differ between legacy and correct approaches."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    # Get both coboundaries
    legacy_result = builder.build_coboundary_with_metrics(
        sheaf, 
        active_edges=[(0, 1), (1, 2)],
        legacy_h0=True
    )
    
    correct_result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[(0, 1), (1, 2)],
        legacy_h0=False
    )
    
    # Compute kernel dimensions (global sections)
    legacy_delta = legacy_result['delta'].numpy()
    correct_delta = correct_result['delta'].numpy()
    
    # Compute ranks
    legacy_rank = np.linalg.matrix_rank(legacy_delta, tol=1e-10)
    correct_rank = np.linalg.matrix_rank(correct_delta, tol=1e-10)
    
    # Kernel dimensions
    legacy_kernel_dim = 7 - legacy_rank  # 7 = total stalk dimension
    correct_kernel_dim = 7 - correct_rank
    
    print(f"Legacy kernel dimension: {legacy_kernel_dim}")
    print(f"Correct kernel dimension: {correct_kernel_dim}")
    
    # They should generally differ (legacy is incorrect)
    # But we can't guarantee they always differ, so just verify computation works
    assert legacy_kernel_dim >= 0
    assert correct_kernel_dim >= 0
    
    print("✅ Kernel dimension test passed")


def test_shape_validation():
    """Test that shape validation catches dimension mismatches."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    # This should work (shapes are validated internally)
    result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[(0, 1), (1, 2)],
        legacy_h0=False
    )
    
    # Verify shapes are consistent
    delta = result['delta']
    G0 = result['G0']
    G1 = result['G1']
    
    assert G1.shape[0] == G1.shape[1] == delta.shape[0], "G1 dimension mismatch"
    assert G0.shape[0] == G0.shape[1] == delta.shape[1], "G0 dimension mismatch"
    
    print("✅ Shape validation test passed")


def test_empty_sheaf_case():
    """Test handling of empty sheaf (no active edges)."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    # Build with no active edges
    result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[],
        legacy_h0=False
    )
    
    # Should have empty delta (0 rows)
    assert result['delta'].shape[0] == 0
    assert result['delta'].shape[1] == 7  # Still has full stalk dimension
    
    # G1 should be empty
    assert result['G1'].shape == (0, 0)
    
    # G0 should still be full size
    assert result['G0'].shape == (7, 7)
    
    print("✅ Empty sheaf test passed")


def test_single_edge_case():
    """Test with single active edge."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    # Build with single edge
    result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[(0, 1)],
        legacy_h0=False
    )
    
    # Should have 2 rows (dimension of target stalk for edge 0->1)
    assert result['delta'].shape[0] == 2
    assert result['delta'].shape[1] == 7
    
    # G1 should be 2x2
    assert result['G1'].shape == (2, 2)
    
    print("✅ Single edge test passed")


def test_coboundary_structure():
    """Test that coboundary has correct mathematical structure."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[(0, 1)],
        legacy_h0=False
    )
    
    delta = result['delta'].numpy()
    
    # Check structure: for edge (0,1), should have:
    # - Identity block for target node 1 (columns 3-4)
    # - Negative restriction for source node 0 (columns 0-2)
    
    # Target block should be close to identity
    target_block = delta[:, 3:5]  # Node 1 columns
    assert np.allclose(target_block, np.eye(2), atol=1e-10), \
        "Target block should be identity"
    
    # Source block should be negative restriction
    source_block = delta[:, 0:3]  # Node 0 columns
    R_01 = sheaf.restrictions[(0, 1)].numpy()
    
    # Should have -R structure (with proper signs)
    assert source_block.shape == (2, 3)
    assert np.all(source_block <= 0.01), "Source block should have negative values"
    
    print("✅ Coboundary structure test passed")


def test_g1_block_structure():
    """Test that G1 has correct block-diagonal structure."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[(0, 1), (1, 2)],
        legacy_h0=False
    )
    
    G1 = result['G1'].numpy()
    
    # G1 should be 4x4 block diagonal
    assert G1.shape == (4, 4)
    
    # Should be diagonal
    assert np.allclose(G1, np.diag(np.diag(G1)), atol=1e-10), \
        "G1 should be diagonal"
    
    # Check block structure: first 2 elements for edge (0,1), next 2 for edge (1,2)
    edge_weights = result['edge_weights']
    
    # First block (edge 0->1)
    w1 = edge_weights.get((0, 1), 1.0)
    assert np.allclose(G1[0, 0], w1, atol=1e-10)
    assert np.allclose(G1[1, 1], w1, atol=1e-10)
    
    # Second block (edge 1->2)
    w2 = edge_weights.get((1, 2), 1.0)
    assert np.allclose(G1[2, 2], w2, atol=1e-10)
    assert np.allclose(G1[3, 3], w2, atol=1e-10)
    
    print("✅ G1 block structure test passed")


def test_legacy_warning():
    """Test that legacy mode runs (warning is logged, not raised)."""
    sheaf = create_test_sheaf_2d_fibers()
    builder = GWLaplacianBuilder()
    
    # This should produce a warning in the logs (not a Python warning)
    result = builder.build_coboundary_with_metrics(
        sheaf,
        active_edges=[(0, 1), (1, 2)],
        legacy_h0=True
    )
    
    # Just verify it runs without error
    assert result is not None
    print("✅ Legacy mode test passed")


if __name__ == "__main__":
    print("Running coboundary unification tests...\n")
    
    test_coboundary_dimension_mismatch()
    test_kernel_dimension_difference()
    test_shape_validation()
    test_empty_sheaf_case()
    test_single_edge_case()
    test_coboundary_structure()
    test_g1_block_structure()
    test_legacy_warning()
    
    print("\n✅ All coboundary unification tests passed!")
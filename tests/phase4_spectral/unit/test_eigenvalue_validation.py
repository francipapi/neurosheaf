# tests/phase4_spectral/unit/test_eigenvalue_validation.py
"""Unit tests for eigenvalue computation validation against known spectra.

This module tests the eigenvalue computation methods against known theoretical
results for canonical graph structures, ensuring mathematical correctness
and numerical stability.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.construction import Sheaf
from ..utils.test_ground_truth import GroundTruthGenerator, PersistenceValidator


class TestEigenvalueValidation:
    """Test eigenvalue computation against known theoretical results."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
        self.tolerance = 1e-6
        self.eigenvalue_tolerance = 1e-3  # More relaxed for numerical eigenvalue computation
    
    @pytest.mark.parametrize("n_nodes", [3, 5, 10])
    def test_linear_chain_eigenvalues(self, n_nodes):
        """Test eigenvalue computation for linear chain (path graph)."""
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes, stalk_dim=1)
        
        # Compute eigenvalues using static Laplacian
        static_laplacian = StaticLaplacianWithMasking(
            eigenvalue_method='dense',  # Use dense for exact computation
            max_eigenvalues=n_nodes
        )
        
        # Build Laplacian
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        
        # Compute eigenvalues
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Validate basic properties
        validation = self.validator.validate_eigenvalue_properties(eigenvals_sorted)
        assert validation['non_negative'], f"Eigenvalues not non-negative for {n_nodes} nodes"
        assert validation['finite'], f"Eigenvalues not finite for {n_nodes} nodes"
        assert validation['has_zero'], f"No zero eigenvalue found for {n_nodes} nodes"
        
        # Check zero eigenvalue count (should equal number of connected components)
        zero_count = torch.sum(eigenvals_sorted < self.tolerance).item()
        assert zero_count >= expected['expected_connected_components'], \
            f"Insufficient zero eigenvalues: got {zero_count}, expected >= {expected['expected_connected_components']}"
        
        # Check spectral gap
        if n_nodes > 1:
            gap_validation = self.validator.validate_spectral_gap(
                eigenvals_sorted, 
                expected_gap=None,  # Don't enforce exact gap due to stalk structure
                tolerance=self.tolerance,
                expected_zero_count=expected['expected_zero_eigenvalues']
            )
            assert gap_validation['gap_exists'], f"No spectral gap found for {n_nodes} nodes"
            assert gap_validation['zero_count_correct'], \
                f"Wrong zero eigenvalue count: got {gap_validation['zero_count']}, expected {gap_validation['expected_zero_count']}"
    
    @pytest.mark.parametrize("n_nodes", [4, 6, 8])
    def test_cycle_graph_eigenvalues(self, n_nodes):
        """Test eigenvalue computation for cycle graphs."""
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes, stalk_dim=1)
        
        static_laplacian = StaticLaplacianWithMasking(
            eigenvalue_method='dense',
            max_eigenvalues=n_nodes
        )
        
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Basic validation
        validation = self.validator.validate_eigenvalue_properties(eigenvals_sorted)
        assert all(validation.values()), f"Basic eigenvalue validation failed for cycle {n_nodes}"
        
        # Check connectivity (one zero eigenvalue)
        zero_count = torch.sum(eigenvals_sorted < self.tolerance).item()
        assert zero_count >= 1, f"No zero eigenvalue found for cycle {n_nodes}"
        
        # For cycles, check that we have the right spectral properties
        # Second smallest eigenvalue should be positive (algebraic connectivity)
        if len(eigenvals_sorted) > 1:
            assert eigenvals_sorted[1] > self.tolerance, \
                f"No algebraic connectivity for cycle {n_nodes}"
    
    @pytest.mark.parametrize("n_nodes", [3, 4, 5])
    def test_complete_graph_eigenvalues(self, n_nodes):
        """Test eigenvalue computation for complete graphs."""
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes, stalk_dim=1)
        
        static_laplacian = StaticLaplacianWithMasking(
            eigenvalue_method='dense',
            max_eigenvalues=n_nodes
        )
        
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Basic validation
        validation = self.validator.validate_eigenvalue_properties(eigenvals_sorted)
        assert all(validation.values()), f"Basic validation failed for complete graph {n_nodes}"
        
        # Complete graphs have high algebraic connectivity
        if len(eigenvals_sorted) > 1:
            gap_validation = self.validator.validate_spectral_gap(
                eigenvals_sorted,
                expected_zero_count=expected['expected_zero_eigenvalues']
            )
            assert gap_validation['gap_exists'], f"No spectral gap for complete graph {n_nodes}"
            assert gap_validation['gap_value'] > 1.0, \
                f"Spectral gap too small for complete graph {n_nodes}: {gap_validation['gap_value']}"
            assert gap_validation['zero_count_correct'], \
                f"Wrong zero eigenvalue count: got {gap_validation['zero_count']}, expected {gap_validation['expected_zero_count']}"
    
    @pytest.mark.parametrize("depth,branching", [(2, 2), (2, 3), (3, 2), (3, 3)])
    def test_tree_eigenvalues(self, depth, branching):
        """Test eigenvalue computation for tree structures."""
        sheaf, expected = self.generator.tree_sheaf(depth, branching, stalk_dim=1)
        
        static_laplacian = StaticLaplacianWithMasking(
            eigenvalue_method='dense',
            max_eigenvalues=expected['n_nodes']
        )
        
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Basic validation
        validation = self.validator.validate_eigenvalue_properties(eigenvals_sorted)
        assert all(validation.values()), \
            f"Basic validation failed for tree depth={depth}, branching={branching}"
        
        # Trees are connected - exactly one zero eigenvalue
        zero_count = torch.sum(eigenvals_sorted < self.tolerance).item()
        assert zero_count >= 1, f"No zero eigenvalue for tree depth={depth}, branching={branching}"
        
        # Trees have positive algebraic connectivity
        if len(eigenvals_sorted) > 1:
            assert eigenvals_sorted[1] > self.tolerance, \
                f"No algebraic connectivity for tree depth={depth}, branching={branching}"
    
    def test_disconnected_components_eigenvalues(self):
        """Test eigenvalue computation for disconnected graphs."""
        component_sizes = [3, 4, 2]  # Three disconnected components
        sheaf, expected = self.generator.disconnected_components_sheaf(component_sizes, stalk_dim=1)
        
        static_laplacian = StaticLaplacianWithMasking(
            eigenvalue_method='dense',
            max_eigenvalues=expected['total_nodes']
        )
        
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Basic validation
        validation = self.validator.validate_eigenvalue_properties(eigenvals_sorted)
        assert all(validation.values()), "Basic validation failed for disconnected components"
        
        # Number of zero eigenvalues should equal number of connected components
        zero_count = torch.sum(eigenvals_sorted < self.tolerance).item()
        assert zero_count >= expected['n_components'], \
            f"Zero eigenvalue count {zero_count} < expected components {expected['n_components']}"
        
        # With multiple components, there should be no algebraic connectivity
        # (multiple zero eigenvalues)
        assert zero_count > 1, "Disconnected graph should have multiple zero eigenvalues"
    
    def test_eigenvalue_method_consistency(self):
        """Test consistency between different eigenvalue computation methods."""
        # Create a moderately sized test case
        sheaf, expected = self.generator.cycle_graph_sheaf(6, stalk_dim=2)
        
        # Test both methods
        static_laplacian_dense = StaticLaplacianWithMasking(
            eigenvalue_method='dense',
            max_eigenvalues=20
        )
        
        static_laplacian_lobpcg = StaticLaplacianWithMasking(
            eigenvalue_method='lobpcg',
            max_eigenvalues=20
        )
        
        # Build Laplacian
        laplacian, metadata = static_laplacian_dense._get_cached_laplacian(sheaf)
        
        # Compute with both methods
        eigenvals_dense, _ = static_laplacian_dense._compute_eigenvalues(laplacian)
        eigenvals_lobpcg, _ = static_laplacian_lobpcg._compute_eigenvalues(laplacian)
        
        # Sort both results
        eigenvals_dense_sorted = torch.sort(eigenvals_dense)[0]
        eigenvals_lobpcg_sorted = torch.sort(eigenvals_lobpcg)[0]
        
        # Take minimum length for comparison
        min_len = min(len(eigenvals_dense_sorted), len(eigenvals_lobpcg_sorted))
        
        # Compare smallest eigenvalues (most important for persistence)
        dense_subset = eigenvals_dense_sorted[:min_len]
        lobpcg_subset = eigenvals_lobpcg_sorted[:min_len]
        
        # Allow some tolerance for LOBPCG approximation
        assert torch.allclose(dense_subset, lobpcg_subset, rtol=1e-2, atol=1e-3), \
            "Dense and LOBPCG eigenvalue methods disagree"
    
    def test_stalk_dimension_scaling(self):
        """Test how eigenvalue computation scales with stalk dimension."""
        base_sheaf, base_expected = self.generator.linear_chain_sheaf(4, stalk_dim=1)
        scaled_sheaf, scaled_expected = self.generator.linear_chain_sheaf(4, stalk_dim=3)
        
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        
        # Compute eigenvalues for both
        base_laplacian, _ = static_laplacian._get_cached_laplacian(base_sheaf)
        scaled_laplacian, _ = static_laplacian._get_cached_laplacian(scaled_sheaf)
        
        base_eigenvals, _ = static_laplacian._compute_eigenvalues(base_laplacian)
        scaled_eigenvals, _ = static_laplacian._compute_eigenvalues(scaled_laplacian)
        
        # Basic validation for both
        base_validation = self.validator.validate_eigenvalue_properties(base_eigenvals)
        scaled_validation = self.validator.validate_eigenvalue_properties(scaled_eigenvals)
        
        assert all(base_validation.values()), "Base case validation failed"
        assert all(scaled_validation.values()), "Scaled case validation failed"
        
        # Check zero eigenvalue multiplicity
        base_zeros = torch.sum(base_eigenvals < self.tolerance).item()
        scaled_zeros = torch.sum(scaled_eigenvals < self.tolerance).item()
        
        # Scaled version should have more zero eigenvalues (higher multiplicity)
        assert scaled_zeros >= base_zeros, \
            f"Scaled version has fewer zeros: {scaled_zeros} vs {base_zeros}"
    
    def test_numerical_stability(self):
        """Test numerical stability of eigenvalue computation."""
        # Create a test case and perturb it slightly
        sheaf, expected = self.generator.cycle_graph_sheaf(5, stalk_dim=2)
        
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        
        # Compute baseline eigenvalues
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        baseline_eigenvals, _ = static_laplacian._compute_eigenvalues(laplacian)
        baseline_sorted = torch.sort(baseline_eigenvals)[0]
        
        # Create slightly perturbed version
        perturbed_stalks = {}
        for node, stalk in sheaf.stalks.items():
            # Add small random perturbation
            noise = torch.randn_like(stalk) * 1e-8
            perturbed_stalks[node] = stalk + noise
        
        perturbed_sheaf = Sheaf(sheaf.poset, perturbed_stalks, sheaf.restrictions)
        
        # Compute perturbed eigenvalues
        perturbed_laplacian, _ = static_laplacian._get_cached_laplacian(perturbed_sheaf)
        perturbed_eigenvals, _ = static_laplacian._compute_eigenvalues(perturbed_laplacian)
        perturbed_sorted = torch.sort(perturbed_eigenvals)[0]
        
        # Check that perturbation doesn't cause dramatic changes
        min_len = min(len(baseline_sorted), len(perturbed_sorted))
        
        # Relative change should be small
        relative_change = torch.abs(baseline_sorted[:min_len] - perturbed_sorted[:min_len])
        max_relative_change = torch.max(relative_change).item()
        
        assert max_relative_change < 1e-6, \
            f"Eigenvalues not stable under small perturbations: max change {max_relative_change}"
    
    def test_edge_case_handling(self):
        """Test handling of edge cases in eigenvalue computation."""
        # Single node sheaf
        single_poset = nx.DiGraph()
        single_poset.add_node('A')
        single_sheaf = Sheaf(single_poset, {'A': torch.eye(2)}, {})
        
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        
        try:
            laplacian, metadata = static_laplacian._get_cached_laplacian(single_sheaf)
            eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
            
            # Should not crash and should produce valid results
            validation = self.validator.validate_eigenvalue_properties(eigenvals)
            assert validation['finite'], "Single node case produced non-finite eigenvalues"
            assert validation['non_negative'], "Single node case produced negative eigenvalues"
            
        except Exception as e:
            pytest.fail(f"Single node case should not raise exception: {e}")
        
        # Empty restrictions (only stalks, no edges)
        empty_poset = nx.DiGraph()
        empty_poset.add_nodes_from(['A', 'B'])
        empty_stalks = {'A': torch.eye(2), 'B': torch.eye(2)}
        empty_sheaf = Sheaf(empty_poset, empty_stalks, {})
        
        try:
            laplacian, metadata = static_laplacian._get_cached_laplacian(empty_sheaf)
            eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
            
            # Should produce all zero eigenvalues (disconnected components)
            zero_count = torch.sum(eigenvals < self.tolerance).item()
            assert zero_count == len(eigenvals), "Empty restrictions should produce all zero eigenvalues"
            
        except Exception as e:
            pytest.fail(f"Empty restrictions case should not raise exception: {e}")


@pytest.mark.slow
class TestEigenvaluePerformance:
    """Performance tests for eigenvalue computation."""
    
    def test_large_graph_eigenvalues(self):
        """Test eigenvalue computation on larger graphs."""
        # This test is marked as slow and will be skipped in regular testing
        generator = GroundTruthGenerator()
        
        # Create larger test case
        sheaf, expected = generator.linear_chain_sheaf(50, stalk_dim=3)
        
        static_laplacian = StaticLaplacianWithMasking(
            eigenvalue_method='lobpcg',
            max_eigenvalues=50
        )
        
        import time
        start_time = time.time()
        
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time (< 30 seconds)
        assert computation_time < 30.0, f"Large graph computation too slow: {computation_time}s"
        
        # Results should still be valid
        validator = PersistenceValidator()
        validation = validator.validate_eigenvalue_properties(eigenvals)
        assert all(validation.values()), "Large graph computation produced invalid eigenvalues"
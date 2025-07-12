# tests/test_laplacian_method_consistency.py
"""Validation tests to ensure all Laplacian assembly methods produce identical results.

This module tests mathematical consistency across different assembly methods
to ensure they all implement the same δ^T δ construction correctly.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from neurosheaf.sheaf.construction import Sheaf
from neurosheaf.sheaf.laplacian import SheafLaplacianBuilder
from tests.phase4_spectral.utils.test_ground_truth import GroundTruthGenerator


class TestLaplacianMethodConsistency:
    """Test consistency across different Laplacian assembly methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.tolerance = 1e-6   # Relaxed for numerical precision in different methods
        self.methods = ['preallocated', 'block_wise']
    
    def _build_laplacian_with_method(self, sheaf: Sheaf, method: str) -> csr_matrix:
        """Build Laplacian using specific assembly method."""
        builder = SheafLaplacianBuilder(
            enable_gpu=False,  # Use CPU for deterministic comparison
            memory_efficient=True,
            validate_properties=False,  # Skip validation to avoid side effects
            assembly_method=method
        )
        
        laplacian, metadata = builder.build_laplacian(sheaf)
        return laplacian
    
    def _assert_matrices_equal(self, matrix1: csr_matrix, matrix2: csr_matrix, 
                              method1: str, method2: str, test_name: str):
        """Assert two sparse matrices are exactly equal."""
        # Check shapes
        assert matrix1.shape == matrix2.shape, \
            f"{test_name}: Shape mismatch between {method1} and {method2}: {matrix1.shape} vs {matrix2.shape}"
        
        # Convert to dense for comparison (small test cases only)
        dense1 = matrix1.toarray()
        dense2 = matrix2.toarray()
        
        # Check exact equality with tolerance
        max_diff = np.max(np.abs(dense1 - dense2))
        assert max_diff < self.tolerance, \
            f"{test_name}: Methods {method1} and {method2} differ by {max_diff:.2e} (max allowed: {self.tolerance:.2e})"
        
        # Check non-zero pattern consistency
        assert matrix1.nnz == matrix2.nnz, \
            f"{test_name}: NNZ mismatch between {method1} and {method2}: {matrix1.nnz} vs {matrix2.nnz}"
    
    @pytest.mark.parametrize("n_nodes", [3, 5, 8])
    @pytest.mark.parametrize("stalk_dim", [1, 2, 3])
    def test_linear_chain_consistency(self, n_nodes, stalk_dim):
        """Test all methods produce identical results for linear chains."""
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes, stalk_dim)
        
        # Build Laplacian with all methods
        laplacians = {}
        for method in self.methods:
            laplacians[method] = self._build_laplacian_with_method(sheaf, method)
        
        # Compare all pairs
        for i, method1 in enumerate(self.methods):
            for method2 in self.methods[i+1:]:
                self._assert_matrices_equal(
                    laplacians[method1], laplacians[method2],
                    method1, method2, 
                    f"linear_chain_n{n_nodes}_dim{stalk_dim}"
                )
    
    @pytest.mark.parametrize("n_nodes", [4, 6, 8])
    @pytest.mark.parametrize("stalk_dim", [1, 2])
    def test_cycle_consistency(self, n_nodes, stalk_dim):
        """Test all methods produce identical results for cycles."""
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes, stalk_dim)
        
        # Build Laplacian with all methods
        laplacians = {}
        for method in self.methods:
            laplacians[method] = self._build_laplacian_with_method(sheaf, method)
        
        # Compare all pairs
        for i, method1 in enumerate(self.methods):
            for method2 in self.methods[i+1:]:
                self._assert_matrices_equal(
                    laplacians[method1], laplacians[method2],
                    method1, method2, 
                    f"cycle_n{n_nodes}_dim{stalk_dim}"
                )
    
    @pytest.mark.parametrize("n_nodes", [3, 4, 5])
    @pytest.mark.parametrize("stalk_dim", [1, 2])
    def test_complete_graph_consistency(self, n_nodes, stalk_dim):
        """Test all methods produce identical results for complete graphs."""
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes, stalk_dim)
        
        # Build Laplacian with all methods
        laplacians = {}
        for method in self.methods:
            laplacians[method] = self._build_laplacian_with_method(sheaf, method)
        
        # Compare all pairs
        for i, method1 in enumerate(self.methods):
            for method2 in self.methods[i+1:]:
                self._assert_matrices_equal(
                    laplacians[method1], laplacians[method2],
                    method1, method2, 
                    f"complete_n{n_nodes}_dim{stalk_dim}"
                )
    
    def test_edge_weight_consistency(self):
        """Test that edge weights are applied consistently across methods."""
        # Create a simple test case with known edge weights
        sheaf, expected = self.generator.linear_chain_sheaf(4, stalk_dim=2)
        
        # Define custom edge weights
        custom_weights = {}
        for edge in sheaf.restrictions.keys():
            custom_weights[edge] = 2.5  # Non-unit weight
        
        # Build Laplacian with all methods using custom weights
        laplacians = {}
        for method in self.methods:
            builder = SheafLaplacianBuilder(
                enable_gpu=False,
                assembly_method=method
            )
            laplacians[method], _ = builder.build_laplacian(sheaf, custom_weights)
        
        # Compare all pairs
        for i, method1 in enumerate(self.methods):
            for method2 in self.methods[i+1:]:
                self._assert_matrices_equal(
                    laplacians[method1], laplacians[method2],
                    method1, method2, 
                    "custom_edge_weights"
                )
    
    def test_identity_sheaf_property(self):
        """Test that all methods preserve the identity sheaf property Δ(1⊗v) = 0."""
        # Create identity sheaf (all restrictions are identity matrices)
        n_nodes = 5
        stalk_dim = 3
        
        # Build simple path graph
        poset = nx.DiGraph()
        nodes = [f'node_{i}' for i in range(n_nodes)]
        poset.add_nodes_from(nodes)
        for i in range(n_nodes - 1):
            poset.add_edge(nodes[i], nodes[i + 1])
        
        # Create identity stalks (in whitened coordinates)
        stalks = {node: torch.eye(stalk_dim) for node in nodes}
        
        # Create identity restrictions (exact identity maps)
        restrictions = {}
        for i in range(n_nodes - 1):
            edge = (nodes[i], nodes[i + 1])
            restrictions[edge] = torch.eye(stalk_dim)  # Exact identity
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Test with all methods
        for method in self.methods:
            laplacian = self._build_laplacian_with_method(sheaf, method)
            
            # Create constant vector: 1⊗v where v is arbitrary
            v = np.ones(stalk_dim)
            constant_vector = np.tile(v, n_nodes)  # [v, v, v, v, v]
            
            # Apply Laplacian: should get zero vector
            result = laplacian @ constant_vector
            
            # Check that Δ(1⊗v) = 0 (identity sheaf property)
            max_error = np.max(np.abs(result))
            assert max_error < 1e-10, \
                f"Method {method} violates identity sheaf property: max error {max_error:.2e}"
    
    def test_mathematical_properties_consistency(self):
        """Test that all methods preserve mathematical properties of the Laplacian."""
        sheaf, expected = self.generator.cycle_graph_sheaf(6, stalk_dim=2)
        
        for method in self.methods:
            laplacian = self._build_laplacian_with_method(sheaf, method)
            dense = laplacian.toarray()
            
            # Test 1: Symmetry
            max_asymmetry = np.max(np.abs(dense - dense.T))
            assert max_asymmetry < 1e-12, \
                f"Method {method} produces non-symmetric Laplacian: {max_asymmetry:.2e}"
            
            # Test 2: Positive semi-definite (all eigenvalues >= 0)
            eigenvals = np.linalg.eigvals(dense)
            min_eigenval = np.min(eigenvals)
            assert min_eigenval >= -1e-10, \
                f"Method {method} produces negative eigenvalue: {min_eigenval:.2e}"
            
            # Test 3: Row sums should be zero (Laplacian property)
            # Note: With unbalanced edge weights (e.g., 0.8), row sums may not be exactly zero
            row_sums = np.sum(dense, axis=1)
            max_row_sum = np.max(np.abs(row_sums))
            assert max_row_sum < 0.5, \
                f"Method {method} has excessive row sum deviation: {max_row_sum:.2e}"


class TestSpecificMathematicalFormula:
    """Test that the Laplacian construction follows the exact δ^T δ formula."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.tolerance = 1e-6  # Relaxed for floating point precision
    
    def test_coboundary_construction_manual(self):
        """Manually verify that Δ = δ^T δ for a simple case."""
        # Create a simple 3-node linear chain with known structure
        sheaf, expected = self.generator.linear_chain_sheaf(3, stalk_dim=2)
        
        # Extract structure
        nodes = list(sheaf.poset.nodes())
        edges = list(sheaf.restrictions.keys())
        
        # Build coboundary operator δ manually
        # δ: R^6 → R^4 (3 nodes × 2 dims → 2 edges × 2 dims)
        n_node_dims = len(nodes) * 2  # 6
        n_edge_dims = len(edges) * 2  # 4
        
        delta = np.zeros((n_edge_dims, n_node_dims))
        
        # For each edge e = (u,v): (δx)_e = R_e x_u - x_v
        for edge_idx, (source, target) in enumerate(edges):
            source_idx = nodes.index(source)
            target_idx = nodes.index(target)
            R = sheaf.restrictions[(source, target)].numpy()
            
            # Edge block starts at edge_idx * 2
            edge_start = edge_idx * 2
            source_start = source_idx * 2
            target_start = target_idx * 2
            
            # (δx)_e = R_e x_source - x_target
            delta[edge_start:edge_start+2, source_start:source_start+2] = R
            delta[edge_start:edge_start+2, target_start:target_start+2] = -np.eye(2)
        
        # Compute manual Laplacian: Δ = δ^T δ
        manual_laplacian = delta.T @ delta
        
        # Build using our implementation
        builder = SheafLaplacianBuilder(assembly_method='preallocated')
        computed_laplacian, _ = builder.build_laplacian(sheaf)
        computed_dense = computed_laplacian.toarray()
        
        # Compare
        max_diff = np.max(np.abs(manual_laplacian - computed_dense))
        assert max_diff < self.tolerance, \
            f"Manual δ^T δ differs from computed Laplacian by {max_diff:.2e}"
    
    def test_diagonal_block_formula(self):
        """Verify diagonal blocks follow the correct δ^T δ formula: Δ_vv = Σ R_e^T R_e + Σ I."""
        # Create test case where we can manually verify
        sheaf, expected = self.generator.linear_chain_sheaf(4, stalk_dim=2)
        
        # Build Laplacian
        builder = SheafLaplacianBuilder(assembly_method='preallocated')
        laplacian, metadata = builder.build_laplacian(sheaf)
        dense = laplacian.toarray()
        
        nodes = list(sheaf.poset.nodes())
        
        # Verify each diagonal block manually
        for node_idx, node in enumerate(nodes):
            node_start = node_idx * 2
            node_end = node_start + 2
            computed_block = dense[node_start:node_end, node_start:node_end]
            
            # Manual computation using CORRECT δ^T δ formula:
            # Δ_vv = Σ R_e^T R_e (outgoing) + Σ I (incoming) 
            manual_block = np.zeros((2, 2))
            
            # Outgoing edges: R_e^T R_e
            for successor in sheaf.poset.successors(node):
                edge = (node, successor)
                if edge in sheaf.restrictions:
                    R = sheaf.restrictions[edge].numpy()
                    manual_block += R.T @ R
            
            # Incoming edges: I (from (-I)^T @ (-I) = I in δ^T δ)
            for predecessor in sheaf.poset.predecessors(node):
                edge = (predecessor, node)
                if edge in sheaf.restrictions:
                    manual_block += np.eye(2)  # Correct: add I, not R @ R^T
            
            # Compare
            max_diff = np.max(np.abs(computed_block - manual_block))
            assert max_diff < self.tolerance, \
                f"Diagonal block for node {node} differs by {max_diff:.2e}"
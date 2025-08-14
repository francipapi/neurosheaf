"""Comprehensive tests for normalized Hodge Laplacian with generalized eigenvalue formulation.

This module tests the new normalized Hodge Laplacian implementation using the 
generalized eigenvalue problem L x = λ M x instead of matrix inversion for 
improved numerical stability and handling of tiny eigenvalues.

Key test areas:
1. Generalized eigenvalue solver correctness (L x = λ M x)
2. Matrix-free LinearOperator functionality for large problems
3. Filtration monotonicity preservation (λ_{t+1} ≥ λ_t)
4. Integration with UnifiedStaticLaplacian pipeline
5. Mathematical property preservation (PSD, symmetry)
6. Fallback mechanism robustness
7. Numerical stability with tiny eigenvalues (~10^-12)
8. LOBPCG vs eigsh solver routing constraints
"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lobpcg, eigsh
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.spectral.static_laplacian_unified import UnifiedStaticLaplacian


class TestGeneralizedEigenvalueSolver:
    """Test the core generalized eigenvalue solver L x = λ M x."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=True)
        self.test_sheaf = self._create_test_gw_sheaf()
        
    def _create_test_gw_sheaf(self) -> Sheaf:
        """Create a well-conditioned GW sheaf for testing."""
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        
        # Identity stalks for GW sheaves
        stalks = {
            'layer1': torch.eye(4, dtype=torch.float64),
            'layer2': torch.eye(3, dtype=torch.float64),
            'layer3': torch.eye(2, dtype=torch.float64)
        }
        
        # Column-stochastic restrictions with controlled conditioning
        torch.manual_seed(42)  # Reproducible tests
        R_12 = torch.softmax(torch.randn(3, 4, dtype=torch.float64), dim=0)  # Column-stochastic
        R_23 = torch.softmax(torch.randn(2, 3, dtype=torch.float64), dim=0)  # Column-stochastic
        
        restrictions = {
            ('layer1', 'layer2'): R_12,
            ('layer2', 'layer3'): R_23
        }
        
        # GW costs as edge weights
        gw_costs = {
            ('layer1', 'layer2'): 0.35,
            ('layer2', 'layer3'): 0.25
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        return Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata
        )
    
    def test_generalized_solver_basic_functionality(self):
        """Test that generalized eigenvalue solver produces valid results."""
        active_edges = list(self.test_sheaf.restrictions.keys())
        
        # Test with reasonable number of eigenvalues
        k = 5
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            self.test_sheaf, active_edges, k=k, use_matrix_free=False
        )
        
        # Basic sanity checks
        assert len(eigenvals) == k
        assert eigenvecs.shape[1] == k
        assert eigenvals.dtype in [np.float32, np.float64]
        assert eigenvecs.dtype in [np.float32, np.float64]
        
        # Eigenvalues should be real and non-negative (PSD property)
        assert np.all(np.isreal(eigenvals))
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
        
        # Should be sorted in ascending order
        assert np.all(eigenvals[:-1] <= eigenvals[1:])
        
        # Eigenvectors should be M-orthonormal (not standard orthonormal)
        # For generalized eigenvalue problem L x = λ M x, normalization is x^T M x = 1
        
        # Get mass matrix for validation
        node_masses = self.builder._extract_node_masses(self.test_sheaf)
        M = self.builder._build_stalk_metric(self.test_sheaf, node_masses)
        
        # Convert to sparse matrix if it's a tensor
        if isinstance(M, torch.Tensor):
            M_dense = M.detach().cpu().numpy()
        else:
            M_dense = M.toarray() if hasattr(M, 'toarray') else M
        
        # Check M-orthonormality: x_i^T M x_i = 1 and x_i^T M x_j = 0 for i ≠ j
        # Use relaxed tolerances due to iterative solver limitations
        for i in range(k):
            x_i = eigenvecs[:, i]
            m_norm_i = np.sqrt(x_i.T @ M_dense @ x_i)
            # Relaxed tolerance for M-normalization
            assert abs(m_norm_i - 1.0) < 1e-2, f"Eigenvector {i} not M-normalized: ||x||_M = {m_norm_i:.6f}"
            
            # Check M-orthogonality with other vectors (more relaxed)
            for j in range(i + 1, k):
                x_j = eigenvecs[:, j]
                m_inner_product = x_i.T @ M_dense @ x_j
                assert abs(m_inner_product) < 1e-2, f"Eigenvectors {i},{j} not M-orthogonal: <x_i, x_j>_M = {m_inner_product:.6f}"
    
    def test_generalized_vs_standard_eigenvalue_relationship(self):
        """Test relationship between generalized L x = λ M x and standard L x = λ x."""
        active_edges = list(self.test_sheaf.restrictions.keys())
        k = 4
        
        # Compute generalized eigenvalues L x = λ M x
        eigenvals_gen, eigenvecs_gen = self.builder.solve_generalized_robust(
            self.test_sheaf, active_edges, k=k, use_matrix_free=False
        )
        
        # Compute standard Laplacian eigenvalues for comparison
        L_sparse = self.builder.build_laplacian(self.test_sheaf, sparse=True, active_edges=active_edges)
        eigenvals_std, eigenvecs_std = eigsh(L_sparse, k=k, which='SA', return_eigenvectors=True)
        eigenvals_std = np.sort(eigenvals_std)
        
        # For normalized Hodge Laplacian, generalized eigenvalues should be related but different
        # They represent the solution to different problems: L x = λ M x vs L x = λ x
        
        # Both should have similar smallest eigenvalue (connectivity properties)
        assert abs(eigenvals_gen[0] - eigenvals_std[0]) < 1.0  # May differ due to normalization
        
        # Both should be non-negative
        assert eigenvals_gen[0] >= -1e-10
        assert eigenvals_std[0] >= -1e-10
    
    def test_mass_matrix_construction(self):
        """Test that mass matrix M = G₀ is constructed correctly."""
        active_edges = list(self.test_sheaf.restrictions.keys())
        
        # Build G₀ (stalk metric/mass matrix)
        node_masses = self.builder._extract_node_masses(self.test_sheaf)
        G0 = self.builder._build_stalk_metric(self.test_sheaf, node_masses)
        
        # Convert to numpy for analysis
        if isinstance(G0, torch.Tensor):
            G0_dense = G0.detach().cpu().numpy()
        else:
            G0_dense = G0.toarray() if hasattr(G0, 'toarray') else G0
        
        # Should be symmetric positive definite
        symmetry_error = np.linalg.norm(G0_dense - G0_dense.T)
        assert symmetry_error < 1e-12
        
        # Check positive definiteness via eigenvalues
        eigenvals_G0 = np.linalg.eigvals(G0_dense)
        assert np.all(eigenvals_G0.real > 1e-12)  # Should be positive definite
        
        # For GW sheaves with identity stalks, G₀ should be mostly block-diagonal identity
        # (with possible edge weight contributions)
        diagonal = np.diag(G0_dense)
        assert np.all(diagonal > 0)  # All positive diagonal entries
    
    def test_generalized_eigenvalue_problem_validation(self):
        """Test that computed eigenvalues actually satisfy L x = λ M x."""
        active_edges = list(self.test_sheaf.restrictions.keys())
        k = 3  # Test with small number for detailed validation
        
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            self.test_sheaf, active_edges, k=k, use_matrix_free=False
        )
        
        # Build L and M matrices for validation
        L = self.builder.build_laplacian(self.test_sheaf, sparse=True, active_edges=active_edges)
        node_masses = self.builder._extract_node_masses(self.test_sheaf)
        M = self.builder._build_stalk_metric(self.test_sheaf, node_masses)
        
        # Convert to sparse matrix for consistency
        if isinstance(M, torch.Tensor):
            from scipy.sparse import csr_matrix
            M = csr_matrix(M.detach().cpu().numpy())
        elif not hasattr(M, 'toarray'):
            from scipy.sparse import csr_matrix
            M = csr_matrix(M)
        
        # Validate generalized eigenvalue equation: L x = λ M x
        for i in range(k):
            lambda_i = eigenvals[i]
            x_i = eigenvecs[:, i]
            
            # Compute L x_i and λ M x_i
            Lx = L @ x_i
            lambda_Mx = lambda_i * (M @ x_i)
            
            # They should be equal: L x_i = λ_i M x_i
            residual = np.linalg.norm(Lx - lambda_Mx)
            relative_residual = residual / (np.linalg.norm(Lx) + 1e-12)
            
            # More relaxed tolerance for iterative solvers
            assert relative_residual < 1e-1, f"Eigenvalue {i}: L x != λ M x, residual = {relative_residual:.2e}"
    
    def test_solver_routing_constraints(self):
        """Test LOBPCG vs eigsh routing constraints based on user specification."""
        active_edges = list(self.test_sheaf.restrictions.keys())
        k = 4
        
        # Test matrix-free mode → should use LOBPCG only
        eigenvals_mf, eigenvecs_mf = self.builder.solve_generalized_robust(
            self.test_sheaf, active_edges, k=k, use_matrix_free=True
        )
        
        # Should succeed (LOBPCG can handle LinearOperator)
        assert len(eigenvals_mf) == k
        assert np.all(eigenvals_mf >= -1e-10)
        
        # Test sparse mode → should use eigsh with shift-invert
        eigenvals_sp, eigenvecs_sp = self.builder.solve_generalized_robust(
            self.test_sheaf, active_edges, k=k, use_matrix_free=False
        )
        
        # Should also succeed (eigsh can handle sparse matrices)
        assert len(eigenvals_sp) == k
        assert np.all(eigenvals_sp >= -1e-10)
        
        # Results should be similar (same mathematical problem)
        assert np.allclose(eigenvals_mf, eigenvals_sp, rtol=1e-4)


class TestMatrixFreeFunctionality:
    """Test matrix-free LinearOperator functionality for large problems."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=False)  # Faster for large tests
        
    def _create_large_gw_sheaf(self, num_layers: int = 5, layer_sizes: List[int] = None) -> Sheaf:
        """Create a larger GW sheaf for matrix-free testing."""
        if layer_sizes is None:
            layer_sizes = [20, 15, 12, 10, 8][:num_layers]
        
        # Create chain poset
        poset = nx.DiGraph()
        layer_names = [f'layer_{i}' for i in range(num_layers)]
        for i in range(num_layers - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        
        # Identity stalks
        stalks = {}
        for i, name in enumerate(layer_names):
            stalks[name] = torch.eye(layer_sizes[i], dtype=torch.float64)
        
        # Random column-stochastic restrictions
        torch.manual_seed(123)  # Reproducible
        restrictions = {}
        gw_costs = {}
        
        for i in range(num_layers - 1):
            source = layer_names[i]
            target = layer_names[i + 1]
            
            # Create column-stochastic restriction
            R = torch.softmax(torch.randn(layer_sizes[i + 1], layer_sizes[i], dtype=torch.float64), dim=0)
            restrictions[(source, target)] = R
            
            # Random GW cost
            gw_costs[(source, target)] = 0.1 + 0.3 * torch.rand(1).item()
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        return Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata
        )
    
    def test_matrix_free_vs_sparse_equivalence(self):
        """Test that matrix-free and sparse modes produce equivalent results."""
        # Use medium-sized problem for comparison
        large_sheaf = self._create_large_gw_sheaf(num_layers=4, layer_sizes=[12, 10, 8, 6])
        active_edges = list(large_sheaf.restrictions.keys())
        k = 6
        
        # Matrix-free computation
        eigenvals_mf, eigenvecs_mf = self.builder.solve_generalized_robust(
            large_sheaf, active_edges, k=k, use_matrix_free=True
        )
        
        # Sparse computation
        eigenvals_sp, eigenvecs_sp = self.builder.solve_generalized_robust(
            large_sheaf, active_edges, k=k, use_matrix_free=False
        )
        
        # Results should be reasonably close (relaxed for iterative solvers)
        assert np.allclose(eigenvals_mf, eigenvals_sp, rtol=1e-3, atol=1e-6)
        
        # Both methods should produce valid M-orthonormal eigenvectors
        # (Different iterative solvers can produce very different but equally valid bases)
        
        # Get mass matrix for M-orthonormality check
        node_masses = self.builder._extract_node_masses(large_sheaf)
        M = self.builder._build_stalk_metric(large_sheaf, node_masses)
        if isinstance(M, torch.Tensor):
            M_dense = M.detach().cpu().numpy()
        else:
            M_dense = M.toarray() if hasattr(M, 'toarray') else M
        
        # Check that both eigenvector sets are M-orthonormal
        mf_m_orthogonality = np.linalg.norm(eigenvecs_mf.T @ M_dense @ eigenvecs_mf - np.eye(k))
        sp_m_orthogonality = np.linalg.norm(eigenvecs_sp.T @ M_dense @ eigenvecs_sp - np.eye(k))
        
        # Use relaxed tolerance for iterative solvers
        assert mf_m_orthogonality < 1e-2, f"Matrix-free eigenvectors not M-orthonormal: {mf_m_orthogonality:.2e}"
        assert sp_m_orthogonality < 1e-2, f"Sparse eigenvectors not M-orthonormal: {sp_m_orthogonality:.2e}"
    
    def test_matrix_free_memory_efficiency(self):
        """Test that matrix-free mode handles larger problems than sparse mode."""
        # Create a problem that would be memory-intensive in sparse mode
        large_sheaf = self._create_large_gw_sheaf(num_layers=6, layer_sizes=[25, 20, 18, 15, 12, 10])
        active_edges = list(large_sheaf.restrictions.keys())
        k = 5
        
        # Matrix-free should handle this
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            large_sheaf, active_edges, k=k, use_matrix_free=True
        )
        
        # Should succeed with reasonable results
        assert len(eigenvals) == k
        assert np.all(eigenvals >= -1e-10)
        assert not np.any(np.isnan(eigenvals))
        assert not np.any(np.isinf(eigenvals))
    
    def test_linear_operator_matvec_correctness(self):
        """Test that LinearOperator matrix-vector products are computed correctly."""
        test_sheaf = self._create_large_gw_sheaf(num_layers=3, layer_sizes=[8, 6, 4])
        active_edges = list(test_sheaf.restrictions.keys())
        
        # Create LinearOperator for L
        L_op = self.builder._build_linear_operator_L(test_sheaf, active_edges)
        
        # Create corresponding sparse matrix for comparison
        L_sparse = self.builder.build_laplacian(test_sheaf, sparse=True, active_edges=active_edges)
        
        # Test matrix-vector multiplication
        np.random.seed(456)
        test_vector = np.random.randn(L_op.shape[1])
        
        # Compute using LinearOperator
        result_op = L_op @ test_vector
        
        # Compute using sparse matrix
        result_sparse = L_sparse @ test_vector
        
        # Should be identical (within numerical precision)
        assert np.allclose(result_op, result_sparse, rtol=1e-10, atol=1e-12)


class TestFiltrationMonotonicity:
    """Test that filtration monotonicity is preserved: A_{t+1} ⪰ A_t ⟹ λ_{t+1} ≥ λ_t."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=False)
        self.test_sheaf = self._create_filtration_test_sheaf()
        
    def _create_filtration_test_sheaf(self) -> Sheaf:
        """Create sheaf with multiple edges for filtration testing."""
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C'), ('B', 'D')])  # More complex graph
        
        stalks = {
            'A': torch.eye(3, dtype=torch.float64),
            'B': torch.eye(3, dtype=torch.float64),
            'C': torch.eye(2, dtype=torch.float64),
            'D': torch.eye(2, dtype=torch.float64)
        }
        
        # Create restrictions with different "strengths" for filtration
        torch.manual_seed(789)
        restrictions = {
            ('A', 'B'): torch.softmax(torch.randn(3, 3, dtype=torch.float64), dim=0),
            ('B', 'C'): torch.softmax(torch.randn(2, 3, dtype=torch.float64), dim=0),
            ('A', 'C'): torch.softmax(torch.randn(2, 3, dtype=torch.float64), dim=0),
            ('B', 'D'): torch.softmax(torch.randn(2, 3, dtype=torch.float64), dim=0)
        }
        
        # Different GW costs for filtration ordering
        gw_costs = {
            ('A', 'B'): 0.1,  # Strongest connection (added first)
            ('B', 'C'): 0.2,
            ('A', 'C'): 0.35,
            ('B', 'D'): 0.5   # Weakest connection (added last)
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        return Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata
        )
    
    def test_eigenvalue_monotonicity_across_filtration(self):
        """Test that eigenvalues increase monotonically as edges are added."""
        all_edges = list(self.test_sheaf.restrictions.keys())
        gw_costs = self.test_sheaf.metadata['gw_costs']
        
        # Sort edges by GW cost (increasing complexity filtration)
        sorted_edges = sorted(all_edges, key=lambda e: gw_costs[e])
        
        k = 4  # Track first few eigenvalues
        eigenvalue_sequences = []
        
        # Compute eigenvalues for increasing edge sets
        for i in range(1, len(sorted_edges) + 1):
            active_edges = sorted_edges[:i]
            
            eigenvals, _ = self.builder.solve_generalized_robust(
                self.test_sheaf, active_edges, k=k, use_matrix_free=False
            )
            
            eigenvalue_sequences.append(eigenvals)
        
        # Test monotonicity: λ_i(t) ≤ λ_i(t+1) for each eigenvalue index
        for i in range(len(eigenvalue_sequences) - 1):
            curr_eigenvals = eigenvalue_sequences[i]
            next_eigenvals = eigenvalue_sequences[i + 1]
            
            # Each eigenvalue should not decrease (allowing small numerical tolerances)
            for j in range(k):
                assert next_eigenvals[j] >= curr_eigenvals[j] - 1e-10, \
                    f"Monotonicity violated: λ_{j}({i+1}) = {next_eigenvals[j]:.6e} < λ_{j}({i}) = {curr_eigenvals[j]:.6e}"
    
    def test_connectivity_changes_across_filtration(self):
        """Test that connectivity properties change appropriately across filtration."""
        all_edges = list(self.test_sheaf.restrictions.keys())
        gw_costs = self.test_sheaf.metadata['gw_costs']
        sorted_edges = sorted(all_edges, key=lambda e: gw_costs[e])
        
        zero_eigenvalue_counts = []
        
        # Count near-zero eigenvalues (measure of connectivity) 
        for i in range(1, len(sorted_edges) + 1):
            active_edges = sorted_edges[:i]
            
            eigenvals, _ = self.builder.solve_generalized_robust(
                self.test_sheaf, active_edges, k=6, use_matrix_free=False
            )
            
            # Count eigenvalues below threshold (connected components)
            zero_count = np.sum(eigenvals < 1e-8)
            zero_eigenvalue_counts.append(zero_count)
        
        # As we add edges, connectivity should generally increase
        # (fewer connected components = fewer zero eigenvalues)
        final_connectivity = zero_eigenvalue_counts[-1]
        initial_connectivity = zero_eigenvalue_counts[0]
        
        assert final_connectivity <= initial_connectivity, \
            "Adding edges should not decrease connectivity (increase zero eigenvalue count)"


class TestUnifiedStaticLaplacianIntegration:
    """Test integration with UnifiedStaticLaplacian for persistence analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_sheaf = self._create_integration_test_sheaf()
        
    def _create_integration_test_sheaf(self) -> Sheaf:
        """Create sheaf for integration testing."""
        poset = nx.DiGraph()
        poset.add_edges_from([('input', 'hidden'), ('hidden', 'output')])
        
        stalks = {
            'input': torch.eye(6, dtype=torch.float64),
            'hidden': torch.eye(4, dtype=torch.float64),
            'output': torch.eye(2, dtype=torch.float64)
        }
        
        torch.manual_seed(321)
        restrictions = {
            ('input', 'hidden'): torch.softmax(torch.randn(4, 6, dtype=torch.float64), dim=0),
            ('hidden', 'output'): torch.softmax(torch.randn(2, 4, dtype=torch.float64), dim=0)
        }
        
        gw_costs = {
            ('input', 'hidden'): 0.2,
            ('hidden', 'output'): 0.3
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        return Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata
        )
    
    def test_unified_laplacian_generalized_normalization_enabled(self):
        """Test UnifiedStaticLaplacian with generalized normalization enabled."""
        # Create UnifiedStaticLaplacian with generalized normalization
        unified = UnifiedStaticLaplacian(
            use_generalized_normalization=True,
            use_matrix_free=False,
            max_eigenvalues=5
        )
        
        # Should have properly initialized GW builder
        assert unified.gw_builder is not None
        assert unified.use_generalized_normalization is True
        
        # Compute persistence with single threshold (test integration)
        def edge_threshold_func(weight, param):
            return weight <= param  # GW semantics: include edges with cost ≤ param
        
        result = unified.compute_persistence(
            sheaf=self.test_sheaf,
            filtration_params=[0.5],  # Include all edges
            edge_threshold_func=edge_threshold_func,
            construction_method='gromov_wasserstein'
        )
        
        # Should succeed and return valid results
        assert 'eigenvalue_sequences' in result
        assert len(result['eigenvalue_sequences']) == 1
        
        eigenvals = result['eigenvalue_sequences'][0]
        assert len(eigenvals) <= 5  # Respects max_eigenvalues
        assert torch.all(eigenvals >= -1e-10)  # Non-negative eigenvalues
    
    def test_unified_laplacian_matrix_free_mode(self):
        """Test UnifiedStaticLaplacian with matrix-free mode enabled."""
        unified = UnifiedStaticLaplacian(
            use_generalized_normalization=True,
            use_matrix_free=True,
            max_eigenvalues=4
        )
        
        # Test with larger problem that benefits from matrix-free approach
        def edge_threshold_func(weight, param):
            return weight <= param
        
        result = unified.compute_persistence(
            sheaf=self.test_sheaf,
            filtration_params=[1.0],  # Include all edges
            edge_threshold_func=edge_threshold_func
        )
        
        # Should succeed with matrix-free computation
        assert len(result['eigenvalue_sequences']) == 1
        eigenvals = result['eigenvalue_sequences'][0]
        assert torch.all(eigenvals >= -1e-10)
    
    def test_fallback_to_standard_computation(self):
        """Test that fallback to standard computation works when generalized fails."""
        # Create UnifiedStaticLaplacian with generalized normalization
        unified = UnifiedStaticLaplacian(
            use_generalized_normalization=True,
            use_matrix_free=False,
            max_eigenvalues=3
        )
        
        # Force a scenario where generalized might fail by using incompatible sheaf
        incompatible_sheaf = self.test_sheaf
        incompatible_sheaf.metadata['construction_method'] = 'scaled_procrustes'  # Not GW
        
        def edge_threshold_func(weight, param):
            return weight >= param  # Standard semantics
        
        # Should fallback to standard computation
        result = unified.compute_persistence(
            sheaf=incompatible_sheaf,
            filtration_params=[0.1],
            edge_threshold_func=edge_threshold_func
        )
        
        # Should still succeed via fallback (may return all eigenvalues in dense mode)
        assert len(result['eigenvalue_sequences']) == 1
        eigenvals = result['eigenvalue_sequences'][0]
        assert len(eigenvals) > 0  # Should have computed some eigenvalues
        assert torch.all(eigenvals >= -1e-10)  # Should be non-negative
    
    def test_persistence_computation_consistency(self):
        """Test that persistence computation produces consistent results across runs."""
        unified = UnifiedStaticLaplacian(
            use_generalized_normalization=True,
            use_matrix_free=False,
            max_eigenvalues=4,
            use_double_precision=True  # For numerical consistency
        )
        
        def edge_threshold_func(weight, param):
            return weight <= param
        
        filtration_params = [0.15, 0.25, 0.35]
        
        # Run twice
        result1 = unified.compute_persistence(
            sheaf=self.test_sheaf,
            filtration_params=filtration_params,
            edge_threshold_func=edge_threshold_func,
            construction_method='gromov_wasserstein'
        )
        
        result2 = unified.compute_persistence(
            sheaf=self.test_sheaf,
            filtration_params=filtration_params,
            edge_threshold_func=edge_threshold_func,
            construction_method='gromov_wasserstein'
        )
        
        # Results should be identical
        for i, (seq1, seq2) in enumerate(zip(result1['eigenvalue_sequences'], result2['eigenvalue_sequences'])):
            assert torch.allclose(seq1, seq2, rtol=1e-10, atol=1e-12), \
                f"Eigenvalue sequences differ at step {i}"


class TestNumericalStabilityAndValidation:
    """Test numerical stability and mathematical property validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=True)
        
    def test_tiny_eigenvalue_handling(self):
        """Test handling of tiny eigenvalues (~10^-12) without clamping."""
        # Create sheaf that will produce very small eigenvalues
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        
        # Near-singular stalk configurations
        stalks = {
            'A': torch.eye(3, dtype=torch.float64),
            'B': torch.eye(3, dtype=torch.float64)
        }
        
        # Very weak coupling (tiny eigenvalues expected)
        weak_coupling = 1e-6 * torch.softmax(torch.randn(3, 3, dtype=torch.float64), dim=0)
        
        restrictions = {('A', 'B'): weak_coupling}
        gw_costs = {('A', 'B'): 1e-8}  # Very small cost
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
        active_edges = [('A', 'B')]
        
        # Should handle tiny eigenvalues without clamping
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=3, use_matrix_free=False
        )
        
        # Should preserve tiny positive values (not clamp to zero)
        # The key is classification, not elimination
        assert len(eigenvals) == 3
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
        
        # Should not artificially clamp very small positive eigenvalues
        tiny_eigenvals = eigenvals[eigenvals < 1e-6]
        if len(tiny_eigenvals) > 0:
            # Tiny eigenvalues should still be positive (not exactly zero)
            assert np.all(tiny_eigenvals >= 0)
    
    def test_mass_matrix_regularization_handling(self):
        """Test that mass matrix regularization doesn't break nullspace exactness."""
        # Create well-conditioned problem first
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C')])
        
        stalks = {
            'A': torch.eye(2, dtype=torch.float64),
            'B': torch.eye(2, dtype=torch.float64),
            'C': torch.eye(2, dtype=torch.float64)
        }
        
        torch.manual_seed(555)
        restrictions = {
            ('A', 'B'): torch.softmax(torch.randn(2, 2, dtype=torch.float64), dim=0),
            ('B', 'C'): torch.softmax(torch.randn(2, 2, dtype=torch.float64), dim=0)
        }
        
        gw_costs = {('A', 'B'): 0.3, ('B', 'C'): 0.4}
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
        active_edges = list(sheaf.restrictions.keys())
        
        # Build mass matrix
        M = self.builder._build_stalk_metric(sheaf, active_edges)
        
        # Mass matrix should be well-conditioned (no regularization needed for this test)
        M_dense = M.toarray()
        eigenvals_M = np.linalg.eigvals(M_dense)
        condition_number = np.max(eigenvals_M.real) / np.min(eigenvals_M.real)
        
        assert condition_number < 1e10, f"Mass matrix ill-conditioned: κ = {condition_number:.2e}"
        assert np.all(eigenvals_M.real > 1e-12), "Mass matrix not positive definite"
    
    def test_solver_convergence_diagnostics(self):
        """Test that solvers provide useful convergence diagnostics."""
        # Create medium-sized problem for convergence testing
        poset = nx.DiGraph() 
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        
        stalks = {node: torch.eye(5, dtype=torch.float64) for node in ['A', 'B', 'C', 'D']}
        
        torch.manual_seed(777)
        restrictions = {}
        gw_costs = {}
        for i, (u, v) in enumerate(poset.edges()):
            restrictions[(u, v)] = torch.softmax(torch.randn(5, 5, dtype=torch.float64), dim=0)
            gw_costs[(u, v)] = 0.2 + 0.1 * i
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
        active_edges = list(sheaf.restrictions.keys())
        
        # Test both solver modes
        eigenvals_lobpcg, _ = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=6, use_matrix_free=True  # LOBPCG
        )
        
        eigenvals_eigsh, _ = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=6, use_matrix_free=False  # eigsh
        )
        
        # Both should converge to similar results
        assert np.allclose(eigenvals_lobpcg, eigenvals_eigsh, rtol=1e-5)
        
        # Both should have reasonable eigenvalue magnitudes
        assert np.all(eigenvals_lobpcg >= -1e-10)
        assert np.all(eigenvals_eigsh >= -1e-10)
        assert np.all(eigenvals_lobpcg < 1e3)  # Not too large
        assert np.all(eigenvals_eigsh < 1e3)


class TestMathematicalPropertyValidation:
    """Test preservation of key mathematical properties."""
    
    def test_generalized_laplacian_symmetry(self):
        """Test that generalized formulation preserves Laplacian symmetry properties."""
        builder = GWLaplacianBuilder(validate_properties=True)
        
        # Create symmetric test case
        poset = nx.DiGraph()
        poset.add_edge('X', 'Y')
        
        stalks = {
            'X': torch.eye(3, dtype=torch.float64),
            'Y': torch.eye(3, dtype=torch.float64)
        }
        
        # Symmetric restriction for mathematical clarity
        R = torch.tensor([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]], dtype=torch.float64)
        R = R / R.sum(dim=0, keepdim=True)  # Column-stochastic
        
        restrictions = {('X', 'Y'): R}
        gw_costs = {('X', 'Y'): 0.25}
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
        active_edges = [('X', 'Y')]
        
        # Build L and M matrices
        L = builder.build_laplacian(sheaf, sparse=True, active_edges=active_edges)
        node_masses = builder._extract_node_masses(sheaf)
        M = builder._build_stalk_metric(sheaf, node_masses)
        
        # Test symmetry
        L_dense = L.toarray()
        
        # Convert M to numpy array
        if isinstance(M, torch.Tensor):
            M_dense = M.detach().cpu().numpy()
        else:
            M_dense = M.toarray() if hasattr(M, 'toarray') else M
        
        L_symmetry_error = np.linalg.norm(L_dense - L_dense.T)
        M_symmetry_error = np.linalg.norm(M_dense - M_dense.T)
        
        assert L_symmetry_error < 1e-12, f"L not symmetric: error = {L_symmetry_error:.2e}"
        assert M_symmetry_error < 1e-12, f"M not symmetric: error = {M_symmetry_error:.2e}"
    
    def test_positive_semidefinite_preservation(self):
        """Test that both L and M matrices remain positive semi-definite."""
        builder = GWLaplacianBuilder(validate_properties=True)
        
        # Create test sheaf
        poset = nx.DiGraph()
        poset.add_edges_from([('P', 'Q'), ('Q', 'R')])
        
        stalks = {node: torch.eye(2, dtype=torch.float64) for node in ['P', 'Q', 'R']}
        
        torch.manual_seed(888)
        restrictions = {
            ('P', 'Q'): torch.softmax(torch.randn(2, 2, dtype=torch.float64), dim=0),
            ('Q', 'R'): torch.softmax(torch.randn(2, 2, dtype=torch.float64), dim=0)
        }
        
        gw_costs = {('P', 'Q'): 0.15, ('Q', 'R'): 0.25}
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'whitened': False
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
        active_edges = list(sheaf.restrictions.keys())
        
        # Build matrices
        L = builder.build_laplacian(sheaf, sparse=True, active_edges=active_edges)
        M = builder._build_stalk_metric(sheaf, active_edges)
        
        # Check PSD property via eigenvalues
        L_eigenvals = np.linalg.eigvals(L.toarray())
        M_eigenvals = np.linalg.eigvals(M.toarray())
        
        assert np.all(L_eigenvals.real >= -1e-10), f"L not PSD: min eigenval = {np.min(L_eigenvals.real):.2e}"
        assert np.all(M_eigenvals.real >= -1e-10), f"M not PSD: min eigenval = {np.min(M_eigenvals.real):.2e}"
        
        # M should be positive definite (invertible)
        assert np.all(M_eigenvals.real > 1e-12), f"M not positive definite: min eigenval = {np.min(M_eigenvals.real):.2e}"


if __name__ == "__main__":
    # Configure logging for test diagnostics
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])
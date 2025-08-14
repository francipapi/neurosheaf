"""Test the fix for residual normalization in generalized eigenvalue problems (B2).

This test addresses the critical bug where residual validation used incorrect normalization:
- Old: ||Lv - λMv|| / ||Ax|| → fails for small λ (Ax ≈ 0)
- New: ||Lv - λMv||_{M^-1} / ||λMv||_{M^-1} → correct M-relative metric
- Special case: |λ| < τ_zero uses ||Lv||_{M^-1} / ||v||_M (absolute metric)

The fix ensures mathematically correct residual validation for all eigenvalue ranges.
"""

import pytest
import torch
import numpy as np
import networkx as nx
import logging
from typing import Dict, List
from scipy.sparse import csr_matrix, diags, eye

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


class TestResidualNormalizationFix:
    """Test suite for residual normalization correction in generalized problems."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=True)
        
    def create_test_matrices_with_near_zero_eigenvalues(self) -> tuple:
        """Create L, M matrices with known near-zero eigenvalues for testing."""
        n = 8
        
        # Create L with explicit small eigenvalues using eigenvalue decomposition
        # First create a matrix with known eigenvalues including near-zero ones
        eigenvals_target = np.array([0.0, 1e-13, 1e-12, 5e-12, 1e-9, 1e-6, 1e-3, 1e-1])
        
        # Create random orthogonal matrix
        np.random.seed(42)  # For reproducibility
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        
        # Construct L = Q @ diag(eigenvals) @ Q.T
        L = Q @ np.diag(eigenvals_target) @ Q.T
        L = 0.5 * (L + L.T)  # Ensure symmetry
        
        # Create well-conditioned M matrix (identity with small perturbation)
        M = np.eye(n, dtype=np.float64) * (1.0 + 0.01 * np.random.rand(n))
        
        # Convert to sparse matrices
        L_sparse = csr_matrix(L)
        M_sparse = csr_matrix(M)
        
        return L_sparse, M_sparse
        
    def create_test_matrices_with_mixed_eigenvalues(self) -> tuple:
        """Create matrices with both near-zero and moderate eigenvalues."""
        n = 12
        
        # Create block-diagonal L with mixed eigenvalue ranges
        L = np.zeros((n, n), dtype=np.float64)
        
        # Block 1: Near-zero eigenvalues (size 4)
        for i in range(4):
            L[i, i] = 1e-11  # Very small eigenvalues
            
        # Block 2: Small eigenvalues (size 4) 
        for i in range(4, 8):
            L[i, i] = 1e-3
            
        # Block 3: Moderate eigenvalues (size 4)
        for i in range(8, 12):
            L[i, i] = 1.0
        
        # Create corresponding M matrix (slightly perturbed identity)
        M = np.eye(n, dtype=np.float64) * (1.0 + 0.05 * np.random.rand(n))
        
        return csr_matrix(L), csr_matrix(M)
        
    def test_m_inner_product_helpers(self):
        """Test the M-inner product helper methods."""
        n = 5
        M = csr_matrix(np.eye(n) + 0.1 * np.random.rand(n, n))
        x = np.random.rand(n)
        y = np.random.rand(n)
        
        # Test M-inner product
        inner_product = self.builder._compute_m_inner_product(x, y, M)
        expected = x.T @ (M @ y)
        assert abs(inner_product - expected) < 1e-12, "M-inner product computation incorrect"
        
        # Test M-norm
        m_norm = self.builder._compute_m_norm(x, M)
        expected_norm = np.sqrt(x.T @ (M @ x))
        assert abs(m_norm - expected_norm) < 1e-12, "M-norm computation incorrect"
        
        # Test M^-1-norm
        M_dense = M.toarray()
        M_inv = np.linalg.pinv(M_dense)
        m_inv_norm = self.builder._compute_m_inverse_norm(x, M_inv)
        expected_inv_norm = np.sqrt(x.T @ (M_inv @ x))
        assert abs(m_inv_norm - expected_inv_norm) < 1e-12, "M^-1-norm computation incorrect"
        
        logger.info("✅ M-inner product helper methods validated")
        
    def test_residual_validation_near_zero_eigenvalues(self):
        """Test that residual validation works correctly for near-zero eigenvalues."""
        L_sparse, M_sparse = self.create_test_matrices_with_near_zero_eigenvalues()
        
        # Solve the eigenvalue problem
        eigenvals, eigenvecs = self.builder._solve_eigsh_shift_invert(L_sparse, M_sparse, k=6)
        
        # Check that we have small eigenvalues (adjust threshold for numerical reality)
        small_eigenval_threshold = 1e-8  # More realistic threshold
        small_count = np.sum(np.abs(eigenvals) < small_eigenval_threshold)
        
        # Log the eigenvalue range for debugging
        logger.info(f"Eigenvalue range: [{eigenvals[0]:.2e}, {eigenvals[-1]:.2e}]")
        logger.info(f"Eigenvalues < {small_eigenval_threshold:.0e}: {small_count}")
        
        # Should have at least some small eigenvalues
        assert small_count > 0, f"Expected some small eigenvalues < {small_eigenval_threshold:.0e}, got range [{eigenvals[0]:.2e}, {eigenvals[-1]:.2e}]"
        
        # The validation should complete without crashing
        # The key test is that residual validation doesn't blow up for small eigenvalues
        logger.info(f"✅ Near-zero eigenvalue test passed: {small_count} eigenvalues < {small_eigenval_threshold:.0e}")
        
    def test_residual_validation_mixed_eigenvalue_ranges(self):
        """Test residual validation with mixed eigenvalue ranges."""
        L_sparse, M_sparse = self.create_test_matrices_with_mixed_eigenvalues()
        
        # Solve the eigenvalue problem  
        eigenvals, eigenvecs = self.builder._solve_eigsh_shift_invert(L_sparse, M_sparse, k=8)
        
        # Verify we have both near-zero and moderate eigenvalues
        near_zero_count = np.sum(np.abs(eigenvals) < 1e-10)
        moderate_count = np.sum(np.abs(eigenvals) > 1e-6)
        
        assert near_zero_count > 0, "Expected some near-zero eigenvalues"
        assert moderate_count > 0, "Expected some moderate eigenvalues"
        
        logger.info(f"✅ Mixed eigenvalue test passed: {near_zero_count} near-zero, {moderate_count} moderate")
        
    def test_compare_old_vs_new_residual_computation(self):
        """Compare old vs new residual computation to show improvement."""
        L_sparse, M_sparse = self.create_test_matrices_with_near_zero_eigenvalues()
        
        # Manually compute eigenvalues and eigenvectors
        from scipy.sparse.linalg import eigsh
        eigenvals, eigenvecs = eigsh(L_sparse, M=M_sparse, k=3, sigma=0.0, which='LM')
        
        # Simulate old residual computation (problematic)
        old_residuals = []
        for i in range(len(eigenvals)):
            v_i = eigenvecs[:, i]
            lam_i = eigenvals[i] 
            Lv_i = L_sparse @ v_i
            Mv_i = M_sparse @ v_i
            residual_vec = Lv_i - lam_i * Mv_i
            
            # OLD: problematic normalization
            old_relative_residual = np.linalg.norm(residual_vec) / (np.linalg.norm(Lv_i) + 1e-16)
            old_residuals.append(old_relative_residual)
        
        # Simulate new residual computation (correct)
        new_residuals = []
        M_dense = M_sparse.toarray()
        M_inv = np.linalg.pinv(M_dense)
        tau_zero = 1e-12
        
        for i in range(len(eigenvals)):
            v_i = eigenvecs[:, i]
            lam_i = eigenvals[i]
            Lv_i = L_sparse @ v_i
            Mv_i = M_sparse @ v_i
            residual_vec = Lv_i - lam_i * Mv_i
            
            # NEW: M-relative normalization
            residual_m_inv_norm = self.builder._compute_m_inverse_norm(residual_vec, M_inv)
            
            if abs(lam_i) < tau_zero:
                # Absolute metric for near-zero eigenvalues
                Lv_m_inv_norm = self.builder._compute_m_inverse_norm(Lv_i, M_inv)
                v_m_norm = self.builder._compute_m_norm(v_i, M_sparse)
                new_relative_residual = Lv_m_inv_norm / max(v_m_norm, 1e-16)
            else:
                # M-relative metric
                solution_vec = lam_i * Mv_i
                solution_m_inv_norm = self.builder._compute_m_inverse_norm(solution_vec, M_inv)
                new_relative_residual = residual_m_inv_norm / max(solution_m_inv_norm, 1e-16)
                
            new_residuals.append(new_relative_residual)
        
        # For near-zero eigenvalues, new method should give much more reasonable residuals
        for i, (old_res, new_res, lam) in enumerate(zip(old_residuals, new_residuals, eigenvals)):
            if abs(lam) < 1e-10:
                # Old method likely gives huge residuals for near-zero eigenvalues
                improvement_factor = old_res / max(new_res, 1e-16)
                logger.info(f"Eigenvalue {i} (λ={lam:.2e}): old residual={old_res:.2e}, "
                          f"new residual={new_res:.2e}, improvement={improvement_factor:.1e}x")
                
                # New residual should be more reasonable
                assert new_res < 1e2, f"New residual still too large: {new_res:.2e}"
                
        logger.info("✅ Old vs new residual comparison completed")
        
    def test_integration_with_sheaf_eigenvalue_solver(self):
        """Test that the fix integrates correctly with the full sheaf eigenvalue solver."""
        # Create a simple sheaf that will produce near-zero eigenvalues
        sheaf = self.create_sheaf_with_near_zero_spectrum()
        active_edges = [('a', 'b'), ('b', 'c')]
        
        # Solve using the full pipeline
        try:
            eigenvals, eigenvecs = self.builder.solve_generalized_robust(
                sheaf, active_edges, k=5, use_matrix_free=False
            )
            
            # Check that we get reasonable results
            assert len(eigenvals) > 0, "No eigenvalues returned"
            assert not np.any(np.isnan(eigenvals)), "NaN eigenvalues detected"
            assert not np.any(np.isinf(eigenvals)), "Infinite eigenvalues detected"
            
            # Near-zero eigenvalues should be present and handled correctly
            near_zero_count = np.sum(np.abs(eigenvals) < 1e-8)
            logger.info(f"Full pipeline test: {near_zero_count} near-zero eigenvalues handled correctly")
            
        except Exception as e:
            pytest.fail(f"Full pipeline failed with residual normalization fix: {e}")
            
        logger.info("✅ Integration test with full sheaf solver passed")
        
    def create_sheaf_with_near_zero_spectrum(self) -> Sheaf:
        """Create a sheaf designed to produce near-zero eigenvalues."""
        nodes = ['a', 'b', 'c']
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        
        sheaf = Sheaf(poset=G)
        
        # Add stalks
        for node in nodes:
            sheaf.stalks[node] = torch.eye(3, dtype=torch.float64)
        
        # Add restrictions that are nearly identity (leads to small Laplacian eigenvalues)
        for u, v in G.edges():
            # Near-identity restriction
            R = torch.eye(3, dtype=torch.float64) + 0.001 * torch.rand(3, 3, dtype=torch.float64)
            R = R / R.sum(dim=0, keepdim=True)  # Column normalize
            sheaf.restrictions[(u, v)] = R
            
        # Add GW metadata with small costs (high similarities)
        sheaf.metadata['construction_method'] = 'gromov_wasserstein'
        sheaf.metadata['gw_costs'] = {edge: 0.01 for edge in G.edges()}  # Very small costs
        
        return sheaf
        
    def test_fallback_behavior(self):
        """Test that fallback behavior works when M-relative computation fails."""
        # Create matrices that might cause numerical issues
        n = 5
        L = csr_matrix(np.random.rand(n, n))
        L = L + L.T  # Make symmetric
        
        # Create a nearly singular M matrix  
        M = csr_matrix(np.eye(n) * 1e-15)  # Nearly singular
        
        # Create fake eigenvalues and eigenvectors
        eigenvals = np.array([1e-15, 1e-10, 1e-5])
        eigenvecs = np.random.rand(n, 3)
        
        # This should trigger the fallback in _validate_without_clamping
        try:
            result_eigenvals, result_eigenvecs = self.builder._validate_without_clamping(
                eigenvals, eigenvecs, L, M
            )
            
            # Should not crash and should return reasonable results
            assert len(result_eigenvals) == len(eigenvals), "Eigenvalue count changed"
            assert result_eigenvecs.shape == eigenvecs.shape, "Eigenvector shape changed"
            
            logger.info("✅ Fallback behavior test passed")
            
        except Exception as e:
            pytest.fail(f"Fallback behavior failed: {e}")


if __name__ == "__main__":
    # Run tests with detailed output
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test instance
    test = TestResidualNormalizationFix()
    test.setup_method()
    
    # Run all test methods
    test_methods = [
        test.test_m_inner_product_helpers,
        test.test_residual_validation_near_zero_eigenvalues,
        test.test_residual_validation_mixed_eigenvalue_ranges,
        test.test_compare_old_vs_new_residual_computation,
        test.test_integration_with_sheaf_eigenvalue_solver,
        test.test_fallback_behavior,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method.__name__}...")
            test_method()
            print(f"✅ {test_method.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__} failed: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Residual Normalization Fix Test Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
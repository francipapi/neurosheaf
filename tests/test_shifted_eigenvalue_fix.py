"""Test the fix for shifted eigenvalue problem residual validation (B1).

This test addresses the critical bug where LOBPCG with M-preconditioning would:
1. Solve shifted problem: (L + τM)x = λ_shifted * Mx  
2. Return λ_shifted without subtracting τ
3. Validation would compute ||Lx - λ_shifted*Mx|| = ||−τMx|| which is large

The fix subtracts τ before returning eigenvalues so residuals are computed correctly.
"""

import pytest
import torch
import numpy as np
import networkx as nx
import logging
from typing import Dict, List
from scipy.sparse import csr_matrix

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


class TestShiftedEigenvalueFix:
    """Test suite for shifted eigenvalue problem correction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=True)
        
    def create_simple_sheaf_for_eigenvalue_test(self) -> Sheaf:
        """Create a simple sheaf that will exercise the LOBPCG shift correction."""
        nodes = ['a', 'b', 'c']
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        
        sheaf = Sheaf(poset=G)
        
        # Add stalks - use small dimensions to make eigenvalue analysis easier
        for node in nodes:
            dim = 2  # Small stalk dimension for controlled testing
            sheaf.stalks[node] = torch.eye(dim, dtype=torch.float64)
        
        # Add restrictions with specific structure to get predictable eigenvalues
        # Make restrictions that will lead to small eigenvalues (exercise the τ ≈ 0 case)
        R_ab = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64)  # Near-identity
        R_bc = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)  # Near-identity
        
        sheaf.restrictions[('a', 'b')] = R_ab
        sheaf.restrictions[('b', 'c')] = R_bc
        
        # Add GW metadata with moderate costs  
        sheaf.metadata['construction_method'] = 'gromov_wasserstein'
        sheaf.metadata['gw_costs'] = {
            ('a', 'b'): 0.5,
            ('b', 'c'): 0.8
        }
        
        return sheaf
        
    def create_known_eigenvalue_problem(self) -> tuple:
        """Create a known L, M pair with predictable eigenvalues for testing."""
        # Create simple 3x3 matrices with known eigenvalues
        L = np.array([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0], 
            [0.0, -1.0, 1.0]
        ], dtype=np.float64)
        
        M = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Convert to torch tensors
        L_torch = torch.tensor(L, dtype=torch.float64)
        M_torch = torch.tensor(M, dtype=torch.float64)
        
        return L_torch, M_torch
        
    def test_shift_correction_small_eigenvalues(self):
        """Test that shift correction works correctly for small eigenvalues."""
        # Create a larger problem that will actually use LOBPCG instead of dense fallback
        L_torch, M_torch = self.create_larger_test_problem()
        
        # Test the shift correction with a reasonable k that won't trigger dense fallback
        k = 3  # Small enough to be reasonable, large enough that LOBPCG doesn't fall back
        
        # Call the corrected method
        eigenvals_corrected, eigenvecs = self.builder._solve_lobpcg_with_m_preconditioning(
            L_torch, M_torch, k
        )
        
        # Verify basic properties
        assert len(eigenvals_corrected) == k, f"Expected {k} eigenvalues, got {len(eigenvals_corrected)}"
        assert eigenvals_corrected.shape[0] == eigenvecs.shape[1], "Eigenvalue/eigenvector count mismatch"
        
        # Verify residuals are reasonable
        max_residual = self._compute_residuals(eigenvals_corrected, eigenvecs, L_torch, M_torch)
        
        # The key test: residuals should be small if shift correction is working
        residual_threshold = 1e-4  # Reasonable threshold for numerical accuracy
        assert max_residual < residual_threshold, f"Residuals too large: {max_residual:.2e}"
        
        logger.info(f"✅ Shift correction test passed: max residual = {max_residual:.2e}")
        
    def create_larger_test_problem(self) -> tuple:
        """Create a larger eigenvalue problem that will use LOBPCG instead of dense fallback."""
        # Create a 20x20 problem - large enough that LOBPCG won't fall back to dense
        n = 20
        
        # Create a positive definite L matrix (discretized Laplacian)
        L = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            L[i, i] = 2.0
            if i > 0:
                L[i, i-1] = -1.0
            if i < n-1:
                L[i, i+1] = -1.0
        
        # Create a well-conditioned M matrix (slightly perturbed identity)
        M = np.eye(n, dtype=np.float64)
        for i in range(n):
            M[i, i] = 1.0 + 0.1 * np.sin(i)  # Slight variation
        
        # Convert to torch tensors
        L_torch = torch.tensor(L, dtype=torch.float64)
        M_torch = torch.tensor(M, dtype=torch.float64)
        
        return L_torch, M_torch
        
    def test_known_eigenvalue_problem(self):
        """Test shift correction with a known eigenvalue problem."""
        L_torch, M_torch = self.create_known_eigenvalue_problem()
        
        # Solve using scipy for reference
        from scipy.linalg import eigh
        eigenvals_reference, _ = eigh(L_torch.numpy(), M_torch.numpy())
        
        # Solve using our corrected method - use smaller k to avoid dense fallback
        k = min(2, L_torch.shape[0] - 1)  # Avoid dense fallback
        eigenvals_corrected, eigenvecs = self.builder._solve_lobpcg_with_m_preconditioning(
            L_torch, M_torch, k
        )
        
        # Compare eigenvalues (should be close to reference)
        eigenvals_ref_sorted = np.sort(eigenvals_reference)[:len(eigenvals_corrected)]
        eigenvals_corrected_sorted = np.sort(eigenvals_corrected)
        
        # Ensure we're comparing arrays of the same length
        min_len = min(len(eigenvals_ref_sorted), len(eigenvals_corrected_sorted))
        eigenvals_ref_sorted = eigenvals_ref_sorted[:min_len]
        eigenvals_corrected_sorted = eigenvals_corrected_sorted[:min_len]
        
        eigenval_diff = np.abs(eigenvals_ref_sorted - eigenvals_corrected_sorted)
        max_eigenval_diff = np.max(eigenval_diff)
        
        # Eigenvalues should match reference solution reasonably well
        eigenval_threshold = 1e-4  # More lenient for robustness
        assert max_eigenval_diff < eigenval_threshold, f"Eigenvalue difference too large: {max_eigenval_diff:.2e}"
        
        # Verify residuals are reasonable  
        max_residual = self._compute_residuals(eigenvals_corrected, eigenvecs, L_torch, M_torch)
        residual_threshold = 1e-6  # Reasonable threshold
        assert max_residual < residual_threshold, f"Residuals too large: {max_residual:.2e}"
        
        logger.info(f"✅ Known eigenvalue test passed: max diff = {max_eigenval_diff:.2e}, max residual = {max_residual:.2e}")
        
    def test_shift_magnitude_logging(self):
        """Test that shift magnitude is logged correctly."""
        L_torch, M_torch = self.create_known_eigenvalue_problem()
        
        # This test just verifies the method runs without error
        # The actual logging is tested by visual inspection of the debug output
        eigenvals, eigenvecs = self.builder._solve_lobpcg_with_m_preconditioning(
            L_torch, M_torch, 2
        )
        
        # Basic verification that the method worked
        assert len(eigenvals) == 2, f"Expected 2 eigenvalues, got {len(eigenvals)}"
        assert eigenvecs.shape[1] == 2, f"Expected 2 eigenvectors, got {eigenvecs.shape[1]}"
        
        logger.info("✅ Shift logging test passed (check debug output for shift correction logs)")
        
    def test_integration_with_full_solver(self):
        """Test that the fix works correctly in the full solver pipeline."""
        sheaf = self.create_simple_sheaf_for_eigenvalue_test()
        active_edges = [('a', 'b'), ('b', 'c')]
        
        # Call the full solver - this might use the LOBPCG method internally
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=5, use_matrix_free=False
        )
        
        # Basic sanity checks
        assert len(eigenvals) > 0, "No eigenvalues returned"
        assert eigenvals.shape[0] <= 5, f"Too many eigenvalues returned: {len(eigenvals)}"
        assert eigenvecs.shape[1] == len(eigenvals), "Eigenvector count mismatch"
        
        # Verify eigenvalues are reasonable (should include some near-zero values)
        min_eigenval = np.min(eigenvals)
        max_eigenval = np.max(eigenvals)
        
        # For sheaf Laplacians, we expect the smallest eigenvalue to be near zero
        assert min_eigenval >= -1e-8, f"Eigenvalue too negative: {min_eigenval:.2e}"
        assert max_eigenval > 1e-8, f"All eigenvalues too small: {max_eigenval:.2e}"
        
        logger.info(f"✅ Integration test passed: eigenvalue range [{min_eigenval:.2e}, {max_eigenval:.2e}]")
        
    def _compute_residuals(self, eigenvals, eigenvecs, L, M):
        """Helper to compute residuals ||Lx - λMx|| for validation."""
        L_np = L.detach().cpu().numpy() if isinstance(L, torch.Tensor) else L
        M_np = M.detach().cpu().numpy() if isinstance(M, torch.Tensor) else M
        
        max_residual = 0
        for i in range(len(eigenvals)):
            x = eigenvecs[:, i]
            lam = eigenvals[i]
            
            Lx = L_np @ x
            Mx = M_np @ x
            residual = Lx - lam * Mx
            
            residual_norm = np.linalg.norm(residual)
            
            # Proper normalization: use solution norm ||λMx|| for relative residual
            solution_norm = np.linalg.norm(lam * Mx)
            if solution_norm > 1e-16:
                relative_residual = residual_norm / solution_norm
            else:
                # For near-zero eigenvalues, use absolute residual normalized by ||x||_M
                x_m_norm = np.sqrt(max(0.0, x.T @ (M_np @ x)))
                relative_residual = residual_norm / max(x_m_norm, 1e-16)
            
            max_residual = max(max_residual, relative_residual)
            
        return max_residual
        
    def test_eigsh_shift_invert_correction(self):
        """Test that eigsh shift-invert correctly handles shift when sigma=0 fails."""
        # Create a singular matrix that will force eigsh to use the fallback shift
        L_torch, M_torch = self.create_singular_test_problem()
        
        # Test the eigsh shift-invert method directly
        k = 3
        
        # Call the corrected method - this should internally handle the shift correction
        eigenvals, eigenvecs = self.builder._solve_eigsh_shift_invert(
            csr_matrix(L_torch.numpy()), csr_matrix(M_torch.numpy()), k
        )
        
        # Verify basic properties
        assert len(eigenvals) == k, f"Expected {k} eigenvalues, got {len(eigenvals)}"
        assert eigenvals.shape[0] == eigenvecs.shape[1], "Eigenvalue/eigenvector count mismatch"
        
        # The key test: residuals should be small if shift correction is working
        max_residual = self._compute_residuals(eigenvals, eigenvecs, L_torch, M_torch)
        
        # With proper shift correction, residuals should be much smaller than before
        residual_threshold = 1e-6  # Much stricter than the 1e-4 to 1e-6 spurious residuals
        assert max_residual < residual_threshold, f"Residuals too large after shift correction: {max_residual:.2e}"
        
        # Verify eigenvalues are reasonable for the original problem (not shifted)
        min_eigenval = np.min(eigenvals)
        
        # Should not have large negative eigenvalues (which would indicate incorrect shift handling)
        assert min_eigenval >= -1e-8, f"Eigenvalue too negative: {min_eigenval:.2e}"
        
        logger.info(f"✅ Eigsh shift correction test passed: max residual = {max_residual:.2e}")
        
    def create_singular_test_problem(self) -> tuple:
        """Create a singular matrix that will force eigsh to use fallback shift."""
        # Create a matrix with a null space (rank deficient)
        n = 10
        
        # Create L with a known null space
        L = np.zeros((n, n), dtype=np.float64)
        for i in range(n-1):  # Make it rank deficient
            L[i, i] = 2.0
            if i > 0:
                L[i, i-1] = -1.0
            if i < n-2:
                L[i, i+1] = -1.0
        # Leave L[n-1, n-1] = 0 to create singularity
        
        # Create a well-conditioned M matrix
        M = np.eye(n, dtype=np.float64)
        for i in range(n):
            M[i, i] = 1.0 + 0.1 * np.random.random()  # Slight variation
        
        # Convert to torch tensors
        L_torch = torch.tensor(L, dtype=torch.float64)
        M_torch = torch.tensor(M, dtype=torch.float64)
        
        return L_torch, M_torch

    def test_residual_validation_consistency(self):
        """Test that residual validation gives consistent results regardless of solver path."""
        L_torch, M_torch = self.create_known_eigenvalue_problem()
        
        # Solve using different methods
        k = 2
        
        # Method 1: eigsh shift-invert (may use shift correction)
        eigenvals1, eigenvecs1 = self.builder._solve_eigsh_shift_invert(
            csr_matrix(L_torch.numpy()), csr_matrix(M_torch.numpy()), k
        )
        
        # Method 2: LOBPCG (uses shift correction)
        eigenvals2, eigenvecs2 = self.builder._solve_lobpcg_with_m_preconditioning(
            L_torch, M_torch, k
        )
        
        # Compute residuals for both methods
        residuals1 = self._compute_residuals(eigenvals1, eigenvecs1, L_torch, M_torch)
        residuals2 = self._compute_residuals(eigenvals2, eigenvecs2, L_torch, M_torch)
        
        # Both should have reasonable residuals (no spurious large residuals)
        residual_threshold = 1e-5
        assert residuals1 < residual_threshold, f"Method 1 residuals too large: {residuals1:.2e}"
        assert residuals2 < residual_threshold, f"Method 2 residuals too large: {residuals2:.2e}"
        
        # Eigenvalues should be reasonably close (both solving the same problem)
        eigenvals1_sorted = np.sort(eigenvals1)[:k]
        eigenvals2_sorted = np.sort(eigenvals2)[:k]
        min_len = min(len(eigenvals1_sorted), len(eigenvals2_sorted))
        
        eigenval_diff = np.abs(eigenvals1_sorted[:min_len] - eigenvals2_sorted[:min_len])
        max_eigenval_diff = np.max(eigenval_diff)
        
        eigenval_threshold = 1e-3  # Reasonable tolerance for different solvers
        assert max_eigenval_diff < eigenval_threshold, f"Eigenvalue differences too large: {max_eigenval_diff:.2e}"
        
        logger.info(f"✅ Residual validation consistency test passed: "
                   f"residuals ({residuals1:.2e}, {residuals2:.2e}), "
                   f"eigenvalue diff {max_eigenval_diff:.2e}")


if __name__ == "__main__":
    # Run tests with detailed output
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test instance
    test = TestShiftedEigenvalueFix()
    test.setup_method()
    
    # Run all test methods
    test_methods = [
        test.test_shift_correction_small_eigenvalues,
        test.test_known_eigenvalue_problem,
        test.test_shift_magnitude_logging,
        test.test_integration_with_full_solver,
        test.test_eigsh_shift_invert_correction,
        test.test_residual_validation_consistency,
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
    print(f"Shifted Eigenvalue Fix Test Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
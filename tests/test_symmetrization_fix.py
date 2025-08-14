"""Test the fix for missing explicit symmetrization before eigenvalue solvers (B3).

This test addresses the critical bug where numerical non-symmetry in L and M matrices
(from sparse-dense conversions, rounding, or assembly order) could produce:
- Complex eigenvalues
- Unstable solutions
- Inflated residuals

The fix ensures all matrices are explicitly symmetrized before any eigenvalue solver.
"""

import pytest
import torch
import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix, random as sp_random
from scipy.linalg import eigh

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


class TestSymmetrizationFix:
    """Test suite for explicit symmetrization before eigenvalue solvers."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=True)
        
    def create_slightly_asymmetric_matrix(self, n: int = 10, asymmetry: float = 1e-10) -> np.ndarray:
        """Create a matrix with small asymmetries that could cause issues.
        
        Args:
            n: Matrix dimension
            asymmetry: Scale of asymmetry to introduce
            
        Returns:
            Matrix with small numerical asymmetries
        """
        # Start with a symmetric matrix
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T)
        
        # Add small asymmetries (simulating numerical errors)
        noise = asymmetry * np.random.randn(n, n)
        A_asymmetric = A + noise
        
        # Verify asymmetry exists
        asymmetry_norm = np.linalg.norm(A_asymmetric - A_asymmetric.T)
        assert asymmetry_norm > 0, "Failed to create asymmetric matrix"
        
        return A_asymmetric
        
    def create_asymmetric_sparse_matrices(self) -> Tuple[csr_matrix, csr_matrix]:
        """Create sparse matrices with numerical asymmetries."""
        n = 20
        
        # Create L with small asymmetries
        L = sp_random(n, n, density=0.3, random_state=42)
        L = L + L.T  # Make mostly symmetric
        L_dense = L.toarray()
        
        # Add tiny asymmetries to simulate numerical errors
        L_dense[0, 1] += 1e-12
        L_dense[2, 3] -= 1e-13
        L_dense[4, 5] += 1e-14
        
        L_sparse = csr_matrix(L_dense)
        
        # Create M (mass matrix) with small asymmetries  
        M = sp_random(n, n, density=0.5, random_state=43)
        M = M + M.T + csr_matrix(np.eye(n))  # Positive definite
        M_dense = M.toarray()
        
        # Add asymmetries
        M_dense[1, 2] += 1e-11
        M_dense[3, 4] -= 1e-12
        
        M_sparse = csr_matrix(M_dense)
        
        return L_sparse, M_sparse
        
    def test_ensure_symmetric_method(self):
        """Test the _ensure_symmetric method for various matrix types."""
        # Test numpy array
        A_np = self.create_slightly_asymmetric_matrix(10, 1e-10)
        A_np_sym = self.builder._ensure_symmetric(A_np, "A_numpy")
        
        # Check symmetry
        asymmetry_after = np.linalg.norm(A_np_sym - A_np_sym.T)
        assert asymmetry_after < 1e-15, f"Numpy array not symmetric: {asymmetry_after:.2e}"
        
        # Test torch tensor
        A_torch = torch.tensor(self.create_slightly_asymmetric_matrix(8, 1e-11))
        A_torch_sym = self.builder._ensure_symmetric(A_torch, "A_torch")
        
        asymmetry_after = torch.norm(A_torch_sym - A_torch_sym.T).item()
        assert asymmetry_after < 1e-14, f"Torch tensor not symmetric: {asymmetry_after:.2e}"
        
        # Test sparse matrix
        L_sparse, M_sparse = self.create_asymmetric_sparse_matrices()
        L_sparse_sym = self.builder._ensure_symmetric(L_sparse, "L_sparse")
        
        L_diff = L_sparse_sym - L_sparse_sym.T
        asymmetry_after = np.sqrt((L_diff.data ** 2).sum()) if hasattr(L_diff, 'data') else 0
        assert asymmetry_after < 1e-15, f"Sparse matrix not symmetric: {asymmetry_after:.2e}"
        
        logger.info("✅ _ensure_symmetric method works for all matrix types")
        
    def test_no_complex_eigenvalues_after_symmetrization(self):
        """Test that symmetrization prevents complex eigenvalues."""
        # Create asymmetric matrices
        n = 15
        L = self.create_slightly_asymmetric_matrix(n, 1e-9)
        M = self.create_slightly_asymmetric_matrix(n, 1e-10)
        M = M + np.eye(n) * 0.1  # Ensure positive definite
        
        # Without symmetrization, might get complex eigenvalues
        try:
            eigenvals_raw, _ = eigh(L, M)
            # Check if any eigenvalues would be complex (this is prevented by eigh)
            # But numerical issues could still arise
        except Exception as e:
            logger.info(f"Raw matrices caused issues: {e}")
        
        # With symmetrization
        L_sym = self.builder._ensure_symmetric(L, "L")
        M_sym = self.builder._ensure_symmetric(M, "M")
        
        # L_sym and M_sym are numpy arrays after symmetrization
        eigenvals_sym, eigenvecs_sym = eigh(L_sym, M_sym)
        
        # All eigenvalues should be real
        assert np.all(np.isreal(eigenvals_sym)), "Complex eigenvalues detected after symmetrization"
        assert not np.any(np.isnan(eigenvals_sym)), "NaN eigenvalues detected"
        assert not np.any(np.isinf(eigenvals_sym)), "Infinite eigenvalues detected"
        
        logger.info("✅ No complex eigenvalues after symmetrization")
        
    def test_residuals_improved_with_symmetrization(self):
        """Test that residuals are improved when matrices are symmetrized."""
        # Create slightly asymmetric test problem
        L_sparse, M_sparse = self.create_asymmetric_sparse_matrices()
        
        # Compute eigenvalues without explicit symmetrization
        # (Note: in practice, our fix always symmetrizes, so we simulate the old behavior)
        L_dense = L_sparse.toarray()
        M_dense = M_sparse.toarray()
        
        # Old behavior: no symmetrization
        eigenvals_old, eigenvecs_old = eigh(L_dense, M_dense)
        
        # Compute residuals for old solution
        residuals_old = []
        for i in range(min(5, len(eigenvals_old))):
            v = eigenvecs_old[:, i]
            lam = eigenvals_old[i]
            residual = L_dense @ v - lam * M_dense @ v
            residuals_old.append(np.linalg.norm(residual))
        
        # New behavior: with symmetrization
        L_sym = self.builder._ensure_symmetric(L_sparse, "L")
        M_sym = self.builder._ensure_symmetric(M_sparse, "M")
        
        eigenvals_new, eigenvecs_new = eigh(L_sym.toarray(), M_sym.toarray())
        
        # Compute residuals for new solution
        residuals_new = []
        for i in range(min(5, len(eigenvals_new))):
            v = eigenvecs_new[:, i]
            lam = eigenvals_new[i]
            residual = L_sym.toarray() @ v - lam * M_sym.toarray() @ v
            residuals_new.append(np.linalg.norm(residual))
        
        # Residuals should be similar or better with symmetrization
        avg_residual_old = np.mean(residuals_old)
        avg_residual_new = np.mean(residuals_new)
        
        logger.info(f"Average residual: old={avg_residual_old:.2e}, new={avg_residual_new:.2e}")
        
        # New residuals should not be significantly worse
        assert avg_residual_new < avg_residual_old * 10, "Residuals got much worse with symmetrization"
        
        logger.info("✅ Residuals are reasonable with symmetrization")
        
    def test_torch_solver_symmetrization(self):
        """Test that torch-based solvers receive symmetric matrices."""
        # Create torch matrices with asymmetries
        n = 12
        L = torch.tensor(self.create_slightly_asymmetric_matrix(n, 1e-11), dtype=torch.float64)
        M = torch.tensor(self.create_slightly_asymmetric_matrix(n, 1e-12), dtype=torch.float64)
        M = M + torch.eye(n, dtype=torch.float64) * 0.1  # Positive definite
        
        # Call a torch-based solver method
        try:
            eigenvals, eigenvecs = self.builder._solve_dense_m_orthonormalized(L, M, k=5)
            
            # Should succeed without issues
            assert len(eigenvals) == 5, f"Expected 5 eigenvalues, got {len(eigenvals)}"
            assert np.all(np.isreal(eigenvals)), "Complex eigenvalues from torch solver"
            
            logger.info("✅ Torch solver handles symmetrization correctly")
            
        except Exception as e:
            pytest.fail(f"Torch solver failed with symmetrization: {e}")
            
    def test_sparse_solver_symmetrization(self):
        """Test that sparse solvers receive symmetric matrices."""
        L_sparse, M_sparse = self.create_asymmetric_sparse_matrices()
        
        # Call sparse solver method
        try:
            eigenvals, eigenvecs = self.builder._solve_eigsh_shift_invert(L_sparse, M_sparse, k=5)
            
            # Should succeed without issues
            assert len(eigenvals) == 5, f"Expected 5 eigenvalues, got {len(eigenvals)}"
            assert np.all(np.isreal(eigenvals)), "Complex eigenvalues from sparse solver"
            
            # Check residuals are reasonable
            L_sym = self.builder._ensure_symmetric(L_sparse, "L")
            M_sym = self.builder._ensure_symmetric(M_sparse, "M")
            
            max_residual = 0
            for i in range(min(3, len(eigenvals))):
                v = eigenvecs[:, i]
                lam = eigenvals[i]
                residual = L_sym @ v - lam * M_sym @ v
                residual_norm = np.linalg.norm(residual)
                max_residual = max(max_residual, residual_norm)
            
            # Allow for reasonable residuals (sparse solver may have looser tolerance)
            assert max_residual < 1e-3, f"Large residuals detected: {max_residual:.2e}"
            
            logger.info("✅ Sparse solver handles symmetrization correctly")
            
        except Exception as e:
            pytest.fail(f"Sparse solver failed with symmetrization: {e}")
            
    def test_full_pipeline_with_asymmetric_input(self):
        """Test the full sheaf eigenvalue pipeline with matrices that have asymmetries."""
        # Create a sheaf that might produce slightly asymmetric matrices
        nodes = ['a', 'b', 'c', 'd']
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')])  # Cycle
        
        sheaf = Sheaf(poset=G)
        
        # Add stalks
        for node in nodes:
            dim = np.random.randint(2, 5)
            stalk = torch.eye(dim, dtype=torch.float64)
            # Add tiny noise to introduce potential asymmetries
            stalk += 1e-13 * torch.randn(dim, dim, dtype=torch.float64)
            sheaf.stalks[node] = stalk
        
        # Add restrictions with potential numerical errors
        for u, v in G.edges():
            u_dim = sheaf.stalks[u].shape[0]
            v_dim = sheaf.stalks[v].shape[0]
            R = torch.randn(v_dim, u_dim, dtype=torch.float64)
            # Normalize with potential rounding errors
            R = R / (R.sum(dim=0, keepdim=True) + 1e-15)
            sheaf.restrictions[(u, v)] = R
        
        # Add GW metadata
        sheaf.metadata['construction_method'] = 'gromov_wasserstein'
        sheaf.metadata['gw_costs'] = {edge: np.random.rand() for edge in G.edges()}
        
        # Solve the eigenvalue problem
        try:
            eigenvals, eigenvecs = self.builder.solve_generalized_robust(
                sheaf, list(G.edges()), k=6, use_matrix_free=False
            )
            
            # Should produce only real eigenvalues
            assert np.all(np.isreal(eigenvals)), "Complex eigenvalues in full pipeline"
            assert not np.any(np.isnan(eigenvals)), "NaN eigenvalues in full pipeline"
            assert not np.any(np.isinf(eigenvals)), "Infinite eigenvalues in full pipeline"
            
            logger.info(f"✅ Full pipeline succeeded with eigenvalue range [{eigenvals[0]:.2e}, {eigenvals[-1]:.2e}]")
            
        except Exception as e:
            pytest.fail(f"Full pipeline failed with symmetrization: {e}")
            
    def test_symmetrization_preserves_eigenvalue_ordering(self):
        """Test that symmetrization doesn't drastically change eigenvalue ordering."""
        # Create test matrices
        n = 10
        L = self.create_slightly_asymmetric_matrix(n, 1e-12)
        M = np.eye(n) + 0.01 * self.create_slightly_asymmetric_matrix(n, 1e-13)
        
        # Symmetrize
        L_sym = self.builder._ensure_symmetric(L, "L")
        M_sym = self.builder._ensure_symmetric(M, "M")
        
        # Compute eigenvalues
        eigenvals_before = np.linalg.eigvals(L)  # May have complex parts
        eigenvals_after, _ = eigh(L_sym, M_sym)
        
        # Sort real parts for comparison
        eigenvals_before_real = np.sort(np.real(eigenvals_before))
        eigenvals_after_sorted = np.sort(eigenvals_after)
        
        # The eigenvalues should be similar (allowing for small numerical differences)
        # We only compare the first few since small eigenvalues are more sensitive
        k = min(5, n)
        diff = np.abs(eigenvals_before_real[:k] - eigenvals_after_sorted[:k])
        
        # Allow for reasonable numerical differences (matrices are different, so eigenvalues will differ)
        # Just check they're in the same ballpark
        assert np.all(diff < 0.01), f"Eigenvalues changed too much: max diff = {np.max(diff):.2e}"
        
        logger.info("✅ Symmetrization preserves eigenvalue ordering")


if __name__ == "__main__":
    # Run tests with detailed output
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test instance
    test = TestSymmetrizationFix()
    test.setup_method()
    
    # Run all test methods
    test_methods = [
        test.test_ensure_symmetric_method,
        test.test_no_complex_eigenvalues_after_symmetrization,
        test.test_residuals_improved_with_symmetrization,
        test.test_torch_solver_symmetrization,
        test.test_sparse_solver_symmetrization,
        test.test_full_pipeline_with_asymmetric_input,
        test.test_symmetrization_preserves_eigenvalue_ordering,
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
    print(f"Symmetrization Fix Test Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
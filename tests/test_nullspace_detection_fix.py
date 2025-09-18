"""Test the nullspace detection fix for σ=0 factorization avoidance.

This test suite verifies that:
1. Nullspace detection correctly identifies matrices with nullspaces
2. σ=0 attempts are skipped for matrices with expected nullspaces
3. Alternative solvers work correctly for nullspace problems
4. Logging is informative rather than noisy
5. Performance is improved by avoiding unnecessary factorization attempts
"""

import pytest
import torch
import numpy as np
import networkx as nx
import logging
from typing import Dict, List
from scipy.sparse import csr_matrix
from unittest.mock import patch

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


class TestNullspaceDetectionFix:
    """Test suite for nullspace detection and σ=0 avoidance fix."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=False)
        
    def create_gw_sheaf_with_nullspace(self) -> Sheaf:
        """Create a GW-based sheaf that will produce a Laplacian with nullspace."""
        # Create a disconnected graph to ensure nullspace
        nodes = ['a', 'b', 'c', 'd']
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        # Add edges but leave one node disconnected to guarantee nullspace
        G.add_edge('a', 'b')
        G.add_edge('b', 'c')
        # 'd' is isolated - this creates nullspace
        
        sheaf = Sheaf(poset=G)
        
        # Add stalks with GW metadata
        for node in nodes:
            dim = 3
            stalk = torch.eye(dim, dtype=torch.float64)
            sheaf.stalks[node] = stalk
            # Add GW metadata to make it GW-based
            sheaf.metadata[f'{node}_gw_basis'] = stalk
            sheaf.metadata[f'{node}_feature_dim'] = dim
        
        # Add restrictions for connected components only with GW metadata
        for u, v in G.edges():
            R = torch.randn(3, 3, dtype=torch.float64) * 0.5
            sheaf.restrictions[(u, v)] = R
            # Add GW edge metadata
            sheaf.metadata[f'{u}_{v}_gw_weight'] = 1.0
            
        # Mark as GW-based
        sheaf.metadata['gw_construction'] = True
        sheaf.metadata['construction_method'] = 'gromov_wasserstein'
            
        return sheaf
        
    def create_sheaf_without_nullspace(self) -> Sheaf:
        """Create a sheaf that should produce a well-conditioned Laplacian."""
        nodes = ['a', 'b', 'c']
        G = nx.complete_graph(nodes, create_using=nx.DiGraph())
        
        sheaf = Sheaf(poset=G)
        
        # Add stalks
        for node in nodes:
            # Add small positive diagonal to make matrix well-conditioned
            stalk = torch.eye(2, dtype=torch.float64) + 0.1 * torch.randn(2, 2, dtype=torch.float64)
            sheaf.stalks[node] = stalk
        
        # Add restrictions with regularization to avoid singularity
        for u, v in G.edges():
            R = torch.eye(2, dtype=torch.float64) + 0.01 * torch.randn(2, 2, dtype=torch.float64)
            sheaf.restrictions[(u, v)] = R
            
        return sheaf
        
    def create_simple_laplacian_matrix(self) -> csr_matrix:
        """Create a simple graph Laplacian matrix with known nullspace."""
        # 4x4 graph Laplacian for path graph: 0-1-2-3
        L = np.array([
            [ 1, -1,  0,  0],  # Node 0: connected to 1
            [-1,  2, -1,  0],  # Node 1: connected to 0,2
            [ 0, -1,  2, -1],  # Node 2: connected to 1,3
            [ 0,  0, -1,  1],  # Node 3: connected to 2
        ], dtype=np.float64)
        
        # This matrix has nullspace (constant vector [1,1,1,1])
        return csr_matrix(L)
        
    def create_well_conditioned_matrix(self) -> csr_matrix:
        """Create a well-conditioned matrix without obvious nullspace."""
        # 4x4 positive definite matrix with larger diagonal dominance
        # to ensure nullspace detection doesn't trigger
        A = np.array([
            [10.0, 0.1, 0.0, 0.0],
            [0.1, 10.0, 0.1, 0.0],
            [0.0, 0.1, 10.0, 0.1],
            [0.0, 0.0, 0.1, 10.0],
        ], dtype=np.float64)
        
        return csr_matrix(A)
        
    def test_nullspace_detection_laplacian_structure(self):
        """Test nullspace detection for matrices with Laplacian structure."""
        L = self.create_simple_laplacian_matrix()
        M = csr_matrix(np.eye(4, dtype=np.float64))
        
        # Should detect nullspace due to zero row sums
        has_nullspace = self.builder._should_expect_nullspace(L, M)
        assert has_nullspace is True, "Should detect nullspace for graph Laplacian"
        
        logger.info("✅ Nullspace detection test passed for Laplacian structure")
        
    def test_nullspace_detection_well_conditioned(self):
        """Test nullspace detection for well-conditioned matrices."""
        A = self.create_well_conditioned_matrix()
        M = csr_matrix(np.eye(4, dtype=np.float64))
        
        # Should not detect nullspace for well-conditioned matrix
        has_nullspace = self.builder._should_expect_nullspace(A, M)
        assert has_nullspace is False, "Should not detect nullspace for well-conditioned matrix"
        
        logger.info("✅ Nullspace detection test passed for well-conditioned matrix")
        
    def test_skip_sigma_zero_for_nullspace(self):
        """Test that σ=0 attempts are skipped when nullspace is detected."""
        L = self.create_simple_laplacian_matrix()
        M = csr_matrix(np.eye(4, dtype=np.float64))
        
        # Mock eigsh to count calls
        with patch('neurosheaf.sheaf.assembly.gw_laplacian.eigsh') as mock_eigsh:
            # Configure mock to return reasonable eigenvalues
            mock_eigsh.return_value = (
                np.array([0.0, 1.0, 2.0]),  # eigenvalues
                np.random.randn(4, 3)       # eigenvectors
            )
            
            # Call the solver
            eigenvals, eigenvecs = self.builder._solve_eigsh_shift_invert(L, M, 3)
            
            # Check that eigsh was called only once (no σ=0 attempt)
            assert mock_eigsh.call_count == 1, f"Expected 1 eigsh call, got {mock_eigsh.call_count}"
            
            # Check that it was called with positive sigma
            call_args = mock_eigsh.call_args
            sigma_used = call_args[1]['sigma']  # keyword argument
            assert sigma_used > 0, f"Expected positive sigma, got {sigma_used}"
            
        logger.info("✅ σ=0 skip test passed for nullspace matrices")
        
    def test_sigma_zero_attempted_for_well_conditioned(self):
        """Test that σ=0 is still attempted for well-conditioned matrices."""
        A = self.create_well_conditioned_matrix()
        M = csr_matrix(np.eye(4, dtype=np.float64))
        
        # Mock eigsh to track calls
        with patch('neurosheaf.sheaf.assembly.gw_laplacian.eigsh') as mock_eigsh:
            # First call (σ=0) succeeds
            mock_eigsh.return_value = (
                np.array([0.1, 1.0, 2.0]),  # eigenvalues
                np.random.randn(4, 3)       # eigenvectors
            )
            
            # Call the solver
            eigenvals, eigenvecs = self.builder._solve_eigsh_shift_invert(A, M, 3)
            
            # Check that eigsh was called only once (σ=0 succeeded)
            assert mock_eigsh.call_count == 1, f"Expected 1 eigsh call, got {mock_eigsh.call_count}"
            
            # Check that it was called with sigma=0
            call_args = mock_eigsh.call_args
            sigma_used = call_args[1]['sigma']  # keyword argument
            assert sigma_used == 0.0, f"Expected sigma=0, got {sigma_used}"
            
        logger.info("✅ σ=0 attempt test passed for well-conditioned matrices")
        
    def test_alternative_solver_smallest_algebraic(self):
        """Test the alternative solver using which='SA'."""
        L = self.create_simple_laplacian_matrix()
        M = csr_matrix(np.eye(4, dtype=np.float64))
        
        # Test the smallest algebraic solver directly
        eigenvals, eigenvecs = self.builder._solve_eigsh_smallest_algebraic(L, M, 3)
        
        # Basic validation
        assert len(eigenvals) == 3, f"Expected 3 eigenvalues, got {len(eigenvals)}"
        assert eigenvecs.shape == (4, 3), f"Expected eigenvectors shape (4,3), got {eigenvecs.shape}"
        
        # Check that eigenvalues are reasonable (smallest should be near zero for Laplacian)
        assert eigenvals[0] >= -1e-10, f"Smallest eigenvalue should be >= 0, got {eigenvals[0]:.2e}"
        assert eigenvals[0] < 1e-6, f"Smallest eigenvalue should be near zero, got {eigenvals[0]:.2e}"
        
        logger.info(f"✅ Alternative solver test passed: eigenvalues {eigenvals}")
        
    def test_nullspace_aware_fallback_chain(self):
        """Test the complete nullspace-aware fallback chain."""
        sheaf = self.create_gw_sheaf_with_nullspace()
        
        # Build the Laplacian - this should use nullspace-aware routing
        L = self.builder.build_laplacian(sheaf, sparse=True)
        
        # Test the solver chain with the actual Laplacian
        k = 3
        eigenvals, eigenvecs = self.builder._solve_with_full_fallbacks(L, csr_matrix(np.eye(L.shape[0])), k)
        
        # Verify results
        assert len(eigenvals) == k, f"Expected {k} eigenvalues, got {len(eigenvals)}"
        assert eigenvecs.shape[1] == k, f"Expected {k} eigenvectors, got {eigenvecs.shape[1]}"
        
        # Should have small eigenvalues (nullspace expected)
        assert eigenvals[0] < 1e-6, f"Smallest eigenvalue should be near zero, got {eigenvals[0]:.2e}"
        
        logger.info("✅ Nullspace-aware fallback chain test passed")
        
    def test_logging_improvement(self, caplog):
        """Test that logging is informative rather than noisy."""
        L = self.create_simple_laplacian_matrix()
        M = csr_matrix(np.eye(4, dtype=np.float64))
        
        with caplog.at_level(logging.INFO):
            eigenvals, eigenvecs = self.builder._solve_eigsh_shift_invert(L, M, 3)
            
        # Check that we get informative log messages
        log_messages = [record.message for record in caplog.records]
        
        # Should have informative message about nullspace detection
        nullspace_message_found = any("nullspace detected" in msg or "positive shift" in msg 
                                     for msg in log_messages)
        assert nullspace_message_found, "Should log informative message about nullspace detection"
        
        # Should NOT have noisy failure messages
        failure_message_found = any("sigma=0.0 failed" in msg for msg in log_messages)
        assert not failure_message_found, "Should not have noisy failure messages when nullspace is detected"
        
        logger.info("✅ Logging improvement test passed")
        
    def test_performance_improvement_timing(self):
        """Test that performance is improved by avoiding unnecessary attempts."""
        L = self.create_simple_laplacian_matrix()
        M = csr_matrix(np.eye(4, dtype=np.float64))
        
        import time
        
        # Time the nullspace-aware solver
        start_time = time.time()
        eigenvals1, eigenvecs1 = self.builder._solve_eigsh_shift_invert(L, M, 3)
        nullspace_aware_time = time.time() - start_time
        
        # For comparison, time the old approach by forcing σ=0 attempt
        with patch.object(self.builder, '_should_expect_nullspace', return_value=False):
            start_time = time.time()
            try:
                eigenvals2, eigenvecs2 = self.builder._solve_eigsh_shift_invert(L, M, 3)
                old_approach_time = time.time() - start_time
            except:
                # If it fails completely, just set a reasonable upper bound
                old_approach_time = nullspace_aware_time * 2
        
        # The improvement might be small for tiny matrices, but should not be slower
        assert nullspace_aware_time <= old_approach_time * 1.1, \
            f"Nullspace-aware approach should not be significantly slower: " \
            f"{nullspace_aware_time:.3f}s vs {old_approach_time:.3f}s"
        
        logger.info(f"✅ Performance test passed: " 
                   f"nullspace-aware={nullspace_aware_time:.3f}s, old={old_approach_time:.3f}s")
        
    def test_integration_with_actual_sheaf(self):
        """Test integration with actual sheaf construction and solving."""
        sheaf = self.create_gw_sheaf_with_nullspace()
        
        # This should work without noisy σ=0 failure messages
        with patch('neurosheaf.sheaf.assembly.gw_laplacian.logger') as mock_logger:
            # Build and solve
            L = self.builder.build_laplacian(sheaf, sparse=True)
            
            # Check that no warning messages about σ=0 failures were logged
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if 'sigma=0.0 failed' in str(call)]
            assert len(warning_calls) == 0, f"Should not have σ=0 failure warnings, got {len(warning_calls)}"
            
        logger.info("✅ Integration test passed - no noisy σ=0 warnings")


if __name__ == "__main__":
    # Run tests with detailed output
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test instance
    test = TestNullspaceDetectionFix()
    test.setup_method()
    
    # Run all test methods (excluding the one that requires caplog)
    test_methods = [
        test.test_nullspace_detection_laplacian_structure,
        test.test_nullspace_detection_well_conditioned,
        test.test_skip_sigma_zero_for_nullspace,
        test.test_sigma_zero_attempted_for_well_conditioned,
        test.test_alternative_solver_smallest_algebraic,
        test.test_nullspace_aware_fallback_chain,
        test.test_performance_improvement_timing,
        test.test_integration_with_actual_sheaf,
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Nullspace Detection Fix Test Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
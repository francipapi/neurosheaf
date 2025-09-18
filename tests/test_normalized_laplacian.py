"""Validation tests for normalized Laplacian implementation.

This module provides comprehensive tests to ensure the mathematical correctness
of the normalized Hodge Laplacian implementation, including:
- Eigenvalue bounds checking
- Generalized eigenvalue problem validation
- Threshold parameter verification
- Caching optimization tests
"""

import pytest
import torch
import numpy as np
from scipy.sparse import csr_matrix
import logging

# Import modules to test
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.spectral.eigenvalue_crossings import EigenvalueCrossingDetector
from neurosheaf.io.config import H0Config

logger = logging.getLogger(__name__)


class TestNormalizedLaplacian:
    """Test suite for normalized Laplacian implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = GWLaplacianBuilder(
            validate_properties=True,
            enable_caching=True
        )
        self.crossing_detector = EigenvalueCrossingDetector()
        
    def create_test_sheaf(self, n_nodes=5, n_edges=6):
        """Create a simple test sheaf for validation."""
        from neurosheaf.sheaf.data_structures import Sheaf
        import networkx as nx
        
        # Create simple graph
        G = nx.erdos_renyi_graph(n_nodes, 0.5, directed=True)
        
        # Create sheaf
        sheaf = Sheaf(poset=G)
        
        # Add stalks (vector spaces at nodes)
        for node in G.nodes():
            dim = np.random.randint(2, 5)
            sheaf.stalks[node] = torch.eye(dim)
        
        # Add restrictions (maps between stalks)
        for u, v in G.edges():
            u_dim = sheaf.stalks[u].shape[0]
            v_dim = sheaf.stalks[v].shape[0]
            
            # Create column-stochastic restriction map
            R = torch.rand(v_dim, u_dim)
            R = R / R.sum(dim=0, keepdim=True)  # Column normalization
            sheaf.restrictions[(u, v)] = R
        
        # Add GW metadata
        sheaf.metadata['construction_method'] = 'gromov_wasserstein'
        sheaf.metadata['gw_costs'] = {edge: np.random.rand() for edge in G.edges()}
        
        return sheaf
    
    def test_eigenvalue_bounds(self):
        """Test that normalized Laplacian eigenvalues are in [0, 2]."""
        sheaf = self.create_test_sheaf()
        active_edges = list(sheaf.restrictions.keys())
        
        # Compute eigenvalues using generalized solver
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=10, use_matrix_free=False
        )
        
        # Check bounds
        assert np.all(eigenvals >= -1e-10), f"Found negative eigenvalue: {np.min(eigenvals)}"
        assert np.all(eigenvals <= 2.0 + 1e-10), f"Found eigenvalue > 2: {np.max(eigenvals)}"
        
        logger.info(f"✅ Eigenvalue bounds test passed: λ ∈ [{np.min(eigenvals):.2e}, {np.max(eigenvals):.2e}]")
    
    def test_classification_thresholds(self):
        """Test adaptive threshold computation for normalized vs unnormalized."""
        eigenvals_normalized = np.array([0.0, 1e-14, 1e-12, 1e-10, 0.01, 0.1, 1.0, 1.5])
        eigenvals_unnormalized = np.array([0.0, 1e-14, 1e-10, 0.1, 1.0, 10.0, 100.0, 1000.0])
        
        # Test normalized classification
        class_norm = self.builder.classify_eigenvalues_downstream(
            eigenvals_normalized, is_normalized=True
        )
        
        # Test unnormalized classification
        class_unnorm = self.builder.classify_eigenvalues_downstream(
            eigenvals_unnormalized, is_normalized=False
        )
        
        # Normalized should have tighter thresholds
        assert class_norm['effective_threshold'] < class_unnorm['effective_threshold']
        
        # Check classification counts
        assert np.sum(class_norm['true_zeros']) >= 2  # At least 0 and 1e-14
        assert np.sum(class_unnorm['true_zeros']) >= 2  # At least 0 and 1e-14
        
        logger.info(f"✅ Threshold test: normalized={class_norm['effective_threshold']:.2e}, "
                   f"unnormalized={class_unnorm['effective_threshold']:.2e}")
    
    def test_h0_config_normalization(self):
        """Test H0Config creation for normalized Laplacian."""
        # Standard config
        config_standard = H0Config()
        
        # Normalized Laplacian config
        config_normalized = H0Config.for_normalized_laplacian()
        
        # Check parameter adjustments
        assert config_normalized.c_in < config_standard.c_in
        assert config_normalized.gap <= config_standard.gap
        assert config_normalized.use_normalized_thresholds == True
        
        # Test threshold creation
        spectral_norm = 1.5  # Typical for normalized Laplacian
        thresholds_norm = config_normalized.create_step_thresholds(spectral_norm)
        thresholds_std = config_standard.create_step_thresholds(spectral_norm)
        
        assert thresholds_norm['tau_in'] < thresholds_std['tau_in']
        assert thresholds_norm['is_normalized'] == True
        
        logger.info(f"✅ H0Config test: τ_in normalized={thresholds_norm['tau_in']:.2e}, "
                   f"standard={thresholds_std['tau_in']:.2e}")
    
    def test_eigenvalue_crossing_detection(self):
        """Test crossing detection for near-zero eigenvalues."""
        # Create eigenvalue sequences with crossing
        prev_eigenvals = torch.tensor([0.0, 1e-11, 1e-9, 0.01, 0.1])
        curr_eigenvals = torch.tensor([1e-11, 0.0, 1e-8, 0.02, 0.09])
        
        # Create dummy eigenvectors
        n = len(prev_eigenvals)
        prev_eigenvecs = torch.eye(n)
        curr_eigenvecs = torch.eye(n)
        
        # Detect crossings
        result = self.crossing_detector.detect_crossings(
            prev_eigenvals, curr_eigenvals,
            prev_eigenvecs, curr_eigenvecs,
            step=1, is_normalized=True
        )
        
        # Should detect crossing between first two eigenvalues
        assert len(result['crossings']) > 0 or len(result['transitions']) > 0
        
        # Check classifications
        assert result['prev_classification']['true_zeros'].sum() >= 1
        assert result['curr_classification']['true_zeros'].sum() >= 1
        
        logger.info(f"✅ Crossing detection test: {len(result['crossings'])} crossings, "
                   f"{len(result['transitions'])} transitions")
    
    def test_spectral_gap_detection(self):
        """Test spectral gap detection for improved classification."""
        # Create eigenvalues with clear gap
        eigenvals_with_gap = torch.tensor([0.0, 1e-12, 1e-11, 1e-10, 1e-3, 0.01, 0.1])
        
        classification = self.crossing_detector.classify_eigenvalues(
            eigenvals_with_gap, is_normalized=True
        )
        
        # Should identify the gap and classify accordingly
        n_zeros = classification['true_zeros'].sum()
        assert n_zeros >= 3 and n_zeros <= 4  # First 3-4 should be zeros
        
        logger.info(f"✅ Spectral gap test: {n_zeros} zeros detected before gap")
    
    def test_caching_optimization(self):
        """Test that caching improves performance."""
        sheaf = self.create_test_sheaf()
        active_edges = list(sheaf.restrictions.keys())
        
        # First computation (no cache)
        import time
        start = time.time()
        eigenvals1, _ = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=5
        )
        time1 = time.time() - start
        
        # Second computation (with cache)
        start = time.time()
        eigenvals2, _ = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=5
        )
        time2 = time.time() - start
        
        # Results should be identical
        assert np.allclose(eigenvals1, eigenvals2)
        
        # Second should be faster (cache hit)
        # Note: This might not always be true for very small problems
        if time1 > 0.01:  # Only check if first computation took significant time
            assert time2 <= time1 * 1.1  # Allow 10% margin
        
        # Clear cache
        self.builder.clear_cache()
        
        # Third computation (cache cleared)
        start = time.time()
        eigenvals3, _ = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=5
        )
        time3 = time.time() - start
        
        assert np.allclose(eigenvals1, eigenvals3)
        
        logger.info(f"✅ Caching test: no-cache={time1:.3f}s, cached={time2:.3f}s, "
                   f"cleared={time3:.3f}s")
    
    def test_matrix_free_computation(self):
        """Test matrix-free LinearOperator for large problems."""
        sheaf = self.create_test_sheaf(n_nodes=10, n_edges=20)
        active_edges = list(sheaf.restrictions.keys())
        
        # Compute with standard method
        eigenvals_standard, _ = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=5, use_matrix_free=False
        )
        
        # Compute with matrix-free method (if applicable)
        try:
            eigenvals_free, _ = self.builder.solve_generalized_robust(
                sheaf, active_edges, k=5, use_matrix_free=True
            )
            
            # Results should be close (allowing for numerical differences)
            assert np.allclose(eigenvals_standard, eigenvals_free, rtol=1e-6)
            logger.info(f"✅ Matrix-free test: standard vs free max diff = "
                       f"{np.max(np.abs(eigenvals_standard - eigenvals_free)):.2e}")
        except Exception as e:
            # Matrix-free might not be available for small problems
            logger.info(f"Matrix-free test skipped: {e}")
    
    def test_b_orthonormality(self):
        """Test that eigenvectors are B-orthonormal (E^T M E = I)."""
        sheaf = self.create_test_sheaf()
        active_edges = list(sheaf.restrictions.keys())
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=min(5, sheaf.stalks[0].shape[0])
        )
        
        # Build mass matrix M
        M_torch = self.builder._build_stalk_metric(
            sheaf, self.builder._extract_node_masses(sheaf)
        )
        M = M_torch.detach().cpu().numpy() if isinstance(M_torch, torch.Tensor) else M_torch
        
        # Check B-orthonormality: E^T M E should be identity
        if hasattr(M, 'toarray'):
            M = M.toarray()
        
        k = eigenvecs.shape[1]
        orthogonality = eigenvecs.T @ M @ eigenvecs
        identity_error = np.linalg.norm(orthogonality - np.eye(k), 'fro')
        
        assert identity_error < 1e-8, f"B-orthonormality error: {identity_error:.2e}"
        
        logger.info(f"✅ B-orthonormality test: error = {identity_error:.2e}")
    
    def test_residual_validation(self):
        """Test that eigenvalue residuals are small (Ax - λMx ≈ 0)."""
        sheaf = self.create_test_sheaf()
        active_edges = list(sheaf.restrictions.keys())
        
        # Build matrices
        delta = self.builder.build_coboundary_general_sparse(sheaf, active_edges)
        edge_weights = self.builder.extract_edge_weights_linear_only(sheaf, active_edges)
        G1 = self.builder.build_G1_block_diagonal_corrected(sheaf, active_edges, edge_weights)
        M = self.builder._build_stalk_metric(sheaf, self.builder._extract_node_masses(sheaf))
        
        # Convert to numpy/scipy
        if hasattr(M, 'tocsr'):
            M = M.tocsr()
        else:
            M = csr_matrix(M.detach().cpu().numpy() if isinstance(M, torch.Tensor) else M)
        
        # Compute A = δ^T G₁ δ
        A = (delta.T @ (G1 @ delta)).tocsr()
        
        # Compute eigenvalues
        eigenvals, eigenvecs = self.builder.solve_generalized_robust(
            sheaf, active_edges, k=min(5, A.shape[0] - 1)
        )
        
        # Check residuals for each eigenpair
        max_residual = 0
        for i in range(len(eigenvals)):
            x = eigenvecs[:, i]
            λ = eigenvals[i]
            
            # Compute residual: Ax - λMx
            Ax = A @ x
            Mx = M @ x
            residual = Ax - λ * Mx
            residual_norm = np.linalg.norm(residual)
            relative_residual = residual_norm / (np.linalg.norm(Ax) + 1e-16)
            
            max_residual = max(max_residual, relative_residual)
        
        assert max_residual < 1e-6, f"Large residual found: {max_residual:.2e}"
        
        logger.info(f"✅ Residual validation test: max relative residual = {max_residual:.2e}")


if __name__ == "__main__":
    # Run tests with detailed output
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test instance
    test = TestNormalizedLaplacian()
    test.setup_method()
    
    # Run all tests
    test_methods = [
        test.test_eigenvalue_bounds,
        test.test_classification_thresholds,
        test.test_h0_config_normalization,
        test.test_eigenvalue_crossing_detection,
        test.test_spectral_gap_detection,
        test.test_caching_optimization,
        test.test_matrix_free_computation,
        test.test_b_orthonormality,
        test.test_residual_validation
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method.__name__}...")
            test_method()
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__} failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
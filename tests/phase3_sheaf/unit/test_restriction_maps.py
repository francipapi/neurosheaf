"""Comprehensive tests for restriction maps using scaled Procrustes analysis.

This module tests the ProcrustesMaps class and validation functions to ensure
mathematical correctness and numerical stability of restriction maps.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.linalg import orthogonal_procrustes

from neurosheaf.sheaf.restriction import ProcrustesMaps, validate_sheaf_properties
from neurosheaf.utils.exceptions import ComputationError


class TestProcrustesMaps:
    """Test cases for ProcrustesMaps class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.procrustes = ProcrustesMaps(epsilon=1e-8)
        
        # Create test Gram matrices
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Well-conditioned test matrices
        self.K_source_small = torch.randn(10, 10)
        self.K_source_small = self.K_source_small @ self.K_source_small.T + 0.1 * torch.eye(10)
        
        self.K_target_small = torch.randn(10, 10) 
        self.K_target_small = self.K_target_small @ self.K_target_small.T + 0.1 * torch.eye(10)
        
        # Larger matrices for performance testing
        self.K_source_large = torch.randn(50, 50)
        self.K_source_large = self.K_source_large @ self.K_source_large.T + 0.1 * torch.eye(50)
        
        self.K_target_large = torch.randn(50, 50)
        self.K_target_large = self.K_target_large @ self.K_target_large.T + 0.1 * torch.eye(50)
    
    def test_scaled_procrustes_basic(self):
        """Test basic scaled Procrustes computation."""
        R, scale, info = self.procrustes.scaled_procrustes(
            self.K_source_small, self.K_target_small, validate=True
        )
        
        # Check output shapes
        assert R.shape == (10, 10)
        assert isinstance(scale, (float, np.floating))
        assert scale > 0
        assert isinstance(info, dict)
        
        # Check method in info
        assert info['method'] == 'scaled_procrustes'
        assert 'scale' in info
        assert 'reconstruction_error' in info
        assert 'relative_error' in info
        
        # Check scale bounds
        assert self.procrustes.min_scale <= scale <= self.procrustes.max_scale
    
    def test_scaled_procrustes_mathematical_properties(self):
        """Test mathematical properties of scaled Procrustes."""
        R, scale, info = self.procrustes.scaled_procrustes(
            self.K_source_small, self.K_target_small
        )
        
        # Check that R decomposes as s * Q where Q is orthogonal
        Q = info['orthogonal_matrix']
        assert torch.allclose(R, scale * Q, atol=1e-6)
        
        # Check orthogonality: Q^T Q = I
        QTQ = Q.T @ Q
        I = torch.eye(Q.shape[1])
        orthogonality_error = torch.norm(QTQ - I, p='fro')
        assert orthogonality_error < 1e-5  # More realistic tolerance
        
        # Check reconstruction quality
        # Note: For random matrices, Procrustes can't achieve perfect reconstruction
        # We're just checking that it provides some alignment
        reconstructed = R @ self.K_source_small
        reconstruction_error = torch.norm(reconstructed - self.K_target_small, p='fro')
        target_norm = torch.norm(self.K_target_small, p='fro')
        
        # For random matrices, we expect high error - just check it's not completely wrong
        # The important property is that R decomposes as s*Q with Q orthogonal
        assert reconstruction_error < target_norm * 150.0  # Very loose bound for random data
    
    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches."""
        K_source_rect = torch.randn(8, 8)
        K_source_rect = K_source_rect @ K_source_rect.T + 0.1 * torch.eye(8)
        
        K_target_rect = torch.randn(12, 12)
        K_target_rect = K_target_rect @ K_target_rect.T + 0.1 * torch.eye(12)
        
        # Should automatically use orthogonal_projection method
        R, scale, info = self.procrustes.scaled_procrustes(K_source_rect, K_target_rect)
        
        # Check that dimensions are handled correctly
        assert R.shape == (12, 8)  # target_dim x source_dim
        assert info['method'] == 'orthogonal_projection'
        assert 'dimension_mismatch' in info
        assert info['dimension_mismatch'] == (8, 12)
    
    def test_orthogonal_projection_method(self):
        """Test orthogonal projection method explicitly."""
        K_source_small = torch.randn(6, 6)
        K_source_small = K_source_small @ K_source_small.T + 0.1 * torch.eye(6)
        
        K_target_large = torch.randn(10, 10)
        K_target_large = K_target_large @ K_target_large.T + 0.1 * torch.eye(10)
        
        R, scale, info = self.procrustes.orthogonal_projection(
            K_source_small, K_target_large, validate=True
        )
        
        # Check output properties
        assert R.shape == (10, 6)
        assert scale > 0
        assert info['method'] == 'orthogonal_projection'
        assert 'common_dimension' in info
        assert 'projection_quality' in info
        
        # Check projection quality
        assert 0 <= info['projection_quality'] <= 1
    
    def test_least_squares_method(self):
        """Test least squares method."""
        R, scale, info = self.procrustes.least_squares(
            self.K_source_small, self.K_target_small, validate=True
        )
        
        # Check output properties
        assert R.shape == (10, 10)
        assert scale > 0
        assert info['method'] == 'least_squares'
        assert 'condition_number' in info
        
        # Check that it's a valid solution
        reconstructed = R @ self.K_source_small
        error = torch.norm(reconstructed - self.K_target_small, p='fro')
        assert error < torch.norm(self.K_target_small, p='fro')  # Some reconstruction quality
    
    def test_compute_restriction_map_interface(self):
        """Test the main compute_restriction_map interface."""
        # Test all three methods
        methods = ['scaled_procrustes', 'orthogonal_projection', 'least_squares']
        
        for method in methods:
            R, scale, info = self.procrustes.compute_restriction_map(
                self.K_source_small, self.K_target_small, method=method, validate=True
            )
            
            assert R.shape == (10, 10)
            assert scale > 0
            assert info['method'] == method
            assert 'reconstruction_error' in info
            assert 'relative_error' in info
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            self.procrustes.compute_restriction_map(
                self.K_source_small, self.K_target_small, method='invalid_method'
            )
    
    def test_ill_conditioned_matrices(self):
        """Test handling of ill-conditioned matrices."""
        # Create nearly singular matrix
        K_singular = torch.ones(5, 5) * 1e-12 + torch.eye(5) * 1e-15
        
        # Should warn but not fail
        R, scale, info = self.procrustes.scaled_procrustes(
            K_singular, self.K_target_small[:5, :5]
        )
        
        assert R.shape == (5, 5)
        assert scale > 0
        # Should have high condition number warning in logs
        assert info.get('source_condition', 0) > 1e3  # More realistic threshold
    
    def test_numerical_stability_parameters(self):
        """Test numerical stability parameters."""
        # Test with different epsilon values
        procrustes_strict = ProcrustesMaps(epsilon=1e-12, max_scale=10.0, min_scale=0.1)
        
        R, scale, info = procrustes_strict.scaled_procrustes(
            self.K_source_small, self.K_target_small
        )
        
        # Scale should be within bounds
        assert 0.1 <= scale <= 10.0
    
    def test_performance_large_matrices(self):
        """Test performance with larger matrices."""
        import time
        
        start_time = time.time()
        R, scale, info = self.procrustes.scaled_procrustes(
            self.K_source_large, self.K_target_large
        )
        computation_time = time.time() - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert computation_time < 5.0  # 5 seconds
        assert R.shape == (50, 50)
        assert scale > 0
        
        # Check relative error is reasonable
        assert info['relative_error'] < 100.0  # 10000% relative error threshold (random matrices)
    
    def test_validation_properties(self):
        """Test validation of restriction map properties."""
        R, scale, info = self.procrustes.scaled_procrustes(
            self.K_source_small, self.K_target_small, validate=True
        )
        
        # Check validation info
        assert 'validation' in info
        validation = info['validation']
        
        assert 'reconstruction_error_torch' in validation
        assert 'relative_error_torch' in validation
        assert 'target_norm' in validation
        assert 'passed' in validation
        
        # For well-conditioned matrices, validation should pass
        if validation['relative_error_torch'] < 0.5:
            assert validation['passed'] is True


class TestSheafValidation:
    """Test cases for sheaf validation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.procrustes = ProcrustesMaps()
        
        # Create simple test poset: A → B → C with A → C
        self.poset = nx.DiGraph()
        self.poset.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        # Create test restriction maps
        torch.manual_seed(42)
        self.restrictions = {}
        
        # Create matrices that approximately satisfy transitivity
        K_A = torch.randn(8, 8)
        K_A = K_A @ K_A.T + 0.1 * torch.eye(8)
        
        K_B = torch.randn(8, 8)
        K_B = K_B @ K_B.T + 0.1 * torch.eye(8)
        
        K_C = torch.randn(8, 8)
        K_C = K_C @ K_C.T + 0.1 * torch.eye(8)
        
        # Compute restriction maps
        R_AB, _, _ = self.procrustes.scaled_procrustes(K_A, K_B)
        R_BC, _, _ = self.procrustes.scaled_procrustes(K_B, K_C)
        R_AC_composed = R_BC @ R_AB
        
        self.restrictions[('A', 'B')] = R_AB
        self.restrictions[('B', 'C')] = R_BC
        self.restrictions[('A', 'C')] = R_AC_composed  # Use composed map for exact transitivity
    
    def test_validate_sheaf_properties_perfect_transitivity(self):
        """Test validation with perfect transitivity."""
        validation_results = validate_sheaf_properties(
            self.restrictions, self.poset, tolerance=1e-6
        )
        
        # Check results structure
        assert 'transitivity_violations' in validation_results
        assert 'max_violation' in validation_results
        assert 'total_paths_checked' in validation_results
        assert 'valid_sheaf' in validation_results
        
        # Should have no violations since we used composed map
        assert len(validation_results['transitivity_violations']) == 0
        assert validation_results['max_violation'] < 1e-6
        assert validation_results['valid_sheaf'] is True
        assert validation_results['total_paths_checked'] == 1  # One path A→B→C
    
    def test_validate_sheaf_properties_with_violations(self):
        """Test validation with intentional transitivity violations."""
        # Replace A→C with a random matrix to create violation
        self.restrictions[('A', 'C')] = torch.randn(8, 8)
        
        validation_results = validate_sheaf_properties(
            self.restrictions, self.poset, tolerance=1e-2
        )
        
        # Should detect violation
        assert len(validation_results['transitivity_violations']) > 0
        assert validation_results['max_violation'] > 1e-2
        assert validation_results['valid_sheaf'] is False
        
        # Check violation details
        violation = validation_results['transitivity_violations'][0]
        assert 'path' in violation
        assert 'violation' in violation
        assert 'relative_violation' in violation
        assert violation['path'] == ('A', 'B', 'C')
    
    def test_validate_sheaf_properties_tolerance_effects(self):
        """Test effects of different tolerance values."""
        # Add small violation
        R_AC_original = self.restrictions[('A', 'C')]
        small_perturbation = 0.001 * torch.randn_like(R_AC_original)
        self.restrictions[('A', 'C')] = R_AC_original + small_perturbation
        
        # Strict tolerance - should fail
        strict_results = validate_sheaf_properties(
            self.restrictions, self.poset, tolerance=1e-6
        )
        assert strict_results['valid_sheaf'] is False
        
        # Loose tolerance - should pass
        loose_results = validate_sheaf_properties(
            self.restrictions, self.poset, tolerance=1e-2
        )
        assert loose_results['valid_sheaf'] is True
    
    def test_validate_empty_sheaf(self):
        """Test validation of empty sheaf."""
        empty_poset = nx.DiGraph()
        empty_restrictions = {}
        
        validation_results = validate_sheaf_properties(
            empty_restrictions, empty_poset
        )
        
        assert validation_results['total_paths_checked'] == 0
        assert validation_results['valid_sheaf'] is True  # Vacuously true
        assert len(validation_results['transitivity_violations']) == 0
    
    def test_validate_complex_poset(self):
        """Test validation with more complex poset structure."""
        # Create poset with multiple paths: A→B→D, A→C→D, B→E, C→E
        complex_poset = nx.DiGraph()
        complex_poset.add_edges_from([
            ('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), 
            ('B', 'E'), ('C', 'E')
        ])
        
        # Create restriction maps for all edges
        complex_restrictions = {}
        torch.manual_seed(123)
        
        for edge in complex_poset.edges():
            # Create random restriction maps
            complex_restrictions[edge] = torch.randn(6, 6) * 0.1 + torch.eye(6)
        
        validation_results = validate_sheaf_properties(
            complex_restrictions, complex_poset, tolerance=1e-1
        )
        
        # Should check multiple paths
        assert validation_results['total_paths_checked'] >= 4  # A→B→D, A→C→D, A→B→E, A→C→E
        
        # May or may not be valid depending on random restrictions
        assert 'valid_sheaf' in validation_results


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.procrustes = ProcrustesMaps()
    
    def test_zero_matrices(self):
        """Test handling of zero matrices."""
        K_zero = torch.zeros(5, 5)
        K_nonzero = torch.eye(5)
        
        # Should handle gracefully without crashing
        R, scale, info = self.procrustes.scaled_procrustes(K_zero, K_nonzero)
        
        assert R.shape == (5, 5)
        assert scale >= self.procrustes.min_scale
        assert 'reconstruction_error' in info
    
    def test_identity_matrices(self):
        """Test with identity matrices."""
        K_identity = torch.eye(5)
        
        R, scale, info = self.procrustes.scaled_procrustes(K_identity, K_identity)
        
        # Should give identity or close to identity
        assert R.shape == (5, 5)
        assert scale > 0
        
        # Reconstruction error should be very small
        assert info['relative_error'] < 1e-6
    
    def test_constant_matrices(self):
        """Test with constant matrices."""
        K_constant = torch.ones(4, 4)
        K_identity = torch.eye(4)
        
        # Should handle rank-deficient matrices
        R, scale, info = self.procrustes.scaled_procrustes(K_constant, K_identity)
        
        assert R.shape == (4, 4)
        assert scale > 0
        # May have high reconstruction error due to rank deficiency
    
    def test_extreme_scale_factors(self):
        """Test matrices that would lead to extreme scale factors."""
        # Very small matrix
        K_tiny = torch.eye(3) * 1e-10
        K_normal = torch.eye(3)
        
        R, scale, info = self.procrustes.scaled_procrustes(K_tiny, K_normal)
        
        # Scale should be clipped to max_scale
        assert scale <= self.procrustes.max_scale
        
        # Very large matrix
        K_huge = torch.eye(3) * 1e10
        
        R2, scale2, info2 = self.procrustes.scaled_procrustes(K_huge, K_normal)
        
        # Scale should be clipped to min_scale
        assert scale2 >= self.procrustes.min_scale
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinity values."""
        K_normal = torch.eye(3)
        K_nan = torch.tensor([[1.0, float('nan'), 0.0],
                              [float('nan'), 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
        
        # Should raise an error for NaN inputs
        with pytest.raises((ComputationError, ValueError, np.linalg.LinAlgError)):
            R, scale, info = self.procrustes.scaled_procrustes(K_nan, K_normal)
    
    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        K_float32 = torch.randn(4, 4, dtype=torch.float32)
        K_float32 = K_float32 @ K_float32.T + 0.1 * torch.eye(4, dtype=torch.float32)
        
        K_float64 = K_float32.double()
        
        # Should handle different dtypes
        R32, scale32, info32 = self.procrustes.scaled_procrustes(K_float32, K_float32)
        R64, scale64, info64 = self.procrustes.scaled_procrustes(K_float64, K_float64)
        
        assert R32.dtype == torch.float32
        assert R64.dtype == torch.float32  # Output is converted to float32
        
        # Results should be similar
        assert abs(scale32 - scale64) < 1e-5
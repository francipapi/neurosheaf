"""Tests for eigenvalue-preserving whitening functionality.

This module tests the core eigenvalue preservation features added in Phase 1:
- WhiteningProcessor with preserve_eigenvalues parameter
- Hodge adjoint computation
- Extended data structures
- Backward compatibility
"""

import torch
import numpy as np
import pytest

from neurosheaf.sheaf.core.whitening import WhiteningProcessor
from neurosheaf.sheaf.data_structures import Sheaf, EigenvalueMetadata, WhiteningInfo


class TestEigenvalueWhitening:
    """Test suite for eigenvalue-preserving whitening."""
    
    @pytest.fixture
    def sample_gram_matrix(self):
        """Create a sample Gram matrix for testing."""
        # Create a well-conditioned symmetric positive definite matrix
        torch.manual_seed(42)
        n = 10
        A = torch.randn(n, n)
        K = A @ A.T  # Guaranteed PSD
        # Add small diagonal term for numerical stability
        K += 0.1 * torch.eye(n)
        return K
    
    @pytest.fixture
    def small_gram_matrix(self):
        """Create a small Gram matrix for detailed testing."""
        # Create 3x3 matrix with known eigenvalues
        K = torch.tensor([
            [4.0, 2.0, 1.0],
            [2.0, 3.0, 0.5],
            [1.0, 0.5, 2.0]
        ], dtype=torch.float32)
        return K
    
    def test_preserve_eigenvalues_parameter(self, sample_gram_matrix):
        """Test that preserve_eigenvalues parameter is properly initialized."""
        # Test default (False)
        wp_default = WhiteningProcessor()
        assert wp_default.preserve_eigenvalues == False
        
        # Test explicit True
        wp_eigen = WhiteningProcessor(preserve_eigenvalues=True)
        assert wp_eigen.preserve_eigenvalues == True
        
        # Test explicit False
        wp_false = WhiteningProcessor(preserve_eigenvalues=False)
        assert wp_false.preserve_eigenvalues == False
    
    def test_eigenvalue_preservation_mode(self, sample_gram_matrix):
        """Test that eigenvalue preservation returns diagonal matrix instead of identity."""
        wp_eigen = WhiteningProcessor(preserve_eigenvalues=True)
        wp_identity = WhiteningProcessor(preserve_eigenvalues=False)
        
        # Get results from both modes
        K_eigen, W_eigen, info_eigen = wp_eigen.whiten_gram_matrix(sample_gram_matrix)
        K_identity, W_identity, info_identity = wp_identity.whiten_gram_matrix(sample_gram_matrix)
        
        # Check that eigenvalue mode returns diagonal matrix, not identity
        assert not torch.allclose(K_eigen, torch.eye(K_eigen.shape[0]))
        assert torch.allclose(K_identity, torch.eye(K_identity.shape[0]))
        
        # Check that eigenvalue matrix is diagonal
        assert torch.allclose(K_eigen, torch.diag(torch.diag(K_eigen)))
        
        # Check that diagonal elements are positive (eigenvalues)
        eigenvals = torch.diag(K_eigen)
        assert torch.all(eigenvals > 0)
        
        # Check metadata
        assert info_eigen['preserve_eigenvalues'] == True
        assert info_identity['preserve_eigenvalues'] == False
    
    def test_eigenvalue_diagonal_correctness(self, small_gram_matrix):
        """Test that eigenvalue diagonal matrix contains correct eigenvalues."""
        wp = WhiteningProcessor(preserve_eigenvalues=True)
        
        K_whitened, W, info = wp.whiten_gram_matrix(small_gram_matrix)
        
        # Get eigenvalues from diagonal matrix
        eigenvals_from_diagonal = torch.diag(K_whitened).detach().cpu().numpy()
        
        # Get eigenvalues from metadata
        eigenvals_from_info = info['eigenvalues']
        
        # They should match (up to ordering and truncation)
        r = len(eigenvals_from_diagonal)
        np.testing.assert_allclose(
            eigenvals_from_diagonal, 
            eigenvals_from_info[:r], 
            rtol=1e-5
        )
    
    def test_whitening_quality_eigenvalue_mode(self, sample_gram_matrix):
        """Test that whitening quality is correctly computed in eigenvalue mode."""
        wp = WhiteningProcessor(preserve_eigenvalues=True)
        
        K_whitened, W, info = wp.whiten_gram_matrix(sample_gram_matrix)
        
        # In eigenvalue preservation mode, W K W^T should still equal identity in standard inner product
        # But we choose to use eigenvalue diagonal matrix for the inner product structure
        K_compute = sample_gram_matrix.to(dtype=W.dtype)
        WKWt = W @ K_compute @ W.T
        identity = torch.eye(WKWt.shape[0], dtype=WKWt.dtype)
        
        # W K W^T should be close to identity (the whitening property still holds)
        whitening_error_identity = torch.norm(WKWt - identity, p='fro').item()
        assert whitening_error_identity < 1e-5, f"Whitening error (vs identity): {whitening_error_identity}"
        
        # The reported error should be the norm of WKWt vs K_whitened
        # This tests the implementation's error calculation method
        expected_error = torch.norm(WKWt - K_whitened, p='fro').item()
        
        # Check that error is reported correctly
        assert 'whitening_error' in info
        assert abs(info['whitening_error'] - expected_error) < 1e-6
    
    def test_backward_compatibility(self, sample_gram_matrix):
        """Test that preserve_eigenvalues=False gives identical results to old behavior."""
        wp_new = WhiteningProcessor(preserve_eigenvalues=False)
        wp_old_style = WhiteningProcessor()  # Default should be False
        
        # Both should give identical results
        K_new, W_new, info_new = wp_new.whiten_gram_matrix(sample_gram_matrix)
        K_old, W_old, info_old = wp_old_style.whiten_gram_matrix(sample_gram_matrix)
        
        # Results should be identical
        assert torch.allclose(K_new, K_old)
        assert torch.allclose(W_new, W_old)
        
        # Both should be identity matrices
        assert torch.allclose(K_new, torch.eye(K_new.shape[0]))
        assert torch.allclose(K_old, torch.eye(K_old.shape[0]))
    
    def test_hodge_adjoint_computation(self):
        """Test Hodge adjoint computation R* = Σₛ⁻¹ R^T Σₜ."""
        wp = WhiteningProcessor(preserve_eigenvalues=True, regularization=1e-12)
        
        # Create test matrices
        R = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float32)
        Sigma_source = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float32))
        Sigma_target = torch.diag(torch.tensor([1.5, 2.5], dtype=torch.float32))
        
        # Compute Hodge adjoint
        R_adjoint = wp.compute_hodge_adjoint(R, Sigma_source, Sigma_target)
        
        # Manually compute expected result: Σₛ⁻¹ R^T Σₜ
        Sigma_source_inv = torch.inverse(Sigma_source)
        expected = Sigma_source_inv @ R.T @ Sigma_target
        
        assert torch.allclose(R_adjoint, expected, atol=1e-6)
    
    def test_hodge_adjoint_fallback(self):
        """Test that Hodge adjoint falls back to transpose when preserve_eigenvalues=False."""
        wp = WhiteningProcessor(preserve_eigenvalues=False)
        
        R = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float32)
        Sigma_source = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float32))
        Sigma_target = torch.diag(torch.tensor([1.5, 2.5], dtype=torch.float32))
        
        # Should fall back to standard transpose
        R_adjoint = wp.compute_hodge_adjoint(R, Sigma_source, Sigma_target)
        
        assert torch.allclose(R_adjoint, R.T)
    
    def test_regularized_inverse(self):
        """Test regularized inverse computation for numerical stability."""
        wp = WhiteningProcessor(preserve_eigenvalues=True, regularization=1e-6)
        
        # Create a matrix with small eigenvalues
        Sigma = torch.diag(torch.tensor([1e-8, 1.0, 2.0], dtype=torch.float32))
        
        # Compute regularized inverse
        Sigma_inv = wp._compute_regularized_inverse(Sigma)
        
        # Check that it's approximately the inverse of regularized matrix
        Sigma_reg = Sigma + wp.regularization * torch.eye(3)
        expected_inv = torch.inverse(Sigma_reg)
        
        assert torch.allclose(Sigma_inv, expected_inv, atol=1e-6)
    
    def test_eigenvalue_metadata_structure(self):
        """Test EigenvalueMetadata dataclass functionality."""
        metadata = EigenvalueMetadata(
            preserve_eigenvalues=True,
            hodge_formulation_active=True
        )
        
        # Test basic properties
        assert metadata.preserve_eigenvalues == True
        assert metadata.hodge_formulation_active == True
        assert len(metadata.eigenvalue_matrices) == 0
        
        # Test summary
        summary = metadata.summary()
        assert "ACTIVE" in summary
        assert "Number of Stalks: 0" in summary
        
        # Test adding matrices
        metadata.eigenvalue_matrices["node1"] = torch.diag(torch.tensor([1.0, 2.0]))
        metadata.condition_numbers["node1"] = 5.0
        metadata.regularization_applied["node1"] = False
        
        # Test regularization summary
        reg_summary = metadata.get_regularization_summary()
        assert reg_summary['total_stalks'] == 1
        assert reg_summary['regularized_stalks'] == 0
        assert reg_summary['regularization_fraction'] == 0.0
    
    def test_whitening_info_extensions(self):
        """Test extended WhiteningInfo with eigenvalue fields."""
        eigenval_diagonal = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
        
        info = WhiteningInfo(
            whitening_matrix=torch.eye(3),
            eigenvalues=torch.tensor([1.0, 2.0, 3.0]),
            condition_number=3.0,
            rank=3,
            eigenvalue_diagonal=eigenval_diagonal,
            preserve_eigenvalues=True
        )
        
        # Test basic properties
        assert info.preserve_eigenvalues == True
        assert torch.allclose(info.eigenvalue_diagonal, eigenval_diagonal)
        
        # Test summary includes eigenvalue status
        summary = info.summary()
        assert "Eigenvalue Preservation: ENABLED" in summary
    
    def test_sheaf_eigenvalue_metadata_integration(self):
        """Test Sheaf dataclass with eigenvalue_metadata field."""
        # Create eigenvalue metadata
        eigenvalue_metadata = EigenvalueMetadata(
            preserve_eigenvalues=True,
            hodge_formulation_active=True
        )
        
        # Create sheaf with eigenvalue metadata
        sheaf = Sheaf(eigenvalue_metadata=eigenvalue_metadata)
        
        # Test that metadata is properly stored
        assert sheaf.eigenvalue_metadata is not None
        assert sheaf.eigenvalue_metadata.preserve_eigenvalues == True
        assert sheaf.eigenvalue_metadata.hodge_formulation_active == True
    
    def test_different_precisions(self, sample_gram_matrix):
        """Test eigenvalue preservation with different precision settings."""
        # Test single precision
        wp_single = WhiteningProcessor(
            preserve_eigenvalues=True, 
            use_double_precision=False
        )
        K_single, W_single, info_single = wp_single.whiten_gram_matrix(sample_gram_matrix)
        
        # Test double precision
        wp_double = WhiteningProcessor(
            preserve_eigenvalues=True, 
            use_double_precision=True
        )
        K_double, W_double, info_double = wp_double.whiten_gram_matrix(sample_gram_matrix)
        
        # Check that both preserve eigenvalues (with appropriate dtype for comparison)
        identity_single = torch.eye(K_single.shape[0], dtype=K_single.dtype)
        identity_double = torch.eye(K_double.shape[0], dtype=K_double.dtype)
        
        assert not torch.allclose(K_single, identity_single)
        assert not torch.allclose(K_double, identity_double)
        
        # Check that eigenvalues are similar (accounting for precision differences)
        eigenvals_single = torch.diag(K_single)
        eigenvals_double = torch.diag(K_double).to(dtype=eigenvals_single.dtype)
        
        assert torch.allclose(eigenvals_single, eigenvals_double, rtol=1e-4)


class TestNumericalStability:
    """Test numerical stability of eigenvalue preservation."""
    
    def test_ill_conditioned_matrix(self):
        """Test eigenvalue preservation with ill-conditioned matrices."""
        # Create ill-conditioned matrix
        eigenvals = torch.tensor([1e-8, 1e-6, 1.0, 10.0])
        n = len(eigenvals)
        Q, _ = torch.linalg.qr(torch.randn(n, n))
        K = Q @ torch.diag(eigenvals) @ Q.T
        
        wp = WhiteningProcessor(
            preserve_eigenvalues=True, 
            regularization=1e-10,
            min_eigenvalue=1e-8
        )
        
        # Should handle ill-conditioning gracefully
        K_whitened, W, info = wp.whiten_gram_matrix(K)
        
        # Check that we get a diagonal matrix
        assert torch.allclose(K_whitened, torch.diag(torch.diag(K_whitened)))
        
        # Check condition number is reported
        assert 'condition_number' in info
        assert info['condition_number'] > 1e6  # Should be ill-conditioned
    
    def test_regularization_effectiveness(self):
        """Test that regularization prevents numerical issues."""
        # Create matrix with very small eigenvalue
        Sigma = torch.diag(torch.tensor([1e-12, 1.0], dtype=torch.float32))
        
        wp = WhiteningProcessor(preserve_eigenvalues=True, regularization=1e-6)
        
        # Compute regularized inverse
        Sigma_inv = wp._compute_regularized_inverse(Sigma)
        
        # Should not have infinite or NaN values
        assert torch.isfinite(Sigma_inv).all()
        assert not torch.isnan(Sigma_inv).any()
        
        # Largest element should be bounded due to regularization
        max_element = torch.max(torch.abs(Sigma_inv))
        assert max_element < 1e6  # Reasonable bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
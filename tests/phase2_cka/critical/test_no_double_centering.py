"""CRITICAL: Tests to ensure NO double-centering occurs in Debiased CKA.

This is the most critical aspect of Phase 2 - ensuring that the debiased CKA
implementation uses raw activations without pre-centering, as the unbiased HSIC
estimator handles centering internally.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestNoDoubleCenteringCritical:
    """CRITICAL: Exhaustive tests to ensure NO double-centering."""
    
    def test_kernel_computation_uses_raw_data(self):
        """Verify kernels are computed from raw, uncentered activations."""
        # Create data with obvious non-zero mean but avoid extreme values that cause numerical issues
        torch.manual_seed(42)  # Ensure reproducibility
        X = torch.ones(50, 30) * 2.0 + torch.randn(50, 30) * 0.5
        Y = torch.ones(50, 30) * 1.5 + torch.randn(50, 30) * 0.5
        
        # Verify the data has non-zero mean
        assert torch.abs(X.mean()) > 1.0, "Test data X should have mean ~2.0"
        assert torch.abs(Y.mean()) > 1.0, "Test data Y should have mean ~1.5"
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Capture intermediate computations
        captured_kernels = {}
        
        # Patch the unbiased HSIC method to capture kernel matrices
        original_unbiased_hsic = cka._unbiased_hsic
        
        def capturing_unbiased_hsic(K, L):
            # Store the first call (K with L)
            if 'K' not in captured_kernels:
                captured_kernels['K'] = K.clone()
                captured_kernels['L'] = L.clone()
            return original_unbiased_hsic(K, L)
        
        cka._unbiased_hsic = capturing_unbiased_hsic
        
        # Compute CKA
        cka_value = cka.compute(X, Y)
        
        # Verify the kernels were computed from raw data
        # Move to same device for comparison
        expected_K = (X @ X.T).to(cka.device)
        expected_L = (Y @ Y.T).to(cka.device)
        
        torch.testing.assert_close(captured_kernels['K'], expected_K, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(captured_kernels['L'], expected_L, rtol=1e-5, atol=1e-5)
        
        # Verify kernels have high values (indicating non-centered data)
        K_mean = captured_kernels['K'].mean().item()
        L_mean = captured_kernels['L'].mean().item()
        
        # For data with mean ~2 and ~1.5, kernel means should be reasonably high
        assert K_mean > 100, f"K mean {K_mean} too low - data may have been centered"
        assert L_mean > 50, f"L mean {L_mean} too low - data may have been centered"
    
    def test_biased_vs_unbiased_difference(self):
        """Ensure biased and unbiased estimators give different results."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        X = torch.randn(100, 50) + 3.0
        Y = torch.randn(100, 40) + 2.0
        
        cka_unbiased = DebiasedCKA(use_unbiased=True)
        cka_biased = DebiasedCKA(use_unbiased=False)
        
        value_unbiased = cka_unbiased.compute(X, Y)
        value_biased = cka_biased.compute(X, Y)
        
        # Should be different
        assert abs(value_unbiased - value_biased) > 0.001, \
            f"Unbiased ({value_unbiased:.6f}) and biased ({value_biased:.6f}) should differ"
    
    def test_mean_shift_invariance(self):
        """CKA should be invariant to mean shifts when computed correctly."""
        torch.manual_seed(42)
        
        X = torch.randn(100, 50)
        Y = torch.randn(100, 40)
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Original
        cka1 = cka.compute(X, Y)
        
        # Shifted by large constants
        X_shifted = X + 100
        Y_shifted = Y + 200
        cka2 = cka.compute(X_shifted, Y_shifted)
        
        # Should be approximately equal (within numerical precision)
        assert abs(cka1 - cka2) < 1e-4, \
            f"CKA not invariant to mean shift: {cka1:.6f} vs {cka2:.6f}"
    
    def test_comparison_with_incorrect_double_centering(self):
        """Compare correct vs incorrect (double-centered) implementation."""
        torch.manual_seed(42)
        
        # Generate data with strong mean bias to make double-centering more obvious
        n_samples = 128
        n_features = 64
        
        # Create data with large means that will be affected by centering
        X = torch.randn(n_samples, n_features) + 10.0  # Large mean
        Y = torch.randn(n_samples, n_features) + 8.0   # Large mean
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Correct: Use raw activations
        cka_correct = cka.compute(X, Y)
        
        # Wrong: Pre-center the data (simulating double-centering)
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        
        # Manually compute what would happen with double-centering
        # Move to device for computation
        X_centered = X_centered.to(cka.device)
        Y_centered = Y_centered.to(cka.device)
        K_centered = X_centered @ X_centered.T
        L_centered = Y_centered @ Y_centered.T
        cka_wrong = cka._compute_unbiased_cka(K_centered, L_centered)
        
        # For data with large mean offsets, pre-centering should give different results
        # The direction depends on the data, but they should be noticeably different
        difference = abs(cka_correct - cka_wrong)
        assert difference > 0.001, \
            f"CKA values too similar ({difference:.6f}) - double-centering effect not detected. " \
            f"Correct: {cka_correct:.4f}, Wrong: {cka_wrong:.4f}"
    
    def test_warning_on_centered_input(self):
        """Test that validation warns about pre-centered data."""
        # Create pre-centered data
        X = torch.randn(100, 50)
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        Y = torch.randn(100, 50)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        
        activations = {
            'layer1': X_centered,
            'layer2': Y_centered
        }
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Should warn about centered data
        with pytest.warns(UserWarning, match="appears to be centered"):
            cka.compute_cka_matrix(activations, warn_preprocessing=True)
    
    def test_gram_matrix_computation(self):
        """Verify Gram matrices are computed correctly from raw data."""
        torch.manual_seed(42)
        
        X = torch.randn(50, 30) + 2.0
        Y = torch.randn(50, 25) + 1.5
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Compute CKA and capture Gram matrices
        gram_matrices = {}
        
        original_compute_unbiased = cka._compute_unbiased_cka
        
        def capture_grams(K, L):
            gram_matrices['K'] = K.clone()
            gram_matrices['L'] = L.clone()
            return original_compute_unbiased(K, L)
        
        cka._compute_unbiased_cka = capture_grams
        
        cka_value = cka.compute(X, Y)
        
        # Verify Gram matrices match expected computation
        # Move to same device for comparison
        expected_K = (X @ X.T).to(cka.device)
        expected_L = (Y @ Y.T).to(cka.device)
        
        torch.testing.assert_close(gram_matrices['K'], expected_K)
        torch.testing.assert_close(gram_matrices['L'], expected_L)
    
    def test_minimum_samples_requirement(self):
        """Unbiased HSIC requires at least 4 samples."""
        cka = DebiasedCKA(use_unbiased=True)
        
        # Should fail with < 4 samples
        X = torch.randn(3, 10)
        Y = torch.randn(3, 10)
        
        with pytest.raises(ValidationError, match="at least 4 samples"):
            cka.compute(X, Y)
        
        # Should work with exactly 4
        X = torch.randn(4, 10)
        Y = torch.randn(4, 10)
        result = cka.compute(X, Y)
        assert 0 <= result <= 1
    
    def test_hsic_formula_correctness(self):
        """Verify the unbiased HSIC formula is implemented correctly."""
        torch.manual_seed(42)
        n = 10  # Small size for manual verification
        
        X = torch.randn(n, 5)
        Y = torch.randn(n, 5)
        
        K = X @ X.T
        L = Y @ Y.T
        
        cka = DebiasedCKA(use_unbiased=True)
        hsic = cka._unbiased_hsic(K, L)
        
        # Manually compute unbiased HSIC
        K_0 = K - torch.diag(torch.diag(K))
        L_0 = L - torch.diag(torch.diag(L))
        
        term1 = torch.sum(K_0 * L_0)
        term2 = torch.sum(K_0) * torch.sum(L_0) / ((n - 1) * (n - 2))
        term3 = 2 * torch.sum(K_0, dim=0) @ torch.sum(L_0, dim=0) / (n - 2)
        
        expected_hsic = (term1 + term2 - term3) / (n * (n - 3))
        
        torch.testing.assert_close(hsic, expected_hsic, rtol=1e-5, atol=1e-7)
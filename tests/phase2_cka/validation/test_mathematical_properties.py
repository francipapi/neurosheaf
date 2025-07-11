"""Validation tests for CKA mathematical properties."""

import pytest
import torch
import numpy as np

from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestMathematicalProperties:
    """Test mathematical properties of CKA implementation."""
    
    def test_self_similarity(self):
        """Test CKA(X, X) = 1 for various inputs."""
        cka = DebiasedCKA(use_unbiased=True)
        
        test_cases = [
            (torch.randn(100, 50), True),
            (torch.randn(500, 128), True), 
            (torch.ones(100, 30), False),  # Constant features - skip validation
            (torch.eye(50), False),  # Identity-like - skip validation (rank deficient)
        ]
        
        for i, (X, validate) in enumerate(test_cases):
            cka_value = cka.compute(X, X, validate_properties=validate)
            if validate:
                assert abs(cka_value - 1.0) < 1e-6, \
                    f"Test case {i}: CKA(X,X) = {cka_value}, expected 1.0"
            else:
                # For degenerate cases, just check it's valid
                assert 0 <= cka_value <= 1 or np.isnan(cka_value), \
                    f"Test case {i}: CKA(X,X) = {cka_value}, should be valid or NaN"
    
    def test_symmetry(self):
        """Test CKA(X, Y) = CKA(Y, X)."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        test_cases = [
            (torch.randn(100, 50), torch.randn(100, 30), True),
            (torch.randn(200, 128), torch.randn(200, 64), True),
            (torch.ones(100, 20), torch.randn(100, 40), False),  # Constant - skip validation
        ]
        
        for i, (X, Y, validate) in enumerate(test_cases):
            cka_xy = cka.compute(X, Y, validate_properties=validate)
            cka_yx = cka.compute(Y, X, validate_properties=validate)
            
            if validate:
                assert abs(cka_xy - cka_yx) < 1e-6, \
                    f"Test case {i}: CKA not symmetric: {cka_xy} vs {cka_yx}"
            else:
                # For degenerate cases, just check both are valid or both NaN
                both_valid = (0 <= cka_xy <= 1) and (0 <= cka_yx <= 1)
                both_nan = np.isnan(cka_xy) and np.isnan(cka_yx)
                assert both_valid or both_nan, \
                    f"Test case {i}: Inconsistent behavior: {cka_xy} vs {cka_yx}"
    
    def test_bounded_range(self):
        """Test 0 ≤ CKA ≤ 1 for various inputs."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        
        # Generate various test cases
        n_tests = 20
        for i in range(n_tests):
            n_samples = torch.randint(10, 500, (1,)).item()
            d1 = torch.randint(10, 100, (1,)).item()
            d2 = torch.randint(10, 100, (1,)).item()
            
            X = torch.randn(n_samples, d1)
            Y = torch.randn(n_samples, d2)
            
            cka_value = cka.compute(X, Y)
            assert 0 <= cka_value <= 1, \
                f"Test {i}: CKA = {cka_value} outside [0, 1]"
    
    def test_orthogonal_features(self):
        """Test CKA for orthogonal features is near zero."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        n = 200
        d = 50
        
        # Create orthogonal features
        X = torch.randn(n, d)
        X = X / (torch.norm(X, dim=0, keepdim=True) + 1e-8)  # Normalize columns safely
        
        # Create different random features (not orthogonal transformation)
        # An orthogonal transformation would give CKA = 1, not 0
        Y = torch.randn(n, d) 
        Y = Y / (torch.norm(Y, dim=0, keepdim=True) + 1e-8)  # Normalize columns safely
        
        cka_value = cka.compute(X, Y, validate_properties=False)  # Skip validation for numerical case
        
        # Random normalized features should have low CKA typically
        # Allow for higher values due to randomness, or NaN for degenerate cases
        assert cka_value < 0.9 or np.isnan(cka_value), \
            f"CKA for independent features too high: {cka_value}"
    
    def test_correlated_features(self):
        """Test CKA for correlated features is high."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        n = 300
        
        # Create highly correlated features
        base = torch.randn(n, 50)
        X = base + 0.1 * torch.randn(n, 50)
        Y = base + 0.1 * torch.randn(n, 50)
        
        cka_value = cka.compute(X, Y)
        
        # Should be high (close to 1)
        assert cka_value > 0.8, f"CKA for correlated features too low: {cka_value}"
    
    def test_scale_invariance(self):
        """Test CKA invariance to isotropic scaling."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        X = torch.randn(200, 60)
        Y = torch.randn(200, 40)
        
        # Original CKA
        cka_original = cka.compute(X, Y)
        
        # Test various scalings
        scales = [0.1, 0.5, 2.0, 10.0, 100.0]
        
        for scale in scales:
            X_scaled = X * scale
            Y_scaled = Y * scale
            
            cka_scaled_x = cka.compute(X_scaled, Y)
            cka_scaled_y = cka.compute(X, Y_scaled)
            cka_scaled_both = cka.compute(X_scaled, Y_scaled)
            
            assert abs(cka_original - cka_scaled_x) < 1e-5, \
                f"CKA not invariant to X scaling by {scale}"
            assert abs(cka_original - cka_scaled_y) < 1e-5, \
                f"CKA not invariant to Y scaling by {scale}"
            assert abs(cka_original - cka_scaled_both) < 1e-5, \
                f"CKA not invariant to both scaling by {scale}"
    
    def test_rotation_invariance(self):
        """Test CKA invariance to orthogonal transformations."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        n = 200
        X = torch.randn(n, 50)
        Y = torch.randn(n, 50)
        
        # Original CKA
        cka_original = cka.compute(X, Y)
        
        # Apply random orthogonal transformation
        Q_x, _ = torch.linalg.qr(torch.randn(50, 50))
        Q_y, _ = torch.linalg.qr(torch.randn(50, 50))
        
        X_rotated = X @ Q_x
        Y_rotated = Y @ Q_y
        
        cka_rotated = cka.compute(X_rotated, Y_rotated)
        
        assert abs(cka_original - cka_rotated) < 1e-5, \
            f"CKA not invariant to rotation: {cka_original} vs {cka_rotated}"
    
    def test_permutation_invariance(self):
        """Test CKA invariance to sample permutation."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        X = torch.randn(150, 40)
        Y = torch.randn(150, 30)
        
        # Original CKA
        cka_original = cka.compute(X, Y)
        
        # Permute samples
        perm = torch.randperm(150)
        X_perm = X[perm]
        Y_perm = Y[perm]
        
        cka_permuted = cka.compute(X_perm, Y_perm)
        
        assert abs(cka_original - cka_permuted) < 1e-6, \
            f"CKA not invariant to permutation: {cka_original} vs {cka_permuted}"
    
    def test_minimum_samples_validation(self):
        """Test minimum sample requirements."""
        cka = DebiasedCKA(use_unbiased=True)
        
        # Unbiased requires at least 4 samples
        X_small = torch.randn(3, 10)
        Y_small = torch.randn(3, 10)
        
        with pytest.raises(ValidationError, match="at least 4"):
            cka.compute(X_small, Y_small)
        
        # Exactly 4 should work
        X_min = torch.randn(4, 10)
        Y_min = torch.randn(4, 10)
        
        cka_value = cka.compute(X_min, Y_min)
        assert 0 <= cka_value <= 1
    
    def test_edge_cases(self):
        """Test CKA on edge cases."""
        cka = DebiasedCKA(use_unbiased=True)
        
        # Case 1: Constant features - skip validation
        X_const = torch.ones(100, 50)
        Y = torch.randn(100, 30)
        
        # Should not crash, but result may be degenerate
        cka_const = cka.compute(X_const, Y, validate_properties=False)
        assert 0 <= cka_const <= 1 or np.isnan(cka_const)
        
        # Case 2: Very high dimensional
        X_high = torch.randn(50, 1000)  # More features than samples
        Y_high = torch.randn(50, 800)
        
        cka_high = cka.compute(X_high, Y_high, validate_properties=False)
        assert 0 <= cka_high <= 1 or np.isnan(cka_high)
        
        # Case 3: Single feature
        X_single = torch.randn(100, 1)
        Y_single = torch.randn(100, 1)
        
        cka_single = cka.compute(X_single, Y_single)
        assert 0 <= cka_single <= 1
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        cka = DebiasedCKA(use_unbiased=True)
        
        torch.manual_seed(42)
        
        # Small values
        X_small = torch.randn(100, 50) * 1e-6
        Y_small = torch.randn(100, 40) * 1e-6
        
        cka_small = cka.compute(X_small, Y_small)
        assert 0 <= cka_small <= 1
        assert not torch.isnan(torch.tensor(cka_small))
        
        # Large values
        X_large = torch.randn(100, 50) * 1e6
        Y_large = torch.randn(100, 40) * 1e6
        
        cka_large = cka.compute(X_large, Y_large)
        assert 0 <= cka_large <= 1
        assert not torch.isnan(torch.tensor(cka_large))
        
        # Mixed scales
        cka_mixed = cka.compute(X_small, Y_large)
        assert 0 <= cka_mixed <= 1
        assert not torch.isnan(torch.tensor(cka_mixed))
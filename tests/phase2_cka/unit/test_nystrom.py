"""Unit tests for Nyström CKA implementation."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from neurosheaf.cka.nystrom import NystromCKA
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestNystromCKA:
    """Test Nyström CKA implementation."""
    
    def test_initialization(self):
        """Test NystromCKA initialization."""
        # Default initialization
        nystrom = NystromCKA()
        assert nystrom.n_landmarks == 256
        assert nystrom.landmark_selection == 'uniform'
        assert nystrom.eps == 1e-6
        
        # Custom initialization
        nystrom = NystromCKA(
            n_landmarks=128,
            landmark_selection='kmeans',
            numerical_stability=1e-5
        )
        assert nystrom.n_landmarks == 128
        assert nystrom.landmark_selection == 'kmeans'
        assert nystrom.eps == 1e-5
    
    def test_initialization_validation(self):
        """Test initialization parameter validation."""
        # Too few landmarks
        with pytest.raises(ValidationError, match="at least 4"):
            NystromCKA(n_landmarks=3)
        
        # Invalid landmark selection
        with pytest.raises(ValidationError, match="Unknown landmark_selection"):
            NystromCKA(landmark_selection='invalid')
    
    def test_device_detection(self):
        """Test device detection for different platforms."""
        nystrom = NystromCKA()
        
        # Device should be detected automatically
        assert nystrom.device.type in ['cpu', 'cuda', 'mps']
        
        # Explicit device setting
        nystrom = NystromCKA(device='cpu')
        assert nystrom.device.type == 'cpu'
    
    def test_uniform_landmark_selection(self):
        """Test uniform landmark selection strategy."""
        torch.manual_seed(42)
        nystrom = NystromCKA(n_landmarks=10, landmark_selection='uniform')
        
        X = torch.randn(100, 20)
        landmarks = nystrom._select_landmarks(X, 10)
        
        assert len(landmarks) == 10
        assert torch.all(landmarks >= 0)
        assert torch.all(landmarks < 100)
        assert len(torch.unique(landmarks)) == 10  # All unique
    
    def test_landmark_selection_edge_cases(self):
        """Test landmark selection edge cases."""
        nystrom = NystromCKA(n_landmarks=100)
        
        # More landmarks than samples
        X = torch.randn(50, 20)
        landmarks = nystrom._select_landmarks(X, 100)
        
        assert len(landmarks) == 50  # Should return all samples
        assert torch.all(landmarks == torch.arange(50, device=X.device))
    
    def test_kmeans_landmark_selection(self):
        """Test k-means landmark selection."""
        torch.manual_seed(42)
        nystrom = NystromCKA(n_landmarks=5, landmark_selection='kmeans')
        
        # Create clustered data
        cluster1 = torch.randn(20, 10) + 2.0  # Shift all features by 2
        cluster2 = torch.randn(20, 10) - 2.0  # Shift all features by -2
        X = torch.cat([cluster1, cluster2], dim=0)
        
        landmarks = nystrom._select_landmarks(X, 5)
        
        assert len(landmarks) == 5
        assert torch.all(landmarks >= 0)
        assert torch.all(landmarks < 40)
        assert len(torch.unique(landmarks)) == 5
    
    def test_kmeans_fallback(self):
        """Test k-means fallback to uniform selection."""
        nystrom = NystromCKA(n_landmarks=5, landmark_selection='kmeans')
        
        # Create normal data but mock k-means to fail
        X = torch.randn(100, 10)
        
        with patch('neurosheaf.cka.nystrom.KMeans') as mock_kmeans:
            mock_kmeans.side_effect = Exception("K-means failed")
            
            landmarks = nystrom._select_landmarks(X, 5)
            
            # Should fall back to uniform selection
            assert len(landmarks) == 5
    
    def test_stable_inverse(self):
        """Test stable matrix inversion."""
        nystrom = NystromCKA()
        
        # Well-conditioned matrix
        A = torch.randn(10, 10)
        A = A @ A.T + torch.eye(10) * 0.1  # Make positive definite
        
        A_inv = nystrom._stable_inverse(A)
        
        # Check that A @ A_inv ≈ I
        product = A @ A_inv
        identity = torch.eye(10, device=A.device)
        assert torch.allclose(product, identity, atol=1e-4)
    
    def test_stable_inverse_singular(self):
        """Test stable inversion with singular matrix."""
        nystrom = NystromCKA()
        
        # Singular matrix (rank deficient)
        A = torch.randn(10, 5)
        A = A @ A.T  # Rank 5 matrix
        
        # Should not crash
        A_inv = nystrom._stable_inverse(A)
        assert A_inv.shape == (10, 10)
    
    def test_nystrom_kernel_computation(self):
        """Test Nyström kernel approximation."""
        torch.manual_seed(42)
        nystrom = NystromCKA(n_landmarks=10)
        
        X = torch.randn(50, 20)
        landmarks = torch.randperm(50)[:10]
        
        K_approx = nystrom._compute_nystrom_kernel(X, landmarks)
        
        # Check output shape
        assert K_approx.shape == (50, 50)
        
        # Check symmetry (Nyström approximation may have small numerical differences)
        assert torch.allclose(K_approx, K_approx.T, atol=1e-5)
        
        # Check positive semi-definite (allow small numerical errors in Nyström approximation)
        eigenvals = torch.linalg.eigvalsh(K_approx)
        assert torch.all(eigenvals >= -3e-5)  # Allow tolerance for Nyström approximation errors
    
    def test_nystrom_vs_exact_small_data(self):
        """Test Nyström approximation vs exact computation on small data."""
        torch.manual_seed(42)
        
        # Small data where we can compare exact vs approximate
        X = torch.randn(30, 15)
        Y = torch.randn(30, 10)
        
        # Exact CKA
        exact_cka = DebiasedCKA(use_unbiased=True)
        exact_value = exact_cka.compute(X, Y)
        
        # Nyström CKA with many landmarks (should be close to exact)
        nystrom = NystromCKA(n_landmarks=25)
        nystrom_value = nystrom.compute(X, Y)
        
        # Should be reasonably close
        assert abs(exact_value - nystrom_value) < 0.1
        assert 0 <= nystrom_value <= 1
    
    def test_self_similarity_property(self):
        """Test that CKA(X, X) ≈ 1 for Nyström approximation."""
        torch.manual_seed(42)
        nystrom = NystromCKA(n_landmarks=50)
        
        X = torch.randn(100, 30)
        
        # Self-similarity should be close to 1
        cka_value = nystrom.compute(X, X, validate_properties=False)
        
        # Allow larger tolerance for approximation (Nyström is low-rank)
        assert abs(cka_value - 1.0) < 0.3
    
    def test_symmetry_property(self):
        """Test that CKA(X, Y) = CKA(Y, X) for Nyström."""
        torch.manual_seed(42)
        nystrom = NystromCKA(n_landmarks=32)
        
        X = torch.randn(80, 25)
        Y = torch.randn(80, 20)
        
        cka_xy = nystrom.compute(X, Y)
        cka_yx = nystrom.compute(Y, X)
        
        # Should be symmetric (allowing for numerical differences)
        assert abs(cka_xy - cka_yx) < 0.01
    
    def test_unbiased_hsic_computation(self):
        """Test unbiased HSIC computation."""
        nystrom = NystromCKA()
        
        # Test with small matrices
        K = torch.randn(10, 10)
        K = K @ K.T  # Make positive semi-definite
        L = torch.randn(10, 10)
        L = L @ L.T
        
        hsic_value = nystrom._unbiased_hsic(K, L)
        
        # Should be a scalar
        assert hsic_value.dim() == 0
        
        # Should not be NaN
        assert not torch.isnan(hsic_value)
    
    def test_unbiased_hsic_minimum_samples(self):
        """Test unbiased HSIC minimum sample requirement."""
        nystrom = NystromCKA()
        
        # Too few samples
        K = torch.randn(3, 3)
        L = torch.randn(3, 3)
        
        with pytest.raises(ValidationError, match="at least 4 samples"):
            nystrom._unbiased_hsic(K, L)
    
    def test_approximation_quality_metrics(self):
        """Test approximation quality computation."""
        torch.manual_seed(42)
        nystrom = NystromCKA(n_landmarks=16)
        
        # Small data for comparison
        X = torch.randn(32, 20)
        Y = torch.randn(32, 15)
        
        cka_value, approx_info = nystrom.compute(
            X, Y, return_approximation_info=True
        )
        
        # Should return approximation info
        assert 'n_landmarks' in approx_info
        assert 'n_samples' in approx_info
        assert approx_info['n_landmarks'] == 16
        assert approx_info['n_samples'] == 32
        
        # For small data, should also have error metrics
        if 'k_approximation_error' in approx_info:
            assert approx_info['k_approximation_error'] >= 0
            assert approx_info['l_approximation_error'] >= 0
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        nystrom = NystromCKA(n_landmarks=100)
        
        memory_info = nystrom.estimate_memory_usage(
            n_samples=1000, n_features=512
        )
        
        # Should contain expected keys
        assert 'exact_cka_mb' in memory_info
        assert 'nystrom_cka_mb' in memory_info
        assert 'memory_reduction_factor' in memory_info
        
        # Nyström should use less memory
        assert memory_info['nystrom_cka_mb'] < memory_info['exact_cka_mb']
        assert memory_info['memory_reduction_factor'] > 1
    
    def test_landmark_recommendation(self):
        """Test landmark count recommendation."""
        nystrom = NystromCKA()
        
        # Small dataset
        rec_small = nystrom.recommend_landmarks(100, target_error=0.01)
        assert 4 <= rec_small <= 50  # Should be reasonable
        
        # Large dataset
        rec_large = nystrom.recommend_landmarks(10000, target_error=0.01)
        assert rec_small < rec_large  # Should recommend more for larger data
        
        # Lower error tolerance
        rec_precise = nystrom.recommend_landmarks(1000, target_error=0.001)
        rec_rough = nystrom.recommend_landmarks(1000, target_error=0.1)
        assert rec_precise > rec_rough  # More landmarks for higher precision
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        nystrom = NystromCKA(n_landmarks=8)
        
        # Constant features
        X = torch.ones(20, 10)
        Y = torch.randn(20, 10)
        
        # Should not crash
        cka_value = nystrom.compute(X, Y, validate_properties=False)
        assert 0 <= cka_value <= 1 or np.isnan(cka_value)
        
        # Very small values
        X = torch.randn(20, 10) * 1e-8
        Y = torch.randn(20, 10) * 1e-8
        
        cka_value = nystrom.compute(X, Y, validate_properties=False)
        assert 0 <= cka_value <= 1 or np.isnan(cka_value)
    
    def test_large_scale_computation(self):
        """Test computation on larger scale data."""
        torch.manual_seed(42)
        nystrom = NystromCKA(n_landmarks=64)
        
        # Larger data that would be expensive for exact computation
        X = torch.randn(1000, 128)
        Y = torch.randn(1000, 64)
        
        cka_value = nystrom.compute(X, Y)
        
        # Should complete without error
        assert 0 <= cka_value <= 1
    
    def test_device_consistency(self):
        """Test that computation works across devices."""
        nystrom = NystromCKA(n_landmarks=16)
        
        # CPU computation
        X_cpu = torch.randn(50, 20)
        Y_cpu = torch.randn(50, 15)
        
        cka_cpu = nystrom.compute(X_cpu, Y_cpu)
        
        # Should work on CPU
        assert 0 <= cka_cpu <= 1
        
        # Test device moves
        if torch.cuda.is_available():
            nystrom_gpu = NystromCKA(n_landmarks=16, device='cuda')
            X_gpu = X_cpu.cuda()
            Y_gpu = Y_cpu.cuda()
            
            cka_gpu = nystrom_gpu.compute(X_gpu, Y_gpu)
            
            # Should give similar results
            assert abs(cka_cpu - cka_gpu) < 0.01
    
    def test_validation_error_handling(self):
        """Test validation and error handling."""
        nystrom = NystromCKA(n_landmarks=10)
        
        # Mismatched sample sizes
        X = torch.randn(50, 20)
        Y = torch.randn(40, 15)  # Different number of samples
        
        with pytest.raises(ValidationError):
            nystrom.compute(X, Y)
        
        # Too few samples
        X = torch.randn(2, 20)
        Y = torch.randn(2, 15)
        
        with pytest.raises(ValidationError):
            nystrom.compute(X, Y)
    
    def test_property_validation(self):
        """Test mathematical property validation."""
        nystrom = NystromCKA(n_landmarks=20)
        
        # Test with identical inputs
        X = torch.randn(50, 25)
        
        # Should validate self-similarity
        cka_value = nystrom.compute(X, X, validate_properties=True)
        
        # Should be close to 1 (allowing for approximation error)
        assert abs(cka_value - 1.0) < 0.1
    
    def test_approximation_convergence(self):
        """Test that approximation improves with more landmarks."""
        torch.manual_seed(42)
        
        # Fixed small dataset for comparison
        X = torch.randn(100, 50)
        Y = torch.randn(100, 40)
        
        # Exact CKA
        exact_cka = DebiasedCKA(use_unbiased=True)
        exact_value = exact_cka.compute(X, Y)
        
        # Test with different landmark counts
        landmark_counts = [10, 20, 40, 80]
        errors = []
        
        for n_landmarks in landmark_counts:
            nystrom = NystromCKA(n_landmarks=n_landmarks)
            approx_value = nystrom.compute(X, Y)
            error = abs(approx_value - exact_value)
            errors.append(error)
        
        # Errors should generally decrease with more landmarks
        # (allowing for some randomness)
        assert errors[-1] < errors[0] * 1.5  # Final error should be reasonably smaller
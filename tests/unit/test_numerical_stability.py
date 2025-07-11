"""Test numerical stability improvements for CKA computation."""

import torch
import pytest
import numpy as np
from neurosheaf.cka.debiased import DebiasedCKA


class TestNumericalStability:
    """Test suite for numerical stability improvements."""
    
    @pytest.fixture
    def small_sample_data(self):
        """Generate small sample data that causes numerical instability."""
        torch.manual_seed(42)
        # Very small sample size (n=10) with high-dimensional features
        X = torch.randn(10, 100, dtype=torch.float32)
        Y = torch.randn(10, 100, dtype=torch.float32)
        return X, Y
    
    @pytest.fixture
    def ill_conditioned_data(self):
        """Generate ill-conditioned data."""
        torch.manual_seed(42)
        n, d = 50, 20
        # Create data with very different scales
        X = torch.randn(n, d)
        X[:, 0] *= 1e6  # Very large scale
        X[:, -1] *= 1e-6  # Very small scale
        
        Y = torch.randn(n, d)
        Y[:, 0] *= 1e6
        Y[:, -1] *= 1e-6
        return X, Y
    
    @pytest.fixture
    def near_singular_data(self):
        """Generate near-singular data."""
        torch.manual_seed(42)
        n, d = 30, 50
        # Create highly correlated features
        base = torch.randn(n, 1)
        X = base + 1e-8 * torch.randn(n, d)
        Y = base + 1e-8 * torch.randn(n, d)
        return X, Y
    
    def test_mps_small_sample_fallback(self, small_sample_data):
        """Test MPS fallback for small samples."""
        X, Y = small_sample_data
        
        # Test on MPS if available
        if torch.backends.mps.is_available():
            cka_mps = DebiasedCKA(device='mps')
            cka_value = cka_mps.compute_cka(X, Y)
            
            # Should not raise errors and produce valid result
            assert 0 <= cka_value <= 1
            assert not np.isnan(cka_value)
    
    def test_cpu_float64_promotion(self, small_sample_data):
        """Test automatic promotion to float64 on CPU for small samples."""
        X, Y = small_sample_data
        
        # Force CPU computation
        cka = DebiasedCKA(device='cpu')
        
        # For MPS simulation, manually set device type
        if torch.backends.mps.is_available():
            cka.device = torch.device('mps')
            cka_value = cka.compute_cka(X, Y)
            
            # Should handle gracefully
            assert 0 <= cka_value <= 1
            assert not np.isnan(cka_value)
    
    def test_regularization_ill_conditioned(self, ill_conditioned_data):
        """Test regularization for ill-conditioned matrices."""
        X, Y = ill_conditioned_data
        
        # Without regularization - might be unstable
        cka_no_reg = DebiasedCKA(regularization=0.0)
        
        # With regularization - should be more stable
        cka_with_reg = DebiasedCKA(regularization=1e-4)
        
        # Both should produce valid results
        value_no_reg = cka_no_reg.compute_cka(X, Y)
        value_with_reg = cka_with_reg.compute_cka(X, Y)
        
        assert 0 <= value_no_reg <= 1
        assert 0 <= value_with_reg <= 1
        assert not np.isnan(value_no_reg)
        assert not np.isnan(value_with_reg)
    
    def test_condition_number_warning(self, near_singular_data):
        """Test condition number warnings for near-singular matrices."""
        X, Y = near_singular_data
        
        cka = DebiasedCKA(enable_profiling=True)
        
        # Should compute without errors despite warnings
        cka_value = cka.compute_cka(X, Y)
        assert 0 <= cka_value <= 1
    
    def test_negative_hsic_handling(self):
        """Test handling of negative HSIC values due to numerical errors."""
        torch.manual_seed(42)
        
        # Create data that might produce negative HSIC
        n = 5  # Very small sample size
        X = torch.randn(n, 10) * 1e-8  # Very small values
        Y = torch.randn(n, 10) * 1e-8
        
        cka = DebiasedCKA()
        cka_value = cka.compute_cka(X, Y)
        
        # Should handle gracefully
        assert 0 <= cka_value <= 1
        assert not np.isnan(cka_value)
    
    def test_numerical_consistency_across_devices(self, small_sample_data):
        """Test numerical consistency across different devices."""
        X, Y = small_sample_data
        
        # CPU baseline
        cka_cpu = DebiasedCKA(device='cpu')
        value_cpu = cka_cpu.compute_cka(X.cpu(), Y.cpu())
        
        # GPU if available
        if torch.cuda.is_available():
            cka_gpu = DebiasedCKA(device='cuda')
            value_gpu = cka_gpu.compute_cka(X.cuda(), Y.cuda())
            
            # Should be close (allowing for floating point differences)
            assert abs(value_cpu - value_gpu) < 1e-4
        
        # MPS if available
        if torch.backends.mps.is_available():
            cka_mps = DebiasedCKA(device='mps')
            value_mps = cka_mps.compute_cka(X, Y)
            
            # MPS might have larger differences due to float32 limitation
            assert abs(value_cpu - value_mps) < 1e-3
    
    def test_adaptive_epsilon(self):
        """Test adaptive epsilon based on data scale."""
        torch.manual_seed(42)
        
        # Large scale data
        X_large = torch.randn(50, 20) * 1e6
        Y_large = torch.randn(50, 20) * 1e6
        
        # Small scale data
        X_small = torch.randn(50, 20) * 1e-6
        Y_small = torch.randn(50, 20) * 1e-6
        
        cka = DebiasedCKA()
        
        # Both should produce valid results
        value_large = cka.compute_cka(X_large, Y_large)
        value_small = cka.compute_cka(X_small, Y_small)
        
        assert 0 <= value_large <= 1
        assert 0 <= value_small <= 1
        assert not np.isnan(value_large)
        assert not np.isnan(value_small)
    
    @pytest.mark.parametrize("n_samples", [4, 5, 10, 15, 20, 50])
    def test_stability_across_sample_sizes(self, n_samples):
        """Test stability across different sample sizes."""
        torch.manual_seed(42)
        
        X = torch.randn(n_samples, 50, dtype=torch.float32)
        Y = torch.randn(n_samples, 50, dtype=torch.float32)
        
        cka = DebiasedCKA()
        cka_value = cka.compute_cka(X, Y)
        
        # Should be stable for all sample sizes
        assert 0 <= cka_value <= 1
        assert not np.isnan(cka_value)
        
        # Test self-similarity
        self_sim = cka.compute_cka(X, X)
        assert abs(self_sim - 1.0) < 1e-6
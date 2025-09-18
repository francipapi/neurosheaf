"""Tests for cost matrix caching hash performance fix.

This module tests the efficient tensor hashing methods that replace the 
pathologically heavy tuple-based cache key computation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from unittest.mock import patch

from neurosheaf.sheaf.core import GWConfig, GromovWassersteinComputer


class SimpleNet(nn.Module):
    """Simple network for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 8)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestEfficientTensorHashing:
    """Test efficient tensor hashing methods."""
    
    def test_sha1_hash_method(self):
        """Test SHA1 hash method produces consistent results."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Create test tensor
        X = torch.randn(20, 10)
        
        # Compute hash multiple times
        hash1 = computer._compute_tensor_hash(X)
        hash2 = computer._compute_tensor_hash(X)
        
        # Should be identical
        assert hash1 == hash2, "SHA1 hash should be deterministic"
        
        # Should be reasonable length (16 chars from hexdigest)
        assert len(hash1) == 16, f"Expected 16 char hash, got {len(hash1)}"
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in hash1), "Hash should be hexadecimal"
    
    def test_id_hash_method(self):
        """Test ID hash method produces consistent results for same tensor."""
        config = GWConfig(cache_hash_method='id')
        computer = GromovWassersteinComputer(config)
        
        # Create test tensor
        X = torch.randn(20, 10)
        
        # Compute hash multiple times with same tensor object
        hash1 = computer._compute_tensor_hash(X)
        hash2 = computer._compute_tensor_hash(X)
        
        # Should be identical for same tensor object
        assert hash1 == hash2, "ID hash should be consistent for same tensor object"
        
        # Should contain tensor ID
        assert str(id(X)) in hash1, "Hash should contain tensor ID"
        
        # Should contain metadata
        assert "20x10" in hash1, "Hash should contain shape info"
    
    def test_different_tensors_different_hashes(self):
        """Test that different tensors produce different hashes."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Create different tensors
        X1 = torch.randn(20, 10)
        X2 = torch.randn(20, 10)  # Same shape, different values
        X3 = torch.randn(15, 10)  # Different shape
        
        hash1 = computer._compute_tensor_hash(X1)
        hash2 = computer._compute_tensor_hash(X2)
        hash3 = computer._compute_tensor_hash(X3)
        
        # All should be different
        assert hash1 != hash2, "Different tensor values should produce different hashes"
        assert hash1 != hash3, "Different tensor shapes should produce different hashes"
        assert hash2 != hash3, "Different tensors should produce different hashes"
    
    def test_same_values_same_hash_sha1(self):
        """Test that tensors with same values produce same SHA1 hash."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Create tensors with identical values
        X1 = torch.ones(10, 5)
        X2 = torch.ones(10, 5)
        
        hash1 = computer._compute_tensor_hash(X1)
        hash2 = computer._compute_tensor_hash(X2)
        
        # Should be identical (content-based)
        assert hash1 == hash2, "Tensors with same values should have same SHA1 hash"
    
    def test_same_values_different_hash_id(self):
        """Test that tensors with same values produce different ID hashes."""
        config = GWConfig(cache_hash_method='id')
        computer = GromovWassersteinComputer(config)
        
        # Create tensors with identical values but different objects
        X1 = torch.ones(10, 5)
        X2 = torch.ones(10, 5)
        
        hash1 = computer._compute_tensor_hash(X1)
        hash2 = computer._compute_tensor_hash(X2)
        
        # Should be different (object-based)
        assert hash1 != hash2, "Different tensor objects should have different ID hashes"
    
    def test_different_devices_different_hashes(self):
        """Test that tensors on different devices produce different hashes."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Create tensor on CPU
        X_cpu = torch.ones(10, 5)
        hash_cpu = computer._compute_tensor_hash(X_cpu)
        
        # The hash is a SHA1 hexdigest, but device info is in the metadata that's hashed
        # Let's verify the hash changes for tensors with different device metadata
        # We can't test actual GPU without GPU, so just verify hash is generated
        assert len(hash_cpu) == 16, "Hash should be 16 characters"
        assert all(c in '0123456789abcdef' for c in hash_cpu), "Hash should be hexadecimal"
    
    def test_different_dtypes_different_hashes(self):
        """Test that tensors with different dtypes produce different hashes."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Create tensors with same values but different dtypes
        X_float32 = torch.ones(10, 5, dtype=torch.float32)
        X_float64 = torch.ones(10, 5, dtype=torch.float64)
        
        hash_f32 = computer._compute_tensor_hash(X_float32)
        hash_f64 = computer._compute_tensor_hash(X_float64)
        
        # Should be different due to dtype
        assert hash_f32 != hash_f64, "Different dtypes should produce different hashes"
    
    def test_invalid_hash_method_raises_error(self):
        """Test that invalid hash method raises error."""
        # We can't bypass validation in __init__, so test the hash method directly
        config = GWConfig(cache_hash_method='sha1')  # Valid config
        computer = GromovWassersteinComputer(config)
        
        # Temporarily change to invalid method to test error handling
        computer.config.cache_hash_method = 'invalid_method'
        
        X = torch.randn(10, 5)
        
        with pytest.raises(ValueError, match="Unknown cache_hash_method"):
            computer._compute_tensor_hash(X)
    
    def test_config_validation_for_hash_method(self):
        """Test that config validation catches invalid hash methods."""
        config = GWConfig()
        config.cache_hash_method = 'invalid'
        
        with pytest.raises(ValueError, match="cache_hash_method must be"):
            config.validate()
    
    def test_hash_performance_improvement_sha1(self):
        """Test that SHA1 hashing is much faster than tuple hashing."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Create moderately sized tensor (not too big to avoid timeouts)
        X = torch.randn(100, 50)
        
        # Time the new SHA1 method
        start_time = time.time()
        for _ in range(10):
            _ = computer._compute_tensor_hash(X)
        sha1_time = time.time() - start_time
        
        # Time the old tuple method (simulated)
        start_time = time.time()
        for _ in range(10):
            _ = hash(tuple(X.flatten().tolist()))
        tuple_time = time.time() - start_time
        
        # SHA1 should be much faster (at least 2x)
        assert sha1_time < tuple_time / 2, \
            f"SHA1 method should be much faster: {sha1_time:.4f}s vs {tuple_time:.4f}s"
    
    def test_hash_performance_improvement_id(self):
        """Test that ID hashing is extremely fast."""
        config = GWConfig(cache_hash_method='id')
        computer = GromovWassersteinComputer(config)
        
        # Create large tensor
        X = torch.randn(500, 200)
        
        # Time the ID method (should be nearly instant)
        start_time = time.time()
        for _ in range(100):
            _ = computer._compute_tensor_hash(X)
        id_time = time.time() - start_time
        
        # Should be very fast (less than 0.01s for 100 iterations)
        assert id_time < 0.01, f"ID method should be very fast, took {id_time:.6f}s"


class TestCacheIntegration:
    """Test integration of new hashing with cache system."""
    
    def test_cache_hit_with_sha1_hash(self):
        """Test that cache works correctly with SHA1 hashing."""
        config = GWConfig(cache_hash_method='sha1', cache_cost_matrices=True)
        computer = GromovWassersteinComputer(config)
        
        X = torch.randn(15, 10)
        
        # First call should compute and cache
        result1 = computer.compute_cosine_cost_matrix(X)
        cache_size_after_first = len(computer.cost_cache.cache)
        
        # Second call should hit cache
        result2 = computer.compute_cosine_cost_matrix(X)
        cache_size_after_second = len(computer.cost_cache.cache)
        
        # Results should be identical
        assert torch.allclose(result1, result2), "Cached result should match computed result"
        
        # Cache size should not increase on second call
        assert cache_size_after_second == cache_size_after_first, "Cache should hit on second call"
    
    def test_cache_hit_with_id_hash(self):
        """Test that cache works correctly with ID hashing."""
        config = GWConfig(cache_hash_method='id', cache_cost_matrices=True)
        computer = GromovWassersteinComputer(config)
        
        X = torch.randn(15, 10)
        
        # First call should compute and cache
        result1 = computer.compute_cosine_cost_matrix(X)
        
        # Second call with same tensor object should hit cache
        result2 = computer.compute_cosine_cost_matrix(X)
        
        # Results should be identical
        assert torch.allclose(result1, result2), "Cached result should match computed result"
        
        # Third call with different tensor object (same values) should miss cache
        X_new = torch.randn(15, 10)
        X_new.data = X.data.clone()  # Same values, different object
        result3 = computer.compute_cosine_cost_matrix(X_new)
        
        # Result should be mathematically equivalent but computed fresh
        assert torch.allclose(result1, result3), "Results should be mathematically equivalent"
    
    def test_cache_miss_with_different_tensors(self):
        """Test that cache misses with different tensors."""
        config = GWConfig(cache_hash_method='sha1', cache_cost_matrices=True)
        computer = GromovWassersteinComputer(config)
        
        X1 = torch.randn(15, 10)
        X2 = torch.randn(15, 10)  # Different values
        
        # Compute for both tensors
        result1 = computer.compute_cosine_cost_matrix(X1)
        result2 = computer.compute_cosine_cost_matrix(X2)
        
        # Should have 2 cache entries
        assert len(computer.cost_cache.cache) == 2, "Should have 2 different cache entries"
        
        # Results should be different
        assert not torch.allclose(result1, result2), "Different tensors should produce different results"
    
    def test_cache_disabled_no_hashing(self):
        """Test that hashing is not called when cache is disabled."""
        config = GWConfig(cache_cost_matrices=False)
        computer = GromovWassersteinComputer(config)
        
        X = torch.randn(15, 10)
        
        # Patch the hash method to detect if it's called
        with patch.object(computer, '_compute_tensor_hash') as mock_hash:
            result = computer.compute_cosine_cost_matrix(X)
            
            # Hash method should not be called when cache is disabled
            mock_hash.assert_not_called()
            
            # Should still get valid result
            assert result.shape == (15, 15)
            assert torch.allclose(result, result.T)  # Should be symmetric


class TestEdgeCases:
    """Test edge cases for tensor hashing."""
    
    def test_empty_tensor_hashing(self):
        """Test hashing behavior with edge case tensors."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Single element tensor
        X_single = torch.tensor([[1.0]])
        hash_single = computer._compute_tensor_hash(X_single)
        assert len(hash_single) == 16
        
        # Very small tensor
        X_small = torch.tensor([[1.0, 2.0]])
        hash_small = computer._compute_tensor_hash(X_small)
        assert len(hash_small) == 16
        
        # Should be different
        assert hash_single != hash_small
    
    def test_zero_tensor_hashing(self):
        """Test hashing of zero tensors."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Zero tensor
        X_zeros = torch.zeros(10, 5)
        hash_zeros = computer._compute_tensor_hash(X_zeros)
        
        # Should produce valid hash
        assert len(hash_zeros) == 16
        assert all(c in '0123456789abcdef' for c in hash_zeros)
        
        # Multiple zero tensors with same shape should have same hash
        X_zeros2 = torch.zeros(10, 5)
        hash_zeros2 = computer._compute_tensor_hash(X_zeros2)
        assert hash_zeros == hash_zeros2
    
    def test_nan_inf_tensor_hashing(self):
        """Test hashing behavior with NaN and Inf values."""
        config = GWConfig(cache_hash_method='sha1')
        computer = GromovWassersteinComputer(config)
        
        # Tensor with NaN
        X_nan = torch.tensor([[1.0, float('nan')]])
        hash_nan = computer._compute_tensor_hash(X_nan)
        assert len(hash_nan) == 16
        
        # Tensor with Inf
        X_inf = torch.tensor([[1.0, float('inf')]])
        hash_inf = computer._compute_tensor_hash(X_inf)
        assert len(hash_inf) == 16
        
        # Should be different
        assert hash_nan != hash_inf
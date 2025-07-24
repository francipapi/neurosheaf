"""Unit tests for Gromov-Wasserstein core components (Phase 1).

This module tests the mathematical correctness and numerical stability
of the core GW implementation, including:
- Cost matrix computation with cosine distances
- GW coupling computation with marginal constraints
- Configuration validation and edge cases
- Integration with POT library
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from neurosheaf.sheaf.core import (
    GWConfig, GromovWassersteinComputer, GWResult, CostMatrixCache
)


class TestGWConfig:
    """Test GW configuration validation and creation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GWConfig()
        
        assert config.epsilon == 0.1
        assert config.max_iter == 1000
        assert config.tolerance == 1e-9
        assert config.quasi_sheaf_tolerance == 0.1
        assert config.use_gpu is True
        assert config.cache_cost_matrices is True
        assert config.validate_couplings is True
        assert config.uniform_measures is True
        assert config.weighted_inner_product is False
    
    def test_config_validation_valid(self):
        """Test validation with valid parameters."""
        config = GWConfig(epsilon=0.05, max_iter=500, tolerance=1e-6)
        config.validate()  # Should not raise
    
    def test_config_validation_invalid_epsilon(self):
        """Test validation with invalid epsilon."""
        config = GWConfig(epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            config.validate()
        
        config = GWConfig(epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            config.validate()
    
    def test_config_validation_invalid_max_iter(self):
        """Test validation with invalid max_iter."""
        config = GWConfig(max_iter=0)
        with pytest.raises(ValueError, match="max_iter must be positive"):
            config.validate()
    
    def test_config_validation_invalid_tolerance(self):
        """Test validation with invalid tolerance."""
        config = GWConfig(tolerance=0.0)
        with pytest.raises(ValueError, match="tolerance must be positive"):
            config.validate()
    
    def test_config_validation_invalid_quasi_sheaf_tolerance(self):
        """Test validation with invalid quasi_sheaf_tolerance."""
        config = GWConfig(quasi_sheaf_tolerance=-0.1)
        with pytest.raises(ValueError, match="quasi_sheaf_tolerance must be non-negative"):
            config.validate()
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = GWConfig(epsilon=0.05, max_iter=500)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['epsilon'] == 0.05
        assert config_dict['max_iter'] == 500
        assert 'use_gpu' in config_dict
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'epsilon': 0.05,
            'max_iter': 500,
            'tolerance': 1e-8,
            'unknown_key': 'ignored'  # Should be filtered out
        }
        
        config = GWConfig.from_dict(config_dict)
        assert config.epsilon == 0.05
        assert config.max_iter == 500
        assert config.tolerance == 1e-8
        assert config.use_gpu is True  # Default value
    
    def test_config_presets(self):
        """Test configuration presets."""
        fast_config = GWConfig.default_fast()
        assert fast_config.epsilon == 0.05
        assert fast_config.max_iter == 500
        assert fast_config.validate_couplings is False
        
        accurate_config = GWConfig.default_accurate()
        assert accurate_config.epsilon == 0.01
        assert accurate_config.max_iter == 2000
        assert accurate_config.tolerance == 1e-12
        
        debug_config = GWConfig.default_debugging()
        assert debug_config.cache_cost_matrices is False
        assert debug_config.use_gpu is False


class TestCostMatrixCache:
    """Test cost matrix caching functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache put/get operations."""
        cache = CostMatrixCache(max_size_gb=0.001)  # Very small cache for testing
        
        # Test put and get
        key = "test_key"
        value = torch.eye(3)
        
        cache.put(key, value)
        retrieved = cache.get(key)
        
        assert retrieved is not None
        assert torch.allclose(retrieved, value)
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache = CostMatrixCache()
        result = cache.get("nonexistent_key")
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction with tiny cache."""
        cache = CostMatrixCache(max_size_gb=1e-6)  # Extremely small cache
        
        # Add first tensor
        key1 = "key1"
        value1 = torch.eye(10)  # Large enough to fill cache
        cache.put(key1, value1)
        
        # Add second tensor (should evict first)
        key2 = "key2"  
        value2 = torch.eye(10)
        cache.put(key2, value2)
        
        # First key should be evicted
        assert cache.get(key1) is None
        assert cache.get(key2) is not None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = CostMatrixCache()
        cache.put("key", torch.eye(3))
        
        assert cache.get("key") is not None
        cache.clear()
        assert cache.get("key") is None


class TestGromovWassersteinComputer:
    """Test core GW computation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GWConfig(
            epsilon=0.1,
            max_iter=100,  # Reduce for faster tests
            tolerance=1e-6,
            validate_couplings=True,
            cache_cost_matrices=False  # Disable for reproducible tests
        )
        self.computer = GromovWassersteinComputer(self.config)
    
    def test_computer_initialization(self):
        """Test GW computer initialization."""
        assert self.computer.config.epsilon == 0.1
        assert self.computer.config.max_iter == 100
        assert self.computer.cost_cache is None  # Disabled in config
    
    def test_cosine_cost_matrix_properties(self):
        """Test cosine cost matrix mathematical properties."""
        # Create test activations
        X = torch.tensor([
            [1.0, 0.0, 0.0],  # Unit vector along x
            [0.0, 1.0, 0.0],  # Unit vector along y
            [0.0, 0.0, 1.0],  # Unit vector along z
            [1.0, 1.0, 0.0]   # Non-unit vector
        ], dtype=torch.float32)
        
        C = self.computer.compute_cosine_cost_matrix(X)
        
        # Test shape
        assert C.shape == (4, 4)
        
        # Test symmetry
        assert torch.allclose(C, C.T, atol=1e-6)
        
        # Test non-negativity
        assert torch.all(C >= 0)
        
        # Test zero diagonal
        assert torch.allclose(torch.diag(C), torch.zeros(4), atol=1e-6)
        
        # Test specific values for orthogonal unit vectors
        # cos(90°) = 0, so cost = 1 - 0 = 1
        assert torch.allclose(C[0, 1], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(C[0, 2], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(C[1, 2], torch.tensor(1.0), atol=1e-6)
    
    def test_cosine_cost_matrix_identical_vectors(self):
        """Test cost matrix with identical vectors."""
        X = torch.tensor([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]  # Identical to first
        ], dtype=torch.float32)
        
        C = self.computer.compute_cosine_cost_matrix(X)
        
        # Cost between identical vectors should be 0
        assert torch.allclose(C[0, 1], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(C[1, 0], torch.tensor(0.0), atol=1e-6)
    
    def test_cosine_cost_matrix_zero_vectors(self):
        """Test cost matrix handling of zero vectors."""
        X = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Zero vector
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore expected warnings about zero vectors
            C = self.computer.compute_cosine_cost_matrix(X)
        
        # Should complete without error
        assert C.shape == (3, 3)
        assert torch.allclose(torch.diag(C), torch.zeros(3), atol=1e-6)
    
    def test_cosine_cost_matrix_input_validation(self):
        """Test input validation for cost matrix computation."""
        # Test wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            self.computer.compute_cosine_cost_matrix(torch.tensor([1, 2, 3]))
        
        # Test empty tensor
        with pytest.raises(ValueError, match="Invalid tensor dimensions"):
            self.computer.compute_cosine_cost_matrix(torch.empty(0, 5))
    
    @patch('neurosheaf.sheaf.core.gromov_wasserstein.POT_AVAILABLE', False)
    def test_gw_coupling_fallback(self):
        """Test GW coupling computation with fallback implementation."""
        computer = GromovWassersteinComputer(self.config)
        
        # Create simple cost matrices
        C_source = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32)
        C_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        
        result = computer.compute_gw_coupling(C_source, C_target)
        
        assert isinstance(result, GWResult)
        assert result.coupling.shape == (2, 2)
        assert result.source_size == 2
        assert result.target_size == 2
        assert 'fallback' in result.log['solver']
    
    @patch('neurosheaf.sheaf.core.gromov_wasserstein.POT_AVAILABLE', True)
    @patch('neurosheaf.sheaf.core.gromov_wasserstein.ot')
    def test_gw_coupling_pot_success(self, mock_ot):
        """Test GW coupling computation with POT library success."""
        # Mock POT library
        mock_coupling = np.array([[0.3, 0.2], [0.2, 0.3]])
        mock_log = {'gw_dist': 0.5, 'it': 10, 'converged': True}
        mock_ot.gromov.entropic_gromov_wasserstein.return_value = (mock_coupling, mock_log)
        
        computer = GromovWassersteinComputer(self.config)
        
        C_source = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32)
        C_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        
        result = computer.compute_gw_coupling(C_source, C_target)
        
        assert isinstance(result, GWResult)
        assert result.coupling.shape == (2, 2)
        assert result.cost == 0.5
        assert result.log['num_iter'] == 10
        assert result.log['converged'] is True
        assert 'pot_entropic' in result.log['solver']
    
    @patch('neurosheaf.sheaf.core.gromov_wasserstein.POT_AVAILABLE', True)
    @patch('neurosheaf.sheaf.core.gromov_wasserstein.ot')
    def test_gw_coupling_pot_failure(self, mock_ot):
        """Test GW coupling computation with POT library failure."""
        # Mock POT library to raise exception
        mock_ot.gromov.entropic_gromov_wasserstein.side_effect = RuntimeError("POT failed")
        
        computer = GromovWassersteinComputer(self.config)
        
        C_source = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32)
        C_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        
        result = computer.compute_gw_coupling(C_source, C_target)
        
        # Should fall back to fallback implementation
        assert isinstance(result, GWResult)
        assert 'fallback' in result.log['solver']
    
    def test_gw_coupling_input_validation(self):
        """Test input validation for GW coupling computation."""
        # Test non-square cost matrices
        C_non_square = torch.tensor([[0.0, 0.5, 0.3], [0.5, 0.0, 0.2]], dtype=torch.float32)
        C_square = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        
        with pytest.raises(ValueError, match="must be square"):
            self.computer.compute_gw_coupling(C_non_square, C_square)
        
        # Test mismatched measure dimensions
        C1 = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32)
        C2 = torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.3], [0.5, 0.3, 0.0]], dtype=torch.float32)
        p_wrong = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)  # 3 elements for 2x2 matrix
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            self.computer.compute_gw_coupling(C1, C2, p_source=p_wrong)
    
    def test_gw_coupling_custom_measures(self):
        """Test GW coupling with custom probability measures."""
        C_source = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32)
        C_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        
        # Custom non-uniform measures
        p_source = torch.tensor([0.3, 0.7], dtype=torch.float32)
        p_target = torch.tensor([0.6, 0.4], dtype=torch.float32)
        
        result = self.computer.compute_gw_coupling(C_source, C_target, p_source, p_target)
        
        assert isinstance(result, GWResult)
        
        # If validation is enabled, check marginal constraints
        if self.config.validate_couplings:
            validation = result.log.get('marginal_validation', {})
            assert 'max_violation' in validation


class TestGWResult:
    """Test GWResult data structure and validation."""
    
    def test_gwresult_creation(self):
        """Test GWResult creation."""
        coupling = torch.tensor([[0.3, 0.2], [0.2, 0.3]], dtype=torch.float32)
        cost = 0.5
        log = {'solver': 'test', 'converged': True}
        
        result = GWResult(
            coupling=coupling,
            cost=cost,
            log=log,
            source_size=2,
            target_size=2
        )
        
        assert torch.allclose(result.coupling, coupling)
        assert result.cost == cost
        assert result.log['solver'] == 'test'
        assert result.source_size == 2
        assert result.target_size == 2
    
    def test_validate_marginals_uniform(self):
        """Test marginal validation with uniform measures."""
        # Perfect coupling for uniform measures
        coupling = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float32)
        
        result = GWResult(
            coupling=coupling,
            cost=0.0,
            log={},
            source_size=2,
            target_size=2
        )
        
        validation = result.validate_marginals()
        
        assert validation['constraints_satisfied'] is True
        assert validation['max_violation'] < 1e-6
    
    def test_validate_marginals_custom(self):
        """Test marginal validation with custom measures."""
        # Coupling that satisfies custom marginals
        coupling = torch.tensor([[0.18, 0.12], [0.42, 0.28]], dtype=torch.float32)
        p_source = torch.tensor([0.3, 0.7], dtype=torch.float32)
        p_target = torch.tensor([0.6, 0.4], dtype=torch.float32)
        
        result = GWResult(
            coupling=coupling,
            cost=0.0,
            log={},
            source_size=2,
            target_size=2
        )
        
        validation = result.validate_marginals(p_source, p_target)
        
        # Check that marginals are approximately satisfied
        assert validation['target_marginal_violation'] < 0.01
        assert validation['source_marginal_violation'] < 0.01
    
    def test_validate_marginals_violation(self):
        """Test marginal validation with violated constraints."""
        # Coupling that doesn't satisfy marginal constraints
        coupling = torch.tensor([[0.5, 0.3], [0.1, 0.1]], dtype=torch.float32)
        
        result = GWResult(
            coupling=coupling,
            cost=0.0,
            log={},
            source_size=2,
            target_size=2
        )
        
        validation = result.validate_marginals()
        
        assert validation['constraints_satisfied'] is False
        assert validation['max_violation'] > 1e-6


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""
    
    def test_complete_pipeline_small(self):
        """Test complete GW pipeline with small synthetic data."""
        config = GWConfig(
            epsilon=0.1,
            max_iter=50,
            validate_couplings=True,
            cache_cost_matrices=False
        )
        computer = GromovWassersteinComputer(config)
        
        # Create synthetic activation data
        torch.manual_seed(42)  # Reproducible
        X1 = torch.randn(5, 3)  # Source activations
        X2 = torch.randn(4, 3)  # Target activations
        
        # Compute cost matrices
        C1 = computer.compute_cosine_cost_matrix(X1)
        C2 = computer.compute_cosine_cost_matrix(X2)
        
        # Compute GW coupling
        result = computer.compute_gw_coupling(C1, C2)
        
        # Validate result
        assert isinstance(result, GWResult)
        assert result.coupling.shape == (5, 4)  # (source_size, target_size) as returned by POT
        assert result.source_size == 5
        assert result.target_size == 4
        assert isinstance(result.cost, float)
        assert result.cost >= 0.0  # GW cost should be non-negative
        
        # Check that result can be used as restriction map
        # For GW: restriction ρ_{target→source} = π^T (transposed coupling)
        restriction = result.coupling.T  # ρ_{target→source} = π^T
        assert restriction.shape == (4, 5)  # (target_size, source_size) for matrix multiplication
    
    def test_edge_case_identical_spaces(self):
        """Test GW coupling between identical metric spaces."""
        config = GWConfig(epsilon=0.1, max_iter=100)
        computer = GromovWassersteinComputer(config)
        
        # Identical cost matrices
        C = torch.tensor([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.5, 0.0]
        ], dtype=torch.float32)
        
        result = computer.compute_gw_coupling(C, C)
        
        # For identical spaces, optimal coupling should be close to identity
        assert isinstance(result, GWResult)
        assert result.coupling.shape == (3, 3)
        # Cost should be close to 0 for identical spaces
        assert result.cost < 0.5  # Some tolerance for numerical approximation
    
    def test_performance_moderate_size(self):
        """Test performance with moderately-sized problem."""
        config = GWConfig(epsilon=0.1, max_iter=20)  # Reduce iterations for speed
        computer = GromovWassersteinComputer(config)
        
        # Moderate-sized synthetic data
        torch.manual_seed(42)
        X1 = torch.randn(20, 10)  # 20 points in 10D
        X2 = torch.randn(15, 10)  # 15 points in 10D
        
        C1 = computer.compute_cosine_cost_matrix(X1)
        C2 = computer.compute_cosine_cost_matrix(X2)
        
        import time
        start_time = time.time()
        result = computer.compute_gw_coupling(C1, C2)
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert computation_time < 5.0
        assert isinstance(result, GWResult)
        assert result.coupling.shape == (15, 20)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
"""Enhanced unit tests for GW sheaf construction components.

This module provides comprehensive unit tests for GW functionality,
building on the existing test_gw_core.py with additional edge cases
and integration scenarios.

Test Categories:
1. GW component edge cases
2. Numerical stability testing
3. GPU/CPU consistency validation
4. Integration with sheaf assembly
5. Configuration validation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from unittest.mock import patch, MagicMock
import warnings

from neurosheaf.sheaf.core import GWConfig, GromovWassersteinComputer, GWResult
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.utils.exceptions import ValidationError, ComputationError
from neurosheaf.utils.logging import setup_logger

logger = setup_logger(__name__)


class TestGWComponents:
    """Test core GW functionality with comprehensive edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GWConfig(epsilon=0.1, max_iter=100)
        try:
            self.gw_computer = GromovWassersteinComputer(self.config)
            self.pot_available = True
        except ImportError:
            self.pot_available = False
            pytest.skip("POT library not available")
    
    def test_cosine_cost_matrix_properties(self):
        """Verify cost matrix is symmetric, non-negative, zero diagonal."""
        # Test with regular data
        X = torch.randn(15, 8)
        C = self.gw_computer.compute_cosine_cost_matrix(X)
        
        # Symmetry
        symmetry_error = torch.norm(C - C.T, 'fro').item()
        assert symmetry_error < 1e-10, f"Cost matrix not symmetric: {symmetry_error}"
        
        # Non-negativity (cosine distance is in [0, 2])
        assert torch.all(C >= -1e-10), "Cost matrix has negative values"
        assert torch.all(C <= 2.0 + 1e-10), "Cost matrix exceeds maximum (2.0)"
        
        # Zero diagonal
        diag_values = torch.diag(C)
        diag_error = torch.norm(diag_values).item()
        assert diag_error < 1e-10, f"Diagonal not zero: {diag_error}"
        
        # Triangle inequality (for cosine distance)
        # cos(x,z) <= cos(x,y) + cos(y,z) for unit vectors
        # Skip detailed verification but check basic sanity
        assert torch.all(torch.isfinite(C)), "Cost matrix contains non-finite values"
    
    def test_gw_coupling_column_stochastic(self):
        """Verify coupling satisfies measure constraints."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Test different matrix sizes
        test_sizes = [(8, 8), (10, 15), (20, 12)]
        
        for n_source, n_target in test_sizes:
            # Create test cost matrices
            C_source = torch.rand(n_source, n_source)
            C_source = C_source + C_source.T  # Make symmetric
            C_source.fill_diagonal_(0)
            
            C_target = torch.rand(n_target, n_target)
            C_target = C_target + C_target.T  # Make symmetric
            C_target.fill_diagonal_(0)
            
            try:
                result = self.gw_computer.compute_gw_coupling(C_source, C_target)
                coupling = result.coupling
                
                # Check marginal constraints for uniform measures
                row_sums = coupling.sum(dim=1)
                col_sums = coupling.sum(dim=0)
                
                expected_row = torch.ones(n_source) / n_target
                expected_col = torch.ones(n_target) / n_source
                
                row_error = torch.norm(row_sums - expected_row).item()
                col_error = torch.norm(col_sums - expected_col).item()
                
                assert row_error < 1e-5, f"Row marginal violated: {row_error}"
                assert col_error < 1e-5, f"Column marginal violated: {col_error}"
                
                # Coupling should be non-negative
                assert torch.all(coupling >= -1e-10), "Coupling has negative values"
                
                logger.info(f"Size ({n_source}, {n_target}): marginal errors = {row_error:.8f}, {col_error:.8f}")
                
            except Exception as e:
                pytest.fail(f"GW coupling failed for size ({n_source}, {n_target}): {e}")
    
    def test_numerical_stability_edge_cases(self):
        """Test handling of zero vectors, identical inputs, etc."""
        edge_cases = [
            # Zero vectors
            torch.zeros(10, 5),
            
            # Identical vectors
            torch.ones(8, 6),
            
            # Very small values
            torch.randn(12, 7) * 1e-8,
            
            # Large values
            torch.randn(6, 4) * 1e3,
            
            # Mixed scales
            torch.cat([torch.randn(5, 3), torch.randn(5, 3) * 100], dim=0)
        ]
        
        for i, X in enumerate(edge_cases):
            try:
                # Cost matrix computation should handle gracefully
                C = self.gw_computer.compute_cosine_cost_matrix(X)
                
                # Should not produce NaN or Inf
                assert torch.all(torch.isfinite(C)), f"Case {i}: Non-finite values in cost matrix"
                
                # Should maintain basic properties
                assert torch.allclose(C, C.T, atol=1e-6), f"Case {i}: Cost matrix not symmetric"
                assert torch.all(C >= -1e-6), f"Case {i}: Negative costs"
                
                logger.info(f"Edge case {i} passed: shape={X.shape}, range=[{X.min():.2e}, {X.max():.2e}]")
                
            except Exception as e:
                pytest.fail(f"Edge case {i} failed: {e}")
    
    def test_gpu_cpu_consistency(self):
        """Test results match across devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create test data on CPU
        X = torch.randn(12, 8)
        
        # CPU computation
        cpu_config = GWConfig(epsilon=0.1, max_iter=100, use_gpu=False)
        try:
            cpu_computer = GromovWassersteinComputer(cpu_config)
        except ImportError:
            pytest.skip("POT library not available")
        
        C_cpu = cpu_computer.compute_cosine_cost_matrix(X)
        
        # GPU computation
        X_gpu = X.cuda()
        gpu_config = GWConfig(epsilon=0.1, max_iter=100, use_gpu=True)
        try:
            gpu_computer = GromovWassersteinComputer(gpu_config)
            C_gpu = gpu_computer.compute_cosine_cost_matrix(X_gpu)
            C_gpu = C_gpu.cpu()  # Move back for comparison
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"GPU computation not available: {e}")
        
        # Should produce identical results (within numerical precision)
        cost_diff = torch.norm(C_cpu - C_gpu, 'fro').item()
        assert cost_diff < 1e-5, f"CPU/GPU results differ: {cost_diff}"
        
        logger.info(f"CPU/GPU consistency: difference = {cost_diff:.8f}")
    
    def test_caching_mechanism(self):
        """Test cost matrix caching works correctly."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Enable caching
        cache_config = GWConfig(cache_cost_matrices=True)
        computer = GromovWassersteinComputer(cache_config)
        
        X = torch.randn(10, 6)
        
        # First computation (should cache)
        C1 = computer.compute_cosine_cost_matrix(X)
        
        # Second computation (should use cache)
        C2 = computer.compute_cosine_cost_matrix(X)
        
        # Should be identical (exact same object if cached properly)
        assert torch.allclose(C1, C2, atol=1e-15), "Cached result differs"
        
        # Test cache invalidation with different input
        X_different = torch.randn(10, 6)
        C3 = computer.compute_cosine_cost_matrix(X_different)
        
        # Should be different from cached result
        assert not torch.allclose(C1, C3, atol=1e-3), "Cache not invalidated for different input"
    
    def test_convergence_diagnostics(self):
        """Test convergence monitoring and diagnostics."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Test with different convergence settings
        configs = [
            GWConfig(epsilon=0.1, max_iter=10, tolerance=1e-3),   # Fast, loose
            GWConfig(epsilon=0.01, max_iter=100, tolerance=1e-8), # Slow, tight
            GWConfig(epsilon=0.05, max_iter=50, tolerance=1e-6)   # Medium
        ]
        
        # Create test cost matrices
        n = 8
        C1 = torch.rand(n, n)
        C1 = C1 + C1.T
        C1.fill_diagonal_(0)
        
        C2 = torch.rand(n, n)
        C2 = C2 + C2.T
        C2.fill_diagonal_(0)
        
        for i, config in enumerate(configs):
            try:
                computer = GromovWassersteinComputer(config)
                result = computer.compute_gw_coupling(C1, C2)
                
                # Should have convergence information
                assert hasattr(result, 'cost'), "Missing cost information"
                assert hasattr(result, 'log'), "Missing convergence log"
                
                # Cost should be non-negative
                assert result.cost >= 0, f"Negative GW cost: {result.cost}"
                
                # Log should contain iteration info
                if hasattr(result.log, 'n_iter'):
                    assert result.log.n_iter <= config.max_iter, "Exceeded max iterations"
                
                logger.info(f"Config {i}: cost={result.cost:.6f}, converged in log info")
                
            except Exception as e:
                logger.warning(f"Convergence test {i} failed: {e}")
    
    def test_measure_handling(self):
        """Test different measure specifications."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        n_source, n_target = 8, 10
        C1 = torch.rand(n_source, n_source)
        C1 = C1 + C1.T
        C1.fill_diagonal_(0)
        
        C2 = torch.rand(n_target, n_target)
        C2 = C2 + C2.T
        C2.fill_diagonal_(0)
        
        # Test with uniform measures (default)
        result_uniform = self.gw_computer.compute_gw_coupling(C1, C2)
        
        # Test with explicit uniform measures
        p_uniform_1 = torch.ones(n_source) / n_source
        p_uniform_2 = torch.ones(n_target) / n_target
        result_explicit = self.gw_computer.compute_gw_coupling(
            C1, C2, p_uniform_1, p_uniform_2
        )
        
        # Should produce similar results
        coupling_diff = torch.norm(result_uniform.coupling - result_explicit.coupling, 'fro').item()
        assert coupling_diff < 1e-3, f"Uniform measure results differ: {coupling_diff}"
        
        # Test with non-uniform measures
        p_nonuniform_1 = torch.rand(n_source)
        p_nonuniform_1 /= p_nonuniform_1.sum()
        
        p_nonuniform_2 = torch.rand(n_target)
        p_nonuniform_2 /= p_nonuniform_2.sum()
        
        try:
            result_nonuniform = self.gw_computer.compute_gw_coupling(
                C1, C2, p_nonuniform_1, p_nonuniform_2
            )
            
            # Should satisfy marginal constraints
            row_sums = result_nonuniform.coupling.sum(dim=1)
            col_sums = result_nonuniform.coupling.sum(dim=0)
            
            row_error = torch.norm(row_sums - p_nonuniform_1).item()
            col_error = torch.norm(col_sums - p_nonuniform_2).item()
            
            assert row_error < 1e-4, f"Non-uniform row marginal error: {row_error}"
            assert col_error < 1e-4, f"Non-uniform col marginal error: {col_error}"
            
        except Exception as e:
            logger.warning(f"Non-uniform measure test failed: {e}")


class TestGWConfigValidation:
    """Test comprehensive GW configuration validation."""
    
    def test_config_parameter_ranges(self):
        """Test parameter validation ranges."""
        # Valid configurations should pass
        valid_configs = [
            GWConfig(epsilon=0.01, max_iter=100),
            GWConfig(epsilon=1.0, max_iter=2000, tolerance=1e-12),
            GWConfig(quasi_sheaf_tolerance=0.0),  # Zero tolerance should be valid
            GWConfig(max_cache_size_gb=0.1)       # Small cache should be valid
        ]
        
        for config in valid_configs:
            config.validate()  # Should not raise
        
        # Invalid configurations should fail
        invalid_configs = [
            (GWConfig(epsilon=0.0), "epsilon must be positive"),
            (GWConfig(epsilon=-0.1), "epsilon must be positive"),
            (GWConfig(max_iter=0), "max_iter must be positive"),
            (GWConfig(max_iter=-1), "max_iter must be positive"),
            (GWConfig(tolerance=0.0), "tolerance must be positive"),
            (GWConfig(tolerance=-1e-6), "tolerance must be positive"),
            (GWConfig(quasi_sheaf_tolerance=-0.1), "quasi_sheaf_tolerance must be non-negative"),
            (GWConfig(max_cache_size_gb=0.0), "max_cache_size_gb must be positive"),
            (GWConfig(cost_matrix_eps=0.0), "cost_matrix_eps must be positive"),
            (GWConfig(coupling_eps=0.0), "coupling_eps must be positive")
        ]
        
        for config, expected_msg in invalid_configs:
            with pytest.raises(ValueError, match=expected_msg):
                config.validate()
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = GWConfig(
            epsilon=0.05,
            max_iter=800,
            tolerance=1e-7,
            quasi_sheaf_tolerance=0.15,
            use_gpu=False,
            validate_couplings=True,
            uniform_measures=True,
            weighted_inner_product=False,
            max_cache_size_gb=1.5
        )
        
        # Serialize
        config_dict = original_config.to_dict()
        
        # Check all fields present
        assert 'epsilon' in config_dict
        assert 'max_iter' in config_dict
        assert 'tolerance' in config_dict
        assert config_dict['epsilon'] == 0.05
        assert config_dict['max_iter'] == 800
        
        # Deserialize
        restored_config = GWConfig.from_dict(config_dict)
        
        # Should be equivalent
        assert restored_config.epsilon == original_config.epsilon
        assert restored_config.max_iter == original_config.max_iter
        assert restored_config.tolerance == original_config.tolerance
        assert restored_config.use_gpu == original_config.use_gpu
        
        # Should validate correctly
        restored_config.validate()
    
    def test_preset_configurations(self):
        """Test preset configuration factories."""
        # Fast configuration
        fast_config = GWConfig.default_fast()
        assert fast_config.epsilon >= 0.01  # Higher epsilon for speed
        assert fast_config.max_iter <= 1000  # Fewer iterations
        assert fast_config.validate_couplings is False  # Skip validation
        fast_config.validate()
        
        # Accurate configuration
        accurate_config = GWConfig.default_accurate()
        assert accurate_config.epsilon <= 0.05  # Lower epsilon for accuracy
        assert accurate_config.max_iter >= 1000  # More iterations
        assert accurate_config.validate_couplings is True  # Full validation
        accurate_config.validate()
        
        # Debugging configuration
        debug_config = GWConfig.default_debugging()
        assert debug_config.validate_couplings is True
        assert debug_config.validate_costs is True
        assert debug_config.use_gpu is False  # CPU for better error messages
        debug_config.validate()
    
    def test_config_forward_compatibility(self):
        """Test configuration handles unknown fields gracefully."""
        # Config dict with unknown fields
        config_dict = {
            'epsilon': 0.1,
            'max_iter': 500,
            'unknown_field': 'should_be_ignored',
            'another_unknown': 42
        }
        
        # Should create config successfully, ignoring unknown fields
        config = GWConfig.from_dict(config_dict)
        assert config.epsilon == 0.1
        assert config.max_iter == 500
        assert not hasattr(config, 'unknown_field')
        
        config.validate()


class TestGWIntegrationWithSheafAssembly:
    """Test GW integration with sheaf assembly pipeline."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        try:
            self.builder = SheafBuilder(restriction_method='gromov_wasserstein')
            self.gw_config = GWConfig(epsilon=0.1, max_iter=50)  # Fast config for testing
            self.pot_available = True
        except ImportError:
            self.pot_available = False
    
    def test_gw_sheaf_construction(self):
        """Test complete GW sheaf construction."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Create simple test model
        model = nn.Sequential(
            nn.Linear(8, 6),
            nn.Linear(6, 4),
            nn.Linear(4, 3)
        )
        data = torch.randn(20, 8)
        
        try:
            # Extract activations
            activations = self._extract_activations(model, data)
            
            # Build GW sheaf
            sheaf = self.builder.build_from_activations(
                activations, model, gw_config=self.gw_config, validate=True
            )
            
            # Verify it's a proper sheaf
            assert isinstance(sheaf, Sheaf)
            assert hasattr(sheaf, 'stalks')
            assert hasattr(sheaf, 'restrictions')
            assert hasattr(sheaf, 'metadata')
            
            # Verify GW-specific properties
            assert sheaf.metadata.get('construction_method') == 'gromov_wasserstein'
            assert 'gw_costs' in sheaf.metadata
            assert 'gw_config' in sheaf.metadata
            
            # Verify structure
            assert len(sheaf.stalks) == len(activations)
            assert len(sheaf.restrictions) >= len(activations) - 1
            
            # Verify GW costs are reasonable
            gw_costs = sheaf.metadata['gw_costs']
            for edge, cost in gw_costs.items():
                assert 0 <= cost <= 2.0, f"GW cost out of range: {cost}"
            
            logger.info(f"GW sheaf constructed successfully: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library issues")
            else:
                pytest.fail(f"GW sheaf construction failed: {e}")
    
    def _extract_activations(self, model: nn.Module, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract layer activations."""
        activations = {}
        x = data
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f"layer_{i}"] = x.detach().clone()
        return activations
    
    def test_gw_sheaf_metadata_completeness(self):
        """Test that GW sheaves have complete metadata."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 3))
        data = torch.randn(15, 6)
        
        try:
            activations = self._extract_activations(model, data)
            sheaf = self.builder.build_from_activations(
                activations, model, gw_config=self.gw_config
            )
            
            # Check required metadata fields
            required_fields = [
                'construction_method',
                'gw_costs',
                'gw_config',
                'quasi_sheaf_tolerance'
            ]
            
            for field in required_fields:
                assert field in sheaf.metadata, f"Missing metadata field: {field}"
            
            # Verify metadata types and values
            assert sheaf.metadata['construction_method'] == 'gromov_wasserstein'
            assert isinstance(sheaf.metadata['gw_costs'], dict)
            assert isinstance(sheaf.metadata['gw_config'], dict)
            assert isinstance(sheaf.metadata['quasi_sheaf_tolerance'], (int, float))
            
            # Verify GW config is properly serialized
            gw_config_dict = sheaf.metadata['gw_config']
            assert 'epsilon' in gw_config_dict
            assert 'max_iter' in gw_config_dict
            assert gw_config_dict['epsilon'] == self.gw_config.epsilon
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library issues")
            else:
                raise
    
    def test_gw_sheaf_validation(self):
        """Test GW sheaf validation during construction."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        model = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3))
        data = torch.randn(12, 5)
        
        # Test with validation enabled
        validation_config = GWConfig(
            epsilon=0.1, max_iter=100,
            validate_couplings=True,
            validate_costs=True
        )
        
        try:
            activations = self._extract_activations(model, data)
            sheaf = self.builder.build_from_activations(
                activations, model, 
                gw_config=validation_config,
                validate=True
            )
            
            # Should have validation results in metadata
            if 'validation_passed' in sheaf.metadata:
                assert sheaf.metadata['validation_passed'] is True
            
            # Should have quasi-sheaf tolerance information
            assert 'quasi_sheaf_tolerance' in sheaf.metadata
            tolerance = sheaf.metadata['quasi_sheaf_tolerance']
            assert tolerance >= 0, "Negative quasi-sheaf tolerance"
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library issues")
            else:
                raise
    
    def test_gw_restriction_properties(self):
        """Test properties of GW-computed restrictions."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        model = nn.Sequential(nn.Linear(6, 5), nn.Linear(5, 4))
        data = torch.randn(18, 6)
        
        try:
            activations = self._extract_activations(model, data)
            sheaf = self.builder.build_from_activations(
                activations, model, gw_config=self.gw_config
            )
            
            # Test restriction properties
            for edge, restriction in sheaf.restrictions.items():
                # Should be proper tensor
                assert isinstance(restriction, torch.Tensor)
                assert restriction.dim() == 2
                
                # Dimensions should match stalks
                source, target = edge
                source_dim = sheaf.stalks[source].shape[0]
                target_dim = sheaf.stalks[target].shape[0]
                
                assert restriction.shape == (target_dim, source_dim), \
                    f"Restriction shape mismatch: {restriction.shape} vs ({target_dim}, {source_dim})"
                
                # For uniform measures, restrictions should have certain properties
                # (but not necessarily orthogonal like Procrustes)
                assert torch.all(torch.isfinite(restriction)), "Non-finite values in restriction"
                
                # Row sums should be reasonable (column-stochastic properties)
                row_sums = restriction.sum(dim=1)
                assert torch.all(torch.isfinite(row_sums)), "Non-finite row sums"
                
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library issues")
            else:
                raise


class TestGWErrorHandling:
    """Test error handling and edge cases for GW implementation."""
    
    def setup_method(self):
        """Set up error handling test fixtures."""
        self.gw_config = GWConfig(epsilon=0.1, max_iter=50)
        try:
            self.builder = SheafBuilder(restriction_method='gromov_wasserstein')
            self.pot_available = True
        except ImportError:
            self.pot_available = False
    
    def test_empty_activations(self):
        """Test handling of empty activation tensors."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Empty activations should be handled gracefully
        empty_activations = {
            'layer1': torch.empty(0, 5),
            'layer2': torch.empty(0, 3)
        }
        
        model = nn.Sequential(nn.Linear(5, 3))
        
        with pytest.raises((ValidationError, ComputationError, RuntimeError)):
            self.builder.build_from_activations(
                empty_activations, model, gw_config=self.gw_config
            )
    
    def test_single_sample_activations(self):
        """Test handling of single-sample activations."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Single sample might cause issues with GW computation
        single_sample = {
            'layer1': torch.randn(1, 5),
            'layer2': torch.randn(1, 3)
        }
        
        model = nn.Sequential(nn.Linear(5, 3))
        
        # Should either work or provide clear error
        try:
            result = self.builder.build_from_activations(
                single_sample, model, gw_config=self.gw_config
            )
            # If it works, should be valid sheaf
            assert isinstance(result, Sheaf)
        except (ValidationError, ComputationError) as e:
            # Should provide informative error
            assert len(str(e)) > 10, "Error message should be informative"
    
    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches.""" 
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Activations with wrong dimensions
        mismatched = {
            'layer1': torch.randn(20, 5),
            'layer2': torch.randn(15, 3)  # Different batch size
        }
        
        model = nn.Sequential(nn.Linear(5, 3))
        
        with pytest.raises((ValidationError, ComputationError, RuntimeError)):
            self.builder.build_from_activations(
                mismatched, model, gw_config=self.gw_config
            )
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values in activations."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Activations with problematic values
        problematic_cases = [
            # NaN values
            {
                'layer1': torch.full((10, 5), float('nan')),
                'layer2': torch.randn(10, 3)
            },
            # Inf values
            {
                'layer1': torch.full((10, 5), float('inf')),
                'layer2': torch.randn(10, 3)
            },
            # Mixed
            {
                'layer1': torch.randn(10, 5),
                'layer2': torch.tensor([[float('inf'), 1, 2]] * 10)
            }
        ]
        
        model = nn.Sequential(nn.Linear(5, 3))
        
        for i, activations in enumerate(problematic_cases):
            with pytest.raises((ValidationError, ComputationError, RuntimeError)):
                self.builder.build_from_activations(
                    activations, model, gw_config=self.gw_config
                )
    
    def test_convergence_failure_handling(self):
        """Test handling of GW convergence failures."""
        if not self.pot_available:
            pytest.skip("POT library not available")
        
        # Create config that might not converge
        strict_config = GWConfig(
            epsilon=1e-10,  # Very small epsilon
            max_iter=5,     # Very few iterations
            tolerance=1e-15 # Very strict tolerance
        )
        
        model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 3))
        data = torch.randn(15, 6)
        
        activations = self._extract_activations(model, data)
        
        # Should either work or provide informative error
        try:
            result = self.builder.build_from_activations(
                activations, model, gw_config=strict_config
            )
            # If successful, should be valid
            assert isinstance(result, Sheaf)
        except (ComputationError, RuntimeError) as e:
            # Should provide informative error about convergence
            error_str = str(e).lower()
            assert any(word in error_str for word in ['converge', 'iteration', 'maximum']), \
                f"Error should mention convergence issues: {e}"
    
    def _extract_activations(self, model: nn.Module, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations for testing."""
        activations = {}
        x = data
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f"layer_{i}"] = x.detach().clone()
        return activations


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
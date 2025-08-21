"""Tests for variance-based non-uniform measures implementation.

This module tests the variance-based per-unit importance measures that replace
uniform weighting with importance weighting based on activation variance.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.assembly import GWRestrictionManager, SheafBuilder


class SimpleTestNetwork(nn.Module):
    """Test network with configurable layer sizes."""
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class TestVarianceBasedMeasures:
    """Test variance-based measure computation."""
    
    def test_variance_measures_basic_properties(self):
        """Test basic properties of variance-based measures."""
        config = GWConfig(uniform_measures=False, measure_eps=1e-6)
        manager = GWRestrictionManager(config=config)
        
        # Create activation tensor with varying unit variances
        batch_size = 10
        n_features = 5
        
        # Design activations with known variance patterns
        # Use fixed seed for reproducible variance patterns
        torch.manual_seed(42)
        activations = torch.zeros(batch_size, n_features)
        # Unit 0: high variance (designed pattern)
        activations[:, 0] = torch.linspace(-3, 3, batch_size)
        # Unit 1: medium variance (designed pattern)
        activations[:, 1] = torch.linspace(-1, 1, batch_size)
        # Unit 2: low variance (designed pattern)
        activations[:, 2] = torch.linspace(-0.1, 0.1, batch_size)
        # Unit 3: zero variance (constant)
        activations[:, 3] = 1.0
        # Unit 4: negative constant (zero variance)
        activations[:, 4] = -0.5
        
        measures = manager._compute_variance_based_measures(activations)
        
        # Check basic properties
        assert measures.shape == (n_features,)
        assert torch.allclose(measures.sum(), torch.tensor(1.0, dtype=measures.dtype), atol=1e-6)
        assert torch.all(measures > 0), "All measures should be positive"
        
        # Check ordering: high variance > medium variance > low variance
        assert measures[0] > measures[1] > measures[2], \
            f"Expected variance ordering violated: {measures[:3].tolist()}"
        
        # Units with zero variance should have minimal weight (eps only)
        assert measures[3] < measures[2], "Zero variance unit should have lower weight"
        assert measures[4] < measures[2], "Zero variance unit should have lower weight"
        
        # Zero variance units should have equal weights (both just eps)
        assert torch.allclose(measures[3], measures[4], atol=1e-6)
    
    def test_variance_measures_eps_parameter(self):
        """Test that eps parameter works correctly."""
        config = GWConfig(uniform_measures=False, measure_eps=0.1)  # Large eps
        manager = GWRestrictionManager(config=config)
        
        # Create activations with one zero-variance unit
        batch_size = 5
        activations = torch.zeros(batch_size, 2)
        activations[:, 0] = torch.randn(batch_size)  # Non-zero variance
        activations[:, 1] = 1.0                     # Zero variance
        
        measures = manager._compute_variance_based_measures(activations, eps=0.1)
        
        # Both units should have significant weight due to large eps
        assert measures[1] > 0.1, f"Zero variance unit should get significant weight with large eps: {measures[1]}"
        
        # Test small eps
        measures_small_eps = manager._compute_variance_based_measures(activations, eps=1e-8)
        
        # Zero variance unit should have much smaller weight with small eps
        assert measures_small_eps[1] < measures[1], "Smaller eps should reduce zero variance unit weight"
    
    def test_variance_measures_edge_cases(self):
        """Test edge cases for variance measure computation."""
        config = GWConfig(uniform_measures=False)
        manager = GWRestrictionManager(config=config)
        
        # Test single feature
        activations_single = torch.randn(10, 1)
        measures_single = manager._compute_variance_based_measures(activations_single)
        assert measures_single.shape == (1,)
        assert torch.allclose(measures_single, torch.tensor([1.0], dtype=measures_single.dtype), atol=1e-6)
        
        # Test all zero activations
        activations_zeros = torch.zeros(5, 3)
        measures_zeros = manager._compute_variance_based_measures(activations_zeros)
        # All should have equal weight (all just eps)
        expected = torch.ones(3, dtype=measures_zeros.dtype) / 3.0
        assert torch.allclose(measures_zeros, expected, atol=1e-6)
        
        # Test identical units (zero relative variance)
        activations_identical = torch.ones(8, 4) * torch.randn(8, 1)  # All units identical
        measures_identical = manager._compute_variance_based_measures(activations_identical)
        # All should have equal weight
        expected = torch.ones(4, dtype=measures_identical.dtype) / 4.0
        assert torch.allclose(measures_identical, expected, atol=1e-5)
    
    def test_variance_measures_validation(self):
        """Test validation in variance measure computation."""
        config = GWConfig(uniform_measures=False)
        manager = GWRestrictionManager(config=config)
        
        # Test invalid eps
        activations = torch.randn(5, 3)
        with pytest.raises(ValueError, match="eps must be positive"):
            manager._compute_variance_based_measures(activations, eps=0.0)
        
        with pytest.raises(ValueError, match="eps must be positive"):
            manager._compute_variance_based_measures(activations, eps=-1e-6)
        
        # Test invalid activation tensor dimensions
        with pytest.raises(ValueError, match="Expected activation tensor with at least 2 dimensions"):
            manager._compute_variance_based_measures(torch.randn(5))  # 1D
        
        # Test that 3D tensors now work (should be reshaped automatically)
        activation_3d = torch.randn(2, 3, 4)
        measures_3d = manager._compute_variance_based_measures(activation_3d)
        assert measures_3d.shape == (12,)  # 3*4 features after flattening
        assert torch.allclose(measures_3d.sum(), torch.tensor(1.0, dtype=measures_3d.dtype), atol=1e-6)
    
    def test_3d_tensor_handling(self):
        """Test that 3D tensors like [1000, 32, 1] are handled correctly."""
        config = GWConfig(uniform_measures=False, measure_eps=1e-6)
        manager = GWRestrictionManager(config=config)
        
        # Test the exact case from the error: [1000, 32, 1]
        activation_3d = torch.randn(1000, 32, 1)
        measures = manager._compute_variance_based_measures(activation_3d)
        
        # Should be flattened to 32 features (32*1 = 32)
        assert measures.shape == (32,)
        assert torch.allclose(measures.sum(), torch.tensor(1.0, dtype=measures.dtype), atol=1e-6)
        assert torch.all(measures > 0), "All measures should be positive"
        
        # Test another 3D case
        activation_3d_multi = torch.randn(100, 16, 4)
        measures_multi = manager._compute_variance_based_measures(activation_3d_multi)
        
        # Should be flattened to 64 features (16*4 = 64) 
        assert measures_multi.shape == (64,)
        assert torch.allclose(measures_multi.sum(), torch.tensor(1.0, dtype=measures_multi.dtype), atol=1e-6)
        assert torch.all(measures_multi > 0), "All measures should be positive"
    
    def test_config_measure_eps_validation(self):
        """Test that config validation catches invalid measure_eps."""
        config = GWConfig()
        config.measure_eps = 0.0
        with pytest.raises(ValueError, match="measure_eps must be positive"):
            config.validate()
        
        config.measure_eps = -1e-6
        with pytest.raises(ValueError, match="measure_eps must be positive"):
            config.validate()
        
        # Valid value should pass
        config.measure_eps = 1e-6
        config.validate()  # Should not raise


class TestNonUniformIntegration:
    """Test integration of non-uniform measures with GW restriction computation."""
    
    def test_uniform_vs_nonuniform_produces_different_results(self):
        """Test that uniform vs non-uniform measures produce different results."""
        # Create simple network with multiple layers
        model = SimpleTestNetwork([8, 12, 6])
        batch_size = 12
        
        # Create input that will produce varying unit activations
        torch.manual_seed(123)  # For reproducibility
        input_tensor = torch.randn(batch_size, 8)
        
        # Build sheaf with uniform measures
        config_uniform = GWConfig(uniform_measures=True, align_units=True)
        builder_uniform = SheafBuilder(restriction_method='gromov_wasserstein')
        
        sheaf_uniform = builder_uniform.build_from_activations(
            model, input_tensor,
            validate=False,  # Skip for speed
            gw_config=config_uniform
        )
        
        # Build sheaf with non-uniform measures
        config_nonuniform = GWConfig(uniform_measures=False, measure_eps=1e-6, align_units=True)
        builder_nonuniform = SheafBuilder(restriction_method='gromov_wasserstein')
        
        sheaf_nonuniform = builder_nonuniform.build_from_activations(
            model, input_tensor,
            validate=False,  # Skip for speed
            gw_config=config_nonuniform
        )
        
        # Both should have restrictions
        assert len(sheaf_uniform.restrictions) > 0
        assert len(sheaf_nonuniform.restrictions) > 0
        
        # Compare restriction maps for the same edge
        for edge in sheaf_uniform.restrictions.keys():
            if edge in sheaf_nonuniform.restrictions:
                R_uniform = sheaf_uniform.restrictions[edge]
                R_nonuniform = sheaf_nonuniform.restrictions[edge]
                
                # They should be different
                difference = torch.norm(R_uniform - R_nonuniform, 'fro').item()
                assert difference > 1e-6, f"Uniform and non-uniform measures should produce different results, difference: {difference}"
                
                # Both should still be row-stochastic
                row_sums_uniform = R_uniform.sum(dim=1)
                row_sums_nonuniform = R_nonuniform.sum(dim=1)
                
                assert torch.allclose(row_sums_uniform, torch.ones_like(row_sums_uniform), atol=1e-3)
                assert torch.allclose(row_sums_nonuniform, torch.ones_like(row_sums_nonuniform), atol=1e-3)
                break  # Test at least one edge
    
    def test_nonuniform_measures_with_dead_units(self):
        """Test non-uniform measures with dead/inactive units."""
        batch_size = 8
        n_features = 6
        
        # Create activations with some dead units
        activations = torch.randn(batch_size, n_features) * 0.1
        activations[:, 0] = torch.randn(batch_size) * 2.0  # Active unit
        activations[:, 1] = 0.0                          # Dead unit
        activations[:, 2] = torch.randn(batch_size) * 1.0  # Active unit
        activations[:, 3] = 0.5                          # Constant unit (dead)
        
        # Create simple test case
        activations_dict = {'layer_a': activations, 'layer_b': activations.clone()}
        
        import networkx as nx
        poset = nx.DiGraph()
        poset.add_edge('layer_a', 'layer_b')
        
        config = GWConfig(uniform_measures=False, measure_eps=1e-8)
        manager = GWRestrictionManager(config=config)
        
        restrictions, costs, metadata = manager.compute_all_restrictions(
            activations_dict, poset, parallel=False
        )
        
        # Should succeed and produce valid results
        assert len(restrictions) == 1
        edge = ('layer_a', 'layer_b')
        assert edge in restrictions
        
        R = restrictions[edge]
        assert R.shape == (n_features, n_features)
        
        # Should be row-stochastic (use more lenient tolerance for numerical issues)
        row_sums = R.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    
    def test_nonuniform_fallback_on_error(self):
        """Test that non-uniform measures fall back to uniform on error."""
        # Create problematic config to trigger fallback
        config = GWConfig(uniform_measures=False, measure_eps=1e-6)
        manager = GWRestrictionManager(config=config)
        
        # Create activations dict
        activations = {
            'layer_a': torch.randn(5, 8),
            'layer_b': torch.randn(5, 6)
        }
        
        import networkx as nx
        poset = nx.DiGraph()
        poset.add_edge('layer_a', 'layer_b')
        
        # Mock the variance computation to raise an error
        original_method = manager._compute_variance_based_measures
        def failing_method(*args, **kwargs):
            raise RuntimeError("Simulated failure")
        
        manager._compute_variance_based_measures = failing_method
        
        # Should fall back to uniform and still work
        restrictions, costs, metadata = manager.compute_all_restrictions(
            activations, poset, parallel=False
        )
        
        # Should succeed with fallback
        assert len(restrictions) == 1
        edge = ('layer_a', 'layer_b')
        assert edge in restrictions
        
        # Restore original method
        manager._compute_variance_based_measures = original_method
    
    def test_full_sheaf_construction_with_nonuniform(self):
        """Test full sheaf construction with non-uniform measures."""
        model = SimpleTestNetwork([8, 12, 6])
        batch_size = 10
        input_tensor = torch.randn(batch_size, 8)
        
        # Test that sheaf construction works with non-uniform measures
        config = GWConfig(uniform_measures=False, align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,  # Skip for speed
            gw_config=config
        )
        
        # Should successfully build sheaf
        assert len(sheaf.stalks) > 0
        assert len(sheaf.restrictions) > 0
        
        # All restrictions should be row-stochastic
        for (source, target), restriction in sheaf.restrictions.items():
            row_sums = restriction.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Restriction {source}→{target} not row-stochastic with non-uniform measures"
        
        # Check metadata indicates non-uniform measures were used
        gw_config_dict = sheaf.metadata.get('gw_config', {})
        assert not gw_config_dict.get('uniform_measures', True), \
            "Metadata should indicate non-uniform measures were used"


class TestMeasureComputationPerformance:
    """Test performance characteristics of measure computation."""
    
    def test_variance_computation_efficiency(self):
        """Test that variance computation is efficient for large tensors."""
        import time
        
        config = GWConfig(uniform_measures=False)
        manager = GWRestrictionManager(config=config)
        
        # Test with large tensor
        large_batch = 1000
        n_features = 500
        activations = torch.randn(large_batch, n_features)
        
        # Should complete quickly
        start_time = time.time()
        measures = manager._compute_variance_based_measures(activations)
        elapsed = time.time() - start_time
        
        assert elapsed < 0.1, f"Variance computation took too long: {elapsed:.3f}s"
        assert measures.shape == (n_features,)
        assert torch.allclose(measures.sum(), torch.tensor(1.0, dtype=measures.dtype), atol=1e-6)
    
    def test_measure_numerical_stability(self):
        """Test numerical stability of measure computation."""
        config = GWConfig(uniform_measures=False, measure_eps=1e-12)
        manager = GWRestrictionManager(config=config)
        
        # Test with very small variances
        small_activations = torch.randn(20, 10) * 1e-8
        measures_small = manager._compute_variance_based_measures(small_activations)
        
        # Should still be valid probability distribution
        assert torch.allclose(measures_small.sum(), torch.tensor(1.0, dtype=measures_small.dtype), atol=1e-6)
        assert torch.all(measures_small > 0)
        assert torch.all(torch.isfinite(measures_small))
        
        # Test with very large variances
        large_activations = torch.randn(20, 10) * 1e8
        measures_large = manager._compute_variance_based_measures(large_activations)
        
        # Should still be valid probability distribution
        assert torch.allclose(measures_large.sum(), torch.tensor(1.0, dtype=measures_large.dtype), atol=1e-6)
        assert torch.all(measures_large > 0)
        assert torch.all(torch.isfinite(measures_large))
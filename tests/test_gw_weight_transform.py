"""Tests for GW cost-to-similarity weight transformations.

This module tests the proper conversion of GW costs (dissimilarity) to 
edge weights (similarity) for correct Laplacian semantics.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.assembly import GWLaplacianBuilder, SheafBuilder
from neurosheaf.sheaf.assembly.gw_laplacian import GWWeightTransform


class SimpleNet(nn.Module):
    """Simple network for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(12, 6)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestGWWeightTransform:
    """Test cost-to-similarity transformations."""
    
    def test_exponential_transform(self):
        """Test exponential transformation: similarity = exp(-β*cost)."""
        builder = GWLaplacianBuilder(
            weight_transform=GWWeightTransform.EXPONENTIAL,
            transform_beta=1.0
        )
        
        # Create mock sheaf with known costs
        model = SimpleNet()
        input_tensor = torch.randn(4, 8)
        
        gw_config = GWConfig(align_units=True)
        sheaf_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = sheaf_builder.build_from_activations(model, input_tensor, gw_config=gw_config)
        
        # Extract weights with exponential transform
        edge_weights = builder.extract_edge_weights(
            sheaf, 
            transform_method=GWWeightTransform.EXPONENTIAL,
            transform_beta=2.0  # Higher sensitivity
        )
        
        # Check that we got weights for all edges
        assert len(edge_weights) == len(sheaf.restrictions)
        
        # All weights should be positive
        for weight in edge_weights.values():
            assert weight > 0, f"Weight should be positive, got {weight}"
        
        # Weights should be <= 1 (since similarity = exp(-β*cost) and cost >= 0)
        for weight in edge_weights.values():
            assert weight <= 1.0, f"Weight should be <= 1.0, got {weight}"
    
    def test_reciprocal_transform(self):
        """Test reciprocal transformation: similarity = 1/(1+cost)."""
        builder = GWLaplacianBuilder(
            weight_transform=GWWeightTransform.RECIPROCAL,
            transform_beta=1.0
        )
        
        model = SimpleNet()
        input_tensor = torch.randn(4, 8)
        
        gw_config = GWConfig(align_units=True)
        sheaf_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = sheaf_builder.build_from_activations(model, input_tensor, gw_config=gw_config)
        
        edge_weights = builder.extract_edge_weights(
            sheaf,
            transform_method=GWWeightTransform.RECIPROCAL
        )
        
        # Check properties of reciprocal transform
        assert len(edge_weights) == len(sheaf.restrictions)
        
        # All weights should be positive and <= 1
        for weight in edge_weights.values():
            assert 0 < weight <= 1.0, f"Reciprocal weight should be in (0,1], got {weight}"
    
    def test_linear_transform(self):
        """Test linear transformation: similarity = max_cost - cost."""
        builder = GWLaplacianBuilder(
            weight_transform=GWWeightTransform.LINEAR,
            transform_beta=1.0
        )
        
        model = SimpleNet()
        input_tensor = torch.randn(4, 8)
        
        gw_config = GWConfig(align_units=True)
        sheaf_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = sheaf_builder.build_from_activations(model, input_tensor, gw_config=gw_config)
        
        edge_weights = builder.extract_edge_weights(
            sheaf,
            transform_method=GWWeightTransform.LINEAR
        )
        
        # Check properties of linear transform
        assert len(edge_weights) == len(sheaf.restrictions)
        
        # All weights should be positive
        for weight in edge_weights.values():
            assert weight > 0, f"Linear weight should be positive, got {weight}"
    
    def test_none_transform_deprecated(self):
        """Test NONE transform (deprecated, backward compatibility)."""
        builder = GWLaplacianBuilder(
            weight_transform=GWWeightTransform.NONE,
            transform_beta=1.0
        )
        
        model = SimpleNet()
        input_tensor = torch.randn(4, 8)
        
        gw_config = GWConfig(align_units=True)
        sheaf_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = sheaf_builder.build_from_activations(model, input_tensor, gw_config=gw_config)
        
        # This should log a deprecation warning
        edge_weights = builder.extract_edge_weights(
            sheaf,
            transform_method=GWWeightTransform.NONE
        )
        
        assert len(edge_weights) == len(sheaf.restrictions)
        # In NONE mode, weights are raw costs (which could be > 1)
    
    def test_transform_comparison(self):
        """Compare different transform methods on the same data."""
        model = SimpleNet()
        input_tensor = torch.randn(4, 8)
        
        gw_config = GWConfig(align_units=True)
        sheaf_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = sheaf_builder.build_from_activations(model, input_tensor, gw_config=gw_config)
        
        builder = GWLaplacianBuilder()
        
        # Extract weights with different transforms
        exponential_weights = builder.extract_edge_weights(
            sheaf, transform_method=GWWeightTransform.EXPONENTIAL, transform_beta=1.0
        )
        
        reciprocal_weights = builder.extract_edge_weights(
            sheaf, transform_method=GWWeightTransform.RECIPROCAL
        )
        
        linear_weights = builder.extract_edge_weights(
            sheaf, transform_method=GWWeightTransform.LINEAR  
        )
        
        # All methods should produce the same edges
        assert set(exponential_weights.keys()) == set(reciprocal_weights.keys()) == set(linear_weights.keys())
        
        # All should produce positive weights
        for weights in [exponential_weights, reciprocal_weights, linear_weights]:
            for weight in weights.values():
                assert weight > 0
    
    def test_square_root_scaling(self):
        """Test that square root is applied correctly for energy scaling."""
        # Create synthetic costs
        costs = {('a', 'b'): 0.0, ('b', 'c'): 1.0}  # Perfect match vs poor match
        
        # Mock sheaf with these costs
        class MockSheaf:
            def is_gw_sheaf(self):
                return True
            
            @property
            def metadata(self):
                return {'gw_costs': costs}
            
            @property  
            def restrictions(self):
                return {('a', 'b'): None, ('b', 'c'): None}
        
        sheaf = MockSheaf()
        builder = GWLaplacianBuilder(weight_transform=GWWeightTransform.EXPONENTIAL)
        
        weights = builder.extract_edge_weights(sheaf, list(costs.keys()))
        
        # For exponential: similarity = exp(-cost), weight = sqrt(similarity)
        expected_sim_ab = np.exp(-0.0)  # = 1.0
        expected_sim_bc = np.exp(-1.0)  # ≈ 0.368
        
        expected_weight_ab = np.sqrt(expected_sim_ab)  # = 1.0
        expected_weight_bc = np.sqrt(expected_sim_bc)  # ≈ 0.606
        
        assert np.isclose(weights[('a', 'b')], expected_weight_ab, atol=1e-6)
        assert np.isclose(weights[('b', 'c')], expected_weight_bc, atol=1e-6)
        
        # Better match should have higher weight
        assert weights[('a', 'b')] > weights[('b', 'c')]
    
    def test_beta_parameter_effect(self):
        """Test that beta parameter affects exponential sensitivity."""
        # Mock sheaf with known cost
        class MockSheaf:
            def is_gw_sheaf(self):
                return True
                
            @property
            def metadata(self):
                return {'gw_costs': {('a', 'b'): 1.0}}
            
            @property
            def restrictions(self):
                return {('a', 'b'): None}
        
        sheaf = MockSheaf()
        
        # Test different beta values
        builder1 = GWLaplacianBuilder(weight_transform=GWWeightTransform.EXPONENTIAL)
        builder2 = GWLaplacianBuilder(weight_transform=GWWeightTransform.EXPONENTIAL)
        
        weights_beta1 = builder1.extract_edge_weights(
            sheaf, [('a', 'b')], transform_beta=1.0
        )
        
        weights_beta2 = builder2.extract_edge_weights(
            sheaf, [('a', 'b')], transform_beta=2.0
        )
        
        # Higher beta should make the transform more sensitive to costs
        # exp(-1.0) vs exp(-2.0), then sqrt
        assert weights_beta2[('a', 'b')] < weights_beta1[('a', 'b')]


class TestGWLaplacianIntegration:
    """Test integration of weight transforms with Laplacian construction."""
    
    def test_laplacian_with_transformed_weights(self):
        """Test that Laplacian construction uses transformed weights correctly."""
        model = SimpleNet()
        input_tensor = torch.randn(4, 8)
        
        gw_config = GWConfig(align_units=True)
        sheaf_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = sheaf_builder.build_from_activations(model, input_tensor, gw_config=gw_config)
        
        # Build Laplacian with exponential transform
        builder = GWLaplacianBuilder(
            weight_transform=GWWeightTransform.EXPONENTIAL,
            transform_beta=1.0
        )
        
        laplacian = builder.build_laplacian(sheaf, sparse=True)
        
        # Basic validation
        assert laplacian.shape[0] == laplacian.shape[1]
        assert laplacian.nnz > 0
        
        # Should be symmetric
        if hasattr(laplacian, 'toarray'):
            L = laplacian.toarray()
            assert np.allclose(L, L.T, atol=1e-10)
    
    def test_different_transforms_produce_different_laplacians(self):
        """Test that different weight transforms produce different Laplacians."""
        model = SimpleNet()
        input_tensor = torch.randn(4, 8)
        
        gw_config = GWConfig(align_units=True)
        sheaf_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = sheaf_builder.build_from_activations(model, input_tensor, gw_config=gw_config)
        
        # Build with different transforms
        builder_exp = GWLaplacianBuilder(weight_transform=GWWeightTransform.EXPONENTIAL)
        builder_rec = GWLaplacianBuilder(weight_transform=GWWeightTransform.RECIPROCAL)
        
        L_exp = builder_exp.build_laplacian(sheaf, sparse=False)
        L_rec = builder_rec.build_laplacian(sheaf, sparse=False)
        
        # Should produce different results
        assert not np.allclose(L_exp, L_rec, atol=1e-6)
        
        # Both should be valid Laplacians
        assert np.allclose(L_exp, L_exp.T)  # Symmetric
        assert np.allclose(L_rec, L_rec.T)  # Symmetric
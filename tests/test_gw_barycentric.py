"""Tests for barycentric normalization in GW restriction maps.

This module tests that restriction maps are properly normalized using
barycentric normalization to ensure row-stochasticity and better
functoriality properties.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from neurosheaf.sheaf.core import GWConfig, GromovWassersteinComputer, GWResult
from neurosheaf.sheaf.assembly import GWRestrictionManager, SheafBuilder


class SimpleNet(nn.Module):
    """Simple network for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 8)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestBarycentricNormalization:
    """Test barycentric normalization of restriction maps."""
    
    def test_gw_result_includes_measures(self):
        """Test that GWResult now includes source and target measures."""
        config = GWConfig()
        computer = GromovWassersteinComputer(config)
        
        # Create cost matrices
        n_source = 10
        n_target = 8
        C_source = torch.rand(n_source, n_source)
        C_source = (C_source + C_source.T) / 2  # Make symmetric
        C_target = torch.rand(n_target, n_target)
        C_target = (C_target + C_target.T) / 2
        
        # Compute GW coupling
        result = computer.compute_gw_coupling(C_source, C_target)
        
        # Check that measures are included
        assert result.p_source is not None
        assert result.p_target is not None
        assert result.p_source.shape == (n_source,)
        assert result.p_target.shape == (n_target,)
        
        # Check that measures are valid distributions
        assert torch.allclose(result.p_source.sum(), torch.tensor(1.0, dtype=result.p_source.dtype), atol=1e-6)
        assert torch.allclose(result.p_target.sum(), torch.tensor(1.0, dtype=result.p_target.dtype), atol=1e-6)
        assert torch.all(result.p_source >= 0)
        assert torch.all(result.p_target >= 0)
    
    def test_restriction_map_row_stochasticity(self):
        """Test that restriction maps are row-stochastic after barycentric normalization."""
        config = GWConfig(align_units=True)
        manager = GWRestrictionManager(config=config)
        
        # Create mock activations
        batch_size = 6
        layer1_units = 20
        layer2_units = 15
        
        activations = {
            'layer1': torch.randn(batch_size, layer1_units),
            'layer2': torch.randn(batch_size, layer2_units)
        }
        
        # Create simple poset
        import networkx as nx
        poset = nx.DiGraph()
        poset.add_edge('layer1', 'layer2')
        
        # Compute restrictions with barycentric normalization
        restrictions, _, _ = manager.compute_all_restrictions(
            activations, poset, parallel=False
        )
        
        # Check that restriction is row-stochastic
        restriction = restrictions[('layer1', 'layer2')]
        row_sums = restriction.sum(dim=1)
        
        # All rows should sum to 1 (within numerical tolerance)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"Row sums not close to 1: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        
        # Check dimensions
        assert restriction.shape == (layer2_units, layer1_units)
    
    def test_barycentric_vs_non_barycentric(self):
        """Compare restriction maps with and without barycentric normalization."""
        # We'll compute the coupling manually and test both approaches
        config = GWConfig(align_units=True)
        computer = GromovWassersteinComputer(config)
        
        # Create cost matrices
        n_source = 12
        n_target = 10
        
        # Create random activations and compute costs
        X_source = torch.randn(n_source, 5)  # 5 features
        X_target = torch.randn(n_target, 5)
        
        C_source = computer.compute_cosine_cost_matrix(X_source)
        C_target = computer.compute_cosine_cost_matrix(X_target)
        
        # Compute GW coupling
        result = computer.compute_gw_coupling(C_source, C_target)
        
        # Method 1: Direct coupling (no normalization)
        # Our coupling already has shape (n_target, n_source), so no transpose needed
        restriction_simple = result.coupling
        
        # Method 2: Barycentric normalization
        p_target = result.p_target if result.p_target is not None else torch.ones(n_target, dtype=result.coupling.dtype) / n_target
        restriction_barycentric = result.coupling / p_target.unsqueeze(1)
        
        # Check properties
        # Simple transpose is NOT row-stochastic in general
        simple_row_sums = restriction_simple.sum(dim=1)
        assert not torch.allclose(simple_row_sums, torch.ones_like(simple_row_sums), atol=1e-5), \
            "Simple transpose should not be row-stochastic"
        
        # Barycentric normalized version IS row-stochastic
        barycentric_row_sums = restriction_barycentric.sum(dim=1)
        assert torch.allclose(barycentric_row_sums, torch.ones_like(barycentric_row_sums), atol=1e-5), \
            f"Barycentric version should be row-stochastic, got row sums: {barycentric_row_sums}"
        
        # The two should differ by a diagonal scaling
        scaling = p_target
        expected_barycentric = restriction_simple / scaling.unsqueeze(1)
        assert torch.allclose(restriction_barycentric, expected_barycentric, atol=1e-6)
    
    def test_functoriality_with_barycentric(self):
        """Test that barycentric normalization improves functoriality."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 8)
        )
        
        batch_size = 5
        input_tensor = torch.randn(batch_size, 10)
        
        # Build sheaf with GW method (uses barycentric normalization)
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(
            model, input_tensor, 
            validate=False,  # Skip for speed
            gw_config=config
        )
        
        # Check that all restrictions are row-stochastic
        for edge, restriction in sheaf.restrictions.items():
            row_sums = restriction.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Restriction {edge} not row-stochastic: row sums range [{row_sums.min():.6f}, {row_sums.max():.6f}]"
    
    def test_uniform_vs_nonuniform_measures(self):
        """Test that non-uniform measures are handled correctly."""
        config = GWConfig(uniform_measures=False)  # Request non-uniform (though not implemented)
        computer = GromovWassersteinComputer(config)
        
        n_source = 8
        n_target = 6
        C_source = torch.rand(n_source, n_source)
        C_source = (C_source + C_source.T) / 2
        C_target = torch.rand(n_target, n_target)
        C_target = (C_target + C_target.T) / 2
        
        # Even with non-uniform requested, should fall back to uniform
        result = computer.compute_gw_coupling(C_source, C_target)
        
        # Check measures are uniform
        expected_p_source = torch.ones(n_source, dtype=result.p_source.dtype) / n_source
        expected_p_target = torch.ones(n_target, dtype=result.p_target.dtype) / n_target
        
        assert torch.allclose(result.p_source, expected_p_source, atol=1e-6)
        assert torch.allclose(result.p_target, expected_p_target, atol=1e-6)
    
    def test_validate_marginals_with_stored_measures(self):
        """Test that validate_marginals uses stored measures correctly."""
        # Create a mock GWResult
        n_source = 5
        n_target = 4
        # Our convention: coupling has shape (n_target, n_source)
        coupling = torch.rand(n_target, n_source)
        
        # Make it a valid coupling with specific marginals
        p_source = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15])
        p_target = torch.tensor([0.3, 0.2, 0.2, 0.3])
        
        # Normalize coupling to have correct marginals
        coupling = coupling / coupling.sum()  # Normalize to sum to 1
        # Use Sinkhorn-like iteration to fix marginals (simplified)
        # For coupling shape (n_target, n_source): rows=target, cols=source
        for _ in range(10):
            # Fix row sums to match p_target
            coupling = coupling * p_target.unsqueeze(1) / coupling.sum(dim=1, keepdim=True)
            # Fix column sums to match p_source
            coupling = coupling * p_source.unsqueeze(0) / coupling.sum(dim=0, keepdim=True)
        
        result = GWResult(
            coupling=coupling,
            cost=0.5,
            log={},
            source_size=n_source,
            target_size=n_target,
            p_source=p_source,
            p_target=p_target
        )
        
        # Validate using stored measures
        validation = result.validate_marginals()
        
        # Should use the stored measures and find small violations
        assert validation['max_violation'] < 1e-3, \
            f"Marginal violation too large: {validation['max_violation']}"
        
        # Test with different measures (should override stored)
        uniform_p_source = torch.ones(n_source) / n_source
        uniform_p_target = torch.ones(n_target) / n_target
        
        validation_uniform = result.validate_marginals(uniform_p_source, uniform_p_target)
        
        # Should have larger violation with uniform measures
        assert validation_uniform['max_violation'] > validation['max_violation']


class TestBarycentricIntegration:
    """Integration tests for barycentric normalization."""
    
    def test_full_pipeline_with_barycentric(self):
        """Test the full GW pipeline with barycentric normalization."""
        model = SimpleNet()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 10)
        
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # Build sheaf
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=True,  # Enable validation
            gw_config=config
        )
        
        # Check metadata
        assert 'gw_config' in sheaf.metadata
        assert sheaf.metadata['align_units'] is True
        
        # Check all restrictions are row-stochastic
        for (source, target), restriction in sheaf.restrictions.items():
            row_sums = restriction.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Restriction {source}→{target} not row-stochastic"
            
            # Check shape consistency with stalks
            source_dim = sheaf.stalks[source].shape[0]
            target_dim = sheaf.stalks[target].shape[0]
            assert restriction.shape == (target_dim, source_dim), \
                f"Restriction shape mismatch: expected ({target_dim}, {source_dim}), got {restriction.shape}"
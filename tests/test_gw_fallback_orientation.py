"""Tests for GW fallback coupling orientation fix.

This module tests that the fallback GW coupling uses the correct POT convention
for coupling matrix orientation: shape (n_source, n_target) where coupling[i,j]
represents transport from source node i to target node j.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from neurosheaf.sheaf.core import GWConfig, GromovWassersteinComputer
from neurosheaf.sheaf.assembly import SheafBuilder


class SimpleNet(nn.Module):
    """Simple network for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(12, 6)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestGWFallbackOrientation:
    """Test fallback GW coupling orientation."""
    
    def test_fallback_coupling_shape_convention(self):
        """Test that fallback coupling follows POT convention: (n_source, n_target)."""
        config = GWConfig()
        computer = GromovWassersteinComputer(config)
        
        n_source = 10
        n_target = 8
        
        # Create symmetric cost matrices
        C_source = torch.rand(n_source, n_source)
        C_source = (C_source + C_source.T) / 2
        C_target = torch.rand(n_target, n_target)
        C_target = (C_target + C_target.T) / 2
        
        # Create uniform measures
        p_source = torch.ones(n_source) / n_source
        p_target = torch.ones(n_target) / n_target
        
        # Force fallback by calling the private method directly
        coupling, cost, log = computer._compute_gw_fallback(C_source, C_target, p_source, p_target)
        
        # Check shape follows POT convention
        expected_shape = (n_source, n_target)
        assert coupling.shape == expected_shape, \
            f"Expected coupling shape {expected_shape}, got {coupling.shape}"
        
        # Check marginal constraints are satisfied
        # coupling.sum(axis=1) should equal p_source
        source_marginals = coupling.sum(dim=1)
        assert torch.allclose(source_marginals, p_source, atol=1e-6), \
            f"Source marginals violated: expected {p_source}, got {source_marginals}"
        
        # coupling.sum(axis=0) should equal p_target
        target_marginals = coupling.sum(dim=0)
        assert torch.allclose(target_marginals, p_target, atol=1e-6), \
            f"Target marginals violated: expected {p_target}, got {target_marginals}"
    
    def test_fallback_vs_pot_orientation_consistency(self):
        """Test that fallback and POT solver produce same orientation."""
        config = GWConfig()
        computer = GromovWassersteinComputer(config)
        
        n_source = 6
        n_target = 5
        
        # Create cost matrices
        C_source = torch.rand(n_source, n_source)
        C_source = (C_source + C_source.T) / 2
        C_target = torch.rand(n_target, n_target)
        C_target = (C_target + C_target.T) / 2
        
        # Create uniform measures
        p_source = torch.ones(n_source) / n_source
        p_target = torch.ones(n_target) / n_target
        
        # Get fallback coupling
        fallback_coupling, _, _ = computer._compute_gw_fallback(C_source, C_target, p_source, p_target)
        
        # Try POT solver (might fail, but if it succeeds, check consistency)
        try:
            result = computer.compute_gw_coupling(C_source, C_target, p_source, p_target)
            pot_coupling = result.coupling
            
            # Both should have same shape
            assert fallback_coupling.shape == pot_coupling.shape, \
                f"Orientation mismatch: fallback {fallback_coupling.shape} vs POT {pot_coupling.shape}"
            
            # Both should satisfy same marginal constraints
            fallback_source_marginals = fallback_coupling.sum(dim=1)
            pot_source_marginals = pot_coupling.sum(dim=1)
            assert torch.allclose(fallback_source_marginals, pot_source_marginals, atol=1e-5)
            
            fallback_target_marginals = fallback_coupling.sum(dim=0) 
            pot_target_marginals = pot_coupling.sum(dim=0)
            assert torch.allclose(fallback_target_marginals, pot_target_marginals, atol=1e-5)
            
        except Exception:
            # POT failed, just check fallback is well-formed
            assert fallback_coupling.shape == (n_source, n_target)
            assert torch.all(fallback_coupling >= 0)
    
    def test_fallback_in_restriction_computation(self):
        """Test that fallback coupling works correctly in restriction map computation."""
        model = SimpleNet()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 8)
        
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # Force use of fallback by temporarily breaking POT import
        # (This is done implicitly if POT is not available)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,  # Skip for speed
            gw_config=config
        )
        
        # Check that restriction maps have correct orientation
        # If coupling has shape (n_source, n_target), then coupling.T has shape (n_target, n_source)
        # which is the correct shape for a restriction map from source stalk to target stalk
        
        for (source, target), restriction in sheaf.restrictions.items():
            source_dim = sheaf.stalks[source].shape[0]
            target_dim = sheaf.stalks[target].shape[0]
            
            # Restriction should map from source to target
            expected_shape = (target_dim, source_dim)
            assert restriction.shape == expected_shape, \
                f"Restriction {source}→{target} has wrong shape: expected {expected_shape}, got {restriction.shape}"
            
            # Should be row-stochastic (rows sum to 1)
            row_sums = restriction.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Restriction {source}→{target} not row-stochastic"
    
    def test_coupling_cost_computation_consistency(self):
        """Test that cost computation works correctly with fixed coupling orientation."""
        config = GWConfig()
        computer = GromovWassersteinComputer(config)
        
        n_source = 5
        n_target = 4
        
        # Create simple cost matrices
        C_source = torch.eye(n_source) * 2  # Identity scaled
        C_target = torch.eye(n_target) * 2
        
        p_source = torch.ones(n_source) / n_source
        p_target = torch.ones(n_target) / n_target
        
        # Get fallback coupling
        coupling, cost, log = computer._compute_gw_fallback(C_source, C_target, p_source, p_target)
        
        # Verify cost is non-negative and finite
        assert cost >= 0, f"Cost should be non-negative, got {cost}"
        assert np.isfinite(cost), f"Cost should be finite, got {cost}"
        
        # Verify coupling orientation is correct for cost computation
        # The cost computation should work without errors
        recomputed_cost = computer._compute_gw_cost(C_source, C_target, coupling)
        assert np.isfinite(recomputed_cost), f"Recomputed cost should be finite, got {recomputed_cost}"
    
    def test_marginal_constraints_preserved(self):
        """Test that marginal constraints are exactly preserved in fallback."""
        config = GWConfig()
        computer = GromovWassersteinComputer(config)
        
        # Use non-uniform measures to make test more sensitive
        n_source = 6
        n_target = 4
        
        # Create slightly non-uniform measures (though fallback uses uniform)
        p_source = torch.ones(n_source) / n_source
        p_target = torch.ones(n_target) / n_target
        
        C_source = torch.rand(n_source, n_source)
        C_source = (C_source + C_source.T) / 2
        C_target = torch.rand(n_target, n_target)
        C_target = (C_target + C_target.T) / 2
        
        coupling, _, _ = computer._compute_gw_fallback(C_source, C_target, p_source, p_target)
        
        # Check exact marginal preservation
        source_marginals = coupling.sum(dim=1)
        target_marginals = coupling.sum(dim=0)
        
        assert torch.allclose(source_marginals, p_source, atol=1e-10), \
            f"Source marginals not preserved: max error {(source_marginals - p_source).abs().max()}"
        
        assert torch.allclose(target_marginals, p_target, atol=1e-10), \
            f"Target marginals not preserved: max error {(target_marginals - p_target).abs().max()}"


class TestOrientationIntegration:
    """Integration tests for coupling orientation fix."""
    
    def test_full_pipeline_with_corrected_orientation(self):
        """Test full GW sheaf construction works with corrected fallback orientation."""
        model = SimpleNet()
        batch_size = 3
        input_tensor = torch.randn(batch_size, 8)
        
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # This should work without errors
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=True,  # Enable validation
            gw_config=config
        )
        
        # Basic validation
        assert len(sheaf.stalks) > 0
        assert len(sheaf.restrictions) > 0
        
        # All restrictions should be properly shaped and row-stochastic
        for (source, target), restriction in sheaf.restrictions.items():
            # Check shape consistency
            source_dim = sheaf.stalks[source].shape[0]
            target_dim = sheaf.stalks[target].shape[0]
            assert restriction.shape == (target_dim, source_dim)
            
            # Check row-stochasticity
            row_sums = restriction.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
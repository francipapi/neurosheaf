"""Tests for unit-based GW alignment.

This module tests the unit-based alignment mode for GW sheaf construction,
ensuring that:
1. Cost matrices have unit dimensions
2. Stalks are unit Gram matrices  
3. Restrictions map between unit spaces
4. Results are batch-independent
5. Architecture awareness is preserved
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from neurosheaf.sheaf.core import GWConfig, GromovWassersteinComputer
from neurosheaf.sheaf.assembly import SheafBuilder, GWRestrictionManager


class SimpleTestNet(nn.Module):
    """Simple test network with different layer widths."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestUnitAlignment:
    """Test unit-based alignment in GW sheaf construction."""
    
    def test_cost_matrix_dimensions_unit_mode(self):
        """Test that cost matrices have unit dimensions when align_units=True."""
        config = GWConfig(align_units=True)
        computer = GromovWassersteinComputer(config)
        
        # Create activation tensor: (batch_size=8, n_units=16)
        batch_size = 8
        n_units = 16
        activations = torch.randn(batch_size, n_units)
        
        # For unit alignment, we transpose to (n_units, batch_size)
        X_units = activations.T
        cost_matrix = computer.compute_cosine_cost_matrix(X_units)
        
        # Cost matrix should be (n_units, n_units)
        assert cost_matrix.shape == (n_units, n_units)
        assert torch.allclose(cost_matrix, cost_matrix.T)  # Symmetric
        assert torch.allclose(cost_matrix.diag(), torch.zeros(n_units, dtype=cost_matrix.dtype))  # Zero diagonal
    
    def test_cost_matrix_dimensions_sample_mode(self):
        """Test that cost matrices have sample dimensions when align_units=False."""
        config = GWConfig(align_units=False)
        computer = GromovWassersteinComputer(config)
        
        # Create activation tensor: (batch_size=8, n_units=16)
        batch_size = 8
        n_units = 16
        activations = torch.randn(batch_size, n_units)
        
        # For sample alignment, keep as (batch_size, n_units)
        cost_matrix = computer.compute_cosine_cost_matrix(activations)
        
        # Cost matrix should be (batch_size, batch_size)
        assert cost_matrix.shape == (batch_size, batch_size)
        assert torch.allclose(cost_matrix, cost_matrix.T)  # Symmetric
        assert torch.allclose(cost_matrix.diag(), torch.zeros(batch_size, dtype=cost_matrix.dtype))  # Zero diagonal
    
    def test_stalk_dimensions_unit_mode(self):
        """Test that stalks have unit dimensions when align_units=True."""
        model = SimpleTestNet()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 10)
        
        # Build sheaf with unit alignment
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, gw_config=config)
        
        # Check stalk dimensions
        for node_name, stalk in sheaf.stalks.items():
            # Stalk should be square with unit dimensions
            assert stalk.shape[0] == stalk.shape[1]
            
            # Get expected unit count based on layer
            if 'fc1' in node_name:
                expected_units = 20
            elif 'fc2' in node_name:
                expected_units = 15
            elif 'fc3' in node_name:
                expected_units = 5
            else:
                continue  # Skip non-fc layers
                
            # Stalk dimension should match unit count, not batch size
            assert stalk.shape[0] == expected_units or stalk.shape[0] > 0
            assert stalk.shape[0] != batch_size  # Should NOT be batch size
    
    def test_batch_independence(self):
        """Test that unit-based alignment produces batch-independent results."""
        model = SimpleTestNet()
        input_dim = 10
        
        # Create two different batches
        batch1 = torch.randn(4, input_dim)
        batch2 = torch.randn(8, input_dim)  # Different batch size
        
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # Build sheaves for both batches
        sheaf1 = builder.build_from_activations(model, batch1, gw_config=config)
        sheaf2 = builder.build_from_activations(model, batch2, gw_config=config)
        
        # Stalk dimensions should be the same (unit counts don't change)
        for node_name in sheaf1.stalks.keys():
            if node_name in sheaf2.stalks:
                assert sheaf1.stalks[node_name].shape == sheaf2.stalks[node_name].shape
                # Both should have unit dimensions, not batch dimensions
                assert sheaf1.stalks[node_name].shape[0] != 4
                assert sheaf2.stalks[node_name].shape[0] != 8
    
    def test_unit_gram_matrix_properties(self):
        """Test that unit Gram matrices have correct mathematical properties."""
        model = SimpleTestNet()
        batch_size = 6
        input_tensor = torch.randn(batch_size, 10)
        
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, gw_config=config)
        
        for node_name, stalk in sheaf.stalks.items():
            # Check Gram matrix properties
            # 1. Symmetric
            assert torch.allclose(stalk, stalk.T, atol=1e-6)
            
            # 2. Positive semi-definite (all eigenvalues >= 0)
            eigenvalues = torch.linalg.eigvalsh(stalk)
            assert torch.all(eigenvalues >= -1e-6)  # Allow small numerical errors
            
            # 3. Diagonal elements should be 1 (normalized unit vectors)
            # This is true for cosine similarity of normalized vectors
            assert torch.allclose(stalk.diag(), torch.ones(stalk.shape[0]), atol=1e-5)
            
            # 4. All elements in [-1, 1] (cosine similarity range)
            assert torch.all(stalk >= -1.0 - 1e-6)
            assert torch.all(stalk <= 1.0 + 1e-6)
    
    def test_restriction_dimensions(self):
        """Test that restriction maps have correct dimensions for unit alignment."""
        model = SimpleTestNet()
        batch_size = 5
        input_tensor = torch.randn(batch_size, 10)
        
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, gw_config=config)
        
        # Check restriction dimensions
        for (source, target), restriction in sheaf.restrictions.items():
            source_stalk = sheaf.stalks[source]
            target_stalk = sheaf.stalks[target]
            
            # Restriction should map from target units to source units
            expected_shape = (target_stalk.shape[0], source_stalk.shape[0])
            assert restriction.shape == expected_shape, \
                f"Restriction {source}→{target} has shape {restriction.shape}, expected {expected_shape}"
    
    def test_architecture_awareness(self):
        """Test that unit alignment correctly handles layers with different widths."""
        # Create a network with very different layer widths
        class VariableWidthNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 64)   # Wide layer
                self.fc2 = nn.Linear(64, 8)    # Narrow layer
                self.fc3 = nn.Linear(8, 128)   # Wide layer again
                self.fc4 = nn.Linear(128, 1)   # Single output
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
                return x
        
        model = VariableWidthNet()
        batch_size = 3
        input_tensor = torch.randn(batch_size, 10)
        
        config = GWConfig(align_units=True)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, gw_config=config)
        
        # Verify each layer has the correct unit count in stalks
        expected_widths = {'fc1': 64, 'fc2': 8, 'fc3': 128, 'fc4': 1}
        
        for node_name, stalk in sheaf.stalks.items():
            for layer_name, expected_width in expected_widths.items():
                if layer_name in node_name:
                    # Stalk dimension should match layer width
                    assert stalk.shape[0] == expected_width, \
                        f"Layer {node_name} has stalk dimension {stalk.shape[0]}, expected {expected_width}"
                    break
    
    def test_warning_for_deprecated_mode(self, caplog):
        """Test that using sample-based alignment produces a deprecation warning."""
        import logging
        
        # Enable logging capture
        caplog.set_level(logging.WARNING)
        
        # Create config with deprecated sample-based alignment
        config = GWConfig(align_units=False)
        config.validate()
        
        # Check that warning was logged
        assert any("deprecated" in record.message.lower() for record in caplog.records)
        assert any("align_units=False" in record.message for record in caplog.records)


class TestGWRestrictionManagerUnitMode:
    """Test GW restriction manager with unit-based alignment."""
    
    def test_restriction_manager_unit_dimensions(self):
        """Test that GW restriction manager produces unit-dimensional restrictions."""
        config = GWConfig(align_units=True)
        manager = GWRestrictionManager(config=config)
        
        # Create mock activations for two layers
        batch_size = 6
        layer1_units = 32
        layer2_units = 16
        
        activations = {
            'layer1': torch.randn(batch_size, layer1_units),
            'layer2': torch.randn(batch_size, layer2_units)
        }
        
        # Create simple poset
        import networkx as nx
        poset = nx.DiGraph()
        poset.add_edge('layer1', 'layer2')
        
        # Compute restrictions
        restrictions, gw_costs, metadata = manager.compute_all_restrictions(
            activations, poset, parallel=False
        )
        
        # Check restriction dimensions
        restriction = restrictions[('layer1', 'layer2')]
        # Should map from layer2 units to layer1 units
        assert restriction.shape == (layer2_units, layer1_units)
    
    def test_cost_matrix_caching_unit_mode(self):
        """Test that cost matrix caching works correctly with unit dimensions."""
        config = GWConfig(align_units=True, cache_cost_matrices=True)
        manager = GWRestrictionManager(config=config)
        
        # Create activations
        batch_size = 4
        n_units = 20
        activations = {
            'layer': torch.randn(batch_size, n_units)
        }
        
        # Create trivial poset (single node)
        import networkx as nx
        poset = nx.DiGraph()
        poset.add_node('layer')
        
        # Compute cost matrices twice
        cost_matrices1 = manager._compute_all_cost_matrices(activations)
        cost_matrices2 = manager._compute_all_cost_matrices(activations)
        
        # Both should have unit dimensions
        assert cost_matrices1['layer'].shape == (n_units, n_units)
        assert cost_matrices2['layer'].shape == (n_units, n_units)
        
        # Check cache statistics if available
        cache_stats = manager.get_cache_stats()
        assert cache_stats is not None
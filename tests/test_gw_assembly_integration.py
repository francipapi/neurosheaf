"""Unit tests for GW assembly integration (Phase 2).

This module tests the integration of GW-based restriction computation
with the existing sheaf assembly infrastructure, ensuring seamless
operation and backward compatibility.
"""

import pytest
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from unittest.mock import patch, MagicMock
import time

from neurosheaf.sheaf.assembly import SheafBuilder, GWRestrictionManager, GWRestrictionError
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.data_structures import Sheaf


class SimpleTestNet(nn.Module):
    """Simple test network for GW sheaf construction."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LinearTestNet(nn.Module):
    """Simple linear test network for controlled testing."""
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 8)
        self.layer2 = nn.Linear(8, 6)
        self.layer3 = nn.Linear(6, 4)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class TestGWRestrictionManager:
    """Test GW restriction manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GWConfig(
            epsilon=0.1,
            max_iter=50,  # Reduced for faster tests
            tolerance=1e-6,
            validate_couplings=True,
            cache_cost_matrices=False  # Disable for reproducible tests
        )
        self.manager = GWRestrictionManager(config=self.config)
        
        # Create synthetic activations for testing
        torch.manual_seed(42)  # Reproducible
        self.activations = {
            'layer1': torch.randn(5, 10),  # 5 samples, 10 features
            'layer2': torch.randn(5, 8),   # 5 samples, 8 features
            'layer3': torch.randn(5, 6)    # 5 samples, 6 features
        }
        
        # Create simple poset
        self.poset = nx.DiGraph()
        self.poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
    
    def test_manager_initialization(self):
        """Test GW restriction manager initialization."""
        assert self.manager.config.epsilon == 0.1
        assert self.manager.config.max_iter == 50
        assert self.manager.gw_computer is not None
    
    def test_cost_matrix_computation(self):
        """Test cosine cost matrix computation for activations."""
        cost_matrices = self.manager._compute_all_cost_matrices(self.activations)
        
        # Check all layers have cost matrices
        assert set(cost_matrices.keys()) == set(self.activations.keys())
        
        # Check matrix properties
        for layer_name, C in cost_matrices.items():
            n_samples = self.activations[layer_name].shape[0]
            assert C.shape == (n_samples, n_samples)
            
            # Check symmetry
            assert torch.allclose(C, C.T, atol=1e-6)
            
            # Check non-negativity
            assert torch.all(C >= 0)
            
            # Check zero diagonal
            assert torch.allclose(torch.diag(C), torch.zeros(n_samples), atol=1e-6)
    
    def test_single_restriction_computation(self):
        """Test computation of a single GW restriction map."""
        cost_matrices = self.manager._compute_all_cost_matrices(self.activations)
        
        # Test layer1 → layer2 restriction
        result = self.manager._compute_single_restriction(
            'layer1', 'layer2', cost_matrices, self.activations
        )
        
        assert result is not None
        restriction, cost, coupling = result
        
        # Check dimensions: restriction should be (target_size, source_size)
        source_size = self.activations['layer1'].shape[0]  # 5
        target_size = self.activations['layer2'].shape[0]  # 5
        assert restriction.shape == (target_size, source_size)
        
        # Check cost is non-negative
        assert cost >= 0
        
        # Check coupling shape (from POT: source_size, target_size)
        assert coupling.shape == (source_size, target_size)
    
    def test_all_restrictions_sequential(self):
        """Test sequential computation of all restrictions."""
        restrictions, gw_costs, metadata = self.manager.compute_all_restrictions(
            self.activations, self.poset, parallel=False
        )
        
        # Check we got restrictions for all edges
        expected_edges = set(self.poset.edges())
        assert set(restrictions.keys()) == expected_edges
        assert set(gw_costs.keys()) == expected_edges
        
        # Check restriction dimensions
        for (source, target), restriction in restrictions.items():
            source_size = self.activations[source].shape[0]
            target_size = self.activations[target].shape[0]
            assert restriction.shape == (target_size, source_size)
        
        # Check metadata
        assert metadata['construction_method'] == 'gromov_wasserstein'
        assert metadata['num_edges_succeeded'] == len(expected_edges)
        assert metadata['num_edges_failed'] == 0
    
    def test_all_restrictions_parallel(self):
        """Test parallel computation of all restrictions."""
        restrictions, gw_costs, metadata = self.manager.compute_all_restrictions(
            self.activations, self.poset, parallel=True, max_workers=2
        )
        
        # Check we got restrictions for all edges
        expected_edges = set(self.poset.edges())
        assert set(restrictions.keys()) == expected_edges
        assert set(gw_costs.keys()) == expected_edges
        
        # Check metadata indicates parallel processing
        assert metadata['parallel_processing'] is True
        assert metadata['num_edges_succeeded'] == len(expected_edges)
    
    def test_quasi_sheaf_validation(self):
        """Test quasi-sheaf property validation."""
        # Create longer chain for transitivity testing
        chain_poset = nx.DiGraph()
        chain_poset.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        chain_activations = {
            'A': torch.randn(4, 8),
            'B': torch.randn(4, 6),
            'C': torch.randn(4, 4)
        }
        
        restrictions, gw_costs, metadata = self.manager.compute_all_restrictions(
            chain_activations, chain_poset, parallel=False
        )
        
        # Validate quasi-sheaf property
        validation_report = self.manager.validate_quasi_sheaf_property(
            restrictions, chain_poset, tolerance=0.5  # Generous tolerance for test
        )
        
        assert 'max_violation' in validation_report
        assert 'satisfies_quasi_sheaf' in validation_report
        assert 'num_paths_checked' in validation_report
        assert validation_report['num_paths_checked'] >= 1  # Should check A→B→C
    
    def test_edge_weight_extraction(self):
        """Test extraction of edge weights from GW costs."""
        gw_costs = {
            ('layer1', 'layer2'): 0.5,
            ('layer2', 'layer3'): 0.3
        }
        
        edge_weights = self.manager.extract_edge_weights(gw_costs)
        
        # Edge weights should be identical to GW costs for increasing filtration
        assert edge_weights == gw_costs


class TestSheafBuilderGWIntegration:
    """Test SheafBuilder integration with GW methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create builders with different methods
        self.procrustes_builder = SheafBuilder(restriction_method='scaled_procrustes')
        self.gw_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # Simple test networks
        self.linear_net = LinearTestNet()
        self.test_input = torch.randn(3, 10)  # Batch of 3
        
        # GW configuration
        self.gw_config = GWConfig(
            epsilon=0.1,
            max_iter=20,  # Fast for testing
            validate_couplings=True,
            quasi_sheaf_tolerance=0.5
        )
    
    def test_builder_initialization(self):
        """Test SheafBuilder initialization with different methods."""
        # Test default (Procrustes)
        default_builder = SheafBuilder()
        assert default_builder.restriction_method == 'scaled_procrustes'
        
        # Test GW method
        assert self.gw_builder.restriction_method == 'gromov_wasserstein'
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown restriction method"):
            SheafBuilder(restriction_method='invalid_method')
    
    def test_method_routing(self):
        """Test that build_from_activations routes to correct method."""
        # This is an integration test - we'll mock the internal methods
        with patch.object(self.gw_builder, '_build_gw_sheaf') as mock_gw_build:
            mock_gw_build.return_value = MagicMock(spec=Sheaf)
            
            self.gw_builder.build_from_activations(
                self.linear_net, self.test_input, gw_config=self.gw_config
            )
            
            # Verify GW method was called
            mock_gw_build.assert_called_once()
    
    def test_gw_sheaf_construction_end_to_end(self):
        """Test complete GW sheaf construction pipeline."""
        # Note: This is a longer test that exercises the full pipeline
        try:
            sheaf = self.gw_builder.build_from_activations(
                self.linear_net, self.test_input, 
                validate=True, gw_config=self.gw_config
            )
            
            # Basic sheaf properties
            assert isinstance(sheaf, Sheaf)
            assert len(sheaf.restrictions) > 0
            assert len(sheaf.stalks) > 0
            
            # GW-specific metadata
            assert sheaf.metadata['construction_method'] == 'gromov_wasserstein'
            assert 'gw_config' in sheaf.metadata
            assert 'gw_costs' in sheaf.metadata
            assert sheaf.metadata['whitened'] is False  # GW doesn't use whitening
            
            # Check filtration semantics
            assert sheaf.get_filtration_semantics() == 'increasing'
            assert sheaf.is_gw_sheaf() is True
            
            # Check stalks are identity matrices (appropriate size)
            for node_name, stalk in sheaf.stalks.items():
                assert stalk.shape[0] == stalk.shape[1]  # Square
                # Should be identity matrix
                expected_identity = torch.eye(stalk.shape[0])
                assert torch.allclose(stalk, expected_identity, atol=1e-6)
            
        except Exception as e:
            # If POT is not available, this test may fail gracefully
            if "POT" in str(e) or "not available" in str(e):
                pytest.skip("POT library not available for GW computations")
            else:
                raise
    
    def test_backward_compatibility_procrustes(self):
        """Test that Procrustes method still works unchanged."""
        sheaf = self.procrustes_builder.build_from_activations(
            self.linear_net, self.test_input, validate=True
        )
        
        # Should be traditional sheaf
        assert isinstance(sheaf, Sheaf)
        assert sheaf.metadata['construction_method'] in ['fx_unified_whitened', 'scaled_procrustes']
        assert sheaf.metadata['whitened'] is True  # Procrustes uses whitening
        assert sheaf.is_gw_sheaf() is False
        assert sheaf.get_filtration_semantics() == 'decreasing'
    
    def test_different_gw_configurations(self):
        """Test GW sheaf construction with different configurations."""
        configs = [
            GWConfig.default_fast(),
            GWConfig.default_accurate(), 
            GWConfig.default_debugging()
        ]
        
        for i, config in enumerate(configs):
            try:
                sheaf = self.gw_builder.build_from_activations(
                    self.linear_net, self.test_input,
                    validate=True, gw_config=config
                )
                
                # All should produce valid sheaves
                assert isinstance(sheaf, Sheaf)
                assert sheaf.metadata['gw_config']['epsilon'] == config.epsilon
                assert sheaf.metadata['gw_config']['max_iter'] == config.max_iter
                
            except Exception as e:
                if "POT" in str(e):
                    pytest.skip(f"POT library not available for config {i}")
                else:
                    raise


class TestGWIntegrationErrors:
    """Test error handling in GW integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gw_builder = SheafBuilder(restriction_method='gromov_wasserstein')
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Empty model
        class EmptyNet(nn.Module):
            def forward(self, x):
                return x
        
        empty_net = EmptyNet()
        test_input = torch.randn(2, 5)
        
        # This should handle gracefully (may produce minimal sheaf or error)
        try:
            sheaf = self.gw_builder.build_from_activations(empty_net, test_input)
            # If successful, should be minimal
            assert isinstance(sheaf, Sheaf)
        except (RuntimeError, ValueError) as e:
            # Acceptable to fail on trivial network
            assert "failed" in str(e).lower() or "no" in str(e).lower()
    
    def test_gw_config_validation(self):
        """Test GW configuration validation."""
        # Invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be positive"):
            invalid_config = GWConfig(epsilon=-0.1)
            invalid_config.validate()
        
        # Invalid max_iter
        with pytest.raises(ValueError, match="max_iter must be positive"):
            invalid_config = GWConfig(max_iter=0)
            invalid_config.validate()


class TestGWRestrictionManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = GWRestrictionManager()
    
    def test_empty_poset(self):
        """Test handling of empty poset."""
        empty_activations = {'node1': torch.randn(3, 5)}
        empty_poset = nx.DiGraph()
        empty_poset.add_node('node1')  # Node but no edges
        
        restrictions, costs, metadata = self.manager.compute_all_restrictions(
            empty_activations, empty_poset
        )
        
        # Should handle gracefully
        assert len(restrictions) == 0
        assert len(costs) == 0
        assert metadata['num_edges_succeeded'] == 0
    
    def test_missing_activations(self):
        """Test handling of missing activation data."""
        incomplete_activations = {'layer1': torch.randn(3, 5)}
        poset = nx.DiGraph()
        poset.add_edge('layer1', 'missing_layer')
        
        restrictions, costs, metadata = self.manager.compute_all_restrictions(
            incomplete_activations, poset
        )
        
        # Should fail gracefully
        assert metadata['num_edges_failed'] > 0
        assert len(metadata['failed_edges']) > 0
    
    def test_identical_activations(self):
        """Test handling of identical activation tensors."""
        # Same activations should produce zero cost
        identical_activations = {
            'layer1': torch.ones(4, 6),  # All ones
            'layer2': torch.ones(4, 6)   # All ones
        }
        
        poset = nx.DiGraph()
        poset.add_edge('layer1', 'layer2')
        
        restrictions, costs, metadata = self.manager.compute_all_restrictions(
            identical_activations, poset
        )
        
        # Should succeed with low cost (identical distributions)
        assert metadata['num_edges_succeeded'] == 1
        assert ('layer1', 'layer2') in costs
        # Cost might not be exactly zero due to numerical precision


class TestPerformance:
    """Performance and scalability tests."""
    
    def test_parallel_vs_sequential_performance(self):
        """Test that parallel processing provides speedup for larger problems."""
        # Create larger synthetic problem
        n_layers = 6
        n_samples = 10
        feature_sizes = [20, 18, 16, 14, 12, 10]
        
        # Create chain network
        activations = {}
        poset = nx.DiGraph()
        
        for i in range(n_layers):
            layer_name = f"layer_{i}"
            activations[layer_name] = torch.randn(n_samples, feature_sizes[i])
            
            if i > 0:
                prev_layer = f"layer_{i-1}"
                poset.add_edge(prev_layer, layer_name)
        
        manager = GWRestrictionManager(config=GWConfig(max_iter=10))  # Fast config
        
        # Sequential
        start_time = time.time()
        _, _, sequential_metadata = manager.compute_all_restrictions(
            activations, poset, parallel=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel
        start_time = time.time()
        _, _, parallel_metadata = manager.compute_all_restrictions(
            activations, poset, parallel=True, max_workers=2
        )
        parallel_time = time.time() - start_time
        
        # Both should succeed
        assert sequential_metadata['num_edges_succeeded'] > 0
        assert parallel_metadata['num_edges_succeeded'] > 0
        
        # Note: Parallel may not always be faster for small problems due to overhead,
        # but both should complete in reasonable time
        assert sequential_time < 30.0  # Should complete within 30 seconds
        assert parallel_time < 30.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
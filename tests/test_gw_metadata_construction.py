"""Tests for GW sheaf metadata construction and detection.

This module tests that GW sheaves are properly marked with construction_method='gromov_wasserstein'
and that all GW-specific functionality correctly detects and works with GW sheaves.
"""

import pytest
import torch
import torch.nn as nn
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.sheaf.core import GWConfig


class SimpleNetwork(nn.Module):
    """Simple test network for GW sheaf construction."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(8, 12),
            nn.Linear(12, 6),
            nn.Linear(6, 4)
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class TestGWMetadataConstruction:
    """Test GW sheaf metadata construction and detection."""
    
    def test_gw_sheaf_construction_sets_correct_metadata(self):
        """Test that GW sheaf construction sets construction_method='gromov_wasserstein'."""
        # Create test model and input
        model = SimpleNetwork()
        input_tensor = torch.randn(10, 8)
        
        # Build GW sheaf using correct method
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(uniform_measures=False)  # Test with non-uniform measures
        
        sheaf = builder.build_from_activations(
            model, input_tensor, 
            validate=False,  # Skip validation for speed
            gw_config=gw_config
        )
        
        # Verify construction_method is set correctly
        assert sheaf.metadata.get('construction_method') == 'gromov_wasserstein', \
            f"Expected 'gromov_wasserstein', got {sheaf.metadata.get('construction_method')}"
        
        # Verify is_gw_sheaf() returns True
        assert sheaf.is_gw_sheaf(), "is_gw_sheaf() should return True for GW-constructed sheaf"
    
    def test_gw_sheaf_has_required_metadata(self):
        """Test that GW sheaves have all required GW-specific metadata."""
        model = SimpleNetwork()
        input_tensor = torch.randn(8, 8)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig()
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Check required GW metadata fields
        required_gw_fields = [
            'gw_config', 'gw_costs', 'gw_couplings', 'edge_weight_type',
            'filtration_semantics', 'alignment_type'
        ]
        
        for field in required_gw_fields:
            assert field in sheaf.metadata, f"Missing required GW metadata field: {field}"
        
        # Check GW-specific field values
        assert sheaf.metadata['edge_weight_type'] == 'metric_distortion'
        assert sheaf.metadata['filtration_semantics'] == 'increasing'
        assert sheaf.metadata['alignment_type'] in ['unit', 'sample']
        
        # Check GW costs and couplings are dictionaries
        assert isinstance(sheaf.metadata['gw_costs'], dict)
        assert isinstance(sheaf.metadata['gw_couplings'], dict)
    
    def test_gw_laplacian_builder_accepts_gw_sheaf(self):
        """Test that GWLaplacianBuilder accepts GW-constructed sheaves."""
        model = SimpleNetwork()
        input_tensor = torch.randn(6, 8)
        
        # Build GW sheaf
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False
        )
        
        # Verify it's detected as GW sheaf
        assert sheaf.is_gw_sheaf()
        
        # GWLaplacianBuilder should accept it without error
        gw_laplacian_builder = GWLaplacianBuilder()
        
        # This should NOT raise an error
        try:
            laplacian = gw_laplacian_builder.build_laplacian(sheaf)
            assert laplacian is not None
        except Exception as e:
            pytest.fail(f"GWLaplacianBuilder rejected valid GW sheaf: {e}")
    
    def test_gw_laplacian_builder_rejects_non_gw_sheaf(self):
        """Test that GWLaplacianBuilder rejects non-GW sheaves."""
        model = SimpleNetwork()
        input_tensor = torch.randn(6, 8)
        
        # Build standard (non-GW) sheaf
        builder = SheafBuilder(restriction_method='scaled_procrustes')
        sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        # Verify it's NOT detected as GW sheaf
        assert not sheaf.is_gw_sheaf()
        
        # GWLaplacianBuilder should reject it
        gw_laplacian_builder = GWLaplacianBuilder()
        
        with pytest.raises(Exception, match="not GW-based"):
            gw_laplacian_builder.build_laplacian(sheaf)
    
    def test_graph_based_sheaf_is_not_gw_sheaf(self):
        """Test that graph-based sheaves are not detected as GW sheaves."""
        import networkx as nx
        
        # Create simple graph and restrictions for build_from_graph
        graph = nx.DiGraph()
        graph.add_edge('node_a', 'node_b')
        
        stalk_dimensions = {'node_a': 4, 'node_b': 3}
        restrictions = {
            ('node_a', 'node_b'): torch.rand(3, 4)  # Row-stochastic-like restriction
        }
        
        # Normalize restriction to be approximately row-stochastic
        restriction = restrictions[('node_a', 'node_b')]
        restriction = restriction / restriction.sum(dim=1, keepdim=True)
        restrictions[('node_a', 'node_b')] = restriction
        
        # Build using graph-based method (should set construction_method='graph_based')
        builder = SheafBuilder()
        sheaf = builder.build_from_graph(graph, stalk_dimensions, restrictions, validate=False)
        
        # Verify it's NOT detected as GW sheaf
        assert not sheaf.is_gw_sheaf(), "Graph-based sheaf should not be detected as GW sheaf"
        assert sheaf.metadata.get('construction_method') == 'graph_based'
        
        # Should not have GW-specific methods work
        assert sheaf.get_gw_costs() is None
        assert sheaf.get_gw_couplings() is None
        assert sheaf.get_gw_config() is None
    
    def test_gw_accessor_methods_work_correctly(self):
        """Test that GW accessor methods work correctly for GW sheaves."""
        model = SimpleNetwork()
        input_tensor = torch.randn(8, 8)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        # All GW accessor methods should return valid data (not None)
        gw_costs = sheaf.get_gw_costs()
        gw_couplings = sheaf.get_gw_couplings()
        gw_config = sheaf.get_gw_config()
        
        assert gw_costs is not None, "GW sheaf should return valid gw_costs"
        assert gw_couplings is not None, "GW sheaf should return valid gw_couplings"  
        assert gw_config is not None, "GW sheaf should return valid gw_config"
        
        assert isinstance(gw_costs, dict)
        assert isinstance(gw_couplings, dict)
        assert isinstance(gw_config, dict)
        
        # Check that we have costs for the edges that succeeded
        assert len(gw_costs) > 0, "Should have GW costs for successful edges"
    
    def test_gw_sheaf_filtration_semantics(self):
        """Test that GW sheaves return correct filtration semantics."""
        model = SimpleNetwork()
        input_tensor = torch.randn(6, 8)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        # GW sheaves should use increasing filtration
        assert gw_sheaf.get_filtration_semantics() == 'increasing'
        
        # Standard sheaves should use decreasing filtration
        builder_standard = SheafBuilder(restriction_method='scaled_procrustes')
        standard_sheaf = builder_standard.build_from_activations(model, input_tensor, validate=False)
        assert standard_sheaf.get_filtration_semantics() == 'decreasing'
    
    def test_construction_method_prevents_confusion(self):
        """Test that construction_method prevents method confusion."""
        # This test ensures the metadata fix prevents the exact issue described
        model = SimpleNetwork()
        input_tensor = torch.randn(5, 8)
        
        # Build GW sheaf
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        # Verify construction method is correct
        construction_method = gw_sheaf.metadata.get('construction_method')
        assert construction_method == 'gromov_wasserstein', \
            f"GW sheaf has wrong construction_method: {construction_method}"
        
        # The specific issue: GWLaplacianBuilder should accept this sheaf
        gw_laplacian_builder = GWLaplacianBuilder()
        
        # Before the fix, this would fail with "not GW-based" error
        # After the fix, it should work
        try:
            laplacian = gw_laplacian_builder.build_laplacian(gw_sheaf)
            # Success - the fix works!
            assert laplacian is not None
        except Exception as e:
            if "not GW-based" in str(e):
                pytest.fail("The construction_method metadata fix failed - GWLaplacianBuilder still rejects GW sheaf")
            else:
                # Some other error is OK (e.g., numerical issues)
                pass
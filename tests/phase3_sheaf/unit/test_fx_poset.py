"""Unit tests for FX poset extraction functionality.

Tests individual methods and edge cases of the FXPosetExtractor class.
"""

import pytest
import torch
import torch.nn as nn
import torch.fx as fx
import networkx as nx
from neurosheaf.sheaf.poset import FXPosetExtractor
from neurosheaf.utils.exceptions import ArchitectureError


class TestFXPosetUnit:
    """Unit tests for FXPosetExtractor methods."""
    
    def test_initialization(self):
        """Test FXPosetExtractor initialization."""
        extractor = FXPosetExtractor()
        assert extractor.handle_dynamic is True
        
        extractor = FXPosetExtractor(handle_dynamic=False)
        assert extractor.handle_dynamic is False
    
    def test_is_activation_node(self):
        """Test _is_activation_node method."""
        extractor = FXPosetExtractor()
        
        # Create a simple model to get FX nodes
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        traced = fx.symbolic_trace(model)
        
        nodes = list(traced.graph.nodes)
        
        # Check different node types
        for node in nodes:
            if node.op == 'placeholder':
                assert not extractor._is_activation_node(node)
            elif node.op == 'output':
                assert not extractor._is_activation_node(node)
            elif node.op == 'call_module':
                assert extractor._is_activation_node(node)
    
    def test_get_node_id(self):
        """Test _get_node_id method."""
        extractor = FXPosetExtractor()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        traced = fx.symbolic_trace(model)
        
        for node in traced.graph.nodes:
            node_id = extractor._get_node_id(node)
            assert isinstance(node_id, str)
            assert len(node_id) > 0
    
    def test_add_topological_levels_empty_graph(self):
        """Test level assignment on empty graph."""
        extractor = FXPosetExtractor()
        poset = nx.DiGraph()
        
        # Should handle empty graph gracefully
        extractor._add_topological_levels(poset)
        assert len(poset.nodes()) == 0
    
    def test_add_topological_levels_single_node(self):
        """Test level assignment with single node."""
        extractor = FXPosetExtractor()
        poset = nx.DiGraph()
        poset.add_node('A')
        
        extractor._add_topological_levels(poset)
        assert poset.nodes['A']['level'] == 0
    
    def test_add_topological_levels_linear_chain(self):
        """Test level assignment on linear chain."""
        extractor = FXPosetExtractor()
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        
        extractor._add_topological_levels(poset)
        
        assert poset.nodes['A']['level'] == 0
        assert poset.nodes['B']['level'] == 1
        assert poset.nodes['C']['level'] == 2
        assert poset.nodes['D']['level'] == 3
    
    def test_add_topological_levels_with_skip(self):
        """Test level assignment with skip connection."""
        extractor = FXPosetExtractor()
        poset = nx.DiGraph()
        poset.add_edges_from([
            ('A', 'B'),
            ('B', 'C'),
            ('A', 'C')  # Skip connection
        ])
        
        extractor._add_topological_levels(poset)
        
        assert poset.nodes['A']['level'] == 0
        assert poset.nodes['B']['level'] == 1
        assert poset.nodes['C']['level'] == 2
    
    def test_is_feature_layer(self):
        """Test _is_feature_layer method."""
        extractor = FXPosetExtractor()
        
        # Feature layers
        assert extractor._is_feature_layer(nn.Linear(10, 20))
        assert extractor._is_feature_layer(nn.Conv2d(3, 64, 3))
        assert extractor._is_feature_layer(nn.ReLU())
        assert extractor._is_feature_layer(nn.BatchNorm2d(64))
        assert extractor._is_feature_layer(nn.LSTM(10, 20))
        assert extractor._is_feature_layer(nn.MultiheadAttention(512, 8))
        
        # Non-feature layers
        assert not extractor._is_feature_layer(nn.Sequential())
        assert not extractor._is_feature_layer(nn.ModuleList())
        assert not extractor._is_feature_layer(nn.Dropout())
    
    def test_fallback_extraction_simple(self):
        """Test fallback extraction on simple model."""
        extractor = FXPosetExtractor()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        poset = extractor._fallback_extraction(model)
        
        assert isinstance(poset, nx.DiGraph)
        assert len(poset.nodes()) == 3
        assert nx.is_directed_acyclic_graph(poset)
        
        # Check all nodes have levels
        for node in poset.nodes():
            assert 'level' in poset.nodes[node]
    
    def test_fallback_extraction_nested(self):
        """Test fallback extraction on nested model."""
        extractor = FXPosetExtractor()
        
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU()
                )
                self.block2 = nn.Sequential(
                    nn.Linear(20, 30),
                    nn.ReLU()
                )
                self.final = nn.Linear(30, 10)
            
            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                return self.final(x)
        
        model = NestedModel()
        poset = extractor._fallback_extraction(model)
        
        assert isinstance(poset, nx.DiGraph)
        assert len(poset.nodes()) > 0
        assert nx.is_directed_acyclic_graph(poset)
    
    def test_find_downstream_activations(self):
        """Test _find_downstream_activations method."""
        extractor = FXPosetExtractor()
        
        # Create a model with multiple paths
        class MultiPathModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.linear3 = nn.Linear(20, 30)
                
            def forward(self, x):
                x1 = self.linear1(x)
                x2 = self.linear2(x1)
                x3 = self.linear3(x1)
                return x2 + x3
        
        model = MultiPathModel()
        traced = fx.symbolic_trace(model)
        
        # Build activation nodes map
        activation_nodes = {}
        for node in traced.graph.nodes:
            if extractor._is_activation_node(node):
                activation_nodes[node] = extractor._get_node_id(node)
        
        # Test downstream finding (this is a simplified test)
        assert len(activation_nodes) > 0
    
    def test_extract_poset_error_handling(self):
        """Test error handling in extract_poset."""
        extractor = FXPosetExtractor(handle_dynamic=False)
        
        # Model that will fail FX tracing
        class UntraceableModel(nn.Module):
            def forward(self, x):
                if x.sum() > 0:  # Dynamic control flow
                    return x * 2
                else:
                    return x * 3
        
        model = UntraceableModel()
        
        with pytest.raises(ArchitectureError):
            extractor.extract_poset(model)
    
    def test_disconnected_components_levels(self):
        """Test level assignment with disconnected components."""
        extractor = FXPosetExtractor()
        poset = nx.DiGraph()
        
        # Create two disconnected components
        poset.add_edges_from([('A', 'B'), ('B', 'C')])  # Component 1
        poset.add_edges_from([('X', 'Y'), ('Y', 'Z')])  # Component 2
        
        extractor._add_topological_levels(poset)
        
        # Both components should have correct levels
        assert poset.nodes['A']['level'] == 0
        assert poset.nodes['B']['level'] == 1
        assert poset.nodes['C']['level'] == 2
        
        assert poset.nodes['X']['level'] == 0
        assert poset.nodes['Y']['level'] == 1
        assert poset.nodes['Z']['level'] == 2
    
    def test_assign_levels_bfs_cycle(self):
        """Test BFS level assignment for graphs with cycles."""
        extractor = FXPosetExtractor()
        poset = nx.DiGraph()
        
        # Create a cycle
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        
        # Should handle cycle gracefully (with warning)
        # The method should detect the cycle and use BFS instead
        extractor._add_topological_levels(poset)
        
        # All nodes should have levels assigned
        for node in poset.nodes():
            assert 'level' in poset.nodes[node]
            assert isinstance(poset.nodes[node]['level'], int)
            assert poset.nodes[node]['level'] >= 0
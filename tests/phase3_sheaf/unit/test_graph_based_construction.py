"""Unit tests for graph-based sheaf construction.

This module tests the new build_from_graph method in SheafBuilder,
which allows manual construction of sheaves from graph structure
and restriction maps for testing and research purposes.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from typing import Dict, Tuple
import sys
import os

# Add neurosheaf to path
sys.path.append('/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.builder import SheafBuilder, BuilderError
from neurosheaf.sheaf.assembly.laplacian import build_sheaf_laplacian


class TestGraphBasedSheafConstruction:
    """Test suite for graph-based sheaf construction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = SheafBuilder()
    
    def test_simple_two_vertex_construction(self):
        """Test basic two-vertex sheaf construction."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 3}
        restrictions = {('u', 'v'): torch.randn(3, 2)}
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Check basic properties
        assert isinstance(sheaf, Sheaf)
        assert len(sheaf.poset.nodes()) == 2
        assert len(sheaf.poset.edges()) == 1
        assert 'u' in sheaf.stalks and 'v' in sheaf.stalks
        assert sheaf.stalks['u'].shape == (2, 2)  # Identity matrix
        assert sheaf.stalks['v'].shape == (3, 3)  # Identity matrix
        assert ('u', 'v') in sheaf.restrictions
        assert sheaf.restrictions[('u', 'v')].shape == (3, 2)
        assert sheaf.metadata['construction_method'] == 'graph_based'
        assert sheaf.metadata['manual_construction'] == True
        assert sheaf.metadata['whitened'] == False
    
    def test_directed_graph_conversion(self):
        """Test conversion of undirected graph to directed."""
        graph = nx.Graph()
        graph.add_edges_from([('a', 'b'), ('b', 'c')])
        
        stalk_dimensions = {'a': 1, 'b': 2, 'c': 1}
        restrictions = {
            ('a', 'b'): torch.randn(2, 1),
            ('b', 'c'): torch.randn(1, 2)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should be converted to directed graph
        assert isinstance(sheaf.poset, nx.DiGraph)
        assert len(sheaf.poset.nodes()) == 3
        assert len(sheaf.poset.edges()) == 2
    
    def test_already_directed_graph(self):
        """Test handling of already directed graph."""
        graph = nx.DiGraph()
        graph.add_edges_from([('a', 'b'), ('b', 'c')])
        
        stalk_dimensions = {'a': 1, 'b': 2, 'c': 1}
        restrictions = {
            ('a', 'b'): torch.randn(2, 1),
            ('b', 'c'): torch.randn(1, 2)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should preserve directed structure
        assert isinstance(sheaf.poset, nx.DiGraph)
        assert len(sheaf.poset.nodes()) == 3
        assert len(sheaf.poset.edges()) == 2
    
    def test_star_graph_construction(self):
        """Test construction of star-shaped graph."""
        graph = nx.star_graph(4)  # Center connected to 4 periphery nodes
        
        stalk_dimensions = {0: 3, 1: 2, 2: 2, 3: 2, 4: 2}
        restrictions = {
            (0, 1): torch.randn(2, 3),
            (0, 2): torch.randn(2, 3),
            (0, 3): torch.randn(2, 3),
            (0, 4): torch.randn(2, 3)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        assert len(sheaf.poset.nodes()) == 5
        assert len(sheaf.poset.edges()) == 4
        assert all(node in sheaf.stalks for node in range(5))
        assert all(edge in sheaf.restrictions for edge in restrictions.keys())
    
    def test_validation_enabled(self):
        """Test sheaf validation when enabled."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {('u', 'v'): torch.eye(2)}
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions, validate=True)
        
        assert 'validation' in sheaf.metadata
        assert 'is_valid' in sheaf.metadata
        assert sheaf.metadata['is_valid'] == True
    
    def test_validation_disabled(self):
        """Test sheaf construction with validation disabled."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {('u', 'v'): torch.eye(2)}
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions, validate=False)
        
        assert 'validation' not in sheaf.metadata
        assert 'is_valid' not in sheaf.metadata
    
    def test_rectangular_restriction_maps(self):
        """Test handling of rectangular restriction maps."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 4, 'v': 2}
        restrictions = {('u', 'v'): torch.randn(2, 4)}  # 4D -> 2D
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        assert sheaf.stalks['u'].shape == (4, 4)
        assert sheaf.stalks['v'].shape == (2, 2)
        assert sheaf.restrictions[('u', 'v')].shape == (2, 4)
    
    def test_single_node_graph(self):
        """Test handling of single node graph."""
        graph = nx.Graph()
        graph.add_node('single')
        
        stalk_dimensions = {'single': 3}
        restrictions = {}
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        assert len(sheaf.poset.nodes()) == 1
        assert len(sheaf.poset.edges()) == 0
        assert sheaf.stalks['single'].shape == (3, 3)
        assert len(sheaf.restrictions) == 0
    
    def test_disconnected_graph(self):
        """Test handling of disconnected graph."""
        graph = nx.Graph()
        graph.add_edge('u1', 'v1')
        graph.add_edge('u2', 'v2')
        # No connection between components
        
        stalk_dimensions = {'u1': 2, 'v1': 2, 'u2': 3, 'v2': 3}
        restrictions = {
            ('u1', 'v1'): torch.eye(2),
            ('u2', 'v2'): torch.eye(3)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        assert len(sheaf.poset.nodes()) == 4
        assert len(sheaf.poset.edges()) == 2
        assert all(node in sheaf.stalks for node in stalk_dimensions.keys())
    
    def test_error_missing_stalk_dimension(self):
        """Test error when stalk dimension is missing."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2}  # Missing 'v'
        restrictions = {('u', 'v'): torch.randn(2, 2)}
        
        with pytest.raises(BuilderError, match="Missing stalk dimension for node v"):
            self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
    
    def test_error_missing_restriction_map(self):
        """Test error when restriction map is missing."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {}  # Missing restriction for edge ('u', 'v')
        
        with pytest.raises(BuilderError, match="Missing restriction map for edge"):
            self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
    
    def test_error_invalid_stalk_dimension(self):
        """Test error when stalk dimension is invalid."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 0, 'v': 2}  # Invalid dimension
        restrictions = {('u', 'v'): torch.randn(2, 2)}
        
        with pytest.raises(BuilderError, match="Stalk dimension for node u must be positive"):
            self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
    
    def test_error_wrong_restriction_dimensions(self):
        """Test error when restriction map has wrong dimensions."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 3}
        restrictions = {('u', 'v'): torch.randn(2, 2)}  # Wrong source dimension
        
        with pytest.raises(BuilderError, match="wrong target dimension"):
            self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
    
    def test_warning_extra_restriction_map(self):
        """Test warning when extra restriction map is provided."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {
            ('u', 'v'): torch.eye(2),
            ('v', 'w'): torch.eye(2)  # Extra restriction not in graph
        }
        
        # Should succeed with warning (not error)
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        assert len(sheaf.poset.edges()) == 1
        assert ('u', 'v') in sheaf.restrictions
        assert ('v', 'w') not in sheaf.restrictions
    
    def test_laplacian_construction_compatibility(self):
        """Test that graph-based sheaves work with Laplacian construction."""
        graph = nx.Graph()
        graph.add_edges_from([('a', 'b'), ('b', 'c')])
        
        stalk_dimensions = {'a': 2, 'b': 3, 'c': 2}
        restrictions = {
            ('a', 'b'): torch.randn(3, 2),
            ('b', 'c'): torch.randn(2, 3)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should be able to build Laplacian without errors
        laplacian, metadata = build_sheaf_laplacian(sheaf)
        
        expected_total_dim = sum(stalk_dimensions.values())
        assert laplacian.shape == (expected_total_dim, expected_total_dim)
        assert metadata.total_dimension == expected_total_dim
    
    def test_global_section_compatibility(self):
        """Test compatibility with global section validation."""
        # Create a sheaf with known global sections
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {('u', 'v'): torch.eye(2)}  # Identity -> 2 global sections
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Build Laplacian with unit edge weights
        edge_weights = {('u', 'v'): 1.0}
        laplacian, metadata = build_sheaf_laplacian(sheaf, edge_weights=edge_weights)
        
        # Check for 2 zero eigenvalues
        eigenvalues = np.linalg.eigvals(laplacian.toarray())
        zero_eigenvalues = eigenvalues[np.abs(eigenvalues) < 1e-12]
        
        assert len(zero_eigenvalues) == 2, f"Expected 2 zero eigenvalues, got {len(zero_eigenvalues)}"
    
    def test_complex_graph_structure(self):
        """Test with a more complex graph structure."""
        # Create a diamond graph: a -> b, a -> c, b -> d, c -> d
        graph = nx.DiGraph()
        graph.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')])
        
        stalk_dimensions = {'a': 3, 'b': 2, 'c': 2, 'd': 1}
        restrictions = {
            ('a', 'b'): torch.randn(2, 3),
            ('a', 'c'): torch.randn(2, 3),
            ('b', 'd'): torch.randn(1, 2),
            ('c', 'd'): torch.randn(1, 2)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Check structure
        assert len(sheaf.poset.nodes()) == 4
        assert len(sheaf.poset.edges()) == 4
        assert all(node in sheaf.stalks for node in stalk_dimensions.keys())
        assert all(edge in sheaf.restrictions for edge in restrictions.keys())
        
        # Should be able to build Laplacian
        laplacian, metadata = build_sheaf_laplacian(sheaf)
        assert laplacian.shape == (8, 8)  # 3 + 2 + 2 + 1 = 8
    
    def test_performance_with_large_graph(self):
        """Test performance with a larger graph."""
        # Create a path graph with 20 nodes
        graph = nx.path_graph(20)
        
        stalk_dimensions = {i: 2 for i in range(20)}
        restrictions = {
            (i, i+1): torch.randn(2, 2) for i in range(19)
        }
        
        import time
        start_time = time.time()
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        construction_time = time.time() - start_time
        
        # Should construct quickly
        assert construction_time < 1.0, f"Construction took too long: {construction_time:.2f}s"
        
        # Check structure
        assert len(sheaf.poset.nodes()) == 20
        assert len(sheaf.poset.edges()) == 19
        
        # Should be able to build Laplacian
        laplacian, metadata = build_sheaf_laplacian(sheaf)
        assert laplacian.shape == (40, 40)  # 20 nodes × 2 dimensions each
    
    def test_tensor_type_handling(self):
        """Test handling of different tensor types."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        
        # Test with different tensor types
        restrictions_float32 = {('u', 'v'): torch.randn(2, 2, dtype=torch.float32)}
        restrictions_float64 = {('u', 'v'): torch.randn(2, 2, dtype=torch.float64)}
        
        sheaf_f32 = self.builder.build_from_graph(graph, stalk_dimensions, restrictions_float32)
        sheaf_f64 = self.builder.build_from_graph(graph, stalk_dimensions, restrictions_float64)
        
        # Should handle both types
        assert sheaf_f32.restrictions[('u', 'v')].dtype == torch.float32
        assert sheaf_f64.restrictions[('u', 'v')].dtype == torch.float64
    
    def test_metadata_completeness(self):
        """Test that metadata is properly populated."""
        graph = nx.Graph()
        graph.add_edges_from([('a', 'b'), ('b', 'c')])
        
        stalk_dimensions = {'a': 2, 'b': 3, 'c': 2}
        restrictions = {
            ('a', 'b'): torch.randn(3, 2),
            ('b', 'c'): torch.randn(2, 3)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions, validate=True)
        
        # Check metadata completeness
        assert 'construction_method' in sheaf.metadata
        assert 'nodes' in sheaf.metadata
        assert 'edges' in sheaf.metadata
        assert 'whitened' in sheaf.metadata
        assert 'manual_construction' in sheaf.metadata
        assert 'validation' in sheaf.metadata
        assert 'is_valid' in sheaf.metadata
        
        assert sheaf.metadata['construction_method'] == 'graph_based'
        assert sheaf.metadata['nodes'] == 3
        assert sheaf.metadata['edges'] == 2
        assert sheaf.metadata['whitened'] == False
        assert sheaf.metadata['manual_construction'] == True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])
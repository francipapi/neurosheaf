"""Tests for deterministic indexing utilities.

This module contains comprehensive tests to verify that the canonical ordering
functions produce consistent, reproducible results across different runs and
architectures. These tests are critical for ensuring cross-architecture
comparability of numerical results.
"""

import pytest
import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple

from neurosheaf.utils.indexing import (
    canonical_node_order,
    canonical_edge_order, 
    create_node_mapping,
    create_edge_mapping,
    validate_round_trip_mapping,
    create_sheaf_indexing_metadata,
    validate_sheaf_indexing,
    get_legacy_ordering_from_metadata,
    compare_orderings
)
from neurosheaf.sheaf.data_structures import Sheaf


class TestCanonicalOrdering:
    """Test canonical ordering functions."""
    
    def test_canonical_node_order_deterministic(self):
        """Test that node ordering is deterministic across multiple calls."""
        nodes = ['layer3', 'layer1', 'layer2', 'layer10']
        
        # Call multiple times
        order1 = canonical_node_order(nodes)
        order2 = canonical_node_order(nodes)
        order3 = canonical_node_order(nodes)
        
        # Should be identical
        assert order1 == order2 == order3
        
        # Should be sorted by string
        expected = ['layer1', 'layer10', 'layer2', 'layer3']
        assert order1 == expected
    
    def test_canonical_node_order_different_input_order(self):
        """Test that output is independent of input order."""
        nodes_a = ['z', 'a', 'm', 'b']
        nodes_b = ['a', 'z', 'b', 'm'] 
        nodes_c = ['m', 'b', 'z', 'a']
        
        order_a = canonical_node_order(nodes_a)
        order_b = canonical_node_order(nodes_b)
        order_c = canonical_node_order(nodes_c)
        
        # All should produce the same sorted order
        assert order_a == order_b == order_c
        assert order_a == ['a', 'b', 'm', 'z']
    
    def test_canonical_node_order_custom_key(self):
        """Test canonical ordering with custom key function."""
        nodes = ['layer3', 'layer1', 'layer10', 'layer2']
        
        # Sort by integer value in layer name
        def layer_key(node: str) -> int:
            return int(node.replace('layer', ''))
        
        order = canonical_node_order(nodes, key=layer_key)
        expected = ['layer1', 'layer2', 'layer3', 'layer10']
        assert order == expected
    
    def test_canonical_edge_order_deterministic(self):
        """Test that edge ordering is deterministic across multiple calls."""
        edges = [('c', 'a'), ('a', 'b'), ('b', 'c'), ('a', 'c')]
        
        # Call multiple times
        order1 = canonical_edge_order(edges)
        order2 = canonical_edge_order(edges)
        order3 = canonical_edge_order(edges)
        
        # Should be identical  
        assert order1 == order2 == order3
        
        # Should be sorted by (source, target)
        expected = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'a')]
        assert order1 == expected
    
    def test_canonical_edge_order_different_input_order(self):
        """Test that output is independent of input order."""
        edges_a = [('z', 'a'), ('a', 'b'), ('b', 'z')]
        edges_b = [('b', 'z'), ('z', 'a'), ('a', 'b')]
        edges_c = [('a', 'b'), ('b', 'z'), ('z', 'a')]
        
        order_a = canonical_edge_order(edges_a)
        order_b = canonical_edge_order(edges_b)
        order_c = canonical_edge_order(edges_c)
        
        # All should produce the same sorted order
        assert order_a == order_b == order_c
        assert order_a == [('a', 'b'), ('b', 'z'), ('z', 'a')]
    
    def test_canonical_edge_order_custom_key(self):
        """Test canonical edge ordering with custom key function."""
        edges = [('layer10', 'layer2'), ('layer1', 'layer3'), ('layer2', 'layer1')]
        
        # Sort by integer values in layer names
        def edge_key(edge: Tuple[str, str]) -> Tuple[int, int]:
            src_num = int(edge[0].replace('layer', ''))
            tgt_num = int(edge[1].replace('layer', ''))
            return (src_num, tgt_num)
        
        order = canonical_edge_order(edges, key=edge_key)
        expected = [('layer1', 'layer3'), ('layer2', 'layer1'), ('layer10', 'layer2')]
        assert order == expected


class TestMappingCreation:
    """Test mapping creation and validation functions."""
    
    def test_create_node_mapping(self):
        """Test node mapping creation."""
        nodes = ['c', 'a', 'b']
        node2idx, idx2node, canonical_nodes = create_node_mapping(nodes)
        
        # Check canonical ordering
        assert canonical_nodes == ['a', 'b', 'c']
        
        # Check forward mapping
        assert node2idx == {'a': 0, 'b': 1, 'c': 2}
        
        # Check reverse mapping
        assert idx2node == {0: 'a', 1: 'b', 2: 'c'}
    
    def test_create_edge_mapping(self):
        """Test edge mapping creation."""
        edges = [('c', 'a'), ('a', 'b'), ('b', 'c')]
        edge2idx, idx2edge, canonical_edges = create_edge_mapping(edges)
        
        # Check canonical ordering
        assert canonical_edges == [('a', 'b'), ('b', 'c'), ('c', 'a')]
        
        # Check forward mapping
        expected_forward = {('a', 'b'): 0, ('b', 'c'): 1, ('c', 'a'): 2}
        assert edge2idx == expected_forward
        
        # Check reverse mapping
        expected_reverse = {0: ('a', 'b'), 1: ('b', 'c'), 2: ('c', 'a')}
        assert idx2edge == expected_reverse
    
    def test_validate_round_trip_mapping_success(self):
        """Test round-trip validation with valid mapping."""
        items = ['a', 'b', 'c']
        mapping = {'a': 0, 'b': 1, 'c': 2}
        
        # Should not raise
        validate_round_trip_mapping(mapping, items)
    
    def test_validate_round_trip_mapping_failure(self):
        """Test round-trip validation with invalid mapping."""
        items = ['a', 'b', 'c']
        
        # Wrong index mapping
        bad_mapping = {'a': 0, 'b': 2, 'c': 1}  # b and c swapped
        
        with pytest.raises(AssertionError, match="Round-trip failed"):
            validate_round_trip_mapping(bad_mapping, items)
    
    def test_validate_round_trip_mapping_size_mismatch(self):
        """Test round-trip validation with size mismatch."""
        items = ['a', 'b', 'c']
        
        # Missing item
        incomplete_mapping = {'a': 0, 'b': 1}
        
        with pytest.raises(AssertionError, match="size mismatch"):
            validate_round_trip_mapping(incomplete_mapping, items)


class TestSheafIndexingMetadata:
    """Test sheaf indexing metadata creation and validation."""
    
    def create_test_graph(self) -> nx.DiGraph:
        """Create a test graph for indexing tests."""
        G = nx.DiGraph()
        G.add_edges_from([
            ('layer2', 'layer1'),
            ('layer3', 'layer1'), 
            ('layer3', 'layer2')
        ])
        return G
    
    def test_create_sheaf_indexing_metadata(self):
        """Test creation of sheaf indexing metadata."""
        G = self.create_test_graph()
        
        metadata = create_sheaf_indexing_metadata(G)
        
        # Check required keys
        required_keys = [
            'node_order', 'edge_order', 'node2idx', 'edge2idx',
            'idx2node', 'idx2edge', 'indexing_version', 'ordering_method'
        ]
        for key in required_keys:
            assert key in metadata
        
        # Check node ordering (should be sorted)
        assert metadata['node_order'] == ['layer1', 'layer2', 'layer3']
        
        # Check edge ordering (should be sorted by (source, target))
        expected_edges = [('layer2', 'layer1'), ('layer3', 'layer1'), ('layer3', 'layer2')]
        assert metadata['edge_order'] == expected_edges
        
        # Check forward mappings
        assert metadata['node2idx'] == {'layer1': 0, 'layer2': 1, 'layer3': 2}
        assert metadata['edge2idx'] == {
            ('layer2', 'layer1'): 0,
            ('layer3', 'layer1'): 1, 
            ('layer3', 'layer2'): 2
        }
    
    def test_validate_sheaf_indexing_success(self):
        """Test successful sheaf indexing validation."""
        G = self.create_test_graph()
        
        # Create a minimal sheaf with indexing metadata
        sheaf = Sheaf()
        sheaf.poset = G
        sheaf.metadata = create_sheaf_indexing_metadata(G)
        
        # Should validate successfully
        assert validate_sheaf_indexing(sheaf) == True
    
    def test_validate_sheaf_indexing_missing_keys(self):
        """Test sheaf indexing validation with missing keys."""
        G = self.create_test_graph()
        
        # Create sheaf with incomplete metadata
        sheaf = Sheaf()
        sheaf.poset = G
        sheaf.metadata = {'node_order': ['layer1', 'layer2']}  # Missing other keys
        
        with pytest.raises(AssertionError, match="Missing indexing metadata"):
            validate_sheaf_indexing(sheaf)
    
    def test_validate_sheaf_indexing_wrong_ordering(self):
        """Test sheaf indexing validation with non-canonical ordering."""
        G = self.create_test_graph()
        
        # Create sheaf with wrong ordering
        sheaf = Sheaf()
        sheaf.poset = G
        metadata = create_sheaf_indexing_metadata(G)
        
        # Corrupt the node ordering (not canonical) and fix the mapping to match
        wrong_node_order = ['layer3', 'layer1', 'layer2']  # Wrong order
        metadata['node_order'] = wrong_node_order
        metadata['node2idx'] = {node: i for i, node in enumerate(wrong_node_order)}
        metadata['idx2node'] = {i: node for i, node in enumerate(wrong_node_order)}
        sheaf.metadata = metadata
        
        with pytest.raises(AssertionError, match="not canonical"):
            validate_sheaf_indexing(sheaf)
    
    def test_validate_sheaf_indexing_poset_mismatch(self):
        """Test sheaf indexing validation with poset mismatch."""
        G1 = self.create_test_graph()
        
        # Create different graph
        G2 = nx.DiGraph()
        G2.add_edge('different', 'nodes')
        
        # Create sheaf with G1 metadata but G2 poset
        sheaf = Sheaf()
        sheaf.poset = G2
        sheaf.metadata = create_sheaf_indexing_metadata(G1)  # Wrong metadata
        
        with pytest.raises(AssertionError, match="doesn't match poset"):
            validate_sheaf_indexing(sheaf)


class TestOrderingComparison:
    """Test ordering comparison utilities."""
    
    def test_compare_orderings_identical(self):
        """Test comparison of identical orderings."""
        nodes = ['a', 'b', 'c']
        edges = [('a', 'b'), ('b', 'c')]
        
        comparison = compare_orderings(nodes, edges, nodes, edges)
        
        assert comparison['nodes_changed'] == False
        assert comparison['edges_changed'] == False
        assert comparison['node_permutation'] is None
        assert comparison['edge_permutation'] is None
        assert comparison['same_node_set'] == True
        assert comparison['same_edge_set'] == True
    
    def test_compare_orderings_different(self):
        """Test comparison of different orderings."""
        old_nodes = ['c', 'a', 'b']
        old_edges = [('c', 'a'), ('a', 'b')]
        
        new_nodes = ['a', 'b', 'c']  # Canonical order
        new_edges = [('a', 'b'), ('c', 'a')]  # Canonical order
        
        comparison = compare_orderings(old_nodes, old_edges, new_nodes, new_edges)
        
        assert comparison['nodes_changed'] == True
        assert comparison['edges_changed'] == True
        assert comparison['same_node_set'] == True
        assert comparison['same_edge_set'] == True
        
        # Check permutation (how to reorder old to get new)
        assert comparison['node_permutation'] == [1, 2, 0]  # old[1], old[2], old[0] = a, b, c
        assert comparison['edge_permutation'] == [1, 0]  # old[1], old[0] = (a,b), (c,a)
    
    def test_get_legacy_ordering_from_metadata(self):
        """Test extraction of legacy ordering from metadata."""
        G = nx.DiGraph()
        G.add_edges_from([('b', 'a'), ('c', 'b')])
        
        sheaf = Sheaf()
        sheaf.poset = G
        sheaf.metadata = {
            'old_node_order': ['c', 'b', 'a'],  # Non-canonical
            'old_edge_order': [('c', 'b'), ('b', 'a')]  # Non-canonical
        }
        
        legacy_nodes, legacy_edges = get_legacy_ordering_from_metadata(sheaf)
        
        assert legacy_nodes == ['c', 'b', 'a']
        assert legacy_edges == [('c', 'b'), ('b', 'a')]
    
    def test_get_legacy_ordering_fallback(self):
        """Test fallback to poset ordering when no legacy metadata."""
        G = nx.DiGraph()
        G.add_edges_from([('b', 'a'), ('c', 'b')])
        
        sheaf = Sheaf()
        sheaf.poset = G
        sheaf.metadata = {}  # No legacy ordering stored
        
        legacy_nodes, legacy_edges = get_legacy_ordering_from_metadata(sheaf)
        
        # Should fall back to whatever poset.nodes()/edges() returns
        assert set(legacy_nodes) == set(G.nodes())
        assert set(legacy_edges) == set(G.edges())


class TestReproducibilityAcrossRuns:
    """Test that ordering is reproducible across multiple runs."""
    
    def test_multiple_runs_identical_results(self):
        """Test that multiple runs produce identical results."""
        # Create the same graph multiple times
        graphs = []
        for _ in range(5):
            G = nx.DiGraph()
            # Add in different orders to test determinism
            nodes = ['layer3', 'layer1', 'layer2', 'layer10']
            edges = [('layer3', 'layer1'), ('layer1', 'layer2'), ('layer10', 'layer3')]
            
            # Add nodes and edges in random order
            import random
            random.shuffle(nodes)
            random.shuffle(edges)
            
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            graphs.append(G)
        
        # Create metadata for all graphs
        metadata_list = []
        for G in graphs:
            metadata = create_sheaf_indexing_metadata(G)
            metadata_list.append(metadata)
        
        # All metadata should be identical
        first_metadata = metadata_list[0]
        for metadata in metadata_list[1:]:
            assert metadata['node_order'] == first_metadata['node_order']
            assert metadata['edge_order'] == first_metadata['edge_order'] 
            assert metadata['node2idx'] == first_metadata['node2idx']
            assert metadata['edge2idx'] == first_metadata['edge2idx']
    
    def test_hash_consistency(self):
        """Test that equivalent orderings produce same hash."""
        nodes = ['a', 'b', 'c']
        edges = [('a', 'b'), ('b', 'c')]
        
        # Create ordering multiple times
        orderings = []
        for _ in range(3):
            node_order = canonical_node_order(nodes)
            edge_order = canonical_edge_order(edges)
            orderings.append((tuple(node_order), tuple(edge_order)))
        
        # All should be identical and have same hash
        first_ordering = orderings[0]
        for ordering in orderings[1:]:
            assert ordering == first_ordering
            assert hash(ordering) == hash(first_ordering)


class TestIntegrationWithSheafBuilder:
    """Integration tests with actual sheaf construction."""
    
    def test_sheaf_creation_has_indexing_metadata(self):
        """Test that sheaf builders create indexing metadata."""
        # This would be an integration test that actually creates a sheaf
        # and verifies it has proper indexing metadata
        
        # Create a simple graph
        G = nx.DiGraph()
        G.add_edges_from([('layer2', 'layer1'), ('layer3', 'layer1')])
        
        # Create minimal sheaf manually (simulating builder output)
        sheaf = Sheaf()
        sheaf.poset = G
        sheaf.stalks = {
            'layer1': torch.eye(3),
            'layer2': torch.eye(2), 
            'layer3': torch.eye(4)
        }
        sheaf.restrictions = {
            ('layer2', 'layer1'): torch.randn(3, 2),
            ('layer3', 'layer1'): torch.randn(3, 4)
        }
        
        # Add indexing metadata (as builders should do)
        from neurosheaf.utils.indexing import create_sheaf_indexing_metadata
        indexing_metadata = create_sheaf_indexing_metadata(G)
        sheaf.metadata.update(indexing_metadata)
        
        # Verify indexing metadata is valid
        assert validate_sheaf_indexing(sheaf) == True
        
        # Check that ordering is canonical
        assert sheaf.metadata['node_order'] == ['layer1', 'layer2', 'layer3']
        assert sheaf.metadata['edge_order'] == [('layer2', 'layer1'), ('layer3', 'layer1')]


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__, '-v'])
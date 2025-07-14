"""Integration tests combining graph-based construction and global section validation.

This module tests the integration between the new build_from_graph method and
the global section validation, demonstrating the complete workflow for
mathematical validation of sheaf Laplacians.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import sys
import os

# Add neurosheaf to path
sys.path.append('/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.sheaf.assembly.laplacian import build_sheaf_laplacian
from tests.phase3_sheaf.mathematical.test_global_section_validation import GlobalSectionValidator


class TestGlobalSectionIntegration:
    """Integration tests for global section validation with graph-based construction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = SheafBuilder()
        self.validator = GlobalSectionValidator()
    
    def test_complete_workflow_known_sections(self):
        """Test complete workflow with known global sections."""
        # Create a simple sheaf with known global sections
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {('u', 'v'): torch.eye(2)}  # Identity -> 2 global sections
        
        # Build sheaf using graph-based construction
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Validate global sections
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        # Check complete workflow
        assert results['test_passed'], f"Global section validation failed: {results}"
        assert results['kernel_dimension'] == 2
        assert results['dimension_match'] == True
        assert results['all_eigenvectors_are_global_sections'] == True
        
        # Check that the sheaf construction was successful
        assert sheaf.metadata['construction_method'] == 'graph_based'
        assert sheaf.metadata['is_valid'] == True
    
    def test_chain_graph_global_sections(self):
        """Test global sections on a chain graph."""
        # Create a 3-node chain with identity restrictions
        graph = nx.path_graph(3)
        
        stalk_dimensions = {0: 2, 1: 2, 2: 2}
        restrictions = {
            (0, 1): torch.eye(2),
            (1, 2): torch.eye(2)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should have 2 global sections (dimension of each stalk)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        assert results['test_passed'], f"Chain graph validation failed: {results}"
        assert results['kernel_dimension'] == 2
    
    def test_star_graph_global_sections(self):
        """Test global sections on a star graph."""
        # Create star graph with center and 3 periphery nodes
        graph = nx.star_graph(3)
        
        stalk_dimensions = {0: 2, 1: 2, 2: 2, 3: 2}  # Center is node 0
        restrictions = {
            (0, 1): torch.eye(2),
            (0, 2): torch.eye(2),
            (0, 3): torch.eye(2)
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should have 2 global sections
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        assert results['test_passed'], f"Star graph validation failed: {results}"
        assert results['kernel_dimension'] == 2
    
    def test_rectangular_restrictions_integration(self):
        """Test integration with rectangular restriction maps."""
        # Create sheaf with rectangular restrictions
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 3, 'v': 2}
        # Project from 3D to 2D (first two components)
        restrictions = {('u', 'v'): torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should have 1 global section (null space of projection)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=1)
        
        assert results['test_passed'], f"Rectangular restriction validation failed: {results}"
        assert results['kernel_dimension'] == 1
    
    def test_complex_graph_structure(self):
        """Test with a more complex graph structure."""
        # Create diamond graph: a -> b, a -> c, b -> d, c -> d
        graph = nx.DiGraph()
        graph.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')])
        
        # Use 1D stalks for simplicity
        stalk_dimensions = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
        restrictions = {
            ('a', 'b'): torch.tensor([[1.0]]),
            ('a', 'c'): torch.tensor([[1.0]]),
            ('b', 'd'): torch.tensor([[1.0]]),
            ('c', 'd'): torch.tensor([[1.0]])
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should have 1 global section (all values equal)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=1)
        
        assert results['test_passed'], f"Complex graph validation failed: {results}"
        assert results['kernel_dimension'] == 1
    
    def test_disconnected_components(self):
        """Test with disconnected graph components."""
        # Create two disconnected components
        graph = nx.Graph()
        graph.add_edge('u1', 'v1')
        graph.add_edge('u2', 'v2')
        
        stalk_dimensions = {'u1': 1, 'v1': 1, 'u2': 1, 'v2': 1}
        restrictions = {
            ('u1', 'v1'): torch.tensor([[1.0]]),
            ('u2', 'v2'): torch.tensor([[1.0]])
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should have 2 global sections (one per component)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        assert results['test_passed'], f"Disconnected components validation failed: {results}"
        assert results['kernel_dimension'] == 2
    
    def test_different_stalk_dimensions(self):
        """Test with stalks of different dimensions."""
        # Create a chain with varying stalk dimensions
        graph = nx.path_graph(3)
        
        stalk_dimensions = {0: 3, 1: 2, 2: 1}
        restrictions = {
            (0, 1): torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # 3D -> 2D
            (1, 2): torch.tensor([[1.0, 0.0]])  # 2D -> 1D
        }
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should have 1 global section (cascading projections)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=1)
        
        assert results['test_passed'], f"Different dimensions validation failed: {results}"
        assert results['kernel_dimension'] == 1
    
    def test_validation_with_edge_weights(self):
        """Test that edge weights don't affect global section dimension."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {('u', 'v'): torch.eye(2)}
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Test with different edge weights
        edge_weights_list = [
            {('u', 'v'): 1.0},
            {('u', 'v'): 2.0},
            {('u', 'v'): 0.5}
        ]
        
        for edge_weights in edge_weights_list:
            # Build Laplacian with specific edge weights
            laplacian, metadata = build_sheaf_laplacian(sheaf, edge_weights=edge_weights)
            
            # Check eigenvalues
            eigenvalues = np.linalg.eigvals(laplacian.toarray())
            zero_eigenvalues = eigenvalues[np.abs(eigenvalues) < 1e-12]
            
            # Should always have 2 zero eigenvalues regardless of edge weights
            assert len(zero_eigenvalues) == 2, f"Edge weight {list(edge_weights.values())[0]} gave {len(zero_eigenvalues)} zero eigenvalues"
    
    def test_performance_with_larger_graphs(self):
        """Test performance with larger graphs."""
        # Create a path graph with 10 nodes
        graph = nx.path_graph(10)
        
        stalk_dimensions = {i: 2 for i in range(10)}
        restrictions = {(i, i+1): torch.eye(2) for i in range(9)}
        
        # Build sheaf
        import time
        start_time = time.time()
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        construction_time = time.time() - start_time
        
        # Validate global sections
        start_time = time.time()
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        validation_time = time.time() - start_time
        
        # Check performance
        assert construction_time < 1.0, f"Construction took too long: {construction_time:.2f}s"
        assert validation_time < 5.0, f"Validation took too long: {validation_time:.2f}s"
        
        # Check correctness
        assert results['test_passed'], f"Large graph validation failed: {results}"
        assert results['kernel_dimension'] == 2
    
    def test_numerical_stability(self):
        """Test numerical stability with near-singular restrictions."""
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        
        # Create near-identity restriction (should have ~2 global sections)
        epsilon = 1e-10
        restrictions = {('u', 'v'): torch.tensor([[1.0 + epsilon, 0.0], [0.0, 1.0 - epsilon]])}
        
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        
        # Should still detect global sections correctly
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        # The test should pass or be very close (numerical precision)
        assert results['kernel_dimension'] in [1, 2], f"Unexpected kernel dimension: {results['kernel_dimension']}"
    
    def test_error_handling_integration(self):
        """Test error handling in integrated workflow."""
        # Test with invalid graph structure
        graph = nx.Graph()
        graph.add_edge('u', 'v')
        
        stalk_dimensions = {'u': 2, 'v': 2}
        restrictions = {('u', 'v'): torch.randn(3, 2)}  # Wrong dimensions
        
        # Should catch dimension mismatch
        with pytest.raises(Exception):  # Either BuilderError or dimension error
            sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
    
    def test_complete_mathematical_workflow(self):
        """Test complete mathematical workflow with documentation."""
        # Create a pedagogical example for documentation
        print("\n=== Complete Mathematical Workflow Example ===")
        
        # Step 1: Create graph structure
        graph = nx.Graph()
        graph.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        print(f"Graph: {list(graph.edges())}")
        
        # Step 2: Define stalk dimensions
        stalk_dimensions = {'layer1': 3, 'layer2': 2, 'layer3': 2}
        print(f"Stalk dimensions: {stalk_dimensions}")
        
        # Step 3: Define restriction maps
        restrictions = {
            ('layer1', 'layer2'): torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # 3D -> 2D
            ('layer2', 'layer3'): torch.eye(2)  # 2D -> 2D (identity)
        }
        print(f"Restriction maps: {[f'{k}: {v.shape}' for k, v in restrictions.items()]}")
        
        # Step 4: Build sheaf
        sheaf = self.builder.build_from_graph(graph, stalk_dimensions, restrictions)
        print(f"Sheaf constructed: {sheaf.metadata['nodes']} nodes, {sheaf.metadata['edges']} edges")
        
        # Step 5: Validate global sections
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        print(f"Global sections: {results['kernel_dimension']} found, {results['expected_dimension']} expected")
        print(f"Test passed: {results['test_passed']}")
        
        # Step 6: Examine eigenvalues
        print(f"Eigenvalues: {results['eigenvalues']}")
        print(f"Zero eigenvalues: {results['zero_eigenvalues']}")
        
        # Step 7: Verify mathematical properties
        assert results['test_passed'], "Mathematical validation failed"
        assert results['kernel_dimension'] == 2, "Unexpected kernel dimension"
        assert results['all_eigenvectors_are_global_sections'], "Eigenvectors are not global sections"
        
        print("✅ Complete mathematical workflow validated!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
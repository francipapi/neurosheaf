"""Global Section Test for Sheaf Laplacian Validation.

This module implements the most rigorous mathematical test for the sheaf Laplacian:
verifying that the dimension of the kernel (zero eigenvalue space) equals the
dimension of the space of global sections.

Mathematical Foundation:
========================

For a cellular sheaf F on a graph G, a global section is a choice of data
x_v ∈ F(v) for each vertex v such that the restriction maps are compatible:
F_u→e(x_u) = F_v→e(x_v) for every edge e = (u,v).

The space of global sections is the kernel of the coboundary operator δ:
Global sections = ker(δ) = {x : δx = 0}

Since the Laplacian Δ = δ^T δ, we have:
- If δx = 0, then Δx = δ^T δx = 0
- Therefore: dim(ker(Δ)) ≥ dim(ker(δ)) = dim(Global sections)

For the degree-0 Laplacian (vertex-to-vertex), equality holds:
dim(ker(Δ)) = dim(Global sections)

Test Strategy:
==============

1. Construct sheaves with known global sections
2. Build the Laplacian using our implementation  
3. Compute eigenvalues and identify zero eigenvalues
4. Verify that dim(kernel) = dim(global_sections)
5. Validate that zero eigenvectors are actual global sections

"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.linalg import null_space
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add neurosheaf to path
sys.path.append('/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.laplacian import build_sheaf_laplacian


class GlobalSectionValidator:
    """Validator for global section properties of sheaf Laplacians."""
    
    def __init__(self, eigenvalue_tolerance: float = 1e-12, 
                 section_tolerance: float = 1e-10):
        """Initialize validator with specified tolerances.
        
        Args:
            eigenvalue_tolerance: Tolerance for identifying zero eigenvalues
            section_tolerance: Tolerance for validating global section properties
        """
        self.eigenvalue_tolerance = eigenvalue_tolerance
        self.section_tolerance = section_tolerance
    
    def validate_global_sections(self, sheaf: Sheaf, 
                               expected_dimension: int) -> Dict[str, any]:
        """Validate that Laplacian kernel dimension equals global sections dimension.
        
        Args:
            sheaf: Sheaf object to validate
            expected_dimension: Expected dimension of global sections space
            
        Returns:
            Dictionary with detailed validation results
        """
        # Build Laplacian with unit edge weights for mathematical clarity
        edge_weights = {edge: 1.0 for edge in sheaf.restrictions.keys()}
        laplacian, metadata = build_sheaf_laplacian(sheaf, edge_weights=edge_weights, validate=False)
        
        # Compute eigenvalues
        if laplacian.shape[0] > 1:
            eigenvalues, eigenvectors = eigsh(laplacian, k=min(10, laplacian.shape[0]-1), 
                                            which='SA', return_eigenvectors=True)
        else:
            eigenvalues = np.array([laplacian.toarray()[0, 0]])
            eigenvectors = np.array([[1.0]])
        
        # Identify zero eigenvalues
        zero_eigenvalue_mask = np.abs(eigenvalues) < self.eigenvalue_tolerance
        zero_eigenvalues = eigenvalues[zero_eigenvalue_mask]
        kernel_dimension = len(zero_eigenvalues)
        
        # Get zero eigenvectors
        zero_eigenvectors = eigenvectors[:, zero_eigenvalue_mask]
        
        # Validate each zero eigenvector is a global section
        global_section_validation = []
        for i in range(kernel_dimension):
            eigenvector = zero_eigenvectors[:, i]
            is_global_section = self._validate_eigenvector_is_global_section(
                sheaf, eigenvector, metadata
            )
            global_section_validation.append(is_global_section)
        
        # Compute actual global sections using restriction map analysis
        actual_global_sections = self._compute_global_sections_analytically(sheaf)
        
        return {
            'laplacian_shape': laplacian.shape,
            'eigenvalues': eigenvalues,
            'zero_eigenvalues': zero_eigenvalues,
            'kernel_dimension': kernel_dimension,
            'expected_dimension': expected_dimension,
            'dimension_match': kernel_dimension == expected_dimension,
            'zero_eigenvectors': zero_eigenvectors,
            'global_section_validation': global_section_validation,
            'all_eigenvectors_are_global_sections': all(global_section_validation),
            'actual_global_sections': actual_global_sections,
            'test_passed': (kernel_dimension == expected_dimension and 
                          all(global_section_validation))
        }
    
    def _validate_eigenvector_is_global_section(self, sheaf: Sheaf, 
                                              eigenvector: np.ndarray,
                                              metadata) -> bool:
        """Validate that an eigenvector satisfies global section property."""
        # Extract vertex data from eigenvector using metadata offsets
        vertex_data = {}
        for node, offset in metadata.stalk_offsets.items():
            dim = metadata.stalk_dimensions[node]
            vertex_data[node] = eigenvector[offset:offset+dim]
        
        # Check restriction compatibility for each edge
        for edge, restriction in sheaf.restrictions.items():
            u, v = edge
            
            if u not in vertex_data or v not in vertex_data:
                continue
                
            # Get restriction map R: u → v
            R = restriction.detach().cpu().numpy() if isinstance(restriction, torch.Tensor) else restriction
            
            # For a global section, we need F_u→e(x_u) = F_v→e(x_v)
            # With our stored restriction R: u → v, this becomes:
            # We need to check if the restriction is compatible with a global section
            
            # Get vertex data
            x_u = vertex_data[u]
            x_v = vertex_data[v]
            
            # Check if R @ x_u ≈ x_v (within tolerance)
            # This is the compatibility condition for our implementation
            residual = R @ x_u - x_v
            error = np.linalg.norm(residual)
            
            if error > self.section_tolerance:
                return False
        
        return True
    
    def _compute_global_sections_analytically(self, sheaf: Sheaf) -> Dict[str, any]:
        """Compute global sections analytically using restriction map constraints."""
        # For small sheaves, we can solve the linear system directly
        # Global sections satisfy: R @ x_u = x_v for all edges (u,v)
        
        nodes = list(sheaf.poset.nodes())
        node_to_index = {node: i for i, node in enumerate(nodes)}
        
        total_dim = sum(sheaf.stalks[node].shape[0] for node in nodes)
        
        # For very simple cases, we can compute analytically
        # For more complex cases, we rely on the eigenvalue analysis
        
        # Simplified analytical computation for basic validation
        if len(nodes) <= 2 and len(sheaf.restrictions) == 1:
            # Simple two-node case
            edge = list(sheaf.restrictions.keys())[0]
            u, v = edge
            R = sheaf.restrictions[edge]
            R_np = R.detach().cpu().numpy() if isinstance(R, torch.Tensor) else R
            
            # Check if R is square identity (full compatibility)
            if (R_np.shape[0] == R_np.shape[1] and 
                np.allclose(R_np, np.eye(R_np.shape[0]))):
                analytical_dimension = R_np.shape[0]  # All vectors are global sections
            else:
                # For general case, compute null space of constraint matrix
                try:
                    # Create constraint matrix [R, -I] where I matches the target dimension
                    I_target = np.eye(R_np.shape[0])
                    constraint_matrix = np.hstack([R_np, -I_target])
                    null_vectors = null_space(constraint_matrix)
                    analytical_dimension = null_vectors.shape[1]
                except:
                    analytical_dimension = -1  # Could not compute
        else:
            # For complex cases, we skip analytical computation
            analytical_dimension = -1
            
        return {
            'analytical_dimension': analytical_dimension,
            'total_dimension': total_dim,
            'num_constraints': len(sheaf.restrictions)
        }


class TestGlobalSectionValidation:
    """Test suite for global section validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = GlobalSectionValidator()
    
    def test_two_vertex_identical_maps(self):
        """Test Case 1: Two vertices with identical restriction maps → 1 global section."""
        # Create simple two-vertex graph
        poset = nx.DiGraph()
        poset.add_edge('u', 'v')
        
        # Create 1D stalks
        stalks = {
            'u': torch.eye(1),
            'v': torch.eye(1)
        }
        
        # Create identical restriction maps F_u→e = F_v→e = [1]
        # This means any vector [a, a] is a global section
        restrictions = {
            ('u', 'v'): torch.tensor([[1.0]])  # Maps u to v with identity
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Validate: Should have exactly 1 global section
        results = self.validator.validate_global_sections(sheaf, expected_dimension=1)
        
        assert results['test_passed'], f"Test failed: {results}"
        assert results['kernel_dimension'] == 1, f"Expected 1 global section, got {results['kernel_dimension']}"
        assert results['dimension_match'], "Kernel dimension doesn't match expected"
        
        # The global section should be proportional to [1, 1]
        zero_eigenvector = results['zero_eigenvectors'][:, 0]
        expected_pattern = np.array([1.0, 1.0])
        expected_pattern = expected_pattern / np.linalg.norm(expected_pattern)
        actual_pattern = zero_eigenvector / np.linalg.norm(zero_eigenvector)
        
        # Check if patterns match (up to sign)
        pattern_match = (np.allclose(actual_pattern, expected_pattern, atol=1e-10) or
                        np.allclose(actual_pattern, -expected_pattern, atol=1e-10))
        
        assert pattern_match, f"Global section pattern incorrect: got {actual_pattern}, expected ±{expected_pattern}"
    
    def test_three_vertex_chain_compatible(self):
        """Test Case 2: Three-vertex chain with compatible restrictions → 1 global section."""
        # Create three-vertex chain: u → v → w
        poset = nx.DiGraph()
        poset.add_edge('u', 'v')
        poset.add_edge('v', 'w')
        
        # Create 2D stalks
        stalks = {
            'u': torch.eye(2),
            'v': torch.eye(2),
            'w': torch.eye(2)
        }
        
        # Create compatible restriction maps
        # For a global section [x, x, x], we need: R_uv @ x = x and R_vw @ x = x
        # So both restrictions should be identity
        restrictions = {
            ('u', 'v'): torch.eye(2),
            ('v', 'w'): torch.eye(2)
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Validate: Should have exactly 2 global sections (since stalks are 2D)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        assert results['test_passed'], f"Test failed: {results}"
        assert results['kernel_dimension'] == 2, f"Expected 2 global sections, got {results['kernel_dimension']}"
    
    def test_two_vertex_incompatible_maps(self):
        """Test Case 3: Two vertices with incompatible restrictions → 0 global sections."""
        # Create two-vertex graph
        poset = nx.DiGraph()
        poset.add_edge('u', 'v')
        
        # Create 2D stalks
        stalks = {
            'u': torch.eye(2),
            'v': torch.eye(2)
        }
        
        # Create incompatible restriction map (no global sections possible)
        # Use a non-identity, non-zero map
        restrictions = {
            ('u', 'v'): torch.tensor([[1.0, 0.0], [0.0, -1.0]])  # Flips second component
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Validate: Should have 0 global sections
        results = self.validator.validate_global_sections(sheaf, expected_dimension=0)
        
        assert results['test_passed'], f"Test failed: {results}"
        assert results['kernel_dimension'] == 0, f"Expected 0 global sections, got {results['kernel_dimension']}"
    
    def test_rectangular_restrictions(self):
        """Test Case 4: Rectangular restriction maps between different dimensions."""
        # Create two-vertex graph with different stalk dimensions
        poset = nx.DiGraph()
        poset.add_edge('u', 'v')
        
        # Create stalks of different dimensions
        stalks = {
            'u': torch.eye(3),  # 3D
            'v': torch.eye(2)   # 2D
        }
        
        # Create rectangular restriction map 2×3
        # Global section must satisfy: R @ x_u = x_v
        # Choose R such that global sections exist
        restrictions = {
            ('u', 'v'): torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Projects to first 2 components
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Validate: Should have 1 global section (null space of projection)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=1)
        
        assert results['test_passed'], f"Test failed: {results}"
        assert results['kernel_dimension'] == 1, f"Expected 1 global section, got {results['kernel_dimension']}"
    
    def test_star_graph_multiple_sections(self):
        """Test Case 5: Star graph with multiple global sections."""
        # Create star graph: center connected to 3 periphery vertices
        poset = nx.DiGraph()
        poset.add_edge('center', 'a')
        poset.add_edge('center', 'b')
        poset.add_edge('center', 'c')
        
        # Create 2D stalks
        stalks = {
            'center': torch.eye(2),
            'a': torch.eye(2),
            'b': torch.eye(2),
            'c': torch.eye(2)
        }
        
        # Create identical restriction maps (all identity)
        # Global sections: [x, x, x, x] for any x ∈ R^2
        restrictions = {
            ('center', 'a'): torch.eye(2),
            ('center', 'b'): torch.eye(2),
            ('center', 'c'): torch.eye(2)
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Validate: Should have 2 global sections
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        assert results['test_passed'], f"Test failed: {results}"
        assert results['kernel_dimension'] == 2, f"Expected 2 global sections, got {results['kernel_dimension']}"
    
    def test_complex_graph_structure(self):
        """Test Case 6: Complex graph with branching and merging."""
        # Create diamond graph: u → v, u → w, v → x, w → x
        poset = nx.DiGraph()
        poset.add_edge('u', 'v')
        poset.add_edge('u', 'w')
        poset.add_edge('v', 'x')
        poset.add_edge('w', 'x')
        
        # Create 1D stalks
        stalks = {
            'u': torch.eye(1),
            'v': torch.eye(1),
            'w': torch.eye(1),
            'x': torch.eye(1)
        }
        
        # Create identity restrictions (compatible)
        restrictions = {
            ('u', 'v'): torch.tensor([[1.0]]),
            ('u', 'w'): torch.tensor([[1.0]]),
            ('v', 'x'): torch.tensor([[1.0]]),
            ('w', 'x'): torch.tensor([[1.0]])
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Validate: Should have 1 global section
        results = self.validator.validate_global_sections(sheaf, expected_dimension=1)
        
        assert results['test_passed'], f"Test failed: {results}"
        assert results['kernel_dimension'] == 1, f"Expected 1 global section, got {results['kernel_dimension']}"
    
    def test_disconnected_graph(self):
        """Test Case 7: Disconnected graph → multiple global sections."""
        # Create two disconnected components
        poset = nx.DiGraph()
        poset.add_edge('u1', 'v1')  # Component 1
        poset.add_edge('u2', 'v2')  # Component 2
        
        # Create 1D stalks
        stalks = {
            'u1': torch.eye(1),
            'v1': torch.eye(1),
            'u2': torch.eye(1),
            'v2': torch.eye(1)
        }
        
        # Create identity restrictions within each component
        restrictions = {
            ('u1', 'v1'): torch.tensor([[1.0]]),
            ('u2', 'v2'): torch.tensor([[1.0]])
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Validate: Should have 2 global sections (one per component)
        results = self.validator.validate_global_sections(sheaf, expected_dimension=2)
        
        assert results['test_passed'], f"Test failed: {results}"
        assert results['kernel_dimension'] == 2, f"Expected 2 global sections, got {results['kernel_dimension']}"
    
    def test_edge_cases_and_robustness(self):
        """Test edge cases and numerical robustness."""
        # Test single vertex (trivial case)
        poset_single = nx.DiGraph()
        poset_single.add_node('single')
        
        stalks_single = {'single': torch.eye(3)}
        restrictions_single = {}
        
        sheaf_single = Sheaf(poset=poset_single, stalks=stalks_single, restrictions=restrictions_single)
        
        # Single vertex should have dim(stalk) global sections
        results_single = self.validator.validate_global_sections(sheaf_single, expected_dimension=3)
        assert results_single['test_passed'], f"Single vertex test failed: {results_single}"
        
        # Test numerical stability with small eigenvalues
        poset_small = nx.DiGraph()
        poset_small.add_edge('u', 'v')
        
        stalks_small = {
            'u': torch.eye(2),
            'v': torch.eye(2)
        }
        
        # Create near-identity restriction (should have near-zero eigenvalues)
        epsilon = 1e-14
        restrictions_small = {
            ('u', 'v'): torch.tensor([[1.0 + epsilon, 0.0], [0.0, 1.0 - epsilon]])
        }
        
        sheaf_small = Sheaf(poset=poset_small, stalks=stalks_small, restrictions=restrictions_small)
        
        # Should still detect global sections correctly
        results_small = self.validator.validate_global_sections(sheaf_small, expected_dimension=0)
        assert results_small['kernel_dimension'] == 0, f"Numerical stability test failed: {results_small}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])
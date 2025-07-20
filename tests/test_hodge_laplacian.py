"""Integration tests for Hodge Laplacian construction with eigenvalue preservation.

This module tests the Phase 2 implementation:
- SheafLaplacianBuilder with Hodge formulation
- SheafBuilder with eigenvalue preservation
- Mathematical properties of Hodge Laplacians
- Integration between eigenvalue preservation and Laplacian construction
"""

import torch
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from neurosheaf.sheaf.data_structures import Sheaf, EigenvalueMetadata
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.sheaf.assembly.laplacian import SheafLaplacianBuilder, LaplacianMetadata
from neurosheaf.sheaf.core.whitening import WhiteningProcessor


class TestHodgeLaplacianConstruction:
    """Test Hodge Laplacian construction with eigenvalue preservation."""
    
    @pytest.fixture
    def simple_eigenvalue_sheaf(self):
        """Create a simple sheaf with eigenvalue preservation for testing."""
        import networkx as nx
        
        # Create simple poset: A -> B -> C
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C')])
        
        # Create eigenvalue-preserving stalks (non-identity diagonal matrices)
        stalks = {
            'A': torch.diag(torch.tensor([3.0, 2.0], dtype=torch.float32)),
            'B': torch.diag(torch.tensor([2.5, 1.5, 1.0], dtype=torch.float32)),
            'C': torch.diag(torch.tensor([2.0, 1.8], dtype=torch.float32))
        }
        
        # Create restriction maps
        restrictions = {
            ('A', 'B'): torch.tensor([[0.8, 0.3], [0.2, 0.9], [0.1, 0.4]], dtype=torch.float32),
            ('B', 'C'): torch.tensor([[0.7, 0.5, 0.2], [0.6, 0.3, 0.8]], dtype=torch.float32)
        }
        
        # Create eigenvalue metadata
        eigenvalue_metadata = EigenvalueMetadata(
            preserve_eigenvalues=True,
            hodge_formulation_active=True
        )
        eigenvalue_metadata.eigenvalue_matrices = stalks.copy()
        eigenvalue_metadata.condition_numbers = {'A': 1.5, 'B': 2.5, 'C': 1.11}
        eigenvalue_metadata.regularization_applied = {'A': False, 'B': False, 'C': False}
        
        return Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            eigenvalue_metadata=eigenvalue_metadata,
            metadata={'preserve_eigenvalues': True}
        )
    
    @pytest.fixture
    def standard_sheaf(self):
        """Create a standard sheaf (identity stalks) for comparison."""
        import networkx as nx
        
        # Same structure as eigenvalue sheaf but with identity matrices
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C')])
        
        stalks = {
            'A': torch.eye(2, dtype=torch.float32),
            'B': torch.eye(3, dtype=torch.float32),
            'C': torch.eye(2, dtype=torch.float32)
        }
        
        restrictions = {
            ('A', 'B'): torch.tensor([[0.8, 0.3], [0.2, 0.9], [0.1, 0.4]], dtype=torch.float32),
            ('B', 'C'): torch.tensor([[0.7, 0.5, 0.2], [0.6, 0.3, 0.8]], dtype=torch.float32)
        }
        
        return Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            eigenvalue_metadata=None,
            metadata={'preserve_eigenvalues': False}
        )
    
    def test_eigenvalue_detection(self, simple_eigenvalue_sheaf, standard_sheaf):
        """Test that eigenvalue preservation is correctly detected."""
        builder = SheafLaplacianBuilder()
        
        # Should detect eigenvalue preservation
        assert builder._uses_eigenvalue_preservation(simple_eigenvalue_sheaf) == True
        
        # Should not detect eigenvalue preservation  
        assert builder._uses_eigenvalue_preservation(standard_sheaf) == False
    
    def test_hodge_vs_standard_construction(self, simple_eigenvalue_sheaf, standard_sheaf):
        """Test that appropriate construction method is chosen."""
        builder = SheafLaplacianBuilder()
        
        # Test eigenvalue-preserving sheaf uses Hodge formulation
        L_hodge, metadata_hodge = builder.build(simple_eigenvalue_sheaf)
        assert metadata_hodge.construction_method == "hodge_formulation"
        assert isinstance(L_hodge, csr_matrix)
        
        # Test standard sheaf uses standard formulation
        L_standard, metadata_standard = builder.build(standard_sheaf)
        assert metadata_standard.construction_method == "standard"
        assert isinstance(L_standard, csr_matrix)
        
        # Dimensions should match
        assert L_hodge.shape == L_standard.shape
    
    def test_hodge_laplacian_symmetry(self, simple_eigenvalue_sheaf):
        """Test that Hodge Laplacian is symmetric."""
        builder = SheafLaplacianBuilder()
        L, metadata = builder.build(simple_eigenvalue_sheaf)
        
        # Convert to dense for testing
        L_dense = L.toarray()
        
        # Check symmetry: L = L^T
        symmetry_error = np.linalg.norm(L_dense - L_dense.T, 'fro')
        assert symmetry_error < 1e-10, f"Hodge Laplacian not symmetric: error = {symmetry_error}"
    
    def test_hodge_laplacian_positive_semidefinite(self, simple_eigenvalue_sheaf):
        """Test that Hodge Laplacian is positive semi-definite."""
        builder = SheafLaplacianBuilder()
        L, metadata = builder.build(simple_eigenvalue_sheaf)
        
        # Convert to dense for eigenvalue computation
        L_dense = L.toarray()
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(L_dense)
        min_eigenval = np.min(eigenvals.real)
        
        # Should be positive semi-definite (relaxed tolerance for numerical precision)
        assert min_eigenval >= -1e-6, f"Hodge Laplacian not PSD: min eigenvalue = {min_eigenval}"
    
    def test_hodge_laplacian_mathematical_structure(self, simple_eigenvalue_sheaf):
        """Test mathematical structure of Hodge Laplacian blocks."""
        builder = SheafLaplacianBuilder()
        L, metadata = builder.build(simple_eigenvalue_sheaf)
        
        # Get stalk dimensions and offsets
        offsets = metadata.stalk_offsets
        dimensions = metadata.stalk_dimensions
        
        # Convert to dense for block analysis
        L_dense = L.toarray()
        
        # Test that diagonal blocks are non-zero (should have eigenvalue contributions)
        for node, offset in offsets.items():
            dim = dimensions[node]
            diagonal_block = L_dense[offset:offset+dim, offset:offset+dim]
            
            # Diagonal blocks should be non-zero for eigenvalue-preserving case
            assert np.any(diagonal_block != 0), f"Diagonal block for {node} is zero"
            
            # Diagonal blocks should be symmetric
            block_symmetry_error = np.linalg.norm(diagonal_block - diagonal_block.T, 'fro')
            assert block_symmetry_error < 1e-10, f"Diagonal block for {node} not symmetric"
    
    def test_eigenvalue_preservation_vs_standard_differences(self, simple_eigenvalue_sheaf, standard_sheaf):
        """Test that eigenvalue preservation produces different results from standard approach."""
        builder = SheafLaplacianBuilder()
        
        # Build both Laplacians
        L_hodge, _ = builder.build(simple_eigenvalue_sheaf)
        L_standard, _ = builder.build(standard_sheaf)
        
        # Convert to dense
        L_hodge_dense = L_hodge.toarray()
        L_standard_dense = L_standard.toarray()
        
        # They should be different (eigenvalue preservation should matter)
        difference = np.linalg.norm(L_hodge_dense - L_standard_dense, 'fro')
        assert difference > 1e-6, "Hodge and standard Laplacians should be different"
    
    def test_backward_compatibility(self, standard_sheaf):
        """Test that preserve_eigenvalues=False maintains backward compatibility."""
        builder = SheafLaplacianBuilder()
        
        # Build Laplacian with standard sheaf
        L, metadata = builder.build(standard_sheaf)
        
        # Should use standard construction method
        assert metadata.construction_method == "standard"
        
        # Should be symmetric and PSD
        L_dense = L.toarray()
        symmetry_error = np.linalg.norm(L_dense - L_dense.T, 'fro')
        assert symmetry_error < 1e-10
        
        eigenvals = np.linalg.eigvals(L_dense)
        assert np.min(eigenvals.real) >= -1e-10


class TestSheafBuilderIntegration:
    """Test SheafBuilder integration with eigenvalue preservation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        import torch.nn as nn
        
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(4, 3)
                self.layer2 = nn.Linear(3, 2)
                self.activation = nn.ReLU()
            
            def forward(self, x):
                x = self.activation(self.layer1(x))
                x = self.layer2(x)
                return x
        
        return SimpleNet()
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        torch.manual_seed(42)
        return torch.randn(8, 4)  # Small batch for testing
    
    def test_sheaf_builder_eigenvalue_preservation(self, simple_model, sample_input):
        """Test SheafBuilder with eigenvalue preservation enabled."""
        # Test with eigenvalue preservation
        builder_eigen = SheafBuilder(preserve_eigenvalues=True)
        sheaf_eigen = builder_eigen.build_from_activations(simple_model, sample_input)
        
        # Should have eigenvalue metadata
        assert sheaf_eigen.eigenvalue_metadata is not None
        assert sheaf_eigen.eigenvalue_metadata.preserve_eigenvalues == True
        assert sheaf_eigen.eigenvalue_metadata.hodge_formulation_active == True
        assert len(sheaf_eigen.eigenvalue_metadata.eigenvalue_matrices) > 0
        
        # Stalks should be eigenvalue diagonal matrices (not identity)
        for node, stalk in sheaf_eigen.stalks.items():
            # Should be diagonal
            assert torch.allclose(stalk, torch.diag(torch.diag(stalk)), atol=1e-6)
            # Should not be identity  
            identity = torch.eye(stalk.shape[0], dtype=stalk.dtype)
            assert not torch.allclose(stalk, identity, atol=1e-3)
    
    def test_sheaf_builder_standard_mode(self, simple_model, sample_input):
        """Test SheafBuilder with standard (identity) whitening."""
        # Test without eigenvalue preservation
        builder_standard = SheafBuilder(preserve_eigenvalues=False)
        sheaf_standard = builder_standard.build_from_activations(simple_model, sample_input)
        
        # Should not have eigenvalue metadata
        assert sheaf_standard.eigenvalue_metadata is None
        
        # Stalks should be identity matrices
        for node, stalk in sheaf_standard.stalks.items():
            identity = torch.eye(stalk.shape[0], dtype=stalk.dtype)
            assert torch.allclose(stalk, identity, atol=1e-6)
    
    def test_end_to_end_laplacian_construction(self, simple_model, sample_input):
        """Test end-to-end: SheafBuilder + SheafLaplacianBuilder."""
        # Build eigenvalue-preserving sheaf
        builder = SheafBuilder(preserve_eigenvalues=True)
        sheaf = builder.build_from_activations(simple_model, sample_input)
        
        # Build Hodge Laplacian
        laplacian_builder = SheafLaplacianBuilder()
        L, metadata = laplacian_builder.build(sheaf)
        
        # Should use Hodge formulation
        assert metadata.construction_method == "hodge_formulation"
        
        # Should be symmetric and PSD
        L_dense = L.toarray()
        
        # Test symmetry (relaxed tolerance for numerical precision)
        symmetry_error = np.linalg.norm(L_dense - L_dense.T, 'fro')
        assert symmetry_error < 1e-6, f"End-to-end Laplacian not symmetric: {symmetry_error}"
        
        # Test positive semi-definiteness (relaxed tolerance for numerical precision)
        eigenvals = np.linalg.eigvals(L_dense)
        min_eigenval = np.min(eigenvals.real)
        assert min_eigenval >= -1e-6, f"End-to-end Laplacian not PSD: {min_eigenval}"
    
    def test_eigenvalue_metadata_extraction(self, simple_model, sample_input):
        """Test that eigenvalue metadata is correctly extracted."""
        builder = SheafBuilder(preserve_eigenvalues=True)
        sheaf = builder.build_from_activations(simple_model, sample_input)
        
        metadata = sheaf.eigenvalue_metadata
        
        # Should have extracted metadata for each node
        assert len(metadata.eigenvalue_matrices) > 0
        assert len(metadata.condition_numbers) > 0
        
        # Check that eigenvalue matrices match stalks
        for node in metadata.eigenvalue_matrices:
            if node in sheaf.stalks:
                eigenval_matrix = metadata.eigenvalue_matrices[node]
                stalk = sheaf.stalks[node]
                
                # They should be the same (eigenvalue matrix is used as stalk)
                assert torch.allclose(eigenval_matrix, stalk, atol=1e-6)


class TestNumericalStability:
    """Test numerical stability of Hodge Laplacian construction."""
    
    def test_ill_conditioned_eigenvalues(self):
        """Test Hodge construction with ill-conditioned eigenvalue matrices."""
        import networkx as nx
        
        # Create sheaf with ill-conditioned eigenvalues
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        
        # Ill-conditioned eigenvalue matrices
        stalks = {
            'A': torch.diag(torch.tensor([1e6, 1e-6], dtype=torch.float32)),
            'B': torch.diag(torch.tensor([1e8, 1e-8], dtype=torch.float32))
        }
        
        restrictions = {
            ('A', 'B'): torch.tensor([[0.5, 0.3], [0.7, 0.4]], dtype=torch.float32)
        }
        
        eigenvalue_metadata = EigenvalueMetadata(preserve_eigenvalues=True)
        eigenvalue_metadata.eigenvalue_matrices = stalks.copy()
        
        sheaf = Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            eigenvalue_metadata=eigenvalue_metadata
        )
        
        # Should handle ill-conditioning gracefully with regularization
        builder = SheafLaplacianBuilder(regularization=1e-8)
        L, metadata = builder.build(sheaf)
        
        # Should still produce finite results
        L_dense = L.toarray()
        assert np.isfinite(L_dense).all(), "Ill-conditioned case produced non-finite values"
        
        # Should still be symmetric
        symmetry_error = np.linalg.norm(L_dense - L_dense.T, 'fro')
        assert symmetry_error < 1e-6, "Ill-conditioned case lost symmetry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
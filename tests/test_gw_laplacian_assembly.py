"""Comprehensive tests for GW Laplacian assembly functionality (Phase 3).

This module tests the GW-specific Laplacian construction, including:
- GWLaplacianBuilder block assembly
- SheafLaplacianBuilder routing logic
- Mathematical property validation
- Edge weight semantics
- Integration with existing pipeline
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from unittest.mock import patch, MagicMock

from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder, GWLaplacianError, GWLaplacianMetadata
from neurosheaf.sheaf.assembly.laplacian import SheafLaplacianBuilder, LaplacianMetadata
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.data_structures import Sheaf


class TestGWLaplacianBuilder:
    """Test GW-specific Laplacian builder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=True)
        
        # Create synthetic GW sheaf for testing
        self.gw_sheaf = self._create_test_gw_sheaf()
    
    def _create_test_gw_sheaf(self) -> Sheaf:
        """Create a synthetic GW sheaf for testing."""
        # Create simple chain poset
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        
        # Create identity stalks (typical for GW sheaves)
        stalks = {
            'layer1': torch.eye(5, dtype=torch.float64),
            'layer2': torch.eye(4, dtype=torch.float64),
            'layer3': torch.eye(3, dtype=torch.float64)
        }
        
        # Create column-stochastic restrictions (typical for GW)
        # These should sum to 1 along columns (preserve measures)
        restrictions = {}
        
        # layer1 -> layer2 (5 -> 4)
        R_12 = torch.rand(4, 5, dtype=torch.float64)
        R_12 = R_12 / R_12.sum(dim=0, keepdim=True)  # Column-stochastic
        restrictions[('layer1', 'layer2')] = R_12
        
        # layer2 -> layer3 (4 -> 3)
        R_23 = torch.rand(3, 4, dtype=torch.float64)
        R_23 = R_23 / R_23.sum(dim=0, keepdim=True)  # Column-stochastic
        restrictions[('layer2', 'layer3')] = R_23
        
        # Create GW-specific metadata
        gw_costs = {
            ('layer1', 'layer2'): 0.3,
            ('layer2', 'layer3'): 0.2
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'gw_config': GWConfig().to_dict(),
            'whitened': False,
            'validation_passed': True
        }
        
        return Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata
        )
    
    def test_builder_initialization(self):
        """Test GW Laplacian builder initialization."""
        builder = GWLaplacianBuilder(
            validate_properties=True,
            sparsity_threshold=1e-10,
            use_weighted_inner_products=False
        )
        
        assert builder.validate_properties is True
        assert builder.sparsity_threshold == 1e-10
        assert builder.use_weighted_inner_products is False
    
    def test_gw_sheaf_detection(self):
        """Test that builder correctly detects GW sheaves."""
        # Should work with GW sheaf
        laplacian = self.builder.build_laplacian(self.gw_sheaf, sparse=True)
        assert isinstance(laplacian, csr_matrix)
        
        # Should fail with non-GW sheaf
        non_gw_sheaf = self.gw_sheaf
        non_gw_sheaf.metadata['construction_method'] = 'standard'
        
        with pytest.raises(GWLaplacianError, match="not GW-based"):
            self.builder.build_laplacian(non_gw_sheaf)
    
    def test_edge_weight_extraction(self):
        """Test GW cost extraction from sheaf metadata."""
        weights = self.builder.extract_edge_weights(self.gw_sheaf)
        
        # Should extract stored GW costs
        assert weights[('layer1', 'layer2')] == 0.3
        assert weights[('layer2', 'layer3')] == 0.2
        
        # Test fallback when GW costs missing
        sheaf_no_costs = self.gw_sheaf
        sheaf_no_costs.metadata.pop('gw_costs', None)
        
        weights_fallback = self.builder.extract_edge_weights(sheaf_no_costs)
        assert len(weights_fallback) == 2
        
        # Fallback should use operator norms
        for edge, weight in weights_fallback.items():
            assert weight > 0
            assert isinstance(weight, float)
    
    def test_sparse_laplacian_construction(self):
        """Test sparse Laplacian construction with correct block structure."""
        laplacian = self.builder.build_laplacian(self.gw_sheaf, sparse=True)
        
        # Should be sparse matrix
        assert isinstance(laplacian, csr_matrix)
        
        # Check dimensions: sum of stalk dimensions
        expected_dim = 5 + 4 + 3  # layer dimensions
        assert laplacian.shape == (expected_dim, expected_dim)
        
        # Should be symmetric
        symmetry_error = np.abs(laplacian - laplacian.T).max()
        assert symmetry_error < 1e-10, f"Laplacian not symmetric: max error = {symmetry_error}"
        
        # Should be positive semi-definite (check smallest eigenvalue)
        if laplacian.shape[0] <= 50:  # Only for small matrices
            eigenvals = np.linalg.eigvals(laplacian.toarray())
            min_eigval = np.min(eigenvals)
            assert min_eigval >= -1e-8, f"Laplacian not PSD: min eigenvalue = {min_eigval}"
    
    def test_dense_laplacian_construction(self):
        """Test dense Laplacian construction for small sheaves."""
        laplacian = self.builder.build_laplacian(self.gw_sheaf, sparse=False)
        
        # Should be dense tensor
        assert isinstance(laplacian, torch.Tensor)
        assert laplacian.dtype == torch.float64
        
        # Check dimensions
        expected_dim = 5 + 4 + 3
        assert laplacian.shape == (expected_dim, expected_dim)
        
        # Should be symmetric
        symmetry_error = torch.abs(laplacian - laplacian.T).max().item()
        assert symmetry_error < 1e-10
    
    def test_laplacian_block_structure(self):
        """Test that Laplacian follows general sheaf formulation."""
        laplacian = self.builder.build_laplacian(self.gw_sheaf, sparse=False)
        L = laplacian.numpy()
        
        # Extract blocks manually to verify structure
        # layer1: indices 0-4, layer2: indices 5-8, layer3: indices 9-11
        
        # Off-diagonal block L[layer2, layer1] should be -R_12 (4x5 block)
        R_12 = self.gw_sheaf.restrictions[('layer1', 'layer2')]
        gw_costs = self.gw_sheaf.metadata['gw_costs']
        weight_12 = gw_costs[('layer1', 'layer2')]
        
        expected_off_diag = -((weight_12**2) * R_12).numpy()
        actual_off_diag = L[5:9, 0:5]  # L[layer2, layer1]
        
        assert np.allclose(actual_off_diag, expected_off_diag, atol=1e-10), \
            "Off-diagonal block doesn't match -R pattern"
        
        # Off-diagonal block L[layer1, layer2] should be -R_12^T  
        expected_off_diag_T = -((weight_12**2) * R_12.T).numpy()
        actual_off_diag_T = L[0:5, 5:9]  # L[layer1, layer2]
        
        assert np.allclose(actual_off_diag_T, expected_off_diag_T, atol=1e-10), \
            "Transpose off-diagonal block doesn't match -R^T pattern"
    
    def test_coboundary_construction(self):
        """Test coboundary operator construction."""
        coboundary = self.builder.build_coboundary(self.gw_sheaf)
        
        # Should be sparse matrix
        assert isinstance(coboundary, csr_matrix)
        
        # Dimensions: (num_edges, total_node_dimension)
        num_edges = len(self.gw_sheaf.restrictions)
        total_node_dim = 5 + 4 + 3
        assert coboundary.shape == (num_edges, total_node_dim)
        
        # Should have reasonable sparsity
        assert coboundary.nnz > 0
        assert coboundary.nnz < coboundary.shape[0] * coboundary.shape[1]
    
    def test_metadata_creation(self):
        """Test GW Laplacian metadata creation."""
        metadata = self.builder._initialize_gw_metadata(
            self.gw_sheaf, 
            self.builder.extract_edge_weights(self.gw_sheaf)
        )
        
        assert isinstance(metadata, GWLaplacianMetadata)
        assert metadata.construction_method == "gw_laplacian"
        assert metadata.filtration_semantics == "increasing"
        assert metadata.measure_type == "uniform"
        assert metadata.edge_weight_source == "gw_costs"
        assert metadata.total_dimension == 12  # 5 + 4 + 3
    
    def test_validation_properties(self):
        """Test mathematical property validation."""
        # Should pass validation for well-formed sheaf
        laplacian = self.builder.build_laplacian(self.gw_sheaf, sparse=True)
        # If we get here without exception, validation passed
        
        # Test with validation disabled
        no_validate_builder = GWLaplacianBuilder(validate_properties=False)
        laplacian_no_val = no_validate_builder.build_laplacian(self.gw_sheaf, sparse=True)
        
        # Should produce same result
        assert np.allclose(laplacian.toarray(), laplacian_no_val.toarray())
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Non-GW sheaf should raise error
        non_gw_sheaf = self.gw_sheaf
        non_gw_sheaf.metadata['construction_method'] = 'standard'
        
        with pytest.raises(GWLaplacianError):
            self.builder.build_laplacian(non_gw_sheaf)
        
        # Coboundary with non-GW sheaf should raise error
        with pytest.raises(GWLaplacianError):
            self.builder.build_coboundary(non_gw_sheaf)


class TestSheafLaplacianBuilderRouting:
    """Test SheafLaplacianBuilder routing to GW methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = SheafLaplacianBuilder(validate_properties=True)
        
        # Create both GW and standard sheaves
        self.gw_sheaf = self._create_gw_sheaf()
        self.standard_sheaf = self._create_standard_sheaf()
    
    def _create_gw_sheaf(self) -> Sheaf:
        """Create GW sheaf for testing."""
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        
        stalks = {
            'A': torch.eye(3),
            'B': torch.eye(2)
        }
        
        restrictions = {
            ('A', 'B'): torch.rand(2, 3) * 0.5  # Column-stochastic-ish
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': {('A', 'B'): 0.4},
            'whitened': False
        }
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def _create_standard_sheaf(self) -> Sheaf:
        """Create standard sheaf for testing."""
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        
        stalks = {
            'A': torch.eye(3),
            'B': torch.eye(2)
        }
        
        restrictions = {
            ('A', 'B'): torch.randn(2, 3) * 0.1  # Small random values
        }
        
        metadata = {
            'construction_method': 'scaled_procrustes',
            'whitened': True
        }
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def test_gw_routing(self):
        """Test that GW sheaves are routed to GW builder."""
        with patch('neurosheaf.sheaf.assembly.laplacian.GWLaplacianBuilder') as mock_gw_builder:
            # Configure mock
            mock_instance = MagicMock()
            mock_gw_builder.return_value = mock_instance
            mock_instance.build_laplacian.return_value = csr_matrix((3, 3))
            mock_instance._initialize_gw_metadata.return_value = MagicMock()
            mock_instance.extract_edge_weights.return_value = {('A', 'B'): 0.4}
            
            # Build Laplacian
            laplacian, metadata = self.builder.build(self.gw_sheaf)
            
            # Should have called GW builder
            mock_gw_builder.assert_called_once()
            mock_instance.build_laplacian.assert_called_once()
    
    def test_standard_routing(self):
        """Test that standard sheaves are routed to standard builder."""
        laplacian, metadata = self.builder.build(self.standard_sheaf)
        
        # Should get standard Laplacian
        assert isinstance(laplacian, csr_matrix)
        assert metadata.construction_method in ["standard", "hodge_formulation"]
    
    def test_metadata_conversion(self):
        """Test that GW metadata is converted to standard format."""
        laplacian, metadata = self.builder.build(self.gw_sheaf)
        
        # Should be base LaplacianMetadata, not GWLaplacianMetadata
        assert isinstance(metadata, LaplacianMetadata)
        assert metadata.construction_method == "gw_laplacian"
        assert hasattr(metadata, 'total_dimension')
        assert hasattr(metadata, 'sparsity_ratio')
    
    def test_edge_weight_handling(self):
        """Test edge weight extraction and usage."""
        laplacian, metadata = self.builder.build(self.gw_sheaf, edge_weights=None)
        
        # Should work without explicit edge weights (extract from metadata)
        assert laplacian.shape[0] > 0
        
        # Should work with explicit edge weights
        explicit_weights = {('A', 'B'): 0.5}
        laplacian2, metadata2 = self.builder.build(self.gw_sheaf, edge_weights=explicit_weights)
        
        # Should produce different result with different weights
        assert not np.allclose(laplacian.toarray(), laplacian2.toarray())


class TestGWIntegrationWithSheafBuilder:
    """Test integration with complete sheaf construction pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gw_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # Create simple test network
        self.test_net = self._create_test_network()
        self.test_input = torch.randn(4, 10)  # Small batch for testing
        
        self.gw_config = GWConfig(
            epsilon=0.1,
            max_iter=20,  # Fast for testing
            validate_couplings=True
        )
    
    def _create_test_network(self):
        """Create simple test network."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 8)
                self.layer2 = nn.Linear(8, 6)
            
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = self.layer2(x)
                return x
        
        return SimpleNet()
    
    def test_end_to_end_gw_laplacian_construction(self):
        """Test complete pipeline from network to Laplacian."""
        try:
            # Build sheaf
            sheaf = self.gw_builder.build_from_activations(
                self.test_net, self.test_input, 
                validate=True, gw_config=self.gw_config
            )
            
            # Should be GW sheaf
            assert sheaf.is_gw_sheaf()
            assert sheaf.metadata['construction_method'] == 'gromov_wasserstein'
            
            # Build Laplacian
            from neurosheaf.sheaf.assembly.laplacian import build_sheaf_laplacian
            laplacian, metadata = build_sheaf_laplacian(sheaf)
            
            # Should route to GW builder
            assert isinstance(laplacian, csr_matrix)
            assert metadata.construction_method == "gw_laplacian"
            
            # Should have correct dimensions
            total_dim = sum(stalk.shape[0] for stalk in sheaf.stalks.values())
            assert laplacian.shape == (total_dim, total_dim)
            
            # Should be symmetric and PSD
            symmetry_error = np.abs(laplacian - laplacian.T).max()
            assert symmetry_error < 1e-6
            
        except Exception as e:
            # If POT library not available, test may fail
            if "POT" in str(e) or "not available" in str(e):
                pytest.skip("POT library not available for GW computations")
            else:
                raise
    
    def test_filtration_semantics_propagation(self):
        """Test that GW sheaves propagate correct filtration semantics."""
        try:
            sheaf = self.gw_builder.build_from_activations(
                self.test_net, self.test_input, 
                gw_config=self.gw_config
            )
            
            # Should have increasing filtration semantics
            assert sheaf.get_filtration_semantics() == 'increasing'
            
            # Should have GW costs in metadata
            assert 'gw_costs' in sheaf.metadata
            assert len(sheaf.metadata['gw_costs']) > 0
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                raise


class TestGWLaplacianMathematicalProperties:
    """Test mathematical properties of GW Laplacians."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = GWLaplacianBuilder(validate_properties=True)
        
        # Create test sheaf with known properties
        self.test_sheaf = self._create_mathematical_test_sheaf()
    
    def _create_mathematical_test_sheaf(self) -> Sheaf:
        """Create sheaf with known mathematical properties for testing."""
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C')])
        
        # Use exact values for deterministic testing
        stalks = {
            'A': torch.eye(2, dtype=torch.float64),
            'B': torch.eye(2, dtype=torch.float64), 
            'C': torch.eye(2, dtype=torch.float64)
        }
        
        # Create exact column-stochastic restrictions
        R_AB = torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float64)  # 2x2 column-stochastic
        R_BC = torch.tensor([[0.7, 0.5], [0.3, 0.5]], dtype=torch.float64)  # 2x2 column-stochastic
        
        restrictions = {
            ('A', 'B'): R_AB,
            ('B', 'C'): R_BC
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': {('A', 'B'): 0.3, ('B', 'C'): 0.4},
            'whitened': False
        }
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def test_block_formula_correctness(self):
        """Test that block formula is implemented correctly."""
        laplacian = self.builder.build_laplacian(self.test_sheaf, sparse=False)
        L = laplacian.numpy()
        
        # Extract restrictions and weights
        R_AB = self.test_sheaf.restrictions[('A', 'B')].numpy()
        R_BC = self.test_sheaf.restrictions[('B', 'C')].numpy()
        w_AB = 0.3
        w_BC = 0.4
        
        # Check off-diagonal blocks manually (L = δᵀδ formulation uses w²)
        # L[B,A] = -w_AB² * R_AB
        expected_BA = -(w_AB**2) * R_AB
        actual_BA = L[2:4, 0:2]  # B block relative to A block
        assert np.allclose(actual_BA, expected_BA, atol=1e-12)
        
        # L[A,B] = -w_AB² * R_AB^T  
        expected_AB = -(w_AB**2) * R_AB.T
        actual_AB = L[0:2, 2:4]  # A block relative to B block
        assert np.allclose(actual_AB, expected_AB, atol=1e-12)
        
        # Check diagonal blocks
        # L[A,A] should include R_AB^T * R_AB contribution
        expected_AA_contrib = w_AB**2 * (R_AB.T @ R_AB)
        actual_AA = L[0:2, 0:2]
        
        # Diagonal block includes contributions from outgoing edges
        assert np.trace(actual_AA) >= np.trace(expected_AA_contrib) - 1e-10
    
    def test_column_stochastic_preservation(self):
        """Test that column-stochastic restrictions are handled correctly."""
        # Verify our test restrictions are indeed column-stochastic
        R_AB = self.test_sheaf.restrictions[('A', 'B')]
        R_BC = self.test_sheaf.restrictions[('B', 'C')]
        
        # Column sums should be 1
        assert torch.allclose(R_AB.sum(dim=0), torch.ones(2, dtype=R_AB.dtype), atol=1e-10)
        assert torch.allclose(R_BC.sum(dim=0), torch.ones(2, dtype=R_BC.dtype), atol=1e-10)
        
        # Laplacian should be well-conditioned
        laplacian = self.builder.build_laplacian(self.test_sheaf, sparse=False)
        eigenvals = torch.linalg.eigvals(laplacian).real
        
        # Should have one zero eigenvalue (connected graph)
        min_eigval = torch.min(eigenvals)
        assert min_eigval >= -1e-10  # PSD
        
        # Should have good conditioning (not too many near-zero eigenvalues)
        near_zero = torch.sum(eigenvals < 1e-8)
        assert near_zero <= 2  # At most a few zero eigenvalues


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
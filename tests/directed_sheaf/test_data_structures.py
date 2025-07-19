"""Comprehensive unit tests for directed sheaf data structures.

This module tests the core data structures of the directed sheaf module:
- DirectedSheaf
- DirectedSheafValidationResult 
- DirectedWhiteningInfo

Tests cover:
- Mathematical correctness
- Complex structure validation
- Real embedding functionality
- Backward compatibility
- Performance characteristics
"""

import pytest
import torch
import numpy as np
import networkx as nx
from typing import Dict, Any

from neurosheaf.directed_sheaf import (
    DirectedSheaf,
    DirectedSheafValidationResult,
    DirectedWhiteningInfo,
    DEFAULT_DIRECTIONALITY_PARAMETER,
    validate_directionality_parameter,
    create_default_directed_sheaf
)
from neurosheaf.sheaf.data_structures import Sheaf, SheafValidationResult


class TestDirectedSheaf:
    """Test suite for DirectedSheaf class."""
    
    def test_initialization_empty(self):
        """Test initialization of empty directed sheaf."""
        sheaf = DirectedSheaf()
        
        assert isinstance(sheaf.poset, nx.DiGraph)
        assert len(sheaf.poset.nodes()) == 0
        assert len(sheaf.complex_stalks) == 0
        assert len(sheaf.directed_restrictions) == 0
        assert sheaf.directional_encoding is None
        assert sheaf.directionality_parameter == DEFAULT_DIRECTIONALITY_PARAMETER
        assert sheaf.base_sheaf is None
        assert sheaf.metadata['directed_sheaf'] is True
    
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        complex_stalks = {
            'a': torch.randn(5, 5, dtype=torch.complex64),
            'b': torch.randn(3, 3, dtype=torch.complex64),
            'c': torch.randn(4, 4, dtype=torch.complex64)
        }
        
        directed_restrictions = {
            ('a', 'b'): torch.randn(3, 5, dtype=torch.complex64),
            ('b', 'c'): torch.randn(4, 3, dtype=torch.complex64)
        }
        
        sheaf = DirectedSheaf(
            poset=poset,
            complex_stalks=complex_stalks,
            directed_restrictions=directed_restrictions,
            directionality_parameter=0.5
        )
        
        assert len(sheaf.poset.nodes()) == 3
        assert len(sheaf.poset.edges()) == 2
        assert len(sheaf.complex_stalks) == 3
        assert len(sheaf.directed_restrictions) == 2
        assert sheaf.directionality_parameter == 0.5
        assert sheaf.metadata['directionality_parameter'] == 0.5
    
    def test_get_dimensions(self):
        """Test dimension calculation methods."""
        complex_stalks = {
            'a': torch.randn(5, 5, dtype=torch.complex64),
            'b': torch.randn(3, 3, dtype=torch.complex64),
            'c': torch.randn(4, 4, dtype=torch.complex64)
        }
        
        sheaf = DirectedSheaf(complex_stalks=complex_stalks)
        
        # Test complex dimension
        assert sheaf.get_complex_dimension() == 5 + 3 + 4
        
        # Test real dimension (2x complex)
        assert sheaf.get_real_dimension() == 2 * (5 + 3 + 4)
        
        # Test node dimensions
        node_dims = sheaf.get_node_dimensions()
        assert node_dims['a'] == 5
        assert node_dims['b'] == 3
        assert node_dims['c'] == 4
        
        # Test real node dimensions
        real_dims = sheaf.get_node_real_dimensions()
        assert real_dims['a'] == 10
        assert real_dims['b'] == 6
        assert real_dims['c'] == 8
    
    def test_get_adjacency_matrix(self):
        """Test adjacency matrix generation."""
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c')])
        
        sheaf = DirectedSheaf(poset=poset)
        adj = sheaf.get_adjacency_matrix()
        
        # Should be 3x3 matrix
        assert adj.shape == (3, 3)
        
        # Check specific entries (order depends on node ordering)
        nodes = list(poset.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Check edges exist
        for u, v in poset.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            assert adj[i, j] == 1.0
        
        # Check diagonal is zero
        assert torch.diag(adj).sum() == 0
    
    def test_to_real_representation_complex(self):
        """Test conversion to real representation with complex data."""
        # Create complex stalks
        complex_stalks = {
            'a': torch.tensor([[1+1j, 2+0j], [0+1j, 1-1j]], dtype=torch.complex64),
            'b': torch.tensor([[0+2j, 1+0j]], dtype=torch.complex64).reshape(1, 2)
        }
        
        # Create complex restrictions
        directed_restrictions = {
            ('a', 'b'): torch.tensor([[1+1j, 0-1j]], dtype=torch.complex64)
        }
        
        sheaf = DirectedSheaf(
            complex_stalks=complex_stalks,
            directed_restrictions=directed_restrictions
        )
        
        real_stalks, real_restrictions = sheaf.to_real_representation()
        
        # Check that real representations exist
        assert 'a' in real_stalks
        assert 'b' in real_stalks
        assert ('a', 'b') in real_restrictions
        
        # Check dimensions are doubled
        assert real_stalks['a'].shape[0] == 2 * complex_stalks['a'].shape[0]
        assert real_stalks['b'].shape[0] == 2 * complex_stalks['b'].shape[0]
        
        # Check restriction dimensions
        assert real_restrictions[('a', 'b')].shape[0] == 2 * directed_restrictions[('a', 'b')].shape[0]
        assert real_restrictions[('a', 'b')].shape[1] == 2 * directed_restrictions[('a', 'b')].shape[1]
        
        # All outputs should be real
        assert real_stalks['a'].dtype in [torch.float32, torch.float64]
        assert real_stalks['b'].dtype in [torch.float32, torch.float64]
        assert real_restrictions[('a', 'b')].dtype in [torch.float32, torch.float64]
    
    def test_to_real_representation_real(self):
        """Test conversion to real representation with real data."""
        # Create real stalks (will be embedded in complex space)
        complex_stalks = {
            'a': torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float32),
            'b': torch.tensor([[0.0, 1.0]], dtype=torch.float32).reshape(1, 2)
        }
        
        directed_restrictions = {
            ('a', 'b'): torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        }
        
        sheaf = DirectedSheaf(
            complex_stalks=complex_stalks,
            directed_restrictions=directed_restrictions
        )
        
        real_stalks, real_restrictions = sheaf.to_real_representation()
        
        # Check that dimensions are still doubled (for complex embedding)
        assert real_stalks['a'].shape[0] == 2 * complex_stalks['a'].shape[0]
        assert real_stalks['b'].shape[0] == 2 * complex_stalks['b'].shape[0]
        
        # All outputs should be real
        assert real_stalks['a'].dtype in [torch.float32, torch.float64]
        assert real_stalks['b'].dtype in [torch.float32, torch.float64]
        assert real_restrictions[('a', 'b')].dtype in [torch.float32, torch.float64]
    
    def test_get_laplacian_structure(self):
        """Test Laplacian structure information."""
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        complex_stalks = {
            'a': torch.randn(5, 5, dtype=torch.complex64),
            'b': torch.randn(3, 3, dtype=torch.complex64),
            'c': torch.randn(4, 4, dtype=torch.complex64)
        }
        
        sheaf = DirectedSheaf(poset=poset, complex_stalks=complex_stalks)
        structure = sheaf.get_laplacian_structure()
        
        assert structure['total_complex_dimension'] == 12
        assert structure['total_real_dimension'] == 24
        assert structure['num_nodes'] == 3
        assert structure['num_edges'] == 2
        assert structure['laplacian_type'] == 'hermitian'
        assert structure['directionality_parameter'] == DEFAULT_DIRECTIONALITY_PARAMETER
        assert structure['real_embedding_overhead'] == 4.0
        assert 'estimated_sparsity' in structure
        assert 'memory_savings' in structure
    
    def test_validate_complex_structure(self):
        """Test complex structure validation."""
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b')])
        
        # Valid complex structure
        complex_stalks = {
            'a': torch.randn(5, 5, dtype=torch.complex64),
            'b': torch.randn(3, 3, dtype=torch.complex64)
        }
        
        directed_restrictions = {
            ('a', 'b'): torch.randn(3, 5, dtype=torch.complex64)
        }
        
        sheaf = DirectedSheaf(
            poset=poset,
            complex_stalks=complex_stalks,
            directed_restrictions=directed_restrictions
        )
        
        validation = sheaf.validate_complex_structure()
        
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
        assert validation['num_stalks'] == 2
        assert validation['num_restrictions'] == 1
        assert validation['directionality_parameter'] == DEFAULT_DIRECTIONALITY_PARAMETER
    
    def test_validate_complex_structure_errors(self):
        """Test complex structure validation with errors."""
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b')])
        
        # Invalid structure - incompatible dimensions
        complex_stalks = {
            'a': torch.randn(5, 5, dtype=torch.complex64),
            'b': torch.randn(3, 3, dtype=torch.complex64)
        }
        
        directed_restrictions = {
            ('a', 'b'): torch.randn(2, 4, dtype=torch.complex64)  # Wrong dimensions
        }
        
        sheaf = DirectedSheaf(
            poset=poset,
            complex_stalks=complex_stalks,
            directed_restrictions=directed_restrictions
        )
        
        validation = sheaf.validate_complex_structure()
        
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
        assert any('incompatible dimensions' in error for error in validation['errors'])
    
    def test_summary(self):
        """Test summary string generation."""
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        complex_stalks = {
            'a': torch.randn(5, 5, dtype=torch.complex64),
            'b': torch.randn(3, 3, dtype=torch.complex64),
            'c': torch.randn(4, 4, dtype=torch.complex64)
        }
        
        sheaf = DirectedSheaf(
            poset=poset,
            complex_stalks=complex_stalks,
            directionality_parameter=0.5
        )
        
        summary = sheaf.summary()
        
        assert 'Directed Sheaf Summary' in summary
        assert 'Nodes: 3' in summary
        assert 'Edges: 2' in summary
        assert 'Complex dimension: 12' in summary
        assert 'Real dimension: 24' in summary
        assert 'Directionality (q): 0.5' in summary
        assert 'Laplacian type: hermitian' in summary
        assert 'Validation: ✗' in summary  # Not validated yet


class TestDirectedSheafValidationResult:
    """Test suite for DirectedSheafValidationResult class."""
    
    def test_initialization_empty(self):
        """Test initialization of empty validation result."""
        result = DirectedSheafValidationResult(valid_directed_sheaf=True)
        
        assert result.valid_directed_sheaf is True
        assert len(result.hermitian_errors) == 0
        assert len(result.complex_structure_errors) == 0
        assert len(result.directional_encoding_errors) == 0
        assert result.max_error == 0.0
        assert result.base_validation is None
        assert result.directionality_parameter == DEFAULT_DIRECTIONALITY_PARAMETER
        assert result.details['validation_type'] == 'directed_sheaf'
    
    def test_initialization_with_errors(self):
        """Test initialization with validation errors."""
        result = DirectedSheafValidationResult(
            valid_directed_sheaf=False,
            hermitian_errors=['Not Hermitian'],
            complex_structure_errors=['Invalid stalk'],
            directional_encoding_errors=['Invalid encoding'],
            max_error=1e-6,
            directionality_parameter=0.5
        )
        
        assert result.valid_directed_sheaf is False
        assert len(result.hermitian_errors) == 1
        assert len(result.complex_structure_errors) == 1
        assert len(result.directional_encoding_errors) == 1
        assert result.max_error == 1e-6
        assert result.directionality_parameter == 0.5
    
    def test_all_errors_property(self):
        """Test all_errors property."""
        # Create base validation with errors
        base_validation = SheafValidationResult(
            valid_sheaf=False,
            transitivity_errors=['Transitivity error'],
            restriction_errors=['Restriction error']
        )
        
        result = DirectedSheafValidationResult(
            valid_directed_sheaf=False,
            hermitian_errors=['Hermitian error'],
            complex_structure_errors=['Complex error'],
            directional_encoding_errors=['Encoding error'],
            base_validation=base_validation
        )
        
        all_errors = result.all_errors
        
        assert len(all_errors) == 5
        assert 'Hermitian error' in all_errors
        assert 'Complex error' in all_errors
        assert 'Encoding error' in all_errors
        assert 'Transitivity error' in all_errors
        assert 'Restriction error' in all_errors
    
    def test_get_error_summary(self):
        """Test error summary generation."""
        base_validation = SheafValidationResult(
            valid_sheaf=False,
            transitivity_errors=['T1', 'T2'],
            restriction_errors=['R1']
        )
        
        result = DirectedSheafValidationResult(
            valid_directed_sheaf=False,
            hermitian_errors=['H1'],
            complex_structure_errors=['C1', 'C2'],
            directional_encoding_errors=['E1'],
            base_validation=base_validation
        )
        
        summary = result.get_error_summary()
        
        assert summary['hermitian_errors'] == 1
        assert summary['complex_structure_errors'] == 2
        assert summary['directional_encoding_errors'] == 1
        assert summary['base_transitivity_errors'] == 2
        assert summary['base_restriction_errors'] == 1
        assert summary['total_errors'] == 7
    
    def test_summary_string(self):
        """Test summary string generation."""
        result = DirectedSheafValidationResult(
            valid_directed_sheaf=False,
            hermitian_errors=['H1'],
            complex_structure_errors=['C1', 'C2'],
            directional_encoding_errors=['E1'],
            max_error=1e-6,
            directionality_parameter=0.5
        )
        
        summary = result.summary()
        
        assert 'Directed Sheaf Validation Status: FAIL' in summary
        assert 'Maximum Error: 0.000001' in summary
        assert 'Directionality Parameter: 0.5' in summary
        assert 'Hermitian Errors: 1' in summary
        assert 'Complex Structure Errors: 2' in summary
        assert 'Directional Encoding Errors: 1' in summary
        assert 'Total Errors: 4' in summary
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = DirectedSheafValidationResult(
            valid_directed_sheaf=True,
            max_error=1e-8,
            directionality_parameter=0.25
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['valid_directed_sheaf'] is True
        assert result_dict['max_error'] == 1e-8
        assert result_dict['directionality_parameter'] == 0.25
        assert 'error_summary' in result_dict
        assert 'details' in result_dict
        assert result_dict['error_summary']['total_errors'] == 0


class TestDirectedWhiteningInfo:
    """Test suite for DirectedWhiteningInfo class."""
    
    def test_initialization(self):
        """Test initialization of whitening info."""
        complex_whitening = torch.randn(5, 5, dtype=torch.complex64)
        real_whitening = torch.randn(10, 10, dtype=torch.float32)
        eigenvalues = torch.randn(5)
        
        info = DirectedWhiteningInfo(
            complex_whitening_matrix=complex_whitening,
            real_whitening_matrix=real_whitening,
            eigenvalues=eigenvalues,
            condition_number=1e3,
            rank=5,
            explained_variance=0.95,
            directionality_parameter=0.5
        )
        
        assert info.complex_whitening_matrix.shape == (5, 5)
        assert info.real_whitening_matrix.shape == (10, 10)
        assert info.eigenvalues.shape == (5,)
        assert info.condition_number == 1e3
        assert info.rank == 5
        assert info.explained_variance == 0.95
        assert info.directionality_parameter == 0.5
    
    def test_summary(self):
        """Test summary string generation."""
        complex_whitening = torch.randn(5, 5, dtype=torch.complex64)
        real_whitening = torch.randn(10, 10, dtype=torch.float32)
        eigenvalues = torch.randn(5)
        
        info = DirectedWhiteningInfo(
            complex_whitening_matrix=complex_whitening,
            real_whitening_matrix=real_whitening,
            eigenvalues=eigenvalues,
            condition_number=1e3,
            rank=5,
            explained_variance=0.95,
            directionality_parameter=0.5
        )
        
        summary = info.summary()
        
        assert 'Directed Whitening Info' in summary
        assert 'Rank: 5' in summary
        assert 'Condition Number: 1.00e+03' in summary
        assert 'Explained Variance: 0.950' in summary
        assert 'Directionality Parameter: 0.5' in summary
        assert 'Complex Whitening: torch.Size([5, 5])' in summary
        assert 'Real Embedding: torch.Size([10, 10])' in summary


class TestModuleFunctions:
    """Test suite for module-level functions."""
    
    def test_validate_directionality_parameter_valid(self):
        """Test validation of valid directionality parameters."""
        assert validate_directionality_parameter(0.0) is True
        assert validate_directionality_parameter(0.25) is True
        assert validate_directionality_parameter(0.5) is True
        assert validate_directionality_parameter(1.0) is True
        assert validate_directionality_parameter(0) is True
        assert validate_directionality_parameter(1) is True
    
    def test_validate_directionality_parameter_invalid(self):
        """Test validation of invalid directionality parameters."""
        assert validate_directionality_parameter(-0.1) is False
        assert validate_directionality_parameter(1.1) is False
        assert validate_directionality_parameter('0.5') is False
        assert validate_directionality_parameter(None) is False
        assert validate_directionality_parameter([0.5]) is False
    
    def test_create_default_directed_sheaf(self):
        """Test creation of default directed sheaf."""
        sheaf = create_default_directed_sheaf()
        
        assert isinstance(sheaf, DirectedSheaf)
        assert sheaf.directionality_parameter == DEFAULT_DIRECTIONALITY_PARAMETER
        assert len(sheaf.complex_stalks) == 0
        assert len(sheaf.directed_restrictions) == 0
        assert sheaf.metadata['construction_method'] == 'default_creation'
        assert sheaf.metadata['directed_sheaf'] is True
        assert sheaf.metadata['validation_passed'] is False


class TestPerformanceCharacteristics:
    """Test suite for performance characteristics."""
    
    def test_memory_scaling(self):
        """Test memory scaling characteristics."""
        # Create increasingly large sheaves
        sizes = [5, 10, 20]
        
        for size in sizes:
            complex_stalks = {
                f'node_{i}': torch.randn(size, size, dtype=torch.complex64)
                for i in range(3)
            }
            
            sheaf = DirectedSheaf(complex_stalks=complex_stalks)
            
            # Check dimension scaling
            assert sheaf.get_complex_dimension() == 3 * size
            assert sheaf.get_real_dimension() == 6 * size
            
            # Check Laplacian structure
            structure = sheaf.get_laplacian_structure()
            assert structure['real_embedding_overhead'] == 4.0
    
    def test_real_embedding_overhead(self):
        """Test real embedding overhead calculation."""
        complex_stalks = {
            'a': torch.randn(10, 10, dtype=torch.complex64),
            'b': torch.randn(5, 5, dtype=torch.complex64)
        }
        
        sheaf = DirectedSheaf(complex_stalks=complex_stalks)
        real_stalks, _ = sheaf.to_real_representation()
        
        # Check that real dimensions are exactly 2x complex
        assert real_stalks['a'].shape[0] == 20
        assert real_stalks['b'].shape[0] == 10
        
        # Check total memory overhead
        complex_elements = 10*10 + 5*5  # 125 complex elements
        real_elements = 20*20 + 10*10   # 500 real elements
        
        # Each complex element becomes 2 real elements
        assert real_elements == 4 * complex_elements


if __name__ == '__main__':
    pytest.main([__file__])
"""Unit tests for DirectionalEncodingComputer class.

Tests the mathematical correctness of directional encoding computation:
- T^{(q)} = exp(i 2π q (A - A^T)) computation
- Mathematical properties validation
- Integration with NetworkX graphs
- Performance characteristics
"""

import pytest
import torch
import numpy as np
import networkx as nx
import math
from typing import Dict, Any

from neurosheaf.directed_sheaf.core.directional_encoding import DirectionalEncodingComputer


class TestDirectionalEncodingComputer:
    """Test suite for DirectionalEncodingComputer class."""
    
    def test_initialization(self):
        """Test initialization of DirectionalEncodingComputer."""
        computer = DirectionalEncodingComputer()
        
        assert computer.q == 0.25
        assert computer.validate_properties is True
        assert computer.tolerance == 1e-12
        assert computer.two_pi_q == 2 * math.pi * 0.25
        
        # Test with custom parameters
        computer_custom = DirectionalEncodingComputer(
            q=0.5, validate_properties=False, tolerance=1e-8
        )
        assert computer_custom.q == 0.5
        assert computer_custom.validate_properties is False
        assert computer_custom.tolerance == 1e-8
    
    def test_initialization_invalid_q(self):
        """Test error handling for invalid q parameters."""
        # Test q out of range
        with pytest.raises(ValueError, match="Directionality parameter q must be in"):
            DirectionalEncodingComputer(q=-0.1)
        
        with pytest.raises(ValueError, match="Directionality parameter q must be in"):
            DirectionalEncodingComputer(q=1.1)
    
    def test_compute_encoding_matrix_basic(self):
        """Test basic encoding matrix computation."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Test with simple adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        # Check properties
        assert encoding.is_complex()
        assert encoding.shape == (3, 3)
        
        # Check unit magnitudes
        magnitudes = torch.abs(encoding)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes))
        
        # Check diagonal is all ones
        diagonal = torch.diag(encoding)
        assert torch.allclose(diagonal, torch.ones_like(diagonal))
    
    def test_compute_encoding_matrix_undirected(self):
        """Test encoding matrix for undirected graph."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create symmetric adjacency matrix (undirected)
        adjacency = torch.tensor([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        # For undirected graph, antisymmetric part is zero, so encoding should be all 1's
        # A - A^T = 0 for symmetric matrix, so exp(i * 2π * q * 0) = 1 for all elements
        ones_matrix = torch.ones(3, 3, dtype=torch.complex64)
        assert torch.allclose(encoding, ones_matrix, atol=1e-12)
    
    def test_compute_encoding_matrix_q_zero(self):
        """Test encoding matrix with q=0 (should be all ones)."""
        computer = DirectionalEncodingComputer(q=0.0)
        
        # Test with directed adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        # Should be all ones matrix regardless of adjacency (exp(i * 0) = 1)
        ones_matrix = torch.ones(3, 3, dtype=torch.complex64)
        assert torch.allclose(encoding, ones_matrix, atol=1e-12)
    
    def test_compute_encoding_matrix_mathematical_properties(self):
        """Test mathematical properties of encoding matrix."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create directed adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        # Test antisymmetric property
        antisymmetric = adjacency - adjacency.T
        expected_phases = 2 * math.pi * 0.25 * antisymmetric
        
        # Check phases
        actual_phases = torch.angle(encoding)
        assert torch.allclose(actual_phases, expected_phases, atol=1e-12)
        
        # Check unit magnitudes
        magnitudes = torch.abs(encoding)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes))
    
    def test_compute_encoding_matrix_validation(self):
        """Test validation during encoding computation."""
        computer = DirectionalEncodingComputer(validate_properties=True)
        
        # Create test adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        # Should pass validation
        encoding = computer.compute_encoding_matrix(adjacency)
        assert encoding.is_complex()
        
        # Test with validation disabled
        computer_no_val = DirectionalEncodingComputer(validate_properties=False)
        encoding_no_val = computer_no_val.compute_encoding_matrix(adjacency)
        
        # Should get same result
        assert torch.allclose(encoding, encoding_no_val)
    
    def test_compute_encoding_matrix_invalid_input(self):
        """Test error handling for invalid inputs."""
        computer = DirectionalEncodingComputer()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Adjacency matrix must be a torch.Tensor"):
            computer.compute_encoding_matrix("not a tensor")
        
        # Test with non-2D tensor
        with pytest.raises(ValueError, match="Adjacency matrix must be 2D"):
            computer.compute_encoding_matrix(torch.randn(3))
        
        # Test with non-square matrix
        with pytest.raises(ValueError, match="Adjacency matrix must be square"):
            computer.compute_encoding_matrix(torch.randn(2, 3))
    
    def test_compute_from_poset_basic(self):
        """Test encoding computation from NetworkX poset."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create simple directed graph
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        encoding = computer.compute_from_poset(poset)
        
        # Check properties
        assert encoding.is_complex()
        assert encoding.shape == (3, 3)
        
        # Check unit magnitudes
        magnitudes = torch.abs(encoding)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes))
    
    def test_compute_from_poset_with_node_ordering(self):
        """Test encoding computation with specified node ordering."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create directed graph
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        
        # Specify node ordering
        node_ordering = ['layer1', 'layer2', 'layer3']
        
        encoding = computer.compute_from_poset(poset, node_ordering)
        
        # Check properties
        assert encoding.is_complex()
        assert encoding.shape == (3, 3)
        
        # Check that ordering is respected
        # layer1 -> layer2 should have phase
        # layer2 -> layer3 should have phase
        # Other entries should be identity
        assert torch.abs(encoding[0, 0] - 1.0) < 1e-12  # Diagonal
        assert torch.abs(encoding[1, 1] - 1.0) < 1e-12  # Diagonal
        assert torch.abs(encoding[2, 2] - 1.0) < 1e-12  # Diagonal
    
    def test_compute_from_poset_invalid_input(self):
        """Test error handling for invalid poset inputs."""
        computer = DirectionalEncodingComputer()
        
        # Test with non-DiGraph
        with pytest.raises(ValueError, match="Input must be a NetworkX DiGraph"):
            computer.compute_from_poset(nx.Graph())
        
        # Test with empty poset
        empty_poset = nx.DiGraph()
        with pytest.raises(ValueError, match="Poset has no nodes"):
            computer.compute_from_poset(empty_poset)
        
        # Test with invalid node ordering
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b')])
        
        with pytest.raises(ValueError, match="Node ordering must contain exactly"):
            computer.compute_from_poset(poset, ['a', 'c'])  # 'c' not in poset
    
    def test_compute_with_metadata(self):
        """Test encoding computation with metadata."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create test adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        encoding, metadata = computer.compute_with_metadata(adjacency)
        
        # Check encoding
        assert encoding.is_complex()
        
        # Check metadata
        assert metadata['directionality_parameter'] == 0.25
        assert metadata['adjacency_shape'] == (3, 3)
        assert metadata['encoding_shape'] == (3, 3)
        assert metadata['num_nodes'] == 3
        assert metadata['num_edges'] == 2
        assert 'antisymmetric_norm' in metadata
        assert 'encoding_properties' in metadata
        assert metadata['validation_passed'] is True
    
    def test_validate_directionality_parameter(self):
        """Test directionality parameter validation."""
        computer = DirectionalEncodingComputer()
        
        # Test valid parameters
        assert computer.validate_directionality_parameter(0.0) is True
        assert computer.validate_directionality_parameter(0.25) is True
        assert computer.validate_directionality_parameter(0.5) is True
        assert computer.validate_directionality_parameter(1.0) is True
        
        # Test invalid parameters
        assert computer.validate_directionality_parameter(-0.1) is False
        assert computer.validate_directionality_parameter(1.1) is False
        assert computer.validate_directionality_parameter("0.5") is False
        assert computer.validate_directionality_parameter(None) is False
    
    def test_compute_phase_matrix(self):
        """Test phase matrix computation."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        phase_matrix = computer.compute_phase_matrix(adjacency)
        
        # Check properties
        assert phase_matrix.shape == (3, 3)
        assert phase_matrix.dtype == torch.float32
        
        # Check mathematical correctness
        antisymmetric = adjacency - adjacency.T
        expected_phase = 2 * math.pi * 0.25 * antisymmetric
        
        assert torch.allclose(phase_matrix, expected_phase)
    
    def test_compare_directionality_parameters(self):
        """Test comparison of different directionality parameters."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create test adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        # Compare different q values
        q_values = [0.0, 0.25, 0.5, 1.0]
        results = computer.compare_directionality_parameters(adjacency, q_values)
        
        # Check results
        assert len(results) == 4
        assert all(q in results for q in q_values)
        assert all(results[q].is_complex() for q in q_values)
        
        # Check that q=0 gives all ones matrix
        ones_matrix = torch.ones(3, 3, dtype=torch.complex64)
        assert torch.allclose(results[0.0], ones_matrix, atol=1e-12)
        
        # Check that different q values give different results for non-zero antisymmetric parts
        # Only check if the adjacency matrix is not symmetric
        antisymmetric = adjacency - adjacency.T
        if torch.any(antisymmetric != 0):
            assert not torch.allclose(results[0.25], results[0.5])
        
        # Check that original q parameter is preserved
        assert computer.q == 0.25
    
    def test_get_node_mapping(self):
        """Test node mapping generation."""
        computer = DirectionalEncodingComputer()
        
        # Create test poset
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        
        # Test default node mapping
        mapping = computer.get_node_mapping(poset)
        
        assert len(mapping) == 3
        assert set(mapping.keys()) == {'layer1', 'layer2', 'layer3'}
        assert set(mapping.values()) == {0, 1, 2}
        
        # Test with custom node ordering
        node_ordering = ['layer3', 'layer1', 'layer2']
        mapping = computer.get_node_mapping(poset, node_ordering)
        
        assert mapping['layer3'] == 0
        assert mapping['layer1'] == 1
        assert mapping['layer2'] == 2
    
    def test_encoding_properties_analysis(self):
        """Test analysis of encoding properties."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        # Analyze properties
        properties = computer._analyze_encoding_properties(encoding)
        
        # Check basic properties
        assert properties['is_complex'] is True
        assert properties['shape'] == (3, 3)
        
        # Check magnitude properties
        assert abs(properties['magnitude_mean'] - 1.0) < 1e-12
        assert properties['magnitude_std'] < 1e-12
        assert abs(properties['magnitude_max'] - 1.0) < 1e-12
        assert abs(properties['magnitude_min'] - 1.0) < 1e-12
        
        # Check diagonal properties
        assert properties['diagonal_all_ones'] is True
        assert properties['diagonal_max_error'] < 1e-12
    
    def test_device_support(self):
        """Test device support for computations."""
        # Test with CPU device
        computer = DirectionalEncodingComputer(device=torch.device('cpu'))
        
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        assert encoding.device == torch.device('cpu')
        assert encoding.is_complex()
        
        # Test device transfer
        if torch.cuda.is_available():
            computer_cuda = DirectionalEncodingComputer(device=torch.device('cuda'))
            encoding_cuda = computer_cuda.compute_encoding_matrix(adjacency)
            assert encoding_cuda.device.type == 'cuda'
    
    def test_large_graph_performance(self):
        """Test performance with larger graphs."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create larger adjacency matrix
        n = 50
        adjacency = torch.randint(0, 2, (n, n), dtype=torch.float32)
        adjacency = adjacency * (1 - torch.eye(n))  # Remove self-loops
        
        # Should complete quickly
        import time
        start_time = time.time()
        encoding = computer.compute_encoding_matrix(adjacency)
        end_time = time.time()
        
        # Check results
        assert encoding.is_complex()
        assert encoding.shape == (n, n)
        assert end_time - start_time < 1.0  # Should be fast
        
        # Check unit magnitudes
        magnitudes = torch.abs(encoding)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes))
    
    def test_numerical_stability(self):
        """Test numerical stability of encoding computation."""
        computer = DirectionalEncodingComputer(q=0.25, tolerance=1e-15)
        
        # Test with various adjacency matrices
        test_cases = [
            torch.zeros(3, 3),  # All zeros
            torch.eye(3),       # Identity (should be zeros after antisymmetric)
            torch.ones(3, 3) - torch.eye(3),  # Fully connected
        ]
        
        for adjacency in test_cases:
            encoding = computer.compute_encoding_matrix(adjacency)
            
            # Check basic properties
            assert encoding.is_complex()
            assert torch.allclose(torch.abs(encoding), torch.ones_like(torch.abs(encoding)))
            
            # Check diagonal is ones
            diagonal = torch.diag(encoding)
            assert torch.allclose(diagonal, torch.ones_like(diagonal))


class TestDirectionalEncodingMathematicalProperties:
    """Test mathematical properties of directional encoding."""
    
    def test_euler_formula_implementation(self):
        """Test that implementation follows Euler's formula."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create test adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        # Manually compute using Euler's formula
        antisymmetric = adjacency - adjacency.T
        phases = 2 * math.pi * 0.25 * antisymmetric
        
        expected_real = torch.cos(phases)
        expected_imag = torch.sin(phases)
        expected_encoding = torch.complex(expected_real, expected_imag)
        
        assert torch.allclose(encoding, expected_encoding, atol=1e-12)
    
    def test_antisymmetric_property(self):
        """Test that encoding depends only on antisymmetric part."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        # Add symmetric part
        symmetric_part = torch.tensor([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=torch.float32)
        
        adjacency_with_symmetric = adjacency + symmetric_part
        
        # Compute antisymmetric parts
        antisymmetric1 = adjacency - adjacency.T
        antisymmetric2 = adjacency_with_symmetric - adjacency_with_symmetric.T
        
        # Antisymmetric parts should be the same
        assert torch.allclose(antisymmetric1, antisymmetric2, atol=1e-12)
        
        # Therefore encodings should be the same
        encoding1 = computer.compute_encoding_matrix(adjacency)
        encoding2 = computer.compute_encoding_matrix(adjacency_with_symmetric)
        
        assert torch.allclose(encoding1, encoding2, atol=1e-12)
    
    def test_phase_scaling(self):
        """Test that phases scale correctly with q parameter."""
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        # Test different q values
        q_values = [0.1, 0.25, 0.5]
        
        for q in q_values:
            computer = DirectionalEncodingComputer(q=q)
            encoding = computer.compute_encoding_matrix(adjacency)
            
            # Check that phases scale correctly
            phases = torch.angle(encoding)
            antisymmetric = adjacency - adjacency.T
            expected_phases = 2 * math.pi * q * antisymmetric
            
            # Normalize phases to [-π, π] range to match torch.angle output
            expected_phases = torch.atan2(torch.sin(expected_phases), torch.cos(expected_phases))
            
            assert torch.allclose(phases, expected_phases, atol=1e-12)
    
    def test_unitary_property(self):
        """Test that encoding matrices have unit magnitude entries."""
        computer = DirectionalEncodingComputer(q=0.25)
        
        # Create test adjacency matrix
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        encoding = computer.compute_encoding_matrix(adjacency)
        
        # Check that all entries have unit magnitude
        magnitudes = torch.abs(encoding)
        expected_magnitudes = torch.ones_like(magnitudes)
        
        assert torch.allclose(magnitudes, expected_magnitudes, atol=1e-12)
        
        # Check that diagonal entries are 1
        diagonal = torch.diag(encoding)
        expected_diagonal = torch.ones_like(diagonal)
        
        assert torch.allclose(diagonal, expected_diagonal, atol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__])
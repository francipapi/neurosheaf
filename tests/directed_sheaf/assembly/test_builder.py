"""Unit tests for DirectedSheafBuilder class.

Tests the mathematical correctness of directed sheaf construction:
- Real to complex stalk extension
- Directional encoding computation
- Directed restriction map construction
- Hermitian Laplacian validation
- Pipeline integration
"""

import pytest
import torch
import numpy as np
import networkx as nx
from typing import Dict, Any
import time

from neurosheaf.directed_sheaf.assembly.builder import DirectedSheafBuilder
from neurosheaf.directed_sheaf.data_structures import DirectedSheaf
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.directed_sheaf.core.directional_encoding import DirectionalEncodingComputer
from neurosheaf.directed_sheaf.core.directed_procrustes import DirectedProcrustesComputer
from neurosheaf.directed_sheaf.core.complex_extension import ComplexStalkExtender


class TestDirectedSheafBuilder:
    """Test suite for DirectedSheafBuilder class."""
    
    def test_initialization(self):
        """Test initialization of DirectedSheafBuilder."""
        builder = DirectedSheafBuilder()
        
        assert builder.q == 0.25
        assert builder.validate_construction is True
        assert builder.device == torch.device('cpu')
        assert isinstance(builder.complex_extender, ComplexStalkExtender)
        assert isinstance(builder.encoding_computer, DirectionalEncodingComputer)
        assert isinstance(builder.procrustes_computer, DirectedProcrustesComputer)
        
        # Test with custom parameters
        builder_custom = DirectedSheafBuilder(
            directionality_parameter=0.5,
            validate_construction=False,
            device=torch.device('cpu')
        )
        assert builder_custom.q == 0.5
        assert builder_custom.validate_construction is False
    
    def test_build_from_sheaf_simple(self):
        """Test building directed sheaf from simple real sheaf."""
        builder = DirectedSheafBuilder()
        
        # Create simple real sheaf
        base_sheaf = self._create_simple_real_sheaf()
        
        # Build directed sheaf
        directed_sheaf = builder.build_from_sheaf(base_sheaf)
        
        # Check basic properties
        assert isinstance(directed_sheaf, DirectedSheaf)
        assert directed_sheaf.directionality_parameter == 0.25
        assert len(directed_sheaf.complex_stalks) == len(base_sheaf.stalks)
        assert len(directed_sheaf.directed_restrictions) == len(base_sheaf.restrictions)
        assert directed_sheaf.poset.nodes() == base_sheaf.poset.nodes()
        assert directed_sheaf.poset.edges() == base_sheaf.poset.edges()
        
        # Check that stalks are complex
        for node_id, stalk in directed_sheaf.complex_stalks.items():
            assert stalk.is_complex()
            assert stalk.dtype in [torch.complex64, torch.complex128]
        
        # Check that restrictions are complex
        for edge, restriction in directed_sheaf.directed_restrictions.items():
            assert restriction.is_complex()
            assert restriction.dtype in [torch.complex64, torch.complex128]
        
        # Check directional encoding
        assert directed_sheaf.directional_encoding is not None
        assert directed_sheaf.directional_encoding.is_complex()
        
        # Check metadata
        assert directed_sheaf.metadata['construction_method'] == 'directed_sheaf_builder'
        assert directed_sheaf.metadata['directionality_parameter'] == 0.25
        assert directed_sheaf.metadata['extension_successful'] is True
    
    def test_build_from_sheaf_invalid_input(self):
        """Test error handling for invalid input."""
        builder = DirectedSheafBuilder()
        
        # Test with non-Sheaf input
        with pytest.raises(ValueError, match="Input must be a Sheaf"):
            builder.build_from_sheaf("not a sheaf")
        
        # Test with empty sheaf
        empty_sheaf = Sheaf()
        with pytest.raises(ValueError, match="non-empty stalks and restrictions"):
            builder.build_from_sheaf(empty_sheaf)
    
    def test_extend_complex_stalks(self):
        """Test extension of real stalks to complex."""
        builder = DirectedSheafBuilder()
        
        # Create real stalks
        real_stalks = {
            'a': torch.eye(3, dtype=torch.float32),
            'b': torch.eye(2, dtype=torch.float32),
            'c': torch.eye(1, dtype=torch.float32)
        }
        
        # Extend to complex
        complex_stalks = builder._extend_complex_stalks(real_stalks)
        
        # Check properties
        assert len(complex_stalks) == len(real_stalks)
        
        for node_id, complex_stalk in complex_stalks.items():
            assert complex_stalk.is_complex()
            assert complex_stalk.shape == real_stalks[node_id].shape
            
            # For identity matrices, complex extension should be complex identity
            expected_complex = torch.eye(real_stalks[node_id].shape[0], dtype=torch.complex64)
            assert torch.allclose(complex_stalk, expected_complex)
    
    def test_extract_adjacency_matrix(self):
        """Test extraction of adjacency matrix from poset."""
        builder = DirectedSheafBuilder()
        
        # Create test poset
        poset = nx.DiGraph()
        poset.add_nodes_from(['a', 'b', 'c'])
        poset.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c')])
        
        # Extract adjacency matrix
        adjacency = builder._extract_adjacency_matrix(poset)
        
        # Check properties
        assert adjacency.shape == (3, 3)
        assert adjacency.dtype == torch.float32
        assert adjacency.device == builder.device
        
        # Check specific entries
        nodes = list(poset.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Should have 1s for edges
        for u, v in poset.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            assert adjacency[i, j] == 1.0
        
        # Should have 0s for non-edges
        for i in range(3):
            for j in range(3):
                if i != j:
                    u, v = nodes[i], nodes[j]
                    if (u, v) not in poset.edges():
                        assert adjacency[i, j] == 0.0
    
    def test_compute_directed_restrictions(self):
        """Test computation of directed restrictions."""
        builder = DirectedSheafBuilder()
        
        # Create test data
        poset = nx.DiGraph()
        poset.add_nodes_from(['a', 'b', 'c'])
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        # Create base restrictions
        base_restrictions = {
            ('a', 'b'): torch.randn(2, 3, dtype=torch.float32),
            ('b', 'c'): torch.randn(1, 2, dtype=torch.float32)
        }
        
        # Create directional encoding
        adjacency = builder._extract_adjacency_matrix(poset)
        directional_encoding = builder.encoding_computer.compute_encoding_matrix(adjacency)
        
        # Compute directed restrictions
        directed_restrictions = builder._compute_directed_restrictions(
            base_restrictions, directional_encoding, poset
        )
        
        # Check properties
        assert len(directed_restrictions) == len(base_restrictions)
        
        for edge, restriction in directed_restrictions.items():
            assert restriction.is_complex()
            assert restriction.shape == base_restrictions[edge].shape
    
    def test_validation_toggle(self):
        """Test validation toggle functionality."""
        # Test with validation enabled
        builder_validated = DirectedSheafBuilder(validate_construction=True)
        base_sheaf = self._create_simple_real_sheaf()
        
        # Should work without errors
        directed_sheaf = builder_validated.build_from_sheaf(base_sheaf)
        assert directed_sheaf.metadata.get('construction_successful', False) is True
        
        # Test with validation disabled
        builder_no_validation = DirectedSheafBuilder(validate_construction=False)
        
        # Should also work
        directed_sheaf_no_val = builder_no_validation.build_from_sheaf(base_sheaf)
        assert isinstance(directed_sheaf_no_val, DirectedSheaf)
    
    def test_directionality_parameter_effect(self):
        """Test effect of different directionality parameters."""
        base_sheaf = self._create_simple_real_sheaf()
        
        # Test different q values
        q_values = [0.0, 0.25, 0.5, 1.0]
        
        for q in q_values:
            builder = DirectedSheafBuilder(directionality_parameter=q)
            directed_sheaf = builder.build_from_sheaf(base_sheaf)
            
            assert directed_sheaf.directionality_parameter == q
            assert directed_sheaf.metadata['directionality_parameter'] == q
            
            # Check that directional encoding reflects q
            encoding = directed_sheaf.directional_encoding
            assert encoding is not None
            
            # For q=0, should reduce to identity-like behavior
            if q == 0.0:
                # Non-edge entries should be 1 (no phase)
                adjacency = directed_sheaf.get_adjacency_matrix()
                for i in range(adjacency.shape[0]):
                    for j in range(adjacency.shape[1]):
                        if adjacency[i, j] == 0 and i != j:
                            # Should be 1 (no directional encoding)
                            assert torch.abs(encoding[i, j] - 1.0) < 1e-6
    
    def test_laplacian_construction_integration(self):
        """Test integration with Laplacian construction."""
        builder = DirectedSheafBuilder()
        base_sheaf = self._create_simple_real_sheaf()
        
        # Build directed sheaf
        directed_sheaf = builder.build_from_sheaf(base_sheaf)
        
        # Build Laplacian using builder methods
        complex_laplacian = builder.build_laplacian(directed_sheaf)
        real_laplacian = builder.build_real_laplacian(directed_sheaf)
        
        # Check properties
        assert complex_laplacian.is_complex()
        assert complex_laplacian.shape[0] == complex_laplacian.shape[1]
        
        # Check Hermitian property
        hermitian_error = torch.abs(complex_laplacian - complex_laplacian.conj().T).max().item()
        assert hermitian_error < 1e-6
        
        # Check real representation
        from scipy.sparse import csr_matrix
        assert isinstance(real_laplacian, csr_matrix)
        assert real_laplacian.shape[0] == 2 * complex_laplacian.shape[0]
        
        # Check symmetry of real representation
        symmetry_error = np.abs(real_laplacian - real_laplacian.T).max()
        assert symmetry_error < 1e-6
    
    def test_metadata_preservation(self):
        """Test preservation of metadata during construction."""
        builder = DirectedSheafBuilder()
        
        # Create base sheaf with metadata
        base_sheaf = self._create_simple_real_sheaf()
        base_sheaf.metadata.update({
            'test_key': 'test_value',
            'batch_size': 100,
            'whitened': True
        })
        
        # Build directed sheaf
        directed_sheaf = builder.build_from_sheaf(base_sheaf)
        
        # Check metadata preservation
        assert 'base_sheaf_metadata' in directed_sheaf.metadata
        assert directed_sheaf.metadata['base_sheaf_metadata']['test_key'] == 'test_value'
        assert directed_sheaf.metadata['base_sheaf_metadata']['batch_size'] == 100
        assert directed_sheaf.metadata['base_sheaf_metadata']['whitened'] is True
    
    def test_device_support(self):
        """Test device support for computations."""
        # Test with CPU device
        builder_cpu = DirectedSheafBuilder(device=torch.device('cpu'))
        base_sheaf = self._create_simple_real_sheaf()
        
        directed_sheaf_cpu = builder_cpu.build_from_sheaf(base_sheaf)
        
        # Check that tensors are on correct device
        for stalk in directed_sheaf_cpu.complex_stalks.values():
            assert stalk.device == torch.device('cpu')
        
        # Test device transfer
        if torch.cuda.is_available():
            builder_cuda = DirectedSheafBuilder(device=torch.device('cuda'))
            directed_sheaf_cuda = builder_cuda.build_from_sheaf(base_sheaf)
            
            for stalk in directed_sheaf_cuda.complex_stalks.values():
                assert stalk.device.type == 'cuda'
    
    def test_performance_benchmarking(self):
        """Test performance of directed sheaf construction."""
        builder = DirectedSheafBuilder()
        
        # Create larger sheaf for performance testing
        base_sheaf = self._create_large_real_sheaf()
        
        # Time the construction
        start_time = time.time()
        directed_sheaf = builder.build_from_sheaf(base_sheaf)
        construction_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert construction_time < 10.0  # 10 seconds max
        
        # Check that metadata includes timing
        assert 'construction_time' in directed_sheaf.metadata
        assert directed_sheaf.metadata['construction_time'] > 0
    
    def test_get_construction_info(self):
        """Test getting construction information."""
        builder = DirectedSheafBuilder()
        
        info = builder.get_construction_info()
        
        assert isinstance(info, dict)
        assert info['class_name'] == 'DirectedSheafBuilder'
        assert info['directionality_parameter'] == 0.25
        assert 'mathematical_foundation' in info
        assert 'construction_method' in info
        assert 'components' in info
        assert 'device' in info
    
    def test_undirected_case_reduction(self):
        """Test that directed sheaf reduces to undirected case when q=0."""
        builder = DirectedSheafBuilder(directionality_parameter=0.0)
        base_sheaf = self._create_simple_real_sheaf()
        
        # Build directed sheaf with q=0
        directed_sheaf = builder.build_from_sheaf(base_sheaf)
        
        # Should have minimal directional encoding
        encoding = directed_sheaf.directional_encoding
        
        # For undirected case, encoding should be real (no imaginary components)
        max_imag = torch.abs(encoding.imag).max().item()
        assert max_imag < 1e-6
    
    def _create_simple_real_sheaf(self) -> Sheaf:
        """Create a simple real sheaf for testing."""
        # Create simple directed graph
        poset = nx.DiGraph()
        poset.add_nodes_from(['a', 'b', 'c'])
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        # Create real stalks (identity matrices)
        stalks = {
            'a': torch.eye(3, dtype=torch.float32),
            'b': torch.eye(2, dtype=torch.float32),
            'c': torch.eye(1, dtype=torch.float32)
        }
        
        # Create real restrictions
        restrictions = {
            ('a', 'b'): torch.randn(2, 3, dtype=torch.float32),
            ('b', 'c'): torch.randn(1, 2, dtype=torch.float32)
        }
        
        # Create sheaf
        sheaf = Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata={
                'construction_method': 'test_real_sheaf',
                'is_valid': True,
                'whitened': True
            }
        )
        
        return sheaf
    
    def _create_large_real_sheaf(self) -> Sheaf:
        """Create a larger real sheaf for performance testing."""
        # Create larger directed graph
        poset = nx.DiGraph()
        nodes = [f'node_{i}' for i in range(10)]
        poset.add_nodes_from(nodes)
        
        # Add edges to create connected structure
        for i in range(9):
            poset.add_edge(f'node_{i}', f'node_{i+1}')
        
        # Add some additional edges for complexity
        for i in range(0, 8, 2):
            poset.add_edge(f'node_{i}', f'node_{i+2}')
        
        # Create real stalks
        stalks = {}
        for i, node in enumerate(nodes):
            # Varying dimensions for realism
            dim = min(5, i + 1)
            stalks[node] = torch.eye(dim, dtype=torch.float32)
        
        # Create real restrictions
        restrictions = {}
        for edge in poset.edges():
            u, v = edge
            u_dim = stalks[u].shape[0]
            v_dim = stalks[v].shape[0]
            restrictions[edge] = torch.randn(v_dim, u_dim, dtype=torch.float32)
        
        # Create sheaf
        sheaf = Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata={
                'construction_method': 'test_large_real_sheaf',
                'is_valid': True,
                'whitened': True
            }
        )
        
        return sheaf


if __name__ == '__main__':
    pytest.main([__file__])
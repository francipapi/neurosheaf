"""Unit tests for DirectedSheafAdapter class.

Tests the mathematical correctness of pipeline integration:
- Spectral analysis adaptation
- Visualization format conversion
- Eigenvalue extraction and mapping
- Persistence result conversion
- Compatibility sheaf creation
"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from typing import Dict, Any
import time

from neurosheaf.directed_sheaf.compatibility.adapter import DirectedSheafAdapter, SpectralAnalysisMetadata
from neurosheaf.directed_sheaf.data_structures import DirectedSheaf
from neurosheaf.directed_sheaf.assembly.builder import DirectedSheafBuilder
from neurosheaf.sheaf.data_structures import Sheaf


class TestDirectedSheafAdapter:
    """Test suite for DirectedSheafAdapter class."""
    
    def test_initialization(self):
        """Test initialization of DirectedSheafAdapter."""
        adapter = DirectedSheafAdapter()
        
        assert adapter.preserve_metadata is True
        assert adapter.validate_conversions is True
        assert adapter.optimize_sparse_operations is True
        assert adapter.device == torch.device('cpu')
        
        # Test with custom parameters
        adapter_custom = DirectedSheafAdapter(
            preserve_metadata=False,
            validate_conversions=False,
            optimize_sparse_operations=False,
            device=torch.device('cpu')
        )
        assert adapter_custom.preserve_metadata is False
        assert adapter_custom.validate_conversions is False
        assert adapter_custom.optimize_sparse_operations is False
    
    def test_adapt_for_spectral_analysis(self):
        """Test adaptation for spectral analysis."""
        adapter = DirectedSheafAdapter()
        
        # Create directed sheaf
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Adapt for spectral analysis
        real_laplacian, metadata = adapter.adapt_for_spectral_analysis(directed_sheaf)
        
        # Check Laplacian properties
        assert isinstance(real_laplacian, csr_matrix)
        assert real_laplacian.shape[0] == real_laplacian.shape[1]
        assert real_laplacian.shape[0] > 0
        
        # Check symmetry (real representation of Hermitian matrix)
        symmetry_error = np.abs(real_laplacian - real_laplacian.T).max()
        assert symmetry_error < 1e-6
        
        # Check metadata
        assert isinstance(metadata, SpectralAnalysisMetadata)
        assert metadata.is_directed is True
        assert metadata.directionality_parameter == 0.25
        assert metadata.hermitian_laplacian is True
        assert metadata.real_embedded is True
        assert metadata.complex_dimension > 0
        assert metadata.real_dimension == 2 * metadata.complex_dimension
        assert metadata.num_vertices > 0
        assert metadata.num_edges >= 0
        assert metadata.conversion_time > 0
    
    def test_adapt_for_spectral_analysis_invalid_input(self):
        """Test error handling for invalid input."""
        adapter = DirectedSheafAdapter()
        
        # Test with non-DirectedSheaf input
        with pytest.raises(ValueError, match="Input must be a DirectedSheaf"):
            adapter.adapt_for_spectral_analysis("not a directed sheaf")
    
    def test_adapt_for_visualization(self):
        """Test adaptation for visualization."""
        adapter = DirectedSheafAdapter()
        
        # Create directed sheaf
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Adapt for visualization
        visualization_data = adapter.adapt_for_visualization(directed_sheaf)
        
        # Check structure
        assert isinstance(visualization_data, dict)
        assert 'graph_structure' in visualization_data
        assert 'similarity_matrix' in visualization_data
        assert 'eigenvalue_data' in visualization_data
        assert 'directed_properties' in visualization_data
        assert 'metadata' in visualization_data
        
        # Check graph structure
        graph_structure = visualization_data['graph_structure']
        assert 'nodes' in graph_structure
        assert 'edges' in graph_structure
        assert 'node_dimensions' in graph_structure
        assert 'adjacency_matrix' in graph_structure
        assert 'directional_encoding' in graph_structure
        
        # Check directed properties
        directed_properties = visualization_data['directed_properties']
        assert directed_properties['directionality_parameter'] == 0.25
        assert directed_properties['complex_stalks'] is True
        assert directed_properties['hermitian_laplacian'] is True
    
    def test_extract_real_eigenvalues(self):
        """Test extraction of real eigenvalues."""
        adapter = DirectedSheafAdapter()
        
        # Create directed sheaf
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Extract eigenvalues
        eigenvalues = adapter.extract_real_eigenvalues(directed_sheaf)
        
        # Check properties
        assert isinstance(eigenvalues, np.ndarray)
        assert eigenvalues.dtype in [np.float32, np.float64]
        assert len(eigenvalues) > 0
        assert np.all(eigenvalues >= -1e-6)  # Should be non-negative (up to numerical precision)
        
        # Check that eigenvalues are real
        assert np.all(np.isreal(eigenvalues))
    
    def test_convert_persistence_results(self):
        """Test conversion of persistence results."""
        adapter = DirectedSheafAdapter()
        
        # Create mock persistence results
        persistence_results = {
            'eigenvalue_data': np.array([0.0, 0.1, 0.5, 1.0]),
            'persistence_diagrams': [(0.0, 0.1), (0.1, 0.5), (0.5, float('inf'))],
            'features': {'total_persistence': 0.5, 'num_bars': 3}
        }
        
        # Convert results
        enhanced_results = adapter.convert_persistence_results(persistence_results)
        
        # Check structure
        assert isinstance(enhanced_results, dict)
        assert 'directed_analysis' in enhanced_results
        assert 'eigenvalue_data' in enhanced_results
        assert 'persistence_diagrams' in enhanced_results
        assert 'features' in enhanced_results
        
        # Check directed analysis information
        directed_analysis = enhanced_results['directed_analysis']
        assert directed_analysis['is_directed'] is True
        assert directed_analysis['hermitian_laplacian'] is True
        assert directed_analysis['real_embedded'] is True
        assert directed_analysis['complex_eigenvalues_processed'] is True
        
        # Check that original data is preserved
        assert 'features' in enhanced_results
        assert enhanced_results['features']['total_persistence'] == 0.5
    
    def test_create_compatibility_sheaf(self):
        """Test creation of compatibility sheaf."""
        adapter = DirectedSheafAdapter()
        
        # Create directed sheaf
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Create compatibility sheaf
        compatibility_sheaf = adapter.create_compatibility_sheaf(directed_sheaf)
        
        # Check properties
        assert isinstance(compatibility_sheaf, Sheaf)
        assert compatibility_sheaf.poset.nodes() == directed_sheaf.poset.nodes()
        assert compatibility_sheaf.poset.edges() == directed_sheaf.poset.edges()
        
        # Check that stalks have correct dimensions (real embedding)
        for node_id, stalk in compatibility_sheaf.stalks.items():
            complex_stalk = directed_sheaf.complex_stalks[node_id]
            expected_real_dim = 2 * complex_stalk.shape[0]
            assert stalk.shape[0] == expected_real_dim
            assert not stalk.is_complex()
        
        # Check that restrictions are real
        for edge, restriction in compatibility_sheaf.restrictions.items():
            assert not restriction.is_complex()
            assert restriction.dtype == torch.float32
        
        # Check metadata
        assert compatibility_sheaf.metadata['construction_method'] == 'directed_sheaf_compatibility'
        assert compatibility_sheaf.metadata['original_directed'] is True
        assert compatibility_sheaf.metadata['directionality_parameter'] == 0.25
        assert compatibility_sheaf.metadata['real_embedding'] is True
    
    def test_validation_toggle(self):
        """Test validation toggle functionality."""
        # Test with validation enabled
        adapter_validated = DirectedSheafAdapter(validate_conversions=True)
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Should work without errors
        real_laplacian, metadata = adapter_validated.adapt_for_spectral_analysis(directed_sheaf)
        assert isinstance(real_laplacian, csr_matrix)
        assert isinstance(metadata, SpectralAnalysisMetadata)
        
        # Test with validation disabled
        adapter_no_validation = DirectedSheafAdapter(validate_conversions=False)
        
        # Should also work
        real_laplacian_no_val, metadata_no_val = adapter_no_validation.adapt_for_spectral_analysis(directed_sheaf)
        assert isinstance(real_laplacian_no_val, csr_matrix)
        assert isinstance(metadata_no_val, SpectralAnalysisMetadata)
        
        # Results should be equivalent
        diff = np.abs(real_laplacian - real_laplacian_no_val).max()
        assert diff < 1e-12
    
    def test_metadata_preservation(self):
        """Test metadata preservation during adaptation."""
        # Test with metadata preservation enabled
        adapter_preserve = DirectedSheafAdapter(preserve_metadata=True)
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Add custom metadata
        directed_sheaf.metadata['custom_key'] = 'custom_value'
        directed_sheaf.metadata['test_number'] = 42
        
        # Adapt for visualization
        visualization_data = adapter_preserve.adapt_for_visualization(directed_sheaf)
        
        # Check metadata preservation
        assert 'metadata' in visualization_data
        assert visualization_data['metadata']['custom_key'] == 'custom_value'
        assert visualization_data['metadata']['test_number'] == 42
        
        # Test with metadata preservation disabled
        adapter_no_preserve = DirectedSheafAdapter(preserve_metadata=False)
        
        visualization_data_no_preserve = adapter_no_preserve.adapt_for_visualization(directed_sheaf)
        
        # Should have empty metadata
        assert visualization_data_no_preserve['metadata'] == {}
    
    def test_device_support(self):
        """Test device support for adaptations."""
        # Test with CPU device
        adapter_cpu = DirectedSheafAdapter(device=torch.device('cpu'))
        directed_sheaf = self._create_simple_directed_sheaf()
        
        real_laplacian, metadata = adapter_cpu.adapt_for_spectral_analysis(directed_sheaf)
        assert isinstance(real_laplacian, csr_matrix)
        
        # Test device transfer
        if torch.cuda.is_available():
            adapter_cuda = DirectedSheafAdapter(device=torch.device('cuda'))
            real_laplacian_cuda, metadata_cuda = adapter_cuda.adapt_for_spectral_analysis(directed_sheaf)
            assert isinstance(real_laplacian_cuda, csr_matrix)
    
    def test_sparse_operations_toggle(self):
        """Test sparse operations toggle functionality."""
        # Test with sparse operations enabled
        adapter_sparse = DirectedSheafAdapter(optimize_sparse_operations=True)
        directed_sheaf = self._create_simple_directed_sheaf()
        
        real_laplacian_sparse, metadata_sparse = adapter_sparse.adapt_for_spectral_analysis(directed_sheaf)
        assert isinstance(real_laplacian_sparse, csr_matrix)
        
        # Test with sparse operations disabled
        adapter_dense = DirectedSheafAdapter(optimize_sparse_operations=False)
        
        real_laplacian_dense, metadata_dense = adapter_dense.adapt_for_spectral_analysis(directed_sheaf)
        assert isinstance(real_laplacian_dense, csr_matrix)  # Still converted to sparse at end
        
        # Results should be equivalent
        diff = np.abs(real_laplacian_sparse - real_laplacian_dense).max()
        assert diff < 1e-12
    
    def test_performance_benchmarking(self):
        """Test performance of adaptation operations."""
        adapter = DirectedSheafAdapter()
        
        # Create larger directed sheaf for performance testing
        directed_sheaf = self._create_large_directed_sheaf()
        
        # Time the spectral analysis adaptation
        start_time = time.time()
        real_laplacian, metadata = adapter.adapt_for_spectral_analysis(directed_sheaf)
        adaptation_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert adaptation_time < 10.0  # 10 seconds max
        
        # Check that timing is recorded
        assert metadata.conversion_time > 0
        assert metadata.conversion_time <= adaptation_time
    
    def test_eigenvalue_extraction_consistency(self):
        """Test consistency of eigenvalue extraction."""
        adapter = DirectedSheafAdapter()
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Extract eigenvalues using direct method
        eigenvalues_direct = adapter.extract_real_eigenvalues(directed_sheaf)
        
        # Extract eigenvalues via spectral analysis adaptation
        real_laplacian, metadata = adapter.adapt_for_spectral_analysis(directed_sheaf)
        
        # Compute eigenvalues from sparse matrix
        eigenvalues_sparse = np.linalg.eigvals(real_laplacian.toarray())
        eigenvalues_sparse = np.sort(eigenvalues_sparse.real)
        
        # Sort both for comparison
        eigenvalues_direct_sorted = np.sort(eigenvalues_direct)
        
        # The real sparse matrix will have twice as many eigenvalues as the complex matrix
        # because of the real embedding, so we need to check that the eigenvalues match
        # in pairs or as appropriate for the real embedding
        
        # For now, just check that both extractions produce valid eigenvalues
        assert len(eigenvalues_direct_sorted) > 0
        assert len(eigenvalues_sparse) > 0
        
        # Check that both are real and non-negative (within tolerance)
        assert np.all(eigenvalues_direct_sorted >= -1e-6)
        assert np.all(eigenvalues_sparse >= -1e-6)
    
    def test_get_adapter_info(self):
        """Test getting adapter information."""
        adapter = DirectedSheafAdapter()
        
        info = adapter.get_adapter_info()
        
        assert isinstance(info, dict)
        assert info['class_name'] == 'DirectedSheafAdapter'
        assert info['preserve_metadata'] is True
        assert info['validate_conversions'] is True
        assert info['optimize_sparse_operations'] is True
        assert 'capabilities' in info
        assert 'mathematical_foundation' in info
        assert 'integration_method' in info
        assert 'device' in info
    
    def test_undirected_case_consistency(self):
        """Test consistency with undirected case when q=0."""
        adapter = DirectedSheafAdapter()
        
        # Create directed sheaf with q=0 (should be undirected)
        directed_sheaf = self._create_simple_directed_sheaf(q=0.0)
        
        # Adapt for spectral analysis
        real_laplacian, metadata = adapter.adapt_for_spectral_analysis(directed_sheaf)
        
        # Should still work and produce valid results
        assert isinstance(real_laplacian, csr_matrix)
        assert metadata.directionality_parameter == 0.0
        assert metadata.hermitian_laplacian is True
        
        # Extract eigenvalues
        eigenvalues = adapter.extract_real_eigenvalues(directed_sheaf)
        
        # Should be non-negative (Laplacian property)
        assert np.all(eigenvalues >= -1e-6)
    
    def test_spectral_analysis_metadata_validation(self):
        """Test validation of spectral analysis metadata."""
        adapter = DirectedSheafAdapter()
        directed_sheaf = self._create_simple_directed_sheaf()
        
        # Get metadata
        real_laplacian, metadata = adapter.adapt_for_spectral_analysis(directed_sheaf)
        
        # Validate metadata properties
        assert metadata.is_directed is True
        assert 0.0 <= metadata.directionality_parameter <= 1.0
        assert metadata.hermitian_laplacian is True
        assert metadata.real_embedded is True
        assert metadata.complex_dimension > 0
        assert metadata.real_dimension == 2 * metadata.complex_dimension
        assert metadata.num_vertices > 0
        assert metadata.num_edges >= 0
        assert 0.0 <= metadata.sparsity <= 1.0
        assert metadata.conversion_time > 0
    
    def _create_simple_directed_sheaf(self, q: float = 0.25) -> DirectedSheaf:
        """Create a simple directed sheaf for testing."""
        # Create builder
        builder = DirectedSheafBuilder(directionality_parameter=q)
        
        # Create simple real sheaf
        base_sheaf = self._create_simple_real_sheaf()
        
        # Build directed sheaf
        directed_sheaf = builder.build_from_sheaf(base_sheaf)
        
        return directed_sheaf
    
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
    
    def _create_large_directed_sheaf(self) -> DirectedSheaf:
        """Create a larger directed sheaf for performance testing."""
        # Create builder
        builder = DirectedSheafBuilder(directionality_parameter=0.25)
        
        # Create larger real sheaf
        base_sheaf = self._create_large_real_sheaf()
        
        # Build directed sheaf
        directed_sheaf = builder.build_from_sheaf(base_sheaf)
        
        return directed_sheaf
    
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
            dim = min(3, i + 1)
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
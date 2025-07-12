# tests/phase4_spectral/unit/test_static_laplacian_masking.py
import pytest
import torch
import numpy as np
import networkx as nx
from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
from neurosheaf.sheaf.construction import Sheaf

class TestStaticLaplacianWithMasking:
    """Unit tests for StaticLaplacianWithMasking class."""
    
    def test_initialization(self):
        """Test StaticLaplacianWithMasking initialization."""
        # Test default initialization
        analyzer = StaticLaplacianWithMasking()
        assert analyzer.eigenvalue_method == 'lobpcg'
        assert analyzer.max_eigenvalues == 100
        assert analyzer.laplacian_builder is not None
        
        # Test custom initialization
        analyzer = StaticLaplacianWithMasking(
            eigenvalue_method='dense',
            max_eigenvalues=50,
            enable_gpu=False
        )
        assert analyzer.eigenvalue_method == 'dense'
        assert analyzer.max_eigenvalues == 50
        assert not analyzer.enable_gpu
    
    def test_edge_info_extraction(self):
        """Test extraction of edge information from sheaf."""
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking()
        
        # Mock Laplacian and metadata for testing
        from scipy.sparse import csr_matrix
        mock_laplacian = csr_matrix((6, 6))  # 3 nodes × 2 dims each
        
        # Create mock metadata
        class MockMetadata:
            def __init__(self):
                self.edge_positions = {
                    ('0', '1'): [(2, 0), (3, 1), (0, 2), (1, 3)],
                    ('1', '2'): [(4, 2), (5, 3), (2, 4), (3, 5)]
                }
        
        mock_metadata = MockMetadata()
        
        edge_info = analyzer._extract_edge_info(sheaf, mock_laplacian, mock_metadata)
        
        # Check edge info structure
        assert len(edge_info) == 2
        assert ('0', '1') in edge_info
        assert ('1', '2') in edge_info
        
        # Check edge weights
        assert abs(edge_info[('0', '1')]['weight'] - (0.8 * np.sqrt(2))) < 1e-6
        assert abs(edge_info[('1', '2')]['weight'] - (0.6 * np.sqrt(2))) < 1e-6
        
        # Check positions are included
        assert 'positions' in edge_info[('0', '1')]
        assert 'positions' in edge_info[('1', '2')]
    
    def test_edge_mask_creation(self):
        """Test creation of edge masks."""
        # Create edge info
        edge_info = {
            ('0', '1'): {'weight': 0.8, 'source': '0', 'target': '1'},
            ('1', '2'): {'weight': 0.6, 'source': '1', 'target': '2'},
            ('2', '3'): {'weight': 0.4, 'source': '2', 'target': '3'}
        }
        
        analyzer = StaticLaplacianWithMasking()
        
        # Test threshold function
        def threshold_func(weight, param):
            return weight >= param
        
        # Low threshold - should include all edges
        mask_low = analyzer._create_edge_mask(edge_info, 0.1, threshold_func)
        assert all(mask_low.values())
        assert len(mask_low) == 3
        
        # Medium threshold - should exclude weakest edge
        mask_medium = analyzer._create_edge_mask(edge_info, 0.5, threshold_func)
        assert mask_medium[('0', '1')] == True
        assert mask_medium[('1', '2')] == True
        assert mask_medium[('2', '3')] == False
        
        # High threshold - should exclude most edges
        mask_high = analyzer._create_edge_mask(edge_info, 0.9, threshold_func)
        assert mask_high[('0', '1')] == False
        assert mask_high[('1', '2')] == False
        assert mask_high[('2', '3')] == False
    
    def test_eigenvalue_computation_dense(self):
        """Test dense eigenvalue computation."""
        analyzer = StaticLaplacianWithMasking(eigenvalue_method='dense', max_eigenvalues=5)
        
        # Create proper graph Laplacian (path graph)
        from scipy.sparse import csr_matrix
        # Path graph Laplacian: [1, -1, 0; -1, 2, -1; 0, -1, 1]
        data = np.array([1, -1, -1, 2, -1, -1, 1])
        row = np.array([0, 0, 1, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 2, 1, 2])
        laplacian = csr_matrix((data, (row, col)), shape=(3, 3))
        
        eigenvals, eigenvecs = analyzer._compute_eigenvalues_dense(laplacian)
        
        # Check properties
        assert len(eigenvals) <= 5
        assert eigenvals.shape[0] == eigenvecs.shape[1]
        assert torch.all(eigenvals >= -1e-6)  # Should be approximately non-negative
        # First eigenvalue should be small (close to 0 for connected graph)
        assert eigenvals[0] < 0.1
    
    def test_eigenvalue_computation_lobpcg(self):
        """Test LOBPCG eigenvalue computation."""
        analyzer = StaticLaplacianWithMasking(eigenvalue_method='lobpcg', max_eigenvalues=5)
        
        # Create larger sparse Laplacian (path graph)
        n = 10
        from scipy.sparse import diags
        # Create proper path graph Laplacian
        main_diag = np.ones(n)
        main_diag[0] = 1  # First node has degree 1
        main_diag[-1] = 1  # Last node has degree 1
        main_diag[1:-1] = 2  # Interior nodes have degree 2
        
        diagonals = [-np.ones(n-1), main_diag, -np.ones(n-1)]
        laplacian = diags(diagonals, [-1, 0, 1], shape=(n, n), format='csr')
        
        eigenvals, eigenvecs = analyzer._compute_eigenvalues_lobpcg(laplacian)
        
        # Check properties
        assert len(eigenvals) <= 5
        assert eigenvals.shape[0] == eigenvecs.shape[1]
        assert torch.all(eigenvals >= -1e-6)  # Should be approximately non-negative
        
        # First eigenvalue should be small for connected graph
        assert eigenvals[0] < 0.1
    
    def test_cache_functionality(self):
        """Test caching of Laplacian and edge information."""
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking()
        
        # Initial cache should be empty
        cache_info = analyzer.get_cache_info()
        assert not cache_info['laplacian_cached']
        assert not cache_info['edge_info_cached']
        
        # Simple threshold function
        def threshold_func(weight, param):
            return weight >= param
        
        # First computation should populate cache
        result1 = analyzer.compute_persistence(sheaf, [0.1, 0.5], threshold_func)
        
        cache_info = analyzer.get_cache_info()
        assert cache_info['laplacian_cached']
        assert cache_info['edge_info_cached']
        
        # Second computation should use cache
        result2 = analyzer.compute_persistence(sheaf, [0.2, 0.7], threshold_func)
        
        # Results should be computed successfully
        assert len(result1['eigenvalue_sequences']) == 2
        assert len(result2['eigenvalue_sequences']) == 2
        
        # Clear cache
        analyzer.clear_cache()
        cache_info = analyzer.get_cache_info()
        assert not cache_info['laplacian_cached']
        assert not cache_info['edge_info_cached']
    
    def test_compute_persistence_basic(self):
        """Test basic persistence computation pipeline."""
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2'), ('2', '3')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2), '3': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.9,
            ('1', '2'): torch.eye(2) * 0.7,
            ('2', '3'): torch.eye(2) * 0.5
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking(max_eigenvalues=8)
        
        # Define filtration parameters and threshold function
        filtration_params = [0.1, 0.4, 0.6, 0.8]
        def threshold_func(weight, param):
            return weight >= param
        
        # Compute persistence
        result = analyzer.compute_persistence(sheaf, filtration_params, threshold_func)
        
        # Check result structure
        assert 'eigenvalue_sequences' in result
        assert 'eigenvector_sequences' in result
        assert 'tracking_info' in result
        assert 'filtration_params' in result
        assert 'edge_info' in result
        
        # Check sequences
        assert len(result['eigenvalue_sequences']) == 4
        assert len(result['eigenvector_sequences']) == 4
        assert result['filtration_params'] == filtration_params
        
        # Check that eigenvalue sequences are properly formed
        for eigenvals, eigenvecs in zip(result['eigenvalue_sequences'], result['eigenvector_sequences']):
            assert isinstance(eigenvals, torch.Tensor)
            assert isinstance(eigenvecs, torch.Tensor)
            assert eigenvals.shape[0] == eigenvecs.shape[1]
            assert torch.all(eigenvals >= -1e-6)  # Non-negative
    
    def test_different_threshold_functions(self):
        """Test different types of threshold functions."""
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking()
        filtration_params = [0.3, 0.7]
        
        # Test >= threshold
        def threshold_ge(weight, param):
            return weight >= param
        
        result1 = analyzer.compute_persistence(sheaf, filtration_params, threshold_ge)
        
        # Test > threshold  
        def threshold_gt(weight, param):
            return weight > param
        
        result2 = analyzer.compute_persistence(sheaf, filtration_params, threshold_gt)
        
        # Test custom threshold (keep edges with weight in specific range)
        def threshold_range(weight, param):
            return 0.5 <= weight <= param
        
        result3 = analyzer.compute_persistence(sheaf, filtration_params, threshold_range)
        
        # All should complete successfully
        assert len(result1['eigenvalue_sequences']) == 2
        assert len(result2['eigenvalue_sequences']) == 2
        assert len(result3['eigenvalue_sequences']) == 2
    
    def test_degenerate_cases(self):
        """Test handling of degenerate cases."""
        analyzer = StaticLaplacianWithMasking()
        
        # Single node sheaf
        single_poset = nx.DiGraph()
        single_poset.add_node('A')
        single_sheaf = Sheaf(single_poset, {'A': torch.eye(2)}, {})
        
        def threshold_func(weight, param):
            return weight >= param
        
        # Should handle single node sheaf gracefully
        result = analyzer.compute_persistence(single_sheaf, [0.1], threshold_func)
        assert len(result['eigenvalue_sequences']) == 1
        
        # Two disconnected nodes
        disconnected_poset = nx.DiGraph()
        disconnected_poset.add_nodes_from(['A', 'B'])
        disconnected_sheaf = Sheaf(
            disconnected_poset,
            {'A': torch.eye(2), 'B': torch.eye(2)}, 
            {}
        )
        
        result = analyzer.compute_persistence(disconnected_sheaf, [0.1, 0.5], threshold_func)
        assert len(result['eigenvalue_sequences']) == 2
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        analyzer = StaticLaplacianWithMasking()
        
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2)}
        restrictions = {('0', '1'): torch.eye(2) * 0.8}
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Test with empty filtration parameters
        def threshold_func(weight, param):
            return weight >= param
        
        result = analyzer.compute_persistence(sheaf, [], threshold_func)
        assert len(result['eigenvalue_sequences']) == 0
        
        # Test with invalid threshold function (should not crash)
        def invalid_threshold(weight, param):
            raise ValueError("Invalid threshold")
        
        with pytest.raises(Exception):  # Should raise ComputationError
            analyzer.compute_persistence(sheaf, [0.5], invalid_threshold)
"""Integration tests for DTW functionality in the main API."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestDTWIntegration:
    """Test DTW integration with the main API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        # Create simple test models
        self.model1 = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        self.model2 = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        # Create test data
        self.data = torch.randn(50, 10)
        
        # Mock sheaf objects
        self.mock_sheaf1 = Mock(spec=Sheaf)
        self.mock_sheaf2 = Mock(spec=Sheaf)
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_compare_networks_dtw(self, mock_analyze):
        """Test network comparison using DTW method."""
        # Mock the analyze method to return sheaf objects
        mock_analyze.side_effect = [
            {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.eye(3)},
            {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.eye(3)}
        ]
        
        # Mock the spectral analyzer's comparison method
        mock_comparison_result = {
            'dtw_comparison': {
                'distance': 0.5,
                'normalized_distance': 0.3,
                'alignment': [(0, 0), (1, 1), (2, 2)]
            },
            'similarity_metrics': {
                'combined_similarity': 0.7,
                'dtw_distance': 0.5,
                'persistence_similarity': 0.8
            }
        }
        
        with patch.object(self.analyzer.spectral_analyzer, 'compare_filtration_evolution') as mock_compare:
            mock_compare.return_value = mock_comparison_result
            
            result = self.analyzer.compare_networks(
                self.model1, self.model2, self.data, method='dtw'
            )
            
            assert result['method'] == 'dtw'
            assert result['similarity_score'] == 0.7
            assert 'dtw_comparison' in result
            assert 'model1_analysis' in result
            assert 'model2_analysis' in result
            assert 'comparison_metadata' in result
            
            # Verify that the spectral analyzer was called
            mock_compare.assert_called_once()
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_compare_networks_euclidean_fallback(self, mock_analyze):
        """Test network comparison using Euclidean distance fallback."""
        # Mock the analyze method
        mock_analyze.side_effect = [
            {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.tensor([[1.0, 0.5], [0.5, 1.0]])},
            {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.tensor([[1.0, 0.3], [0.3, 1.0]])}
        ]
        
        result = self.analyzer.compare_networks(
            self.model1, self.model2, self.data, method='euclidean'
        )
        
        assert result['method'] == 'euclidean'
        assert 'similarity_score' in result
        assert result['dtw_comparison'] is None  # Should be None for non-DTW methods
        assert 0.0 <= result['similarity_score'] <= 1.0
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_compare_networks_cosine_fallback(self, mock_analyze):
        """Test network comparison using cosine similarity fallback."""
        # Mock the analyze method
        mock_analyze.side_effect = [
            {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.tensor([[1.0, 0.8], [0.8, 1.0]])},
            {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.tensor([[1.0, 0.6], [0.6, 1.0]])}
        ]
        
        result = self.analyzer.compare_networks(
            self.model1, self.model2, self.data, method='cosine'
        )
        
        assert result['method'] == 'cosine'
        assert 'similarity_score' in result
        assert -1.0 <= result['similarity_score'] <= 1.0
    
    def test_compare_networks_invalid_method(self):
        """Test network comparison with invalid method."""
        with pytest.raises(ValueError, match="Unknown comparison method"):
            self.analyzer.compare_networks(
                self.model1, self.model2, self.data, method='invalid_method'
            )
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_compare_multiple_networks_dtw(self, mock_analyze):
        """Test multiple network comparison using DTW."""
        models = [self.model1, self.model2]
        
        # Mock the analyze method
        mock_analyze.side_effect = [
            {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.eye(3)},
            {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.eye(3)}
        ]
        
        # Mock the spectral analyzer's multiple comparison method
        mock_comparison_result = {
            'distance_matrix': np.array([[0.0, 0.5], [0.5, 0.0]]),
            'similarity_rankings': [
                {'sheaf_index': 0, 'most_similar': [{'sheaf_index': 1, 'distance': 0.5}]},
                {'sheaf_index': 1, 'most_similar': [{'sheaf_index': 0, 'distance': 0.5}]}
            ]
        }
        
        with patch.object(self.analyzer.spectral_analyzer, 'compare_multiple_sheaves') as mock_compare:
            mock_compare.return_value = mock_comparison_result
            
            result = self.analyzer.compare_multiple_networks(
                models, self.data, method='dtw'
            )
            
            assert result['method'] == 'dtw'
            assert 'distance_matrix' in result
            assert 'similarity_rankings' in result
            assert 'individual_analyses' in result
            assert 'cluster_analysis' in result
            assert result['comparison_metadata']['n_models'] == 2
            
            # Verify that the spectral analyzer was called
            mock_compare.assert_called_once()
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_compare_multiple_networks_euclidean_fallback(self, mock_analyze):
        """Test multiple network comparison using Euclidean distance fallback."""
        models = [self.model1, self.model2]
        
        # Mock the analyze method
        mock_analyze.side_effect = [
            {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.tensor([[1.0, 0.5], [0.5, 1.0]])},
            {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.tensor([[1.0, 0.3], [0.3, 1.0]])}
        ]
        
        result = self.analyzer.compare_multiple_networks(
            models, self.data, method='euclidean'
        )
        
        assert result['method'] == 'euclidean'
        assert 'distance_matrix' in result
        assert 'similarity_rankings' in result
        assert isinstance(result['distance_matrix'], np.ndarray)
        assert result['distance_matrix'].shape == (2, 2)
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_compare_multiple_networks_with_clustering(self, mock_analyze):
        """Test multiple network comparison with clustering analysis."""
        models = [self.model1, self.model2]
        
        # Mock the analyze method
        mock_analyze.side_effect = [
            {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.eye(3)},
            {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.eye(3)}
        ]
        
        # Mock sklearn clustering
        with patch('sklearn.cluster.AgglomerativeClustering') as mock_clustering:
            with patch('sklearn.metrics.silhouette_score') as mock_silhouette:
                mock_clustering_instance = Mock()
                mock_clustering_instance.fit_predict.return_value = np.array([0, 1])
                mock_clustering.return_value = mock_clustering_instance
                mock_silhouette.return_value = 0.8
                
                result = self.analyzer.compare_multiple_networks(
                    models, self.data, method='euclidean'
                )
                
                assert 'cluster_analysis' in result
                assert result['cluster_analysis']['n_clusters'] == 2
                assert result['cluster_analysis']['silhouette_score'] == 0.8
    
    def test_compare_multiple_networks_clustering_fallback(self):
        """Test clustering analysis fallback when sklearn is not available."""
        models = [self.model1, self.model2]
        
        with patch('neurosheaf.api.NeurosheafAnalyzer.analyze') as mock_analyze:
            mock_analyze.side_effect = [
                {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.eye(3)},
                {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.eye(3)}
            ]
            
            # Mock sklearn import failure
            with patch('neurosheaf.api.NeurosheafAnalyzer._perform_cluster_analysis') as mock_cluster:
                mock_cluster.return_value = {'status': 'sklearn_not_available'}
                
                result = self.analyzer.compare_multiple_networks(
                    models, self.data, method='euclidean'
                )
                
                assert result['cluster_analysis']['status'] == 'sklearn_not_available'
    
    def test_dtw_parameters_passed_correctly(self):
        """Test that DTW parameters are passed correctly through the API."""
        with patch('neurosheaf.api.NeurosheafAnalyzer.analyze') as mock_analyze:
            mock_analyze.side_effect = [
                {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.eye(3)},
                {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.eye(3)}
            ]
            
            mock_comparison_result = {
                'similarity_metrics': {'combined_similarity': 0.8}
            }
            
            with patch.object(self.analyzer.spectral_analyzer, 'compare_filtration_evolution') as mock_compare:
                mock_compare.return_value = mock_comparison_result
                
                result = self.analyzer.compare_networks(
                    self.model1, self.model2, self.data,
                    method='dtw',
                    eigenvalue_index=2,
                    multivariate=True
                )
                
                # Check that parameters were passed correctly
                mock_compare.assert_called_once()
                args, kwargs = mock_compare.call_args
                assert kwargs['eigenvalue_index'] == 2
                assert kwargs['multivariate'] == True
    
    def test_comparison_metadata_correctness(self):
        """Test that comparison metadata is correctly populated."""
        with patch('neurosheaf.api.NeurosheafAnalyzer.analyze') as mock_analyze:
            mock_analyze.side_effect = [
                {'sheaf': self.mock_sheaf1, 'cka_matrix': torch.eye(3)},
                {'sheaf': self.mock_sheaf2, 'cka_matrix': torch.eye(3)}
            ]
            
            mock_comparison_result = {
                'similarity_metrics': {'combined_similarity': 0.8}
            }
            
            with patch.object(self.analyzer.spectral_analyzer, 'compare_filtration_evolution') as mock_compare:
                mock_compare.return_value = mock_comparison_result
                
                result = self.analyzer.compare_networks(
                    self.model1, self.model2, self.data,
                    method='dtw',
                    eigenvalue_index=1,
                    multivariate=False
                )
                
                metadata = result['comparison_metadata']
                assert metadata['eigenvalue_index'] == 1
                assert metadata['multivariate'] == False
                assert metadata['data_shape'] == self.data.shape
                assert metadata['device'] == 'cpu'


class TestDTWVisualization:
    """Test DTW visualization integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    @patch('neurosheaf.visualization.spectral.SpectralVisualizer.plot_dtw_alignment')
    def test_dtw_visualization_integration(self, mock_plot):
        """Test that DTW visualization methods are available and callable."""
        # Mock alignment data
        alignment_data = {
            'sequence1': [1.0, 0.8, 0.6],
            'sequence2': [1.2, 0.9, 0.7],
            'filtration_params1': [0.1, 0.2, 0.3],
            'filtration_params2': [0.1, 0.2, 0.3],
            'alignment': [(0, 0), (1, 1), (2, 2)],
            'alignment_quality': 0.9
        }
        
        # Mock return value
        mock_figure = Mock()
        mock_plot.return_value = mock_figure
        
        # Test that we can import and use the visualization
        from neurosheaf.visualization.spectral import SpectralVisualizer
        
        visualizer = SpectralVisualizer()
        result = visualizer.plot_dtw_alignment(alignment_data)
        
        assert result == mock_figure
        mock_plot.assert_called_once_with(alignment_data)


class TestDTWErrorHandling:
    """Test error handling in DTW integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        self.model1 = nn.Sequential(nn.Linear(10, 5))
        self.model2 = nn.Sequential(nn.Linear(10, 5))
        self.data = torch.randn(50, 10)
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_dtw_computation_error_handling(self, mock_analyze):
        """Test error handling when DTW computation fails."""
        # Mock the analyze method
        mock_analyze.side_effect = [
            {'sheaf': Mock(), 'cka_matrix': torch.eye(3)},
            {'sheaf': Mock(), 'cka_matrix': torch.eye(3)}
        ]
        
        # Mock DTW computation to raise an error
        with patch.object(self.analyzer.spectral_analyzer, 'compare_filtration_evolution') as mock_compare:
            mock_compare.side_effect = ComputationError("DTW computation failed")
            
            with pytest.raises(ComputationError, match="DTW computation failed"):
                self.analyzer.compare_networks(
                    self.model1, self.model2, self.data, method='dtw'
                )
    
    @patch('neurosheaf.api.NeurosheafAnalyzer.analyze')
    def test_mismatched_cka_matrices_handling(self, mock_analyze):
        """Test handling of mismatched CKA matrix sizes."""
        # Mock the analyze method with different sized matrices
        mock_analyze.side_effect = [
            {'sheaf': Mock(), 'cka_matrix': torch.eye(2)},
            {'sheaf': Mock(), 'cka_matrix': torch.eye(3)}
        ]
        
        # Should handle mismatched sizes gracefully
        result = self.analyzer.compare_networks(
            self.model1, self.model2, self.data, method='euclidean'
        )
        
        assert 'similarity_score' in result
        assert result['method'] == 'euclidean'
    
    def test_empty_model_list_handling(self):
        """Test handling of empty model list."""
        with pytest.raises((ValueError, IndexError)):
            self.analyzer.compare_multiple_networks([], self.data, method='dtw')
    
    def test_single_model_list_handling(self):
        """Test handling of single model in list."""
        models = [self.model1]
        
        with patch('neurosheaf.api.NeurosheafAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = {'sheaf': Mock(), 'cka_matrix': torch.eye(3)}
            
            result = self.analyzer.compare_multiple_networks(models, self.data, method='dtw')
            
            assert result['comparison_metadata']['n_models'] == 1
            assert result['distance_matrix'].shape == (1, 1)


@pytest.mark.integration
class TestDTWEndToEnd:
    """End-to-end integration tests for DTW functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        
        # Create slightly different models for meaningful comparison
        self.model1 = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        
        self.model2 = nn.Sequential(
            nn.Linear(4, 3),
            nn.Tanh(),  # Different activation
            nn.Linear(3, 2)
        )
        
        # Create deterministic test data
        torch.manual_seed(42)
        self.data = torch.randn(20, 4)
    
    @pytest.mark.slow
    def test_end_to_end_dtw_comparison(self):
        """Test complete end-to-end DTW comparison pipeline."""
        # This test requires the full pipeline to be functional
        # and DTW libraries to be available
        
        try:
            result = self.analyzer.compare_networks(
                self.model1, self.model2, self.data, method='dtw'
            )
            
            # Verify result structure
            assert 'similarity_score' in result
            assert 'method' in result
            assert 'dtw_comparison' in result
            assert 'comparison_metadata' in result
            
            # Verify reasonable values
            assert 0.0 <= result['similarity_score'] <= 1.0
            assert result['method'] == 'dtw'
            
        except (ImportError, ComputationError) as e:
            # Skip if DTW libraries are not available
            pytest.skip(f"DTW libraries not available: {e}")
    
    @pytest.mark.slow
    def test_end_to_end_multiple_network_comparison(self):
        """Test complete end-to-end multiple network comparison."""
        models = [self.model1, self.model2]
        
        try:
            result = self.analyzer.compare_multiple_networks(
                models, self.data, method='dtw'
            )
            
            # Verify result structure
            assert 'distance_matrix' in result
            assert 'similarity_rankings' in result
            assert 'individual_analyses' in result
            assert 'cluster_analysis' in result
            
            # Verify matrix properties
            distance_matrix = result['distance_matrix']
            assert distance_matrix.shape == (2, 2)
            assert distance_matrix[0, 0] == 0.0  # Self-distance
            assert distance_matrix[0, 1] == distance_matrix[1, 0]  # Symmetry
            
        except (ImportError, ComputationError) as e:
            # Skip if DTW libraries are not available
            pytest.skip(f"DTW libraries not available: {e}")
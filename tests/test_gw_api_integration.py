"""Comprehensive integration tests for GW API functionality.

This module tests the complete GW integration through the high-level API,
ensuring that all components work together correctly and that the API
provides the expected interface and behavior.

Test Categories:
1. Basic GW API functionality
2. Configuration handling
3. Method routing and validation
4. Network comparison with GW
5. Mixed-method workflows
6. Error handling and edge cases
7. Performance and regression tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestGWAPIBasicFunctionality:
    """Test basic GW API functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.simple_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4)
        )
        self.test_data = torch.randn(30, 10)
        
    def test_gw_method_parameter(self):
        """Test that GW method parameter is accepted."""
        # Should accept gromov_wasserstein method
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(
                self.simple_model, self.test_data,
                method='gromov_wasserstein'
            )
            
            # Should have called SheafBuilder with GW method
            mock_builder.assert_called_once_with(restriction_method='gromov_wasserstein')
            
    def test_default_gw_config_creation(self):
        """Test that default GW config is created when method='gromov_wasserstein'."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance 
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(
                self.simple_model, self.test_data,
                method='gromov_wasserstein'
            )
            
            # Should have created default GW config
            assert result['gw_config'] is not None
            assert result['gw_config']['epsilon'] == 0.1  # Default value
            
    def test_custom_gw_config_handling(self):
        """Test custom GW configuration handling."""
        custom_config = GWConfig(
            epsilon=0.05,
            max_iter=800,
            tolerance=1e-10
        )
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(
                self.simple_model, self.test_data,
                method='gromov_wasserstein',
                gw_config=custom_config
            )
            
            # Should use custom config
            assert result['gw_config']['epsilon'] == 0.05
            assert result['gw_config']['max_iter'] == 800
            
            # Should pass config to builder
            mock_instance.build_from_activations.assert_called_once()
            call_kwargs = mock_instance.build_from_activations.call_args[1]
            assert call_kwargs['gw_config'] == custom_config
            
    def test_result_structure_with_gw(self):
        """Test that GW analysis results have correct structure."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_sheaf = MagicMock()
            mock_instance.build_from_activations.return_value = mock_sheaf
            
            result = self.analyzer.analyze(
                self.simple_model, self.test_data,
                method='gromov_wasserstein'
            )
            
            # Check required fields are present
            required_fields = [
                'analysis_type', 'sheaf', 'construction_method', 
                'gw_config', 'device_info', 'performance'
            ]
            for field in required_fields:
                assert field in result
                
            assert result['construction_method'] == 'gromov_wasserstein'
            assert result['analysis_type'] == 'undirected'


class TestGWAPIValidation:
    """Test API validation and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.simple_model = nn.Sequential(nn.Linear(5, 3))
        self.test_data = torch.randn(20, 5)
        
    def test_invalid_method_validation(self):
        """Test validation of invalid method names."""
        with pytest.raises(ValidationError, match="Unknown method"):
            self.analyzer.analyze(
                self.simple_model, self.test_data,
                method='invalid_method'
            )
            
    def test_valid_method_names(self):
        """Test that all valid method names are accepted."""
        valid_methods = ['procrustes', 'gromov_wasserstein', 'whitened_procrustes']
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder'):
            for method in valid_methods:
                # Should not raise validation error
                try:
                    self.analyzer.analyze(
                        self.simple_model, self.test_data,
                        method=method
                    )
                except ValidationError:
                    pytest.fail(f"Valid method '{method}' was rejected")
                except Exception:
                    # Other exceptions are fine, we're just testing validation
                    pass
                    
    def test_gw_config_validation(self):
        """Test GW config validation."""
        # Invalid epsilon
        invalid_config = GWConfig(epsilon=-0.1)
        
        with pytest.raises(ValidationError):
            self.analyzer.analyze(
                self.simple_model, self.test_data,
                method='gromov_wasserstein',
                gw_config=invalid_config
            )
            
    def test_gw_config_ignored_for_other_methods(self):
        """Test that GW config is ignored for non-GW methods."""
        gw_config = GWConfig(epsilon=0.05)
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(
                self.simple_model, self.test_data,
                method='procrustes',  # Not GW method
                gw_config=gw_config   # Should be ignored
            )
            
            # Config should not be passed for procrustes method
            assert result['gw_config'] is None


class TestGWNetworkComparison:
    """Test network comparison with GW methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        
        # Create simple models with different architectures
        self.model1 = nn.Sequential(
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4)
        )
        
        self.model2 = nn.Sequential(
            nn.Linear(8, 5),  # Different intermediate size
            nn.ReLU(),
            nn.Linear(5, 4)
        )
        
        self.test_data = torch.randn(50, 8)
        
    def test_compare_networks_with_gw(self):
        """Test network comparison using GW method."""
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            # Mock analyze results
            mock_sheaf = MagicMock()
            mock_analyze.return_value = {
                'sheaf': mock_sheaf,
                'construction_method': 'gromov_wasserstein',
                'gw_config': {'epsilon': 0.1}
            }
            
            with patch.object(self.analyzer, '_compare_networks_simple') as mock_simple:
                mock_simple.return_value = {'similarity_score': 0.75}
                
                result = self.analyzer.compare_networks(
                    self.model1, self.model2, self.test_data,
                    comparison_method='euclidean',
                    sheaf_method='gromov_wasserstein'
                )
                
                # Should call analyze with GW method for both models
                assert mock_analyze.call_count == 2
                for call in mock_analyze.call_args_list:
                    kwargs = call[1]
                    assert kwargs['method'] == 'gromov_wasserstein'
                    
                # Check result structure
                assert 'similarity_score' in result
                assert result['sheaf_method'] == 'gromov_wasserstein'
                assert result['comparison_method'] == 'euclidean'
                
    def test_compare_multiple_networks_with_gw(self):
        """Test multiple network comparison with GW."""
        models = [self.model1, self.model2]
        
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            mock_sheaf = MagicMock()
            mock_analyze.return_value = {
                'sheaf': mock_sheaf,
                'construction_method': 'gromov_wasserstein'
            }
            
            with patch.object(self.analyzer, '_compare_multiple_simple') as mock_multiple:
                mock_multiple.return_value = (
                    np.array([[0, 0.5], [0.5, 0]]),  # Distance matrix
                    [{'model_index': 0, 'most_similar': []}]  # Rankings
                )
                
                result = self.analyzer.compare_multiple_networks(
                    models, self.test_data,
                    comparison_method='cosine',
                    sheaf_method='gromov_wasserstein'
                )
                
                # Should analyze each model with GW
                assert mock_analyze.call_count == len(models)
                
                # Check result structure
                assert 'distance_matrix' in result
                assert result['sheaf_method'] == 'gromov_wasserstein'
                assert result['comparison_method'] == 'cosine'
                
    def test_mixed_method_comparison_parameters(self):
        """Test that comparison method and sheaf method are handled separately."""
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'sheaf': MagicMock()}
            
            with patch.object(self.analyzer, '_compare_networks_simple') as mock_simple:
                mock_simple.return_value = {'similarity_score': 0.8}
                
                self.analyzer.compare_networks(
                    self.model1, self.model2, self.test_data,
                    comparison_method='cosine',      # Comparison algorithm
                    sheaf_method='gromov_wasserstein'  # Sheaf construction
                )
                
                # Should call simple comparison with 'cosine' method
                mock_simple.assert_called_once()
                args = mock_simple.call_args[0]
                assert args[2] == 'cosine'  # comparison_method parameter


class TestGWDirectedAnalysis:
    """Test GW integration with directed analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        self.test_data = torch.randn(25, 6)
        
    def test_directed_analysis_with_gw(self):
        """Test directed analysis with GW method."""
        # Note: Current implementation doesn't support GW for directed analysis
        # but we should test the parameter passing
        
        with patch('neurosheaf.directed_sheaf.DirectedSheafBuilder') as mock_directed:
            mock_instance = MagicMock()
            mock_directed.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            with patch('neurosheaf.directed_sheaf.DirectedSheafAdapter') as mock_adapter:
                mock_adapter_instance = MagicMock()
                mock_adapter.return_value = mock_adapter_instance
                mock_adapter_instance.adapt_for_spectral_analysis.return_value = (MagicMock(), MagicMock())
                
                result = self.analyzer.analyze(
                    self.model, self.test_data,
                    method='gromov_wasserstein',
                    directed=True
                )
                
                # Should include method information in results
                assert result['construction_method'] == 'gromov_wasserstein'
                assert result['analysis_type'] == 'directed'
                
    def test_analyze_directed_convenience_method(self):
        """Test analyze_directed convenience method with GW."""
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'result': 'test'}
            
            result = self.analyzer.analyze_directed(
                self.model, self.test_data,
                method='gromov_wasserstein'
            )
            
            # Should call main analyze method with correct parameters
            mock_analyze.assert_called_once()
            call_kwargs = mock_analyze.call_args[1]
            assert call_kwargs['method'] == 'gromov_wasserstein'
            assert call_kwargs['directed'] is True
            
    def test_compare_directed_undirected_with_gw(self):
        """Test directed vs undirected comparison with GW."""
        with patch.object(self.analyzer, 'analyze_directed') as mock_directed:
            mock_directed.return_value = {'analysis_type': 'directed'}
            
            with patch.object(self.analyzer, 'analyze') as mock_undirected:
                mock_undirected.return_value = {'analysis_type': 'undirected'}
                
                with patch.object(self.analyzer, '_compute_comparison_metrics') as mock_compare:
                    mock_compare.return_value = {'comparison': 'test'}
                    
                    result = self.analyzer.compare_directed_undirected(
                        self.model, self.test_data,
                        method='gromov_wasserstein'
                    )
                    
                    # Should call both directed and undirected with GW method
                    directed_call = mock_directed.call_args[1]
                    undirected_call = mock_undirected.call_args[1]
                    
                    assert directed_call['method'] == 'gromov_wasserstein'
                    assert undirected_call['method'] == 'gromov_wasserstein'


class TestGWAPIErrorHandling:
    """Test error handling in GW API integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(nn.Linear(4, 2))
        self.test_data = torch.randn(15, 4)
        
    def test_sheaf_builder_error_handling(self):
        """Test handling of errors from SheafBuilder."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.side_effect = Exception("GW computation failed")
            
            with pytest.raises(ComputationError, match="Analysis failed"):
                self.analyzer.analyze(
                    self.model, self.test_data,
                    method='gromov_wasserstein'
                )
                
    def test_pot_library_unavailable_handling(self):
        """Test graceful handling when POT library is unavailable."""
        # This would typically be handled at the SheafBuilder level,
        # but we should ensure the API propagates the error correctly
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.side_effect = ImportError("POT library not available")
            
            with pytest.raises(ComputationError, match="Analysis failed"):
                self.analyzer.analyze(
                    self.model, self.test_data,
                    method='gromov_wasserstein'
                )
                
    def test_invalid_model_architecture(self):
        """Test handling of invalid model architectures."""
        # Test with empty model
        empty_model = nn.Sequential()
        
        with pytest.raises(ValidationError, match="Data cannot be empty"):
            self.analyzer.analyze(empty_model, torch.empty(0, 4))
            
    def test_mismatched_data_dimensions(self):
        """Test handling of mismatched data dimensions."""
        wrong_size_data = torch.randn(20, 10)  # Model expects 4, data has 10
        
        # Should be caught at the model execution level
        with pytest.raises(ComputationError):
            self.analyzer.analyze(self.model, wrong_size_data, method='gromov_wasserstein')


class TestGWAPIBackwardCompatibility:
    """Test backward compatibility of GW API integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(
            nn.Linear(7, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        self.test_data = torch.randn(40, 7)
        
    def test_default_method_unchanged(self):
        """Test that default behavior is unchanged (uses Procrustes)."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            # Call without method parameter (should default to procrustes)
            result = self.analyzer.analyze(self.model, self.test_data)
            
            # Should use default scaled_procrustes method
            mock_builder.assert_called_once_with(restriction_method='scaled_procrustes')
            assert result['construction_method'] == 'procrustes'
            
    def test_existing_parameters_still_work(self):
        """Test that existing parameters continue to work."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            # Call with traditional parameters
            result = self.analyzer.analyze(
                self.model, self.test_data,
                preserve_eigenvalues=True,
                use_gram_regularization=True
            )
            
            # Should pass through existing parameters
            call_kwargs = mock_instance.build_from_activations.call_args[1]
            assert call_kwargs['preserve_eigenvalues'] is True
            assert call_kwargs['use_gram_regularization'] is True
            
    def test_compare_networks_backward_compatibility(self):
        """Test that old compare_networks calls still work."""
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'sheaf': MagicMock()}
            
            with patch.object(self.analyzer, '_compare_networks_simple') as mock_simple:
                mock_simple.return_value = {'similarity_score': 0.85}
                
                # Old-style call (should still work with new parameter names)
                result = self.analyzer.compare_networks(
                    self.model, self.model, self.test_data,
                    comparison_method='euclidean'  # Was 'method' before
                )
                
                # Should work and default to procrustes sheaf method
                assert 'similarity_score' in result
                
                # Should call analyze with default method
                for call in mock_analyze.call_args_list:
                    kwargs = call[1]
                    assert kwargs['method'] == 'procrustes'  # Default


class TestGWAPIPerformance:
    """Performance and regression tests for GW API."""
    
    def setup_method(self):
        """Set up test fixtures.""" 
        self.analyzer = NeurosheafAnalyzer()
        
    def test_gw_config_serialization(self):
        """Test GW config serialization in results."""
        custom_config = GWConfig(
            epsilon=0.07,
            max_iter=600,
            use_gpu=False
        )
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(
                nn.Sequential(nn.Linear(3, 2)), torch.randn(10, 3),
                method='gromov_wasserstein',
                gw_config=custom_config
            )
            
            # Config should be serialized in results
            assert result['gw_config']['epsilon'] == 0.07
            assert result['gw_config']['max_iter'] == 600
            assert result['gw_config']['use_gpu'] is False
            
    def test_memory_cleanup(self):
        """Test that GW analysis doesn't leak memory."""
        # This is more of a smoke test - real memory testing would need more infrastructure
        model = nn.Sequential(
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 8)
        )
        data = torch.randn(60, 12)
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            # Multiple analyses should not accumulate state
            for i in range(3):
                result = self.analyzer.analyze(
                    model, data,
                    method='gromov_wasserstein'
                )
                
                # Each should create fresh config if none provided
                assert result['gw_config'] is not None
                
    def test_api_overhead(self):
        """Test that API overhead is minimal."""
        model = nn.Sequential(nn.Linear(5, 3))
        data = torch.randn(20, 5)
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            import time
            start_time = time.time()
            
            result = self.analyzer.analyze(
                model, data,
                method='gromov_wasserstein'
            )
            
            api_time = time.time() - start_time
            
            # API overhead should be minimal (< 1ms for this simple case)
            assert api_time < 0.001 or api_time < result['construction_time'] * 0.1


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
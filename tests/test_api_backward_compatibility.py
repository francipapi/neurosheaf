"""Backward compatibility tests for NeurosheafAnalyzer API.

This module ensures that the integration of GW functionality doesn't break
existing API behavior. All existing code should continue to work unchanged
after adding GW support.

Test Categories:
1. Default behavior preservation
2. Existing parameter handling
3. Return value structure compatibility
4. Method signature compatibility
5. Error handling consistency
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestDefaultBehaviorPreservation:
    """Test that default behavior is preserved after GW integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4)
        )
        self.test_data = torch.randn(30, 8)
        
    def test_default_analyze_method(self):
        """Test that analyze() without method parameter uses Procrustes."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            # Call analyze without method parameter (old behavior)
            result = self.analyzer.analyze(self.model, self.test_data)
            
            # Should use scaled_procrustes (default)
            mock_builder.assert_called_once_with(restriction_method='scaled_procrustes')
            assert result['construction_method'] == 'procrustes'
            
    def test_default_result_structure(self):
        """Test that default result structure is preserved."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_sheaf = MagicMock()
            mock_instance.build_from_activations.return_value = mock_sheaf
            
            result = self.analyzer.analyze(self.model, self.test_data)
            
            # Check that all original fields are present
            expected_fields = [
                'analysis_type', 'sheaf', 'device_info', 'memory_info',
                'performance', 'construction_time', 'preserve_eigenvalues',
                'use_gram_regularization', 'regularization_config'
            ]
            
            for field in expected_fields:
                assert field in result, f"Missing field: {field}"
                
            # New fields should also be present but with appropriate defaults
            assert result['construction_method'] == 'procrustes'
            assert result['gw_config'] is None  # Should be None for non-GW methods
            
    def test_directed_analysis_default(self):
        """Test that directed analysis default behavior is preserved."""
        with patch('neurosheaf.directed_sheaf.DirectedSheafBuilder') as mock_directed:
            mock_instance = MagicMock()
            mock_directed.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            with patch('neurosheaf.directed_sheaf.DirectedSheafAdapter') as mock_adapter:
                mock_adapter_instance = MagicMock()
                mock_adapter.return_value = mock_adapter_instance
                mock_adapter_instance.adapt_for_spectral_analysis.return_value = (MagicMock(), MagicMock())
                
                # Old-style directed analysis call
                result = self.analyzer.analyze(
                    self.model, self.test_data,
                    directed=True
                )
                
                assert result['analysis_type'] == 'directed'
                assert result['construction_method'] == 'procrustes'  # Default
                

class TestExistingParameterHandling:
    """Test that existing parameters continue to work correctly."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(nn.Linear(5, 3))
        self.test_data = torch.randn(20, 5)
        
    def test_preserve_eigenvalues_parameter(self):
        """Test that preserve_eigenvalues parameter still works."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(
                self.model, self.test_data,
                preserve_eigenvalues=True
            )
            
            # Should pass parameter through to builder
            call_kwargs = mock_instance.build_from_activations.call_args[1]
            assert call_kwargs['preserve_eigenvalues'] is True
            
            # Should appear in results
            assert result['preserve_eigenvalues'] is True
            
    def test_regularization_parameters(self):
        """Test that regularization parameters still work."""
        reg_config = {'lambda': 0.01, 'method': 'tikhonov'}
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(
                self.model, self.test_data,
                use_gram_regularization=True,
                regularization_config=reg_config
            )
            
            # Should pass parameters through
            call_kwargs = mock_instance.build_from_activations.call_args[1]
            assert call_kwargs['use_gram_regularization'] is True
            assert call_kwargs['regularization_config'] == reg_config
            
            # Should appear in results
            assert result['use_gram_regularization'] is True
            assert result['regularization_config'] == reg_config
            
    def test_directed_parameters(self):
        """Test that directed analysis parameters still work."""
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
                    directed=True,
                    directionality_parameter=0.3
                )
                
                assert result['directionality_parameter'] == 0.3
                

class TestMethodSignatureCompatibility:
    """Test that method signatures remain compatible."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(nn.Linear(4, 2))
        self.test_data = torch.randn(15, 4)
        
    def test_analyze_method_signature(self):
        """Test that analyze method accepts old parameter patterns."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            # All these calls should work (different parameter combinations)
            test_calls = [
                # Minimal call
                lambda: self.analyzer.analyze(self.model, self.test_data),
                
                # With some optional parameters
                lambda: self.analyzer.analyze(
                    self.model, self.test_data,
                    preserve_eigenvalues=True
                ),
                
                # With directed analysis
                lambda: self.analyzer.analyze(
                    self.model, self.test_data,
                    directed=True,
                    directionality_parameter=0.25
                ),
                
                # With regularization
                lambda: self.analyzer.analyze(
                    self.model, self.test_data,
                    use_gram_regularization=True
                ),
            ]
            
            for call_func in test_calls:
                try:
                    result = call_func()
                    # Should return dict with expected structure
                    assert isinstance(result, dict)
                    assert 'analysis_type' in result
                except Exception as e:
                    pytest.fail(f"Backward compatible call failed: {e}")
                    
    def test_analyze_directed_method_signature(self):
        """Test that analyze_directed method signature is backward compatible."""
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'analysis_type': 'directed'}
            
            # Old-style call should work
            result = self.analyzer.analyze_directed(
                self.model, self.test_data,
                directionality_parameter=0.4
            )
            
            # Should call main analyze method
            mock_analyze.assert_called_once()
            call_kwargs = mock_analyze.call_args[1]
            assert call_kwargs['directed'] is True
            assert call_kwargs['directionality_parameter'] == 0.4
            
    def test_compare_directed_undirected_signature(self):
        """Test backward compatibility of compare_directed_undirected."""
        with patch.object(self.analyzer, 'analyze_directed') as mock_directed:
            mock_directed.return_value = {'analysis_type': 'directed'}
            
            with patch.object(self.analyzer, 'analyze') as mock_undirected:
                mock_undirected.return_value = {'analysis_type': 'undirected'}
                
                with patch.object(self.analyzer, '_compute_comparison_metrics') as mock_compare:
                    mock_compare.return_value = {'comparison': 'test'}
                    
                    # Old-style call
                    result = self.analyzer.compare_directed_undirected(
                        self.model, self.test_data,
                        directionality_parameter=0.2
                    )
                    
                    # Should work
                    assert 'directed_results' in result
                    assert 'undirected_results' in result


class TestCompareMethodsBackwardCompatibility:
    """Test backward compatibility of network comparison methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model1 = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 2))
        self.model2 = nn.Sequential(nn.Linear(6, 3), nn.ReLU(), nn.Linear(3, 2))
        self.test_data = torch.randn(25, 6)
        
    def test_compare_networks_old_parameter_style(self):
        """Test that old compare_networks parameter style still works."""
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'sheaf': MagicMock()}
            
            with patch.object(self.analyzer, '_compare_networks_simple') as mock_simple:
                mock_simple.return_value = {'similarity_score': 0.8}
                
                # Note: The old 'method' parameter is now 'comparison_method'
                # We need to test the new interface, but ensure old-style usage patterns work
                result = self.analyzer.compare_networks(
                    self.model1, self.model2, self.test_data,
                    comparison_method='euclidean',  # This is the new name
                    eigenvalue_index=1
                )
                
                # Should work with default sheaf method
                assert 'similarity_score' in result
                assert result['comparison_method'] == 'euclidean'
                assert result['sheaf_method'] == 'procrustes'  # Default
                
    def test_compare_multiple_networks_compatibility(self):
        """Test backward compatibility of compare_multiple_networks."""
        models = [self.model1, self.model2]
        
        with patch.object(self.analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'sheaf': MagicMock()}
            
            with patch.object(self.analyzer, '_compare_multiple_simple') as mock_multiple:
                mock_multiple.return_value = (
                    torch.tensor([[0, 0.5], [0.5, 0]]),  # Distance matrix
                    []  # Rankings
                )
                
                result = self.analyzer.compare_multiple_networks(
                    models, self.test_data,
                    comparison_method='cosine'
                )
                
                # Should work with defaults
                assert 'distance_matrix' in result
                assert result['comparison_method'] == 'cosine'
                assert result['sheaf_method'] == 'procrustes'  # Default


class TestErrorHandlingConsistency:
    """Test that error handling behavior is consistent with old API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        
    def test_validation_error_types(self):
        """Test that validation errors are consistent."""
        model = nn.Sequential(nn.Linear(3, 1))
        
        # Invalid data (empty)
        with pytest.raises(ValidationError, match="Data cannot be empty"):
            self.analyzer.analyze(model, torch.empty(0, 3))
            
        # Invalid model type
        with pytest.raises(ValidationError, match="Model must be a PyTorch nn.Module"):
            self.analyzer.analyze("not a model", torch.randn(10, 3))
            
        # Invalid data type
        with pytest.raises(ValidationError, match="Data must be a torch.Tensor"):
            self.analyzer.analyze(model, "not tensor data")
            
    def test_computation_error_handling(self):
        """Test that computation errors are handled consistently."""
        model = nn.Sequential(nn.Linear(3, 1))
        data = torch.randn(10, 3)
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.side_effect = RuntimeError("Computation failed")
            
            with pytest.raises(ComputationError, match="Analysis failed"):
                self.analyzer.analyze(model, data)
                
    def test_device_handling_consistency(self):
        """Test that device handling is consistent."""
        model = nn.Sequential(nn.Linear(3, 1))
        data = torch.randn(10, 3)
        
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(model, data)
            
            # Should include device info like before
            assert 'device_info' in result
            assert 'memory_info' in result
            

class TestRegressionTests:
    """Regression tests to ensure no functionality was broken."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        
    def test_system_info_method(self):
        """Test that get_system_info method still works."""
        info = self.analyzer.get_system_info()
        
        expected_keys = ['device_info', 'memory_info', 'analyzer_config']
        for key in expected_keys:
            assert key in info
            
    def test_profile_memory_usage_method(self):
        """Test that profile_memory_usage method still works."""
        model = nn.Sequential(nn.Linear(2, 1))
        data = torch.randn(5, 2)
        
        # Should work regardless of profiling enabled/disabled
        result = self.analyzer.profile_memory_usage(model, data)
        assert 'status' in result
        
    def test_initialization_parameters(self):
        """Test that initialization parameters still work."""
        # Test different initialization options
        analyzers = [
            NeurosheafAnalyzer(),
            NeurosheafAnalyzer(device='cpu'),
            NeurosheafAnalyzer(memory_limit_gb=4.0),
            NeurosheafAnalyzer(enable_profiling=False),
            NeurosheafAnalyzer(log_level="DEBUG")
        ]
        
        # All should initialize successfully
        for analyzer in analyzers:
            assert analyzer is not None
            assert hasattr(analyzer, 'analyze')
            

class TestResultFormatConsistency:
    """Test that result formats are consistent and backward compatible."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(nn.Linear(4, 2))
        self.test_data = torch.randn(15, 4)
        
    def test_undirected_result_format(self):
        """Test that undirected analysis result format is consistent."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_sheaf = MagicMock()
            mock_instance.build_from_activations.return_value = mock_sheaf
            
            result = self.analyzer.analyze(self.model, self.test_data)
            
            # Original fields should be present
            original_fields = [
                'analysis_type', 'sheaf', 'preserve_eigenvalues',
                'use_gram_regularization', 'regularization_config',
                'construction_time', 'device_info', 'memory_info', 'performance'
            ]
            
            for field in original_fields:
                assert field in result
                
            # New fields should be present with appropriate values
            assert result['construction_method'] == 'procrustes'
            assert result['gw_config'] is None
            assert result['analysis_type'] == 'undirected'
            
    def test_performance_metrics_format(self):
        """Test that performance metrics format is preserved."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_sheaf = MagicMock()
            mock_instance.build_from_activations.return_value = mock_sheaf
            
            result = self.analyzer.analyze(self.model, self.test_data)
            
            perf = result['performance']
            assert isinstance(perf, dict)
            assert 'construction_time' in perf
            
            # Should be numeric
            assert isinstance(perf['construction_time'], (int, float))
            
    def test_device_memory_info_format(self):
        """Test that device and memory info format is preserved."""
        with patch('neurosheaf.sheaf.assembly.builder.SheafBuilder') as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.build_from_activations.return_value = MagicMock()
            
            result = self.analyzer.analyze(self.model, self.test_data)
            
            # Device info format
            device_info = result['device_info']
            expected_device_fields = ['device', 'platform', 'processor', 'python_version', 'torch_version']
            for field in expected_device_fields:
                assert field in device_info
                
            # Memory info format
            memory_info = result['memory_info']
            expected_memory_fields = ['system_total_gb', 'system_available_gb', 'system_used_gb', 'system_percent']
            for field in expected_memory_fields:
                assert field in memory_info


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
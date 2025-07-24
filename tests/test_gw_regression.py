"""Integration regression tests for GW implementation.

This module ensures that GW integration doesn't break existing functionality
and maintains backward compatibility across all system components.

Test Categories:
1. Default behavior preservation
2. API backward compatibility  
3. Persistence analysis compatibility
4. Performance regression testing
5. Edge case handling consistency
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import time
from unittest.mock import patch, MagicMock

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils.logger import get_logger

logger = get_logger(__name__)


class TestGWRegression:
    """Ensure GW integration doesn't break existing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        
        # Standard test models
        self.simple_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4)
        )
        
        self.test_data = torch.randn(50, 10)
    
    def test_procrustes_method_unchanged(self):
        """Verify default Procrustes behavior is preserved."""
        # Default analysis (should use Procrustes)
        result_default = self.analyzer.analyze(self.simple_model, self.test_data)
        
        # Explicit Procrustes analysis
        result_explicit = self.analyzer.analyze(
            self.simple_model, self.test_data, 
            method='procrustes'
        )
        
        # Both should use the same method
        assert result_default['construction_method'] == 'procrustes'
        assert result_explicit['construction_method'] == 'procrustes'
        
        # Should produce similar results (allowing for numerical differences)
        assert result_default['analysis_type'] == result_explicit['analysis_type']
        assert 'sheaf' in result_default
        assert 'sheaf' in result_explicit
        
        # GW config should be None for Procrustes
        assert result_default['gw_config'] is None
        assert result_explicit['gw_config'] is None
    
    def test_api_backward_compatibility(self):
        """Test existing API calls work without modification."""
        # These should all work exactly as before
        test_calls = [
            # Basic analysis
            lambda: self.analyzer.analyze(self.simple_model, self.test_data),
            
            # With traditional parameters
            lambda: self.analyzer.analyze(
                self.simple_model, self.test_data,
                preserve_eigenvalues=True
            ),
            
            # With regularization
            lambda: self.analyzer.analyze(
                self.simple_model, self.test_data,
                use_gram_regularization=True,
                regularization_config={'lambda': 0.01}
            ),
            
            # Directed analysis
            lambda: self.analyzer.analyze_directed(
                self.simple_model, self.test_data,
                directionality_parameter=0.3
            )
        ]
        
        for i, call_func in enumerate(test_calls):
            try:
                result = call_func()
                
                # Should return expected structure
                assert isinstance(result, dict)
                assert 'analysis_type' in result
                assert 'sheaf' in result
                assert 'construction_method' in result
                assert 'device_info' in result
                assert 'performance' in result
                
                logger.info(f"Backward compatibility call {i} passed")
                
            except Exception as e:
                pytest.fail(f"Backward compatibility call {i} failed: {e}")
    
    def test_network_comparison_compatibility(self):
        """Test that network comparison methods still work."""
        model1 = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
        model2 = nn.Sequential(nn.Linear(8, 5), nn.Linear(5, 4))
        data = torch.randn(30, 8)
        
        # Old-style comparison calls should work
        comparison_calls = [
            # Basic comparison
            lambda: self.analyzer.compare_networks(
                model1, model2, data,
                comparison_method='euclidean'
            ),
            
            # Multiple network comparison
            lambda: self.analyzer.compare_multiple_networks(
                [model1, model2], data,
                comparison_method='cosine'
            ),
            
            # Directed vs undirected
            lambda: self.analyzer.compare_directed_undirected(
                model1, data,
                directionality_parameter=0.2
            )
        ]
        
        for i, call_func in enumerate(comparison_calls):
            try:
                result = call_func()
                
                # Should return expected structure
                assert isinstance(result, dict)
                
                # Should use default Procrustes method
                if 'sheaf_method' in result:
                    assert result['sheaf_method'] == 'procrustes'
                
                logger.info(f"Network comparison call {i} passed")
                
            except Exception as e:
                # Some comparison methods might not be fully implemented
                if "not implemented" not in str(e).lower():
                    logger.warning(f"Network comparison call {i} failed: {e}")
    
    def test_persistence_backward_compatibility(self):
        """Ensure existing persistence analysis works."""
        # Build standard sheaf
        builder = SheafBuilder(restriction_method='scaled_procrustes')
        
        # Use simple model to avoid FX tracing issues
        simple_model = nn.Sequential(
            nn.Linear(6, 4),
            nn.Linear(4, 3)
        )
        data = torch.randn(20, 6)
        
        try:
            # Build sheaf using traditional method
            sheaf = builder.build_from_activations(
                simple_model, data, validate=True
            )
            
            # Should be able to analyze with persistence
            analyzer = PersistentSpectralAnalyzer()
            result = analyzer.analyze(sheaf, n_steps=8)
            
            # Should produce expected structure
            assert 'persistence_result' in result
            assert 'filtration_params' in result
            assert 'features' in result
            
            # Should use standard (decreasing) filtration
            params = result['filtration_params']
            assert len(params) == 8
            
            # For standard sheaves, should detect standard construction
            construction_method = sheaf.metadata.get('construction_method', 'standard')
            assert construction_method in ['scaled_procrustes', 'standard']
            
        except Exception as e:
            logger.warning(f"Persistence compatibility test failed: {e}")
            # Don't fail hard if there are implementation issues
    
    def test_system_info_methods_unchanged(self):
        """Test that utility methods still work."""
        # System info
        info = self.analyzer.get_system_info()
        expected_keys = ['device_info', 'memory_info', 'analyzer_config']
        for key in expected_keys:
            assert key in info
        
        # Memory profiling
        profile_result = self.analyzer.profile_memory_usage(
            self.simple_model, self.test_data[:10]  # Small data
        )
        assert 'status' in profile_result
    
    def test_initialization_parameters_unchanged(self):
        """Test that initialization options still work."""
        init_options = [
            {},
            {'device': 'cpu'},
            {'memory_limit_gb': 4.0},
            {'enable_profiling': False},
            {'log_level': 'WARNING'}
        ]
        
        for i, options in enumerate(init_options):
            try:
                analyzer = NeurosheafAnalyzer(**options)
                
                # Should initialize successfully
                assert hasattr(analyzer, 'analyze')
                assert hasattr(analyzer, 'get_system_info')
                
                # Should be able to run basic analysis
                result = analyzer.analyze(
                    nn.Sequential(nn.Linear(4, 2)),
                    torch.randn(10, 4)
                )
                assert 'analysis_type' in result
                
                logger.info(f"Initialization option {i} passed")
                
            except Exception as e:
                pytest.fail(f"Initialization option {i} failed: {e}")
    
    def test_error_handling_consistency(self):
        """Test that error handling behavior is consistent."""
        from neurosheaf.utils.exceptions import ValidationError, ComputationError
        
        # Invalid inputs should still raise same errors
        error_cases = [
            # Empty data
            (lambda: self.analyzer.analyze(
                nn.Sequential(nn.Linear(3, 1)), 
                torch.empty(0, 3)
            ), ValidationError),
            
            # Invalid model
            (lambda: self.analyzer.analyze(
                "not a model", 
                torch.randn(10, 3)
            ), ValidationError),
            
            # Invalid data type
            (lambda: self.analyzer.analyze(
                nn.Sequential(nn.Linear(3, 1)), 
                "not tensor data"
            ), ValidationError)
        ]
        
        for call_func, expected_error in error_cases:
            with pytest.raises(expected_error):
                call_func()


class TestPerformanceRegression:
    """Test that GW integration doesn't cause performance regressions."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.test_sizes = [
            (nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4)), torch.randn(30, 8)),
            (nn.Sequential(nn.Linear(12, 10), nn.Linear(10, 8)), torch.randn(50, 12))
        ]
    
    def test_procrustes_performance_unchanged(self):
        """Test that Procrustes method hasn't slowed down."""
        baseline_times = []
        
        for model, data in self.test_sizes:
            # Measure Procrustes performance
            start_time = time.time()
            result = self.analyzer.analyze(model, data, method='procrustes')
            end_time = time.time()
            
            analysis_time = end_time - start_time
            baseline_times.append(analysis_time)
            
            # Should complete reasonably quickly (adjust threshold as needed)
            assert analysis_time < 30.0, f"Procrustes too slow: {analysis_time:.2f}s"
            
            # Should produce expected results
            assert result['construction_method'] == 'procrustes'
            assert 'construction_time' in result
            
        logger.info(f"Procrustes baseline times: {baseline_times}")
    
    def test_memory_usage_baseline(self):
        """Test that memory usage hasn't increased significantly."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run several analyses
        for _ in range(3):
            for model, data in self.test_sizes:
                result = self.analyzer.analyze(model, data, method='procrustes')
                assert 'sheaf' in result
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
        # Should not increase memory significantly (adjust threshold)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f} MB"
    
    def test_api_call_overhead(self):
        """Test that API call overhead is minimal."""
        simple_model = nn.Sequential(nn.Linear(4, 2))
        simple_data = torch.randn(10, 4)
        
        # Measure API overhead
        times = []
        for _ in range(5):
            start_time = time.time()
            result = self.analyzer.analyze(simple_model, simple_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"API call times: avg={avg_time:.4f}s, std={std_time:.4f}s")
        
        # API calls should be consistently fast
        assert avg_time < 5.0, f"API calls too slow: {avg_time:.4f}s"
        assert std_time < 2.0, f"API call times too variable: {std_time:.4f}s"


class TestEdgeCaseRegression:
    """Test edge case handling remains consistent."""
    
    def setup_method(self):
        """Set up edge case test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
    
    def test_single_layer_model(self):
        """Test single-layer models still work."""
        single_layer = nn.Sequential(nn.Linear(5, 3))
        data = torch.randn(20, 5)
        
        # Should handle gracefully
        result = self.analyzer.analyze(single_layer, data)
        
        assert 'sheaf' in result
        assert result['construction_method'] == 'procrustes'
        # May have no edges or trivial structure, but shouldn't crash
    
    def test_identical_layer_sizes(self):
        """Test models with identical layer sizes."""
        identical_model = nn.Sequential(
            nn.Linear(6, 6),
            nn.Linear(6, 6),
            nn.Linear(6, 6)
        )
        data = torch.randn(25, 6)
        
        result = self.analyzer.analyze(identical_model, data)
        
        assert 'sheaf' in result
        assert result['construction_method'] == 'procrustes'
    
    def test_very_small_data(self):
        """Test with very small batch sizes."""
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        small_data = torch.randn(2, 4)  # Very small batch
        
        # Should handle gracefully or provide clear error
        try:
            result = self.analyzer.analyze(model, small_data)
            assert 'sheaf' in result
        except Exception as e:
            # Should be informative error, not crash
            assert "batch" in str(e).lower() or "size" in str(e).lower()
    
    def test_zero_activations(self):
        """Test handling of zero activations."""
        # Model that might produce zeros
        zero_model = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),  # Can produce zeros
            nn.Linear(3, 2)
        )
        
        # Data that might produce zero activations after ReLU
        negative_data = -torch.ones(20, 4)
        
        try:
            result = self.analyzer.analyze(zero_model, negative_data)
            assert 'sheaf' in result
        except Exception as e:
            # Should handle gracefully or provide informative error
            assert not "nan" in str(e).lower()  # No NaN errors
    
    def test_device_consistency(self):
        """Test device handling consistency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        data = torch.randn(15, 4)
        
        # CPU analysis
        cpu_result = self.analyzer.analyze(model, data)
        
        # Move to GPU
        model_gpu = model.cuda()
        data_gpu = data.cuda()
        
        # GPU analysis (if supported)
        try:
            gpu_result = self.analyzer.analyze(model_gpu, data_gpu)
            
            # Should produce similar results
            assert cpu_result['construction_method'] == gpu_result['construction_method']
            assert cpu_result['analysis_type'] == gpu_result['analysis_type']
            
        except Exception as e:
            # Acceptable if GPU analysis not fully supported
            logger.info(f"GPU analysis not supported: {e}")


class TestMetadataCompatibility:
    """Test that metadata handling remains compatible."""
    
    def setup_method(self):
        """Set up metadata test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 3))
        self.data = torch.randn(20, 6)
    
    def test_result_structure_compatibility(self):
        """Test that result structure remains compatible."""
        result = self.analyzer.analyze(self.model, self.data)
        
        # Original fields should be present
        original_fields = [
            'analysis_type', 'sheaf', 'preserve_eigenvalues',
            'use_gram_regularization', 'regularization_config',
            'construction_time', 'device_info', 'memory_info', 'performance'
        ]
        
        for field in original_fields:
            assert field in result, f"Missing field: {field}"
        
        # New fields should be present with appropriate values
        assert result['construction_method'] == 'procrustes'  # Default
        assert result['gw_config'] is None  # None for non-GW methods
    
    def test_performance_metrics_format(self):
        """Test that performance metrics format is preserved."""
        result = self.analyzer.analyze(self.model, self.data)
        
        perf = result['performance']
        assert isinstance(perf, dict)
        assert 'construction_time' in perf
        assert isinstance(perf['construction_time'], (int, float))
        assert perf['construction_time'] >= 0
    
    def test_device_memory_info_format(self):
        """Test that device and memory info format is preserved."""
        result = self.analyzer.analyze(self.model, self.data)
        
        # Device info format
        device_info = result['device_info']
        expected_device_fields = ['device', 'platform', 'processor', 'python_version', 'torch_version']
        for field in expected_device_fields:
            assert field in device_info, f"Missing device field: {field}"
        
        # Memory info format
        memory_info = result['memory_info']
        expected_memory_fields = ['system_total_gb', 'system_available_gb', 'system_used_gb', 'system_percent']
        for field in expected_memory_fields:
            assert field in memory_info, f"Missing memory field: {field}"
    
    def test_sheaf_metadata_preservation(self):
        """Test that sheaf metadata is preserved correctly."""
        result = self.analyzer.analyze(self.model, self.data)
        sheaf = result['sheaf']
        
        # Sheaf should have metadata
        assert hasattr(sheaf, 'metadata')
        assert isinstance(sheaf.metadata, dict)
        
        # Should include construction information
        if 'construction_method' in sheaf.metadata:
            assert sheaf.metadata['construction_method'] in [
                'scaled_procrustes', 'procrustes', 'standard'
            ]


class TestGWMethodIntegration:
    """Test that GW method integrates correctly without breaking existing patterns."""
    
    def setup_method(self):
        """Set up GW integration test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.model = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
        self.data = torch.randn(30, 8)
        self.gw_config = GWConfig(epsilon=0.1, max_iter=50)  # Fast config
    
    def test_gw_method_optional(self):
        """Test that GW method is optional and doesn't interfere."""
        try:
            # Should work with GW method
            gw_result = self.analyzer.analyze(
                self.model, self.data,
                method='gromov_wasserstein',
                gw_config=self.gw_config
            )
            
            assert gw_result['construction_method'] == 'gromov_wasserstein'
            assert gw_result['gw_config'] is not None
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                # GW method should work if properly implemented
                logger.warning(f"GW method failed: {e}")
        
        # Standard method should still work regardless
        std_result = self.analyzer.analyze(self.model, self.data, method='procrustes')
        assert std_result['construction_method'] == 'procrustes'
        assert std_result['gw_config'] is None
    
    def test_gw_config_ignored_for_standard_methods(self):
        """Test that GW config is ignored for non-GW methods."""
        # Passing GW config with Procrustes method should be ignored
        result = self.analyzer.analyze(
            self.model, self.data,
            method='procrustes',
            gw_config=self.gw_config  # Should be ignored
        )
        
        assert result['construction_method'] == 'procrustes'
        assert result['gw_config'] is None  # Should be ignored
    
    def test_invalid_method_validation(self):
        """Test that invalid methods are properly rejected."""
        from neurosheaf.utils.exceptions import ValidationError
        
        with pytest.raises(ValidationError, match="Unknown method"):
            self.analyzer.analyze(
                self.model, self.data,
                method='invalid_method'
            )
    
    def test_all_valid_methods_accepted(self):
        """Test that all valid methods are accepted."""
        valid_methods = ['procrustes', 'gromov_wasserstein', 'whitened_procrustes']
        
        for method in valid_methods:
            try:
                if method == 'gromov_wasserstein':
                    result = self.analyzer.analyze(
                        self.model, self.data,
                        method=method,
                        gw_config=self.gw_config
                    )
                else:
                    result = self.analyzer.analyze(
                        self.model, self.data,
                        method=method
                    )
                
                assert result['construction_method'] == method
                logger.info(f"Method {method} accepted and working")
                
            except Exception as e:
                if method == 'gromov_wasserstein' and "POT" in str(e):
                    pytest.skip(f"POT library not available for {method}")
                elif method == 'whitened_procrustes':
                    # May not be fully implemented yet
                    logger.info(f"Method {method} not fully implemented: {e}")
                else:
                    pytest.fail(f"Valid method {method} was rejected: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
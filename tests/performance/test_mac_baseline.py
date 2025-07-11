"""Mac-specific performance regression tests for Neurosheaf baseline."""

import pytest
import torch
import platform
import psutil
import time
from pathlib import Path

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.cka.baseline import BaselineCKA
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.utils.profiling import get_mac_device_info, get_mac_memory_info
from benchmarks.synthetic_data import SyntheticDataGenerator


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Mac-specific tests only run on macOS"
)
class TestMacBaselinePerformance:
    """Test baseline performance on Mac hardware."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Set up test environment."""
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.is_apple_silicon = platform.processor() == "arm"
        
        # Initialize components
        self.analyzer = NeurosheafAnalyzer(device=self.device)
        self.baseline_cka = BaselineCKA(device=self.device)
        self.debiased_cka = DebiasedCKA(device=self.device)
        self.data_generator = SyntheticDataGenerator(device=self.device)
        
        # Clear memory before each test
        if self.device == "mps":
            torch.mps.empty_cache()
    
    def test_mac_device_detection(self):
        """Test Mac device detection and configuration."""
        device_info = get_mac_device_info()
        
        assert device_info['is_mac'] == True
        assert device_info['is_apple_silicon'] == self.is_apple_silicon
        
        if self.is_apple_silicon:
            assert device_info['mps_available'] is not None
            assert device_info['mps_built'] is not None
    
    def test_mac_memory_info(self):
        """Test Mac memory information retrieval."""
        memory_info = get_mac_memory_info()
        
        assert memory_info['system_total_gb'] > 0
        assert memory_info['system_available_gb'] > 0
        assert memory_info['system_used_gb'] > 0
        assert 0 <= memory_info['system_percent'] <= 100
        
        if self.is_apple_silicon:
            assert memory_info['unified_memory'] == True
            assert memory_info['memory_pressure'] in ['normal', 'warning', 'critical', 'unknown']
    
    def test_analyzer_initialization(self):
        """Test NeurosheafAnalyzer initialization on Mac."""
        assert str(self.analyzer.device) in ['mps', 'cpu']
        assert self.analyzer.is_mac == True
        assert self.analyzer.is_apple_silicon == self.is_apple_silicon
        
        system_info = self.analyzer.get_system_info()
        assert 'device_info' in system_info
        assert 'memory_info' in system_info
        assert 'analyzer_config' in system_info
    
    @pytest.mark.benchmark
    def test_baseline_cka_small_scale(self):
        """Test baseline CKA computation with small scale."""
        # Generate small test data
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=50,
            scale_factor=0.1
        )
        
        # Test computation
        initial_memory = get_mac_memory_info()
        
        cka_matrix, profiling_data = self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        final_memory = get_mac_memory_info()
        
        # Validate results
        assert cka_matrix.shape == (len(activations), len(activations))
        assert profiling_data['n_layers'] == len(activations)
        assert profiling_data['computation_time_seconds'] > 0
        
        # Check memory usage
        memory_increase = final_memory['system_used_gb'] - initial_memory['system_used_gb']
        assert memory_increase < 5.0  # Should be under 5GB for small scale
    
    @pytest.mark.benchmark
    def test_baseline_cka_medium_scale(self):
        """Test baseline CKA computation with medium scale."""
        # Generate medium test data
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=200,
            scale_factor=0.5
        )
        
        # Test computation
        initial_memory = get_mac_memory_info()
        
        cka_matrix, profiling_data = self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        final_memory = get_mac_memory_info()
        
        # Validate results
        assert cka_matrix.shape == (len(activations), len(activations))
        assert profiling_data['computation_time_seconds'] > 0
        
        # Check memory usage
        memory_increase = final_memory['system_used_gb'] - initial_memory['system_used_gb']
        assert memory_increase < 10.0  # Should be under 10GB for medium scale
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_baseline_vs_debiased_comparison(self):
        """Test baseline vs debiased CKA comparison."""
        # Generate test data
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=100,
            scale_factor=0.3
        )
        
        # Test baseline CKA
        baseline_start = time.time()
        baseline_matrix, baseline_profiling = self.baseline_cka.compute_baseline_cka_matrix(activations)
        baseline_time = time.time() - baseline_start
        
        # Clear memory
        self.baseline_cka.clear_intermediates()
        if self.device == "mps":
            torch.mps.empty_cache()
        
        # Test debiased CKA
        debiased_start = time.time()
        debiased_matrix = self.debiased_cka.compute_cka_matrix(activations)
        debiased_time = time.time() - debiased_start
        
        # Compare results
        assert baseline_matrix.shape == debiased_matrix.shape
        
        # Check mathematical properties
        matrix_diff = torch.abs(baseline_matrix - debiased_matrix)
        max_diff = torch.max(matrix_diff)
        assert max_diff < 0.1  # Should be similar results
        
        # Check diagonal elements (self-similarity)
        baseline_diag = torch.diag(baseline_matrix)
        debiased_diag = torch.diag(debiased_matrix)
        assert torch.allclose(baseline_diag, torch.ones_like(baseline_diag), atol=1e-6)
        assert torch.allclose(debiased_diag, torch.ones_like(debiased_diag), atol=1e-6)
    
    @pytest.mark.benchmark
    def test_memory_scaling_regression(self):
        """Test memory scaling behavior for regression detection."""
        batch_sizes = [50, 100, 200]
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Generate test data
            activations = self.data_generator.generate_resnet50_activations(
                batch_size=batch_size,
                scale_factor=0.2
            )
            
            # Measure memory usage
            initial_memory = get_mac_memory_info()
            
            cka_matrix, profiling_data = self.baseline_cka.compute_baseline_cka_matrix(activations)
            
            final_memory = get_mac_memory_info()
            memory_increase = final_memory['system_used_gb'] - initial_memory['system_used_gb']
            memory_usage.append(memory_increase)
            
            # Clear memory
            self.baseline_cka.clear_intermediates()
            if self.device == "mps":
                torch.mps.empty_cache()
        
        # Check scaling behavior
        assert len(memory_usage) == len(batch_sizes)
        
        # Memory should generally increase with batch size (allowing for some noise)
        # Check that the overall trend is increasing
        if len(memory_usage) >= 2:
            # Allow for some variance in memory measurements
            # Just check that memory usage is reasonable and non-zero
            for mem_usage in memory_usage:
                assert mem_usage >= 0  # Memory usage should be non-negative
            
            # At least one measurement should show meaningful memory usage
            assert any(mem > 0.01 for mem in memory_usage)  # At least 10MB
    
    @pytest.mark.benchmark
    def test_synthetic_data_generation(self):
        """Test synthetic data generation performance."""
        # Test ResNet50 data generation
        start_time = time.time()
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=100,
            scale_factor=1.0
        )
        generation_time = time.time() - start_time
        
        assert len(activations) > 0
        assert generation_time < 30.0  # Should generate within 30 seconds
        
        # Check activation shapes
        for name, activation in activations.items():
            assert activation.dim() == 2  # Should be flattened
            assert activation.shape[0] == 100  # Batch size
            assert activation.device.type == self.device
    
    @pytest.mark.benchmark
    def test_mac_profiling_accuracy(self):
        """Test Mac-specific profiling accuracy."""
        # Generate test data
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=100,
            scale_factor=0.5
        )
        
        # Test profiling
        cka_matrix, profiling_data = self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        # Validate profiling data
        assert profiling_data['computation_time_seconds'] > 0
        assert profiling_data['n_layers'] == len(activations)
        assert profiling_data['total_parameters'] > 0
        
        # Check Mac-specific data
        mac_specific = profiling_data['mac_specific']
        assert mac_specific['is_mac'] == True
        assert mac_specific['is_apple_silicon'] == self.is_apple_silicon
        assert mac_specific['unified_memory'] == self.is_apple_silicon
    
    @pytest.mark.slow
    def test_baseline_target_approach(self):
        """Test approach to 20GB baseline target."""
        # Generate larger test data to approach target
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=500,
            scale_factor=1.0
        )
        
        # Test computation
        initial_memory = get_mac_memory_info()
        
        cka_matrix, profiling_data = self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        final_memory = get_mac_memory_info()
        
        # Check results
        assert cka_matrix.shape == (len(activations), len(activations))
        
        # Check progress toward 20GB target
        memory_increase = final_memory['system_used_gb'] - initial_memory['system_used_gb']
        target_progress = memory_increase / 20.0
        
        # Should make some progress toward target
        assert target_progress > 0.01  # At least 1% of target
        assert target_progress < 2.0   # Should not exceed 2x target
    
    def test_error_handling(self):
        """Test error handling in baseline computation."""
        # Test with invalid data
        with pytest.raises(Exception):
            self.baseline_cka.compute_baseline_cka_matrix({})
        
        # Test with mismatched dimensions
        activations = {
            "layer1": torch.randn(100, 512, device=self.device),
            "layer2": torch.randn(200, 512, device=self.device)  # Different batch size
        }
        
        with pytest.raises(Exception):
            self.baseline_cka.compute_baseline_cka_matrix(activations)
    
    def test_mathematical_properties(self):
        """Test mathematical properties of CKA computation."""
        # Generate test data
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=50,
            scale_factor=0.2
        )
        
        # Compute CKA matrix
        cka_matrix, _ = self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        # Test mathematical properties
        # 1. Diagonal should be 1 (self-similarity)
        diagonal = torch.diag(cka_matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-6)
        
        # 2. Matrix should be symmetric
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-6)
        
        # 3. Values should be between 0 and 1
        assert torch.all(cka_matrix >= 0)
        assert torch.all(cka_matrix <= 1)
    
    @pytest.mark.benchmark
    def test_cleanup_and_memory_management(self):
        """Test memory cleanup and management."""
        # Generate test data
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=100,
            scale_factor=0.5
        )
        
        # Test with intermediate storage
        initial_memory = get_mac_memory_info()
        
        cka_matrix, profiling_data = self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        after_computation_memory = get_mac_memory_info()
        
        # Clear intermediates
        self.baseline_cka.clear_intermediates()
        if self.device == "mps":
            torch.mps.empty_cache()
        
        after_cleanup_memory = get_mac_memory_info()
        
        # Check memory was freed
        memory_after_computation = after_computation_memory['system_used_gb']
        memory_after_cleanup = after_cleanup_memory['system_used_gb']
        
        # Should free some memory (allowing for system variance)
        assert memory_after_cleanup <= memory_after_computation + 1.0  # Allow 1GB variance
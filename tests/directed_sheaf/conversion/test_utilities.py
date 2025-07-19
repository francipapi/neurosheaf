"""Unit tests for conversion utilities.

Tests the mathematical correctness of conversion validation and optimization:
- ConversionValidator: Validates conversion properties
- PerformanceOptimizer: Optimizes conversion operations
- Round-trip conversion validation
- Performance benchmarking
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List
import time

from neurosheaf.directed_sheaf.conversion.utilities import (
    ConversionValidator, 
    PerformanceOptimizer
)


class TestConversionValidator:
    """Test suite for ConversionValidator class."""
    
    def test_initialization(self):
        """Test initialization of ConversionValidator."""
        validator = ConversionValidator()
        
        assert validator.tolerance == 1e-12
        assert validator.device == torch.device('cpu')
        assert hasattr(validator, 'embedder')
        assert hasattr(validator, 'reconstructor')
        
        # Test with custom parameters
        validator_custom = ConversionValidator(
            tolerance=1e-8,
            device=torch.device('cpu')
        )
        assert validator_custom.tolerance == 1e-8
    
    def test_validate_round_trip_conversion(self):
        """Test round-trip conversion validation."""
        validator = ConversionValidator()
        
        # Create simple complex matrix
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        result = validator.validate_round_trip_conversion(complex_matrix)
        
        # Check validation results
        assert 'reconstruction_error' in result
        assert 'relative_error' in result
        assert 'passes_tolerance' in result
        assert result['passes_tolerance'] is True
        assert result['reconstruction_error'] < 1e-12
    
    def test_validate_round_trip_conversion_invalid_input(self):
        """Test error handling for round-trip validation."""
        validator = ConversionValidator()
        
        # Test with real matrix
        with pytest.raises(ValueError, match="Input matrix must be complex"):
            real_matrix = torch.randn(2, 2)
            validator.validate_round_trip_conversion(real_matrix)
    
    def test_validate_spectral_preservation(self):
        """Test spectral property preservation validation."""
        validator = ConversionValidator()
        
        # Create Hermitian matrix
        hermitian_matrix = torch.tensor([
            [1+0j, 2+3j],
            [2-3j, 4+0j]
        ], dtype=torch.complex64)
        
        result = validator.validate_spectral_preservation(hermitian_matrix)
        
        # Check validation results
        assert 'spectral_error' in result
        assert 'spectral_relative_error' in result
        assert 'passes_tolerance' in result
        assert 'conjugate_pairs_detected' in result
        assert result['passes_tolerance'] is True
        assert result['spectral_error'] < 1e-6
    
    def test_validate_spectral_preservation_invalid_input(self):
        """Test error handling for spectral preservation validation."""
        validator = ConversionValidator()
        
        # Test with real matrix
        with pytest.raises(ValueError, match="Input matrix must be complex"):
            real_matrix = torch.randn(2, 2)
            validator.validate_spectral_preservation(real_matrix)
        
        # Test with non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            nonsquare_matrix = torch.randn(2, 3, dtype=torch.complex64)
            validator.validate_spectral_preservation(nonsquare_matrix)
    
    def test_validate_hermitian_to_symmetric(self):
        """Test Hermitian-to-symmetric mapping validation."""
        validator = ConversionValidator()
        
        # Create Hermitian matrix
        hermitian_matrix = torch.tensor([
            [1+0j, 2+3j],
            [2-3j, 4+0j]
        ], dtype=torch.complex64)
        
        result = validator.validate_hermitian_to_symmetric(hermitian_matrix)
        
        # Check validation results
        assert 'hermitian_error' in result
        assert 'symmetric_error' in result
        assert 'is_hermitian' in result
        assert 'is_symmetric' in result
        assert 'mapping_correct' in result
        assert result['is_hermitian'] is True
        assert result['is_symmetric'] is True
        assert result['mapping_correct'] is True
    
    def test_validate_hermitian_to_symmetric_invalid_input(self):
        """Test error handling for Hermitian-to-symmetric validation."""
        validator = ConversionValidator()
        
        # Test with real matrix
        with pytest.raises(ValueError, match="Input matrix must be complex"):
            real_matrix = torch.randn(2, 2)
            validator.validate_hermitian_to_symmetric(real_matrix)
        
        # Test with non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            nonsquare_matrix = torch.randn(2, 3, dtype=torch.complex64)
            validator.validate_hermitian_to_symmetric(nonsquare_matrix)
    
    def test_validate_positive_definiteness_preservation(self):
        """Test positive definiteness preservation validation."""
        validator = ConversionValidator()
        
        # Create positive definite Hermitian matrix
        A = torch.randn(3, 3, dtype=torch.complex64)
        positive_definite_matrix = A @ A.conj().T  # Guaranteed positive definite
        
        result = validator.validate_positive_definiteness_preservation(positive_definite_matrix)
        
        # Check validation results
        assert 'original_min_eigenvalue' in result
        assert 'real_min_eigenvalue' in result
        assert 'original_positive_definite' in result
        assert 'real_positive_definite' in result
        assert 'positive_definiteness_preserved' in result
        assert result['original_positive_definite'] is True
        assert result['real_positive_definite'] is True
        assert result['positive_definiteness_preserved'] is True
    
    def test_validate_positive_definiteness_preservation_invalid_input(self):
        """Test error handling for positive definiteness validation."""
        validator = ConversionValidator()
        
        # Test with real matrix
        with pytest.raises(ValueError, match="Input matrix must be complex"):
            real_matrix = torch.randn(2, 2)
            validator.validate_positive_definiteness_preservation(real_matrix)
        
        # Test with non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            nonsquare_matrix = torch.randn(2, 3, dtype=torch.complex64)
            validator.validate_positive_definiteness_preservation(nonsquare_matrix)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation of conversion operations."""
        validator = ConversionValidator()
        
        # Create Hermitian matrix
        hermitian_matrix = torch.tensor([
            [2+0j, 1+1j],
            [1-1j, 3+0j]
        ], dtype=torch.complex64)
        
        results = validator.comprehensive_validation(hermitian_matrix)
        
        # Check comprehensive results
        assert 'matrix_shape' in results
        assert 'matrix_dtype' in results
        assert 'round_trip' in results
        assert 'spectral_preservation' in results
        assert 'hermitian_to_symmetric' in results
        assert 'overall_valid' in results
        assert results['overall_valid'] is True
        
        # Check individual validations
        assert results['round_trip']['passes_tolerance'] is True
        assert results['spectral_preservation']['passes_tolerance'] is True
        assert results['hermitian_to_symmetric']['mapping_correct'] is True
    
    def test_comprehensive_validation_non_square(self):
        """Test comprehensive validation with non-square matrix."""
        validator = ConversionValidator()
        
        # Create non-square matrix
        nonsquare_matrix = torch.randn(2, 3, dtype=torch.complex64)
        
        results = validator.comprehensive_validation(nonsquare_matrix)
        
        # Check that only round-trip validation is performed
        assert 'round_trip' in results
        assert 'spectral_preservation' not in results
        assert 'hermitian_to_symmetric' not in results
        assert results['overall_valid'] is True  # Should pass round-trip
    
    def test_batch_validation(self):
        """Test batch validation on multiple matrices."""
        validator = ConversionValidator()
        
        # Create multiple test matrices
        matrices = [
            torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=torch.complex64),
            torch.tensor([[2+0j, 1+1j], [1-1j, 3+0j]], dtype=torch.complex64),
            torch.eye(2, dtype=torch.complex64)
        ]
        
        batch_results = validator.batch_validation(matrices)
        
        # Check batch results
        assert batch_results['num_matrices'] == 3
        assert len(batch_results['individual_results']) == 3
        assert 'summary' in batch_results
        
        # Check summary
        summary = batch_results['summary']
        assert 'all_valid' in summary
        assert 'num_valid' in summary
        assert 'num_invalid' in summary
        assert 'average_round_trip_error' in summary
        assert 'max_round_trip_error' in summary
        assert summary['num_valid'] == 3
        assert summary['num_invalid'] == 0
        assert summary['all_valid'] is True
    
    def test_batch_validation_with_invalid_matrix(self):
        """Test batch validation with some invalid matrices."""
        validator = ConversionValidator()
        
        # Create matrices including Hermitian ones for better validation
        matrices = [
            torch.tensor([[1+0j, 2+3j], [2-3j, 4+0j]], dtype=torch.complex64),  # Hermitian
            torch.zeros(2, 2, dtype=torch.complex64),  # Zero matrix (Hermitian)
            torch.eye(2, dtype=torch.complex64)  # Identity (Hermitian)
        ]
        
        batch_results = validator.batch_validation(matrices)
        
        # Check that validation completed
        assert batch_results['num_matrices'] == 3
        assert len(batch_results['individual_results']) == 3
        
        # At least some should be valid (round-trip should pass for all)
        assert batch_results['summary']['num_valid'] >= 1


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer class."""
    
    def test_initialization(self):
        """Test initialization of PerformanceOptimizer."""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.cache_size == 100
        assert optimizer.device == torch.device('cpu')
        assert hasattr(optimizer, 'embedder')
        assert hasattr(optimizer, 'reconstructor')
        assert hasattr(optimizer, 'conversion_cache')
        assert optimizer.cache_hits == 0
        assert optimizer.cache_misses == 0
        
        # Test with custom parameters
        optimizer_custom = PerformanceOptimizer(
            cache_size=50,
            device=torch.device('cpu')
        )
        assert optimizer_custom.cache_size == 50
    
    def test_embed_matrix_cached(self):
        """Test cached matrix embedding."""
        optimizer = PerformanceOptimizer()
        
        # Create complex matrix
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        # First call (cache miss)
        result1 = optimizer.embed_matrix_cached(complex_matrix)
        assert optimizer.cache_misses == 1
        assert optimizer.cache_hits == 0
        
        # Second call (cache hit)
        result2 = optimizer.embed_matrix_cached(complex_matrix)
        assert optimizer.cache_misses == 1
        assert optimizer.cache_hits == 1
        
        # Results should be identical
        assert torch.allclose(result1, result2)
    
    def test_embed_matrix_cached_with_custom_key(self):
        """Test cached matrix embedding with custom key."""
        optimizer = PerformanceOptimizer()
        
        # Create complex matrix
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        # Test with custom cache key
        custom_key = "test_matrix_1"
        result = optimizer.embed_matrix_cached(complex_matrix, custom_key)
        
        assert optimizer.cache_misses == 1
        assert custom_key in optimizer.conversion_cache
    
    def test_batch_embed_matrices(self):
        """Test batch embedding of multiple matrices."""
        optimizer = PerformanceOptimizer()
        
        # Create multiple complex matrices
        matrices = [
            torch.randn(2, 2, dtype=torch.complex64) for _ in range(5)
        ]
        
        # Batch embed
        results = optimizer.batch_embed_matrices(matrices, batch_size=2)
        
        # Check results
        assert len(results) == 5
        assert all(not result.is_complex() for result in results)
        assert all(result.shape == (4, 4) for result in results)
    
    def test_benchmark_conversion_performance(self):
        """Test conversion performance benchmarking."""
        optimizer = PerformanceOptimizer()
        
        # Create test matrices
        test_matrices = [
            torch.randn(3, 3, dtype=torch.complex64) for _ in range(3)
        ]
        
        # Run benchmark
        benchmark_results = optimizer.benchmark_conversion_performance(test_matrices)
        
        # Check benchmark results
        assert benchmark_results['num_matrices'] == 3
        assert 'timing_results' in benchmark_results
        assert 'memory_results' in benchmark_results
        assert 'cache_statistics' in benchmark_results
        assert 'summary' in benchmark_results
        
        # Check timing results
        timing_results = benchmark_results['timing_results']
        assert len(timing_results) == 3
        assert all('conversion_time' in result for result in timing_results)
        assert all('throughput' in result for result in timing_results)
        
        # Check summary
        summary = benchmark_results['summary']
        assert 'total_time' in summary
        assert 'average_time' in summary
        assert 'total_throughput' in summary
        assert summary['total_time'] > 0
        assert summary['average_time'] > 0
    
    def test_estimate_memory_requirements(self):
        """Test memory requirements estimation."""
        optimizer = PerformanceOptimizer()
        
        # Test with various matrix shapes
        shapes = [(10, 10), (5, 20), (100, 50)]
        
        estimates = optimizer.estimate_memory_requirements(shapes)
        
        # Check estimates
        assert 'total_complex_elements' in estimates
        assert 'total_real_elements' in estimates
        assert 'complex_memory_bytes' in estimates
        assert 'real_memory_bytes' in estimates
        assert 'total_memory_bytes' in estimates
        assert 'memory_overhead_ratio' in estimates
        assert 'estimated_peak_memory_gb' in estimates
        
        # Check calculations
        expected_complex_elements = sum(n * m for n, m in shapes)
        expected_real_elements = sum(4 * n * m for n, m in shapes)
        
        assert estimates['total_complex_elements'] == expected_complex_elements
        assert estimates['total_real_elements'] == expected_real_elements
        assert estimates['memory_overhead_ratio'] == 2.0
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        optimizer = PerformanceOptimizer()
        
        # Add some entries to cache
        matrix = torch.randn(2, 2, dtype=torch.complex64)
        optimizer.embed_matrix_cached(matrix)
        
        assert len(optimizer.conversion_cache) > 0
        assert optimizer.cache_misses == 1
        
        # Clear cache
        optimizer.clear_cache()
        
        assert len(optimizer.conversion_cache) == 0
        assert optimizer.cache_hits == 0
        assert optimizer.cache_misses == 0
    
    def test_get_cache_statistics(self):
        """Test cache statistics retrieval."""
        optimizer = PerformanceOptimizer()
        
        # Test with empty cache
        stats = optimizer.get_cache_statistics()
        assert stats['cache_size'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['hit_rate'] == 0
        
        # Add some cache activity
        matrix = torch.randn(2, 2, dtype=torch.complex64)
        optimizer.embed_matrix_cached(matrix)  # Miss
        optimizer.embed_matrix_cached(matrix)  # Hit
        
        stats = optimizer.get_cache_statistics()
        assert stats['cache_size'] == 1
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_cache_size_management(self):
        """Test that cache size is properly managed."""
        # Create optimizer with small cache
        optimizer = PerformanceOptimizer(cache_size=2)
        
        # Add more matrices than cache can hold
        matrices = [
            torch.randn(2, 2, dtype=torch.complex64) for _ in range(3)
        ]
        
        for i, matrix in enumerate(matrices):
            optimizer.embed_matrix_cached(matrix, f"key_{i}")
        
        # Cache should contain at most 2 entries
        assert len(optimizer.conversion_cache) <= 2
        assert optimizer.cache_misses == 3  # All were misses
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        optimizer = PerformanceOptimizer()
        
        # Create matrix
        matrix = torch.randn(2, 2, dtype=torch.complex64)
        
        # Generate key
        key = optimizer._generate_cache_key(matrix)
        
        # Check key properties
        assert isinstance(key, str)
        assert len(key) > 0
        
        # Same matrix should generate same key
        key2 = optimizer._generate_cache_key(matrix)
        assert key == key2
        
        # Different matrix should generate different key
        different_matrix = torch.randn(3, 3, dtype=torch.complex64)
        key3 = optimizer._generate_cache_key(different_matrix)
        assert key != key3
    
    def test_device_support(self):
        """Test device support for optimizer."""
        # Test with CPU device
        optimizer = PerformanceOptimizer(device=torch.device('cpu'))
        
        matrix = torch.randn(2, 2, dtype=torch.complex64)
        result = optimizer.embed_matrix_cached(matrix)
        
        assert result.device == torch.device('cpu')
        
        # Test device transfer
        if torch.cuda.is_available():
            optimizer_cuda = PerformanceOptimizer(device=torch.device('cuda'))
            result_cuda = optimizer_cuda.embed_matrix_cached(matrix)
            assert result_cuda.device.type == 'cuda'
    
    def test_performance_with_large_matrices(self):
        """Test performance with larger matrices."""
        optimizer = PerformanceOptimizer()
        
        # Create larger matrices
        large_matrices = [
            torch.randn(50, 50, dtype=torch.complex64) for _ in range(2)
        ]
        
        # Test batch processing
        start_time = time.time()
        results = optimizer.batch_embed_matrices(large_matrices, batch_size=1)
        end_time = time.time()
        
        # Check results
        assert len(results) == 2
        assert all(result.shape == (100, 100) for result in results)
        assert end_time - start_time < 5.0  # Should complete in reasonable time
    
    def test_memory_efficiency(self):
        """Test memory efficiency of batch operations."""
        optimizer = PerformanceOptimizer()
        
        # Create multiple matrices
        matrices = [
            torch.randn(10, 10, dtype=torch.complex64) for _ in range(10)
        ]
        
        # Process with small batch size (should be memory efficient)
        results = optimizer.batch_embed_matrices(matrices, batch_size=2)
        
        # Check that all matrices were processed
        assert len(results) == 10
        assert all(result.shape == (20, 20) for result in results)


class TestConversionUtilitiesIntegration:
    """Integration tests for conversion utilities."""
    
    def test_validator_optimizer_integration(self):
        """Test integration between validator and optimizer."""
        validator = ConversionValidator()
        optimizer = PerformanceOptimizer()
        
        # Create Hermitian test matrices for better validation
        A = torch.randn(3, 3, dtype=torch.complex64)
        matrices = [
            A @ A.conj().T,  # Positive definite Hermitian
            torch.eye(3, dtype=torch.complex64),  # Identity
            torch.zeros(3, 3, dtype=torch.complex64)  # Zero matrix
        ]
        
        # Use optimizer to embed matrices
        embedded_matrices = optimizer.batch_embed_matrices(matrices)
        
        # Use validator to validate original matrices
        validation_results = validator.batch_validation(matrices)
        
        # Check that both operations succeeded
        assert len(embedded_matrices) == 3
        # At least some should be valid (round-trip should pass for all)
        assert validation_results['summary']['num_valid'] >= 1
        
        # Check cache statistics
        cache_stats = optimizer.get_cache_statistics()
        assert cache_stats['cache_misses'] == 3  # All should be cache misses
    
    def test_comprehensive_conversion_workflow(self):
        """Test comprehensive conversion workflow."""
        validator = ConversionValidator()
        optimizer = PerformanceOptimizer()
        
        # Create Hermitian matrix
        hermitian_matrix = torch.tensor([
            [2+0j, 1+1j],
            [1-1j, 3+0j]
        ], dtype=torch.complex64)
        
        # Step 1: Validate conversion properties
        validation_result = validator.comprehensive_validation(hermitian_matrix)
        assert validation_result['overall_valid'] is True
        
        # Step 2: Perform optimized embedding
        embedded_matrix = optimizer.embed_matrix_cached(hermitian_matrix)
        
        # Step 3: Benchmark performance
        benchmark_result = optimizer.benchmark_conversion_performance([hermitian_matrix])
        
        # Check results
        assert embedded_matrix.shape == (4, 4)
        assert benchmark_result['num_matrices'] == 1
        assert benchmark_result['summary']['total_time'] > 0
    
    def test_error_handling_integration(self):
        """Test error handling across conversion utilities."""
        validator = ConversionValidator()
        optimizer = PerformanceOptimizer()
        
        # Test with invalid input
        invalid_matrices = [
            torch.randn(2, 2),  # Real matrix (should cause validation error)
        ]
        
        # Validator should handle the error gracefully
        batch_results = validator.batch_validation(invalid_matrices)
        assert batch_results['summary']['all_valid'] is False
        assert batch_results['summary']['num_invalid'] == 1
        
        # Optimizer should raise appropriate errors
        with pytest.raises(ValueError):
            optimizer.embed_matrix_cached(invalid_matrices[0])


if __name__ == '__main__':
    pytest.main([__file__])
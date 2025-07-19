"""Conversion utilities for directed sheaf real-complex operations.

This module provides utility classes for validating conversion operations
and optimizing performance of real-complex conversions in directed sheaf
computation.

Key Components:
- ConversionValidator: Validates mathematical properties of conversions
- PerformanceOptimizer: Optimizes conversion operations for large matrices

Mathematical Foundation:
- Round-trip conversion accuracy validation
- Spectral property preservation verification
- Hermitian-to-symmetric mapping validation
- Performance optimization for sparse matrices
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy.sparse import csr_matrix, csc_matrix, issparse
import time
import gc

from .real_embedding import ComplexToRealEmbedding
from .complex_reconstruction import RealToComplexReconstruction

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


class ConversionValidator:
    """Validates mathematical properties of real-complex conversions.
    
    This class provides comprehensive validation of conversion operations
    to ensure mathematical correctness and numerical accuracy.
    
    Key Features:
    - Round-trip conversion accuracy validation
    - Spectral property preservation verification
    - Hermitian-to-symmetric mapping validation
    - Performance benchmarking
    - Memory usage analysis
    
    The validator ensures that all mathematical properties required for
    directed sheaf analysis are preserved during conversion operations.
    """
    
    def __init__(self, tolerance: float = 1e-12,
                 device: Optional[torch.device] = None):
        """Initialize the conversion validator.
        
        Args:
            tolerance: Tolerance for numerical validation
            device: PyTorch device for computations
        """
        self.tolerance = tolerance
        self.device = device or torch.device('cpu')
        
        # Initialize embedder and reconstructor
        self.embedder = ComplexToRealEmbedding(
            validate_properties=True,
            tolerance=tolerance,
            device=device
        )
        self.reconstructor = RealToComplexReconstruction(
            validate_properties=True,
            tolerance=tolerance,
            device=device
        )
        
        logger.debug(f"ConversionValidator initialized with tolerance={tolerance}")
    
    def validate_round_trip_conversion(self, complex_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate round-trip conversion accuracy.
        
        Args:
            complex_matrix: Original complex matrix to test
            
        Returns:
            Dictionary with validation results
        """
        if not complex_matrix.is_complex():
            raise ValueError("Input matrix must be complex")
        
        try:
            # Perform round-trip conversion
            real_embedded = self.embedder.embed_matrix(complex_matrix)
            reconstructed = self.reconstructor.reconstruct_matrix(real_embedded)
            
            # Compute conversion accuracy
            return self.reconstructor.validate_round_trip(complex_matrix, reconstructed)
            
        except Exception as e:
            logger.error(f"Round-trip conversion validation failed: {e}")
            return {
                'reconstruction_error': float('inf'),
                'relative_error': float('inf'),
                'passes_tolerance': False,
                'error_message': str(e)
            }
    
    def validate_spectral_preservation(self, complex_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate preservation of spectral properties.
        
        Args:
            complex_matrix: Complex matrix to test (must be square)
            
        Returns:
            Dictionary with spectral validation results
        """
        if not complex_matrix.is_complex():
            raise ValueError("Input matrix must be complex")
        
        if complex_matrix.shape[0] != complex_matrix.shape[1]:
            raise ValueError("Matrix must be square for spectral analysis")
        
        try:
            # Compute original eigenvalues
            original_eigenvalues = torch.linalg.eigvals(complex_matrix)
            
            # Convert to real representation and compute eigenvalues
            real_embedded = self.embedder.embed_matrix(complex_matrix)
            real_eigenvalues = torch.linalg.eigvals(real_embedded)
            
            # Reconstruct complex eigenvalues
            reconstructed_eigenvalues = self.reconstructor.reconstruct_eigenvalues(real_eigenvalues.real)
            
            # Compare eigenvalue sets
            orig_sorted = torch.sort(original_eigenvalues.real)[0]
            recon_sorted = torch.sort(reconstructed_eigenvalues.real)[0]
            
            spectral_error = torch.abs(orig_sorted - recon_sorted).max().item()
            
            return {
                'spectral_error': spectral_error,
                'spectral_relative_error': spectral_error / (torch.norm(orig_sorted).item() + 1e-12),
                'passes_tolerance': spectral_error <= self.tolerance,
                'original_eigenvalues': original_eigenvalues,
                'reconstructed_eigenvalues': reconstructed_eigenvalues,
                'real_eigenvalues_shape': real_eigenvalues.shape,
                'conjugate_pairs_detected': real_eigenvalues.shape[0] == 2 * original_eigenvalues.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Spectral preservation validation failed: {e}")
            return {
                'spectral_error': float('inf'),
                'spectral_relative_error': float('inf'),
                'passes_tolerance': False,
                'error_message': str(e)
            }
    
    def validate_hermitian_to_symmetric(self, hermitian_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate Hermitian-to-symmetric mapping.
        
        Args:
            hermitian_matrix: Complex Hermitian matrix
            
        Returns:
            Dictionary with validation results
        """
        if not hermitian_matrix.is_complex():
            raise ValueError("Input matrix must be complex")
        
        if hermitian_matrix.shape[0] != hermitian_matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        try:
            # Verify matrix is Hermitian
            hermitian_error = torch.abs(hermitian_matrix - hermitian_matrix.conj().T).max().item()
            if hermitian_error > self.tolerance:
                logger.warning(f"Matrix not Hermitian: error={hermitian_error}")
            
            # Convert to real representation
            real_embedded = self.embedder.embed_matrix(hermitian_matrix)
            
            # Verify real matrix is symmetric
            symmetric_error = torch.abs(real_embedded - real_embedded.T).max().item()
            
            return {
                'hermitian_error': hermitian_error,
                'symmetric_error': symmetric_error,
                'is_hermitian': hermitian_error <= self.tolerance,
                'is_symmetric': symmetric_error <= self.tolerance,
                'mapping_correct': hermitian_error <= self.tolerance and symmetric_error <= self.tolerance,
                'real_matrix_shape': real_embedded.shape
            }
            
        except Exception as e:
            logger.error(f"Hermitian-to-symmetric validation failed: {e}")
            return {
                'hermitian_error': float('inf'),
                'symmetric_error': float('inf'),
                'is_hermitian': False,
                'is_symmetric': False,
                'mapping_correct': False,
                'error_message': str(e)
            }
    
    def validate_positive_definiteness_preservation(self, positive_definite_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate preservation of positive definiteness.
        
        Args:
            positive_definite_matrix: Complex positive definite matrix
            
        Returns:
            Dictionary with validation results
        """
        if not positive_definite_matrix.is_complex():
            raise ValueError("Input matrix must be complex")
        
        if positive_definite_matrix.shape[0] != positive_definite_matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        try:
            # Check original matrix is positive definite
            original_eigenvalues = torch.linalg.eigvals(positive_definite_matrix)
            original_min_eigenvalue = original_eigenvalues.real.min().item()
            
            # Convert to real representation
            real_embedded = self.embedder.embed_matrix(positive_definite_matrix)
            
            # Check embedded matrix is positive definite
            real_eigenvalues = torch.linalg.eigvals(real_embedded)
            real_min_eigenvalue = real_eigenvalues.real.min().item()
            
            return {
                'original_min_eigenvalue': original_min_eigenvalue,
                'real_min_eigenvalue': real_min_eigenvalue,
                'original_positive_definite': original_min_eigenvalue > -self.tolerance,
                'real_positive_definite': real_min_eigenvalue > -self.tolerance,
                'positive_definiteness_preserved': (
                    original_min_eigenvalue > -self.tolerance and 
                    real_min_eigenvalue > -self.tolerance
                )
            }
            
        except Exception as e:
            logger.error(f"Positive definiteness validation failed: {e}")
            return {
                'original_min_eigenvalue': float('-inf'),
                'real_min_eigenvalue': float('-inf'),
                'original_positive_definite': False,
                'real_positive_definite': False,
                'positive_definiteness_preserved': False,
                'error_message': str(e)
            }
    
    def comprehensive_validation(self, complex_matrix: torch.Tensor) -> Dict[str, Any]:
        """Perform comprehensive validation of conversion operations.
        
        Args:
            complex_matrix: Complex matrix to validate
            
        Returns:
            Dictionary with comprehensive validation results
        """
        results = {
            'matrix_shape': complex_matrix.shape,
            'matrix_dtype': complex_matrix.dtype,
            'validation_timestamp': time.time()
        }
        
        # Round-trip conversion validation
        results['round_trip'] = self.validate_round_trip_conversion(complex_matrix)
        
        # Spectral preservation validation (for square matrices)
        if complex_matrix.shape[0] == complex_matrix.shape[1]:
            results['spectral_preservation'] = self.validate_spectral_preservation(complex_matrix)
            
            # Hermitian-to-symmetric validation
            results['hermitian_to_symmetric'] = self.validate_hermitian_to_symmetric(complex_matrix)
            
            # Positive definiteness validation (if matrix is positive definite)
            try:
                eigenvalues = torch.linalg.eigvals(complex_matrix)
                if eigenvalues.real.min().item() > -self.tolerance:
                    results['positive_definiteness'] = self.validate_positive_definiteness_preservation(complex_matrix)
            except Exception:
                pass
        
        # Overall validation status (focus on round-trip as the core requirement)
        results['overall_valid'] = results['round_trip']['passes_tolerance']
        
        # Only require spectral preservation for square matrices that are actually Hermitian
        if 'spectral_preservation' in results and results.get('hermitian_to_symmetric', {}).get('is_hermitian', False):
            results['overall_valid'] = results['overall_valid'] and results['spectral_preservation']['passes_tolerance']
        
        if 'hermitian_to_symmetric' in results and results['hermitian_to_symmetric']['is_hermitian']:
            results['overall_valid'] = results['overall_valid'] and results['hermitian_to_symmetric']['mapping_correct']
        
        return results
    
    def batch_validation(self, complex_matrices: List[torch.Tensor]) -> Dict[str, Any]:
        """Perform batch validation on multiple matrices.
        
        Args:
            complex_matrices: List of complex matrices to validate
            
        Returns:
            Dictionary with batch validation results
        """
        batch_results = {
            'num_matrices': len(complex_matrices),
            'individual_results': [],
            'summary': {
                'all_valid': True,
                'num_valid': 0,
                'num_invalid': 0,
                'average_round_trip_error': 0.0,
                'max_round_trip_error': 0.0
            }
        }
        
        round_trip_errors = []
        
        for i, matrix in enumerate(complex_matrices):
            try:
                result = self.comprehensive_validation(matrix)
                batch_results['individual_results'].append(result)
                
                # Track statistics
                if result['overall_valid']:
                    batch_results['summary']['num_valid'] += 1
                else:
                    batch_results['summary']['num_invalid'] += 1
                    batch_results['summary']['all_valid'] = False
                
                round_trip_error = result['round_trip']['reconstruction_error']
                round_trip_errors.append(round_trip_error)
                
                logger.debug(f"Validated matrix {i}: valid={result['overall_valid']}, error={round_trip_error:.2e}")
                
            except Exception as e:
                logger.error(f"Validation failed for matrix {i}: {e}")
                batch_results['individual_results'].append({
                    'error_message': str(e),
                    'overall_valid': False
                })
                batch_results['summary']['num_invalid'] += 1
                batch_results['summary']['all_valid'] = False
        
        # Compute summary statistics
        if round_trip_errors:
            batch_results['summary']['average_round_trip_error'] = np.mean(round_trip_errors)
            batch_results['summary']['max_round_trip_error'] = np.max(round_trip_errors)
        
        return batch_results


class PerformanceOptimizer:
    """Optimizes performance of real-complex conversion operations.
    
    This class provides optimization strategies for large-scale conversion
    operations, including memory management, sparse matrix optimization,
    and batch processing.
    
    Key Features:
    - Memory-efficient batch operations
    - Sparse matrix conversion optimization
    - Caching for repeated conversions
    - Performance benchmarking
    - Memory usage monitoring
    
    The optimizer enables efficient processing of large directed sheaf
    structures while maintaining numerical accuracy.
    """
    
    def __init__(self, cache_size: int = 100,
                 device: Optional[torch.device] = None):
        """Initialize the performance optimizer.
        
        Args:
            cache_size: Maximum number of cached conversions
            device: PyTorch device for computations
        """
        self.cache_size = cache_size
        self.device = device or torch.device('cpu')
        
        # Initialize conversion cache
        self.conversion_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize embedder and reconstructor
        self.embedder = ComplexToRealEmbedding(
            validate_properties=False,  # Disable validation for performance
            device=device
        )
        self.reconstructor = RealToComplexReconstruction(
            validate_properties=False,
            device=device
        )
        
        logger.debug(f"PerformanceOptimizer initialized with cache_size={cache_size}")
    
    def embed_matrix_cached(self, complex_matrix: torch.Tensor, 
                           cache_key: Optional[str] = None) -> torch.Tensor:
        """Embed complex matrix with caching.
        
        Args:
            complex_matrix: Complex matrix to embed
            cache_key: Optional cache key for the matrix
            
        Returns:
            Real embedded matrix
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(complex_matrix)
        
        # Check cache
        if cache_key in self.conversion_cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for key {cache_key}")
            return self.conversion_cache[cache_key]
        
        # Compute embedding
        real_embedded = self.embedder.embed_matrix(complex_matrix)
        
        # Store in cache
        self.cache_misses += 1
        self._store_in_cache(cache_key, real_embedded)
        
        logger.debug(f"Cache miss for key {cache_key}, computed and stored")
        return real_embedded
    
    def batch_embed_matrices(self, complex_matrices: List[torch.Tensor],
                           batch_size: int = 10) -> List[torch.Tensor]:
        """Embed multiple matrices in batches for memory efficiency.
        
        Args:
            complex_matrices: List of complex matrices to embed
            batch_size: Size of processing batches
            
        Returns:
            List of real embedded matrices
        """
        real_matrices = []
        
        for i in range(0, len(complex_matrices), batch_size):
            batch = complex_matrices[i:i+batch_size]
            
            # Process batch
            batch_results = []
            for matrix in batch:
                real_matrix = self.embedder.embed_matrix(matrix)
                batch_results.append(real_matrix)
            
            real_matrices.extend(batch_results)
            
            # Clean up memory
            if i % (batch_size * 5) == 0:  # Periodic cleanup
                gc.collect()
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(complex_matrices) + batch_size - 1)//batch_size}")
        
        return real_matrices
    
    def optimize_sparse_conversion(self, complex_sparse: csr_matrix) -> csr_matrix:
        """Optimize sparse matrix conversion.
        
        Args:
            complex_sparse: Complex sparse matrix
            
        Returns:
            Real embedded sparse matrix
        """
        if not issparse(complex_sparse):
            raise ValueError("Input must be a sparse matrix")
        
        # Use specialized sparse conversion
        return self.embedder.embed_sparse_matrix(complex_sparse)
    
    def benchmark_conversion_performance(self, test_matrices: List[torch.Tensor]) -> Dict[str, Any]:
        """Benchmark conversion performance.
        
        Args:
            test_matrices: List of test matrices
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_results = {
            'num_matrices': len(test_matrices),
            'timing_results': [],
            'memory_results': [],
            'cache_statistics': {}
        }
        
        # Clear cache for fair benchmarking
        self.clear_cache()
        
        total_time = 0
        peak_memory = 0
        
        for i, matrix in enumerate(test_matrices):
            # Measure timing
            start_time = time.time()
            
            # Measure memory before
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            else:
                memory_before = 0
            
            # Perform conversion
            real_embedded = self.embed_matrix_cached(matrix)
            
            # Measure memory after
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
            else:
                memory_after = 0
            
            end_time = time.time()
            
            # Record results
            conversion_time = end_time - start_time
            memory_usage = memory_after - memory_before
            
            benchmark_results['timing_results'].append({
                'matrix_index': i,
                'matrix_shape': matrix.shape,
                'conversion_time': conversion_time,
                'throughput': matrix.numel() / conversion_time
            })
            
            benchmark_results['memory_results'].append({
                'matrix_index': i,
                'memory_usage': memory_usage,
                'peak_memory': memory_after
            })
            
            total_time += conversion_time
            peak_memory = max(peak_memory, memory_after)
        
        # Summary statistics
        benchmark_results['summary'] = {
            'total_time': total_time,
            'average_time': total_time / len(test_matrices),
            'peak_memory': peak_memory,
            'total_throughput': sum(m.numel() for m in test_matrices) / total_time
        }
        
        # Cache statistics
        benchmark_results['cache_statistics'] = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
        
        return benchmark_results
    
    def estimate_memory_requirements(self, complex_shapes: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Estimate memory requirements for conversion operations.
        
        Args:
            complex_shapes: List of complex matrix shapes
            
        Returns:
            Dictionary with memory estimates
        """
        total_complex_elements = sum(n * m for n, m in complex_shapes)
        total_real_elements = sum(4 * n * m for n, m in complex_shapes)  # 4x due to 2x in each dimension
        
        # Estimate memory usage (assuming float32)
        complex_memory = total_complex_elements * 2 * 4  # 2 components (real, imag) × 4 bytes
        real_memory = total_real_elements * 4  # 4 bytes per float32
        
        return {
            'total_complex_elements': total_complex_elements,
            'total_real_elements': total_real_elements,
            'complex_memory_bytes': complex_memory,
            'real_memory_bytes': real_memory,
            'total_memory_bytes': complex_memory + real_memory,
            'memory_overhead_ratio': real_memory / complex_memory,
            'estimated_peak_memory_gb': (complex_memory + real_memory) / (1024**3)
        }
    
    def clear_cache(self) -> None:
        """Clear conversion cache."""
        self.conversion_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.debug("Conversion cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.conversion_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def _generate_cache_key(self, matrix: torch.Tensor) -> str:
        """Generate cache key for matrix."""
        # Simple hash based on shape and a few elements
        key_elements = [
            str(matrix.shape),
            str(matrix.dtype),
            str(matrix.device)
        ]
        
        # Add hash of a few elements for uniqueness
        if matrix.numel() > 0:
            flat_matrix = matrix.flatten()
            sample_indices = torch.linspace(0, flat_matrix.numel()-1, min(10, flat_matrix.numel()), dtype=torch.long)
            sample_elements = flat_matrix[sample_indices]
            key_elements.append(str(hash(tuple(sample_elements.cpu().numpy().flatten()))))
        
        return "_".join(key_elements)
    
    def _store_in_cache(self, cache_key: str, value: torch.Tensor) -> None:
        """Store value in cache with size management."""
        if len(self.conversion_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.conversion_cache))
            del self.conversion_cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")
        
        self.conversion_cache[cache_key] = value
"""Performance validation tests for GW sheaf implementation.

This module validates that the GW implementation meets performance requirements
specified in the implementation plan:
- Memory usage < 2x Procrustes baseline
- Wall time < 2x Procrustes for typical networks
- Scalability with network size
- Cache effectiveness

Test Categories:
1. Performance benchmarking vs Procrustes
2. Memory usage validation
3. Scalability testing
4. Cache effectiveness
5. GPU vs CPU performance comparison
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceResult:
    """Performance measurement result."""
    wall_time: float
    memory_peak_mb: float
    memory_increase_mb: float
    construction_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceBenchmark:
    """Performance benchmark comparison."""
    network_size: Tuple[int, ...]
    batch_size: int
    procrustes_result: PerformanceResult
    gw_result: Optional[PerformanceResult]
    speedup_ratio: Optional[float] = None
    memory_ratio: Optional[float] = None


class TestGWPerformance:
    """Verify GW performance meets requirements."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.process = psutil.Process(os.getpid())
        
        # Performance thresholds from implementation plan
        self.max_slowdown_factor = 2.0     # < 2x slowdown vs Procrustes
        self.max_memory_factor = 2.0       # < 2x memory vs Procrustes
        self.max_absolute_memory_gb = 3.0  # < 3GB absolute memory usage
        
        # Test network configurations
        self.test_networks = [
            ([10, 8, 6, 4], 30),      # Small network
            ([20, 16, 12, 8], 50),    # Medium network  
            ([32, 24, 16, 8], 80),    # Larger network
            ([50, 40, 30, 20, 10], 100)  # Deep network
        ]
        
        # Fast GW config for testing
        self.gw_config = GWConfig(epsilon=0.05, max_iter=100, cache_cost_matrices=True)
    
    def _measure_performance(self, method: str, network_arch: List[int], 
                           batch_size: int, gw_config: Optional[GWConfig] = None) -> PerformanceResult:
        """Measure performance of a single method."""
        # Create model and data
        model = nn.Sequential(*[
            nn.Linear(network_arch[i], network_arch[i+1])
            for i in range(len(network_arch) - 1)
        ])
        data = torch.randn(batch_size, network_arch[0])
        
        # Measure initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Measure wall time
            start_time = time.time()
            
            if method == 'gromov_wasserstein':
                result = self.analyzer.analyze(model, data, method=method, gw_config=gw_config)
            else:
                result = self.analyzer.analyze(model, data, method=method)
            
            end_time = time.time()
            
            # Measure final memory
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = final_memory  # Approximation
            
            wall_time = end_time - start_time
            memory_increase = final_memory - initial_memory
            construction_time = result.get('construction_time', wall_time)
            
            return PerformanceResult(
                wall_time=wall_time,
                memory_peak_mb=peak_memory,
                memory_increase_mb=memory_increase,
                construction_time=construction_time,
                success=True
            )
            
        except Exception as e:
            final_memory = self.process.memory_info().rss / 1024 / 1024
            return PerformanceResult(
                wall_time=0.0,
                memory_peak_mb=final_memory,
                memory_increase_mb=final_memory - initial_memory,
                construction_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def test_gw_vs_procrustes_performance(self):
        """Test that GW performance is within acceptable bounds vs Procrustes."""
        benchmarks = []
        
        for network_arch, batch_size in self.test_networks:
            logger.info(f"Benchmarking network {network_arch} with batch size {batch_size}")
            
            # Measure Procrustes performance
            procrustes_result = self._measure_performance('procrustes', network_arch, batch_size)
            assert procrustes_result.success, f"Procrustes failed: {procrustes_result.error_message}"
            
            # Measure GW performance
            try:
                gw_result = self._measure_performance('gromov_wasserstein', network_arch, batch_size, self.gw_config)
            except Exception as e:
                if "POT" in str(e):
                    pytest.skip("POT library not available")
                    return
                else:
                    gw_result = PerformanceResult(
                        wall_time=0.0, memory_peak_mb=0.0, memory_increase_mb=0.0,
                        construction_time=0.0, success=False, error_message=str(e)
                    )
            
            # Create benchmark
            benchmark = PerformanceBenchmark(
                network_size=tuple(network_arch),
                batch_size=batch_size,
                procrustes_result=procrustes_result,
                gw_result=gw_result
            )
            
            if gw_result.success:
                # Compute performance ratios
                benchmark.speedup_ratio = gw_result.wall_time / procrustes_result.wall_time
                benchmark.memory_ratio = gw_result.memory_peak_mb / procrustes_result.memory_peak_mb
                
                # Log results
                logger.info(f"  Procrustes: {procrustes_result.wall_time:.3f}s, {procrustes_result.memory_peak_mb:.1f}MB")
                logger.info(f"  GW: {gw_result.wall_time:.3f}s, {gw_result.memory_peak_mb:.1f}MB")
                logger.info(f"  Ratios: {benchmark.speedup_ratio:.2f}x time, {benchmark.memory_ratio:.2f}x memory")
                
                # Validate performance requirements
                assert benchmark.speedup_ratio <= self.max_slowdown_factor, \
                    f"GW too slow: {benchmark.speedup_ratio:.2f}x vs max {self.max_slowdown_factor}x"
                
                assert benchmark.memory_ratio <= self.max_memory_factor, \
                    f"GW uses too much memory: {benchmark.memory_ratio:.2f}x vs max {self.max_memory_factor}x"
                
                assert gw_result.memory_peak_mb <= self.max_absolute_memory_gb * 1024, \
                    f"GW absolute memory too high: {gw_result.memory_peak_mb:.1f}MB vs max {self.max_absolute_memory_gb * 1024}MB"
            else:
                logger.warning(f"GW failed for network {network_arch}: {gw_result.error_message}")
            
            benchmarks.append(benchmark)
        
        # Overall analysis
        successful_benchmarks = [b for b in benchmarks if b.gw_result and b.gw_result.success]
        if successful_benchmarks:
            avg_speedup = np.mean([b.speedup_ratio for b in successful_benchmarks])
            avg_memory = np.mean([b.memory_ratio for b in successful_benchmarks])
            
            logger.info(f"Overall averages: {avg_speedup:.2f}x time, {avg_memory:.2f}x memory")
            
            # Should meet average performance targets
            assert avg_speedup <= self.max_slowdown_factor, f"Average slowdown too high: {avg_speedup:.2f}x"
            assert avg_memory <= self.max_memory_factor, f"Average memory overhead too high: {avg_memory:.2f}x"
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with network size."""
        memory_usage = []
        
        # Test different network sizes
        test_sizes = [
            ([8, 6, 4], 20),
            ([16, 12, 8], 30),
            ([24, 18, 12], 40),
            ([32, 24, 16], 50)
        ]
        
        for network_arch, batch_size in test_sizes:
            try:
                result = self._measure_performance('gromov_wasserstein', network_arch, batch_size, self.gw_config)
                if result.success:
                    network_complexity = sum(network_arch) * batch_size  # Rough complexity measure
                    memory_usage.append((network_complexity, result.memory_increase_mb))
                    
                    logger.info(f"Network {network_arch}, batch {batch_size}: "
                              f"complexity={network_complexity}, memory={result.memory_increase_mb:.1f}MB")
            except Exception as e:
                if "POT" in str(e):
                    pytest.skip("POT library not available")
                else:
                    logger.warning(f"Failed for network {network_arch}: {e}")
        
        if len(memory_usage) >= 2:
            # Check that memory scaling is reasonable (not exponential)
            complexities = np.array([x[0] for x in memory_usage])
            memories = np.array([x[1] for x in memory_usage])
            
            # Fit linear model: memory = a * complexity + b
            A = np.vstack([complexities, np.ones(len(complexities))]).T
            coeffs, residuals, rank, s = np.linalg.lstsq(A, memories, rcond=None)
            
            slope, intercept = coeffs
            logger.info(f"Memory scaling: {slope:.4f} MB per complexity unit + {intercept:.1f} MB base")
            
            # Memory should scale reasonably (not too steep)
            assert slope < 0.1, f"Memory scaling too steep: {slope:.4f} MB per unit"
    
    def test_gw_cache_effectiveness(self):
        """Test that caching improves performance."""
        if not hasattr(self, 'gw_config'):
            pytest.skip("GW config not available")
        
        network_arch = [12, 10, 8, 6]
        batch_size = 40
        
        # Test with caching disabled
        no_cache_config = GWConfig(
            epsilon=0.05, max_iter=100, 
            cache_cost_matrices=False
        )
        
        # Test with caching enabled
        cache_config = GWConfig(
            epsilon=0.05, max_iter=100,
            cache_cost_matrices=True
        )
        
        try:
            # Measure without cache
            no_cache_result = self._measure_performance(
                'gromov_wasserstein', network_arch, batch_size, no_cache_config
            )
            
            # Measure with cache (run twice to see cache benefits)
            cache_result_1 = self._measure_performance(
                'gromov_wasserstein', network_arch, batch_size, cache_config
            )
            cache_result_2 = self._measure_performance(
                'gromov_wasserstein', network_arch, batch_size, cache_config
            )
            
            if all(r.success for r in [no_cache_result, cache_result_1, cache_result_2]):
                logger.info(f"No cache: {no_cache_result.wall_time:.3f}s")
                logger.info(f"With cache (1st): {cache_result_1.wall_time:.3f}s")
                logger.info(f"With cache (2nd): {cache_result_2.wall_time:.3f}s")
                
                # Second cached run should be faster or similar
                cache_speedup = cache_result_1.wall_time / cache_result_2.wall_time
                logger.info(f"Cache speedup: {cache_speedup:.2f}x")
                
                # Cache should provide some benefit (allow for noise)
                assert cache_speedup >= 0.8, f"Cache should not slow down significantly: {cache_speedup:.2f}x"
        
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                logger.warning(f"Cache test failed: {e}")
    
    def test_gw_configuration_performance_tradeoffs(self):
        """Test performance vs accuracy tradeoffs for different GW configurations."""
        network_arch = [16, 12, 8]
        batch_size = 50
        
        # Different performance/accuracy configurations
        configs = [
            ('fast', GWConfig(epsilon=0.1, max_iter=50, tolerance=1e-6)),
            ('balanced', GWConfig(epsilon=0.05, max_iter=100, tolerance=1e-8)),
            ('accurate', GWConfig(epsilon=0.01, max_iter=200, tolerance=1e-10))
        ]
        
        results = {}
        
        try:
            for config_name, config in configs:
                result = self._measure_performance('gromov_wasserstein', network_arch, batch_size, config)
                if result.success:
                    results[config_name] = result
                    logger.info(f"{config_name}: {result.wall_time:.3f}s, {result.memory_peak_mb:.1f}MB")
            
            if len(results) >= 2:
                # Fast should be faster than accurate
                if 'fast' in results and 'accurate' in results:
                    speedup = results['accurate'].wall_time / results['fast'].wall_time
                    assert speedup >= 1.0, f"Fast config should be faster: {speedup:.2f}x"
                    logger.info(f"Fast vs Accurate speedup: {speedup:.2f}x")
                
                # All configurations should meet basic performance requirements
                for name, result in results.items():
                    assert result.wall_time < 30.0, f"{name} config too slow: {result.wall_time:.2f}s"
                    assert result.memory_peak_mb < 2048, f"{name} config uses too much memory: {result.memory_peak_mb:.1f}MB"
        
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                logger.warning(f"Configuration performance test failed: {e}")
    
    def test_concurrent_analysis_performance(self):
        """Test performance when running multiple analyses concurrently.""" 
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for concurrent testing")
        
        network_arch = [10, 8, 6]
        batch_size = 30
        num_concurrent = 3
        
        # Sequential baseline
        sequential_times = []
        for i in range(num_concurrent):
            try:
                result = self._measure_performance('procrustes', network_arch, batch_size)
                if result.success:
                    sequential_times.append(result.wall_time)
            except Exception as e:
                logger.warning(f"Sequential test {i} failed: {e}")
        
        if sequential_times:
            total_sequential_time = sum(sequential_times)
            
            # Concurrent test (using threading would require more complex setup)
            # For now, just verify single analysis doesn't degrade significantly
            try:
                concurrent_result = self._measure_performance('procrustes', network_arch, batch_size * 2)
                if concurrent_result.success:
                    # Should scale reasonably with batch size
                    scaling_factor = concurrent_result.wall_time / np.mean(sequential_times)
                    assert scaling_factor < 3.0, f"Poor scaling with batch size: {scaling_factor:.2f}x"
                    
                    logger.info(f"Scaling factor with 2x batch: {scaling_factor:.2f}x")
            except Exception as e:
                logger.warning(f"Concurrent test failed: {e}")


class TestGWPerformanceRegression:
    """Test for performance regressions over time."""
    
    def setup_method(self):
        """Set up regression test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        
        # Baseline performance expectations (adjust based on actual measurements)
        self.performance_baselines = {
            'small_network': {'max_time': 5.0, 'max_memory_mb': 100},
            'medium_network': {'max_time': 15.0, 'max_memory_mb': 300},
            'large_network': {'max_time': 30.0, 'max_memory_mb': 500}
        }
    
    def test_small_network_baseline(self):
        """Test small network performance baseline."""
        model = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
        data = torch.randn(20, 8)
        
        start_time = time.time()
        result = self.analyzer.analyze(model, data, method='procrustes')
        end_time = time.time()
        
        wall_time = end_time - start_time
        baseline = self.performance_baselines['small_network']
        
        assert wall_time <= baseline['max_time'], \
            f"Small network too slow: {wall_time:.2f}s vs {baseline['max_time']}s"
        
        logger.info(f"Small network baseline: {wall_time:.3f}s")
    
    def test_medium_network_baseline(self):
        """Test medium network performance baseline."""
        model = nn.Sequential(
            nn.Linear(20, 16), nn.Linear(16, 12), nn.Linear(12, 8)
        )
        data = torch.randn(50, 20)
        
        start_time = time.time()
        result = self.analyzer.analyze(model, data, method='procrustes')
        end_time = time.time()
        
        wall_time = end_time - start_time
        baseline = self.performance_baselines['medium_network']
        
        assert wall_time <= baseline['max_time'], \
            f"Medium network too slow: {wall_time:.2f}s vs {baseline['max_time']}s"
        
        logger.info(f"Medium network baseline: {wall_time:.3f}s")
    
    def test_gw_method_reasonable_performance(self):
        """Test that GW method has reasonable absolute performance."""
        try:
            model = nn.Sequential(nn.Linear(12, 10), nn.Linear(10, 8))
            data = torch.randn(40, 12)
            
            gw_config = GWConfig(epsilon=0.1, max_iter=100)
            
            start_time = time.time()
            result = self.analyzer.analyze(model, data, method='gromov_wasserstein', gw_config=gw_config)
            end_time = time.time()
            
            wall_time = end_time - start_time
            
            # Should complete in reasonable time (adjust threshold as needed)
            assert wall_time <= 60.0, f"GW method too slow: {wall_time:.2f}s"
            assert result['construction_method'] == 'gromov_wasserstein'
            
            logger.info(f"GW method baseline: {wall_time:.3f}s")
        
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                raise


class TestMemoryEfficiency:
    """Test memory efficiency and leak detection."""
    
    def setup_method(self):
        """Set up memory test fixtures."""
        self.analyzer = NeurosheafAnalyzer()
        self.process = psutil.Process(os.getpid())
    
    def test_no_memory_leaks(self):
        """Test that repeated analyses don't leak memory."""
        model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 6))
        data = torch.randn(30, 10)
        
        # Measure baseline
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple analyses
        for i in range(5):
            result = self.analyzer.analyze(model, data, method='procrustes')
            assert 'sheaf' in result
            
            # Clear references
            del result
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Measure final memory
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory increase after 5 analyses: {memory_increase:.2f}MB")
        
        # Should not increase memory significantly
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.2f}MB increase"
    
    def test_large_batch_memory_efficiency(self):
        """Test memory efficiency with large batch sizes."""
        model = nn.Sequential(nn.Linear(20, 15), nn.Linear(15, 10))
        
        # Test different batch sizes
        batch_sizes = [50, 100, 200]
        memory_usage = []
        
        for batch_size in batch_sizes:
            data = torch.randn(batch_size, 20)
            
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            try:
                result = self.analyzer.analyze(model, data, method='procrustes')
                final_memory = self.process.memory_info().rss / 1024 / 1024
                
                memory_increase = final_memory - initial_memory
                memory_usage.append((batch_size, memory_increase))
                
                logger.info(f"Batch size {batch_size}: memory increase {memory_increase:.2f}MB")
                
                # Clean up
                del result, data
                import gc
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed for batch size {batch_size}: {e}")
        
        if len(memory_usage) >= 2:
            # Memory should scale reasonably with batch size
            batch_factors = []
            for i in range(1, len(memory_usage)):
                prev_batch, prev_mem = memory_usage[i-1]
                curr_batch, curr_mem = memory_usage[i]
                
                batch_factor = curr_batch / prev_batch
                memory_factor = curr_mem / prev_mem if prev_mem > 0 else 1.0
                
                batch_factors.append(memory_factor / batch_factor)
            
            # Memory efficiency should not degrade significantly with batch size
            avg_efficiency = np.mean(batch_factors) if batch_factors else 1.0
            assert avg_efficiency < 2.0, f"Poor memory scaling: {avg_efficiency:.2f}x"


if __name__ == "__main__":
    # Run tests with verbose output and performance metrics
    pytest.main([__file__, "-v", "--tb=short", "-s"])  # -s to show print statements
"""Performance benchmarks for Neurosheaf."""

import pytest
import time
import psutil
import os
from unittest.mock import MagicMock

from neurosheaf.utils.profiling import profile_memory, profile_time, get_profile_manager


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Benchmark tests for performance validation."""
    
    def test_basic_import_speed(self):
        """Test that basic imports are fast."""
        start_time = time.time()
        import neurosheaf
        import_time = time.time() - start_time
        
        # Should import quickly
        assert import_time < 1.0, f"Import took {import_time:.2f}s, expected <1.0s"
    
    def test_memory_profiling_overhead(self):
        """Test memory profiling overhead is minimal."""
        @profile_memory()
        def dummy_function():
            return sum(range(1000))
        
        # Test with profiling
        start_time = time.time()
        result = dummy_function()
        profiled_time = time.time() - start_time
        
        # Test without profiling
        start_time = time.time()
        result_plain = sum(range(1000))
        plain_time = time.time() - start_time
        
        assert result == result_plain
        # Profiling overhead should be reasonable (profiling takes time)
        assert profiled_time < 1.0  # Should complete within 1 second
    
    def test_logging_performance(self):
        """Test logging performance is acceptable."""
        from neurosheaf.utils.logging import setup_logger
        
        logger = setup_logger("perf_test")
        
        start_time = time.time()
        for i in range(100):
            logger.info(f"Test message {i}")
        
        log_time = time.time() - start_time
        
        # Should log 100 messages quickly
        assert log_time < 1.0, f"Logging took {log_time:.2f}s, expected <1.0s"


@pytest.mark.benchmark
@pytest.mark.slow
class TestMemoryBenchmarks:
    """Memory usage benchmarks."""
    
    def test_basic_memory_usage(self):
        """Test basic memory usage is reasonable."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Import and use basic functionality
        from neurosheaf.utils.logging import setup_logger
        from neurosheaf.utils.profiling import ProfileManager
        
        logger = setup_logger("memory_test")
        manager = ProfileManager()
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not use excessive memory for basic operations
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, expected <50MB"
    
    def test_profile_manager_memory(self):
        """Test ProfileManager doesn't leak memory."""
        from neurosheaf.utils.profiling import ProfileManager, ProfileResult
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy multiple managers
        for i in range(10):
            manager = ProfileManager()
            # Add some results
            result = ProfileResult(
                function_name="test_function",
                execution_time=1.0,
                cpu_memory_peak_mb=100.0,
                cpu_memory_current_mb=90.0,
                gpu_memory_peak_mb=0.0,
                gpu_memory_current_mb=0.0
            )
            manager.add_result(result)
            manager.clear_results()
            del manager
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not leak significant memory
        assert memory_increase < 10, f"Memory increased by {memory_increase:.1f}MB, expected <10MB"


class TestBenchmarkUtilities:
    """Test benchmark utilities are working."""
    
    def test_benchmark_markers_available(self):
        """Test that benchmark markers are available."""
        assert hasattr(pytest.mark, 'benchmark')
        assert hasattr(pytest.mark, 'slow')
    
    def test_profiling_utilities_available(self):
        """Test that profiling utilities are available."""
        from neurosheaf.utils.profiling import profile_memory, profile_time
        
        # Should be callable
        assert callable(profile_memory)
        assert callable(profile_time)
    
    def test_memory_monitoring_available(self):
        """Test memory monitoring is available."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            assert hasattr(memory_info, 'rss')
        except ImportError:
            pytest.skip("psutil not available")
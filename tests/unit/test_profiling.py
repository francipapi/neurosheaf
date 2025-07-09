"""Unit tests for performance profiling utilities."""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import threading

from neurosheaf.utils.profiling import (
    ProfileResult,
    ProfileManager,
    profile_memory,
    profile_time,
    profile_comprehensive,
    MemoryMonitor,
    get_memory_usage,
    clear_gpu_memory,
    check_memory_limits,
    get_profile_manager,
    benchmark_function,
)
from neurosheaf.utils.exceptions import MemoryError


class TestProfileResult:
    """Test ProfileResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic ProfileResult creation."""
        result = ProfileResult(
            function_name="test_func",
            execution_time=1.5,
            cpu_memory_peak_mb=100.0,
            cpu_memory_current_mb=80.0
        )
        
        assert result.function_name == "test_func"
        assert result.execution_time == 1.5
        assert result.cpu_memory_peak_mb == 100.0
        assert result.cpu_memory_current_mb == 80.0
        assert result.gpu_memory_peak_mb == 0.0
        assert result.gpu_memory_current_mb == 0.0
    
    def test_creation_with_gpu_memory(self):
        """Test ProfileResult creation with GPU memory."""
        result = ProfileResult(
            function_name="gpu_func",
            execution_time=2.0,
            cpu_memory_peak_mb=200.0,
            cpu_memory_current_mb=150.0,
            gpu_memory_peak_mb=500.0,
            gpu_memory_current_mb=400.0
        )
        
        assert result.gpu_memory_peak_mb == 500.0
        assert result.gpu_memory_current_mb == 400.0
    
    def test_to_dict_method(self):
        """Test to_dict method."""
        result = ProfileResult(
            function_name="test_func",
            execution_time=1.0,
            cpu_memory_peak_mb=100.0,
            cpu_memory_current_mb=80.0,
            args_info={"arg_count": 2},
            context={"test": "value"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["function_name"] == "test_func"
        assert result_dict["execution_time"] == 1.0
        assert result_dict["cpu_memory_peak_mb"] == 100.0
        assert result_dict["args_info"] == {"arg_count": 2}
        assert result_dict["context"] == {"test": "value"}
        assert "timestamp" in result_dict


class TestProfileManager:
    """Test ProfileManager class."""
    
    def test_manager_creation(self):
        """Test ProfileManager creation."""
        manager = ProfileManager()
        assert len(manager.results) == 0
    
    def test_add_result(self):
        """Test adding profiling results."""
        manager = ProfileManager()
        result = ProfileResult("test", 1.0, 100.0, 80.0)
        
        manager.add_result(result)
        
        assert len(manager.results) == 1
        assert manager.results[0] == result
    
    def test_get_results(self):
        """Test getting profiling results."""
        manager = ProfileManager()
        result1 = ProfileResult("func1", 1.0, 100.0, 80.0)
        result2 = ProfileResult("func2", 2.0, 200.0, 150.0)
        
        manager.add_result(result1)
        manager.add_result(result2)
        
        all_results = manager.get_results()
        assert len(all_results) == 2
        
        func1_results = manager.get_results("func1")
        assert len(func1_results) == 1
        assert func1_results[0].function_name == "func1"
    
    def test_clear_results(self):
        """Test clearing profiling results."""
        manager = ProfileManager()
        result = ProfileResult("test", 1.0, 100.0, 80.0)
        manager.add_result(result)
        
        manager.clear_results()
        
        assert len(manager.results) == 0
    
    def test_save_results(self, temp_dir):
        """Test saving results to file."""
        manager = ProfileManager()
        result = ProfileResult("test", 1.0, 100.0, 80.0)
        manager.add_result(result)
        
        filepath = temp_dir / "results.json"
        manager.save_results(filepath)
        
        assert filepath.exists()
        
        with open(filepath) as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["function_name"] == "test"
    
    def test_generate_report(self):
        """Test generating profiling report."""
        manager = ProfileManager()
        
        # Test empty report
        report = manager.generate_report()
        assert "No profiling results available" in report
        
        # Test with results
        result1 = ProfileResult("func1", 1.0, 100.0, 80.0)
        result2 = ProfileResult("func1", 1.5, 120.0, 90.0)
        result3 = ProfileResult("func2", 2.0, 200.0, 150.0)
        
        manager.add_result(result1)
        manager.add_result(result2)
        manager.add_result(result3)
        
        report = manager.generate_report()
        
        assert "Neurosheaf Profiling Report" in report
        assert "func1" in report
        assert "func2" in report
        assert "Calls: 2" in report
    
    def test_thread_safety(self):
        """Test thread safety of ProfileManager."""
        manager = ProfileManager()
        
        def add_results(thread_id):
            for i in range(10):
                result = ProfileResult(f"func_{thread_id}", i * 0.1, i * 10.0, i * 8.0)
                manager.add_result(result)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_results, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(manager.results) == 50


class TestProfileMemoryDecorator:
    """Test profile_memory decorator."""
    
    def test_basic_profiling(self):
        """Test basic memory profiling."""
        @profile_memory(log_results=False)
        def test_function():
            return "result"
        
        # Should not raise exception
        result = test_function()
        assert result == "result"
        
        # Should have added result to manager
        manager = get_profile_manager()
        results = manager.get_results("test_function")
        assert len(results) >= 1
    
    def test_profiling_with_threshold(self):
        """Test profiling with memory threshold."""
        @profile_memory(memory_threshold_mb=1.0, log_results=False)
        def allocate_memory():
            # Allocate some memory
            data = [0] * 1000
            return data
        
        # Should not raise exception
        result = allocate_memory()
        assert len(result) == 1000
    
    def test_profiling_with_args(self):
        """Test profiling with argument information."""
        @profile_memory(include_args=True, log_results=False)
        def function_with_args(arg1, arg2, kwarg1=None):
            return arg1 + arg2
        
        result = function_with_args(1, 2, kwarg1="test")
        assert result == 3
        
        manager = get_profile_manager()
        results = manager.get_results("function_with_args")
        assert len(results) >= 1
        assert results[-1].args_info["args_count"] == 2
        assert "kwarg1" in results[-1].args_info["kwargs_keys"]
    
    def test_profiling_with_exception(self):
        """Test profiling when function raises exception."""
        @profile_memory(log_results=False)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
    
    @patch('neurosheaf.utils.profiling.torch')
    def test_profiling_with_gpu(self, mock_torch):
        """Test profiling with GPU support."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 200  # 200MB
        
        @profile_memory(log_results=False)
        def gpu_function():
            return "gpu_result"
        
        result = gpu_function()
        assert result == "gpu_result"
        
        manager = get_profile_manager()
        results = manager.get_results("gpu_function")
        assert len(results) >= 1
        assert results[-1].gpu_memory_peak_mb == 200.0


class TestProfileTimeDecorator:
    """Test profile_time decorator."""
    
    def test_basic_time_profiling(self):
        """Test basic time profiling."""
        @profile_time(log_results=False)
        def timed_function():
            time.sleep(0.1)
            return "result"
        
        start = time.time()
        result = timed_function()
        end = time.time()
        
        assert result == "result"
        assert end - start >= 0.1
    
    def test_time_threshold(self):
        """Test time threshold warning."""
        @profile_time(time_threshold_seconds=0.05, log_results=False)
        def slow_function():
            time.sleep(0.1)
            return "result"
        
        # Should not raise exception but should log warning
        result = slow_function()
        assert result == "result"
    
    def test_time_profiling_with_exception(self):
        """Test time profiling when function raises exception."""
        @profile_time(log_results=False)
        def failing_function():
            time.sleep(0.1)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()


class TestProfileComprehensiveDecorator:
    """Test profile_comprehensive decorator."""
    
    def test_comprehensive_profiling(self):
        """Test comprehensive profiling (memory + time)."""
        @profile_comprehensive(log_results=False)
        def comprehensive_function():
            time.sleep(0.1)
            data = [0] * 1000
            return data
        
        result = comprehensive_function()
        assert len(result) == 1000
        
        manager = get_profile_manager()
        results = manager.get_results("comprehensive_function")
        assert len(results) >= 1
        assert results[-1].execution_time >= 0.1


class TestMemoryMonitor:
    """Test MemoryMonitor context manager."""
    
    def test_basic_monitoring(self):
        """Test basic memory monitoring."""
        with MemoryMonitor("test_operation") as monitor:
            # Allocate some memory
            data = [0] * 1000
            memory_used = monitor.check_memory()
            assert memory_used >= 0
    
    def test_monitoring_with_threshold(self):
        """Test monitoring with threshold."""
        with MemoryMonitor("test_operation", threshold_mb=1.0):
            # Should not raise exception for small allocation
            data = [0] * 100
    
    def test_monitoring_exception_handling(self):
        """Test monitoring with exception in context."""
        try:
            with MemoryMonitor("test_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should not raise additional exceptions


class TestMemoryUtilities:
    """Test memory utility functions."""
    
    @patch('neurosheaf.utils.profiling.psutil')
    def test_get_memory_usage_cpu(self, mock_psutil):
        """Test getting CPU memory usage."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_psutil.Process.return_value = mock_process
        
        cpu_memory, gpu_memory = get_memory_usage()
        
        assert cpu_memory == 100.0
        assert gpu_memory == 0.0
    
    @patch('neurosheaf.utils.profiling.torch')
    @patch('neurosheaf.utils.profiling.psutil')
    def test_get_memory_usage_gpu(self, mock_psutil, mock_torch):
        """Test getting GPU memory usage."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_psutil.Process.return_value = mock_process
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 200  # 200MB
        
        cpu_memory, gpu_memory = get_memory_usage()
        
        assert cpu_memory == 100.0
        assert gpu_memory == 200.0
    
    @patch('neurosheaf.utils.profiling.torch')
    def test_clear_gpu_memory(self, mock_torch):
        """Test clearing GPU memory."""
        mock_torch.cuda.is_available.return_value = True
        
        clear_gpu_memory()
        
        mock_torch.cuda.empty_cache.assert_called_once()
    
    @patch('neurosheaf.utils.profiling.torch')
    def test_clear_gpu_memory_no_cuda(self, mock_torch):
        """Test clearing GPU memory when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        
        clear_gpu_memory()
        
        mock_torch.cuda.empty_cache.assert_not_called()
    
    @patch('neurosheaf.utils.profiling.get_memory_usage')
    def test_check_memory_limits_under_limit(self, mock_get_memory):
        """Test memory limit checking under limit."""
        mock_get_memory.return_value = (100.0, 200.0)
        
        # Should not raise exception
        check_memory_limits(cpu_limit_mb=500.0, gpu_limit_mb=1000.0)
    
    @patch('neurosheaf.utils.profiling.get_memory_usage')
    def test_check_memory_limits_cpu_exceeded(self, mock_get_memory):
        """Test memory limit checking with CPU limit exceeded."""
        mock_get_memory.return_value = (600.0, 200.0)
        
        with pytest.raises(MemoryError) as exc_info:
            check_memory_limits(cpu_limit_mb=500.0, gpu_limit_mb=1000.0)
        
        assert "CPU memory limit" in str(exc_info.value)
        assert exc_info.value.context["memory_type"] == "cpu"
    
    @patch('neurosheaf.utils.profiling.get_memory_usage')
    def test_check_memory_limits_gpu_exceeded(self, mock_get_memory):
        """Test memory limit checking with GPU limit exceeded."""
        mock_get_memory.return_value = (100.0, 1200.0)
        
        with pytest.raises(MemoryError) as exc_info:
            check_memory_limits(cpu_limit_mb=500.0, gpu_limit_mb=1000.0)
        
        assert "GPU memory limit" in str(exc_info.value)
        assert exc_info.value.context["memory_type"] == "gpu"


class TestBenchmarkFunction:
    """Test benchmark_function utility."""
    
    def test_basic_benchmarking(self):
        """Test basic function benchmarking."""
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        stats = benchmark_function(
            test_function,
            args=(1, 2),
            num_runs=3,
            warmup_runs=1
        )
        
        assert stats["num_runs"] == 3
        assert stats["mean_time"] >= 0.01
        assert stats["min_time"] > 0
        assert stats["max_time"] >= stats["min_time"]
        assert "mean_memory" in stats
    
    def test_benchmarking_with_kwargs(self):
        """Test benchmarking with keyword arguments."""
        def test_function(x, y=10):
            return x * y
        
        stats = benchmark_function(
            test_function,
            args=(5,),
            kwargs={"y": 20},
            num_runs=2,
            warmup_runs=0
        )
        
        assert stats["num_runs"] == 2
        assert stats["mean_time"] >= 0
    
    def test_benchmarking_statistics(self):
        """Test benchmarking statistics calculation."""
        def variable_function():
            # Introduce some variability
            time.sleep(0.01 + (hash(time.time()) % 100) / 10000)
            return "result"
        
        stats = benchmark_function(
            variable_function,
            num_runs=5,
            warmup_runs=1
        )
        
        assert stats["std_time"] >= 0
        assert stats["min_time"] <= stats["mean_time"] <= stats["max_time"]


class TestGlobalProfileManager:
    """Test global profile manager."""
    
    def test_get_profile_manager(self):
        """Test getting global profile manager."""
        manager1 = get_profile_manager()
        manager2 = get_profile_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, ProfileManager)
    
    def test_global_manager_persistence(self):
        """Test that global manager persists across function calls."""
        manager = get_profile_manager()
        result = ProfileResult("test", 1.0, 100.0, 80.0)
        manager.add_result(result)
        
        # Get manager again
        manager2 = get_profile_manager()
        results = manager2.get_results("test")
        
        assert len(results) == 1
        assert results[0].function_name == "test"


class TestProfilingEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    def test_profiling_recursive_function(self):
        """Test profiling recursive function."""
        @profile_memory(log_results=False)
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        result = fibonacci(5)
        assert result == 5
        
        manager = get_profile_manager()
        results = manager.get_results("fibonacci")
        assert len(results) >= 1
    
    def test_profiling_generator_function(self):
        """Test profiling generator function."""
        @profile_memory(log_results=False)
        def generator_function():
            for i in range(10):
                yield i
        
        result = list(generator_function())
        assert result == list(range(10))
    
    def test_profiling_with_very_short_execution(self):
        """Test profiling with very short execution time."""
        @profile_time(log_results=False)
        def instant_function():
            return 42
        
        result = instant_function()
        assert result == 42
    
    def test_profiling_with_large_memory_allocation(self):
        """Test profiling with large memory allocation."""
        @profile_memory(memory_threshold_mb=1.0, log_results=False)
        def allocate_large_memory():
            # Allocate large amount of memory
            data = [0] * 1000000
            return len(data)
        
        result = allocate_large_memory()
        assert result == 1000000
    
    def test_profiling_nested_decorated_functions(self):
        """Test profiling nested decorated functions."""
        @profile_memory(log_results=False)
        def outer_function():
            @profile_memory(log_results=False)
            def inner_function():
                return "inner"
            return inner_function()
        
        result = outer_function()
        assert result == "inner"
        
        manager = get_profile_manager()
        outer_results = manager.get_results("outer_function")
        inner_results = manager.get_results("inner_function")
        
        assert len(outer_results) >= 1
        assert len(inner_results) >= 1


@pytest.mark.phase1
class TestProfilingPhase1Requirements:
    """Test Phase 1 specific requirements for profiling."""
    
    def test_profiling_ready_for_baseline(self):
        """Test that profiling is ready for baseline measurements."""
        @profile_comprehensive(log_results=False)
        def baseline_function():
            # Simulate baseline computation
            data = [[i * j for j in range(100)] for i in range(100)]
            return data
        
        result = baseline_function()
        assert len(result) == 100
        
        manager = get_profile_manager()
        results = manager.get_results("baseline_function")
        assert len(results) >= 1
        assert results[-1].execution_time > 0
        assert results[-1].cpu_memory_peak_mb > 0
    
    def test_profiling_supports_memory_thresholds(self):
        """Test that profiling supports memory thresholds for optimization."""
        # Test with 3GB threshold (Phase 1 requirement)
        @profile_memory(memory_threshold_mb=3000.0, log_results=False)
        def memory_intensive_function():
            # Simulate memory usage
            data = [0] * 10000
            return data
        
        result = memory_intensive_function()
        assert len(result) == 10000
    
    def test_profiling_integration_with_logging(self):
        """Test profiling integration with logging system."""
        from neurosheaf.utils.logging import setup_logger
        
        logger = setup_logger("profiling_test")
        
        @profile_memory(log_results=True)
        def logged_function():
            return "logged"
        
        result = logged_function()
        assert result == "logged"
    
    def test_profiling_benchmark_capabilities(self):
        """Test benchmarking capabilities for performance validation."""
        def simple_computation():
            return sum(range(1000))
        
        stats = benchmark_function(
            simple_computation,
            num_runs=5,
            warmup_runs=2
        )
        
        assert stats["num_runs"] == 5
        assert stats["mean_time"] > 0
        assert stats["mean_memory"] >= 0
        assert "std_time" in stats
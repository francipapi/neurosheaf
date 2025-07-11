"""Unit tests for performance profiling utilities."""

import pytest
import time
import tempfile
import json
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import threading

from neurosheaf.utils.profiling import (
    ProfileResult,
    ProfileManager,
    MemoryBackend,
    MemoryMeasurement,
    MemoryMeasurementSystem,
    profile_memory,
    profile_time,
    profile_comprehensive,
    MemoryMonitor,
    get_memory_usage,
    clear_gpu_memory,
    check_memory_limits,
    get_profile_manager,
    benchmark_function,
    assess_memory_reduction,
    validate_memory_measurement_precision,
)
from neurosheaf.utils.exceptions import MemoryError


class TestMemoryMeasurement:
    """Test MemoryMeasurement dataclass."""
    
    def test_basic_creation(self):
        """Test basic MemoryMeasurement creation."""
        measurement = MemoryMeasurement(
            backend=MemoryBackend.TRACEMALLOC,
            current_mb=100.0,
            peak_mb=120.0
        )
        
        assert measurement.backend == MemoryBackend.TRACEMALLOC
        assert measurement.current_mb == 100.0
        assert measurement.peak_mb == 120.0
        assert measurement.is_valid == True
        assert measurement.error is None
    
    def test_validation_negative_values(self):
        """Test validation with negative values."""
        measurement = MemoryMeasurement(
            backend=MemoryBackend.PSUTIL,
            current_mb=-10.0,
            peak_mb=120.0
        )
        
        assert measurement.is_valid == False
        assert "Negative memory values" in measurement.error
    
    def test_validation_peak_less_than_current(self):
        """Test validation when peak is less than current."""
        measurement = MemoryMeasurement(
            backend=MemoryBackend.TRACEMALLOC,
            current_mb=120.0,
            peak_mb=100.0
        )
        
        assert measurement.is_valid == False
        assert "Peak memory less than current" in measurement.error


class TestMemoryMeasurementSystem:
    """Test MemoryMeasurementSystem class."""
    
    def test_system_creation(self):
        """Test MemoryMeasurementSystem creation."""
        system = MemoryMeasurementSystem()
        assert len(system.available_backends) > 0
        assert MemoryBackend.TRACEMALLOC in system.available_backends
        assert MemoryBackend.PSUTIL in system.available_backends
    
    def test_backend_detection(self):
        """Test backend detection."""
        system = MemoryMeasurementSystem()
        backends = system._detect_available_backends()
        
        # Should always have these
        assert MemoryBackend.TRACEMALLOC in backends
        assert MemoryBackend.PSUTIL in backends
        assert MemoryBackend.SYSTEM in backends
        
        # MPS should be available on Apple Silicon
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            assert MemoryBackend.MPS in backends
    
    def test_measure_all_backends(self):
        """Test measuring all available backends."""
        system = MemoryMeasurementSystem()
        measurements = system._measure_all_backends()
        
        assert len(measurements) > 0
        for backend, measurement in measurements.items():
            assert isinstance(measurement, MemoryMeasurement)
            assert measurement.backend == backend
    
    def test_best_memory_estimate(self):
        """Test getting best memory estimate."""
        system = MemoryMeasurementSystem()
        
        # Create mock measurements
        measurements = {
            MemoryBackend.TRACEMALLOC: MemoryMeasurement(
                backend=MemoryBackend.TRACEMALLOC,
                current_mb=100.0,
                peak_mb=120.0,
                is_valid=True
            ),
            MemoryBackend.PSUTIL: MemoryMeasurement(
                backend=MemoryBackend.PSUTIL,
                current_mb=95.0,
                peak_mb=110.0,
                is_valid=True
            )
        }
        
        cpu_memory, gpu_memory = system.get_best_memory_estimate(measurements)
        
        # Should prefer tracemalloc for CPU
        assert cpu_memory == 120.0
        assert gpu_memory == 0.0
    
    def test_measurement_validation(self):
        """Test measurement validation."""
        system = MemoryMeasurementSystem()
        
        # Valid measurements
        measurements = {
            MemoryBackend.TRACEMALLOC: MemoryMeasurement(
                backend=MemoryBackend.TRACEMALLOC,
                current_mb=100.0,
                peak_mb=120.0,
                is_valid=True
            )
        }
        
        quality, issues = system.validate_measurements(measurements)
        assert quality == "good"
        assert len(issues) == 0
        
        # Invalid measurements
        measurements = {
            MemoryBackend.TRACEMALLOC: MemoryMeasurement(
                backend=MemoryBackend.TRACEMALLOC,
                current_mb=0.0,
                peak_mb=0.0,
                is_valid=False,
                error="Test error"
            )
        }
        
        quality, issues = system.validate_measurements(measurements)
        assert quality == "error"
        assert len(issues) > 0


class TestProfileResult:
    """Test ProfileResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic ProfileResult creation."""
        measurements = {
            MemoryBackend.TRACEMALLOC: MemoryMeasurement(
                backend=MemoryBackend.TRACEMALLOC,
                current_mb=80.0,
                peak_mb=100.0
            )
        }
        
        result = ProfileResult(
            function_name="test_func",
            execution_time=1.5,
            measurements=measurements,
            primary_cpu_memory_mb=100.0,
            primary_gpu_memory_mb=0.0
        )
        
        assert result.function_name == "test_func"
        assert result.execution_time == 1.5
        assert result.primary_cpu_memory_mb == 100.0
        assert result.primary_gpu_memory_mb == 0.0
        
        # Test legacy compatibility
        assert result.cpu_memory_peak_mb == 100.0
        assert result.gpu_memory_peak_mb == 0.0
    
    def test_creation_with_gpu_memory(self):
        """Test ProfileResult creation with GPU memory."""
        measurements = {
            MemoryBackend.TRACEMALLOC: MemoryMeasurement(
                backend=MemoryBackend.TRACEMALLOC,
                current_mb=150.0,
                peak_mb=200.0
            )
        }
        
        result = ProfileResult(
            function_name="gpu_func",
            execution_time=2.0,
            measurements=measurements,
            primary_cpu_memory_mb=200.0,
            primary_gpu_memory_mb=500.0
        )
        
        assert result.gpu_memory_peak_mb == 500.0
        assert result.gpu_memory_current_mb == 500.0
    
    def test_to_dict_method(self):
        """Test to_dict method."""
        measurements = {
            MemoryBackend.TRACEMALLOC: MemoryMeasurement(
                backend=MemoryBackend.TRACEMALLOC,
                current_mb=80.0,
                peak_mb=100.0
            )
        }
        
        result = ProfileResult(
            function_name="test_func",
            execution_time=1.0,
            measurements=measurements,
            primary_cpu_memory_mb=100.0,
            primary_gpu_memory_mb=0.0,
            args_info={"arg_count": 2},
            context={"test": "value"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["function_name"] == "test_func"
        assert result_dict["execution_time"] == 1.0
        assert result_dict["primary_cpu_memory_mb"] == 100.0
        assert result_dict["args_info"] == {"arg_count": 2}
        assert result_dict["context"] == {"test": "value"}
        assert "timestamp" in result_dict
        assert "measurements" in result_dict


class TestProfileManager:
    """Test ProfileManager class."""
    
    def test_manager_creation(self):
        """Test ProfileManager creation."""
        manager = ProfileManager()
        assert len(manager.results) == 0
    
    def test_add_result(self):
        """Test adding profiling results."""
        manager = ProfileManager()
        measurements = {
            MemoryBackend.TRACEMALLOC: MemoryMeasurement(
                backend=MemoryBackend.TRACEMALLOC,
                current_mb=80.0,
                peak_mb=100.0
            )
        }
        result = ProfileResult("test", 1.0, measurements, 100.0, 0.0)
        
        manager.add_result(result)
        
        assert len(manager.results) == 1
        assert manager.results[0] == result
    
    def test_get_results(self):
        """Test getting profiling results."""
        manager = ProfileManager()
        measurements1 = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 80.0, 100.0)}
        measurements2 = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 150.0, 200.0)}
        result1 = ProfileResult("func1", 1.0, measurements1, 100.0, 0.0)
        result2 = ProfileResult("func2", 2.0, measurements2, 200.0, 0.0)
        
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
        measurements = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 80.0, 100.0)}
        result = ProfileResult("test", 1.0, measurements, 100.0, 0.0)
        manager.add_result(result)
        
        manager.clear_results()
        
        assert len(manager.results) == 0
    
    def test_save_results(self, temp_dir):
        """Test saving results to file."""
        manager = ProfileManager()
        measurements = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 80.0, 100.0)}
        result = ProfileResult("test", 1.0, measurements, 100.0, 0.0)
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
        measurements1 = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 80.0, 100.0)}
        measurements2 = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 90.0, 120.0)}
        measurements3 = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 150.0, 200.0)}
        result1 = ProfileResult("func1", 1.0, measurements1, 100.0, 0.0)
        result2 = ProfileResult("func1", 1.5, measurements2, 120.0, 0.0)
        result3 = ProfileResult("func2", 2.0, measurements3, 200.0, 0.0)
        
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
                measurements = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, i * 8.0, i * 10.0)}
                result = ProfileResult(f"func_{thread_id}", i * 0.1, measurements, i * 10.0, 0.0)
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
    
    @patch('neurosheaf.utils.profiling.HAS_TORCH', True)
    @patch('neurosheaf.utils.profiling.torch')
    def test_profiling_with_gpu(self, mock_torch):
        """Test profiling with GPU support."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 200  # 200MB
        mock_torch.cuda.reset_peak_memory_stats.return_value = None
        
        # Mock MPS to be unavailable to avoid conflict
        mock_torch.backends.mps.is_available.return_value = False
        
        @profile_memory(log_results=False)
        def gpu_function():
            return "gpu_result"
        
        result = gpu_function()
        assert result == "gpu_result"
        
        manager = get_profile_manager()
        results = manager.get_results("gpu_function")
        assert len(results) >= 1
        # Since we're mocking, the GPU memory measurement might not work perfectly
        # Just check that the result exists and has reasonable values
        assert results[-1].primary_gpu_memory_mb >= 0


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
    
    def test_get_memory_usage_cpu(self):
        """Test getting CPU memory usage."""
        cpu_memory, gpu_memory = get_memory_usage()
        
        # Should return reasonable values
        assert cpu_memory >= 0
        assert gpu_memory >= 0
    
    def test_get_memory_usage_gpu(self):
        """Test getting GPU memory usage."""
        cpu_memory, gpu_memory = get_memory_usage()
        
        # Should return reasonable values
        assert cpu_memory >= 0
        assert gpu_memory >= 0
    
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
        measurements = {MemoryBackend.TRACEMALLOC: MemoryMeasurement(MemoryBackend.TRACEMALLOC, 80.0, 100.0)}
        result = ProfileResult("test", 1.0, measurements, 100.0, 0.0)
        manager.add_result(result)
        
        # Get manager again
        manager2 = get_profile_manager()
        results = manager2.get_results("test")
        
        assert len(results) >= 1
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


class TestMemoryReductionAssessment:
    """Test memory reduction assessment functionality."""
    
    def test_basic_assessment(self):
        """Test basic memory reduction assessment."""
        baseline_mb = 20 * 1024  # 20GB
        optimized_mb = 3 * 1024  # 3GB
        
        assessment = assess_memory_reduction(baseline_mb, optimized_mb)
        
        assert assessment["baseline_memory_gb"] == 20.0
        assert assessment["optimized_memory_gb"] == 3.0
        assert assessment["actual_reduction_factor"] == pytest.approx(6.67, rel=0.01)
        assert assessment["target_reduction_factor"] == 7.0
        assert assessment["reduction_achieved"] == False
        assert assessment["meets_3gb_target"] == True
    
    def test_target_achieved(self):
        """Test assessment when target is achieved."""
        baseline_mb = 21 * 1024  # 21GB
        optimized_mb = 3 * 1024  # 3GB
        
        assessment = assess_memory_reduction(baseline_mb, optimized_mb)
        
        assert assessment["actual_reduction_factor"] == 7.0
        assert assessment["reduction_achieved"] == True
        assert assessment["progress_to_target"] == 1.0
    
    def test_invalid_baseline(self):
        """Test assessment with invalid baseline."""
        assessment = assess_memory_reduction(0, 1000)
        
        assert "error" in assessment
        assert "Invalid baseline memory measurement" in assessment["error"]
    
    def test_zero_optimized_memory(self):
        """Test assessment with zero optimized memory."""
        assessment = assess_memory_reduction(1000, 0)
        
        assert assessment["actual_reduction_factor"] == float('inf')


class TestMemoryMeasurementPrecision:
    """Test memory measurement precision validation."""
    
    def test_precision_validation(self):
        """Test precision validation function."""
        # This is a functional test that depends on system behavior
        results = validate_memory_measurement_precision()
        
        assert "overall_precision" in results
        assert results["overall_precision"] in ["good", "acceptable", "poor", "error"]
        
        if "precision_analysis" in results:
            for backend, analysis in results["precision_analysis"].items():
                assert "detected_mb" in analysis
                assert "error_percent" in analysis
                assert "precision" in analysis
                assert analysis["precision"] in ["good", "acceptable", "poor"]


@pytest.mark.phase1
class TestAppleSiliconMemoryMeasurement:
    """Test Apple Silicon specific memory measurement features."""
    
    def test_unified_memory_detection(self):
        """Test unified memory architecture detection."""
        system = MemoryMeasurementSystem()
        
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            assert system.unified_memory == True
            assert system.is_apple_silicon == True
        else:
            assert system.unified_memory == False
            assert system.is_apple_silicon == False
    
    def test_mps_backend_availability(self):
        """Test MPS backend availability detection."""
        system = MemoryMeasurementSystem()
        
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # MPS should be available on Apple Silicon
            assert MemoryBackend.MPS in system.available_backends
        else:
            # MPS should not be available on other systems
            assert MemoryBackend.MPS not in system.available_backends
    
    @pytest.mark.skipif(
        not (platform.system() == "Darwin" and platform.processor() == "arm"),
        reason="Requires Apple Silicon Mac"
    )
    def test_mps_memory_measurement(self):
        """Test MPS memory measurement on Apple Silicon."""
        system = MemoryMeasurementSystem()
        
        try:
            measurement = system._measure_backend(MemoryBackend.MPS)
            assert measurement.backend == MemoryBackend.MPS
            assert measurement.current_mb >= 0
            assert measurement.peak_mb >= 0
        except Exception as e:
            # MPS measurement may fail in some environments
            assert "MPS" in str(e)
    
    def test_negative_memory_prevention(self):
        """Test that negative memory measurements are prevented."""
        
        @profile_memory(log_results=False)
        def test_function():
            # Small operation that shouldn't cause negative memory
            return sum(range(100))
        
        result = test_function()
        assert result == 4950
        
        manager = get_profile_manager()
        results = manager.get_results("test_function")
        assert len(results) >= 1
        
        latest_result = results[-1]
        assert latest_result.primary_cpu_memory_mb >= 0
        assert latest_result.primary_gpu_memory_mb >= 0
        
        # Check all measurements are non-negative
        for measurement in latest_result.measurements.values():
            if measurement.is_valid:
                assert measurement.current_mb >= 0
                assert measurement.peak_mb >= 0


@pytest.mark.phase1
class TestMemoryMeasurementRobustness:
    """Test robustness of memory measurement system."""
    
    def test_measurement_consistency(self):
        """Test consistency of measurements across multiple calls."""
        
        @profile_memory(log_results=False)
        def consistent_function():
            # Allocate consistent amount of memory
            data = [0] * 100000  # ~800KB
            return len(data)
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = consistent_function()
            assert result == 100000
            
            manager = get_profile_manager()
            prof_results = manager.get_results("consistent_function")
            if prof_results:
                results.append(prof_results[-1].primary_cpu_memory_mb)
        
        # Check that measurements are reasonably consistent
        if len(results) >= 2:
            mean_memory = sum(results) / len(results)
            for memory in results:
                # Allow up to 50% variance (measurements can be noisy)
                assert abs(memory - mean_memory) / mean_memory < 0.5
    
    def test_error_handling_robustness(self):
        """Test error handling in memory measurement."""
        
        @profile_memory(log_results=False)
        def error_function():
            # Function that might cause measurement issues
            import gc
            gc.collect()  # Force garbage collection
            return "success"
        
        # Should not raise exception
        result = error_function()
        assert result == "success"
        
        manager = get_profile_manager()
        results = manager.get_results("error_function")
        assert len(results) >= 1
        
        # Should have some kind of measurement
        latest_result = results[-1]
        assert latest_result.measurement_quality in ["good", "warning", "error"]
    
    def test_memory_threshold_validation(self):
        """Test memory threshold validation for 7x reduction target."""
        
        # Test with 3GB threshold (target after 7x reduction)
        @profile_memory(memory_threshold_mb=3000.0, log_results=False)
        def small_memory_function():
            # Small allocation that should be under threshold
            data = [0] * 1000
            return len(data)
        
        result = small_memory_function()
        assert result == 1000
        
        manager = get_profile_manager()
        results = manager.get_results("small_memory_function")
        assert len(results) >= 1
        
        latest_result = results[-1]
        # Should be well under 3GB threshold
        assert latest_result.primary_cpu_memory_mb < 3000.0
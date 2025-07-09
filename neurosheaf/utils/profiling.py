"""Performance profiling utilities for Neurosheaf.

This module provides decorators and utilities for monitoring memory usage,
execution time, and other performance metrics throughout the Neurosheaf framework.
It supports both CPU and GPU profiling with detailed reporting.

Key Features:
- Memory usage tracking (CPU and GPU)
- Execution time measurement
- Performance regression detection
- Automatic threshold-based warnings
- Detailed profiling reports
- Context-aware profiling
"""

import functools
import time
import tracemalloc
from typing import Callable, Tuple, Any, Dict, Optional, List
import threading
from dataclasses import dataclass, field
from pathlib import Path
import json
import psutil
import os

# Try to import torch for GPU profiling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .logging import get_logger
from .exceptions import MemoryError


@dataclass
class ProfileResult:
    """Container for profiling results."""
    
    function_name: str
    execution_time: float
    cpu_memory_peak_mb: float
    cpu_memory_current_mb: float
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_current_mb: float = 0.0
    args_info: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "execution_time": self.execution_time,
            "cpu_memory_peak_mb": self.cpu_memory_peak_mb,
            "cpu_memory_current_mb": self.cpu_memory_current_mb,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "gpu_memory_current_mb": self.gpu_memory_current_mb,
            "args_info": self.args_info,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class ProfileManager:
    """Manages profiling results and provides reporting functionality."""
    
    def __init__(self):
        self.results: List[ProfileResult] = []
        self.lock = threading.Lock()
        self.logger = get_logger("neurosheaf.profiling")
    
    def add_result(self, result: ProfileResult):
        """Add a profiling result."""
        with self.lock:
            self.results.append(result)
    
    def get_results(self, function_name: Optional[str] = None) -> List[ProfileResult]:
        """Get profiling results, optionally filtered by function name."""
        with self.lock:
            if function_name:
                return [r for r in self.results if r.function_name == function_name]
            return self.results.copy()
    
    def clear_results(self):
        """Clear all profiling results."""
        with self.lock:
            self.results.clear()
    
    def save_results(self, filepath: Path):
        """Save profiling results to JSON file."""
        with self.lock:
            data = [result.to_dict() for result in self.results]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a human-readable profiling report."""
        with self.lock:
            if not self.results:
                return "No profiling results available."
            
            report = ["Neurosheaf Profiling Report", "=" * 30, ""]
            
            # Summary statistics
            functions = set(r.function_name for r in self.results)
            total_time = sum(r.execution_time for r in self.results)
            max_memory = max(r.cpu_memory_peak_mb for r in self.results)
            
            report.extend([
                f"Total functions profiled: {len(functions)}",
                f"Total execution time: {total_time:.2f}s",
                f"Peak memory usage: {max_memory:.2f}MB",
                ""
            ])
            
            # Per-function summary
            for func_name in sorted(functions):
                func_results = [r for r in self.results if r.function_name == func_name]
                avg_time = sum(r.execution_time for r in func_results) / len(func_results)
                avg_memory = sum(r.cpu_memory_peak_mb for r in func_results) / len(func_results)
                
                report.extend([
                    f"Function: {func_name}",
                    f"  Calls: {len(func_results)}",
                    f"  Avg time: {avg_time:.2f}s",
                    f"  Avg memory: {avg_memory:.2f}MB",
                    ""
                ])
            
            return "\n".join(report)


# Global profile manager
_profile_manager = ProfileManager()


def profile_memory(
    memory_threshold_mb: float = 1000.0,
    log_results: bool = True,
    include_args: bool = False
) -> Callable:
    """Decorator to profile memory usage of a function.
    
    Args:
        memory_threshold_mb: Threshold for memory usage warnings
        log_results: Whether to log profiling results
        include_args: Whether to include function arguments in results
        
    Returns:
        Decorated function with memory profiling
        
    Example:
        @profile_memory(memory_threshold_mb=500.0)
        def compute_cka_matrix(X, Y):
            return X @ X.T, Y @ Y.T
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"neurosheaf.profiling.{func.__name__}")
            
            # Get initial memory state
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Start CPU memory tracing
            tracemalloc.start()
            
            # GPU memory tracking
            gpu_initial = gpu_peak = 0.0
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_initial = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Get peak memory usage
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Convert to MB
                cpu_memory_peak = peak_memory / 1024 / 1024
                cpu_memory_current = current_memory / 1024 / 1024
                
                # GPU memory
                if HAS_TORCH and torch.cuda.is_available():
                    gpu_current = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    gpu_current = gpu_peak = 0.0
                
                # Prepare arguments info
                args_info = {}
                if include_args:
                    args_info = {
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "args_types": [type(arg).__name__ for arg in args]
                    }
                
                # Create profiling result
                profile_result = ProfileResult(
                    function_name=func.__name__,
                    execution_time=execution_time,
                    cpu_memory_peak_mb=cpu_memory_peak,
                    cpu_memory_current_mb=cpu_memory_current,
                    gpu_memory_peak_mb=gpu_peak,
                    gpu_memory_current_mb=gpu_current,
                    args_info=args_info
                )
                
                # Add to global manager
                _profile_manager.add_result(profile_result)
                
                # Log results
                if log_results:
                    logger.info(f"{func.__name__} execution:")
                    logger.info(f"  Time: {execution_time:.2f}s")
                    logger.info(f"  CPU Memory Peak: {cpu_memory_peak:.2f}MB")
                    if HAS_TORCH and torch.cuda.is_available():
                        logger.info(f"  GPU Memory Peak: {gpu_peak:.2f}MB")
                
                # Check memory threshold
                if cpu_memory_peak > memory_threshold_mb:
                    logger.warning(
                        f"{func.__name__} exceeded memory threshold: "
                        f"{cpu_memory_peak:.2f}MB > {memory_threshold_mb:.2f}MB"
                    )
                
                return result
                
            except Exception as e:
                # Clean up tracing on error
                tracemalloc.stop()
                logger.error(f"Error in {func.__name__}: {e}")
                raise
                
        return wrapper
    return decorator


def profile_time(
    time_threshold_seconds: float = 60.0,
    log_results: bool = True
) -> Callable:
    """Decorator to profile execution time of a function.
    
    Args:
        time_threshold_seconds: Threshold for execution time warnings
        log_results: Whether to log profiling results
        
    Returns:
        Decorated function with time profiling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"neurosheaf.profiling.{func.__name__}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if log_results:
                    logger.info(f"{func.__name__} execution time: {execution_time:.2f}s")
                
                if execution_time > time_threshold_seconds:
                    logger.warning(
                        f"{func.__name__} exceeded time threshold: "
                        f"{execution_time:.2f}s > {time_threshold_seconds:.2f}s"
                    )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Error in {func.__name__} after {execution_time:.2f}s: {e}")
                raise
                
        return wrapper
    return decorator


def profile_comprehensive(
    memory_threshold_mb: float = 1000.0,
    time_threshold_seconds: float = 60.0,
    log_results: bool = True,
    include_args: bool = False
) -> Callable:
    """Decorator combining memory and time profiling.
    
    Args:
        memory_threshold_mb: Memory threshold for warnings
        time_threshold_seconds: Time threshold for warnings
        log_results: Whether to log profiling results
        include_args: Whether to include function arguments in results
        
    Returns:
        Decorated function with comprehensive profiling
    """
    def decorator(func: Callable) -> Callable:
        # Apply both decorators
        func = profile_memory(
            memory_threshold_mb=memory_threshold_mb,
            log_results=log_results,
            include_args=include_args
        )(func)
        func = profile_time(
            time_threshold_seconds=time_threshold_seconds,
            log_results=False  # Avoid duplicate logging
        )(func)
        return func
    return decorator


class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, name: str = "operation", threshold_mb: float = 1000.0):
        self.name = name
        self.threshold_mb = threshold_mb
        self.logger = get_logger("neurosheaf.profiling.memory")
        self.start_memory = 0.0
        self.peak_memory = 0.0
    
    def __enter__(self):
        """Start memory monitoring."""
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        tracemalloc.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop memory monitoring and report results."""
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.peak_memory = peak_memory / 1024 / 1024  # MB
        
        self.logger.info(f"{self.name} memory usage: {self.peak_memory:.2f}MB")
        
        if self.peak_memory > self.threshold_mb:
            self.logger.warning(
                f"{self.name} exceeded memory threshold: "
                f"{self.peak_memory:.2f}MB > {self.threshold_mb:.2f}MB"
            )
    
    def check_memory(self) -> float:
        """Check current memory usage."""
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        return current_memory - self.start_memory


def get_memory_usage() -> Tuple[float, float]:
    """Get current CPU and GPU memory usage.
    
    Returns:
        Tuple of (cpu_memory_mb, gpu_memory_mb)
    """
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # GPU memory
    gpu_memory = 0.0
    if HAS_TORCH and torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    return cpu_memory, gpu_memory


def clear_gpu_memory():
    """Clear GPU memory cache if available."""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_memory_limits(
    cpu_limit_mb: float = 3000.0,
    gpu_limit_mb: float = 8000.0,
    operation: str = "operation"
):
    """Check current memory usage against limits.
    
    Args:
        cpu_limit_mb: CPU memory limit in MB
        gpu_limit_mb: GPU memory limit in MB
        operation: Name of the operation for error messages
        
    Raises:
        MemoryError: If memory usage exceeds limits
    """
    cpu_memory, gpu_memory = get_memory_usage()
    
    if cpu_memory > cpu_limit_mb:
        raise MemoryError(
            f"{operation} would exceed CPU memory limit",
            memory_used_mb=cpu_memory,
            memory_limit_mb=cpu_limit_mb,
            memory_type="cpu"
        )
    
    if gpu_memory > gpu_limit_mb:
        raise MemoryError(
            f"{operation} would exceed GPU memory limit",
            memory_used_mb=gpu_memory,
            memory_limit_mb=gpu_limit_mb,
            memory_type="gpu"
        )


def get_profile_manager() -> ProfileManager:
    """Get the global profile manager."""
    return _profile_manager


def benchmark_function(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict = None,
    num_runs: int = 10,
    warmup_runs: int = 2
) -> Dict[str, float]:
    """Benchmark a function with multiple runs.
    
    Args:
        func: Function to benchmark
        args: Function arguments
        kwargs: Function keyword arguments
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted)
        
    Returns:
        Dictionary with benchmark statistics
    """
    if kwargs is None:
        kwargs = {}
    
    logger = get_logger("neurosheaf.profiling.benchmark")
    
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Benchmark runs
    times = []
    memories = []
    
    for i in range(num_runs):
        # Memory before
        memory_before = get_memory_usage()[0]
        
        # Time execution
        start_time = time.time()
        func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Memory after
        memory_after = get_memory_usage()[0]
        memory_used = memory_after - memory_before
        
        times.append(execution_time)
        memories.append(memory_used)
    
    # Calculate statistics
    stats = {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "mean_memory": sum(memories) / len(memories),
        "min_memory": min(memories),
        "max_memory": max(memories),
        "num_runs": num_runs,
    }
    
    logger.info(f"Benchmark results for {func.__name__}:")
    logger.info(f"  Mean time: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
    logger.info(f"  Mean memory: {stats['mean_memory']:.2f}MB")
    
    return stats
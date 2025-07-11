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
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
import tracemalloc
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Tuple, Any, Dict, Optional, List, Union

import psutil

# Try to import torch for GPU profiling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .logging import get_logger
from .exceptions import MemoryError


class MemoryBackend(Enum):
    """Available memory measurement backends."""
    TRACEMALLOC = "tracemalloc"
    PSUTIL = "psutil"
    MPS = "mps"
    CUDA = "cuda"
    SYSTEM = "system"


@dataclass
class MemoryMeasurement:
    """Container for memory measurement from a specific backend."""
    
    backend: MemoryBackend
    current_mb: float
    peak_mb: float
    is_valid: bool = True
    error: Optional[str] = None
    measurement_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate measurement after initialization."""
        if self.current_mb < 0 or self.peak_mb < 0:
            self.is_valid = False
            self.error = f"Negative memory values: current={self.current_mb:.2f}, peak={self.peak_mb:.2f}"
        elif self.peak_mb < self.current_mb:
            self.is_valid = False
            self.error = f"Peak memory less than current: peak={self.peak_mb:.2f}, current={self.current_mb:.2f}"


@dataclass
class ProfileResult:
    """Container for profiling results with multiple backend measurements."""
    
    function_name: str
    execution_time: float
    measurements: Dict[MemoryBackend, MemoryMeasurement]
    primary_cpu_memory_mb: float  # Best estimate of CPU memory
    primary_gpu_memory_mb: float  # Best estimate of GPU memory
    unified_memory_mb: float = 0.0  # For Apple Silicon unified memory
    args_info: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    measurement_quality: str = "good"  # good, warning, error
    validation_issues: List[str] = field(default_factory=list)
    
    # Legacy compatibility
    @property
    def cpu_memory_peak_mb(self) -> float:
        return self.primary_cpu_memory_mb
    
    @property
    def cpu_memory_current_mb(self) -> float:
        return self.primary_cpu_memory_mb
    
    @property
    def gpu_memory_peak_mb(self) -> float:
        return self.primary_gpu_memory_mb
    
    @property
    def gpu_memory_current_mb(self) -> float:
        return self.primary_gpu_memory_mb
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        measurements_dict = {}
        for backend, measurement in self.measurements.items():
            measurements_dict[backend.value] = {
                "current_mb": measurement.current_mb,
                "peak_mb": measurement.peak_mb,
                "is_valid": measurement.is_valid,
                "error": measurement.error,
                "measurement_time": measurement.measurement_time
            }
        
        return {
            "function_name": self.function_name,
            "execution_time": self.execution_time,
            "measurements": measurements_dict,
            "primary_cpu_memory_mb": self.primary_cpu_memory_mb,
            "primary_gpu_memory_mb": self.primary_gpu_memory_mb,
            "unified_memory_mb": self.unified_memory_mb,
            "args_info": self.args_info,
            "context": self.context,
            "timestamp": self.timestamp,
            "measurement_quality": self.measurement_quality,
            "validation_issues": self.validation_issues,
            # Legacy compatibility
            "cpu_memory_peak_mb": self.cpu_memory_peak_mb,
            "cpu_memory_current_mb": self.cpu_memory_current_mb,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "gpu_memory_current_mb": self.gpu_memory_current_mb,
        }


# NOTE: ProfileManager class is defined later in this file to avoid duplication


class MemoryMeasurementSystem:
    """Precise memory measurement system with multiple backends."""
    
    def __init__(self):
        self.logger = get_logger("neurosheaf.profiling.memory")
        self.is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"
        self.unified_memory = self.is_apple_silicon
        
        # Initialize available backends
        self.available_backends = self._detect_available_backends()
        self.logger.info(f"Available memory backends: {[b.value for b in self.available_backends]}")
    
    def _detect_available_backends(self) -> List[MemoryBackend]:
        """Detect available memory measurement backends."""
        backends = [MemoryBackend.TRACEMALLOC, MemoryBackend.PSUTIL, MemoryBackend.SYSTEM]
        
        if HAS_TORCH:
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                backends.append(MemoryBackend.MPS)
            if torch.cuda.is_available():
                backends.append(MemoryBackend.CUDA)
        
        return backends
    
    @contextmanager
    def measure_memory(self, operation_name: str = "operation"):
        """Context manager for precise memory measurement."""
        try:
            initial_measurements = self._measure_all_backends()
            yield
        finally:
            final_measurements = self._measure_all_backends()
            
            # Calculate differences and validate
            results = {}
            for backend in self.available_backends:
                if backend in initial_measurements and backend in final_measurements:
                    initial = initial_measurements[backend]
                    final = final_measurements[backend]
                    
                    if initial.is_valid and final.is_valid:
                        current_diff = final.current_mb - initial.current_mb
                        peak_diff = final.peak_mb - initial.peak_mb
                        
                        results[backend] = MemoryMeasurement(
                            backend=backend,
                            current_mb=max(0, current_diff),  # Ensure non-negative
                            peak_mb=max(0, peak_diff),
                            is_valid=True
                        )
                    else:
                        results[backend] = MemoryMeasurement(
                            backend=backend,
                            current_mb=0,
                            peak_mb=0,
                            is_valid=False,
                            error=f"Backend measurement failed: {initial.error or final.error}"
                        )
            
            self._log_memory_results(operation_name, results)
    
    def _measure_all_backends(self) -> Dict[MemoryBackend, MemoryMeasurement]:
        """Measure memory usage across all available backends."""
        measurements = {}
        
        for backend in self.available_backends:
            try:
                measurement = self._measure_backend(backend)
                measurements[backend] = measurement
            except Exception as e:
                self.logger.warning(f"Failed to measure {backend.value}: {e}")
                measurements[backend] = MemoryMeasurement(
                    backend=backend,
                    current_mb=0,
                    peak_mb=0,
                    is_valid=False,
                    error=str(e)
                )
        
        return measurements
    
    def _measure_backend(self, backend: MemoryBackend) -> MemoryMeasurement:
        """Measure memory usage for a specific backend."""
        if backend == MemoryBackend.TRACEMALLOC:
            return self._measure_tracemalloc()
        elif backend == MemoryBackend.PSUTIL:
            return self._measure_psutil()
        elif backend == MemoryBackend.MPS:
            return self._measure_mps()
        elif backend == MemoryBackend.CUDA:
            return self._measure_cuda()
        elif backend == MemoryBackend.SYSTEM:
            return self._measure_system()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _measure_tracemalloc(self) -> MemoryMeasurement:
        """Measure using tracemalloc (most precise for Python objects)."""
        if not tracemalloc.is_tracing():
            # Start tracing if not already active
            tracemalloc.start()
            current_mb = peak_mb = 0
        else:
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024
        
        return MemoryMeasurement(
            backend=MemoryBackend.TRACEMALLOC,
            current_mb=current_mb,
            peak_mb=peak_mb
        )
    
    def _measure_psutil(self) -> MemoryMeasurement:
        """Measure using psutil (process-wide memory)."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # RSS (Resident Set Size) is the physical memory currently used
        current_mb = memory_info.rss / 1024 / 1024
        
        # For psutil, we use current as peak since we can't track peak over time
        return MemoryMeasurement(
            backend=MemoryBackend.PSUTIL,
            current_mb=current_mb,
            peak_mb=current_mb
        )
    
    def _measure_mps(self) -> MemoryMeasurement:
        """Measure MPS memory usage on Apple Silicon."""
        if not (HAS_TORCH and hasattr(torch, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS not available")
        
        try:
            # Current allocated memory
            current_mb = torch.mps.current_allocated_memory() / 1024 / 1024
            
            # Driver allocated memory (includes cached)
            driver_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
            
            # Use driver allocated as peak estimate
            return MemoryMeasurement(
                backend=MemoryBackend.MPS,
                current_mb=current_mb,
                peak_mb=driver_mb
            )
        except Exception as e:
            # MPS functions may not be available in all PyTorch versions
            raise RuntimeError(f"MPS memory measurement failed: {e}")
    
    def _measure_cuda(self) -> MemoryMeasurement:
        """Measure CUDA memory usage."""
        if not (HAS_TORCH and torch.cuda.is_available()):
            raise RuntimeError("CUDA not available")
        
        current_mb = torch.cuda.memory_allocated() / 1024 / 1024
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return MemoryMeasurement(
            backend=MemoryBackend.CUDA,
            current_mb=current_mb,
            peak_mb=peak_mb
        )
    
    def _measure_system(self) -> MemoryMeasurement:
        """Measure system-wide memory usage."""
        vm = psutil.virtual_memory()
        
        # System memory in use
        used_mb = vm.used / 1024 / 1024
        
        return MemoryMeasurement(
            backend=MemoryBackend.SYSTEM,
            current_mb=used_mb,
            peak_mb=used_mb
        )
    
    def _log_memory_results(self, operation_name: str, results: Dict[MemoryBackend, MemoryMeasurement]):
        """Log memory measurement results."""
        self.logger.info(f"Memory measurement for {operation_name}:")
        for backend, measurement in results.items():
            if measurement.is_valid:
                self.logger.info(f"  {backend.value}: {measurement.current_mb:.2f}MB (peak: {measurement.peak_mb:.2f}MB)")
            else:
                self.logger.warning(f"  {backend.value}: INVALID - {measurement.error}")
    
    def get_best_memory_estimate(self, measurements: Dict[MemoryBackend, MemoryMeasurement]) -> Tuple[float, float]:
        """Get best estimate of CPU and GPU memory usage with fallback handling."""
        cpu_memory = 0.0
        gpu_memory = 0.0
        
        # Priority order for CPU measurement with fallback
        cpu_priority = [MemoryBackend.TRACEMALLOC, MemoryBackend.PSUTIL, MemoryBackend.SYSTEM]
        cpu_fallback_used = False
        
        for backend in cpu_priority:
            if backend in measurements and measurements[backend].is_valid:
                cpu_memory = measurements[backend].peak_mb
                break
            elif backend in measurements:
                # Backend available but measurement invalid - try fallback
                cpu_fallback_used = True
                self.logger.warning(f"CPU measurement backend {backend.value} failed, trying fallback")
        
        # If all CPU backends failed, try emergency fallback
        if cpu_memory == 0.0 and cpu_fallback_used:
            cpu_memory = self._emergency_cpu_memory_estimate()
            self.logger.warning(f"Using emergency CPU memory estimate: {cpu_memory:.2f}MB")
        
        # Priority order for GPU measurement with fallback
        gpu_priority = [MemoryBackend.MPS, MemoryBackend.CUDA]
        gpu_fallback_used = False
        
        for backend in gpu_priority:
            if backend in measurements and measurements[backend].is_valid:
                gpu_memory = measurements[backend].peak_mb
                break
            elif backend in measurements:
                # Backend available but measurement invalid - try fallback
                gpu_fallback_used = True
                self.logger.warning(f"GPU measurement backend {backend.value} failed, trying fallback")
        
        # If all GPU backends failed, try emergency fallback
        if gpu_memory == 0.0 and gpu_fallback_used:
            gpu_memory = self._emergency_gpu_memory_estimate()
            self.logger.warning(f"Using emergency GPU memory estimate: {gpu_memory:.2f}MB")
        
        return cpu_memory, gpu_memory
    
    def validate_measurements(self, measurements: Dict[MemoryBackend, MemoryMeasurement]) -> Tuple[str, List[str]]:
        """Validate memory measurements and return quality assessment."""
        issues = []
        valid_count = sum(1 for m in measurements.values() if m.is_valid)
        total_count = len(measurements)
        
        if valid_count == 0:
            return "error", ["No valid measurements available"]
        
        if valid_count < total_count / 2:
            issues.append(f"Only {valid_count}/{total_count} measurements are valid")
        
        # Check for consistency between backends
        valid_measurements = {k: v for k, v in measurements.items() if v.is_valid}
        
        if len(valid_measurements) >= 2:
            # Check if tracemalloc and psutil measurements are reasonably consistent
            if (MemoryBackend.TRACEMALLOC in valid_measurements and 
                MemoryBackend.PSUTIL in valid_measurements):
                
                tm_mem = valid_measurements[MemoryBackend.TRACEMALLOC].peak_mb
                ps_mem = valid_measurements[MemoryBackend.PSUTIL].peak_mb
                
                if tm_mem > 0 and ps_mem > 0:
                    ratio = abs(tm_mem - ps_mem) / max(tm_mem, ps_mem)
                    if ratio > 0.5:  # More than 50% difference
                        issues.append(f"Large discrepancy between tracemalloc ({tm_mem:.2f}MB) and psutil ({ps_mem:.2f}MB)")
        
        # Determine quality level
        if not issues:
            quality = "good"
        elif len(issues) == 1:
            quality = "warning"
        else:
            quality = "error"
        
        return quality, issues
    
    def _emergency_cpu_memory_estimate(self) -> float:
        """Emergency CPU memory estimate when all backends fail."""
        try:
            # Try basic process memory info
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            try:
                # Try system memory as last resort
                import psutil
                vm = psutil.virtual_memory()
                return vm.used / 1024 / 1024 * 0.1  # Estimate 10% of system memory
            except Exception:
                # If all else fails, return a conservative estimate
                return 100.0  # 100MB conservative estimate
    
    def _emergency_gpu_memory_estimate(self) -> float:
        """Emergency GPU memory estimate when all backends fail."""
        try:
            # Try alternative GPU memory detection
            if HAS_TORCH:
                if torch.cuda.is_available():
                    # Try basic CUDA memory info
                    return torch.cuda.memory_allocated() / 1024 / 1024
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    # Try basic MPS memory info
                    return torch.mps.current_allocated_memory() / 1024 / 1024
        except Exception:
            pass
        
        # If all else fails, return 0 (no GPU memory detected)
        return 0.0


class ProfileManager:
    """Manages profiling results and provides reporting functionality."""
    
    def __init__(self):
        self.results: List[ProfileResult] = []
        self.lock = threading.Lock()
        self.logger = get_logger("neurosheaf.profiling")
        self.memory_system = MemoryMeasurementSystem()
    
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
            max_memory = max(r.primary_cpu_memory_mb for r in self.results)
            
            # Quality assessment
            quality_counts = defaultdict(int)
            for r in self.results:
                quality_counts[r.measurement_quality] += 1
            
            report.extend([
                f"Total functions profiled: {len(functions)}",
                f"Total execution time: {total_time:.2f}s",
                f"Peak memory usage: {max_memory:.2f}MB",
                f"Measurement quality: {dict(quality_counts)}",
                ""
            ])
            
            # Per-function summary
            for func_name in sorted(functions):
                func_results = [r for r in self.results if r.function_name == func_name]
                avg_time = sum(r.execution_time for r in func_results) / len(func_results)
                avg_memory = sum(r.primary_cpu_memory_mb for r in func_results) / len(func_results)
                
                # Quality assessment for this function
                quality_issues = []
                for r in func_results:
                    quality_issues.extend(r.validation_issues)
                
                quality_str = "" if not quality_issues else f" (issues: {len(quality_issues)})"
                
                report.extend([
                    f"Function: {func_name}{quality_str}",
                    f"  Calls: {len(func_results)}",
                    f"  Avg time: {avg_time:.2f}s",
                    f"  Avg memory: {avg_memory:.2f}MB",
                    ""
                ])
            
            return "\n".join(report)


# Global profile manager with thread-safe initialization
_profile_manager = None
_profile_manager_lock = threading.Lock()


def profile_memory(
    memory_threshold_mb: float = 1000.0,
    log_results: bool = True,
    include_args: bool = False,
    validate_measurements: bool = True
) -> Callable:
    """Decorator to profile memory usage of a function with robust multi-backend measurement.
    
    Args:
        memory_threshold_mb: Threshold for memory usage warnings
        log_results: Whether to log profiling results
        include_args: Whether to include function arguments in results
        validate_measurements: Whether to validate measurement consistency
        
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
            profile_manager = get_profile_manager()
            memory_system = profile_manager.memory_system
            
            # Get initial memory measurements from all backends
            initial_measurements = memory_system._measure_all_backends()
            
            # Start tracemalloc if not already running
            tracemalloc_was_running = tracemalloc.is_tracing()
            if not tracemalloc_was_running:
                tracemalloc.start()
            
            # Reset GPU memory stats if available
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Get final memory measurements
                final_measurements = memory_system._measure_all_backends()
                
                # Calculate memory usage for each backend
                measurements = {}
                for backend in memory_system.available_backends:
                    if backend in initial_measurements and backend in final_measurements:
                        initial = initial_measurements[backend]
                        final = final_measurements[backend]
                        
                        if initial.is_valid and final.is_valid:
                            # Calculate memory difference
                            current_diff = final.current_mb - initial.current_mb
                            peak_diff = final.peak_mb - initial.peak_mb
                            
                            # For tracemalloc, use the traced memory directly
                            if backend == MemoryBackend.TRACEMALLOC and tracemalloc.is_tracing():
                                current_mem, peak_mem = tracemalloc.get_traced_memory()
                                current_diff = current_mem / 1024 / 1024
                                peak_diff = peak_mem / 1024 / 1024
                            
                            measurements[backend] = MemoryMeasurement(
                                backend=backend,
                                current_mb=max(0, current_diff),
                                peak_mb=max(0, peak_diff),
                                is_valid=True
                            )
                        else:
                            measurements[backend] = MemoryMeasurement(
                                backend=backend,
                                current_mb=0,
                                peak_mb=0,
                                is_valid=False,
                                error=f"Invalid measurement: {initial.error or final.error}"
                            )
                
                # Stop tracemalloc if we started it
                if not tracemalloc_was_running and tracemalloc.is_tracing():
                    tracemalloc.stop()
                
                # Get best estimates
                cpu_memory, gpu_memory = memory_system.get_best_memory_estimate(measurements)
                
                # Calculate unified memory for Apple Silicon
                unified_memory = 0.0
                if memory_system.unified_memory:
                    # For Apple Silicon, total memory includes both CPU and GPU
                    unified_memory = cpu_memory + gpu_memory
                
                # Validate measurements
                quality, issues = memory_system.validate_measurements(measurements) if validate_measurements else ("good", [])
                
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
                    measurements=measurements,
                    primary_cpu_memory_mb=cpu_memory,
                    primary_gpu_memory_mb=gpu_memory,
                    unified_memory_mb=unified_memory,
                    args_info=args_info,
                    measurement_quality=quality,
                    validation_issues=issues
                )
                
                # Add to global manager
                profile_manager.add_result(profile_result)
                
                # Log results
                if log_results:
                    logger.info(f"{func.__name__} execution:")
                    logger.info(f"  Time: {execution_time:.2f}s")
                    logger.info(f"  CPU Memory: {cpu_memory:.2f}MB")
                    if gpu_memory > 0:
                        logger.info(f"  GPU Memory: {gpu_memory:.2f}MB")
                    if unified_memory > 0:
                        logger.info(f"  Unified Memory: {unified_memory:.2f}MB")
                    if quality != "good":
                        logger.warning(f"  Measurement quality: {quality}")
                        for issue in issues:
                            logger.warning(f"    - {issue}")
                
                # Check memory threshold
                threshold_memory = unified_memory if unified_memory > 0 else cpu_memory
                if threshold_memory > memory_threshold_mb:
                    logger.warning(
                        f"{func.__name__} exceeded memory threshold: "
                        f"{threshold_memory:.2f}MB > {memory_threshold_mb:.2f}MB"
                    )
                
                return result
                
            except Exception as e:
                # Clean up tracing on error
                if not tracemalloc_was_running and tracemalloc.is_tracing():
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
    include_args: bool = False,
    validate_measurements: bool = True
) -> Callable:
    """Decorator combining memory and time profiling with robust measurement.
    
    Args:
        memory_threshold_mb: Memory threshold for warnings
        time_threshold_seconds: Time threshold for warnings
        log_results: Whether to log profiling results
        include_args: Whether to include function arguments in results
        validate_measurements: Whether to validate measurement consistency
        
    Returns:
        Decorated function with comprehensive profiling
    """
    def decorator(func: Callable) -> Callable:
        # Apply both decorators
        func = profile_memory(
            memory_threshold_mb=memory_threshold_mb,
            log_results=log_results,
            include_args=include_args,
            validate_measurements=validate_measurements
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
    """Get current CPU and GPU memory usage using the robust measurement system.
    
    Returns:
        Tuple of (cpu_memory_mb, gpu_memory_mb)
    """
    profile_manager = get_profile_manager()
    memory_system = profile_manager.memory_system
    measurements = memory_system._measure_all_backends()
    cpu_memory, gpu_memory = memory_system.get_best_memory_estimate(measurements)
    return cpu_memory, gpu_memory


def clear_gpu_memory():
    """Clear GPU memory cache if available (includes MPS support)."""
    if HAS_TORCH:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()


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
    """Get the global profile manager with thread-safe initialization."""
    global _profile_manager
    
    if _profile_manager is None:
        with _profile_manager_lock:
            # Double-check pattern to avoid race conditions
            if _profile_manager is None:
                _profile_manager = ProfileManager()
    
    return _profile_manager


def get_mac_device_info() -> Dict[str, Any]:
    """Get Mac-specific device information.
    
    Returns:
        Dictionary with Mac hardware information
    """
    import platform
    
    info = {
        'is_mac': platform.system() == "Darwin",
        'is_apple_silicon': platform.processor() == "arm",
        'machine': platform.machine(),
        'processor': platform.processor(),
        'platform': platform.platform()
    }
    
    # MPS availability
    if HAS_TORCH and hasattr(torch.backends, 'mps'):
        info['mps_available'] = torch.backends.mps.is_available()
        info['mps_built'] = torch.backends.mps.is_built()
    else:
        info['mps_available'] = False
        info['mps_built'] = False
    
    return info


def get_mac_memory_info() -> Dict[str, float]:
    """Get Mac-specific memory information with robust measurement.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    import platform
    
    # System memory
    vm = psutil.virtual_memory()
    memory_info = {
        'system_total_gb': vm.total / (1024**3),
        'system_available_gb': vm.available / (1024**3),
        'system_used_gb': vm.used / (1024**3),
        'system_percent': vm.percent
    }
    
    # Mac-specific memory information
    is_apple_silicon = platform.processor() == "arm"
    
    if is_apple_silicon:
        # Apple Silicon unified memory
        memory_info['unified_memory'] = True
        memory_info['memory_pressure'] = _get_mac_memory_pressure()
        
        # MPS memory if available
        if HAS_TORCH and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            try:
                memory_info['mps_allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
                memory_info['mps_driver_allocated_gb'] = torch.mps.driver_allocated_memory() / (1024**3)
            except Exception as e:
                # Handle cases where MPS memory functions are not available
                memory_info['mps_allocated_gb'] = 0.0
                memory_info['mps_driver_allocated_gb'] = 0.0
                memory_info['mps_error'] = str(e)
        else:
            memory_info['mps_allocated_gb'] = 0.0
            memory_info['mps_driver_allocated_gb'] = 0.0
    else:
        # Intel Mac
        memory_info['unified_memory'] = False
        memory_info['memory_pressure'] = 'unknown'
    
    return memory_info


def _get_mac_memory_pressure() -> str:
    """Get Mac memory pressure status.
    
    Returns:
        Memory pressure status string
    """
    try:
        import subprocess
        result = subprocess.run(['memory_pressure'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse memory pressure output
            output = result.stdout.lower()
            if 'critical' in output:
                return 'critical'
            elif 'warn' in output:
                return 'warning'
            else:
                return 'normal'
    except:
        pass
    
    return 'unknown'


def profile_mac_memory(func: Callable) -> Callable:
    """Mac-specific memory profiling decorator with robust measurement.
    
    Args:
        func: Function to profile
        
    Returns:
        Decorated function with Mac-specific memory profiling
    """
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if platform.system() != "Darwin":
            # Fall back to regular profiling on non-Mac systems
            return profile_memory()(func)(*args, **kwargs)
        
        logger = get_logger("neurosheaf.profiling.mac")
        
        # Get initial memory state
        initial_memory = get_mac_memory_info()
        
        # Clear caches
        clear_gpu_memory()
        
        # Execute function with robust memory measurement
        start_time = time.time()
        try:
            # Use the robust profiling system
            result = profile_memory(log_results=False)(func)(*args, **kwargs)
        finally:
            end_time = time.time()
            
            # Get final memory state
            final_memory = get_mac_memory_info()
            
            # Calculate memory usage
            memory_increase = final_memory['system_used_gb'] - initial_memory['system_used_gb']
            
            # Get the detailed profiling result
            manager = get_profile_manager()
            recent_results = manager.get_results(func.__name__)
            if recent_results:
                recent_result = recent_results[-1]
                
                # Log Mac-specific profiling info
                logger.info(f"Mac Memory Profiling - {func.__name__}:")
                logger.info(f"  Execution time: {end_time - start_time:.2f}s")
                logger.info(f"  System memory increase: {memory_increase:.3f}GB")
                logger.info(f"  Measured memory: {recent_result.primary_cpu_memory_mb:.2f}MB")
                logger.info(f"  Unified memory: {final_memory.get('unified_memory', 'unknown')}")
                logger.info(f"  Memory pressure: {final_memory.get('memory_pressure', 'unknown')}")
                logger.info(f"  Measurement quality: {recent_result.measurement_quality}")
                
                if 'mps_allocated_gb' in final_memory:
                    mps_increase = final_memory['mps_allocated_gb'] - initial_memory.get('mps_allocated_gb', 0)
                    logger.info(f"  MPS memory increase: {mps_increase:.3f}GB")
                
                if recent_result.validation_issues:
                    logger.warning("  Measurement issues:")
                    for issue in recent_result.validation_issues:
                        logger.warning(f"    - {issue}")
        
        return result
    
    return wrapper


def benchmark_function(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict = None,
    num_runs: int = 10,
    warmup_runs: int = 2,
    validate_measurements: bool = True
) -> Dict[str, float]:
    """Benchmark a function with multiple runs and robust memory measurement.
    
    Args:
        func: Function to benchmark
        args: Function arguments
        kwargs: Function keyword arguments
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted)
        validate_measurements: Whether to validate measurement consistency
        
    Returns:
        Dictionary with benchmark statistics
    """
    if kwargs is None:
        kwargs = {}
    
    logger = get_logger("neurosheaf.profiling.benchmark")
    
    # Clear memory before benchmarking
    clear_gpu_memory()
    gc.collect()
    
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
        clear_gpu_memory()
        gc.collect()
    
    # Benchmark runs with robust profiling
    times = []
    cpu_memories = []
    gpu_memories = []
    measurement_qualities = []
    
    for i in range(num_runs):
        # Use robust memory measurement
        memory_system = get_profile_manager().memory_system
        initial_measurements = memory_system._measure_all_backends()
        
        start_time = time.time()
        func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        final_measurements = memory_system._measure_all_backends()
        
        # Calculate memory usage
        measurements = {}
        for backend in memory_system.available_backends:
            if backend in initial_measurements and backend in final_measurements:
                initial = initial_measurements[backend]
                final = final_measurements[backend]
                
                if initial.is_valid and final.is_valid:
                    current_diff = final.current_mb - initial.current_mb
                    peak_diff = final.peak_mb - initial.peak_mb
                    
                    measurements[backend] = MemoryMeasurement(
                        backend=backend,
                        current_mb=max(0, current_diff),
                        peak_mb=max(0, peak_diff),
                        is_valid=True
                    )
        
        # Get best estimates
        cpu_memory, gpu_memory = memory_system.get_best_memory_estimate(measurements)
        
        # Validate measurements
        quality, issues = memory_system.validate_measurements(measurements) if validate_measurements else ("good", [])
        
        cpu_memories.append(cpu_memory)
        gpu_memories.append(gpu_memory)
        measurement_qualities.append(quality)
        
        times.append(execution_time)
        
        # Clear memory between runs
        clear_gpu_memory()
        gc.collect()
    
    # Calculate statistics
    stats = {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "mean_cpu_memory": sum(cpu_memories) / len(cpu_memories),
        "min_cpu_memory": min(cpu_memories),
        "max_cpu_memory": max(cpu_memories),
        "mean_gpu_memory": sum(gpu_memories) / len(gpu_memories),
        "min_gpu_memory": min(gpu_memories),
        "max_gpu_memory": max(gpu_memories),
        "num_runs": num_runs,
        "measurement_quality_counts": {q: measurement_qualities.count(q) for q in set(measurement_qualities)}
    }
    
    # Legacy compatibility
    stats["mean_memory"] = stats["mean_cpu_memory"]
    stats["min_memory"] = stats["min_cpu_memory"]
    stats["max_memory"] = stats["max_cpu_memory"]
    
    logger.info(f"Benchmark results for {func.__name__}:")
    logger.info(f"  Mean time: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
    logger.info(f"  Mean CPU memory: {stats['mean_cpu_memory']:.2f}MB")
    if stats['mean_gpu_memory'] > 0:
        logger.info(f"  Mean GPU memory: {stats['mean_gpu_memory']:.2f}MB")
    logger.info(f"  Measurement quality: {stats['measurement_quality_counts']}")
    
    return stats


def assess_memory_reduction(
    baseline_memory_mb: float,
    optimized_memory_mb: float,
    target_reduction_factor: float = 7.0
) -> Dict[str, Any]:
    """Assess memory reduction against target (7x improvement from 20GB to 3GB).
    
    Args:
        baseline_memory_mb: Memory usage of baseline implementation
        optimized_memory_mb: Memory usage of optimized implementation
        target_reduction_factor: Target reduction factor (default 7.0)
        
    Returns:
        Dictionary with assessment results
    """
    if baseline_memory_mb <= 0:
        return {
            "error": "Invalid baseline memory measurement",
            "baseline_memory_mb": baseline_memory_mb,
            "optimized_memory_mb": optimized_memory_mb
        }
    
    actual_reduction_factor = baseline_memory_mb / optimized_memory_mb if optimized_memory_mb > 0 else float('inf')
    
    # Assessment against 20GB -> 3GB target
    baseline_target_gb = 20.0
    optimized_target_gb = 3.0
    
    baseline_gb = baseline_memory_mb / 1024
    optimized_gb = optimized_memory_mb / 1024
    
    assessment = {
        "baseline_memory_mb": baseline_memory_mb,
        "optimized_memory_mb": optimized_memory_mb,
        "baseline_memory_gb": baseline_gb,
        "optimized_memory_gb": optimized_gb,
        "actual_reduction_factor": actual_reduction_factor,
        "target_reduction_factor": target_reduction_factor,
        "reduction_achieved": actual_reduction_factor >= target_reduction_factor,
        "progress_to_target": min(actual_reduction_factor / target_reduction_factor, 1.0),
        "baseline_target_gb": baseline_target_gb,
        "optimized_target_gb": optimized_target_gb,
        "baseline_vs_target": baseline_gb / baseline_target_gb,
        "optimized_vs_target": optimized_gb / optimized_target_gb,
        "meets_3gb_target": optimized_gb <= optimized_target_gb
    }
    
    return assessment


def create_memory_context_manager(
    name: str = "operation",
    memory_threshold_mb: float = 3000.0,
    validate_measurements: bool = True
):
    """Create a context manager for memory measurement and validation.
    
    Args:
        name: Name of the operation
        memory_threshold_mb: Memory threshold for warnings
        validate_measurements: Whether to validate measurement consistency
        
    Returns:
        Context manager for memory measurement
    """
    @contextmanager
    def memory_context():
        """Context manager for precise memory measurement."""
        profile_manager = get_profile_manager()
        memory_system = profile_manager.memory_system
        logger = get_logger(f"neurosheaf.profiling.{name}")
        
        # Get initial measurements
        initial_measurements = memory_system._measure_all_backends()
        
        try:
            yield memory_system
        finally:
            # Get final measurements
            final_measurements = memory_system._measure_all_backends()
            
            # Calculate memory usage
            measurements = {}
            for backend in memory_system.available_backends:
                if backend in initial_measurements and backend in final_measurements:
                    initial = initial_measurements[backend]
                    final = final_measurements[backend]
                    
                    if initial.is_valid and final.is_valid:
                        current_diff = final.current_mb - initial.current_mb
                        peak_diff = final.peak_mb - initial.peak_mb
                        
                        measurements[backend] = MemoryMeasurement(
                            backend=backend,
                            current_mb=max(0, current_diff),
                            peak_mb=max(0, peak_diff),
                            is_valid=True
                        )
                    else:
                        measurements[backend] = MemoryMeasurement(
                            backend=backend,
                            current_mb=0,
                            peak_mb=0,
                            is_valid=False,
                            error=f"Invalid measurement: {initial.error or final.error}"
                        )
            
            # Get best estimates and validate
            cpu_memory, gpu_memory = memory_system.get_best_memory_estimate(measurements)
            quality, issues = memory_system.validate_measurements(measurements) if validate_measurements else ("good", [])
            
            # Log results
            logger.info(f"Memory measurement for {name}:")
            logger.info(f"  CPU Memory: {cpu_memory:.2f}MB")
            if gpu_memory > 0:
                logger.info(f"  GPU Memory: {gpu_memory:.2f}MB")
            if quality != "good":
                logger.warning(f"  Quality: {quality}")
                for issue in issues:
                    logger.warning(f"    - {issue}")
            
            # Check threshold
            total_memory = cpu_memory + gpu_memory
            if total_memory > memory_threshold_mb:
                logger.warning(f"Memory threshold exceeded: {total_memory:.2f}MB > {memory_threshold_mb:.2f}MB")
    
    return memory_context


def validate_memory_measurement_precision() -> Dict[str, Any]:
    """Validate the precision of memory measurements across backends.
    
    Returns:
        Dictionary with validation results
    """
    profile_manager = get_profile_manager()
    memory_system = profile_manager.memory_system
    logger = get_logger("neurosheaf.profiling.validation")
    
    # Test with a known memory allocation
    test_data_mb = 100.0  # 100MB
    test_data_size = int(test_data_mb * 1024 * 1024 / 8)  # 8 bytes per float64
    
    results = {}
    
    try:
        # Measure before allocation
        before_measurements = memory_system._measure_all_backends()
        
        # Allocate test data
        import numpy as np
        test_array = np.random.random(test_data_size)
        
        # Measure after allocation
        after_measurements = memory_system._measure_all_backends()
        
        # Calculate detected memory usage
        detected_usage = {}
        for backend in memory_system.available_backends:
            if backend in before_measurements and backend in after_measurements:
                before = before_measurements[backend]
                after = after_measurements[backend]
                
                if before.is_valid and after.is_valid:
                    usage_mb = after.current_mb - before.current_mb
                    detected_usage[backend] = usage_mb
                else:
                    detected_usage[backend] = None
        
        # Validate against expected usage
        results = {
            "expected_usage_mb": test_data_mb,
            "detected_usage": detected_usage,
            "precision_analysis": {},
            "overall_precision": "good"
        }
        
        # Analyze precision for each backend
        good_measurements = 0
        total_measurements = 0
        
        for backend, usage in detected_usage.items():
            if usage is not None:
                total_measurements += 1
                error_percent = abs(usage - test_data_mb) / test_data_mb * 100
                
                if error_percent < 10:  # Within 10%
                    precision = "good"
                    good_measurements += 1
                elif error_percent < 30:  # Within 30%
                    precision = "acceptable"
                else:
                    precision = "poor"
                
                results["precision_analysis"][backend.value] = {
                    "detected_mb": usage,
                    "error_percent": error_percent,
                    "precision": precision
                }
        
        # Overall precision assessment
        if total_measurements == 0:
            results["overall_precision"] = "error"
        elif good_measurements / total_measurements >= 0.5:
            results["overall_precision"] = "good"
        else:
            results["overall_precision"] = "poor"
        
        # Clean up
        del test_array
        
    except Exception as e:
        results = {
            "error": str(e),
            "overall_precision": "error"
        }
    
    logger.info(f"Memory measurement validation: {results.get('overall_precision', 'unknown')}")
    
    return results
"""Memory monitoring utilities for the neurosheaf package."""

import psutil
import torch
import platform
from typing import Dict, Optional

from .logging import setup_logger


logger = setup_logger(__name__)


class MemoryMonitor:
    """Monitor memory usage during computation.
    
    Provides utilities to track memory consumption and available memory
    across different platforms (CPU, CUDA, MPS).
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory monitor.
        
        Args:
            device: Device to monitor (auto-detected if None)
        """
        self.device = device or self._detect_device()
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = platform.processor() == "arm"
        
        logger.info(f"Initialized MemoryMonitor for device: {self.device}")
    
    def _detect_device(self) -> torch.device:
        """Auto-detect the compute device."""
        if platform.system() == "Darwin":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information in MB.
        
        Returns:
            Dictionary with memory statistics:
                - available_mb: Available memory
                - used_mb: Used memory
                - total_mb: Total memory
                - percent: Percentage used
        """
        if self.device.type == 'cuda':
            # CUDA memory
            used_bytes = torch.cuda.memory_allocated(self.device)
            reserved_bytes = torch.cuda.memory_reserved(self.device)
            total_bytes = torch.cuda.get_device_properties(self.device).total_memory
            
            return {
                'available_mb': (total_bytes - used_bytes) / (1024**2),
                'used_mb': used_bytes / (1024**2),
                'reserved_mb': reserved_bytes / (1024**2),
                'total_mb': total_bytes / (1024**2),
                'percent': (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
            }
        
        elif self.device.type == 'mps':
            # MPS (Apple Silicon) - use system memory as unified memory
            vm = psutil.virtual_memory()
            
            # Try to get MPS-specific memory if available
            mps_used_mb = 0
            try:
                if hasattr(torch.mps, 'current_allocated_memory'):
                    mps_used_mb = torch.mps.current_allocated_memory() / (1024**2)
            except:
                pass
            
            return {
                'available_mb': vm.available / (1024**2),
                'used_mb': vm.used / (1024**2),
                'mps_used_mb': mps_used_mb,
                'total_mb': vm.total / (1024**2),
                'percent': vm.percent
            }
        
        else:
            # CPU memory
            vm = psutil.virtual_memory()
            return {
                'available_mb': vm.available / (1024**2),
                'used_mb': vm.used / (1024**2),
                'total_mb': vm.total / (1024**2),
                'percent': vm.percent
            }
    
    def available_mb(self) -> float:
        """Get available memory in MB."""
        return self.get_memory_info()['available_mb']
    
    def used_mb(self) -> float:
        """Get used memory in MB."""
        return self.get_memory_info()['used_mb']
    
    def check_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if sufficient memory is available
        """
        available = self.available_mb()
        # Leave 20% buffer for system stability
        return available * 0.8 >= required_mb
    
    def clear_cache(self) -> None:
        """Clear device memory cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared CUDA cache")
        elif self.device.type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.info("Cleared MPS cache")
        else:
            # CPU - trigger garbage collection
            import gc
            gc.collect()
            logger.info("Triggered garbage collection")
    
    def log_memory_stats(self, prefix: str = "") -> None:
        """Log current memory statistics.
        
        Args:
            prefix: Prefix for log message
        """
        info = self.get_memory_info()
        
        if prefix:
            prefix = f"{prefix}: "
        
        logger.info(
            f"{prefix}Memory - Available: {info['available_mb']:.1f}MB, "
            f"Used: {info['used_mb']:.1f}MB ({info['percent']:.1f}%)"
        )
    
    def estimate_tensor_memory(self, shape: tuple, dtype: torch.dtype = torch.float32) -> float:
        """Estimate memory usage for a tensor in MB.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Memory usage in MB
        """
        # Get bytes per element for different dtypes
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.int16: 2,
            torch.int8: 1,
            torch.uint8: 1,
            torch.bool: 1
        }
        
        bytes_per_element = dtype_sizes.get(dtype, 4)  # Default to float32
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        return (total_elements * bytes_per_element) / (1024**2)
    
    def can_fit_tensors(self, *tensor_shapes: tuple, dtype: torch.dtype = torch.float32) -> bool:
        """Check if multiple tensors can fit in memory.
        
        Args:
            *tensor_shapes: Variable number of tensor shapes
            dtype: Tensor data type
            
        Returns:
            True if all tensors can fit in memory
        """
        total_mb = sum(self.estimate_tensor_memory(shape, dtype) for shape in tensor_shapes)
        return self.check_memory_available(total_mb)
    
    def optimal_chunk_size(self, total_size: int, min_chunk: int = 100, max_chunk: int = 10000) -> int:
        """Determine optimal chunk size for memory-efficient processing.
        
        Args:
            total_size: Total number of elements to process
            min_chunk: Minimum chunk size
            max_chunk: Maximum chunk size
            
        Returns:
            Optimal chunk size
        """
        available_mb = self.available_mb() * 0.8  # Use 80% of available memory
        
        # Estimate memory per element (assuming float32 matrix operations)
        bytes_per_element = 4 * total_size  # Approximate for matrix operations
        elements_per_mb = (1024**2) / bytes_per_element
        
        # Calculate chunk size that fits in available memory
        optimal_chunk = int(available_mb * elements_per_mb)
        
        # Apply bounds
        chunk_size = max(min_chunk, min(optimal_chunk, max_chunk))
        
        logger.info(f"Determined optimal chunk size: {chunk_size} (total: {total_size})")
        return chunk_size
"""Main API for Neurosheaf analysis.

This module provides the high-level interface for conducting neural network
similarity analysis using persistent sheaf Laplacians.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import platform

from .utils.logging import setup_logger
from .utils.exceptions import ValidationError, ComputationError, ArchitectureError
from .utils.profiling import profile_memory, profile_time


class NeurosheafAnalyzer:
    """Main interface for neural network similarity analysis.
    
    This class provides a high-level API for analyzing neural networks using
    persistent sheaf Laplacians. It automatically handles device detection,
    memory management, and provides Mac-specific optimizations.
    
    Examples:
        >>> analyzer = NeurosheafAnalyzer()
        >>> model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU())
        >>> data = torch.randn(1000, 100)
        >>> results = analyzer.analyze(model, data)
        >>> cka_matrix = results['cka_matrix']
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        memory_limit_gb: float = 8.0,
        enable_profiling: bool = True,
        log_level: str = "INFO"
    ):
        """Initialize the Neurosheaf analyzer.
        
        Args:
            device: Device to use ('cpu', 'mps', 'cuda', or None for auto-detection)
            memory_limit_gb: Memory limit in GB for computations
            enable_profiling: Whether to enable performance profiling
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.logger = setup_logger("neurosheaf.analyzer", level=log_level)
        self.device = self._detect_device(device)
        self.memory_limit_gb = memory_limit_gb
        self.enable_profiling = enable_profiling
        
        # Mac-specific initialization
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = platform.processor() == "arm"
        
        self.logger.info(f"Initialized NeurosheafAnalyzer on {self.device}")
        if self.is_mac:
            self.logger.info(f"Mac detected: Apple Silicon = {self.is_apple_silicon}")
    
    def _detect_device(self, device: Optional[str] = None) -> torch.device:
        """Detect the optimal device for computation.
        
        Args:
            device: Optional device specification
            
        Returns:
            torch.device: The selected device
        """
        if device is not None:
            return torch.device(device)
        
        # Mac-specific device detection
        if platform.system() == "Darwin":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        
        # Other platforms
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        return torch.device("cpu")
    
    def analyze(
        self,
        model: nn.Module,
        data: torch.Tensor,
        batch_size: Optional[int] = None,
        layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform complete neurosheaf analysis.
        
        This is a placeholder implementation for Phase 1 Week 2.
        The full implementation will be completed in Phase 2.
        
        Args:
            model: PyTorch neural network model
            data: Input data tensor
            batch_size: Batch size for processing (auto-detected if None)
            layers: Specific layers to analyze (all if None)
            
        Returns:
            Dictionary containing analysis results:
                - 'cka_matrix': CKA similarity matrix
                - 'activations': Layer activations
                - 'device_info': Device and hardware information
                - 'performance': Performance metrics
                
        Raises:
            ValidationError: If input validation fails
            ComputationError: If analysis computation fails
            ArchitectureError: If model architecture is unsupported
        """
        self.logger.info("Starting neurosheaf analysis...")
        
        # Validate inputs
        self._validate_inputs(model, data)
        
        # Move model and data to device
        model = model.to(self.device)
        data = data.to(self.device)
        
        # Placeholder results for Phase 1 Week 2
        results = {
            'status': 'placeholder_implementation',
            'phase': 'Phase 1 Week 2 - Baseline Setup',
            'device_info': self._get_device_info(),
            'data_shape': data.shape,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'memory_info': self._get_memory_info()
        }
        
        self.logger.info("Analysis completed (placeholder implementation)")
        return results
    
    def _validate_inputs(self, model: nn.Module, data: torch.Tensor) -> None:
        """Validate input parameters.
        
        Args:
            model: PyTorch model to validate
            data: Input data to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(model, nn.Module):
            raise ValidationError("Model must be a PyTorch nn.Module")
        
        if not isinstance(data, torch.Tensor):
            raise ValidationError("Data must be a torch.Tensor")
        
        if data.dim() < 2:
            raise ValidationError("Data must have at least 2 dimensions")
        
        if data.shape[0] == 0:
            raise ValidationError("Data cannot be empty")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device and hardware information.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'device': str(self.device),
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__
        }
        
        # Mac-specific information
        if self.is_mac:
            info['is_apple_silicon'] = self.is_apple_silicon
            if self.device.type == 'mps':
                info['mps_available'] = torch.backends.mps.is_available()
        
        # CUDA information
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['cuda_device_count'] = torch.cuda.device_count()
            if self.device.type == 'cuda':
                info['cuda_device_name'] = torch.cuda.get_device_name()
        
        return info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information.
        
        Returns:
            Dictionary with memory information
        """
        import psutil
        
        # System memory
        memory_info = {
            'system_total_gb': psutil.virtual_memory().total / (1024**3),
            'system_available_gb': psutil.virtual_memory().available / (1024**3),
            'system_used_gb': psutil.virtual_memory().used / (1024**3),
            'system_percent': psutil.virtual_memory().percent
        }
        
        # Device-specific memory
        if self.device.type == 'cuda' and torch.cuda.is_available():
            memory_info['cuda_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_info['cuda_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['cuda_cached_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        elif self.device.type == 'mps' and self.is_apple_silicon:
            # Apple Silicon uses unified memory
            memory_info['unified_memory'] = True
            memory_info['mps_allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
        
        return memory_info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.
        
        Returns:
            Dictionary with system and hardware information
        """
        return {
            'device_info': self._get_device_info(),
            'memory_info': self._get_memory_info(),
            'analyzer_config': {
                'device': str(self.device),
                'memory_limit_gb': self.memory_limit_gb,
                'enable_profiling': self.enable_profiling,
                'is_mac': self.is_mac,
                'is_apple_silicon': self.is_apple_silicon
            }
        }
    
    def profile_memory_usage(self, model: nn.Module, data: torch.Tensor) -> Dict[str, Any]:
        """Profile memory usage for the given model and data.
        
        Args:
            model: PyTorch model to profile
            data: Input data for profiling
            
        Returns:
            Dictionary with memory profiling results
        """
        if not self.enable_profiling:
            self.logger.warning("Profiling is disabled")
            return {'status': 'disabled'}
        
        @profile_memory()
        def _profile_forward_pass():
            model.eval()
            with torch.no_grad():
                return model(data)
        
        # Run profiling
        output = _profile_forward_pass()
        
        return {
            'status': 'completed',
            'output_shape': output.shape,
            'memory_info': self._get_memory_info(),
            'device_info': self._get_device_info()
        }
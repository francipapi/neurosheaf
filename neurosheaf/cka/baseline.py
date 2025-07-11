"""Baseline CKA implementation for profiling and benchmarking.

This module provides a memory-intensive baseline implementation specifically
designed for establishing performance baselines and identifying optimization
opportunities. This is used for Phase 1 Week 2 baseline measurements.
"""

import time
import psutil
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ..utils.device import detect_optimal_device, get_device_info, clear_device_cache
from ..utils.exceptions import ValidationError, ComputationError, MemoryError
from ..utils.logging import setup_logger
from ..utils.profiling import profile_memory, profile_time


class BaselineCKA:
    """Baseline CKA implementation for profiling and benchmarking.
    
    This implementation is intentionally memory-intensive to establish
    baseline performance metrics. It serves as the reference point for
    optimization in Phase 2.
    
    Key characteristics:
    - Memory-intensive (stores all intermediate matrices)
    - Unoptimized computation paths
    - Comprehensive profiling and measurement
    - Mac-specific optimizations for accurate baseline measurement
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        store_intermediates: bool = True,
        enable_detailed_profiling: bool = True
    ):
        """Initialize the baseline CKA computer.
        
        Args:
            device: Device to use for computation (auto-detected if None)
            store_intermediates: Whether to store intermediate matrices for analysis
            enable_detailed_profiling: Whether to enable detailed profiling
        """
        self.logger = setup_logger("neurosheaf.cka.baseline")
        self.device = detect_optimal_device(device)
        self.store_intermediates = store_intermediates
        self.enable_detailed_profiling = enable_detailed_profiling
        
        # Device information
        self.device_info = get_device_info()
        self.is_mac = self.device_info['is_mac']
        self.is_apple_silicon = self.device_info['is_apple_silicon']
        
        # Storage for intermediate results and profiling data
        self.intermediates = {}
        self.profiling_data = {}
        
        self.logger.info(f"Initialized BaselineCKA on {self.device}")
        if self.is_mac:
            self.logger.info(f"Mac baseline profiling: Apple Silicon = {self.is_apple_silicon}")
    
    
    @profile_memory(memory_threshold_mb=1000.0, log_results=True)
    def compute_baseline_cka_matrix(
        self,
        activations: Dict[str, torch.Tensor],
        target_memory_gb: float = 20.0  # 20 GB target
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """Compute CKA matrix using baseline (memory-intensive) approach.
        
        This method is designed to consume significant memory to establish
        baseline performance metrics. It aims for the 20GB target.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
            target_memory_gb: Target memory usage in GB (default: 20GB)
            
        Returns:
            Tuple of (CKA matrix, profiling data)
        """
        self.logger.info("Starting baseline CKA matrix computation...")
        
        # Validate inputs
        if not activations:
            raise ValidationError("Activations dictionary cannot be empty")
        
        # Record initial state
        initial_memory = self._get_memory_usage()
        start_time = time.time()
        
        layer_names = list(activations.keys())
        n_layers = len(layer_names)
        
        if n_layers < 2:
            raise ValidationError("Need at least 2 layers to compute CKA matrix")
        
        # Move all activations to device (memory-intensive)
        device_activations = {}
        for name, activation in activations.items():
            device_activations[name] = activation.to(self.device)
            if self.store_intermediates:
                self.intermediates[f"activation_{name}"] = activation.clone()
        
        # Pre-compute all Gram matrices (memory-intensive)
        gram_matrices = {}
        for name, activation in device_activations.items():
            self.logger.debug(f"Computing Gram matrix for {name}: {activation.shape}")
            gram_matrix = self._compute_gram_matrix_baseline(activation)
            gram_matrices[name] = gram_matrix
            
            if self.store_intermediates:
                self.intermediates[f"gram_{name}"] = gram_matrix.clone()
        
        # Compute CKA matrix (memory-intensive)
        cka_matrix = torch.zeros(n_layers, n_layers, device=self.device)
        
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                if i <= j:  # Compute upper triangle
                    cka_value = self._compute_cka_from_grams_baseline(
                        gram_matrices[layer_i],
                        gram_matrices[layer_j],
                        device_activations[layer_i].shape[0]
                    )
                    cka_matrix[i, j] = cka_value
                    cka_matrix[j, i] = cka_value  # Symmetry
        
        # Record final measurements
        final_memory = self._get_memory_usage()
        end_time = time.time()
        
        # Compile profiling data
        profiling_data = {
            'initial_memory_gb': initial_memory,
            'final_memory_gb': final_memory,
            'peak_memory_gb': final_memory,  # Approximation
            'memory_increase_gb': final_memory - initial_memory,
            'computation_time_seconds': end_time - start_time,
            'n_layers': n_layers,
            'total_parameters': sum(act.numel() for act in activations.values()),
            'device': str(self.device),
            'device_info': self.device_info
        }
        
        self.profiling_data = profiling_data
        
        self.logger.info(f"Baseline CKA computation completed:")
        self.logger.info(f"  Memory usage: {profiling_data['memory_increase_gb']:.2f}GB")
        self.logger.info(f"  Computation time: {profiling_data['computation_time_seconds']:.2f}s")
        
        return cka_matrix, profiling_data
    
    def _compute_gram_matrix_baseline(self, activation: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix using baseline (memory-intensive) approach.
        
        Args:
            activation: Activation tensor [n_samples, n_features]
            
        Returns:
            torch.Tensor: Gram matrix [n_samples, n_samples]
        """
        # Memory-intensive: store full activation matrix
        n_samples = activation.shape[0]
        
        # Compute Gram matrix K = X @ X.T (no pre-centering)
        gram_matrix = activation @ activation.T
        
        # Additional memory-intensive operations for baseline
        if self.store_intermediates:
            # Store additional intermediate matrices
            self.intermediates[f"gram_diagonal"] = torch.diag(gram_matrix).clone()
            self.intermediates[f"gram_trace"] = torch.trace(gram_matrix).clone()
        
        return gram_matrix
    
    def _compute_cka_from_grams_baseline(
        self,
        gram_x: torch.Tensor,
        gram_y: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """Compute CKA from pre-computed Gram matrices (baseline approach).
        
        Args:
            gram_x: Gram matrix for first activation
            gram_y: Gram matrix for second activation
            n_samples: Number of samples
            
        Returns:
            torch.Tensor: CKA value
        """
        # Create centering matrix H = I - (1/n) * ones
        H = torch.eye(n_samples, device=self.device) - torch.ones(n_samples, n_samples, device=self.device) / n_samples
        
        # Compute HSIC components (memory-intensive)
        hsic_xy = torch.trace(gram_x @ gram_y @ H) / (n_samples * n_samples)
        hsic_xx = torch.trace(gram_x @ gram_x @ H) / (n_samples * n_samples)
        hsic_yy = torch.trace(gram_y @ gram_y @ H) / (n_samples * n_samples)
        
        # Store intermediate HSIC values
        if self.store_intermediates:
            self.intermediates[f"hsic_xy"] = hsic_xy.clone()
            self.intermediates[f"hsic_xx"] = hsic_xx.clone()
            self.intermediates[f"hsic_yy"] = hsic_yy.clone()
        
        # Compute CKA
        denominator = torch.sqrt(hsic_xx * hsic_yy)
        
        if denominator < 1e-8:
            return torch.tensor(0.0, device=self.device)
        
        cka_value = hsic_xy / denominator
        return cka_value
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB.
        
        Returns:
            float: Memory usage in GB
        """
        if self.is_mac and self.is_apple_silicon:
            # Apple Silicon unified memory
            system_memory = psutil.virtual_memory().used / (1024**3)
            if self.device.type == 'mps':
                try:
                    mps_memory = torch.mps.current_allocated_memory() / (1024**3)
                    return system_memory  # Unified memory, so system memory is more accurate
                except:
                    return system_memory
            return system_memory
        
        elif self.device.type == 'cuda':
            # CUDA memory
            return torch.cuda.memory_allocated() / (1024**3)
        
        else:
            # CPU memory
            return psutil.virtual_memory().used / (1024**3)
    
    def profile_resnet50_baseline(
        self,
        batch_size: int = 1000,
        num_layers: int = 50,
        feature_dims: List[int] = None
    ) -> Dict[str, any]:
        """Profile ResNet50-like model to establish 1.5TB baseline.
        
        Args:
            batch_size: Number of samples
            num_layers: Number of layers to simulate
            feature_dims: Feature dimensions for each layer
            
        Returns:
            Dictionary with profiling results
        """
        if feature_dims is None:
            # ResNet50-like feature dimensions
            feature_dims = [64, 256, 512, 1024, 2048] * (num_layers // 5)
            feature_dims = feature_dims[:num_layers]
        
        self.logger.info(f"Profiling ResNet50-like model: {num_layers} layers, {batch_size} samples")
        
        # Generate synthetic activations
        activations = {}
        for i, dim in enumerate(feature_dims):
            layer_name = f"layer_{i:02d}"
            # Create large activation tensors to reach memory targets
            activation = torch.randn(batch_size, dim, device=self.device)
            activations[layer_name] = activation
            
            self.logger.debug(f"Generated {layer_name}: {activation.shape}")
        
        # Compute baseline CKA matrix
        cka_matrix, profiling_data = self.compute_baseline_cka_matrix(activations)
        
        # Add ResNet50-specific profiling data
        profiling_data['model_type'] = 'resnet50_baseline'
        profiling_data['batch_size'] = batch_size
        profiling_data['num_layers'] = num_layers
        profiling_data['feature_dims'] = feature_dims
        profiling_data['cka_matrix_shape'] = cka_matrix.shape
        
        return profiling_data
    
    def generate_baseline_report(self) -> str:
        """Generate a comprehensive baseline performance report.
        
        Returns:
            str: Formatted baseline report
        """
        if not self.profiling_data:
            return "No profiling data available. Run profiling first."
        
        data = self.profiling_data
        
        device_info = data['device_info']
        report = [
            "Neurosheaf Baseline CKA Performance Report",
            "=" * 50,
            "",
            f"Device: {data['device']}",
            f"Platform: {device_info['platform']}",
            f"Mac System: {device_info['is_mac']}",
            f"Apple Silicon: {device_info['is_apple_silicon']}",
            "",
            "Memory Usage:",
            f"  Initial: {data['initial_memory_gb']:.2f} GB",
            f"  Final: {data['final_memory_gb']:.2f} GB",
            f"  Increase: {data['memory_increase_gb']:.2f} GB",
            f"  Target (20GB): {data['memory_increase_gb']/20:.1%} of target",
            "",
            "Computation Performance:",
            f"  Total Time: {data['computation_time_seconds']:.2f} seconds",
            f"  Layers: {data['n_layers']}",
            f"  Parameters: {data['total_parameters']:,}",
            "",
            "Bottleneck Analysis:",
            f"  Memory per layer: {data['memory_increase_gb']/data['n_layers']:.3f} GB",
            f"  Time per layer: {data['computation_time_seconds']/data['n_layers']:.3f} seconds",
            f"  Memory per parameter: {data['memory_increase_gb']/data['total_parameters']*1e9:.2f} GB/billion params",
            "",
            "Optimization Targets:",
            "  1. Gram matrix computation (O(n²) memory)",
            "  2. Matrix multiplication optimization",
            "  3. Memory-efficient HSIC computation",
            "  4. Batch processing for large networks",
            "",
            "Next Steps:",
            "  - Implement Nyström approximation (Phase 2)",
            "  - Add adaptive sampling (Phase 2)",
            "  - Optimize for sparse operations (Phase 2)",
            ""
        ]
        
        return "\n".join(report)
    
    def clear_intermediates(self):
        """Clear stored intermediate results to free memory."""
        self.intermediates.clear()
        clear_device_cache(self.device)
    
    def get_memory_breakdown(self) -> Dict[str, float]:
        """Get detailed memory breakdown by component.
        
        Returns:
            Dictionary with memory usage by component
        """
        breakdown = {}
        
        for name, tensor in self.intermediates.items():
            if isinstance(tensor, torch.Tensor):
                memory_gb = tensor.element_size() * tensor.numel() / (1024**3)
                breakdown[name] = memory_gb
        
        return breakdown
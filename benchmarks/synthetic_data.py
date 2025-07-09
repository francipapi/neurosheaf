"""Synthetic dataset generation for Mac-optimized benchmarking.

This module generates synthetic neural network activations that mimic real
network behavior for baseline performance measurements and benchmarking.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import platform
import math

from neurosheaf.utils.logging import setup_logger
from neurosheaf.utils.exceptions import ValidationError


class SyntheticDataGenerator:
    """Generator for synthetic neural network activation data.
    
    This class creates realistic synthetic activations that mimic the
    statistical properties of real neural network layers. It's optimized
    for Mac hardware and includes Apple Silicon MPS support.
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize the synthetic data generator.
        
        Args:
            device: Device to use for generation (auto-detected if None)
            seed: Random seed for reproducibility
            dtype: Data type for generated tensors
        """
        self.logger = setup_logger("neurosheaf.benchmarks.synthetic")
        self.device = self._detect_device(device)
        self.seed = seed
        self.dtype = dtype
        
        # Mac-specific initialization
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = platform.processor() == "arm"
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.logger.info(f"Initialized SyntheticDataGenerator on {self.device}")
        if self.is_mac:
            self.logger.info(f"Mac optimization enabled: Apple Silicon = {self.is_apple_silicon}")
    
    def _detect_device(self, device: Optional[Union[str, torch.device]] = None) -> torch.device:
        """Detect optimal device for Mac and other platforms."""
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
    
    def generate_resnet50_activations(
        self,
        batch_size: int = 1000,
        scale_factor: float = 1.0,
        realistic_distributions: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic activations mimicking ResNet50 architecture.
        
        Args:
            batch_size: Number of samples
            scale_factor: Scale factor for memory usage (1.0 = normal, 10.0 = 10x memory)
            realistic_distributions: Whether to use realistic activation distributions
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self.logger.info(f"Generating ResNet50 activations: batch_size={batch_size}, scale={scale_factor}")
        
        # ResNet50 layer specifications
        layer_specs = [
            ("conv1", 64, 112, 112),
            ("layer1.0", 256, 56, 56),
            ("layer1.1", 256, 56, 56),
            ("layer1.2", 256, 56, 56),
            ("layer2.0", 512, 28, 28),
            ("layer2.1", 512, 28, 28),
            ("layer2.2", 512, 28, 28),
            ("layer2.3", 512, 28, 28),
            ("layer3.0", 1024, 14, 14),
            ("layer3.1", 1024, 14, 14),
            ("layer3.2", 1024, 14, 14),
            ("layer3.3", 1024, 14, 14),
            ("layer3.4", 1024, 14, 14),
            ("layer3.5", 1024, 14, 14),
            ("layer4.0", 2048, 7, 7),
            ("layer4.1", 2048, 7, 7),
            ("layer4.2", 2048, 7, 7),
            ("avgpool", 2048, 1, 1),
            ("fc", 1000, 1, 1)
        ]
        
        activations = {}
        
        for layer_name, channels, height, width in layer_specs:
            # Scale dimensions for memory testing
            scaled_batch = int(batch_size * scale_factor)
            scaled_channels = int(channels * scale_factor)
            
            # Generate activation tensor
            if realistic_distributions:
                activation = self._generate_realistic_activation(
                    scaled_batch, scaled_channels, height, width, layer_name
                )
            else:
                activation = self._generate_gaussian_activation(
                    scaled_batch, scaled_channels, height, width
                )
            
            activations[layer_name] = activation
            
            memory_mb = activation.element_size() * activation.numel() / (1024**2)
            self.logger.debug(f"Generated {layer_name}: {activation.shape} ({memory_mb:.1f}MB)")
        
        total_memory_gb = sum(
            act.element_size() * act.numel() for act in activations.values()
        ) / (1024**3)
        
        self.logger.info(f"Total activations memory: {total_memory_gb:.2f}GB")
        
        return activations
    
    def _generate_realistic_activation(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        layer_name: str
    ) -> torch.Tensor:
        """Generate realistic activation tensor with proper distributions.
        
        Args:
            batch_size: Batch size
            channels: Number of channels
            height: Height dimension
            width: Width dimension
            layer_name: Layer name for distribution selection
            
        Returns:
            torch.Tensor: Realistic activation tensor
        """
        shape = (batch_size, channels * height * width)
        
        # Different distributions for different layer types
        if "conv" in layer_name.lower():
            # Convolutional layers: ReLU-like distribution
            activation = torch.randn(shape, device=self.device, dtype=self.dtype)
            activation = torch.relu(activation)  # ReLU activation
            
        elif "layer" in layer_name.lower():
            # ResNet blocks: Mix of positive and negative with skip connections
            base = torch.randn(shape, device=self.device, dtype=self.dtype)
            skip = torch.randn(shape, device=self.device, dtype=self.dtype) * 0.1
            activation = torch.relu(base + skip)
            
        elif "avgpool" in layer_name.lower():
            # Average pooling: Positive values, smaller variance
            activation = torch.abs(torch.randn(shape, device=self.device, dtype=self.dtype)) * 0.5
            
        elif "fc" in layer_name.lower():
            # Fully connected: Pre-softmax logits
            activation = torch.randn(shape, device=self.device, dtype=self.dtype) * 2.0
            
        else:
            # Default: Standard normal
            activation = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        return activation
    
    def _generate_gaussian_activation(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Generate simple Gaussian activation tensor.
        
        Args:
            batch_size: Batch size
            channels: Number of channels  
            height: Height dimension
            width: Width dimension
            
        Returns:
            torch.Tensor: Gaussian activation tensor
        """
        shape = (batch_size, channels * height * width)
        return torch.randn(shape, device=self.device, dtype=self.dtype)
    
    def generate_transformer_activations(
        self,
        batch_size: int = 1000,
        seq_length: int = 512,
        scale_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic activations mimicking Transformer architecture.
        
        Args:
            batch_size: Number of samples
            seq_length: Sequence length
            scale_factor: Scale factor for memory usage
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self.logger.info(f"Generating Transformer activations: batch_size={batch_size}, seq_length={seq_length}")
        
        # Transformer layer specifications
        layer_specs = [
            ("embedding", 768),
            ("encoder.0.attention", 768),
            ("encoder.0.feedforward", 3072),
            ("encoder.1.attention", 768),
            ("encoder.1.feedforward", 3072),
            ("encoder.2.attention", 768),
            ("encoder.2.feedforward", 3072),
            ("encoder.3.attention", 768),
            ("encoder.3.feedforward", 3072),
            ("encoder.4.attention", 768),
            ("encoder.4.feedforward", 3072),
            ("encoder.5.attention", 768),
            ("encoder.5.feedforward", 3072),
            ("pooler", 768),
            ("classifier", 768)
        ]
        
        activations = {}
        
        for layer_name, hidden_size in layer_specs:
            # Scale dimensions
            scaled_batch = int(batch_size * scale_factor)
            scaled_hidden = int(hidden_size * scale_factor)
            scaled_seq = int(seq_length * scale_factor)
            
            # Generate activation tensor
            shape = (scaled_batch, scaled_seq, scaled_hidden)
            
            if "attention" in layer_name:
                # Attention patterns: softmax-like distribution
                activation = torch.randn(shape, device=self.device, dtype=self.dtype)
                activation = torch.softmax(activation, dim=-1)
                activation = activation.view(scaled_batch, scaled_seq * scaled_hidden)
            elif "feedforward" in layer_name:
                # Feedforward: GELU-like distribution
                activation = torch.randn(shape, device=self.device, dtype=self.dtype)
                activation = torch.gelu(activation)
                activation = activation.view(scaled_batch, scaled_seq * scaled_hidden)
            else:
                # Default: reshape to 2D
                activation = torch.randn(shape, device=self.device, dtype=self.dtype)
                activation = activation.view(scaled_batch, scaled_seq * scaled_hidden)
            
            activations[layer_name] = activation
            
            memory_mb = activation.element_size() * activation.numel() / (1024**2)
            self.logger.debug(f"Generated {layer_name}: {activation.shape} ({memory_mb:.1f}MB)")
        
        total_memory_gb = sum(
            act.element_size() * act.numel() for act in activations.values()
        ) / (1024**3)
        
        self.logger.info(f"Total activations memory: {total_memory_gb:.2f}GB")
        
        return activations
    
    def generate_memory_stress_test(
        self,
        target_memory_gb: float = 20.0,  # 20 GB
        batch_size: int = 1000,
        num_layers: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic data to stress test memory usage.
        
        Args:
            target_memory_gb: Target memory usage in GB
            batch_size: Number of samples
            num_layers: Number of layers to generate
            
        Returns:
            Dictionary with memory-intensive activations
        """
        self.logger.info(f"Generating memory stress test: target={target_memory_gb:.1f}GB")
        
        # Calculate required dimensions to reach target memory
        bytes_per_float = 4  # float32
        target_elements = int(target_memory_gb * (1024**3) / bytes_per_float)
        elements_per_layer = target_elements // num_layers
        
        # Calculate feature dimension
        feature_dim = int(math.sqrt(elements_per_layer / batch_size))
        
        self.logger.info(f"Using feature_dim={feature_dim} for {num_layers} layers")
        
        activations = {}
        
        for i in range(num_layers):
            layer_name = f"stress_layer_{i:03d}"
            
            # Generate large activation tensor
            activation = torch.randn(
                batch_size, feature_dim, 
                device=self.device, 
                dtype=self.dtype
            )
            
            activations[layer_name] = activation
            
            memory_gb = activation.element_size() * activation.numel() / (1024**3)
            self.logger.debug(f"Generated {layer_name}: {activation.shape} ({memory_gb:.2f}GB)")
        
        total_memory_gb = sum(
            act.element_size() * act.numel() for act in activations.values()
        ) / (1024**3)
        
        self.logger.info(f"Total stress test memory: {total_memory_gb:.2f}GB")
        
        return activations
    
    def generate_scaling_test_data(
        self,
        sizes: List[int],
        feature_dim: int = 512,
        architecture: str = "resnet"
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Generate data for scaling analysis.
        
        Args:
            sizes: List of batch sizes to generate
            feature_dim: Feature dimension for each layer
            architecture: Architecture type ('resnet' or 'transformer')
            
        Returns:
            Dictionary mapping sizes to activation dictionaries
        """
        self.logger.info(f"Generating scaling test data for sizes: {sizes}")
        
        scaling_data = {}
        
        for size in sizes:
            self.logger.info(f"Generating data for size {size}...")
            
            if architecture == "resnet":
                activations = self.generate_resnet50_activations(
                    batch_size=size, 
                    scale_factor=feature_dim/512
                )
            elif architecture == "transformer":
                activations = self.generate_transformer_activations(
                    batch_size=size,
                    seq_length=feature_dim
                )
            else:
                raise ValidationError(f"Unknown architecture: {architecture}")
            
            scaling_data[size] = activations
        
        return scaling_data
    
    def estimate_memory_usage(
        self,
        batch_size: int,
        feature_dims: List[int],
        include_intermediates: bool = True
    ) -> Dict[str, float]:
        """Estimate memory usage for given dimensions.
        
        Args:
            batch_size: Number of samples
            feature_dims: List of feature dimensions
            include_intermediates: Whether to include intermediate matrices
            
        Returns:
            Dictionary with memory usage estimates
        """
        bytes_per_float = 4  # float32
        
        # Activation memory
        activation_memory = sum(
            batch_size * dim * bytes_per_float 
            for dim in feature_dims
        ) / (1024**3)
        
        # Gram matrix memory (n x n for each layer)
        gram_memory = sum(
            batch_size * batch_size * bytes_per_float 
            for _ in feature_dims
        ) / (1024**3)
        
        # CKA matrix memory
        n_layers = len(feature_dims)
        cka_memory = n_layers * n_layers * bytes_per_float / (1024**3)
        
        estimates = {
            'activation_memory_gb': activation_memory,
            'gram_memory_gb': gram_memory,
            'cka_memory_gb': cka_memory,
            'total_memory_gb': activation_memory + gram_memory + cka_memory
        }
        
        if include_intermediates:
            # Additional memory for intermediate computations
            intermediate_memory = gram_memory * 2  # HSIC computations
            estimates['intermediate_memory_gb'] = intermediate_memory
            estimates['total_memory_gb'] += intermediate_memory
        
        return estimates
#!/usr/bin/env python3
"""Memory scaling analysis for Mac-optimized Neurosheaf CKA computation.

This script analyzes how memory usage scales with different input sizes
and provides insights for optimization in Phase 2.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add neurosheaf to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import platform

from neurosheaf.cka.baseline import BaselineCKA
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.utils.logging import setup_logger
from neurosheaf.utils.profiling import get_mac_memory_info
from benchmarks.synthetic_data import SyntheticDataGenerator


class MemoryScalingAnalyzer:
    """Analyzer for memory scaling behavior on Mac systems."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        log_level: str = "INFO",
        output_dir: str = "scaling_results"
    ):
        """Initialize the memory scaling analyzer.
        
        Args:
            device: Device to use (auto-detected if None)
            log_level: Logging level
            output_dir: Directory to save results
        """
        self.logger = setup_logger("neurosheaf.benchmarks.scaling", level=log_level)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.baseline_cka = BaselineCKA(device=device)
        self.debiased_cka = DebiasedCKA(device=device)
        self.data_generator = SyntheticDataGenerator(device=device)
        
        # Mac-specific initialization
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = platform.processor() == "arm"
        self.device = self.baseline_cka.device
        
        self.logger.info(f"Initialized MemoryScalingAnalyzer on {self.device}")
    
    def analyze_batch_size_scaling(
        self,
        batch_sizes: List[int] = None,
        feature_dim: int = 512,
        num_layers: int = 10
    ) -> Dict[str, any]:
        """Analyze memory scaling with batch size.
        
        Args:
            batch_sizes: List of batch sizes to test
            feature_dim: Feature dimension for synthetic data
            num_layers: Number of layers to generate
            
        Returns:
            Dictionary with scaling analysis results
        """
        if batch_sizes is None:
            batch_sizes = [50, 100, 200, 500, 1000, 2000]
        
        self.logger.info(f"Analyzing batch size scaling: {batch_sizes}")
        
        results = {
            'experiment': 'batch_size_scaling',
            'batch_sizes': batch_sizes,
            'feature_dim': feature_dim,
            'num_layers': num_layers,
            'measurements': []
        }
        
        for batch_size in batch_sizes:
            self.logger.info(f"Testing batch size: {batch_size}")
            
            try:
                # Generate synthetic data
                activations = self._generate_synthetic_layers(
                    batch_size=batch_size,
                    feature_dim=feature_dim,
                    num_layers=num_layers
                )
                
                # Measure memory usage
                measurement = self._measure_memory_usage(activations, batch_size)
                results['measurements'].append(measurement)
                
                # Clear memory
                del activations
                self.baseline_cka.clear_intermediates()
                self._clear_device_memory()
                
            except Exception as e:
                self.logger.error(f"Failed at batch size {batch_size}: {e}")
                results['measurements'].append({
                    'batch_size': batch_size,
                    'error': str(e),
                    'success': False
                })
        
        # Analyze scaling behavior
        results['analysis'] = self._analyze_scaling_behavior(results['measurements'])
        
        return results
    
    def analyze_layer_scaling(
        self,
        layer_counts: List[int] = None,
        batch_size: int = 1000,
        feature_dim: int = 512
    ) -> Dict[str, any]:
        """Analyze memory scaling with number of layers.
        
        Args:
            layer_counts: List of layer counts to test
            batch_size: Batch size for testing
            feature_dim: Feature dimension for synthetic data
            
        Returns:
            Dictionary with scaling analysis results
        """
        if layer_counts is None:
            layer_counts = [5, 10, 20, 50, 100]
        
        self.logger.info(f"Analyzing layer count scaling: {layer_counts}")
        
        results = {
            'experiment': 'layer_scaling',
            'layer_counts': layer_counts,
            'batch_size': batch_size,
            'feature_dim': feature_dim,
            'measurements': []
        }
        
        for num_layers in layer_counts:
            self.logger.info(f"Testing {num_layers} layers")
            
            try:
                # Generate synthetic data
                activations = self._generate_synthetic_layers(
                    batch_size=batch_size,
                    feature_dim=feature_dim,
                    num_layers=num_layers
                )
                
                # Measure memory usage
                measurement = self._measure_memory_usage(activations, batch_size)
                measurement['num_layers'] = num_layers
                results['measurements'].append(measurement)
                
                # Clear memory
                del activations
                self.baseline_cka.clear_intermediates()
                self._clear_device_memory()
                
            except Exception as e:
                self.logger.error(f"Failed at {num_layers} layers: {e}")
                results['measurements'].append({
                    'num_layers': num_layers,
                    'error': str(e),
                    'success': False
                })
        
        # Analyze scaling behavior
        results['analysis'] = self._analyze_scaling_behavior(results['measurements'])
        
        return results
    
    def analyze_feature_dim_scaling(
        self,
        feature_dims: List[int] = None,
        batch_size: int = 1000,
        num_layers: int = 10
    ) -> Dict[str, any]:
        """Analyze memory scaling with feature dimension.
        
        Args:
            feature_dims: List of feature dimensions to test
            batch_size: Batch size for testing
            num_layers: Number of layers to generate
            
        Returns:
            Dictionary with scaling analysis results
        """
        if feature_dims is None:
            feature_dims = [128, 256, 512, 1024, 2048]
        
        self.logger.info(f"Analyzing feature dimension scaling: {feature_dims}")
        
        results = {
            'experiment': 'feature_dim_scaling',
            'feature_dims': feature_dims,
            'batch_size': batch_size,
            'num_layers': num_layers,
            'measurements': []
        }
        
        for feature_dim in feature_dims:
            self.logger.info(f"Testing feature dimension: {feature_dim}")
            
            try:
                # Generate synthetic data
                activations = self._generate_synthetic_layers(
                    batch_size=batch_size,
                    feature_dim=feature_dim,
                    num_layers=num_layers
                )
                
                # Measure memory usage
                measurement = self._measure_memory_usage(activations, batch_size)
                measurement['feature_dim'] = feature_dim
                results['measurements'].append(measurement)
                
                # Clear memory
                del activations
                self.baseline_cka.clear_intermediates()
                self._clear_device_memory()
                
            except Exception as e:
                self.logger.error(f"Failed at feature dimension {feature_dim}: {e}")
                results['measurements'].append({
                    'feature_dim': feature_dim,
                    'error': str(e),
                    'success': False
                })
        
        # Analyze scaling behavior
        results['analysis'] = self._analyze_scaling_behavior(results['measurements'])
        
        return results
    
    def _generate_synthetic_layers(
        self,
        batch_size: int,
        feature_dim: int,
        num_layers: int
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic layer activations.
        
        Args:
            batch_size: Number of samples
            feature_dim: Feature dimension
            num_layers: Number of layers
            
        Returns:
            Dictionary with synthetic activations
        """
        activations = {}
        
        for i in range(num_layers):
            layer_name = f"layer_{i:02d}"
            activation = torch.randn(
                batch_size, feature_dim,
                device=self.device,
                dtype=torch.float32
            )
            activations[layer_name] = activation
        
        return activations
    
    def _measure_memory_usage(
        self,
        activations: Dict[str, torch.Tensor],
        batch_size: int
    ) -> Dict[str, any]:
        """Measure memory usage for given activations.
        
        Args:
            activations: Dictionary of activations
            batch_size: Batch size
            
        Returns:
            Dictionary with memory measurements
        """
        # Get initial memory state
        initial_memory = get_mac_memory_info()
        
        # Compute CKA matrix
        try:
            cka_matrix, profiling_data = self.baseline_cka.compute_baseline_cka_matrix(activations)
            
            # Get final memory state
            final_memory = get_mac_memory_info()
            
            # Calculate memory components
            activation_memory = sum(
                act.element_size() * act.numel() for act in activations.values()
            ) / (1024**3)
            
            gram_memory = batch_size * batch_size * len(activations) * 4 / (1024**3)  # float32
            
            measurement = {
                'batch_size': batch_size,
                'num_layers': len(activations),
                'activation_memory_gb': activation_memory,
                'gram_memory_gb': gram_memory,
                'total_memory_increase_gb': profiling_data['memory_increase_gb'],
                'computation_time_seconds': profiling_data['computation_time_seconds'],
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'cka_matrix_shape': cka_matrix.shape,
                'success': True
            }
            
            return measurement
            
        except Exception as e:
            return {
                'batch_size': batch_size,
                'error': str(e),
                'success': False
            }
    
    def _analyze_scaling_behavior(
        self,
        measurements: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Analyze scaling behavior from measurements.
        
        Args:
            measurements: List of measurement dictionaries
            
        Returns:
            Dictionary with scaling analysis
        """
        # Filter successful measurements
        successful = [m for m in measurements if m.get('success', False)]
        
        if len(successful) < 2:
            return {'error': 'Not enough successful measurements for analysis'}
        
        # Extract data arrays
        if 'batch_size' in successful[0]:
            x_values = [m['batch_size'] for m in successful]
            x_name = 'batch_size'
        elif 'num_layers' in successful[0]:
            x_values = [m['num_layers'] for m in successful]
            x_name = 'num_layers'
        elif 'feature_dim' in successful[0]:
            x_values = [m['feature_dim'] for m in successful]
            x_name = 'feature_dim'
        else:
            return {'error': 'Unknown scaling variable'}
        
        memory_values = [m['total_memory_increase_gb'] for m in successful]
        time_values = [m['computation_time_seconds'] for m in successful]
        
        # Fit polynomial models
        memory_fit = np.polyfit(x_values, memory_values, 2)  # Quadratic fit
        time_fit = np.polyfit(x_values, time_values, 2)
        
        # Calculate R-squared
        memory_pred = np.polyval(memory_fit, x_values)
        time_pred = np.polyval(time_fit, x_values)
        
        memory_r2 = 1 - np.sum((memory_values - memory_pred)**2) / np.sum((memory_values - np.mean(memory_values))**2)
        time_r2 = 1 - np.sum((time_values - time_pred)**2) / np.sum((time_values - np.mean(time_values))**2)
        
        # Theoretical scaling analysis
        if x_name == 'batch_size':
            expected_memory_scaling = 'O(n²)'  # Gram matrices
            expected_time_scaling = 'O(n²)'
        elif x_name == 'num_layers':
            expected_memory_scaling = 'O(L²)'  # CKA matrix
            expected_time_scaling = 'O(L²)'
        elif x_name == 'feature_dim':
            expected_memory_scaling = 'O(d)'  # Linear in features
            expected_time_scaling = 'O(d)'
        else:
            expected_memory_scaling = 'Unknown'
            expected_time_scaling = 'Unknown'
        
        analysis = {
            'scaling_variable': x_name,
            'num_measurements': len(successful),
            'memory_analysis': {
                'polynomial_coefficients': memory_fit.tolist(),
                'r_squared': memory_r2,
                'expected_scaling': expected_memory_scaling,
                'min_memory_gb': min(memory_values),
                'max_memory_gb': max(memory_values),
                'memory_range_gb': max(memory_values) - min(memory_values)
            },
            'time_analysis': {
                'polynomial_coefficients': time_fit.tolist(),
                'r_squared': time_r2,
                'expected_scaling': expected_time_scaling,
                'min_time_seconds': min(time_values),
                'max_time_seconds': max(time_values),
                'time_range_seconds': max(time_values) - min(time_values)
            },
            'optimization_targets': self._identify_optimization_targets(successful)
        }
        
        return analysis
    
    def _identify_optimization_targets(
        self,
        measurements: List[Dict[str, any]]
    ) -> List[str]:
        """Identify optimization targets based on measurements.
        
        Args:
            measurements: List of measurement dictionaries
            
        Returns:
            List of optimization target strings
        """
        targets = []
        
        # Check if Gram matrix memory dominates
        for m in measurements:
            if m.get('gram_memory_gb', 0) > m.get('activation_memory_gb', 0) * 2:
                targets.append("Gram matrix computation (high memory usage)")
                break
        
        # Check for quadratic scaling
        memory_values = [m['total_memory_increase_gb'] for m in measurements]
        if len(memory_values) >= 3:
            # Check if memory grows quadratically
            ratios = [memory_values[i+1] / memory_values[i] for i in range(len(memory_values)-1)]
            if all(r > 1.5 for r in ratios):  # Rapid growth
                targets.append("Quadratic memory scaling needs optimization")
        
        # Check computation time
        time_values = [m['computation_time_seconds'] for m in measurements]
        if max(time_values) > 60:  # More than 1 minute
            targets.append("Computation time optimization needed")
        
        return targets
    
    def _clear_device_memory(self):
        """Clear device memory caches."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
    
    def generate_scaling_plots(
        self,
        results: Dict[str, any],
        save_plots: bool = True
    ) -> None:
        """Generate scaling analysis plots.
        
        Args:
            results: Results dictionary from scaling analysis
            save_plots: Whether to save plots to disk
        """
        if not results.get('measurements'):
            self.logger.warning("No measurements to plot")
            return
        
        # Filter successful measurements
        successful = [m for m in results['measurements'] if m.get('success', False)]
        
        if len(successful) < 2:
            self.logger.warning("Not enough successful measurements to plot")
            return
        
        # Determine x-axis variable
        if 'batch_size' in successful[0]:
            x_values = [m['batch_size'] for m in successful]
            x_label = 'Batch Size'
        elif 'num_layers' in successful[0]:
            x_values = [m['num_layers'] for m in successful]
            x_label = 'Number of Layers'
        elif 'feature_dim' in successful[0]:
            x_values = [m['feature_dim'] for m in successful]
            x_label = 'Feature Dimension'
        else:
            self.logger.warning("Unknown scaling variable for plotting")
            return
        
        memory_values = [m['total_memory_increase_gb'] for m in successful]
        time_values = [m['computation_time_seconds'] for m in successful]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory scaling plot
        ax1.plot(x_values, memory_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title(f'Memory Scaling vs {x_label}')
        ax1.grid(True, alpha=0.3)
        
        # Time scaling plot
        ax2.plot(x_values, time_values, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title(f'Time Scaling vs {x_label}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_filename = f"scaling_analysis_{results['experiment']}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Scaling plot saved to {plot_path}")
        
        plt.show()
    
    def save_results(self, results: Dict[str, any], filename: str) -> None:
        """Save results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        filepath = self.output_dir / f"{filename}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")


def main():
    """Main entry point for memory scaling analysis."""
    analyzer = MemoryScalingAnalyzer()
    
    print("Running memory scaling analysis...")
    
    # 1. Batch size scaling
    print("\n1. Analyzing batch size scaling...")
    batch_results = analyzer.analyze_batch_size_scaling(
        batch_sizes=[50, 100, 200, 500, 1000]
    )
    analyzer.save_results(batch_results, "batch_size_scaling")
    analyzer.generate_scaling_plots(batch_results)
    
    # 2. Layer count scaling
    print("\n2. Analyzing layer count scaling...")
    layer_results = analyzer.analyze_layer_scaling(
        layer_counts=[5, 10, 20, 50]
    )
    analyzer.save_results(layer_results, "layer_count_scaling")
    analyzer.generate_scaling_plots(layer_results)
    
    # 3. Feature dimension scaling
    print("\n3. Analyzing feature dimension scaling...")
    feature_results = analyzer.analyze_feature_dim_scaling(
        feature_dims=[128, 256, 512, 1024]
    )
    analyzer.save_results(feature_results, "feature_dim_scaling")
    analyzer.generate_scaling_plots(feature_results)
    
    print("\nMemory scaling analysis complete!")


if __name__ == "__main__":
    main()
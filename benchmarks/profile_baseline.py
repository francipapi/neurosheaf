#!/usr/bin/env python3
"""Main baseline profiling script for Mac-optimized benchmarking.

This script establishes the 1.5TB baseline performance metrics for
Neurosheaf CKA computation on Mac hardware, with specific optimizations
for Apple Silicon MPS support.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add neurosheaf to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import platform

from neurosheaf.cka.baseline import BaselineCKA
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.utils.logging import setup_logger
from neurosheaf.utils.profiling import get_mac_device_info, get_mac_memory_info, profile_mac_memory
from benchmarks.synthetic_data import SyntheticDataGenerator


class MacBaselineProfiler:
    """Mac-optimized baseline profiler for Neurosheaf CKA computation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        log_level: str = "INFO",
        output_dir: str = "baseline_results"
    ):
        """Initialize the Mac baseline profiler.
        
        Args:
            device: Device to use (auto-detected if None)
            log_level: Logging level
            output_dir: Directory to save results
        """
        self.logger = setup_logger("neurosheaf.benchmarks.baseline", level=log_level)
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
        
        self.logger.info(f"Initialized MacBaselineProfiler on {self.device}")
        self.logger.info(f"Mac system: {self.is_mac}, Apple Silicon: {self.is_apple_silicon}")
    
    def profile_resnet50_baseline(
        self,
        batch_size: int = 1000,
        scale_factor: float = 10.0,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Profile ResNet50 baseline to establish 1.5TB target.
        
        Args:
            batch_size: Batch size for profiling
            scale_factor: Scale factor to reach memory target
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with profiling results
        """
        self.logger.info("Starting ResNet50 baseline profiling...")
        
        # Get initial system state
        initial_memory = get_mac_memory_info()
        device_info = get_mac_device_info()
        
        # Generate synthetic ResNet50 data
        self.logger.info("Generating synthetic ResNet50 activations...")
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=batch_size,
            scale_factor=scale_factor
        )
        
        # Profile baseline CKA computation
        self.logger.info("Computing baseline CKA matrix...")
        
        @profile_mac_memory
        def _compute_baseline():
            return self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        cka_matrix, cka_profiling = _compute_baseline()
        
        # Get final system state
        final_memory = get_mac_memory_info()
        
        # Compile comprehensive results
        results = {
            'timestamp': time.time(),
            'experiment': 'resnet50_baseline',
            'parameters': {
                'batch_size': batch_size,
                'scale_factor': scale_factor,
                'num_layers': len(activations),
                'total_activations_gb': sum(
                    act.element_size() * act.numel() for act in activations.values()
                ) / (1024**3)
            },
            'system_info': {
                'device_info': device_info,
                'initial_memory': initial_memory,
                'final_memory': final_memory
            },
            'cka_profiling': cka_profiling,
            'cka_matrix_shape': cka_matrix.shape,
            'baseline_report': self.baseline_cka.generate_baseline_report()
        }
        
        if save_results:
            self._save_results(results, 'resnet50_baseline')
        
        return results
    
    def profile_memory_scaling(
        self,
        sizes: list = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Profile memory scaling across different input sizes.
        
        Args:
            sizes: List of batch sizes to test
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with scaling results
        """
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]
        
        self.logger.info(f"Starting memory scaling profiling for sizes: {sizes}")
        
        scaling_results = {
            'timestamp': time.time(),
            'experiment': 'memory_scaling',
            'sizes': sizes,
            'results': {}
        }
        
        for size in sizes:
            self.logger.info(f"Profiling size {size}...")
            
            # Generate data
            activations = self.data_generator.generate_resnet50_activations(
                batch_size=size,
                scale_factor=1.0
            )
            
            # Profile memory usage
            initial_memory = get_mac_memory_info()
            
            @profile_mac_memory
            def _compute_scaled():
                return self.baseline_cka.compute_baseline_cka_matrix(activations)
            
            cka_matrix, profiling_data = _compute_scaled()
            
            final_memory = get_mac_memory_info()
            
            # Store results
            scaling_results['results'][size] = {
                'profiling_data': profiling_data,
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'cka_matrix_shape': cka_matrix.shape
            }
            
            # Clear memory
            del activations, cka_matrix
            self.baseline_cka.clear_intermediates()
            
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        if save_results:
            self._save_results(scaling_results, 'memory_scaling')
        
        return scaling_results
    
    def profile_debiased_vs_baseline(
        self,
        batch_size: int = 1000,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Compare debiased CKA vs baseline performance.
        
        Args:
            batch_size: Batch size for comparison
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Starting debiased vs baseline comparison...")
        
        # Generate test data
        activations = self.data_generator.generate_resnet50_activations(
            batch_size=batch_size,
            scale_factor=1.0
        )
        
        comparison_results = {
            'timestamp': time.time(),
            'experiment': 'debiased_vs_baseline',
            'batch_size': batch_size,
            'results': {}
        }
        
        # Profile baseline CKA
        self.logger.info("Profiling baseline CKA...")
        
        @profile_mac_memory
        def _compute_baseline():
            return self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        baseline_matrix, baseline_profiling = _compute_baseline()
        
        comparison_results['results']['baseline'] = {
            'profiling_data': baseline_profiling,
            'matrix_shape': baseline_matrix.shape
        }
        
        # Clear memory
        self.baseline_cka.clear_intermediates()
        
        # Profile debiased CKA
        self.logger.info("Profiling debiased CKA...")
        
        @profile_mac_memory
        def _compute_debiased():
            return self.debiased_cka.compute_cka_matrix(activations)
        
        debiased_matrix = _compute_debiased()
        
        comparison_results['results']['debiased'] = {
            'matrix_shape': debiased_matrix.shape
        }
        
        # Compare results
        if baseline_matrix.shape == debiased_matrix.shape:
            matrix_diff = torch.abs(baseline_matrix - debiased_matrix)
            comparison_results['comparison'] = {
                'max_difference': float(torch.max(matrix_diff)),
                'mean_difference': float(torch.mean(matrix_diff)),
                'correlation': float(torch.corrcoef(torch.stack([
                    baseline_matrix.flatten(), 
                    debiased_matrix.flatten()
                ]))[0, 1])
            }
        
        if save_results:
            self._save_results(comparison_results, 'debiased_vs_baseline')
        
        return comparison_results
    
    def profile_stress_test(
        self,
        target_memory_gb: float = 20.0,  # 20GB target
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run memory stress test to approach target memory usage.
        
        Args:
            target_memory_gb: Target memory usage in GB
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with stress test results
        """
        self.logger.info(f"Starting memory stress test: target={target_memory_gb:.1f}GB")
        
        # Generate memory-intensive data
        activations = self.data_generator.generate_memory_stress_test(
            target_memory_gb=target_memory_gb,
            batch_size=1000,
            num_layers=20
        )
        
        # Profile stress test
        initial_memory = get_mac_memory_info()
        
        @profile_mac_memory
        def _compute_stress():
            return self.baseline_cka.compute_baseline_cka_matrix(activations)
        
        try:
            cka_matrix, profiling_data = _compute_stress()
            success = True
            error_message = None
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            success = False
            error_message = str(e)
            cka_matrix = None
            profiling_data = None
        
        final_memory = get_mac_memory_info()
        
        results = {
            'timestamp': time.time(),
            'experiment': 'memory_stress_test',
            'target_memory_gb': target_memory_gb,
            'success': success,
            'error_message': error_message,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'profiling_data': profiling_data,
            'cka_matrix_shape': cka_matrix.shape if cka_matrix is not None else None
        }
        
        if save_results:
            self._save_results(results, 'memory_stress_test')
        
        return results
    
    def run_comprehensive_benchmark(self, save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with all benchmark results
        """
        self.logger.info("Starting comprehensive benchmark suite...")
        
        comprehensive_results = {
            'timestamp': time.time(),
            'experiment': 'comprehensive_benchmark',
            'system_info': {
                'device_info': get_mac_device_info(),
                'initial_memory': get_mac_memory_info()
            },
            'benchmarks': {}
        }
        
        # 1. ResNet50 baseline
        self.logger.info("Running ResNet50 baseline...")
        comprehensive_results['benchmarks']['resnet50_baseline'] = self.profile_resnet50_baseline(
            batch_size=500,  # Reduced for practical testing
            scale_factor=2.0,
            save_results=False
        )
        
        # 2. Memory scaling
        self.logger.info("Running memory scaling...")
        comprehensive_results['benchmarks']['memory_scaling'] = self.profile_memory_scaling(
            sizes=[100, 500, 1000, 2000],
            save_results=False
        )
        
        # 3. Debiased vs baseline
        self.logger.info("Running debiased vs baseline comparison...")
        comprehensive_results['benchmarks']['debiased_vs_baseline'] = self.profile_debiased_vs_baseline(
            batch_size=500,
            save_results=False
        )
        
        # 4. Stress test
        self.logger.info("Running memory stress test...")
        comprehensive_results['benchmarks']['stress_test'] = self.profile_stress_test(
            target_memory_gb=20.0,  # 20GB target
            save_results=False
        )
        
        # Final system state
        comprehensive_results['system_info']['final_memory'] = get_mac_memory_info()
        
        if save_results:
            self._save_results(comprehensive_results, 'comprehensive_benchmark')
        
        return comprehensive_results
    
    def _save_results(self, results: Dict[str, Any], experiment_name: str) -> None:
        """Save results to JSON file.
        
        Args:
            results: Results dictionary to save
            experiment_name: Name of the experiment
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert any torch tensors to lists for JSON serialization
        results_serializable = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        else:
            return obj


def main():
    """Main entry point for baseline profiling."""
    parser = argparse.ArgumentParser(description="Neurosheaf Mac Baseline Profiler")
    parser.add_argument(
        "--experiment",
        choices=["resnet50", "scaling", "comparison", "stress", "comprehensive"],
        default="comprehensive",
        help="Experiment to run"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Device to use (auto-detected if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for profiling"
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Scale factor for memory usage"
    )
    parser.add_argument(
        "--target-memory",
        type=float,
        default=20.0,
        help="Target memory usage in GB"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = MacBaselineProfiler(
        device=args.device,
        log_level=args.log_level,
        output_dir=args.output_dir
    )
    
    # Run selected experiment
    if args.experiment == "resnet50":
        results = profiler.profile_resnet50_baseline(
            batch_size=args.batch_size,
            scale_factor=args.scale_factor
        )
    elif args.experiment == "scaling":
        results = profiler.profile_memory_scaling()
    elif args.experiment == "comparison":
        results = profiler.profile_debiased_vs_baseline(
            batch_size=args.batch_size
        )
    elif args.experiment == "stress":
        results = profiler.profile_stress_test(
            target_memory_gb=args.target_memory
        )
    elif args.experiment == "comprehensive":
        results = profiler.run_comprehensive_benchmark()
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE PROFILING SUMMARY")
    print("="*50)
    print(f"Experiment: {args.experiment}")
    print(f"Device: {profiler.device}")
    print(f"Mac System: {profiler.is_mac}")
    print(f"Apple Silicon: {profiler.is_apple_silicon}")
    print("="*50)


if __name__ == "__main__":
    main()
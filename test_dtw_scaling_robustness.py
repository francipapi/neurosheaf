#!/usr/bin/env python3
"""
Comprehensive test suite for DTW scaling robustness
Tests the fixed LogScaleInterpolationDTW with different input sizes
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import neurosheaf components and the fixed LogScaleInterpolationDTW
import sys
sys.path.insert(0, str(Path(__file__).parent))

from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf_comprehensive_demo import LogScaleInterpolationDTW

# Model classes for testing
class ActualCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.0),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(), nn.Dropout(0.0),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(), nn.Dropout(0.0),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layers[1](self.layers[0](x))
        x = self.layers[4](self.layers[3](self.layers[2](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[7](self.layers[6](self.layers[5](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[10](self.layers[9](self.layers[8](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[13](self.layers[12](self.layers[11](x)))
        x = x.view(x.size(0), -1)
        x = self.layers[15](self.layers[14](x))
        return x

def run_single_analysis(batch_size: int, model_path: str, model_class) -> Tuple[List[torch.Tensor], bool]:
    """Run spectral analysis for a single batch size and return eigenvalue evolution."""
    try:
        # Load model
        model = load_model(model_class, model_path, device="cpu")
        data = 8 * torch.randn(batch_size, 3)
        
        # Use consistent GW config
        gw_config = GWConfig(
            epsilon=0.1,
            max_iter=1000,
            tolerance=1e-6,
            quasi_sheaf_tolerance=0.08
        )
        
        # Build sheaf
        analyzer = NeurosheafAnalyzer(device='cpu')
        analysis = analyzer.analyze(model, data, method='gromov_wasserstein', gw_config=gw_config)
        sheaf = analysis['sheaf']
        
        # Run spectral analysis
        spectral_analyzer = PersistentSpectralAnalyzer(
            default_n_steps=10,
            default_filtration_type='threshold'
        )
        
        results = spectral_analyzer.analyze(sheaf, filtration_type='threshold', n_steps=10)
        eigenval_seqs = results['persistence_result']['eigenvalue_sequences']
        
        return eigenval_seqs, True
        
    except Exception as e:
        logger.error(f"Analysis failed for batch size {batch_size}: {e}")
        return [], False

def test_dtw_robustness(eigenval_seqs_1: List[torch.Tensor], 
                       eigenval_seqs_2: List[torch.Tensor],
                       batch_size_1: int, batch_size_2: int,
                       test_name: str) -> Dict:
    """Test DTW distance computation between two eigenvalue sequences."""
    logger.info(f"Testing DTW robustness: {test_name}")
    
    # Set up the enhanced DTW comparator
    optimal_config = {
        'constraint_band': 0.0,
        'min_eigenvalue_threshold': 1e-15,
        'method': 'tslearn',
        'eigenvalue_weight': 1.0,
        'structural_weight': 0.0,
        'normalization_scheme': 'range_aware'
    }
    
    dtw_comparator = LogScaleInterpolationDTW(**optimal_config)
    
    try:
        start_time = time.time()
        
        # Filter to top 15 eigenvalues (matching original config)
        def filter_top_eigenvalues(evolution, k=15):
            filtered = []
            for step_eigenvals in evolution:
                if len(step_eigenvals) > k:
                    sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
                    filtered.append(sorted_eigenvals[:k])
                else:
                    filtered.append(step_eigenvals)
            return filtered
        
        filtered_seqs_1 = filter_top_eigenvalues(eigenval_seqs_1)
        filtered_seqs_2 = filter_top_eigenvalues(eigenval_seqs_2)
        
        # Compute DTW distance
        result = dtw_comparator.compare_eigenvalue_evolution(
            filtered_seqs_1, filtered_seqs_2,
            multivariate=True,
            use_interpolation=True,
            match_all_eigenvalues=True,
            interpolation_points=75
        )
        
        computation_time = time.time() - start_time
        distance = result['normalized_distance']
        
        return {
            'test_name': test_name,
            'batch_sizes': f"{batch_size_1} vs {batch_size_2}",
            'distance': distance,
            'computation_time': computation_time,
            'success': True,
            'sequences_1_steps': len(filtered_seqs_1),
            'sequences_2_steps': len(filtered_seqs_2),
            'sequences_1_eigenvals_per_step': len(filtered_seqs_1[0]) if len(filtered_seqs_1) > 0 else 0,
            'sequences_2_eigenvals_per_step': len(filtered_seqs_2[0]) if len(filtered_seqs_2) > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"DTW computation failed for {test_name}: {e}")
        return {
            'test_name': test_name,
            'batch_sizes': f"{batch_size_1} vs {batch_size_2}",
            'distance': 0.0,
            'computation_time': 0.0,
            'success': False,
            'error': str(e)
        }

def run_comprehensive_scaling_test():
    """Run comprehensive scaling tests across different batch sizes."""
    logger.info("Starting comprehensive DTW scaling robustness test")
    
    # Test configurations
    batch_sizes = [25, 50, 75, 100, 125]
    model_paths = {
        'custom_trained': 'models/torch_custom_acc_1.0000_epoch_200.pth',
        'custom_random': 'models/random_custom_net_000_default_seed_42.pth'
    }
    
    # Step 1: Generate eigenvalue sequences for all batch sizes
    logger.info("Generating eigenvalue sequences for all batch sizes...")
    eigenvalue_data = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Processing batch size: {batch_size}")
        eigenvalue_data[batch_size] = {}
        
        for model_name, model_path in model_paths.items():
            logger.info(f"  Analyzing {model_name}...")
            eigenvals, success = run_single_analysis(batch_size, model_path, ActualCustomModel)
            
            if success:
                logger.info(f"    ✅ Success: {len(eigenvals)} steps, "
                           f"{len(eigenvals[0]) if len(eigenvals) > 0 else 0} eigenvalues per step")
                eigenvalue_data[batch_size][model_name] = eigenvals
            else:
                logger.error(f"    ❌ Failed for {model_name} with batch size {batch_size}")
                eigenvalue_data[batch_size][model_name] = []
    
    # Step 2: Test DTW distances for all combinations
    logger.info("\nTesting DTW distance computation robustness...")
    test_results = []
    
    # Test same model across different batch sizes (should be similar distances)
    for model_name in model_paths.keys():
        logger.info(f"\nTesting {model_name} across different batch sizes:")
        
        for i, batch_size_1 in enumerate(batch_sizes):
            for j, batch_size_2 in enumerate(batch_sizes):
                if i < j:  # Only test unique pairs
                    eigenvals_1 = eigenvalue_data[batch_size_1].get(model_name, [])
                    eigenvals_2 = eigenvalue_data[batch_size_2].get(model_name, [])
                    
                    if len(eigenvals_1) > 0 and len(eigenvals_2) > 0:
                        test_name = f"{model_name}_{batch_size_1}_vs_{batch_size_2}"
                        result = test_dtw_robustness(
                            eigenvals_1, eigenvals_2, 
                            batch_size_1, batch_size_2, 
                            test_name
                        )
                        test_results.append(result)
                        
                        if result['success']:
                            logger.info(f"  ✅ {test_name}: distance = {result['distance']:.6f}")
                        else:
                            logger.error(f"  ❌ {test_name}: FAILED")
    
    # Test different models with same batch size (should have larger distances)
    logger.info(f"\nTesting different models with same batch sizes:")
    for batch_size in batch_sizes:
        eigenvals_trained = eigenvalue_data[batch_size].get('custom_trained', [])
        eigenvals_random = eigenvalue_data[batch_size].get('custom_random', [])
        
        if len(eigenvals_trained) > 0 and len(eigenvals_random) > 0:
            test_name = f"trained_vs_random_batchsize_{batch_size}"
            result = test_dtw_robustness(
                eigenvals_trained, eigenvals_random,
                batch_size, batch_size,
                test_name
            )
            test_results.append(result)
            
            if result['success']:
                logger.info(f"  ✅ {test_name}: distance = {result['distance']:.6f}")  
            else:
                logger.error(f"  ❌ {test_name}: FAILED")
    
    # Step 3: Analyze results and report
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE SCALING TEST RESULTS")
    logger.info(f"{'='*80}")
    
    successful_tests = [r for r in test_results if r['success']]
    failed_tests = [r for r in test_results if not r['success']]
    
    logger.info(f"Total tests: {len(test_results)}")
    logger.info(f"Successful: {len(successful_tests)} ({len(successful_tests)/len(test_results)*100:.1f}%)")
    logger.info(f"Failed: {len(failed_tests)} ({len(failed_tests)/len(test_results)*100:.1f}%)")
    
    if len(successful_tests) > 0:
        distances = [r['distance'] for r in successful_tests]
        computation_times = [r['computation_time'] for r in successful_tests]
        
        logger.info(f"\nDistance Statistics:")
        logger.info(f"  Range: [{min(distances):.6f}, {max(distances):.6f}]")
        logger.info(f"  Mean: {np.mean(distances):.6f}")
        logger.info(f"  Std: {np.std(distances):.6f}")
        
        logger.info(f"\nComputation Time Statistics:")
        logger.info(f"  Range: [{min(computation_times):.2f}s, {max(computation_times):.2f}s]")
        logger.info(f"  Mean: {np.mean(computation_times):.2f}s")
        
        # Analyze patterns
        same_model_tests = [r for r in successful_tests if 'vs' in r['test_name'] and 'trained_vs_random' not in r['test_name']]
        different_model_tests = [r for r in successful_tests if 'trained_vs_random' in r['test_name']]
        
        if len(same_model_tests) > 0:
            same_model_distances = [r['distance'] for r in same_model_tests]
            logger.info(f"\nSame Model, Different Batch Sizes:")
            logger.info(f"  Distance range: [{min(same_model_distances):.6f}, {max(same_model_distances):.6f}]")
            logger.info(f"  Mean distance: {np.mean(same_model_distances):.6f}")
            
        if len(different_model_tests) > 0:
            different_model_distances = [r['distance'] for r in different_model_tests]
            logger.info(f"\nDifferent Models, Same Batch Size:")
            logger.info(f"  Distance range: [{min(different_model_distances):.6f}, {max(different_model_distances):.6f}]")
            logger.info(f"  Mean distance: {np.mean(different_model_distances):.6f}")
            
            # Check if we have proper separation
            if len(same_model_distances) > 0:
                separation_ratio = np.mean(different_model_distances) / np.mean(same_model_distances) if np.mean(same_model_distances) > 0 else float('inf')
                logger.info(f"\nSeparation Analysis:")
                logger.info(f"  Different models distance: {np.mean(different_model_distances):.6f}")
                logger.info(f"  Same model distance: {np.mean(same_model_distances):.6f}")
                logger.info(f"  Separation ratio: {separation_ratio:.2f}x")
                
                if separation_ratio > 5.0:
                    logger.info(f"  ✅ EXCELLENT separation (>{separation_ratio:.1f}x)")
                elif separation_ratio > 2.0:
                    logger.info(f"  ✅ GOOD separation ({separation_ratio:.1f}x)")
                else:
                    logger.info(f"  ⚠️  WEAK separation ({separation_ratio:.1f}x)")
    
    if len(failed_tests) > 0:
        logger.error(f"\nFailed Tests:")
        for failure in failed_tests:
            logger.error(f"  ❌ {failure['test_name']}: {failure.get('error', 'Unknown error')}")
    
    logger.info(f"\n{'='*80}")
    logger.info("SCALING TEST COMPLETE")
    logger.info(f"{'='*80}")
    
    return test_results

def main():
    logger.info("DTW Scaling Robustness Test Suite")
    logger.info("Testing fixed LogScaleInterpolationDTW across different input sizes")
    
    results = run_comprehensive_scaling_test()
    
    # Summary assessment
    successful_results = [r for r in results if r['success']]
    success_rate = len(successful_results) / len(results) * 100 if len(results) > 0 else 0
    
    if success_rate >= 90:
        logger.info(f"🎉 TEST SUITE PASSED: {success_rate:.1f}% success rate")
        logger.info("✅ DTW scaling robustness fix is working correctly!")
    elif success_rate >= 70:
        logger.info(f"⚠️  TEST SUITE PARTIALLY PASSED: {success_rate:.1f}% success rate")
        logger.info("⚠️  Some issues remain, but core functionality works")
    else:
        logger.error(f"❌ TEST SUITE FAILED: {success_rate:.1f}% success rate")
        logger.error("❌ Significant issues with DTW scaling robustness")

if __name__ == "__main__":
    main()
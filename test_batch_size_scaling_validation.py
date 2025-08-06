#!/usr/bin/env python3
"""
Batch Size Scaling Validation Test

This script validates that the DTW pipeline produces consistent results
across different input batch sizes (50 vs 100) using the same real models
from the comprehensive demo. It ensures that:

1. DTW separation ratios are maintained across scales
2. Trained vs trained distances remain low
3. Trained vs random distances remain high  
4. Same models produce consistent measurements regardless of batch size

Expected Results:
- Separation ratio ~17.68x for both 50 and 100 inputs
- Trained vs trained: 0-10K distance range
- Trained vs random: 25K-145K distance range
- Cross-batch consistency for same model pairs
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import logging
from datetime import datetime

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.core.gw_config import GWConfig

# Import the enhanced DTW class and models from comprehensive demo
from neurosheaf_comprehensive_demo import LogScaleInterpolationDTW, MLPModel, ActualCustomModel

# Optimal DTW configuration (17.68x separation)
OPTIMAL_DTW_CONFIG = {
    'constraint_band': 0.0,           # No path constraints (best performance)
    'min_eigenvalue_threshold': 1e-15,
    'method': 'tslearn',              # Multivariate with log-scale
    'eigenvalue_weight': 1.0,
    'structural_weight': 0.0,         # Pure functional similarity
    'normalization_scheme': 'range_aware'
}

EIGENVALUE_SELECTION = 15             # Top 15 eigenvalues (17.68x separation)
INTERPOLATION_POINTS = 75             # Good resolution

# Real model configurations (from comprehensive demo)
MODEL_CONFIGS = {
    'mlp_trained_100': {
        'name': 'MLP Trained (100% Acc)',
        'path': 'models/torch_mlp_acc_1.0000_epoch_200.pth',
        'model_class': MLPModel,
        'is_trained': True,
        'architecture_type': 'MLP',
    },
    'mlp_trained_98': {
        'name': 'MLP Trained (98.57% Acc)',
        'path': 'models/torch_mlp_acc_0.9857_epoch_100.pth',
        'model_class': MLPModel,
        'is_trained': True,
        'architecture_type': 'MLP',
    },
    'custom_trained': {
        'name': 'Custom Trained (100% Acc)',
        'path': 'models/torch_custom_acc_1.0000_epoch_200.pth',
        'model_class': ActualCustomModel,
        'is_trained': True,
        'architecture_type': 'Custom',
    },
    'mlp_random': {
        'name': 'MLP Random',
        'path': 'models/random_mlp_net_000_default_seed_42.pth',
        'model_class': MLPModel,
        'is_trained': False,
        'architecture_type': 'MLP',
    },
    'custom_random': {
        'name': 'Custom Random',
        'path': 'models/random_custom_net_000_default_seed_42.pth',
        'model_class': ActualCustomModel,
        'is_trained': False,
        'architecture_type': 'Custom',
    }
}


def analyze_single_model(model_id: str, model_config: Dict, batch_size: int) -> List[torch.Tensor]:
    """Analyze a single model with given batch size and return eigenvalue evolution."""
    logger.info(f"Analyzing {model_id} with batch size {batch_size}")
    
    try:
        # Load model
        model = load_model(model_config['model_class'], model_config['path'], device="cpu")
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
        
        # Run spectral analysis with parameters that produce meaningful eigenvalue diversity
        spectral_analyzer = PersistentSpectralAnalyzer(
            default_n_steps=50,  # More steps for better eigenvalue evolution
            default_filtration_type='threshold'
        )
        
        results = spectral_analyzer.analyze(sheaf, filtration_type='threshold', n_steps=50)
        eigenval_seqs = results['persistence_result']['eigenvalue_sequences']
        
        logger.info(f"  ✅ Success: {len(eigenval_seqs)} steps, {len(eigenval_seqs[0]) if len(eigenval_seqs) > 0 else 0} eigenvals/step")
        return eigenval_seqs
        
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")
        return []


def filter_top_eigenvalues(evolution: List[torch.Tensor], k: int = 15) -> List[torch.Tensor]:
    """Filter eigenvalue evolution to top-k eigenvalues per step."""
    filtered = []
    for step_eigenvals in evolution:
        if len(step_eigenvals) > k:
            sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
            filtered.append(sorted_eigenvals[:k])
        else:
            filtered.append(step_eigenvals)
    return filtered


def compute_dtw_distance(evolution1: List[torch.Tensor], evolution2: List[torch.Tensor]) -> Dict:
    """Compute DTW distance between two eigenvalue evolutions."""
    dtw_comparator = LogScaleInterpolationDTW(**OPTIMAL_DTW_CONFIG)
    
    # Filter to top eigenvalues
    filtered_ev1 = filter_top_eigenvalues(evolution1, EIGENVALUE_SELECTION)
    filtered_ev2 = filter_top_eigenvalues(evolution2, EIGENVALUE_SELECTION)
    
    # Compute DTW distance with optimal parameters
    result = dtw_comparator.compare_eigenvalue_evolution(
        filtered_ev1, filtered_ev2,
        multivariate=True,
        use_interpolation=True,
        match_all_eigenvalues=True,
        interpolation_points=INTERPOLATION_POINTS
    )
    
    return {
        'distance': result['normalized_distance'],
        'raw_distance': result.get('distance', 0.0),
        'interpolation_used': result.get('interpolation_used', False)
    }


def run_batch_size_scaling_validation():
    """Run comprehensive batch size scaling validation."""
    logger.info("🚀 BATCH SIZE SCALING VALIDATION")
    logger.info("=" * 60)
    logger.info("Testing DTW consistency across 50 vs 100 input batch sizes")
    logger.info("Using real models from comprehensive demo")
    logger.info("Expected: ~17.68x separation ratio for both batch sizes")
    logger.info("")
    
    batch_sizes = [50, 100]
    
    # Step 1: Analyze all models with both batch sizes
    logger.info("📊 STEP 1: Analyzing all models with different batch sizes...")
    eigenvalue_data = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\n--- Batch Size: {batch_size} ---")
        eigenvalue_data[batch_size] = {}
        
        for model_id, model_config in MODEL_CONFIGS.items():
            eigenvals = analyze_single_model(model_id, model_config, batch_size)
            eigenvalue_data[batch_size][model_id] = eigenvals
    
    # Step 2: Compute DTW distances for each batch size
    logger.info("\n📏 STEP 2: Computing DTW distances for each batch size...")
    
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\n--- DTW Analysis for Batch Size {batch_size} ---")
        
        # Compute all pairwise distances
        model_ids = list(MODEL_CONFIGS.keys())
        distances = {}
        
        for i, model_i in enumerate(model_ids):
            for j, model_j in enumerate(model_ids):
                if i < j:  # Avoid duplicates
                    if (len(eigenvalue_data[batch_size][model_i]) > 0 and 
                        len(eigenvalue_data[batch_size][model_j]) > 0):
                        
                        dtw_result = compute_dtw_distance(
                            eigenvalue_data[batch_size][model_i],
                            eigenvalue_data[batch_size][model_j]
                        )
                        
                        pair_key = f"{model_i}_{model_j}"
                        distances[pair_key] = dtw_result['distance']
                        
                        logger.info(f"  {model_i} ↔ {model_j}: {dtw_result['distance']:.6f}")
        
        # Categorize distances
        trained_vs_trained = []
        trained_vs_random = []
        random_vs_random = []
        
        for pair_key, distance in distances.items():
            model_i, model_j = pair_key.split('_', 1)
            config_i = MODEL_CONFIGS[model_i]
            config_j = MODEL_CONFIGS[model_j]
            
            if config_i['is_trained'] and config_j['is_trained']:
                trained_vs_trained.append(distance)
            elif config_i['is_trained'] != config_j['is_trained']:
                trained_vs_random.append(distance)
            else:
                random_vs_random.append(distance)
        
        # Compute separation ratio
        separation_ratio = (np.mean(trained_vs_random) / np.mean(trained_vs_trained) 
                          if trained_vs_trained and np.mean(trained_vs_trained) > 0 else float('inf'))
        
        results[batch_size] = {
            'distances': distances,
            'trained_vs_trained': trained_vs_trained,
            'trained_vs_random': trained_vs_random,
            'random_vs_random': random_vs_random,
            'separation_ratio': separation_ratio
        }
        
        logger.info(f"  Separation Ratio: {separation_ratio:.2f}x")
        logger.info(f"  Trained vs Trained (mean): {np.mean(trained_vs_trained):.2f}")
        logger.info(f"  Trained vs Random (mean): {np.mean(trained_vs_random):.2f}")
    
    # Step 3: Cross-batch size consistency analysis
    logger.info(f"\n🔍 STEP 3: Cross-batch size consistency analysis...")
    
    consistency_results = {}
    
    # Compare same model pairs across different batch sizes
    common_pairs = set(results[50]['distances'].keys()) & set(results[100]['distances'].keys())
    
    for pair_key in common_pairs:
        distance_50 = results[50]['distances'][pair_key]
        distance_100 = results[100]['distances'][pair_key]
        
        if distance_50 > 0 and distance_100 > 0:
            ratio = max(distance_50, distance_100) / min(distance_50, distance_100)
            consistency_results[pair_key] = {
                'distance_50': distance_50,
                'distance_100': distance_100,
                'ratio': ratio
            }
            
            logger.info(f"  {pair_key}: 50→{distance_50:.2f}, 100→{distance_100:.2f}, ratio={ratio:.2f}x")
    
    # Step 4: Generate summary report
    logger.info(f"\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY REPORT")
    logger.info("=" * 60)
    
    # Separation ratio comparison
    sep_50 = results[50]['separation_ratio']
    sep_100 = results[100]['separation_ratio']
    logger.info(f"Separation Ratios:")
    logger.info(f"  50 inputs:  {sep_50:.2f}x")
    logger.info(f"  100 inputs: {sep_100:.2f}x")
    logger.info(f"  Target:     17.68x")
    
    # Pattern validation
    success_50 = sep_50 > 5.0  # Good separation threshold
    success_100 = sep_100 > 5.0
    
    logger.info(f"\nPattern Validation:")
    logger.info(f"  50 inputs:  {'✅ PASS' if success_50 else '❌ FAIL'}")
    logger.info(f"  100 inputs: {'✅ PASS' if success_100 else '❌ FAIL'}")
    
    # Consistency analysis
    if consistency_results:
        ratios = [r['ratio'] for r in consistency_results.values()]
        avg_consistency = np.mean(ratios)
        logger.info(f"\nCross-batch Consistency:")
        logger.info(f"  Average ratio: {avg_consistency:.2f}x")
        logger.info(f"  Consistency:   {'✅ GOOD' if avg_consistency < 3.0 else '⚠️ MODERATE' if avg_consistency < 10.0 else '❌ POOR'}")
    
    # Overall assessment
    overall_success = success_50 and success_100 and (avg_consistency < 5.0 if consistency_results else True)
    
    logger.info(f"\n🎯 OVERALL ASSESSMENT:")
    logger.info(f"  Status: {'✅ VALIDATION PASSED' if overall_success else '❌ VALIDATION FAILED'}")
    logger.info(f"  The DTW pipeline {'is' if overall_success else 'is NOT'} robust to batch size scaling")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_size_scaling_validation_{timestamp}.json"
    
    detailed_results = {
        'timestamp': timestamp,
        'batch_sizes_tested': batch_sizes,
        'models_tested': list(MODEL_CONFIGS.keys()),
        'dtw_config': OPTIMAL_DTW_CONFIG,
        'eigenvalue_selection': EIGENVALUE_SELECTION,
        'interpolation_points': INTERPOLATION_POINTS,
        'results_by_batch_size': results,
        'consistency_analysis': consistency_results,
        'summary': {
            'separation_ratio_50': sep_50,
            'separation_ratio_100': sep_100,
            'average_consistency_ratio': avg_consistency if consistency_results else None,
            'validation_passed': overall_success
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    logger.info(f"  Detailed results saved: {results_file}")
    
    return detailed_results


def main():
    """Main execution function."""
    try:
        results = run_batch_size_scaling_validation()
        
        if results['summary']['validation_passed']:
            logger.info("\n🎉 SUCCESS: Batch size scaling validation completed successfully!")
            return 0
        else:
            logger.error("\n❌ FAILURE: Batch size scaling validation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
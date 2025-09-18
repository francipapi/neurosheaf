#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Grid Search for Elastic Distance Optimization

Systematic search through all parameter combinations to find optimal configuration
that achieves positive margin while maintaining high separation ratio.
"""

import json
import subprocess
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from itertools import product
import pandas as pd
from scipy import stats

def calculate_separation_metrics(distance_matrix: np.ndarray, model_names: List[str]) -> Dict[str, Any]:
    """Calculate separation metrics from distance matrix."""
    
    # Classify models
    trained_indices = []
    random_indices = []
    
    for i, name in enumerate(model_names):
        name_lower = name.lower()
        if any(x in name_lower for x in ['trained', 'acc']):
            trained_indices.append(i)
        elif 'random' in name_lower:
            random_indices.append(i)
    
    if len(trained_indices) < 2 or len(random_indices) < 2:
        return {'error': 'Insufficient trained or random models'}
    
    # Calculate distances
    within_trained = []
    for i in range(len(trained_indices)):
        for j in range(i + 1, len(trained_indices)):
            within_trained.append(distance_matrix[trained_indices[i], trained_indices[j]])
    
    cross_distances = []
    for i in trained_indices:
        for j in random_indices:
            cross_distances.append(distance_matrix[i, j])
    
    within_random = []
    for i in range(len(random_indices)):
        for j in range(i + 1, len(random_indices)):
            within_random.append(distance_matrix[random_indices[i], random_indices[j]])
    
    if not within_trained or not cross_distances:
        return {'error': 'No valid distance pairs found'}
    
    # Calculate metrics
    margin = np.min(cross_distances) - np.max(within_trained)
    separation_ratio = np.mean(cross_distances) / np.mean(within_trained) if np.mean(within_trained) > 0 else float('inf')
    
    # Statistical test
    try:
        t_stat, p_value = stats.ttest_ind(cross_distances, within_trained)
    except:
        t_stat, p_value = None, None
    
    return {
        'margin': margin,
        'separation_ratio': separation_ratio,
        'within_trained_mean': np.mean(within_trained),
        'within_trained_std': np.std(within_trained),
        'within_random_mean': np.mean(within_random),
        'within_random_std': np.std(within_random),
        'cross_mean': np.mean(cross_distances),
        'cross_std': np.std(cross_distances),
        'min_cross': np.min(cross_distances),
        'max_within_trained': np.max(within_trained),
        'n_trained': len(trained_indices),
        'n_random': len(random_indices),
        't_statistic': t_stat,
        'p_value': p_value
    }

def run_elastic_computation(data_dir: str, config: Dict[str, Any], output_prefix: str) -> Dict[str, Any]:
    """Run elastic distance computation with given configuration."""
    
    # Build command
    cmd = [
        'python', 'elastic_optimized.py',
        '--data-dir', data_dir,
        '--pattern', '*eigenvalues.npz',
        '--topk', '0',  # Use full spectrum as requested
        '--amp-norm', 'zscore',
        '--n-jobs', '-1',
        '--out-prefix', output_prefix
    ]
    
    # Add core parameters
    cmd.extend(['--lambda-warp', str(config['lambda_warp'])])
    cmd.extend(['--window-frac', str(config['window_frac'])])
    cmd.extend(['--step-penalty', str(config['step_penalty'])])
    cmd.extend(['--resample', str(config['resample'])])
    cmd.extend(['--smooth', config['smooth']])
    cmd.extend(['--smooth-win', str(config['smooth_win'])])
    
    # Add enhancement features
    if config.get('knee_weight', False):
        cmd.append('--knee-weight')
        cmd.extend(['--knee-weight-power', str(config.get('knee_weight_power', 1.0))])
    
    if config.get('derivative_ensemble', False):
        cmd.append('--derivative-ensemble')
        cmd.extend(['--ensemble-ratio', str(config.get('ensemble_ratio', 0.6))])
    
    if config.get('isotonic', False):
        cmd.append('--isotonic')
    
    # Run computation
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        computation_time = time.time() - start_time
        
        if result.returncode != 0:
            return {
                'error': f"Command failed with return code {result.returncode}",
                'stderr': result.stderr,
                'computation_time': computation_time
            }
        
        # Load results
        distance_matrix = np.load(f"{output_prefix}_distance.npy")
        with open(f"{output_prefix}_index.json", "r") as f:
            model_names = json.load(f)
        
        # Calculate metrics
        metrics = calculate_separation_metrics(distance_matrix, model_names)
        metrics['computation_time'] = computation_time
        metrics['stdout'] = result.stdout
        
        # Clean up temporary files
        for suffix in ['_distance.npy', '_distance.csv', '_index.json', '_config.json']:
            try:
                Path(f"{output_prefix}{suffix}").unlink()
            except:
                pass
        
        return metrics
        
    except subprocess.TimeoutExpired:
        return {'error': 'Computation timed out', 'computation_time': 1800}
    except Exception as e:
        return {'error': str(e), 'computation_time': time.time() - start_time}

def stage1_core_parameters():
    """Stage 1: Search core parameters (lambda_warp, window_frac, step_penalty)."""
    
    print("="*80)
    print("STAGE 1: CORE PARAMETER SEARCH")
    print("="*80)
    
    # Define parameter grid
    parameter_grid = {
        'lambda_warp': [0.05, 0.10, 0.15, 0.20, 0.25],
        'window_frac': [0.10, 0.15, 0.20, 0.25],
        'step_penalty': [0.0, 0.01, 0.02, 0.04, 0.06],
        # Fixed parameters for stage 1
        'resample': [400],
        'smooth': ['moving'],
        'smooth_win': [11],
        'knee_weight': [False],
        'derivative_ensemble': [False],
        'isotonic': [False]
    }
    
    # Generate all combinations
    keys, values = zip(*parameter_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Testing {len(combinations)} core parameter combinations...")
    
    results = []
    for i, config in enumerate(combinations):
        print(f"\nTest {i+1}/{len(combinations)}: λ={config['lambda_warp']}, w={config['window_frac']}, step={config['step_penalty']}")
        
        # Run computation
        output_prefix = f"temp_stage1_{i}"
        metrics = run_elastic_computation("./eigenvalueData", config, output_prefix)
        
        # Store results
        result_entry = {
            'config_id': f"stage1_{i:03d}",
            'stage': 1,
            'parameters': config.copy(),
            **metrics
        }
        results.append(result_entry)
        
        # Print immediate results
        if 'error' not in metrics:
            margin = metrics.get('margin', -999)
            sep_ratio = metrics.get('separation_ratio', -999)
            print(f"  Result: Margin = {margin:+.6f}, Separation = {sep_ratio:.4f}")
            if margin > 0:
                print(f"  🎉 POSITIVE MARGIN ACHIEVED!")
        else:
            print(f"  ❌ Error: {metrics['error']}")
    
    # Save stage 1 results
    with open('stage1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find top performers
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        # Sort by margin first, then separation ratio
        valid_results.sort(key=lambda x: (x.get('margin', -999), x.get('separation_ratio', 0)), reverse=True)
        
        print(f"\n🏆 STAGE 1 TOP 10 RESULTS:")
        print(f"{'Rank':<4} {'Config':<12} {'Margin':<12} {'Sep Ratio':<10} {'λ':<6} {'w':<6} {'step':<6}")
        print("-" * 70)
        
        for i, result in enumerate(valid_results[:10], 1):
            config_id = result['config_id']
            margin = result.get('margin', -999)
            sep_ratio = result.get('separation_ratio', -999)
            params = result['parameters']
            
            print(f"{i:<4} {config_id:<12} {margin:+10.6f} {sep_ratio:8.4f} {params['lambda_warp']:<6} {params['window_frac']:<6} {params['step_penalty']:<6}")
    
    return results

def stage2_enhancements(stage1_results: List[Dict[str, Any]], top_n: int = 10):
    """Stage 2: Test enhancement features on top configurations."""
    
    print(f"\n{'='*80}")
    print("STAGE 2: ENHANCEMENT FEATURES")
    print(f"{'='*80}")
    
    # Get top N configurations from stage 1
    valid_results = [r for r in stage1_results if 'error' not in r]
    if not valid_results:
        print("No valid results from Stage 1!")
        return []
    
    valid_results.sort(key=lambda x: (x.get('margin', -999), x.get('separation_ratio', 0)), reverse=True)
    top_configs = valid_results[:top_n]
    
    print(f"Testing enhancement features on top {len(top_configs)} configurations from Stage 1...")
    
    # Enhancement combinations to test
    enhancements = [
        {'knee_weight': False, 'derivative_ensemble': False},
        {'knee_weight': True, 'knee_weight_power': 0.5, 'derivative_ensemble': False},
        {'knee_weight': True, 'knee_weight_power': 1.0, 'derivative_ensemble': False},
        {'knee_weight': True, 'knee_weight_power': 1.5, 'derivative_ensemble': False},
        {'knee_weight': False, 'derivative_ensemble': True, 'ensemble_ratio': 0.5},
        {'knee_weight': False, 'derivative_ensemble': True, 'ensemble_ratio': 0.6},
        {'knee_weight': False, 'derivative_ensemble': True, 'ensemble_ratio': 0.7},
        {'knee_weight': True, 'knee_weight_power': 1.0, 'derivative_ensemble': True, 'ensemble_ratio': 0.6},
        {'knee_weight': True, 'knee_weight_power': 1.5, 'derivative_ensemble': True, 'ensemble_ratio': 0.6}
    ]
    
    results = []
    config_counter = 0
    
    for base_config in top_configs:
        base_params = base_config['parameters']
        
        for enhancement in enhancements:
            config_counter += 1
            print(f"\nTest {config_counter}: Base {base_config['config_id']} + {enhancement}")
            
            # Merge parameters
            test_config = base_params.copy()
            test_config.update(enhancement)
            
            # Run computation
            output_prefix = f"temp_stage2_{config_counter}"
            metrics = run_elastic_computation("./eigenvalueData", test_config, output_prefix)
            
            # Store results
            result_entry = {
                'config_id': f"stage2_{config_counter:03d}",
                'stage': 2,
                'base_config': base_config['config_id'],
                'parameters': test_config.copy(),
                **metrics
            }
            results.append(result_entry)
            
            # Print immediate results
            if 'error' not in metrics:
                margin = metrics.get('margin', -999)
                sep_ratio = metrics.get('separation_ratio', -999)
                print(f"  Result: Margin = {margin:+.6f}, Separation = {sep_ratio:.4f}")
                if margin > 0:
                    print(f"  🎉 POSITIVE MARGIN ACHIEVED!")
            else:
                print(f"  ❌ Error: {metrics['error']}")
    
    # Save stage 2 results
    with open('stage2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Show best results
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        valid_results.sort(key=lambda x: (x.get('margin', -999), x.get('separation_ratio', 0)), reverse=True)
        
        print(f"\n🏆 STAGE 2 TOP 10 RESULTS:")
        print(f"{'Rank':<4} {'Config':<12} {'Margin':<12} {'Sep Ratio':<10} {'Knee':<5} {'Deriv':<5}")
        print("-" * 60)
        
        for i, result in enumerate(valid_results[:10], 1):
            config_id = result['config_id']
            margin = result.get('margin', -999)
            sep_ratio = result.get('separation_ratio', -999)
            params = result['parameters']
            knee = "Yes" if params.get('knee_weight', False) else "No"
            deriv = "Yes" if params.get('derivative_ensemble', False) else "No"
            
            print(f"{i:<4} {config_id:<12} {margin:+10.6f} {sep_ratio:8.4f} {knee:<5} {deriv:<5}")
    
    return results

def stage3_fine_tuning(stage2_results: List[Dict[str, Any]]):
    """Stage 3: Fine-tune preprocessing on best configurations."""
    
    print(f"\n{'='*80}")
    print("STAGE 3: FINE-TUNING")
    print(f"{'='*80}")
    
    # Get configurations with positive margin
    positive_margin_configs = [r for r in stage2_results 
                              if 'error' not in r and r.get('margin', -999) > 0]
    
    if not positive_margin_configs:
        print("No configurations with positive margin found in Stage 2!")
        print("Testing fine-tuning on top 5 configurations...")
        valid_results = [r for r in stage2_results if 'error' not in r]
        valid_results.sort(key=lambda x: (x.get('margin', -999), x.get('separation_ratio', 0)), reverse=True)
        positive_margin_configs = valid_results[:5]
    
    print(f"Testing fine-tuning on {len(positive_margin_configs)} configurations...")
    
    # Fine-tuning combinations
    fine_tuning_options = [
        {'resample': 300, 'smooth': 'moving', 'smooth_win': 9, 'isotonic': False},
        {'resample': 400, 'smooth': 'moving', 'smooth_win': 11, 'isotonic': False},
        {'resample': 500, 'smooth': 'moving', 'smooth_win': 13, 'isotonic': False},
        {'resample': 400, 'smooth': 'moving', 'smooth_win': 15, 'isotonic': False},
        {'resample': 400, 'smooth': 'savgol', 'smooth_win': 11, 'isotonic': False},
        {'resample': 400, 'smooth': 'moving', 'smooth_win': 11, 'isotonic': True}
    ]
    
    results = []
    config_counter = 0
    
    for base_config in positive_margin_configs:
        base_params = base_config['parameters']
        
        for fine_tune in fine_tuning_options:
            config_counter += 1
            print(f"\nTest {config_counter}: Base {base_config['config_id']} + fine-tuning")
            
            # Merge parameters
            test_config = base_params.copy()
            test_config.update(fine_tune)
            
            # Run computation
            output_prefix = f"temp_stage3_{config_counter}"
            metrics = run_elastic_computation("./eigenvalueData", test_config, output_prefix)
            
            # Store results
            result_entry = {
                'config_id': f"stage3_{config_counter:03d}",
                'stage': 3,
                'base_config': base_config['config_id'],
                'parameters': test_config.copy(),
                **metrics
            }
            results.append(result_entry)
            
            # Print immediate results
            if 'error' not in metrics:
                margin = metrics.get('margin', -999)
                sep_ratio = metrics.get('separation_ratio', -999)
                print(f"  Result: Margin = {margin:+.6f}, Separation = {sep_ratio:.4f}")
                if margin > 0:
                    print(f"  🎉 POSITIVE MARGIN ACHIEVED!")
            else:
                print(f"  ❌ Error: {metrics['error']}")
    
    # Save stage 3 results
    with open('stage3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def analyze_final_results():
    """Analyze all results and identify the optimal configuration."""
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # Load all results
    all_results = []
    
    for stage in [1, 2, 3]:
        filename = f'stage{stage}_results.json'
        if Path(filename).exists():
            with open(filename, 'r') as f:
                stage_results = json.load(f)
                all_results.extend(stage_results)
        else:
            print(f"Warning: {filename} not found")
    
    if not all_results:
        print("No results to analyze!")
        return
    
    # Filter valid results
    valid_results = [r for r in all_results if 'error' not in r]
    print(f"Total valid configurations tested: {len(valid_results)}")
    
    # Find best configurations
    valid_results.sort(key=lambda x: (x.get('margin', -999), x.get('separation_ratio', 0)), reverse=True)
    
    # Summary statistics
    margins = [r.get('margin', -999) for r in valid_results]
    sep_ratios = [r.get('separation_ratio', 0) for r in valid_results]
    positive_margin_count = sum(1 for m in margins if m > 0)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"  Configurations with positive margin: {positive_margin_count}/{len(valid_results)} ({100*positive_margin_count/len(valid_results):.1f}%)")
    print(f"  Best margin achieved: {max(margins):+.6f}")
    print(f"  Best separation ratio: {max(sep_ratios):.4f}")
    print(f"  Average margin: {np.mean(margins):+.6f}")
    print(f"  Average separation ratio: {np.mean(sep_ratios):.4f}")
    
    # Top 20 configurations
    print(f"\n🏆 TOP 20 CONFIGURATIONS:")
    print(f"{'Rank':<4} {'Stage':<6} {'Config':<12} {'Margin':<12} {'Sep Ratio':<10} {'λ':<6} {'w':<6} {'step':<6} {'Knee':<5} {'Deriv':<5}")
    print("-" * 90)
    
    for i, result in enumerate(valid_results[:20], 1):
        config_id = result['config_id']
        stage = result['stage']
        margin = result.get('margin', -999)
        sep_ratio = result.get('separation_ratio', -999)
        params = result['parameters']
        
        lambda_w = params.get('lambda_warp', 0)
        window = params.get('window_frac', 0)
        step = params.get('step_penalty', 0)
        knee = "Yes" if params.get('knee_weight', False) else "No"
        deriv = "Yes" if params.get('derivative_ensemble', False) else "No"
        
        print(f"{i:<4} {stage:<6} {config_id:<12} {margin:+10.6f} {sep_ratio:8.4f} {lambda_w:<6} {window:<6} {step:<6} {knee:<5} {deriv:<5}")
    
    # Best configuration details
    if valid_results:
        best_config = valid_results[0]
        print(f"\n🥇 OPTIMAL CONFIGURATION:")
        print(f"  Config ID: {best_config['config_id']}")
        print(f"  Stage: {best_config['stage']}")
        print(f"  Margin: {best_config.get('margin', 'N/A'):+.6f}")
        print(f"  Separation Ratio: {best_config.get('separation_ratio', 'N/A'):.4f}")
        print(f"  Statistical significance: p = {best_config.get('p_value', 'N/A')}")
        
        print(f"\n  📋 PARAMETERS:")
        params = best_config['parameters']
        for key, value in sorted(params.items()):
            print(f"    {key}: {value}")
        
        # Save best configuration
        with open('optimal_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n✅ Optimal configuration saved to 'optimal_config.json'")
    
    # Create comparison with baseline
    baseline_margin = -0.522118  # From original analysis
    baseline_sep_ratio = 3.6386
    
    if valid_results:
        best_margin = valid_results[0].get('margin', -999)
        best_sep_ratio = valid_results[0].get('separation_ratio', 0)
        
        print(f"\n📈 IMPROVEMENT OVER BASELINE:")
        print(f"  Margin:          {baseline_margin:+.6f} → {best_margin:+.6f} (Δ = {best_margin - baseline_margin:+.6f})")
        print(f"  Separation:      {baseline_sep_ratio:.4f} → {best_sep_ratio:.4f} (Δ = {best_sep_ratio - baseline_sep_ratio:+.4f})")
        
        if best_margin > 0 and baseline_margin <= 0:
            print(f"  🎯 SUCCESS: Achieved positive margin!")

def main():
    print("COMPREHENSIVE ELASTIC DISTANCE OPTIMIZATION")
    print("Using full eigenvalue spectrum (topk=0) as requested")
    print("="*80)
    
    # Stage 1: Core parameters
    stage1_results = stage1_core_parameters()
    
    # Stage 2: Enhancement features
    stage2_results = stage2_enhancements(stage1_results, top_n=10)
    
    # Stage 3: Fine-tuning
    stage3_results = stage3_fine_tuning(stage2_results)
    
    # Final analysis
    analyze_final_results()
    
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE!")
    print("Check 'optimal_config.json' for the best configuration")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter tuning script for elastic distance computation to maximize separation
between trained and random models.

This script systematically tests different parameter combinations for the elastic
distance computation and measures separation quality using various metrics.

Usage:
    python tune_elastic_parameters.py --data-dir eigenvalueData --output-dir results
"""

import argparse
import json
import numpy as np
import pandas as pd
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

try:
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from scipy.stats import ttest_ind
    from scipy.spatial.distance import pdist, squareform
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("[WARN] sklearn/scipy not available. Some metrics will be skipped.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False
    print("[WARN] matplotlib/seaborn not available. Plotting disabled.")


def classify_model_type(filename: str) -> str:
    """Classify model as trained or random based on filename."""
    filename_lower = filename.lower()
    if 'random' in filename_lower:
        return 'random'
    elif 'seed' in filename_lower:
        return 'trained'
    else:
        return 'unknown'


def get_parameter_combinations() -> List[Dict[str, Any]]:
    """Generate parameter combinations to test."""
    
    # Parameter grids - start with focused ranges based on typical values
    resample_values = [100, 200, 300, 500]  # Reasonable range for computational efficiency
    topk_values = [0, 5, 10, -5, -10]  # Focus on all, top-k, or bottom-k eigenvalues
    amp_norm_values = ['zscore', 'unit', 'p95']
    smooth_win_values = [1, 5, 9, 15]  # 1 means no smoothing
    lambda_warp_values = [0.01, 0.05, 0.1, 0.5]
    window_frac_values = [0.1, 0.2, 0.3]
    srvf_norm_values = ['none', 'unit', 'robust']
    
    combinations = []
    
    for (resample, topk, amp_norm, smooth_win, lambda_warp, window_frac, srvf_norm) in product(
        resample_values, topk_values, amp_norm_values, smooth_win_values, 
        lambda_warp_values, window_frac_values, srvf_norm_values
    ):
        combinations.append({
            'resample': resample,
            'topk': topk,
            'amp_norm': amp_norm,
            'smooth_win': smooth_win,
            'lambda_warp': lambda_warp,
            'window_frac': window_frac,
            'srvf_norm': srvf_norm
        })
    
    print(f"Generated {len(combinations)} parameter combinations to test")
    return combinations


def run_elastic_distance(data_dir: Path, params: Dict[str, Any], output_prefix: str) -> Dict[str, Any]:
    """Run elastic distance computation with given parameters."""
    
    cmd = [
        'python', 'elastic_mean_eigs_distance.py',
        '--data-dir', str(data_dir),
        '--pattern', '*eigenvalues.npz',
        '--resample', str(params['resample']),
        '--topk', str(params['topk']),
        '--amp-norm', params['amp_norm'],
        '--smooth', 'moving' if params['smooth_win'] > 1 else 'none',
        '--smooth-win', str(params['smooth_win']),
        '--lambda-warp', str(params['lambda_warp']),
        '--window-frac', str(params['window_frac']),
        '--no-time-scaling',  # Use actual time ranges (per user recommendation)
        '--pad-with-last',   # Pad with last values (per user recommendation)
        '--out-prefix', output_prefix,
        '--n-jobs', '1'  # Sequential to avoid conflicts in parallel execution
    ]
    
    # Add SRVF normalization if specified
    if params['srvf_norm'] != 'none':
        if params['srvf_norm'] == 'robust':
            cmd.append('--use-robust-norm')
        else:
            cmd.extend(['--srvf-norm', params['srvf_norm']])
    
    try:
        # Set environment for conda activation
        env = os.environ.copy()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=300,  # 5 minute timeout
                              env=env)
        
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {result.stderr}")
            return None
            
        # Load the resulting distance matrix and index
        distance_file = f"{output_prefix}_distance.npy"
        index_file = f"{output_prefix}_index.json"
        
        if not (Path(distance_file).exists() and Path(index_file).exists()):
            print(f"Output files not found: {distance_file}, {index_file}")
            return None
            
        distance_matrix = np.load(distance_file)
        with open(index_file, 'r') as f:
            file_index = json.load(f)
            
        return {
            'distance_matrix': distance_matrix,
            'file_index': file_index,
            'params': params
        }
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def compute_separation_metrics(distance_matrix: np.ndarray, file_index: List[str], params: Dict[str, Any]) -> Dict[str, float]:
    """Compute various separation quality metrics."""
    
    # Classify files as trained/random
    labels = [classify_model_type(f) for f in file_index]
    trained_mask = np.array([l == 'trained' for l in labels])
    random_mask = np.array([l == 'random' for l in labels])
    
    # Skip if we don't have both types
    if not (trained_mask.any() and random_mask.any()):
        return {'error': 'missing_model_types'}
    
    metrics = {}
    
    try:
        # 1. Inter-class vs intra-class distance ratio
        # Intra-class distances (trained-trained, random-random)
        trained_indices = np.where(trained_mask)[0]
        random_indices = np.where(random_mask)[0]
        
        if len(trained_indices) > 1:
            trained_distances = distance_matrix[np.ix_(trained_indices, trained_indices)]
            # Get upper triangle excluding diagonal
            triu_mask = np.triu(np.ones_like(trained_distances, dtype=bool), k=1)
            intra_trained = trained_distances[triu_mask]
        else:
            intra_trained = np.array([])
            
        if len(random_indices) > 1:
            random_distances = distance_matrix[np.ix_(random_indices, random_indices)]
            triu_mask = np.triu(np.ones_like(random_distances, dtype=bool), k=1)
            intra_random = random_distances[triu_mask]
        else:
            intra_random = np.array([])
            
        # Inter-class distances (trained-random)
        inter_distances = distance_matrix[np.ix_(trained_indices, random_indices)]
        inter_class = inter_distances.flatten()
        
        # Combine intra-class distances
        intra_class = np.concatenate([intra_trained, intra_random]) if len(intra_trained) > 0 and len(intra_random) > 0 else np.array([0])
        
        if len(intra_class) > 0 and len(inter_class) > 0:
            mean_intra = np.mean(intra_class)
            mean_inter = np.mean(inter_class)
            
            metrics['mean_intra_distance'] = mean_intra
            metrics['mean_inter_distance'] = mean_inter
            metrics['inter_intra_ratio'] = mean_inter / (mean_intra + 1e-8)
            
            # Statistical test
            if len(intra_class) > 1 and len(inter_class) > 1:
                t_stat, p_value = ttest_ind(inter_class, intra_class)
                metrics['t_statistic'] = t_stat
                metrics['p_value'] = p_value
        
        # 2. Silhouette coefficient (if sklearn available)
        if _HAS_SKLEARN and len(set(labels)) == 2:
            # Convert labels to numeric
            numeric_labels = [0 if l == 'trained' else 1 for l in labels]
            
            # Convert distance matrix to distance metric
            # Silhouette score expects a distance matrix format
            try:
                silhouette = silhouette_score(distance_matrix, numeric_labels, metric='precomputed')
                metrics['silhouette_score'] = silhouette
            except Exception as e:
                print(f"Silhouette computation failed: {e}")
                
        # 3. Davies-Bouldin like index (manual computation)
        if len(trained_indices) > 0 and len(random_indices) > 0:
            # Average distance within each cluster
            avg_intra_trained = np.mean(intra_trained) if len(intra_trained) > 0 else 0
            avg_intra_random = np.mean(intra_random) if len(intra_random) > 0 else 0
            
            # Average distance between cluster centers (use medians as centers)
            trained_center_dist = np.median(distance_matrix[trained_indices, :], axis=0)
            random_center_dist = np.median(distance_matrix[random_indices, :], axis=0)
            inter_center_dist = np.linalg.norm(trained_center_dist - random_center_dist)
            
            if inter_center_dist > 0:
                db_index = (avg_intra_trained + avg_intra_random) / inter_center_dist
                metrics['davies_bouldin_index'] = db_index
        
        # 4. Separation quality score (custom metric)
        # Higher inter-class, lower intra-class, higher t-statistic is better
        if 'inter_intra_ratio' in metrics and 't_statistic' in metrics:
            separation_score = metrics['inter_intra_ratio'] * abs(metrics['t_statistic']) / (1 + abs(metrics.get('p_value', 1)))
            metrics['separation_score'] = separation_score
            
    except Exception as e:
        print(f"Error computing metrics: {e}")
        metrics['error'] = str(e)
    
    # Add parameter info
    metrics['params'] = params
    return metrics


def run_single_experiment(args_tuple: Tuple[Path, Dict[str, Any], str]) -> Dict[str, Any]:
    """Run a single parameter combination experiment."""
    data_dir, params, temp_dir = args_tuple
    
    # Create unique output prefix
    param_hash = hash(str(sorted(params.items())))
    output_prefix = f"{temp_dir}/test_{param_hash}"
    
    print(f"Testing params: {params}")
    
    # Run elastic distance computation
    result = run_elastic_distance(data_dir, params, output_prefix)
    
    if result is None:
        return {'params': params, 'error': 'computation_failed'}
    
    # Compute separation metrics
    metrics = compute_separation_metrics(
        result['distance_matrix'], 
        result['file_index'], 
        params
    )
    
    # Cleanup temporary files
    try:
        for suffix in ['_distance.npy', '_distance.csv', '_index.json']:
            temp_file = Path(f"{output_prefix}{suffix}")
            if temp_file.exists():
                temp_file.unlink()
    except Exception as e:
        print(f"Cleanup error: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Tune elastic distance parameters for optimal separation")
    parser.add_argument('--data-dir', type=str, default='eigenvalueData', 
                       help='Directory containing eigenvalue files')
    parser.add_argument('--output-dir', type=str, default='parameter_tuning_results',
                       help='Directory to save results')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of parallel processes')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of parameter combinations to test (None for all)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print("Generating parameter combinations...")
    all_combinations = get_parameter_combinations()
    
    if args.sample_size and args.sample_size < len(all_combinations):
        # Random sample for faster experimentation
        import random
        random.seed(42)
        combinations = random.sample(all_combinations, args.sample_size)
        print(f"Randomly selected {len(combinations)} combinations to test")
    else:
        combinations = all_combinations
    
    print(f"Running {len(combinations)} experiments with {args.n_jobs} parallel processes...")
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Prepare arguments for parallel processing
        experiment_args = [(data_dir, params, temp_dir) for params in combinations]
        
        results = []
        completed = 0
        
        # Run experiments in parallel
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            # Submit all jobs
            future_to_params = {executor.submit(run_single_experiment, args): args[1] 
                              for args in experiment_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % max(1, len(combinations) // 20) == 0:
                        print(f"Completed {completed}/{len(combinations)} experiments ({100*completed/len(combinations):.1f}%)")
                        
                except Exception as e:
                    print(f"Experiment failed: {e}")
                    params = future_to_params[future]
                    results.append({'params': params, 'error': str(e)})
    
    print(f"\nCompleted all {len(results)} experiments")
    
    # Filter out failed experiments
    successful_results = [r for r in results if 'error' not in r or r.get('separation_score') is not None]
    failed_count = len(results) - len(successful_results)
    
    if failed_count > 0:
        print(f"Note: {failed_count} experiments failed")
    
    if len(successful_results) == 0:
        print("No successful experiments! Check your setup.")
        return
    
    # Save raw results
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = []
        for r in results:
            serializable_r = {}
            for k, v in r.items():
                if isinstance(v, np.floating):
                    serializable_r[k] = float(v)
                elif isinstance(v, np.integer):
                    serializable_r[k] = int(v)
                else:
                    serializable_r[k] = v
            serializable_results.append(serializable_r)
        json.dump(serializable_results, f, indent=2)
    
    print(f"Raw results saved to {results_file}")
    
    # Analyze and rank results
    df_results = pd.DataFrame(successful_results)
    
    # Define the primary ranking metric (higher is better)
    if 'separation_score' in df_results.columns:
        ranking_metric = 'separation_score'
    elif 'inter_intra_ratio' in df_results.columns:
        ranking_metric = 'inter_intra_ratio'
    elif 'silhouette_score' in df_results.columns:
        ranking_metric = 'silhouette_score'
    else:
        print("No suitable ranking metric found!")
        return
    
    # Sort by ranking metric (descending for most metrics)
    df_sorted = df_results.sort_values(ranking_metric, ascending=False)
    
    # Save top results
    top_results_file = output_dir / 'top_results.csv'
    df_sorted.head(50).to_csv(top_results_file, index=False)
    
    print(f"\nTop results saved to {top_results_file}")
    print("\n" + "="*80)
    print("TOP 10 PARAMETER COMBINATIONS:")
    print("="*80)
    
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"\n{i+1}. RANK {i+1}:")
        print(f"   {ranking_metric.upper()}: {row[ranking_metric]:.4f}")
        
        if 'params' in row and isinstance(row['params'], dict):
            params = row['params']
            print("   Parameters:")
            for param, value in params.items():
                print(f"     --{param.replace('_', '-')} {value}")
        
        # Show other metrics
        other_metrics = ['inter_intra_ratio', 'silhouette_score', 't_statistic', 'p_value']
        for metric in other_metrics:
            if metric in row and pd.notna(row[metric]) and metric != ranking_metric:
                print(f"   {metric}: {row[metric]:.4f}")
    
    # Generate summary statistics
    print(f"\n" + "="*80)
    print("PARAMETER IMPACT ANALYSIS:")
    print("="*80)
    
    if 'params' in df_results.columns:
        # Extract parameter columns
        param_columns = {}
        for idx, row in df_results.iterrows():
            if isinstance(row['params'], dict):
                for param, value in row['params'].items():
                    if param not in param_columns:
                        param_columns[param] = []
                    param_columns[param].append(value)
        
        # Add parameter columns to dataframe
        for param, values in param_columns.items():
            df_results[f'param_{param}'] = values
        
        # Analyze parameter importance
        for param in param_columns.keys():
            if len(set(param_columns[param])) > 1:  # Only if parameter varies
                grouped = df_results.groupby(f'param_{param}')[ranking_metric].agg(['mean', 'std', 'count'])
                print(f"\n{param.upper()}:")
                for value, stats in grouped.iterrows():
                    print(f"  {value}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['count']}")
    
    # Save optimal configuration
    best_result = df_sorted.iloc[0]
    optimal_config = {
        'best_score': float(best_result[ranking_metric]),
        'ranking_metric': ranking_metric,
        'parameters': best_result['params'] if 'params' in best_result else {},
        'all_metrics': {k: float(v) for k, v in best_result.items() 
                       if isinstance(v, (int, float, np.number)) and k != 'params'}
    }
    
    config_file = output_dir / 'optimal_configuration.json'
    with open(config_file, 'w') as f:
        json.dump(optimal_config, f, indent=2)
    
    print(f"\nOptimal configuration saved to {config_file}")
    print(f"Best {ranking_metric}: {optimal_config['best_score']:.4f}")


if __name__ == '__main__':
    main()
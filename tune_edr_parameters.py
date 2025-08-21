#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Tuning Script for EDR/LCSS Metrics

Systematically explores different parameter configurations to find optimal settings
for capturing functional similarity in eigenvalue evolution data.

Usage:
    python tune_edr_parameters.py                    # Run coarse grid search
    python tune_edr_parameters.py --fine-tune        # Run fine-tuned search
    python tune_edr_parameters.py --custom-config config.json  # Use custom parameter grid
"""

from __future__ import annotations
import os
import glob
import pathlib
import numpy as np
import multiprocessing as mp
from functools import partial
import argparse
import time
import json
import itertools
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

# Import functions from the original parallel script
# We'll copy the core functions here to maintain independence

# ========================
# Core functions (copied from mean_dtw_parallel.py)
# ========================

def _load_one(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one file and return (L, t, name)."""
    p = pathlib.Path(path)
    name = p.stem

    if p.suffix.lower() == ".npz":
        d = np.load(p, allow_pickle=False)
        if "eigenvalue_matrix" not in d or "time_vector" not in d:
            raise ValueError(f"{p}: .npz must have 'eigenvalue_matrix' and 'time_vector'.")
        L = np.asarray(d["eigenvalue_matrix"], dtype=float)
        t = np.asarray(d["time_vector"], dtype=float)

    elif p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=True)
        if isinstance(arr, dict):
            L = np.asarray(arr["eigenvalue_matrix"], dtype=float)
            t = np.asarray(arr["time_vector"], dtype=float)
        else:
            L = np.asarray(arr, dtype=float)
            if L.ndim != 2:
                raise ValueError(f"{p}: raw .npy must be 2D (T x n).")
            T = L.shape[0]
            t = np.linspace(0.0, 1.0, T)

    elif p.suffix.lower() in (".csv", ".txt"):
        L = np.loadtxt(p, delimiter="," if p.suffix.lower() == ".csv" else None, dtype=float)
        if L.ndim != 2:
            raise ValueError(f"{p}: CSV/TXT must be a 2D table (T x n).")
        T = L.shape[0]
        t = np.linspace(0.0, 1.0, T)
    else:
        raise ValueError(f"Unsupported file type: {p}")

    if L.shape[0] != t.shape[0] and L.shape[1] == t.shape[0]:
        L = L.T

    order = np.argsort(t)
    return L[order, :], t[order], name

def _preprocess(L: np.ndarray, eps: float, log1p: bool) -> np.ndarray:
    A = np.asarray(L, float).copy()
    A[A < 0] = 0.0
    if eps > 0:
        A[A < eps] = eps
    if log1p:
        A = np.log1p(A)
    return A

def _global_normalize(mats: list[np.ndarray], mode: str | None) -> list[np.ndarray]:
    if mode is None or mode == "none":
        return mats
    vals = np.concatenate([M.ravel() for M in mats])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        s = 1.0
    elif mode == "p95":
        s = float(np.percentile(vals, 95.0))
    elif mode == "max":
        s = float(np.max(vals))
    else:
        raise ValueError("NORMALIZE_MODE must be one of {'p95','max','none'}")
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return [M / s for M in mats]

def _resample_curve(y: np.ndarray, t: np.ndarray, t_common: np.ndarray) -> np.ndarray:
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    return np.interp(t_common, t, y, left=y[0], right=y[-1])

def _mean_curve(L: np.ndarray) -> np.ndarray:
    return np.nanmean(L, axis=1)

def edr_distance(a: np.ndarray, b: np.ndarray, eps: float, gap_cost: float = 1.0) -> float:
    """Normalized Edit Distance on Real sequences (EDR)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    N, M = len(a), len(b)
    dp = np.zeros((N + 1, M + 1), dtype=float)
    for i in range(1, N + 1):
        dp[i, 0] = i * gap_cost
    for j in range(1, M + 1):
        dp[0, j] = j * gap_cost
    for i in range(1, N + 1):
        ai = a[i - 1]
        for j in range(1, M + 1):
            sub = dp[i - 1, j - 1] + (0.0 if abs(ai - b[j - 1]) <= eps else 1.0)
            dp[i, j] = min(sub, dp[i - 1, j] + gap_cost, dp[i, j - 1] + gap_cost)
    return float(dp[N, M] / max(N, M))

def lcss_distance(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    """Normalized LCSS dissimilarity: 1 - LCSS / min(N, M)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    N, M = len(a), len(b)
    dp = np.zeros((N + 1, M + 1), dtype=int)
    for i in range(1, N + 1):
        ai = a[i - 1]
        for j in range(1, M + 1):
            if abs(ai - b[j - 1]) <= eps:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    lcss = dp[N, M]
    return 1.0 - (lcss / float(min(N, M)))

# ========================
# Parameter configuration
# ========================

# Default parameter grids
COARSE_GRID = {
    'K': [250, 500, 750],
    'USE_LOG1P': [True, False],
    'NORMALIZE_MODE': ['none', 'p95', 'max'],
    'TOP_K_EIGS': [None, 25, 50, -25],
    'REL_EPS_MULT': [0.05, 0.1, 0.2],
    'EDR_GAP_COST': [1.0, 1.5]
}

FINE_GRID = {
    'K': [500],
    'USE_LOG1P': [True],
    'NORMALIZE_MODE': ['p95', 'max'],
    'TOP_K_EIGS': [25, 50],
    'REL_EPS_MULT': [0.03, 0.05, 0.07, 0.1, 0.15],
    'EDR_GAP_COST': [0.8, 1.0, 1.2]
}

# Fixed parameters
FIXED_PARAMS = {
    'DATA_DIR': 'eigenvalueData',
    'EPS_FLOOR': 1e-10,
    'ABS_EPS': None
}

# ========================
# Parallel worker functions for distance computation
# ========================

def compute_edr_pair_worker(args: tuple) -> tuple[int, int, float]:
    """Worker function for computing a single EDR distance pair."""
    i, j, mean_i, mean_j, eps, gap_cost = args
    d = edr_distance(mean_i, mean_j, eps=eps, gap_cost=gap_cost)
    return i, j, d

def evaluate_configuration(config: Dict[str, Any], data_cache: Dict, workers: int = 4, verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single parameter configuration using parallel distance computation.
    
    Args:
        config: Parameter configuration to test
        data_cache: Pre-loaded data to avoid repeated file loading
        workers: Number of parallel workers for distance computation
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with configuration and metrics
    """
    try:
        if verbose:
            print(f"Evaluating config: {config}")
        
        # Extract parameters
        K = config['K']
        USE_LOG1P = config['USE_LOG1P']
        NORMALIZE_MODE = config['NORMALIZE_MODE']
        TOP_K_EIGS = config['TOP_K_EIGS']
        REL_EPS_MULT = config['REL_EPS_MULT']
        EDR_GAP_COST = config['EDR_GAP_COST']
        
        # Use cached data
        raw = data_cache['raw']
        names = data_cache['names']
        
        # Apply TOP_K_EIGS selection
        if TOP_K_EIGS is not None:
            if TOP_K_EIGS > 0:
                processed_raw = [(L[:, :TOP_K_EIGS], t) for L, t in raw]
            else:
                k = abs(TOP_K_EIGS)
                processed_raw = [(L[:, -k:], t) for L, t in raw]
        else:
            processed_raw = raw
        
        # Common time grid
        t_min = max(t[0] for _, t in processed_raw)
        t_max = min(t[-1] for _, t in processed_raw)
        if not (t_max > t_min):
            return {'config': config, 'error': 'No overlapping time domain'}
        tg = np.linspace(t_min, t_max, K)
        
        # Preprocess
        pre = [_preprocess(L, FIXED_PARAMS['EPS_FLOOR'], USE_LOG1P) for L, _ in processed_raw]
        pre = _global_normalize(pre, NORMALIZE_MODE)
        
        # Mean curves
        means = []
        for (L, t), P in zip(processed_raw, pre):
            m = _mean_curve(P)
            means.append(_resample_curve(m, t, tg))
        
        # Epsilon calculation
        if FIXED_PARAMS['ABS_EPS'] is not None:
            eps = float(FIXED_PARAMS['ABS_EPS'])
        else:
            amps = [float(np.std(m)) for m in means]
            median_std = float(np.median(amps)) if amps else 1.0
            eps = float(REL_EPS_MULT * median_std)
        if not np.isfinite(eps) or eps <= 0:
            eps = 1e-3
        
        # Compute pairwise distances in parallel (FIXED: Now using parallel computation)
        N = len(means)
        D = np.zeros((N, N), dtype=float)
        
        # Generate all unique pairs for parallel computation
        pairs = [
            (i, j, means[i], means[j], eps, EDR_GAP_COST)
            for i in range(N) for j in range(i + 1, N)
        ]
        
        if pairs:  # Only use parallel if we have pairs to compute
            # Compute distances in parallel
            with mp.Pool(workers) as pool:
                results = pool.map(compute_edr_pair_worker, pairs)
            
            # Fill distance matrix (symmetric)
            for i, j, d in results:
                D[i, j] = D[j, i] = d
        
        # Calculate separation metrics
        metrics = calculate_separation_metrics(D, names)
        metrics['epsilon'] = eps
        
        return {
            'config': config,
            'metrics': metrics,
            'distance_matrix': D.tolist(),
            'model_names': names
        }
        
    except Exception as e:
        return {'config': config, 'error': str(e)}

def calculate_separation_metrics(D: np.ndarray, names: List[str]) -> Dict[str, float]:
    """Calculate separation metrics for functional similarity assessment."""
    # Classify models
    trained_indices = [i for i, name in enumerate(names) if 'trained' in name.lower()]
    random_indices = [i for i, name in enumerate(names) if 'random' in name.lower()]
    
    if not trained_indices or not random_indices:
        return {'separation_ratio': 0.0, 'margin': -999.0, 'error': 'Insufficient model types'}
    
    # Within-trained distances
    trained_dists = [D[i, j] for i in trained_indices for j in trained_indices if i < j]
    
    # Cross distances (trained vs random)
    cross_dists = [D[i, j] for i in trained_indices for j in random_indices]
    
    if not trained_dists or not cross_dists:
        return {'separation_ratio': 0.0, 'margin': -999.0, 'error': 'No valid distances'}
    
    # Core metrics
    mean_within = np.mean(trained_dists)
    mean_cross = np.mean(cross_dists)
    max_within = np.max(trained_dists)
    min_cross = np.min(cross_dists)
    
    separation_ratio = mean_cross / mean_within if mean_within > 0 else 0.0
    margin = min_cross - max_within
    
    # Additional metrics
    std_within = np.std(trained_dists)
    std_cross = np.std(cross_dists)
    
    return {
        'separation_ratio': separation_ratio,
        'margin': margin,
        'mean_within_trained': mean_within,
        'mean_cross': mean_cross,
        'std_within_trained': std_within,
        'std_cross': std_cross,
        'max_within_trained': max_within,
        'min_cross': min_cross,
        'n_trained': len(trained_indices),
        'n_random': len(random_indices),
        'n_within_pairs': len(trained_dists),
        'n_cross_pairs': len(cross_dists)
    }

def load_file_worker(f: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Worker function for parallel file loading."""
    try:
        return _load_one(f)
    except Exception as e:
        print(f"[WARN] Skipping {f}: {e}")
        return None

def load_data(data_dir: str, workers: int = 4) -> Dict[str, Any]:
    """Load all data files and return cached data."""
    print("[INFO] Loading data files...")
    
    patterns = ["*.npz", "*.npy", "*.csv", "*.txt"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat)))
    files = sorted(files)
    
    if not files:
        raise ValueError(f"No files found in {data_dir}")
    
    # Load in parallel
    with mp.Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(load_file_worker, files),
            total=len(files),
            desc="Loading files"
        ))
    
    raw = []
    names = []
    for r in results:
        if r is not None:
            L, t, name = r
            raw.append((L, t))
            names.append(name)
    
    print(f"[INFO] Loaded {len(raw)} files successfully")
    
    return {
        'raw': raw,
        'names': names,
        'n_files': len(raw)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Tune EDR/LCSS parameters for optimal functional similarity capture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help='Number of worker processes')
    parser.add_argument('--fine-tune', action='store_true', help='Use fine-tuned parameter grid')
    parser.add_argument('--custom-config', type=str, help='Path to custom parameter grid JSON file')
    parser.add_argument('--output-dir', type=str, default='tuning_results', help='Output directory for results')
    parser.add_argument('--max-configs', type=int, help='Maximum number of configurations to test')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] Parameter tuning with {args.workers} workers")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Load parameter grid
    if args.custom_config:
        with open(args.custom_config, 'r') as f:
            param_grid = json.load(f)
        print(f"[INFO] Loaded custom parameter grid from {args.custom_config}")
    elif args.fine_tune:
        param_grid = FINE_GRID
        print("[INFO] Using fine-tuned parameter grid")
    else:
        param_grid = COARSE_GRID
        print("[INFO] Using coarse parameter grid")
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    if args.max_configs and len(combinations) > args.max_configs:
        combinations = combinations[:args.max_configs]
        print(f"[INFO] Limited to {args.max_configs} configurations")
    
    print(f"[INFO] Total configurations to test: {len(combinations)}")
    
    # Load data once
    data_cache = load_data(FIXED_PARAMS['DATA_DIR'], args.workers)
    
    # Evaluate all configurations
    print("[INFO] Starting parameter evaluation...")
    start_time = time.time()
    
    results = []
    best_separation = 0.0
    best_config = None
    
    for i, combo in enumerate(tqdm(combinations, desc="Evaluating configurations")):
        config = dict(zip(param_names, combo))
        
        result = evaluate_configuration(config, data_cache, workers=args.workers, verbose=args.verbose)
        results.append(result)
        
        # Track best configuration
        if 'metrics' in result and result['metrics']['separation_ratio'] > best_separation:
            best_separation = result['metrics']['separation_ratio']
            best_config = result
        
        # Save intermediate results every 10 configurations
        if (i + 1) % 10 == 0 or i == len(combinations) - 1:
            results_file = output_dir / f"results_batch_{i//10 + 1}.json"
            with open(results_file, 'w') as f:
                json.dump(results[max(0, i-9):i+1], f, indent=2)
    
    # Save all results
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best configuration
    if best_config:
        best_file = output_dir / "best_config.json"
        with open(best_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n[BEST] Configuration found!")
        print(f"Separation ratio: {best_separation:.4f}")
        print(f"Margin: {best_config['metrics']['margin']:.4f}")
        print("Parameters:")
        for key, value in best_config['config'].items():
            print(f"  {key}: {value}")
    
    # Summary statistics
    valid_results = [r for r in results if 'metrics' in r]
    if valid_results:
        separations = [r['metrics']['separation_ratio'] for r in valid_results]
        margins = [r['metrics']['margin'] for r in valid_results]
        
        print(f"\n[SUMMARY]")
        print(f"Valid configurations: {len(valid_results)}/{len(results)}")
        print(f"Separation ratio - Best: {max(separations):.4f}, Mean: {np.mean(separations):.4f}, Std: {np.std(separations):.4f}")
        print(f"Margin - Best: {max(margins):.4f}, Mean: {np.mean(margins):.4f}, Std: {np.std(margins):.4f}")
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Tuning completed in {total_time:.2f} seconds")
    print(f"[INFO] Results saved to {output_dir}")

if __name__ == "__main__":
    main()
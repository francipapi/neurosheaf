#!/usr/bin/env python3
"""
Alternative distance metrics for mean eigenvalue curves.

This script uses the EXACT same preprocessing pipeline as elastic_mean_eigs_distance.py
but computes L2, Wasserstein, and DTW distances instead of elastic distances.

The preprocessing includes:
- Loading eigenvalue data from .npz files
- Computing mean curves (all eigenvalues if topk=0)
- Time range handling (no normalization, padding with last values)
- Resampling to common grid
- Smoothing with moving average
- Z-score amplitude normalization

Usage:
    python compute_alternative_distances.py \
      --data-dir eigenvalueData \
      --pattern "*eigenvalues.npz" \
      --resample 200 \
      --topk 0 \
      --amp-norm zscore \
      --smooth moving \
      --smooth-win 15 \
      --no-time-scaling \
      --pad-with-last \
      --out-prefix comparison \
      --n-jobs 8
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

try:
    from scipy.spatial.distance import euclidean
    from scipy.stats import wasserstein_distance
    from dtaidistance import dtw
    _HAS_SCIPY = True
    _HAS_DTW = True
except Exception as e:
    _HAS_SCIPY = False
    _HAS_DTW = False
    print(f"Warning: Missing dependencies: {e}")
    print("Install with: pip install scipy dtaidistance")

# Import the exact preprocessing functions from elastic_mean_eigs_distance.py
from elastic_mean_eigs_distance import (
    _find_files, load_run, remove_tinycnn_outliers, mean_curve,
    find_maximum_range, resample_to_grid, smooth_series, normalize_amplitude
)


def _dtw_distance_numpy(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[Union[float, int]] = None,
    *,
    cost: str = "l1",        # "l1" or "l2"
    normalize: bool = False  # divide by warping-path length
) -> float:
    """
    Classic DTW with optional Sakoe–Chiba band and normalization.
    CORRECTED IMPLEMENTATION:
    - Supports unequal-length sequences
    - Proper Sakoe-Chiba window calculation considering |n-m|
    - Standard L1/L2 costs without final sqrt
    - Optional path-length normalization

    Args:
        x, y: 1D arrays (unequal lengths allowed).
        window:
            - None  -> no band (full O(n*m))
            - float in (0,1] -> fraction of max(n, m) as radius
            - int   -> radius in indices
        cost: "l1" (|x-y|) or "l2" ((x-y)^2)
        normalize: if True, returns average per step along the optimal path.

    Returns:
        DTW distance (scalar).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("inf")

    # Determine Sakoe–Chiba radius (CORRECTED)
    if window is None:
        w = max(n, m)
    else:
        if isinstance(window, float) and window <= 1.0:
            w = int(math.ceil(window * max(n, m)))
        else:
            w = int(window)
        # CRITICAL FIX: Always at least the length difference to keep a feasible path
        w = max(w, abs(n - m))

    # DP matrix (n+1) x (m+1)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0

    # CORRECTED: Standard DTW cost functions
    if cost == "l2":
        def local(a, b):  # squared difference (no sqrt at end)
            d = a - b
            return d * d
    elif cost == "l1":
        def local(a, b):  # absolute difference (standard DTW default)
            return abs(a - b)
    else:
        raise ValueError("cost must be 'l1' or 'l2'.")

    # CORRECTED: Proper DTW recurrence with window constraints
    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        xi = x[i - 1]
        for j in range(j_lo, j_hi + 1):
            c = local(xi, y[j - 1])
            # Standard DTW recurrence: insertion, deletion, match
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    dist = D[n, m]

    # CORRECTED: Optional path-length normalization
    if normalize:
        # Backtrack to compute actual path length
        i, j = n, m
        path_length = 0
        while i > 0 or j > 0:
            path_length += 1
            if i == 0:
                j -= 1
                continue
            if j == 0:
                i -= 1
                continue
            # Choose the path that led to current cell
            a, b, c = D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]
            if c <= a and c <= b:
                i, j = i - 1, j - 1  # diagonal (match)
            elif a < b:
                i -= 1  # vertical (insertion)
            else:
                j -= 1  # horizontal (deletion)

        if path_length > 0:
            dist /= path_length

    return float(dist)


def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two curves."""
    return float(np.sqrt(np.sum((x - y) ** 2)))


def wasserstein_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute 1D Wasserstein distance between two curves.
    Treats curves as empirical distributions with uniform weights.
    """
    if not _HAS_SCIPY:
        # Fallback to L2 distance if scipy not available
        return l2_distance(x, y)

    # Create uniform weights
    n = len(x)
    weights = np.ones(n) / n

    # Use scipy's wasserstein_distance
    return float(wasserstein_distance(x, y, weights, weights))


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float = 0.1,
    *,
    cost: str = "l1",
    normalize: bool = False
) -> float:
    """
    Wrapper DTW. Uses dtaidistance if available; otherwise NumPy implementation.

    Args:
        x, y: 1D arrays
        window: Sakoe–Chiba band as fraction (default 0.1). You can pass an int radius.
        cost: "l1" or "l2" (l2 means squared local cost)
        normalize: average by warping-path length if True

    Returns:
        DTW distance.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    # Try fast library path with equivalent semantics
    if _HAS_DTW and isinstance(window, float) and window <= 1.0 and not normalize and cost == "l1":
        # dtaidistance expects an integer radius in indices
        radius = int(math.ceil(window * max(len(x), len(y))))
        radius = max(radius, abs(len(x) - len(y)))
        try:
            return float(dtw.distance_fast(x, y, window=radius))
        except Exception:
            pass  # fall back to NumPy

    # Fallback: robust NumPy reference
    return _dtw_distance_numpy(x, y, window=window, cost=cost, normalize=normalize)


def compute_distance_matrix(curves: List[np.ndarray], distance_func, n_jobs: int = -1, **kwargs) -> np.ndarray:
    """Compute pairwise distance matrix using specified distance function."""
    N = len(curves)
    D = np.zeros((N, N), dtype=np.float64)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def _compute_pair(i: int, j: int) -> Tuple[int, int, float]:
        try:
            d = distance_func(curves[i], curves[j], **kwargs)
            return i, j, d
        except Exception as e:
            print(f"Warning: Distance computation failed for pair ({i}, {j}): {e}")
            return i, j, 0.0

    if _HAS_JOBLIB and n_jobs != 1 and len(pairs) > 10:
        print(f"Computing {len(pairs)} pairwise distances in parallel...")
        results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
            delayed(_compute_pair)(i, j) for (i, j) in pairs
        )
    else:
        print(f"Computing {len(pairs)} pairwise distances sequentially...")
        results = []
        for idx, (i, j) in enumerate(pairs):
            if idx % 100 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(pairs)} pairs computed")
            results.append(_compute_pair(i, j))

    # Fill symmetric matrix
    for (i, j, d) in results:
        D[i, j] = d
        D[j, i] = d

    return D


def save_distance_matrix(distance_matrix: np.ndarray, filenames: List[str],
                        metric_name: str, out_prefix: str):
    """Save distance matrix and index in the same format as elastic script."""

    # Save .npy file
    npy_file = f"{out_prefix}_{metric_name}_distance.npy"
    np.save(npy_file, distance_matrix)
    print(f"Saved distance matrix to {npy_file}")

    # Save .csv file
    csv_file = f"{out_prefix}_{metric_name}_distance.csv"
    np.savetxt(csv_file, distance_matrix, delimiter=',', fmt='%.8e')
    print(f"Saved distance matrix to {csv_file}")

    # Save index file
    index_file = f"{out_prefix}_{metric_name}_index.json"
    with open(index_file, 'w') as f:
        json.dump(filenames, f, indent=2)
    print(f"Saved file index to {index_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute L2, Wasserstein, and DTW distances with identical preprocessing to elastic_mean_eigs_distance.py"
    )

    # Same arguments as elastic script
    parser.add_argument('--data-dir', type=str, required=True, help='Directory to scan recursively for data files')
    parser.add_argument('--pattern', type=str, default='*', help='Glob pattern to match (e.g., "*eigenvalues.npz")')
    parser.add_argument('--resample', type=int, default=200, help='Number of points in common time grid')
    parser.add_argument('--topk', type=int, default=0, help='>0: mean of top-k eigs; <0: mean of bottom-k; 0: mean of all')
    parser.add_argument('--amp-norm', type=str, default='zscore', choices=['zscore','unit','p95','none'],
                       help='Amplitude normalization for scale-free comparison')
    parser.add_argument('--smooth', type=str, default='moving', choices=['moving','none'], help='Smoothing method')
    parser.add_argument('--smooth-win', type=int, default=15, help='Window size for smoothing')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs (joblib). Use 1 for sequential.')
    parser.add_argument('--outlier-method', type=str, default='none',
                       choices=['none', 'trajectory_end'],
                       help='Method for outlier detection')
    parser.add_argument('--out-prefix', type=str, default='alternative', help='Prefix for output files')
    parser.add_argument('--no-time-scaling', action='store_true',
                       help='Do not normalize time to [0,1]; use actual time ranges')
    parser.add_argument('--pad-with-last', action='store_true',
                       help='When using --no-time-scaling, pad curves with last values to cover maximum range')

    # Distance-specific arguments
    parser.add_argument('--dtw-window', type=float, default=0.1,
                       help='DTW window constraint as fraction of sequence length')
    parser.add_argument('--metrics', type=str, nargs='+', default=['l2', 'wasserstein', 'dtw'],
                       choices=['l2', 'wasserstein', 'dtw'],
                       help='Distance metrics to compute')

    args = parser.parse_args()

    # Find and process files using exact same logic as elastic script
    data_dir = Path(args.data_dir)
    files = _find_files(data_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files found in {data_dir} matching pattern '{args.pattern}'. Supported: .npz, .npy")

    print(f"Found {len(files)} files.")

    # Preprocessing pipeline - IDENTICAL to elastic script
    curves: List[np.ndarray] = []
    index: List[str] = []
    common_time: Optional[np.ndarray] = None

    # Determine time normalization settings
    normalize_time = not args.no_time_scaling
    use_padding = args.pad_with_last
    common_range = None

    # If not normalizing time, determine the common range first
    if not normalize_time:
        # First pass: collect all time arrays to determine common range
        files_data = []
        for p in files:
            try:
                E, t = load_run(p)
                E, t = remove_tinycnn_outliers(E, t, p.name, outlier_method=args.outlier_method)
                files_data.append((t, E))
            except Exception as e:
                print(f"[WARN] Failed to load {p.name}: {e}")
                continue

        if use_padding:
            common_range = find_maximum_range(files_data)
            print(f"[INFO] Maximum time range: [{common_range[0]:.6f}, {common_range[1]:.6f}]")
        else:
            # For overlapping mode, we'd use find_overlapping_range here
            # but since we're matching the elastic script exactly, use maximum range
            common_range = find_maximum_range(files_data)

    # Second pass: process all files with identical preprocessing
    for p in files:
        try:
            # Load data
            E, t = load_run(p)
            E, t = remove_tinycnn_outliers(E, t, p.name, outlier_method=args.outlier_method)

            # Compute mean curve
            y = mean_curve(E, topk=args.topk)

            # Resample to common grid
            t_new, y_new = resample_to_grid(
                t, y, N=args.resample,
                normalize_time=normalize_time,
                common_range=common_range,
                use_padding=use_padding
            )

            # Smooth
            y_smooth = smooth_series(y_new, method=args.smooth, win=args.smooth_win)

            # Normalize amplitude
            y_final = normalize_amplitude(y_smooth, mode=args.amp_norm)

            curves.append(y_final)
            index.append(p.name)

            if common_time is None:
                common_time = t_new

        except Exception as e:
            print(f"[WARN] Failed to process {p.name}: {e}")
            continue

    if len(curves) == 0:
        raise SystemExit("No curves could be processed successfully")

    print(f"Successfully processed {len(curves)} curves")

    # Compute distance matrices for each requested metric
    for metric in args.metrics:
        print(f"\n{'='*60}")
        print(f"Computing {metric.upper()} distances")
        print(f"{'='*60}")

        if metric == 'l2':
            distance_matrix = compute_distance_matrix(curves, l2_distance, n_jobs=args.n_jobs)
        elif metric == 'wasserstein':
            distance_matrix = compute_distance_matrix(curves, wasserstein_distance_1d, n_jobs=args.n_jobs)
        elif metric == 'dtw':
            distance_matrix = compute_distance_matrix(curves, dtw_distance, n_jobs=args.n_jobs,
                                                     window=args.dtw_window)
        else:
            print(f"Warning: Unknown metric {metric}, skipping")
            continue

        # Save results
        save_distance_matrix(distance_matrix, index, metric, args.out_prefix)

        # Print basic statistics
        upper_tri = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        print(f"{metric.upper()} distance statistics:")
        print(f"  Mean: {np.mean(upper_tri):.6f}")
        print(f"  Std:  {np.std(upper_tri):.6f}")
        print(f"  Min:  {np.min(upper_tri):.6f}")
        print(f"  Max:  {np.max(upper_tri):.6f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed {len(curves)} curves with identical preprocessing to elastic_mean_eigs_distance.py")
    print(f"Computed {len(args.metrics)} distance matrices: {', '.join(args.metrics)}")
    print(f"Output files: {args.out_prefix}_{{metric}}_distance.npy/csv and {args.out_prefix}_{{metric}}_index.json")
    print(f"\nTo compute ARI, run:")
    for metric in args.metrics:
        print(f"  python compute_elastic_ari_enhanced.py --distance-file {args.out_prefix}_{metric}_distance.npy --index-file {args.out_prefix}_{metric}_index.json")


if __name__ == '__main__':
    main()
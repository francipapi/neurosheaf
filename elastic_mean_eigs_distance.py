#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elastic distance between mean eigenvalue evolution curves.

This script:
  • Recursively scans a data directory for eigenvalue run files (.npz, .npy).
  • Loads each run, computes the mean eigenvalue evolution curve (optionally top-k/bottom-k).
  • Resamples curves to a common grid [0,1] with N points.
  • (Optional) Smooths.
  • Normalizes amplitude (z-score by default) for scale-free comparison.
  • Converts curves to SRVF (SRSF) representation: q(t) = sign(f') * sqrt(|f'|).
  • (NEW) Optionally applies robust SRVF normalization (median/MAD) right before distances.
  • Computes an elastic distance between all pairs via DP warping on SRVFs with a warp penalty.
  • Parallelizes pairwise distance computation.
  • Saves the distance matrix (.npy and .csv) and an index file (.json).

The elastic distance here is a practical approximation: we use a DTW-like dynamic program
on SRVFs under a window constraint, add a penalty for off-diagonal alignment (warp size),
then measure the L2 distance between q1 and the q2 aligned to q1's grid.

Usage (example):
---------------
# Standard normalized time [0,1] (default behavior)
python elastic_mean_eigs_distance.py --data-dir ./eigenvalueData --pattern "*eigenvalues.npz" \
    --resample 300 --topk 0 --window-frac 0.2 --lambda-warp 0.05 \
    --amp-norm zscore --smooth moving --smooth-win 9 \
    --use-robust-norm \
    --out-prefix elastic_eigs

# Use actual time ranges with overlapping intersection
python elastic_mean_eigs_distance.py --data-dir ./eigenvalueData --pattern "*eigenvalues.npz" \
    --no-time-scaling --resample 300 --out-prefix elastic_eigs_unnorm

# Use actual time ranges with maximum range and padding
python elastic_mean_eigs_distance.py --data-dir ./eigenvalueData --pattern "*eigenvalues.npz" \
    --no-time-scaling --pad-with-last --resample 300 --out-prefix elastic_eigs_padded

# Generate plot to visualize the curves being compared
python elastic_mean_eigs_distance.py --data-dir ./eigenvalueData --pattern "*eigenvalues.npz" \
    --plot --plot-output my_curves.png --resample 300 --out-prefix elastic_eigs

Inputs:
-------
Supported file types: .npz, .npy
  - .npz expected keys (we try in order): ('eigenvalue_matrix','time_vector'), ('E','t'), or fallbacks.
  - .npy may contain either a dict with those keys, or a 2D array (time x eigen)
Heuristics try to infer orientation. If no time vector is present we assume uniform in [0,1].

Mean curve:
-----------
If --topk > 0  => per-time-step mean of the top-k eigenvalues
If --topk < 0  => per-time-step mean of the bottom-k eigenvalues
If --topk == 0 => mean of all eigenvalues

Outputs:
--------
  {out-prefix}_distance.npy : NxN symmetric matrix of elastic distances (float64)
  {out-prefix}_distance.csv : same in CSV
  {out-prefix}_index.json   : list of file basenames in matrix order
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


def _find_files(data_dir: Path, pattern: str) -> List[Path]:
    exts = {'.npz', '.npy'}
    files = []
    for p in data_dir.rglob(pattern):
        if p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (E, t) with E shape (T, K) time-by-eigen; t shape (T,).
    """
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        E = None
        t = None

        # Common key patterns
        for Ek in ('eigenvalue_matrix', 'E', 'eigenvalues', 'matrix'):
            if Ek in keys:
                E = z[Ek]
                break
        for tk in ('time_vector', 't', 'times', 'time'):
            if tk in keys:
                t = z[tk]
                break

        # Try to infer if raw array stored without keys
        if E is None and len(keys) == 1:
            only = list(keys)[0]
            arr = z[only]
            if arr.ndim == 2:
                E = arr
            elif arr.ndim == 1:
                # Single curve; treat as 1 eigenvalue across time
                E = arr[:, None]

        if E is None:
            raise ValueError(f"{path.name}: could not find eigenvalue matrix in keys={keys}")

        # Ensure 2D
        if E.ndim != 2:
            raise ValueError(f"{path.name}: expected 2D eigenvalue matrix, got shape {E.shape}")

        # Orientation: prefer (T, K). If K > T, we *guess* E is (K, T) and transpose.
        T, K = E.shape
        if T < K:
            E = E.T
            T, K = E.shape

        # Time vector
        if t is None:
            t = np.linspace(0.0, 1.0, T)
        else:
            t = np.asarray(t).reshape(-1)
            if t.size != T:
                # Try the other orientation
                if t.size == E.shape[1]:
                    E = E.T
                    T, K = E.shape
                else:
                    raise ValueError(f"{path.name}: time vector length {t.size} does not match matrix (T={T})")

        # Ensure strictly increasing t
        if not np.all(np.diff(t) > 0):
            idx = np.argsort(t)
            t = t[idx]
            E = E[idx, :]

        return E.astype(np.float64), t.astype(np.float64)


def _load_npy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        E = obj
        T, K = E.shape
        if T < K:
            E = E.T
            T, K = E.shape
        t = np.linspace(0.0, 1.0, T)
        return E.astype(np.float64), t.astype(np.float64)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        # maybe dict-like
        if obj.size == 1 and isinstance(obj.item(), dict):
            d = obj.item()
            if 'eigenvalue_matrix' in d:
                E = np.array(d['eigenvalue_matrix'])
                t = np.array(d.get('time_vector', np.linspace(0.0, 1.0, E.shape[0])))
                if E.ndim != 2:
                    raise ValueError(f"{path.name}: eigenvalue_matrix not 2D")
                if t.size != E.shape[0]:
                    if t.size == E.shape[1]:
                        E = E.T
                    else:
                        raise ValueError(f"{path.name}: time size mismatch")
                return E.astype(np.float64), t.astype(np.float64)
    elif isinstance(obj, dict):
        d = obj
        if 'eigenvalue_matrix' in d:
            E = np.array(d['eigenvalue_matrix'])
            t = np.array(d.get('time_vector', np.linspace(0.0, 1.0, E.shape[0])))
            if E.ndim != 2:
                raise ValueError(f"{path.name}: eigenvalue_matrix not 2D")
            if t.size != E.shape[0]:
                if t.size == E.shape[1]:
                    E = E.T
                else:
                    raise ValueError(f"{path.name}: time size mismatch")
            return E.astype(np.float64), t.astype(np.float64)

    raise ValueError(f"{path.name}: unsupported .npy content (expect 2D array or dict)")


def load_run(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == '.npz':
        return _load_npz(path)
    elif path.suffix.lower() == '.npy':
        return _load_npy(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def remove_tinycnn_outliers(E: np.ndarray, t: np.ndarray, filename: str,
                           outlier_method: str = 'none') -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from eigenvalue data, specifically the last point from TRAINED TinyCNN models.
    """
    if outlier_method == 'trajectory_end':
        filename_lower = filename.lower()
        is_tinycnn = 'tinycnn' in filename_lower
        is_trained = (is_tinycnn and 'mnist' in filename_lower and 'random' not in filename_lower)
        if is_trained and len(t) > 1:
            print(f"[INFO] {filename}: Removing last datapoint (trained TinyCNN model)")
            return E[:-1, :], t[:-1]
        elif is_tinycnn and 'random' in filename_lower:
            print(f"[INFO] {filename}: Keeping all points (random TinyCNN model)")
    return E, t


def mean_curve(E: np.ndarray, topk: int = 0) -> np.ndarray:
    """Compute per-time-step mean eigenvalue curve."""
    if E.ndim != 2:
        raise ValueError("E must be 2D (T, K)")
    if topk == 0:
        return np.mean(E, axis=1)
    T, K = E.shape
    k = min(abs(topk), K)
    if topk > 0:
        idx = np.argpartition(E, K - k, axis=1)[:, K - k:]
        vals = np.take_along_axis(E, idx, axis=1)
        return np.mean(vals, axis=1)
    else:
        idx = np.argpartition(E, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(E, idx, axis=1)
        return np.mean(vals, axis=1)


def classify_model(model_name: str) -> Dict[str, str]:
    """
    Classify model based on filename.
    """
    name_lower = model_name.lower()

    # Digits dataset models first - actual current patterns
    if name_lower.startswith('digits_hourglass_random'):
        architecture = 'Digits-Hourglass'
        status = 'Random'
    elif name_lower.startswith('digits_pyramid_seed'):
        architecture = 'Digits-Pyramid'
        status = 'Trained'
    elif name_lower.startswith('digits_hourglass') and 'seed' in name_lower:
        architecture = 'Digits-Hourglass'
        status = 'Trained'
    elif name_lower.startswith('digits_pyramid_random') or name_lower.startswith('digits_pyramid') and 'random' in name_lower:
        architecture = 'Digits-Pyramid'
        status = 'Random'
    # Legacy patterns
    elif name_lower.startswith('digits_mlp_seed') or (name_lower.startswith('digits_mlp') and 'seed' in name_lower):
        architecture = 'Digits-MLP'
        status = 'Trained'
    elif name_lower.startswith('digits_mlp_random') or (name_lower.startswith('digits_mlp') and 'random' in name_lower):
        architecture = 'Digits-MLP'
        status = 'Random'
    elif name_lower.startswith('slimcnn_digits_seed') or (name_lower.startswith('slimcnn_digits') and 'seed' in name_lower):
        architecture = 'Digits-CNN'
        status = 'Trained'
    elif name_lower.startswith('slimcnn_digits_random') or (name_lower.startswith('slimcnn_digits') and 'random' in name_lower):
        architecture = 'Digits-CNN'
        status = 'Random'
    # MNIST-specific 
    elif name_lower.startswith('mlp4layer_mnist'):
        architecture = 'MNIST-MLP4'
        status = 'Trained'
    elif name_lower.startswith('tinycnn_mnist'):
        architecture = 'MNIST-CNN'
        status = 'Trained'
    elif name_lower.startswith('tinycnn_random'):
        architecture = 'MNIST-CNN'
        status = 'Random'
    elif name_lower.startswith('tinycnn'):
        architecture = 'MNIST-CNN'
        if any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        elif any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        else:
            status = 'Trained'
    elif name_lower.startswith('mnist_mlp_random'):
        architecture = 'MNIST-MLP'
        status = 'Random'
    elif name_lower.startswith('mnist_mlp'):
        architecture = 'MNIST-MLP'
        if any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        elif any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        else:
            status = 'Unknown'
    # Generic
    elif any(x in name_lower for x in ['custom', 'conv']):
        architecture = 'Custom'
        if any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        elif any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        else:
            status = 'Unknown'
    elif 'mlp' in name_lower:
        architecture = 'MLP'
        if any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        elif any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        else:
            status = 'Unknown'
    else:
        architecture = 'Other'
        if any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        elif any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        else:
            status = 'Unknown'

    return {
        'architecture': architecture,
        'status': status,
        'category': f"{architecture}-{status}"
    }


def find_overlapping_range(files_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    """
    Find the overlapping time range across all loaded files.
    
    Args:
        files_data: List of (time, _) tuples from all loaded files
        
    Returns:
        Tuple of (min_overlap, max_overlap) representing the common time range
    """
    if not files_data:
        return 0.0, 1.0
    
    # Collect time ranges from all files
    time_ranges = []
    for t, _ in files_data:
        if len(t) > 0 and np.isfinite(t).any():
            valid_time = t[np.isfinite(t)]
            if len(valid_time) > 0:
                time_ranges.append((valid_time.min(), valid_time.max()))
    
    if not time_ranges:
        print("[WARN] No valid time ranges found")
        return 0.0, 1.0
    
    # Find intersection of all time ranges
    min_starts = [t_range[0] for t_range in time_ranges]
    max_ends = [t_range[1] for t_range in time_ranges]
    
    overlap_start = max(min_starts)  # Latest start time
    overlap_end = min(max_ends)      # Earliest end time
    
    # Ensure we have a valid range
    if overlap_start >= overlap_end:
        print(f"[WARN] No overlapping time range found. Using full range.")
        overlap_start = min(min_starts)
        overlap_end = max(max_ends)
    
    print(f"[INFO] Overlapping time range: [{overlap_start:.6f}, {overlap_end:.6f}]")
    print(f"[INFO] Found {len(time_ranges)} files with valid time data")
    
    return overlap_start, overlap_end


def find_maximum_range(files_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    """
    Find the maximum time range across all files (union instead of intersection).
    
    Args:
        files_data: List of (time, _) tuples from all loaded files
        
    Returns:
        Tuple of (global_min, global_max) representing the full time range
    """
    if not files_data:
        return 0.0, 1.0
    
    # Collect time ranges from all files
    time_ranges = []
    for t, _ in files_data:
        if len(t) > 0 and np.isfinite(t).any():
            valid_time = t[np.isfinite(t)]
            if len(valid_time) > 0:
                time_ranges.append((valid_time.min(), valid_time.max()))
    
    if not time_ranges:
        print("[WARN] No valid time ranges found")
        return 0.0, 1.0
    
    # Find union of all time ranges (maximum extent)
    min_starts = [t_range[0] for t_range in time_ranges]
    max_ends = [t_range[1] for t_range in time_ranges]
    
    global_start = min(min_starts)  # Earliest start time
    global_end = max(max_ends)      # Latest end time
    
    print(f"[INFO] Maximum time range: [{global_start:.6f}, {global_end:.6f}]")
    print(f"[INFO] Found {len(time_ranges)} files with valid time data")
    
    return global_start, global_end


def filter_to_overlapping_range(t: np.ndarray, E: np.ndarray, 
                                overlap_start: float, overlap_end: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter time and eigenvalue data to keep only points within the overlapping range.
    
    Args:
        t: Time array
        E: Eigenvalue matrix (time x eigenvalues)
        overlap_start: Start of overlapping range
        overlap_end: End of overlapping range
        
    Returns:
        Tuple of (filtered_time, filtered_eigenvalues)
    """
    if t.size == 0 or E.size == 0:
        return t, E
        
    # Keep only data within overlapping range
    mask = (t >= overlap_start) & (t <= overlap_end)
    if not mask.any():
        return np.array([]), np.array([]).reshape(0, E.shape[1] if E.ndim == 2 else 0)
    
    t_filtered = t[mask]
    E_filtered = E[mask, :] if E.ndim == 2 else E[mask]
    
    return t_filtered, E_filtered


def resample_to_grid(t: np.ndarray, y: np.ndarray, N: int = 300, 
                     normalize_time: bool = True, common_range: Optional[Tuple[float, float]] = None,
                     use_padding: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample curve to a common grid.
    
    Args:
        t: Time array
        y: Values array 
        N: Number of points in output grid
        normalize_time: If True, normalize individual curve to [0,1]. If False, use common_range.
        common_range: (start, end) for common grid when normalize_time=False
        use_padding: If True, pad with first/last values for extrapolation
        
    Returns:
        Tuple of (t_new, y_new) arrays
    """
    t = np.asarray(t).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if t.size != y.size:
        raise ValueError("t and y must have same length")
    
    # Handle edge cases with empty or single-point data
    if t.size == 0:
        return np.array([]), np.array([])
    if t.size == 1:
        # Single point: replicate across the grid
        if normalize_time:
            t_new = np.linspace(0.0, 1.0, N)
        else:
            if common_range is None:
                raise ValueError("common_range must be provided when normalize_time=False")
            start, end = common_range
            t_new = np.linspace(start, end, N)
        y_new = np.full_like(t_new, y[0])
        return t_new, y_new
    
    if normalize_time:
        # Original behavior: normalize to [0,1]
        t_new = np.linspace(0.0, 1.0, N)
        # Map original t to [0,1]
        t01 = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.linspace(0.0, 1.0, t.size)
        y_new = np.interp(t_new, t01, y)
    else:
        # New behavior: use actual time ranges
        if common_range is None:
            raise ValueError("common_range must be provided when normalize_time=False")
        
        start, end = common_range
        t_new = np.linspace(start, end, N)
        
        if use_padding:
            # Use padding with first and last values for extrapolation
            left_value = y[0] if len(y) > 0 else 0.0
            right_value = y[-1] if len(y) > 0 else 0.0
            y_new = np.interp(t_new, t, y, left=left_value, right=right_value)
        else:
            # Standard interpolation (NaN for extrapolation)
            # When using overlapping mode, data should already be filtered,
            # so interpolation should stay within valid range
            y_new = np.interp(t_new, t, y)
    
    return t_new, y_new


def smooth_series(y: np.ndarray, method: str = 'moving', win: int = 9) -> np.ndarray:
    if method == 'none' or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    if win < 3:
        return y
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(ypad, kernel, mode='valid')


def normalize_amplitude(y: np.ndarray, mode: str = 'zscore') -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if mode == 'none':
        return y
    if mode == 'zscore':
        mu = np.mean(y)
        sd = np.std(y)
        if sd < 1e-12:
            return y * 0.0
        return (y - mu) / sd
    elif mode == 'unit':
        dt = 1.0 / max(1, (y.size - 1))
        norm = math.sqrt(np.sum(y * y) * dt)
        if norm < 1e-12:
            return y * 0.0
        return y / norm
    elif mode == 'p95':
        p95 = np.percentile(np.abs(y), 95.0)
        if p95 < 1e-12:
            return y * 0.0
        return y / p95
    else:
        raise ValueError("Unknown amplitude normalization mode")


def srvf(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Square-root velocity (slope) function for 1D time-series: q(t) = sign(f') * sqrt(|f'|)."""
    dy = np.gradient(y, 1.0 / max(1, (y.size - 1)))
    q = np.sign(dy) * np.sqrt(np.abs(dy) + eps)
    return q


# === NEW: SRVF normalization utilities =======================================

def _robust_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Median/MAD robust standardization."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # 1.4826 makes MAD consistent with std for a normal distribution
    scale = max(1.4826 * mad, eps)
    return (x - med) / scale


def normalize_srvf(q: np.ndarray, mode: str = 'none') -> np.ndarray:
    """
    Normalize SRVF vector prior to elastic distance.
    Modes:
      - 'none'   : no change
      - 'unit'   : unit L2 norm over [0,1]
      - 'zscore' : mean/std z-score on q
      - 'robust' : median/MAD robust z-score on q   (RECOMMENDED)
    """
    q = np.asarray(q, dtype=np.float64)
    if mode == 'none':
        return q
    if mode == 'unit':
        dt = 1.0 / max(1, (q.size - 1))
        n = math.sqrt(np.sum(q * q) * dt)
        return q / n if n > 1e-12 else q * 0.0
    if mode == 'zscore':
        mu = float(np.mean(q))
        sd = float(np.std(q))
        return (q - mu) / sd if sd > 1e-12 else q * 0.0
    if mode == 'robust':
        return _robust_zscore(q)
    raise ValueError("Unknown SRVF normalization mode")


# =============================================================================

def _dtw_path(cost: np.ndarray, window: int) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute minimal-cost DTW path with Sakoe–Chiba window."""
    N = cost.shape[0]
    big = 1e30
    D = np.full((N, N), big, dtype=np.float64)
    ptr = np.full((N, N), 0, dtype=np.uint8)  # 1: diag, 2: up, 3: left

    D[0, 0] = cost[0, 0]
    for i in range(N):
        jmin = max(0, i - window)
        jmax = min(N - 1, i + window)
        for j in range(jmin, jmax + 1):
            if i == 0 and j == 0:
                continue
            best = big
            move = 0
            if i > 0 and j > 0:
                v = D[i - 1, j - 1]
                if v < best:
                    best = v
                    move = 1  # diag
            if i > 0 and (j >= max(0, i - 1 - window)) and (j <= min(N - 1, i - 1 + window)):
                v = D[i - 1, j]
                if v < best:
                    best = v
                    move = 2  # up
            if j > 0 and (j - 1 >= max(0, i - window)) and (j - 1 <= min(N - 1, i + window)):
                v = D[i, j - 1]
                if v < best:
                    best = v
                    move = 3  # left
            D[i, j] = cost[i, j] + best
            ptr[i, j] = move

    # backtrack
    path: List[Tuple[int, int]] = []
    i, j = N - 1, N - 1
    path.append((i, j))
    while not (i == 0 and j == 0):
        m = ptr[i, j]
        if m == 1:
            i -= 1; j -= 1
        elif m == 2:
            i -= 1
        elif m == 3:
            j -= 1
        else:
            # Safeguard fallback
            if i > 0 and j > 0:
                i -= 1; j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return D[-1, -1], path


def elastic_distance(q1: np.ndarray, q2: np.ndarray, lambda_warp: float = 0.05, window_frac: float = 0.25) -> float:
    """Approximate elastic distance between SRVFs via DP warping with penalty."""
    assert q1.shape == q2.shape
    N = q1.size
    w = max(0, min(N - 1, int(round(window_frac * N))))

    # Local cost
    ii = np.arange(N, dtype=np.float64)
    A = (q1.reshape(-1, 1) - q2.reshape(1, -1))**2
    JJ = (ii.reshape(-1, 1) - ii.reshape(1, -1)) / float(N)
    C = A + (lambda_warp * JJ * JJ)

    total_cost, path = _dtw_path(C, w)

    # Align q2 to q1 grid via the path
    q2_aligned = np.empty_like(q1)
    from collections import defaultdict
    bucket: Dict[int, List[int]] = defaultdict(list)
    for (i, j) in path:
        bucket[i].append(j)
    for i in range(N):
        js = bucket.get(i, None)
        if not js:
            q2_aligned[i] = q2[i]  # fallback to diagonal
        else:
            q2_aligned[i] = np.mean(q2[np.array(js, dtype=int)])

    # L2 integral on [0,1]
    diff2 = (q1 - q2_aligned)**2
    dt = 1.0 / max(1, (N - 1))
    l2 = math.sqrt(np.sum(diff2) * dt)
    return float(l2)


def get_color_scheme() -> Dict[str, Dict]:
    """Define color scheme for different model categories."""
    return {
        # Generic
        'Custom-Trained':  {'color': '#1f77b4', 'linestyle': '-',  'alpha': 0.7},
        'Custom-Random':   {'color': '#1f77b4', 'linestyle': '--', 'alpha': 0.6},
        'MLP-Trained':     {'color': '#d62728', 'linestyle': '-',  'alpha': 0.7},
        'MLP-Random':      {'color': '#d62728', 'linestyle': '--', 'alpha': 0.6},
        'Other-Trained':   {'color': '#2ca02c', 'linestyle': '-',  'alpha': 0.7},
        'Other-Random':    {'color': '#2ca02c', 'linestyle': '--', 'alpha': 0.6},
        'Other-Unknown':   {'color': '#808080', 'linestyle': ':',  'alpha': 0.5},
        # Digits dataset models
        'Digits-MLP-Trained': {'color': '#e377c2', 'linestyle': '-',  'alpha': 0.8},
        'Digits-MLP-Random':  {'color': '#e377c2', 'linestyle': '--', 'alpha': 0.7},
        'Digits-CNN-Trained': {'color': '#8c564b', 'linestyle': '-',  'alpha': 0.8},
        'Digits-CNN-Random':  {'color': '#8c564b', 'linestyle': '--', 'alpha': 0.7},
        # MNIST-specific
        'MNIST-MLP4-Trained': {'color': '#ff7f0e', 'linestyle': '-',  'alpha': 0.8},
        'MNIST-MLP-Random':   {'color': '#9467bd', 'linestyle': '--', 'alpha': 0.7},
        'MNIST-MLP-Trained':  {'color': '#9467bd', 'linestyle': '-',  'alpha': 0.8},
        'MNIST-MLP-Unknown':  {'color': '#9467bd', 'linestyle': ':',  'alpha': 0.6},
        'MNIST-CNN-Trained':  {'color': '#2ca02c', 'linestyle': '-',  'alpha': 0.8},
        'MNIST-CNN-Random':   {'color': '#17becf', 'linestyle': '--', 'alpha': 0.7},
        'MNIST-CNN-Unknown':  {'color': '#808080', 'linestyle': ':',  'alpha': 0.6},
    }


def plot_mean_curves(curves: List[np.ndarray], 
                     common_time: np.ndarray,
                     model_names: List[str],
                     normalize_time: bool = True,
                     show_stats: bool = True,
                     output_file: str = "elastic_mean_curves.png") -> None:
    """
    Plot the mean eigenvalue curves used for elastic distance computation.
    
    Args:
        curves: List of processed mean curves (after all transformations)
        common_time: Common time grid used for all curves
        model_names: List of model names corresponding to curves
        normalize_time: Whether time was normalized to [0,1]
        show_stats: Whether to show mean/std statistics per category
        output_file: Output filename for the plot
    """
    if not _HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available. Skipping plot generation.")
        return
        
    if len(curves) == 0:
        print("[WARN] No curves to plot.")
        return
        
    color_scheme = get_color_scheme()
    fig, ax = plt.subplots(figsize=(14, 10))

    # Classify models and group by category
    categories: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for name, curve in zip(model_names, curves):
        classification = classify_model(name)
        category = classification['category']
        categories.setdefault(category, []).append((name, curve))

    legend_handles = []
    category_curves: Dict[str, np.ndarray] = {}

    # Plot individual curves
    for category, model_curves in categories.items():
        style = color_scheme.get(category, {'color': '#808080', 'linestyle': '-', 'alpha': 0.5})
        curves_for_stats: List[np.ndarray] = []

        for _, curve in model_curves:
            try:
                ax.plot(common_time, curve,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        alpha=style['alpha'],
                        linewidth=0.8)
                curves_for_stats.append(curve)
            except Exception as e:
                print(f"[WARN] Failed to plot curve: {e}")

        if curves_for_stats:
            category_curves[category] = np.vstack(curves_for_stats)
            legend_handles.append(mpatches.Patch(color=style['color'],
                                                 label=f"{category} (n={len(curves_for_stats)})"))

    # Plot category statistics (mean ± std)
    if show_stats and category_curves:
        for category, curves_array in category_curves.items():
            style = color_scheme.get(category, {'color': '#808080', 'linestyle': '-', 'alpha': 1.0})
            mean_curve = np.nanmean(curves_array, axis=0)
            std_curve = np.nanstd(curves_array, axis=0)
            ax.plot(common_time, mean_curve,
                    color=style['color'],
                    linestyle=style.get('linestyle', '-'),
                    alpha=1.0,
                    linewidth=2.5,
                    label=f"{category} Mean")
            ax.fill_between(common_time,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            color=style['color'],
                            alpha=0.15)

    # Configure plot
    # Determine x-axis label based on time range
    time_min, time_max = common_time.min(), common_time.max()
    if abs(time_min - 0.0) < 1e-6 and abs(time_max - 1.0) < 1e-6:
        xlabel = 'Filtration Parameter (Normalized)'
    else:
        xlabel = 'Filtration Parameter'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Mean Eigenvalue (Processed)', fontsize=12)
    ax.set_title('Mean Eigenvalue Curves Used for Elastic Distance Computation', fontsize=14, fontweight='bold')
    ax.legend(handles=legend_handles, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot saved to {output_file}")
    plt.show()


def pairwise_distance_matrix(Q: List[np.ndarray], lambda_warp: float, window_frac: float,
                             n_jobs: int = -1) -> np.ndarray:
    N = len(Q)
    D = np.zeros((N, N), dtype=np.float64)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def _compute(i: int, j: int) -> Tuple[int, int, float]:
        d = elastic_distance(Q[i], Q[j], lambda_warp=lambda_warp, window_frac=window_frac)
        return i, j, d

    if _HAS_JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10)(
            delayed(_compute)(i, j) for (i, j) in pairs
        )
    else:
        results = list(map(lambda ij: _compute(*ij), pairs))

    for (i, j, d) in results:
        D[i, j] = d
        D[j, i] = d
    return D


def main():
    ap = argparse.ArgumentParser(description="Elastic distance between mean eigenvalue evolution curves (SRVF + DP warping)")
    ap.add_argument('--data-dir', type=str, required=True, help='Directory to scan recursively for data files')
    ap.add_argument('--pattern', type=str, default='*', help='Glob pattern to match (e.g., \"*eigenvalues.npz\")')
    ap.add_argument('--resample', type=int, default=200, help='Number of points in common time grid (OPTIMIZED: was 300)')
    ap.add_argument('--topk', type=int, default=0, help='>0: mean of top-k eigs; <0: mean of bottom-k; 0: mean of all (OPTIMAL: use all)')
    ap.add_argument('--amp-norm', type=str, default='zscore', choices=['zscore','unit','p95','none'],
                    help='Amplitude normalization for scale-free comparison (OPTIMAL: zscore)')
    ap.add_argument('--smooth', type=str, default='moving', choices=['moving','none'], help='Smoothing method (OPTIMAL: moving)')
    ap.add_argument('--smooth-win', type=int, default=15, help='Window size for smoothing (OPTIMIZED: was 9, now 15)')
    ap.add_argument('--lambda-warp', type=float, default=0.1, help='Penalty strength for time warping (OPTIMIZED: was 0.05, now 0.1)')
    ap.add_argument('--window-frac', type=float, default=0.1, help='Sakoe–Chiba band as fraction of length (OPTIMAL: 0.1)')
    ap.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs (joblib). Use 1 for sequential.')
    ap.add_argument('--outlier-method', type=str, default='none',
                    choices=['none', 'trajectory_end'],
                    help='Method for outlier detection (trajectory_end removes last point from TRAINED TinyCNN only)')
    ap.add_argument('--srvf-norm', type=str, default='unit', choices=['none','unit','zscore','robust'],
                    help='Normalization applied to SRVF q before computing elastic distance (OPTIMIZED: was none, now unit)')
    ap.add_argument('--use-robust-norm', action='store_true',
                    help='Shortcut to force robust SRVF normalization (median/MAD) before distance (overrides --srvf-norm)')
    ap.add_argument('--out-prefix', type=str, default='elastic_eigs', help='Prefix for output files')
    ap.add_argument('--no-time-scaling', action='store_true',
                    help='Do not normalize time to [0,1]; use actual time ranges')
    ap.add_argument('--pad-with-last', action='store_true',
                    help='When using --no-time-scaling, pad curves with last values to cover maximum range')
    ap.add_argument('--plot', action='store_true',
                    help='Generate plot of mean eigenvalue curves used for distance computation')
    ap.add_argument('--plot-output', type=str, default='elastic_mean_curves.png',
                    help='Output filename for the plot (default: elastic_mean_curves.png)')
    ap.add_argument('--no-plot-stats', action='store_true',
                    help='Disable plotting of mean/std statistics per category')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = _find_files(data_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files found in {data_dir} matching pattern '{args.pattern}'. Supported: .npz, .npy")

    print(f"Found {len(files)} files.")
    curves: List[np.ndarray] = []
    srvfs: List[np.ndarray] = []
    index: List[str] = []
    common_time: Optional[np.ndarray] = None

    srvf_norm_mode = 'robust' if args.use_robust_norm else args.srvf_norm
    
    # Determine time normalization settings
    normalize_time = not args.no_time_scaling
    use_padding = args.pad_with_last
    common_range = None
    
    # If not normalizing time, we need to determine the common range first
    if not normalize_time:
        # First pass: collect all time arrays to determine common range
        files_data = []
        for p in files:
            try:
                E, t = load_run(p)
                # Optional data-specific outlier handling
                E, t = remove_tinycnn_outliers(E, t, p.name, outlier_method=args.outlier_method)
                files_data.append((t, E))
            except Exception as e:
                print(f"[WARN] Skipping {p.name} for range determination: {e}")
                continue
        
        if use_padding:
            common_range = find_maximum_range(files_data)
        else:
            common_range = find_overlapping_range(files_data)

    for p in files:
        try:
            E, t = load_run(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        # Optional data-specific outlier handling
        E, t = remove_tinycnn_outliers(E, t, p.name, outlier_method=args.outlier_method)

        # Filter to overlapping range if using overlapping mode (not padding)
        if not normalize_time and not use_padding:
            t, E = filter_to_overlapping_range(t, E, common_range[0], common_range[1])
            if t.size == 0:
                print(f"[WARN] {p.name}: no data in overlapping range; skipping")
                continue

        # Mean curve -> resample -> smooth -> amplitude normalize
        y = mean_curve(E, topk=args.topk)
        t_resampled, y = resample_to_grid(t, y, N=args.resample, 
                                         normalize_time=normalize_time, 
                                         common_range=common_range,
                                         use_padding=use_padding)
        
        # Store common time grid from first successful curve
        if common_time is None:
            common_time = t_resampled
        
        # Skip if resampling resulted in empty curve
        if y.size == 0:
            print(f"[WARN] {p.name}: empty curve after resampling; skipping")
            continue
            
        y = smooth_series(y, method=args.smooth, win=args.smooth_win)
        y = normalize_amplitude(y, mode=args.amp_norm)

        # SRVF
        q = srvf(y)

        # (NEW) Normalize SRVF before elastic distance
        q = normalize_srvf(q, mode=srvf_norm_mode)

        curves.append(y)
        srvfs.append(q)
        index.append(p.name)

    N = len(srvfs)
    if N < 2:
        raise SystemExit("Need at least two valid runs to compute distances.")

    # Generate plot if requested
    if args.plot and common_time is not None:
        show_stats = not args.no_plot_stats
        plot_mean_curves(curves, common_time, index, normalize_time, show_stats, args.plot_output)

    time_mode = "normalized" if normalize_time else ("padded" if use_padding else "overlapping")
    print(
        f"Computing pairwise elastic distances for N={N} curves "
        f"(resample={args.resample}, time_mode={time_mode}, "
        f"outlier_method={args.outlier_method}, srvf_norm={srvf_norm_mode})..."
    )

    D = pairwise_distance_matrix(srvfs, lambda_warp=args.lambda_warp,
                                 window_frac=args.window_frac, n_jobs=args.n_jobs)

    out_prefix = Path(args.out_prefix)
    np.save(f"{out_prefix}_distance.npy", D)
    np.savetxt(f"{out_prefix}_distance.csv", D, delimiter=",", fmt="%.6f")
    with open(f"{out_prefix}_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Saved distance matrix to {out_prefix}_distance.npy / .csv and indices to {out_prefix}_index.json")


if __name__ == '__main__':
    main()

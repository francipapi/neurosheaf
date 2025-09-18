#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Elastic Distance with Comprehensive Parameter Search

Enhanced version of elastic distance computation with all optimizations:
- Slope penalty in DTW path
- Knee-weighted cost matrix 
- Derivative channel ensemble
- Isotonic regression
- Enhanced smoothing options

All features are toggleable via CLI parameters for comprehensive grid search.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
from scipy import stats
from scipy.stats import rankdata
try:
    from scipy.isotonic import isotonic_regression
    _HAS_ISOTONIC = True
except ImportError:
    _HAS_ISOTONIC = False

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

def _find_files(data_dir: Path, pattern: str) -> List[Path]:
    exts = {'.npz', '.npy'}
    files = []
    for p in data_dir.rglob(pattern):
        if p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files

def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (E, t) with E shape (T, K) time-by-eigen; t shape (T,)."""
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

def mean_curve(E: np.ndarray, topk: int = 0) -> np.ndarray:
    """Compute per-time-step mean eigenvalue curve.
    E shape (T, K). If topk > 0: mean of top-k per row (largest values).
    If topk < 0: mean of bottom-k per row. If topk == 0: mean across all K.
    """
    if E.ndim != 2:
        raise ValueError("E must be 2D (T, K)")
    if topk == 0:
        return np.mean(E, axis=1)
    T, K = E.shape
    k = min(abs(topk), K)
    # Partial top-k via argpartition (faster than full sort)
    if topk > 0:
        idx = np.argpartition(E, K - k, axis=1)[:, K - k:]
        vals = np.take_along_axis(E, idx, axis=1)
        return np.mean(vals, axis=1)
    else:
        idx = np.argpartition(E, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(E, idx, axis=1)
        return np.mean(vals, axis=1)

def resample_to_grid(t: np.ndarray, y: np.ndarray, N: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if t.size != y.size:
        raise ValueError("t and y must have same length")
    t_new = np.linspace(0.0, 1.0, N)
    # Interpolate to [0,1] by mapping original t to [0,1]
    t01 = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.linspace(0.0, 1.0, t.size)
    y_new = np.interp(t_new, t01, y)
    return t_new, y_new

def smooth_series(y: np.ndarray, method: str = 'moving', win: int = 9) -> np.ndarray:
    if method == 'none' or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    if win < 3:
        return y
    
    if method == 'savgol':
        # Savitzky-Golay filter (requires scipy)
        try:
            from scipy.signal import savgol_filter
            if win >= len(y):
                win = len(y) - 2 if len(y) >= 3 else 1
                if win % 2 == 0:
                    win -= 1
            poly_order = min(3, win - 1)
            return savgol_filter(y, win, poly_order)
        except ImportError:
            method = 'moving'  # Fallback to moving average
    
    if method == 'moving':
        # Simple symmetric moving average
        pad = win // 2
        ypad = np.pad(y, (pad, pad), mode='edge')
        kernel = np.ones(win, dtype=np.float64) / win
        return np.convolve(ypad, kernel, mode='valid')
    
    return y

def normalize_amplitude(y: np.ndarray, mode: str = 'zscore') -> np.ndarray:
    if mode == 'none':
        return y
    y = np.asarray(y, dtype=np.float64)
    if mode == 'zscore':
        mu = np.mean(y)
        sd = np.std(y)
        if sd < 1e-12:
            return y * 0.0
        return (y - mu) / sd
    elif mode == 'unit':
        # Unit L2 norm on [0,1]
        dt = 1.0 / (y.size - 1)
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

def apply_isotonic(y: np.ndarray) -> np.ndarray:
    """Apply isotonic regression to enforce monotonicity."""
    if not _HAS_ISOTONIC:
        return y  # Skip if not available
    try:
        return isotonic_regression(y, increasing=True)
    except:
        return y  # Return original if isotonic regression fails

def srvf(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Square-root velocity (slope) function for 1D time-series.
    q(t) = sign(f'(t)) * sqrt(|f'(t)|).
    """
    dy = np.gradient(y, 1.0 / (y.size - 1))
    q = np.sign(dy) * np.sqrt(np.abs(dy) + eps)
    return q

def compute_knee_weights(curves: List[np.ndarray], power: float = 1.0) -> np.ndarray:
    """Compute weights emphasizing high-derivative (knee) regions."""
    if not curves:
        return np.ones(300)  # Default length
    
    N = len(curves[0])
    derivs = []
    for y in curves:
        dy = np.gradient(y, 1.0 / (len(y) - 1))
        derivs.append(np.abs(dy))
    
    # Average absolute derivative across all curves
    avg_deriv = np.mean(derivs, axis=0)
    
    # Normalize and apply power
    weights = avg_deriv / (avg_deriv.mean() + 1e-12)
    weights = np.power(weights, power)
    
    # Scale to [0.5, 1.0] range to avoid zero weights
    weights = 0.5 + 0.5 * (weights / (weights.max() + 1e-12))
    
    return weights

def enhanced_dtw_path(cost: np.ndarray, window: int, step_penalty: float = 0.0) -> Tuple[float, List[Tuple[int, int]]]:
    """Enhanced DTW path with slope penalty for non-diagonal steps."""
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
            # From candidates within window
            best = big
            move = 0
            
            # Diagonal move (preferred)
            if i > 0 and j > 0:
                v = D[i - 1, j - 1]
                if v < best:
                    best = v
                    move = 1  # diag
            
            # Vertical move (with penalty)
            if i > 0 and (j >= max(0, i - 1 - window)) and (j <= min(N - 1, i - 1 + window)):
                v = D[i - 1, j] + step_penalty
                if v < best:
                    best = v
                    move = 2  # up
            
            # Horizontal move (with penalty)
            if j > 0 and (j - 1 >= max(0, i - window)) and (j - 1 <= min(N - 1, i + window)):
                v = D[i, j - 1] + step_penalty
                if v < best:
                    best = v
                    move = 3  # left
            
            D[i, j] = cost[i, j] + best
            ptr[i, j] = move

    # Backtrack
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
            # Should not happen; safeguard
            if i > 0 and j > 0:
                i -= 1; j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return D[-1, -1], path

def elastic_distance(q1: np.ndarray, q2: np.ndarray, 
                    lambda_warp: float = 0.05, 
                    window_frac: float = 0.2,
                    step_penalty: float = 0.0,
                    knee_weights: Optional[np.ndarray] = None) -> float:
    """Enhanced elastic distance with knee weighting and step penalty."""
    assert q1.shape == q2.shape
    N = q1.size
    w = max(0, min(N - 1, int(round(window_frac * N))))
    
    # Local cost
    ii = np.arange(N, dtype=np.float64)
    # (q1[i]-q2[j])^2 term
    A = (q1.reshape(-1, 1) - q2.reshape(1, -1))**2
    # warp penalty term
    JJ = (ii.reshape(-1, 1) - ii.reshape(1, -1)) / float(N)
    C = A + (lambda_warp * JJ * JJ)
    
    # Apply knee weights if provided
    if knee_weights is not None:
        weight_matrix = knee_weights.reshape(-1, 1) * knee_weights.reshape(1, -1)
        C = C * weight_matrix

    total_cost, path = enhanced_dtw_path(C, w, step_penalty)

    # Build q2 aligned to q1 grid via the path
    q2_aligned = np.empty_like(q1)
    from collections import defaultdict
    bucket: Dict[int, List[int]] = defaultdict(list)
    for (i, j) in path:
        bucket[i].append(j)
    for i in range(N):
        js = bucket.get(i, None)
        if not js:
            # Fallback: nearest j along diagonal
            q2_aligned[i] = q2[min(N - 1, max(0, i))]
        else:
            q2_aligned[i] = np.mean(q2[np.array(js, dtype=int)])

    # L2 integral on [0,1]
    diff2 = (q1 - q2_aligned)**2
    dt = 1.0 / (N - 1)
    l2 = math.sqrt(np.sum(diff2) * dt)
    return float(l2)

def compute_derivative_distances(curves: List[np.ndarray], 
                               lambda_warp: float = 0.05,
                               window_frac: float = 0.15,
                               n_jobs: int = -1) -> np.ndarray:
    """Compute distance matrix based on derivatives."""
    # Compute and normalize derivatives
    derivs = []
    for y in curves:
        dy = np.gradient(y, 1.0 / (len(y) - 1))
        # Z-score normalize
        dy_norm = (dy - np.mean(dy)) / (np.std(dy) + 1e-12)
        derivs.append(dy_norm)
    
    # Convert to SRVF representation
    Q_deriv = [srvf(d) for d in derivs]
    
    # Compute pairwise distances
    N = len(Q_deriv)
    D = np.zeros((N, N), dtype=np.float64)
    
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def _compute_deriv(i: int, j: int) -> Tuple[int, int, float]:
        d = elastic_distance(Q_deriv[i], Q_deriv[j], 
                           lambda_warp=lambda_warp, 
                           window_frac=window_frac)
        return i, j, d
    
    if _HAS_JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, prefer='processes')(
            delayed(_compute_deriv)(i, j) for (i, j) in pairs
        )
    else:
        results = list(map(lambda ij: _compute_deriv(*ij), pairs))
    
    for (i, j, d) in results:
        D[i, j] = d
        D[j, i] = d
    
    return D

def ensemble_distances(D_elastic: np.ndarray, D_derivative: np.ndarray, 
                      ratio: float = 0.6) -> np.ndarray:
    """Combine distances using rank normalization."""
    # Get upper triangle indices for ranking
    triu_idx = np.triu_indices_from(D_elastic, k=1)
    
    # Rank the distances (scale-free combination)
    elastic_ranks = rankdata(D_elastic[triu_idx])
    deriv_ranks = rankdata(D_derivative[triu_idx])
    
    # Weighted combination
    combined_ranks = ratio * elastic_ranks + (1 - ratio) * deriv_ranks
    
    # Create combined matrix
    D_ensemble = np.zeros_like(D_elastic)
    D_ensemble[triu_idx] = combined_ranks
    D_ensemble[triu_idx[1], triu_idx[0]] = combined_ranks  # Make symmetric
    
    return D_ensemble

def pairwise_distance_matrix(Q: List[np.ndarray], 
                           lambda_warp: float,
                           window_frac: float,
                           step_penalty: float = 0.0,
                           knee_weights: Optional[np.ndarray] = None,
                           n_jobs: int = -1) -> np.ndarray:
    N = len(Q)
    D = np.zeros((N, N), dtype=np.float64)

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def _compute(i: int, j: int) -> Tuple[int, int, float]:
        d = elastic_distance(Q[i], Q[j], 
                           lambda_warp=lambda_warp, 
                           window_frac=window_frac,
                           step_penalty=step_penalty,
                           knee_weights=knee_weights)
        return i, j, d

    if _HAS_JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, prefer='processes')(
            delayed(_compute)(i, j) for (i, j) in pairs
        )
    else:
        results = list(map(lambda ij: _compute(*ij), pairs))

    for (i, j, d) in results:
        D[i, j] = d
        D[j, i] = d
    return D

def main():
    ap = argparse.ArgumentParser(description="Optimized elastic distance with all enhancement features")
    ap.add_argument('--data-dir', type=str, required=True, help='Directory to scan recursively for data files')
    ap.add_argument('--pattern', type=str, default='*', help='Glob pattern to match')
    ap.add_argument('--resample', type=int, default=300, help='Number of points in common time grid')
    ap.add_argument('--topk', type=int, default=0, help='Use full spectrum (0) or top-k eigenvalues')
    ap.add_argument('--amp-norm', type=str, default='zscore', choices=['zscore','unit','p95','none'])
    ap.add_argument('--smooth', type=str, default='moving', choices=['moving','savgol','none'])
    ap.add_argument('--smooth-win', type=int, default=9, help='Window size for smoothing')
    
    # Core optimization parameters
    ap.add_argument('--lambda-warp', type=float, default=0.05, help='Penalty strength for time warping')
    ap.add_argument('--window-frac', type=float, default=0.20, help='Sakoe–Chiba band fraction')
    ap.add_argument('--step-penalty', type=float, default=0.0, help='Penalty for non-diagonal DTW steps')
    
    # Enhancement features
    ap.add_argument('--knee-weight', action='store_true', help='Enable knee region weighting')
    ap.add_argument('--knee-weight-power', type=float, default=1.0, help='Power for knee weighting')
    ap.add_argument('--derivative-ensemble', action='store_true', help='Enable derivative channel ensemble')
    ap.add_argument('--ensemble-ratio', type=float, default=0.6, help='Elastic weight in ensemble')
    ap.add_argument('--isotonic', action='store_true', help='Apply isotonic regression')
    
    ap.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs')
    ap.add_argument('--out-prefix', type=str, default='elastic_opt', help='Prefix for output files')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = _find_files(data_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files found in {data_dir} matching pattern '{args.pattern}'")

    print(f"Found {len(files)} files.")
    start_time = time.time()
    
    curves = []
    srvfs = []
    index = []

    for p in files:
        try:
            E, t = load_run(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        y = mean_curve(E, topk=args.topk)  # Use full spectrum (topk=0)
        _, y = resample_to_grid(t, y, N=args.resample)
        y = smooth_series(y, method=args.smooth, win=args.smooth_win)
        
        # Apply isotonic regression if requested
        if args.isotonic:
            y = apply_isotonic(y)
            
        y = normalize_amplitude(y, mode=args.amp_norm)
        q = srvf(y)
        curves.append(y)
        srvfs.append(q)
        index.append(p.name)

    N = len(srvfs)
    if N < 2:
        raise SystemExit("Need at least two valid runs to compute distances.")

    print(f"Computing distances for N={N} curves...")
    print(f"Parameters: λ={args.lambda_warp}, w={args.window_frac}, step_pen={args.step_penalty}")
    
    # Compute knee weights if requested
    knee_weights = None
    if args.knee_weight:
        print("Computing knee weights...")
        knee_weights = compute_knee_weights(curves, power=args.knee_weight_power)
    
    # Compute elastic distances
    D_elastic = pairwise_distance_matrix(srvfs, 
                                       lambda_warp=args.lambda_warp,
                                       window_frac=args.window_frac,
                                       step_penalty=args.step_penalty,
                                       knee_weights=knee_weights,
                                       n_jobs=args.n_jobs)
    
    # Compute derivative ensemble if requested
    if args.derivative_ensemble:
        print("Computing derivative distances...")
        D_derivative = compute_derivative_distances(curves,
                                                  lambda_warp=args.lambda_warp * 0.5,  # Softer for derivatives
                                                  window_frac=args.window_frac,
                                                  n_jobs=args.n_jobs)
        print("Creating ensemble...")
        D = ensemble_distances(D_elastic, D_derivative, ratio=args.ensemble_ratio)
    else:
        D = D_elastic
    
    computation_time = time.time() - start_time

    # Save results
    out_prefix = Path(args.out_prefix)
    np.save(f"{out_prefix}_distance.npy", D)
    np.savetxt(f"{out_prefix}_distance.csv", D, delimiter=",", fmt="%.6f")
    with open(f"{out_prefix}_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    
    # Save configuration and timing
    config = {
        'parameters': vars(args),
        'n_models': N,
        'computation_time': computation_time,
        'enhancement_features': {
            'knee_weight': args.knee_weight,
            'derivative_ensemble': args.derivative_ensemble,
            'isotonic': args.isotonic
        }
    }
    with open(f"{out_prefix}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved results to {out_prefix}_* files")
    print(f"Computation time: {computation_time:.1f} seconds")

if __name__ == '__main__':
    main()
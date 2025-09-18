#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elastic distance between **median** eigenvalue evolution curves.

This is a drop-in sibling of `elastic_mean_eigs_distance_torus.py`, but it
uses the per-time-step **median across all eigenvalues** instead of a mean.
Outputs and CLI remain compatible with the original script:
  {out-prefix}_distance.npy : NxN symmetric matrix of elastic distances (float64)
  {out-prefix}_distance.csv : same in CSV
  {out-prefix}_index.json   : list of file basenames in matrix order

Best-performing defaults (based on experiments you requested):
  --resample      60
  --amp-norm      zscore
  --smooth        moving
  --smooth-win    7
  --lambda-warp   0.2
  --window-frac   0.2

Notes:
- `--topk` is accepted for compatibility but ignored (we always use the median).
- Outlier handling for trained TinyCNN runs is preserved (drop last sample).
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
                E = arr[:, None]

        if E is None:
            raise ValueError(f"{path.name}: could not find eigenvalue matrix in keys={keys}")

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

    outlier_method:
        'none'            : do nothing
        'trajectory_end'  : if filename looks like trained TinyCNN, drop final sample
    """
    if outlier_method != 'trajectory_end':
        return E, t

    fname = filename.lower()
    is_tinycnn = ('tinycnn' in fname) and ('random' not in fname)
    if is_tinycnn and E.shape[0] > 1:
        E = E[:-1, :]
        t = t[:-1]
    return E, t


def median_curve(E: np.ndarray) -> np.ndarray:
    """Per-time-step **median** across eigenvalues. E shape (T, K)."""
    if E.ndim != 2:
        raise ValueError("E must be 2D (T, K)")
    return np.median(E, axis=1)


def resample_to_grid(t: np.ndarray, y: np.ndarray, N: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if t.size != y.size:
        raise ValueError("t and y must have same length")
    t_new = np.linspace(0.0, 1.0, N)
    t01 = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.linspace(0.0, 1.0, t.size)
    y_new = np.interp(t_new, t01, y)
    return t_new, y_new


def smooth_series(y: np.ndarray, method: str = 'moving', win: int = 7) -> np.ndarray:
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


def srvf(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Square-root velocity function for 1D time-series: q(t) = sign(f') * sqrt(|f'|)."""
    dy = np.gradient(y, 1.0 / (y.size - 1))
    q = np.sign(dy) * np.sqrt(np.abs(dy) + eps)
    return q


def _dtw_path(cost: np.ndarray, window: int) -> Tuple[float, List[Tuple[int, int]]]:
    """Minimal-cost DTW path with Sakoe–Chiba window. Returns (total_cost, path)."""
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
            if i > 0 and j > 0:
                i -= 1; j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return D[-1, -1], path


def elastic_distance(q1: np.ndarray, q2: np.ndarray, lambda_warp: float = 0.2, window_frac: float = 0.2) -> float:
    """Elastic distance between SRVFs via DP warping with penalty (best defaults baked in)."""
    assert q1.shape == q2.shape
    N = q1.size
    w = max(0, min(N - 1, int(round(window_frac * N))))
    ii = np.arange(N, dtype=np.float64)

    # (q1[i]-q2[j])^2 term
    A = (q1.reshape(-1, 1) - q2.reshape(1, -1))**2
    # warp penalty term
    JJ = (ii.reshape(-1, 1) - ii.reshape(1, -1)) / float(N)
    C = A + (lambda_warp * JJ * JJ)

    total_cost, path = _dtw_path(C, w)

    # Align q2 to q1 grid
    from collections import defaultdict
    bucket: Dict[int, List[int]] = defaultdict(list)
    for (i, j) in path:
        bucket[i].append(j)
    q2_aligned = np.empty_like(q1)
    for i in range(N):
        js = bucket.get(i, None)
        q2_aligned[i] = np.mean(q2[np.array(js, dtype=int)]) if js else q2[min(N - 1, max(0, i))]

    # L2 on [0,1]
    diff2 = (q1 - q2_aligned)**2
    dt = 1.0 / (N - 1)
    l2 = math.sqrt(np.sum(diff2) * dt)
    return float(l2)


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
    ap = argparse.ArgumentParser(description="Elastic distance between **median** eigenvalue evolution curves (SRVF + DP warping)")
    ap.add_argument('--data-dir', type=str, required=True, help='Directory to scan recursively for data files')
    ap.add_argument('--pattern', type=str, default='*eigenvalues.npz', help='Glob pattern to match (e.g., \"*eigenvalues.npz\")')
    ap.add_argument('--resample', type=int, default=60, help='Number of points in common time grid')
    ap.add_argument('--topk', type=int, default=0, help='(compat only) ignored: median uses all eigenvalues')
    ap.add_argument('--amp-norm', type=str, default='zscore', choices=['zscore','unit','p95','none'],
                    help='Amplitude normalization mode')
    ap.add_argument('--smooth', type=str, default='moving', choices=['moving','none'], help='Smoothing method')
    ap.add_argument('--smooth-win', type=int, default=7, help='Window size for smoothing (odd integer)')
    ap.add_argument('--lambda-warp', type=float, default=0.2, help='Penalty strength for time warping')
    ap.add_argument('--window-frac', type=float, default=0.2, help='Sakoe–Chiba band as fraction of length')
    ap.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs (joblib). Use 1 for sequential.')
    ap.add_argument('--outlier-method', type=str, default='trajectory_end',
                    choices=['none', 'trajectory_end'],
                    help='Outlier method (trajectory_end removes last point from TRAINED TinyCNN only)')
    ap.add_argument('--out-prefix', type=str, default='elastic_eigs', help='Prefix for output files')
    args = ap.parse_args()

    if args.topk != 0:
        print(f"[INFO] --topk={args.topk} provided but ignored: using median across all eigenvalues.")

    data_dir = Path(args.data_dir)
    files = _find_files(data_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files found in {data_dir} matching pattern '{args.pattern}'. Supported: .npz, .npy")

    print(f"Found {len(files)} files.")
    curves = []
    srvfs = []
    index = []

    for p in files:
        try:
            E, t = load_run(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        # Outlier removal (trained TinyCNN only)
        E, t = remove_tinycnn_outliers(E, t, p.name, outlier_method=args.outlier_method)

        y = median_curve(E)                                 # <-- median instead of mean
        _, y = resample_to_grid(t, y, N=args.resample)
        y = smooth_series(y, method=args.smooth, win=args.smooth_win)
        y = normalize_amplitude(y, mode=args.amp_norm)
        q = srvf(y)

        curves.append(y)
        srvfs.append(q)
        index.append(p.name)

    N = len(srvfs)
    if N < 2:
        raise SystemExit("Need at least two valid runs to compute distances.")

    print(f"Computing pairwise elastic distances for N={N} curves "
          f"(resample={args.resample}, lambda={args.lambda_warp}, window_frac={args.window_frac})...")
    D = pairwise_distance_matrix(srvfs, lambda_warp=args.lambda_warp,
                                 window_frac=args.window_frac, n_jobs=args.n_jobs)

    out_prefix = Path(args.out_prefix)
    np.save(f"{out_prefix}_distance.npy", D)
    np.savetxt(f"{out_prefix}_distance.csv", D, delimiter=",", fmt="%.6f")
    with open(f"{out_prefix}_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Saved: {out_prefix}_distance.[npy,csv] and {out_prefix}_index.json")


if __name__ == "__main__":
    main()

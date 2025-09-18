#!/usr/bin/env python3
# ablate_eig_curves.py
"""
Full ablation study on mean-eigenvalue trajectories (fingerprints).

- Scans a folder of *eigenvalues.npz files (one per model). Any filename
  containing 'random' (case-insensitive) is labeled 'random'; otherwise 'trained'.
- Derives or loads a mean eigenvalue curve per model.
- Builds pairwise distance matrices for multiple curve distances and
  elastic/SRVT variants, then evaluates clustering metrics vs. ground truth.

Distances implemented:
  - elastic (SRVT with DTW-style reparameterization; options: amplitude norm,
    smoothing, SRVF norm, warp penalty, window band, time scaling, padding)
  - l2 (Euclidean)
  - cosine
  - dtw (classic DP; optional Sakoe–Chiba window band)
  - wasserstein (1D Earth Mover's Distance on value distributions; SciPy)

Outputs:
  - ablation_out/
      dist_<tag>.csv         (distance matrix for each configuration)
      summary.csv            (one row per configuration with metrics)
      config_<tag>.json      (exact config used)

Notes:
  - No external packages beyond numpy/scipy/sklearn.
  - Robust loading of mean curve: tries keys ['mean_curve', 'mean', 'curve'].
    If absent, computes mean across eigenvalue index (auto axis heuristic).
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from glob import glob
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.signal import savgol_filter
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score, v_measure_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# ----------------------------- Utilities ---------------------------------

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    w = int(w)
    if w % 2 == 0:
        w += 1  # prefer odd window
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kern = np.ones(w) / w
    return np.convolve(xp, kern, mode="valid")

def find_maximum_range(files_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    """Find the maximum time range across all files (union instead of intersection)."""
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
        return 0.0, 1.0
    
    # Find union of all time ranges (maximum extent)
    min_starts = [t_range[0] for t_range in time_ranges]
    max_ends = [t_range[1] for t_range in time_ranges]
    
    global_start = min(min_starts)  # Earliest start time
    global_end = max(max_ends)      # Latest end time
    
    return global_start, global_end

def resample_to_grid(t: np.ndarray, y: np.ndarray, N: int = 200, 
                     normalize_time: bool = True, common_range: Optional[Tuple[float, float]] = None,
                     use_padding: bool = False, use_arc_length: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Resample curve to a common grid."""
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
        if use_arc_length:
            # Arc-length parameterization: use cumulative |dy| as parameterization
            dy = np.diff(y)
            speed = np.sqrt(np.abs(dy) + 1e-12)
            s = np.zeros(t.size)
            s[1:] = np.cumsum(speed)
            s = s / (s[-1] + 1e-12) if s[-1] > 0 else np.linspace(0.0, 1.0, t.size)
            y_new = np.interp(t_new, s, y)
        else:
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
            y_new = np.interp(t_new, t, y)
    
    return t_new, y_new

def resample_curve(y: np.ndarray, n: int, time_scaling: bool = False) -> np.ndarray:
    """Resample 1D curve y to n points via linear interpolation.
    If time_scaling is True, pre-warp by cumulative arc-length."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    if T == n:
        return y.copy()
    t = np.linspace(0, 1, T)
    if time_scaling:
        # arc-length parameterization
        dy = np.diff(y)
        speed = np.sqrt(np.abs(dy) + 1e-12)
        s = np.zeros(T)
        s[1:] = np.cumsum(speed)
        s = s / (s[-1] + 1e-12)
        t = s
    t_new = np.linspace(0, 1, n)
    return np.interp(t_new, t, y)

def amp_normalize(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "none":
        return y
    if mode == "zscore":
        mu = y.mean()
        sd = y.std()
        return (y - mu) / (sd + 1e-12)
    if mode == "minmax":
        lo, hi = y.min(), y.max()
        if hi - lo < 1e-12:
            return y * 0.0
        return (y - lo) / (hi - lo)
    raise ValueError(f"Unknown amp-norm: {mode}")

def smooth_curve(y: np.ndarray, mode: str, win: int) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "moving":
        return moving_average(y, win)
    if mode == "savgol":
        w = max(3, int(win) if int(win) % 2 == 1 else int(win) + 1)
        poly = 2 if w >= 5 else 1
        return savgol_filter(y, window_length=w, polyorder=poly, mode="interp")
    raise ValueError(f"Unknown smoothing mode: {mode}")

def pad_to_same_length(a: np.ndarray, b: np.ndarray, pad_with_last: bool) -> Tuple[np.ndarray, np.ndarray]:
    la, lb = len(a), len(b)
    if la == lb:
        return a, b
    L = max(la, lb)
    if pad_with_last:
        if la < L:
            a = np.pad(a, (0, L - la), mode="edge")
        if lb < L:
            b = np.pad(b, (0, L - lb), mode="edge")
        return a, b
    # else resample both to max length
    return resample_curve(a, L), resample_curve(b, L)

# --------------------------- Distances ------------------------------------

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    x, y = pad_to_same_length(a, b, pad_with_last=True)
    return float(np.linalg.norm(x - y))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    x, y = pad_to_same_length(a, b, pad_with_last=True)
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return 1.0
    return float(1.0 - np.dot(x, y) / (nx * ny))

def dtw_distance(a: np.ndarray, b: np.ndarray, window_frac: float = 0.1) -> float:
    """Classic O(n^2) DTW with optional Sakoe–Chiba band (fraction of length)."""
    x, y = pad_to_same_length(a, b, pad_with_last=True)
    n, m = len(x), len(y)
    w = int(max(n, m) * window_frac)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jmin = 1 if window_frac <= 0 else max(1, i - w)
        jmax = m if window_frac <= 0 else min(m, i + w)
        for j in range(jmin, jmax + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))

def wasserstein1d_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Treat sample values as 1D distributions (ignores time order)
    return float(wasserstein_distance(a, b))

def srvf(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Square-root velocity (slope) function for 1D time-series: q(t) = sign(f') * sqrt(|f'|)."""
    dy = np.gradient(y, 1.0 / max(1, (y.size - 1)))
    q = np.sign(dy) * np.sqrt(np.abs(dy) + eps)
    return q

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

def elastic_distance(
    a: np.ndarray,
    b: np.ndarray,
    srvf_norm: str = "unit",
    window_frac: float = 0.1,
    lambda_warp: float = 0.1,
) -> float:
    """Approximate elastic distance between SRVFs via DP warping with penalty."""
    # Curves should already be on same grid from preprocessing
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        # Fallback to padding if shapes don't match
        x, y = pad_to_same_length(a, b, pad_with_last=True)
    
    # Convert to SRVF
    qa = srvf(x)
    qb = srvf(y)
    
    # Apply SRVF normalization
    qa = normalize_srvf(qa, mode=srvf_norm)
    qb = normalize_srvf(qb, mode=srvf_norm)
    
    assert qa.shape == qb.shape
    N = qa.size
    w = max(0, min(N - 1, int(round(window_frac * N))))

    # Local cost
    ii = np.arange(N, dtype=np.float64)
    A = (qa.reshape(-1, 1) - qb.reshape(1, -1))**2
    JJ = (ii.reshape(-1, 1) - ii.reshape(1, -1)) / float(N)
    C = A + (lambda_warp * JJ * JJ)

    total_cost, path = _dtw_path(C, w)

    # Align qb to qa grid via the path
    qb_aligned = np.empty_like(qa)
    from collections import defaultdict
    bucket: Dict[int, List[int]] = defaultdict(list)
    for (i, j) in path:
        bucket[i].append(j)
    for i in range(N):
        js = bucket.get(i, None)
        if not js:
            qb_aligned[i] = qb[i]  # fallback to diagonal
        else:
            qb_aligned[i] = np.mean(qb[np.array(js, dtype=int)])

    # L2 integral on [0,1]
    diff2 = (qa - qb_aligned)**2
    dt = 1.0 / max(1, (N - 1))
    l2 = math.sqrt(np.sum(diff2) * dt)
    return float(l2)

# --------------------------- Metrics --------------------------------------

def dunn_index(D: np.ndarray, labels: np.ndarray) -> float:
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)
    diameters = []
    for members in clusters.values():
        if len(members) < 2:
            diameters.append(0.0)
        else:
            sub = D[np.ix_(members, members)]
            diameters.append(sub.max())
    delta_intra = max(diameters) if diameters else 0.0
    deltas = []
    keys = list(clusters.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = clusters[keys[i]]
            b = clusters[keys[j]]
            sub = D[np.ix_(a, b)]
            deltas.append(sub.min())
    delta_inter = min(deltas) if deltas else 0.0
    if delta_intra == 0:
        return math.inf if delta_inter > 0 else 0.0
    return float(delta_inter / delta_intra)

def cophenetic_corr(D: np.ndarray, method: str = "average") -> float:
    Y = squareform(D, checks=False)
    Z = linkage(Y, method=method)
    c, _ = cophenet(Z, Y)
    return float(c)

def cohen_d_within_between(D: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    n = D.shape[0]
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                within.append(D[i, j])
            else:
                between.append(D[i, j])
    within = np.array(within); between = np.array(between)
    m1, m2 = between.mean(), within.mean()
    s1, s2 = between.std(ddof=1), within.std(ddof=1)
    n1, n2 = len(between), len(within)
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2 + 1e-12))
    d = (m1 - m2) / (sp + 1e-12)
    return {"within_mean": float(m2), "between_mean": float(m1), "cohen_d": float(d)}

def permutation_pvalue_within_between(D: np.ndarray, y_true: np.ndarray, n_perm: int = 3000, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    def diff_mean(labels):
        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    within.append(D[i, j])
                else:
                    between.append(D[i, j])
        return float(np.mean(between) - np.mean(within))
    y = np.array(y_true)
    obs = diff_mean(y)
    cnt = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if diff_mean(y_perm) >= obs - 1e-12:
            cnt += 1
    pval = (cnt + 1) / (n_perm + 1)
    return obs, float(pval)

def jaccard_stability(D: np.ndarray, base_pred: np.ndarray, B: int = 300, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    N = len(base_pred)
    base_pairs = {(i, j) for i in range(N) for j in range(i + 1, N) if base_pred[i] == base_pred[j]}
    scores = []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        uniq = sorted(set(idx.tolist()))
        if len(uniq) < 2:
            continue
        Dsub = D[np.ix_(uniq, uniq)]
        agg = AgglomerativeClustering(n_clusters=len(set(base_pred)), metric="precomputed", linkage="average")
        pred_sub = agg.fit_predict(Dsub)
        sub_pairs = {(uniq[a], uniq[b]) for a in range(len(uniq)) for b in range(a + 1, len(uniq)) if pred_sub[a] == pred_sub[b]}
        base_sub = {(i, j) for (i, j) in base_pairs if i in uniq and j in uniq}
        union = len(sub_pairs.union(base_sub)); inter = len(sub_pairs.intersection(base_sub))
        if union == 0:
            continue
        scores.append(inter / union)
    if not scores:
        return {"mean": None, "ci95_lo": None, "ci95_hi": None, "n": 0}
    arr = np.array(scores)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {"mean": float(arr.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi), "n": int(len(arr))}

# ------------------------- Data loading -----------------------------------

def load_mean_curve_from_npz(path: Path, compute_mean_axis: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Try to load a mean eigenvalue curve and time vector from an .npz file."""
    data = np.load(path, allow_pickle=True)
    
    # First check for eigenvalue_matrix (specific to this dataset)
    if "eigenvalue_matrix" in data:
        E = np.array(data["eigenvalue_matrix"]).astype(float)
        if E.ndim == 2:
            # Get time vector to determine correct orientation
            if "time_vector" in data:
                t = np.array(data["time_vector"]).astype(float)
                # Match elastic_mean_eigs_distance.py behavior: transpose if needed
                if t.size != E.shape[0] and t.size == E.shape[1]:
                    E = E.T
                # Now compute mean over eigenvalues (axis=1 after potential transpose)
                y = E.mean(axis=1)
            else:
                # No time vector, assume we need to transpose based on shape heuristics
                if E.shape[0] > E.shape[1]:  # More eigenvalues than time points
                    E = E.T
                y = E.mean(axis=1)
                t = np.linspace(0.0, 1.0, len(y))
            return np.array(y).squeeze(), t
    
    # Try common keys next
    for key in ["mean_curve", "mean", "curve"]:
        if key in data:
            y = np.array(data[key]).astype(float).squeeze()
            if y.ndim == 1:
                # Default time vector if no time info available
                t = np.linspace(0.0, 1.0, len(y))
                return y, t
    # Otherwise infer from eigenvalue array(s)
    # choose the first array-like entry
    candidate = None
    for k in data.files:
        arr = np.array(data[k])
        if arr.ndim >= 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            candidate = arr
            break
    if candidate is None:
        raise ValueError(f"No numeric array found in {path}")
    arr = candidate.astype(float)
    if arr.ndim == 1:
        t = np.linspace(0.0, 1.0, len(arr))
        return arr, t
    # Heuristic: assume shape (n_eigs, n_t) or (n_t, n_eigs). We want mean over eigenvalues.
    if compute_mean_axis == "0":
        axis = 0
    elif compute_mean_axis == "1":
        axis = 1
    else:
        # auto: assume smaller axis is eigenvalue index
        axis = int(np.argmin(arr.shape))
    y = arr.mean(axis=axis)
    y = np.array(y).squeeze()
    t = np.linspace(0.0, 1.0, len(y))
    return y, t

def list_models(data_dir: Path, pattern: str) -> List[Path]:
    return sorted(map(Path, glob(str(data_dir / pattern))))

def label_from_name(name: str) -> str:
    return "random" if "random" in name.lower() else "trained"

# --------------------------- Config / CLI ---------------------------------

@dataclass
class Config:
    # IO
    data_dir: str
    pattern: str = "*eigenvalues.npz"
    out_dir: str = "ablation_out"
    # selection
    distances: List[str] = field(default_factory=lambda: ["elastic", "l2", "cosine", "dtw", "wasserstein"])
    # common preprocessing
    resample: int = 200
    amp_norm: List[str] = field(default_factory=lambda: ["zscore"])
    smooth: List[str] = field(default_factory=lambda: ["moving"])
    smooth_win: List[int] = field(default_factory=lambda: [15])
    time_scaling: List[bool] = field(default_factory=lambda: [False])
    pad_with_last: bool = True
    # elastic-specific
    srvf_norm: List[str] = field(default_factory=lambda: ["unit"])
    window_frac: List[float] = field(default_factory=lambda: [0.1])
    lambda_warp: List[float] = field(default_factory=lambda: [0.1])
    # dtw-specific
    dtw_window_frac: List[float] = field(default_factory=lambda: [0.1])
    # admin
    n_jobs: int = 8
    stability: bool = False
    compute_mean_axis: str = "auto"  # 'auto'|'0'|'1'

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation study on eigenvalue-evolution fingerprints.")
    p.add_argument("--data-dir", required=True, type=str)
    p.add_argument("--pattern", default="*eigenvalues.npz", type=str)
    p.add_argument("--out-dir", default="ablation_out", type=str)

    # distance families
    p.add_argument("--distances", nargs="+", default=["elastic", "l2", "cosine", "dtw", "wasserstein"],
                   help="Which distances to run.")
    # preprocessing
    p.add_argument("--resample", type=int, default=200)
    p.add_argument("--amp-norm", nargs="+", default=["zscore"], choices=["none", "zscore", "minmax"])
    p.add_argument("--smooth", nargs="+", default=["moving"], choices=["none", "moving", "savgol"])
    p.add_argument("--smooth-win", nargs="+", type=int, default=[15])
    p.add_argument("--time-scaling", nargs="+", default=["false"], choices=["true", "false"])
    p.add_argument("--no-time-scaling", action="store_true", help="Use actual time ranges instead of normalizing to [0,1]")
    p.add_argument("--use-arc-length", nargs="+", default=["false"], choices=["true", "false"], help="Use arc-length parameterization")
    p.add_argument("--pad-with-last", action="store_true")
    p.add_argument("--compute-mean-axis", default="auto", choices=["auto", "0", "1"])

    # elastic-specific
    p.add_argument("--srvf-norm", nargs="+", default=["unit"], choices=["unit", "none"])
    p.add_argument("--window-frac", nargs="+", type=float, default=[0.1])
    p.add_argument("--lambda-warp", nargs="+", type=float, default=[0.1])

    # dtw-specific
    p.add_argument("--dtw-window-frac", nargs="+", type=float, default=[0.1])

    # misc
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--stability", action="store_true", help="Compute bootstrap Jaccard stability.")
    p.add_argument("--preset", type=str, default=None, help="Use 'elastic_optimal' to mirror your current best config.")
    return p

def apply_preset(args: argparse.Namespace):
    if args.preset is None:
        return
    if args.preset == "elastic_optimal":
        # mirror your provided call
        args.distances = ["elastic"]
        args.resample = 200
        args.amp_norm = ["zscore"]
        args.smooth = ["moving"]
        args.smooth_win = [15]
        args.lambda_warp = [0.1]
        args.window_frac = [0.1]
        args.time_scaling = ["false"]
        args.pad_with_last = True
        args.srvf_norm = ["unit"]
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

# ---------------------- Pipeline: build distances -------------------------

def preprocess_curve(t: np.ndarray, y: np.ndarray, resample_n: int, amp_mode: str, smooth_mode: str, smooth_win: int, 
                     normalize_time: bool = True, common_range: Optional[Tuple[float, float]] = None, use_padding: bool = False,
                     use_arc_length: bool = False) -> np.ndarray:
    # Use new grid-based resampling
    t_new, y2 = resample_to_grid(t, y, N=resample_n, normalize_time=normalize_time, 
                                common_range=common_range, use_padding=use_padding, use_arc_length=use_arc_length)
    y2 = amp_normalize(y2, amp_mode)
    y2 = smooth_curve(y2, smooth_mode, smooth_win)
    return y2

def pairwise_distance_matrix(
    curves: List[np.ndarray],
    distance: str,
    cfg: Dict,
) -> np.ndarray:
    n = len(curves)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if distance == "l2":
                d = l2_distance(curves[i], curves[j])
            elif distance == "cosine":
                d = cosine_distance(curves[i], curves[j])
            elif distance == "dtw":
                d = dtw_distance(curves[i], curves[j], window_frac=cfg.get("dtw_window_frac", 0.1))
            elif distance == "wasserstein":
                d = wasserstein1d_distance(curves[i], curves[j])
            elif distance == "elastic":
                d = elastic_distance(
                    curves[i],
                    curves[j],
                    srvf_norm=cfg.get("srvf_norm", "unit"),
                    window_frac=cfg.get("window_frac", 0.1),
                    lambda_warp=cfg.get("lambda_warp", 0.1),
                )
            else:
                raise ValueError(f"Unknown distance: {distance}")
            D[i, j] = D[j, i] = float(d)
    np.fill_diagonal(D, 0.0)
    return D

# ---------------------- Evaluation & reporting ----------------------------

def evaluate_distance_matrix(D: np.ndarray, keys: List[str], y_true_str: List[str], stability: bool) -> Dict[str, float]:
    # Sanity check: ensure we have multiple classes
    classes = sorted(set(y_true_str))
    if len(classes) < 2:
        raise SystemExit(f"Only {len(classes)} class found in filenames. Make sure 'random' appears in the random model names.")
    
    # map to ints for external metrics
    c2i = {c: i for i, c in enumerate(classes)}
    y_true = np.array([c2i[c] for c in y_true_str])
    
    # Log class balance
    class_counts = {c: sum(1 for label in y_true_str if label == c) for c in classes}
    print(f"    Classes: {class_counts}")

    # clustering with sklearn version compatibility
    try:
        from packaging import version
        import sklearn
        if version.parse(sklearn.__version__) < version.parse("1.2"):
            agg = AgglomerativeClustering(n_clusters=len(classes), affinity="precomputed", linkage="average")
        else:
            agg = AgglomerativeClustering(n_clusters=len(classes), metric="precomputed", linkage="average")
    except ImportError:
        # Fallback - try both and use whichever works
        try:
            agg = AgglomerativeClustering(n_clusters=len(classes), metric="precomputed", linkage="average")
        except TypeError:
            agg = AgglomerativeClustering(n_clusters=len(classes), affinity="precomputed", linkage="average")
    
    y_pred = agg.fit_predict(D)
    
    # Log prediction balance
    pred_counts = dict(zip(*np.unique(y_pred, return_counts=True)))
    print(f"    Predictions: {pred_counts}")

    # external
    ARI = adjusted_rand_score(y_true, y_pred)
    AMI = adjusted_mutual_info_score(y_true, y_pred)
    V = v_measure_score(y_true, y_pred)

    # internal
    try:
        sil = silhouette_score(D, y_pred, metric="precomputed")
    except Exception:
        sil = np.nan
    dunn = dunn_index(D, y_pred)
    coph = cophenetic_corr(D, method="average")

    # MDS2 for DBI/CH
    try:
        X2 = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4, max_iter=300).fit_transform(D)
        dbi = davies_bouldin_score(X2, y_pred)
        ch = calinski_harabasz_score(X2, y_pred)
    except Exception:
        dbi = np.nan; ch = np.nan

    eff = cohen_d_within_between(D, y_true)
    diff, pval = permutation_pvalue_within_between(D, y_true, n_perm=3000, seed=123)

    out = {
        "ARI": float(ARI), "AMI": float(AMI), "V_measure": float(V),
        "Silhouette": float(sil) if np.isfinite(sil) else None,
        "Dunn": float(dunn) if np.isfinite(dunn) else None,
        "Cophenetic": float(coph),
        "DaviesBouldin_on_MDS2": float(dbi) if np.isfinite(dbi) else None,
        "CalinskiHarabasz_on_MDS2": float(ch) if np.isfinite(ch) else None,
        "WithinMean": eff["within_mean"], "BetweenMean": eff["between_mean"],
        "Cohen_d": eff["cohen_d"], "PermTest_diffMean": diff, "PermTest_p": pval,
    }

    if stability:
        stab = jaccard_stability(D, y_pred, B=300, seed=42)
        out.update({
            "Jaccard_mean": stab["mean"],
            "Jaccard_ci95_lo": stab["ci95_lo"],
            "Jaccard_ci95_hi": stab["ci95_hi"]
        })
    return out

def config_tag(cfg: Dict) -> str:
    items = []
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, float):
            v = f"{v:.3g}"
        items.append(f"{k}={v}")
    return "_".join(items)

# ------------------------------ Main --------------------------------------

def main():
    parser = make_parser()
    args = parser.parse_args()
    apply_preset(args)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_models(data_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files matched {data_dir / args.pattern}")

    # Determine time normalization settings
    normalize_time = not args.no_time_scaling
    use_padding = args.pad_with_last
    common_range = None
    
    # Load curves and labels with time information
    keys, curves_raw, time_vectors, labels = [], [], [], []
    for f in files:
        key = f.name
        lbl = label_from_name(key)
        try:
            y, t = load_mean_curve_from_npz(f, compute_mean_axis=args.compute_mean_axis)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
            continue
        keys.append(key); curves_raw.append(y); time_vectors.append(t); labels.append(lbl)
        
    # If not normalizing time, determine common range
    if not normalize_time:
        files_data = [(t, y) for t, y in zip(time_vectors, curves_raw)]
        common_range = find_maximum_range(files_data)

    n_models = len(keys)
    print(f"Loaded {n_models} models.")

    # Build ablation grid
    # Convert 'true'/'false' strings to bools for time_scaling and arc_length lists
    ts_options = [s.lower() == "true" for s in args.time_scaling]
    arc_length_options = [s.lower() == "true" for s in args.use_arc_length]

    # distance-specific grid configs
    grids = []
    for dist in args.distances:
        if dist == "elastic":
            for amp, sm, sw, srvf_n, wf, lw, ts, arc in product(
                args.amp_norm, args.smooth, args.smooth_win, args.srvf_norm, args.window_frac, args.lambda_warp, ts_options, arc_length_options
            ):
                grids.append({
                    "distance": "elastic",
                    "resample": args.resample,
                    "amp_norm": amp, "smooth": sm, "smooth_win": sw,
                    "srvf_norm": srvf_n, "window_frac": wf, "lambda_warp": lw,
                    "time_scaling": ts, "use_arc_length": arc, "pad_with_last": args.pad_with_last
                })
        elif dist == "dtw":
            for amp, sm, sw, ts, arc, dwf in product(
                args.amp_norm, args.smooth, args.smooth_win, ts_options, arc_length_options, args.dtw_window_frac
            ):
                grids.append({
                    "distance": "dtw",
                    "resample": args.resample,
                    "amp_norm": amp, "smooth": sm, "smooth_win": sw,
                    "dtw_window_frac": dwf,
                    "time_scaling": ts, "use_arc_length": arc, "pad_with_last": args.pad_with_last
                })
        else:
            # l2, cosine, wasserstein share preprocessing knobs
            for amp, sm, sw, ts, arc in product(args.amp_norm, args.smooth, args.smooth_win, ts_options, arc_length_options):
                grids.append({
                    "distance": dist,
                    "resample": args.resample,
                    "amp_norm": amp, "smooth": sm, "smooth_win": sw,
                    "time_scaling": ts, "use_arc_length": arc, "pad_with_last": args.pad_with_last
                })

    summary_rows = []
    D0 = None  # Reference distance matrix for comparison
    for cfg_idx, cfg in enumerate(grids):
        # Use config-specific time normalization, falling back to global setting
        cfg_normalize_time = not cfg.get("time_scaling", False) if "time_scaling" in cfg else normalize_time
        cfg_use_arc_length = cfg.get("use_arc_length", False)
        
        # Calculate common range if this config needs actual time ranges
        cfg_common_range = common_range
        if not cfg_normalize_time and cfg_common_range is None:
            # Need to calculate common range for this specific config
            files_data = [(t, y) for t, y in zip(time_vectors, curves_raw)]
            cfg_common_range = find_maximum_range(files_data)
        
        # Preprocess curves for this cfg
        pre = [
            preprocess_curve(t, y, resample_n=cfg["resample"],
                             amp_mode=cfg["amp_norm"],
                             smooth_mode=cfg["smooth"], smooth_win=cfg["smooth_win"],
                             normalize_time=cfg_normalize_time, common_range=cfg_common_range, use_padding=use_padding,
                             use_arc_length=cfg_use_arc_length)
            for t, y in zip(time_vectors, curves_raw)
        ]

        # Build distance matrix
        if cfg["distance"] == "elastic":
            D = pairwise_distance_matrix(pre, "elastic", {
                "srvf_norm": cfg["srvf_norm"],
                "window_frac": cfg["window_frac"],
                "lambda_warp": cfg["lambda_warp"],
            })
        elif cfg["distance"] == "dtw":
            D = pairwise_distance_matrix(pre, "dtw", {
                "dtw_window_frac": cfg["dtw_window_frac"]
            })
        else:
            D = pairwise_distance_matrix(pre, cfg["distance"], {})

        # Diagnostic checks
        tag = config_tag(cfg)
        print(f"[CONFIG] {tag}")
        print(f"    Distance stats: min={D.min():.4f}, max={D.max():.4f}, mean={D.mean():.4f}, std={D.std():.4f}")
        
        # Check if this distance matrix is identical to the first one
        if cfg_idx == 0:
            D0 = D.copy()
        else:
            delta = np.abs(D - D0).mean()
            corr = np.corrcoef(D.flatten(), D0.flatten())[0, 1]
            print(f"    vs first matrix: mean_abs_diff={delta:.6f}, correlation={corr:.6f}")
            if delta < 1e-12:
                print(f"    [WARN] Matrix is numerically identical to the first one!")

        # Evaluate
        stats = evaluate_distance_matrix(D, keys, labels, stability=args.stability)

        # Save matrix and config
        tag = config_tag(cfg)
        dist_path = out_dir / f"dist_{tag}.csv"
        pd.DataFrame(D, index=keys, columns=keys).to_csv(dist_path)

        (out_dir / f"config_{tag}.json").write_text(json.dumps(cfg, indent=2))

        row = {**cfg, **stats}
        summary_rows.append(row)
        print(f"[OK] {tag}  ARI={stats['ARI']:.3f}  Silh={stats['Silhouette']:.3f}  d={stats['Cohen_d']:.3f}  p={stats['PermTest_p']:.3g}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv}")
    print("Done.")

if __name__ == "__main__":
    main()
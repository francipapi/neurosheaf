#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Pairwise EDR & LCSS distances (unsupervised) for eigenvalue evolution runs.

Parallel version of mean_dtw.py that maintains exact same functionality and parameters
but leverages multiprocessing for improved performance.

What it does
------------
- Loads all runs from DATA_DIR matching: .npz (with keys 'eigenvalue_matrix','time_vector'),
  .npy (dict with those keys or a raw 2D array), and .csv/.txt (2D numeric table).
- Preprocess: floor at EPS_FLOOR, optional log1p, global normalization ('p95' or 'max').
- Resample all to a COMMON time grid of K points.
- Reduce to the MEAN eigenvalue curve for each run.
- Compute pairwise:
    * EDR (Edit Distance on Real sequences)   → edr_matrix.csv
    * LCSS (Longest Common Subsequence-based) → lcss_matrix.csv
  with a fixed, label-free epsilon (see CONFIG).

Notes
-----
- No prototypes, no class-informed tuning — fully unsupervised.
- Epsilon ε is chosen by a simple scale rule: REL_EPS_MULT × median(std of mean curves).
  You can instead hard-set ABS_EPS to a numeric value to make ε completely fixed.
- Uses multiprocessing to parallelize computationally intensive operations.

How to run
----------
    python mean_dtw_parallel.py
    python mean_dtw_parallel.py --workers 4    # Use 4 worker processes

Outputs
-------
- edr_matrix.csv
- lcss_matrix.csv
  Each CSV has header row/col with run names (file stems).

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
from tqdm import tqdm

# ========================
# CONFIG (edit here) - EXACT SAME AS ORIGINAL
# ========================
DATA_DIR         = "eigenvalueData"   # folder with your runs
OUTPUT_EDR_CSV   = "edr_matrix.csv"
OUTPUT_LCSS_CSV  = "lcss_matrix.csv"

K                = 500        # common time samples
EPS_FLOOR        = 1e-10      # floor before log1p
USE_LOG1P        = True       # apply log1p
NORMALIZE_MODE   = "none"     # 'p95', 'max', or 'none'
TOP_K_EIGS       = None       # None = all eigs; >0 = top-k; <0 = bottom-k

# Epsilon choice for EDR/LCSS (set exactly one mode)
REL_EPS_MULT     = 0.1
ABS_EPS          = None

# Gap penalty for EDR (usually 1.0)
EDR_GAP_COST     = 1.0

# ========================
# IO helpers - IDENTICAL TO ORIGINAL
# ========================

def _load_one(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load one file and return (L, t, name).
    L: (T, n_eigs), t: (T,)
    """
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

    # Ensure orientation matches time
    if L.shape[0] != t.shape[0] and L.shape[1] == t.shape[0]:
        L = L.T

    # Sort time just in case
    order = np.argsort(t)
    return L[order, :], t[order], name


# ========================
# Processing helpers - IDENTICAL TO ORIGINAL
# ========================

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
    # piecewise-linear interpolation with edge holding
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    return np.interp(t_common, t, y, left=y[0], right=y[-1])

def _mean_curve(L: np.ndarray) -> np.ndarray:
    return np.nanmean(L, axis=1)


# ========================
# EDR & LCSS (elastic, label-free) - IDENTICAL TO ORIGINAL
# ========================

def edr_distance(a: np.ndarray, b: np.ndarray, eps: float, gap_cost: float = 1.0) -> float:
    """
    Normalized Edit Distance on Real sequences (EDR).
    Match if |a[i]-b[j]| <= eps; substitution cost=1, gap cost=gap_cost.
    Returns dp[N,M] / max(N,M).
    """
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
    """
    Normalized LCSS dissimilarity: 1 - LCSS / min(N, M).
    Match if |a[i]-b[j]| <= eps. No costs for skipping non-matches.
    """
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
# Parallel worker functions
# ========================

def load_file_worker(f: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Worker function for parallel file loading."""
    try:
        return _load_one(f)
    except Exception as e:
        print(f"[WARN] Skipping {f}: {e}")
        return None

def compute_distances_worker(args: tuple) -> tuple[int, int, float, float]:
    """Worker function for parallel distance computation."""
    i, j, mean_i, mean_j, eps, gap_cost = args
    d_edr = edr_distance(mean_i, mean_j, eps=eps, gap_cost=gap_cost)
    d_lcss = lcss_distance(mean_i, mean_j, eps=eps)
    return i, j, d_edr, d_lcss

def preprocess_worker(args: tuple) -> np.ndarray:
    """Worker function for parallel preprocessing."""
    L, eps_floor, use_log1p = args
    return _preprocess(L, eps_floor, use_log1p)

def compute_mean_worker(args: tuple) -> np.ndarray:
    """Worker function for parallel mean curve computation."""
    (L, t), P, tg = args
    m = _mean_curve(P)
    return _resample_curve(m, t, tg)


# ========================
# Main - PARALLEL VERSION
# ========================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compute pairwise EDR & LCSS distances with parallel processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=mp.cpu_count(),
        help='Number of worker processes'
    )
    args = parser.parse_args()
    
    print(f"[INFO] Using {args.workers} worker processes")
    start_time = time.time()
    
    # 1) Collect files
    patterns = ["*.npz", "*.npy", "*.csv", "*.txt"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(DATA_DIR, pat)))
    files = sorted(files)
    if not files:
        raise SystemExit(f"No files found in {DATA_DIR} matching {patterns}")
    
    print(f"[INFO] Found {len(files)} files to process")

    # 2) Load all files in parallel
    print("[INFO] Loading files in parallel...")
    raw = []   # (L, t)
    names = [] # stems
    
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(load_file_worker, files),
            total=len(files),
            desc="Loading files"
        ))
    
    for r in results:
        if r is not None:
            L, t, name = r
            raw.append((L, t))
            names.append(name)

    if not raw:
        raise SystemExit("No valid inputs loaded.")
    
    print(f"[INFO] Successfully loaded {len(raw)} files")
    
    # 2b) Restrict to top-k or bottom-k eigenvalues if requested
    if TOP_K_EIGS is not None:
        new_raw = []
        if TOP_K_EIGS > 0:
            for L, t in raw:
                Lsel = L[:, :TOP_K_EIGS]
                new_raw.append((Lsel, t))
            print(f"[INFO] Using top {TOP_K_EIGS} eigenvalues per run.")
        else:
            k = abs(TOP_K_EIGS)
            for L, t in raw:
                Lsel = L[:, -k:]
                new_raw.append((Lsel, t))
            print(f"[INFO] Using bottom {k} eigenvalues per run.")
        raw = new_raw

    # 3) Common time grid (intersection)
    t_min = max(t[0] for _, t in raw)
    t_max = min(t[-1] for _, t in raw)
    if not (t_max > t_min):
        raise SystemExit("No overlapping time domain across inputs.")
    tg = np.linspace(t_min, t_max, K)

    # 4) Preprocess in parallel and global normalization
    print("[INFO] Preprocessing matrices in parallel...")
    preprocess_args = [(L, EPS_FLOOR, USE_LOG1P) for L, _ in raw]
    
    with mp.Pool(args.workers) as pool:
        pre = list(tqdm(
            pool.imap(preprocess_worker, preprocess_args),
            total=len(preprocess_args),
            desc="Preprocessing"
        ))
    
    pre = _global_normalize(pre, NORMALIZE_MODE)

    # 5) Compute mean curves on common grid in parallel
    print("[INFO] Computing mean curves in parallel...")
    mean_args = [(raw_item, P, tg) for raw_item, P in zip(raw, pre)]
    
    with mp.Pool(args.workers) as pool:
        means = list(tqdm(
            pool.imap(compute_mean_worker, mean_args),
            total=len(mean_args),
            desc="Computing means"
        ))

    # 6) Epsilon (either absolute or relative scale rule) - IDENTICAL TO ORIGINAL
    if ABS_EPS is not None:
        eps = float(ABS_EPS)
    else:
        amps = [float(np.std(m)) for m in means]
        median_std = float(np.median(amps)) if amps else 1.0
        eps = float(REL_EPS_MULT * median_std)
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-3  # tiny fallback
    print(f"[INFO] EDR/LCSS epsilon = {eps:.6g} "
          f"({'ABS' if ABS_EPS is not None else f'REL={REL_EPS_MULT}×median-std'})")

    # 7) Pairwise EDR & LCSS on mean curves - PARALLEL VERSION
    print("[INFO] Computing pairwise distances in parallel...")
    N = len(means)
    D_edr  = np.zeros((N, N), dtype=float)
    D_lcss = np.zeros((N, N), dtype=float)

    # Generate all unique pairs (i, j) where i < j
    pairs = [
        (i, j, means[i], means[j], eps, EDR_GAP_COST)
        for i in range(N) for j in range(i + 1, N)
    ]
    
    total_pairs = len(pairs)
    print(f"[INFO] Computing {total_pairs} pairwise distances...")
    
    # Compute distances in parallel
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(compute_distances_worker, pairs),
            total=total_pairs,
            desc="Computing distances"
        ))
    
    # Fill distance matrices (symmetric)
    for i, j, d_edr, d_lcss in results:
        D_edr[i, j] = D_edr[j, i] = d_edr
        D_lcss[i, j] = D_lcss[j, i] = d_lcss

    # 8) Save CSVs with header row & col - IDENTICAL TO ORIGINAL
    def save_csv(path, D, labels):
        header = ",".join([""] + labels)
        np.savetxt(path, D, delimiter=",", header=header, comments="")
        # patch in row labels (numpy.savetxt doesn't do row headers)
        with open(path, "r") as f:
            lines = f.read().splitlines()
        lines = [lines[0]] + [f"{labels[i]},{lines[i+1]}" for i in range(len(labels))]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"[OK] wrote {path}  ({D.shape[0]}×{D.shape[1]})")

    save_csv(OUTPUT_EDR_CSV,  D_edr,  names)
    save_csv(OUTPUT_LCSS_CSV, D_lcss, names)

    # 9) Optional quick print: if names use 'r' prefix for random, show separation stats - IDENTICAL TO ORIGINAL
    trained = [i for i,n in enumerate(names) if not n.lower().startswith('r')]
    random  = [i for i,n in enumerate(names) if     n.lower().startswith('r')]
    if trained and random:
        import itertools as it
        def sep_stats(D):
            within = [D[i,j] for i,j in it.combinations(trained, 2)] or [np.nan]
            cross  = [D[i,j] for i in trained for j in random] or [np.nan]
            ratio  = np.nanmean(cross) / np.nanmean(within)
            margin = np.nanmin(cross) - np.nanmax(within)
            return ratio, margin
        r1, m1 = sep_stats(D_edr)
        r2, m2 = sep_stats(D_lcss)
        print(f"[EDR ] ratio={r1:.3f}  margin(min cross - max within)={m1:+.6f}")
        print(f"[LCSS] ratio={r2:.3f}  margin(min cross - max within)={m2:+.6f}")
    
    # Show timing information
    total_time = time.time() - start_time
    print(f"[INFO] Total processing time: {total_time:.2f} seconds")
    print(f"[INFO] Processed {len(names)} runs with {args.workers} workers")

if __name__ == "__main__":
    main()
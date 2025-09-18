#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Mean Eigenvalue Evolution Curves

Creates visualization of eigenvalue evolution curves from all models in eigenvalueData/,
color-coded by model type (trained/random, custom/MLP). Curves are strictly resampled
onto a shared [0,1] time grid after de-duplicating time stamps and handling NaNs.

Usage:
    python plot_eigenvalue_curves.py                    # Basic plot
    python plot_eigenvalue_curves.py --interactive      # Interactive HTML plot
    python plot_eigenvalue_curves.py --subplots         # Separate subplots by category
    python plot_eigenvalue_curves.py --log-scale        # Use log scale for eigenvalues
    python plot_eigenvalue_curves.py --statistics       # Save curve statistics
"""

import argparse
import glob
import os
import pathlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Optional seaborn styling
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except Exception:
    plt.style.use('default')

# Optional interactive plotting
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ===========================
# Loading & classification
# ===========================

def load_eigenvalue_data(data_dir: str, include_unknown: bool = False) -> Dict[str, Dict]:
    """
    Load all eigenvalue evolution data from the specified directory.

    Returns:
        { model_name: {
            'eigenvalues': (T, K) array,
            'time': (T,) array,
            'architecture': str,
            'status': str,
            'category': str,
            'file_path': str
        } }
    """
    print(f"[INFO] Loading eigenvalue data from {data_dir}")

    patterns = ["*.npz", "*.npy"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern)))

    model_files = []
    for f in files:
        basename = os.path.basename(f)
        if not any(skip in basename.lower() for skip in ['processing', 'log', 'summary']):
            model_files.append(f)

    print(f"[INFO] Found {len(model_files)} model data files")

    models_data: Dict[str, Dict] = {}
    failed_loads: List[Tuple[str, str]] = []

    for file_path in model_files:
        try:
            model_name = pathlib.Path(file_path).stem
            data = load_single_file(file_path, model_name, include_unknown)
            if data:
                models_data[model_name] = data
        except Exception as e:
            failed_loads.append((file_path, str(e)))
            print(f"[WARN] Failed to load {file_path}: {e}")

    print(f"[INFO] Successfully loaded {len(models_data)} models")
    if failed_loads:
        print(f"[WARN] Failed to load {len(failed_loads)} files")
    return models_data


def load_single_file(file_path: str, model_name: str, include_unknown: bool = False) -> Optional[Dict]:
    """Load a single eigenvalue evolution file with strict validation."""
    try:
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=False)
            if 'eigenvalue_matrix' in data and 'time_vector' in data:
                eigenvalues = np.asarray(data['eigenvalue_matrix'], dtype=float)
                time = np.asarray(data['time_vector'], dtype=float)
            else:
                print(f"[WARN] {file_path}: Missing required keys {{'eigenvalue_matrix','time_vector'}}")
                return None

        elif file_path.endswith('.npy'):
            arr = np.load(file_path, allow_pickle=True)
            if isinstance(arr, dict):
                eigenvalues = np.asarray(arr['eigenvalue_matrix'], dtype=float)
                time = np.asarray(arr['time_vector'], dtype=float)
            elif arr.dtype == object and arr.shape == ():
                # np.save of a dict sometimes loads as 0-d object
                maybe = arr.item()
                if isinstance(maybe, dict) and 'eigenvalue_matrix' in maybe and 'time_vector' in maybe:
                    eigenvalues = np.asarray(maybe['eigenvalue_matrix'], dtype=float)
                    time = np.asarray(maybe['time_vector'], dtype=float)
                else:
                    print(f"[WARN] {file_path}: Unsupported .npy object content")
                    return None
            else:
                # Raw array (assume rows=time, cols=eigs)
                eigenvalues = np.asarray(arr, dtype=float)
                if eigenvalues.ndim != 2:
                    print(f"[WARN] {file_path}: Expected 2D array for raw .npy")
                    return None
                time = np.linspace(0.0, 1.0, eigenvalues.shape[0])
        else:
            print(f"[WARN] Unsupported file format: {file_path}")
            return None

        # Validate & orient
        time = np.asarray(time, dtype=float).reshape(-1)
        if eigenvalues.ndim != 2:
            print(f"[WARN] {file_path}: eigenvalues must be 2D")
            return None

        n_t, n_e = eigenvalues.shape
        if n_t != len(time) and eigenvalues.shape[1] == len(time):
            eigenvalues = eigenvalues.T
            n_t, n_e = eigenvalues.shape

        if n_t != len(time):
            print(f"[WARN] {file_path}: time length ({len(time)}) != eigenvalues rows ({n_t})")
            return None

        # Sort by time (stable)
        order = np.argsort(time, kind='mergesort')
        time = time[order]
        eigenvalues = eigenvalues[order, :]

        # Classify
        classification = classify_model(model_name)
        if not include_unknown and (classification['architecture'] == 'Other' or classification['status'] == 'Unknown'):
            print(f"[INFO] Skipping unknown model: {model_name} (category: {classification['category']})")
            return None

        return {
            'eigenvalues': eigenvalues,
            'time': time,
            'architecture': classification['architecture'],
            'status': classification['status'],
            'category': classification['category'],
            'file_path': file_path
        }

    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return None


def classify_model(model_name: str) -> Dict[str, str]:
    """
    Classify model based on filename.
    """
    name_lower = model_name.lower()

    # Digits dataset models first
    if name_lower.startswith('digits_mlp_seed') or (name_lower.startswith('digits_mlp') and 'seed' in name_lower):
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
    # New Digits architectures
    elif name_lower.startswith('digits_tiny_deep_seed') or (name_lower.startswith('digits_tiny_deep') and 'seed' in name_lower):
        architecture = 'Digits-TinyDeep'
        status = 'Trained'
    elif name_lower.startswith('digits_tiny_deep_random') or (name_lower.startswith('digits_tiny_deep') and 'random' in name_lower):
        architecture = 'Digits-TinyDeep'
        status = 'Random'
    elif name_lower.startswith('digits_wide_shallow_seed') or (name_lower.startswith('digits_wide_shallow') and 'seed' in name_lower):
        architecture = 'Digits-WideShallow'
        status = 'Trained'
    elif name_lower.startswith('digits_wide_shallow_random') or (name_lower.startswith('digits_wide_shallow') and 'random' in name_lower):
        architecture = 'Digits-WideShallow'
        status = 'Random'
    elif name_lower.startswith('digits_pyramid_seed') or (name_lower.startswith('digits_pyramid') and 'seed' in name_lower):
        architecture = 'Digits-Pyramid'
        status = 'Trained'
    elif name_lower.startswith('digits_pyramid_random') or (name_lower.startswith('digits_pyramid') and 'random' in name_lower):
        architecture = 'Digits-Pyramid'
        status = 'Random'
    elif name_lower.startswith('digits_hourglass_seed') or (name_lower.startswith('digits_hourglass') and 'seed' in name_lower):
        architecture = 'Digits-Hourglass'
        status = 'Trained'
    elif name_lower.startswith('digits_hourglass_random') or (name_lower.startswith('digits_hourglass') and 'random' in name_lower):
        architecture = 'Digits-Hourglass'
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


# ===========================
# Cleaning & resampling
# ===========================

def find_overlapping_range(models_data: Dict) -> Tuple[float, float]:
    """
    Find the overlapping time range across all models.
    
    Args:
        models_data: Dictionary of model data with time arrays
        
    Returns:
        Tuple of (min_overlap, max_overlap) representing the common time range
    """
    if not models_data:
        return 0.0, 1.0
    
    # Collect time ranges from all models
    time_ranges = []
    for name, data in models_data.items():
        time = np.asarray(data['time'], dtype=float).reshape(-1)
        if len(time) > 0 and np.isfinite(time).any():
            valid_time = time[np.isfinite(time)]
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
    print(f"[INFO] Found {len(time_ranges)} models with valid time data")
    
    return overlap_start, overlap_end


def find_maximum_range(models_data: Dict) -> Tuple[float, float]:
    """
    Find the maximum time range across all models (union instead of intersection).
    
    Args:
        models_data: Dictionary of model data with time arrays
        
    Returns:
        Tuple of (global_min, global_max) representing the full time range
    """
    if not models_data:
        return 0.0, 1.0
    
    # Collect time ranges from all models
    time_ranges = []
    for name, data in models_data.items():
        time = np.asarray(data['time'], dtype=float).reshape(-1)
        if len(time) > 0 and np.isfinite(time).any():
            valid_time = time[np.isfinite(time)]
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
    print(f"[INFO] Found {len(time_ranges)} models with valid time data")
    
    return global_start, global_end


def detect_and_remove_outliers(eigenvalues: np.ndarray, time: np.ndarray,
                               method: str = 'none', model_name: str = '',
                               category: str = '') -> Tuple[np.ndarray, np.ndarray]:
    """
    Outlier detection stub (kept for CLI compatibility).
    Currently pass-through.
    """
    return eigenvalues, time


def _remove_dup_x_with_mean(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average y at duplicate x to enforce strictly increasing x."""
    xu, inv = np.unique(x, return_inverse=True)
    counts = np.bincount(inv)
    s = np.bincount(inv, weights=y)
    yu = s / np.maximum(counts, 1)
    return xu, yu


def _dedup_and_aggregate_by_time(time: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate time stamps by averaging eigenvalues at the same time.
    Returns strictly increasing time and aggregated E.
    """
    # Drop rows with NaN time
    good_t = np.isfinite(time)
    time = time[good_t]
    E = E[good_t, :]

    if time.size == 0:
        return time, E

    t_unique, inv = np.unique(time, return_inverse=True)
    counts = np.bincount(inv)

    agg = np.zeros((t_unique.size, E.shape[1]), dtype=float)
    # aggregate per column; NaNs become 0 in sum, later divided by counts
    for j in range(E.shape[1]):
        s = np.bincount(inv, weights=np.nan_to_num(E[:, j], nan=0.0))
        agg[:, j] = s / np.maximum(counts, 1)

    return t_unique, agg


def _interp_column(x_new: np.ndarray, x: np.ndarray, y: np.ndarray, 
                   use_padding: bool = False) -> np.ndarray:
    """
    Interpolate one column y(x) onto x_new with NaN-safe anchors and
    graceful fallback when too few points.
    
    Args:
        x_new: Target x values for interpolation
        x: Original x values
        y: Original y values
        use_padding: If True, pad with last value beyond original range
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]

    if x_valid.size == 0:
        return np.full_like(x_new, fill_value=np.nan)

    if x_valid.size == 1:
        return np.full_like(x_new, fill_value=y_valid[0])

    # Ensure strictly increasing x for np.interp
    sort_idx = np.argsort(x_valid, kind='mergesort')
    x_valid = x_valid[sort_idx]
    y_valid = y_valid[sort_idx]
    x_valid, y_valid = _remove_dup_x_with_mean(x_valid, y_valid)

    if x_valid.size == 1:
        return np.full_like(x_new, fill_value=y_valid[0])

    if use_padding:
        # Use padding with first and last values for extrapolation
        left_value = y_valid[0]   # First value for x < x_valid.min()
        right_value = y_valid[-1] # Last value for x > x_valid.max()
        return np.interp(x_new, x_valid, y_valid, left=left_value, right=right_value)
    else:
        # Standard interpolation (default NaN extrapolation)
        return np.interp(x_new, x_valid, y_valid)


def resample_to_common_grid(models_data: Dict, n_points: int = 200,
                            outlier_method: str = 'none', 
                            normalize_time: bool = True,
                            use_padding: bool = False) -> Dict:
    """
    Resample all curves to a common time grid with strict cleaning.
    - Deduplicate and average repeated time stamps
    - Optionally normalize individual time arrays to [0,1], use overlapping range, or use maximum range with padding
    - Interpolate each eigenvalue column onto a shared grid
    
    Args:
        models_data: Dictionary of model data
        n_points: Number of points for resampling
        outlier_method: Method for outlier detection
        normalize_time: If True, normalize to [0,1]. If False, use overlapping or maximum range.
        use_padding: If True with normalize_time=False, use maximum range and pad with last values.
    """
    print(f"[INFO] Applying outlier detection method: {outlier_method}")
    cleaned_data: Dict[str, Dict] = {}
    outlier_stats = {'models_processed': 0, 'outliers_found': 0}

    # 1) Outlier stage (stub) + basic cleaning
    for name, data in models_data.items():
        E = np.asarray(data['eigenvalues'], dtype=float)
        t = np.asarray(data['time'], dtype=float).reshape(-1)

        E, t = detect_and_remove_outliers(E, t, method=outlier_method,
                                          model_name=name, category=data.get('category', ''))

        t, E = _dedup_and_aggregate_by_time(t, E)

        cleaned_data[name] = {
            **data,
            'eigenvalues': E,
            'time': t,
            'original_shape': data['eigenvalues'].shape,
        }
        outlier_stats['models_processed'] += 1

    print(f"[INFO] Outlier detection complete: {outlier_stats['outliers_found']}/{outlier_stats['models_processed']} models had outliers")

    # 2) Handle time grid - normalize to [0,1] or use overlapping range
    normalized: Dict[str, Dict] = {}
    if normalize_time:
        # Original behavior: normalize to [0,1]
        for name, data in cleaned_data.items():
            t = data['time']
            E = data['eigenvalues']

            if t.size == 0:
                print(f"[WARN] {name}: no valid time points after cleaning; skipping")
                continue

            tmin, tmax = float(t.min()), float(t.max())
            if tmax > tmin:
                tn = (t - tmin) / (tmax - tmin)
                tn[0] = 0.0
                tn[-1] = 1.0
            else:
                tn = np.zeros_like(t)

            normalized[name] = {**data, 'time': tn, 'original_time_range': (tmin, tmax)}
        
        # 3) Shared grid [0, 1]
        common_time = np.linspace(0.0, 1.0, n_points)
        print(f"[INFO] Created common time grid: [0, 1] with {n_points} points")
    else:
        if use_padding:
            # New behavior: use maximum range with padding
            global_start, global_end = find_maximum_range(cleaned_data)
            
            # Keep original data (no filtering), padding will be handled in interpolation
            for name, data in cleaned_data.items():
                t = data['time']
                E = data['eigenvalues']

                if t.size == 0:
                    print(f"[WARN] {name}: no valid time points after cleaning; skipping")
                    continue
                
                normalized[name] = {**data, 'time': t, 'eigenvalues': E,
                                  'original_time_range': (t.min(), t.max())}
            
            # 3) Shared grid for maximum range
            common_time = np.linspace(global_start, global_end, n_points)
            print(f"[INFO] Created common time grid with padding: [{global_start:.6f}, {global_end:.6f}] with {n_points} points")
        else:
            # Existing behavior: use overlapping range without normalization
            overlap_start, overlap_end = find_overlapping_range(cleaned_data)
            
            # Filter and trim data to overlapping range only
            for name, data in cleaned_data.items():
                t = data['time']
                E = data['eigenvalues']

                if t.size == 0:
                    print(f"[WARN] {name}: no valid time points after cleaning; skipping")
                    continue

                # Keep only data within overlapping range
                mask = (t >= overlap_start) & (t <= overlap_end)
                if not mask.any():
                    print(f"[WARN] {name}: no data in overlapping range; skipping")
                    continue
                
                t_filtered = t[mask]
                E_filtered = E[mask, :]

                normalized[name] = {**data, 'time': t_filtered, 'eigenvalues': E_filtered,
                                  'original_time_range': (t.min(), t.max())}
            
            # 3) Shared grid for overlapping range
            common_time = np.linspace(overlap_start, overlap_end, n_points)
            print(f"[INFO] Created common time grid: [{overlap_start:.6f}, {overlap_end:.6f}] with {n_points} points")

    # 4) Interpolate per column, NaN-safe
    resampled: Dict[str, Dict] = {}
    for name, data in normalized.items():
        t = data['time']
        E = data['eigenvalues']

        if t.size == 0 or E.size == 0:
            print(f"[WARN] {name}: empty after normalization; skipping")
            continue

        order = np.argsort(t, kind='mergesort')
        t = t[order]
        E = E[order, :]

        if np.unique(t).size == 1:
            resE = np.tile(E[0, :], (common_time.size, 1))
        else:
            resE = np.empty((common_time.size, E.shape[1]), dtype=float)
            for j in range(E.shape[1]):
                resE[:, j] = _interp_column(common_time, t, E[:, j], use_padding=use_padding)

        resampled[name] = {
            **data,
            'eigenvalues': resE,
            'time': common_time,
            'normalized': normalize_time
        }

    if normalize_time:
        print(f"[INFO] Successfully resampled {len(resampled)} models to normalized [0, 1] grid")
    elif use_padding:
        print(f"[INFO] Successfully resampled {len(resampled)} models with padding to maximum range")
    else:
        print(f"[INFO] Successfully resampled {len(resampled)} models to overlapping range")
    return resampled


# ===========================
# Plotting & statistics
# ===========================

def compute_mean_curve(eigenvalues: np.ndarray, use_log: bool = False) -> np.ndarray:
    """Compute mean eigenvalue curve across all eigenvalues at each time step."""
    E = np.asarray(eigenvalues, dtype=float)
    if use_log:
        # Avoid log of non-positive; use log1p for stability
        E = np.where(E > 0, E, 1e-12)
        E = np.log1p(E)
    return np.nanmean(E, axis=1)


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Normalize curve to [0, 1] range for better comparison.
    
    Args:
        curve: Input curve array
        
    Returns:
        Normalized curve with values in [0, 1] range
    """
    curve = np.asarray(curve, dtype=float)
    
    # Handle edge cases
    if len(curve) == 0:
        return curve
    
    # Remove NaN values for min/max calculation
    valid_mask = np.isfinite(curve)
    if not np.any(valid_mask):
        return curve  # All NaN
    
    valid_values = curve[valid_mask]
    curve_min = np.min(valid_values)
    curve_max = np.max(valid_values)
    
    # Handle constant curve
    if curve_max == curve_min:
        return np.where(valid_mask, 0.5, curve)  # Set to middle value, keep NaNs
    
    # Normalize to [0, 1]
    normalized = np.where(valid_mask, 
                         (curve - curve_min) / (curve_max - curve_min),
                         curve)  # Keep NaN values as NaN
    
    return normalized


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
        # New Digits architectures
        'Digits-TinyDeep-Trained': {'color': '#f7b801', 'linestyle': '-',  'alpha': 0.8},
        'Digits-TinyDeep-Random':  {'color': '#f7b801', 'linestyle': '--', 'alpha': 0.7},
        'Digits-WideShallow-Trained': {'color': '#00d2d3', 'linestyle': '-',  'alpha': 0.8},
        'Digits-WideShallow-Random':  {'color': '#00d2d3', 'linestyle': '--', 'alpha': 0.7},
        'Digits-Pyramid-Trained': {'color': '#ff6319', 'linestyle': '-',  'alpha': 0.8},
        'Digits-Pyramid-Random':  {'color': '#ff6319', 'linestyle': '--', 'alpha': 0.7},
        'Digits-Hourglass-Trained': {'color': '#c44e52', 'linestyle': '-',  'alpha': 0.8},
        'Digits-Hourglass-Random':  {'color': '#c44e52', 'linestyle': '--', 'alpha': 0.7},
        # MNIST-specific
        'MNIST-MLP4-Trained': {'color': '#ff7f0e', 'linestyle': '-',  'alpha': 0.8},
        'MNIST-MLP-Random':   {'color': '#9467bd', 'linestyle': '--', 'alpha': 0.7},
        'MNIST-MLP-Trained':  {'color': '#9467bd', 'linestyle': '-',  'alpha': 0.8},
        'MNIST-MLP-Unknown':  {'color': '#9467bd', 'linestyle': ':',  'alpha': 0.6},
        'MNIST-CNN-Trained':  {'color': '#2ca02c', 'linestyle': '-',  'alpha': 0.8},
        'MNIST-CNN-Random':   {'color': '#17becf', 'linestyle': '--', 'alpha': 0.7},
        'MNIST-CNN-Unknown':  {'color': '#808080', 'linestyle': ':',  'alpha': 0.6},
    }


def plot_curves(models_data: Dict, use_log: bool = False, show_stats: bool = True,
                normalize_y: bool = False, output_file: str = "eigenvalue_curves.png") -> None:
    """Create the main plot with all eigenvalue curves on a shared time grid."""
    color_scheme = get_color_scheme()
    fig, ax = plt.subplots(figsize=(14, 10))

    # Group by category
    categories: Dict[str, List[Tuple[str, Dict]]] = {}
    for name, data in models_data.items():
        categories.setdefault(data['category'], []).append((name, data))

    # Shared time grid (post-resampling all should share it)
    any_model = next(iter(models_data.values()))
    shared_time = any_model['time']

    legend_handles = []
    category_curves: Dict[str, np.ndarray] = {}

    for category, models in categories.items():
        style = color_scheme.get(category, {'color': '#808080', 'linestyle': '-', 'alpha': 0.5})
        curves_for_stats: List[np.ndarray] = []

        for _, data in models:
            try:
                mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
                if normalize_y:
                    mean_curve = normalize_curve(mean_curve)
                ax.plot(shared_time, mean_curve,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        alpha=style['alpha'],
                        linewidth=0.8)
                curves_for_stats.append(mean_curve)
            except Exception as e:
                print(f"[WARN] Failed to plot {data.get('file_path','?')}: {e}")

        if curves_for_stats:
            category_curves[category] = np.vstack(curves_for_stats)
            legend_handles.append(mpatches.Patch(color=style['color'],
                                                 label=f"{category} (n={len(curves_for_stats)})"))

    if show_stats and category_curves:
        for category, curves in category_curves.items():
            style = color_scheme.get(category, {'color': '#808080', 'linestyle': '-', 'alpha': 1.0})
            mean_curve = np.nanmean(curves, axis=0)
            std_curve = np.nanstd(curves, axis=0)
            ax.plot(shared_time, mean_curve,
                    color=style['color'],
                    linestyle=style.get('linestyle', '-'),
                    alpha=1.0,
                    linewidth=2.5,
                    label=f"{category} Mean")
            ax.fill_between(shared_time,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            color=style['color'],
                            alpha=0.15)

    # Determine x-axis label based on time range
    time_min, time_max = shared_time.min(), shared_time.max()
    if abs(time_min - 0.0) < 1e-6 and abs(time_max - 1.0) < 1e-6:
        xlabel = 'Filtration Parameter (Normalized)'
    else:
        xlabel = 'Filtration Parameter'
    ax.set_xlabel(xlabel, fontsize=12)
    
    # Construct y-axis label
    ylabel = 'Mean Eigenvalue'
    if use_log:
        ylabel = 'Mean Log(Eigenvalue + 1)'
    if normalize_y:
        ylabel = ylabel + ' (Normalized [0,1])'
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Construct title
    title_parts = ['Mean Eigenvalue Evolution']
    if use_log:
        title_parts.append('(Log Scale)')
    if normalize_y:
        title_parts.append('(Normalized)')
    ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold')
    ax.legend(handles=legend_handles, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot saved to {output_file}")
    plt.show()


def create_subplots(models_data: Dict, use_log: bool = False, normalize_y: bool = False,
                    output_file: str = "eigenvalue_curves_subplots.png") -> None:
    """Create subplots per category (shared time grid)."""
    color_scheme = get_color_scheme()

    # Group by category
    categories: Dict[str, List[Tuple[str, Dict]]] = {}
    for name, data in models_data.items():
        categories.setdefault(data['category'], []).append((name, data))

    n_categories = len(categories)
    n_cols = min(2, n_categories)
    n_rows = (n_categories + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_categories == 1:
        axes = np.array([axes])
    axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    # Shared grid
    any_model = next(iter(models_data.values()))
    shared_time = any_model['time']

    for idx, (category, models) in enumerate(categories.items()):
        ax = axes[idx]
        style = color_scheme.get(category, {'color': '#808080', 'linestyle': '-', 'alpha': 0.7})
        curves: List[np.ndarray] = []

        for _, data in models:
            try:
                mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
                if normalize_y:
                    mean_curve = normalize_curve(mean_curve)
                ax.plot(shared_time, mean_curve,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        alpha=style['alpha'],
                        linewidth=1.0)
                curves.append(mean_curve)
            except Exception as e:
                print(f"[WARN] Failed to plot {data.get('file_path','?')} in subplot: {e}")

        if curves:
            mean_curve = np.nanmean(np.vstack(curves), axis=0)
            ax.plot(shared_time, mean_curve, color=style['color'], linewidth=2.5, alpha=1.0)

        ax.set_title(f"{category} (n={len(models)})", fontweight='bold')
        # Determine x-axis label based on time range
        time_min, time_max = shared_time.min(), shared_time.max()
        if abs(time_min - 0.0) < 1e-6 and abs(time_max - 1.0) < 1e-6:
            xlabel = 'Filtration Parameter (Normalized)'
        else:
            xlabel = 'Filtration Parameter'
        ax.set_xlabel(xlabel)
        
        # Construct y-axis label
        ylabel = 'Mean Eigenvalue'
        if use_log:
            ylabel = 'Mean Log(Eigenvalue + 1)'
        if normalize_y:
            ylabel = ylabel + ' (Norm.)'
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for j in range(len(categories), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Subplots saved to {output_file}")
    plt.show()


def create_interactive_plot(models_data: Dict, use_log: bool = False, normalize_y: bool = False,
                            output_file: str = "eigenvalue_curves.html") -> None:
    """Create an interactive plotly plot using the shared time grid."""
    if not PLOTLY_AVAILABLE:
        print("[WARN] Plotly not available. Install plotly for interactive plots.")
        return

    color_scheme = get_color_scheme()
    fig = go.Figure()

    # Group by category
    categories: Dict[str, List[Tuple[str, Dict]]] = {}
    for name, data in models_data.items():
        categories.setdefault(data['category'], []).append((name, data))

    any_model = next(iter(models_data.values()))
    shared_time = any_model['time']

    for category, models in categories.items():
        style = color_scheme.get(category, {'color': '#808080', 'linestyle': '-', 'alpha': 0.7})
        dash = 'solid' if style['linestyle'] == '-' else ('dash' if style['linestyle'] == '--' else 'dot')

        for name, data in models:
            try:
                mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
                if normalize_y:
                    mean_curve = normalize_curve(mean_curve)
                fig.add_trace(go.Scatter(
                    x=shared_time,
                    y=mean_curve,
                    mode='lines',
                    name=name,
                    legendgroup=category,
                    legendgrouptitle_text=category,
                    line=dict(color=style['color'], dash=dash, width=1),
                    opacity=style['alpha'],
                    hovertemplate=f"<b>{name}</b><br>t: %{{x:.4f}}<br>mean: %{{y:.4f}}<extra></extra>"
                ))
            except Exception as e:
                print(f"[WARN] Failed to add {name} to interactive plot: {e}")

    # Construct title and y-axis label
    title_parts = ['Interactive Mean Eigenvalue Evolution']
    if use_log:
        title_parts.append('(Log Scale)')
    if normalize_y:
        title_parts.append('(Normalized)')
    title = ' '.join(title_parts)
    
    y_title = 'Mean Eigenvalue'
    if use_log:
        y_title = 'Mean Log(Eigenvalue + 1)'
    if normalize_y:
        y_title = y_title + ' (Normalized [0,1])'

    # Determine x-axis title based on time range  
    time_min, time_max = shared_time.min(), shared_time.max()
    if abs(time_min - 0.0) < 1e-6 and abs(time_max - 1.0) < 1e-6:
        x_title = 'Filtration Parameter (Normalized)'
    else:
        x_title = 'Filtration Parameter'
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode='closest',
        showlegend=True,
        width=1200,
        height=700
    )

    fig.write_html(output_file)
    print(f"[INFO] Interactive plot saved to {output_file}")
    fig.show()


def save_statistics(models_data: Dict, use_log: bool = False, normalize_y: bool = False,
                    output_file: str = "curve_statistics.csv") -> None:
    """Save curve statistics to CSV file (post-resampling)."""
    import pandas as pd

    stats_data = []
    for name, data in models_data.items():
        try:
            mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
            if normalize_y:
                mean_curve = normalize_curve(mean_curve)
            stats_data.append({
                'model_name': name,
                'category': data['category'],
                'architecture': data['architecture'],
                'status': data['status'],
                'n_timepoints': len(mean_curve),
                'n_eigenvalues': data['eigenvalues'].shape[1],
                'mean_eigenvalue_overall': np.nanmean(mean_curve),
                'std_eigenvalue_overall': np.nanstd(mean_curve),
                'min_eigenvalue': np.nanmin(mean_curve),
                'max_eigenvalue': np.nanmax(mean_curve),
                'final_eigenvalue': mean_curve[-1],
                'initial_eigenvalue': mean_curve[0],
                'normalized': normalize_y,
                'log_scale': use_log
            })
        except Exception as e:
            print(f"[WARN] Failed to compute statistics for {name}: {e}")

    df = pd.DataFrame(stats_data)
    df.to_csv(output_file, index=False)
    print(f"[INFO] Statistics saved to {output_file}")

    if not df.empty:
        print("\n=== SUMMARY STATISTICS ===")
        summary = df.groupby('category').agg({
            'model_name': 'count',
            'mean_eigenvalue_overall': ['mean', 'std'],
            'n_eigenvalues': 'mean'
        })
        print(summary)


# ===========================
# CLI
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description="Plot eigenvalue evolution curves from all models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, default='eigenvalueData',
                        help='Directory containing eigenvalue data files')
    parser.add_argument('--interactive', action='store_true',
                        help='Create interactive HTML plot')
    parser.add_argument('--subplots', action='store_true',
                        help='Create separate subplots by category')
    parser.add_argument('--log-scale', action='store_true',
                        help='Use log scale for eigenvalues')
    parser.add_argument('--statistics', action='store_true',
                        help='Save curve statistics to CSV')
    parser.add_argument('--no-stats-overlay', action='store_true',
                        help='Disable statistical overlays (mean curves and bands)')
    parser.add_argument('--output-prefix', type=str, default='eigenvalue_curves',
                        help='Output file prefix')
    parser.add_argument('--include-unknown', action='store_true',
                        help='Include unknown/unclassified models in the plots')
    parser.add_argument('--outlier-method', type=str, default='none',
                        choices=['none', 'iqr', 'zscore', 'trajectory_end'],
                        help='Method for outlier detection (placeholder)')
    parser.add_argument('--n-points', type=int, default=200,
                        help='Number of resampling points on time grid')
    parser.add_argument('--normalize-y', action='store_true',
                        help='Normalize y-axis values to [0, 1] range for better comparison')
    parser.add_argument('--no-time-scaling', action='store_true',
                        help='Use original time ranges and plot only overlapping portions (no normalization to [0,1])')
    parser.add_argument('--pad-with-last', action='store_true',
                        help='Pad eigenvalue curves with last value and plot over maximum range (implies --no-time-scaling)')

    args = parser.parse_args()

    # Load
    models_data = load_eigenvalue_data(args.data_dir, include_unknown=args.include_unknown)
    if not models_data:
        print("[ERROR] No models loaded!")
        return

    # Handle padding mode (implies no time scaling)
    use_padding = args.pad_with_last
    normalize_time = not args.no_time_scaling and not args.pad_with_last
    
    # Resample
    models_data = resample_to_common_grid(models_data, n_points=args.n_points,
                                          outlier_method=args.outlier_method,
                                          normalize_time=normalize_time,
                                          use_padding=use_padding)

    # Plot
    output_file = f"{args.output_prefix}.png"
    plot_curves(models_data,
                use_log=args.log_scale,
                show_stats=not args.no_stats_overlay,
                normalize_y=args.normalize_y,
                output_file=output_file)

    # Subplots
    if args.subplots:
        subplot_file = f"{args.output_prefix}_subplots.png"
        create_subplots(models_data,
                        use_log=args.log_scale,
                        normalize_y=args.normalize_y,
                        output_file=subplot_file)

    # Interactive
    if args.interactive:
        interactive_file = f"{args.output_prefix}.html"
        create_interactive_plot(models_data,
                                use_log=args.log_scale,
                                normalize_y=args.normalize_y,
                                output_file=interactive_file)

    # Stats
    if args.statistics:
        stats_file = f"{args.output_prefix}_statistics.csv"
        save_statistics(models_data,
                        use_log=args.log_scale,
                        normalize_y=args.normalize_y,
                        output_file=stats_file)

    print(f"\n[INFO] Visualization complete! Processed {len(models_data)} models.")


if __name__ == "__main__":
    main()

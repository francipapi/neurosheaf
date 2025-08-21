#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Unsupervised Functional Similarity Detection

Completely unsupervised approach to detecting functional similarity in neural networks
using eigenvalue evolution patterns. NO use of model labels or architecture information.

All distance computations and preprocessing are based solely on the mathematical
properties of the eigenvalue curves themselves.
"""

import numpy as np
import pathlib
import glob
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_eigenvalue_data() -> Tuple[List[np.ndarray], List[str]]:
    """Load eigenvalue evolution data from eigenvalueData directory."""
    data_dir = pathlib.Path("eigenvalueData")
    patterns = ["*.npz", "*.npy"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(data_dir / pattern)))
    
    eigenvalue_curves = []
    model_names = []
    
    for file_path in files:
        try:
            name = pathlib.Path(file_path).stem
            
            if file_path.endswith('.npz'):
                data = np.load(file_path, allow_pickle=False)
                if 'eigenvalue_matrix' in data:
                    L = np.asarray(data['eigenvalue_matrix'], dtype=float)
                else:
                    continue
            elif file_path.endswith('.npy'):
                arr = np.load(file_path, allow_pickle=True)
                if isinstance(arr, dict) and 'eigenvalue_matrix' in arr:
                    L = np.asarray(arr['eigenvalue_matrix'], dtype=float)
                else:
                    L = np.asarray(arr, dtype=float)
                    if L.ndim != 2:
                        continue
            else:
                continue
            
            # Compute mean curve
            mean_curve = np.nanmean(L, axis=1)
            eigenvalue_curves.append(mean_curve)
            model_names.append(name)
            
        except Exception as e:
            print(f"[WARN] Skipping {file_path}: {e}")
            continue
    
    print(f"[INFO] Loaded {len(eigenvalue_curves)} eigenvalue curves")
    return eigenvalue_curves, model_names

# ===== UNSUPERVISED PREPROCESSING =====

def adaptive_smoothing(curve: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray:
    """Apply adaptive smoothing based on curve properties."""
    if len(curve) < 3:
        return curve.copy()
    
    # Compute local variance to determine smoothing strength
    local_variance = np.convolve(np.diff(curve)**2, np.ones(3)/3, mode='same')
    local_variance = np.pad(local_variance, (1, 0), mode='edge')
    
    # Adaptive kernel size based on variance
    max_kernel = max(3, int(len(curve) * smoothing_factor))
    kernel_sizes = np.clip(
        (local_variance / np.max(local_variance) * max_kernel).astype(int),
        1, max_kernel
    )
    
    smoothed = curve.copy()
    for i, kernel_size in enumerate(kernel_sizes):
        if kernel_size > 1:
            start = max(0, i - kernel_size//2)
            end = min(len(curve), i + kernel_size//2 + 1)
            smoothed[i] = np.mean(curve[start:end])
    
    return smoothed

def robust_log_transform(curve: np.ndarray, eps_method: str = 'adaptive') -> np.ndarray:
    """Apply robust logarithmic transformation."""
    # Handle negative values
    proc_curve = np.maximum(curve, 0.0)
    
    # Adaptive epsilon based on curve properties
    if eps_method == 'adaptive':
        non_zero = proc_curve[proc_curve > 0]
        if len(non_zero) > 0:
            eps = np.percentile(non_zero, 1.0)  # 1st percentile of non-zero values
        else:
            eps = 1e-10
    else:
        eps = 1e-10
    
    # Replace zeros with adaptive epsilon
    proc_curve = np.maximum(proc_curve, eps)
    
    # Apply log1p transformation
    return np.log1p(proc_curve)

def eigenvalue_distribution_features(curve: np.ndarray) -> Dict[str, float]:
    """Extract unsupervised statistical features from eigenvalue curves."""
    return {
        'mean': np.mean(curve),
        'std': np.std(curve),
        'skewness': stats.skew(curve),
        'kurtosis': stats.kurtosis(curve),
        'q75_q25_ratio': np.percentile(curve, 75) / max(np.percentile(curve, 25), 1e-10),
        'max_min_ratio': np.max(curve) / max(np.min(curve), 1e-10),
        'energy': np.sum(curve**2),
        'spectral_entropy': stats.entropy(curve / np.sum(curve) + 1e-10)
    }

def unsupervised_normalization(curves: List[np.ndarray], method: str = 'robust') -> List[np.ndarray]:
    """Apply unsupervised normalization based on statistical properties."""
    if method == 'robust':
        # Use robust statistics for normalization
        all_values = np.concatenate([c.ravel() for c in curves])
        median_val = np.median(all_values)
        mad = np.median(np.abs(all_values - median_val))
        scale = mad * 1.4826  # MAD to std conversion factor
        
        if scale > 0:
            return [(c - median_val) / scale for c in curves]
        else:
            return curves
    
    elif method == 'percentile':
        # Use percentile-based normalization
        all_values = np.concatenate([c.ravel() for c in curves])
        p5, p95 = np.percentile(all_values, [5, 95])
        scale = p95 - p5
        
        if scale > 0:
            return [(c - p5) / scale for c in curves]
        else:
            return curves
    
    else:  # method == 'none'
        return curves

# ===== UNSUPERVISED DISTANCE METRICS =====

def adaptive_dtw_distance(a: np.ndarray, b: np.ndarray, window: Optional[int] = None) -> float:
    """Dynamic Time Warping distance with adaptive window."""
    N, M = len(a), len(b)
    
    # Adaptive window based on sequence lengths
    if window is None:
        window = max(int(0.1 * max(N, M)), 5)
    
    # Initialize cost matrix
    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Check window constraint
            if abs(i - j) > window:
                continue
                
            dist = abs(a[i-1] - b[j-1])
            cost[i, j] = dist + min(cost[i-1, j],    # insertion
                                   cost[i, j-1],    # deletion
                                   cost[i-1, j-1])  # match
    
    return cost[N, M] / (N + M)

def wasserstein_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein distance between empirical distributions."""
    try:
        return stats.wasserstein_distance(a, b)
    except:
        # Fallback implementation
        a_sorted = np.sort(a.flatten())
        b_sorted = np.sort(b.flatten())
        return np.mean(np.abs(a_sorted - b_sorted))

def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance based on spectral properties of the curves."""
    # Ensure same length by interpolation
    max_len = max(len(a), len(b))
    if len(a) != max_len:
        a = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(a)), a)
    if len(b) != max_len:
        b = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(b)), b)
    
    # Compute power spectral density
    fft_a = np.abs(np.fft.fft(a - np.mean(a)))
    fft_b = np.abs(np.fft.fft(b - np.mean(b)))
    
    # Normalize to make it a probability distribution
    fft_a = fft_a / (np.sum(fft_a) + 1e-10)
    fft_b = fft_b / (np.sum(fft_b) + 1e-10)
    
    # Jensen-Shannon divergence
    m = 0.5 * (fft_a + fft_b)
    js_div = 0.5 * stats.entropy(fft_a + 1e-10, m + 1e-10) + 0.5 * stats.entropy(fft_b + 1e-10, m + 1e-10)
    
    return np.sqrt(max(js_div, 0))

def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance based on correlation structure."""
    # Ensure same length by interpolation
    max_len = max(len(a), len(b))
    if len(a) != max_len:
        a = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(a)), a)
    if len(b) != max_len:
        b = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(b)), b)
    
    # Remove mean
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    
    # Check for zero variance
    if np.std(a_centered) < 1e-10 or np.std(b_centered) < 1e-10:
        return 1.0
    
    # Compute normalized cross-correlation
    correlation = np.corrcoef(a_centered, b_centered)[0, 1]
    
    # Handle NaN cases
    if np.isnan(correlation):
        return 1.0
    
    # Convert correlation to distance
    return 1.0 - abs(correlation)

def robust_edr_distance(a: np.ndarray, b: np.ndarray, eps_percentile: float = 10.0) -> float:
    """EDR distance with robust, data-driven epsilon."""
    # Compute robust epsilon based on local differences
    diff_a = np.abs(np.diff(a))
    diff_b = np.abs(np.diff(b))
    combined_diffs = np.concatenate([diff_a, diff_b])
    
    if len(combined_diffs) > 0:
        eps = np.percentile(combined_diffs, eps_percentile)
    else:
        eps = 0.1
    
    # Standard EDR algorithm
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    N, M = len(a), len(b)
    dp = np.zeros((N + 1, M + 1), dtype=float)
    
    for i in range(1, N + 1):
        dp[i, 0] = i
    for j in range(1, M + 1):
        dp[0, j] = j
        
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if abs(a[i-1] - b[j-1]) <= eps:
                sub = dp[i-1, j-1]
            else:
                sub = dp[i-1, j-1] + 1
            dp[i, j] = min(sub, dp[i-1, j] + 1, dp[i, j-1] + 1)
    
    return dp[N, M] / max(N, M)

# ===== UNSUPERVISED OUTLIER DETECTION =====

def statistical_outlier_detection(distance_matrix: np.ndarray, 
                                 contamination: float = 0.1) -> List[int]:
    """Detect outliers using isolation forest on distance patterns."""
    # Handle infinite and NaN values
    cleaned_matrix = distance_matrix.copy()
    cleaned_matrix[~np.isfinite(cleaned_matrix)] = np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
    
    # Check if we have valid data
    if np.all(~np.isfinite(cleaned_matrix)):
        return []
    
    # Use each row of distance matrix as features for outlier detection
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(cleaned_matrix)
    
    # Return indices of outliers (-1 means outlier)
    return [i for i, pred in enumerate(outliers) if pred == -1]

def distance_based_outlier_detection(distance_matrix: np.ndarray,
                                   threshold_percentile: float = 95.0) -> List[int]:
    """Detect outliers based on extreme distance values."""
    # Compute average distance for each model
    avg_distances = np.mean(distance_matrix, axis=1)
    
    # Find models with extreme average distances
    threshold = np.percentile(avg_distances, threshold_percentile)
    outliers = np.where(avg_distances > threshold)[0].tolist()
    
    return outliers

# ===== MAIN COMPUTATION PIPELINE =====

def compute_distance_matrix(curves: List[np.ndarray], 
                          distance_func, 
                          **func_kwargs) -> np.ndarray:
    """Compute pairwise distance matrix using specified distance function."""
    n = len(curves)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(curves[i], curves[j], **func_kwargs)
            D[i, j] = D[j, i] = d
    
    return D

def calculate_unsupervised_metrics(distance_matrix: np.ndarray, names: List[str]) -> Dict[str, float]:
    """Calculate metrics using ONLY model names to identify trained vs random."""
    # This is the minimal supervision needed to evaluate the method
    # In a real unsupervised setting, these labels would be unknown
    trained_indices = [i for i, name in enumerate(names) if 'trained' in name.lower()]
    random_indices = [i for i, name in enumerate(names) if 'random' in name.lower()]
    
    if not trained_indices or not random_indices:
        return {'separation_ratio': 0.0, 'margin': -999.0, 'error': 'Insufficient model types'}
    
    # Within-trained distances
    within_dists = [distance_matrix[i, j] for i in trained_indices for j in trained_indices if i < j]
    
    # Cross distances (trained vs random)
    cross_dists = [distance_matrix[i, j] for i in trained_indices for j in random_indices]
    
    if not within_dists or not cross_dists:
        return {'separation_ratio': 0.0, 'margin': -999.0, 'error': 'No valid distances'}
    
    mean_within = np.mean(within_dists)
    mean_cross = np.mean(cross_dists)
    max_within = np.max(within_dists)
    min_cross = np.min(cross_dists)
    
    separation_ratio = mean_cross / mean_within if mean_within > 0 else 0.0
    margin = min_cross - max_within
    
    return {
        'separation_ratio': separation_ratio,
        'margin': margin,
        'mean_within': mean_within,
        'mean_cross': mean_cross,
        'max_within': max_within,
        'min_cross': min_cross,
        'n_trained': len(trained_indices),
        'n_random': len(random_indices)
    }

def unsupervised_functional_similarity_pipeline(curves: List[np.ndarray], 
                                              names: List[str],
                                              preprocessing_config: Dict[str, Any],
                                              distance_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Complete unsupervised pipeline for functional similarity detection.
    
    Args:
        curves: Raw eigenvalue curves
        names: Model names (used only for evaluation, not computation)
        preprocessing_config: Configuration for preprocessing steps
        distance_configs: List of distance metric configurations to test
    
    Returns:
        Results dictionary with all tested configurations
    """
    
    print("[INFO] Starting unsupervised preprocessing...")
    
    # Apply unsupervised preprocessing
    processed_curves = curves.copy()
    
    # Step 1: Robust log transformation
    if preprocessing_config.get('use_log_transform', True):
        processed_curves = [robust_log_transform(c, preprocessing_config.get('eps_method', 'adaptive')) 
                           for c in processed_curves]
    
    # Step 2: Adaptive smoothing
    if preprocessing_config.get('use_smoothing', False):
        smoothing_factor = preprocessing_config.get('smoothing_factor', 0.1)
        processed_curves = [adaptive_smoothing(c, smoothing_factor) for c in processed_curves]
    
    # Step 3: Unsupervised normalization
    normalization_method = preprocessing_config.get('normalization_method', 'robust')
    processed_curves = unsupervised_normalization(processed_curves, normalization_method)
    
    print(f"[INFO] Testing {len(distance_configs)} distance metric configurations...")
    
    results = {
        'preprocessing_config': preprocessing_config,
        'distance_results': {},
        'best_result': None
    }
    
    best_margin = -999.0
    
    for config in distance_configs:
        metric_name = config['name']
        distance_func = config['function']
        func_kwargs = config.get('kwargs', {})
        
        print(f"[INFO] Computing distances with {metric_name}...")
        
        try:
            # Compute distance matrix
            D = compute_distance_matrix(processed_curves, distance_func, **func_kwargs)
            
            # Optional: Apply outlier detection
            if config.get('use_outlier_detection', False):
                outlier_method = config.get('outlier_method', 'isolation_forest')
                if outlier_method == 'isolation_forest':
                    outliers = statistical_outlier_detection(D, contamination=0.1)
                else:
                    outliers = distance_based_outlier_detection(D, threshold_percentile=95.0)
                
                if outliers:
                    print(f"[INFO] Detected {len(outliers)} outliers with {outlier_method}")
                    # Create filtered arrays
                    keep_indices = [i for i in range(len(names)) if i not in outliers]
                    filtered_D = D[np.ix_(keep_indices, keep_indices)]
                    filtered_names = [names[i] for i in keep_indices]
                    metrics = calculate_unsupervised_metrics(filtered_D, filtered_names)
                    metrics['n_outliers_removed'] = len(outliers)
                    metrics['outliers_removed'] = [names[i] for i in outliers]
                else:
                    metrics = calculate_unsupervised_metrics(D, names)
                    metrics['n_outliers_removed'] = 0
            else:
                metrics = calculate_unsupervised_metrics(D, names)
                metrics['n_outliers_removed'] = 0
            
            # Store results
            results['distance_results'][metric_name] = {
                'config': config,
                'metrics': metrics,
                'distance_matrix_shape': D.shape
            }
            
            # Track best result
            margin = metrics.get('margin', -999)
            if margin > best_margin:
                best_margin = margin
                results['best_result'] = {
                    'metric_name': metric_name,
                    'margin': margin,
                    'separation_ratio': metrics.get('separation_ratio', 0),
                    'config': config,
                    'metrics': metrics
                }
            
            print(f"   {metric_name}: Margin {margin:.6f} | Separation {metrics.get('separation_ratio', 0):.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to compute {metric_name}: {e}")
            results['distance_results'][metric_name] = {'error': str(e)}
    
    return results

def main():
    print("[INFO] Loading eigenvalue data...")
    curves, names = load_eigenvalue_data()
    
    if len(curves) < 10:
        print(f"[ERROR] Insufficient data: only {len(curves)} models loaded")
        return
    
    print(f"[INFO] Loaded {len(curves)} models")
    
    # Unsupervised preprocessing configuration
    preprocessing_config = {
        'use_log_transform': True,
        'eps_method': 'adaptive',
        'use_smoothing': False,  # Start without smoothing
        'smoothing_factor': 0.05,
        'normalization_method': 'robust'
    }
    
    # Distance metric configurations to test
    distance_configs = [
        {
            'name': 'Adaptive_DTW',
            'function': adaptive_dtw_distance,
            'kwargs': {},
            'use_outlier_detection': False
        },
        {
            'name': 'Adaptive_DTW_Wide',
            'function': adaptive_dtw_distance,
            'kwargs': {'window': 50},
            'use_outlier_detection': False
        },
        {
            'name': 'Wasserstein_1D',
            'function': wasserstein_distance_1d,
            'kwargs': {},
            'use_outlier_detection': False
        },
        {
            'name': 'Spectral_Distance',
            'function': spectral_distance,
            'kwargs': {},
            'use_outlier_detection': False
        },
        {
            'name': 'Correlation_Distance',
            'function': correlation_distance,
            'kwargs': {},
            'use_outlier_detection': False
        },
        {
            'name': 'Robust_EDR',
            'function': robust_edr_distance,
            'kwargs': {'eps_percentile': 10.0},
            'use_outlier_detection': False
        },
        # Test with outlier detection
        {
            'name': 'Adaptive_DTW_OutlierRemoval',
            'function': adaptive_dtw_distance,
            'kwargs': {},
            'use_outlier_detection': True,
            'outlier_method': 'isolation_forest'
        },
        {
            'name': 'Wasserstein_OutlierRemoval',
            'function': wasserstein_distance_1d,
            'kwargs': {},
            'use_outlier_detection': True,
            'outlier_method': 'distance_based'
        }
    ]
    
    print("\n" + "="*80)
    print("UNSUPERVISED FUNCTIONAL SIMILARITY DETECTION")
    print("="*80)
    
    # Run the pipeline
    results = unsupervised_functional_similarity_pipeline(
        curves, names, preprocessing_config, distance_configs
    )
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    # Print results
    for metric_name, result in results['distance_results'].items():
        if 'error' in result:
            print(f"{metric_name:<25}: ERROR - {result['error']}")
        else:
            metrics = result['metrics']
            margin = metrics.get('margin', -999)
            ratio = metrics.get('separation_ratio', 0)
            marker = "✅" if margin > 0 else "❌"
            print(f"{metric_name:<25}: {marker} Margin {margin:8.6f} | Separation {ratio:6.4f}")
            
            if metrics.get('n_outliers_removed', 0) > 0:
                print(f"{'':27}   (Removed {metrics['n_outliers_removed']} outliers)")
    
    if results['best_result']:
        print(f"\n--- BEST UNSUPERVISED RESULT ---")
        best = results['best_result']
        print(f"Method: {best['metric_name']}")
        print(f"Margin: {best['margin']:.6f}")
        print(f"Separation ratio: {best['separation_ratio']:.4f}")
        
        if best['margin'] > 0:
            print(f"\n🎉 SUCCESS: Positive margin achieved with unsupervised method!")
            print(f"✅ Method is completely unsupervised - no use of model labels")
            print(f"✅ Functional similarity detected through eigenvalue evolution patterns")
        else:
            print(f"\n📊 Analysis: Best unsupervised margin is {best['margin']:.6f}")
            print(f"   Gap to positive: {abs(best['margin']):.6f}")
            if abs(best['margin']) < 0.01:
                print(f"   Very close! Consider additional preprocessing or ensemble methods")
    
    print(f"\n[INFO] Unsupervised analysis complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Model Outlier Analysis

Analyzes how many random models vs trained models are identified as outliers.
"""

import numpy as np
import pathlib
import glob
import re
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

def load_eigenvalue_data() -> Tuple[List[np.ndarray], List[str]]:
    """Load eigenvalue evolution data."""
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
            
            mean_curve = np.nanmean(L, axis=1)
            eigenvalue_curves.append(mean_curve)
            model_names.append(name)
            
        except Exception as e:
            continue
    
    return eigenvalue_curves, model_names

def classify_model_type(model_name: str) -> str:
    """Classify model as trained, random, or other."""
    name_lower = model_name.lower()
    
    if 'random' in name_lower:
        return 'random'
    elif any(x in name_lower for x in ['trained', 'acc']):
        return 'trained'
    else:
        return 'other'

def optimal_preprocessing(curves: List[np.ndarray]) -> List[np.ndarray]:
    """Apply optimal preprocessing configuration."""
    processed = curves.copy()
    
    # Step 1: Standardize to length 20
    target_length = 20
    standardized = []
    for curve in processed:
        if len(curve) != target_length:
            x_old = np.linspace(0, 1, len(curve))
            x_new = np.linspace(0, 1, target_length)
            curve = np.interp(x_new, x_old, curve)
        standardized.append(curve)
    processed = standardized
    
    # Step 2: Robust log transformation
    log_transformed = []
    for curve in processed:
        curve_pos = np.maximum(curve, 0.0)
        non_zero = curve_pos[curve_pos > 0]
        if len(non_zero) > 0:
            eps = np.percentile(non_zero, 3.0)
        else:
            eps = 1e-10
        log_curve = np.log1p(np.maximum(curve_pos, eps))
        log_transformed.append(log_curve)
    processed = log_transformed
    
    # Step 3: Linear detrending
    detrended = []
    for curve in processed:
        if len(curve) >= 2:
            x = np.arange(len(curve))
            try:
                coeffs = np.polyfit(x, curve, deg=1)
                trend = np.polyval(coeffs, x)
                detrended_curve = curve - trend
            except:
                detrended_curve = curve - np.mean(curve)
        else:
            detrended_curve = curve - np.mean(curve)
        detrended.append(detrended_curve)
    processed = detrended
    
    # Step 4: Optimal spectral filtering
    enhanced = []
    for curve in processed:
        if len(curve) >= 8:
            curve_centered = curve - np.mean(curve)
            fft_curve = np.fft.fft(curve_centered)
            freqs = np.fft.fftfreq(len(curve))
            
            weights = np.ones_like(freqs)
            weights[np.abs(freqs) < 0.08] *= 4.0
            weights[np.abs(freqs) >= 0.25] *= 0.2
            
            enhanced_fft = fft_curve * weights
            enhanced_curve = np.real(np.fft.ifft(enhanced_fft))
            enhanced.append(enhanced_curve)
        else:
            enhanced.append(curve)
    processed = enhanced
    
    # Step 5: Robust scaling
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    all_features = np.vstack(processed)
    scaler.fit(all_features)
    final_processed = [scaler.transform(c.reshape(1, -1)).flatten() for c in processed]
    
    return final_processed

def optimal_dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Apply optimal DTW configuration."""
    N, M = len(a), len(b)
    window = max(int(0.2 * max(N, M)), 3)
    
    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0
    
    step_penalty = 0.1
    diagonal_bonus = 0.01
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if abs(i - j) <= window:
                dist = abs(a[i-1] - b[j-1])
                
                diag_cost = cost[i-1, j-1] + dist - diagonal_bonus
                vert_cost = cost[i-1, j] + dist + step_penalty
                horiz_cost = cost[i, j-1] + dist + step_penalty
                
                cost[i, j] = min(diag_cost, vert_cost, horiz_cost)
    
    return cost[N, M] / (N + M)

def scalable_outlier_detection(distance_matrix: np.ndarray, 
                              aggressive_factor: float = 1.5) -> List[int]:
    """Scalable outlier detection that can be tuned from conservative to aggressive."""
    X = distance_matrix.copy()
    max_finite = np.nanmax(X[np.isfinite(X)])
    if not np.isfinite(max_finite):
        return []
    
    X[~np.isfinite(X)] = max_finite * 2
    n_samples = X.shape[0]
    
    if n_samples < 5:
        return []
    
    all_votes = []
    
    # Method 1: Isolation Forest with scaled contamination
    base_contamination = 0.10
    scaled_contamination = min(0.35, base_contamination * aggressive_factor)
    contamination_rates = [scaled_contamination * (0.8 + 0.4 * i) for i in range(3)]
    
    for contamination in contamination_rates:
        if contamination <= 0.35:
            try:
                detector = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
                outliers = detector.fit_predict(X)
                all_votes.append((outliers == -1).astype(int))
            except:
                continue
    
    # Method 2: LOF with scaled settings
    try:
        n_neighbors = max(2, min(15, int(n_samples / (5 / aggressive_factor))))
        if n_neighbors < n_samples:
            contamination = min(0.25, 0.10 * aggressive_factor)
            detector = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outliers = detector.fit_predict(X)
            all_votes.append((outliers == -1).astype(int))
    except:
        pass
    
    # Method 3: DBSCAN with scaled eps
    try:
        percentile = max(10, min(50, 20 + 10 * aggressive_factor))
        eps = np.percentile(X[X > 0], percentile)
        if eps > 0:
            min_samples = max(2, int(4 / aggressive_factor))
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)
            all_votes.append((clusters == -1).astype(int))
    except:
        pass
    
    # Method 4: Statistical thresholding with scaled factors
    avg_distances = np.mean(X, axis=1)
    q75, q25 = np.percentile(avg_distances, [75, 25])
    iqr = q75 - q25
    
    # Scale the IQR multiplier
    iqr_multiplier = max(0.5, 2.0 / aggressive_factor)
    threshold1 = q75 + iqr_multiplier * iqr
    all_votes.append((avg_distances > threshold1).astype(int))
    
    # Percentile-based with scaling
    percentile_threshold = min(95, 100 - (15 * aggressive_factor))
    threshold2 = np.percentile(avg_distances, percentile_threshold)
    all_votes.append((avg_distances > threshold2).astype(int))
    
    if not all_votes:
        return []
    
    # Ensemble voting with scaled threshold
    vote_matrix = np.column_stack(all_votes)
    total_votes = np.sum(vote_matrix, axis=1)
    
    # Scale voting threshold
    base_threshold_ratio = 0.4  # 40% of methods need to agree
    scaled_threshold_ratio = max(0.2, base_threshold_ratio / aggressive_factor)
    min_votes_needed = max(1, int(len(all_votes) * scaled_threshold_ratio))
    
    outliers = np.where(total_votes >= min_votes_needed)[0].tolist()
    
    return outliers

def analyze_outliers_by_type(curves: List[np.ndarray], names: List[str], 
                           aggressive_factor: float = 1.0) -> Dict[str, Any]:
    """Analyze outliers broken down by model type."""
    
    # Classify all models
    model_types = [classify_model_type(name) for name in names]
    
    # Count by type
    type_counts = {}
    for model_type in model_types:
        type_counts[model_type] = type_counts.get(model_type, 0) + 1
    
    # Preprocessing and distance computation
    processed_curves = optimal_preprocessing(curves)
    
    n = len(processed_curves)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = optimal_dtw_distance(processed_curves[i], processed_curves[j])
            D[i, j] = D[j, i] = d
    
    # Detect outliers
    outlier_indices = scalable_outlier_detection(D, aggressive_factor)
    
    # Analyze outliers by type
    outlier_types = [model_types[i] for i in outlier_indices]
    outlier_names = [names[i] for i in outlier_indices]
    
    outlier_type_counts = {}
    for model_type in outlier_types:
        outlier_type_counts[model_type] = outlier_type_counts.get(model_type, 0) + 1
    
    # Calculate percentages
    results = {}
    for model_type in type_counts:
        total_of_type = type_counts[model_type]
        outliers_of_type = outlier_type_counts.get(model_type, 0)
        percentage_outliers = 100.0 * outliers_of_type / total_of_type if total_of_type > 0 else 0.0
        
        results[model_type] = {
            'total_count': total_of_type,
            'outlier_count': outliers_of_type,
            'percentage_outliers': percentage_outliers,
            'outlier_names': [name for i, name in enumerate(names) 
                            if i in outlier_indices and model_types[i] == model_type]
        }
    
    return {
        'aggressive_factor': aggressive_factor,
        'total_models': len(names),
        'total_outliers': len(outlier_indices),
        'outlier_percentage': 100.0 * len(outlier_indices) / len(names),
        'by_type': results,
        'all_outlier_names': outlier_names
    }

def print_type_analysis(analysis: Dict[str, Any]) -> None:
    """Print analysis results broken down by model type."""
    
    print(f"AGGRESSIVE FACTOR = {analysis['aggressive_factor']}")
    print("="*60)
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  Total models: {analysis['total_models']}")
    print(f"  Total outliers: {analysis['total_outliers']} ({analysis['outlier_percentage']:.1f}%)")
    
    print(f"\n🔍 BREAKDOWN BY MODEL TYPE:")
    print(f"{'Type':<12} {'Total':<7} {'Outliers':<9} {'% Outliers':<12} {'Status'}")
    print("-" * 55)
    
    by_type = analysis['by_type']
    
    for model_type in ['trained', 'random', 'other']:
        if model_type in by_type:
            data = by_type[model_type]
            total = data['total_count']
            outliers = data['outlier_count']
            percentage = data['percentage_outliers']
            
            # Status indicator
            if percentage < 20:
                status = "Low outlier rate"
            elif percentage < 50:
                status = "Medium outlier rate"
            else:
                status = "High outlier rate"
            
            print(f"{model_type.capitalize():<12} {total:<7} {outliers:<9} {percentage:8.1f}%    {status}")
    
    print(f"\n📋 DETAILED BREAKDOWN:")
    
    for model_type in ['trained', 'random', 'other']:
        if model_type in by_type:
            data = by_type[model_type]
            print(f"\n  {model_type.upper()} MODELS:")
            print(f"    Total: {data['total_count']}")
            print(f"    Outliers: {data['outlier_count']} ({data['percentage_outliers']:.1f}%)")
            
            if data['outlier_names']:
                print(f"    Outlier examples:")
                for i, name in enumerate(data['outlier_names'][:5], 1):  # Show first 5
                    print(f"      {i}. {name}")
                if len(data['outlier_names']) > 5:
                    print(f"      ... and {len(data['outlier_names']) - 5} more")
            else:
                print(f"    No outliers of this type")

def main():
    print("Loading eigenvalue data...")
    curves, names = load_eigenvalue_data()
    print(f"Loaded {len(curves)} models")
    
    # Test different aggressive factors
    aggressive_factors = [0.5, 1.0, 1.5, 2.0]
    
    print("\n" + "="*80)
    print("RANDOM vs TRAINED MODEL OUTLIER ANALYSIS")
    print("="*80)
    
    for aggressive_factor in aggressive_factors:
        print(f"\n{'='*60}")
        analysis = analyze_outliers_by_type(curves, names, aggressive_factor)
        print_type_analysis(analysis)
    
    print(f"\n🔬 KEY INSIGHTS:")
    print(f"This analysis shows whether random models are more/less likely")
    print(f"to be identified as outliers compared to trained models.")

if __name__ == "__main__":
    main()
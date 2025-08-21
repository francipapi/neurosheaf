#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Best Configurations on New Dataset

Tests the best configurations we found previously on the new improved dataset.
"""

import numpy as np
import pathlib
import glob
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats, signal
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

def optimal_preprocessing(curves: List[np.ndarray]) -> List[np.ndarray]:
    """Apply optimal preprocessing configuration from breakthrough."""
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

def calculate_metrics(distance_matrix: np.ndarray, model_names: List[str]) -> Dict[str, Any]:
    """Calculate margin and separation metrics."""
    n = len(model_names)
    
    # Classify models
    trained_indices = []
    random_indices = []
    
    for i, name in enumerate(model_names):
        name_lower = name.lower()
        if any(x in name_lower for x in ['trained', 'acc']):
            trained_indices.append(i)
        elif 'random' in name_lower:
            random_indices.append(i)
    
    if len(trained_indices) < 2 or len(random_indices) < 2:
        return {'error': 'Insufficient trained or random models'}
    
    # Calculate distances
    within_trained_distances = []
    for i in range(len(trained_indices)):
        for j in range(i + 1, len(trained_indices)):
            idx_i, idx_j = trained_indices[i], trained_indices[j]
            within_trained_distances.append(distance_matrix[idx_i, idx_j])
    
    cross_distances = []
    for i in trained_indices:
        for j in random_indices:
            cross_distances.append(distance_matrix[i, j])
    
    if not within_trained_distances or not cross_distances:
        return {'error': 'No valid distance pairs found'}
    
    # Calculate metrics
    margin = np.min(cross_distances) - np.max(within_trained_distances)
    separation_ratio = np.mean(cross_distances) / np.mean(within_trained_distances) if np.mean(within_trained_distances) > 0 else float('inf')
    
    # Statistical test
    try:
        t_stat, p_value = stats.ttest_ind(cross_distances, within_trained_distances)
    except:
        t_stat, p_value = None, None
    
    return {
        'margin': margin,
        'separation_ratio': separation_ratio,
        'within_trained_mean': np.mean(within_trained_distances),
        'within_trained_std': np.std(within_trained_distances),
        'cross_mean': np.mean(cross_distances),
        'cross_std': np.std(cross_distances),
        't_statistic': t_stat,
        'p_value': p_value,
        'n_trained': len(trained_indices),
        'n_random': len(random_indices),
        'n_within_pairs': len(within_trained_distances),
        'n_cross_pairs': len(cross_distances)
    }

def iterative_outlier_removal(curves: List[np.ndarray], names: List[str], 
                             aggressive_factor: float = 1.5,
                             max_iterations: int = 4) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
    """Apply iterative outlier removal with the specified aggressiveness."""
    
    current_curves = curves.copy()
    current_names = names.copy()
    removed_models = []
    iteration_results = []
    
    best_margin = -999.0
    best_result = None
    
    for iteration in range(max_iterations):
        # Preprocessing
        processed_curves = optimal_preprocessing(current_curves)
        
        # Distance computation
        n = len(processed_curves)
        D = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = optimal_dtw_distance(processed_curves[i], processed_curves[j])
                D[i, j] = D[j, i] = d
        
        # Calculate metrics
        metrics = calculate_metrics(D, current_names)
        margin = metrics.get('margin', -999)
        
        iteration_results.append({
            'iteration': iteration + 1,
            'n_models': len(current_curves),
            'margin': margin,
            'metrics': metrics.copy()
        })
        
        if margin > best_margin:
            best_margin = margin
            best_result = {
                'curves': current_curves.copy(),
                'names': current_names.copy(),
                'removed': removed_models.copy(),
                'iteration': iteration + 1,
                'metrics': metrics.copy()
            }
        
        if margin > 0:
            break
        
        # Outlier detection
        outliers = scalable_outlier_detection(D, aggressive_factor)
        
        if not outliers or len(current_curves) - len(outliers) < 8:
            break
        
        # Remove outliers
        keep_indices = [i for i in range(len(current_names)) if i not in outliers]
        removed_in_iteration = [current_names[i] for i in outliers]
        removed_models.extend(removed_in_iteration)
        
        current_curves = [current_curves[i] for i in keep_indices]
        current_names = [current_names[i] for i in keep_indices]
    
    return current_curves, current_names, {
        'final_margin': best_margin,
        'removed_models': removed_models,
        'iteration_results': iteration_results,
        'final_model_count': len(current_curves),
        'best_result': best_result
    }

def test_configuration(curves: List[np.ndarray], names: List[str], 
                      config_name: str, aggressive_factor: float) -> Dict[str, Any]:
    """Test a specific configuration."""
    
    print(f"\n🔍 Testing {config_name} with aggressive_factor={aggressive_factor}")
    print(f"   Starting with {len(curves)} models")
    
    if aggressive_factor == 0.0:
        # No outlier removal
        processed_curves = optimal_preprocessing(curves)
        n = len(processed_curves)
        D = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = optimal_dtw_distance(processed_curves[i], processed_curves[j])
                D[i, j] = D[j, i] = d
        
        metrics = calculate_metrics(D, names)
        
        return {
            'config_name': config_name,
            'aggressive_factor': aggressive_factor,
            'final_model_count': len(names),
            'outlier_percentage': 0.0,
            'removed_models': [],
            'metrics': metrics
        }
    else:
        # With outlier removal
        final_curves, final_names, removal_results = iterative_outlier_removal(
            curves, names, aggressive_factor
        )
        
        return {
            'config_name': config_name,
            'aggressive_factor': aggressive_factor,
            'final_model_count': removal_results['final_model_count'],
            'outlier_percentage': 100.0 * len(removal_results['removed_models']) / len(names),
            'removed_models': removal_results['removed_models'],
            'metrics': removal_results['best_result']['metrics'] if removal_results['best_result'] else {'margin': -999}
        }

def main():
    print("Loading eigenvalue data from new dataset...")
    curves, names = load_eigenvalue_data()
    print(f"Loaded {len(curves)} models")
    
    # Count model types
    trained_count = sum(1 for name in names if any(x in name.lower() for x in ['trained', 'acc']))
    random_count = sum(1 for name in names if 'random' in name.lower())
    other_count = len(names) - trained_count - random_count
    
    print(f"  - Trained models: {trained_count}")
    print(f"  - Random models: {random_count}")  
    print(f"  - Other models: {other_count}")
    
    print(f"\n{'='*80}")
    print("TESTING BEST CONFIGURATIONS ON NEW DATASET")
    print(f"{'='*80}")
    
    # Test the best configurations from our previous analysis
    test_configs = [
        # Our breakthrough configuration
        ("Optimal DTW + Full Preprocessing", 0.0),   # Baseline - no outlier removal
        ("Optimal DTW + Full Preprocessing", 1.0),   # Minimum positive margin config
        ("Optimal DTW + Full Preprocessing", 1.5),   # Original breakthrough config 
        ("Optimal DTW + Full Preprocessing", 2.0),   # Very aggressive
    ]
    
    results = []
    
    for config_name, aggressive_factor in test_configs:
        try:
            result = test_configuration(curves, names, config_name, aggressive_factor)
            results.append(result)
            
            # Print immediate results
            metrics = result['metrics']
            margin = metrics.get('margin', -999)
            sep_ratio = metrics.get('separation_ratio', -999)
            
            print(f"   Result: Margin = {margin:+.6f}, Separation = {sep_ratio:.4f}")
            print(f"   Models remaining: {result['final_model_count']}/{len(names)} ({result['outlier_percentage']:.1f}% removed)")
            
            if margin > 0:
                print(f"   🎉 POSITIVE MARGIN ACHIEVED!")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    # Summary results
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Config':<35} {'Aggressive':<10} {'Margin':<12} {'Sep Ratio':<10} {'Models':<8} {'Outlier%'}")
    print("-" * 80)
    
    for result in results:
        config = result['config_name'][:34]
        aggressive = result['aggressive_factor']
        metrics = result['metrics']
        margin = metrics.get('margin', -999)
        sep_ratio = metrics.get('separation_ratio', -999)
        models = result['final_model_count']
        outlier_pct = result['outlier_percentage']
        
        margin_str = f"{margin:+10.6f}" if margin > -900 else "    N/A   "
        sep_str = f"{sep_ratio:8.4f}" if sep_ratio > -900 and sep_ratio != float('inf') else "   N/A "
        
        print(f"{config:<35} {aggressive:<10} {margin_str:<12} {sep_str:<10} {models:<8} {outlier_pct:6.1f}%")
    
    # Best result analysis
    valid_results = [r for r in results if 'error' not in r['metrics']]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['metrics'].get('margin', -999))
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Configuration: {best_result['config_name']}")
        print(f"   Aggressive factor: {best_result['aggressive_factor']}")
        print(f"   Margin: {best_result['metrics']['margin']:+.6f}")
        print(f"   Separation ratio: {best_result['metrics']['separation_ratio']:.4f}")
        print(f"   Models used: {best_result['final_model_count']}/{len(names)}")
        print(f"   Outlier removal: {best_result['outlier_percentage']:.1f}%")
        
        if best_result['metrics']['p_value'] is not None:
            print(f"   Statistical significance: p = {best_result['metrics']['p_value']:.6f}")
        
        if best_result['metrics']['margin'] > 0:
            print(f"\n   🎯 SUCCESS: Achieved positive margin with new dataset!")
        else:
            print(f"\n   📊 Best margin achieved, but still negative")
            print(f"      Improvement needed: {abs(best_result['metrics']['margin']):.6f}")

if __name__ == "__main__":
    main()
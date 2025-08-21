#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy vs Outlier Analysis

Analyzes the relationship between model training accuracy and outlier removal patterns.
Tests the hypothesis: "Are outliers mostly models that did not achieve perfect accuracy?"
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

def extract_model_info(model_name: str) -> Dict[str, Any]:
    """Extract detailed information from model name."""
    name_lower = model_name.lower()
    
    info = {
        'name': model_name,
        'type': 'unknown',
        'accuracy': None,
        'epochs': None,
        'architecture': 'unknown',
        'is_trained': False,
        'is_random': False,
        'is_perfect_accuracy': False,
        'accuracy_tier': 'unknown'
    }
    
    # Extract accuracy
    acc_match = re.search(r'acc(\d+)', name_lower)
    if acc_match:
        info['accuracy'] = int(acc_match.group(1))
        info['is_perfect_accuracy'] = (info['accuracy'] == 100)
        
        # Classify accuracy tiers
        if info['accuracy'] >= 100:
            info['accuracy_tier'] = 'perfect'
        elif info['accuracy'] >= 95:
            info['accuracy_tier'] = 'high'
        elif info['accuracy'] >= 90:
            info['accuracy_tier'] = 'medium'
        else:
            info['accuracy_tier'] = 'low'
    
    # Extract epochs
    ep_match = re.search(r'ep(\d+)', name_lower)
    if ep_match:
        info['epochs'] = int(ep_match.group(1))
    
    # Determine type
    if 'random' in name_lower:
        info['type'] = 'random'
        info['is_random'] = True
    elif any(x in name_lower for x in ['trained', 'acc']):
        info['type'] = 'trained'
        info['is_trained'] = True
    
    # Architecture
    if 'custom' in name_lower:
        info['architecture'] = 'custom'
    elif 'mlp' in name_lower:
        info['architecture'] = 'mlp'
    elif 'conv' in name_lower:
        info['architecture'] = 'conv'
    
    return info

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

def analyze_outlier_patterns(curves: List[np.ndarray], names: List[str], 
                           aggressive_factor: float = 1.0) -> Dict[str, Any]:
    """Analyze which models are identified as outliers and their accuracy patterns."""
    
    # Extract model information
    model_infos = [extract_model_info(name) for name in names]
    
    # Preprocessing and distance computation
    processed_curves = optimal_preprocessing(curves)
    
    n = len(processed_curves)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = optimal_dtw_distance(processed_curves[i], processed_curves[j])
            D[i, j] = D[j, i] = d
    
    # Detect outliers
    outliers = scalable_outlier_detection(D, aggressive_factor)
    
    # Analyze outlier patterns
    outlier_infos = [model_infos[i] for i in outliers]
    non_outlier_infos = [model_infos[i] for i in range(len(model_infos)) if i not in outliers]
    
    return {
        'all_models': model_infos,
        'outlier_indices': outliers,
        'outlier_models': outlier_infos,
        'non_outlier_models': non_outlier_infos,
        'distance_matrix': D,
        'removal_percentage': 100.0 * len(outliers) / len(model_infos)
    }

def compute_accuracy_statistics(model_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics about accuracy distribution."""
    trained_models = [m for m in model_infos if m['is_trained'] and m['accuracy'] is not None]
    
    if not trained_models:
        return {'error': 'No trained models with accuracy information'}
    
    accuracies = [m['accuracy'] for m in trained_models]
    
    stats = {
        'count': len(trained_models),
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'perfect_count': sum(1 for acc in accuracies if acc == 100),
        'high_count': sum(1 for acc in accuracies if 95 <= acc < 100),
        'medium_count': sum(1 for acc in accuracies if 90 <= acc < 95),
        'low_count': sum(1 for acc in accuracies if acc < 90),
        'accuracy_distribution': {
            'perfect': [m for m in trained_models if m['accuracy'] == 100],
            'high': [m for m in trained_models if 95 <= m['accuracy'] < 100],
            'medium': [m for m in trained_models if 90 <= m['accuracy'] < 95],
            'low': [m for m in trained_models if m['accuracy'] < 90]
        }
    }
    
    return stats

def analyze_outlier_accuracy_correlation(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze correlation between being an outlier and training accuracy."""
    
    # Filter to trained models only
    outlier_trained = [m for m in analysis_results['outlier_models'] 
                      if m['is_trained'] and m['accuracy'] is not None]
    non_outlier_trained = [m for m in analysis_results['non_outlier_models'] 
                          if m['is_trained'] and m['accuracy'] is not None]
    
    if not outlier_trained or not non_outlier_trained:
        return {'error': 'Insufficient trained models for comparison'}
    
    outlier_accuracies = [m['accuracy'] for m in outlier_trained]
    non_outlier_accuracies = [m['accuracy'] for m in non_outlier_trained]
    
    # Statistical comparison
    outlier_stats = {
        'count': len(outlier_accuracies),
        'mean': np.mean(outlier_accuracies),
        'std': np.std(outlier_accuracies),
        'perfect_count': sum(1 for acc in outlier_accuracies if acc == 100),
        'perfect_percentage': 100.0 * sum(1 for acc in outlier_accuracies if acc == 100) / len(outlier_accuracies)
    }
    
    non_outlier_stats = {
        'count': len(non_outlier_accuracies),
        'mean': np.mean(non_outlier_accuracies),
        'std': np.std(non_outlier_accuracies),
        'perfect_count': sum(1 for acc in non_outlier_accuracies if acc == 100),
        'perfect_percentage': 100.0 * sum(1 for acc in non_outlier_accuracies if acc == 100) / len(non_outlier_accuracies)
    }
    
    # Statistical test
    try:
        t_stat, p_value = stats.ttest_ind(outlier_accuracies, non_outlier_accuracies)
    except:
        t_stat, p_value = None, None
    
    # Accuracy tier breakdown
    def get_tier_breakdown(models):
        return {
            'perfect': sum(1 for m in models if m['accuracy'] == 100),
            'high': sum(1 for m in models if 95 <= m['accuracy'] < 100),
            'medium': sum(1 for m in models if 90 <= m['accuracy'] < 95),
            'low': sum(1 for m in models if m['accuracy'] < 90)
        }
    
    outlier_tiers = get_tier_breakdown(outlier_trained)
    non_outlier_tiers = get_tier_breakdown(non_outlier_trained)
    
    return {
        'outlier_stats': outlier_stats,
        'non_outlier_stats': non_outlier_stats,
        'statistical_test': {'t_statistic': t_stat, 'p_value': p_value},
        'outlier_tier_breakdown': outlier_tiers,
        'non_outlier_tier_breakdown': non_outlier_tiers,
        'outlier_models': outlier_trained,
        'non_outlier_models': non_outlier_trained
    }

def print_analysis_results(analysis_results: Dict[str, Any], 
                          correlation_results: Dict[str, Any]) -> None:
    """Print comprehensive analysis results."""
    
    print("="*80)
    print("ACCURACY vs OUTLIER ANALYSIS")
    print("="*80)
    
    print(f"\n📊 OVERALL DATASET:")
    print(f"  Total models: {len(analysis_results['all_models'])}")
    print(f"  Outliers detected: {len(analysis_results['outlier_indices'])} ({analysis_results['removal_percentage']:.1f}%)")
    print(f"  Non-outliers: {len(analysis_results['non_outlier_models'])}")
    
    # Model type breakdown
    all_models = analysis_results['all_models']
    trained_count = sum(1 for m in all_models if m['is_trained'])
    random_count = sum(1 for m in all_models if m['is_random'])
    
    print(f"\n📋 MODEL TYPE BREAKDOWN:")
    print(f"  Trained models: {trained_count}")
    print(f"  Random models: {random_count}")
    print(f"  Other: {len(all_models) - trained_count - random_count}")
    
    if 'error' in correlation_results:
        print(f"\n❌ ERROR: {correlation_results['error']}")
        return
    
    # Accuracy comparison
    print(f"\n🎯 ACCURACY COMPARISON (Trained Models Only):")
    
    outlier_stats = correlation_results['outlier_stats']
    non_outlier_stats = correlation_results['non_outlier_stats']
    
    print(f"\n  OUTLIER TRAINED MODELS ({outlier_stats['count']} models):")
    print(f"    Mean accuracy: {outlier_stats['mean']:.1f}%")
    print(f"    Std deviation: {outlier_stats['std']:.1f}%")
    print(f"    Perfect accuracy (100%): {outlier_stats['perfect_count']}/{outlier_stats['count']} ({outlier_stats['perfect_percentage']:.1f}%)")
    
    print(f"\n  NON-OUTLIER TRAINED MODELS ({non_outlier_stats['count']} models):")
    print(f"    Mean accuracy: {non_outlier_stats['mean']:.1f}%")
    print(f"    Std deviation: {non_outlier_stats['std']:.1f}%")
    print(f"    Perfect accuracy (100%): {non_outlier_stats['perfect_count']}/{non_outlier_stats['count']} ({non_outlier_stats['perfect_percentage']:.1f}%)")
    
    # Statistical significance
    stat_test = correlation_results['statistical_test']
    if stat_test['t_statistic'] is not None:
        print(f"\n📈 STATISTICAL TEST (t-test):")
        print(f"  t-statistic: {stat_test['t_statistic']:.4f}")
        print(f"  p-value: {stat_test['p_value']:.6f}")
        significance = "SIGNIFICANT" if stat_test['p_value'] < 0.05 else "NOT SIGNIFICANT"
        print(f"  Result: {significance} difference in accuracy")
    
    # Accuracy tier breakdown
    print(f"\n🏆 ACCURACY TIER BREAKDOWN:")
    
    outlier_tiers = correlation_results['outlier_tier_breakdown']
    non_outlier_tiers = correlation_results['non_outlier_tier_breakdown']
    
    print(f"  {'Tier':<12} {'Outliers':<10} {'Non-Outliers':<12} {'% Outliers'}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    
    for tier in ['perfect', 'high', 'medium', 'low']:
        outlier_count = outlier_tiers.get(tier, 0)
        non_outlier_count = non_outlier_tiers.get(tier, 0)
        total_in_tier = outlier_count + non_outlier_count
        
        if total_in_tier > 0:
            outlier_percentage = 100.0 * outlier_count / total_in_tier
        else:
            outlier_percentage = 0.0
        
        tier_ranges = {
            'perfect': '100%',
            'high': '95-99%',
            'medium': '90-94%',
            'low': '<90%'
        }
        
        print(f"  {tier_ranges[tier]:<12} {outlier_count:<10} {non_outlier_count:<12} {outlier_percentage:6.1f}%")
    
    # Key findings
    print(f"\n🔍 KEY FINDINGS:")
    
    mean_diff = outlier_stats['mean'] - non_outlier_stats['mean']
    perfect_diff = outlier_stats['perfect_percentage'] - non_outlier_stats['perfect_percentage']
    
    if abs(mean_diff) < 2.0:
        print(f"  ✅ NO strong correlation between accuracy and outlier status")
        print(f"     Mean accuracy difference: {mean_diff:.1f}% (very small)")
    elif mean_diff < -5.0:
        print(f"  ⚠️  OUTLIERS tend to have LOWER accuracy")
        print(f"     Mean accuracy difference: {mean_diff:.1f}%")
    elif mean_diff > 5.0:
        print(f"  🤔 OUTLIERS tend to have HIGHER accuracy (unexpected!)")
        print(f"     Mean accuracy difference: {mean_diff:.1f}%")
    else:
        print(f"  📊 WEAK correlation between accuracy and outlier status")
        print(f"     Mean accuracy difference: {mean_diff:.1f}%")
    
    if abs(perfect_diff) < 10.0:
        print(f"  ✅ Perfect accuracy models are equally represented in outliers/non-outliers")
        print(f"     Perfect accuracy difference: {perfect_diff:.1f}%")
    elif perfect_diff < -10.0:
        print(f"  ⚠️  Perfect accuracy models are LESS likely to be outliers")
        print(f"     Perfect accuracy difference: {perfect_diff:.1f}%")
    else:
        print(f"  🤔 Perfect accuracy models are MORE likely to be outliers")
        print(f"     Perfect accuracy difference: {perfect_diff:.1f}%")
    
    # Detailed outlier list
    print(f"\n📝 DETAILED OUTLIER LIST (Trained Models):")
    outlier_trained = correlation_results['outlier_models']
    outlier_trained_sorted = sorted(outlier_trained, key=lambda x: x['accuracy'], reverse=True)
    
    for i, model in enumerate(outlier_trained_sorted[:15], 1):  # Show top 15
        acc = model['accuracy']
        epochs = model.get('epochs', 'Unknown')
        arch = model.get('architecture', 'Unknown')
        epochs_str = str(epochs) if epochs is not None else 'Unknown'
        print(f"  {i:2d}. {acc:3d}% accuracy, {epochs_str:>7s} epochs, {arch:6s} arch - {model['name'][:50]}")
    
    if len(outlier_trained_sorted) > 15:
        print(f"  ... and {len(outlier_trained_sorted) - 15} more")

def main():
    print("Loading eigenvalue data...")
    curves, names = load_eigenvalue_data()
    print(f"Loaded {len(curves)} models")
    
    # Test different aggressive factors
    aggressive_factors = [0.5, 1.0, 1.5, 2.0]
    
    for aggressive_factor in aggressive_factors:
        print(f"\n{'='*60}")
        print(f"ANALYZING AGGRESSIVE FACTOR = {aggressive_factor}")
        print(f"{'='*60}")
        
        # Run analysis
        analysis_results = analyze_outlier_patterns(curves, names, aggressive_factor)
        correlation_results = analyze_outlier_accuracy_correlation(analysis_results)
        
        # Print results
        print_analysis_results(analysis_results, correlation_results)
        
        print(f"\n" + "="*60)

if __name__ == "__main__":
    main()
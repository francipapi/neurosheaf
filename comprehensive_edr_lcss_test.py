#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive EDR & LCSS Testing Suite

Tests EDR (Edit Distance on Real sequences) and LCSS (Longest Common Subsequence) 
distance measures with all preprocessing techniques and outlier removal strategies 
on the new improved model dataset.
"""

import numpy as np
import pathlib
import glob
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats, signal
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import itertools
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

def edr_distance(s1: np.ndarray, s2: np.ndarray, epsilon: float = 0.2, gap_penalty: float = 1.0) -> float:
    """
    Compute Edit Distance on Real sequences (EDR).
    
    Args:
        s1, s2: Input sequences
        epsilon: Matching threshold
        gap_penalty: Cost of insertions/deletions
    """
    n, m = len(s1), len(s2)
    
    # DP matrix
    dp = np.zeros((n + 1, m + 1))
    
    # Initialize with gap penalties
    for i in range(1, n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = j * gap_penalty
    
    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(s1[i-1] - s2[j-1]) <= epsilon:
                # Match
                dp[i][j] = dp[i-1][j-1]
            else:
                # Mismatch, insertion, deletion
                dp[i][j] = min(
                    dp[i-1][j-1] + 1,  # substitution
                    dp[i-1][j] + gap_penalty,  # deletion
                    dp[i][j-1] + gap_penalty   # insertion
                )
    
    return dp[n][m]

def lcss_distance(s1: np.ndarray, s2: np.ndarray, epsilon: float = 0.2, window: Optional[int] = None) -> float:
    """
    Compute distance based on Longest Common Subsequence (LCSS).
    
    Args:
        s1, s2: Input sequences
        epsilon: Matching threshold
        window: Window constraint for alignment
    """
    n, m = len(s1), len(s2)
    
    # DP matrix for LCSS
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Check window constraint
            if window is not None and abs(i - j) > window:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            elif abs(s1[i-1] - s2[j-1]) <= epsilon:
                # Match found
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                # No match
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Convert LCSS to distance (normalized by average length)
    lcss_length = dp[n][m]
    avg_length = (n + m) / 2.0
    return 1.0 - (lcss_length / avg_length)

class PreprocessingConfig:
    """Configuration for preprocessing steps."""
    def __init__(self, 
                 target_length: Optional[int] = None,
                 eps_percentile: Optional[float] = None,
                 detrend_degree: Optional[int] = None,
                 spectral_filtering: bool = False,
                 low_freq_mult: float = 1.0,
                 high_freq_mult: float = 1.0,
                 low_freq_cutoff: float = 0.08,
                 high_freq_cutoff: float = 0.25,
                 scaler_type: str = 'robust',
                 quantile_range: Tuple[float, float] = (25.0, 75.0)):
        
        self.target_length = target_length
        self.eps_percentile = eps_percentile
        self.detrend_degree = detrend_degree
        self.spectral_filtering = spectral_filtering
        self.low_freq_mult = low_freq_mult
        self.high_freq_mult = high_freq_mult
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_cutoff = high_freq_cutoff
        self.scaler_type = scaler_type
        self.quantile_range = quantile_range
    
    def __repr__(self):
        parts = []
        if self.target_length:
            parts.append(f"len={self.target_length}")
        if self.eps_percentile:
            parts.append(f"eps={self.eps_percentile}")
        if self.detrend_degree is not None:
            parts.append(f"detrend={self.detrend_degree}")
        if self.spectral_filtering:
            parts.append("spectral")
        parts.append(f"scale={self.scaler_type}")
        return f"Preproc({','.join(parts)})"

def apply_preprocessing(curves: List[np.ndarray], config: PreprocessingConfig) -> List[np.ndarray]:
    """Apply preprocessing configuration to eigenvalue curves."""
    processed = [curve.copy() for curve in curves]
    
    # Step 1: Length standardization
    if config.target_length is not None:
        standardized = []
        for curve in processed:
            if len(curve) != config.target_length:
                x_old = np.linspace(0, 1, len(curve))
                x_new = np.linspace(0, 1, config.target_length)
                curve = np.interp(x_new, x_old, curve)
            standardized.append(curve)
        processed = standardized
    
    # Step 2: Log transformation
    if config.eps_percentile is not None:
        log_transformed = []
        for curve in processed:
            curve_pos = np.maximum(curve, 0.0)
            non_zero = curve_pos[curve_pos > 0]
            if len(non_zero) > 0:
                eps = np.percentile(non_zero, config.eps_percentile)
            else:
                eps = 1e-10
            log_curve = np.log1p(np.maximum(curve_pos, eps))
            log_transformed.append(log_curve)
        processed = log_transformed
    
    # Step 3: Detrending
    if config.detrend_degree is not None:
        detrended = []
        for curve in processed:
            if len(curve) >= config.detrend_degree + 1:
                x = np.arange(len(curve))
                try:
                    coeffs = np.polyfit(x, curve, deg=config.detrend_degree)
                    trend = np.polyval(coeffs, x)
                    detrended_curve = curve - trend
                except:
                    detrended_curve = curve - np.mean(curve)
            else:
                detrended_curve = curve - np.mean(curve)
            detrended.append(detrended_curve)
        processed = detrended
    
    # Step 4: Spectral filtering
    if config.spectral_filtering:
        enhanced = []
        for curve in processed:
            if len(curve) >= 8:
                curve_centered = curve - np.mean(curve)
                fft_curve = np.fft.fft(curve_centered)
                freqs = np.fft.fftfreq(len(curve))
                
                weights = np.ones_like(freqs)
                weights[np.abs(freqs) < config.low_freq_cutoff] *= config.low_freq_mult
                weights[np.abs(freqs) >= config.high_freq_cutoff] *= config.high_freq_mult
                
                enhanced_fft = fft_curve * weights
                enhanced_curve = np.real(np.fft.ifft(enhanced_fft))
                enhanced.append(enhanced_curve)
            else:
                enhanced.append(curve)
        processed = enhanced
    
    # Step 5: Scaling
    if processed and len(processed[0]) > 0:
        all_features = np.vstack(processed)
        
        if config.scaler_type == 'robust':
            scaler = RobustScaler(quantile_range=config.quantile_range)
        elif config.scaler_type == 'standard':
            scaler = StandardScaler()
        elif config.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            return processed
        
        try:
            scaler.fit(all_features)
            final_processed = [scaler.transform(c.reshape(1, -1)).flatten() for c in processed]
            return final_processed
        except:
            return processed
    
    return processed

def scalable_outlier_detection(distance_matrix: np.ndarray, 
                              aggressive_factor: float = 1.5) -> List[int]:
    """Scalable outlier detection with ensemble methods."""
    X = distance_matrix.copy()
    max_finite = np.nanmax(X[np.isfinite(X)])
    if not np.isfinite(max_finite):
        return []
    
    X[~np.isfinite(X)] = max_finite * 2
    n_samples = X.shape[0]
    
    if n_samples < 5:
        return []
    
    all_votes = []
    
    # Method 1: Isolation Forest
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
    
    # Method 2: LOF
    try:
        n_neighbors = max(2, min(15, int(n_samples / (5 / aggressive_factor))))
        if n_neighbors < n_samples:
            contamination = min(0.25, 0.10 * aggressive_factor)
            detector = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outliers = detector.fit_predict(X)
            all_votes.append((outliers == -1).astype(int))
    except:
        pass
    
    # Method 3: DBSCAN
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
    
    # Method 4: Statistical thresholding
    avg_distances = np.mean(X, axis=1)
    q75, q25 = np.percentile(avg_distances, [75, 25])
    iqr = q75 - q25
    
    iqr_multiplier = max(0.5, 2.0 / aggressive_factor)
    threshold1 = q75 + iqr_multiplier * iqr
    all_votes.append((avg_distances > threshold1).astype(int))
    
    # Method 5: Percentile-based
    percentile_threshold = min(95, 100 - (15 * aggressive_factor))
    threshold2 = np.percentile(avg_distances, percentile_threshold)
    all_votes.append((avg_distances > threshold2).astype(int))
    
    if not all_votes:
        return []
    
    # Ensemble voting
    vote_matrix = np.column_stack(all_votes)
    total_votes = np.sum(vote_matrix, axis=1)
    
    base_threshold_ratio = 0.4
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
                             distance_func, distance_params: Dict[str, Any],
                             preprocessing_config: PreprocessingConfig,
                             aggressive_factor: float = 1.5,
                             max_iterations: int = 4) -> Dict[str, Any]:
    """Apply iterative outlier removal with specified distance function."""
    
    current_curves = curves.copy()
    current_names = names.copy()
    removed_models = []
    best_margin = -999.0
    best_results = None
    
    for iteration in range(max_iterations):
        # Preprocessing
        processed_curves = apply_preprocessing(current_curves, preprocessing_config)
        
        # Distance computation
        n = len(processed_curves)
        D = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = distance_func(processed_curves[i], processed_curves[j], **distance_params)
                D[i, j] = D[j, i] = d
        
        # Calculate metrics
        metrics = calculate_metrics(D, current_names)
        
        if 'error' not in metrics:
            margin = metrics.get('margin', -999)
            
            if margin > best_margin:
                best_margin = margin
                best_results = {
                    'iteration': iteration + 1,
                    'n_models': len(current_curves),
                    'removed_models': removed_models.copy(),
                    'metrics': metrics.copy(),
                    'final_names': current_names.copy()
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
    
    return best_results or {
        'iteration': 0,
        'n_models': len(curves),
        'removed_models': [],
        'metrics': {'margin': -999, 'error': 'No valid results'},
        'final_names': names.copy()
    }

def comprehensive_test():
    """Run comprehensive test of EDR and LCSS with all configurations."""
    
    print("Loading eigenvalue data...")
    curves, names = load_eigenvalue_data()
    print(f"Loaded {len(curves)} models")
    
    # Define test configurations
    
    # Distance function configurations
    distance_configs = []
    
    # EDR configurations
    for epsilon in [0.1, 0.2, 0.3, 0.5]:
        for gap_penalty in [0.5, 1.0, 2.0]:
            distance_configs.append({
                'name': 'EDR',
                'function': edr_distance,
                'params': {'epsilon': epsilon, 'gap_penalty': gap_penalty},
                'description': f'EDR(ε={epsilon},gap={gap_penalty})'
            })
    
    # LCSS configurations
    for epsilon in [0.1, 0.2, 0.3, 0.5]:
        for window in [None, 5, 10, 20]:
            distance_configs.append({
                'name': 'LCSS',
                'function': lcss_distance,
                'params': {'epsilon': epsilon, 'window': window},
                'description': f'LCSS(ε={epsilon},win={window})'
            })
    
    # Preprocessing configurations
    preprocessing_configs = [
        # Optimal configuration
        PreprocessingConfig(
            target_length=20,
            eps_percentile=3.0,
            detrend_degree=1,
            spectral_filtering=True,
            low_freq_mult=4.0,
            high_freq_mult=0.2,
            low_freq_cutoff=0.08,
            high_freq_cutoff=0.25,
            scaler_type='robust',
            quantile_range=(10.0, 90.0)
        ),
        # Alternative length
        PreprocessingConfig(
            target_length=30,
            eps_percentile=5.0,
            detrend_degree=1,
            spectral_filtering=True,
            low_freq_mult=2.0,
            high_freq_mult=0.5,
            scaler_type='robust'
        ),
        # Minimal preprocessing
        PreprocessingConfig(
            target_length=20,
            detrend_degree=1,
            scaler_type='standard'
        ),
        # No preprocessing
        PreprocessingConfig(
            scaler_type='robust'
        )
    ]
    
    # Outlier removal levels
    aggressive_factors = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Run comprehensive test
    results = []
    total_tests = len(distance_configs) * len(preprocessing_configs) * len(aggressive_factors)
    test_count = 0
    
    print(f"\nRunning {total_tests} test configurations...")
    
    for dist_config in distance_configs:
        for prep_config in preprocessing_configs:
            for aggressive_factor in aggressive_factors:
                test_count += 1
                print(f"Test {test_count}/{total_tests}: {dist_config['description']} + {prep_config} + aggressive={aggressive_factor}")
                
                try:
                    if aggressive_factor == 0.0:
                        # No outlier removal
                        processed_curves = apply_preprocessing(curves, prep_config)
                        n = len(processed_curves)
                        D = np.zeros((n, n))
                        
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = dist_config['function'](processed_curves[i], processed_curves[j], **dist_config['params'])
                                D[i, j] = D[j, i] = d
                        
                        metrics = calculate_metrics(D, names)
                        
                        result = {
                            'distance_name': dist_config['name'],
                            'distance_description': dist_config['description'],
                            'distance_params': dist_config['params'],
                            'preprocessing': str(prep_config),
                            'aggressive_factor': aggressive_factor,
                            'n_models': len(names),
                            'outlier_percentage': 0.0,
                            'removed_models': [],
                            'metrics': metrics if 'error' not in metrics else {'margin': -999, 'error': metrics.get('error', 'Unknown error')}
                        }
                    else:
                        # With outlier removal
                        result = iterative_outlier_removal(
                            curves, names,
                            dist_config['function'],
                            dist_config['params'],
                            prep_config,
                            aggressive_factor
                        )
                        
                        result.update({
                            'distance_name': dist_config['name'],
                            'distance_description': dist_config['description'],
                            'distance_params': dist_config['params'],
                            'preprocessing': str(prep_config),
                            'aggressive_factor': aggressive_factor,
                            'outlier_percentage': 100.0 * len(result['removed_models']) / len(names)
                        })
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue
    
    # Sort results by margin
    valid_results = [r for r in results if 'error' not in r.get('metrics', {})]
    valid_results.sort(key=lambda x: x['metrics'].get('margin', -999), reverse=True)
    
    return valid_results

def print_results(results: List[Dict[str, Any]], top_n: int = 20):
    """Print comprehensive results."""
    
    print(f"\n{'='*120}")
    print("COMPREHENSIVE EDR & LCSS TEST RESULTS")
    print(f"{'='*120}")
    
    print(f"\nTotal configurations tested: {len(results)}")
    positive_margin_count = sum(1 for r in results if r['metrics'].get('margin', -999) > 0)
    print(f"Configurations with positive margin: {positive_margin_count}")
    
    if positive_margin_count > 0:
        print(f"\n🎉 SUCCESS! Found {positive_margin_count} configuration(s) with positive margin!")
    else:
        print(f"\n📊 Best configurations (top {top_n}):")
    
    print(f"\n{'Rank':<4} {'Distance':<20} {'Margin':<8} {'SepRatio':<8} {'Models':<7} {'Outlier%':<8} {'AggrFact':<8} {'Preprocessing'}")
    print("-" * 120)
    
    for i, result in enumerate(results[:top_n], 1):
        metrics = result['metrics']
        margin = metrics.get('margin', -999)
        sep_ratio = metrics.get('separation_ratio', -999)
        n_models = result['n_models']
        outlier_pct = result['outlier_percentage']
        aggr_factor = result['aggressive_factor']
        dist_desc = result['distance_description'][:20]
        prep_short = str(result['preprocessing'])[:50] if len(str(result['preprocessing'])) <= 50 else str(result['preprocessing'])[:47] + "..."
        
        margin_str = f"{margin:+7.4f}" if margin > -900 else "   N/A  "
        sep_str = f"{sep_ratio:7.3f}" if sep_ratio > -900 and sep_ratio != float('inf') else "   N/A "
        
        print(f"{i:<4} {dist_desc:<20} {margin_str:<8} {sep_str:<8} {n_models:<7} {outlier_pct:6.1f}% {aggr_factor:<8} {prep_short}")
    
    # Detailed analysis of best results
    if results:
        print(f"\n🏆 BEST CONFIGURATION DETAILS:")
        best = results[0]
        print(f"  Distance: {best['distance_description']}")
        print(f"  Parameters: {best['distance_params']}")
        print(f"  Preprocessing: {best['preprocessing']}")
        print(f"  Aggressive factor: {best['aggressive_factor']}")
        print(f"  Models remaining: {best['n_models']} / {best['n_models'] + len(best.get('removed_models', []))}")
        print(f"  Outlier removal: {best['outlier_percentage']:.1f}%")
        
        metrics = best['metrics']
        print(f"\n  📊 Performance Metrics:")
        print(f"    Margin: {metrics.get('margin', 'N/A'):+.6f}")
        print(f"    Separation ratio: {metrics.get('separation_ratio', 'N/A'):.4f}")
        print(f"    Within-trained mean: {metrics.get('within_trained_mean', 'N/A'):.6f}")
        print(f"    Cross-distance mean: {metrics.get('cross_mean', 'N/A'):.6f}")
        if metrics.get('p_value') is not None:
            print(f"    Statistical significance: p = {metrics['p_value']:.6f}")
    
    # Summary by distance type
    print(f"\n📈 SUMMARY BY DISTANCE TYPE:")
    distance_summary = {}
    for result in results:
        dist_name = result['distance_name']
        margin = result['metrics'].get('margin', -999)
        
        if dist_name not in distance_summary:
            distance_summary[dist_name] = {'count': 0, 'best_margin': -999, 'positive_count': 0}
        
        distance_summary[dist_name]['count'] += 1
        distance_summary[dist_name]['best_margin'] = max(distance_summary[dist_name]['best_margin'], margin)
        if margin > 0:
            distance_summary[dist_name]['positive_count'] += 1
    
    for dist_name, summary in distance_summary.items():
        print(f"  {dist_name}: {summary['count']} configs, best margin = {summary['best_margin']:+.4f}, {summary['positive_count']} positive")

if __name__ == "__main__":
    results = comprehensive_test()
    print_results(results, top_n=25)
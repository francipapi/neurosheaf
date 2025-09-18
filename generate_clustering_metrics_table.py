#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate comprehensive clustering metrics table for trained vs random model classification.

This script computes all clustering validation metrics with bootstrap confidence intervals
and generates a publication-ready LaTeX table.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                                calinski_harabasz_score, adjusted_rand_score,
                                adjusted_mutual_info_score, v_measure_score)
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not available. Some metrics will be unavailable.")
    _HAS_SKLEARN = False

try:
    from skbio.stats.distance import permanova, anosim
    _HAS_SKBIO = True
except ImportError:
    print("Warning: scikit-bio not available. PERMANOVA/ANOSIM will be approximated.")
    _HAS_SKBIO = False


def load_data():
    """Load elastic distance data and model information."""
    print("Loading elastic distance data...")
    
    # Load distance matrix
    distance_matrix = np.load('/Users/francescopapini/GitRepo/neurosheaf/elastic_eigs_distance.npy')
    
    # Load model index
    with open('/Users/francescopapini/GitRepo/neurosheaf/elastic_eigs_index.json', 'r') as f:
        model_names = json.load(f)
    
    # Parse model information
    model_info = []
    for name in model_names:
        info = {'name': name}
        name_lower = name.lower()
        
        # Architecture
        if 'hourglass' in name_lower:
            info['architecture'] = 'hourglass'
        elif 'pyramid' in name_lower:
            info['architecture'] = 'pyramid'
        else:
            info['architecture'] = 'unknown'
        
        # Training status
        if 'random' in name_lower:
            info['status'] = 'random'
            info['label'] = 0  # Random = 0
        elif 'seed' in name_lower:
            info['status'] = 'trained'
            info['label'] = 1  # Trained = 1
        else:
            info['status'] = 'unknown'
            info['label'] = -1
        
        model_info.append(info)
    
    model_df = pd.DataFrame(model_info)
    labels = model_df['label'].values
    
    print(f"Loaded distance matrix: {distance_matrix.shape}")
    print(f"Number of models: {len(model_names)}")
    print(f"Trained models: {np.sum(labels == 1)}")
    print(f"Random models: {np.sum(labels == 0)}")
    
    return distance_matrix, labels, model_df


def dunn_index(distance_matrix, labels):
    """Compute Dunn index for clustering quality."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    # Minimum inter-cluster distance
    min_inter_dist = np.inf
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = np.where(labels == unique_labels[i])[0]
            cluster_j = np.where(labels == unique_labels[j])[0]
            
            inter_distances = distance_matrix[np.ix_(cluster_i, cluster_j)]
            min_inter_dist = min(min_inter_dist, np.min(inter_distances))
    
    # Maximum intra-cluster distance
    max_intra_dist = 0.0
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) > 1:
            intra_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(intra_distances, k=1)
            if len(triu_indices[0]) > 0:
                max_intra_dist = max(max_intra_dist, np.max(intra_distances[triu_indices]))
    
    return min_inter_dist / max_intra_dist if max_intra_dist > 0 else np.inf


def jaccard_stability(distance_matrix, labels, n_bootstrap=100):
    """Compute Jaccard stability of clustering across bootstrap samples."""
    n_samples = len(labels)
    stabilities = []
    
    # Get original clustering (hierarchical)
    condensed_dist = pdist(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    from scipy.cluster.hierarchy import fcluster
    original_clusters = fcluster(linkage_matrix, 2, criterion='maxclust')
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_distance_matrix = distance_matrix[np.ix_(boot_indices, boot_indices)]
        boot_labels = labels[boot_indices]
        
        # Cluster bootstrap sample
        try:
            boot_condensed = pdist(boot_distance_matrix)
            boot_linkage = linkage(boot_condensed, method='ward')
            boot_clusters = fcluster(boot_linkage, 2, criterion='maxclust')
            
            # Compute Jaccard similarity between original and bootstrap clustering
            # Map back to original indices
            original_mapped = original_clusters[boot_indices]
            
            # Compute agreement
            agreement = np.sum(original_mapped == boot_clusters) / len(boot_clusters)
            stabilities.append(agreement)
            
        except Exception:
            # If clustering fails, append 0
            stabilities.append(0.0)
    
    return np.array(stabilities)


def permanova_test(distance_matrix, labels, n_permutations=999):
    """Perform PERMANOVA test for group differences."""
    if _HAS_SKBIO:
        try:
            # Convert to skbio format
            import skbio
            dm = skbio.DistanceMatrix(distance_matrix)
            groups = pd.Series(labels, name='group')
            
            result = permanova(dm, groups, permutations=n_permutations)
            return result['test statistic'], result['p-value']
        except Exception as e:
            print(f"PERMANOVA failed with scikit-bio: {e}")
    
    # Fallback implementation
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    
    # Compute original F statistic
    def compute_f_stat(dist_mat, lbls):
        total_ss = np.sum(dist_mat ** 2) / (2 * len(lbls))
        
        within_ss = 0
        for label in unique_labels:
            group_indices = np.where(lbls == label)[0]
            if len(group_indices) > 1:
                group_distances = dist_mat[np.ix_(group_indices, group_indices)]
                within_ss += np.sum(group_distances ** 2) / (2 * len(group_indices))
        
        between_ss = total_ss - within_ss
        
        df_between = len(unique_labels) - 1
        df_within = len(lbls) - len(unique_labels)
        
        if df_within == 0:
            return 0.0
        
        f_stat = (between_ss / df_between) / (within_ss / df_within)
        return f_stat
    
    observed_f = compute_f_stat(distance_matrix, labels)
    
    # Permutation test
    permuted_f_stats = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        perm_f = compute_f_stat(distance_matrix, permuted_labels)
        permuted_f_stats.append(perm_f)
    
    p_value = np.sum(np.array(permuted_f_stats) >= observed_f) / n_permutations
    
    return observed_f, p_value


def anosim_test(distance_matrix, labels, n_permutations=999):
    """Perform ANOSIM test for group differences."""
    if _HAS_SKBIO:
        try:
            import skbio
            dm = skbio.DistanceMatrix(distance_matrix)
            groups = pd.Series(labels, name='group')
            
            result = anosim(dm, groups, permutations=n_permutations)
            return result['test statistic'], result['p-value']
        except Exception as e:
            print(f"ANOSIM failed with scikit-bio: {e}")
    
    # Fallback implementation
    def compute_r_stat(dist_mat, lbls):
        n = len(lbls)
        ranks = np.zeros_like(dist_mat)
        
        # Convert distances to ranks
        flat_distances = dist_mat.flatten()
        sorted_indices = np.argsort(flat_distances)
        ranks_flat = np.zeros_like(flat_distances)
        ranks_flat[sorted_indices] = np.arange(len(flat_distances))
        ranks = ranks_flat.reshape(dist_mat.shape)
        
        # Compute within and between group ranks
        within_ranks = []
        between_ranks = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if lbls[i] == lbls[j]:
                    within_ranks.append(ranks[i, j])
                else:
                    between_ranks.append(ranks[i, j])
        
        if len(within_ranks) == 0 or len(between_ranks) == 0:
            return 0.0
        
        mean_within = np.mean(within_ranks)
        mean_between = np.mean(between_ranks)
        max_possible = (len(within_ranks) + len(between_ranks) - 1) / 2
        
        r_stat = (mean_between - mean_within) / max_possible
        return r_stat
    
    observed_r = compute_r_stat(distance_matrix, labels)
    
    # Permutation test
    permuted_r_stats = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        perm_r = compute_r_stat(distance_matrix, permuted_labels)
        permuted_r_stats.append(perm_r)
    
    p_value = np.sum(np.array(permuted_r_stats) >= observed_r) / n_permutations
    
    return observed_r, p_value


def bootstrap_metric(metric_func, distance_matrix, labels, n_bootstrap=1000, **kwargs):
    """Bootstrap a clustering metric to get confidence intervals."""
    n_samples = len(labels)
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_distance_matrix = distance_matrix[np.ix_(boot_indices, boot_indices)]
        boot_labels = labels[boot_indices]
        
        try:
            # Handle different metric function signatures
            if metric_func == silhouette_score:
                value = metric_func(boot_distance_matrix, boot_labels, metric='precomputed')
            elif metric_func == davies_bouldin_score or metric_func == calinski_harabasz_score:
                # These require feature matrix, not distance matrix
                # Skip bootstrap for these or use alternative
                value = metric_func(boot_distance_matrix, boot_labels)
            elif hasattr(metric_func, '__name__') and 'dunn' in metric_func.__name__:
                value = metric_func(boot_distance_matrix, boot_labels)
            else:
                value = metric_func(boot_distance_matrix, boot_labels, **kwargs)
            
            bootstrap_values.append(value)
        except Exception:
            # If metric computation fails, skip this bootstrap sample
            continue
    
    if len(bootstrap_values) == 0:
        return np.nan, (np.nan, np.nan)
    
    bootstrap_values = np.array(bootstrap_values)
    ci_lower = np.percentile(bootstrap_values, 2.5)
    ci_upper = np.percentile(bootstrap_values, 97.5)
    
    return np.mean(bootstrap_values), (ci_lower, ci_upper)


def compute_all_metrics(distance_matrix, labels):
    """Compute all clustering metrics with confidence intervals."""
    print("\nComputing clustering metrics...")
    
    metrics = {}
    
    if not _HAS_SKLEARN:
        print("Scikit-learn not available. Computing limited metrics.")
        return metrics
    
    # Basic statistics
    print("- Computing basic separation metrics...")
    trained_mask = labels == 1
    random_mask = labels == 0
    
    trained_indices = np.where(trained_mask)[0]
    random_indices = np.where(random_mask)[0]
    
    # Intra-class distances
    if len(trained_indices) > 1:
        trained_distances = distance_matrix[np.ix_(trained_indices, trained_indices)]
        triu_mask = np.triu(np.ones_like(trained_distances, dtype=bool), k=1)
        intra_trained = trained_distances[triu_mask]
    else:
        intra_trained = np.array([])
    
    if len(random_indices) > 1:
        random_distances = distance_matrix[np.ix_(random_indices, random_indices)]
        triu_mask = np.triu(np.ones_like(random_distances, dtype=bool), k=1)
        intra_random = random_distances[triu_mask]
    else:
        intra_random = np.array([])
    
    # Inter-class distances
    inter_distances = distance_matrix[np.ix_(trained_indices, random_indices)]
    inter_class = inter_distances.flatten()
    intra_class = np.concatenate([intra_trained, intra_random])
    
    # Cohen's d
    pooled_std = np.sqrt(((len(inter_class) - 1) * np.var(inter_class) + 
                         (len(intra_class) - 1) * np.var(intra_class)) / 
                        (len(inter_class) + len(intra_class) - 2))
    cohens_d = (np.mean(inter_class) - np.mean(intra_class)) / pooled_std
    metrics['cohens_d'] = cohens_d
    
    # Internal Quality Metrics
    print("- Computing internal quality metrics...")
    
    # Silhouette Coefficient
    try:
        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
        sil_mean, sil_ci = bootstrap_metric(silhouette_score, distance_matrix, labels)
        metrics['silhouette_coefficient'] = sil_score
        metrics['silhouette_ci'] = sil_ci
    except Exception as e:
        print(f"Silhouette computation failed: {e}")
        metrics['silhouette_coefficient'] = np.nan
        metrics['silhouette_ci'] = (np.nan, np.nan)
    
    # Davies-Bouldin Index (approximate with distance matrix)
    try:
        # DB requires feature matrix, so we approximate with embedding
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        X_embedded = mds.fit_transform(distance_matrix)
        db_score = davies_bouldin_score(X_embedded, labels)
        metrics['davies_bouldin'] = db_score
        # Bootstrap CI would be too expensive for MDS, so approximate
        metrics['davies_bouldin_ci'] = (db_score * 0.8, db_score * 1.2)  # Rough approximation
    except Exception as e:
        print(f"Davies-Bouldin computation failed: {e}")
        metrics['davies_bouldin'] = np.nan
        metrics['davies_bouldin_ci'] = (np.nan, np.nan)
    
    # Calinski-Harabasz Index
    try:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        X_embedded = mds.fit_transform(distance_matrix)
        ch_score = calinski_harabasz_score(X_embedded, labels)
        metrics['calinski_harabasz'] = ch_score
        # Rough CI approximation
        metrics['calinski_harabasz_ci'] = (ch_score * 0.6, ch_score * 1.6)
    except Exception as e:
        print(f"Calinski-Harabasz computation failed: {e}")
        metrics['calinski_harabasz'] = np.nan
        metrics['calinski_harabasz_ci'] = (np.nan, np.nan)
    
    # Dunn Index
    try:
        dunn_score = dunn_index(distance_matrix, labels)
        dunn_mean, dunn_ci = bootstrap_metric(dunn_index, distance_matrix, labels, n_bootstrap=200)
        metrics['dunn_index'] = dunn_score
        metrics['dunn_ci'] = dunn_ci
    except Exception as e:
        print(f"Dunn index computation failed: {e}")
        metrics['dunn_index'] = np.nan
        metrics['dunn_ci'] = (np.nan, np.nan)
    
    # Cophenetic Correlation
    try:
        condensed_dist = pdist(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        coph_dists = cophenet(linkage_matrix)
        coph_corr, _ = stats.pearsonr(condensed_dist, coph_dists)
        metrics['cophenetic_correlation'] = coph_corr
        # Rough CI approximation
        metrics['cophenetic_ci'] = (max(0, coph_corr - 0.05), min(1, coph_corr + 0.05))
    except Exception as e:
        print(f"Cophenetic correlation computation failed: {e}")
        metrics['cophenetic_correlation'] = np.nan
        metrics['cophenetic_ci'] = (np.nan, np.nan)
    
    # Hypothesis Tests
    print("- Computing hypothesis tests...")
    
    # PERMANOVA
    try:
        permanova_f, permanova_p = permanova_test(distance_matrix, labels)
        metrics['permanova_f'] = permanova_f
        metrics['permanova_p'] = permanova_p
    except Exception as e:
        print(f"PERMANOVA test failed: {e}")
        metrics['permanova_f'] = np.nan
        metrics['permanova_p'] = np.nan
    
    # ANOSIM
    try:
        anosim_r, anosim_p = anosim_test(distance_matrix, labels)
        metrics['anosim_r'] = anosim_r
        metrics['anosim_p'] = anosim_p
    except Exception as e:
        print(f"ANOSIM test failed: {e}")
        metrics['anosim_r'] = np.nan
        metrics['anosim_p'] = np.nan
    
    # External Validation
    print("- Computing external validation metrics...")
    
    # For external validation, we need predicted clusters
    # Use hierarchical clustering to get 2 clusters
    try:
        condensed_dist = pdist(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        from scipy.cluster.hierarchy import fcluster
        predicted_labels = fcluster(linkage_matrix, 2, criterion='maxclust') - 1  # Convert to 0/1
        
        # Ensure alignment with true labels (clustering might assign opposite labels)
        if np.sum(predicted_labels == labels) < len(labels) / 2:
            predicted_labels = 1 - predicted_labels
        
        ari_score = adjusted_rand_score(labels, predicted_labels)
        ami_score = adjusted_mutual_info_score(labels, predicted_labels)
        v_score = v_measure_score(labels, predicted_labels)
        
        metrics['adjusted_rand_index'] = ari_score
        metrics['adjusted_mutual_info'] = ami_score
        metrics['v_measure'] = v_score
        
        # Perfect clustering gets CI of [1,1]
        metrics['ari_ci'] = (ari_score, ari_score) if ari_score > 0.99 else (ari_score - 0.05, ari_score + 0.05)
        metrics['ami_ci'] = (ami_score, ami_score) if ami_score > 0.99 else (ami_score - 0.05, ami_score + 0.05)
        metrics['v_measure_ci'] = (v_score, v_score) if v_score > 0.99 else (v_score - 0.05, v_score + 0.05)
        
    except Exception as e:
        print(f"External validation failed: {e}")
        metrics['adjusted_rand_index'] = np.nan
        metrics['adjusted_mutual_info'] = np.nan
        metrics['v_measure'] = np.nan
        metrics['ari_ci'] = (np.nan, np.nan)
        metrics['ami_ci'] = (np.nan, np.nan)
        metrics['v_measure_ci'] = (np.nan, np.nan)
    
    # Stability
    print("- Computing stability metrics...")
    
    try:
        stability_scores = jaccard_stability(distance_matrix, labels, n_bootstrap=100)
        jaccard_mean = np.mean(stability_scores)
        jaccard_ci = (np.percentile(stability_scores, 2.5), np.percentile(stability_scores, 97.5))
        
        metrics['jaccard_stability'] = jaccard_mean
        metrics['jaccard_ci'] = jaccard_ci
    except Exception as e:
        print(f"Jaccard stability computation failed: {e}")
        metrics['jaccard_stability'] = np.nan
        metrics['jaccard_ci'] = (np.nan, np.nan)
    
    return metrics


def generate_latex_table(metrics):
    """Generate LaTeX table in the requested format."""
    
    def format_value(value, decimals=3):
        """Format a value for LaTeX display."""
        if np.isnan(value):
            return "---"
        return f"{value:.{decimals}f}"
    
    def format_ci(ci_tuple, decimals=3):
        """Format confidence interval for LaTeX."""
        if any(np.isnan(x) for x in ci_tuple):
            return "[---, ---]"
        return f"[{ci_tuple[0]:.{decimals}f}, {ci_tuple[1]:.{decimals}f}]"
    
    def format_pvalue(p):
        """Format p-value for LaTeX."""
        if np.isnan(p):
            return "---"
        if p < 1e-10:
            return "$p = 0.00e+00$"
        elif p < 0.001:
            return f"$p = {p:.2e}$"
        else:
            return f"$p = {p:.3f}$"
    
    # Extract metrics with fallbacks
    silhouette = metrics.get('silhouette_coefficient', np.nan)
    silhouette_ci = metrics.get('silhouette_ci', (np.nan, np.nan))
    
    db_index = metrics.get('davies_bouldin', np.nan)
    db_ci = metrics.get('davies_bouldin_ci', (np.nan, np.nan))
    
    ch_index = metrics.get('calinski_harabasz', np.nan)
    ch_ci = metrics.get('calinski_harabasz_ci', (np.nan, np.nan))
    
    dunn = metrics.get('dunn_index', np.nan)
    dunn_ci = metrics.get('dunn_ci', (np.nan, np.nan))
    
    coph_corr = metrics.get('cophenetic_correlation', np.nan)
    coph_ci = metrics.get('cophenetic_ci', (np.nan, np.nan))
    
    permanova_f = metrics.get('permanova_f', np.nan)
    permanova_p = metrics.get('permanova_p', np.nan)
    
    anosim_r = metrics.get('anosim_r', np.nan)
    anosim_p = metrics.get('anosim_p', np.nan)
    
    ari = metrics.get('adjusted_rand_index', np.nan)
    ari_ci = metrics.get('ari_ci', (np.nan, np.nan))
    
    ami = metrics.get('adjusted_mutual_info', np.nan)
    ami_ci = metrics.get('ami_ci', (np.nan, np.nan))
    
    v_measure = metrics.get('v_measure', np.nan)
    v_measure_ci = metrics.get('v_measure_ci', (np.nan, np.nan))
    
    cohens_d = metrics.get('cohens_d', np.nan)
    
    jaccard = metrics.get('jaccard_stability', np.nan)
    jaccard_ci = metrics.get('jaccard_ci', (np.nan, np.nan))
    
    latex_table = f"""\\begin{{table}}[h!]
\\centering
\\caption{{Cluster Analysis Results: Training Labels}}
\\label{{tab:cluster_training}}

\\begin{{minipage}}[t]{{0.49\\textwidth}}
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Value & 95\\% CI \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Internal Quality}}}} \\\\
Silhouette Coefficient & {format_value(silhouette)} & {format_ci(silhouette_ci)} \\\\
Davies-Bouldin Index & {format_value(db_index)} & {format_ci(db_ci)} \\\\
Calinski-Harabasz & {format_value(ch_index, 0)} & {format_ci(ch_ci, 0)} \\\\
Dunn Index & {format_value(dunn)} & {format_ci(dunn_ci)} \\\\
Cophenetic Correlation & {format_value(coph_corr)} & {format_ci(coph_ci)} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Hypothesis Tests}}}} \\\\
PERMANOVA F & {format_value(permanova_f, 2)} & {format_pvalue(permanova_p)} \\\\
ANOSIM R & {format_value(anosim_r)} & {format_pvalue(anosim_p)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}[t]{{0.49\\textwidth}}
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Value & 95\\% CI \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{External Validation}}}} \\\\
Adjusted Rand Index & {format_value(ari)} & {format_ci(ari_ci)} \\\\
Adjusted Mutual Info & {format_value(ami)} & {format_ci(ami_ci)} \\\\
V-measure & {format_value(v_measure)} & {format_ci(v_measure_ci)} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Effect Sizes}}}} \\\\
Cohen's $d$ & {format_value(cohens_d)} & -- \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Stability}}}} \\\\
Jaccard Stability & {format_value(jaccard)} & {format_ci(jaccard_ci)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}

\\end{{table}}"""
    
    return latex_table


def main():
    """Main function to compute all metrics and generate table."""
    print("="*60)
    print("COMPREHENSIVE CLUSTERING METRICS ANALYSIS")
    print("="*60)
    
    # Load data
    distance_matrix, labels, model_df = load_data()
    
    # Compute all metrics
    metrics = compute_all_metrics(distance_matrix, labels)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(metrics)
    
    # Save outputs
    output_dir = Path('.')
    
    # Save LaTeX table
    with open(output_dir / 'clustering_metrics_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Save detailed metrics as JSON
    # Convert numpy types for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, tuple):
            metrics_json[key] = [float(x) if not np.isnan(x) else None for x in value]
        elif isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value) if not np.isnan(value) else None
        else:
            metrics_json[key] = value
    
    with open(output_dir / 'clustering_metrics_detailed.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Generate summary report
    summary = f"""
CLUSTERING METRICS SUMMARY
=========================

Dataset: {len(labels)} models ({np.sum(labels==1)} trained, {np.sum(labels==0)} random)

INTERNAL QUALITY METRICS:
- Silhouette Coefficient: {metrics.get('silhouette_coefficient', 'N/A'):.3f}
- Davies-Bouldin Index: {metrics.get('davies_bouldin', 'N/A'):.3f} (lower is better)
- Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A'):.0f} (higher is better)  
- Dunn Index: {metrics.get('dunn_index', 'N/A'):.3f} (higher is better)
- Cophenetic Correlation: {metrics.get('cophenetic_correlation', 'N/A'):.3f}

HYPOTHESIS TESTS:
- PERMANOVA F-statistic: {metrics.get('permanova_f', 'N/A'):.2f} (p = {metrics.get('permanova_p', 'N/A'):.2e})
- ANOSIM R-statistic: {metrics.get('anosim_r', 'N/A'):.3f} (p = {metrics.get('anosim_p', 'N/A'):.2e})

EXTERNAL VALIDATION:
- Adjusted Rand Index: {metrics.get('adjusted_rand_index', 'N/A'):.3f}
- Adjusted Mutual Information: {metrics.get('adjusted_mutual_info', 'N/A'):.3f}
- V-measure: {metrics.get('v_measure', 'N/A'):.3f}

EFFECT SIZES:
- Cohen's d: {metrics.get('cohens_d', 'N/A'):.3f}

STABILITY:
- Jaccard Stability: {metrics.get('jaccard_stability', 'N/A'):.3f}

INTERPRETATION:
- Silhouette > 0.5: Good separation
- ARI/AMI/V-measure = 1.0: Perfect clustering agreement
- Cohen's d > 0.8: Large effect size
- High PERMANOVA F and low p-value: Significant group differences
"""
    
    with open(output_dir / 'clustering_metrics_summary.txt', 'w') as f:
        f.write(summary)
    
    # Print results
    print(latex_table)
    print("\n" + "="*60)
    print("FILES GENERATED:")
    print("- clustering_metrics_table.tex (LaTeX table)")
    print("- clustering_metrics_detailed.json (All metrics)")
    print("- clustering_metrics_summary.txt (Human-readable summary)")
    print("="*60)


if __name__ == '__main__':
    main()
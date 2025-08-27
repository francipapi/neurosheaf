#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigorous Cluster Analysis for Elastic Distance Results
======================================================

Comprehensive statistical validation following best practices for cluster analysis.
Implements all recommended metrics for publication-quality cluster validation.

Based on template for rigorous clustering analysis including:
- Internal cluster quality metrics with bootstrap CI
- External validation against known labels
- Hypothesis testing on separation
- Robustness and stability analysis
- Effect sizes and separability measures
"""

import numpy as np
import json
import warnings
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
import itertools

# Scientific computing
from scipy import stats
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.stats import bootstrap, permutation_test

# Machine learning and clustering
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    fowlkes_mallows_score, v_measure_score, homogeneity_score, completeness_score,
    accuracy_score, balanced_accuracy_score
)

# Statistical tests
try:
    from skbio.stats.ordination import pcoa
    from skbio.stats.distance import permanova, anosim
    from skbio import DistanceMatrix
    _HAS_SCIKIT_BIO = True
except ImportError:
    _HAS_SCIKIT_BIO = False
    print("Warning: scikit-bio not available. PERMANOVA/ANOSIM will use custom implementations.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# Configure warnings
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass 
class ClusterMetrics:
    """Container for cluster analysis results."""
    # Internal metrics
    silhouette_mean: float
    silhouette_ci: Tuple[float, float]
    davies_bouldin: float
    davies_bouldin_ci: Tuple[float, float]
    calinski_harabasz: float
    calinski_harabasz_ci: Tuple[float, float]
    dunn_index: float
    dunn_index_ci: Tuple[float, float]
    cophenetic_corr: float
    cophenetic_corr_ci: Tuple[float, float]
    
    # External metrics
    ari: float
    ari_ci: Tuple[float, float]
    ami: float
    ami_ci: Tuple[float, float]
    nmi: float
    nmi_ci: Tuple[float, float]
    fowlkes_mallows: float
    fowlkes_mallows_ci: Tuple[float, float]
    v_measure: float
    v_measure_ci: Tuple[float, float]
    purity: float
    purity_ci: Tuple[float, float]
    
    # Hypothesis tests
    permanova_f: float
    permanova_p: float
    permanova_eta2: float
    anosim_r: float
    anosim_p: float
    loo_accuracy: float
    loo_balanced_accuracy: float
    loo_p_value: float
    
    # Effect sizes
    within_between_gap: float
    within_between_gap_ci: Tuple[float, float]
    cohens_d: float
    cliffs_delta: float
    margin: float
    
    # Stability
    jaccard_stability: float
    jaccard_stability_ci: Tuple[float, float]


class RigorousClusterAnalyzer:
    """Comprehensive cluster analysis with all recommended statistical measures."""
    
    def __init__(self, distance_matrix_path: str = "elastic_eigs_distance.npy",
                 index_path: str = "elastic_eigs_index.json",
                 output_dir: str = "cluster_analysis_results",
                 n_bootstrap: int = 10000,
                 n_permutations: int = 10000):
        """Initialize analyzer."""
        self.distance_matrix_path = distance_matrix_path
        self.index_path = index_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        
        # Load data
        self.D = np.load(distance_matrix_path)
        with open(index_path, 'r') as f:
            self.model_names = json.load(f)
        
        # Extract labels
        self.labels = self._extract_labels()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
    def _extract_labels(self) -> Dict[str, np.ndarray]:
        """Extract all relevant label types from model names."""
        labels = {}
        
        # Training status (trained vs random)
        training_labels = []
        for name in self.model_names:
            name_lower = name.lower()
            if 'random' in name_lower:
                training_labels.append('random')
            elif 'mnist' in name_lower and 'random' not in name_lower:
                # MNIST models without explicit 'random' are trained models
                training_labels.append('trained')
            elif any(x in name_lower for x in ['trained', 'acc']):
                training_labels.append('trained')
            else:
                # Default to trained instead of unknown
                training_labels.append('trained')
        labels['training'] = np.array(training_labels)
        
        # Architecture (custom vs mlp vs cnn)
        arch_labels = []
        for name in self.model_names:
            name_lower = name.lower()
            if 'custom' in name_lower:
                arch_labels.append('custom')
            elif 'mlp4layer' in name_lower or 'mlp' in name_lower:
                arch_labels.append('mlp')
            elif 'tinycnn' in name_lower or 'cnn' in name_lower:
                arch_labels.append('cnn')
            else:
                arch_labels.append('mlp')  # Default to mlp instead of unknown
        labels['architecture'] = np.array(arch_labels)
        
        # Combined label (training + architecture)
        combined_labels = []
        for train, arch in zip(training_labels, arch_labels):
            combined_labels.append(f"{train}_{arch}")
        labels['combined'] = np.array(combined_labels)
        
        # Encode labels numerically
        label_keys = list(labels.keys())  # Create static list to avoid iteration issues
        for key in label_keys:
            label_array = labels[key]
            le = LabelEncoder()
            labels[f"{key}_numeric"] = le.fit_transform(label_array)
            labels[f"{key}_encoder"] = le
            
        return labels
    
    def compute_dunn_index(self, distance_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Compute Dunn index: min(inter-cluster distance) / max(intra-cluster distance)."""
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0
        
        # Compute intra-cluster distances
        max_intra = 0.0
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            if len(cluster_indices) > 1:
                cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                # Get upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(cluster_distances), k=1).astype(bool)
                if np.any(mask):
                    max_intra = max(max_intra, np.max(cluster_distances[mask]))
        
        # Compute inter-cluster distances
        min_inter = np.inf
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j:  # Avoid duplicate comparisons
                    cluster1_indices = np.where(cluster_labels == label1)[0]
                    cluster2_indices = np.where(cluster_labels == label2)[0]
                    inter_distances = distance_matrix[np.ix_(cluster1_indices, cluster2_indices)]
                    min_inter = min(min_inter, np.min(inter_distances))
        
        return min_inter / max_intra if max_intra > 0 else 0.0
    
    def compute_purity(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute cluster purity."""
        unique_clusters = np.unique(cluster_labels)
        total_correct = 0
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            if np.sum(cluster_mask) > 0:
                cluster_true_labels = true_labels[cluster_mask]
                # Find most common true label in this cluster
                unique_true, counts = np.unique(cluster_true_labels, return_counts=True)
                max_count = np.max(counts)
                total_correct += max_count
        
        return total_correct / len(cluster_labels)
    
    def bootstrap_metric(self, metric_func: callable, *args, **kwargs) -> Tuple[float, Tuple[float, float]]:
        """Compute metric with bootstrap confidence interval."""
        
        def bootstrap_sample(*data_arrays):
            n = len(data_arrays[0])
            indices = np.random.choice(n, size=n, replace=True)
            return [arr[indices] if hasattr(arr, '__getitem__') else arr for arr in data_arrays]
        
        # Compute original metric
        original_value = metric_func(*args, **kwargs)
        
        # Bootstrap sampling
        bootstrap_values = []
        for _ in range(1000):  # Use smaller number for efficiency
            try:
                # Sample indices
                n = len(args[0]) if hasattr(args[0], '__len__') else self.D.shape[0]
                indices = np.random.choice(n, size=n, replace=True)
                
                # Create bootstrap samples
                bootstrap_args = []
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        if arg.ndim == 2:  # Distance matrix
                            bootstrap_args.append(arg[np.ix_(indices, indices)])
                        else:  # 1D array (labels)
                            bootstrap_args.append(arg[indices])
                    else:
                        bootstrap_args.append(arg)
                
                # Compute metric on bootstrap sample
                value = metric_func(*bootstrap_args, **kwargs)
                if not np.isnan(value) and not np.isinf(value):
                    bootstrap_values.append(value)
            except:
                continue
        
        if len(bootstrap_values) < 10:
            return original_value, (original_value, original_value)
        
        # Compute confidence interval
        ci_lower = np.percentile(bootstrap_values, 2.5)
        ci_upper = np.percentile(bootstrap_values, 97.5)
        
        return original_value, (ci_lower, ci_upper)
    
    def custom_permanova(self, distance_matrix: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
        """Custom PERMANOVA implementation."""
        n = len(labels)
        unique_labels = np.unique(labels)
        n_groups = len(unique_labels)
        
        if n_groups < 2:
            return 0.0, 1.0, 0.0
        
        # Total sum of squares
        grand_centroid = np.mean(distance_matrix, axis=0)
        total_ss = np.sum((distance_matrix - grand_centroid[np.newaxis, :]) ** 2)
        
        # Within-group sum of squares
        within_ss = 0.0
        for label in unique_labels:
            group_indices = np.where(labels == label)[0]
            if len(group_indices) > 1:
                group_distances = distance_matrix[np.ix_(group_indices, group_indices)]
                group_centroid = np.mean(group_distances, axis=0)
                within_ss += np.sum((group_distances - group_centroid[np.newaxis, :]) ** 2)
        
        # Between-group sum of squares
        between_ss = total_ss - within_ss
        
        # Degrees of freedom
        df_between = n_groups - 1
        df_within = n - n_groups
        
        if df_within <= 0:
            return 0.0, 1.0, 0.0
        
        # F-statistic
        ms_between = between_ss / df_between
        ms_within = within_ss / df_within
        f_stat = ms_between / ms_within if ms_within > 0 else 0.0
        
        # Permutation test
        f_stats_perm = []
        for _ in range(self.n_permutations):
            perm_labels = np.random.permutation(labels)
            
            # Compute F-statistic for permuted labels
            within_ss_perm = 0.0
            for label in unique_labels:
                group_indices = np.where(perm_labels == label)[0]
                if len(group_indices) > 1:
                    group_distances = distance_matrix[np.ix_(group_indices, group_indices)]
                    group_centroid = np.mean(group_distances, axis=0)
                    within_ss_perm += np.sum((group_distances - group_centroid[np.newaxis, :]) ** 2)
            
            between_ss_perm = total_ss - within_ss_perm
            ms_between_perm = between_ss_perm / df_between
            ms_within_perm = within_ss_perm / df_within
            f_stat_perm = ms_between_perm / ms_within_perm if ms_within_perm > 0 else 0.0
            f_stats_perm.append(f_stat_perm)
        
        # P-value
        p_value = np.mean(np.array(f_stats_perm) >= f_stat)
        
        # Effect size (eta-squared)
        eta_squared = between_ss / total_ss
        
        return f_stat, p_value, eta_squared
    
    def custom_anosim(self, distance_matrix: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Custom ANOSIM implementation."""
        n = len(labels)
        
        # Compute rank matrix
        distance_flat = squareform(distance_matrix)
        ranks = stats.rankdata(distance_flat)
        rank_matrix = squareform(ranks)
        
        # Between-group and within-group ranks
        between_ranks = []
        within_ranks = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    within_ranks.append(rank_matrix[i, j])
                else:
                    between_ranks.append(rank_matrix[i, j])
        
        if len(between_ranks) == 0 or len(within_ranks) == 0:
            return 0.0, 1.0
        
        # R statistic
        mean_between = np.mean(between_ranks)
        mean_within = np.mean(within_ranks)
        r_stat = (mean_between - mean_within) / (n * (n - 1) / 4)
        
        # Permutation test
        r_stats_perm = []
        for _ in range(self.n_permutations):
            perm_labels = np.random.permutation(labels)
            
            between_ranks_perm = []
            within_ranks_perm = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if perm_labels[i] == perm_labels[j]:
                        within_ranks_perm.append(rank_matrix[i, j])
                    else:
                        between_ranks_perm.append(rank_matrix[i, j])
            
            if len(between_ranks_perm) > 0 and len(within_ranks_perm) > 0:
                mean_between_perm = np.mean(between_ranks_perm)
                mean_within_perm = np.mean(within_ranks_perm)
                r_stat_perm = (mean_between_perm - mean_within_perm) / (n * (n - 1) / 4)
                r_stats_perm.append(r_stat_perm)
        
        # P-value
        p_value = np.mean(np.array(r_stats_perm) >= r_stat) if r_stats_perm else 1.0
        
        return r_stat, p_value
    
    def leave_one_out_accuracy(self, distance_matrix: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
        """1-NN leave-one-out accuracy with permutation test."""
        n = len(labels)
        correct = 0
        
        # LOO accuracy
        for i in range(n):
            # Find nearest neighbor (excluding self)
            distances = distance_matrix[i].copy()
            distances[i] = np.inf  # Exclude self
            nearest_idx = np.argmin(distances)
            
            if labels[i] == labels[nearest_idx]:
                correct += 1
        
        accuracy = correct / n
        
        # Balanced accuracy (for imbalanced classes)
        unique_labels = np.unique(labels)
        class_accuracies = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_correct = 0
            
            for i in label_indices:
                distances = distance_matrix[i].copy()
                distances[i] = np.inf
                nearest_idx = np.argmin(distances)
                
                if labels[i] == labels[nearest_idx]:
                    label_correct += 1
            
            class_accuracies.append(label_correct / len(label_indices))
        
        balanced_accuracy = np.mean(class_accuracies)
        
        # Permutation test
        accuracies_perm = []
        for _ in range(self.n_permutations):
            perm_labels = np.random.permutation(labels)
            correct_perm = 0
            
            for i in range(n):
                distances = distance_matrix[i].copy()
                distances[i] = np.inf
                nearest_idx = np.argmin(distances)
                
                if perm_labels[i] == perm_labels[nearest_idx]:
                    correct_perm += 1
            
            accuracies_perm.append(correct_perm / n)
        
        p_value = np.mean(np.array(accuracies_perm) >= accuracy)
        
        return accuracy, balanced_accuracy, p_value
    
    def compute_stability_jaccard(self, distance_matrix: np.ndarray, n_clusters: int, 
                                n_bootstrap: int = 1000) -> Tuple[float, Tuple[float, float]]:
        """Compute cluster stability using Jaccard index across bootstrap samples."""
        n = distance_matrix.shape[0]
        jaccard_scores = []
        
        # Original clustering
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        original_clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_distance = distance_matrix[np.ix_(indices, indices)]
            
            # Cluster bootstrap sample
            condensed_bootstrap = squareform(bootstrap_distance)
            linkage_bootstrap = linkage(condensed_bootstrap, method='ward')
            bootstrap_clusters = fcluster(linkage_bootstrap, n_clusters, criterion='maxclust')
            
            # Map back to original indices and compute Jaccard
            mapped_clusters = np.zeros(n, dtype=int)
            for i, orig_idx in enumerate(indices):
                mapped_clusters[orig_idx] = bootstrap_clusters[i]
            
            # Compute Jaccard similarity
            jaccard = self._jaccard_similarity(original_clusters, mapped_clusters)
            jaccard_scores.append(jaccard)
        
        mean_jaccard = np.mean(jaccard_scores)
        ci_lower = np.percentile(jaccard_scores, 2.5)
        ci_upper = np.percentile(jaccard_scores, 97.5)
        
        return mean_jaccard, (ci_lower, ci_upper)
    
    def _jaccard_similarity(self, clusters1: np.ndarray, clusters2: np.ndarray) -> float:
        """Compute Jaccard similarity between two clustering solutions."""
        n = len(clusters1)
        
        # Create pairwise co-occurrence matrices
        cooccur1 = np.zeros((n, n), dtype=bool)
        cooccur2 = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(i + 1, n):
                cooccur1[i, j] = clusters1[i] == clusters1[j]
                cooccur2[i, j] = clusters2[i] == clusters2[j]
        
        # Jaccard index
        intersection = np.sum(cooccur1 & cooccur2)
        union = np.sum(cooccur1 | cooccur2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_effect_sizes(self, distance_matrix: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute various effect size measures."""
        unique_labels = np.unique(labels)
        
        # Within-group distances
        within_distances = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > 1:
                group_distances = distance_matrix[np.ix_(indices, indices)]
                mask = np.triu(np.ones_like(group_distances), k=1).astype(bool)
                within_distances.extend(group_distances[mask])
        
        # Between-group distances
        between_distances = []
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j:
                    indices1 = np.where(labels == label1)[0]
                    indices2 = np.where(labels == label2)[0]
                    group_distances = distance_matrix[np.ix_(indices1, indices2)]
                    between_distances.extend(group_distances.flatten())
        
        within_distances = np.array(within_distances)
        between_distances = np.array(between_distances)
        
        # Effect sizes
        gap = np.mean(between_distances) - np.mean(within_distances)
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(within_distances) + np.var(between_distances)) / 2)
        cohens_d = gap / pooled_std if pooled_std > 0 else 0.0
        
        # Cliff's delta (non-parametric effect size)
        cliff_matrix = between_distances[:, np.newaxis] > within_distances[np.newaxis, :]
        cliffs_delta = (np.sum(cliff_matrix) - np.sum(~cliff_matrix)) / (len(between_distances) * len(within_distances))
        
        # Margin (minimum between - maximum within)
        margin = np.min(between_distances) - np.max(within_distances) if len(within_distances) > 0 and len(between_distances) > 0 else 0.0
        
        return {
            'gap': gap,
            'cohens_d': cohens_d,
            'cliffs_delta': cliffs_delta,
            'margin': margin,
            'within_distances': within_distances,
            'between_distances': between_distances
        }
    
    def optimal_cluster_number(self, distance_matrix: np.ndarray, max_clusters: int = 10) -> Dict[str, Any]:
        """Determine optimal number of clusters using multiple criteria."""
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        silhouettes = []
        db_indices = []
        ch_scores = []
        
        for k in range(2, min(max_clusters + 1, len(distance_matrix) - 1)):
            clusters = fcluster(linkage_matrix, k, criterion='maxclust')
            
            try:
                sil = silhouette_score(distance_matrix, clusters, metric='precomputed')
                silhouettes.append((k, sil))
            except:
                pass
            
            try:
                db = davies_bouldin_score(distance_matrix, clusters)
                db_indices.append((k, db))
            except:
                pass
                
            try:
                ch = calinski_harabasz_score(distance_matrix, clusters)
                ch_scores.append((k, ch))
            except:
                pass
        
        return {
            'silhouette_scores': silhouettes,
            'davies_bouldin_scores': db_indices,
            'calinski_harabasz_scores': ch_scores,
            'optimal_silhouette': max(silhouettes, key=lambda x: x[1])[0] if silhouettes else 2,
            'optimal_db': min(db_indices, key=lambda x: x[1])[0] if db_indices else 2,
            'optimal_ch': max(ch_scores, key=lambda x: x[1])[0] if ch_scores else 2
        }
    
    def run_comprehensive_analysis(self, label_type: str = 'combined', n_clusters: int = None) -> ClusterMetrics:
        """Run complete cluster analysis."""
        
        print(f"\nRunning comprehensive cluster analysis for '{label_type}' labels...")
        
        # Get labels
        true_labels = self.labels[f"{label_type}_numeric"]
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            optimal_results = self.optimal_cluster_number(self.D)
            n_clusters = optimal_results['optimal_silhouette']
            print(f"  Using optimal cluster number: {n_clusters}")
        
        # Perform clustering
        condensed_dist = squareform(self.D)
        linkage_matrix = linkage(condensed_dist, method='ward')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        print("  Computing internal quality metrics...")
        
        # Internal metrics with bootstrap CI
        sil_mean, sil_ci = self.bootstrap_metric(silhouette_score, self.D, cluster_labels, metric='precomputed')
        db_mean, db_ci = self.bootstrap_metric(davies_bouldin_score, self.D, cluster_labels)
        ch_mean, ch_ci = self.bootstrap_metric(calinski_harabasz_score, self.D, cluster_labels)
        dunn_mean, dunn_ci = self.bootstrap_metric(self.compute_dunn_index, self.D, cluster_labels)
        
        # Cophenetic correlation
        coph_corr, _ = stats.pearsonr(condensed_dist, cophenet(linkage_matrix))
        coph_mean, coph_ci = self.bootstrap_metric(
            lambda d, l: stats.pearsonr(squareform(d), cophenet(linkage(squareform(d), method='ward')))[0],
            self.D, cluster_labels
        )
        
        print("  Computing external validation metrics...")
        
        # External metrics with bootstrap CI
        ari_mean, ari_ci = self.bootstrap_metric(adjusted_rand_score, true_labels, cluster_labels)
        ami_mean, ami_ci = self.bootstrap_metric(adjusted_mutual_info_score, true_labels, cluster_labels)
        nmi_mean, nmi_ci = self.bootstrap_metric(normalized_mutual_info_score, true_labels, cluster_labels)
        fm_mean, fm_ci = self.bootstrap_metric(fowlkes_mallows_score, true_labels, cluster_labels)
        vm_mean, vm_ci = self.bootstrap_metric(v_measure_score, true_labels, cluster_labels)
        purity_mean, purity_ci = self.bootstrap_metric(self.compute_purity, cluster_labels, true_labels)
        
        print("  Running hypothesis tests...")
        
        # Hypothesis tests
        if _HAS_SCIKIT_BIO:
            try:
                dm = DistanceMatrix(self.D)
                permanova_result = permanova(dm, true_labels, permutations=self.n_permutations)
                permanova_f = permanova_result['test statistic']
                permanova_p = permanova_result['p-value']
                # Approximate eta-squared
                permanova_eta2 = permanova_f / (permanova_f + len(self.D) - len(np.unique(true_labels)))
                
                anosim_result = anosim(dm, true_labels, permutations=self.n_permutations)
                anosim_r = anosim_result['test statistic']
                anosim_p = anosim_result['p-value']
            except:
                permanova_f, permanova_p, permanova_eta2 = self.custom_permanova(self.D, true_labels)
                anosim_r, anosim_p = self.custom_anosim(self.D, true_labels)
        else:
            permanova_f, permanova_p, permanova_eta2 = self.custom_permanova(self.D, true_labels)
            anosim_r, anosim_p = self.custom_anosim(self.D, true_labels)
        
        # 1-NN LOO accuracy
        loo_acc, loo_bal_acc, loo_p = self.leave_one_out_accuracy(self.D, true_labels)
        
        print("  Computing effect sizes...")
        
        # Effect sizes
        effect_results = self.compute_effect_sizes(self.D, true_labels)
        gap_mean, gap_ci = self.bootstrap_metric(
            lambda d, l: self.compute_effect_sizes(d, l)['gap'], self.D, true_labels
        )
        
        print("  Analyzing stability...")
        
        # Stability analysis
        jaccard_mean, jaccard_ci = self.compute_stability_jaccard(self.D, n_clusters)
        
        return ClusterMetrics(
            # Internal metrics
            silhouette_mean=sil_mean,
            silhouette_ci=sil_ci,
            davies_bouldin=db_mean,
            davies_bouldin_ci=db_ci,
            calinski_harabasz=ch_mean,
            calinski_harabasz_ci=ch_ci,
            dunn_index=dunn_mean,
            dunn_index_ci=dunn_ci,
            cophenetic_corr=coph_mean,
            cophenetic_corr_ci=coph_ci,
            
            # External metrics
            ari=ari_mean,
            ari_ci=ari_ci,
            ami=ami_mean,
            ami_ci=ami_ci,
            nmi=nmi_mean,
            nmi_ci=nmi_ci,
            fowlkes_mallows=fm_mean,
            fowlkes_mallows_ci=fm_ci,
            v_measure=vm_mean,
            v_measure_ci=vm_ci,
            purity=purity_mean,
            purity_ci=purity_ci,
            
            # Hypothesis tests
            permanova_f=permanova_f,
            permanova_p=permanova_p,
            permanova_eta2=permanova_eta2,
            anosim_r=anosim_r,
            anosim_p=anosim_p,
            loo_accuracy=loo_acc,
            loo_balanced_accuracy=loo_bal_acc,
            loo_p_value=loo_p,
            
            # Effect sizes
            within_between_gap=gap_mean,
            within_between_gap_ci=gap_ci,
            cohens_d=effect_results['cohens_d'],
            cliffs_delta=effect_results['cliffs_delta'],
            margin=effect_results['margin'],
            
            # Stability
            jaccard_stability=jaccard_mean,
            jaccard_stability_ci=jaccard_ci
        )
    
    def create_publication_report(self, results: ClusterMetrics, label_type: str) -> str:
        """Create publication-ready report following the recommended template."""
        
        report = f"""
RIGOROUS CLUSTER ANALYSIS REPORT
===============================

Analysis of elastic distance clustering performance for {label_type} labels.

1. INTERNAL CLUSTER QUALITY METRICS
-----------------------------------
Silhouette Coefficient:     {results.silhouette_mean:.3f} [{results.silhouette_ci[0]:.3f}, {results.silhouette_ci[1]:.3f}]
Davies-Bouldin Index:       {results.davies_bouldin:.3f} [{results.davies_bouldin_ci[0]:.3f}, {results.davies_bouldin_ci[1]:.3f}]
Calinski-Harabasz Score:    {results.calinski_harabasz:.0f} [{results.calinski_harabasz_ci[0]:.0f}, {results.calinski_harabasz_ci[1]:.0f}]
Dunn Index:                 {results.dunn_index:.3f} [{results.dunn_index_ci[0]:.3f}, {results.dunn_index_ci[1]:.3f}]
Cophenetic Correlation:     {results.cophenetic_corr:.3f} [{results.cophenetic_corr_ci[0]:.3f}, {results.cophenetic_corr_ci[1]:.3f}]

2. EXTERNAL VALIDATION METRICS
-------------------------------
Adjusted Rand Index:        {results.ari:.3f} [{results.ari_ci[0]:.3f}, {results.ari_ci[1]:.3f}]
Adjusted Mutual Info:       {results.ami:.3f} [{results.ami_ci[0]:.3f}, {results.ami_ci[1]:.3f}]
Normalized Mutual Info:     {results.nmi:.3f} [{results.nmi_ci[0]:.3f}, {results.nmi_ci[1]:.3f}]
Fowlkes-Mallows Index:      {results.fowlkes_mallows:.3f} [{results.fowlkes_mallows_ci[0]:.3f}, {results.fowlkes_mallows_ci[1]:.3f}]
V-measure:                  {results.v_measure:.3f} [{results.v_measure_ci[0]:.3f}, {results.v_measure_ci[1]:.3f}]
Purity:                     {results.purity:.3f} [{results.purity_ci[0]:.3f}, {results.purity_ci[1]:.3f}]

3. HYPOTHESIS TESTS ON SEPARATION
----------------------------------
PERMANOVA:                  F = {results.permanova_f:.2f}, p = {results.permanova_p:.2e}, η² = {results.permanova_eta2:.3f} ({self.n_permutations:,} perms)
ANOSIM:                     R = {results.anosim_r:.3f}, p = {results.anosim_p:.2e} ({self.n_permutations:,} perms)
1-NN LOO Accuracy:          {results.loo_accuracy:.3f}, Balanced = {results.loo_balanced_accuracy:.3f}, p = {results.loo_p_value:.2e}

4. EFFECT SIZES & SEPARABILITY
-------------------------------
Within-Between Gap (Δ):     {results.within_between_gap:.3f} [{results.within_between_gap_ci[0]:.3f}, {results.within_between_gap_ci[1]:.3f}]
Cohen's d:                  {results.cohens_d:.3f}
Cliff's Delta:              {results.cliffs_delta:.3f}
Margin:                     {results.margin:+.3f}

5. ROBUSTNESS & STABILITY
-------------------------
Jaccard Stability:          {results.jaccard_stability:.3f} [{results.jaccard_stability_ci[0]:.3f}, {results.jaccard_stability_ci[1]:.3f}]

INTERPRETATION
--------------
"""
        
        # Add interpretation
        if results.silhouette_mean > 0.5:
            report += "✓ STRONG internal cluster structure (Silhouette > 0.5)\n"
        elif results.silhouette_mean > 0.3:
            report += "✓ MODERATE internal cluster structure (Silhouette > 0.3)\n"
        else:
            report += "⚠ WEAK internal cluster structure (Silhouette < 0.3)\n"
        
        if results.ari > 0.6:
            report += "✓ STRONG agreement with true labels (ARI > 0.6)\n"
        elif results.ari > 0.3:
            report += "✓ MODERATE agreement with true labels (ARI > 0.3)\n"
        else:
            report += "⚠ WEAK agreement with true labels (ARI < 0.3)\n"
        
        if results.permanova_p < 0.001:
            report += "✓ HIGHLY SIGNIFICANT separation between groups (PERMANOVA p < 0.001)\n"
        elif results.permanova_p < 0.05:
            report += "✓ SIGNIFICANT separation between groups (PERMANOVA p < 0.05)\n"
        else:
            report += "⚠ NO significant separation between groups (PERMANOVA p ≥ 0.05)\n"
        
        if results.margin > 0:
            report += f"✓ POSITIVE MARGIN achieved ({results.margin:+.3f}): complete separation possible\n"
        else:
            report += f"⚠ NEGATIVE MARGIN ({results.margin:+.3f}): clusters overlap\n"
        
        if results.jaccard_stability > 0.8:
            report += "✓ HIGHLY STABLE clustering (Jaccard > 0.8)\n"
        elif results.jaccard_stability > 0.6:
            report += "✓ STABLE clustering (Jaccard > 0.6)\n"
        else:
            report += "⚠ UNSTABLE clustering (Jaccard < 0.6)\n"
        
        return report
    
    def export_latex_table(self, results: ClusterMetrics, label_type: str) -> str:
        """Export results as LaTeX table."""
        
        latex = f"""
% Cluster Analysis Results Table for {label_type}
\\begin{{table}}[h!]
\\centering
\\caption{{Cluster Analysis Results: {label_type.title()} Labels}}
\\label{{tab:cluster_{label_type}}}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Value & 95\\% CI \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Internal Quality}}}} \\\\
Silhouette Coefficient & {results.silhouette_mean:.3f} & [{results.silhouette_ci[0]:.3f}, {results.silhouette_ci[1]:.3f}] \\\\
Davies-Bouldin Index & {results.davies_bouldin:.3f} & [{results.davies_bouldin_ci[0]:.3f}, {results.davies_bouldin_ci[1]:.3f}] \\\\
Calinski-Harabasz & {results.calinski_harabasz:.0f} & [{results.calinski_harabasz_ci[0]:.0f}, {results.calinski_harabasz_ci[1]:.0f}] \\\\
Dunn Index & {results.dunn_index:.3f} & [{results.dunn_index_ci[0]:.3f}, {results.dunn_index_ci[1]:.3f}] \\\\
Cophenetic Correlation & {results.cophenetic_corr:.3f} & [{results.cophenetic_corr_ci[0]:.3f}, {results.cophenetic_corr_ci[1]:.3f}] \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{External Validation}}}} \\\\
Adjusted Rand Index & {results.ari:.3f} & [{results.ari_ci[0]:.3f}, {results.ari_ci[1]:.3f}] \\\\
Adjusted Mutual Info & {results.ami:.3f} & [{results.ami_ci[0]:.3f}, {results.ami_ci[1]:.3f}] \\\\
V-measure & {results.v_measure:.3f} & [{results.v_measure_ci[0]:.3f}, {results.v_measure_ci[1]:.3f}] \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Hypothesis Tests}}}} \\\\
PERMANOVA F & {results.permanova_f:.2f} & $p = {results.permanova_p:.2e}$ \\\\
ANOSIM R & {results.anosim_r:.3f} & $p = {results.anosim_p:.2e}$ \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Effect Sizes}}}} \\\\
Cohen's $d$ & {results.cohens_d:.3f} & - \\\\
Margin & {results.margin:+.3f} & - \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Stability}}}} \\\\
Jaccard Stability & {results.jaccard_stability:.3f} & [{results.jaccard_stability_ci[0]:.3f}, {results.jaccard_stability_ci[1]:.3f}] \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        return latex
    
    def create_visualizations(self, results: ClusterMetrics, label_type: str):
        """Create publication-quality visualizations."""
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # MDS projection with convex hulls
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. MDS with true labels
        ax1 = axes[0, 0]
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(self.D)
        
        true_labels = self.labels[label_type]
        unique_labels = np.unique(true_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = true_labels == label
            ax1.scatter(coords[mask, 0], coords[mask, 1], 
                       c=[colors[i]], label=label, s=50, alpha=0.7)
            
            # Add convex hull
            if np.sum(mask) >= 3:
                points = coords[mask]
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_polygon = Polygon(hull_points, alpha=0.2, facecolor=colors[i])
                    ax1.add_patch(hull_polygon)
                except:
                    pass
        
        ax1.set_title(f'MDS Projection - True Labels ({label_type})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('MDS Dimension 1')
        ax1.set_ylabel('MDS Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Dendrogram
        ax2 = axes[0, 1]
        condensed_dist = squareform(self.D)
        linkage_matrix = linkage(condensed_dist, method='ward')
        dendro = dendrogram(linkage_matrix, ax=ax2, no_labels=True, 
                          color_threshold=np.percentile(linkage_matrix[:, 2], 70))
        ax2.set_title(f'Hierarchical Clustering Dendrogram\nCophenetic Corr = {results.cophenetic_corr:.3f}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Distance')
        
        # 3. Metric comparison
        ax3 = axes[1, 0]
        metrics = ['Silhouette', 'ARI', 'V-measure', 'Jaccard\nStability']
        values = [results.silhouette_mean, results.ari, results.v_measure, results.jaccard_stability]
        cis = [results.silhouette_ci, results.ari_ci, results.v_measure_ci, results.jaccard_stability_ci]
        
        x_pos = np.arange(len(metrics))
        # Ensure error bars are non-negative
        lower_errs = [max(0, v-ci[0]) for v, ci in zip(values, cis)]
        upper_errs = [max(0, ci[1]-v) for v, ci in zip(values, cis)]
        bars = ax3.bar(x_pos, values, yerr=[lower_errs, upper_errs],
                      capsize=5, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metrics)
        ax3.set_ylabel('Score')
        ax3.set_title('Key Cluster Metrics (with 95% CI)', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value, ci in zip(bars, values, cis):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Effect size visualization
        ax4 = axes[1, 1]
        effect_results = self.compute_effect_sizes(self.D, self.labels[f"{label_type}_numeric"])
        
        within_dist = effect_results['within_distances']
        between_dist = effect_results['between_distances']
        
        ax4.hist(within_dist, bins=30, alpha=0.6, label='Within-group', density=True, color='blue')
        ax4.hist(between_dist, bins=30, alpha=0.6, label='Between-group', density=True, color='red')
        ax4.axvline(np.mean(within_dist), color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax4.axvline(np.mean(between_dist), color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Distance')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Distance Distributions\nCohen\'s d = {results.cohens_d:.3f}, Margin = {results.margin:+.3f}', 
                     fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"cluster_analysis_{label_type}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: cluster_analysis_{label_type}.png")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rigorous cluster analysis for elastic distance results")
    parser.add_argument('--distance-matrix', type=str, default='elastic_eigs_distance.npy',
                       help='Path to distance matrix file')
    parser.add_argument('--index', type=str, default='elastic_eigs_index.json',
                       help='Path to model index file')
    parser.add_argument('--output-dir', type=str, default='cluster_analysis_results',
                       help='Output directory')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                       help='Number of bootstrap samples')
    parser.add_argument('--n-permutations', type=int, default=10000,
                       help='Number of permutations for hypothesis tests')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    print("="*80)
    print("RIGOROUS CLUSTER ANALYSIS")
    print("="*80)
    
    analyzer = RigorousClusterAnalyzer(
        distance_matrix_path=args.distance_matrix,
        index_path=args.index,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations
    )
    
    print(f"Loaded {len(analyzer.model_names)} models")
    print(f"Available label types: {list(analyzer.labels.keys())}")
    
    # Run analysis for different label types
    label_types = ['training', 'architecture', 'combined']
    
    for label_type in label_types:
        print(f"\n" + "="*60)
        print(f"ANALYZING {label_type.upper()} LABELS")
        print("="*60)
        
        try:
            # Run comprehensive analysis
            results = analyzer.run_comprehensive_analysis(label_type)
            
            # Generate reports
            print("\nGenerating publication report...")
            report = analyzer.create_publication_report(results, label_type)
            
            # Save report
            report_file = analyzer.output_dir / f"cluster_report_{label_type}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"Report saved: {report_file}")
            
            # Save LaTeX table
            latex_table = analyzer.export_latex_table(results, label_type)
            latex_file = analyzer.output_dir / f"cluster_table_{label_type}.tex"
            with open(latex_file, 'w') as f:
                f.write(latex_table)
            print(f"LaTeX table saved: {latex_file}")
            
            # Create visualizations
            print("Creating visualizations...")
            analyzer.create_visualizations(results, label_type)
            
            # Print summary to console
            print("\nSUMMARY RESULTS:")
            print("-" * 40)
            print(f"Silhouette:  {results.silhouette_mean:.3f} [{results.silhouette_ci[0]:.3f}, {results.silhouette_ci[1]:.3f}]")
            print(f"ARI:         {results.ari:.3f} [{results.ari_ci[0]:.3f}, {results.ari_ci[1]:.3f}]")
            print(f"PERMANOVA:   F={results.permanova_f:.2f}, p={results.permanova_p:.2e}, η²={results.permanova_eta2:.3f}")
            print(f"Cohen's d:   {results.cohens_d:.3f}")
            print(f"Margin:      {results.margin:+.3f}")
            print(f"Stability:   {results.jaccard_stability:.3f} [{results.jaccard_stability_ci[0]:.3f}, {results.jaccard_stability_ci[1]:.3f}]")
            
        except Exception as e:
            print(f"Error analyzing {label_type}: {e}")
            continue
    
    print(f"\n✅ Analysis complete! Results saved to: {analyzer.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
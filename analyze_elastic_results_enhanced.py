#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Elastic Distance Analysis for Publication
===================================================

Comprehensive statistical analysis of elastic distance results between
eigenvalue evolution curves for trained and random neural network models.

Features:
- Advanced statistical measures with effect sizes and confidence intervals
- Bootstrap analysis for robust uncertainty quantification  
- Hierarchical clustering with dendrograms
- Publication-quality visualizations
- LaTeX table export for direct paper inclusion
- Multiple hypothesis testing corrections
- ROC/AUC analysis for classification performance
"""

import numpy as np
import json
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd

# Scientific computing
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import bootstrap

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px

# Machine learning
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Configure style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class StatisticalResults:
    """Container for comprehensive statistical analysis results."""
    mean: float
    std: float
    median: float
    mad: float  # Median absolute deviation
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float
    bootstrap_mean: float
    bootstrap_std: float
    q25: float
    q75: float
    iqr: float
    min_val: float
    max_val: float
    n_samples: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for export."""
        return {
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'mad': self.mad,
            'ci_95': (self.ci_95_lower, self.ci_95_upper),
            'ci_99': (self.ci_99_lower, self.ci_99_upper),
            'bootstrap_mean': self.bootstrap_mean,
            'bootstrap_std': self.bootstrap_std,
            'q25': self.q25,
            'q75': self.q75,
            'iqr': self.iqr,
            'range': (self.min_val, self.max_val),
            'n': self.n_samples
        }
    
    def to_latex_row(self, label: str) -> str:
        """Format as LaTeX table row."""
        return (f"{label} & {self.mean:.4f} & {self.std:.4f} & {self.median:.4f} & "
                f"[{self.ci_95_lower:.4f}, {self.ci_95_upper:.4f}] & {self.n_samples} \\\\")


class EnhancedElasticAnalyzer:
    """Enhanced analyzer for elastic distance results with publication-quality outputs."""
    
    def __init__(self, distance_matrix_path: str = "elastic_eigs_distance.npy",
                 index_path: str = "elastic_eigs_index.json",
                 output_dir: str = "elastic_analysis_results"):
        """Initialize analyzer with data paths."""
        self.distance_matrix_path = distance_matrix_path
        self.index_path = index_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.D, self.model_names = self._load_results()
        self.model_indices = self._classify_models()
        self.model_infos = [self._extract_model_info(name) for name in self.model_names]
        
    def _load_results(self) -> Tuple[np.ndarray, List[str]]:
        """Load distance matrix and model index."""
        D = np.load(self.distance_matrix_path)
        with open(self.index_path, "r") as f:
            model_names = json.load(f)
        return D, model_names
    
    def _classify_models(self) -> Dict[str, List[int]]:
        """Classify models by type."""
        trained_indices = []
        random_indices = []
        other_indices = []
        
        for i, name in enumerate(self.model_names):
            name_lower = name.lower()
            
            if 'random' in name_lower:
                random_indices.append(i)
            elif any(x in name_lower for x in ['trained', 'acc']):
                trained_indices.append(i)
            elif 'mnist' in name_lower and 'random' not in name_lower:
                # MNIST models without explicit 'trained' or 'acc' are trained models
                # This handles patterns like tinycnn_mnist1, mlp4layer_mnist_seed42
                trained_indices.append(i)
            else:
                other_indices.append(i)
        
        return {
            'trained': trained_indices,
            'random': random_indices,
            'other': other_indices
        }
    
    def _extract_model_info(self, model_name: str) -> Dict[str, Any]:
        """Extract detailed information from model name."""
        import re
        
        name_lower = model_name.lower()
        info = {
            'name': model_name,
            'type': 'unknown',
            'accuracy': None,
            'epochs': None,
            'architecture': 'unknown',
            'dataset': None,
            'seed': None,
            'variant': None
        }
        
        # Extract accuracy
        acc_match = re.search(r'acc(\d+)', name_lower)
        if acc_match:
            info['accuracy'] = int(acc_match.group(1))
        
        # Extract epochs
        ep_match = re.search(r'ep(\d+)', name_lower)
        if ep_match:
            info['epochs'] = int(ep_match.group(1))
        
        # Extract seed
        seed_match = re.search(r'seed(\d+)', name_lower)
        if seed_match:
            info['seed'] = int(seed_match.group(1))
        
        # Extract variant number (for patterns like mnist1, mnist2, etc.)
        variant_match = re.search(r'mnist(\d+)', name_lower)
        if variant_match:
            info['variant'] = int(variant_match.group(1))
        
        # Determine dataset
        if 'mnist' in name_lower:
            info['dataset'] = 'mnist'
        elif 'cifar' in name_lower:
            info['dataset'] = 'cifar'
        
        # Determine type
        if 'random' in name_lower:
            info['type'] = 'random'
        elif any(x in name_lower for x in ['trained', 'acc']):
            info['type'] = 'trained'
        elif 'mnist' in name_lower and 'random' not in name_lower:
            # MNIST models without explicit 'trained' or 'acc' are trained models
            info['type'] = 'trained'
        
        # Architecture classification
        if 'tinycnn' in name_lower:
            info['architecture'] = 'tinycnn'
        elif 'mlp4layer' in name_lower:
            info['architecture'] = 'mlp4layer'
        elif 'custom' in name_lower:
            info['architecture'] = 'custom'
        elif 'mlp' in name_lower:
            info['architecture'] = 'mlp'
        elif any(x in name_lower for x in ['conv', 'cnn']):
            info['architecture'] = 'conv'
        elif 'resnet' in name_lower:
            info['architecture'] = 'resnet'
        elif 'vgg' in name_lower:
            info['architecture'] = 'vgg'
        
        return info
    
    def compute_comprehensive_stats(self, data: np.ndarray, n_bootstrap: int = 10000) -> StatisticalResults:
        """Compute comprehensive statistics with bootstrap confidence intervals."""
        
        # Basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        median_val = np.median(data)
        mad_val = np.median(np.abs(data - median_val))
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Bootstrap confidence intervals
        rng = np.random.RandomState(42)
        
        def statistic(x, axis):
            return np.mean(x, axis=axis)
        
        res = bootstrap((data,), statistic, n_resamples=n_bootstrap, 
                       confidence_level=0.95, random_state=rng, method='percentile')
        ci_95_lower, ci_95_upper = res.confidence_interval
        
        res_99 = bootstrap((data,), statistic, n_resamples=n_bootstrap,
                          confidence_level=0.99, random_state=rng, method='percentile')
        ci_99_lower, ci_99_upper = res_99.confidence_interval
        
        bootstrap_samples = res.bootstrap_distribution
        bootstrap_mean = np.mean(bootstrap_samples)
        bootstrap_std = np.std(bootstrap_samples)
        
        return StatisticalResults(
            mean=mean_val, std=std_val, median=median_val, mad=mad_val,
            ci_95_lower=ci_95_lower, ci_95_upper=ci_95_upper,
            ci_99_lower=ci_99_lower, ci_99_upper=ci_99_upper,
            bootstrap_mean=bootstrap_mean, bootstrap_std=bootstrap_std,
            q25=q25, q75=q75, iqr=iqr,
            min_val=min_val, max_val=max_val,
            n_samples=len(data)
        )
    
    def calculate_effect_sizes(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """Calculate various effect size measures."""
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Hedge's g (corrected Cohen's d for small samples)
        n1, n2 = len(group1), len(group2)
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohen_d * correction
        
        # Glass's delta (using control group std)
        glass_delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1) if np.std(group2, ddof=1) > 0 else 0
        
        # Common language effect size (probability that random value from group1 > group2)
        from scipy.stats import norm
        cles = norm.cdf(cohen_d / np.sqrt(2))
        
        # Rank-biserial correlation (non-parametric effect size)
        u_stat, _ = stats.mannwhitneyu(group1, group2)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        
        return {
            'cohen_d': cohen_d,
            'hedges_g': hedges_g,
            'glass_delta': glass_delta,
            'cles': cles,
            'rank_biserial': rank_biserial,
            'interpretation': self._interpret_effect_size(cohen_d)
        }
    
    def _interpret_effect_size(self, cohen_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        elif abs_d < 1.2:
            return "large"
        else:
            return "very large"
    
    def perform_statistical_tests(self, group1: np.ndarray, group2: np.ndarray,
                                 group3: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform comprehensive statistical tests with corrections."""
        
        results = {}
        
        # Normality tests
        _, p_norm1 = stats.shapiro(group1) if len(group1) <= 5000 else stats.normaltest(group1)
        _, p_norm2 = stats.shapiro(group2) if len(group2) <= 5000 else stats.normaltest(group2)
        results['normality'] = {'group1': p_norm1, 'group2': p_norm2}
        
        # Parametric tests
        t_stat, p_ttest = stats.ttest_ind(group1, group2)
        results['ttest'] = {'statistic': t_stat, 'p_value': p_ttest}
        
        # Non-parametric tests
        u_stat, p_mann = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        results['mann_whitney'] = {'statistic': u_stat, 'p_value': p_mann}
        
        # Levene's test for variance equality
        _, p_levene = stats.levene(group1, group2)
        results['levene'] = {'p_value': p_levene}
        
        # Welch's t-test (doesn't assume equal variances)
        t_welch, p_welch = stats.ttest_ind(group1, group2, equal_var=False)
        results['welch_ttest'] = {'statistic': t_welch, 'p_value': p_welch}
        
        # If three groups provided, perform ANOVA/Kruskal-Wallis
        if group3 is not None:
            f_stat, p_anova = stats.f_oneway(group1, group2, group3)
            results['anova'] = {'statistic': f_stat, 'p_value': p_anova}
            
            h_stat, p_kruskal = stats.kruskal(group1, group2, group3)
            results['kruskal_wallis'] = {'statistic': h_stat, 'p_value': p_kruskal}
        
        # Permutation test
        def statistic(x, y):
            return np.mean(x) - np.mean(y)
        
        res = stats.permutation_test((group1, group2), statistic, n_resamples=10000,
                                    alternative='two-sided', random_state=42)
        results['permutation'] = {'statistic': res.statistic, 'p_value': res.pvalue}
        
        # Multiple comparison corrections
        p_values = [results['ttest']['p_value'], results['mann_whitney']['p_value'],
                   results['welch_ttest']['p_value'], results['permutation']['p_value']]
        
        # Bonferroni correction
        bonferroni_alpha = 0.05 / len(p_values)
        results['bonferroni'] = {
            'corrected_alpha': bonferroni_alpha,
            'significant': [p < bonferroni_alpha for p in p_values]
        }
        
        # Holm-Bonferroni correction
        from statsmodels.stats.multitest import multipletests
        reject_holm, p_holm, _, _ = multipletests(p_values, method='holm')
        results['holm_bonferroni'] = {
            'corrected_p': p_holm.tolist(),
            'reject': reject_holm.tolist()
        }
        
        # FDR (Benjamini-Hochberg) correction
        reject_fdr, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
        results['fdr'] = {
            'corrected_p': p_fdr.tolist(),
            'reject': reject_fdr.tolist()
        }
        
        return results
    
    def perform_clustering_analysis(self) -> Dict[str, Any]:
        """Perform hierarchical clustering and compute validity indices."""
        
        # Prepare distance matrix for clustering
        condensed = squareform(self.D)
        
        # Hierarchical clustering
        linkage_matrix = linkage(condensed, method='ward')
        
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        for n_clusters in range(2, min(10, len(self.model_names))):
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            score = silhouette_score(self.D, clusters, metric='precomputed')
            silhouette_scores.append((n_clusters, score))
        
        optimal_n = max(silhouette_scores, key=lambda x: x[1])[0]
        optimal_clusters = fcluster(linkage_matrix, optimal_n, criterion='maxclust')
        
        # Compute clustering validity indices
        validity_indices = {
            'silhouette': silhouette_score(self.D, optimal_clusters, metric='precomputed'),
            'calinski_harabasz': calinski_harabasz_score(self.D, optimal_clusters),
            'davies_bouldin': davies_bouldin_score(self.D, optimal_clusters)
        }
        
        # Cluster composition analysis
        cluster_composition = defaultdict(lambda: {'trained': 0, 'random': 0, 'other': 0})
        for i, cluster_id in enumerate(optimal_clusters):
            model_type = self.model_infos[i]['type']
            if model_type == 'trained':
                cluster_composition[cluster_id]['trained'] += 1
            elif model_type == 'random':
                cluster_composition[cluster_id]['random'] += 1
            else:
                cluster_composition[cluster_id]['other'] += 1
        
        return {
            'linkage_matrix': linkage_matrix,
            'optimal_n_clusters': optimal_n,
            'cluster_assignments': optimal_clusters.tolist(),
            'silhouette_scores': silhouette_scores,
            'validity_indices': validity_indices,
            'cluster_composition': dict(cluster_composition)
        }
    
    def perform_roc_analysis(self) -> Dict[str, Any]:
        """Perform ROC analysis for distance-based classification."""
        
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        if len(trained_idx) < 2 or len(random_idx) < 2:
            return {'error': 'Insufficient samples for ROC analysis'}
        
        # Create labels and distances for ROC
        labels = []
        distances = []
        
        # For each model, compute average distance to trained models
        for i in range(len(self.model_names)):
            if i in trained_idx:
                # Average distance to other trained models
                other_trained = [j for j in trained_idx if j != i]
                if other_trained:
                    avg_dist = np.mean([self.D[i, j] for j in other_trained])
                    distances.append(avg_dist)
                    labels.append(1)  # 1 for trained
            elif i in random_idx:
                # Average distance to trained models
                if trained_idx:
                    avg_dist = np.mean([self.D[i, j] for j in trained_idx])
                    distances.append(avg_dist)
                    labels.append(0)  # 0 for random
        
        if len(set(labels)) < 2:
            return {'error': 'Only one class present'}
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, distances)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Classification metrics at optimal threshold
        predictions = (np.array(distances) <= optimal_threshold).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1_score': f1_score(labels, predictions)
        }
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx],
            'classification_metrics': metrics
        }
    
    def create_publication_figures(self):
        """Create all publication-quality figures."""
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf'
        })
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Enhanced distance matrix heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_distance_matrix(ax1)
        
        # 2. Dendrogram
        ax2 = fig.add_subplot(gs[0, 2])
        clustering_results = self.perform_clustering_analysis()
        self._plot_dendrogram(ax2, clustering_results['linkage_matrix'])
        
        # 3. Violin plots with statistics
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_violin_comparison(ax3)
        
        # 4. ROC curve
        ax4 = fig.add_subplot(gs[1, 1])
        roc_results = self.perform_roc_analysis()
        if 'error' not in roc_results:
            self._plot_roc_curve(ax4, roc_results)
        
        # 5. Effect size comparison
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_effect_sizes(ax5)
        
        # 6. MDS projection
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_mds_projection(ax6)
        
        # 7. Bootstrap distributions
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_bootstrap_distributions(ax7)
        
        # 8. Statistical summary table
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_summary_table(ax8)
        
        # Save figure
        output_path = self.output_dir / "elastic_analysis_publication.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.savefig(self.output_dir / "elastic_analysis_publication.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved publication figure to {output_path}")
    
    def _plot_distance_matrix(self, ax):
        """Plot enhanced distance matrix heatmap."""
        
        # Reorder by type
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        other_idx = self.model_indices['other']
        ordered_idx = trained_idx + random_idx + other_idx
        
        D_ordered = self.D[np.ix_(ordered_idx, ordered_idx)]
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(D_ordered), k=1)
        
        # Plot heatmap
        sns.heatmap(D_ordered, mask=mask, cmap='RdBu_r', center=np.median(self.D),
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=np.percentile(self.D, 5), vmax=np.percentile(self.D, 95))
        
        # Add dividing lines
        n_trained = len(trained_idx)
        n_random = len(random_idx)
        ax.axhline(y=n_trained, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=n_trained, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(y=n_trained + n_random, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=n_trained + n_random, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title('Elastic Distance Matrix\n(Lower Triangle: Trained | Random | Other)', fontweight='bold')
        ax.set_xlabel('Model Index')
        ax.set_ylabel('Model Index')
    
    def _plot_dendrogram(self, ax, linkage_matrix):
        """Plot hierarchical clustering dendrogram."""
        
        # Create dendrogram
        dendro = dendrogram(linkage_matrix, ax=ax, orientation='left',
                          no_labels=True, color_threshold=0)
        
        ax.set_title('Hierarchical Clustering\nDendrogram', fontweight='bold')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Model Index')
    
    def _plot_violin_comparison(self, ax):
        """Plot violin plots comparing distance distributions."""
        
        # Get distance groups
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        within_trained = []
        for i in range(len(trained_idx)):
            for j in range(i + 1, len(trained_idx)):
                within_trained.append(self.D[trained_idx[i], trained_idx[j]])
        
        within_random = []
        for i in range(len(random_idx)):
            for j in range(i + 1, len(random_idx)):
                within_random.append(self.D[random_idx[i], random_idx[j]])
        
        cross_distances = []
        for i in trained_idx:
            for j in random_idx:
                cross_distances.append(self.D[i, j])
        
        # Create violin plot
        data = [within_trained, within_random, cross_distances]
        parts = ax.violinplot(data, positions=[1, 2, 3], widths=0.7,
                             showmeans=True, showmedians=True, showextrema=True)
        
        # Customize colors
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Within\nTrained', 'Within\nRandom', 'Cross\n(T vs R)'])
        ax.set_ylabel('Elastic Distance')
        ax.set_title('Distance Distribution\nComparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_roc_curve(self, ax, roc_results):
        """Plot ROC curve."""
        
        ax.plot(roc_results['fpr'], roc_results['tpr'], 
               color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_results["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Mark optimal point
        ax.scatter(roc_results['optimal_fpr'], roc_results['optimal_tpr'],
                  color='red', s=100, marker='o', 
                  label=f'Optimal (thresh={roc_results["optimal_threshold"]:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve\nDistance-based Classification', fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    
    def _plot_effect_sizes(self, ax):
        """Plot effect size comparison."""
        
        # Calculate effect sizes
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        within_trained = []
        for i in range(len(trained_idx)):
            for j in range(i + 1, len(trained_idx)):
                within_trained.append(self.D[trained_idx[i], trained_idx[j]])
        
        cross_distances = []
        for i in trained_idx:
            for j in random_idx:
                cross_distances.append(self.D[i, j])
        
        if within_trained and cross_distances:
            effect_sizes = self.calculate_effect_sizes(
                np.array(cross_distances), np.array(within_trained)
            )
            
            # Create bar plot
            measures = ['Cohen\'s d', 'Hedge\'s g', 'Glass\'s Δ', 'CLES', 'Rank-biserial']
            values = [
                effect_sizes['cohen_d'],
                effect_sizes['hedges_g'],
                effect_sizes['glass_delta'],
                effect_sizes['cles'],
                effect_sizes['rank_biserial']
            ]
            
            bars = ax.bar(range(len(measures)), values, color=['blue', 'green', 'orange', 'red', 'purple'])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add reference lines for Cohen's d interpretation
            ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_ylabel('Effect Size')
            ax.set_title(f'Effect Sizes\n({effect_sizes["interpretation"].title()} Effect)', 
                        fontweight='bold')
            ax.set_xticks(range(len(measures)))
            ax.set_xticklabels(measures, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_mds_projection(self, ax):
        """Plot MDS projection of distance matrix."""
        
        # Compute MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(self.D)
        
        # Plot by type
        for model_type, color in [('trained', 'blue'), ('random', 'red'), ('other', 'gray')]:
            indices = self.model_indices[model_type]
            if indices:
                ax.scatter(coords[indices, 0], coords[indices, 1],
                         c=color, label=model_type.title(), s=50, alpha=0.7)
        
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        ax.set_title('MDS Projection\nof Distance Matrix', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_bootstrap_distributions(self, ax):
        """Plot bootstrap distributions of key metrics."""
        
        # Get distance groups
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        within_trained = []
        for i in range(len(trained_idx)):
            for j in range(i + 1, len(trained_idx)):
                within_trained.append(self.D[trained_idx[i], trained_idx[j]])
        
        cross_distances = []
        for i in trained_idx:
            for j in random_idx:
                cross_distances.append(self.D[i, j])
        
        if within_trained and cross_distances:
            # Bootstrap difference in means
            n_bootstrap = 5000
            rng = np.random.RandomState(42)
            
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                sample_cross = rng.choice(cross_distances, size=len(cross_distances), replace=True)
                sample_within = rng.choice(within_trained, size=len(within_trained), replace=True)
                bootstrap_diffs.append(np.mean(sample_cross) - np.mean(sample_within))
            
            ax.hist(bootstrap_diffs, bins=50, density=True, alpha=0.7, color='steelblue')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            
            # Add confidence intervals
            ci_95 = np.percentile(bootstrap_diffs, [2.5, 97.5])
            ax.axvline(x=ci_95[0], color='orange', linestyle=':', linewidth=1.5)
            ax.axvline(x=ci_95[1], color='orange', linestyle=':', linewidth=1.5)
            
            ax.set_xlabel('Mean Difference (Cross - Within)')
            ax.set_ylabel('Density')
            ax.set_title('Bootstrap Distribution\nMean Difference', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_summary_table(self, ax):
        """Plot statistical summary table."""
        
        ax.axis('off')
        
        # Calculate key metrics
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        within_trained = []
        for i in range(len(trained_idx)):
            for j in range(i + 1, len(trained_idx)):
                within_trained.append(self.D[trained_idx[i], trained_idx[j]])
        
        within_random = []
        for i in range(len(random_idx)):
            for j in range(i + 1, len(random_idx)):
                within_random.append(self.D[random_idx[i], random_idx[j]])
        
        cross_distances = []
        for i in trained_idx:
            for j in random_idx:
                cross_distances.append(self.D[i, j])
        
        if within_trained and cross_distances:
            # Calculate statistics
            margin = np.min(cross_distances) - np.max(within_trained) if within_trained else 0
            sep_ratio = np.mean(cross_distances) / np.mean(within_trained) if within_trained and np.mean(within_trained) > 0 else 0
            
            # Perform statistical test
            _, p_value = stats.mannwhitneyu(cross_distances, within_trained, alternative='greater')
            
            # Create summary text
            summary_data = [
                ['Metric', 'Value'],
                ['', ''],
                ['Dataset Size', f'{len(self.model_names)} models'],
                ['Trained Models', f'{len(trained_idx)}'],
                ['Random Models', f'{len(random_idx)}'],
                ['', ''],
                ['Margin', f'{margin:+.4f}'],
                ['Separation Ratio', f'{sep_ratio:.3f}'],
                ['P-value (Cross > Within)', f'{p_value:.2e}'],
                ['', ''],
                ['Within Trained', f'{np.mean(within_trained):.3f} ± {np.std(within_trained):.3f}'],
                ['Within Random', f'{np.mean(within_random):.3f} ± {np.std(within_random):.3f}'],
                ['Cross Distance', f'{np.mean(cross_distances):.3f} ± {np.std(cross_distances):.3f}'],
            ]
            
            # Create table
            table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Style header row
            for i in range(len(summary_data[0])):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    def export_latex_tables(self):
        """Export results as LaTeX tables for paper inclusion."""
        
        output_file = self.output_dir / "latex_tables.tex"
        
        # Get distance groups
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        within_trained = []
        for i in range(len(trained_idx)):
            for j in range(i + 1, len(trained_idx)):
                within_trained.append(self.D[trained_idx[i], trained_idx[j]])
        
        within_random = []
        for i in range(len(random_idx)):
            for j in range(i + 1, len(random_idx)):
                within_random.append(self.D[random_idx[i], random_idx[j]])
        
        cross_distances = []
        for i in trained_idx:
            for j in random_idx:
                cross_distances.append(self.D[i, j])
        
        if not within_trained or not cross_distances:
            print("❌ Insufficient data for LaTeX export")
            return
        
        # Compute comprehensive stats
        stats_trained = self.compute_comprehensive_stats(np.array(within_trained))
        stats_random = self.compute_comprehensive_stats(np.array(within_random))
        stats_cross = self.compute_comprehensive_stats(np.array(cross_distances))
        
        # Calculate effect sizes
        effect_sizes = self.calculate_effect_sizes(
            np.array(cross_distances), np.array(within_trained)
        )
        
        # Perform statistical tests
        test_results = self.perform_statistical_tests(
            np.array(cross_distances), np.array(within_trained), np.array(within_random)
        )
        
        with open(output_file, 'w') as f:
            # Table 1: Distance Statistics
            f.write("% Table 1: Distance Statistics with Bootstrap Confidence Intervals\n")
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\caption{Elastic Distance Statistics}\n")
            f.write("\\label{tab:distance_stats}\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\toprule\n")
            f.write("Group & Mean & Std & Median & 95\\% CI & N \\\\\n")
            f.write("\\midrule\n")
            f.write(stats_trained.to_latex_row("Within Trained") + "\n")
            f.write(stats_random.to_latex_row("Within Random") + "\n")
            f.write(stats_cross.to_latex_row("Cross (T vs R)") + "\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Table 2: Effect Sizes
            f.write("% Table 2: Effect Size Measures\n")
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\caption{Effect Size Analysis (Cross vs Within Trained)}\n")
            f.write("\\label{tab:effect_sizes}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Measure & Value & Interpretation \\\\\n")
            f.write("\\midrule\n")
            f.write(f"Cohen's $d$ & {effect_sizes['cohen_d']:.3f} & {effect_sizes['interpretation']} \\\\\n")
            f.write(f"Hedge's $g$ & {effect_sizes['hedges_g']:.3f} & - \\\\\n")
            f.write(f"Glass's $\\Delta$ & {effect_sizes['glass_delta']:.3f} & - \\\\\n")
            f.write(f"CLES & {effect_sizes['cles']:.3f} & - \\\\\n")
            f.write(f"Rank-biserial & {effect_sizes['rank_biserial']:.3f} & - \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Table 3: Statistical Tests
            f.write("% Table 3: Statistical Test Results\n")
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\caption{Statistical Significance Tests}\n")
            f.write("\\label{tab:stat_tests}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write("Test & Statistic & P-value & Significant \\\\\n")
            f.write("\\midrule\n")
            
            # Add test results
            f.write(f"Student's t & {test_results['ttest']['statistic']:.3f} & "
                   f"{test_results['ttest']['p_value']:.3e} & "
                   f"{'Yes' if test_results['ttest']['p_value'] < 0.05 else 'No'} \\\\\n")
            
            f.write(f"Welch's t & {test_results['welch_ttest']['statistic']:.3f} & "
                   f"{test_results['welch_ttest']['p_value']:.3e} & "
                   f"{'Yes' if test_results['welch_ttest']['p_value'] < 0.05 else 'No'} \\\\\n")
            
            f.write(f"Mann-Whitney U & {test_results['mann_whitney']['statistic']:.1f} & "
                   f"{test_results['mann_whitney']['p_value']:.3e} & "
                   f"{'Yes' if test_results['mann_whitney']['p_value'] < 0.05 else 'No'} \\\\\n")
            
            f.write(f"Permutation & {test_results['permutation']['statistic']:.3f} & "
                   f"{test_results['permutation']['p_value']:.3e} & "
                   f"{'Yes' if test_results['permutation']['p_value'] < 0.05 else 'No'} \\\\\n")
            
            if 'anova' in test_results:
                f.write(f"ANOVA & {test_results['anova']['statistic']:.3f} & "
                       f"{test_results['anova']['p_value']:.3e} & "
                       f"{'Yes' if test_results['anova']['p_value'] < 0.05 else 'No'} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\multicolumn{4}{l}{\\footnotesize $\\alpha = 0.05$, ")
            f.write(f"Bonferroni corrected $\\alpha = {test_results['bonferroni']['corrected_alpha']:.4f}$")
            f.write("}\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"📝 Exported LaTeX tables to {output_file}")
    
    def export_results_json(self):
        """Export all results to JSON for reproducibility."""
        
        # Gather all results
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        within_trained = []
        for i in range(len(trained_idx)):
            for j in range(i + 1, len(trained_idx)):
                within_trained.append(self.D[trained_idx[i], trained_idx[j]])
        
        within_random = []
        for i in range(len(random_idx)):
            for j in range(i + 1, len(random_idx)):
                within_random.append(self.D[random_idx[i], random_idx[j]])
        
        cross_distances = []
        for i in trained_idx:
            for j in random_idx:
                cross_distances.append(self.D[i, j])
        
        results = {
            'dataset': {
                'n_models': len(self.model_names),
                'n_trained': len(trained_idx),
                'n_random': len(random_idx),
                'n_other': len(self.model_indices['other'])
            },
            'distances': {
                'within_trained': {
                    'values': within_trained,
                    'stats': self.compute_comprehensive_stats(np.array(within_trained)).to_dict() if within_trained else {}
                },
                'within_random': {
                    'values': within_random,
                    'stats': self.compute_comprehensive_stats(np.array(within_random)).to_dict() if within_random else {}
                },
                'cross': {
                    'values': cross_distances,
                    'stats': self.compute_comprehensive_stats(np.array(cross_distances)).to_dict() if cross_distances else {}
                }
            }
        }
        
        if within_trained and cross_distances:
            results['effect_sizes'] = self.calculate_effect_sizes(
                np.array(cross_distances), np.array(within_trained)
            )
            results['statistical_tests'] = self.perform_statistical_tests(
                np.array(cross_distances), np.array(within_trained), np.array(within_random)
            )
            results['clustering'] = self.perform_clustering_analysis()
            results['roc_analysis'] = self.perform_roc_analysis()
        
        # Save to JSON
        output_file = self.output_dir / "analysis_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                # Convert dict keys to strings if they're numpy types
                new_dict = {}
                for k, v in obj.items():
                    if isinstance(k, (np.integer, np.int32, np.int64)):
                        new_dict[str(int(k))] = convert_numpy(v)
                    elif isinstance(k, (np.floating, np.float32, np.float64)):
                        new_dict[str(float(k))] = convert_numpy(v)
                    else:
                        new_dict[str(k)] = convert_numpy(v)
                return new_dict
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            return obj
        
        results = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Exported results to {output_file}")
    
    def create_interactive_visualization(self):
        """Create interactive 3D visualization using plotly."""
        
        # Compute 3D MDS projection
        mds_3d = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        coords_3d = mds_3d.fit_transform(self.D)
        
        # Prepare data for plotly
        model_types = [self.model_infos[i]['type'] for i in range(len(self.model_names))]
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=[{'trained': 0, 'random': 1, 'unknown': 2}[t] for t in model_types],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Model Type",
                    ticktext=["Trained", "Random", "Other"],
                    tickvals=[0, 1, 2]
                ),
                line=dict(width=0.5, color='white')
            ),
            text=[f"{name}<br>Type: {t}" for name, t in zip(self.model_names, model_types)],
            hovertemplate='%{text}<br>MDS1: %{x:.3f}<br>MDS2: %{y:.3f}<br>MDS3: %{z:.3f}'
        )])
        
        fig.update_layout(
            title='3D MDS Projection of Elastic Distance Matrix',
            scene=dict(
                xaxis_title='MDS Dimension 1',
                yaxis_title='MDS Dimension 2',
                zaxis_title='MDS Dimension 3'
            ),
            width=900,
            height=700
        )
        
        output_file = self.output_dir / "interactive_3d_projection.html"
        fig.write_html(str(output_file))
        print(f"🌐 Saved interactive 3D visualization to {output_file}")
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report."""
        
        output_file = self.output_dir / "analysis_report.md"
        
        # Calculate all metrics
        trained_idx = self.model_indices['trained']
        random_idx = self.model_indices['random']
        
        within_trained = []
        for i in range(len(trained_idx)):
            for j in range(i + 1, len(trained_idx)):
                within_trained.append(self.D[trained_idx[i], trained_idx[j]])
        
        within_random = []
        for i in range(len(random_idx)):
            for j in range(i + 1, len(random_idx)):
                within_random.append(self.D[random_idx[i], random_idx[j]])
        
        cross_distances = []
        for i in trained_idx:
            for j in random_idx:
                cross_distances.append(self.D[i, j])
        
        with open(output_file, 'w') as f:
            f.write("# Elastic Distance Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            
            if within_trained and cross_distances:
                margin = np.min(cross_distances) - np.max(within_trained)
                sep_ratio = np.mean(cross_distances) / np.mean(within_trained) if np.mean(within_trained) > 0 else 0
                
                if margin > 0:
                    f.write("✅ **SUCCESS**: The elastic distance metric successfully separates trained from random models.\n\n")
                    f.write(f"- **Positive margin achieved**: {margin:+.4f}\n")
                    f.write(f"- **Separation ratio**: {sep_ratio:.3f} (values > 1.0 indicate separation)\n")
                else:
                    f.write("⚠️ **PARTIAL SUCCESS**: The elastic distance shows some separation between model types.\n\n")
                    f.write(f"- **Margin**: {margin:+.4f} (negative indicates overlap)\n")
                    f.write(f"- **Separation ratio**: {sep_ratio:.3f}\n")
                
                f.write("\n## Dataset Overview\n\n")
                f.write(f"- Total models analyzed: **{len(self.model_names)}**\n")
                f.write(f"- Trained models: **{len(trained_idx)}**\n")
                f.write(f"- Random models: **{len(random_idx)}**\n")
                f.write(f"- Other models: **{len(self.model_indices['other'])}**\n")
                
                f.write("\n## Statistical Analysis\n\n")
                f.write("### Distance Statistics\n\n")
                
                stats_trained = self.compute_comprehensive_stats(np.array(within_trained))
                stats_cross = self.compute_comprehensive_stats(np.array(cross_distances))
                
                f.write("| Metric | Within Trained | Cross (T vs R) | Difference |\n")
                f.write("|--------|---------------|----------------|------------|\n")
                f.write(f"| Mean | {stats_trained.mean:.4f} | {stats_cross.mean:.4f} | "
                       f"{stats_cross.mean - stats_trained.mean:+.4f} |\n")
                f.write(f"| Std | {stats_trained.std:.4f} | {stats_cross.std:.4f} | "
                       f"{stats_cross.std - stats_trained.std:+.4f} |\n")
                f.write(f"| Median | {stats_trained.median:.4f} | {stats_cross.median:.4f} | "
                       f"{stats_cross.median - stats_trained.median:+.4f} |\n")
                f.write(f"| 95% CI | [{stats_trained.ci_95_lower:.4f}, {stats_trained.ci_95_upper:.4f}] | "
                       f"[{stats_cross.ci_95_lower:.4f}, {stats_cross.ci_95_upper:.4f}] | - |\n")
                
                f.write("\n### Effect Sizes\n\n")
                effect_sizes = self.calculate_effect_sizes(np.array(cross_distances), np.array(within_trained))
                
                f.write(f"- **Cohen's d**: {effect_sizes['cohen_d']:.3f} ({effect_sizes['interpretation']})\n")
                f.write(f"- **Hedge's g**: {effect_sizes['hedges_g']:.3f}\n")
                f.write(f"- **CLES**: {effect_sizes['cles']:.3f} (probability that a random cross distance > within distance)\n")
                
                f.write("\n### Statistical Significance\n\n")
                test_results = self.perform_statistical_tests(
                    np.array(cross_distances), np.array(within_trained)
                )
                
                f.write("| Test | P-value | Significant (α=0.05) |\n")
                f.write("|------|---------|---------------------|\n")
                f.write(f"| Mann-Whitney U | {test_results['mann_whitney']['p_value']:.3e} | "
                       f"{'Yes' if test_results['mann_whitney']['p_value'] < 0.05 else 'No'} |\n")
                f.write(f"| Welch's t-test | {test_results['welch_ttest']['p_value']:.3e} | "
                       f"{'Yes' if test_results['welch_ttest']['p_value'] < 0.05 else 'No'} |\n")
                f.write(f"| Permutation test | {test_results['permutation']['p_value']:.3e} | "
                       f"{'Yes' if test_results['permutation']['p_value'] < 0.05 else 'No'} |\n")
                
                f.write("\n## Clustering Analysis\n\n")
                clustering = self.perform_clustering_analysis()
                
                f.write(f"- **Optimal number of clusters**: {clustering['optimal_n_clusters']}\n")
                f.write(f"- **Silhouette score**: {clustering['validity_indices']['silhouette']:.3f}\n")
                f.write(f"- **Davies-Bouldin index**: {clustering['validity_indices']['davies_bouldin']:.3f}\n")
                
                f.write("\n### Cluster Composition\n\n")
                f.write("| Cluster | Trained | Random | Other |\n")
                f.write("|---------|---------|--------|-------|\n")
                for cluster_id, composition in clustering['cluster_composition'].items():
                    f.write(f"| {cluster_id} | {composition['trained']} | "
                           f"{composition['random']} | {composition['other']} |\n")
                
                f.write("\n## Classification Performance\n\n")
                roc_results = self.perform_roc_analysis()
                if 'error' not in roc_results:
                    f.write(f"- **AUC-ROC**: {roc_results['auc']:.3f}\n")
                    f.write(f"- **Optimal threshold**: {roc_results['optimal_threshold']:.3f}\n")
                    f.write(f"- **Accuracy at optimal threshold**: "
                           f"{roc_results['classification_metrics']['accuracy']:.3f}\n")
                    f.write(f"- **F1 score**: {roc_results['classification_metrics']['f1_score']:.3f}\n")
                
                f.write("\n## Conclusions\n\n")
                f.write("The elastic distance metric based on eigenvalue evolution curves ")
                if margin > 0:
                    f.write("**successfully distinguishes** between trained and random neural networks. ")
                    f.write("This validates the hypothesis that functional similarity in neural networks ")
                    f.write("can be detected through spectral analysis of their eigenvalue trajectories.\n")
                else:
                    f.write("shows **promising separation** between trained and random models, ")
                    f.write("though complete separation was not achieved. Further refinement of ")
                    f.write("the distance metric or feature extraction may improve performance.\n")
        
        print(f"📄 Generated markdown report at {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        
        print("="*80)
        print("ENHANCED ELASTIC DISTANCE ANALYSIS FOR PUBLICATION")
        print("="*80)
        
        print(f"\n📊 Dataset: {len(self.model_names)} models")
        print(f"  - Trained: {len(self.model_indices['trained'])}")
        print(f"  - Random: {len(self.model_indices['random'])}")
        print(f"  - Other: {len(self.model_indices['other'])}")
        
        # Create visualizations
        print("\n🎨 Creating publication-quality figures...")
        self.create_publication_figures()
        
        # Export LaTeX tables
        print("\n📝 Exporting LaTeX tables...")
        self.export_latex_tables()
        
        # Export JSON results
        print("\n💾 Exporting JSON results...")
        self.export_results_json()
        
        # Create interactive visualization
        print("\n🌐 Creating interactive 3D visualization...")
        self.create_interactive_visualization()
        
        # Generate markdown report
        print("\n📄 Generating markdown report...")
        self.generate_markdown_report()
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}/")
        print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced elastic distance analysis for publication"
    )
    parser.add_argument(
        '--distance-matrix', type=str, default='elastic_eigs_distance.npy',
        help='Path to distance matrix .npy file'
    )
    parser.add_argument(
        '--index', type=str, default='elastic_eigs_index.json',
        help='Path to model index .json file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='elastic_analysis_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = EnhancedElasticAnalyzer(
        distance_matrix_path=args.distance_matrix,
        index_path=args.index,
        output_dir=args.output_dir
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Compute cluster separation metrics between random and trained models.

This script analyzes the separation between random and trained neural network models
using distance matrices from various metrics (elastic, L2, Wasserstein, DTW).

Instead of clustering algorithms, it directly computes separation metrics:
- Intra-cluster vs inter-cluster distances
- Separation ratios and statistical significance
- Distribution overlap analysis
- Comprehensive visualizations

Usage:
    python compute_cluster_separation.py \
      --distance-files elastic_eigs_optimal_distance.npy comparison_l2_distance.npy \
      --index-files elastic_eigs_optimal_index.json comparison_l2_index.json \
      --output-prefix separation_analysis
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import MDS
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_labels_from_filenames(filenames: List[str]) -> Tuple[List[str], List[int], Dict[str, List[int]]]:
    """Extract training labels and indices from filenames."""
    labels = []
    indices = {'random': [], 'trained': []}

    for i, filename in enumerate(filenames):
        filename_lower = filename.lower()
        if 'random' in filename_lower:
            label = 'random'
        elif 'trained' in filename_lower or 'seed' in filename_lower or 'acc' in filename_lower:
            label = 'trained'
        else:
            label = 'unknown'

        labels.append(label)
        if label in indices:
            indices[label].append(i)

    # Convert to numeric labels
    label_map = {'random': 0, 'trained': 1, 'unknown': -1}
    numeric_labels = [label_map[label] for label in labels]

    return labels, numeric_labels, indices


def compute_distance_statistics(distance_matrix: np.ndarray, indices: Dict[str, List[int]]) -> Dict[str, Any]:
    """Compute intra and inter cluster distance statistics."""

    random_idx = indices['random']
    trained_idx = indices['trained']

    if len(random_idx) == 0 or len(trained_idx) == 0:
        raise ValueError("Need both random and trained samples for separation analysis")

    # Intra-cluster distances (within groups)
    random_intra = []
    for i in range(len(random_idx)):
        for j in range(i + 1, len(random_idx)):
            random_intra.append(distance_matrix[random_idx[i], random_idx[j]])

    trained_intra = []
    for i in range(len(trained_idx)):
        for j in range(i + 1, len(trained_idx)):
            trained_intra.append(distance_matrix[trained_idx[i], trained_idx[j]])

    # Inter-cluster distances (between groups)
    inter_distances = []
    for i in random_idx:
        for j in trained_idx:
            inter_distances.append(distance_matrix[i, j])

    # Convert to numpy arrays
    random_intra = np.array(random_intra)
    trained_intra = np.array(trained_intra)
    inter_distances = np.array(inter_distances)

    # Combined intra-cluster distances
    all_intra = np.concatenate([random_intra, trained_intra])

    return {
        'random_intra': random_intra,
        'trained_intra': trained_intra,
        'all_intra': all_intra,
        'inter': inter_distances,
        'n_random': len(random_idx),
        'n_trained': len(trained_idx)
    }


def compute_separation_metrics(stats: Dict[str, Any]) -> Dict[str, float]:
    """Compute various cluster separation metrics."""

    random_intra = stats['random_intra']
    trained_intra = stats['trained_intra']
    all_intra = stats['all_intra']
    inter = stats['inter']

    metrics = {}

    # Basic statistics
    metrics['random_intra_mean'] = float(np.mean(random_intra))
    metrics['random_intra_std'] = float(np.std(random_intra))
    metrics['trained_intra_mean'] = float(np.mean(trained_intra))
    metrics['trained_intra_std'] = float(np.std(trained_intra))
    metrics['all_intra_mean'] = float(np.mean(all_intra))
    metrics['all_intra_std'] = float(np.std(all_intra))
    metrics['inter_mean'] = float(np.mean(inter))
    metrics['inter_std'] = float(np.std(inter))

    # Separation ratios
    if metrics['all_intra_mean'] > 0:
        metrics['separation_ratio'] = metrics['inter_mean'] / metrics['all_intra_mean']
    else:
        metrics['separation_ratio'] = np.inf

    # Silhouette-like coefficient
    # Silhouette = (b - a) / max(a, b) where a=intra, b=inter
    a = metrics['all_intra_mean']
    b = metrics['inter_mean']
    if max(a, b) > 0:
        metrics['silhouette_like'] = (b - a) / max(a, b)
    else:
        metrics['silhouette_like'] = 0.0

    # Davies-Bouldin like index (lower is better)
    # DB = (sigma_intra_1 + sigma_intra_2) / distance_between_centers
    if metrics['inter_mean'] > 0:
        metrics['davies_bouldin_like'] = (metrics['random_intra_std'] + metrics['trained_intra_std']) / metrics['inter_mean']
    else:
        metrics['davies_bouldin_like'] = np.inf

    # Statistical tests
    if len(all_intra) > 0 and len(inter) > 0:
        try:
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(inter, all_intra, alternative='greater')
            metrics['mann_whitney_u_statistic'] = float(statistic)
            metrics['mann_whitney_u_pvalue'] = float(p_value)
        except Exception:
            metrics['mann_whitney_u_statistic'] = np.nan
            metrics['mann_whitney_u_pvalue'] = 1.0

        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(inter) - 1) * np.var(inter) + (len(all_intra) - 1) * np.var(all_intra)) /
                           (len(inter) + len(all_intra) - 2))
        if pooled_std > 0:
            metrics['cohens_d'] = (np.mean(inter) - np.mean(all_intra)) / pooled_std
        else:
            metrics['cohens_d'] = 0.0

    # Percentile analysis
    if len(all_intra) > 0 and len(inter) > 0:
        intra_95th = np.percentile(all_intra, 95)
        metrics['intra_95th_percentile'] = float(intra_95th)
        metrics['inter_above_intra_95th_pct'] = float(np.mean(inter > intra_95th) * 100)

        # Overlap coefficient (area under minimum of two distributions)
        # Approximate using histograms
        all_distances = np.concatenate([all_intra, inter])
        hist_range = (np.min(all_distances), np.max(all_distances))
        bins = 50

        intra_hist, bin_edges = np.histogram(all_intra, bins=bins, range=hist_range, density=True)
        inter_hist, _ = np.histogram(inter, bins=bins, range=hist_range, density=True)

        # Normalize to get probabilities
        bin_width = bin_edges[1] - bin_edges[0]
        intra_prob = intra_hist * bin_width
        inter_prob = inter_hist * bin_width

        # Overlap coefficient = sum of minimum probabilities
        overlap = np.sum(np.minimum(intra_prob, inter_prob))
        metrics['distribution_overlap'] = float(overlap)

    return metrics


def create_visualization(distance_matrix: np.ndarray, filenames: List[str],
                        labels: List[str], stats: Dict[str, Any],
                        metrics: Dict[str, float], metric_name: str,
                        output_file: str):
    """Create comprehensive visualization of cluster separation."""

    fig = plt.figure(figsize=(16, 12))

    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Distance matrix heatmap with cluster annotations
    ax1 = fig.add_subplot(gs[0, 0])

    # Reorder matrix to group by clusters
    random_idx = [i for i, label in enumerate(labels) if label == 'random']
    trained_idx = [i for i, label in enumerate(labels) if label == 'trained']
    reorder_idx = random_idx + trained_idx

    reordered_matrix = distance_matrix[np.ix_(reorder_idx, reorder_idx)]
    im = ax1.imshow(reordered_matrix, cmap='viridis', aspect='auto')

    # Add cluster boundaries
    boundary = len(random_idx)
    ax1.axhline(y=boundary - 0.5, color='red', linewidth=2, linestyle='--')
    ax1.axvline(x=boundary - 0.5, color='red', linewidth=2, linestyle='--')

    ax1.set_title(f'{metric_name} Distance Matrix\n(Reordered by Clusters)')
    ax1.set_xlabel('Model Index')
    ax1.set_ylabel('Model Index')
    plt.colorbar(im, ax=ax1, shrink=0.6)

    # Add cluster labels
    ax1.text(len(random_idx)//2, -1, 'Random', ha='center', fontweight='bold')
    ax1.text(len(random_idx) + len(trained_idx)//2, -1, 'Trained', ha='center', fontweight='bold')
    ax1.text(-1, len(random_idx)//2, 'Random', ha='center', va='center', rotation=90, fontweight='bold')
    ax1.text(-1, len(random_idx) + len(trained_idx)//2, 'Trained', ha='center', va='center', rotation=90, fontweight='bold')

    # 2. Distance distribution histograms
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.hist(stats['all_intra'], bins=20, alpha=0.6, label='Intra-cluster', color='blue', density=True)
    ax2.hist(stats['inter'], bins=20, alpha=0.6, label='Inter-cluster', color='orange', density=True)
    ax2.axvline(metrics['all_intra_mean'], color='blue', linestyle='--', label=f"Intra mean: {metrics['all_intra_mean']:.3f}")
    ax2.axvline(metrics['inter_mean'], color='orange', linestyle='--', label=f"Inter mean: {metrics['inter_mean']:.3f}")

    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Density')
    ax2.set_title('Distance Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Box plot comparison
    ax3 = fig.add_subplot(gs[0, 2])

    box_data = [stats['random_intra'], stats['trained_intra'], stats['inter']]
    box_labels = ['Random\nIntra', 'Trained\nIntra', 'Inter-cluster']

    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax3.set_ylabel('Distance')
    ax3.set_title('Distance Distributions')
    ax3.grid(True, alpha=0.3)

    # 4. MDS projection
    ax4 = fig.add_subplot(gs[1, :])

    # Compute MDS embedding
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=1000)
        coords = mds.fit_transform(distance_matrix)

        # Plot points colored by cluster
        random_coords = coords[[i for i, label in enumerate(labels) if label == 'random']]
        trained_coords = coords[[i for i, label in enumerate(labels) if label == 'trained']]

        ax4.scatter(random_coords[:, 0], random_coords[:, 1],
                   c='blue', label='Random', s=100, alpha=0.7, marker='o')
        ax4.scatter(trained_coords[:, 0], trained_coords[:, 1],
                   c='orange', label='Trained', s=100, alpha=0.7, marker='s')

        ax4.set_xlabel('MDS Dimension 1')
        ax4.set_ylabel('MDS Dimension 2')
        ax4.set_title(f'MDS Projection - {metric_name} Distances')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add stress value if available
        if hasattr(mds, 'stress_'):
            ax4.text(0.02, 0.98, f'Stress: {mds.stress_:.3f}',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception as e:
        ax4.text(0.5, 0.5, f'MDS failed: {str(e)}', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'MDS Projection - {metric_name} Distances (Failed)')

    # 5. Metrics summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Create metrics table
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Separation Ratio', f"{metrics['separation_ratio']:.3f}", 'Higher = Better separation'],
        ['Silhouette-like', f"{metrics['silhouette_like']:.3f}", 'Higher = Better separation'],
        ['Davies-Bouldin-like', f"{metrics['davies_bouldin_like']:.3f}", 'Lower = Better separation'],
        ["Cohen's d", f"{metrics['cohens_d']:.3f}", 'Effect size (>0.8 = large)'],
        ['Mann-Whitney p-value', f"{metrics['mann_whitney_u_pvalue']:.2e}", 'Significance of separation'],
        ['Inter > Intra 95th %', f"{metrics['inter_above_intra_95th_pct']:.1f}%", '% inter-cluster above 95th percentile'],
        ['Distribution Overlap', f"{metrics['distribution_overlap']:.3f}", 'Lower = Better separation']
    ]

    # Create table
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='left', loc='center', colWidths=[0.3, 0.2, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 1:  # Value column
                    cell.set_facecolor('#f0f0f0')
                cell.set_text_props(wrap=True)

    plt.suptitle(f'Cluster Separation Analysis - {metric_name}', fontsize=16, fontweight='bold')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_distance_metric(distance_file: str, index_file: str, metric_name: str,
                          output_prefix: str) -> Dict[str, Any]:
    """Analyze separation for a single distance metric."""

    print(f"\n{'='*60}")
    print(f"Analyzing {metric_name.upper()} separation")
    print(f"{'='*60}")

    # Load data
    print(f"Loading distance matrix from {distance_file}")
    distance_matrix = np.load(distance_file)

    print(f"Loading file index from {index_file}")
    with open(index_file, 'r') as f:
        filenames = json.load(f)

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Number of files: {len(filenames)}")

    # Extract labels
    labels, numeric_labels, indices = extract_labels_from_filenames(filenames)

    print(f"Random models: {len(indices['random'])}")
    print(f"Trained models: {len(indices['trained'])}")

    if len(indices['random']) == 0 or len(indices['trained']) == 0:
        raise ValueError(f"Need both random and trained models for analysis. Found: {indices}")

    # Compute statistics
    stats = compute_distance_statistics(distance_matrix, indices)

    # Compute metrics
    metrics = compute_separation_metrics(stats)

    # Print results
    print(f"\nSeparation Metrics for {metric_name}:")
    print(f"  Intra-cluster mean: {metrics['all_intra_mean']:.4f} ± {metrics['all_intra_std']:.4f}")
    print(f"  Inter-cluster mean: {metrics['inter_mean']:.4f} ± {metrics['inter_std']:.4f}")
    print(f"  Separation ratio: {metrics['separation_ratio']:.4f}")
    print(f"  Silhouette-like: {metrics['silhouette_like']:.4f}")
    print(f"  Cohen's d: {metrics['cohens_d']:.4f}")
    print(f"  Mann-Whitney p-value: {metrics['mann_whitney_u_pvalue']:.2e}")
    print(f"  Inter > Intra 95th %: {metrics['inter_above_intra_95th_pct']:.1f}%")

    # Create visualization
    viz_file = f"{output_prefix}_{metric_name}_separation.png"
    create_visualization(distance_matrix, filenames, labels, stats, metrics,
                        metric_name, viz_file)
    print(f"Visualization saved to {viz_file}")

    return {
        'metric_name': metric_name,
        'metrics': metrics,
        'stats': {
            'n_random': int(stats['n_random']),
            'n_trained': int(stats['n_trained']),
            'n_intra_random': len(stats['random_intra']),
            'n_intra_trained': len(stats['trained_intra']),
            'n_inter': len(stats['inter'])
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Compute cluster separation metrics between random and trained models')

    parser.add_argument('--distance-files', nargs='+', required=True,
                       help='Paths to distance matrix files (.npy)')
    parser.add_argument('--index-files', nargs='+', required=True,
                       help='Paths to corresponding index files (.json)')
    parser.add_argument('--metric-names', nargs='+',
                       help='Names for each distance metric (optional)')
    parser.add_argument('--output-prefix', default='separation_analysis',
                       help='Prefix for output files')

    args = parser.parse_args()

    if len(args.distance_files) != len(args.index_files):
        raise ValueError("Number of distance files must match number of index files")

    # Default metric names
    if args.metric_names is None:
        args.metric_names = [f"Metric_{i+1}" for i in range(len(args.distance_files))]
    elif len(args.metric_names) != len(args.distance_files):
        raise ValueError("Number of metric names must match number of distance files")

    # Analyze each distance metric
    results = []
    for dist_file, idx_file, metric_name in zip(args.distance_files, args.index_files, args.metric_names):
        try:
            result = analyze_distance_metric(dist_file, idx_file, metric_name, args.output_prefix)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {metric_name}: {e}")
            continue

    if not results:
        raise SystemExit("No metrics could be analyzed successfully")

    # Create comparison summary
    print(f"\n{'='*80}")
    print("SEPARATION COMPARISON SUMMARY")
    print(f"{'='*80}")

    comparison_data = []
    for result in results:
        metrics = result['metrics']
        comparison_data.append({
            'Metric': result['metric_name'],
            'Separation_Ratio': f"{metrics['separation_ratio']:.3f}",
            'Silhouette_Like': f"{metrics['silhouette_like']:.3f}",
            'Cohens_d': f"{metrics['cohens_d']:.3f}",
            'Mann_Whitney_p': f"{metrics['mann_whitney_u_pvalue']:.2e}",
            'Inter_Above_95th_Pct': f"{metrics['inter_above_intra_95th_pct']:.1f}%"
        })

    # Print comparison table
    df = pd.DataFrame(comparison_data)
    print("\nComparison Table:")
    print(df.to_string(index=False))

    # Save results
    results_file = f"{args.output_prefix}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")

    # Save comparison CSV
    csv_file = f"{args.output_prefix}_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"Comparison table saved to {csv_file}")

    # Find best performing metric
    best_metric = None
    best_separation_ratio = 0

    for result in results:
        sep_ratio = result['metrics']['separation_ratio']
        if sep_ratio > best_separation_ratio:
            best_separation_ratio = sep_ratio
            best_metric = result['metric_name']

    print(f"\nBest separation metric: {best_metric} (ratio: {best_separation_ratio:.3f})")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Compute ARI (Adjusted Rand Index) from elastic distance matrix.

This script:
- Loads the distance matrix and file index from elastic_mean_eigs_distance.py output
- Extracts ground truth labels from filenames (hourglass vs pyramid architectures)
- Performs hierarchical clustering on the distance matrix
- Computes ARI between cluster assignments and ground truth labels
- Tries different numbers of clusters to find optimal clustering
- Outputs clustering quality metrics

Usage:
    python compute_elastic_ari.py --distance-file elastic_eigs_optimal_distance.npy
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re

from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns


def extract_ground_truth_labels(filenames: List[str]) -> Tuple[List[str], List[int]]:
    """Extract ground truth architecture labels from filenames."""
    labels = []
    label_map = {}
    label_counter = 0

    for filename in filenames:
        filename_lower = filename.lower()
        if 'hourglass' in filename_lower:
            arch = 'hourglass'
        elif 'pyramid' in filename_lower:
            arch = 'pyramid'
        else:
            # Fallback: try to extract from filename pattern
            arch = 'unknown'

        if arch not in label_map:
            label_map[arch] = label_counter
            label_counter += 1

        labels.append(arch)

    # Convert to numeric labels
    numeric_labels = [label_map[label] for label in labels]

    return labels, numeric_labels


def perform_clustering(distance_matrix: np.ndarray, n_clusters: int, method: str = 'ward') -> np.ndarray:
    """Perform hierarchical clustering on distance matrix."""
    if method == 'ward':
        # Ward linkage requires Euclidean distances, but we can use it with precomputed
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        # Convert distance to similarity for ward (not ideal, but workable)
        similarity_matrix = np.max(distance_matrix) - distance_matrix
        cluster_labels = clustering.fit_predict(similarity_matrix)
    elif method == 'complete':
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='complete',
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
    elif method == 'average':
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='average',
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
    elif method == 'spectral':
        # Convert distance to similarity matrix
        similarity_matrix = np.exp(-distance_matrix / np.std(distance_matrix))
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        cluster_labels = clustering.fit_predict(similarity_matrix)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return cluster_labels


def compute_clustering_metrics(distance_matrix: np.ndarray, true_labels: List[int],
                             cluster_labels: np.ndarray) -> Dict[str, float]:
    """Compute various clustering quality metrics."""
    metrics = {}

    # ARI
    metrics['ari'] = adjusted_rand_score(true_labels, cluster_labels)

    # Silhouette score (need to handle precomputed distances)
    try:
        # Convert distance to similarity for silhouette score
        similarity_matrix = np.max(distance_matrix) - distance_matrix
        np.fill_diagonal(similarity_matrix, 0)  # Ensure diagonal is 0
        metrics['silhouette'] = silhouette_score(similarity_matrix, cluster_labels, metric='precomputed')
    except:
        metrics['silhouette'] = np.nan

    return metrics


def plot_clustering_results(distance_matrix: np.ndarray, true_labels: List[str],
                          cluster_labels: np.ndarray, output_file: str):
    """Create visualization of clustering results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Distance matrix heatmap
    im1 = ax1.imshow(distance_matrix, cmap='viridis')
    ax1.set_title('Elastic Distance Matrix')
    plt.colorbar(im1, ax=ax1)

    # 2. MDS projection colored by true labels
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)

    unique_true_labels = list(set(true_labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_true_labels)))
    for i, label in enumerate(unique_true_labels):
        mask = [tl == label for tl in true_labels]
        ax2.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=f'True: {label}', s=50)
    ax2.set_title('MDS Projection (True Labels)')
    ax2.legend()

    # 3. MDS projection colored by cluster labels
    unique_cluster_labels = list(set(cluster_labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cluster_labels)))
    for i, label in enumerate(unique_cluster_labels):
        mask = cluster_labels == label
        ax3.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=f'Cluster {label}', s=50)
    ax3.set_title('MDS Projection (Cluster Labels)')
    ax3.legend()

    # 4. Confusion matrix
    true_labels_mapped = [unique_true_labels.index(tl) for tl in true_labels]
    cm = confusion_matrix(true_labels_mapped, cluster_labels)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax4, cmap='Blues')
    ax4.set_title('Confusion Matrix')
    ax4.set_xlabel('Predicted Cluster')
    ax4.set_ylabel('True Architecture')

    # Set y-axis labels properly
    ax4.set_yticks(range(len(unique_true_labels)))
    ax4.set_yticklabels(unique_true_labels)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute ARI from elastic distance matrix')
    parser.add_argument('--distance-file', default='elastic_eigs_optimal_distance.npy',
                       help='Path to distance matrix (.npy file)')
    parser.add_argument('--index-file', default='elastic_eigs_optimal_index.json',
                       help='Path to file index (.json file)')
    parser.add_argument('--method', default='complete', choices=['ward', 'complete', 'average', 'spectral'],
                       help='Clustering method to use')
    parser.add_argument('--max-clusters', type=int, default=5,
                       help='Maximum number of clusters to try')
    parser.add_argument('--plot-output', default='clustering_results.png',
                       help='Output file for clustering visualization')

    args = parser.parse_args()

    # Load distance matrix
    print(f"Loading distance matrix from {args.distance_file}")
    distance_matrix = np.load(args.distance_file)
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # Load file index
    print(f"Loading file index from {args.index_file}")
    with open(args.index_file, 'r') as f:
        filenames = json.load(f)
    print(f"Number of files: {len(filenames)}")

    # Extract ground truth labels
    print("Extracting ground truth labels from filenames...")
    true_labels, true_labels_numeric = extract_ground_truth_labels(filenames)

    # Print label distribution
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    print("Ground truth label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")

    # Try different numbers of clusters
    print(f"\nTrying clustering with method '{args.method}':")
    best_ari = -1
    best_n_clusters = 2
    best_cluster_labels = None
    results = []

    for n_clusters in range(2, min(args.max_clusters + 1, len(set(true_labels)) + 3)):
        print(f"\nClustering with {n_clusters} clusters...")

        cluster_labels = perform_clustering(distance_matrix, n_clusters, args.method)
        metrics = compute_clustering_metrics(distance_matrix, true_labels_numeric, cluster_labels)

        print(f"  ARI: {metrics['ari']:.4f}")
        print(f"  Silhouette: {metrics['silhouette']:.4f}")

        results.append({
            'n_clusters': n_clusters,
            'ari': metrics['ari'],
            'silhouette': metrics['silhouette'],
            'cluster_labels': cluster_labels
        })

        if metrics['ari'] > best_ari:
            best_ari = metrics['ari']
            best_n_clusters = n_clusters
            best_cluster_labels = cluster_labels

    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Best ARI: {best_ari:.4f} (with {best_n_clusters} clusters)")

    # Show confusion matrix for best clustering
    print(f"\nConfusion Matrix (Best Clustering - {best_n_clusters} clusters):")
    cm = confusion_matrix(true_labels_numeric, best_cluster_labels)
    print(cm)

    # Create detailed results table
    print(f"\nDetailed Results:")
    print(f"{'N_Clusters':<12} {'ARI':<8} {'Silhouette':<12}")
    print("-" * 32)
    for result in results:
        print(f"{result['n_clusters']:<12} {result['ari']:<8.4f} {result['silhouette']:<12.4f}")

    # Create visualization
    print(f"\nCreating visualization: {args.plot_output}")
    plot_clustering_results(distance_matrix, true_labels, best_cluster_labels, args.plot_output)

    # Save results
    results_file = args.distance_file.replace('.npy', '_ari_results.json')
    results_data = {
        'best_ari': float(best_ari),
        'best_n_clusters': int(best_n_clusters),
        'method': args.method,
        'all_results': [
            {
                'n_clusters': int(r['n_clusters']),
                'ari': float(r['ari']),
                'silhouette': float(r['silhouette'])
            }
            for r in results
        ],
        'confusion_matrix': cm.tolist(),
        'ground_truth_distribution': {str(label): int(count) for label, count in zip(unique_labels, counts)}
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
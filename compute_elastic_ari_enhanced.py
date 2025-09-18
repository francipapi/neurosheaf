#!/usr/bin/env python3
"""
Enhanced ARI computation with trained/random separation.

This script:
- Loads the distance matrix and file index from elastic_mean_eigs_distance.py output
- Extracts multiple label schemes: architecture, training status, and combined
- Performs hierarchical clustering on the distance matrix
- Computes ARI for each labeling scheme
- Generates comprehensive visualizations and reports

Usage:
    python compute_elastic_ari_enhanced.py --distance-file elastic_eigs_optimal_distance.npy
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import re

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns


def extract_enhanced_labels(filenames: List[str]) -> Dict[str, Any]:
    """Extract multiple labeling schemes from filenames."""

    # Initialize label lists
    architecture_labels = []
    training_labels = []
    combined_labels = []

    # Label mappings
    arch_map = {}
    train_map = {}
    combined_map = {}

    arch_counter = 0
    train_counter = 0
    combined_counter = 0

    for filename in filenames:
        filename_lower = filename.lower()

        # Extract architecture
        if 'hourglass' in filename_lower:
            arch = 'hourglass'
        elif 'pyramid' in filename_lower:
            arch = 'pyramid'
        elif 'mlp' in filename_lower:
            arch = 'mlp'
        elif 'custom' in filename_lower:
            arch = 'custom'
        else:
            arch = 'unknown'

        # Extract training status
        if 'random' in filename_lower:
            train = 'random'
        elif 'seed' in filename_lower or 'trained' in filename_lower or 'acc' in filename_lower:
            train = 'trained'
        else:
            train = 'unknown'

        # Combined label
        combined = f"{arch}_{train}"

        # Map to numeric labels
        if arch not in arch_map:
            arch_map[arch] = arch_counter
            arch_counter += 1

        if train not in train_map:
            train_map[train] = train_counter
            train_counter += 1

        if combined not in combined_map:
            combined_map[combined] = combined_counter
            combined_counter += 1

        architecture_labels.append(arch)
        training_labels.append(train)
        combined_labels.append(combined)

    # Convert to numeric
    arch_numeric = [arch_map[label] for label in architecture_labels]
    train_numeric = [train_map[label] for label in training_labels]
    combined_numeric = [combined_map[label] for label in combined_labels]

    return {
        'architecture': {
            'labels': architecture_labels,
            'numeric': arch_numeric,
            'map': arch_map,
            'unique': list(arch_map.keys())
        },
        'training': {
            'labels': training_labels,
            'numeric': train_numeric,
            'map': train_map,
            'unique': list(train_map.keys())
        },
        'combined': {
            'labels': combined_labels,
            'numeric': combined_numeric,
            'map': combined_map,
            'unique': list(combined_map.keys())
        }
    }


def perform_clustering(distance_matrix: np.ndarray, n_clusters: int, method: str = 'complete') -> np.ndarray:
    """Perform hierarchical clustering on distance matrix."""
    if method == 'ward':
        # Convert distance to similarity for ward
        similarity_matrix = np.max(distance_matrix) - distance_matrix
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        cluster_labels = clustering.fit_predict(similarity_matrix)
    elif method in ['complete', 'average', 'single']:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=method,
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

    # Silhouette score
    try:
        # Convert distance to similarity for silhouette score
        similarity_matrix = np.max(distance_matrix) - distance_matrix
        np.fill_diagonal(similarity_matrix, 0)
        metrics['silhouette'] = silhouette_score(similarity_matrix, cluster_labels, metric='precomputed')
    except:
        metrics['silhouette'] = np.nan

    return metrics


def plot_clustering_results(distance_matrix: np.ndarray, label_info: Dict[str, Any],
                          cluster_labels: np.ndarray, scheme_name: str, output_file: str):
    """Create visualization of clustering results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    true_labels = label_info['labels']
    unique_true_labels = label_info['unique']

    # 1. Distance matrix heatmap
    im1 = ax1.imshow(distance_matrix, cmap='viridis')
    ax1.set_title(f'Elastic Distance Matrix\n({scheme_name} Labels)')
    plt.colorbar(im1, ax=ax1)

    # 2. MDS projection colored by true labels
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)

    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_true_labels)))
    for i, label in enumerate(unique_true_labels):
        mask = [tl == label for tl in true_labels]
        ax2.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]],
                   label=f'True: {label}', s=50, alpha=0.7)
    ax2.set_title(f'MDS Projection (True {scheme_name} Labels)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. MDS projection colored by cluster labels
    unique_cluster_labels = sorted(list(set(cluster_labels)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cluster_labels)))
    for i, label in enumerate(unique_cluster_labels):
        mask = cluster_labels == label
        ax3.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]],
                   label=f'Cluster {label}', s=50, alpha=0.7)
    ax3.set_title('MDS Projection (Cluster Labels)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. Confusion matrix
    true_labels_mapped = [unique_true_labels.index(tl) for tl in true_labels]
    cm = confusion_matrix(true_labels_mapped, cluster_labels)

    # Ensure confusion matrix has the right dimensions
    max_cluster = max(cluster_labels)
    if cm.shape[1] <= max_cluster:
        # Pad confusion matrix if needed
        cm_padded = np.zeros((len(unique_true_labels), max_cluster + 1), dtype=int)
        cm_padded[:cm.shape[0], :cm.shape[1]] = cm
        cm = cm_padded

    sns.heatmap(cm, annot=True, fmt='d', ax=ax4, cmap='Blues')
    ax4.set_title(f'Confusion Matrix ({scheme_name})')
    ax4.set_xlabel('Predicted Cluster')
    ax4.set_ylabel(f'True {scheme_name}')
    ax4.set_yticks(range(len(unique_true_labels)))
    ax4.set_yticklabels(unique_true_labels)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return cm


def analyze_scheme(distance_matrix: np.ndarray, label_info: Dict[str, Any],
                  scheme_name: str, method: str = 'complete', max_clusters: int = 6) -> Dict[str, Any]:
    """Analyze clustering for a specific labeling scheme."""

    print(f"\n{'='*60}")
    print(f"ANALYZING {scheme_name.upper()} LABELS")
    print(f"{'='*60}")

    true_labels = label_info['labels']
    true_labels_numeric = label_info['numeric']
    unique_labels = label_info['unique']

    # Print label distribution
    unique_labels_list, counts = np.unique(true_labels, return_counts=True)
    print(f"Ground truth {scheme_name} distribution:")
    for label, count in zip(unique_labels_list, counts):
        print(f"  {label}: {count}")

    # Try different numbers of clusters
    print(f"\nTrying clustering with method '{method}':")
    results = []
    best_ari = -1
    best_n_clusters = 2
    best_cluster_labels = None
    best_cm = None

    for n_clusters in range(2, min(max_clusters + 1, len(unique_labels) + 3)):
        print(f"\nClustering with {n_clusters} clusters...")

        cluster_labels = perform_clustering(distance_matrix, n_clusters, method)
        metrics = compute_clustering_metrics(distance_matrix, true_labels_numeric, cluster_labels)

        print(f"  ARI: {metrics['ari']:.4f}")
        print(f"  Silhouette: {metrics['silhouette']:.4f}")

        results.append({
            'n_clusters': n_clusters,
            'ari': metrics['ari'],
            'silhouette': metrics['silhouette'],
            'cluster_labels': cluster_labels.copy()
        })

        if metrics['ari'] > best_ari:
            best_ari = metrics['ari']
            best_n_clusters = n_clusters
            best_cluster_labels = cluster_labels.copy()

    # Create visualization for best clustering
    output_file = f'clustering_results_{scheme_name.lower()}.png'
    print(f"\nCreating visualization: {output_file}")
    best_cm = plot_clustering_results(distance_matrix, label_info, best_cluster_labels,
                                     scheme_name, output_file)

    # Print summary
    print(f"\nBest ARI: {best_ari:.4f} (with {best_n_clusters} clusters)")
    print(f"Confusion Matrix (Best Clustering - {best_n_clusters} clusters):")
    print(best_cm)

    return {
        'scheme_name': scheme_name,
        'best_ari': best_ari,
        'best_n_clusters': best_n_clusters,
        'best_cluster_labels': best_cluster_labels.tolist(),
        'confusion_matrix': best_cm.tolist(),
        'all_results': results,
        'ground_truth_distribution': {str(label): int(count)
                                    for label, count in zip(unique_labels_list, counts)}
    }


def main():
    parser = argparse.ArgumentParser(description='Enhanced ARI computation with trained/random separation')
    parser.add_argument('--distance-file', default='elastic_eigs_optimal_distance.npy',
                       help='Path to distance matrix (.npy file)')
    parser.add_argument('--index-file', default='elastic_eigs_optimal_index.json',
                       help='Path to file index (.json file)')
    parser.add_argument('--method', default='complete', choices=['ward', 'complete', 'average', 'spectral'],
                       help='Clustering method to use')
    parser.add_argument('--max-clusters', type=int, default=6,
                       help='Maximum number of clusters to try')

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

    # Extract enhanced labels
    print("Extracting enhanced labels from filenames...")
    label_data = extract_enhanced_labels(filenames)

    # Analyze each labeling scheme
    results = {}

    # 1. Architecture labels (hourglass vs pyramid)
    results['architecture'] = analyze_scheme(
        distance_matrix, label_data['architecture'], 'Architecture', args.method, args.max_clusters
    )

    # 2. Training labels (random vs trained)
    results['training'] = analyze_scheme(
        distance_matrix, label_data['training'], 'Training', args.method, args.max_clusters
    )

    # 3. Combined labels (4 classes)
    results['combined'] = analyze_scheme(
        distance_matrix, label_data['combined'], 'Combined', args.method, args.max_clusters
    )

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Architecture ARI: {results['architecture']['best_ari']:.4f} "
          f"({results['architecture']['best_n_clusters']} clusters)")
    print(f"Training ARI:     {results['training']['best_ari']:.4f} "
          f"({results['training']['best_n_clusters']} clusters)")
    print(f"Combined ARI:     {results['combined']['best_ari']:.4f} "
          f"({results['combined']['best_n_clusters']} clusters)")

    # Save comprehensive results
    results_file = args.distance_file.replace('.npy', '_ari_enhanced_results.json')

    # Convert numpy arrays to lists for JSON serialization
    for scheme in results:
        for i, result in enumerate(results[scheme]['all_results']):
            results[scheme]['all_results'][i]['cluster_labels'] = result['cluster_labels'].tolist()

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nComprehensive results saved to: {results_file}")
    print(f"Visualizations saved as:")
    print(f"  - clustering_results_architecture.png")
    print(f"  - clustering_results_training.png")
    print(f"  - clustering_results_combined.png")


if __name__ == '__main__':
    main()
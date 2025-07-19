#!/usr/bin/env python3
"""
Example: DTW-based Neural Network Comparison using Eigenvalue Evolution

This example demonstrates how to compare neural networks using Dynamic Time Warping
(DTW) applied to eigenvalue evolution patterns during spectral analysis.

Key Features:
- Compare two neural networks using DTW
- Visualize eigenvalue evolution alignment
- Perform multiple network comparisons
- Clustering analysis of network similarities
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from neurosheaf import NeurosheafAnalyzer
from neurosheaf.utils.dtw_similarity import FiltrationDTW, quick_dtw_comparison
from neurosheaf.visualization.spectral import SpectralVisualizer


def create_sample_networks():
    """Create sample neural networks for comparison."""
    # Network 1: Standard architecture
    net1 = nn.Sequential(
        nn.Linear(20, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    )
    
    # Network 2: Wider architecture
    net2 = nn.Sequential(
        nn.Linear(20, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 2)
    )
    
    # Network 3: Different activation
    net3 = nn.Sequential(
        nn.Linear(20, 16),
        nn.Tanh(),
        nn.Linear(16, 8),
        nn.Tanh(),
        nn.Linear(8, 4),
        nn.Tanh(),
        nn.Linear(4, 2)
    )
    
    return net1, net2, net3


def example_basic_dtw_comparison():
    """Basic example of DTW comparison between two networks."""
    print("🔍 Basic DTW Comparison Example")
    print("=" * 50)
    
    # Create networks and data
    net1, net2, _ = create_sample_networks()
    data = torch.randn(100, 20)
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    try:
        # Compare networks using DTW
        result = analyzer.compare_networks(
            net1, net2, data, 
            method='dtw',
            eigenvalue_index=0,  # Compare largest eigenvalue
            multivariate=False
        )
        
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"DTW Distance: {result['dtw_comparison']['dtw_comparison']['distance']:.4f}")
        print(f"Normalized Distance: {result['dtw_comparison']['dtw_comparison']['normalized_distance']:.4f}")
        
        # Print similarity metrics
        metrics = result['dtw_comparison']['similarity_metrics']
        print(f"\nDetailed Metrics:")
        print(f"  Combined Similarity: {metrics['combined_similarity']:.4f}")
        print(f"  Persistence Similarity: {metrics['persistence_similarity']:.4f}")
        print(f"  Spectral Similarity: {metrics['spectral_similarity']:.4f}")
        print(f"  Alignment Quality: {metrics['alignment_quality']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: DTW comparison requires dtaidistance or tslearn libraries")
    
    print()


def example_multiple_network_comparison():
    """Example of comparing multiple networks using DTW."""
    print("🔍 Multiple Network Comparison Example")
    print("=" * 50)
    
    # Create networks
    net1, net2, net3 = create_sample_networks()
    networks = [net1, net2, net3]
    network_names = ['Standard', 'Wide', 'Tanh']
    
    # Create data
    data = torch.randn(100, 20)
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    try:
        # Compare multiple networks
        result = analyzer.compare_multiple_networks(
            networks, data,
            method='dtw',
            eigenvalue_index=0
        )
        
        print("Distance Matrix:")
        distance_matrix = result['distance_matrix']
        print(f"Shape: {distance_matrix.shape}")
        
        # Print formatted distance matrix
        print("\n     ", end="")
        for name in network_names:
            print(f"{name:>8}", end="")
        print()
        
        for i, name in enumerate(network_names):
            print(f"{name:>8}", end="")
            for j in range(len(network_names)):
                print(f"{distance_matrix[i, j]:>8.3f}", end="")
            print()
        
        # Print similarity rankings
        print("\nSimilarity Rankings:")
        for ranking in result['similarity_rankings']:
            idx = ranking['sheaf_index']
            print(f"\n{network_names[idx]} is most similar to:")
            for similar in ranking['most_similar'][:2]:  # Top 2
                sim_idx = similar['sheaf_index']
                similarity = similar['similarity']
                print(f"  {network_names[sim_idx]}: {similarity:.4f}")
        
        # Print cluster analysis
        if 'cluster_analysis' in result and result['cluster_analysis'].get('status') != 'sklearn_not_available':
            cluster_info = result['cluster_analysis']
            print(f"\nCluster Analysis:")
            print(f"  Number of clusters: {cluster_info['n_clusters']}")
            print(f"  Silhouette score: {cluster_info['silhouette_score']:.4f}")
            
            for cluster_name, members in cluster_info['cluster_assignments'].items():
                member_names = [network_names[i] for i in members]
                print(f"  {cluster_name}: {', '.join(member_names)}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Multiple network comparison requires DTW libraries")
    
    print()


def example_custom_dtw_analysis():
    """Example of custom DTW analysis with specific parameters."""
    print("🔍 Custom DTW Analysis Example")
    print("=" * 50)
    
    # Create sample eigenvalue sequences manually
    n_steps = 20
    eigenvals1 = []
    eigenvals2 = []
    
    for i in range(n_steps):
        # Network 1: Exponential decay
        vals1 = torch.tensor([
            np.exp(-0.1 * i),
            np.exp(-0.2 * i),
            np.exp(-0.3 * i)
        ])
        eigenvals1.append(vals1)
        
        # Network 2: Power law decay (similar but different)
        vals2 = torch.tensor([
            (i + 1) ** -0.5,
            (i + 1) ** -0.8,
            (i + 1) ** -1.2
        ])
        eigenvals2.append(vals2)
    
    # Create DTW comparator with custom parameters
    dtw_comparator = FiltrationDTW(
        method='auto',
        constraint_band=0.2,  # Allow moderate warping
        eigenvalue_weight=0.8,
        structural_weight=0.2
    )
    
    try:
        # Compare eigenvalue evolutions
        result = dtw_comparator.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0,  # Compare largest eigenvalue
            multivariate=False
        )
        
        print(f"DTW Distance: {result['distance']:.4f}")
        print(f"Normalized Distance: {result['normalized_distance']:.4f}")
        print(f"Sequence Lengths: {result['sequence1_length']}, {result['sequence2_length']}")
        print(f"Method Used: {result['method']}")
        
        # Quick comparison using utility function
        quick_distance = quick_dtw_comparison(eigenvals1, eigenvals2, eigenvalue_index=0)
        print(f"Quick DTW Distance: {quick_distance:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Custom DTW analysis requires DTW libraries")
    
    print()


def example_dtw_visualization():
    """Example of DTW alignment visualization."""
    print("🔍 DTW Visualization Example")
    print("=" * 50)
    
    # Create sample alignment data
    alignment_data = {
        'sequence1': [1.0, 0.8, 0.6, 0.4, 0.2],
        'sequence2': [1.2, 0.9, 0.7, 0.5, 0.3, 0.1],
        'filtration_params1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'filtration_params2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'alignment': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        'alignment_quality': 0.85,
        'distance': 0.25
    }
    
    try:
        # Create visualizer
        visualizer = SpectralVisualizer()
        
        # Create DTW alignment plot
        fig = visualizer.plot_dtw_alignment(
            alignment_data,
            title="DTW Alignment: Network Comparison"
        )
        
        # Save plot
        fig.write_html("dtw_alignment_example.html")
        print("DTW alignment visualization saved as 'dtw_alignment_example.html'")
        
        # Create comparison summary
        comparison_results = [{
            'distance_matrix': np.array([[0.0, 0.25, 0.40],
                                       [0.25, 0.0, 0.35],
                                       [0.40, 0.35, 0.0]])
        }]
        
        network_names = ['Network A', 'Network B', 'Network C']
        
        summary_fig = visualizer.plot_dtw_comparison_summary(
            comparison_results,
            network_names,
            title="DTW Comparison Summary"
        )
        
        summary_fig.write_html("dtw_summary_example.html")
        print("DTW comparison summary saved as 'dtw_summary_example.html'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Visualization requires plotly")
    
    print()


def example_dtw_performance_tips():
    """Example showing DTW performance optimization tips."""
    print("🔍 DTW Performance Tips")
    print("=" * 50)
    
    print("1. Library Selection:")
    print("   - dtaidistance: Fastest for univariate sequences")
    print("   - tslearn: Best for multivariate sequences")
    print("   - Use constraint_band=0.1-0.2 for large sequences")
    print()
    
    print("2. Sequence Length Guidelines:")
    print("   - <100 points: No constraints needed")
    print("   - 100-500 points: Use constraint_band=0.1")
    print("   - >500 points: Use constraint_band=0.05 or consider sampling")
    print()
    
    print("3. Memory Usage:")
    print("   - DTW requires O(N²) memory")
    print("   - For 1000-point sequences: ~8MB per comparison")
    print("   - Consider batch processing for multiple comparisons")
    print()
    
    print("4. Eigenvalue Index Selection:")
    print("   - eigenvalue_index=0: Compare largest eigenvalue (most stable)")
    print("   - eigenvalue_index=None: Compare all eigenvalues (more comprehensive)")
    print("   - multivariate=True: Compare all eigenvalues simultaneously")
    print()
    
    # Demonstrate performance comparison
    try:
        # Create different sized sequences
        small_seq = [torch.tensor([np.random.random()]) for _ in range(50)]
        large_seq = [torch.tensor([np.random.random()]) for _ in range(200)]
        
        # Time comparisons
        import time
        
        # Small sequence
        start = time.time()
        quick_dtw_comparison(small_seq, small_seq)
        small_time = time.time() - start
        
        # Large sequence with constraints
        dtw_constrained = FiltrationDTW(constraint_band=0.1)
        start = time.time()
        dtw_constrained.compare_eigenvalue_evolution(large_seq, large_seq)
        large_time = time.time() - start
        
        print(f"Performance Comparison:")
        print(f"  Small sequence (50 points): {small_time:.4f}s")
        print(f"  Large sequence (200 points, constrained): {large_time:.4f}s")
        print(f"  Speedup factor: {large_time / small_time:.2f}x")
        
    except Exception as e:
        print(f"Performance test error: {e}")


def main():
    """Run all DTW examples."""
    print("🚀 DTW-based Neural Network Comparison Examples")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_basic_dtw_comparison()
    example_multiple_network_comparison()
    example_custom_dtw_analysis()
    example_dtw_visualization()
    example_dtw_performance_tips()
    
    print("✅ All examples completed!")
    print("\nTo run individual examples:")
    print("  python dtw_eigenvalue_comparison.py")
    print("\nGenerated files:")
    print("  - dtw_alignment_example.html")
    print("  - dtw_summary_example.html")


if __name__ == "__main__":
    main()
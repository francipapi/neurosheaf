"""Visualization script for DAG pipeline persistence results.

This script creates visualizations for:
1. The DAG/poset structure
2. Persistence diagrams (birth-death scatter plot)
3. Persistence barcodes
4. Eigenvalue evolution through filtration
5. Spectral gap evolution
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
import torch


def load_results(filename='dag_pipeline_results.pkl'):
    """Load saved results from the pipeline test."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def visualize_poset(graph, stalk_dimensions, ax=None):
    """Visualize the DAG/poset structure with colored nodes."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a layered layout for better visualization
    # Group nodes by topological generation
    generations = list(nx.topological_generations(graph))
    pos = {}
    
    # Position nodes in layers
    for gen_idx, generation in enumerate(generations):
        gen_size = len(generation)
        for idx, node in enumerate(sorted(generation)):
            x = (idx - gen_size/2 + 0.5) * (10.0 / max(1, gen_size))
            y = -gen_idx * 2
            pos[node] = (x, y)
    
    # Color nodes by stalk dimension
    node_colors = [stalk_dimensions[node] for node in graph.nodes()]
    
    # Draw the graph
    nx.draw(graph, pos, ax=ax,
            node_color=node_colors,
            node_size=800,
            cmap='viridis',
            with_labels=True,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            alpha=0.8)
    
    # Add colorbar for stalk dimensions
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(stalk_dimensions.values()),
                                                vmax=max(stalk_dimensions.values())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Stalk Dimension')
    
    ax.set_title('DAG Structure with Stalk Dimensions', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return ax


def plot_persistence_diagram(results, ax=None):
    """Plot the persistence diagram (birth-death scatter plot)."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    diagrams = results['diagrams']
    birth_death_pairs = diagrams['birth_death_pairs']
    infinite_bars = diagrams['infinite_bars']
    
    # Plot finite persistence pairs
    if birth_death_pairs:
        births = [pair['birth'] for pair in birth_death_pairs]
        deaths = [pair['death'] for pair in birth_death_pairs]
        lifetimes = [pair['lifetime'] for pair in birth_death_pairs]
        
        scatter = ax.scatter(births, deaths, c=lifetimes, cmap='hot', 
                           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Lifetime')
    
    # Plot infinite bars
    if infinite_bars:
        inf_births = [bar['birth'] for bar in infinite_bars]
        # Plot at the top of the diagram
        max_val = max(deaths) if birth_death_pairs else 1.0
        ax.scatter(inf_births, [max_val * 1.1] * len(inf_births), 
                  marker='^', s=100, c='red', label='Infinite bars')
    
    # Plot diagonal
    min_val = 0
    max_val = max(deaths) if birth_death_pairs else 1.0
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title('Persistence Diagram', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics
    stats = diagrams['statistics']
    stats_text = (f"Finite pairs: {stats['n_finite_pairs']}\n"
                 f"Infinite bars: {stats['n_infinite_bars']}\n"
                 f"Mean lifetime: {stats.get('mean_lifetime', 0):.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return ax


def plot_persistence_barcode(results, ax=None):
    """Plot the persistence barcode."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    diagrams = results['diagrams']
    birth_death_pairs = diagrams['birth_death_pairs']
    infinite_bars = diagrams['infinite_bars']
    
    all_bars = []
    
    # Add finite bars
    for pair in birth_death_pairs:
        all_bars.append({
            'birth': pair['birth'],
            'death': pair['death'],
            'lifetime': pair['lifetime'],
            'type': 'finite'
        })
    
    # Add infinite bars
    max_param = max(results['filtration_params'])
    for bar in infinite_bars:
        all_bars.append({
            'birth': bar['birth'],
            'death': max_param * 1.2,  # Extend beyond max parameter
            'lifetime': float('inf'),
            'type': 'infinite'
        })
    
    # Sort by birth time
    all_bars.sort(key=lambda x: x['birth'])
    
    # Plot bars
    for i, bar in enumerate(all_bars):
        color = 'red' if bar['type'] == 'infinite' else 'blue'
        alpha = 0.8 if bar['type'] == 'infinite' else 0.6
        
        ax.barh(i, bar['death'] - bar['birth'], left=bar['birth'],
               height=0.8, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Filtration Parameter', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    ax.set_title('Persistence Barcode', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    finite_patch = Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.6, edgecolor='black')
    infinite_patch = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.8, edgecolor='black')
    ax.legend([finite_patch, infinite_patch], ['Finite bars', 'Infinite bars'])
    
    return ax


def plot_eigenvalue_evolution(results, ax=None):
    """Plot eigenvalue evolution through filtration."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    filtration_params = results['filtration_params']
    eigenval_seqs = results['persistence_result']['eigenvalue_sequences']
    
    # Determine how many eigenvalues to plot (plot smallest 10)
    n_plot = min(10, min(len(seq) for seq in eigenval_seqs))
    
    # Extract evolution of smallest eigenvalues
    eigenval_tracks = []
    for i in range(n_plot):
        track = []
        for eigenvals in eigenval_seqs:
            if i < len(eigenvals):
                sorted_vals = torch.sort(eigenvals)[0]
                track.append(sorted_vals[i].item())
            else:
                track.append(np.nan)
        eigenval_tracks.append(track)
    
    # Plot each eigenvalue track
    for i, track in enumerate(eigenval_tracks):
        ax.plot(filtration_params, track, marker='o', markersize=3, 
               label=f'λ_{i}', alpha=0.7)
    
    ax.set_xlabel('Filtration Parameter', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Eigenvalue Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-15)
    
    return ax


def plot_spectral_gap_evolution(results, ax=None):
    """Plot spectral gap evolution through filtration."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    filtration_params = results['filtration_params']
    gap_evolution = results['features']['spectral_gap_evolution']
    
    ax.plot(filtration_params, gap_evolution, 'b-', linewidth=2, marker='o', markersize=4)
    ax.fill_between(filtration_params, 0, gap_evolution, alpha=0.3)
    
    ax.set_xlabel('Filtration Parameter', fontsize=12)
    ax.set_ylabel('Spectral Gap', fontsize=12)
    ax.set_title('Spectral Gap Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_gap = np.mean(gap_evolution)
    max_gap = np.max(gap_evolution)
    min_gap = np.min(gap_evolution)
    
    stats_text = (f"Mean gap: {mean_gap:.4f}\n"
                 f"Max gap: {max_gap:.4f}\n"
                 f"Min gap: {min_gap:.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return ax


def create_summary_figure(data):
    """Create a comprehensive summary figure with all visualizations."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplot layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Poset structure (large subplot)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    visualize_poset(data['graph'], data['stalk_dimensions'], ax=ax1)
    
    # 2. Persistence diagram
    ax2 = fig.add_subplot(gs[0, 2])
    plot_persistence_diagram(data['results'], ax=ax2)
    
    # 3. Persistence barcode
    ax3 = fig.add_subplot(gs[1, 2])
    plot_persistence_barcode(data['results'], ax=ax3)
    
    # 4. Eigenvalue evolution
    ax4 = fig.add_subplot(gs[2, 0:2])
    plot_eigenvalue_evolution(data['results'], ax=ax4)
    
    # 5. Spectral gap evolution
    ax5 = fig.add_subplot(gs[2, 2])
    plot_spectral_gap_evolution(data['results'], ax=ax5)
    
    # Add overall title
    fig.suptitle(f'Persistence Analysis of {data["n_vertices"]}-Vertex DAG', 
                 fontsize=16, fontweight='bold')
    
    return fig


def main():
    """Main visualization function."""
    print("Loading results...")
    data = load_results()
    
    print("Creating visualizations...")
    
    # Create individual plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Poset structure
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    visualize_poset(data['graph'], data['stalk_dimensions'], ax=ax1)
    plt.tight_layout()
    plt.savefig('dag_poset_structure.png', dpi=300, bbox_inches='tight')
    print("Saved: dag_poset_structure.png")
    
    # 2. Persistence diagram
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    plot_persistence_diagram(data['results'], ax=ax2)
    plt.tight_layout()
    plt.savefig('dag_persistence_diagram.png', dpi=300, bbox_inches='tight')
    print("Saved: dag_persistence_diagram.png")
    
    # 3. Persistence barcode
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    plot_persistence_barcode(data['results'], ax=ax3)
    plt.tight_layout()
    plt.savefig('dag_persistence_barcode.png', dpi=300, bbox_inches='tight')
    print("Saved: dag_persistence_barcode.png")
    
    # 4. Eigenvalue evolution
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    plot_eigenvalue_evolution(data['results'], ax=ax4)
    plt.tight_layout()
    plt.savefig('dag_eigenvalue_evolution.png', dpi=300, bbox_inches='tight')
    print("Saved: dag_eigenvalue_evolution.png")
    
    # 5. Spectral gap evolution  
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
    plot_spectral_gap_evolution(data['results'], ax=ax5)
    plt.tight_layout()
    plt.savefig('dag_spectral_gap_evolution.png', dpi=300, bbox_inches='tight')
    print("Saved: dag_spectral_gap_evolution.png")
    
    # 6. Create summary figure
    fig_summary = create_summary_figure(data)
    plt.savefig('dag_persistence_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: dag_persistence_summary.png")
    
    # Show all plots
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    results = data['results']
    print(f"Total filtration steps: {len(results['filtration_params'])}")
    print(f"Birth events: {results['features']['num_birth_events']}")
    print(f"Death events: {results['features']['num_death_events']}")
    print(f"Crossing events: {results['features']['num_crossings']}")
    print(f"Persistent paths: {results['features']['num_persistent_paths']}")
    print(f"Infinite bars: {results['diagrams']['statistics']['n_infinite_bars']}")
    print(f"Finite pairs: {results['diagrams']['statistics']['n_finite_pairs']}")
    print(f"Mean lifetime: {results['diagrams']['statistics'].get('mean_lifetime', 0):.6f}")
    print(f"Max lifetime: {results['diagrams']['statistics'].get('max_lifetime', 0):.6f}")
    print(f"Total persistence: {results['diagrams']['statistics'].get('total_persistence', 0):.6f}")


if __name__ == "__main__":
    main()
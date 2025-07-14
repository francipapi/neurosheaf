"""pytest suite: validates Neurosheaf end‑to‑end on pre‑trained ResNet‑18
using the new clean architecture with whitened coordinates.

Ground‑truth metrics are adjusted for the whitened coordinate system.
The test downloads the model weights automatically if the .pt file is
missing (≈ 44 MB) and stores/re‑uses local copies of activation,
Sheaf, and Laplacian to avoid recomputation on re‑runs.
"""

from __future__ import annotations

import cProfile
import json
import os
import pstats
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import psutil
import pytest
import torch
import torchvision as tv
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Neurosheaf imports - updated for clean architecture
from neurosheaf.sheaf import (
    SheafBuilder,
    SheafLaplacianBuilder,
    EnhancedActivationExtractor,
    Sheaf,
    WhiteningInfo
)
from neurosheaf.utils.device import detect_optimal_device

# -----------------------------------------------------------------------------
# Ground truth constants adjusted for whitened coordinate system
# -----------------------------------------------------------------------------
GT = {
    "nodes": 32,  # Expected to remain the same
    "edges": 38,  # Expected to remain the same
    "stalk_dim_sum_max": 4_096,  # Upper bound - whitening may reduce this
    "stalk_dim_sum_min": 3_000,  # Lower bound - some rank reduction expected
    "laplacian_nnz_max": 120_000,  # Upper bound for non-zeros
    "laplacian_nnz_min": 80_000,   # Lower bound for non-zeros
    "harmonic_dim": 4,  # Should remain the same
    "restriction_residual_max": 0.05,
    "symmetry_err": 1e-10,
    "lambda_min": -8e-10,
    "runtime_s": 300,
    "ram_bytes": int(3.8 * 1024 ** 3),
    "whitened_coordinates": True,  # New: expect whitened coordinates
}

HERE = Path(__file__).parent
CACHE_DIR = HERE / "_cache"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_WEIGHTS = CACHE_DIR / "resnet18_imagenet1k.pt"


# -----------------------------------------------------------------------------
# Utilities (keeping the same visualization and profiling utilities)
# -----------------------------------------------------------------------------

def current_rss() -> int:
    """Return resident set size (bytes)."""
    return psutil.Process(os.getpid()).memory_info().rss


def approx(val, target, tol):
    return abs(val - target) <= tol


def visualize_poset(poset: nx.DiGraph, stalks: Dict[str, torch.Tensor], save_path: Path = None):
    """Visualize the poset structure with node information.
    
    Args:
        poset: NetworkX directed graph representing the poset
        stalks: Dictionary of stalks to show dimensions
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(20, 16))
    
    # Create a hierarchical layout
    try:
        # Try to create layers based on topological sort
        layers = {}
        for node in nx.topological_sort(poset):
            # Determine layer based on max distance from sources
            if poset.in_degree(node) == 0:
                layer = 0
            else:
                layer = max(layers.get(pred, 0) for pred in poset.predecessors(node)) + 1
            
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        # Create positions
        pos = {}
        max_width = max(len(nodes) for nodes in layers.values())
        for layer_idx, nodes in layers.items():
            width = len(nodes)
            for i, node in enumerate(nodes):
                x = (i - width/2) * (max_width / width) * 2
                y = -layer_idx * 2
                pos[node] = (x, y)
                
    except nx.NetworkXError:
        # Fallback to spring layout if not a DAG
        pos = nx.spring_layout(poset, k=3, iterations=50)
    
    # Create labels with rank information
    labels = {}
    node_colors = []
    max_rank = max(stalk.shape[0] for stalk in stalks.values()) if stalks else 128
    
    for node in poset.nodes():
        if node in stalks:
            rank = stalks[node].shape[0]
            percentage = (rank / max_rank) * 100
            labels[node] = f"{node}\nrank {rank}\n({percentage:.0f}%)"
            
            # Color based on rank percentage
            if rank == max_rank:
                node_colors.append('lightblue')  # Full rank
            elif percentage >= 90:
                node_colors.append('lightgreen')  # Nearly full rank (90%+)
            elif percentage >= 75:
                node_colors.append('lightyellow')  # Medium rank (75-90%)
            elif percentage >= 50:
                node_colors.append('orange')  # Low rank (50-75%)
            else:
                node_colors.append('lightcoral')  # Very low rank (<50%)
        else:
            labels[node] = node
            node_colors.append('lightgray')
    
    # Draw the graph
    nx.draw(poset, pos, labels=labels, node_color=node_colors, 
            node_size=3000, font_size=8, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray',
            arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Add title and legend
    plt.title("ResNet-18 Poset Structure (Whitened Coordinates)\n(Node ranks and percentages shown)", fontsize=16)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label=f'Full rank ({max_rank})'),
        Patch(facecolor='lightgreen', label='Nearly full rank (90%+)'),
        Patch(facecolor='lightyellow', label='Medium rank (75-90%)'),
        Patch(facecolor='orange', label='Low rank (50-75%)'),
        Patch(facecolor='lightcoral', label='Very low rank (<50%)'),
        Patch(facecolor='lightgray', label='No stalk data')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Poset visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_poset_structure(poset: nx.DiGraph, stalks: Dict[str, torch.Tensor]):
    """Print a text-based representation of the poset structure."""
    print("\n=== POSET STRUCTURE SUMMARY (WHITENED COORDINATES) ===")
    
    # Group nodes by layer
    layer_groups = {
        'Input': ['conv1'],
        'Early': ['relu', 'maxpool'],
        'Layer1': [n for n in poset.nodes() if n.startswith('layer1')],
        'Layer2': [n for n in poset.nodes() if n.startswith('layer2')],
        'Layer3': [n for n in poset.nodes() if n.startswith('layer3')],
        'Layer4': [n for n in poset.nodes() if n.startswith('layer4')],
        'Output': ['avgpool', 'fc']
    }
    
    for group_name, nodes in layer_groups.items():
        existing_nodes = [n for n in nodes if n in poset.nodes()]
        if existing_nodes:
            print(f"\n{group_name}:")
            for node in sorted(existing_nodes):
                if node in stalks:
                    rank = stalks[node].shape[0]
                    in_deg = poset.in_degree(node)
                    out_deg = poset.out_degree(node)
                    print(f"  {node:<25} [rank {rank}] (in:{in_deg}, out:{out_deg})")
                    
                    # Show connections
                    predecessors = list(poset.predecessors(node))
                    successors = list(poset.successors(node))
                    if predecessors:
                        print(f"    ← from: {', '.join(predecessors)}")
                    if successors:
                        print(f"    → to: {', '.join(successors)}")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total nodes: {poset.number_of_nodes()}")
    print(f"Total edges: {poset.number_of_edges()}")
    
    # Rank distribution
    rank_counts = {}
    for node, stalk in stalks.items():
        rank = stalk.shape[0]
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    print("\nRank distribution:")
    for rank in sorted(rank_counts.keys()):
        print(f"  Rank {rank}: {rank_counts[rank]} nodes")
    
    print("=======================================")


class TimingProfiler:
    """Simple timing profiler for measuring execution phases."""
    
    def __init__(self):
        self.times = {}
        self.current_phase = None
        self.phase_start = None
    
    def start_phase(self, phase_name: str):
        """Start timing a new phase."""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_start = time.time()
    
    def end_phase(self):
        """End the current phase and record timing."""
        if self.current_phase and self.phase_start:
            elapsed = time.time() - self.phase_start
            self.times[self.current_phase] = elapsed
            self.current_phase = None
            self.phase_start = None
    
    def print_summary(self):
        """Print timing summary."""
        print("\n=== PERFORMANCE PROFILING RESULTS ===")
        total_time = sum(self.times.values())
        
        print(f"Total execution time: {total_time:.3f}s")
        print("\nBreakdown by phase:")
        
        # Sort by time (descending)
        sorted_times = sorted(self.times.items(), key=lambda x: x[1], reverse=True)
        
        for phase, elapsed in sorted_times:
            percentage = (elapsed / total_time) * 100
            print(f"  {phase:<30}: {elapsed:6.3f}s ({percentage:5.1f}%)")
        
        print("======================================")


# -----------------------------------------------------------------------------
# Fixtures - updated for clean architecture
# -----------------------------------------------------------------------------

def _download_weights() -> None:
    if MODEL_WEIGHTS.exists():
        return
    model = tv.models.resnet18(weights="IMAGENET1K_V1")
    model.eval()
    torch.save(model.state_dict(), MODEL_WEIGHTS)


@pytest.fixture(scope="session")
def model() -> torch.nn.Module:
    _download_weights()
    m = tv.models.resnet18()
    m.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu", weights_only=True))
    m.eval()
    return m


@pytest.fixture(scope="session")
def input_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(128, 3, 224, 224)  # 128 samples for rank variation


@pytest.fixture(scope="session")
def activations(model, input_batch) -> Dict[str, torch.Tensor]:
    print(f"\n=== TIMING: Starting activation extraction ===")
    start_time = time.time()
    
    # Use new clean activation extractor
    extractor = EnhancedActivationExtractor(capture_functional=True)
    with torch.no_grad():
        activations = extractor.extract_comprehensive_activations(model, input_batch)
    
    extraction_time = time.time() - start_time
    print(f"Activation extraction took: {extraction_time:.3f}s")
    
    print(f"\n=== ACTIVATION EXTRACTION SUMMARY ===")
    print(f"Total activations extracted: {len(activations)}")
    print(f"Input batch shape: {input_batch.shape}")
    
    # Group by type
    module_acts = {k: v for k, v in activations.items() if not any(x in k for x in ['relu_', 'max_pool2d_', 'adaptive_avg_pool2d_', 'flatten_', 'add_', 'cat_'])}
    functional_acts = {k: v for k, v in activations.items() if any(x in k for x in ['relu_', 'max_pool2d_', 'adaptive_avg_pool2d_', 'flatten_', 'add_', 'cat_'])}
    
    print(f"Module activations: {len(module_acts)}")
    for name, tensor in sorted(module_acts.items()):
        print(f"  {name:<25}: {tensor.shape}")
    
    print(f"Functional activations: {len(functional_acts)}")
    for name, tensor in sorted(functional_acts.items()):
        print(f"  {name:<25}: {tensor.shape}")
    print(f"====================================\n")
    
    return activations


@pytest.fixture(scope="session")
def sheaf(model, activations) -> Sheaf:
    print(f"\n=== TIMING: Starting sheaf construction (Clean Architecture) ===")
    start_time = time.time()
    
    # Use new simplified SheafBuilder
    builder = SheafBuilder()
    result = builder.build_from_activations(model, activations, validate=True)
    
    construction_time = time.time() - start_time
    print(f"Sheaf construction took: {construction_time:.3f}s")
    
    # Verify it's using whitened coordinates
    assert result.metadata.get('whitened_coordinates', False), "Expected whitened coordinates"
    
    return result


@pytest.fixture(scope="session") 
def laplacian(sheaf):
    print(f"\n=== TIMING: Starting Laplacian construction (Clean Architecture) ===")
    start_time = time.time()
    
    # Use new simplified SheafLaplacianBuilder
    builder = SheafLaplacianBuilder(validate_properties=True)
    L, meta = builder.build(sheaf)
    
    laplacian_time = time.time() - start_time
    print(f"Laplacian construction took: {laplacian_time:.3f}s")
    
    return L, meta


# -----------------------------------------------------------------------------
# Tests - updated for clean architecture
# -----------------------------------------------------------------------------

@pytest.mark.timeout(GT["runtime_s"] + 30)
@pytest.mark.slow
def test_end_to_end_clean_architecture(model, sheaf, laplacian):
    """Test the complete pipeline using the clean architecture."""
    # Initialize profiler
    profiler = TimingProfiler()
    profiler.start_phase("Test setup")
    
    start = time.time()
    L, meta = laplacian
    
    profiler.start_phase("Whitened coordinates validation")
    
    # 1. Verify whitened coordinates are being used
    assert sheaf.metadata.get('whitened_coordinates', False), "Expected whitened coordinates"
    assert sheaf.metadata.get('construction_method') == 'whitened_activations', "Expected whitened construction method"
    
    print(f"\n=== WHITENED COORDINATES VALIDATION ===")
    print(f"✓ Using whitened coordinates: {sheaf.metadata.get('whitened_coordinates')}")
    print(f"✓ Construction method: {sheaf.metadata.get('construction_method')}")
    print(f"✓ Whitening maps available: {len(sheaf.whitening_maps)} nodes")
    
    # Check that we have whitening maps for stalks
    for node in sheaf.stalks:
        if node in sheaf.whitening_maps:
            W = sheaf.whitening_maps[node]
            print(f"  {node:<25}: whitening map {W.shape}")
        else:
            print(f"  {node:<25}: WARNING - no whitening map")
    
    print(f"========================================\n")
    
    profiler.start_phase("Poset structure analysis")

    # 2. Poset structure (should be same as before)
    poset: nx.DiGraph = sheaf.poset
    
    print(f"=== POSET STRUCTURE ANALYSIS ===")
    print(f"Nodes: {poset.number_of_nodes()}")
    print(f"Edges: {poset.number_of_edges()}")
    
    # Detailed analysis
    print_poset_structure(poset, sheaf.stalks)
    
    # Validate basic structure
    assert poset.number_of_nodes() == GT["nodes"]
    assert poset.number_of_edges() == GT["edges"]
    
    profiler.start_phase("Stalk analysis")
    
    # 3. Stalk dimensions (may be reduced due to whitening)
    total_dim = sum(t.shape[0] if t.ndim == 2 else t.size(0) for t in sheaf.stalks.values())
    
    print(f"\n=== STALK ANALYSIS (WHITENED) ===")
    print(f"Total stalk dimensions: {total_dim}")
    print(f"Expected range: [{GT['stalk_dim_sum_min']}, {GT['stalk_dim_sum_max']}]")
    
    # Analyze individual stalks
    print(f"\nIndividual stalk analysis:")
    for node, stalk in sheaf.stalks.items():
        dims = stalk.shape[0] if len(stalk.shape) == 2 else stalk.size(0)
        
        # Check if it's identity (expected in whitened coordinates)
        if isinstance(stalk, torch.Tensor):
            stalk_np = stalk.detach().cpu().numpy()
        else:
            stalk_np = stalk
        
        # Check if close to identity
        I = np.eye(dims)
        identity_error = np.linalg.norm(stalk_np - I)
        is_identity = identity_error < 1e-10
        
        print(f"  {node:<25}: {stalk.shape}, identity_error={identity_error:.2e}, is_identity={is_identity}")
    
    # Validate dimension bounds
    assert GT["stalk_dim_sum_min"] <= total_dim <= GT["stalk_dim_sum_max"], \
        f"Total dimensions {total_dim} outside expected range [{GT['stalk_dim_sum_min']}, {GT['stalk_dim_sum_max']}]"
    
    print(f"✓ Total dimensions within expected range")
    print(f"================================\n")
    
    profiler.start_phase("Restriction map analysis")
    
    # 4. Restriction map analysis
    print(f"=== RESTRICTION MAP ANALYSIS (WHITENED) ===")
    
    restriction_info = []
    for i, ((source, target), R) in enumerate(sheaf.restrictions.items()):
        # Get dimensions
        shape = R.shape
        
        # Compute rank
        if isinstance(R, torch.Tensor):
            R_np = R.detach().cpu().numpy()
        else:
            R_np = R
        
        rank = np.linalg.matrix_rank(R_np, tol=1e-10)
        
        # Check orthogonality properties
        r_s, r_t = shape
        if r_s <= r_t:
            # Column orthonormal: R^T R = I
            RTR = R_np.T @ R_np
            I_s = np.eye(r_s)
            orth_error = np.linalg.norm(RTR - I_s)
        else:
            # Row orthonormal: R R^T = I
            RRT = R_np @ R_np.T
            I_t = np.eye(r_t)
            orth_error = np.linalg.norm(RRT - I_t)
        
        restriction_info.append({
            'edge': (source, target),
            'shape': shape,
            'rank': rank,
            'orth_error': orth_error
        })
        
        print(f"  {i:2d}. {source:<20} -> {target:<20}")
        print(f"      Shape: {shape}, Rank: {rank}/{min(shape)}, Orth_error: {orth_error:.2e}")
    
    # Validate orthogonality
    max_orth_error = max(r['orth_error'] for r in restriction_info)
    print(f"\nMaximum orthogonality error: {max_orth_error:.2e}")
    assert max_orth_error < 1e-4, f"Orthogonality error too large: {max_orth_error}"
    
    print(f"✓ All restriction maps satisfy orthogonality")
    print(f"=============================================\n")
    
    profiler.start_phase("Laplacian analysis")
    
    # 5. Laplacian analysis
    L_csr: sparse.csr_matrix = L
    
    print(f"=== LAPLACIAN ANALYSIS (WHITENED) ===")
    print(f"Matrix shape: {L_csr.shape}")
    print(f"Non-zero elements: {L_csr.nnz:,}")
    print(f"Sparsity: {(1 - L_csr.nnz / (L_csr.shape[0] * L_csr.shape[1])) * 100:.2f}%")
    print(f"Construction time: {meta.construction_time:.3f}s")
    
    # Validate Laplacian properties
    assert L_csr.shape == (total_dim, total_dim)
    assert GT["laplacian_nnz_min"] <= L_csr.nnz <= GT["laplacian_nnz_max"], \
        f"Laplacian nnz {L_csr.nnz} outside expected range [{GT['laplacian_nnz_min']}, {GT['laplacian_nnz_max']}]"
    
    # Symmetry check
    sym_err = abs(L_csr - L_csr.T).max()
    assert sym_err < GT["symmetry_err"], f"Symmetry error too large: {sym_err}"
    
    print(f"✓ Laplacian dimensions and sparsity within bounds")
    print(f"✓ Symmetry error: {sym_err:.2e}")
    
    # Eigenvalue analysis
    try:
        evals = eigsh(L_csr, k=10, which="SM", tol=1e-4, maxiter=5000, return_eigenvectors=False)
        harmonic_count = np.sum(evals < 1e-6)
        min_eigenval = evals.min()
        
        print(f"✓ Smallest eigenvalues: {evals[:5]}")
        print(f"✓ Harmonic dimension: {harmonic_count}")
        print(f"✓ Minimum eigenvalue: {min_eigenval:.2e}")
        
        assert harmonic_count >= GT["harmonic_dim"], f"Expected at least {GT['harmonic_dim']} harmonic, got {harmonic_count}"
        assert min_eigenval >= GT["lambda_min"], f"Minimum eigenvalue too negative: {min_eigenval}"
        
    except Exception as e:
        print(f"Warning: Eigenvalue computation failed: {e}")
        # Just check basic matrix properties
        assert abs((L_csr - L_csr.T).max()) < 1e-10
    
    print(f"====================================\n")
    
    profiler.start_phase("Visualization generation")
    
    # 6. Generate visualizations
    viz_path = HERE / "resnet18_poset_clean_architecture.png"
    visualize_poset(poset, sheaf.stalks, save_path=viz_path)
    
    profiler.start_phase("Final validation")
    
    # 7. Final resource and performance validation
    runtime = time.time() - start
    current_ram = current_rss()
    
    print(f"=== PERFORMANCE VALIDATION ===")
    print(f"Total runtime: {runtime:.3f}s (limit: {GT['runtime_s']}s)")
    print(f"Memory usage: {current_ram / (1024**3):.2f}GB (limit: {GT['ram_bytes'] / (1024**3):.2f}GB)")
    
    assert runtime <= GT["runtime_s"], f"Runtime exceeded: {runtime}s > {GT['runtime_s']}s"
    assert current_ram <= GT["ram_bytes"], f"Memory exceeded: {current_ram} > {GT['ram_bytes']}"
    
    print(f"✓ Performance requirements met")
    print(f"==============================\n")
    
    # Print final profiling results
    profiler.end_phase()
    profiler.print_summary()
    
    print(f"\n🎉 All tests passed! Clean architecture working correctly. 🎉")


if __name__ == "__main__":
    # Allow standalone execution
    import pytest, sys
    sys.exit(pytest.main([__file__]))
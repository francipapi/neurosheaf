#!/usr/bin/env python3
"""
Parallel DTW Neural Network Comparison Script (Centralized DTW defaults)

This script reproduces the functionality of simple_dtw_comparison.py but:
- Uses the centralized DTW preprocessing/normalization in FiltrationDTW (no per-script transforms)
- Parallelizes per-model analysis (sheaf + spectral) and pairwise DTW computations
- Uses the exact models/architectures and weights referenced in simple_dtw_comparison.py

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv
    python parallel_dtw_comparison.py
"""

import os
import sys
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import concurrent.futures as futures

import numpy as np
import torch
import torch.nn as nn

# Environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Neurosheaf imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils.dtw_similarity import FiltrationDTW
from neurosheaf.utils import load_model
from neurosheaf.sheaf.core.gw_config import GWConfig


# =========================
# Model architectures (copied from simple_dtw_comparison.py)
# =========================
class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        num_hidden_layers: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 1,
        activation_fn_name: str = 'relu',
        output_activation_fn_name: str = 'sigmoid',
        dropout_rate: float = 0.0012
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)

        layers_list = []
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(self.output_activation_fn)
        self.layers = nn.Sequential(*layers_list)

    def _get_activation_fn(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'softmax': nn.Softmax(dim=-1),
            'none': nn.Identity()
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        return activations[name.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ActualCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers[1](self.layers[0](x))
        x = self.layers[4](self.layers[3](self.layers[2](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[7](self.layers[6](self.layers[5](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[10](self.layers[9](self.layers[8](x)))
        x = x.view(-1, 16, 2)
        x = self.layers[13](self.layers[12](self.layers[11](x)))
        x = x.view(x.size(0), -1)
        x = self.layers[15](self.layers[14](x))
        return x


# =========================
# Config
# =========================
CONFIG = {
    'data_batch_size': 100,
    'random_seed': 42,
    'top_n_eigenvalues': 20,
    'interpolation_points': None,
    'num_workers_models': 5,
    'num_workers_pairs': 10,
    'plots_dir': 'model_summaries_parallel',
}

# Model checkpoints (same as simple_dtw_comparison.py)
custom_path = "models/torch_custom_acc_1.0000_epoch_200.pth"
mlp_path = "models/torch_mlp_acc_1.0000_epoch_200.pth"
mlp_path1 = "models/torch_mlp_acc_0.9857_epoch_100.pth"
rand_custom_path = "models/random_custom_net_000_default_seed_42.pth"
rand_mlp_path = "models/random_mlp_net_000_default_seed_42.pth"

MODEL_REGISTRY: Dict[str, Tuple[Any, str]] = {
    'mlp_trained': (MLPModel, mlp_path),
    'mlp_trained_98': (MLPModel, mlp_path1),
    'custom_trained': (ActualCustomModel, custom_path),
    'rand_mlp': (MLPModel, rand_mlp_path),
    'rand_custom': (ActualCustomModel, rand_custom_path),
}

# GW config (copied from simple_dtw_comparison.py)
gw_config = GWConfig(
    epsilon=0.05,
    max_iter=100,
    tolerance=1e-8,
    quasi_sheaf_tolerance=0.08,
    adaptive_epsilon=True,
    base_epsilon=0.05,
    reference_n=50,
    epsilon_scaling_method='sqrt',
    epsilon_min=0.01,
    epsilon_max=0.2,
)


# =========================
# Utilities
# =========================
def filter_top_eigenvalues(evolution: List[torch.Tensor], k: int) -> List[torch.Tensor]:
    filtered = []
    for step_eigenvals in evolution:
        if len(step_eigenvals) > k:
            sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
            filtered.append(sorted_eigenvals[:k])
        else:
            filtered.append(step_eigenvals)
    return filtered


def _rebuild_model(class_name: str) -> nn.Module:
    if class_name == 'MLPModel':
        return MLPModel()
    if class_name == 'ActualCustomModel':
        return ActualCustomModel()
    raise ValueError(f"Unknown model class: {class_name}")


def _analyze_model_worker(args: Tuple[str, str, str, int, Tuple[int, ...], Dict[str, Any], str]) -> Dict[str, Any]:
    """Worker to analyze a single model and return eigenvalue evolution."""
    (
        model_key,
        model_class_name,
        weights_path,
        data_seed,
        data_shape,
        spectral_cfg,
        plots_dir,
    ) = args

    # Deterministic seeds per worker
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)

    # Rebuild model and load weights
    model = _rebuild_model(model_class_name)
    model = load_model(lambda: model, weights_path, device='cpu')  # load_model expects a factory or class

    # Regenerate data deterministically in worker
    batch_size, input_dim = data_shape
    data = 12 * torch.randn(batch_size, input_dim)

    analyzer = NeurosheafAnalyzer(device='cpu')
    result = analyzer.analyze(
        model,
        data,
        method='gromov_wasserstein',
        gw_config=gw_config,
        exclude_final_single_output=True,
    )

    spectral_analyzer = PersistentSpectralAnalyzer(
        default_n_steps=spectral_cfg.get('default_n_steps', 50),
        default_filtration_type=spectral_cfg.get('default_filtration_type', 'threshold'),
    )
    spectral = spectral_analyzer.analyze(
        result['sheaf'], filtration_type='threshold', n_steps=spectral_cfg.get('n_steps', 100)
    )

    # Create per-model summary plots
    try:
        from neurosheaf.visualization import EnhancedVisualizationFactory
        os.makedirs(plots_dir, exist_ok=True)
        model_dir = os.path.join(plots_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)
        vf = EnhancedVisualizationFactory(theme='neurosheaf_default')
        summary_plots = vf.create_analysis_summary(spectral)
        for plot_name, figure in summary_plots.items():
            filename = os.path.join(model_dir, f"{plot_name}.html")
            figure.write_html(filename)
    except Exception:
        pass

    evolution = spectral['persistence_result']['eigenvalue_sequences']
    filtration_params = spectral['filtration_params']

    # Apply consistent top-k filtering if requested
    top_k = spectral_cfg.get('top_k', None)
    if top_k is not None:
        evolution = filter_top_eigenvalues(evolution, top_k)

    # Convert tensors to lightweight lists for cross-process passing
    evolution_serializable: List[List[float]] = []
    for step in evolution:
        evolution_serializable.append([float(x.item()) for x in step])

    return {
        'key': model_key,
        'evolution': evolution_serializable,
        'filtration_params': [float(p) for p in filtration_params],
    }


def _compute_pairwise_worker(args: Tuple[str, Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]) -> Tuple[str, float]:
    """Worker to compute a single pairwise DTW normalized distance."""
    (key1, data1, key2, data2, comparator_cfg) = args

    # Build DTW comparator with centralized defaults
    dtw_comparator = FiltrationDTW(
        method=comparator_cfg.get('method', 'tslearn'),
        constraint_band=comparator_cfg.get('constraint_band', 0.05),
        normalization_scheme=comparator_cfg.get('normalization_scheme', 'average_cost'),
        use_log_scale=comparator_cfg.get('use_log_scale', True),
        use_persistence_weighting=comparator_cfg.get('use_persistence_weighting', True),
        matching_strategy=comparator_cfg.get('matching_strategy', 'correlation'),
        eigen_ordering=comparator_cfg.get('eigen_ordering', 'descending'),
        use_zscore=comparator_cfg.get('use_zscore', True),
    )

    # Convert serialized evolution to torch tensors expected by the API
    evo1: List[torch.Tensor] = [torch.tensor(step, dtype=torch.float64) for step in data1['evolution']]
    evo2: List[torch.Tensor] = [torch.tensor(step, dtype=torch.float64) for step in data2['evolution']]

    res = dtw_comparator.compare_eigenvalue_evolution(
        evo1,
        evo2,
        filtration_params1=data1['filtration_params'],
        filtration_params2=data2['filtration_params'],
        multivariate=True,
        use_interpolation=True,
        match_all_eigenvalues=True,
        interpolation_points=comparator_cfg.get('interpolation_points', CONFIG['interpolation_points']),
    )

    return (f"{key1}_vs_{key2}", float(res['normalized_distance']))


def _build_distance_matrix(keys: List[str], distances: Dict[str, float]) -> np.ndarray:
    n = len(keys)
    M = np.zeros((n, n))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i == j:
                M[i, j] = 0.0
            elif i < j:
                val = distances.get(f"{k1}_vs_{k2}")
                if val is not None:
                    M[i, j] = val
                    M[j, i] = val
    return M


def main() -> int:
    print("🚀 Parallel DTW Neural Network Comparison (Centralized DTW)")
    print("=" * 60)

    # Seeds
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # Prepare synthetic input data (broadcast or regenerate per worker)
    batch_size = CONFIG['data_batch_size']
    input_dim = 3

    # Prepare spectral and comparator configs
    spectral_cfg = {
        'default_n_steps': 50,
        'default_filtration_type': 'threshold',
        'n_steps': 100,
        'top_k': CONFIG['top_n_eigenvalues'],
    }
    comparator_cfg = {
        'method': 'tslearn',
        'constraint_band': 0,
        'normalization_scheme': 'average_cost',
        'use_log_scale': True,
        'use_persistence_weighting': False,
        'matching_strategy': 'correlation',
        'eigen_ordering': 'descending',
        'use_zscore': False,
        'interpolation_points': CONFIG['interpolation_points'],
    }

    # Phase 1: Parallel per-model analysis
    print("\n📥 Analyzing models in parallel...")
    model_args: List[Tuple[str, str, str, int, Tuple[int, ...], Dict[str, Any], str]] = []
    for key, (cls, path) in MODEL_REGISTRY.items():
        class_name = cls.__name__
        model_args.append((key, class_name, path, CONFIG['random_seed'], (batch_size, input_dim), spectral_cfg, CONFIG['plots_dir']))

    analyzed: Dict[str, Dict[str, Any]] = {}
    start = time.time()
    with futures.ProcessPoolExecutor(max_workers=CONFIG['num_workers_models']) as pool:
        results = list(pool.map(_analyze_model_worker, model_args))
    for r in results:
        analyzed[r['key']] = {'evolution': r['evolution'], 'filtration_params': r['filtration_params']}
    print(f"✅ Analyzed {len(analyzed)} models in {time.time()-start:.2f}s")
    print(f"🖼️  Saved per-model summaries to '{CONFIG['plots_dir']}'")

    # Phase 2: Parallel pairwise DTW distances
    print("\n📐 Computing pairwise DTW distances in parallel...")
    keys = list(MODEL_REGISTRY.keys())
    pair_args: List[Tuple[str, Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]] = []
    for i, k1 in enumerate(keys):
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            pair_args.append((k1, analyzed[k1], k2, analyzed[k2], comparator_cfg))

    all_distances: Dict[str, float] = {}
    start = time.time()
    with futures.ProcessPoolExecutor(max_workers=CONFIG['num_workers_pairs']) as pool:
        for pair_key, dist in pool.map(_compute_pairwise_worker, pair_args):
            all_distances[pair_key] = dist
    print(f"✅ Computed {len(all_distances)} distances in {time.time()-start:.2f}s")

    # Phase 3: Reporting
    print("\n🎯 COMPLETE PAIRWISE DISTANCE ANALYSIS")
    print("=" * 60)
    for pair_key in sorted(all_distances.keys()):
        a, b = pair_key.split('_vs_')
        if ('trained' in a and 'rand' not in a) and ('trained' in b and 'rand' not in b):
            category = "🟢 INTRA-TRAINED"
        elif (('trained' in a and 'rand' in b) or ('rand' in a and 'trained' in b)):
            category = "🔴 INTER-GROUP"
        elif ('rand' in a and 'rand' in b):
            category = "🟡 INTRA-RANDOM"
        else:
            category = "❓ OTHER"
        print(f"   {category:15} {a:20} vs {b:20} = {all_distances[pair_key]:8.3f}")

    # Stats by category
    keys_set = set(keys)
    intra_trained = []
    inter_group = []
    intra_random = []

    def add_if(k):
        if k in all_distances:
            return all_distances[k]
        return None

    # Build lists
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                k = f"{a}_vs_{b}"
                val = all_distances.get(k)
                if val is None:
                    continue
                if ('trained' in a and 'rand' not in a) and ('trained' in b and 'rand' not in b):
                    intra_trained.append(val)
                elif (('trained' in a and 'rand' in b) or ('rand' in a and 'trained' in b)):
                    inter_group.append(val)
                elif ('rand' in a and 'rand' in b):
                    intra_random.append(val)

    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {'count': 0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        return {
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'std': float(np.std(arr)),
        }

    print("\n📊 DISTANCE STATISTICS BY CATEGORY")
    print("=" * 60)
    s_intra_tr = stats(intra_trained)
    s_inter = stats(inter_group)
    s_intra_r = stats(intra_random)
    print(f"   🟢 Intra-Trained: {s_intra_tr}")
    print(f"   🔴 Inter-Group:  {s_inter}")
    print(f"   🟡 Intra-Random: {s_intra_r}")

    separation_ratio = (s_inter['mean'] / s_intra_tr['mean']) if s_intra_tr['mean'] > 0 else float('inf')
    print(f"\n🏆 SEPARATION RATIO: {separation_ratio:.2f}x")

    # Distance matrix
    print("\n📋 COMPLETE DISTANCE MATRIX")
    print("=" * 60)
    M = _build_distance_matrix(keys, all_distances)
    header = "Model".ljust(20) + ''.join([f"{k[:12]:>12}" for k in keys])
    print(header)
    print("-" * len(header))
    for i, k1 in enumerate(keys):
        row = f"{k1[:20]:20}" + ''.join([f"{M[i,j]:12.3f}" if i == j or M[i,j] != 0.0 else f"{'---':>12}" for j in range(len(keys))])
        print(row)

    # Configuration echo
    print("\n📋 Configuration Used:")
    print(f"   Batch size: {CONFIG['data_batch_size']}")
    print(f"   Top eigenvalues: {CONFIG['top_n_eigenvalues']}")
    print(f"   Interpolation points: {CONFIG['interpolation_points']}")
    print(f"   DTW method: tslearn")
    print(f"   Constraint band: {comparator_cfg['constraint_band']}")
    print("\n✅ Parallel DTW analysis complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""
DTW Parameter Optimizer (centralized DTW, cached evolutions, parallel evaluation)

This script performs a staged grid/random search to find parameters
that maximize separation between trained and random models while keeping runtime reasonable.

Key improvements:
- Uses centralized DTW preprocessing/normalization (FiltrationDTW) with new defaults
- Caches per-spectral eigenvalue evolutions to avoid recomputation across configs
- Parallelizes across configurations using cached evolutions
- Adds comparator parameters (constraint_band, normalization_scheme, use_zscore, interpolation_points)

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv
    python dtw_parameter_optimizer.py [--quick] [--no-parallel]
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from tqdm import tqdm

# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Neurosheaf imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils.dtw_similarity import FiltrationDTW
from neurosheaf.utils import load_model
from neurosheaf.sheaf.core.gw_config import GWConfig

# Local copies of model architectures to avoid side effects from importing scripts
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
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'softmax': nn.Softmax(dim=-1),
            'none': nn.Identity()
        }
        act = activations[activation_fn_name]
        out_act = activations[output_activation_fn_name]
        layers_list = []
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(act)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(act)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(out_act)
        self.layers = nn.Sequential(*layers_list)
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


def filter_top_eigenvalues(evolution: List[torch.Tensor], k: int) -> List[torch.Tensor]:
    filtered = []
    for step_eigenvals in evolution:
        if len(step_eigenvals) > k:
            sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
            filtered.append(sorted_eigenvals[:k])
        else:
            filtered.append(step_eigenvals)
    return filtered


def _compute_pairwise_distances_with_config(
    evolutions_serialized: Dict[str, Dict[str, Any]],
    comparator_cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], float, float]:
    """Compute all pairwise DTW distances and separation using cached evolutions.
    evolutions_serialized: { key: { 'evolution': List[List[float]], 'filtration_params': List[float] } }
    Returns: (distances dict, mean_intra_trained, mean_inter_group)
    """
    # Build comparator
    dtw_comparator = FiltrationDTW(
        method=comparator_cfg.get('method', 'tslearn'),
        constraint_band=comparator_cfg.get('constraint_band', 0.05),
        normalization_scheme=comparator_cfg.get('normalization_scheme', 'average_cost'),
        use_log_scale=comparator_cfg.get('use_log_scale', True),
        use_persistence_weighting=comparator_cfg.get('use_persistence_weighting', True),
        matching_strategy=comparator_cfg.get('matching_strategy', 'correlation'),
        eigen_ordering=comparator_cfg.get('eigen_ordering', 'descending'),
        use_zscore=comparator_cfg.get('use_zscore', True),
        min_eigenvalue_threshold=comparator_cfg.get('min_eigenvalue_threshold', 1e-12),
    )

    keys = list(evolutions_serialized.keys())
    distances: Dict[str, float] = {}

    # Convert serialized to tensors once
    tensors: Dict[str, Dict[str, Any]] = {}
    for k, payload in evolutions_serialized.items():
        evo = [torch.tensor(step, dtype=torch.float64) for step in payload['evolution']]
        tensors[k] = {
            'evolution': evo,
            'filtration_params': payload['filtration_params'],
        }

    for i, k1 in enumerate(keys):
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            res = dtw_comparator.compare_eigenvalue_evolution(
                tensors[k1]['evolution'],
                tensors[k2]['evolution'],
                filtration_params1=tensors[k1]['filtration_params'],
                filtration_params2=tensors[k2]['filtration_params'],
                multivariate=True,
                use_interpolation=True,
                match_all_eigenvalues=True,
                interpolation_points=comparator_cfg.get('interpolation_points', 100),
            )
            distances[f"{k1}_vs_{k2}"] = float(res['normalized_distance'])

    # Categorize distances
    intra_trained: List[float] = []
    inter_group: List[float] = []
    for key, dist in distances.items():
        a, b = key.split('_vs_')
        is_a_trained = ('trained' in a and 'rand' not in a)
        is_b_trained = ('trained' in b and 'rand' not in b)
        is_a_rand = 'rand' in a
        is_b_rand = 'rand' in b
        if is_a_trained and is_b_trained:
            intra_trained.append(dist)
        elif (is_a_trained and is_b_rand) or (is_a_rand and is_b_trained):
            inter_group.append(dist)

    mean_intra = float(np.mean(intra_trained)) if intra_trained else 0.0
    mean_inter = float(np.mean(inter_group)) if inter_group else 0.0
    return distances, mean_intra, mean_inter


def _evaluate_config_job(args: Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Top-level worker to evaluate a single config with provided cached evolutions."""
    cfg, evol_ser = args
    try:
        comparator_cfg = {
            'method': 'tslearn',
            'constraint_band': cfg['constraint_band'],
            'normalization_scheme': cfg['normalization_scheme'],
            'use_log_scale': True,
            'use_persistence_weighting': bool(cfg.get('use_persistence_weighting', True)),
            'matching_strategy': 'correlation',
            'eigen_ordering': 'descending',
            'use_zscore': bool(cfg['use_zscore']),
            'min_eigenvalue_threshold': float(cfg['min_eigenvalue_threshold']),
            'interpolation_points': int(cfg['interpolation_points']),
        }
        distances, mean_intra, mean_inter = _compute_pairwise_distances_with_config(evol_ser, comparator_cfg)
        separation = (mean_inter / mean_intra) if mean_intra > 0 else 0.0
        return {
            'config': cfg.copy(),
            'separation_ratio': separation,
            'mean_intra_trained': mean_intra,
            'mean_inter_group': mean_inter,
            'num_pairs': len(distances),
        }
    except Exception as e:
        return {'config': cfg.copy(), 'separation_ratio': 0.0, 'error': str(e)}


def _analyze_models_for_spectral_key(
    spectral_key: Tuple[int, float, int, float],
    model_paths: Dict[str, str],
    top_k: int,
    base_seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Compute and serialize evolutions for a given spectral config.
    spectral_key = (batch_size, data_scaling, n_steps, gw_epsilon)
    Returns { key: { 'evolution': List[List[float]], 'filtration_params': List[float] } }
    """
    batch_size, data_scaling, n_steps, gw_eps = spectral_key

    # Seeds
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    # Generate data
    data = data_scaling * torch.randn(batch_size, 3)

    # GW config
    gw_config = GWConfig(
        epsilon=gw_eps,
        max_iter=100,
        tolerance=1e-8,
        adaptive_epsilon=True,
        base_epsilon=gw_eps,
        reference_n=batch_size,
        epsilon_scaling_method='sqrt',
        epsilon_min=0.01,
        epsilon_max=0.2,
    )

    # Load models on CPU
    models: Dict[str, nn.Module] = {}
    for key, path in model_paths.items():
        if 'custom' in key:
            models[key] = load_model(ActualCustomModel, path, device='cpu')
        else:
            models[key] = load_model(MLPModel, path, device='cpu')

    # Analyze and serialize evolutions
    evolutions_serialized: Dict[str, Dict[str, Any]] = {}
    for model_key, model in models.items():
        analyzer = NeurosheafAnalyzer(device='cpu')
        result = analyzer.analyze(
            model,
            data,
            method='gromov_wasserstein',
            gw_config=gw_config,
            exclude_final_single_output=True,
        )
        spectral_analyzer = PersistentSpectralAnalyzer(
            default_n_steps=n_steps,
            default_filtration_type='threshold',
        )
        spectral = spectral_analyzer.analyze(
            result['sheaf'], filtration_type='threshold', n_steps=n_steps
        )
        evolution = spectral['persistence_result']['eigenvalue_sequences']
        # Apply top-k filtering
        evolution = filter_top_eigenvalues(evolution, top_k)
        # Serialize to Python lists of floats
        ev_ser: List[List[float]] = []
        for step in evolution:
            ev_ser.append([float(x.item()) for x in step])
        evolutions_serialized[model_key] = {
            'evolution': ev_ser,
            'filtration_params': [float(p) for p in spectral['filtration_params']],
        }

    return evolutions_serialized


class DTWParameterOptimizer:
    """Systematic parameter optimization for DTW separation with caching."""
    
    def __init__(self, quick_mode=False, parallel=True):
        self.quick_mode = quick_mode
        self.parallel = parallel
        self.results = []
        self.best_config = None
        self.best_separation = 0.0

        # Fixed model paths matching simple_dtw_comparison.py
        self.model_paths = {
            'mlp_trained': 'models/torch_mlp_acc_1.0000_epoch_200.pth',
            'mlp_trained_98': 'models/torch_mlp_acc_0.9857_epoch_100.pth',
            'custom_trained': 'models/torch_custom_acc_1.0000_epoch_200.pth',
            'rand_mlp': 'models/random_mlp_net_000_default_seed_42.pth',
            'rand_custom': 'models/random_custom_net_000_default_seed_42.pth',
        }

        # Search space
        if quick_mode:
            self.param_grid = {
                # Spectral/data (coarse)
                'batch_size': [50],
                'data_scaling': [8, 12],
                'n_steps': [100],
                'gw_epsilon': [0.05],
                'top_eigenvalues': [15, 20],
                # Comparator (high impact)
                'interpolation_points': [100, 125],
                'constraint_band': [0.05, 0.1],
                'normalization_scheme': ['average_cost'],
                'use_zscore': [True, False],
                'use_persistence_weighting': [True],
                'min_eigenvalue_threshold': [1e-15],
            }
        else:
            self.param_grid = {
                # Spectral/data (fixed/narrow to exploit caching)
                'batch_size': [50, 75],
                'data_scaling': [8, 12],
                'n_steps': [100],
                'gw_epsilon': [0.05],
                'top_eigenvalues': [15, 20, 25],
                # Comparator (wider)
                'interpolation_points': [100, 125, 150],
                'constraint_band': [0.02, 0.05, 0.1],
                'normalization_scheme': ['average_cost'],
                'use_zscore': [True, False],
                'use_persistence_weighting': [True],
                'min_eigenvalue_threshold': [1e-16, 1e-15, 1e-12],
            }

    def _generate_configurations(self) -> List[Dict[str, Any]]:
        names = list(self.param_grid.keys())
        values = [self.param_grid[n] for n in names]
        configs: List[Dict[str, Any]] = []
        for vals in itertools.product(*values):
            configs.append(dict(zip(names, vals)))
        return configs

    def _spectral_key(self, cfg: Dict[str, Any]) -> Tuple[int, float, int, float]:
        return (
            int(cfg['batch_size']),
            float(cfg['data_scaling']),
            int(cfg['n_steps']),
            float(cfg['gw_epsilon']),
        )

    def optimize(self) -> Dict[str, Any]:
        print("🚀 Starting DTW Parameter Optimization (centralized DTW, cached evolutions)")
        print(f"   Mode: {'Quick' if self.quick_mode else 'Full'} | Parallel: {self.parallel}")

        configs = self._generate_configurations()
        print(f"   Testing {len(configs)} configurations")

        # Build spectral cache keys
        spectral_keys = sorted({self._spectral_key(c) for c in configs})
        print(f"   Unique spectral settings to compute: {len(spectral_keys)}")

        # Compute evolutions for each spectral key (parallel)
        spectral_cache: Dict[Tuple[int, float, int, float], Dict[str, Dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=min(8, len(spectral_keys))) as pool:
            fut_to_key = {
                pool.submit(_analyze_models_for_spectral_key, sk, self.model_paths, int(configs[0]['top_eigenvalues'])): sk
                for sk in spectral_keys
            }
            for fut in tqdm(as_completed(fut_to_key), total=len(fut_to_key), desc="Analyzing spectral sets"):
                sk = fut_to_key[fut]
                try:
                    spectral_cache[sk] = fut.result()
                except Exception as e:
                    print(f"   ❌ Failed spectral analysis for {sk}: {e}")

        # Build jobs with cached evolutions for process-safe evaluation
        jobs: List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]] = []
        for cfg in configs:
            sk = self._spectral_key(cfg)
            jobs.append((cfg, spectral_cache[sk]))

        self.results = []
        if self.parallel:
            with ProcessPoolExecutor(max_workers=8) as pool:
                for res in tqdm(pool.map(_evaluate_config_job, jobs), total=len(jobs), desc="Evaluating configs"):
                    self.results.append(res)
                    if res.get('separation_ratio', 0.0) > self.best_separation:
                        self.best_separation = res['separation_ratio']
                        self.best_config = res['config']
        else:
            for job in tqdm(jobs, desc="Evaluating configs"):
                res = _evaluate_config_job(job)
                self.results.append(res)
                if res.get('separation_ratio', 0.0) > self.best_separation:
                    self.best_separation = res['separation_ratio']
                    self.best_config = res['config']

        # Sort results by separation ratio
        self.results.sort(key=lambda x: x.get('separation_ratio', 0.0), reverse=True)
        return {
            'best_config': self.best_config,
            'best_separation': self.best_separation,
            'top_10_configs': self.results[:10],
            'all_results': self.results,
        }

    def save_results(self, output_dir: str = "."):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'optimal_dtw_config.json', 'w') as f:
            json.dump({
                'config': self.best_config,
                'separation_ratio': self.best_separation,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        with open(output_dir / 'dtw_optimization_results.json', 'w') as f:
            json.dump({
                'best_config': self.best_config,
                'best_separation': self.best_separation,
                'top_10_configs': self.results[:10] if len(self.results) >= 10 else self.results,
                'num_configs_tested': len(self.results),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        self.generate_html_report(output_dir / 'dtw_optimization_report.html')
        print(f"\n📁 Results saved: optimal_dtw_config.json, dtw_optimization_results.json, dtw_optimization_report.html")

    def generate_html_report(self, filepath: Path):
        # Minimal changes to the original report; reuse separation metrics
        html_content = f"""
<!DOCTYPE html>
<html>
<head><title>DTW Parameter Optimization Report</title></head>
<body>
<h1>DTW Parameter Optimization Report</h1>
<p>Configurations tested: {len(self.results)}</p>
<p>Best separation ratio: {self.best_separation:.2f}x</p>
<h2>Top 10 Configurations</h2>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Rank</th><th>Separation</th><th>Constraint</th><th>Norm</th><th>Z-Score</th><th>Interp</th><th>TopK</th></tr>
"""
        for i, r in enumerate(self.results[:10], 1):
            c = r['config']
            html_content += f"<tr><td>{i}</td><td>{r.get('separation_ratio', 0):.2f}</td><td>{c.get('constraint_band')}</td><td>{c.get('normalization_scheme')}</td><td>{c.get('use_zscore')}</td><td>{c.get('interpolation_points')}</td><td>{c.get('top_eigenvalues')}</td></tr>\n"
        html_content += "</table></body></html>"
        with open(filepath, 'w') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description='DTW Parameter Optimizer')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer parameter combinations')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    args = parser.parse_args()

    optimizer = DTWParameterOptimizer(quick_mode=args.quick, parallel=not args.no_parallel)
    results = optimizer.optimize()
    optimizer.save_results()

    print(f"\n{'='*60}")
    print("🎯 OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best Separation Ratio: {results['best_separation']:.2f}x")
    print("Best Configuration:")
    if results['best_config']:
        for param, value in results['best_config'].items():
            print(f"  {param}: {value}")
    else:
        print("  None")
    if results['best_separation'] >= 10:
        print("\n✅ SUCCESS: Achieved target 10x+ separation!")
    else:
        print(f"\n⚠️  Current best is {results['best_separation']:.2f}x, target is 10x+")
        print("   Consider expanding comparator ranges or running non-quick mode.")


if __name__ == "__main__":
    main()
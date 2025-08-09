#!/usr/bin/env python3
"""
Parallel DTW Parameter Optimizer

This script systematically optimizes DTW parameters for maximum separation between
trained and random neural networks using parallel processing.

Target: Beat current best separation ratio of 4.607x by testing previously untested parameters

NEW FOCUS: Untested high-impact parameters with potential for improvement:
- eigen_ordering: 'ascending' (may be better for eigenvalue collapse detection)  
- Higher eigenvalue counts: 30-50 (capture more structural information)
- min_eigenvalue_threshold: 1e-14 (user requested, finer precision)
- Extended constraint_band: 0.12-0.25 (more temporal flexibility)
- eigenvalue_weight/structural_weight: variations (emphasis tuning)
- More interpolation points: 175-300 (higher temporal detail)

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv
    python parallel_parameter_optimizer.py [--mode coarse|fine|advanced|validate] [--max-configs 200]

Modes:
- coarse: Focus on untested high-impact parameters
- fine: Fine-tune around promising untested areas  
- advanced: Test highest-impact untested parameters (eigen_ordering, high eigenvalue counts)
- validate: Validate best configurations with both orderings
"""

import os
import sys
import json
import time
import math
import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import concurrent.futures as futures
from itertools import product
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Neurosheaf imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils.dtw_similarity import FiltrationDTW
from neurosheaf.utils import load_model
from neurosheaf.sheaf.core.gw_config import GWConfig

# Import model classes
from parallel_dtw_comparison import MLPModel, ActualCustomModel, filter_top_eigenvalues, _rebuild_model


class ModelAnalysisCache:
    """Intelligent caching system for model analyses."""
    
    def __init__(self, cache_dir: str = "cache/model_analysis"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _hash_config(self, model_key: str, spectral_config: Dict[str, Any], 
                     data_config: Dict[str, Any]) -> str:
        """Generate hash for configuration."""
        config_str = f"{model_key}_{json.dumps(spectral_config, sort_keys=True)}_{json.dumps(data_config, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get(self, model_key: str, spectral_config: Dict[str, Any], 
            data_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis if available."""
        cache_key = self._hash_config(model_key, spectral_config, data_config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                cache_file.unlink(missing_ok=True)
        return None
    
    def set(self, model_key: str, spectral_config: Dict[str, Any], 
            data_config: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache analysis result."""
        cache_key = self._hash_config(model_key, spectral_config, data_config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    def clear(self) -> None:
        """Clear all cached analyses."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


class ParameterGrid:
    """Smart parameter grid generator with priorities."""
    
    def __init__(self, mode: str = 'coarse'):
        self.mode = mode
        
        # Define parameter search spaces
        if mode == 'coarse':
            self.param_space = {
                # High impact parameters - focus on untested values
                'constraint_band': [0.12, 0.15, 0.18, 0.2, 0.25],  # UNTESTED: Higher than optimal 0.1
                'normalization_scheme': ['mad_aware', 'std_aware'],  # UNTESTED: Alternative schemes
                'use_log_scale': [True],  # Keep optimal
                'use_zscore': [False],    # Keep optimal  
                'use_persistence_weighting': [True],  # Keep optimal
                'matching_strategy': ['correlation'],  # Keep optimal
                'eigen_ordering': ['ascending'],  # UNTESTED: Opposite of default
                
                # Medium impact parameters - extend ranges
                'top_n_eigenvalues': [30, 35, 40, 50],  # UNTESTED: Higher counts
                'interpolation_points': [175, 200, 250, 300],  # UNTESTED: More temporal detail
                'min_eigenvalue_threshold': [1e-16, 1e-14, 1e-10],  # UNTESTED: Including 1e-14
                'eigenvalue_weight': [0.8, 0.9, 1.1, 1.2],  # UNTESTED: Weight variations
                'structural_weight': [0.1, 0.2, 0.3],  # UNTESTED: Include structural info
                
                # Keep optimal values for these
                'data_batch_size': [75],     # Optimal value
                'data_scaling': [12],        # Optimal value  
                'spectral_n_steps': [100]    # Optimal value
            }
        elif mode == 'fine':
            # Fine-grained exploration of untested high-impact parameters
            self.param_space = {
                # Fine-tune untested constraint bands around optimal 0.1
                'constraint_band': [0.08, 0.09, 0.11, 0.12, 0.13, 0.14],  # UNTESTED: Near-optimal range
                'normalization_scheme': ['average_cost'],  # Keep optimal
                'use_log_scale': [True],  # Keep optimal
                'use_zscore': [False],    # Keep optimal
                'use_persistence_weighting': [True],  # Keep optimal
                'matching_strategy': ['correlation'],  # Keep optimal
                'eigen_ordering': ['ascending', 'descending'],  # UNTESTED: Both orderings
                
                # Fine-tune eigenvalue parameters
                'top_n_eigenvalues': [15, 20, 25, 28, 32, 35],  # Mix of tested and untested
                'interpolation_points': [125, 150, 175, 200],   # Extend beyond tested
                'min_eigenvalue_threshold': [1e-15, 1e-14, 1e-13, 1e-12],  # Include requested 1e-14
                'eigenvalue_weight': [0.9, 1.0, 1.05, 1.1],    # UNTESTED: Fine variations
                'structural_weight': [0.0, 0.05, 0.1, 0.15],   # UNTESTED: Gradual structural weight
                
                # Keep optimal values
                'data_batch_size': [75],     # Optimal
                'data_scaling': [12],        # Optimal
                'spectral_n_steps': [100]    # Optimal
            }
        elif mode == 'advanced':
            # Focus on highest-impact untested parameters with potential for >4.607x separation  
            self.param_space = {
                # Test eigen_ordering - potentially major impact on eigenvalue collapse detection
                'eigen_ordering': ['ascending'],  # UNTESTED: May be better for trained model collapse patterns
                'constraint_band': [0.1, 0.12, 0.15],  # Extend beyond tested optimal
                'normalization_scheme': ['average_cost'],  # Keep optimal
                'use_log_scale': [True],           # Keep optimal
                'use_zscore': [False],             # Keep optimal  
                'use_persistence_weighting': [True],  # Keep optimal
                'matching_strategy': ['correlation'], # Keep optimal
                
                # Test higher eigenvalue counts - capture more structure
                'top_n_eigenvalues': [35, 40, 50],    # UNTESTED: Much higher counts
                'interpolation_points': [200, 250],   # UNTESTED: High temporal detail
                'min_eigenvalue_threshold': [1e-14],  # UNTESTED: User requested value
                'eigenvalue_weight': [1.1, 1.2],     # UNTESTED: Emphasize eigenvalue patterns
                'structural_weight': [0.0],          # Keep structural weight minimal
                
                # Keep all optimal values
                'data_batch_size': [75],
                'data_scaling': [12], 
                'spectral_n_steps': [100]
            }
        else:  # validate
            # Test specific high-performing configurations
            self.param_space = {
                'constraint_band': [0.1, 0.12],      # Around optimal
                'normalization_scheme': ['average_cost'], 
                'use_log_scale': [True],
                'use_zscore': [False],                # Optimal value
                'use_persistence_weighting': [True],
                'matching_strategy': ['correlation'],
                'eigen_ordering': ['descending', 'ascending'],  # Test both
                
                'top_n_eigenvalues': [15, 20, 25],
                'interpolation_points': [125, 150],
                'min_eigenvalue_threshold': [1e-14, 1e-12],
                
                'data_batch_size': [75],              # Optimal
                'data_scaling': [12],                 # Optimal
                'spectral_n_steps': [100]             # Optimal
            }
    
    def generate_combinations(self, max_combinations: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate parameter combinations with smart ordering."""
        # Get all parameter combinations
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        
        combinations = []
        for combo in product(*values):
            config = dict(zip(keys, combo))
            combinations.append(config)
        
        # Smart ordering: prioritize configurations likely to perform well
        combinations = self._prioritize_combinations(combinations)
        
        if max_combinations and len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
            
        print(f"Generated {len(combinations)} parameter combinations for {self.mode} mode")
        return combinations
    
    def _prioritize_combinations(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order combinations by expected performance, prioritizing untested high-impact parameters."""
        def priority_score(config: Dict[str, Any]) -> float:
            score = 0.0
            
            # HIGH PRIORITY: Untested eigen_ordering (potentially major impact)
            if config.get('eigen_ordering') == 'ascending':
                score += 25.0  # Major untested parameter
            
            # HIGH PRIORITY: Higher eigenvalue counts (capture more structure)
            n_eigen = config.get('top_n_eigenvalues', 15)
            if n_eigen >= 30:
                score += 20.0  # Untested high counts
            elif n_eigen >= 25:
                score += 15.0  # Moderately high counts
            
            # HIGH PRIORITY: Requested threshold value 1e-14
            threshold = config.get('min_eigenvalue_threshold', 1e-12)
            if threshold == 1e-14:
                score += 18.0  # User specifically requested
            elif threshold == 1e-16:
                score += 12.0  # Even finer precision
                
            # HIGH PRIORITY: More interpolation points (untested)  
            interp_points = config.get('interpolation_points', 125)
            if interp_points >= 200:
                score += 15.0  # High temporal detail
            elif interp_points >= 175:
                score += 10.0  # Medium-high detail
            
            # MEDIUM PRIORITY: Extended constraint bands
            cb = config.get('constraint_band', 0.1)
            if 0.12 <= cb <= 0.18:
                score += 12.0  # Untested range beyond optimal
            elif cb == 0.1:
                score += 8.0   # Known optimal
                
            # MEDIUM PRIORITY: Eigenvalue weight variations (untested)
            ev_weight = config.get('eigenvalue_weight', 1.0)
            if ev_weight != 1.0:
                score += 10.0  # Any variation is untested
            
            # MEDIUM PRIORITY: Structural weight inclusion (untested)
            struct_weight = config.get('structural_weight', 0.0)
            if 0.05 <= struct_weight <= 0.2:
                score += 8.0   # Moderate structural information
            
            # MEDIUM PRIORITY: Alternative normalization schemes
            if config.get('normalization_scheme') in ['mad_aware', 'std_aware']:
                score += 7.0   # Untested alternatives
            elif config.get('normalization_scheme') == 'average_cost':
                score += 5.0   # Known optimal
            
            # BASELINE: Keep proven optimal settings
            if config.get('use_log_scale', False):
                score += 4.0
            if config.get('use_persistence_weighting', False):
                score += 4.0
            if config.get('matching_strategy') == 'correlation':
                score += 4.0
            if config.get('use_zscore', True) == False:  # Optimal is False
                score += 3.0
                
            # Add small randomization to break ties
            score += np.random.uniform(-0.5, 0.5)
            
            return score
        
        # Sort by priority score (descending)
        combinations.sort(key=priority_score, reverse=True)
        return combinations


class ParallelParameterOptimizer:
    """Main optimizer class."""
    
    def __init__(self, mode: str = 'coarse', max_configs: int = 100, 
                 num_workers_models: int = 5, num_workers_configs: int = 3):
        self.mode = mode
        self.max_configs = max_configs
        self.num_workers_models = num_workers_models
        self.num_workers_configs = num_workers_configs
        
        # Initialize components
        self.cache = ModelAnalysisCache()
        self.param_grid = ParameterGrid(mode)
        
        # Results tracking
        self.results = []
        self.best_config = None
        self.best_separation = 0.0
        
        # Model registry (same as parallel_dtw_comparison.py)
        self.model_registry = {
            'mlp_trained': (MLPModel, "models/torch_mlp_acc_1.0000_epoch_200.pth"),
            'mlp_trained_98': (MLPModel, "models/torch_mlp_acc_0.9857_epoch_100.pth"),
            'custom_trained': (ActualCustomModel, "models/torch_custom_acc_1.0000_epoch_200.pth"),
            'rand_mlp': (MLPModel, "models/random_mlp_net_000_default_seed_42.pth"),
            'rand_custom': (ActualCustomModel, "models/random_custom_net_000_default_seed_42.pth"),
        }
        
        print(f"🚀 ParallelParameterOptimizer initialized:")
        print(f"   Mode: {mode}")
        print(f"   Max configs: {max_configs}")
        print(f"   Workers: {num_workers_models} (models), {num_workers_configs} (configs)")
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run the complete optimization process."""
        print(f"\n{'='*60}")
        print(f"🎯 STARTING {self.mode.upper()} PARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Generate parameter combinations
        combinations = self.param_grid.generate_combinations(self.max_configs)
        
        # Run optimization with progress tracking
        results = self._optimize_parallel(combinations)
        
        # Analyze and rank results
        analysis = self._analyze_results(results)
        
        # Save results
        self._save_results(analysis)
        
        total_time = time.time() - start_time
        print(f"\n✅ Optimization complete in {total_time:.1f}s")
        print(f"🏆 Best separation: {self.best_separation:.2f}x")
        
        return analysis
    
    def _optimize_parallel(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run parallel optimization across parameter combinations."""
        print(f"\n📊 Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        # Use ProcessPoolExecutor for parameter combinations
        with futures.ProcessPoolExecutor(max_workers=self.num_workers_configs) as executor:
            # Submit all combinations
            future_to_config = {
                executor.submit(self._test_single_configuration, config): config 
                for config in combinations
            }
            
            # Process completed futures with progress bar
            with tqdm(total=len(combinations), desc="Testing configs") as pbar:
                for future in futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update best configuration tracking
                        separation = result['separation_ratio']
                        if separation > self.best_separation:
                            self.best_separation = separation
                            self.best_config = config
                            
                        # Early stopping for very poor configurations
                        if separation < 1.5 and len(results) > 20:
                            pbar.set_postfix({'best': f'{self.best_separation:.2f}x', 'current': f'{separation:.2f}x'})
                        else:
                            pbar.set_postfix({'best': f'{self.best_separation:.2f}x', 'current': f'{separation:.2f}x'})
                            
                    except Exception as e:
                        print(f"❌ Configuration failed: {e}")
                        
                    pbar.update(1)
        
        return results
    
    def _test_single_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single parameter configuration."""
        config_start = time.time()
        
        try:
            # Extract configuration components
            data_config = {
                'batch_size': config['data_batch_size'],
                'data_scaling': config['data_scaling'],
                'random_seed': 42
            }
            
            spectral_config = {
                'n_steps': config['spectral_n_steps'],
                'top_k': config['top_n_eigenvalues']
            }
            
            dtw_config = {
                'constraint_band': config['constraint_band'],
                'normalization_scheme': config['normalization_scheme'],
                'use_log_scale': config['use_log_scale'],
                'use_zscore': config['use_zscore'],
                'use_persistence_weighting': config['use_persistence_weighting'],
                'matching_strategy': config['matching_strategy'],
                'interpolation_points': config['interpolation_points'],
                'min_eigenvalue_threshold': config['min_eigenvalue_threshold'],
                'eigen_ordering': config.get('eigen_ordering', 'descending'),
                'eigenvalue_weight': config.get('eigenvalue_weight', 1.0),
                'structural_weight': config.get('structural_weight', 0.0)
            }
            
            # Analyze all models (with caching)
            analyzed_models = self._analyze_models_cached(spectral_config, data_config)
            
            # Compute DTW distances
            distances = self._compute_dtw_distances(analyzed_models, dtw_config)
            
            # Calculate separation metrics
            metrics = self._calculate_separation_metrics(distances)
            
            config_time = time.time() - config_start
            
            return {
                'config': config,
                'data_config': data_config,
                'spectral_config': spectral_config,
                'dtw_config': dtw_config,
                'metrics': metrics,
                'separation_ratio': metrics['separation_ratio'],
                'computation_time': config_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'config': config,
                'error': str(e),
                'separation_ratio': 0.0,
                'computation_time': time.time() - config_start,
                'success': False
            }
    
    def _analyze_models_cached(self, spectral_config: Dict[str, Any], 
                             data_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze all models with caching."""
        analyzed = {}
        
        for model_key, (model_class, model_path) in self.model_registry.items():
            # Check cache first
            cached_result = self.cache.get(model_key, spectral_config, data_config)
            
            if cached_result is not None:
                analyzed[model_key] = cached_result
            else:
                # Analyze model
                result = self._analyze_single_model(model_key, model_class, model_path, 
                                                 spectral_config, data_config)
                analyzed[model_key] = result
                
                # Cache the result
                self.cache.set(model_key, spectral_config, data_config, result)
        
        return analyzed
    
    def _analyze_single_model(self, model_key: str, model_class: type, model_path: str,
                            spectral_config: Dict[str, Any], 
                            data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single model."""
        # Set deterministic seeds
        torch.manual_seed(data_config['random_seed'])
        np.random.seed(data_config['random_seed'])
        
        # Load model
        model = _rebuild_model(model_class.__name__)
        model = load_model(lambda: model, model_path, device='cpu')
        
        # Generate data
        batch_size = data_config['batch_size']
        data_scaling = data_config['data_scaling']
        data = data_scaling * torch.randn(batch_size, 3)
        
        # GW configuration
        gw_config = GWConfig(
            epsilon=0.05,
            max_iter=100,
            tolerance=1e-8,
            adaptive_epsilon=True
        )
        
        # Analyze with neurosheaf
        analyzer = NeurosheafAnalyzer(device='cpu')
        result = analyzer.analyze(
            model, data,
            method='gromov_wasserstein',
            gw_config=gw_config,
            exclude_final_single_output=True
        )
        
        # Spectral analysis
        spectral_analyzer = PersistentSpectralAnalyzer(
            default_n_steps=50,
            default_filtration_type='threshold'
        )
        spectral_result = spectral_analyzer.analyze(
            result['sheaf'], 
            filtration_type='threshold',
            n_steps=spectral_config['n_steps']
        )
        
        evolution = spectral_result['persistence_result']['eigenvalue_sequences']
        filtration_params = spectral_result['filtration_params']
        
        # Apply top-k filtering
        if spectral_config.get('top_k'):
            evolution = filter_top_eigenvalues(evolution, spectral_config['top_k'])
        
        # Convert to serializable format
        evolution_serializable = []
        for step in evolution:
            evolution_serializable.append([float(x.item()) for x in step])
        
        return {
            'evolution': evolution_serializable,
            'filtration_params': [float(p) for p in filtration_params]
        }
    
    def _compute_dtw_distances(self, analyzed_models: Dict[str, Dict[str, Any]], 
                             dtw_config: Dict[str, Any]) -> Dict[str, float]:
        """Compute all pairwise DTW distances."""
        # Create DTW comparator
        comparator = FiltrationDTW(
            method='tslearn',
            constraint_band=dtw_config['constraint_band'],
            normalization_scheme=dtw_config['normalization_scheme'],
            use_log_scale=dtw_config['use_log_scale'],
            use_persistence_weighting=dtw_config['use_persistence_weighting'],
            matching_strategy=dtw_config['matching_strategy'],
            use_zscore=dtw_config['use_zscore'],
            min_eigenvalue_threshold=dtw_config['min_eigenvalue_threshold'],
            eigen_ordering=dtw_config.get('eigen_ordering', 'descending'),
            eigenvalue_weight=dtw_config.get('eigenvalue_weight', 1.0),
            structural_weight=dtw_config.get('structural_weight', 0.0)
        )
        
        distances = {}
        model_keys = list(analyzed_models.keys())
        
        for i, key1 in enumerate(model_keys):
            for j in range(i + 1, len(model_keys)):
                key2 = model_keys[j]
                
                # Convert to tensors
                evo1 = [torch.tensor(step, dtype=torch.float64) 
                       for step in analyzed_models[key1]['evolution']]
                evo2 = [torch.tensor(step, dtype=torch.float64) 
                       for step in analyzed_models[key2]['evolution']]
                
                # Compute DTW distance
                result = comparator.compare_eigenvalue_evolution(
                    evo1, evo2,
                    filtration_params1=analyzed_models[key1]['filtration_params'],
                    filtration_params2=analyzed_models[key2]['filtration_params'],
                    multivariate=True,
                    use_interpolation=True,
                    match_all_eigenvalues=True,
                    interpolation_points=dtw_config['interpolation_points']
                )
                
                distances[f"{key1}_vs_{key2}"] = result['normalized_distance']
        
        return distances
    
    def _calculate_separation_metrics(self, distances: Dict[str, float]) -> Dict[str, Any]:
        """Calculate separation metrics from distance matrix."""
        trained_vs_trained = []
        trained_vs_random = []
        random_vs_random = []
        
        for pair_key, distance in distances.items():
            key1, key2 = pair_key.split('_vs_')
            
            # Classify based on model types
            is_trained_1 = 'trained' in key1 and 'rand' not in key1
            is_trained_2 = 'trained' in key2 and 'rand' not in key2
            
            if is_trained_1 and is_trained_2:
                trained_vs_trained.append(distance)
            elif is_trained_1 != is_trained_2:  # One trained, one random
                trained_vs_random.append(distance)
            else:  # Both random
                random_vs_random.append(distance)
        
        # Calculate statistics
        def safe_stats(arr):
            if not arr:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        
        tt_stats = safe_stats(trained_vs_trained)
        tr_stats = safe_stats(trained_vs_random)
        rr_stats = safe_stats(random_vs_random)
        
        # Calculate separation ratio
        separation_ratio = (tr_stats['mean'] / tt_stats['mean'] 
                          if tt_stats['mean'] > 0 else float('inf'))
        
        return {
            'trained_vs_trained': tt_stats,
            'trained_vs_random': tr_stats,
            'random_vs_random': rr_stats,
            'separation_ratio': separation_ratio,
            'distances': distances
        }
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and rank optimization results."""
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful configurations found'}
        
        # Sort by separation ratio
        successful_results.sort(key=lambda x: x['separation_ratio'], reverse=True)
        
        # Get top configurations
        top_10 = successful_results[:10]
        
        print(f"\n📈 OPTIMIZATION RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total configurations tested: {len(results)}")
        print(f"Successful configurations: {len(successful_results)}")
        print(f"Best separation ratio: {successful_results[0]['separation_ratio']:.2f}x")
        
        return {
            'mode': self.mode,
            'total_tested': len(results),
            'successful': len(successful_results),
            'best_separation': successful_results[0]['separation_ratio'],
            'best_config': successful_results[0]['config'],
            'top_10': top_10,
            'all_results': successful_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_results(self, analysis: Dict[str, Any]) -> None:
        """Save optimization results."""
        # Create results directory
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_file = results_dir / f"{self.mode}_optimization_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save best configuration
        if 'best_config' in analysis:
            best_config_file = results_dir / f"best_config_{self.mode}_{timestamp}.json"
            with open(best_config_file, 'w') as f:
                json.dump(analysis['best_config'], f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        print(f"🏆 Best config saved to: {best_config_file}")


def main():
    parser = argparse.ArgumentParser(description='Parallel DTW Parameter Optimizer')
    parser.add_argument('--mode', choices=['coarse', 'fine', 'validate'], 
                       default='coarse', help='Optimization mode')
    parser.add_argument('--max-configs', type=int, default=100, 
                       help='Maximum configurations to test')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear analysis cache before starting')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ParallelParameterOptimizer(
        mode=args.mode,
        max_configs=args.max_configs
    )
    
    # Clear cache if requested
    if args.clear_cache:
        print("🗑️  Clearing analysis cache...")
        optimizer.cache.clear()
    
    # Run optimization
    results = optimizer.run_optimization()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
DTW Batch Size Validation Script for MLP Models

This script systematically tests DTW distance computation between MLP models
across different batch sizes to validate expected behavior and identify any
batch size dependencies that might affect discrimination quality.

Expected Behavior:
- mlp_trained_100 ↔ mlp_trained_98: LOW distances (~0) across all batch sizes
- mlp_trained_100 ↔ mlp_random: HIGH distances (>50K) across all batch sizes  
- mlp_trained_98 ↔ mlp_random: HIGH distances (>50K) across all batch sizes

Focus: Establish DTW algorithm stability and identify any systematic biases.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.utils.dtw_similarity import FiltrationDTW
from neurosheaf.utils.exceptions import ValidationError, ComputationError
from neurosheaf.sheaf.core.gw_config import GWConfig
# Define MLPModel directly to avoid import side effects
import torch.nn as nn

class MLPModel(nn.Module):
    """MLP model architecture matching the configuration:
    - input_dim: 3 (torus data)
    - num_hidden_layers: 8 
    - hidden_dim: 32
    - output_dim: 1 (binary classification)
    - activation_fn: relu
    - output_activation_fn: sigmoid
    - dropout_rate: 0.0012
    """
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
        
        # Store configuration
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Get activation functions
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        # Build the network
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(self.output_activation_fn)
        
        # Use 'layers' as the attribute name to match saved weights
        self.layers = nn.Sequential(*layers_list)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        """Get activation function by name."""
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Logging setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DTWBatchSizeValidator:
    """Comprehensive DTW batch size validation for MLP models."""
    
    def __init__(self):
        self.device = 'cpu'  # Force CPU for consistency
        self.analyzer = NeurosheafAnalyzer(device=self.device)
        
        # Model paths (from test_all.py)
        self.model_paths = {
            'mlp_trained_100': 'models/torch_mlp_acc_1.0000_epoch_200.pth',
            'mlp_trained_98': 'models/torch_mlp_acc_0.9857_epoch_100.pth', 
            'mlp_random': 'models/random_mlp_net_000_default_seed_42.pth'
        }
        
        # Batch sizes to test
        self.batch_sizes = [25, 50, 75, 100, 150, 200]
        
        # GW config for consistent analysis
        self.gw_config = GWConfig(
            epsilon=0.02,
            max_iter=200,
            tolerance=1e-10,
            adaptive_epsilon=True,
            base_epsilon=0.02,
            reference_n=50,
            epsilon_scaling_method='sqrt'
        )
        
        # DTW configuration
        self.dtw_config = {
            'method': 'tslearn',
            'constraint_band': 0.0,  # No constraints for maximum sensitivity
            'eigenvalue_selection': 15,
            'min_eigenvalue_threshold': 1e-15,
            'interpolation_points': 75,
            'eigenvalue_weight': 1.0,
            'structural_weight': 0.0
        }
        
        # Results storage
        self.results = {}
        self.eigenvalue_data = {}
        
    def load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all MLP models for testing."""
        models = {}
        
        print("🔄 Loading MLP Models:")
        for name, path in self.model_paths.items():
            try:
                model = load_model(MLPModel, path, device=self.device)
                models[name] = model
                param_count = sum(p.numel() for p in model.parameters())
                print(f"  ✅ {name}: {param_count:,} parameters")
            except Exception as e:
                print(f"  ❌ Failed to load {name}: {e}")
                raise
                
        return models
    
    def analyze_model_at_batch_size(self, model: torch.nn.Module, model_name: str, 
                                  batch_size: int) -> Dict[str, Any]:
        """Analyze a single model at specific batch size."""
        # Generate data
        torch.manual_seed(42)  # Fixed seed for reproducibility
        data = 8 * torch.randn(batch_size, 3)
        
        # Run analysis
        try:
            analysis = self.analyzer.analyze(
                model, data,
                method='gromov_wasserstein',
                gw_config=self.gw_config
            )
            
            # Extract eigenvalue evolution
            spectral_results = analysis['spectral_results']
            eigenvalue_sequences = spectral_results['persistence_result']['eigenvalue_sequences']
            filtration_params = spectral_results['filtration_params']
            
            return {
                'eigenvalue_sequences': eigenvalue_sequences,
                'filtration_params': filtration_params,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {model_name} at batch size {batch_size}: {e}")
            return {
                'eigenvalue_sequences': None,
                'filtration_params': None,
                'success': False,
                'error': str(e)
            }
    
    def compute_dtw_distance(self, evolution1: List[torch.Tensor], evolution2: List[torch.Tensor],
                           params1: List[float], params2: List[float]) -> float:
        """Compute DTW distance between two eigenvalue evolution sequences."""
        try:
            dtw_analyzer = FiltrationDTW(**self.dtw_config)
            
            result = dtw_analyzer.compare_eigenvalue_evolution(
                evolution1=evolution1,
                evolution2=evolution2,
                filtration_params1=params1,
                filtration_params2=params2,
                multivariate=True,
                use_interpolation=True,
                match_all_eigenvalues=True,
                interpolation_points=self.dtw_config['interpolation_points']
            )
            
            return result['distance']
            
        except Exception as e:
            logger.warning(f"DTW computation failed: {e}")
            return float('nan')
    
    def run_batch_size_analysis(self) -> None:
        """Run complete batch size analysis across all models and batch sizes."""
        models = self.load_models()
        
        print(f"\n🔍 Running DTW Batch Size Validation:")
        print(f"  Batch sizes: {self.batch_sizes}")
        print(f"  Model pairs: 3 (focusing on MLP models)")
        print(f"  Total analyses: {len(self.batch_sizes) * len(models)} individual model analyses")
        
        # Step 1: Analyze all models at all batch sizes
        print(f"\n📊 Step 1: Analyzing Models Across Batch Sizes")
        for batch_size in self.batch_sizes:
            print(f"\n  Batch Size {batch_size}:")
            self.eigenvalue_data[batch_size] = {}
            
            for model_name, model in models.items():
                print(f"    Analyzing {model_name}...", end=' ')
                result = self.analyze_model_at_batch_size(model, model_name, batch_size)
                
                if result['success']:
                    print("✅")
                    self.eigenvalue_data[batch_size][model_name] = result
                else:
                    print(f"❌ {result['error']}")
                    self.eigenvalue_data[batch_size][model_name] = result
        
        # Step 2: Compute DTW distances for all pairs at all batch sizes
        print(f"\n📏 Step 2: Computing DTW Distances")
        model_names = list(models.keys())
        
        for batch_size in self.batch_sizes:
            print(f"\n  Batch Size {batch_size}:")
            self.results[batch_size] = {}
            
            batch_data = self.eigenvalue_data[batch_size]
            
            # Check if all models analyzed successfully
            failed_models = [name for name, data in batch_data.items() if not data['success']]
            if failed_models:
                print(f"    ⚠️ Skipping due to failed models: {failed_models}")
                continue
            
            # Compute pairwise distances
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i <= j:  # Only compute upper triangle + diagonal
                        if i == j:
                            distance = 0.0  # Self-distance
                        else:
                            data1 = batch_data[model1]
                            data2 = batch_data[model2]
                            
                            distance = self.compute_dtw_distance(
                                data1['eigenvalue_sequences'], data2['eigenvalue_sequences'],
                                data1['filtration_params'], data2['filtration_params']
                            )
                        
                        pair_name = f"{model1} ↔ {model2}"
                        self.results[batch_size][pair_name] = distance
                        
                        if not np.isnan(distance):
                            print(f"    {pair_name}: {distance:,.1f}")
                        else:
                            print(f"    {pair_name}: FAILED")
        
        print(f"\n✅ Batch size analysis complete!")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze results for batch size dependencies and validation."""
        print(f"\n📈 Analyzing Results for Batch Size Dependencies:")
        
        # Extract all successful results
        successful_batches = [bs for bs in self.batch_sizes if bs in self.results and self.results[bs]]
        
        if not successful_batches:
            print("  ❌ No successful analyses to evaluate")
            return {}
        
        print(f"  Successful batch sizes: {successful_batches}")
        
        # Create DataFrame for analysis
        all_data = []
        for batch_size in successful_batches:
            for pair_name, distance in self.results[batch_size].items():
                if not np.isnan(distance):
                    all_data.append({
                        'batch_size': batch_size,
                        'pair': pair_name,
                        'distance': distance,
                        'pair_type': self._classify_pair(pair_name)
                    })
        
        if not all_data:
            print("  ❌ No valid distance measurements")
            return {}
        
        df = pd.DataFrame(all_data)
        
        # Analysis 1: Distance stability across batch sizes
        print(f"\n  📊 Distance Stability Analysis:")
        stability_stats = {}
        
        for pair in df['pair'].unique():
            pair_data = df[df['pair'] == pair]['distance']
            if len(pair_data) > 1:
                cv = pair_data.std() / pair_data.mean() if pair_data.mean() > 0 else float('inf')
                stability_stats[pair] = {
                    'mean': pair_data.mean(),
                    'std': pair_data.std(),
                    'cv': cv,
                    'min': pair_data.min(),
                    'max': pair_data.max()
                }
                
                print(f"    {pair}:")
                print(f"      Mean: {pair_data.mean():,.1f} ± {pair_data.std():,.1f}")
                print(f"      CV: {cv:.3f} ({'STABLE' if cv < 0.1 else 'UNSTABLE' if cv < 0.3 else 'HIGHLY UNSTABLE'})")
                print(f"      Range: [{pair_data.min():,.1f}, {pair_data.max():,.1f}]")
        
        # Analysis 2: Expected behavior validation
        print(f"\n  ✅ Expected Behavior Validation:")
        
        # Define expected patterns
        expected_low = ['mlp_trained_100 ↔ mlp_trained_98']
        expected_high = ['mlp_trained_100 ↔ mlp_random', 'mlp_trained_98 ↔ mlp_random']
        
        validation_results = {}
        
        for pair in expected_low:
            if pair in stability_stats:
                mean_dist = stability_stats[pair]['mean']
                is_valid = mean_dist < 10000  # Low threshold
                validation_results[pair] = {'type': 'low', 'mean': mean_dist, 'valid': is_valid}
                print(f"    {pair}: {mean_dist:,.1f} ({'✅ LOW' if is_valid else '❌ NOT LOW'})")
        
        for pair in expected_high:
            if pair in stability_stats:
                mean_dist = stability_stats[pair]['mean']
                is_valid = mean_dist > 50000  # High threshold
                validation_results[pair] = {'type': 'high', 'mean': mean_dist, 'valid': is_valid}
                print(f"    {pair}: {mean_dist:,.1f} ({'✅ HIGH' if is_valid else '❌ NOT HIGH'})")
        
        # Analysis 3: Separation ratio analysis
        print(f"\n  🎯 Separation Ratio Analysis:")
        separation_ratios = {}
        
        for batch_size in successful_batches:
            batch_results = self.results[batch_size]
            
            # Get trained-trained distances (should be low)
            trained_trained = [batch_results.get(pair, np.nan) for pair in expected_low]
            trained_trained = [d for d in trained_trained if not np.isnan(d)]
            
            # Get trained-random distances (should be high) 
            trained_random = [batch_results.get(pair, np.nan) for pair in expected_high]
            trained_random = [d for d in trained_random if not np.isnan(d)]
            
            if trained_trained and trained_random:
                mean_low = np.mean(trained_trained)
                mean_high = np.mean(trained_random)
                
                if mean_low > 0:
                    ratio = mean_high / mean_low
                    separation_ratios[batch_size] = ratio
                    print(f"    Batch {batch_size}: {ratio:.2f}x ({'✅ GOOD' if ratio > 5.0 else '❌ POOR'})")
        
        # Summary
        valid_low = sum(1 for r in validation_results.values() if r['type'] == 'low' and r['valid'])
        valid_high = sum(1 for r in validation_results.values() if r['type'] == 'high' and r['valid'])
        total_expected = len(expected_low) + len(expected_high)
        
        print(f"\n  📋 Validation Summary:")
        print(f"    Expected behavior maintained: {valid_low + valid_high}/{total_expected}")
        print(f"    Low distance pairs correct: {valid_low}/{len(expected_low)}")  
        print(f"    High distance pairs correct: {valid_high}/{len(expected_high)}")
        
        if separation_ratios:
            mean_separation = np.mean(list(separation_ratios.values()))
            print(f"    Average separation ratio: {mean_separation:.2f}x")
        
        return {
            'stability_stats': stability_stats,
            'validation_results': validation_results,
            'separation_ratios': separation_ratios,
            'summary': {
                'valid_pairs': valid_low + valid_high,
                'total_pairs': total_expected,
                'mean_separation': np.mean(list(separation_ratios.values())) if separation_ratios else 0
            }
        }
    
    def _classify_pair(self, pair_name: str) -> str:
        """Classify pair type for analysis."""
        if 'mlp_trained_100 ↔ mlp_trained_98' in pair_name:
            return 'trained_trained'
        elif 'mlp_random' in pair_name:
            return 'trained_random'
        else:
            return 'other'
    
    def create_visualizations(self, analysis_results: Dict[str, Any]) -> None:
        """Create visualization plots for batch size analysis."""
        print(f"\n📊 Creating Visualizations:")
        
        if not analysis_results:
            print("  ⚠️ No results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DTW Batch Size Validation Results (MLP Models)', fontsize=16)
        
        # Prepare data
        successful_batches = [bs for bs in self.batch_sizes if bs in self.results and self.results[bs]]
        
        # Plot 1: Distance vs Batch Size for each pair
        ax1 = axes[0, 0]
        for pair in analysis_results['stability_stats'].keys():
            batch_distances = []
            batch_sizes_for_pair = []
            
            for bs in successful_batches:
                if pair in self.results[bs]:
                    distance = self.results[bs][pair]
                    if not np.isnan(distance):
                        batch_distances.append(distance)
                        batch_sizes_for_pair.append(bs)
            
            if batch_distances:
                ax1.plot(batch_sizes_for_pair, batch_distances, 'o-', label=pair, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('DTW Distance')
        ax1.set_title('DTW Distance Stability Across Batch Sizes')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Coefficient of Variation
        ax2 = axes[0, 1]
        pairs = list(analysis_results['stability_stats'].keys())
        cvs = [analysis_results['stability_stats'][pair]['cv'] for pair in pairs]
        
        colors = ['green' if cv < 0.1 else 'orange' if cv < 0.3 else 'red' for cv in cvs]
        bars = ax2.bar(range(len(pairs)), cvs, color=colors, alpha=0.7)
        ax2.set_xlabel('Model Pairs')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('Distance Stability (Lower = More Stable)')
        ax2.set_xticks(range(len(pairs)))
        ax2.set_xticklabels([p.split(' ↔ ')[0][:10] + '...' for p in pairs], rotation=45)
        ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Stable threshold')
        ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Unstable threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Separation Ratios
        ax3 = axes[1, 0]
        if analysis_results['separation_ratios']:
            batch_sizes_sep = list(analysis_results['separation_ratios'].keys())
            ratios = list(analysis_results['separation_ratios'].values())
            
            bars = ax3.bar(batch_sizes_sep, ratios, color='steelblue', alpha=0.7)
            ax3.set_xlabel('Batch Size')
            ax3.set_ylabel('Separation Ratio (High/Low)')
            ax3.set_title('Discrimination Quality Across Batch Sizes')
            ax3.axhline(y=5.0, color='green', linestyle='--', alpha=0.7, label='Good threshold (5x)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, ratio in zip(bars, ratios):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{ratio:.1f}x', ha='center', va='bottom')
        
        # Plot 4: Validation Summary
        ax4 = axes[1, 1]
        validation_summary = analysis_results['summary']
        
        categories = ['Valid Pairs', 'Invalid Pairs']
        values = [validation_summary['valid_pairs'], 
                 validation_summary['total_pairs'] - validation_summary['valid_pairs']]
        colors = ['green', 'red']
        
        wedges, texts, autotexts = ax4.pie(values, labels=categories, colors=colors, 
                                          autopct='%1.0f', startangle=90)
        ax4.set_title(f'Validation Results\n(Mean Separation: {validation_summary["mean_separation"]:.1f}x)')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'dtw_batch_size_validation_results.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  ✅ Visualization saved as '{plot_filename}'")
        
        plt.close()
    
    def save_detailed_results(self, analysis_results: Dict[str, Any]) -> None:
        """Save detailed results to JSON and CSV files."""
        print(f"\n💾 Saving Detailed Results:")
        
        # Save raw results as JSON
        json_filename = 'dtw_batch_size_validation_raw.json'
        json_data = {
            'batch_sizes': self.batch_sizes,
            'dtw_config': self.dtw_config,
            'results': self.results,
            'timestamp': datetime.now().isoformat(),
            'analysis_results': analysis_results
        }
        
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"  ✅ Raw results saved as '{json_filename}'")
        
        # Save distance matrix as CSV
        if self.results:
            # Create distance matrix DataFrame
            all_pairs = set()
            for batch_results in self.results.values():
                all_pairs.update(batch_results.keys())
            
            distance_matrix = []
            for batch_size in self.batch_sizes:
                row = {'batch_size': batch_size}
                if batch_size in self.results:
                    for pair in sorted(all_pairs):
                        distance = self.results[batch_size].get(pair, np.nan)
                        row[pair] = distance
                else:
                    for pair in sorted(all_pairs):
                        row[pair] = np.nan
                distance_matrix.append(row)
            
            df = pd.DataFrame(distance_matrix)
            csv_filename = 'dtw_batch_size_validation_matrix.csv'
            df.to_csv(csv_filename, index=False)
            print(f"  ✅ Distance matrix saved as '{csv_filename}'")
    
    def run_full_validation(self) -> None:
        """Run complete DTW batch size validation pipeline."""
        print("🚀 DTW Batch Size Validation for MLP Models")
        print("=" * 80)
        
        try:
            # Run analysis
            self.run_batch_size_analysis()
            
            # Analyze results  
            analysis_results = self.analyze_results()
            
            # Create visualizations
            self.create_visualizations(analysis_results)
            
            # Save results
            self.save_detailed_results(analysis_results)
            
            print(f"\n🎉 DTW Batch Size Validation Complete!")
            print("=" * 80)
            
            if analysis_results and analysis_results['summary']:
                summary = analysis_results['summary']
                print(f"📊 Final Summary:")
                print(f"  • Validation success rate: {summary['valid_pairs']}/{summary['total_pairs']} pairs")
                print(f"  • Average separation ratio: {summary['mean_separation']:.2f}x")
                print(f"  • Expected behavior: {'✅ MAINTAINED' if summary['valid_pairs'] == summary['total_pairs'] else '⚠️ ISSUES DETECTED'}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    validator = DTWBatchSizeValidator()
    validator.run_full_validation()

if __name__ == "__main__":
    main()
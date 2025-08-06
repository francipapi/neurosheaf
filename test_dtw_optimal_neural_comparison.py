#!/usr/bin/env python3
"""
Optimal DTW Parameter Tuning for Neural Network Comparison

This script finds optimal DTW parameters for comparing neural networks using 
multivariate DTW distance on eigenvalue evolution. The goal is to achieve:

Expected Results:
- LOW distance between similar models: mlp_path, mlp_path1, custom_path (all trained)
- HIGH distance between trained vs random: trained models vs rand_mlp_path, rand_custom_path

Models tested (from test_all.py):
- mlp_path: models/torch_mlp_acc_1.0000_epoch_200.pth (trained MLP, 100% acc)
- mlp_path1: models/torch_mlp_acc_0.9857_epoch_100.pth (trained MLP, 98.57% acc)  
- custom_path: models/torch_custom_acc_1.0000_epoch_200.pth (trained Custom, 100% acc)
- rand_mlp_path: models/random_mlp_net_000_default_seed_42.pth (random MLP)
- rand_custom_path: models/random_custom_net_000_default_seed_42.pth (random Custom)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import itertools
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
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

# Custom DTW class that forces log scale and proper interpolation
class LogScaleInterpolationDTW(FiltrationDTW):
    """Enhanced DTW that forces log scale transformation and proper interpolation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Using LogScaleInterpolationDTW with forced log transformation")
    
    def _tslearn_multivariate(self, seq1: np.ndarray, seq2: np.ndarray):
        """Override multivariate DTW to force log transformation."""
        try:
            # Validate input shapes for multivariate DTW
            if seq1.ndim != 2 or seq2.ndim != 2:
                raise ValidationError(f"Multivariate sequences must be 2D, got shapes {seq1.shape}, {seq2.shape}")
            
            if seq1.shape[1] != seq2.shape[1]:
                raise ValidationError(f"Sequences must have same number of features: {seq1.shape[1]} vs {seq2.shape[1]}")
            
            # FORCE log transformation for eigenvalues (they're always very small)
            logger.debug("Forcing log transformation for eigenvalue sequences")
            
            # Apply log transformation with more aggressive thresholding
            seq1_processed = np.log(np.maximum(seq1, self.min_eigenvalue_threshold))
            seq2_processed = np.log(np.maximum(seq2, self.min_eigenvalue_threshold))
            
            # Ensure finite values with better handling
            if not (np.isfinite(seq1_processed).all() and np.isfinite(seq2_processed).all()):
                logger.warning("Non-finite values detected in log-transformed sequences, cleaning...")
                seq1_processed = np.nan_to_num(seq1_processed, nan=-30.0, posinf=10.0, neginf=-30.0)
                seq2_processed = np.nan_to_num(seq2_processed, nan=-30.0, posinf=10.0, neginf=-30.0)
            
            # Additional validation: check for constant sequences after log transform
            seq1_var = np.var(seq1_processed)
            seq2_var = np.var(seq2_processed)
            
            if seq1_var < 1e-10:
                logger.warning(f"Log-transformed sequence 1 is nearly constant (var: {seq1_var:.2e})")
                # Add small random noise to break ties
                seq1_processed += np.random.normal(0, 1e-6, seq1_processed.shape)
            
            if seq2_var < 1e-10:
                logger.warning(f"Log-transformed sequence 2 is nearly constant (var: {seq2_var:.2e})")
                # Add small random noise to break ties
                seq2_processed += np.random.normal(0, 1e-6, seq2_processed.shape)
            
            logger.debug(f"Log transformation applied - seq1 range: [{np.min(seq1_processed):.2f}, {np.max(seq1_processed):.2f}], "
                        f"seq2 range: [{np.min(seq2_processed):.2f}, {np.max(seq2_processed):.2f}]")
            
            # Import DTW functions
            try:
                from tslearn.metrics import dtw_path
            except ImportError:
                raise ComputationError("tslearn not available for multivariate DTW")
            
            # Compute DTW with constraints
            if self.constraint_band > 0:
                global_constraint = "sakoe_chiba"
                sakoe_chiba_radius = int(max(len(seq1_processed), len(seq2_processed)) * self.constraint_band)
                path, distance = dtw_path(seq1_processed, seq2_processed, 
                                        global_constraint=global_constraint,
                                        sakoe_chiba_radius=sakoe_chiba_radius)
            else:
                path, distance = dtw_path(seq1_processed, seq2_processed)
            
            alignment = [(int(i), int(j)) for i, j in path]
            
            logger.debug(f"DTW completed - distance: {distance:.6f}, alignment length: {len(alignment)}")
            
            return float(distance), alignment
            
        except Exception as e:
            logger.warning(f"Enhanced multivariate DTW failed: {e}, falling back to univariate")
            # Fallback to univariate DTW on first dimension with log transform
            seq1_log = np.log(np.maximum(seq1[:, 0], self.min_eigenvalue_threshold))
            seq2_log = np.log(np.maximum(seq2[:, 0], self.min_eigenvalue_threshold))
            return self._tslearn_univariate(seq1_log, seq2_log)

from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.utils.logging import setup_logger

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

logger = setup_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a neural network model."""
    name: str
    path: str
    model_class: type
    description: str
    is_trained: bool
    architecture_type: str  # 'MLP' or 'Custom'


@dataclass 
class ParameterConfig:
    """DTW parameter configuration."""
    constraint_band: float
    eigenvalue_selection: Optional[int]  # None = all eigenvalues
    min_eigenvalue_threshold: float
    interpolation_points: Optional[int]  # None = auto
    name: str


class MLPModel(nn.Module):
    """MLP model architecture matching test_all.py."""
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
            'none': nn.Identity()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x):
        return self.layers(x)


class ActualCustomModel(nn.Module):
    """Custom model architecture matching test_all.py with Conv1D layers."""
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(3, 32),                                    # layers.0
            nn.ReLU(),                                           # layers.1
            nn.Linear(32, 32),                                   # layers.2
            nn.ReLU(),                                           # layers.3
            nn.Dropout(0.0),                                     # layers.4
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.5
            nn.ReLU(),                                           # layers.6
            nn.Dropout(0.0),                                     # layers.7
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.8
            nn.ReLU(),                                           # layers.9
            nn.Dropout(0.0),                                     # layers.10
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.11
            nn.ReLU(),                                           # layers.12
            nn.Dropout(0.0),                                     # layers.13
            nn.Linear(32, 1),                                    # layers.14
            nn.Sigmoid()                                         # layers.15
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


class ModelManager:
    """Manages loading and validation of all neural network models."""
    
    def __init__(self):
        self.models = {}
        self.eigenvalue_evolutions = {}
        self.spectral_results = {}
        self.model_infos = self._define_model_info()
        
        # Shared analysis configuration
        self.gw_config = GWConfig(
            epsilon=0.05,
            max_iter=100, 
            tolerance=1e-8,
            quasi_sheaf_tolerance=0.08
        )
        self.batch_size = 50
        self.data = 8 * torch.randn(self.batch_size, 3)
    
    def _define_model_info(self) -> Dict[str, ModelInfo]:
        """Define information for all models to be tested."""
        return {
            'mlp_trained_100': ModelInfo(
                name='mlp_trained_100',
                path='models/torch_mlp_acc_1.0000_epoch_200.pth',
                model_class=MLPModel,
                description='MLP 100% accuracy (200 epochs)',
                is_trained=True,
                architecture_type='MLP'
            ),
            'mlp_trained_98': ModelInfo(
                name='mlp_trained_98',
                path='models/torch_mlp_acc_0.9857_epoch_100.pth', 
                model_class=MLPModel,
                description='MLP 98.57% accuracy (100 epochs)',
                is_trained=True,
                architecture_type='MLP'
            ),
            'custom_trained': ModelInfo(
                name='custom_trained',
                path='models/torch_custom_acc_1.0000_epoch_200.pth',
                model_class=ActualCustomModel,
                description='Custom 100% accuracy (200 epochs)',
                is_trained=True,
                architecture_type='Custom'
            ),
            'mlp_random': ModelInfo(
                name='mlp_random',
                path='models/random_mlp_net_000_default_seed_42.pth',
                model_class=MLPModel,
                description='Random MLP (untrained)',
                is_trained=False,
                architecture_type='MLP'
            ),
            'custom_random': ModelInfo(
                name='custom_random',
                path='models/random_custom_net_000_default_seed_42.pth',
                model_class=ActualCustomModel,
                description='Random Custom (untrained)',
                is_trained=False,
                architecture_type='Custom'
            )
        }
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load all models and extract eigenvalue evolutions."""
        logger.info("Loading all models and extracting eigenvalue evolutions...")
        
        for model_id, model_info in self.model_infos.items():
            try:
                logger.info(f"Loading {model_id}: {model_info.description}")
                
                # Load model
                model = load_model(model_info.model_class, model_info.path, device="cpu")
                model.eval()
                self.models[model_id] = model
                
                # Extract eigenvalue evolution
                evolution, spectral_result = self._extract_eigenvalue_evolution(
                    model, model_id, model_info.description
                )
                self.eigenvalue_evolutions[model_id] = evolution
                self.spectral_results[model_id] = spectral_result
                
                logger.info(f"✅ {model_id}: {sum(p.numel() for p in model.parameters()):,} parameters, "
                           f"{len(evolution)} filtration steps")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_id}: {e}")
                raise
        
        return self.models
    
    def _extract_eigenvalue_evolution(self, model, model_id: str, description: str):  
        """Extract eigenvalue evolution from a model using the exact approach from test_all.py."""
        logger.debug(f"Extracting eigenvalue evolution for {model_id}")
        
        # Use high-level API like test_all.py
        analyzer = NeurosheafAnalyzer(device='cpu')
        analysis = analyzer.analyze(model, self.data, method='gromov_wasserstein', gw_config=self.gw_config)
        sheaf = analysis['sheaf']
        
        logger.debug(f"   Sheaf: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
        
        # Run spectral analysis with same parameters as test_all.py
        from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
        
        spectral_analyzer = PersistentSpectralAnalyzer(
            default_n_steps=50,
            default_filtration_type='threshold'
        )
        
        spectral_results = spectral_analyzer.analyze(
            sheaf,
            filtration_type='threshold',
            n_steps=100
        )
        
        # Extract eigenvalue evolution
        eigenvalue_evolution = spectral_results['persistence_result']['eigenvalue_sequences']
        
        logger.debug(f"   ✅ Eigenvalue sequences: {len(eigenvalue_evolution)} steps")
        logger.debug(f"   Eigenvalues per step: {[len(seq) for seq in eigenvalue_evolution[:5]]}...")
        
        return eigenvalue_evolution, spectral_results
    
    def get_model_groups(self) -> Dict[str, List[str]]:
        """Get model groups for validation."""
        trained_models = [k for k, v in self.model_infos.items() if v.is_trained]
        random_models = [k for k, v in self.model_infos.items() if not v.is_trained]
        mlp_models = [k for k, v in self.model_infos.items() if v.architecture_type == 'MLP']
        custom_models = [k for k, v in self.model_infos.items() if v.architecture_type == 'Custom']
        
        return {
            'trained': trained_models,
            'random': random_models,
            'mlp': mlp_models,
            'custom': custom_models,
            'all': list(self.model_infos.keys())
        }


class ParameterTuner:
    """Systematic parameter space exploration for DTW optimization."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.results = {}
        
    def create_parameter_configs(self) -> List[ParameterConfig]:
        """Create parameter configurations including FULL SPECTRUM tests."""
        configs = []
        
        # FULL SPECTRUM CONFIGURATIONS - Test using ALL eigenvalues
        # This leverages the interpolation method to handle different spectrum sizes
        
        # 1. Full spectrum with varying constraint bands
        full_spectrum_constraint_bands = [0.0, 0.1]  # Best performers from limited tests
        for cb in full_spectrum_constraint_bands:
            configs.append(ParameterConfig(
                constraint_band=cb,
                eigenvalue_selection=None,  # Use ALL eigenvalues
                min_eigenvalue_threshold=1e-15,
                interpolation_points=150,  # Higher resolution for full spectrum
                name=f'full_spectrum_cb_{cb}'
            ))
        
        # 2. Full spectrum with different interpolation resolutions
        interp_points = [100, 200, 300]  # Test different resolutions
        for points in interp_points:
            configs.append(ParameterConfig(
                constraint_band=0.0,  # Best constraint from previous results
                eigenvalue_selection=None,  # Use ALL eigenvalues
                min_eigenvalue_threshold=1e-15,
                interpolation_points=points,
                name=f'full_spectrum_{points}_points'
            ))
        
        # 3. Full spectrum with ultra-low threshold for maximum sensitivity
        configs.append(ParameterConfig(
            constraint_band=0.0,
            eigenvalue_selection=None,  # Use ALL eigenvalues
            min_eigenvalue_threshold=1e-18,  # Ultra-low threshold
            interpolation_points=200,
            name='full_spectrum_ultra_sensitive'
        ))
        
        # 4. Compare with LIMITED spectrum configurations (for baseline)
        # These are the best performers from previous results
        configs.append(ParameterConfig(
            constraint_band=0.0,
            eigenvalue_selection=15,  # Previous best
            min_eigenvalue_threshold=1e-15,
            interpolation_points=75,
            name='limited_top15_baseline'
        ))
        
        configs.append(ParameterConfig(
            constraint_band=0.0,
            eigenvalue_selection=30,  # Medium selection
            min_eigenvalue_threshold=1e-15,
            interpolation_points=100,
            name='limited_top30_comparison'
        ))
        
        logger.info(f"Created {len(configs)} parameter configurations")
        logger.info("Testing FULL SPECTRUM (all eigenvalues) vs LIMITED spectrum approaches")
        logger.info("Full spectrum uses interpolation to handle varying eigenvalue counts")
        return configs
    
    def test_parameter_config(self, config: ParameterConfig) -> Dict[str, Any]:
        """Test a single parameter configuration across all model pairs."""
        logger.info(f"Testing configuration: {config.name}")
        
        try:
            # Create enhanced DTW comparator with forced log scale and interpolation
            dtw_comparator = LogScaleInterpolationDTW(
                method='tslearn',  # Force tslearn for multivariate with log transform
                constraint_band=config.constraint_band,
                eigenvalue_weight=1.0,
                structural_weight=0.0,
                normalization_scheme='range_aware',
                min_eigenvalue_threshold=config.min_eigenvalue_threshold
            )
            
            # Compute distance matrix for all model pairs
            model_ids = list(self.model_manager.model_infos.keys())
            n_models = len(model_ids)
            distance_matrix = np.zeros((n_models, n_models))
            comparison_details = {}
            
            for i, model_i in enumerate(model_ids):
                for j, model_j in enumerate(model_ids):
                    if i <= j:  # Compute upper triangle + diagonal
                        if i == j:
                            distance_matrix[i, j] = 0.0
                        else:
                            # Get eigenvalue evolutions
                            evolution_i = self.model_manager.eigenvalue_evolutions[model_i]
                            evolution_j = self.model_manager.eigenvalue_evolutions[model_j]
                            
                            # Apply eigenvalue selection if specified
                            if config.eigenvalue_selection is not None:
                                evolution_i = self._filter_top_eigenvalues(evolution_i, config.eigenvalue_selection)
                                evolution_j = self._filter_top_eigenvalues(evolution_j, config.eigenvalue_selection)
                            
                            # Compute DTW distance - FORCE interpolation mode
                            result = dtw_comparator.compare_eigenvalue_evolution(
                                evolution_i, evolution_j,
                                multivariate=True,  # Always use multivariate for log-scale benefits
                                use_interpolation=True,  # Always use interpolation to avoid padding
                                match_all_eigenvalues=True,  # Match all available eigenvalues
                                interpolation_points=config.interpolation_points
                            )
                            
                            distance = result['normalized_distance']
                            distance_matrix[i, j] = distance
                            distance_matrix[j, i] = distance  # Symmetric
                            
                            # Store comparison details
                            pair_key = f"{model_i}_{model_j}"
                            comparison_details[pair_key] = {
                                'distance': distance,
                                'raw_distance': result['distance'],
                                'sequence_lengths': (result['sequence1_length'], result['sequence2_length']),
                                'interpolation_used': result['interpolation_used']
                            }
            
            return {
                'config': config,
                'distance_matrix': distance_matrix,
                'model_ids': model_ids,
                'comparison_details': comparison_details,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to test configuration {config.name}: {e}")
            return {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    def _filter_top_eigenvalues(self, evolution: List[torch.Tensor], k: int) -> List[torch.Tensor]:
        """Filter eigenvalue evolution to top-k eigenvalues per step."""
        filtered_evolution = []
        
        for step_eigenvals in evolution:
            if len(step_eigenvals) > k:
                sorted_eigenvals, _ = torch.sort(step_eigenvals, descending=True)
                filtered_evolution.append(sorted_eigenvals[:k])
            else:
                filtered_evolution.append(step_eigenvals)
        
        return filtered_evolution
    
    def run_parameter_exploration(self) -> Dict[str, Any]:
        """Run comprehensive parameter exploration."""
        logger.info("Starting comprehensive parameter exploration...")
        
        configs = self.create_parameter_configs()
        results = {}
        
        for i, config in enumerate(configs):
            logger.info(f"Testing configuration {i+1}/{len(configs)}: {config.name}")
            result = self.test_parameter_config(config)
            results[config.name] = result
            
            if result['success']:
                logger.info(f"✅ {config.name}: Successfully computed distance matrix")
            else:
                logger.warning(f"❌ {config.name}: Failed - {result.get('error', 'Unknown error')}")
        
        self.results = results
        return results


class ValidationEngine:
    """Validates DTW results against expected similarity patterns."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.model_groups = model_manager.get_model_groups()
    
    def validate_distance_matrix(self, distance_matrix: np.ndarray, model_ids: List[str]) -> Dict[str, Any]:
        """Validate distance matrix against expected patterns."""
        
        # Create model info lookup
        id_to_info = {model_id: self.model_manager.model_infos[model_id] for model_id in model_ids}
        
        # Analyze different comparison types
        intra_trained_distances = []  # trained vs trained
        inter_group_distances = []    # trained vs random
        mlp_distances = []           # MLP vs MLP
        custom_distances = []        # Custom vs Custom
        cross_arch_distances = []    # MLP vs Custom
        
        for i, model_i in enumerate(model_ids):
            for j, model_j in enumerate(model_ids):
                if i < j:  # Upper triangle only
                    distance = distance_matrix[i, j]
                    info_i = id_to_info[model_i]
                    info_j = id_to_info[model_j]
                    
                    # Categorize comparison
                    if info_i.is_trained and info_j.is_trained:
                        intra_trained_distances.append(distance)
                    elif info_i.is_trained != info_j.is_trained: 
                        inter_group_distances.append(distance)
                    
                    if info_i.architecture_type == 'MLP' and info_j.architecture_type == 'MLP':
                        mlp_distances.append(distance)
                    elif info_i.architecture_type == 'Custom' and info_j.architecture_type == 'Custom':
                        custom_distances.append(distance)
                    elif info_i.architecture_type != info_j.architecture_type:
                        cross_arch_distances.append(distance)
        
        # Compute validation metrics
        validation_metrics = {
            'intra_trained_stats': self._compute_stats(intra_trained_distances, 'intra_trained'),
            'inter_group_stats': self._compute_stats(inter_group_distances, 'inter_group'),
            'mlp_stats': self._compute_stats(mlp_distances, 'mlp_vs_mlp'),
            'custom_stats': self._compute_stats(custom_distances, 'custom_vs_custom'),
            'cross_arch_stats': self._compute_stats(cross_arch_distances, 'cross_architecture'),
        }
        
        # Compute separation metrics
        if intra_trained_distances and inter_group_distances:
            mean_intra = np.mean(intra_trained_distances)
            mean_inter = np.mean(inter_group_distances)
            separation_ratio = mean_inter / mean_intra if mean_intra > 0 else float('inf')
            
            validation_metrics['separation_ratio'] = separation_ratio
            validation_metrics['expected_pattern'] = mean_inter > mean_intra  # Should be True
            
        return validation_metrics
    
    def _compute_stats(self, distances: List[float], category: str) -> Dict[str, Any]:
        """Compute statistics for a category of distances."""
        if not distances:
            return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'count': len(distances),
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'median': float(np.median(distances)),
            'distances': distances
        }
    
    def rank_configurations(self, all_results: Dict[str, Any]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Rank parameter configurations by validation performance."""
        rankings = []
        
        for config_name, result in all_results.items():
            if not result['success']:
                continue
                
            # Validate distance matrix
            distance_matrix = result['distance_matrix'] 
            model_ids = result['model_ids']
            validation = self.validate_distance_matrix(distance_matrix, model_ids)
            
            # Compute overall performance score
            score = self._compute_performance_score(validation)
            
            rankings.append((config_name, score, validation))
        
        # Sort by score (higher is better)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _compute_performance_score(self, validation: Dict[str, Any]) -> float:
        """Compute overall performance score for a configuration."""
        score = 0.0
        
        # Primary objective: separation ratio (trained vs random)
        if 'separation_ratio' in validation:
            separation_ratio = validation['separation_ratio']
            if separation_ratio > 1.0:  # Good: inter-group > intra-group
                score += min(10.0, separation_ratio)  # Cap at 10 to avoid dominance
            else:
                score -= 5.0  # Penalty for wrong pattern
        
        # Secondary: low variance within expected similar groups
        intra_stats = validation.get('intra_trained_stats', {})
        if intra_stats.get('count', 0) > 0:
            # Reward low standard deviation in intra-trained comparisons
            intra_std = intra_stats.get('std', 1.0)
            score += max(0, 2.0 - intra_std)  # Bonus for std < 2.0
        
        # Tertiary: reasonable absolute distances
        inter_stats = validation.get('inter_group_stats', {})
        if inter_stats.get('count', 0) > 0:
            inter_mean = inter_stats.get('mean', 0.0)
            # Bonus for moderate inter-group distances (not too high/low)
            if 0.3 < inter_mean < 2.0:
                score += 1.0
        
        return score


class ComprehensiveAnalyzer:
    """Comprehensive analysis and reporting of DTW parameter tuning results."""
    
    def __init__(self, model_manager: ModelManager, validation_engine: ValidationEngine):
        self.model_manager = model_manager
        self.validation_engine = validation_engine
    
    def analyze_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of all results."""
        logger.info("Performing comprehensive analysis of results...")
        
        # Rank configurations
        rankings = self.validation_engine.rank_configurations(all_results)
        
        # Find best configuration
        if rankings:
            best_config_name, best_score, best_validation = rankings[0]
            best_result = all_results[best_config_name]
        else:
            best_config_name = None
            best_score = 0.0
            best_validation = {}
            best_result = {}
        
        # Analyze parameter impact
        parameter_impact = self._analyze_parameter_impact(all_results, rankings)
        
        # Generate distance matrix analysis for best configuration
        distance_analysis = {}
        if best_result.get('success', False):
            distance_analysis = self._analyze_distance_matrix(
                best_result['distance_matrix'],
                best_result['model_ids'],
                f"Best Configuration: {best_config_name}"
            )
        
        return {
            'rankings': rankings[:10],  # Top 10 configurations
            'best_configuration': {
                'name': best_config_name,
                'score': best_score,
                'validation': best_validation,
                'config': best_result.get('config') if best_result else None
            },
            'parameter_impact': parameter_impact,
            'distance_analysis': distance_analysis,
            'summary_stats': self._compute_summary_stats(all_results)
        }
    
    def _analyze_parameter_impact(self, all_results: Dict[str, Any], rankings: List) -> Dict[str, Any]:
        """Analyze the impact of different parameters."""
        parameter_impacts = {
            'constraint_band': {},
            'eigenvalue_selection': {},
            'threshold': {},
            'interpolation': {}
        }
        
        # Group results by parameter type
        for config_name, score, validation in rankings:
            if not all_results[config_name]['success']:
                continue
                
            config = all_results[config_name]['config']
            
            # Constraint band impact
            if config_name.startswith('constraint_band_'):
                cb_value = config.constraint_band
                parameter_impacts['constraint_band'][cb_value] = {
                    'score': score,
                    'config_name': config_name,
                    'separation_ratio': validation.get('separation_ratio', 0.0)
                }
            
            # Eigenvalue selection impact
            elif config_name.startswith('top_') and 'eigenvalues' in config_name:
                k_value = config.eigenvalue_selection
                parameter_impacts['eigenvalue_selection'][k_value] = {
                    'score': score,
                    'config_name': config_name,
                    'separation_ratio': validation.get('separation_ratio', 0.0)
                }
            
            # Threshold impact
            elif config_name.startswith('threshold_'):
                threshold_value = config.min_eigenvalue_threshold
                parameter_impacts['threshold'][threshold_value] = {
                    'score': score,
                    'config_name': config_name,
                    'separation_ratio': validation.get('separation_ratio', 0.0)
                }
            
            # Interpolation impact
            elif config_name.startswith('interp_'):
                interp_value = config.interpolation_points
                parameter_impacts['interpolation'][interp_value] = {
                    'score': score,
                    'config_name': config_name,
                    'separation_ratio': validation.get('separation_ratio', 0.0)
                }
        
        return parameter_impacts
    
    def _analyze_distance_matrix(self, distance_matrix: np.ndarray, model_ids: List[str], title: str) -> Dict[str, Any]:
        """Analyze distance matrix structure."""
        n_models = len(model_ids)
        
        # Create model info lookup
        model_infos = [self.model_manager.model_infos[model_id] for model_id in model_ids]
        
        # Create labels for visualization
        labels = [f"{info.name}\n({info.description})" for info in model_infos]
        
        analysis = {
            'title': title,
            'distance_matrix': distance_matrix.tolist(),
            'model_ids': model_ids,
            'labels': labels,
            'matrix_stats': {
                'mean': float(np.mean(distance_matrix)),
                'std': float(np.std(distance_matrix)),
                'min': float(np.min(distance_matrix)),
                'max': float(np.max(distance_matrix))
            }
        }
        
        # Hierarchical clustering analysis
        try:
            # Convert to condensed distance matrix for clustering
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            analysis['clustering'] = {
                'linkage_matrix': linkage_matrix.tolist(),
                'available': True
            }
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            analysis['clustering'] = {'available': False, 'error': str(e)}
        
        return analysis
    
    def _compute_summary_stats(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all configurations."""
        successful_configs = sum(1 for r in all_results.values() if r['success'])
        total_configs = len(all_results)
        
        return {
            'total_configurations_tested': total_configs,
            'successful_configurations': successful_configs,
            'success_rate': successful_configs / total_configs if total_configs > 0 else 0.0
        }
    
    def create_visualizations(self, analysis_results: Dict[str, Any], output_dir: str = ".") -> Dict[str, str]:
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        output_files = {}
        
        # 1. Best configuration distance matrix heatmap
        if analysis_results['distance_analysis']:
            output_files['heatmap'] = self._create_distance_heatmap(
                analysis_results['distance_analysis'], output_dir
            )
        
        # 2. Parameter impact plots
        output_files['parameter_impact'] = self._create_parameter_impact_plots(
            analysis_results['parameter_impact'], output_dir
        )
        
        # 3. Configuration ranking plot
        output_files['rankings'] = self._create_ranking_plot(
            analysis_results['rankings'], output_dir
        )
        
        return output_files
    
    def _create_distance_heatmap(self, distance_analysis: Dict[str, Any], output_dir: str) -> str:
        """Create distance matrix heatmap."""
        distance_matrix = np.array(distance_analysis['distance_matrix'])
        model_ids = distance_analysis['model_ids']
        title = distance_analysis['title']
        
        # Create short labels for the heatmap
        short_labels = []
        for model_id in model_ids:
            info = self.model_manager.model_infos[model_id]
            if info.is_trained:
                short_labels.append(f"{info.architecture_type}_T")
            else:
                short_labels.append(f"{info.architecture_type}_R")
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)  # Mask upper triangle
        
        sns.heatmap(
            distance_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='viridis_r',
            square=True,
            xticklabels=short_labels,
            yticklabels=short_labels,
            cbar_kws={'label': 'DTW Distance'}
        )
        
        plt.title(f'Neural Network DTW Distance Matrix\n{title}', fontsize=14, pad=20)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'dtw_optimal_distance_matrix.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_parameter_impact_plots(self, parameter_impact: Dict[str, Any], output_dir: str) -> str:
        """Create parameter impact visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_data = [
            ('constraint_band', 'Constraint Band', parameter_impact['constraint_band']),
            ('eigenvalue_selection', 'Top-K Eigenvalues', parameter_impact['eigenvalue_selection']),
            ('threshold', 'Min Eigenvalue Threshold', parameter_impact['threshold']),
            ('interpolation', 'Interpolation Points', parameter_impact['interpolation'])
        ]
        
        for idx, (param_name, param_title, data) in enumerate(plot_data):
            ax = axes[idx]
            
            if not data:
                ax.text(0.5, 0.5, f'No data for\n{param_title}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(param_title)
                continue
            
            # Extract data for plotting
            x_values = list(data.keys())
            scores = [info['score'] for info in data.values()]
            separation_ratios = [info['separation_ratio'] for info in data.values()]
            
            # Plot score vs parameter value
            ax2 = ax.twinx()
            
            line1 = ax.plot(x_values, scores, 'bo-', label='Performance Score', linewidth=2, markersize=6)
            line2 = ax2.plot(x_values, separation_ratios, 'r^-', label='Separation Ratio', linewidth=2, markersize=6)
            
            ax.set_xlabel(param_title)
            ax.set_ylabel('Performance Score', color='b')
            ax2.set_ylabel('Separation Ratio', color='r')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Handle x-axis formatting
            if param_name in ['threshold']:
                ax.set_xscale('log')
            
            ax.set_title(f'{param_title} Impact')
            ax.grid(True, alpha=0.3)
            
            # Add combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.suptitle('DTW Parameter Impact Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'dtw_parameter_impact.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_ranking_plot(self, rankings: List, output_dir: str) -> str:
        """Create configuration ranking plot."""
        if not rankings:
            return ""
        
        # Extract top 15 configurations for visualization
        top_configs = rankings[:15]
        config_names = [name for name, score, validation in top_configs]
        scores = [score for name, score, validation in top_configs]
        separation_ratios = [validation.get('separation_ratio', 0.0) for name, score, validation in top_configs]
        
        # Shorten config names for better visualization
        short_names = []
        for name in config_names:
            if len(name) > 20:
                short_names.append(name[:17] + '...')
            else:
                short_names.append(name)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Performance scores
        bars1 = ax1.bar(range(len(short_names)), scores, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Top DTW Configuration Rankings by Performance Score')
        ax1.set_xticks(range(len(short_names)))
        ax1.set_xticklabels(short_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Separation ratios
        bars2 = ax2.bar(range(len(short_names)), separation_ratios, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Separation Ratio (Inter/Intra)')
        ax2.set_title('Separation Ratio: Trained vs Random Model Distances')
        ax2.set_xticks(range(len(short_names)))
        ax2.set_xticklabels(short_names, rotation=45, ha='right')
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Threshold (Inter > Intra)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add value labels on bars
        for bar, ratio in zip(bars2, separation_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'dtw_configuration_rankings.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename


def generate_comprehensive_report(analysis_results: Dict[str, Any], model_manager: ModelManager, 
                                output_file: str = "dtw_optimal_neural_comparison_report.json") -> str:
    """Generate comprehensive JSON report of the analysis."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare model information
    model_info_summary = {}
    for model_id, info in model_manager.model_infos.items():
        model_info_summary[model_id] = {
            'name': info.name,
            'path': info.path,
            'description': info.description,
            'is_trained': info.is_trained,
            'architecture_type': info.architecture_type,
            'parameters': sum(p.numel() for p in model_manager.models[model_id].parameters())
        }
    
    # Prepare best configuration details
    best_config = analysis_results['best_configuration']
    best_config_details = {}
    if best_config['config']:
        config = best_config['config']
        best_config_details = {
            'constraint_band': config.constraint_band,
            'eigenvalue_selection': config.eigenvalue_selection,
            'min_eigenvalue_threshold': config.min_eigenvalue_threshold,
            'interpolation_points': config.interpolation_points,
            'name': config.name
        }
    
    report = {
        'metadata': {
            'timestamp': timestamp,
            'analysis_type': 'DTW Optimal Neural Network Comparison',
            'objective': 'Find optimal DTW parameters for distinguishing trained vs random neural networks'
        },
        'models_analyzed': model_info_summary,
        'experimental_setup': {
            'total_configurations_tested': len(analysis_results.get('summary_stats', {})),
            'dtw_method': 'multivariate with interpolation',
            'eigenvalue_weight': 1.0,
            'structural_weight': 0.0,
            'analysis_focus': 'pure functional similarity'
        },
        'results': {
            'best_configuration': {
                'name': best_config['name'],
                'performance_score': best_config['score'],
                'parameters': best_config_details,
                'validation_metrics': best_config['validation']
            },
            'top_configurations': [
                {
                    'rank': i + 1,
                    'name': name,
                    'score': score,
                    'separation_ratio': validation.get('separation_ratio', 0.0),
                    'expected_pattern_satisfied': validation.get('expected_pattern', False)
                }
                for i, (name, score, validation) in enumerate(analysis_results['rankings'])
            ],
            'parameter_impact_analysis': analysis_results['parameter_impact']
        },
        'key_findings': {
            'optimal_constraint_band': None,
            'optimal_eigenvalue_selection': None,
            'optimal_threshold': None,
            'separation_achieved': best_config['validation'].get('separation_ratio', 0.0),
            'pattern_validation': best_config['validation'].get('expected_pattern', False)
        },
        'recommendations': {
            'dtw_configuration': best_config_details,
            'use_case': 'Neural network functional similarity analysis',
            'expected_performance': f"Separation ratio: {best_config['validation'].get('separation_ratio', 0.0):.2f}"
        },
        'summary_statistics': analysis_results.get('summary_stats', {})
    }
    
    # Extract key findings from parameter impact
    param_impact = analysis_results['parameter_impact']
    
    # Find best parameter values
    if param_impact['constraint_band']:
        best_cb = max(param_impact['constraint_band'].items(), key=lambda x: x[1]['score'])
        report['key_findings']['optimal_constraint_band'] = best_cb[0]
    
    if param_impact['eigenvalue_selection']:
        best_ev = max(param_impact['eigenvalue_selection'].items(), key=lambda x: x[1]['score'])
        report['key_findings']['optimal_eigenvalue_selection'] = best_ev[0]
    
    if param_impact['threshold']:
        best_th = max(param_impact['threshold'].items(), key=lambda x: x[1]['score'])
        report['key_findings']['optimal_threshold'] = best_th[0]
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return output_file


def main():
    """Main execution function."""
    print("🚀 DTW Optimal Parameter Tuning: FULL SPECTRUM vs LIMITED SPECTRUM")
    print("=" * 80)
    print("TESTING: Full eigenvalue spectrum (ALL eigenvalues) vs Limited selection")
    print("Finding optimal DTW parameters for distinguishing trained vs random networks")
    print("Expected: LOW distance between trained models, HIGH distance between trained vs random")
    print("Key Features:")
    print("  • FULL SPECTRUM: Using ALL eigenvalues with interpolation")
    print("  • Forced log-scale transformation for all eigenvalues")
    print("  • Interpolation-based multivariate DTW (handles different spectrum sizes)")
    print("  • Comparing full spectrum vs top-15/top-30 eigenvalues")
    print("  • Testing 8 configurations: 6 full spectrum + 2 limited baseline")
    
    try:
        # Initialize components
        print("\n📦 Initializing components...")
        model_manager = ModelManager()
        
        # Load all models and extract eigenvalue evolutions
        print("\n📥 Loading models and extracting eigenvalue evolutions...")
        models = model_manager.load_all_models()
        print(f"✅ Successfully loaded {len(models)} models")
        
        # Initialize parameter tuner
        print("\n🔧 Initializing parameter tuner...")
        parameter_tuner = ParameterTuner(model_manager)
        
        # Run parameter exploration
        print("\n🔍 Running comprehensive parameter exploration...")
        print("This may take several minutes as we test many parameter combinations...")
        results = parameter_tuner.run_parameter_exploration()
        print(f"✅ Completed parameter exploration: {len(results)} configurations tested")
        
        # Initialize validation engine
        print("\n✅ Initializing validation engine...")
        validation_engine = ValidationEngine(model_manager)
        
        # Initialize comprehensive analyzer
        print("\n📊 Performing comprehensive analysis...")
        analyzer = ComprehensiveAnalyzer(model_manager, validation_engine)
        analysis_results = analyzer.analyze_results(results)
        
        # Print preliminary results
        best_config = analysis_results['best_configuration']
        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"   Name: {best_config['name']}")
        print(f"   Performance Score: {best_config['score']:.3f}")
        if 'separation_ratio' in best_config['validation']:
            sep_ratio = best_config['validation']['separation_ratio']
            print(f"   Separation Ratio: {sep_ratio:.3f} {'✅' if sep_ratio > 1.0 else '❌'}")
            print(f"   Pattern Validated: {'✅' if best_config['validation'].get('expected_pattern', False) else '❌'}")
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        output_files = analyzer.create_visualizations(analysis_results)
        for viz_name, filename in output_files.items():
            if filename:
                print(f"   ✅ {viz_name}: {filename}")
        
        # Generate comprehensive report
        print("\n📄 Generating comprehensive report...")
        report_file = generate_comprehensive_report(analysis_results, model_manager)
        print(f"   ✅ Report saved: {report_file}")
        
        # Print summary recommendations
        print("\n" + "="*80)
        print("🎯 FINAL RECOMMENDATIONS FOR OPTIMAL DTW NEURAL NETWORK COMPARISON")
        print("="*80)
        
        if best_config['config']:
            config = best_config['config']
            print(f"Optimal DTW Configuration:")
            print(f"  • constraint_band: {config.constraint_band}")
            print(f"  • eigenvalue_selection: {config.eigenvalue_selection if config.eigenvalue_selection else 'All eigenvalues'}")
            print(f"  • min_eigenvalue_threshold: {config.min_eigenvalue_threshold}")
            print(f"  • interpolation_points: {config.interpolation_points if config.interpolation_points else 'Auto'}")
            print(f"  • Use multivariate=True, use_interpolation=True, match_all_eigenvalues=True")
        
        validation = best_config['validation'] 
        if validation:
            print(f"\nExpected Performance:")
            if 'separation_ratio' in validation:
                print(f"  • Separation Ratio: {validation['separation_ratio']:.3f} (higher is better)")
            if 'intra_trained_stats' in validation:
                intra_mean = validation['intra_trained_stats'].get('mean', 0.0)
                print(f"  • Intra-trained Distance: {intra_mean:.3f} (trained vs trained)")
            if 'inter_group_stats' in validation:
                inter_mean = validation['inter_group_stats'].get('mean', 0.0)
                print(f"  • Inter-group Distance: {inter_mean:.3f} (trained vs random)")
        
        print(f"\n📁 Generated Files:")
        print(f"  • Analysis Report: {report_file}")
        for viz_name, filename in output_files.items():
            if filename:
                print(f"  • {viz_name.replace('_', ' ').title()}: {filename}")
        
        print(f"\n✅ DTW Optimal Parameter Tuning Complete!")
        print(f"🎉 Use the recommended configuration for neural network functional similarity analysis")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
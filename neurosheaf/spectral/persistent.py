# neurosheaf/spectral/persistent.py
"""Persistent spectral analysis of neural sheaves.

This module provides the main high-level interface for persistent spectral
analysis, combining edge masking, eigenspace tracking, and persistence
computation into a unified analysis pipeline.

Key Features:
- Complete persistent spectral analysis pipeline
- Automatic filtration parameter generation
- Feature extraction from persistence results
- Persistence diagram generation
- Integration with existing sheaf construction
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError
from ..sheaf.data_structures import Sheaf
from .static_laplacian_unified import UnifiedStaticLaplacian as StaticLaplacianWithMasking
from .tracker import SubspaceTracker

logger = setup_logger(__name__)


class PersistentSpectralAnalyzer:
    """Main class for persistent spectral analysis of neural sheaves.
    
    This class provides a high-level interface for performing complete
    persistent spectral analysis, from sheaf input to persistence diagrams
    and extracted features.
    
    The analysis pipeline consists of:
    1. Filtration parameter generation
    2. Edge masking and Laplacian computation
    3. Eigenvalue/eigenvector computation
    4. Subspace tracking through filtration
    5. Feature extraction and persistence diagram generation
    
    Attributes:
        static_laplacian: StaticLaplacianWithMasking instance
        subspace_tracker: SubspaceTracker instance
        default_n_steps: Default number of filtration steps
        default_filtration_type: Default filtration type
    """
    
    def __init__(self,
                 static_laplacian: Optional[StaticLaplacianWithMasking] = None,
                 subspace_tracker: Optional[SubspaceTracker] = None,
                 default_n_steps: int = 50,
                 default_filtration_type: str = 'threshold'):
        """Initialize PersistentSpectralAnalyzer.
        
        Args:
            static_laplacian: StaticLaplacianWithMasking instance (auto-created if None)
            subspace_tracker: SubspaceTracker instance (auto-created if None)
            default_n_steps: Default number of filtration steps
            default_filtration_type: Default filtration type
        """
        self.static_laplacian = static_laplacian or StaticLaplacianWithMasking()
        self.subspace_tracker = subspace_tracker or SubspaceTracker()
        self.default_n_steps = default_n_steps
        self.default_filtration_type = default_filtration_type
        
        logger.info(f"PersistentSpectralAnalyzer initialized: "
                   f"default {default_n_steps} steps, {default_filtration_type} filtration")
    
    def analyze(self,
               sheaf: Sheaf,
               filtration_type: str = None,
               n_steps: int = None,
               param_range: Optional[Tuple[float, float]] = None,
               custom_threshold_func: Optional[Callable] = None) -> Dict:
        """Perform complete persistent spectral analysis.
        
        Args:
            sheaf: Sheaf object to analyze
            filtration_type: Type of filtration ('threshold', 'cka_based', 'custom')
            n_steps: Number of filtration steps (default: use default_n_steps)
            param_range: Range of filtration parameters (auto-detected if None)
            custom_threshold_func: Custom threshold function for 'custom' filtration type
            
        Returns:
            Complete analysis results with:
            - persistence_result: Raw persistence computation results
            - features: Extracted persistence features
            - diagrams: Persistence diagrams (birth-death pairs, infinite bars)
            - filtration_params: Parameter values used
            - filtration_type: Filtration type used
            - analysis_metadata: Timing and other metadata
        """
        # Use defaults if not specified
        filtration_type = filtration_type or self.default_filtration_type
        n_steps = n_steps or self.default_n_steps
        
        logger.info(f"Starting persistent spectral analysis: {filtration_type} filtration, "
                   f"{n_steps} steps")
        start_time = time.time()
        
        try:
            # Determine filtration parameters
            filtration_params = self._generate_filtration_params(
                sheaf, filtration_type, n_steps, param_range
            )
            
            # Create edge threshold function
            edge_threshold_func = self._create_edge_threshold_func(
                filtration_type, custom_threshold_func
            )
            
            # Compute persistence using static Laplacian with masking
            persistence_result = self.static_laplacian.compute_persistence(
                sheaf, filtration_params, edge_threshold_func
            )
            
            # Extract persistence features
            features = self._extract_persistence_features(persistence_result)
            
            # Generate persistence diagrams
            diagrams = self._generate_persistence_diagrams(
                persistence_result['tracking_info'],
                filtration_params
            )
            
            # Create analysis metadata
            analysis_time = time.time() - start_time
            analysis_metadata = {
                'analysis_time': analysis_time,
                'computation_time': persistence_result.get('computation_time', 0.0),
                'n_eigenvalue_sequences': len(persistence_result['eigenvalue_sequences']),
                'n_filtration_steps': len(filtration_params),
                'sheaf_nodes': len(sheaf.stalks),
                'sheaf_edges': len(sheaf.restrictions)
            }
            
            logger.info(f"Persistent spectral analysis completed in {analysis_time:.2f}s")
            
            return {
                'persistence_result': persistence_result,
                'features': features,
                'diagrams': diagrams,
                'filtration_params': filtration_params,
                'filtration_type': filtration_type,
                'analysis_metadata': analysis_metadata
            }
            
        except Exception as e:
            raise ComputationError(f"Persistent spectral analysis failed: {e}",
                                 operation="analyze")
    
    def _generate_filtration_params(self,
                                  sheaf: Sheaf,
                                  filtration_type: str,
                                  n_steps: int,
                                  param_range: Optional[Tuple[float, float]]) -> List[float]:
        """Generate filtration parameter sequence.
        
        Args:
            sheaf: Sheaf object
            filtration_type: Type of filtration
            n_steps: Number of steps
            param_range: Parameter range (auto-detected if None)
            
        Returns:
            List of filtration parameter values
        """
        if param_range is None:
            # Auto-detect range based on edge weights
            edge_weights = []
            for edge, restriction in sheaf.restrictions.items():
                weight = torch.norm(restriction, 'fro').item()
                edge_weights.append(weight)
            
            if not edge_weights:
                logger.warning("No edges found in sheaf, using default parameter range")
                param_range = (0.0, 1.0)
            else:
                min_weight = min(edge_weights)
                max_weight = max(edge_weights)
                # Extend range slightly to ensure all edges are captured
                param_range = (min_weight * 0.1, max_weight * 1.1)
                
                logger.info(f"Auto-detected parameter range: [{param_range[0]:.4f}, {param_range[1]:.4f}]")
        
        # Generate parameter sequence based on filtration type
        if filtration_type == 'threshold':
            # Linear sequence for threshold filtration
            params = np.linspace(param_range[0], param_range[1], n_steps)
        elif filtration_type == 'cka_based':
            # CKA values are typically in [0, 1]
            params = np.linspace(0.0, 1.0, n_steps)
        elif filtration_type == 'custom':
            # Use provided range with linear spacing
            params = np.linspace(param_range[0], param_range[1], n_steps)
        else:
            # Default to linear spacing
            logger.warning(f"Unknown filtration type '{filtration_type}', using linear spacing")
            params = np.linspace(param_range[0], param_range[1], n_steps)
        
        # Ensure parameters are sorted
        params = np.sort(params)
        
        logger.debug(f"Generated {len(params)} filtration parameters: "
                    f"[{params[0]:.4f}, ..., {params[-1]:.4f}]")
        
        return params.tolist()
    
    def _create_edge_threshold_func(self,
                                   filtration_type: str,
                                   custom_func: Optional[Callable] = None) -> Callable:
        """Create edge threshold function based on filtration type.
        
        Args:
            filtration_type: Type of filtration
            custom_func: Custom threshold function (used if filtration_type='custom')
            
        Returns:
            Function with signature (edge_weight, param) -> bool
        """
        if filtration_type == 'threshold':
            # Standard threshold: keep edges with weight >= parameter
            return lambda weight, param: weight >= param
        
        elif filtration_type == 'cka_based':
            # CKA-based: normalize by maximum weight then threshold
            return lambda weight, param: weight >= param
        
        elif filtration_type == 'custom':
            if custom_func is None:
                raise ValueError("Custom threshold function required for 'custom' filtration type")
            return custom_func
        
        else:
            logger.warning(f"Unknown filtration type '{filtration_type}', using default threshold")
            return lambda weight, param: weight >= param
    
    def _extract_persistence_features(self, persistence_result: Dict) -> Dict:
        """Extract features from persistence computation.
        
        Args:
            persistence_result: Results from StaticLaplacianWithMasking
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Get eigenvalue sequences
        eigenval_sequences = persistence_result['eigenvalue_sequences']
        
        # Initialize feature lists
        features['eigenvalue_evolution'] = []
        features['spectral_gap_evolution'] = []
        features['effective_dimension'] = []
        features['eigenvalue_statistics'] = []
        
        # Extract features for each filtration step
        for i, eigenvals in enumerate(eigenval_sequences):
            if len(eigenvals) == 0:
                # Handle empty eigenvalue case
                features['eigenvalue_evolution'].append({
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                })
                features['spectral_gap_evolution'].append(0.0)
                features['effective_dimension'].append(0.0)
                features['eigenvalue_statistics'].append({
                    'n_eigenvals': 0, 'n_zero': 0, 'n_small': 0
                })
                continue
            
            # Basic eigenvalue statistics
            eigenval_stats = {
                'mean': torch.mean(eigenvals).item(),
                'std': torch.std(eigenvals).item() if len(eigenvals) > 1 else 0.0,
                'min': torch.min(eigenvals).item(),
                'max': torch.max(eigenvals).item()
            }
            features['eigenvalue_evolution'].append(eigenval_stats)
            
            # Spectral gap (difference between first two eigenvalues)
            if len(eigenvals) > 1:
                gap = eigenvals[1] - eigenvals[0]
                features['spectral_gap_evolution'].append(gap.item())
            else:
                features['spectral_gap_evolution'].append(0.0)
            
            # Effective dimension (participation ratio)
            if len(eigenvals) > 0 and torch.sum(eigenvals) > 1e-12:
                normalized = eigenvals / torch.sum(eigenvals)
                eff_dim = 1.0 / torch.sum(normalized ** 2)
                features['effective_dimension'].append(eff_dim.item())
            else:
                features['effective_dimension'].append(0.0)
            
            # Additional eigenvalue statistics
            n_zero = torch.sum(eigenvals < 1e-10).item()
            n_small = torch.sum(eigenvals < 1e-6).item()
            features['eigenvalue_statistics'].append({
                'n_eigenvals': len(eigenvals),
                'n_zero': n_zero,
                'n_small': n_small
            })
        
        # Persistence-specific features from tracking
        tracking_info = persistence_result['tracking_info']
        features['num_birth_events'] = len(tracking_info['birth_events'])
        features['num_death_events'] = len(tracking_info['death_events'])
        features['num_crossings'] = len(tracking_info['crossings'])
        features['num_persistent_paths'] = len(tracking_info['eigenvalue_paths'])
        
        # Summary statistics
        features['summary'] = {
            'total_filtration_steps': len(eigenval_sequences),
            'mean_eigenvals_per_step': np.mean([len(seq) for seq in eigenval_sequences]),
            'mean_spectral_gap': np.mean(features['spectral_gap_evolution']),
            'mean_effective_dimension': np.mean(features['effective_dimension'])
        }
        
        logger.debug(f"Extracted features: {features['num_birth_events']} births, "
                    f"{features['num_death_events']} deaths, "
                    f"{features['num_persistent_paths']} paths")
        
        return features
    
    def _generate_persistence_diagrams(self,
                                     tracking_info: Dict,
                                     filtration_params: List[float]) -> Dict:
        """Generate persistence diagrams from continuous path tracking.
        
        MATHEMATICAL CORRECTION: Uses proper continuous eigenvalue paths instead of 
        the incorrect birth/death event pairing. This ensures mathematically valid
        persistence diagrams based on actual eigenvalue evolution.
        
        Args:
            tracking_info: Eigenspace tracking results from SubspaceTracker
            filtration_params: Filtration parameter values
            
        Returns:
            Dictionary with persistence diagrams based on continuous paths
        """
        diagrams = {
            'birth_death_pairs': [],
            'infinite_bars': [],
            'continuous_paths': tracking_info.get('continuous_paths', []),
            'path_based_computation': True  # Flag indicating correct method
        }
        
        # Use the mathematically correct continuous paths from tracker
        if 'continuous_paths' in tracking_info:
            # Direct extraction from continuous paths (correct approach)
            continuous_paths = tracking_info['continuous_paths']
            
            for path in continuous_paths:
                if path['death_param'] is not None:
                    # Finite persistence pair
                    pair = {
                        'birth': path['birth_param'],
                        'death': path['death_param'],
                        'lifetime': path['death_param'] - path['birth_param'],
                        'birth_step': path['birth_step'],
                        'death_step': path['death_step'],
                        'path_id': path['path_id'],
                        'eigenvalue_trace': path.get('eigenvalue_trace', [])
                    }
                    diagrams['birth_death_pairs'].append(pair)
                else:
                    # Infinite persistence bar
                    infinite_bar = {
                        'birth': path['birth_param'],
                        'death': float('inf'),
                        'birth_step': path['birth_step'],
                        'path_id': path['path_id'],
                        'eigenvalue_trace': path.get('eigenvalue_trace', [])
                    }
                    diagrams['infinite_bars'].append(infinite_bar)
        
        # Fallback to finite/infinite pairs if available
        elif 'finite_pairs' in tracking_info and 'infinite_pairs' in tracking_info:
            logger.warning("Using fallback finite/infinite pairs - consider updating SubspaceTracker")
            
            for pair in tracking_info['finite_pairs']:
                diagrams['birth_death_pairs'].append({
                    'birth': pair['birth_param'],
                    'death': pair['death_param'],
                    'lifetime': pair['lifetime'],
                    'path_id': pair.get('path_id', -1)
                })
            
            for pair in tracking_info['infinite_pairs']:
                diagrams['infinite_bars'].append({
                    'birth': pair['birth_param'],
                    'death': float('inf'),
                    'path_id': pair.get('path_id', -1)
                })
        
        # Emergency fallback to old event-based method (should not happen with updated tracker)
        else:
            logger.error("No continuous paths found - falling back to deprecated event pairing")
            logger.error("This indicates SubspaceTracker is not providing proper path tracking")
            
            # Keep old logic as emergency fallback only
            birth_events = tracking_info.get('birth_events', [])
            death_events = tracking_info.get('death_events', [])
            
            # Basic pairing for compatibility
            used_death_indices = set()
            for birth in birth_events:
                corresponding_death = None
                corresponding_death_idx = None
                min_death_step = float('inf')
                
                for i, death in enumerate(death_events):
                    if (death['step'] > birth['step'] and 
                        death['step'] < min_death_step and
                        i not in used_death_indices):
                        corresponding_death = death
                        corresponding_death_idx = i
                        min_death_step = death['step']
                
                if corresponding_death is not None:
                    pair = {
                        'birth': birth['filtration_param'],
                        'death': corresponding_death['filtration_param'],
                        'lifetime': corresponding_death['filtration_param'] - birth['filtration_param'],
                        'birth_step': birth['step'],
                        'death_step': corresponding_death['step'],
                        'deprecated_pairing': True
                    }
                    diagrams['birth_death_pairs'].append(pair)
                    used_death_indices.add(corresponding_death_idx)
                else:
                    infinite_bar = {
                        'birth': birth['filtration_param'],
                        'death': float('inf'),
                        'birth_step': birth['step'],
                        'deprecated_pairing': True
                    }
                    diagrams['infinite_bars'].append(infinite_bar)
        
        # Sort by birth time for consistency
        diagrams['birth_death_pairs'].sort(key=lambda x: x['birth'])
        diagrams['infinite_bars'].sort(key=lambda x: x['birth'])
        
        # Compute comprehensive statistics
        if diagrams['birth_death_pairs']:
            lifetimes = [pair['lifetime'] for pair in diagrams['birth_death_pairs']]
            diagrams['statistics'] = {
                'n_finite_pairs': len(diagrams['birth_death_pairs']),
                'n_infinite_bars': len(diagrams['infinite_bars']),
                'mean_lifetime': np.mean(lifetimes),
                'max_lifetime': max(lifetimes),
                'min_lifetime': min(lifetimes),
                'total_persistence': sum(lifetimes),
                'lifetime_std': np.std(lifetimes) if len(lifetimes) > 1 else 0.0
            }
        else:
            diagrams['statistics'] = {
                'n_finite_pairs': 0,
                'n_infinite_bars': len(diagrams['infinite_bars']),
                'mean_lifetime': 0.0,
                'max_lifetime': 0.0,
                'min_lifetime': 0.0,
                'total_persistence': 0.0,
                'lifetime_std': 0.0
            }
        
        # Add path-based validation metrics
        if 'continuous_paths' in tracking_info:
            total_paths = len(tracking_info['continuous_paths'])
            active_paths = len([p for p in tracking_info['continuous_paths'] if p.get('is_alive', True)])
            diagrams['path_statistics'] = {
                'total_paths': total_paths,
                'finite_paths': diagrams['statistics']['n_finite_pairs'],
                'infinite_paths': diagrams['statistics']['n_infinite_bars'],
                'path_completion_rate': (total_paths - active_paths) / max(total_paths, 1)
            }
        
        computation_method = "continuous_paths" if 'continuous_paths' in tracking_info else "event_pairing"
        logger.info(f"Generated persistence diagrams using {computation_method}: "
                   f"{diagrams['statistics']['n_finite_pairs']} finite pairs, "
                   f"{diagrams['statistics']['n_infinite_bars']} infinite bars")
        
        return diagrams
    
    def analyze_multiple_sheaves(self,
                                sheaves: List[Sheaf],
                                **analysis_kwargs) -> List[Dict]:
        """Analyze multiple sheaves with the same parameters.
        
        Args:
            sheaves: List of Sheaf objects to analyze
            **analysis_kwargs: Arguments passed to analyze() method
            
        Returns:
            List of analysis results, one per sheaf
        """
        logger.info(f"Analyzing {len(sheaves)} sheaves")
        
        results = []
        for i, sheaf in enumerate(sheaves):
            logger.debug(f"Analyzing sheaf {i+1}/{len(sheaves)}")
            result = self.analyze(sheaf, **analysis_kwargs)
            results.append(result)
        
        logger.info(f"Completed analysis of {len(sheaves)} sheaves")
        return results
    
    def clear_cache(self):
        """Clear cached data in underlying components."""
        self.static_laplacian.clear_cache()
        logger.info("Cleared PersistentSpectralAnalyzer cache")
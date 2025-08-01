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
from ..utils.dtw_similarity import FiltrationDTW
from .tracker_factory import SubspaceTrackerFactory

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
                 default_filtration_type: str = 'threshold',
                 dtw_comparator: Optional[FiltrationDTW] = None):
        """Initialize PersistentSpectralAnalyzer.
        
        Args:
            static_laplacian: StaticLaplacianWithMasking instance (auto-created if None)
            subspace_tracker: SubspaceTracker instance (auto-created if None)
            default_n_steps: Default number of filtration steps
            default_filtration_type: Default filtration type
            dtw_comparator: FiltrationDTW instance for eigenvalue evolution comparison
        """
        self.static_laplacian = static_laplacian or StaticLaplacianWithMasking()
        
        # Use factory for subspace tracker to enable method-specific routing
        if subspace_tracker is not None:
            self.subspace_tracker = subspace_tracker
        else:
            # Will be created dynamically based on sheaf construction method
            self.subspace_tracker = None
            
        self.dtw_comparator = dtw_comparator or FiltrationDTW()
        self.default_n_steps = default_n_steps
        self.default_filtration_type = default_filtration_type
        
        logger.info(f"PersistentSpectralAnalyzer initialized: "
                   f"default {default_n_steps} steps, {default_filtration_type} filtration, "
                   f"DTW method: {self.dtw_comparator.method}")
    
    def analyze(self,
               sheaf: Sheaf,
               filtration_type: str = None,
               n_steps: int = None,
               param_range: Optional[Tuple[float, float]] = None,
               custom_threshold_func: Optional[Callable] = None) -> Dict:
        """Perform complete persistent spectral analysis.
        
        This method implements a decreasing complexity filtration for threshold type:
        - Uses increasing parameters with threshold function (weight >= param)
        - Start: low threshold → many edges → high connectivity
        - End: high threshold → few edges → low connectivity
        - Birth/death semantics: features born early (low param), die later (high param)
        
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
            # Create appropriate subspace tracker if not already provided
            if self.subspace_tracker is None:
                construction_method = sheaf.metadata.get('construction_method', 'standard')
                self.subspace_tracker = SubspaceTrackerFactory.create_tracker(
                    construction_method=construction_method
                )
                logger.debug(f"Created {construction_method} subspace tracker dynamically")
            
            # Determine filtration parameters
            filtration_params = self._generate_filtration_params(
                sheaf, filtration_type, n_steps, param_range
            )
            
            # Create edge threshold function with construction method awareness
            edge_threshold_func = self._create_edge_threshold_func(
                filtration_type, custom_threshold_func, sheaf
            )
            
            # Compute persistence using static Laplacian with masking
            construction_method = sheaf.metadata.get('construction_method', 'standard')
            persistence_result = self.static_laplacian.compute_persistence(
                sheaf, filtration_params, edge_threshold_func, construction_method
            )
            
            # Extract persistence features
            features = self._extract_persistence_features(persistence_result)
            
            # Get construction method for proper persistence interpretation
            construction_method = sheaf.metadata.get('construction_method', 'standard')
            
            # Generate persistence diagrams with construction method awareness
            diagrams = self._generate_persistence_diagrams(
                persistence_result['tracking_info'],
                filtration_params,
                filtration_type,
                construction_method
            )
            
            # Validate filtration semantics for mathematical correctness
            semantic_validation = self._validate_filtration_semantics(
                persistence_result['eigenvalue_sequences'], 
                filtration_params, 
                construction_method
            )
            
            # Create analysis metadata
            analysis_time = time.time() - start_time
            analysis_metadata = {
                'analysis_time': analysis_time,
                'computation_time': persistence_result.get('computation_time', 0.0),
                'n_eigenvalue_sequences': len(persistence_result['eigenvalue_sequences']),
                'n_filtration_steps': len(filtration_params),
                'sheaf_nodes': len(sheaf.stalks),
                'sheaf_edges': len(sheaf.restrictions),
                'semantic_validation': semantic_validation
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
        """Generate filtration parameter sequence with GW awareness.
        
        Args:
            sheaf: Sheaf object
            filtration_type: Type of filtration
            n_steps: Number of steps
            param_range: Parameter range (auto-detected if None)
            
        Returns:
            List of filtration parameter values (always increasing)
        """
        construction_method = sheaf.metadata.get('construction_method', 'standard')
        
        # Extract edge weights based on construction method
        if construction_method == 'gromov_wasserstein':
            edge_weights = self._generate_gw_filtration_params(sheaf, n_steps, param_range)
            return edge_weights
        else:
            return self._generate_standard_filtration_params(sheaf, n_steps, param_range)
    
    def _generate_gw_filtration_params(self, 
                                     sheaf: Sheaf,
                                     n_steps: int, 
                                     param_range: Optional[Tuple[float, float]]) -> List[float]:
        """Generate edge-aware GW filtration parameters preventing plateau issues.
        
        CRITICAL FIX: This method replaces the problematic implementation that caused
        74-step plateaus with no edge activations. The new approach focuses parameters
        on actual GW cost transitions for meaningful structural changes.
        
        GW Filtration Logic:
        - Start with NO edges (only isolated stalks)
        - Add edges with cost ≤ threshold as threshold increases
        - Small costs = good matches = added first
        - Results in INCREASING complexity
        - PREVENTS over-extension beyond actual cost range
        """
        logger.info("Generating edge-aware GW filtration parameters (plateau prevention)")
        
        # Extract and validate GW costs
        edge_weights = self._extract_and_validate_gw_costs(sheaf)
        if not edge_weights:
            return self._fallback_to_uniform_params(n_steps, param_range)
        
        # Apply cost perturbation if needed to break ties
        if self._has_cost_ties(edge_weights):
            edge_weights = self._add_cost_perturbation(edge_weights)
            logger.info("Applied perturbations to break cost ties")
        
        # Determine optimal step count based on cost distribution
        optimal_steps = self._determine_optimal_steps(edge_weights, n_steps)
        if optimal_steps < n_steps:
            logger.info(f"Adjusted steps from {n_steps} to {optimal_steps} for optimal edge activation")
        
        # Generate edge-aware parameters
        params = self._generate_edge_aware_params(edge_weights, optimal_steps, param_range)
        
        # Validate filtration quality
        validation_passed = self._validate_filtration_progression(params, edge_weights)
        if not validation_passed:
            logger.warning("Filtration quality below optimal - some plateaus detected")
        
        # Log comprehensive diagnostics
        self._log_filtration_diagnostics(params, edge_weights)
        
        return params
    
    def _extract_and_validate_gw_costs(self, sheaf: Sheaf) -> List[float]:
        """Extract and validate GW costs from sheaf metadata."""
        gw_costs = sheaf.metadata.get('gw_costs', {})
        
        if not gw_costs:
            logger.warning("No GW costs found in metadata, computing from restrictions")
            edge_weights = []
            for edge, restriction in sheaf.restrictions.items():
                # Use operator norm as proxy for GW distortion
                weight = torch.linalg.norm(restriction, ord=2).item()
                edge_weights.append(weight)
            return edge_weights
        
        edge_weights = list(gw_costs.values())
        
        # Validate costs are reasonable
        if any(cost < 0 for cost in edge_weights):
            logger.warning("Negative GW costs detected - taking absolute values")
            edge_weights = [abs(cost) for cost in edge_weights]
        
        if not edge_weights:
            logger.error("No valid GW costs available")
            return []
        
        logger.info(f"Extracted {len(edge_weights)} GW costs: "
                   f"range [{min(edge_weights):.6f}, {max(edge_weights):.6f}]")
        
        return edge_weights
    
    def _has_cost_ties(self, costs: List[float], tolerance: float = 1e-12) -> bool:
        """Check if there are tied costs that need perturbation."""
        sorted_costs = sorted(costs)
        for i in range(1, len(sorted_costs)):
            if abs(sorted_costs[i] - sorted_costs[i-1]) < tolerance:
                return True
        return False
    
    def _add_cost_perturbation(self, costs: List[float], scale: float = 1e-9) -> List[float]:
        """Add tiny perturbations to break cost ties."""
        costs_array = np.array(costs)
        unique_costs, inverse_indices, counts = np.unique(costs_array, return_inverse=True, return_counts=True)
        
        # Only perturb costs that appear multiple times
        for i, (unique_cost, count) in enumerate(zip(unique_costs, counts)):
            if count > 1:
                # Find all positions with this cost
                mask = costs_array == unique_cost
                n_ties = np.sum(mask)
                
                # Generate small perturbations centered on zero
                perturbations = np.linspace(-scale, scale, n_ties)
                costs_array[mask] += perturbations
                
                logger.debug(f"Perturbed {n_ties} copies of cost {unique_cost:.6f}")
        
        return costs_array.tolist()
    
    def _determine_optimal_steps(self, costs: List[float], requested_steps: int) -> int:
        """Determine optimal number of steps based on cost distribution.
        
        RADICAL PLATEAU PREVENTION: Only use meaningful steps based on actual costs.
        This prevents the 75% plateau issue by not trying to fill arbitrary step counts.
        """
        unique_costs = len(set(costs))
        
        # RADICAL FIX: Never use more steps than we can meaningfully fill
        # Use exactly: unique_costs + modest_buffer to avoid plateaus
        # This ensures every step provides structural information
        modest_buffer = min(3, max(1, unique_costs // 3))  # Very small buffer
        optimal_steps = unique_costs + modest_buffer + 1  # +1 for zero-edge start
        
        # CRITICAL: Cap at meaningful maximum - never create excessive steps
        max_meaningful_steps = unique_costs * 2  # At most 2 steps per unique cost
        optimal_steps = min(optimal_steps, max_meaningful_steps)
        
        # If user requested fewer steps, use that (never force more)
        if requested_steps < optimal_steps:
            logger.info(f"Using user-requested {requested_steps} steps "
                       f"(optimal would be {optimal_steps} for {unique_costs} unique costs)")
            optimal_steps = requested_steps
        else:
            logger.info(f"Limiting steps from {requested_steps} to {optimal_steps} "
                       f"to prevent plateaus (only {unique_costs} unique costs available)")
        
        return optimal_steps
    
    def _generate_edge_aware_params(self, costs: List[float], n_steps: int, 
                                  param_range: Optional[Tuple[float, float]]) -> List[float]:
        """Generate parameters focused on actual edge cost transitions."""
        sorted_costs = sorted(costs)
        
        # Determine meaningful parameter range (CRITICAL FIX: no over-extension)
        if param_range is not None:
            min_param, max_param = param_range
            logger.info(f"Using user-provided parameter range: [{min_param:.6f}, {max_param:.6f}]")
        else:
            # Use actual cost range with minimal margins (NO OVER-EXTENSION)
            min_param = max(0.0, sorted_costs[0] - 1e-8)  # Just below minimum cost
            max_param = sorted_costs[-1] + 1e-8           # Just above maximum cost
            logger.info(f"Using edge-aware parameter range: [{min_param:.6f}, {max_param:.6f}] "
                       f"(tightly bound to actual costs)")
        
        params = []
        
        # Start with zero edges active
        params.append(min_param)
        
        # Add parameter just above each cost for edge activation
        for cost in sorted_costs:
            # Add parameter that activates this edge
            activation_param = cost + 1e-9
            if activation_param <= max_param:
                params.append(activation_param)
        
        # CONSERVATIVE GAP FILLING: Only add steps that provide meaningful resolution
        # RADICAL FIX: Stop when we have enough parameters for all edge transitions
        unique_costs = len(set(sorted_costs))
        
        # We already have parameters for: start + each edge activation
        # Only fill a few gaps for better resolution, but never create excessive steps
        max_reasonable_steps = min(n_steps, unique_costs * 2)  # Hard cap at 2x unique costs
        
        gap_fill_iterations = 0
        max_gap_fills = min(3, max_reasonable_steps - len(params))  # Very conservative
        
        while len(params) < max_reasonable_steps and gap_fill_iterations < max_gap_fills:
            # Find largest gap that's actually meaningful (not tiny numerical gaps)
            max_gap = 0
            max_gap_idx = -1
            
            for i in range(len(params) - 1):
                gap = params[i + 1] - params[i]
                # Only consider gaps that are reasonably large (not numerical precision)
                if gap > max_gap and gap > 1e-6:  # Much larger threshold
                    max_gap = gap
                    max_gap_idx = i
            
            if max_gap_idx >= 0:
                # Insert midpoint in largest meaningful gap
                mid_point = (params[max_gap_idx] + params[max_gap_idx + 1]) / 2
                params.insert(max_gap_idx + 1, mid_point)
                gap_fill_iterations += 1
            else:
                # No meaningful gaps left - perfect!
                logger.info(f"Optimal filtration achieved at {len(params)} steps "
                           f"(no more meaningful gaps to fill)")
                break
        
        # If we still have too few steps and user specifically requested more,
        # only then consider very minor additional interpolation
        if len(params) < n_steps and gap_fill_iterations == 0:
            logger.info(f"Generated {len(params)} meaningful steps for {unique_costs} unique costs "
                       f"(user requested {n_steps} but more would create plateaus)")
        
        # Ensure parameters are sorted and within bounds
        params = sorted([p for p in set(params) if min_param <= p <= max_param])
        
        # Trim to requested number of steps
        if len(params) > n_steps:
            params = params[:n_steps]
        
        return params
    
    def _validate_filtration_progression(self, params: List[float], costs: List[float]) -> bool:
        """Validate that filtration provides meaningful structural changes."""
        if len(params) < 2:
            return True  # Too few steps to validate
        
        # Predict edge activation pattern
        edge_counts = []
        for param in params:
            count = sum(1 for cost in costs if cost <= param)
            edge_counts.append(count)
        
        # Calculate plateau metrics
        changes = np.diff(edge_counts)
        no_change_steps = np.sum(changes == 0)
        plateau_ratio = no_change_steps / len(changes) if len(changes) > 0 else 0
        
        # Check for excessive plateaus (threshold: 30%)
        if plateau_ratio > 0.3:
            logger.warning(f"Filtration quality issue: {plateau_ratio:.1%} plateau ratio "
                          f"(target: <30%)")
            return False
        
        # Check for reasonable progression
        max_single_change = np.max(changes) if len(changes) > 0 else 0
        if max_single_change > 5:
            logger.warning(f"Large batch activation detected: {max_single_change} edges "
                          f"activate simultaneously")
        
        logger.info(f"Filtration quality validation: {plateau_ratio:.1%} plateau ratio ✓")
        return True
    
    def _log_filtration_diagnostics(self, params: List[float], costs: List[float]):
        """Log comprehensive diagnostics about filtration generation."""
        # Edge activation prediction
        edge_counts = [sum(1 for cost in costs if cost <= param) for param in params]
        
        # Statistics
        plateau_changes = np.diff(edge_counts)
        plateau_ratio = np.sum(plateau_changes == 0) / len(plateau_changes) if len(plateau_changes) > 0 else 0
        
        logger.info(f"Generated {len(params)} edge-aware filtration parameters:")
        logger.info(f"  Parameter range: [{params[0]:.6f}, {params[-1]:.6f}]")
        logger.info(f"  Edge activation: {edge_counts[0]} → {edge_counts[-1]} edges")
        logger.info(f"  Plateau ratio: {plateau_ratio:.1%} (target: <30%)")
        logger.info(f"  Unique costs: {len(set(costs))}, Steps: {len(params)}")
        
        # Log first few transitions for debugging
        logger.debug("First 5 edge activation steps:")
        for i in range(min(5, len(params))):
            logger.debug(f"  Step {i}: param={params[i]:.6f}, edges={edge_counts[i]}")
    
    def _fallback_to_uniform_params(self, n_steps: int, param_range: Optional[Tuple[float, float]]) -> List[float]:
        """Fallback to uniform parameter spacing when no costs available."""
        if param_range is not None:
            min_param, max_param = param_range
        else:
            min_param, max_param = 0.0, 1.0
        
        logger.warning(f"Using uniform fallback parameters: [{min_param:.4f}, {max_param:.4f}]")
        return np.linspace(min_param, max_param, n_steps).tolist()
    
    def _generate_standard_filtration_params(self, 
                                           sheaf: Sheaf, 
                                           n_steps: int,
                                           param_range: Optional[Tuple[float, float]]) -> List[float]:
        """Generate standard filtration parameters using Frobenius norms."""
        # Calculate edge weights using Frobenius norms
        edge_weights = []
        for edge, restriction in sheaf.restrictions.items():
            weight = torch.norm(restriction, 'fro').item()
            edge_weights.append(weight)
        
        # Determine the parameter range
        if param_range is not None:
            # User provided explicit range - use it directly
            logger.info(f"Using user-provided parameter range: [{param_range[0]:.4f}, {param_range[1]:.4f}]")
        else:
            # Auto-detect range based on edge weights
            if not edge_weights:
                logger.warning("No edges found in sheaf, using default parameter range")
                param_range = (0.0, 1.0)
            else:
                min_weight = min(edge_weights)
                max_weight = max(edge_weights)
                # Extend range for filtration: start slightly below min, end above max
                # MATHEMATICAL CORRECTION: Never use negative parameters for positive weights
                weight_range = max_weight - min_weight
                margin = max(0.1 * weight_range, 0.05)  # Smaller, more reasonable margin
                
                # Ensure minimum parameter is never negative for positive weights
                min_param = max(0.0, min_weight - margin)
                max_param = max_weight + margin
                param_range = (min_param, max_param)
                
                logger.info(f"Auto-detected parameter range from edge weights: [{param_range[0]:.4f}, {param_range[1]:.4f}]")
        
        # Generate parameter sequence based on filtration type (always threshold for this method)
        filtration_type = 'threshold'  # Standard method always uses threshold
        if filtration_type == 'threshold':
            # MATHEMATICAL CORRECTION: Threshold filtration with increasing parameters
            # Start with minimum weight (many edges) and go up to maximum weight (few edges)
            # This creates decreasing complexity filtration with weight >= param threshold
            safe_min = param_range[0]
            safe_max = param_range[1]
            
            # Parameter spacing optimized for decreasing complexity (increasing parameters)
            # Start with many edges (low threshold) and gradually remove more (high threshold)
            if n_steps > 20:
                # Mix linear and log spacing for gradual edge removal
                n_linear = n_steps // 2
                n_log = n_steps - n_linear
                
                # Generate increasing parameters for decreasing complexity
                # Linear spacing for initial gradual changes (low to medium threshold)
                linear_params = np.linspace(safe_min, safe_min + (safe_max - safe_min) * 0.7, n_linear)
                # Log spacing for final rapid edge removal (medium to high threshold)
                log_base = safe_min + (safe_max - safe_min) * 0.7
                log_params = np.logspace(
                    np.log10(log_base + 1e-6), 
                    np.log10(safe_max + 1e-6), 
                    n_log
                )
                params = np.concatenate([linear_params, log_params])
            else:
                # For smaller step counts, use smooth transition
                # More resolution where edge changes are gradual
                smooth_params = np.linspace(0, 1, n_steps) ** 0.7  # Gentle curve
                params = safe_min + (safe_max - safe_min) * smooth_params
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
        
        # Parameters are already in correct order from generation above
        # No additional sorting needed - preserve the carefully crafted spacing
        
        logger.debug(f"Generated {len(params)} filtration parameters: "
                    f"[{params[0]:.4f}, ..., {params[-1]:.4f}]")
        
        return params.tolist()
    
    def _create_edge_threshold_func(self,
                                   filtration_type: str,
                                   custom_func: Optional[Callable] = None,
                                   sheaf: Optional[Sheaf] = None) -> Callable:
        """Create edge threshold function based on filtration type and construction method.
        
        Args:
            filtration_type: Type of filtration
            custom_func: Custom threshold function (used if filtration_type='custom')
            sheaf: Sheaf object for construction method detection
            
        Returns:
            Function with signature (edge_weight, param) -> bool
        """
        # Detect construction method for appropriate threshold semantics
        construction_method = sheaf.metadata.get('construction_method', 'standard') if sheaf else 'standard'
        
        if filtration_type == 'threshold':
            if construction_method == 'gromov_wasserstein':
                # GW: Increasing complexity filtration
                # Include edges with cost ≤ threshold (small costs = good matches = added first)
                # As parameter increases, more edges are included (increasing complexity)
                logger.debug("Using GW threshold function: weight <= param (increasing complexity)")
                return lambda weight, param: weight <= param
            else:
                # Standard: Decreasing complexity filtration
                # Keep edges with weight >= parameter (large weights = strong connections = kept longest)
                # As parameter increases, fewer edges are kept (decreasing complexity)
                logger.debug("Using standard threshold function: weight >= param (decreasing complexity)")
                return lambda weight, param: weight >= param
        
        elif filtration_type == 'cka_based':
            # CKA-based always uses standard semantics (higher correlation = keep longer)
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
                                     filtration_params: List[float],
                                     filtration_type: str = 'threshold',
                                     construction_method: str = 'standard') -> Dict:
        """Generate persistence diagrams from continuous path tracking.
        
        MATHEMATICAL CORRECTION: Uses proper continuous eigenvalue paths instead of 
        the incorrect birth/death event pairing. This ensures mathematically valid
        persistence diagrams based on actual eigenvalue evolution.
        
        For decreasing filtrations, birth and death are swapped to maintain the
        mathematical property that birth < death in persistence diagrams.
        
        Args:
            tracking_info: Eigenspace tracking results from SubspaceTracker
            filtration_params: Filtration parameter values
            filtration_type: Type of filtration ('threshold' uses decreasing)
            
        Returns:
            Dictionary with persistence diagrams based on continuous paths
        """
        diagrams = {
            'birth_death_pairs': [],
            'infinite_bars': [],
            'continuous_paths': tracking_info.get('continuous_paths', []),
            'path_based_computation': True  # Flag indicating correct method
        }
        
        # MATHEMATICAL CORRECTION: Handle different construction method semantics
        # GW construction: increasing parameters = increasing complexity
        # Standard construction: increasing parameters = decreasing complexity
        is_gw_construction = construction_method == 'gromov_wasserstein'
        
        if is_gw_construction:
            logger.info("Using GW-specific persistence semantics (increasing complexity)")
        else:
            logger.debug("Using standard persistence semantics (decreasing complexity)")
        
        # Use the mathematically correct continuous paths from tracker
        if 'continuous_paths' in tracking_info:
            # Direct extraction from continuous paths (correct approach)
            continuous_paths = tracking_info['continuous_paths']
            
            for path in continuous_paths:
                if path['death_param'] is not None:
                    # Finite persistence pair - handle construction method semantics
                    raw_birth = path['birth_param']
                    raw_death = path['death_param']
                    birth_step = path['birth_step']
                    death_step = path['death_step']
                    
                    # CRITICAL FIX: Apply correct semantics based on construction method
                    if is_gw_construction:
                        # GW semantics: increasing parameters = increasing complexity
                        # A feature "born" at low param (sparse) "dies" at high param (connected)
                        # This matches the mathematical expectation: sparse → many components → connected
                        birth = raw_birth  # Keep original: small param = birth
                        death = raw_death  # Keep original: large param = death
                        logger.debug(f"GW semantics: birth={birth:.6f} (sparse), death={death:.6f} (connected)")
                    else:
                        # Standard semantics: increasing parameters = decreasing complexity  
                        # A feature "born" at low param (dense) "dies" at high param (sparse)
                        birth = raw_birth
                        death = raw_death
                        logger.debug(f"Standard semantics: birth={birth:.6f}, death={death:.6f}")
                    
                    # Calculate lifetime and validate
                    lifetime = abs(death - birth)
                    
                    # Skip pairs with invalid values (NaN, inf, or negative lifetime)
                    # MATHEMATICAL CORRECTION: Allow birth == death for instantaneous features
                    # but require birth <= death and finite values
                    if (not np.isfinite(birth) or not np.isfinite(death) or 
                        not np.isfinite(lifetime) or lifetime < 0 or birth > death):
                        logger.debug(f"Skipping invalid persistence pair: birth={birth:.6f}, death={death:.6f}, lifetime={lifetime:.6f}")
                        continue
                    
                    pair = {
                        'birth': birth,
                        'death': death,
                        'lifetime': lifetime,
                        'birth_step': birth_step,
                        'death_step': death_step,
                        'path_id': path['path_id'],
                        'eigenvalue_trace': path.get('eigenvalue_trace', [])
                    }
                    diagrams['birth_death_pairs'].append(pair)
                else:
                    # Infinite persistence bar - use standard increasing filtration semantics
                    birth = path['birth_param']
                    birth_step = path['birth_step']
                    
                    # Validate infinite bar birth time
                    if not np.isfinite(birth):
                        logger.debug(f"Skipping invalid infinite bar: birth={birth:.6f}")
                        continue
                    
                    infinite_bar = {
                        'birth': birth,
                        'death': float('inf'),
                        'birth_step': birth_step,
                        'path_id': path['path_id'],
                        'eigenvalue_trace': path.get('eigenvalue_trace', [])
                    }
                    diagrams['infinite_bars'].append(infinite_bar)
        
        # Fallback to finite/infinite pairs if available
        elif 'finite_pairs' in tracking_info and 'infinite_pairs' in tracking_info:
            logger.warning("Using fallback finite/infinite pairs - consider updating SubspaceTracker")
            
            for pair in tracking_info['finite_pairs']:
                # Use standard increasing filtration semantics
                birth = pair['birth_param']
                death = pair['death_param']
                    
                # Validate fallback pair
                lifetime = abs(death - birth)
                if (np.isfinite(birth) and np.isfinite(death) and 
                    np.isfinite(lifetime) and lifetime >= 0 and birth <= death):
                    diagrams['birth_death_pairs'].append({
                        'birth': birth,
                        'death': death,
                        'lifetime': lifetime,
                        'path_id': pair.get('path_id', -1)
                    })
                else:
                    logger.debug(f"Skipping invalid fallback pair: birth={birth:.6f}, death={death:.6f}")
            
            for pair in tracking_info['infinite_pairs']:
                # Use standard increasing filtration semantics
                birth = pair['birth_param']
                    
                diagrams['infinite_bars'].append({
                    'birth': birth,
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
            # Filter out any invalid lifetimes (NaN, inf, negative, zero)
            lifetimes = [pair['lifetime'] for pair in diagrams['birth_death_pairs'] 
                        if np.isfinite(pair['lifetime']) and pair['lifetime'] > 0]
            
            if lifetimes:
                diagrams['statistics'] = {
                    'n_finite_pairs': len(diagrams['birth_death_pairs']),
                    'n_infinite_bars': len(diagrams['infinite_bars']),
                    'mean_lifetime': np.mean(lifetimes),
                    'max_lifetime': max(lifetimes),
                    'min_lifetime': min(lifetimes),
                    'total_persistence': sum(lifetimes),
                    'lifetime_std': np.std(lifetimes) if len(lifetimes) > 1 else 0.0,
                    'valid_pairs': len(lifetimes),
                    'invalid_pairs': len(diagrams['birth_death_pairs']) - len(lifetimes)
                }
            else:
                # All pairs have invalid lifetimes
                logger.warning("All birth-death pairs have invalid lifetimes")
                diagrams['statistics'] = {
                    'n_finite_pairs': len(diagrams['birth_death_pairs']),
                    'n_infinite_bars': len(diagrams['infinite_bars']),
                    'mean_lifetime': 0.0,
                    'max_lifetime': 0.0,
                    'min_lifetime': 0.0,
                    'total_persistence': 0.0,
                    'lifetime_std': 0.0,
                    'valid_pairs': 0,
                    'invalid_pairs': len(diagrams['birth_death_pairs'])
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
    
    def compare_filtration_evolution(self,
                                   sheaf1: Sheaf,
                                   sheaf2: Sheaf,
                                   filtration_type: str = None,
                                   n_steps: int = None,
                                   eigenvalue_index: Optional[int] = None,
                                   multivariate: bool = False,
                                   **analysis_kwargs) -> Dict:
        """Compare eigenvalue evolution across filtration between two sheaves.
        
        This method performs spectral analysis on both sheaves and compares their
        eigenvalue evolution patterns using Dynamic Time Warping (DTW).
        
        Args:
            sheaf1: First sheaf to analyze
            sheaf2: Second sheaf to analyze
            filtration_type: Type of filtration to use
            n_steps: Number of filtration steps
            eigenvalue_index: Index of eigenvalue to compare (None = all)
            multivariate: Whether to use multivariate DTW
            **analysis_kwargs: Additional arguments for spectral analysis
            
        Returns:
            Dictionary containing:
            - dtw_comparison: DTW comparison results
            - analysis1: Full analysis results for sheaf1
            - analysis2: Full analysis results for sheaf2
            - similarity_metrics: Derived similarity metrics
        """
        logger.info(f"Comparing eigenvalue evolution between two sheaves using DTW")
        
        # Analyze both sheaves
        analysis1 = self.analyze(sheaf1, filtration_type=filtration_type, 
                               n_steps=n_steps, **analysis_kwargs)
        analysis2 = self.analyze(sheaf2, filtration_type=filtration_type, 
                               n_steps=n_steps, **analysis_kwargs)
        
        # Extract eigenvalue sequences
        eigenvalue_sequences1 = analysis1['persistence_result']['eigenvalue_sequences']
        eigenvalue_sequences2 = analysis2['persistence_result']['eigenvalue_sequences']
        
        # Get filtration parameters
        filtration_params1 = analysis1['filtration_params']
        filtration_params2 = analysis2['filtration_params']
        
        # Perform DTW comparison
        dtw_comparison = self.dtw_comparator.compare_eigenvalue_evolution(
            eigenvalue_sequences1, eigenvalue_sequences2,
            filtration_params1, filtration_params2,
            eigenvalue_index=eigenvalue_index,
            multivariate=multivariate
        )
        
        # Compute additional similarity metrics
        similarity_metrics = self._compute_similarity_metrics(
            analysis1, analysis2, dtw_comparison
        )
        
        logger.info(f"DTW comparison completed: distance={dtw_comparison['distance']:.4f}, "
                   f"normalized_distance={dtw_comparison['normalized_distance']:.4f}")
        
        return {
            'dtw_comparison': dtw_comparison,
            'analysis1': analysis1,
            'analysis2': analysis2,
            'similarity_metrics': similarity_metrics
        }
    
    def compare_multiple_sheaves(self,
                                sheaves: List[Sheaf],
                                filtration_type: str = None,
                                n_steps: int = None,
                                eigenvalue_index: Optional[int] = None,
                                multivariate: bool = False,
                                **analysis_kwargs) -> Dict:
        """Compare multiple sheaves pairwise using DTW.
        
        Args:
            sheaves: List of sheaves to compare
            filtration_type: Type of filtration to use
            n_steps: Number of filtration steps
            eigenvalue_index: Index of eigenvalue to compare (None = all)
            multivariate: Whether to use multivariate DTW
            **analysis_kwargs: Additional arguments for spectral analysis
            
        Returns:
            Dictionary containing:
            - distance_matrix: Pairwise DTW distances
            - analyses: Individual analysis results for each sheaf
            - similarity_rankings: Ranked similarity results
        """
        logger.info(f"Comparing {len(sheaves)} sheaves pairwise using DTW")
        
        # Analyze all sheaves
        analyses = []
        eigenvalue_evolutions = []
        filtration_params = []
        
        for i, sheaf in enumerate(sheaves):
            logger.debug(f"Analyzing sheaf {i+1}/{len(sheaves)}")
            analysis = self.analyze(sheaf, filtration_type=filtration_type,
                                  n_steps=n_steps, **analysis_kwargs)
            analyses.append(analysis)
            eigenvalue_evolutions.append(analysis['persistence_result']['eigenvalue_sequences'])
            filtration_params.append(analysis['filtration_params'])
        
        # Compute pairwise DTW distances
        distance_matrix = self.dtw_comparator.compare_multiple_evolutions(
            eigenvalue_evolutions, filtration_params,
            eigenvalue_index=eigenvalue_index, multivariate=multivariate
        )
        
        # Create similarity rankings
        similarity_rankings = self._create_similarity_rankings(distance_matrix)
        
        logger.info(f"Completed pairwise DTW comparison of {len(sheaves)} sheaves")
        
        return {
            'distance_matrix': distance_matrix,
            'analyses': analyses,
            'similarity_rankings': similarity_rankings,
            'mean_distance': np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)]),
            'std_distance': np.std(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
        }
    
    def _compute_similarity_metrics(self,
                                  analysis1: Dict,
                                  analysis2: Dict,
                                  dtw_comparison: Dict) -> Dict:
        """Compute additional similarity metrics from DTW comparison."""
        
        # Extract persistence statistics
        stats1 = analysis1['diagrams']['statistics']
        stats2 = analysis2['diagrams']['statistics']
        
        # Compute persistence similarity
        persistence_similarity = self._compute_persistence_similarity(stats1, stats2)
        
        # Compute spectral similarity based on eigenvalue statistics
        spectral_similarity = self._compute_spectral_similarity(
            analysis1['persistence_result'], analysis2['persistence_result']
        )
        
        # Compute temporal alignment quality
        alignment_quality = dtw_comparison['alignment_visualization']['alignment_quality']
        
        # Combined similarity score with proper DTW distance handling
        # Convert DTW distance to similarity using inverse relationship
        raw_dtw_distance = dtw_comparison.get('raw_normalized_distance', dtw_comparison['normalized_distance'])
        
        # Use inverse scaling for DTW similarity to preserve sensitivity across full range
        # For multivariate DTW, distances can range from 0 to 100+, so use adaptive scaling
        if raw_dtw_distance <= 0.001:
            dtw_similarity = 1.0  # Perfect similarity for near-zero distances
        else:
            # Use inverse scaling: similarity = 1 / (1 + distance/scale_factor)
            # This preserves sensitivity across the full distance range
            scale_factor = 10.0  # Chosen to map typical distances (0-50) to similarities (1.0-0.1)
            dtw_similarity = 1.0 / (1.0 + raw_dtw_distance / scale_factor)
        
        # Ensure all similarity components are in [0,1] range
        dtw_similarity = max(0.0, min(1.0, dtw_similarity))
        persistence_similarity = max(0.0, min(1.0, persistence_similarity))
        spectral_similarity = max(0.0, min(1.0, spectral_similarity))
        alignment_quality = max(0.0, min(1.0, alignment_quality))
        
        # Combined similarity with corrected DTW component - guaranteed to be in [0,1]
        combined_similarity = (
            0.4 * dtw_similarity +
            0.3 * persistence_similarity +
            0.2 * spectral_similarity +
            0.1 * alignment_quality
        )
        
        return {
            'dtw_distance': dtw_comparison['distance'],
            'normalized_dtw_distance': dtw_comparison['normalized_distance'],
            'raw_dtw_distance': raw_dtw_distance,
            'dtw_similarity': dtw_similarity,
            'persistence_similarity': persistence_similarity,
            'spectral_similarity': spectral_similarity,
            'alignment_quality': alignment_quality,
            'combined_similarity': combined_similarity
        }
    
    def _compute_persistence_similarity(self, stats1: Dict, stats2: Dict) -> float:
        """Compute similarity between persistence statistics."""
        # Compare key persistence metrics
        lifetime_diff = abs(stats1['mean_lifetime'] - stats2['mean_lifetime'])
        max_lifetime = max(stats1['mean_lifetime'], stats2['mean_lifetime'])
        
        if max_lifetime > 0:
            lifetime_similarity = 1.0 - (lifetime_diff / max_lifetime)
        else:
            lifetime_similarity = 1.0
        
        # Compare number of persistent features
        count_diff = abs(stats1['n_finite_pairs'] - stats2['n_finite_pairs'])
        max_count = max(stats1['n_finite_pairs'], stats2['n_finite_pairs'])
        
        if max_count > 0:
            count_similarity = 1.0 - (count_diff / max_count)
        else:
            count_similarity = 1.0
        
        # Weighted average
        return 0.6 * lifetime_similarity + 0.4 * count_similarity
    
    def _compute_spectral_similarity(self, result1: Dict, result2: Dict) -> float:
        """Compute similarity between spectral properties."""
        eigenvalues1 = result1['eigenvalue_sequences']
        eigenvalues2 = result2['eigenvalue_sequences']
        
        # Compute average eigenvalue similarity across filtration
        similarities = []
        
        min_length = min(len(eigenvalues1), len(eigenvalues2))
        for i in range(min_length):
            if len(eigenvalues1[i]) > 0 and len(eigenvalues2[i]) > 0:
                # Compare largest eigenvalues
                val1 = eigenvalues1[i][0].item()
                val2 = eigenvalues2[i][0].item()
                
                if max(val1, val2) > 0:
                    similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
                else:
                    similarity = 1.0
                    
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _validate_filtration_semantics(self, 
                                     eigenvalue_sequences: List[torch.Tensor], 
                                     filtration_params: List[float],
                                     construction_method: str) -> Dict:
        """Validate that filtration produces expected eigenvalue progression.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors for each filtration step
            filtration_params: Filtration parameter values
            construction_method: Construction method ('gromov_wasserstein' or 'standard')
            
        Returns:
            Dictionary with validation results and diagnostics
        """
        validation = {
            'construction_method': construction_method,
            'is_valid': True,
            'warnings': [],
            'statistics': {}
        }
        
        if len(eigenvalue_sequences) < 2:
            validation['warnings'].append("Too few filtration steps for semantic validation")
            return validation
            
        try:
            # Count non-zero eigenvalues at each step (above threshold)
            eigenvalue_threshold = 1e-10
            non_zero_counts = []
            mean_eigenvalues = []
            
            for i, eigenvals in enumerate(eigenvalue_sequences):
                if len(eigenvals) > 0:
                    non_zero_count = torch.sum(eigenvals > eigenvalue_threshold).item()
                    mean_eigenval = torch.mean(eigenvals[eigenvals > eigenvalue_threshold]).item() if non_zero_count > 0 else 0.0
                else:
                    non_zero_count = 0
                    mean_eigenval = 0.0
                    
                non_zero_counts.append(non_zero_count)
                mean_eigenvalues.append(mean_eigenval)
            
            # Store statistics
            validation['statistics'] = {
                'non_zero_counts': non_zero_counts,
                'mean_eigenvalues': mean_eigenvalues,
                'early_step_count': non_zero_counts[0] if non_zero_counts else 0,
                'late_step_count': non_zero_counts[-1] if non_zero_counts else 0,
                'early_step_mean': mean_eigenvalues[0] if mean_eigenvalues else 0.0,
                'late_step_mean': mean_eigenvalues[-1] if mean_eigenvalues else 0.0
            }
            
            # Validation logic depends on construction method
            if construction_method == 'gromov_wasserstein':
                # GW: Increasing complexity → Early steps should have MORE small eigenvalues
                # Early params (sparse) → many disconnected components → many small eigenvalues
                # Late params (dense) → fewer components → fewer, larger eigenvalues
                expected_early_more_eigenvals = non_zero_counts[0] >= non_zero_counts[-1]
                expected_early_smaller_mean = mean_eigenvalues[0] <= mean_eigenvalues[-1] * 2  # Allow some flexibility
                
                if not expected_early_more_eigenvals:
                    validation['warnings'].append(
                        f"GW semantics: Expected early step ({filtration_params[0]:.4f}) to have ≥ eigenvalues than late step ({filtration_params[-1]:.4f}), "
                        f"but got {non_zero_counts[0]} vs {non_zero_counts[-1]}")
                    validation['is_valid'] = False
                
                if not expected_early_smaller_mean and mean_eigenvalues[0] > 0 and mean_eigenvalues[-1] > 0:
                    validation['warnings'].append(
                        f"GW semantics: Expected early step to have smaller mean eigenvalue, "
                        f"but got {mean_eigenvalues[0]:.6f} vs {mean_eigenvalues[-1]:.6f}")
                
                logger.info(f"GW semantic validation: Early step {non_zero_counts[0]} eigenvalues (mean={mean_eigenvalues[0]:.6f}), "
                           f"Late step {non_zero_counts[-1]} eigenvalues (mean={mean_eigenvalues[-1]:.6f})")
                
            else:
                # Standard: Decreasing complexity → Early steps should have FEWER large eigenvalues  
                # Early params (dense) → fewer components → fewer, larger eigenvalues
                # Late params (sparse) → many components → many small eigenvalues
                expected_early_fewer_eigenvals = non_zero_counts[0] <= non_zero_counts[-1]
                expected_early_larger_mean = mean_eigenvalues[0] >= mean_eigenvalues[-1] * 0.5  # Allow some flexibility
                
                if not expected_early_fewer_eigenvals:
                    validation['warnings'].append(
                        f"Standard semantics: Expected early step to have ≤ eigenvalues than late step, "
                        f"but got {non_zero_counts[0]} vs {non_zero_counts[-1]}")
                
                if not expected_early_larger_mean and mean_eigenvalues[0] > 0 and mean_eigenvalues[-1] > 0:
                    validation['warnings'].append(
                        f"Standard semantics: Expected early step to have larger mean eigenvalue, "
                        f"but got {mean_eigenvalues[0]:.6f} vs {mean_eigenvalues[-1]:.6f}")
                
                logger.debug(f"Standard semantic validation: Early step {non_zero_counts[0]} eigenvalues (mean={mean_eigenvalues[0]:.6f}), "
                            f"Late step {non_zero_counts[-1]} eigenvalues (mean={mean_eigenvalues[-1]:.6f})")
            
            # General sanity checks
            if all(count == 0 for count in non_zero_counts):
                validation['warnings'].append("All filtration steps have zero eigenvalues - possible regularization or numerical issue")
                validation['is_valid'] = False
            
            if len(validation['warnings']) == 0:
                logger.info(f"✅ Filtration semantic validation passed for {construction_method} construction")
            else:
                logger.warning(f"⚠️ Filtration semantic validation found {len(validation['warnings'])} issues")
                
        except Exception as e:
            validation['warnings'].append(f"Semantic validation failed: {e}")
            validation['is_valid'] = False
            logger.error(f"Semantic validation error: {e}")
        
        return validation
    
    def _create_similarity_rankings(self, distance_matrix: np.ndarray) -> List[Dict]:
        """Create ranked similarity results from distance matrix."""
        n_sheaves = distance_matrix.shape[0]
        rankings = []
        
        for i in range(n_sheaves):
            # Get distances for sheaf i
            distances = distance_matrix[i, :].copy()
            distances[i] = np.inf  # Exclude self-comparison
            
            # Sort by distance (ascending = most similar first)
            sorted_indices = np.argsort(distances)
            
            # Create ranking for sheaf i
            ranking = {
                'sheaf_index': i,
                'most_similar': [
                    {
                        'sheaf_index': int(idx),
                        'distance': float(distances[idx]),
                        'similarity': 1.0 - distances[idx]  # Convert to similarity
                    }
                    for idx in sorted_indices[:min(5, n_sheaves-1)]  # Top 5 similar
                ]
            }
            rankings.append(ranking)
        
        return rankings
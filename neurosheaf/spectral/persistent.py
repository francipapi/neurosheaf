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
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Sequence, Literal
import time
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError
from ..sheaf.data_structures import Sheaf
from .static_laplacian_unified import UnifiedStaticLaplacian as StaticLaplacianWithMasking
from .tracker import SubspaceTracker
from ..utils.dtw_similarity import FiltrationDTW
from .tracker_factory import SubspaceTrackerFactory
from .h0_persistence import H0PersistenceTracker
from ..io.config import H0Config, DEFAULT_H0_CONFIG
from .flows.alpha_flow import AlphaGroupingPolicy, AlphaFlowBuilder
from .flows.diffusion_flow import DiffusionSpec, DiffusionSummaries, DiffusionFlowAnalyzer

logger = setup_logger(__name__)


@dataclass(frozen=True)
class StaticBuildConfig:
    """Configuration for flow-based static Laplacian building.
    
    This configuration controls how Laplacian operators are constructed
    for α-flow and t-flow analysis, with emphasis on cross-architecture
    comparability and numerical stability.
    
    Attributes:
        mass_mode: Mass matrix mode ('fixed' for consistency, 'adaptive' for accuracy)
        precision: Computation precision ('double' or 'single')
        normalization: Applied to summaries only, not operators (None, 'trace', 'frobenius')
        random_state: Random seed for reproducible computations
    """
    mass_mode: Literal['fixed', 'adaptive'] = 'fixed'
    precision: Literal['double', 'single'] = 'double'
    normalization: Optional[str] = None  # Applied to summaries only, not operators
    random_state: Optional[int] = None


@dataclass(frozen=True)
class AlphaFlowSpec:
    """Specification for α-flow analysis.
    
    The α-flow method analyzes network structure using baseline/residual
    Laplacian decomposition: L(α) = L_base + α*L_resid.
    
    Attributes:
        alpha_grid: Sequence of α values for analysis
        k_small: Number of smallest eigenvalues to compute
        probes: Number of Hutchinson probes for trace estimation
        moments: Powers k for computing Tr(L^k) moments
        grouping: Policy for partitioning edges into base/residual sets
        sigma: Shift-invert parameter for eigenvalue computation
        eigen_use_csr: Use explicit CSR matrices for eigenvalue computation (faster shift-invert)
    """
    alpha_grid: Sequence[float] = (0.0, 0.1, 0.3, 1.0, 3.0)
    k_small: int = 16
    probes: int = 64
    moments: Sequence[int] = (1, 2, 3, 4, 5, 6)
    grouping: AlphaGroupingPolicy = AlphaGroupingPolicy()
    sigma: float = 1e-6
    eigen_use_csr: bool = True


@dataclass(frozen=True)
class DiffusionFlowSpec:
    """Specification for t-flow (diffusion) analysis.
    
    The t-flow method analyzes multi-scale structure using heat kernel
    summaries: h(t) = Tr(exp(-t*L))/n over different time scales.
    
    Attributes:
        t_grid: Time points for analysis ('auto' for automatic generation or explicit sequence)
        k_small: Number of smallest eigenvalues to compute
        probes: Number of Hutchinson probes for SLQ estimation
        slq_iters: Number of Lanczos iterations for SLQ computation
        sigma: Shift-invert parameter for eigenvalue computation
    """
    t_grid: Union[Sequence[float], Literal['auto']] = 'auto'
    k_small: int = 16
    probes: int = 64
    slq_iters: int = 30
    sigma: float = 1e-6


@dataclass
class AlphaFlowPoint:
    """Results for a single α value in α-flow analysis.
    
    Attributes:
        alpha: The α parameter value
        eigenvalues: k smallest eigenvalues of L(α)
        moments: Hutchinson moment estimates Tr(L(α)^k)
        moment_stds: Standard deviations of moment estimates
        trace_normalized: Normalized trace Tr(L(α))/n
        frobenius_normalized: Normalized Frobenius norm ||L(α)||²_F/n²
        meta: Computation metadata (timing, convergence, etc.)
    """
    alpha: float
    eigenvalues: np.ndarray
    moments: Dict[int, float]  # {k: Tr(L^k)}
    moment_stds: Dict[int, float]  # {k: std(Tr(L^k))}
    trace_normalized: float
    frobenius_normalized: float
    meta: Dict[str, Any]


@dataclass
class AlphaFlowResult:
    """Complete α-flow analysis results.
    
    Attributes:
        points: Results for each α value
        grouping_meta: Edge partitioning metadata
        build_meta: Laplacian construction metadata
        analysis_time: Total analysis time in seconds
        spec: Original analysis specification
        config: Build configuration used
    """
    points: List[AlphaFlowPoint]
    grouping_meta: Dict[str, Any]
    build_meta: Dict[str, Any]
    analysis_time: float
    spec: AlphaFlowSpec
    config: StaticBuildConfig


@dataclass
class DiffusionFlowResult:
    """Complete t-flow (diffusion) analysis results.
    
    Attributes:
        summaries: Heat trace summaries from diffusion analysis
        build_meta: Laplacian construction metadata  
        analysis_time: Total analysis time in seconds
        spec: Original analysis specification
        config: Build configuration used
    """
    summaries: DiffusionSummaries
    build_meta: Dict[str, Any]
    analysis_time: float
    spec: DiffusionFlowSpec
    config: StaticBuildConfig


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
               custom_threshold_func: Optional[Callable] = None,
               h0_config: Optional[H0Config] = None) -> Dict:
        """Perform complete persistent spectral analysis.
        
        This method implements both standard eigenvalue-based persistence and H⁰ 
        Global Section persistence for Gromov-Wasserstein sheaves:
        
        Standard construction: decreasing complexity filtration
        - Uses increasing parameters with threshold function (weight >= param)
        - Start: low threshold → many edges → high connectivity
        - End: high threshold → few edges → low connectivity
        
        GW construction: H⁰ Global Section persistence
        - Transport-informed kernel tracking with RRQR-based persistence
        - Uses whitened coboundary operators and GW coupling transport maps
        - More mathematically principled for optimal transport-based sheaves
        
        Args:
            sheaf: Sheaf object to analyze
            filtration_type: Type of filtration ('threshold', 'cka_based', 'custom')
            n_steps: Number of filtration steps (default: use default_n_steps)
            param_range: Range of filtration parameters (auto-detected if None)
            custom_threshold_func: Custom threshold function for 'custom' filtration type
            h0_config: Optional H⁰ configuration for GW sheaves
            
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
        
        # Detect construction method for routing to appropriate persistence method
        construction_method = sheaf.metadata.get('construction_method', 'standard')
        
        logger.info(f"Starting persistent spectral analysis: {construction_method} construction, "
                   f"{filtration_type} filtration, {n_steps} steps")
        start_time = time.time()
        
        try:
            # ROUTING DECISION: Use H⁰ tracking for GW sheaves, standard for others
            if construction_method == 'gromov_wasserstein':
                logger.info("Routing to H⁰ Global Section persistence for GW sheaf")
                return self._analyze_with_h0_tracking(
                    sheaf, filtration_type, n_steps, param_range, 
                    custom_threshold_func, h0_config, start_time
                )
            else:
                logger.info("Routing to standard eigenvalue-based persistence")
                return self._analyze_with_standard_persistence(
                    sheaf, filtration_type, n_steps, param_range, 
                    custom_threshold_func, start_time
                )
                
        except Exception as e:
            raise ComputationError(f"Persistent spectral analysis failed: {e}",
                                 operation="analyze")
    
    def _analyze_with_standard_persistence(self,
                                         sheaf: Sheaf,
                                         filtration_type: str,
                                         n_steps: int,
                                         param_range: Optional[Tuple[float, float]],
                                         custom_threshold_func: Optional[Callable],
                                         start_time: float) -> Dict:
        """Standard eigenvalue-based persistence analysis."""
        construction_method = sheaf.metadata.get('construction_method', 'standard')
        
        # Create appropriate subspace tracker if not already provided
        if self.subspace_tracker is None:
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
        persistence_result = self.static_laplacian.compute_persistence(
            sheaf, filtration_params, edge_threshold_func, construction_method
        )
        
        # Extract persistence features
        features = self._extract_persistence_features(persistence_result)
        
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
            'semantic_validation': semantic_validation,
            'persistence_method': 'eigenvalue_based'
        }
        
        logger.info(f"Standard persistent spectral analysis completed in {analysis_time:.2f}s")
        
        return {
            'persistence_result': persistence_result,
            'features': features,
            'diagrams': diagrams,
            'filtration_params': filtration_params,
            'filtration_type': filtration_type,
            'analysis_metadata': analysis_metadata
        }
    
    def _analyze_with_h0_tracking(self,
                                sheaf: Sheaf,
                                filtration_type: str,
                                n_steps: int,
                                param_range: Optional[Tuple[float, float]],
                                custom_threshold_func: Optional[Callable],
                                h0_config: Optional[H0Config],
                                start_time: float) -> Dict:
        """H⁰ Global Section persistence analysis for GW sheaves.
        
        This method implements the complete H⁰ persistence pipeline from the production plan:
        1. Prepare filtration steps with GW metadata
        2. Set up builders for coboundary and transport extraction
        3. Run H⁰ persistence tracking with RRQR-based updates
        4. Convert results to standard analysis format
        
        Args:
            sheaf: GW sheaf to analyze
            filtration_type: Type of filtration (should be 'threshold' for GW)
            n_steps: Number of filtration steps
            param_range: Range of filtration parameters (auto-detected for GW)
            custom_threshold_func: Custom threshold function (ignored for GW)
            h0_config: H⁰ configuration parameters
            start_time: Analysis start time for timing
            
        Returns:
            Analysis results in standard format compatible with visualization
        """
        construction_method = sheaf.metadata.get('construction_method', 'gromov_wasserstein')
        h0_config = h0_config or DEFAULT_H0_CONFIG
        
        logger.info(f"Starting H⁰ Global Section analysis: {n_steps} steps, dtype={h0_config.dtype}")
        
        # Step 1: Generate GW-aware filtration parameters
        filtration_params = self._generate_filtration_params(
            sheaf, filtration_type, n_steps, param_range
        )
        
        # Step 2: Prepare filtration data for H⁰ tracking
        filtration_data = self._prepare_gw_filtration_data(sheaf, filtration_params, filtration_type)
        
        # Step 3: Set up builders for coboundary and transport extraction
        builders = self._setup_h0_builders(sheaf)
        
        # Step 4: Initialize and run H⁰ persistence tracker
        h0_tracker = H0PersistenceTracker(h0_config)
        h0_result = h0_tracker.run_h0_persistence_pipeline(
            filtration_data, builders, h0_config
        )
        
        # Step 5: Convert H⁰ results to standard analysis format
        features = self._extract_h0_features(h0_result)
        diagrams = self._convert_h0_to_diagrams(h0_result, filtration_params, construction_method)
        
        # Step 6: Create analysis metadata
        analysis_time = time.time() - start_time
        analysis_metadata = {
            'analysis_time': analysis_time,
            'computation_time': h0_result.total_time,
            'n_filtration_steps': h0_result.n_steps,
            'sheaf_nodes': len(sheaf.stalks),
            'sheaf_edges': len(sheaf.restrictions),
            'persistence_method': 'transport_informed_h0',
            'h0_config': h0_config.__dict__,
            'semantic_validation': {'is_valid': True, 'method': 'transport_informed'}
        }
        
        logger.info(f"H⁰ Global Section analysis completed in {analysis_time:.2f}s: "
                   f"{len(h0_result.intervals)} intervals")
        
        # Step 7: Compute eigenvalue and eigenvector sequences for eigenvalue tracking
        logger.info("Computing eigenvalue and eigenvector sequences for eigenvalue tracking")
        eigenvalue_sequences, eigenvector_sequences = self._compute_eigenvalues_and_eigenvectors_for_h0_steps(
            sheaf, filtration_params, filtration_type, builders
        )
        logger.info(f"Computed eigenvalue and eigenvector sequences for {len(eigenvalue_sequences)} steps")
        
        return {
            'persistence_result': {
                'h0_result': h0_result,
                'tracking_info': {'method': 'transport_informed_h0'},
                'eigenvalue_sequences': eigenvalue_sequences  # Now populated for visualization!
            },
            'features': features,
            'diagrams': diagrams,
            'filtration_params': filtration_params,
            'filtration_type': filtration_type,
            'analysis_metadata': analysis_metadata
        }
    
    def _prepare_gw_filtration_data(self,
                                  sheaf: Sheaf,
                                  filtration_params: List[float],
                                  filtration_type: str) -> List[Dict]:
        """Prepare filtration steps with GW-specific metadata.
        
        Each step contains:
        - param: filtration parameter value
        - active_edges: edges with cost <= param (GW semantics)
        - sheaf: reference to the original sheaf for metadata access
        
        Args:
            sheaf: GW sheaf with metadata
            filtration_params: List of filtration parameter values
            filtration_type: Type of filtration (typically 'threshold' for GW)
            
        Returns:
            List of filtration step dictionaries
        """
        filtration_data = []
        edge_threshold_func = self._create_edge_threshold_func(filtration_type, sheaf=sheaf)
        
        # Extract GW costs for edge filtering
        gw_costs = sheaf.metadata.get('gw_costs', {})
        
        for step_idx, param in enumerate(filtration_params):
            # Determine active edges based on GW cost threshold
            active_edges = []
            
            if gw_costs:
                # Use GW costs for filtering
                for edge, cost in gw_costs.items():
                    if edge_threshold_func(cost, param):
                        active_edges.append(edge)
            else:
                # Fallback: use restriction norms
                logger.warning("No GW costs found, using restriction norms as proxy")
                for edge, restriction in sheaf.restrictions.items():
                    cost = torch.linalg.norm(restriction, ord=2).item()
                    if edge_threshold_func(cost, param):
                        active_edges.append(edge)
            
            step_data = {
                'step': step_idx,
                'param': param,
                'active_edges': active_edges,
                'sheaf': sheaf,  # Reference for metadata access
                'n_active_edges': len(active_edges)
            }
            
            filtration_data.append(step_data)
            
            logger.debug(f"Step {step_idx}: param={param:.6f}, "
                        f"active_edges={len(active_edges)}/{len(sheaf.restrictions)}")
        
        logger.info(f"Prepared {len(filtration_data)} GW filtration steps")
        return filtration_data
    
    def _setup_h0_builders(self, sheaf: Sheaf) -> Dict:
        """Set up builders for H⁰ tracking pipeline.
        
        Creates the necessary builders for:
        - Coboundary operator construction with metrics
        - Transport map extraction from GW couplings
        
        Args:
            sheaf: GW sheaf with construction metadata
            
        Returns:
            Dictionary with builder instances
        """
        builders = {}
        
        # Check if GW Laplacian builder is available in metadata
        if hasattr(sheaf, 'metadata') and 'gw_laplacian_builder' in sheaf.metadata:
            builders['coboundary_builder'] = sheaf.metadata['gw_laplacian_builder']
            logger.info("Using GW Laplacian builder from sheaf metadata")
        else:
            # Import GW builder and create instance
            try:
                from ..sheaf.assembly.gw_laplacian import GWLaplacianBuilder
                
                # Extract normalized Laplacian flag from sheaf metadata
                gw_config = sheaf.metadata.get('gw_config', {})
                use_normalized = gw_config.get('use_normalized_laplacian', False) if isinstance(gw_config, dict) else False
                
                builders['coboundary_builder'] = GWLaplacianBuilder(
                    use_normalized_laplacian=use_normalized
                )
                logger.info(f"Created new GW Laplacian builder (normalized: {use_normalized})")
            except ImportError:
                logger.error("GW Laplacian builder not available")
                raise RuntimeError("Cannot perform H⁰ tracking without GW Laplacian builder")
        
        # Transport extractor (same as coboundary builder for GW)
        builders['transport_extractor'] = builders['coboundary_builder']
        
        # Add any other builders from metadata
        if hasattr(sheaf, 'metadata'):
            for key, value in sheaf.metadata.items():
                if key.endswith('_builder') and key not in builders:
                    builders[key] = value
        
        logger.debug(f"Set up {len(builders)} builders for H⁰ tracking")
        return builders
    
    def _extract_h0_features(self, h0_result) -> Dict:
        """Extract features from H⁰ persistence result.
        
        Converts H⁰-specific results to standard feature format
        compatible with existing visualization systems.
        
        Args:
            h0_result: PersistenceResult from H⁰ tracking
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Basic persistence statistics
        intervals = h0_result.intervals
        features['num_birth_events'] = len([i for i in intervals if i.get('birth_step') is not None])
        features['num_death_events'] = len([i for i in intervals if i.get('death_step') is not None])
        features['num_crossings'] = 0  # Not applicable for H⁰ tracking
        features['num_persistent_paths'] = len(intervals)
        
        # Betti curve evolution
        if hasattr(h0_result, 'betti_curve') and h0_result.betti_curve:
            betti_values = [step['beta0'] for step in h0_result.betti_curve]
            features['betti_evolution'] = betti_values
            features['max_betti_number'] = max(betti_values) if betti_values else 0
            features['mean_betti_number'] = np.mean(betti_values) if betti_values else 0
            features['betti_variance'] = np.var(betti_values) if len(betti_values) > 1 else 0
        else:
            features['betti_evolution'] = []
            features['max_betti_number'] = 0
            features['mean_betti_number'] = 0
            features['betti_variance'] = 0
        
        # Lifetime statistics  
        finite_lifetimes = []
        for interval in intervals:
            if interval.get('death_step') is not None and interval.get('birth_step') is not None:
                lifetime = abs(interval.get('death_param', 0) - interval.get('birth_param', 0))
                if np.isfinite(lifetime) and lifetime > 0:
                    finite_lifetimes.append(lifetime)
        
        if finite_lifetimes:
            features['mean_lifetime'] = np.mean(finite_lifetimes)
            features['max_lifetime'] = max(finite_lifetimes)
            features['min_lifetime'] = min(finite_lifetimes)
            features['lifetime_std'] = np.std(finite_lifetimes)
        else:
            features['mean_lifetime'] = 0.0
            features['max_lifetime'] = 0.0
            features['min_lifetime'] = 0.0
            features['lifetime_std'] = 0.0
        
        # Transport-specific features
        transport_informed_count = len([i for i in intervals if i.get('transport_informed', False)])
        features['transport_informed_ratio'] = (
            transport_informed_count / len(intervals) if intervals else 0.0
        )
        
        # Diagnostic features
        if hasattr(h0_result, 'diagnostics') and h0_result.diagnostics:
            diagnostics = h0_result.diagnostics
            features['mean_kernel_dimension'] = np.mean([d.kernel_dimension for d in diagnostics])
            features['mean_computational_time'] = np.mean([d.computational_time for d in diagnostics])
            
            # Certificate success rate
            certificates_passed = [d for d in diagnostics if getattr(d, 'certificate_passed', True)]
            features['certificate_success_rate'] = len(certificates_passed) / len(diagnostics)
        
        # Summary
        features['summary'] = {
            'total_filtration_steps': h0_result.n_steps,
            'total_intervals': len(intervals),
            'infinite_intervals': len([i for i in intervals if i.get('death_step') is None]),
            'finite_intervals': len(finite_lifetimes),
            'method': 'transport_informed_h0'
        }
        
        logger.debug(f"Extracted H⁰ features: {len(intervals)} intervals, "
                    f"max β₀={features['max_betti_number']}")
        
        return features
    
    def _convert_h0_to_diagrams(self,
                              h0_result,
                              filtration_params: List[float],
                              construction_method: str) -> Dict:
        """Convert H⁰ persistence result to standard diagram format.
        
        Args:
            h0_result: PersistenceResult from H⁰ tracking
            filtration_params: Filtration parameter values
            construction_method: Construction method (should be 'gromov_wasserstein')
            
        Returns:
            Dictionary with persistence diagrams in standard format
        """
        diagrams = {
            'birth_death_pairs': [],
            'infinite_bars': [],
            'continuous_paths': [],  # H⁰ tracking provides different path concept
            'path_based_computation': True,
            'h0_based_computation': True  # Flag for H⁰ method
        }
        
        # Convert H⁰ intervals to standard format
        for interval in h0_result.intervals:
            birth_param = interval.get('birth_param')
            death_param = interval.get('death_param')
            birth_step = interval.get('birth_step')
            death_step = interval.get('death_step')
            
            if birth_param is None:
                continue  # Skip invalid intervals
            
            if death_param is not None and death_step is not None:
                # Finite interval
                lifetime = abs(death_param - birth_param)
                
                # Validate interval
                if (np.isfinite(birth_param) and np.isfinite(death_param) and 
                    np.isfinite(lifetime) and lifetime >= 0 and birth_param <= death_param):
                    
                    pair = {
                        'birth': birth_param,
                        'death': death_param,
                        'lifetime': lifetime,
                        'birth_step': birth_step,
                        'death_step': death_step,
                        'transport_informed': interval.get('transport_informed', False),
                        'confirmed': interval.get('confirmed', True)
                    }
                    diagrams['birth_death_pairs'].append(pair)
            else:
                # Infinite interval  
                if np.isfinite(birth_param):
                    infinite_bar = {
                        'birth': birth_param,
                        'death': float('inf'),
                        'birth_step': birth_step,
                        'transport_informed': interval.get('transport_informed', False),
                        'confirmed': interval.get('confirmed', True)
                    }
                    diagrams['infinite_bars'].append(infinite_bar)
        
        # Sort by birth time
        diagrams['birth_death_pairs'].sort(key=lambda x: x['birth'])
        diagrams['infinite_bars'].sort(key=lambda x: x['birth'])
        
        # Compute statistics
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
        
        # Add H⁰-specific statistics
        total_intervals = len(h0_result.intervals)
        transport_informed = len([i for i in h0_result.intervals 
                                if i.get('transport_informed', False)])
        
        diagrams['h0_statistics'] = {
            'total_h0_intervals': total_intervals,
            'transport_informed_intervals': transport_informed,
            'transport_informed_ratio': transport_informed / max(total_intervals, 1),
            'computation_method': 'rrqr_based_h0_tracking'
        }
        
        logger.info(f"Converted H⁰ result to diagrams: "
                   f"{diagrams['statistics']['n_finite_pairs']} finite pairs, "
                   f"{diagrams['statistics']['n_infinite_bars']} infinite bars")
        
        return diagrams
        
        # Summary
        features['summary'] = {
            'total_filtration_steps': len(h0_result.betti_curve),
            'persistence_method': 'h0_global_section',
            'mean_betti0': features['mean_betti0'],
            'n_transport_informed_intervals': len([i for i in h0_result.intervals if i.get('transport_informed', False)])
        }
        
        return features
    
    def _generate_h0_persistence_diagrams(self, h0_result, construction_method: str) -> Dict:
        """Generate persistence diagrams from H⁰ intervals."""
        diagrams = {
            'birth_death_pairs': [],
            'infinite_bars': [],
            'continuous_paths': [],
            'path_based_computation': True,
            'h0_based': True
        }
        
        # Process finite intervals
        for interval in h0_result.intervals:
            if interval['death_param'] is not None:
                pair = {
                    'birth': interval['birth_param'],
                    'death': interval['death_param'],
                    'lifetime': interval.get('lifetime', interval['death_param'] - interval['birth_param']),
                    'birth_step': interval['birth_step'],
                    'death_step': interval['death_step'],
                    'transport_informed': interval.get('transport_informed', True),
                    'confirmed': interval.get('confirmed', True)
                }
                diagrams['birth_death_pairs'].append(pair)
            else:
                # Infinite interval
                bar = {
                    'birth': interval['birth_param'],
                    'death': float('inf'),
                    'birth_step': interval['birth_step'],
                    'transport_informed': interval.get('transport_informed', True),
                    'confirmed': interval.get('confirmed', True)
                }
                diagrams['infinite_bars'].append(bar)
        
        # Sort by birth time
        diagrams['birth_death_pairs'].sort(key=lambda x: x['birth'])
        diagrams['infinite_bars'].sort(key=lambda x: x['birth'])
        
        # Compute statistics
        if diagrams['birth_death_pairs']:
            lifetimes = [p['lifetime'] for p in diagrams['birth_death_pairs'] if p['lifetime'] is not None]
            if lifetimes:
                diagrams['statistics'] = {
                    'n_finite_pairs': len(diagrams['birth_death_pairs']),
                    'n_infinite_bars': len(diagrams['infinite_bars']),
                    'mean_lifetime': np.mean(lifetimes),
                    'max_lifetime': max(lifetimes),
                    'total_persistence': sum(lifetimes),
                    'transport_informed_pairs': len([p for p in diagrams['birth_death_pairs'] if p.get('transport_informed', False)])
                }
            else:
                diagrams['statistics'] = {
                    'n_finite_pairs': len(diagrams['birth_death_pairs']),
                    'n_infinite_bars': len(diagrams['infinite_bars']),
                    'mean_lifetime': 0.0,
                    'max_lifetime': 0.0,
                    'total_persistence': 0.0,
                    'transport_informed_pairs': 0
                }
        else:
            diagrams['statistics'] = {
                'n_finite_pairs': 0,
                'n_infinite_bars': len(diagrams['infinite_bars']),
                'mean_lifetime': 0.0,
                'max_lifetime': 0.0,
                'total_persistence': 0.0,
                'transport_informed_pairs': 0
            }
        
        logger.info(f"Generated H⁰ persistence diagrams: "
                   f"{diagrams['statistics']['n_finite_pairs']} finite pairs, "
                   f"{diagrams['statistics']['n_infinite_bars']} infinite bars, "
                   f"{diagrams['statistics']['transport_informed_pairs']} transport-informed")
        
        return diagrams
    
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
        
        🔧 STEP-EDGE COORDINATION FIX: Ensure we never exceed the number of available edges.
        This prevents the wraparound behavior that causes critical principal angles.
        
        Key principle: Every filtration step must correspond to adding exactly one edge.
        """
        # 🔧 CRITICAL FIX: Fundamental limit is number of edges, not unique costs
        # Step 0: no edges active
        # Steps 1-N: each step activates exactly one edge (ordered by cost)
        max_possible_steps = len(costs) + 1  # +1 for zero-edge initial state
        
        # Respect the fundamental edge limit
        optimal_steps = min(max_possible_steps, requested_steps)
        
        # Log coordination information
        unique_costs = len(set(costs))
        if optimal_steps < requested_steps:
            logger.info(f"🔧 COORDINATION FIX: Limiting steps from {requested_steps} to {optimal_steps} "
                       f"to match {len(costs)} available edges ({unique_costs} unique costs)")
        else:
            logger.info(f"Using {optimal_steps} steps for {len(costs)} edges "
                       f"({unique_costs} unique costs)")
        
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
    
    def analyze_alpha_flow(self, 
                          sheaf: Sheaf,
                          spec: AlphaFlowSpec,
                          config: StaticBuildConfig, 
                          use_normalized: Union[str, bool] = False) -> AlphaFlowResult:
        """Perform α-flow analysis on a sheaf.
        
        The α-flow method analyzes network structure using baseline/residual
        Laplacian decomposition: L(α) = L_base + α*L_resid. This provides
        architecture-invariant comparison by avoiding trivial uniform scaling.
        
        Process:
        1. Build grouped operators (L_base, L_resid, D) with no masking
        2. Handle degenerate splits with guard rails  
        3. Freeze D in fixed mode with proper ridge regularization
        4. For each α: compute eigenvalues, Hutchinson moments, norms
        5. Apply normalization to summaries only (not operators)
        
        Args:
            sheaf: GW sheaf containing edge costs and restrictions
            spec: α-flow analysis specification
            config: Build configuration for Laplacian construction
            
        Returns:
            AlphaFlowResult with complete analysis results and metadata
            
        Raises:
            ComputationError: If analysis fails due to numerical issues
            ValueError: If sheaf is not suitable for α-flow analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting α-flow analysis: {len(spec.alpha_grid)} α values, "
                   f"{spec.k_small} eigenvalues, {spec.probes} probes")
        
        try:
            # Import numerical utilities
            from .utils_numerical import hutchinson_trace_power, smallest_eigs_generalized
            
            # Step 1: Build grouped operators using AlphaFlowBuilder
            alpha_builder = AlphaFlowBuilder(sheaf, self._get_gw_laplacian_builder(), use_normalized_laplacian=use_normalized)
            
            # Configure random seed for reproducible results
            if config.random_state is not None:
                np.random.seed(config.random_state)
                rng = np.random.default_rng(config.random_state)
            else:
                rng = np.random.default_rng()
            
            # Build the α-flow operators - choose format based on efficiency flag
            if spec.eigen_use_csr:
                # Build CSR matrices for efficient eigenvalue computation
                alpha_build_csr = alpha_builder.get_csr_matrices(
                    grouping=spec.grouping,
                    mass_mode=config.mass_mode
                )
                # Also need LinearOperator version for Hutchinson moments
                alpha_build_op = alpha_builder.build(
                    grouping=spec.grouping,
                    mass_mode=config.mass_mode,
                    as_linear_operator=True
                )
                logger.info(f"Built α-flow operators (CSR+LinearOperator): {alpha_build_csr.meta['n_base_edges']} base edges, "
                           f"{alpha_build_csr.meta['n_resid_edges']} residual edges")
            else:
                # Traditional LinearOperator approach
                alpha_build_op = alpha_builder.build(
                    grouping=spec.grouping,
                    mass_mode=config.mass_mode,
                    as_linear_operator=True
                )
                alpha_build_csr = None
                logger.info(f"Built α-flow operators (LinearOperator): {alpha_build_op.meta['n_base_edges']} base edges, "
                           f"{alpha_build_op.meta['n_resid_edges']} residual edges")
            
            # Step 2: Analyze each α value
            points = []
            
            for alpha in spec.alpha_grid:
                logger.debug(f"Computing α = {alpha}")
                
                # Form L(α) for eigenvalue computation (CSR or LinearOperator)
                if spec.eigen_use_csr:
                    # Use CSR matrices for more efficient shift-invert
                    L_alpha_csr = alpha_builder.as_csr_combined(alpha_build_csr, alpha)
                    L_alpha_eigen = L_alpha_csr
                    D_eigen = alpha_build_csr.D
                    logger.debug(f"Using CSR matrix for eigenvalues: nnz={L_alpha_csr.nnz}")
                else:
                    # Traditional LinearOperator approach
                    L_alpha_eigen = alpha_builder.as_operator(alpha_build_op, alpha)
                    D_eigen = alpha_build_op.D
                
                # Form L(α) for Hutchinson moments (always use LinearOperator - efficient for matvec)
                L_alpha_moments = alpha_builder.as_operator(alpha_build_op, alpha)
                
                # Compute k smallest generalized eigenvalues
                try:
                    eig_result = smallest_eigs_generalized(
                        L_alpha_eigen, D_eigen, 
                        k=spec.k_small, 
                        sigma=spec.sigma,
                        random_state=rng,
                        return_vecs=False
                    )
                    
                    if not eig_result.converged or len(eig_result.eigenvalues) == 0:
                        logger.warning(f"Eigenvalue computation failed for α={alpha}")
                        eigenvalues = np.array([])
                    else:
                        # Clip negative eigenvalues as per plan
                        eigenvalues = np.maximum(eig_result.eigenvalues, -1e-12)
                        
                        # Log eigenvalue range for monitoring
                        if len(eigenvalues) > 0:
                            eig_min = eigenvalues.min()
                            eig_max = eigenvalues.max()
                            eig_mean = eigenvalues.mean()
                            eig_std = eigenvalues.std()
                            logger.info(f"α={alpha}: Eigenvalue range: [{eig_min:.6e}, {eig_max:.6e}], "
                                       f"mean={eig_mean:.6e}, std={eig_std:.6e}, count={len(eigenvalues)}")
                            
                            # Log warning if eigenvalues seem unusual
                            if eig_max > 1e6:
                                logger.warning(f"α={alpha}: Large eigenvalues detected (max={eig_max:.6e})")
                            if eig_min < -1e-10:
                                logger.warning(f"α={alpha}: Negative eigenvalues detected (min={eig_min:.6e})")
                        else:
                            logger.info(f"α={alpha}: No eigenvalues computed")
                        
                except Exception as e:
                    logger.warning(f"Eigenvalue computation failed for α={alpha}: {e}")
                    eigenvalues = np.array([])
                    logger.info(f"α={alpha}: No eigenvalues computed")
                
                # Compute Hutchinson moments Tr(L^k) using LinearOperator (efficient for matvec)
                moments = {}
                moment_stds = {}
                
                for k in spec.moments:
                    try:
                        trace_est, trace_std = hutchinson_trace_power(
                            L_alpha_moments, power=k, probes=spec.probes, rng=rng
                        )
                        moments[k] = trace_est
                        moment_stds[k] = trace_std
                    except Exception as e:
                        logger.warning(f"Moment computation failed for α={alpha}, k={k}: {e}")
                        moments[k] = np.nan
                        moment_stds[k] = np.nan
                
                # Compute trace and Frobenius norm (k=1 and k=2 moments)
                n = L_alpha_moments.shape[0]
                trace_normalized = moments.get(1, np.nan) / n if n > 0 else np.nan
                frobenius_normalized = moments.get(2, np.nan) / (n * n) if n > 0 else np.nan
                
                # Apply additional normalization if specified
                if config.normalization == 'trace' and not np.isnan(trace_normalized):
                    for k in moments:
                        if not np.isnan(moments[k]):
                            moments[k] /= trace_normalized
                
                # Compile point results
                point_meta = {
                    'eigenvalue_converged': eig_result.converged if 'eig_result' in locals() else False,
                    'eigenvalue_iterations': eig_result.num_iterations if 'eig_result' in locals() else 0,
                    'moment_computation_success': {k: not np.isnan(moments[k]) for k in moments},
                    'alpha': alpha
                }
                
                point = AlphaFlowPoint(
                    alpha=alpha,
                    eigenvalues=eigenvalues,
                    moments=moments,
                    moment_stds=moment_stds,
                    trace_normalized=trace_normalized,
                    frobenius_normalized=frobenius_normalized,
                    meta=point_meta
                )
                points.append(point)
                
                if len(eigenvalues) > 0:
                    logger.debug(f"α={alpha}: {len(eigenvalues)} eigenvalues [min={eigenvalues.min():.3e}, max={eigenvalues.max():.3e}], "
                               f"{sum(1 for v in moments.values() if not np.isnan(v))}/{len(moments)} moments")
                else:
                    logger.debug(f"α={alpha}: 0 eigenvalues, "
                               f"{sum(1 for v in moments.values() if not np.isnan(v))}/{len(moments)} moments")
            
            # Step 3: Compile final results
            analysis_time = time.time() - start_time
            
            # Use alpha_build_op for metadata (always exists)
            active_build = alpha_build_op
            
            result = AlphaFlowResult(
                points=points,
                grouping_meta=active_build.meta,
                build_meta={
                    'matrix_size': active_build.L_base.shape[0],
                    'eigen_use_csr': spec.eigen_use_csr,
                    'mass_mode': config.mass_mode,
                    'precision': config.precision,
                    'normalization': config.normalization
                },
                analysis_time=analysis_time,
                spec=spec,
                config=config
            )
            
            # Log overall eigenvalue statistics
            all_eigenvalues = []
            for point in points:
                if point.eigenvalues.size > 0:
                    all_eigenvalues.extend(point.eigenvalues.tolist())
            
            if all_eigenvalues:
                all_eigs = np.array(all_eigenvalues)
                logger.info(f"Overall eigenvalue statistics across all α values:")
                logger.info(f"  Range: [{all_eigs.min():.6e}, {all_eigs.max():.6e}]")
                logger.info(f"  Mean: {all_eigs.mean():.6e}, Std: {all_eigs.std():.6e}")
                logger.info(f"  Total eigenvalues computed: {len(all_eigenvalues)}")
            
            logger.info(f"α-flow analysis completed: {analysis_time:.3f}s, "
                       f"{len(points)} α values, matrix size {alpha_build_op.L_base.shape[0]}")
            
            return result
            
        except Exception as e:
            logger.error(f"α-flow analysis failed: {e}")
            raise ComputationError(f"α-flow analysis failed: {e}") from e
    
    def analyze_diffusion_flow(self,
                              sheaf: Sheaf,
                              spec: DiffusionFlowSpec, 
                              config: StaticBuildConfig) -> DiffusionFlowResult:
        """Perform t-flow (diffusion) analysis on a sheaf.
        
        The t-flow method uses heat kernel summaries to probe multi-scale
        structure: h(t) = Tr(exp(-t*L))/n over different time scales.
        
        Process:
        1. Build single L (all edges) and D with no masking
        2. Freeze D in fixed mode with ridge regularization  
        3. Compute k smallest eigenvalues once (cached for efficiency)
        4. Auto-generate t-grid if needed using λ_max estimation
        5. For each t: estimate heat trace using SLQ with variance tracking
        
        Args:
            sheaf: GW sheaf for analysis
            spec: Diffusion flow analysis specification
            config: Build configuration for Laplacian construction
            
        Returns:
            DiffusionFlowResult with heat trace summaries and metadata
            
        Raises:
            ComputationError: If analysis fails due to numerical issues
            ValueError: If sheaf is not suitable for t-flow analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting t-flow analysis: {spec.probes} probes, "
                   f"{spec.slq_iters} SLQ iterations")
        
        try:
            # Step 1: Initialize diffusion analyzer
            diffusion_analyzer = DiffusionFlowAnalyzer(sheaf, self._get_gw_laplacian_builder())
            
            # Step 2: Create DiffusionSpec for the analyzer
            diffusion_spec = DiffusionSpec(
                t_grid=spec.t_grid,
                k_small=spec.k_small,
                probes=spec.probes,
                slq_iters=spec.slq_iters
            )
            
            # Step 3: Perform the analysis
            summaries = diffusion_analyzer.analyze(diffusion_spec, config.mass_mode)
            
            # Step 4: Apply normalization if specified
            if config.normalization == 'trace' and len(summaries.heat_trace) > 0:
                # Normalize by the first (largest) heat trace value
                max_trace = np.nanmax(summaries.heat_trace)
                if max_trace > 0:
                    summaries.heat_trace = summaries.heat_trace / max_trace
            
            # Step 5: Compile final results
            analysis_time = time.time() - start_time
            
            result = DiffusionFlowResult(
                summaries=summaries,
                build_meta={
                    'matrix_size': summaries.meta.get('matrix_size', 0),
                    'mass_mode': config.mass_mode,
                    'precision': config.precision,
                    'normalization': config.normalization,
                    **summaries.meta
                },
                analysis_time=analysis_time,
                spec=spec,
                config=config
            )
            
            logger.info(f"t-flow analysis completed: {analysis_time:.3f}s, "
                       f"{len(summaries.heat_trace)} time points, "
                       f"monotonic: {summaries.meta.get('is_monotonic', False)}")
            
            return result
            
        except Exception as e:
            logger.error(f"t-flow analysis failed: {e}")
            raise ComputationError(f"t-flow analysis failed: {e}") from e
    
    def _get_gw_laplacian_builder(self):
        """Get or create GWLaplacianBuilder instance."""
        # Import here to avoid circular imports
        from ..sheaf.assembly.gw_laplacian import GWLaplacianBuilder
        
        if not hasattr(self, '_gw_laplacian_builder'):
            self._gw_laplacian_builder = GWLaplacianBuilder()
        
        return self._gw_laplacian_builder
    
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
                # GW: CORRECT BEHAVIOR: 0 eigenvalues → many eigenvalues
                # Early params (sparse/disconnected) → 0 or few eigenvalues 
                # Late params (dense/connected) → many eigenvalues
                # This matches GW construction: sparse graphs → dense graphs
                
                expected_progression = non_zero_counts[0] <= non_zero_counts[-1]
                expected_mean_increase = True  # Allow any mean progression since structure matters more
                
                # Allow for legitimate 0→many progression
                if non_zero_counts[0] == 0 and non_zero_counts[-1] > 0:
                    logger.info(f"✅ GW semantic validation: Correct 0→many eigenvalue progression "
                               f"(start: {non_zero_counts[0]} eigenvalues, end: {non_zero_counts[-1]} eigenvalues)")
                elif expected_progression:
                    logger.info(f"✅ GW semantic validation: Correct eigenvalue progression "
                               f"(start: {non_zero_counts[0]} eigenvalues, end: {non_zero_counts[-1]} eigenvalues)")
                else:
                    validation['warnings'].append(
                        f"GW semantics: Expected eigenvalue count increase from sparse→dense, "
                        f"but got {non_zero_counts[0]} → {non_zero_counts[-1]} eigenvalues")
                    validation['is_valid'] = False
                
                # Additional check: monotonic or at least non-decreasing trend
                decreases = 0
                for i in range(1, len(non_zero_counts)):
                    if non_zero_counts[i] < non_zero_counts[i-1]:
                        decreases += 1
                
                # Allow some minor fluctuations but not major decreases
                if decreases > len(non_zero_counts) * 0.3:  # More than 30% steps decrease
                    validation['warnings'].append(
                        f"GW semantics: Too many eigenvalue count decreases ({decreases}/{len(non_zero_counts)} steps) - "
                        f"expected mostly increasing trend for sparse→dense construction")
                    
                logger.info(f"GW semantic validation: Early step {non_zero_counts[0]} eigenvalues (mean={mean_eigenvalues[0]:.6f}), "
                           f"Late step {non_zero_counts[-1]} eigenvalues (mean={mean_eigenvalues[-1]:.6f}), "
                           f"Decreases: {decreases}/{len(non_zero_counts)}")
                
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
    
    def _compute_eigenvalues_and_eigenvectors_for_h0_steps(self,
                                        sheaf: Sheaf,
                                        filtration_params: List[float],
                                        filtration_type: str,
                                        builders: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute eigenvalue and eigenvector sequences for eigenvalue tracking.
        
        This method computes both eigenvalues and eigenvectors at each filtration step,
        enabling eigenvalue tracking alongside H⁰ Global Section persistence analysis.
        
        Args:
            sheaf: GW sheaf to analyze
            filtration_params: Filtration parameter values from H⁰ analysis
            filtration_type: Type of filtration (typically 'threshold' for GW)
            builders: Dictionary with coboundary builders (GWLaplacianBuilder)
            
        Returns:
            Tuple of (eigenvalue_sequences, eigenvector_sequences)
            - eigenvalue_sequences: List of eigenvalue tensors, one per filtration step
            - eigenvector_sequences: List of eigenvector tensors, one per filtration step
        """
        logger.info(f"Computing eigenvalue sequences for {len(filtration_params)} H⁰ filtration steps")
        
        if 'coboundary_builder' not in builders or builders['coboundary_builder'] is None:
            logger.warning("No coboundary builder available for eigenvalue computation")
            # Return empty eigenvalues and eigenvectors for each step
            empty_eigenvals = [torch.tensor([]) for _ in filtration_params]
            empty_eigenvecs = [torch.tensor([]) for _ in filtration_params]
            return empty_eigenvals, empty_eigenvecs
        
        builder = builders['coboundary_builder']
        eigenvalue_sequences = []
        eigenvector_sequences = []
        
        # Use unified Laplacian construction (same as standard persistence)
        from .static_laplacian_unified import UnifiedStaticLaplacian
        
        logger.info("Using unified Laplacian construction for consistent eigenvalue computation")
        
        # Determine if we need GW support based on sheaf type
        is_gw_sheaf = sheaf.is_gw_sheaf()
        use_normalized = False
        if is_gw_sheaf:
            gw_config = sheaf.metadata.get('gw_config', {})
            if isinstance(gw_config, dict):
                use_normalized = gw_config.get('use_normalized_laplacian', False)
        
        # Create unified Laplacian computer with appropriate configuration
        unified_computer = UnifiedStaticLaplacian(
            eigenvalue_method='auto',
            max_eigenvalues=10000,
            enable_gpu=False,
            enable_caching=True,
            use_generalized_normalization=use_normalized  # Enable GW support when needed
        )
        
        # Build base Laplacian once (same as standard persistence)
        static_laplacian, laplacian_metadata = unified_computer._get_or_build_laplacian(sheaf)
        
        # Extract edge information (same as standard persistence)
        edge_info = unified_computer._get_or_extract_edge_info(
            sheaf, static_laplacian, laplacian_metadata
        )
        
        # Create edge threshold function (same as standard persistence)
        edge_threshold_func = unified_computer._create_edge_threshold_func(
            sheaf.metadata.get('construction_method', 'gromov_wasserstein')
        )
        
        try:
            for i, param in enumerate(filtration_params):
                try:
                    # Create edge mask (same as standard persistence)
                    edge_mask = unified_computer._create_edge_mask(edge_info, param, edge_threshold_func)
                    
                    # Apply correct masking with block reconstruction (same as standard persistence)
                    masked_laplacian = unified_computer._apply_correct_masking(
                        static_laplacian, edge_mask, edge_info, laplacian_metadata
                    )
                    
                    # Set context for generalized eigenvalue computation if needed
                    if unified_computer.use_generalized_normalization and unified_computer.gw_builder is not None:
                        # Extract active edges from mask
                        active_edges = [edge for edge, keep in edge_mask.items() if keep]
                        unified_computer.sheaf = sheaf
                        unified_computer.active_edges = active_edges
                    
                    try:
                        # Compute eigenvalues AND eigenvectors for PES tracking
                        eigenvals, eigenvecs = unified_computer._compute_eigenvalues(masked_laplacian)
                    finally:
                        # Clear context after computation
                        if unified_computer.use_generalized_normalization and unified_computer.gw_builder is not None:
                            if hasattr(unified_computer, 'sheaf'):
                                delattr(unified_computer, 'sheaf')
                            if hasattr(unified_computer, 'active_edges'):
                                delattr(unified_computer, 'active_edges')
                    
                    # Store both eigenvalues and eigenvectors for PES tracking
                    # ✅ DEFENSIVE: Validate that we have proper tensors, not tuples
                    if isinstance(eigenvals, tuple):
                        logger.error(f"Step {i}: eigenvals is tuple instead of tensor: {type(eigenvals)}")
                        # Handle tuple case - extract first element if it's a tensor
                        if len(eigenvals) > 0 and torch.is_tensor(eigenvals[0]):
                            eigenvals = eigenvals[0]
                            logger.warning(f"Step {i}: extracted eigenvals tensor from tuple")
                        else:
                            eigenvals = torch.tensor([])
                            logger.warning(f"Step {i}: fallback to empty eigenvals tensor")
                    
                    if isinstance(eigenvecs, tuple):
                        logger.error(f"Step {i}: eigenvecs is tuple instead of tensor: {type(eigenvecs)}")
                        # Handle tuple case - extract second element if it's a tensor  
                        if len(eigenvecs) > 1 and torch.is_tensor(eigenvecs[1]):
                            eigenvecs = eigenvecs[1]
                            logger.warning(f"Step {i}: extracted eigenvecs tensor from tuple")
                        else:
                            eigenvecs = torch.tensor([])
                            logger.warning(f"Step {i}: fallback to empty eigenvecs tensor")
                    
                    # Ensure we have proper tensors before appending
                    if not torch.is_tensor(eigenvals):
                        logger.error(f"Step {i}: eigenvals is not a tensor: {type(eigenvals)}")
                        eigenvals = torch.tensor([])
                    if not torch.is_tensor(eigenvecs):
                        logger.error(f"Step {i}: eigenvecs is not a tensor: {type(eigenvecs)}")
                        eigenvecs = torch.tensor([])
                    
                    eigenvalue_sequences.append(eigenvals)
                    eigenvector_sequences.append(eigenvecs)
                    
                    # Debug: Log filtration dynamics for validation
                    if (i + 1) % 5 == 0 or len(eigenvals) > 0:
                        n_active = torch.sum(edge_mask).item() if torch.is_tensor(edge_mask) else sum(edge_mask.values())
                        n_total = len(edge_info)
                        eigenval_range = f"[{eigenvals.min().item():.2e}, {eigenvals.max().item():.2e}]" if len(eigenvals) > 0 else "[]"
                        logger.debug(f"Step {i+1}/{len(filtration_params)}: param={param:.6f}, "
                                   f"active_edges={n_active}/{n_total}, eigenvals={eigenval_range}")
                        
                        # Detect eigenvalue changes between consecutive steps
                        if i > 0 and len(eigenvalue_sequences) >= 2:
                            prev_eigenvals = eigenvalue_sequences[-2]
                            if len(prev_eigenvals) > 0 and len(eigenvals) > 0:
                                min_len = min(len(prev_eigenvals), len(eigenvals))
                                if min_len > 0:
                                    eigenval_change = torch.norm(eigenvals[:min_len] - prev_eigenvals[:min_len]).item()
                                    logger.debug(f"  Eigenvalue change from step {i}: {eigenval_change:.2e}")
                                    
                                    if eigenval_change < 1e-10:
                                        logger.warning(f"  ⚠️  Very small eigenvalue change detected - check edge activation")
                            elif len(prev_eigenvals) != len(eigenvals):
                                logger.debug(f"  Eigenvalue count changed: {len(prev_eigenvals)} → {len(eigenvals)}")
                                
                except Exception as step_error:
                    logger.warning(f"Failed to compute eigenvalues for step {i} (param={param}): {step_error}")
                    # Add empty tensors for failed step
                    eigenvalue_sequences.append(torch.tensor([]))
                    eigenvector_sequences.append(torch.tensor([]))
            
            logger.info(f"Successfully computed eigenvalue sequences for {len(eigenvalue_sequences)} H⁰ steps")
            
            # Validation summary: Check for proper eigenvalue evolution
            non_empty_sequences = [seq for seq in eigenvalue_sequences if len(seq) > 0]
            if len(non_empty_sequences) >= 2:
                # Check if eigenvalues are actually changing between steps
                total_changes = 0
                significant_changes = 0
                for i in range(1, len(non_empty_sequences)):
                    prev_seq = non_empty_sequences[i-1]
                    curr_seq = non_empty_sequences[i]
                    min_len = min(len(prev_seq), len(curr_seq))
                    if min_len > 0:
                        change = torch.norm(curr_seq[:min_len] - prev_seq[:min_len]).item()
                        total_changes += change
                        if change > 1e-6:  # Significant change threshold
                            significant_changes += 1
                
                avg_change = total_changes / max(len(non_empty_sequences) - 1, 1)
                change_ratio = significant_changes / max(len(non_empty_sequences) - 1, 1)
                
                logger.info(f"Eigenvalue evolution validation:")
                logger.info(f"  Average eigenvalue change: {avg_change:.2e}")
                logger.info(f"  Significant changes: {significant_changes}/{len(non_empty_sequences)-1} steps ({change_ratio:.1%})")
                
                if avg_change < 1e-8:
                    logger.warning("⚠️  Eigenvalues appear too stable - check GW cost extraction and filtration")
                elif change_ratio > 0.7:
                    logger.info("✅ Good eigenvalue dynamics detected")
                else:
                    logger.info("⚠️  Mixed eigenvalue dynamics - some steps may have redundant edges")
            else:
                logger.warning("⚠️  Insufficient eigenvalue sequences for validation")
            
            # Add standard tracking for eigenvalue evolution alongside H⁰ global section tracking
            # H⁰ provides birth-death pairs for persistence diagrams, standard tracking provides eigenvalue evolution
            logger.info("Adding standard eigenvalue tracking after H⁰ Global Section analysis")
            # Note: H⁰ analysis uses generator birth-death tracking for persistence diagrams,
            # Standard tracking provides eigenvalue evolution analysis
            if True:  # Enable standard tracking for eigenvalue evolution
                logger.info("Applying standard subspace tracker for GW eigenvalue persistence")
                try:
                    from .tracker_factory import SubspaceTrackerFactory
                    
                    # Use standard tracker for eigenvalue evolution (disable PES)
                    tracker = SubspaceTrackerFactory.create_tracker('standard')
                    
                    # Apply eigenvalue tracking using computed eigenvectors
                    tracking_info = tracker.track_eigenspaces(
                        eigenvalue_sequences,
                        eigenvector_sequences, 
                        filtration_params,
                        construction_method='gromov_wasserstein',
                        sheaf_metadata=sheaf.metadata
                    )
                    
                    logger.info(f"Standard tracking completed: {len(tracking_info.get('tracked_eigenvalues', []))} tracked eigenvalues")
                    
                except Exception as tracking_error:
                    logger.warning(f"Standard tracking failed, using raw eigenvalues: {tracking_error}")
            else:
                logger.info("Skipping PES tracking - no valid eigenvalue sequences")
            
            return eigenvalue_sequences, eigenvector_sequences
            
        except Exception as e:
            logger.error(f"Eigenvalue computation for H⁰ steps failed: {e}")
            # Return empty eigenvalues and eigenvectors for all steps as fallback
            empty_eigenvals = [torch.tensor([]) for _ in filtration_params]
            empty_eigenvecs = [torch.tensor([]) for _ in filtration_params]
            return empty_eigenvals, empty_eigenvecs
    
    def _compute_laplacian_eigenvalues(self, 
                                     laplacian: Union[torch.Tensor, 'csr_matrix'],
                                     max_eigenvalues: int = 100) -> torch.Tensor:
        """Compute eigenvalues of a Laplacian matrix.
        
        Args:
            laplacian: Laplacian matrix (sparse or dense)
            max_eigenvalues: Maximum number of eigenvalues to compute
            
        Returns:
            Tensor of eigenvalues (smallest first)
        """
        try:
            # Convert to appropriate format for eigenvalue computation
            if hasattr(laplacian, 'todense'):
                # Sparse matrix - convert to dense for eigenvalue computation
                laplacian_dense = torch.from_numpy(laplacian.todense()).float()
            elif isinstance(laplacian, torch.Tensor):
                laplacian_dense = laplacian.float()
            else:
                # Try to convert whatever format we have
                laplacian_dense = torch.tensor(laplacian).float()
            
            # Ensure matrix is symmetric (should be by construction)
            if laplacian_dense.shape[0] != laplacian_dense.shape[1]:
                logger.warning(f"Non-square Laplacian: {laplacian_dense.shape}")
                return torch.tensor([])
            
            # Limit size for performance
            n = laplacian_dense.shape[0]
            if n > 1000:
                logger.debug(f"Large Laplacian ({n}x{n}), computing subset of eigenvalues")
                # For large matrices, compute only the smallest eigenvalues
                max_eigenvalues = min(max_eigenvalues, n // 10)
            
            # Compute eigenvalues
            if n <= 100:
                # Small matrix - compute all eigenvalues
                eigenvals = torch.linalg.eigvals(laplacian_dense).real
            else:
                # Large matrix - compute subset using scipy for efficiency
                try:
                    from scipy.sparse.linalg import eigsh
                    import numpy as np
                    
                    # Convert back to sparse for scipy
                    if hasattr(laplacian, 'todense'):
                        sparse_laplacian = laplacian
                    else:
                        # Convert dense to sparse
                        sparse_laplacian = csr_matrix(laplacian_dense.numpy())
                    
                    # Compute smallest eigenvalues
                    k = min(max_eigenvalues, n - 2)  # scipy requirement: k < n
                    eigenvals_np, _ = eigsh(sparse_laplacian, k=k, which='SM', return_eigenvectors=False)
                    eigenvals = torch.from_numpy(eigenvals_np).float()
                    
                except Exception as scipy_error:
                    logger.debug(f"scipy eigsh failed: {scipy_error}, using dense computation")
                    # Fallback to dense computation with truncation
                    eigenvals_all = torch.linalg.eigvals(laplacian_dense).real
                    eigenvals, _ = torch.sort(eigenvals_all)
                    eigenvals = eigenvals[:max_eigenvalues]
            
            # Sort eigenvalues (smallest first) and ensure non-negative
            eigenvals, _ = torch.sort(torch.clamp(eigenvals, min=0.0))
            
            return eigenvals
            
        except Exception as e:
            logger.warning(f"Eigenvalue computation failed: {e}")
            return torch.tensor([])
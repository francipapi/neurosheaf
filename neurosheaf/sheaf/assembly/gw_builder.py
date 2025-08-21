"""GW-based sheaf assembly and restriction map computation.

This module provides the GWRestrictionManager class for computing restriction maps
using Gromov-Wasserstein optimal transport. It extends the existing sheaf assembly
infrastructure with GW-specific functionality while maintaining backward compatibility.
"""

import torch
import networkx as nx
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List, Tuple, Optional, Any, Union

from ..core import GWConfig, GromovWassersteinComputer, GWResult
from ..core.validation import validate_restriction_maps_gw
from ..data_structures import Sheaf, GWCouplingInfo
from ..extraction import extract_activations_fx

logger = logging.getLogger(__name__)


class GWRestrictionError(Exception):
    """Exception raised during GW restriction map computation."""
    pass


class GWRestrictionManager:
    """Manages GW-based restriction map computation with validation.
    
    This class orchestrates the complete GW-based sheaf construction process:
    1. Computes cosine cost matrices for each layer
    2. Solves entropic GW problems for each edge
    3. Extracts restriction maps from transport couplings
    4. Validates quasi-sheaf properties
    
    The manager handles parallel edge processing, cost matrix caching,
    and comprehensive error recovery to ensure robust operation.
    
    Mathematical Foundation:
    For edge e = (i → j), computes entropic GW coupling:
    π_{j→i} = argmin_{π∈Π(p_j,p_i)} ∑_{k,ℓ,k',ℓ'} |C_j[k,k'] - C_i[ℓ,ℓ']|² π[k,ℓ]π[k',ℓ'] - ε H(π)
    
    Then extracts backward restriction map: ρ_{j→i} = π_{j→i}^T
    """
    
    def __init__(self, 
                 gw_computer: Optional[GromovWassersteinComputer] = None,
                 config: Optional[GWConfig] = None):
        """Initialize GW restriction manager.
        
        Args:
            gw_computer: GW computation engine (creates default if None)
            config: GW configuration (uses defaults if None)
        """
        self.config = config or GWConfig()
        self.gw_computer = gw_computer or GromovWassersteinComputer(self.config)
        
        # Cache for expensive cost matrices
        self._cost_cache = {}
        # Store both couplings and costs for each edge with enhanced metadata
        self._gw_results = {}
        # Temporal tracking for transport metadata availability
        self._transport_metadata = {}
        self._edge_creation_timestamps = {}
        
        logger.info(f"GWRestrictionManager initialized: epsilon={self.config.epsilon}, "
                   f"max_iter={self.config.max_iter}, gpu={self.config.use_gpu}")
    
    def compute_all_restrictions(self, 
                               activations: Dict[str, torch.Tensor],
                               poset: nx.DiGraph,
                               parallel: bool = True,
                               max_workers: Optional[int] = None) -> Tuple[Dict[Tuple[str, str], torch.Tensor], Dict[Tuple[str, str], float], Dict[str, Any]]:
        """Compute all restriction maps with parallel edge processing.
        
        This is the main entry point for GW-based restriction computation.
        It computes cosine cost matrices, solves GW problems for all edges,
        and returns restriction maps along with comprehensive metadata.
        
        Args:
            activations: Dictionary mapping node names to activation tensors
            poset: Network structure as directed graph
            parallel: Whether to use parallel processing for edges
            max_workers: Maximum number of worker threads (None = auto)
            
        Returns:
            Tuple of (restrictions, gw_costs, metadata):
            - restrictions: Dict mapping edges to restriction map tensors ρ_{j→i} = π_{j→i}^T
            - gw_costs: Dict mapping edges to GW distortion costs
            - metadata: Comprehensive information about the computation process
            
        Raises:
            GWRestrictionError: If computation fails critically
        """
        logger.info(f"Computing GW restrictions for {len(poset.edges())} edges, "
                   f"parallel={parallel}")
        
        start_time = time.time()
        
        try:
            # 1. Compute cost matrices for all nodes
            logger.info("Computing cosine cost matrices for all nodes")
            cost_matrices = self._compute_all_cost_matrices(activations)
            
            # 2. Process all edges to compute GW couplings and restrictions
            from ...utils.indexing import canonical_edge_order
            edges = canonical_edge_order(poset.edges())
            restrictions = {}
            gw_costs = {}
            gw_couplings = {}
            failed_edges = []
            
            if parallel and len(edges) > 1:
                # Parallel processing for multiple edges
                logger.info(f"Processing {len(edges)} edges in parallel")
                restrictions, gw_costs, gw_couplings, gw_quality_scores, failed_edges = self._compute_restrictions_parallel(
                    edges, cost_matrices, activations, max_workers
                )
            else:
                # Sequential processing
                logger.info(f"Processing {len(edges)} edges sequentially")
                restrictions, gw_costs, gw_couplings, gw_quality_scores, failed_edges = self._compute_restrictions_sequential(
                    edges, cost_matrices, activations
                )
            
            # 3. Validate GW restriction maps if requested
            restriction_validation_metadata = None
            if self.config.validate_restrictions and restrictions:
                logger.info("Validating GW restriction maps (finiteness and stochasticity)")
                
                validated_restrictions, restriction_validation_metadata = validate_restriction_maps_gw(
                    restrictions=restrictions,
                    stochastic_tolerance=self.config.stochastic_tolerance,
                    correction_threshold=self.config.correction_threshold,
                    strict_threshold=self.config.strict_validation_threshold,
                    strict_mode=self.config.strict_validation_mode,
                    auto_correct=self.config.auto_correct_restrictions,
                    check_stochasticity=True  # Always check for GW restrictions
                )
                
                # Replace original restrictions with validated ones
                restrictions = validated_restrictions
                
                # Log validation results
                if restriction_validation_metadata['validation_passed']:
                    logger.info(f"✓ GW restriction validation passed: "
                              f"{len(validated_restrictions)}/{restriction_validation_metadata['total_restrictions']} maps valid")
                else:
                    logger.warning(f"⚠ GW restriction validation issues: "
                                 f"{len(restriction_validation_metadata['dropped_edges'])} edges dropped, "
                                 f"{len(restriction_validation_metadata['corrections_applied'])} corrections applied")
                    
                if restriction_validation_metadata['max_stochasticity_error'] > self.config.stochastic_tolerance:
                    logger.warning(f"Maximum stochasticity error: "
                                 f"{restriction_validation_metadata['max_stochasticity_error']:.6f}")
            
            # 4. Validate quasi-sheaf property if requested
            validation_report = None
            if self.config.quasi_sheaf_tolerance > 0:
                logger.info("Validating quasi-sheaf property")
                validation_report = self.validate_quasi_sheaf_property(
                    restrictions, poset, self.config.quasi_sheaf_tolerance
                )
                logger.info(f"Quasi-sheaf validation: max_violation={validation_report['max_violation']:.2e}, "
                           f"satisfies_quasi_sheaf={validation_report['satisfies_quasi_sheaf']}")
            
            # 5. Generate standardized transport metadata
            transport_metadata = self._generate_transport_metadata(
                gw_couplings, gw_costs, gw_quality_scores, failed_edges
            )
            
            # 6. Prepare comprehensive metadata
            total_time = time.time() - start_time
            metadata = {
                'computation_time': total_time,
                'num_edges_processed': len(edges),
                'num_edges_succeeded': len(restrictions),
                'num_edges_failed': len(failed_edges),
                'failed_edges': failed_edges,
                'gw_config': self.config.to_dict(),
                'gw_couplings': gw_couplings,
                'gw_quality_scores': gw_quality_scores,  # Quality scores for Laplacian filtering
                'restriction_validation': restriction_validation_metadata,  # GW restriction validation results
                'validation_report': validation_report,
                'cost_matrix_cache_hits': getattr(self.gw_computer.cost_cache, 'hits', 0) if self.gw_computer.cost_cache else 0,
                'parallel_processing': parallel,
                'construction_method': 'gromov_wasserstein',
                'transport_metadata': transport_metadata  # Enhanced transport availability info
            }
            
            logger.info(f"GW restriction computation complete: {len(restrictions)}/{len(edges)} edges "
                       f"succeeded in {total_time:.2f}s")
            
            return restrictions, gw_costs, metadata
            
        except Exception as e:
            logger.error(f"GW restriction computation failed: {e}", exc_info=True)
            raise GWRestrictionError(f"Failed to compute GW restrictions: {e}")
    
    def _compute_variance_based_measures(self, activation_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute per-unit importance based on activation variance across batch.
        
        This method computes variance-based probability measures for units, giving higher
        weight to units with more variable activations across the batch. Dead or noisy
        units (low variance) receive lower weights, while informative units (high variance)
        receive higher weights.
        
        Args:
            activation_tensor: Activation tensor with shape (batch_size, n_features)
            eps: Floor value to prevent zero weights (default: 1e-6)
            
        Returns:
            Probability distribution over units with shape (n_features,)
            where p[i] = (var(unit_i) + eps) / sum(var(unit_j) + eps for all j)
            
        Raises:
            ValueError: If eps <= 0 or activation tensor is invalid
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
            
        # Ensure consistent dtype
        target_dtype = self.config.get_torch_dtype()
        activation_tensor = activation_tensor.to(dtype=target_dtype)
        
        # Reshape activation tensor to (batch_size, n_features) if needed
        if activation_tensor.dim() > 2:
            # Flatten all dimensions except first (batch dimension)
            activation_flat = activation_tensor.view(activation_tensor.shape[0], -1)
            logger.debug(f"Reshaped activation tensor from {activation_tensor.shape} to {activation_flat.shape}")
        elif activation_tensor.dim() == 2:
            activation_flat = activation_tensor
        else:
            raise ValueError(f"Expected activation tensor with at least 2 dimensions, got shape {activation_tensor.shape}")
        
        batch_size, n_features = activation_flat.shape
        if batch_size < 2:
            logger.warning(f"Batch size {batch_size} < 2, variance computation may be unreliable")
        
        # Transpose to (n_features, batch_size) for unit-wise variance computation
        # Each row represents a unit's activations across the batch
        units_activations = activation_flat.T  # Shape: (n_features, batch_size)
        
        # Check for zero vectors (units with zero norm across all samples)
        unit_norms = torch.norm(units_activations, dim=1)  # Shape: (n_features,)
        zero_units = (unit_norms < 1e-12)
        n_zero_units = zero_units.sum().item()
        
        if n_zero_units > 0:
            logger.debug(f"Found {n_zero_units} zero/near-zero units out of {n_features} total units")
        
        # Compute variance for each unit across the batch dimension
        unit_variances = torch.var(units_activations, dim=1, unbiased=False)  # Shape: (n_features,)
        
        # Handle zero variance units (constant units) - these should get low but non-zero weight
        zero_var_units = (unit_variances < 1e-12)
        n_zero_var_units = zero_var_units.sum().item()
        
        if n_zero_var_units > 0:
            logger.debug(f"Found {n_zero_var_units} constant units (zero variance) out of {n_features} total units")
        
        # Apply floor to prevent zero weights and ensure numerical stability
        # Use a larger floor value for zero/constant units to give them minimal but non-zero weight
        floored_variances = unit_variances + eps
        
        # Give constant/zero units slightly higher weight than pure eps to maintain some signal
        # but much lower than variable units
        constant_unit_weight = eps * 2.0
        floored_variances = torch.where(zero_var_units | zero_units, constant_unit_weight, floored_variances)
        
        # Normalize to probability distribution
        measures = floored_variances / floored_variances.sum()
        
        # Validate result
        assert torch.allclose(measures.sum(), torch.tensor(1.0, dtype=target_dtype), atol=1e-6), \
            f"Measures should sum to 1.0, got {measures.sum():.6f}"
        assert torch.all(measures > 0), "All measures should be positive"
        
        logger.debug(f"Computed variance-based measures: "
                    f"min={measures.min().item():.6f}, "
                    f"max={measures.max().item():.6f}, "
                    f"entropy={-(measures * torch.log(measures + 1e-12)).sum().item():.3f}")
        
        return measures
    
    def _compute_all_cost_matrices(self, activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute cosine cost matrices for all activation layers.
        
        For unit-based alignment (align_units=True), computes costs between units/neurons.
        For sample-based alignment (align_units=False, deprecated), computes costs between samples.
        
        Args:
            activations: Dictionary mapping node names to activation tensors
            
        Returns:
            Dictionary mapping node names to cost matrices
        """
        cost_matrices = {}
        target_dtype = self.config.get_torch_dtype()
        
        for node_name, activation_tensor in activations.items():
            try:
                # Convert to target dtype early for consistency
                activation_tensor = activation_tensor.to(dtype=target_dtype)
                
                # Reshape activation tensor to (n_samples, n_features) if needed
                if activation_tensor.dim() > 2:
                    # Flatten all dimensions except first (batch dimension)
                    activation_flat = activation_tensor.view(activation_tensor.shape[0], -1)
                else:
                    activation_flat = activation_tensor
                
                # Choose alignment mode based on configuration
                if self.config.align_units:
                    # Unit-based alignment: transpose to (n_units, n_samples)
                    # Each row is a unit's activation vector across the batch
                    X_units = activation_flat.T  # (n_features, n_samples)
                    cost_matrix = self.gw_computer.compute_cosine_cost_matrix(X_units)
                    logger.debug(f"Computed unit cost matrix for {node_name}: {cost_matrix.shape} "
                               f"(n_units={X_units.shape[0]})")
                else:
                    # Sample-based alignment (deprecated): keep as (n_samples, n_features)
                    cost_matrix = self.gw_computer.compute_cosine_cost_matrix(activation_flat)
                    logger.debug(f"Computed sample cost matrix for {node_name}: {cost_matrix.shape} "
                               f"(n_samples={activation_flat.shape[0]})")
                
                cost_matrices[node_name] = cost_matrix
                
            except Exception as e:
                logger.error(f"Failed to compute cost matrix for {node_name}: {e}")
                raise GWRestrictionError(f"Cost matrix computation failed for {node_name}: {e}")
        
        logger.info(f"Computed cost matrices for {len(cost_matrices)} nodes "
                   f"(align_units={self.config.align_units})")
        return cost_matrices
    
    def _compute_restrictions_sequential(self, 
                                       edges: List[Tuple[str, str]],
                                       cost_matrices: Dict[str, torch.Tensor],
                                       activations: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict, Dict, Dict, List]:
        """Compute restrictions sequentially for all edges.
        
        Args:
            edges: List of edges to process
            cost_matrices: Pre-computed cost matrices
            activations: Original activation tensors
            
        Returns:
            Tuple of (restrictions, gw_costs, gw_couplings, gw_quality_scores, failed_edges)
        """
        restrictions = {}
        gw_costs = {}
        gw_couplings = {}
        gw_quality_scores = {}
        failed_edges = []
        
        for i, (source, target) in enumerate(edges):
            logger.debug(f"Processing edge {i+1}/{len(edges)}: {source} → {target}")
            
            try:
                result = self._compute_single_restriction(
                    source, target, cost_matrices, activations
                )
                
                if result is not None:
                    if len(result) == 4:
                        restriction, cost, coupling, quality = result
                    else:
                        # Backward compatibility
                        restriction, cost, coupling = result
                        quality = 1.0
                    restrictions[(source, target)] = restriction
                    gw_costs[(source, target)] = cost
                    gw_couplings[(source, target)] = coupling
                    gw_quality_scores[(source, target)] = quality
                    logger.debug(f"✓ Computed restriction {source} → {target}: {restriction.shape}, quality={quality:.4f}")
                else:
                    failed_edges.append((source, target))
                    
            except Exception as e:
                logger.warning(f"Failed to compute restriction {source} → {target}: {e}")
                failed_edges.append((source, target))
        
        # Check if we should fail fast in strict mode
        if self.config.strict_quality_mode and failed_edges:
            raise RuntimeError(f"Strict quality mode: {len(failed_edges)} edges failed quality checks")
        
        return restrictions, gw_costs, gw_couplings, gw_quality_scores, failed_edges
    
    def _compute_restrictions_parallel(self, 
                                     edges: List[Tuple[str, str]],
                                     cost_matrices: Dict[str, torch.Tensor],
                                     activations: Dict[str, torch.Tensor],
                                     max_workers: Optional[int] = None) -> Tuple[Dict, Dict, Dict, Dict, List]:
        """Compute restrictions in parallel for all edges.
        
        Args:
            edges: List of edges to process
            cost_matrices: Pre-computed cost matrices
            activations: Original activation tensors
            max_workers: Maximum number of worker threads
            
        Returns:
            Tuple of (restrictions, gw_costs, gw_couplings, gw_quality_scores, failed_edges)
        """
        restrictions = {}
        gw_costs = {}
        gw_couplings = {}
        gw_quality_scores = {}
        failed_edges = []
        quality_issues = []  # Track quality problems
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(len(edges), 4)  # Cap at 4 to avoid overwhelming the system
        
        # Submit all edge computations to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_edge = {
                executor.submit(self._compute_single_restriction, source, target, cost_matrices, activations): (source, target)
                for source, target in edges
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_edge)):
                edge = future_to_edge[future]
                source, target = edge
                
                logger.debug(f"Completed edge {i+1}/{len(edges)}: {source} → {target}")
                
                try:
                    result = future.result()
                    
                    if result is not None:
                        if len(result) == 4:
                            restriction, cost, coupling, quality = result
                        else:
                            # Backward compatibility
                            restriction, cost, coupling = result
                            quality = 1.0
                        restrictions[(source, target)] = restriction
                        gw_costs[(source, target)] = cost
                        gw_couplings[(source, target)] = coupling
                        gw_quality_scores[(source, target)] = quality
                        logger.debug(f"✓ Computed restriction {source} → {target}: {restriction.shape}, quality={quality:.4f}")
                    else:
                        failed_edges.append((source, target))
                        quality_issues.append((source, target, "Quality threshold not met or strict mode failure"))
                        
                except Exception as e:
                    logger.warning(f"Parallel computation failed for {source} → {target}: {e}")
                    failed_edges.append((source, target))
                    quality_issues.append((source, target, str(e)))
        
        # Report quality summary if there were issues
        if quality_issues:
            logger.warning(f"\nQuality issues detected in {len(quality_issues)}/{len(edges)} edges:")
            for src, tgt, reason in quality_issues[:5]:  # Show first 5
                logger.warning(f"  {src} → {tgt}: {reason}")
            if len(quality_issues) > 5:
                logger.warning(f"  ... and {len(quality_issues) - 5} more")
            
            if self.config.strict_quality_mode and quality_issues:
                raise RuntimeError(f"Strict quality mode: {len(quality_issues)} edges failed quality checks")
        
        return restrictions, gw_costs, gw_couplings, gw_quality_scores, failed_edges
    
    def _compute_single_restriction(self, 
                                  source: str, 
                                  target: str,
                                  cost_matrices: Dict[str, torch.Tensor],
                                  activations: Dict[str, torch.Tensor]) -> Optional[Tuple[torch.Tensor, float, torch.Tensor]]:
        """Compute restriction map for a single edge.
        
        Args:
            source: Source node name
            target: Target node name
            cost_matrices: Pre-computed cost matrices
            activations: Original activation tensors
            
        Returns:
            Tuple of (restriction_map, gw_cost, gw_coupling) or None if failed
        """
        try:
            # Check that we have the required data
            if source not in cost_matrices or target not in cost_matrices:
                logger.warning(f"Missing cost matrices for edge {source} → {target}")
                return None
            
            if source not in activations or target not in activations:
                logger.warning(f"Missing activations for edge {source} → {target}")
                return None
            
            # Get cost matrices
            C_source = cost_matrices[source]
            C_target = cost_matrices[target]
            
            # Set up measures (uniform by default)
            p_source = None
            p_target = None
            if not self.config.uniform_measures:
                # Implement variance-based importance sampling
                logger.debug(f"Computing non-uniform measures for edge {source} → {target}")
                try:
                    # Compute variance-based measures for source and target
                    # Use eps from config
                    eps = self.config.measure_eps
                    
                    p_source = self._compute_variance_based_measures(activations[source], eps)
                    p_target = self._compute_variance_based_measures(activations[target], eps)
                    
                    logger.debug(f"Source measures: min={p_source.min():.6f}, max={p_source.max():.6f}")
                    logger.debug(f"Target measures: min={p_target.min():.6f}, max={p_target.max():.6f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to compute non-uniform measures for {source} → {target}: {e}. "
                                 f"Falling back to uniform measures.")
                    p_source = None
                    p_target = None
            
            # Compute GW coupling
            gw_result = self.gw_computer.compute_gw_coupling(
                C_source, C_target, p_source, p_target
            )
            
            # Store quality score for this edge (will be added to sheaf metadata)
            quality_score = gw_result.quality_score
            
            # Check quality if strict mode is enabled
            if self.config.strict_quality_mode:
                if gw_result.coupling_quality != 'optimal':
                    logger.error(f"Edge {source} → {target} has suboptimal quality: {gw_result.coupling_quality}")
                    logger.error(f"Quality score: {quality_score:.4f}, solver: {gw_result.solver_type}")
                    return None  # Fail fast in strict mode
                    
            # Check minimum quality threshold
            if quality_score < self.config.min_coupling_quality:
                logger.warning(f"Edge {source} → {target} below quality threshold: "
                             f"{quality_score:.4f} < {self.config.min_coupling_quality}")
                if self.config.exclude_fallback_edges:
                    return None  # Exclude low-quality edges
            
            # Extract restriction map R_{source→target} with barycentric normalization
            # 
            # MATHEMATICAL FOUNDATION:
            # For edge (source → target), we extract a restriction map R: F(source) → F(target)
            # 
            # POT CONVENTION: GW coupling π has shape (n_source, n_target)
            # - Rows sum to p_source: π @ 1 = p_source  
            # - Columns sum to p_target: π.T @ 1 = p_target
            #
            # RESTRICTION MAP CONSTRUCTION:
            # R = diag(1/p_target) @ π^T  
            # - Shape: (n_target, n_source) 
            # - Row-stochastic: each row sums to 1
            # - Maps source stalk F(source) to target stalk F(target)
            
            # Get the measures from the result (will be uniform if not specified)
            p_target_measure = gw_result.p_target
            if p_target_measure is None:
                # Fallback to uniform if not in result
                target_size = gw_result.target_size
                p_target_measure = torch.ones(target_size) / target_size
            
            # Apply barycentric normalization: R = diag(1/p_target) @ π
            # Our coupling π already has shape (n_target, n_source), so no transpose needed
            coupling_matrix = gw_result.coupling  # Shape: (target_size, source_size)
            
            # Normalize each row by dividing by p_target (broadcasting)
            # This ensures each row sums to 1 (row-stochastic)
            restriction_map = coupling_matrix / p_target_measure.unsqueeze(1)
            
            # Validate row-stochasticity
            row_sums = restriction_map.sum(dim=1)
            expected_ones = torch.ones_like(row_sums)
            if not torch.allclose(row_sums, expected_ones, atol=1e-5):
                stochasticity_error = torch.abs(row_sums - expected_ones).max().item()
                logger.warning(f"Restriction map not perfectly row-stochastic for {source}→{target}: "
                             f"max deviation = {stochasticity_error:.6f}, "
                             f"row sums range [{row_sums.min().item():.6f}, {row_sums.max().item():.6f}]")
            else:
                logger.debug(f"✓ Restriction map is row-stochastic for {source}→{target}")
            
            # Validate dimensions
            source_size = C_source.shape[0]
            target_size = C_target.shape[0]
            expected_shape = (target_size, source_size)
            
            if restriction_map.shape != expected_shape:
                logger.error(f"Restriction map dimension mismatch for {source} → {target}: "
                           f"expected {expected_shape}, got {restriction_map.shape}")
                return None
            
            # Store enhanced GW result with temporal tracking
            self._store_enhanced_gw_result(
                (source, target), gw_result, quality_score
            )
            
            # Return restriction map, GW cost, coupling, and quality score
            return restriction_map, gw_result.cost, gw_result.coupling, quality_score
            
        except Exception as e:
            logger.error(f"Single restriction computation failed for {source} → {target}: {e}")
            return None
    
    def validate_quasi_sheaf_property(self, 
                                    restrictions: Dict[Tuple[str, str], torch.Tensor],
                                    poset: nx.DiGraph,
                                    tolerance: float = 0.1) -> Dict[str, Any]:
        """Validate quasi-sheaf property for GW-constructed restrictions.
        
        For GW sheaves with barycentric-normalized restrictions, checks functoriality:
        ||R_{ik} - R_{ij} @ R_{jk}||_F ≤ tolerance
        
        where R_{uv} are row-stochastic restriction maps from node u to node v.
        
        Args:
            restrictions: Dictionary mapping edges (u,v) to restriction maps R_{uv}
            poset: Network structure
            tolerance: Maximum allowed functoriality violation (ε-sheaf threshold)
            
        Returns:
            Validation results dictionary with violation statistics
        """
        violations = []
        max_violation = 0.0
        paths_checked = 0
        
        # Find all 3-node paths for transitivity check
        for node_i in poset.nodes():
            for node_j in poset.successors(node_i):
                for node_k in poset.successors(node_j):
                    # Check path i → j → k
                    edge_ij = (node_i, node_j)
                    edge_jk = (node_j, node_k)
                    edge_ik = (node_i, node_k)
                    
                    if (edge_ij in restrictions and 
                        edge_jk in restrictions and 
                        edge_ik in restrictions):
                        
                        paths_checked += 1
                        
                        try:
                            # Check functoriality: R_{ik} ≈ R_{ij} @ R_{jk}
                            R_ij = restrictions[edge_ij]  # Restriction i→j
                            R_jk = restrictions[edge_jk]  # Restriction j→k
                            R_ik = restrictions[edge_ik]  # Restriction i→k (direct)
                            
                            # Composition: R_{ij} @ R_{jk} should equal R_{ik}
                            composed = R_ij @ R_jk
                            
                            # Functoriality violation (should be small for valid sheaf)
                            violation = torch.norm(R_ik - composed, 'fro').item()
                            violations.append({
                                'path': f"{node_i} → {node_j} → {node_k}",
                                'violation': violation,
                                'nodes': (node_i, node_j, node_k)
                            })
                            max_violation = max(max_violation, violation)
                            
                        except Exception as e:
                            logger.warning(f"Functoriality check failed for {node_i}→{node_j}→{node_k}: {e}")
        
        # Compute statistics
        mean_violation = sum(v['violation'] for v in violations) / len(violations) if violations else 0.0
        satisfies_quasi_sheaf = max_violation <= tolerance
        
        # Count violations by severity
        severe_violations = [v for v in violations if v['violation'] > tolerance]
        moderate_violations = [v for v in violations if tolerance/2 < v['violation'] <= tolerance]
        
        return {
            'max_violation': max_violation,
            'mean_violation': mean_violation,
            'num_paths_checked': paths_checked,
            'violations': violations,
            'severe_violations': severe_violations,
            'moderate_violations': moderate_violations,
            'satisfies_quasi_sheaf': satisfies_quasi_sheaf,
            'tolerance_used': tolerance,
            'violation_rate': len(severe_violations) / paths_checked if paths_checked > 0 else 0.0
        }
    
    def extract_edge_weights(self, 
                           gw_costs: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        """Extract GW costs as edge weights for persistence analysis.
        
        Important: GW costs represent metric distortion (lower = better match)
        This is opposite to Procrustes norms (higher = stronger connection)
        
        Args:
            gw_costs: Dictionary mapping edges to GW distortion costs
            
        Returns:
            Dictionary mapping edges to weights for increasing filtration
        """
        logger.info(f"Extracting edge weights from {len(gw_costs)} GW costs")
        
        if not gw_costs:
            logger.warning("No GW costs provided for edge weight extraction")
            return {}
        
        # GW costs are already in the correct format for increasing filtration
        # Lower costs = better matches = added first in increasing complexity filtration
        edge_weights = dict(gw_costs)
        
        # Log statistics
        costs = list(gw_costs.values())
        min_cost = min(costs)
        max_cost = max(costs)
        mean_cost = sum(costs) / len(costs)
        
        logger.info(f"GW edge weights: min={min_cost:.4f}, max={max_cost:.4f}, "
                   f"mean={mean_cost:.4f} (for INCREASING filtration)")
        
        return edge_weights
    
    def _generate_transport_metadata(self, 
                                   gw_couplings: Dict[Tuple[str, str], torch.Tensor],
                                   gw_costs: Dict[Tuple[str, str], float],
                                   gw_quality_scores: Dict[Tuple[str, str], float],
                                   failed_edges: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Generate standardized transport metadata for availability tracking.
        
        Creates structured metadata that enables robust transport access
        for inclusion mappings in sparse GW filtrations.
        
        Args:
            gw_couplings: Dictionary of GW coupling matrices by edge
            gw_costs: Dictionary of GW costs by edge  
            gw_quality_scores: Dictionary of quality scores by edge
            failed_edges: List of edges that failed GW computation
            
        Returns:
            Standardized transport metadata dictionary
        """
        current_timestamp = time.time()
        
        # Build per-edge transport availability information
        edge_transport_info = {}
        available_edges = set(gw_couplings.keys())
        
        for edge in available_edges:
            coupling = gw_couplings[edge]
            cost = gw_costs.get(edge, float('inf'))
            quality = gw_quality_scores.get(edge, 0.0)
            creation_time = self._edge_creation_timestamps.get(edge, current_timestamp)
            
            # Determine transport availability status
            if edge in failed_edges:
                availability_status = 'failed'
            elif quality < self.config.min_coupling_quality:
                availability_status = 'low_quality'
            else:
                availability_status = 'available'
            
            edge_transport_info[edge] = {
                'availability_status': availability_status,
                'coupling_shape': tuple(coupling.shape) if coupling is not None else None,
                'gw_cost': cost,
                'quality_score': quality,
                'creation_timestamp': creation_time,
                'measures_uniform': True,  # Default for current implementation
                'solver_type': getattr(self._gw_results.get(edge, {}), 'solver_type', 'unknown'),
                'coupling_quality': getattr(self._gw_results.get(edge, {}), 'coupling_quality', 'unknown'),
                'has_valid_coupling': coupling is not None and coupling.numel() > 0
            }
        
        # Add failed edges with minimal info
        for edge in failed_edges:
            if edge not in edge_transport_info:
                edge_transport_info[edge] = {
                    'availability_status': 'failed',
                    'coupling_shape': None,
                    'gw_cost': float('inf'),
                    'quality_score': 0.0,
                    'creation_timestamp': self._edge_creation_timestamps.get(edge, current_timestamp),
                    'measures_uniform': True,
                    'solver_type': 'failed',
                    'coupling_quality': 'failed',
                    'has_valid_coupling': False
                }
        
        # Compute availability statistics
        total_edges = len(edge_transport_info)
        available_count = sum(1 for info in edge_transport_info.values() 
                            if info['availability_status'] == 'available')
        failed_count = len(failed_edges)
        low_quality_count = sum(1 for info in edge_transport_info.values() 
                              if info['availability_status'] == 'low_quality')
        
        # Create nearest neighbor mapping for fallback policy
        nearest_neighbors = self._compute_nearest_neighbor_mapping(
            edge_transport_info, available_edges
        )
        
        transport_metadata = {
            'version': '1.0',
            'generation_timestamp': current_timestamp,
            'edge_transport_info': edge_transport_info,
            'availability_statistics': {
                'total_edges': total_edges,
                'available_count': available_count,
                'failed_count': failed_count,
                'low_quality_count': low_quality_count,
                'availability_rate': available_count / total_edges if total_edges > 0 else 0.0
            },
            'nearest_neighbor_mapping': nearest_neighbors,
            'fallback_policy': 'nearest_neighbor_quality_weighted',
            'quality_statistics': {
                'min_quality': min((info['quality_score'] for info in edge_transport_info.values() 
                                  if info['has_valid_coupling']), default=0.0),
                'max_quality': max((info['quality_score'] for info in edge_transport_info.values() 
                                  if info['has_valid_coupling']), default=0.0),
                'mean_quality': sum(info['quality_score'] for info in edge_transport_info.values() 
                                  if info['has_valid_coupling']) / max(available_count, 1)
            }
        }
        
        # Store for later access
        self._transport_metadata = transport_metadata
        
        logger.info(f"Generated transport metadata: {available_count}/{total_edges} edges available, "
                   f"{failed_count} failed, {low_quality_count} low quality")
        
        return transport_metadata
    
    def _compute_nearest_neighbor_mapping(self, 
                                        edge_transport_info: Dict[Tuple[str, str], Dict[str, Any]],
                                        available_edges: set) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """Compute nearest neighbor mapping for fallback policy.
        
        For each edge, finds the k nearest available edges based on 
        quality score similarity and coupling shape compatibility.
        
        Args:
            edge_transport_info: Per-edge transport information
            available_edges: Set of edges with available transport
            
        Returns:
            Dictionary mapping each edge to ordered list of nearest neighbors
        """
        nearest_neighbors = {}
        
        # Filter to only high-quality available edges for fallback
        high_quality_edges = [
            edge for edge in available_edges 
            if (edge_transport_info[edge]['availability_status'] == 'available' and
                edge_transport_info[edge]['quality_score'] >= self.config.min_coupling_quality)
        ]
        
        if not high_quality_edges:
            # No high-quality edges available - use any available edges as fallback
            high_quality_edges = list(available_edges)
        
        for target_edge in edge_transport_info.keys():
            if target_edge in high_quality_edges:
                # Available edges map to themselves first, then others
                neighbors = [target_edge] + [e for e in high_quality_edges if e != target_edge]
            else:
                # Missing/failed edges get all available edges sorted by quality
                neighbors = sorted(high_quality_edges, 
                                 key=lambda e: edge_transport_info[e]['quality_score'], 
                                 reverse=True)
            
            # Limit to top 5 neighbors for performance
            nearest_neighbors[target_edge] = neighbors[:5]
        
        return nearest_neighbors
    
    def _store_enhanced_gw_result(self, 
                                edge: Tuple[str, str],
                                gw_result: GWResult,
                                quality_score: float) -> None:
        """Store enhanced GW result with temporal tracking metadata.
        
        Args:
            edge: Edge tuple (source, target)
            gw_result: GW computation result
            quality_score: Computed quality score
        """
        current_timestamp = time.time()
        
        # Update temporal tracking
        self._edge_creation_timestamps[edge] = current_timestamp
        
        # Enhance GW result with temporal metadata if needed
        if not hasattr(gw_result, 'creation_timestamp') or gw_result.creation_timestamp is None:
            gw_result.creation_timestamp = current_timestamp
        
        if not hasattr(gw_result, 'edge_info') or gw_result.edge_info is None:
            gw_result.edge_info = {
                'source_node': edge[0],
                'target_node': edge[1],
                'computation_timestamp': current_timestamp,
                'quality_score': quality_score
            }
        
        # Store enhanced result
        self._gw_results[edge] = gw_result
        
        logger.debug(f"Stored enhanced GW result for edge {edge}: quality={quality_score:.4f}")
    
    def get_transport_metadata(self) -> Optional[Dict[str, Any]]:
        """Get the current transport metadata for external access.
        
        Returns:
            Transport metadata dictionary or None if not available
        """
        return self._transport_metadata.copy() if self._transport_metadata else None
    
    def get_transport_info_for_edge(self, edge: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        """Get transport information for a specific edge.
        
        Args:
            edge: Edge tuple (source, target)
            
        Returns:
            Transport info dictionary or None if not available
        """
        if not self._transport_metadata:
            return None
        
        return self._transport_metadata.get('edge_transport_info', {}).get(edge)
    
    def get_fallback_edges_for_edge(self, edge: Tuple[str, str]) -> List[Tuple[str, str]]:
        """Get ordered list of fallback edges for a given edge.
        
        Args:
            edge: Edge tuple to find fallbacks for
            
        Returns:
            List of fallback edges in priority order
        """
        if not self._transport_metadata:
            return []
        
        nearest_neighbors = self._transport_metadata.get('nearest_neighbor_mapping', {})
        return nearest_neighbors.get(edge, [])
    
    def clear_cache(self) -> None:
        """Clear all cached data including transport metadata."""
        self._cost_cache.clear()
        self._gw_results.clear()
        self._transport_metadata.clear()
        self._edge_creation_timestamps.clear()
        if self.gw_computer.cost_cache:
            self.gw_computer.cost_cache.clear()
        logger.info("GW restriction manager cache and transport metadata cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        stats = {
            'cost_cache_entries': len(self._cost_cache),
            'gw_results_entries': len(self._gw_results),
            'transport_metadata_available': len(self._transport_metadata) > 0,
            'edge_timestamps_tracked': len(self._edge_creation_timestamps)
        }
        
        if self.gw_computer.cost_cache:
            stats['computer_cache_entries'] = len(self.gw_computer.cost_cache.cache)
            stats['computer_cache_bytes'] = self.gw_computer.cost_cache.current_bytes
            stats['computer_cache_max_bytes'] = self.gw_computer.cost_cache.max_bytes
        
        return stats
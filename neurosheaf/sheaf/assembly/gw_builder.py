"""GW-based sheaf assembly and restriction map computation.

This module provides the GWRestrictionManager class for computing restriction maps
using Gromov-Wasserstein optimal transport. It extends the existing sheaf assembly
infrastructure with GW-specific functionality while maintaining backward compatibility.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import networkx as nx
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..core import GWConfig, GromovWassersteinComputer, GWResult
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
        # Store both couplings and costs for each edge
        self._gw_results = {}
        
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
            edges = list(poset.edges())
            restrictions = {}
            gw_costs = {}
            gw_couplings = {}
            failed_edges = []
            
            if parallel and len(edges) > 1:
                # Parallel processing for multiple edges
                logger.info(f"Processing {len(edges)} edges in parallel")
                restrictions, gw_costs, gw_couplings, failed_edges = self._compute_restrictions_parallel(
                    edges, cost_matrices, activations, max_workers
                )
            else:
                # Sequential processing
                logger.info(f"Processing {len(edges)} edges sequentially")
                restrictions, gw_costs, gw_couplings, failed_edges = self._compute_restrictions_sequential(
                    edges, cost_matrices, activations
                )
            
            # 3. Validate quasi-sheaf property if requested
            validation_report = None
            if self.config.quasi_sheaf_tolerance > 0:
                logger.info("Validating quasi-sheaf property")
                validation_report = self.validate_quasi_sheaf_property(
                    restrictions, poset, self.config.quasi_sheaf_tolerance
                )
                logger.info(f"Quasi-sheaf validation: max_violation={validation_report['max_violation']:.2e}, "
                           f"satisfies_quasi_sheaf={validation_report['satisfies_quasi_sheaf']}")
            
            # 4. Prepare comprehensive metadata
            total_time = time.time() - start_time
            metadata = {
                'computation_time': total_time,
                'num_edges_processed': len(edges),
                'num_edges_succeeded': len(restrictions),
                'num_edges_failed': len(failed_edges),
                'failed_edges': failed_edges,
                'gw_config': self.config.to_dict(),
                'gw_couplings': gw_couplings,
                'validation_report': validation_report,
                'cost_matrix_cache_hits': getattr(self.gw_computer.cost_cache, 'hits', 0) if self.gw_computer.cost_cache else 0,
                'parallel_processing': parallel,
                'construction_method': 'gromov_wasserstein'
            }
            
            logger.info(f"GW restriction computation complete: {len(restrictions)}/{len(edges)} edges "
                       f"succeeded in {total_time:.2f}s")
            
            return restrictions, gw_costs, metadata
            
        except Exception as e:
            logger.error(f"GW restriction computation failed: {e}", exc_info=True)
            raise GWRestrictionError(f"Failed to compute GW restrictions: {e}")
    
    def _compute_all_cost_matrices(self, activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute cosine cost matrices for all activation layers.
        
        Args:
            activations: Dictionary mapping node names to activation tensors
            
        Returns:
            Dictionary mapping node names to cost matrices
        """
        cost_matrices = {}
        
        for node_name, activation_tensor in activations.items():
            try:
                # Reshape activation tensor to (n_samples, n_features) if needed
                if activation_tensor.dim() > 2:
                    # Flatten all dimensions except first (batch dimension)
                    activation_tensor = activation_tensor.view(activation_tensor.shape[0], -1)
                
                # Compute cosine cost matrix
                cost_matrix = self.gw_computer.compute_cosine_cost_matrix(activation_tensor)
                cost_matrices[node_name] = cost_matrix
                
                logger.debug(f"Computed cost matrix for {node_name}: {cost_matrix.shape}")
                
            except Exception as e:
                logger.error(f"Failed to compute cost matrix for {node_name}: {e}")
                raise GWRestrictionError(f"Cost matrix computation failed for {node_name}: {e}")
        
        logger.info(f"Computed cost matrices for {len(cost_matrices)} nodes")
        return cost_matrices
    
    def _compute_restrictions_sequential(self, 
                                       edges: List[Tuple[str, str]],
                                       cost_matrices: Dict[str, torch.Tensor],
                                       activations: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict, Dict, List]:
        """Compute restrictions sequentially for all edges.
        
        Args:
            edges: List of edges to process
            cost_matrices: Pre-computed cost matrices
            activations: Original activation tensors
            
        Returns:
            Tuple of (restrictions, gw_costs, gw_couplings, failed_edges)
        """
        restrictions = {}
        gw_costs = {}
        gw_couplings = {}
        failed_edges = []
        
        for i, (source, target) in enumerate(edges):
            logger.debug(f"Processing edge {i+1}/{len(edges)}: {source} → {target}")
            
            try:
                result = self._compute_single_restriction(
                    source, target, cost_matrices, activations
                )
                
                if result is not None:
                    restriction, cost, coupling = result
                    restrictions[(source, target)] = restriction
                    gw_costs[(source, target)] = cost
                    gw_couplings[(source, target)] = coupling
                    logger.debug(f"✓ Computed restriction {source} → {target}: {restriction.shape}")
                else:
                    failed_edges.append((source, target))
                    
            except Exception as e:
                logger.warning(f"Failed to compute restriction {source} → {target}: {e}")
                failed_edges.append((source, target))
        
        return restrictions, gw_costs, gw_couplings, failed_edges
    
    def _compute_restrictions_parallel(self, 
                                     edges: List[Tuple[str, str]],
                                     cost_matrices: Dict[str, torch.Tensor],
                                     activations: Dict[str, torch.Tensor],
                                     max_workers: Optional[int] = None) -> Tuple[Dict, Dict, Dict, List]:
        """Compute restrictions in parallel for all edges.
        
        Args:
            edges: List of edges to process
            cost_matrices: Pre-computed cost matrices
            activations: Original activation tensors
            max_workers: Maximum number of worker threads
            
        Returns:
            Tuple of (restrictions, gw_costs, gw_couplings, failed_edges)
        """
        restrictions = {}
        gw_costs = {}
        gw_couplings = {}
        failed_edges = []
        
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
                        restriction, cost, coupling = result
                        restrictions[(source, target)] = restriction
                        gw_costs[(source, target)] = cost
                        gw_couplings[(source, target)] = coupling
                        logger.debug(f"✓ Computed restriction {source} → {target}: {restriction.shape}")
                    else:
                        failed_edges.append((source, target))
                        
                except Exception as e:
                    logger.warning(f"Parallel computation failed for {source} → {target}: {e}")
                    failed_edges.append((source, target))
        
        return restrictions, gw_costs, gw_couplings, failed_edges
    
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
                # Could implement importance sampling here if needed
                logger.debug("Non-uniform measures requested but not implemented, using uniform")
            
            # Compute GW coupling
            gw_result = self.gw_computer.compute_gw_coupling(
                C_source, C_target, p_source, p_target
            )
            
            # Extract restriction map: ρ_{target→source} = π^T
            # The GW coupling π has shape (source_size, target_size) from POT
            # We need the restriction map to have shape (target_size, source_size)
            restriction_map = gw_result.coupling.T  # Transpose to get correct shape
            
            # Validate dimensions
            source_size = C_source.shape[0]
            target_size = C_target.shape[0]
            expected_shape = (target_size, source_size)
            
            if restriction_map.shape != expected_shape:
                logger.error(f"Restriction map dimension mismatch for {source} → {target}: "
                           f"expected {expected_shape}, got {restriction_map.shape}")
                return None
            
            # Return restriction map, GW cost, and coupling
            return restriction_map, gw_result.cost, gw_result.coupling
            
        except Exception as e:
            logger.error(f"Single restriction computation failed for {source} → {target}: {e}")
            return None
    
    def validate_quasi_sheaf_property(self, 
                                    restrictions: Dict[Tuple[str, str], torch.Tensor],
                                    poset: nx.DiGraph,
                                    tolerance: float = 0.1) -> Dict[str, Any]:
        """Validate quasi-sheaf property for GW-constructed restrictions.
        
        For GW sheaves, checks functoriality violations:
        ||ρ_{k→i} - ρ_{j→i} ∘ ρ_{k→j}||_F ≤ tolerance
        
        Args:
            restrictions: Dictionary mapping edges to restriction tensors
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
                            # Compute ρ_{k→i} - ρ_{j→i} ∘ ρ_{k→j}
                            R_ij = restrictions[edge_ij]  # ρ_{j→i}
                            R_jk = restrictions[edge_jk]  # ρ_{k→j}  
                            R_ik = restrictions[edge_ik]  # ρ_{k→i}
                            
                            # Composition: ρ_{j→i} ∘ ρ_{k→j}
                            composed = R_ij @ R_jk
                            
                            # Functoriality violation
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
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cost_cache.clear()
        self._gw_results.clear()
        if self.gw_computer.cost_cache:
            self.gw_computer.cost_cache.clear()
        logger.info("GW restriction manager cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        stats = {
            'cost_cache_entries': len(self._cost_cache),
            'gw_results_entries': len(self._gw_results),
        }
        
        if self.gw_computer.cost_cache:
            stats['computer_cache_entries'] = len(self.gw_computer.cost_cache.cache)
            stats['computer_cache_bytes'] = self.gw_computer.cost_cache.current_bytes
            stats['computer_cache_max_bytes'] = self.gw_computer.cost_cache.max_bytes
        
        return stats
"""Unified edge weight extraction for different sheaf construction methods.

This module provides a centralized system for extracting edge weights from sheaves
constructed using different methods (Procrustes, GW, etc.), ensuring consistent
handling of filtration semantics and weight interpretation across the pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
import torch
import numpy as np
import logging

from ..sheaf.data_structures import Sheaf

logger = logging.getLogger(__name__)


@dataclass 
class EdgeWeightMetadata:
    """Metadata about extracted edge weights.
    
    Attributes:
        construction_method: Method used to construct the sheaf
        weight_type: Type of weights ('gw_costs', 'frobenius_norms', etc.)
        weight_range: (min_weight, max_weight) tuple
        filtration_semantics: 'increasing' or 'decreasing' complexity
        fallback_used: Whether fallback method was used due to missing data
        num_edges: Number of edges processed
    """
    construction_method: str = "unknown"
    weight_type: str = "unknown"
    weight_range: Tuple[float, float] = (0.0, 1.0)
    filtration_semantics: str = "decreasing"
    fallback_used: bool = False
    num_edges: int = 0


class EdgeWeightExtractor:
    """Unified edge weight extraction for different sheaf types.
    
    This class provides a single interface for extracting edge weights
    from sheaves constructed using different methods, with proper handling
    of the different semantic interpretations:
    
    - Standard sheaves: Frobenius norms (higher = stronger connection)
    - GW sheaves: GW costs (lower = better match)
    - Whitened sheaves: Scaled norms with whitening factors
    
    The extractor also handles fallback scenarios where metadata is missing
    or incomplete, ensuring robust operation across different use cases.
    """
    
    def __init__(self, 
                 validate_weights: bool = True,
                 log_statistics: bool = True):
        """Initialize edge weight extractor.
        
        Args:
            validate_weights: Whether to validate extracted weights
            log_statistics: Whether to log weight statistics
        """
        self.validate_weights = validate_weights
        self.log_statistics = log_statistics
        
        logger.info(f"EdgeWeightExtractor initialized: validate={validate_weights}, "
                   f"log_stats={log_statistics}")
    
    def extract_weights(self, sheaf: Sheaf) -> Tuple[Dict[Tuple[str, str], float], EdgeWeightMetadata]:
        """Extract edge weights based on construction method.
        
        This is the main entry point for edge weight extraction. It automatically
        detects the construction method and routes to the appropriate extraction
        approach, ensuring consistent semantics across the pipeline.
        
        Args:
            sheaf: Sheaf object with restrictions and metadata
            
        Returns:
            Tuple of (edge_weights, metadata)
            - edge_weights: Dictionary mapping edges to weight values
            - metadata: Information about the extraction process
        """
        logger.info(f"Extracting edge weights from sheaf with {len(sheaf.restrictions)} edges")
        
        # Detect construction method
        construction_method = sheaf.metadata.get('construction_method', 'unknown')
        
        # Route to appropriate extraction method
        if construction_method == 'gromov_wasserstein' or sheaf.is_gw_sheaf():
            weights, metadata = self._extract_gw_weights(sheaf)
        elif construction_method in ['scaled_procrustes', 'whitened_procrustes', 'fx_unified_whitened']:
            weights, metadata = self._extract_procrustes_weights(sheaf)
        else:
            weights, metadata = self._extract_standard_weights(sheaf)
        
        # Validate weights if requested
        if self.validate_weights:
            self._validate_weights(weights, metadata)
        
        # Log statistics if requested
        if self.log_statistics:
            self._log_weight_statistics(weights, metadata)
        
        return weights, metadata
    
    def _extract_gw_weights(self, sheaf: Sheaf) -> Tuple[Dict, EdgeWeightMetadata]:
        """Extract GW costs as weights for GW-based sheaves.
        
        GW costs represent metric distortion where lower values indicate
        better matches. This creates an INCREASING complexity filtration.
        """
        logger.debug("Extracting GW costs as edge weights")
        
        # Primary source: stored GW costs from construction
        gw_costs = sheaf.metadata.get('gw_costs', {})
        fallback_used = False
        
        if gw_costs and len(gw_costs) == len(sheaf.restrictions):
            # Use stored GW costs directly
            weights = dict(gw_costs)
            weight_type = "gw_costs"
        else:
            # Fallback: compute from restriction properties
            logger.warning("GW costs missing/incomplete, computing from restriction norms")
            weights = {}
            for edge, restriction in sheaf.restrictions.items():
                # Use operator 2-norm as proxy for distortion
                if isinstance(restriction, torch.Tensor):
                    weight = torch.linalg.norm(restriction, ord=2).item()
                else:
                    weight = np.linalg.norm(restriction, ord=2)
                weights[edge] = weight
            
            weight_type = "operator_norms_fallback"
            fallback_used = True
        
        # Create metadata
        weight_values = list(weights.values())
        metadata = EdgeWeightMetadata(
            construction_method="gromov_wasserstein",
            weight_type=weight_type,
            weight_range=(min(weight_values), max(weight_values)) if weight_values else (0.0, 1.0),
            filtration_semantics="increasing",
            fallback_used=fallback_used,
            num_edges=len(weights)
        )
        
        return weights, metadata
    
    def _extract_procrustes_weights(self, sheaf: Sheaf) -> Tuple[Dict, EdgeWeightMetadata]:
        """Extract weights for Procrustes-based sheaves using Frobenius norms.
        
        Procrustes-based sheaves use Frobenius norms where higher values
        indicate stronger connections. This creates a DECREASING complexity filtration.
        """
        logger.debug("Extracting Procrustes weights using Frobenius norms")
        
        weights = {}
        for edge, restriction in sheaf.restrictions.items():
            # Use Frobenius norm for Procrustes-based restrictions
            if isinstance(restriction, torch.Tensor):
                weight = torch.norm(restriction, p='fro').item()
            else:
                weight = np.linalg.norm(restriction, ord='fro')
            weights[edge] = weight
        
        # Check for whitening scaling factors
        whitening_metadata = sheaf.metadata.get('whitening_metadata', {})
        weight_type = "frobenius_norms_whitened" if whitening_metadata else "frobenius_norms"
        
        # Create metadata
        weight_values = list(weights.values())
        metadata = EdgeWeightMetadata(
            construction_method=sheaf.metadata.get('construction_method', 'procrustes'),
            weight_type=weight_type,
            weight_range=(min(weight_values), max(weight_values)) if weight_values else (0.0, 1.0),
            filtration_semantics="decreasing",
            fallback_used=False,
            num_edges=len(weights)
        )
        
        return weights, metadata
    
    def _extract_standard_weights(self, sheaf: Sheaf) -> Tuple[Dict, EdgeWeightMetadata]:
        """Extract weights for standard/unknown sheaves using Frobenius norms."""
        logger.debug("Extracting standard weights using Frobenius norms")
        
        weights = {}
        for edge, restriction in sheaf.restrictions.items():
            # Use Frobenius norm as default
            if isinstance(restriction, torch.Tensor):
                weight = torch.norm(restriction, p='fro').item()
            else:
                weight = np.linalg.norm(restriction, ord='fro')
            weights[edge] = weight
        
        # Create metadata
        weight_values = list(weights.values())
        metadata = EdgeWeightMetadata(
            construction_method=sheaf.metadata.get('construction_method', 'standard'),
            weight_type="frobenius_norms",
            weight_range=(min(weight_values), max(weight_values)) if weight_values else (0.0, 1.0),
            filtration_semantics="decreasing",
            fallback_used=False,
            num_edges=len(weights)
        )
        
        return weights, metadata
    
    def _validate_weights(self, weights: Dict, metadata: EdgeWeightMetadata):
        """Validate extracted edge weights for consistency and correctness."""
        if not weights:
            logger.warning("No edge weights extracted")
            return
        
        weight_values = list(weights.values())
        
        # Check for non-positive weights (problematic for most applications)
        non_positive = [w for w in weight_values if w <= 0]
        if non_positive:
            logger.warning(f"Found {len(non_positive)} non-positive weights: "
                          f"min={min(non_positive):.6f}")
        
        # Check for infinite or NaN weights
        invalid = [w for w in weight_values if not np.isfinite(w)]
        if invalid:
            logger.error(f"Found {len(invalid)} invalid weights (inf/NaN)")
        
        # Check weight range consistency with construction method
        min_weight, max_weight = metadata.weight_range
        if metadata.construction_method == 'gromov_wasserstein':
            # GW costs are typically in [0, 2] for cosine distance
            if max_weight > 10.0:
                logger.warning(f"Unusually large GW costs detected: max={max_weight:.4f}")
        else:
            # Frobenius norms can vary widely but should be positive
            if min_weight == max_weight:
                logger.warning(f"All edge weights are identical: {min_weight:.6f}")
        
        logger.debug(f"Weight validation completed: {len(weights)} weights, "
                    f"range=[{min_weight:.4f}, {max_weight:.4f}]")
    
    def _log_weight_statistics(self, weights: Dict, metadata: EdgeWeightMetadata):
        """Log detailed statistics about extracted edge weights."""
        if not weights:
            return
        
        weight_values = list(weights.values())
        min_weight = min(weight_values)
        max_weight = max(weight_values)
        mean_weight = np.mean(weight_values)
        std_weight = np.std(weight_values)
        
        logger.info(f"Edge weight statistics ({metadata.weight_type}):")
        logger.info(f"  Range: [{min_weight:.4f}, {max_weight:.4f}]")
        logger.info(f"  Mean±Std: {mean_weight:.4f}±{std_weight:.4f}")
        logger.info(f"  Filtration: {metadata.filtration_semantics} complexity")
        logger.info(f"  Method: {metadata.construction_method}")
        
        if metadata.fallback_used:
            logger.warning(f"  Fallback method used due to missing metadata")
        
        # Log distribution statistics for detailed analysis
        if len(weight_values) > 1:
            percentiles = np.percentile(weight_values, [25, 50, 75])
            logger.debug(f"  Quartiles: [Q1={percentiles[0]:.4f}, "
                        f"Q2={percentiles[1]:.4f}, Q3={percentiles[2]:.4f}]")
            
            # Check for outliers (simple IQR method)
            iqr = percentiles[2] - percentiles[0]
            lower_bound = percentiles[0] - 1.5 * iqr
            upper_bound = percentiles[2] + 1.5 * iqr
            outliers = [w for w in weight_values if w < lower_bound or w > upper_bound]
            
            if outliers:
                logger.debug(f"  Outliers: {len(outliers)} weights outside "
                            f"[{lower_bound:.4f}, {upper_bound:.4f}]")
    
    def get_filtration_direction(self, construction_method: str) -> str:
        """Get appropriate filtration direction for construction method.
        
        Args:
            construction_method: Sheaf construction method
            
        Returns:
            'increasing' for GW (add edges with low cost first)
            'decreasing' for standard (keep edges with high weight longest)
        """
        if construction_method == 'gromov_wasserstein':
            return 'increasing'
        else:
            return 'decreasing'
    
    def create_threshold_function(self, filtration_semantics: str) -> callable:
        """Create appropriate threshold function for filtration semantics.
        
        Args:
            filtration_semantics: 'increasing' or 'decreasing' complexity
            
        Returns:
            Function with signature (weight, param) -> bool
        """
        if filtration_semantics == 'increasing':
            # Include edges with weight ≤ threshold (GW: small costs first)
            return lambda weight, param: weight <= param
        else:
            # Include edges with weight ≥ threshold (Standard: large weights first)
            return lambda weight, param: weight >= param


# Convenience functions for common use cases
def extract_edge_weights(sheaf: Sheaf) -> Dict[Tuple[str, str], float]:
    """Convenience function to extract edge weights without metadata.
    
    Args:
        sheaf: Sheaf object
        
    Returns:
        Dictionary mapping edges to weights
    """
    extractor = EdgeWeightExtractor(log_statistics=False)
    weights, _ = extractor.extract_weights(sheaf)
    return weights


def get_filtration_semantics(sheaf: Sheaf) -> str:
    """Get filtration semantics for a sheaf.
    
    Args:
        sheaf: Sheaf object
        
    Returns:
        'increasing' or 'decreasing' complexity filtration
    """
    if hasattr(sheaf, 'get_filtration_semantics'):
        return sheaf.get_filtration_semantics()
    
    # Fallback to construction method detection
    construction_method = sheaf.metadata.get('construction_method', 'standard')
    extractor = EdgeWeightExtractor(log_statistics=False)
    return 'increasing' if construction_method == 'gromov_wasserstein' else 'decreasing'
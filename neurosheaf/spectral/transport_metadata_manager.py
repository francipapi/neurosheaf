"""
Transport metadata management for GW filtrations.

This module provides the TransportMetadataManager utility for centralized
access to transport information with intelligent fallback policies for
sparse filtrations where some edges lack transport data.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError

logger = setup_logger(__name__)


class TransportNotAvailableError(ComputationError):
    """Exception raised when transport information is not available for an edge."""
    
    def __init__(self, edge: Tuple[str, str], reason: str, operation: str = "transport_access"):
        self.edge = edge
        self.reason = reason
        message = f"Transport not available for edge {edge}: {reason}"
        super().__init__(message, operation=operation)


class TransportQualityError(ComputationError):
    """Exception raised when transport quality is below threshold."""
    
    def __init__(self, edge: Tuple[str, str], quality: float, threshold: float, operation: str = "transport_access"):
        self.edge = edge
        self.quality = quality
        self.threshold = threshold
        message = f"Transport quality {quality:.4f} below threshold {threshold:.4f} for edge {edge}"
        super().__init__(message, operation=operation)


class TransportMetadataManager:
    """
    Centralized manager for transport metadata access with fallback policies.
    
    Provides intelligent access to GW transport information with fallback
    strategies for sparse filtrations where some edges lack transport data.
    
    Key Features:
    - Centralized transport metadata access
    - Quality-aware transport selection
    - Nearest neighbor fallback policy
    - Temporal tracking of transport availability
    - Comprehensive error reporting with specific transport error types
    """
    
    def __init__(self, 
                 transport_metadata: Optional[Dict[str, Any]] = None,
                 fallback_policy: str = 'nearest_neighbor_quality_weighted',
                 quality_threshold: float = 0.5,
                 strict_mode: bool = False,
                 max_fallback_attempts: int = 3):
        """
        Initialize transport metadata manager.
        
        Args:
            transport_metadata: Standardized transport metadata from GWRestrictionManager
            fallback_policy: Policy for handling missing transport data
                - 'nearest_neighbor_quality_weighted': Use quality-weighted nearest neighbors
                - 'best_available': Use highest quality available transport
                - 'strict': Fail if exact transport not available
            quality_threshold: Minimum quality threshold for transport selection
            strict_mode: If True, raise errors for missing transport; if False, use fallbacks
            max_fallback_attempts: Maximum number of fallback edges to try
        """
        self.transport_metadata = transport_metadata or {}
        self.fallback_policy = fallback_policy
        self.quality_threshold = quality_threshold
        self.strict_mode = strict_mode
        self.max_fallback_attempts = max_fallback_attempts
        
        # Extract edge transport info for quick access
        self.edge_transport_info = self.transport_metadata.get('edge_transport_info', {})
        self.nearest_neighbor_mapping = self.transport_metadata.get('nearest_neighbor_mapping', {})
        self.availability_stats = self.transport_metadata.get('availability_statistics', {})
        
        # Validate fallback policy
        valid_policies = ['nearest_neighbor_quality_weighted', 'best_available', 'strict']
        if fallback_policy not in valid_policies:
            raise ValueError(f"Invalid fallback policy '{fallback_policy}'. Valid options: {valid_policies}")
        
        logger.info(f"TransportMetadataManager initialized: policy={fallback_policy}, "
                   f"quality_threshold={quality_threshold}, strict_mode={strict_mode}")
        
        if self.availability_stats:
            available_count = self.availability_stats.get('available_count', 0)
            total_edges = self.availability_stats.get('total_edges', 0)
            availability_rate = self.availability_stats.get('availability_rate', 0.0)
            logger.info(f"Transport availability: {available_count}/{total_edges} edges available "
                       f"({availability_rate:.1%})")
    
    def get_transport_matrix(self, 
                           edge: Tuple[str, str],
                           gw_couplings: Dict[Tuple[str, str], torch.Tensor],
                           allow_fallback: bool = True) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Get transport matrix for edge with intelligent fallback.
        
        Returns the GW coupling matrix for the specified edge, with fallback
        to nearest neighbor edges if the direct edge is not available.
        
        Args:
            edge: Target edge (source, target)
            gw_couplings: Dictionary of GW coupling matrices by edge
            allow_fallback: Whether to use fallback policy if direct edge not available
            
        Returns:
            Tuple of (transport_matrix, selection_info):
            - transport_matrix: Transport coupling matrix or None if not found
            - selection_info: Information about transport selection including fallback
            
        Raises:
            TransportNotAvailableError: If transport not available and strict_mode=True
            TransportQualityError: If transport quality below threshold and strict_mode=True
        """
        logger.debug(f"Requesting transport matrix for edge {edge}")
        
        selection_info = {
            'requested_edge': edge,
            'selected_edge': None,
            'selection_method': None,
            'quality_score': None,
            'availability_status': None,
            'fallback_used': False,
            'fallback_attempts': 0,
            'error': None
        }
        
        # Check if direct edge is available
        direct_info = self.edge_transport_info.get(edge)
        if direct_info:
            selection_info['availability_status'] = direct_info['availability_status']
            
            if (direct_info['availability_status'] == 'available' and 
                direct_info['has_valid_coupling'] and
                edge in gw_couplings):
                
                quality = direct_info['quality_score']
                selection_info['quality_score'] = quality
                
                # Check quality threshold
                if quality >= self.quality_threshold:
                    selection_info.update({
                        'selected_edge': edge,
                        'selection_method': 'direct',
                        'fallback_used': False
                    })
                    
                    transport_matrix = gw_couplings[edge]
                    logger.debug(f"✓ Direct transport matrix for {edge}: shape {transport_matrix.shape}, "
                               f"quality={quality:.4f}")
                    return transport_matrix, selection_info
                
                elif self.strict_mode:
                    error_msg = f"quality {quality:.4f} below threshold {self.quality_threshold}"
                    selection_info['error'] = error_msg
                    raise TransportQualityError(edge, quality, self.quality_threshold)
                
                elif not allow_fallback:
                    selection_info['error'] = f"quality {quality:.4f} below threshold, fallback disabled"
                    logger.warning(f"Transport quality {quality:.4f} below threshold {self.quality_threshold} "
                                 f"for {edge}, fallback disabled")
                    return None, selection_info
        
        # Direct edge not available or below quality threshold - try fallback
        if not allow_fallback:
            error_msg = f"not available, fallback disabled"
            selection_info['error'] = error_msg
            if self.strict_mode:
                raise TransportNotAvailableError(edge, error_msg)
            logger.warning(f"Transport not available for {edge}, fallback disabled")
            return None, selection_info
        
        # Apply fallback policy
        fallback_result = self._apply_fallback_policy(edge, gw_couplings, selection_info)
        
        if fallback_result[0] is None and self.strict_mode:
            error_msg = selection_info.get('error', 'no suitable fallback found')
            raise TransportNotAvailableError(edge, error_msg)
        
        return fallback_result
    
    def _apply_fallback_policy(self, 
                             edge: Tuple[str, str],
                             gw_couplings: Dict[Tuple[str, str], torch.Tensor],
                             selection_info: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Apply the configured fallback policy to find alternative transport.
        
        Args:
            edge: Target edge
            gw_couplings: Available GW coupling matrices
            selection_info: Information dictionary to update
            
        Returns:
            Tuple of (transport_matrix, selection_info)
        """
        logger.debug(f"Applying fallback policy '{self.fallback_policy}' for edge {edge}")
        
        if self.fallback_policy == 'strict':
            selection_info['error'] = 'strict policy, no fallback allowed'
            return None, selection_info
        
        elif self.fallback_policy == 'nearest_neighbor_quality_weighted':
            return self._nearest_neighbor_fallback(edge, gw_couplings, selection_info)
        
        elif self.fallback_policy == 'best_available':
            return self._best_available_fallback(edge, gw_couplings, selection_info)
        
        else:
            selection_info['error'] = f'unknown fallback policy: {self.fallback_policy}'
            return None, selection_info
    
    def _nearest_neighbor_fallback(self, 
                                 edge: Tuple[str, str],
                                 gw_couplings: Dict[Tuple[str, str], torch.Tensor],
                                 selection_info: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Use nearest neighbor fallback policy.
        
        Tries edges from the nearest neighbor mapping in order of preference.
        """
        neighbors = self.nearest_neighbor_mapping.get(edge, [])
        
        if not neighbors:
            selection_info['error'] = 'no neighbors available for fallback'
            logger.warning(f"No nearest neighbors available for fallback for edge {edge}")
            return None, selection_info
        
        # Try each neighbor in order
        for attempt, neighbor_edge in enumerate(neighbors[:self.max_fallback_attempts]):
            selection_info['fallback_attempts'] = attempt + 1
            
            if neighbor_edge not in gw_couplings:
                logger.debug(f"Neighbor {neighbor_edge} not in couplings, trying next")
                continue
            
            neighbor_info = self.edge_transport_info.get(neighbor_edge)
            if not neighbor_info:
                logger.debug(f"No transport info for neighbor {neighbor_edge}, trying next")
                continue
            
            if (neighbor_info['availability_status'] == 'available' and 
                neighbor_info['has_valid_coupling']):
                
                quality = neighbor_info['quality_score']
                
                if quality >= self.quality_threshold:
                    # Found suitable fallback
                    selection_info.update({
                        'selected_edge': neighbor_edge,
                        'selection_method': 'nearest_neighbor_fallback',
                        'quality_score': quality,
                        'fallback_used': True
                    })
                    
                    transport_matrix = gw_couplings[neighbor_edge]
                    logger.info(f"✓ Fallback transport matrix for {edge} → {neighbor_edge}: "
                              f"shape {transport_matrix.shape}, quality={quality:.4f}")
                    return transport_matrix, selection_info
                else:
                    logger.debug(f"Neighbor {neighbor_edge} quality {quality:.4f} below threshold")
        
        selection_info['error'] = f'no suitable neighbors found after {selection_info["fallback_attempts"]} attempts'
        logger.warning(f"No suitable nearest neighbor found for {edge} after {selection_info['fallback_attempts']} attempts")
        return None, selection_info
    
    def _best_available_fallback(self, 
                               edge: Tuple[str, str],
                               gw_couplings: Dict[Tuple[str, str], torch.Tensor],
                               selection_info: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Use best available transport regardless of edge proximity.
        
        Selects the highest quality available transport matrix.
        """
        best_edge = None
        best_quality = -1.0
        
        # Find best quality available transport
        for candidate_edge, coupling in gw_couplings.items():
            candidate_info = self.edge_transport_info.get(candidate_edge)
            if not candidate_info:
                continue
            
            if (candidate_info['availability_status'] == 'available' and 
                candidate_info['has_valid_coupling']):
                
                quality = candidate_info['quality_score']
                
                if quality >= self.quality_threshold and quality > best_quality:
                    best_edge = candidate_edge
                    best_quality = quality
        
        if best_edge is not None:
            selection_info.update({
                'selected_edge': best_edge,
                'selection_method': 'best_available_fallback',
                'quality_score': best_quality,
                'fallback_used': True,
                'fallback_attempts': 1
            })
            
            transport_matrix = gw_couplings[best_edge]
            logger.info(f"✓ Best available transport matrix for {edge} → {best_edge}: "
                      f"shape {transport_matrix.shape}, quality={best_quality:.4f}")
            return transport_matrix, selection_info
        
        selection_info['error'] = 'no transport matrices meet quality threshold'
        logger.warning(f"No available transport matrices meet quality threshold {self.quality_threshold} "
                     f"for fallback to {edge}")
        return None, selection_info
    
    def get_transport_costs(self, 
                          edge: Tuple[str, str],
                          gw_costs: Dict[Tuple[str, str], float],
                          allow_fallback: bool = True) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Get transport costs for edge with fallback support.
        
        Args:
            edge: Target edge
            gw_costs: Dictionary of GW costs by edge
            allow_fallback: Whether to use fallback policy
            
        Returns:
            Tuple of (cost, selection_info)
        """
        # First try to get transport matrix selection info
        _, selection_info = self.get_transport_matrix(edge, {}, allow_fallback=allow_fallback)
        
        selected_edge = selection_info.get('selected_edge')
        if selected_edge and selected_edge in gw_costs:
            cost = gw_costs[selected_edge]
            logger.debug(f"Transport cost for {edge}: {cost:.6f} (via {selected_edge})")
            return cost, selection_info
        
        return None, selection_info
    
    def get_availability_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of transport availability.
        
        Returns:
            Dictionary with availability statistics and diagnostics
        """
        summary = {
            'metadata_available': len(self.transport_metadata) > 0,
            'availability_statistics': self.availability_stats.copy(),
            'fallback_policy': self.fallback_policy,
            'quality_threshold': self.quality_threshold,
            'strict_mode': self.strict_mode,
            'max_fallback_attempts': self.max_fallback_attempts
        }
        
        if self.edge_transport_info:
            # Compute detailed statistics
            total_edges = len(self.edge_transport_info)
            available = sum(1 for info in self.edge_transport_info.values() 
                          if info.get('availability_status') == 'available')
            failed = sum(1 for info in self.edge_transport_info.values() 
                        if info.get('availability_status') == 'failed')
            low_quality = sum(1 for info in self.edge_transport_info.values() 
                            if info.get('availability_status') == 'low_quality')
            
            quality_scores = [info.get('quality_score', 0.0) for info in self.edge_transport_info.values() 
                            if info.get('has_valid_coupling', False)]
            
            summary.update({
                'edge_counts': {
                    'total': total_edges,
                    'available': available,
                    'failed': failed,
                    'low_quality': low_quality
                },
                'quality_statistics': {
                    'min_quality': min(quality_scores) if quality_scores else 0.0,
                    'max_quality': max(quality_scores) if quality_scores else 0.0,
                    'mean_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                    'above_threshold': sum(1 for q in quality_scores if q >= self.quality_threshold)
                },
                'fallback_coverage': {
                    'edges_with_neighbors': sum(1 for edge in self.edge_transport_info.keys() 
                                              if edge in self.nearest_neighbor_mapping and 
                                              len(self.nearest_neighbor_mapping[edge]) > 0),
                    'total_neighbor_mappings': len(self.nearest_neighbor_mapping)
                }
            })
        
        return summary
    
    def validate_transport_metadata(self) -> Dict[str, Any]:
        """
        Validate the transport metadata for completeness and consistency.
        
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check if metadata exists
        if not self.transport_metadata:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No transport metadata available")
            return validation_results
        
        # Check required fields
        required_fields = ['edge_transport_info', 'availability_statistics', 'nearest_neighbor_mapping']
        for field in required_fields:
            if field not in self.transport_metadata:
                validation_results['warnings'].append(f"Missing field: {field}")
        
        # Check availability statistics
        if self.availability_stats:
            availability_rate = self.availability_stats.get('availability_rate', 0.0)
            if availability_rate < 0.5:
                validation_results['warnings'].append(
                    f"Low transport availability: {availability_rate:.1%} of edges have transport"
                )
                validation_results['recommendations'].append(
                    "Consider using more lenient GW quality thresholds or fallback policies"
                )
            
            failed_count = self.availability_stats.get('failed_count', 0)
            total_edges = self.availability_stats.get('total_edges', 1)
            if failed_count / total_edges > 0.2:
                validation_results['warnings'].append(
                    f"High failure rate: {failed_count}/{total_edges} edges failed GW computation"
                )
        
        # Check quality threshold vs available quality
        if self.edge_transport_info:
            available_qualities = [
                info['quality_score'] for info in self.edge_transport_info.values()
                if info['availability_status'] == 'available'
            ]
            
            if available_qualities:
                max_available_quality = max(available_qualities)
                if self.quality_threshold > max_available_quality:
                    validation_results['warnings'].append(
                        f"Quality threshold {self.quality_threshold} exceeds maximum available "
                        f"quality {max_available_quality:.4f}"
                    )
                    validation_results['recommendations'].append(
                        f"Consider lowering quality threshold to {max_available_quality * 0.9:.3f}"
                    )
        
        # Check fallback coverage
        if self.fallback_policy != 'strict':
            edges_without_fallback = sum(
                1 for edge in self.edge_transport_info.keys()
                if edge not in self.nearest_neighbor_mapping or 
                len(self.nearest_neighbor_mapping[edge]) == 0
            )
            
            if edges_without_fallback > 0:
                validation_results['warnings'].append(
                    f"{edges_without_fallback} edges have no fallback options"
                )
        
        logger.info(f"Transport metadata validation: {'✓ PASSED' if validation_results['is_valid'] else '✗ FAILED'}, "
                   f"{len(validation_results['warnings'])} warnings, {len(validation_results['errors'])} errors")
        
        return validation_results
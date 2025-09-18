"""Sheaf property validation for mathematical correctness.

This module validates the mathematical properties required for a valid sheaf:
- Transitivity: R_AC = R_BC @ R_AB for all paths A → B → C  
- Consistency: All stalks and restrictions are well-defined
- Numerical stability: Restriction maps have reasonable condition numbers

The validation functions ensure that constructed sheaves satisfy the
mathematical requirements for topological data analysis.
"""

from typing import Dict, Tuple, Any, List

import torch

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)


def validate_sheaf_properties(
    restrictions: Dict[Tuple[str, str], torch.Tensor],
    poset: 'nx.DiGraph', 
    tolerance: float = 1e-2
) -> Dict[str, Any]:
    """Validate mathematical properties required for a valid sheaf.
    
    This function checks the transitivity property: R_AC = R_BC @ R_AB
    for all valid paths in the poset.
    
    Args:
        restrictions: Dictionary mapping edges to restriction maps
        poset: NetworkX directed graph representing the poset structure
        tolerance: Tolerance for approximate equality
        
    Returns:
        Dictionary with validation results
    """
    import networkx as nx
    
    validation_results = {
        'transitivity_violations': [],
        'max_violation': 0.0,
        'total_paths_checked': 0,
        'valid_sheaf': True
    }
    
    # Check transitivity for all paths of length 2
    for node_a in poset.nodes():
        for node_b in poset.successors(node_a):
            for node_c in poset.successors(node_b):
                # We have path A → B → C
                edge_ab = (node_a, node_b)
                edge_bc = (node_b, node_c)
                edge_ac = (node_a, node_c) if poset.has_edge(node_a, node_c) else None
                
                if edge_ab in restrictions and edge_bc in restrictions:
                    R_ab = restrictions[edge_ab]
                    R_bc = restrictions[edge_bc]
                    R_composed = R_bc @ R_ab
                    
                    validation_results['total_paths_checked'] += 1
                    
                    if edge_ac and edge_ac in restrictions:
                        # Direct path exists, check transitivity
                        R_ac = restrictions[edge_ac]
                        violation = torch.norm(R_composed - R_ac, p='fro').item()
                        
                        if violation > tolerance:
                            validation_results['transitivity_violations'].append({
                                'path': (node_a, node_b, node_c),
                                'violation': violation,
                                'relative_violation': violation / (torch.norm(R_ac, p='fro').item() + 1e-8)
                            })
                            validation_results['valid_sheaf'] = False
                        
                        validation_results['max_violation'] = max(validation_results['max_violation'], violation)
    
    logger.info(f"Sheaf validation: {validation_results['total_paths_checked']} paths checked, "
                f"{len(validation_results['transitivity_violations'])} violations found, "
                f"max violation: {validation_results['max_violation']:.6f}")
    
    return validation_results


def validate_restriction_orthogonality(
    restrictions: Dict[Tuple[str, str], torch.Tensor],
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """Validate orthogonality properties of restriction maps.
    
    Checks that restriction maps satisfy appropriate orthogonality conditions:
    - Column orthonormal when r_source ≤ r_target (R^T R = I)
    - Row orthonormal when r_source > r_target (R R^T = I)
    
    Args:
        restrictions: Dictionary mapping edges to restriction maps
        tolerance: Tolerance for orthogonality errors
        
    Returns:
        Dictionary with orthogonality validation results
    """
    orthogonality_results = {
        'total_restrictions': len(restrictions),
        'orthogonality_violations': [],
        'max_orthogonality_error': 0.0,
        'all_orthogonal': True
    }
    
    for edge, R in restrictions.items():
        r_target, r_source = R.shape
        
        if r_source <= r_target:
            # Check column orthogonality: R^T R = I
            RtR = R.T @ R
            identity = torch.eye(r_source, device=R.device)
            orth_error = torch.norm(RtR - identity, p='fro').item()
            orth_type = 'column'
        else:
            # Check row orthogonality: R R^T = I
            RRt = R @ R.T
            identity = torch.eye(r_target, device=R.device)
            orth_error = torch.norm(RRt - identity, p='fro').item()
            orth_type = 'row'
        
        orthogonality_results['max_orthogonality_error'] = max(
            orthogonality_results['max_orthogonality_error'], orth_error
        )
        
        if orth_error > tolerance:
            orthogonality_results['orthogonality_violations'].append({
                'edge': edge,
                'error': orth_error,
                'tolerance': tolerance,
                'type': orth_type,
                'dimensions': (r_target, r_source)
            })
            orthogonality_results['all_orthogonal'] = False
    
    logger.info(f"Orthogonality validation: {len(orthogonality_results['orthogonality_violations'])} "
                f"violations out of {orthogonality_results['total_restrictions']} restrictions, "
                f"max error: {orthogonality_results['max_orthogonality_error']:.2e}")
    
    return orthogonality_results


def validate_restriction_maps_gw(
    restrictions: Dict[Tuple[str, str], torch.Tensor],
    stochastic_tolerance: float = 1e-6,
    correction_threshold: float = 1e-3,
    strict_threshold: float = 0.1,
    strict_mode: bool = False,
    auto_correct: bool = True,
    check_stochasticity: bool = True
) -> Tuple[Dict[Tuple[str, str], torch.Tensor], Dict[str, Any]]:
    """Validate and optionally correct GW-derived restriction maps.
    
    Performs comprehensive validation of restriction maps derived from 
    Gromov-Wasserstein optimal transport, including finiteness checks,
    row-stochasticity validation, and automated correction of small violations.
    
    Args:
        restrictions: Dictionary mapping edges to restriction maps
        stochastic_tolerance: Tolerance for row-stochasticity check (default: 1e-6)
        correction_threshold: Threshold below which violations are auto-corrected (default: 1e-3)
        strict_threshold: Threshold above which strict mode errors are raised (default: 0.1)
        strict_mode: Whether to raise errors for large violations instead of warnings
        auto_correct: Whether to automatically correct small violations
        check_stochasticity: Whether to validate row-stochasticity (GW-specific)
        
    Returns:
        Tuple of (validated_restrictions, validation_metadata)
        where validated_restrictions may contain corrected maps
        and validation_metadata contains detailed per-edge validation results
        
    Raises:
        ValueError: If finiteness check fails or strict_mode violations occur
    """
    validated_restrictions = {}
    validation_metadata = {
        'total_restrictions': len(restrictions),
        'finite_violations': [],
        'stochasticity_violations': [],
        'corrections_applied': [],
        'dropped_edges': [],
        'max_stochasticity_error': 0.0,
        'validation_passed': True,
        'config': {
            'stochastic_tolerance': stochastic_tolerance,
            'correction_threshold': correction_threshold,
            'strict_threshold': strict_threshold,
            'strict_mode': strict_mode,
            'auto_correct': auto_correct,
            'check_stochasticity': check_stochasticity
        }
    }
    
    logger.info(f"Validating {len(restrictions)} GW-derived restriction maps...")
    logger.debug(f"Config: stochastic_tol={stochastic_tolerance:.2e}, "
                f"correction_thresh={correction_threshold:.3f}, auto_correct={auto_correct}")
    
    for edge, R in restrictions.items():
        edge_validation = {
            'edge': edge,
            'original_shape': R.shape,
            'finite_check_passed': True,
            'stochasticity_error': 0.0,
            'correction_applied': False,
            'action_taken': 'passed'
        }
        
        try:
            # Step 1: Finiteness check
            if not torch.isfinite(R).all():
                edge_validation['finite_check_passed'] = False
                edge_validation['action_taken'] = 'dropped_finite_violation'
                validation_metadata['finite_violations'].append(edge)
                validation_metadata['dropped_edges'].append(edge)  # Add to dropped edges
                validation_metadata['validation_passed'] = False
                
                error_msg = f"Restriction map for edge {edge} contains NaN or Inf values"
                if strict_mode:
                    raise ValueError(error_msg)
                else:
                    logger.error(error_msg + " - dropping edge")
                    continue
            
            # Step 2: Row-stochasticity validation (GW-specific)
            if check_stochasticity:
                stochasticity_error = _validate_row_stochasticity(R)
                edge_validation['stochasticity_error'] = stochasticity_error
                validation_metadata['max_stochasticity_error'] = max(
                    validation_metadata['max_stochasticity_error'], stochasticity_error
                )
                
                if stochasticity_error > stochastic_tolerance:
                    violation_info = {
                        'edge': edge,
                        'error': stochasticity_error,
                        'tolerance': stochastic_tolerance,
                        'row_sums_range': (R.sum(dim=1).min().item(), R.sum(dim=1).max().item()),
                        'corrected': False
                    }
                    
                    # Decide on correction vs error based on violation severity
                    if stochasticity_error <= correction_threshold and auto_correct:
                        # Apply automatic correction
                        R_corrected = _correct_stochasticity_violations(R)
                        corrected_error = _validate_row_stochasticity(R_corrected)
                        
                        violation_info.update({
                            'corrected': True,
                            'corrected_error': corrected_error,
                            'corrected_row_sums_range': (
                                R_corrected.sum(dim=1).min().item(), 
                                R_corrected.sum(dim=1).max().item()
                            )
                        })
                        
                        validated_restrictions[edge] = R_corrected
                        validation_metadata['corrections_applied'].append(violation_info)
                        edge_validation['correction_applied'] = True
                        edge_validation['action_taken'] = 'corrected'
                        
                        logger.debug(f"Auto-corrected stochasticity violation for {edge}: "
                                   f"{stochasticity_error:.6f} → {corrected_error:.6f}")
                        
                    elif stochasticity_error > strict_threshold:
                        # Large violation - handle based on strict mode
                        edge_validation['action_taken'] = 'dropped_stochastic_violation'
                        validation_metadata['dropped_edges'].append(edge)
                        validation_metadata['validation_passed'] = False
                        
                        error_msg = (f"Large stochasticity violation for edge {edge}: "
                                   f"error={stochasticity_error:.6f} > threshold={strict_threshold}")
                        
                        if strict_mode:
                            raise ValueError(error_msg)
                        else:
                            logger.warning(error_msg + " - dropping edge")
                            continue
                    else:
                        # Medium violation - warn but keep
                        validated_restrictions[edge] = R
                        edge_validation['action_taken'] = 'kept_with_warning'
                        logger.warning(f"Stochasticity violation for {edge}: "
                                     f"error={stochasticity_error:.6f} (no correction applied)")
                    
                    validation_metadata['stochasticity_violations'].append(violation_info)
                else:
                    # Stochasticity check passed
                    validated_restrictions[edge] = R
            else:
                # Stochasticity check disabled - just copy
                validated_restrictions[edge] = R
            
        except Exception as e:
            edge_validation['action_taken'] = 'dropped_validation_error'
            validation_metadata['dropped_edges'].append(edge)
            validation_metadata['validation_passed'] = False
            
            error_msg = f"Validation failed for edge {edge}: {e}"
            if strict_mode:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg + " - dropping edge")
                continue
    
    # Final validation summary
    n_original = len(restrictions)
    n_validated = len(validated_restrictions)
    n_corrected = len(validation_metadata['corrections_applied'])
    n_dropped = len(validation_metadata['dropped_edges'])
    
    logger.info(f"GW restriction validation complete: {n_validated}/{n_original} maps passed, "
               f"{n_corrected} corrected, {n_dropped} dropped")
    
    if validation_metadata['max_stochasticity_error'] > stochastic_tolerance:
        logger.warning(f"Maximum stochasticity error: {validation_metadata['max_stochasticity_error']:.6f}")
    
    return validated_restrictions, validation_metadata


def _validate_row_stochasticity(R: torch.Tensor, eps: float = 1e-12) -> float:
    """Check row-stochasticity of restriction map.
    
    For GW-derived restrictions, each row should sum to 1.0.
    
    Args:
        R: Restriction map tensor [target_dim x source_dim]
        eps: Small epsilon for numerical stability
        
    Returns:
        Maximum deviation from row-stochastic property
    """
    row_sums = R.sum(dim=1)
    expected_ones = torch.ones_like(row_sums)
    stochasticity_error = torch.abs(row_sums - expected_ones).max().item()
    return stochasticity_error


def _correct_stochasticity_violations(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Correct small stochasticity violations in restriction maps.
    
    Applies two-step correction:
    1. Clip negative values to zero
    2. Renormalize rows to sum to 1.0
    
    Args:
        R: Restriction map tensor to correct
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Corrected restriction map with proper row-stochastic property
    """
    # Step 1: Clip negative values (common from imperfect OT)
    R_clipped = torch.clamp(R, min=0.0)
    
    # Step 2: Renormalize rows to sum to 1.0
    row_sums = R_clipped.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=eps)  # Avoid division by zero
    R_corrected = R_clipped / row_sums
    
    return R_corrected
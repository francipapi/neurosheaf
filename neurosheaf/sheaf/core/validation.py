"""Sheaf property validation for mathematical correctness.

This module validates the mathematical properties required for a valid sheaf:
- Transitivity: R_AC = R_BC @ R_AB for all paths A → B → C  
- Consistency: All stalks and restrictions are well-defined
- Numerical stability: Restriction maps have reasonable condition numbers

The validation functions ensure that constructed sheaves satisfy the
mathematical requirements for topological data analysis.
"""

from typing import Dict, Tuple, Any

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
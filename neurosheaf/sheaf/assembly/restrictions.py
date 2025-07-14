"""Restriction map computation for sheaf assembly.

This module provides utilities for computing restriction maps between
different layers in whitened coordinate space using scaled Procrustes
analysis for optimal mathematical properties.
"""

from typing import Dict, Tuple, List, Optional
import torch
import networkx as nx

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)

from ..core import scaled_procrustes_whitened, WhiteningProcessor


class RestrictionError(Exception):
    """Exception raised during restriction map computation."""
    pass


class RestrictionManager:
    """Manages computation of restriction maps in whitened coordinates.
    
    This class handles the computation of restriction maps between layers
    using the scaled Procrustes method in whitened coordinate space for
    optimal mathematical properties.
    """
    
    def __init__(self):
        """Initialize the restriction manager."""
        self.whitening_processor = WhiteningProcessor()
    
    def compute_all_restrictions(self, 
                               gram_matrices: Dict[str, torch.Tensor],
                               poset: nx.DiGraph,
                               validate: bool = True) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute all restriction maps for a poset.
        
        Args:
            gram_matrices: Dictionary of original Gram matrices
            poset: Network structure as directed graph
            validate: Whether to validate restriction orthogonality
            
        Returns:
            Dictionary mapping edges to restriction tensors
            
        Raises:
            RestrictionError: If restriction computation fails
        """
        logger.info(f"Computing restrictions for {len(poset.edges())} edges")
        
        restrictions = {}
        failed_edges = []
        
        for source, target in poset.edges():
            if source in gram_matrices and target in gram_matrices:
                try:
                    restriction = self.compute_restriction(
                        gram_matrices[source],
                        gram_matrices[target],
                        validate=validate
                    )
                    restrictions[(source, target)] = restriction
                    logger.debug(f"Computed restriction {source} → {target}: {restriction.shape}")
                    
                except Exception as e:
                    logger.warning(f"Failed to compute restriction {source} → {target}: {e}")
                    failed_edges.append((source, target))
                    continue
            else:
                logger.warning(f"Missing Gram matrices for edge {source} → {target}")
                failed_edges.append((source, target))
        
        if failed_edges:
            logger.warning(f"Failed to compute {len(failed_edges)} restrictions")
        
        logger.info(f"Successfully computed {len(restrictions)} restriction maps")
        return restrictions
    
    def compute_restriction(self,
                          K_source: torch.Tensor,
                          K_target: torch.Tensor,
                          validate: bool = True) -> torch.Tensor:
        """Compute restriction map between two Gram matrices.
        
        Args:
            K_source: Source Gram matrix
            K_target: Target Gram matrix  
            validate: Whether to validate orthogonality
            
        Returns:
            Restriction map tensor in whitened coordinates
            
        Raises:
            RestrictionError: If computation fails
        """
        try:
            # Use scaled Procrustes in whitened space
            restriction, scale, info = scaled_procrustes_whitened(
                K_source, 
                K_target, 
                validate=validate
            )
            return restriction
            
        except Exception as e:
            raise RestrictionError(f"Restriction computation failed: {e}")
    
    def validate_restriction_properties(self,
                                      restrictions: Dict[Tuple[str, str], torch.Tensor],
                                      poset: nx.DiGraph,
                                      tolerance: float = 1e-5) -> Dict[str, any]:
        """Validate mathematical properties of restriction maps.
        
        Args:
            restrictions: Dictionary of restriction maps
            poset: Network structure
            tolerance: Numerical tolerance for validation
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating restriction map properties")
        
        results = {
            'valid': True,
            'orthogonality_errors': [],
            'transitivity_errors': [],
            'max_orthogonality_error': 0.0,
            'max_transitivity_error': 0.0
        }
        
        # Check orthogonality properties
        for (source, target), R in restrictions.items():
            try:
                orth_error = self._check_orthogonality(R, tolerance)
                if orth_error > tolerance:
                    results['orthogonality_errors'].append({
                        'edge': (source, target),
                        'error': orth_error
                    })
                    results['valid'] = False
                results['max_orthogonality_error'] = max(
                    results['max_orthogonality_error'], orth_error
                )
            except Exception as e:
                logger.warning(f"Orthogonality check failed for {source} → {target}: {e}")
        
        # Check transitivity where applicable
        trans_errors = self._check_transitivity(restrictions, poset, tolerance)
        if trans_errors:
            results['transitivity_errors'] = trans_errors
            results['max_transitivity_error'] = max(err['error'] for err in trans_errors)
            results['valid'] = False
        
        logger.info(f"Validation complete: {'PASS' if results['valid'] else 'FAIL'}")
        return results
    
    def _check_orthogonality(self, R: torch.Tensor, tolerance: float) -> float:
        """Check orthogonality property of restriction map.
        
        Args:
            R: Restriction map tensor
            tolerance: Numerical tolerance
            
        Returns:
            Maximum orthogonality error
        """
        r_s, r_t = R.shape
        
        if r_s <= r_t:
            # Column orthonormal case: R^T R = I
            RTR = R.T @ R
            I = torch.eye(r_s, device=R.device, dtype=R.dtype)
            error = torch.norm(RTR - I).item()
        else:
            # Row orthonormal case: R R^T = I  
            RRT = R @ R.T
            I = torch.eye(r_t, device=R.device, dtype=R.dtype)
            error = torch.norm(RRT - I).item()
        
        return error
    
    def _check_transitivity(self,
                           restrictions: Dict[Tuple[str, str], torch.Tensor],
                           poset: nx.DiGraph,
                           tolerance: float) -> List[Dict]:
        """Check transitivity property: R_AC = R_BC @ R_AB.
        
        Args:
            restrictions: Dictionary of restriction maps
            poset: Network structure
            tolerance: Numerical tolerance
            
        Returns:
            List of transitivity violations
        """
        errors = []
        
        # Find all paths of length 2
        for A in poset.nodes():
            for B in poset.successors(A):
                for C in poset.successors(B):
                    # Check if all required restrictions exist
                    if ((A, B) in restrictions and 
                        (B, C) in restrictions and 
                        (A, C) in restrictions):
                        
                        R_AB = restrictions[(A, B)]
                        R_BC = restrictions[(B, C)]  
                        R_AC = restrictions[(A, C)]
                        
                        # Check R_AC = R_BC @ R_AB
                        try:
                            expected = R_BC @ R_AB
                            error = torch.norm(R_AC - expected).item()
                            
                            if error > tolerance:
                                errors.append({
                                    'path': (A, B, C),
                                    'error': error,
                                    'expected_norm': torch.norm(expected).item(),
                                    'actual_norm': torch.norm(R_AC).item()
                                })
                        except Exception as e:
                            logger.warning(f"Transitivity check failed for {A}→{B}→{C}: {e}")
        
        return errors


def compute_restrictions_for_sheaf(gram_matrices: Dict[str, torch.Tensor],
                                 poset: nx.DiGraph,
                                 validate: bool = True) -> Dict[Tuple[str, str], torch.Tensor]:
    """Convenience function to compute all restrictions for sheaf construction.
    
    Args:
        gram_matrices: Dictionary of Gram matrices
        poset: Network structure  
        validate: Whether to validate restriction properties
        
    Returns:
        Dictionary of restriction maps
    """
    manager = RestrictionManager()
    return manager.compute_all_restrictions(gram_matrices, poset, validate)
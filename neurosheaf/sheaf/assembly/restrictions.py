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

from ..core import scaled_procrustes_whitened, scaled_procrustes_adaptive, WhiteningProcessor


class RestrictionError(Exception):
    """Exception raised during restriction map computation."""
    pass


class RestrictionManager:
    """Manages computation of restriction maps in whitened coordinates.
    
    This class handles the computation of restriction maps between layers
    using the scaled Procrustes method in whitened coordinate space for
    optimal mathematical properties.
    """
    
    def __init__(self, use_double_precision: bool = False, batch_size: int = None):
        """Initialize the restriction manager.
        
        Args:
            use_double_precision: Whether to use double precision for computations
            batch_size: Hint for adaptive precision selection
        """
        self.use_double_precision = use_double_precision
        self.batch_size = batch_size
        self.whitening_processor = WhiteningProcessor(use_double_precision=use_double_precision)
    
    def compute_all_restrictions(self, 
                               gram_matrices: Dict[str, torch.Tensor],
                               poset: nx.DiGraph,
                               validate: bool = True,
                               regularization_info: Optional[Dict[str, Dict]] = None) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute all restriction maps for a poset.
        
        Args:
            gram_matrices: Dictionary of Gram matrices (possibly regularized)
            poset: Network structure as directed graph
            validate: Whether to validate restriction orthogonality
            regularization_info: Optional regularization metadata for adaptive handling
            
        Returns:
            Dictionary mapping edges to restriction tensors
            
        Raises:
            RestrictionError: If restriction computation fails
        """
        logger.info(f"Computing restrictions for {len(poset.edges())} edges")
        
        # Check if any matrices were regularized
        regularized_layers = set()
        if regularization_info:
            for layer_name, reg_info in regularization_info.items():
                if reg_info.get('regularized', False):
                    regularized_layers.add(layer_name)
                    logger.debug(f"Layer {layer_name} was regularized with λ={reg_info.get('regularization_strength', 'N/A'):.2e}")
        
        if regularized_layers:
            logger.info(f"Processing {len(regularized_layers)} regularized layers")
        
        restrictions = {}
        failed_edges = []
        
        for source, target in poset.edges():
            if source in gram_matrices and target in gram_matrices:
                try:
                    # Get regularization info for source and target layers
                    source_reg_info = regularization_info.get(source, {}) if regularization_info else {}
                    target_reg_info = regularization_info.get(target, {}) if regularization_info else {}
                    
                    restriction = self.compute_restriction(
                        gram_matrices[source],
                        gram_matrices[target],
                        validate=validate,
                        source_regularization_info=source_reg_info,
                        target_regularization_info=target_reg_info
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
    
    def compute_all_restrictions_with_whitening(self, 
                                               gram_matrices: Dict[str, torch.Tensor],
                                               whitening_info: Dict[str, Dict],
                                               poset: nx.DiGraph,
                                               validate: bool = True,
                                               regularization_info: Optional[Dict[str, Dict]] = None) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute all restriction maps using pre-computed whitening information.
        
        This method ensures dimensional consistency between stalks and restrictions by
        using the same whitening transformations for both.
        
        Args:
            gram_matrices: Dictionary of original Gram matrices
            whitening_info: Pre-computed whitening information for each node
            poset: Network structure as directed graph
            validate: Whether to validate restriction orthogonality
            regularization_info: Optional regularization metadata for adaptive handling
            
        Returns:
            Dictionary mapping edges to restriction tensors with correct dimensions
            
        Raises:
            RestrictionError: If restriction computation fails
        """
        logger.info(f"Computing restrictions with pre-computed whitening for {len(poset.edges())} edges")
        
        # Check if any matrices were regularized
        regularized_layers = set()
        if regularization_info:
            for layer_name, reg_info in regularization_info.items():
                if reg_info.get('regularized', False):
                    regularized_layers.add(layer_name)
                    logger.debug(f"Layer {layer_name} was regularized with λ={reg_info.get('regularization_strength', 'N/A'):.2e}")
        
        if regularized_layers:
            logger.info(f"Processing {len(regularized_layers)} regularized layers")
        
        restrictions = {}
        failed_edges = []
        
        for source, target in poset.edges():
            if source in gram_matrices and target in gram_matrices and \
               source in whitening_info and target in whitening_info:
                try:
                    # Get pre-computed whitening information
                    source_whitening = whitening_info[source]
                    target_whitening = whitening_info[target]
                    
                    source_rank = source_whitening['rank']
                    target_rank = target_whitening['rank']
                    
                    logger.debug(f"Computing restriction {source} → {target}: ({target_rank}, {source_rank})")
                    
                    # Get regularization info for source and target layers
                    source_reg_info = regularization_info.get(source, {}) if regularization_info else {}
                    target_reg_info = regularization_info.get(target, {}) if regularization_info else {}
                    
                    # Compute restriction using the consistent whitening processor
                    restriction = self.compute_restriction_with_whitening(
                        gram_matrices[source],
                        gram_matrices[target],
                        source_whitening,
                        target_whitening,
                        validate=validate,
                        source_regularization_info=source_reg_info,
                        target_regularization_info=target_reg_info
                    )
                    
                    # Verify dimensions match expected
                    expected_shape = (target_rank, source_rank)
                    if restriction.shape != expected_shape:
                        logger.error(f"Dimension mismatch for {source} → {target}: "
                                   f"expected {expected_shape}, got {restriction.shape}")
                        failed_edges.append((source, target))
                        continue
                    
                    restrictions[(source, target)] = restriction
                    logger.debug(f"✓ Computed restriction {source} → {target}: {restriction.shape}")
                    
                except Exception as e:
                    logger.warning(f"Failed to compute restriction {source} → {target}: {e}")
                    failed_edges.append((source, target))
                    continue
            else:
                missing_items = []
                if source not in gram_matrices: missing_items.append(f"gram_matrix[{source}]")
                if target not in gram_matrices: missing_items.append(f"gram_matrix[{target}]")
                if source not in whitening_info: missing_items.append(f"whitening_info[{source}]")
                if target not in whitening_info: missing_items.append(f"whitening_info[{target}]")
                logger.warning(f"Missing data for edge {source} → {target}: {', '.join(missing_items)}")
                failed_edges.append((source, target))
        
        if failed_edges:
            logger.warning(f"Failed to compute {len(failed_edges)} restrictions")
        
        logger.info(f"Successfully computed {len(restrictions)} restriction maps with consistent dimensions")
        return restrictions
    
    def compute_restriction_with_whitening(self,
                                          K_source: torch.Tensor,
                                          K_target: torch.Tensor,
                                          source_whitening: Dict,
                                          target_whitening: Dict,
                                          validate: bool = True,
                                          source_regularization_info: Optional[Dict] = None,
                                          target_regularization_info: Optional[Dict] = None) -> torch.Tensor:
        """Compute restriction map using pre-computed whitening information.
        
        This method ensures dimensional consistency by using the same whitening
        transformations that were used to create the stalks.
        
        Args:
            K_source: Source Gram matrix
            K_target: Target Gram matrix
            source_whitening: Pre-computed whitening info for source
            target_whitening: Pre-computed whitening info for target
            validate: Whether to validate orthogonality
            source_regularization_info: Regularization metadata for source matrix
            target_regularization_info: Regularization metadata for target matrix
            
        Returns:
            Restriction map tensor with dimensions (target_rank, source_rank)
            
        Raises:
            RestrictionError: If computation fails
        """
        try:
            # Extract whitening matrices
            W_source = source_whitening['whitening_matrix']
            W_target = target_whitening['whitening_matrix']
            
            source_rank = source_whitening['rank']
            target_rank = target_whitening['rank']
            
            logger.debug(f"Using pre-computed whitening: source_rank={source_rank}, target_rank={target_rank}")
            
            # Compute the cross-covariance matrix in whitened space
            # This is the core computation from scaled_procrustes_whitened
            # Use double precision for better numerical stability
            M = W_target.double() @ W_source.double().T  # (target_rank × source_rank)
            
            # Compute SVD to get optimal restriction map in double precision
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            
            # Optimal restriction map (convert back to float32)
            R_w = (U @ Vh).float()  # (target_rank × source_rank)
            
            # Compute scale (edge weight)
            scale = S.sum().item() / (source_rank + 1e-9)
            
            # Validation (optional)
            if validate:
                eye_source = torch.eye(source_rank, device=R_w.device, dtype=R_w.dtype)
                eye_target = torch.eye(target_rank, device=R_w.device, dtype=R_w.dtype)
                
                col_orth = torch.norm(R_w.T @ R_w - eye_source)
                row_orth = torch.norm(R_w @ R_w.T - eye_target)
                
                orth_tol = 1e-10  # Tighter tolerance for double precision computation
                
                if source_rank <= target_rank:
                    # Column orthonormal case
                    if col_orth > orth_tol:
                        logger.warning(f"Column orthogonality tolerance exceeded: {col_orth:.2e} > {orth_tol:.2e}")
                else:
                    # Row orthonormal case
                    if row_orth > orth_tol:
                        logger.warning(f"Row orthogonality tolerance exceeded: {row_orth:.2e} > {orth_tol:.2e}")
            
            # Verify output dimensions
            expected_shape = (target_rank, source_rank)
            if R_w.shape != expected_shape:
                raise RestrictionError(f"Computed restriction has wrong dimensions: "
                                     f"expected {expected_shape}, got {R_w.shape}")
            
            logger.debug(f"✓ Restriction computed successfully: {R_w.shape}, scale={scale:.4f}")
            return R_w
            
        except Exception as e:
            raise RestrictionError(f"Restriction computation failed: {e}")
    
    def compute_restriction(self,
                          K_source: torch.Tensor,
                          K_target: torch.Tensor,
                          validate: bool = True,
                          source_regularization_info: Optional[Dict] = None,
                          target_regularization_info: Optional[Dict] = None) -> torch.Tensor:
        """Compute restriction map between two Gram matrices.
        
        Args:
            K_source: Source Gram matrix (possibly regularized)
            K_target: Target Gram matrix (possibly regularized)
            validate: Whether to validate orthogonality
            source_regularization_info: Regularization metadata for source matrix
            target_regularization_info: Regularization metadata for target matrix
            
        Returns:
            Restriction map tensor in whitened coordinates
            
        Raises:
            RestrictionError: If computation fails
        """
        try:
            # Check if either matrix was regularized to inform processing
            source_regularized = source_regularization_info.get('regularized', False) if source_regularization_info else False
            target_regularized = target_regularization_info.get('regularized', False) if target_regularization_info else False
            
            # Log regularization status for debugging
            if source_regularized or target_regularized:
                logger.debug(f"Computing restriction with regularized matrices: source={source_regularized}, target={target_regularized}")
            
            # For regularized matrices, use consistent precision but avoid forced double precision
            # to prevent precision mismatch errors in the Procrustes functions
            if source_regularized or target_regularized:
                # Use adaptive precision for regularized matrices (more robust)
                restriction, scale, info = scaled_procrustes_adaptive(
                    K_source, 
                    K_target, 
                    validate=validate,
                    batch_size=self.batch_size or 64  # Ensure batch_size is set for adaptive behavior
                )
                logger.debug(f"Used adaptive precision for regularized matrices")
            elif self.use_double_precision or (self.batch_size and self.batch_size >= 64):
                # Only use forced double precision when not dealing with regularized matrices
                restriction, scale, info = scaled_procrustes_whitened(
                    K_source, 
                    K_target, 
                    validate=validate,
                    use_double_precision=True,
                    whitening_processor=self.whitening_processor
                )
                logger.debug(f"Used double precision for restriction computation")
            else:
                # Use adaptive precision based on condition numbers
                restriction, scale, info = scaled_procrustes_adaptive(
                    K_source, 
                    K_target, 
                    validate=validate,
                    batch_size=self.batch_size
                )
                logger.debug(f"Used adaptive precision for restriction computation")
            
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
                                 validate: bool = True,
                                 regularization_info: Optional[Dict[str, Dict]] = None) -> Dict[Tuple[str, str], torch.Tensor]:
    """Convenience function to compute all restrictions for sheaf construction.
    
    Args:
        gram_matrices: Dictionary of Gram matrices (possibly regularized)
        poset: Network structure  
        validate: Whether to validate restriction properties
        regularization_info: Optional regularization metadata for adaptive handling
        
    Returns:
        Dictionary of restriction maps
    """
    manager = RestrictionManager()
    return manager.compute_all_restrictions(gram_matrices, poset, validate, regularization_info)
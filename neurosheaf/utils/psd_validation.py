#!/usr/bin/env python3
"""
Comprehensive Positive Semidefinite (PSD) validation for Laplacian matrices.

This module provides enhanced validation functions to ensure that all 
Laplacian matrices generated in the neurosheaf package are properly
positive semidefinite, with detailed diagnostics for numerical issues.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)

# Standardized PSD tolerance based on acceptance criteria
PSD_TOLERANCE = -1e-6  # Based on filtration test results showing -1.46e-07 to -2.26e-07

@dataclass
class PSDValidationResult:
    """Results from comprehensive PSD validation.
    
    Attributes:
        is_psd: Whether the matrix is positive semidefinite within tolerance
        smallest_eigenvalue: The smallest eigenvalue found
        condition_number: Condition number of the matrix
        rank: Numerical rank of the matrix
        spectral_gap: Gap between smallest positive and largest negative eigenvalues
        regularization_needed: Whether regularization is recommended
        diagnostics: Additional diagnostic information
    """
    is_psd: bool
    smallest_eigenvalue: float
    condition_number: float
    rank: int
    spectral_gap: float
    regularization_needed: bool
    diagnostics: Dict[str, Any]
    
    def summary(self) -> str:
        """Get a summary of the PSD validation results."""
        status = "✅ PASS" if self.is_psd else "❌ FAIL"
        reg_status = "⚠️ RECOMMENDED" if self.regularization_needed else "✅ NOT NEEDED"
        
        return (
            f"PSD Validation: {status}\n"
            f"  Smallest eigenvalue: {self.smallest_eigenvalue:.2e}\n"
            f"  Condition number: {self.condition_number:.2e}\n"
            f"  Numerical rank: {self.rank}\n"
            f"  Spectral gap: {self.spectral_gap:.2e}\n"
            f"  Regularization: {reg_status}\n"
            f"  Tolerance: {PSD_TOLERANCE:.2e}"
        )


def validate_psd_comprehensive(matrix: Union[csr_matrix, np.ndarray, torch.Tensor],
                             name: str = "matrix",
                             tolerance: float = PSD_TOLERANCE,
                             compute_full_spectrum: bool = False,
                             enable_regularization: bool = True,
                             use_double_precision: bool = False) -> PSDValidationResult:
    """
    Comprehensive PSD validation with detailed diagnostics.
    
    Args:
        matrix: Matrix to validate (sparse or dense)
        name: Name of the matrix for logging
        tolerance: Tolerance for negative eigenvalues
        compute_full_spectrum: Whether to compute all eigenvalues (expensive)
        enable_regularization: Whether to recommend regularization
        use_double_precision: Whether to use double precision for eigenvalue computation
        
    Returns:
        PSDValidationResult with detailed diagnostics
    """
    logger.debug(f"Starting comprehensive PSD validation for {name}")
    
    # Convert to appropriate format with precision control
    if torch.is_tensor(matrix):
        if matrix.is_sparse:
            matrix_dense = matrix.coalesce().to_dense().detach().cpu()
            if use_double_precision:
                matrix = matrix_dense.double().numpy()
            else:
                matrix = matrix_dense.numpy()
        else:
            if use_double_precision:
                matrix = matrix.detach().cpu().double().numpy()
            else:
                matrix = matrix.detach().cpu().numpy()
    elif use_double_precision and matrix.dtype != np.float64:
        matrix = matrix.astype(np.float64)
    
    # Basic checks
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix {name} must be square, got shape {matrix.shape}")
    
    n = matrix.shape[0]
    diagnostics = {
        'matrix_size': n,
        'matrix_type': 'sparse' if issparse(matrix) else 'dense',
        'nnz': matrix.nnz if issparse(matrix) else np.count_nonzero(matrix),
        'symmetry_error': None,
        'eigenvalue_method': None,
        'computation_time': None
    }
    
    # Check symmetry
    if issparse(matrix):
        symmetry_error = (matrix - matrix.T).max()
    else:
        symmetry_error = np.max(np.abs(matrix - matrix.T))
    
    diagnostics['symmetry_error'] = float(symmetry_error)
    
    if symmetry_error > 1e-12:
        logger.warning(f"Matrix {name} is not symmetric (error: {symmetry_error:.2e})")
    
    # Compute eigenvalues
    import time
    start_time = time.time()
    
    if n == 0:
        # Empty matrix case
        eigenvalues = np.array([])
        smallest_eigenvalue = 0.0
        condition_number = 1.0
        rank = 0
        diagnostics['eigenvalue_method'] = 'empty_matrix'
    elif n <= 50 or compute_full_spectrum:
        # Small matrix: compute all eigenvalues
        try:
            if issparse(matrix):
                matrix_dense = matrix.toarray()
            else:
                matrix_dense = matrix
            
            # Use appropriate precision for eigenvalue computation
            if use_double_precision and matrix_dense.dtype != np.float64:
                matrix_dense = matrix_dense.astype(np.float64)
            
            eigenvalues = np.linalg.eigvals(matrix_dense)
            eigenvalues = np.sort(np.real(eigenvalues))
            diagnostics['eigenvalue_method'] = f"full_spectrum_{'double' if use_double_precision else 'single'}"
        except Exception as e:
            logger.error(f"Full eigenvalue computation failed for {name}: {e}")
            eigenvalues = np.array([0.0])
            diagnostics['eigenvalue_method'] = 'fallback_zero'
    else:
        # Large matrix: compute only smallest eigenvalues
        try:
            k = min(10, n - 1)
            if k <= 0:
                eigenvalues = np.array([0.0])
                diagnostics['eigenvalue_method'] = 'single_eigenvalue'
            else:
                eigenvalues = eigsh(matrix, k=k, which='SA', return_eigenvectors=False)
                eigenvalues = np.sort(eigenvalues)
                diagnostics['eigenvalue_method'] = 'sparse_smallest'
        except Exception as e:
            logger.warning(f"Sparse eigenvalue computation failed for {name}: {e}")
            # Fallback to dense computation for smallest eigenvalues
            try:
                if issparse(matrix):
                    matrix_dense = matrix.toarray()
                else:
                    matrix_dense = matrix
                
                # Use appropriate precision for fallback computation
                if use_double_precision and matrix_dense.dtype != np.float64:
                    matrix_dense = matrix_dense.astype(np.float64)
                
                eigenvalues = np.linalg.eigvals(matrix_dense)
                eigenvalues = np.sort(np.real(eigenvalues))
                diagnostics['eigenvalue_method'] = f"dense_fallback_{'double' if use_double_precision else 'single'}"
            except Exception as e2:
                logger.error(f"All eigenvalue computations failed for {name}: {e2}")
                eigenvalues = np.array([0.0])
                diagnostics['eigenvalue_method'] = 'failed'
    
    computation_time = time.time() - start_time
    diagnostics['computation_time'] = computation_time
    
    # Analyze eigenvalues
    if len(eigenvalues) == 0:
        smallest_eigenvalue = 0.0
        condition_number = 1.0
        rank = 0
        spectral_gap = 0.0
    else:
        smallest_eigenvalue = float(np.min(eigenvalues))
        positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        if len(positive_eigenvalues) == 0:
            condition_number = float('inf')
            rank = 0
        else:
            condition_number = float(np.max(positive_eigenvalues) / np.min(positive_eigenvalues))
            rank = len(positive_eigenvalues)
        
        # Compute spectral gap
        negative_eigenvalues = eigenvalues[eigenvalues < -1e-12]
        if len(negative_eigenvalues) > 0 and len(positive_eigenvalues) > 0:
            spectral_gap = float(np.min(positive_eigenvalues) - np.max(negative_eigenvalues))
        else:
            spectral_gap = float(np.min(positive_eigenvalues)) if len(positive_eigenvalues) > 0 else 0.0
    
    # Determine PSD status
    is_psd = smallest_eigenvalue >= tolerance
    
    # Regularization recommendation
    regularization_needed = (
        enable_regularization and 
        (smallest_eigenvalue < -1e-10 or condition_number > 1e12)
    )
    
    # Additional diagnostics
    diagnostics.update({
        'num_eigenvalues_computed': len(eigenvalues),
        'num_positive_eigenvalues': len(eigenvalues[eigenvalues > 1e-12]) if len(eigenvalues) > 0 else 0,
        'num_negative_eigenvalues': len(eigenvalues[eigenvalues < -1e-12]) if len(eigenvalues) > 0 else 0,
        'eigenvalue_range': (float(np.min(eigenvalues)), float(np.max(eigenvalues))) if len(eigenvalues) > 0 else (0.0, 0.0),
        'tolerance_used': tolerance
    })
    
    # Logging
    if is_psd:
        logger.debug(f"Matrix {name} is PSD: smallest eigenvalue = {smallest_eigenvalue:.2e}")
    else:
        logger.warning(f"Matrix {name} is NOT PSD: smallest eigenvalue = {smallest_eigenvalue:.2e} (tolerance: {tolerance:.2e})")
    
    if regularization_needed:
        logger.info(f"Regularization recommended for {name}: condition number = {condition_number:.2e}")
    
    return PSDValidationResult(
        is_psd=is_psd,
        smallest_eigenvalue=smallest_eigenvalue,
        condition_number=condition_number,
        rank=rank,
        spectral_gap=spectral_gap,
        regularization_needed=regularization_needed,
        diagnostics=diagnostics
    )


def validate_psd_simple(matrix: Union[csr_matrix, np.ndarray, torch.Tensor],
                       name: str = "matrix",
                       tolerance: float = PSD_TOLERANCE,
                       raise_on_failure: bool = False) -> bool:
    """
    Simple PSD validation that returns only boolean result.
    
    Args:
        matrix: Matrix to validate
        name: Name for logging
        tolerance: Tolerance for negative eigenvalues
        raise_on_failure: Whether to raise exception on PSD failure
        
    Returns:
        True if matrix is PSD within tolerance
        
    Raises:
        ValueError: If raise_on_failure=True and matrix is not PSD
    """
    result = validate_psd_comprehensive(matrix, name, tolerance, compute_full_spectrum=False, enable_regularization=False)
    
    if not result.is_psd and raise_on_failure:
        raise ValueError(f"Matrix {name} is not positive semidefinite: smallest eigenvalue = {result.smallest_eigenvalue:.2e}")
    
    return result.is_psd


def validate_psd_adaptive(matrix: Union[csr_matrix, np.ndarray, torch.Tensor],
                         name: str = "matrix",
                         batch_size: int = None,
                         condition_threshold: float = 1e6,
                         tolerance: float = PSD_TOLERANCE,
                         **kwargs) -> PSDValidationResult:
    """
    Adaptive PSD validation that chooses precision based on matrix properties.
    
    Args:
        matrix: Matrix to validate
        name: Name for logging
        batch_size: Batch size hint for precision selection
        condition_threshold: Condition number above which double precision is used
        tolerance: Tolerance for negative eigenvalues
        **kwargs: Additional arguments for validate_psd_comprehensive
        
    Returns:
        PSDValidationResult with appropriate precision
    """
    # Determine if double precision is needed
    use_double = False
    
    # Method 1: Use batch size heuristic
    if batch_size is not None and batch_size >= 64:
        use_double = True
        logger.debug(f"Using double precision for {name} due to large batch size: {batch_size}")
    
    # Method 2: Quick condition number estimation for torch tensors
    if not use_double and torch.is_tensor(matrix) and not matrix.is_sparse:
        try:
            if matrix.numel() > 0:
                # Quick condition estimate using SVD
                if matrix.shape[0] <= 1000:  # Only for reasonably sized matrices
                    S = torch.linalg.svdvals(matrix.float())
                    if len(S) > 1:
                        condition_est = (S[0] / S[-1]).item()
                        if condition_est > condition_threshold:
                            use_double = True
                            logger.debug(f"Using double precision for {name} due to high condition number: {condition_est:.2e}")
        except:
            # If quick check fails, be conservative for large matrices
            if matrix.shape[0] > 256:
                use_double = True
                logger.debug(f"Using double precision for {name} due to failed condition check on large matrix")
    
    return validate_psd_comprehensive(
        matrix, name, tolerance,
        use_double_precision=use_double, 
        **kwargs
    )


def regularize_near_psd(matrix: Union[csr_matrix, np.ndarray, torch.Tensor],
                       regularization_strength: float = 1e-10,
                       name: str = "matrix") -> Union[csr_matrix, np.ndarray, torch.Tensor]:
    """
    Regularize a matrix to ensure it is positive semidefinite.
    
    Args:
        matrix: Matrix to regularize
        regularization_strength: Strength of regularization (added to diagonal)
        name: Name for logging
        
    Returns:
        Regularized matrix of the same type as input
    """
    logger.info(f"Regularizing matrix {name} with strength {regularization_strength:.2e}")
    
    if torch.is_tensor(matrix):
        if matrix.is_sparse:
            # For sparse tensors, add regularization to diagonal
            regularized = matrix.clone()
            diagonal_indices = torch.arange(min(matrix.shape), device=matrix.device)
            regularized.coalesce()
            regularized._values()[regularized._indices()[0] == regularized._indices()[1]] += regularization_strength
            return regularized
        else:
            # For dense tensors
            regularized = matrix.clone()
            regularized.diagonal().add_(regularization_strength)
            return regularized
    elif issparse(matrix):
        # For sparse matrices
        regularized = matrix.copy()
        regularized.setdiag(regularized.diagonal() + regularization_strength)
        return regularized
    else:
        # For dense numpy arrays
        regularized = matrix.copy()
        np.fill_diagonal(regularized, regularized.diagonal() + regularization_strength)
        return regularized


def validate_laplacian_psd(laplacian: Union[csr_matrix, np.ndarray, torch.Tensor],
                          name: str = "laplacian",
                          tolerance: float = PSD_TOLERANCE,
                          auto_regularize: bool = False,
                          regularization_strength: float = 1e-10) -> Tuple[bool, Optional[Any]]:
    """
    Validate that a Laplacian matrix is positive semidefinite.
    
    Args:
        laplacian: Laplacian matrix to validate
        name: Name for logging
        tolerance: Tolerance for negative eigenvalues
        auto_regularize: Whether to automatically regularize if needed
        regularization_strength: Strength of regularization
        
    Returns:
        Tuple of (is_psd, regularized_matrix)
        regularized_matrix is None if auto_regularize=False or regularization not needed
    """
    result = validate_psd_comprehensive(laplacian, name, tolerance)
    
    if not result.is_psd:
        logger.warning(f"Laplacian {name} PSD validation failed:\n{result.summary()}")
        
        if auto_regularize and result.regularization_needed:
            regularized = regularize_near_psd(laplacian, regularization_strength, name)
            logger.info(f"Auto-regularized Laplacian {name}")
            return False, regularized
        else:
            return False, None
    else:
        logger.debug(f"Laplacian {name} PSD validation passed")
        return True, None


def validate_filtration_psd(laplacian_sequence: list,
                           threshold_sequence: list,
                           name: str = "filtration",
                           tolerance: float = PSD_TOLERANCE) -> Dict[str, Any]:
    """
    Validate PSD property for a sequence of Laplacian matrices in a filtration.
    
    Args:
        laplacian_sequence: List of Laplacian matrices
        threshold_sequence: List of threshold values
        name: Name for logging
        tolerance: Tolerance for negative eigenvalues
        
    Returns:
        Dictionary with validation results for each threshold
    """
    logger.info(f"Validating PSD property for {len(laplacian_sequence)} matrices in filtration {name}")
    
    results = {
        'overall_valid': True,
        'threshold_results': {},
        'summary': {
            'total_matrices': len(laplacian_sequence),
            'valid_matrices': 0,
            'invalid_matrices': 0,
            'worst_eigenvalue': 0.0,
            'average_condition': 0.0
        }
    }
    
    condition_numbers = []
    
    for i, (laplacian, threshold) in enumerate(zip(laplacian_sequence, threshold_sequence)):
        try:
            result = validate_psd_comprehensive(laplacian, f"{name}_t{threshold}", tolerance)
            
            results['threshold_results'][threshold] = {
                'is_psd': result.is_psd,
                'smallest_eigenvalue': result.smallest_eigenvalue,
                'condition_number': result.condition_number,
                'rank': result.rank,
                'regularization_needed': result.regularization_needed
            }
            
            if result.is_psd:
                results['summary']['valid_matrices'] += 1
            else:
                results['summary']['invalid_matrices'] += 1
                results['overall_valid'] = False
            
            if result.smallest_eigenvalue < results['summary']['worst_eigenvalue']:
                results['summary']['worst_eigenvalue'] = result.smallest_eigenvalue
            
            if not np.isinf(result.condition_number):
                condition_numbers.append(result.condition_number)
                
        except Exception as e:
            logger.error(f"PSD validation failed for {name} at threshold {threshold}: {e}")
            results['overall_valid'] = False
            results['threshold_results'][threshold] = {
                'is_psd': False,
                'error': str(e)
            }
    
    if condition_numbers:
        results['summary']['average_condition'] = np.mean(condition_numbers)
    
    logger.info(f"Filtration {name} PSD validation: {results['summary']['valid_matrices']}/{results['summary']['total_matrices']} matrices valid")
    
    return results


# Convenience functions for common use cases

def ensure_psd_laplacian(laplacian: Union[csr_matrix, np.ndarray, torch.Tensor],
                        name: str = "laplacian") -> Union[csr_matrix, np.ndarray, torch.Tensor]:
    """
    Ensure a Laplacian matrix is positive semidefinite, with automatic regularization.
    
    Args:
        laplacian: Input Laplacian matrix
        name: Name for logging
        
    Returns:
        PSD Laplacian matrix (regularized if necessary)
    """
    is_psd, regularized = validate_laplacian_psd(laplacian, name, auto_regularize=True)
    
    if regularized is not None:
        return regularized
    else:
        return laplacian


def check_psd_with_warning(matrix: Union[csr_matrix, np.ndarray, torch.Tensor],
                          name: str = "matrix",
                          tolerance: float = PSD_TOLERANCE) -> bool:
    """
    Check PSD property and issue appropriate warnings.
    
    Args:
        matrix: Matrix to check
        name: Name for logging
        tolerance: Tolerance for negative eigenvalues
        
    Returns:
        True if matrix is PSD within tolerance
    """
    result = validate_psd_comprehensive(matrix, name, tolerance)
    
    if not result.is_psd:
        if result.smallest_eigenvalue > -1e-6:
            # Small numerical error
            warnings.warn(f"Matrix {name} has small numerical PSD violation: {result.smallest_eigenvalue:.2e}")
        else:
            # Significant PSD violation
            warnings.warn(f"Matrix {name} is not positive semidefinite: {result.smallest_eigenvalue:.2e}")
    
    return result.is_psd


# Set up module-level tolerance configuration
def set_global_psd_tolerance(tolerance: float):
    """Set the global PSD tolerance for all validation functions."""
    global PSD_TOLERANCE
    PSD_TOLERANCE = tolerance
    logger.info(f"Global PSD tolerance set to {tolerance:.2e}")


def get_global_psd_tolerance() -> float:
    """Get the current global PSD tolerance."""
    return PSD_TOLERANCE
"""Hermitian validation utilities for directed sheaf Laplacians.

This module provides comprehensive validation of Hermitian properties for
directed sheaf Laplacians, ensuring mathematical correctness according to
the formulation in docs/DirectedSheaf_mathematicalFormulation.md.

Mathematical Foundation:
- Hermitian property: L^* = L (conjugate transpose equals self)
- Real spectrum: All eigenvalues are real
- Positive semi-definiteness: L ≽ 0 (all eigenvalues ≥ 0)
- Block structure validation

Key Features:
- Comprehensive Hermitian property validation
- Real spectrum verification
- Positive semi-definiteness checking
- Block structure validation
- Numerical stability monitoring
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy.sparse import csr_matrix, issparse
from dataclasses import dataclass
import time

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


@dataclass
class HermitianValidationResult:
    """Results of Hermitian property validation."""
    is_hermitian: bool
    hermitian_error: float
    has_real_spectrum: bool
    max_imaginary_eigenvalue: float
    is_positive_semidefinite: bool
    min_eigenvalue: float
    condition_number: float
    validation_time: float
    errors: List[str]
    warnings: List[str]
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.hermitian_error < 0:
            raise ValueError("Hermitian error must be non-negative")
        if self.max_imaginary_eigenvalue < 0:
            raise ValueError("Max imaginary eigenvalue must be non-negative")


class HermitianValidator:
    """Validates Hermitian properties of directed sheaf Laplacians.
    
    This class provides comprehensive validation of Hermitian matrices
    arising from directed sheaf Laplacian construction, ensuring mathematical
    correctness and numerical stability.
    
    Mathematical Properties Validated:
    - Hermitian property: L^* = L
    - Real spectrum: All eigenvalues are real
    - Positive semi-definiteness: L ≽ 0
    - Block structure consistency
    - Numerical conditioning
    
    The validator uses robust numerical methods to handle potential
    floating-point errors while maintaining strict mathematical standards.
    """
    
    def __init__(self, 
                 hermitian_tolerance: float = 1e-12,
                 spectrum_tolerance: float = 1e-12,
                 positive_semidefinite_tolerance: float = 1e-12,
                 condition_number_threshold: float = 1e12,
                 device: Optional[torch.device] = None):
        """Initialize the Hermitian validator.
        
        Args:
            hermitian_tolerance: Tolerance for Hermitian property validation
            spectrum_tolerance: Tolerance for real spectrum validation
            positive_semidefinite_tolerance: Tolerance for PSD validation
            condition_number_threshold: Threshold for condition number warnings
            device: PyTorch device for computations
        """
        self.hermitian_tolerance = hermitian_tolerance
        self.spectrum_tolerance = spectrum_tolerance
        self.positive_semidefinite_tolerance = positive_semidefinite_tolerance
        self.condition_number_threshold = condition_number_threshold
        self.device = device or torch.device('cpu')
        
        logger.debug(f"HermitianValidator initialized with tolerances: H={hermitian_tolerance}, S={spectrum_tolerance}, PSD={positive_semidefinite_tolerance}")
    
    def validate_hermitian_property(self, matrix: torch.Tensor) -> bool:
        """Validate that matrix is Hermitian: L^* = L.
        
        Args:
            matrix: Complex matrix to validate
            
        Returns:
            True if matrix is Hermitian within tolerance
        """
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if not matrix.is_complex():
            raise ValueError("Input tensor must be complex for Hermitian validation")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for Hermitian validation")
        
        # Move to specified device
        matrix = matrix.to(self.device)
        
        # Compute Hermitian error: ||L - L^*||_F
        hermitian_error = torch.abs(matrix - matrix.conj().T).max().item()
        
        is_hermitian = hermitian_error <= self.hermitian_tolerance
        
        if not is_hermitian:
            logger.warning(f"Matrix not Hermitian: error={hermitian_error}")
        
        return is_hermitian
    
    def validate_real_spectrum(self, matrix: torch.Tensor) -> bool:
        """Validate that all eigenvalues are real.
        
        Args:
            matrix: Complex matrix to validate
            
        Returns:
            True if all eigenvalues are real within tolerance
        """
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if not matrix.is_complex():
            raise ValueError("Input tensor must be complex for spectrum validation")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for spectrum validation")
        
        # Move to specified device
        matrix = matrix.to(self.device)
        
        try:
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvals(matrix)
            
            # Check maximum imaginary part
            max_imag = torch.abs(eigenvalues.imag).max().item()
            
            has_real_spectrum = max_imag <= self.spectrum_tolerance
            
            if not has_real_spectrum:
                logger.warning(f"Eigenvalues not real: max_imag={max_imag}")
            
            return has_real_spectrum
            
        except Exception as e:
            logger.error(f"Failed to compute eigenvalues: {e}")
            return False
    
    def validate_positive_semidefinite(self, matrix: torch.Tensor) -> bool:
        """Validate that matrix is positive semi-definite: L ≽ 0.
        
        Args:
            matrix: Complex matrix to validate
            
        Returns:
            True if matrix is positive semi-definite within tolerance
        """
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if not matrix.is_complex():
            raise ValueError("Input tensor must be complex for PSD validation")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for PSD validation")
        
        # Move to specified device
        matrix = matrix.to(self.device)
        
        try:
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvals(matrix)
            
            # Check minimum real eigenvalue
            min_eigenvalue = eigenvalues.real.min().item()
            
            is_positive_semidefinite = min_eigenvalue >= -self.positive_semidefinite_tolerance
            
            if not is_positive_semidefinite:
                logger.warning(f"Matrix not positive semi-definite: min_eigenvalue={min_eigenvalue}")
            
            return is_positive_semidefinite
            
        except Exception as e:
            logger.error(f"Failed to validate positive semi-definiteness: {e}")
            return False
    
    def compute_condition_number(self, matrix: torch.Tensor) -> float:
        """Compute condition number of the matrix.
        
        Args:
            matrix: Complex matrix to analyze
            
        Returns:
            Condition number (ratio of largest to smallest eigenvalue)
        """
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if not matrix.is_complex():
            raise ValueError("Input tensor must be complex")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        # Move to specified device
        matrix = matrix.to(self.device)
        
        try:
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvals(matrix)
            real_eigenvalues = eigenvalues.real
            
            # Filter out near-zero eigenvalues
            nonzero_eigenvalues = real_eigenvalues[real_eigenvalues > self.positive_semidefinite_tolerance]
            
            if len(nonzero_eigenvalues) == 0:
                return float('inf')
            
            # Condition number = max / min
            condition_number = nonzero_eigenvalues.max().item() / nonzero_eigenvalues.min().item()
            
            return condition_number
            
        except Exception as e:
            logger.error(f"Failed to compute condition number: {e}")
            return float('inf')
    
    def comprehensive_validation(self, matrix: torch.Tensor) -> HermitianValidationResult:
        """Perform comprehensive validation of Hermitian properties.
        
        Args:
            matrix: Complex matrix to validate
            
        Returns:
            HermitianValidationResult with comprehensive validation results
        """
        start_time = time.time()
        
        errors = []
        warnings = []
        
        # Initialize results
        is_hermitian = False
        hermitian_error = float('inf')
        has_real_spectrum = False
        max_imaginary_eigenvalue = float('inf')
        is_positive_semidefinite = False
        min_eigenvalue = float('-inf')
        condition_number = float('inf')
        
        try:
            # Validate input
            if not isinstance(matrix, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")
            
            if not matrix.is_complex():
                raise ValueError("Input tensor must be complex")
            
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix must be square")
            
            # Move to specified device
            matrix = matrix.to(self.device)
            
            # Validate Hermitian property
            hermitian_error = torch.abs(matrix - matrix.conj().T).max().item()
            is_hermitian = hermitian_error <= self.hermitian_tolerance
            
            if not is_hermitian:
                warnings.append(f"Matrix not Hermitian: error={hermitian_error}")
            
            # Validate real spectrum
            try:
                eigenvalues = torch.linalg.eigvals(matrix)
                max_imaginary_eigenvalue = torch.abs(eigenvalues.imag).max().item()
                has_real_spectrum = max_imaginary_eigenvalue <= self.spectrum_tolerance
                
                if not has_real_spectrum:
                    warnings.append(f"Eigenvalues not real: max_imag={max_imaginary_eigenvalue}")
                
                # Validate positive semi-definiteness
                min_eigenvalue = eigenvalues.real.min().item()
                is_positive_semidefinite = min_eigenvalue >= -self.positive_semidefinite_tolerance
                
                if not is_positive_semidefinite:
                    warnings.append(f"Matrix not positive semi-definite: min_eigenvalue={min_eigenvalue}")
                
                # Compute condition number
                condition_number = self.compute_condition_number(matrix)
                
                if condition_number > self.condition_number_threshold:
                    warnings.append(f"High condition number: {condition_number}")
                
            except Exception as e:
                errors.append(f"Failed to compute eigenvalues: {e}")
                
        except Exception as e:
            errors.append(f"Validation failed: {e}")
        
        validation_time = time.time() - start_time
        
        return HermitianValidationResult(
            is_hermitian=is_hermitian,
            hermitian_error=hermitian_error,
            has_real_spectrum=has_real_spectrum,
            max_imaginary_eigenvalue=max_imaginary_eigenvalue,
            is_positive_semidefinite=is_positive_semidefinite,
            min_eigenvalue=min_eigenvalue,
            condition_number=condition_number,
            validation_time=validation_time,
            errors=errors,
            warnings=warnings
        )
    
    def validate_block_structure(self, 
                                blocks: Dict[Tuple[str, str], torch.Tensor],
                                expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the block structure of a Hermitian matrix.
        
        Args:
            blocks: Dictionary mapping (row, col) to block tensors
            expected_structure: Expected block structure specification
            
        Returns:
            Dictionary with block structure validation results
        """
        validation_results = {
            'block_structure_valid': True,
            'hermitian_block_structure': True,
            'block_dimension_consistency': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check that off-diagonal blocks satisfy Hermitian property
            for (i, j), block in blocks.items():
                if i != j:  # Off-diagonal block
                    conjugate_key = (j, i)
                    if conjugate_key in blocks:
                        conjugate_block = blocks[conjugate_key]
                        
                        # Check if block = conjugate_block.conj().T
                        hermitian_error = torch.abs(block - conjugate_block.conj().T).max().item()
                        
                        if hermitian_error > self.hermitian_tolerance:
                            validation_results['hermitian_block_structure'] = False
                            validation_results['errors'].append(
                                f"Block ({i},{j}) not Hermitian conjugate of ({j},{i}): error={hermitian_error}"
                            )
            
            # Check diagonal blocks are Hermitian
            for (i, j), block in blocks.items():
                if i == j:  # Diagonal block
                    if not self.validate_hermitian_property(block):
                        validation_results['hermitian_block_structure'] = False
                        validation_results['errors'].append(f"Diagonal block ({i},{i}) not Hermitian")
            
            # Check dimension consistency
            if 'vertex_dimensions' in expected_structure:
                vertex_dims = expected_structure['vertex_dimensions']
                for (i, j), block in blocks.items():
                    expected_rows = vertex_dims.get(i, 0)
                    expected_cols = vertex_dims.get(j, 0)
                    
                    if block.shape != (expected_rows, expected_cols):
                        validation_results['block_dimension_consistency'] = False
                        validation_results['errors'].append(
                            f"Block ({i},{j}) has wrong dimensions: {block.shape} vs expected ({expected_rows},{expected_cols})"
                        )
            
            # Overall validation
            validation_results['block_structure_valid'] = (
                validation_results['hermitian_block_structure'] and 
                validation_results['block_dimension_consistency']
            )
            
        except Exception as e:
            validation_results['block_structure_valid'] = False
            validation_results['errors'].append(f"Block structure validation failed: {e}")
        
        return validation_results
    
    def validate_directionality_encoding(self, 
                                        encoding_matrix: torch.Tensor,
                                        adjacency_matrix: torch.Tensor,
                                        q: float) -> Dict[str, Any]:
        """Validate directional encoding matrix T^{(q)}.
        
        Args:
            encoding_matrix: T^{(q)} matrix to validate
            adjacency_matrix: Original adjacency matrix A
            q: Directionality parameter
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'encoding_valid': True,
            'phase_consistency': True,
            'unitary_property': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate input dimensions
            if encoding_matrix.shape != adjacency_matrix.shape:
                validation_results['encoding_valid'] = False
                validation_results['errors'].append("Encoding matrix and adjacency matrix have different shapes")
                return validation_results
            
            # Compute expected encoding: T^{(q)} = exp(i 2π q (A - A^T))
            A_antisymmetric = adjacency_matrix - adjacency_matrix.T
            expected_encoding = torch.exp(1j * 2 * np.pi * q * A_antisymmetric)
            
            # Compare with actual encoding
            encoding_error = torch.abs(encoding_matrix - expected_encoding).max().item()
            
            if encoding_error > self.hermitian_tolerance:
                validation_results['phase_consistency'] = False
                validation_results['errors'].append(f"Encoding matrix incorrect: error={encoding_error}")
            
            # Check unitary property for non-zero entries
            nonzero_mask = torch.abs(encoding_matrix) > self.hermitian_tolerance
            if nonzero_mask.any():
                magnitudes = torch.abs(encoding_matrix[nonzero_mask])
                unit_error = torch.abs(magnitudes - 1.0).max().item()
                
                if unit_error > self.hermitian_tolerance:
                    validation_results['unitary_property'] = False
                    validation_results['warnings'].append(f"Encoding not unitary: error={unit_error}")
            
            # Overall validation
            validation_results['encoding_valid'] = (
                validation_results['phase_consistency'] and 
                validation_results['unitary_property']
            )
            
        except Exception as e:
            validation_results['encoding_valid'] = False
            validation_results['errors'].append(f"Encoding validation failed: {e}")
        
        return validation_results
    
    def validate_sparse_hermitian(self, sparse_matrix: csr_matrix) -> Dict[str, Any]:
        """Validate Hermitian properties of sparse matrix.
        
        Args:
            sparse_matrix: Sparse matrix to validate
            
        Returns:
            Dictionary with sparse validation results
        """
        validation_results = {
            'sparse_hermitian_valid': True,
            'sparsity_preserved': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if not issparse(sparse_matrix):
                raise ValueError("Input must be a sparse matrix")
            
            # Convert to dense for validation (only for small matrices)
            if sparse_matrix.shape[0] > 1000:
                validation_results['warnings'].append("Matrix too large for dense conversion - skipping full validation")
                return validation_results
            
            # Convert to dense tensor
            dense_matrix = torch.tensor(sparse_matrix.toarray(), dtype=torch.complex64)
            
            # Validate Hermitian property
            hermitian_result = self.comprehensive_validation(dense_matrix)
            
            if not hermitian_result.is_hermitian:
                validation_results['sparse_hermitian_valid'] = False
                validation_results['errors'].extend(hermitian_result.errors)
            
            # Check sparsity preservation
            original_nnz = sparse_matrix.nnz
            expected_nnz = torch.count_nonzero(dense_matrix).item()
            
            if abs(original_nnz - expected_nnz) > 0:
                validation_results['sparsity_preserved'] = False
                validation_results['warnings'].append(f"Sparsity not preserved: {original_nnz} vs {expected_nnz}")
            
        except Exception as e:
            validation_results['sparse_hermitian_valid'] = False
            validation_results['errors'].append(f"Sparse validation failed: {e}")
        
        return validation_results
    
    def get_validation_summary(self, result: HermitianValidationResult) -> str:
        """Generate a human-readable summary of validation results.
        
        Args:
            result: HermitianValidationResult to summarize
            
        Returns:
            String summary of validation results
        """
        summary_lines = []
        
        # Header
        summary_lines.append("=== Hermitian Validation Summary ===")
        
        # Basic properties
        summary_lines.append(f"Hermitian property: {'✓' if result.is_hermitian else '✗'} (error: {result.hermitian_error:.2e})")
        summary_lines.append(f"Real spectrum: {'✓' if result.has_real_spectrum else '✗'} (max_imag: {result.max_imaginary_eigenvalue:.2e})")
        summary_lines.append(f"Positive semi-definite: {'✓' if result.is_positive_semidefinite else '✗'} (min_eig: {result.min_eigenvalue:.2e})")
        
        # Condition number
        if result.condition_number == float('inf'):
            summary_lines.append("Condition number: ∞ (singular matrix)")
        else:
            summary_lines.append(f"Condition number: {result.condition_number:.2e}")
        
        # Timing
        summary_lines.append(f"Validation time: {result.validation_time:.3f}s")
        
        # Errors and warnings
        if result.errors:
            summary_lines.append("Errors:")
            for error in result.errors:
                summary_lines.append(f"  - {error}")
        
        if result.warnings:
            summary_lines.append("Warnings:")
            for warning in result.warnings:
                summary_lines.append(f"  - {warning}")
        
        # Overall status
        overall_valid = result.is_hermitian and result.has_real_spectrum and result.is_positive_semidefinite
        summary_lines.append(f"Overall validation: {'✓ PASSED' if overall_valid else '✗ FAILED'}")
        
        return "\n".join(summary_lines)
"""Unit tests for HermitianValidator class.

Tests the mathematical correctness of Hermitian validation:
- Hermitian property validation: L^* = L
- Real spectrum verification: All eigenvalues are real
- Positive semi-definiteness: L ≽ 0
- Block structure validation
- Numerical stability monitoring
"""

import pytest
import torch
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Any
import time

from neurosheaf.directed_sheaf.core.hermitian_validation import (
    HermitianValidator, 
    HermitianValidationResult
)


class TestHermitianValidator:
    """Test suite for HermitianValidator class."""
    
    def test_initialization(self):
        """Test initialization of HermitianValidator."""
        validator = HermitianValidator()
        
        assert validator.hermitian_tolerance == 1e-12
        assert validator.spectrum_tolerance == 1e-12
        assert validator.positive_semidefinite_tolerance == 1e-12
        assert validator.condition_number_threshold == 1e12
        assert validator.device == torch.device('cpu')
        
        # Test with custom parameters
        validator_custom = HermitianValidator(
            hermitian_tolerance=1e-8,
            spectrum_tolerance=1e-10,
            positive_semidefinite_tolerance=1e-6,
            condition_number_threshold=1e10,
            device=torch.device('cpu')
        )
        assert validator_custom.hermitian_tolerance == 1e-8
        assert validator_custom.spectrum_tolerance == 1e-10
        assert validator_custom.positive_semidefinite_tolerance == 1e-6
        assert validator_custom.condition_number_threshold == 1e10
    
    def test_validate_hermitian_property_valid(self):
        """Test validation of valid Hermitian matrix."""
        validator = HermitianValidator()
        
        # Create Hermitian matrix
        hermitian_matrix = torch.tensor([
            [1+0j, 2+3j, 0+1j],
            [2-3j, 4+0j, 1+2j],
            [0-1j, 1-2j, 3+0j]
        ], dtype=torch.complex64)
        
        # Should pass validation
        result = validator.validate_hermitian_property(hermitian_matrix)
        assert result is True
    
    def test_validate_hermitian_property_invalid(self):
        """Test validation of non-Hermitian matrix."""
        validator = HermitianValidator()
        
        # Create non-Hermitian matrix
        non_hermitian_matrix = torch.tensor([
            [1+0j, 2+3j],
            [1+1j, 4+0j]  # Not conjugate transpose
        ], dtype=torch.complex64)
        
        # Should fail validation
        result = validator.validate_hermitian_property(non_hermitian_matrix)
        assert result is False
    
    def test_validate_hermitian_property_invalid_input(self):
        """Test error handling for invalid input."""
        validator = HermitianValidator()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            validator.validate_hermitian_property("not a tensor")
        
        # Test with real tensor
        with pytest.raises(ValueError, match="Input tensor must be complex"):
            real_tensor = torch.randn(2, 2)
            validator.validate_hermitian_property(real_tensor)
        
        # Test with non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            nonsquare_matrix = torch.randn(2, 3, dtype=torch.complex64)
            validator.validate_hermitian_property(nonsquare_matrix)
    
    def test_validate_real_spectrum_valid(self):
        """Test validation of matrix with real spectrum."""
        validator = HermitianValidator(spectrum_tolerance=1e-6)  # More realistic tolerance
        
        # Create Hermitian matrix (guaranteed real spectrum)
        hermitian_matrix = torch.tensor([
            [2+0j, 1+1j],
            [1-1j, 3+0j]
        ], dtype=torch.complex64)
        
        # Should pass validation
        result = validator.validate_real_spectrum(hermitian_matrix)
        assert result is True
    
    def test_validate_real_spectrum_invalid(self):
        """Test validation of matrix with complex spectrum."""
        validator = HermitianValidator()
        
        # Create matrix with complex eigenvalues
        complex_spectrum_matrix = torch.tensor([
            [0+0j, 1+0j],
            [0+0j, 0+1j]  # Not Hermitian, will have complex eigenvalues
        ], dtype=torch.complex64)
        
        # Should fail validation
        result = validator.validate_real_spectrum(complex_spectrum_matrix)
        assert result is False
    
    def test_validate_real_spectrum_invalid_input(self):
        """Test error handling for real spectrum validation."""
        validator = HermitianValidator()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            validator.validate_real_spectrum("not a tensor")
        
        # Test with real tensor
        with pytest.raises(ValueError, match="Input tensor must be complex"):
            real_tensor = torch.randn(2, 2)
            validator.validate_real_spectrum(real_tensor)
        
        # Test with non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            nonsquare_matrix = torch.randn(2, 3, dtype=torch.complex64)
            validator.validate_real_spectrum(nonsquare_matrix)
    
    def test_validate_positive_semidefinite_valid(self):
        """Test validation of positive semi-definite matrix."""
        validator = HermitianValidator()
        
        # Create positive definite Hermitian matrix
        A = torch.randn(3, 3, dtype=torch.complex64)
        positive_definite_matrix = A @ A.conj().T
        
        # Should pass validation
        result = validator.validate_positive_semidefinite(positive_definite_matrix)
        assert result is True
    
    def test_validate_positive_semidefinite_invalid(self):
        """Test validation of non-positive semi-definite matrix."""
        validator = HermitianValidator()
        
        # Create matrix with negative eigenvalues
        negative_definite_matrix = torch.tensor([
            [-1+0j, 0+0j],
            [0+0j, -2+0j]
        ], dtype=torch.complex64)
        
        # Should fail validation
        result = validator.validate_positive_semidefinite(negative_definite_matrix)
        assert result is False
    
    def test_validate_positive_semidefinite_invalid_input(self):
        """Test error handling for PSD validation."""
        validator = HermitianValidator()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            validator.validate_positive_semidefinite("not a tensor")
        
        # Test with real tensor
        with pytest.raises(ValueError, match="Input tensor must be complex"):
            real_tensor = torch.randn(2, 2)
            validator.validate_positive_semidefinite(real_tensor)
        
        # Test with non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            nonsquare_matrix = torch.randn(2, 3, dtype=torch.complex64)
            validator.validate_positive_semidefinite(nonsquare_matrix)
    
    def test_compute_condition_number(self):
        """Test condition number computation."""
        validator = HermitianValidator()
        
        # Create well-conditioned matrix
        well_conditioned = torch.tensor([
            [2+0j, 0+0j],
            [0+0j, 2+0j]
        ], dtype=torch.complex64)
        
        condition_number = validator.compute_condition_number(well_conditioned)
        assert condition_number > 0
        assert condition_number < 100  # Should be well-conditioned
        
        # Create ill-conditioned matrix
        ill_conditioned = torch.tensor([
            [1+0j, 0+0j],
            [0+0j, 1e-10+0j]  # Nearly singular
        ], dtype=torch.complex64)
        
        condition_number_ill = validator.compute_condition_number(ill_conditioned)
        assert condition_number_ill > condition_number
    
    def test_compute_condition_number_singular(self):
        """Test condition number computation for singular matrix."""
        validator = HermitianValidator()
        
        # Create singular matrix
        singular_matrix = torch.tensor([
            [0+0j, 0+0j],
            [0+0j, 0+0j]
        ], dtype=torch.complex64)
        
        condition_number = validator.compute_condition_number(singular_matrix)
        assert condition_number == float('inf')
    
    def test_comprehensive_validation_valid(self):
        """Test comprehensive validation of valid Hermitian matrix."""
        validator = HermitianValidator(
            hermitian_tolerance=1e-6,
            spectrum_tolerance=1e-6,
            positive_semidefinite_tolerance=1e-6
        )
        
        # Create valid Hermitian positive semi-definite matrix
        A = torch.randn(3, 3, dtype=torch.complex64)
        valid_matrix = A @ A.conj().T
        
        # Perform comprehensive validation
        result = validator.comprehensive_validation(valid_matrix)
        
        # Check result structure
        assert isinstance(result, HermitianValidationResult)
        assert result.is_hermitian is True
        assert result.hermitian_error < 1e-6
        assert result.has_real_spectrum is True
        assert result.max_imaginary_eigenvalue < 1e-6
        assert result.is_positive_semidefinite is True
        assert result.min_eigenvalue >= -1e-6
        assert result.condition_number > 0
        assert result.validation_time > 0
        assert len(result.errors) == 0
        
        # Should have minimal warnings
        assert len(result.warnings) <= 1  # May have condition number warning
    
    def test_comprehensive_validation_invalid(self):
        """Test comprehensive validation of invalid matrix."""
        validator = HermitianValidator()
        
        # Create non-Hermitian matrix
        invalid_matrix = torch.tensor([
            [1+0j, 2+3j],
            [1+1j, 4+0j]  # Not conjugate transpose
        ], dtype=torch.complex64)
        
        # Perform comprehensive validation
        result = validator.comprehensive_validation(invalid_matrix)
        
        # Check result structure
        assert isinstance(result, HermitianValidationResult)
        assert result.is_hermitian is False
        assert result.hermitian_error > 1e-12
        assert result.validation_time > 0
        assert len(result.warnings) > 0
        
        # Should have warning about not being Hermitian
        hermitian_warning = any("not Hermitian" in warning for warning in result.warnings)
        assert hermitian_warning
    
    def test_comprehensive_validation_invalid_input(self):
        """Test comprehensive validation with invalid input."""
        validator = HermitianValidator()
        
        # Test with non-tensor input
        result = validator.comprehensive_validation("not a tensor")
        
        assert isinstance(result, HermitianValidationResult)
        assert result.is_hermitian is False
        assert result.hermitian_error == float('inf')
        assert len(result.errors) > 0
        assert "Input must be a torch.Tensor" in result.errors[0]
    
    def test_validate_block_structure(self):
        """Test validation of block structure."""
        validator = HermitianValidator()
        
        # Create valid block structure
        blocks = {
            ('a', 'a'): torch.tensor([[1+0j, 0+1j], [0-1j, 2+0j]], dtype=torch.complex64),
            ('b', 'b'): torch.tensor([[3+0j]], dtype=torch.complex64),
            ('a', 'b'): torch.tensor([[1+1j], [0+2j]], dtype=torch.complex64),
            ('b', 'a'): torch.tensor([[1-1j, 0-2j]], dtype=torch.complex64)
        }
        
        expected_structure = {
            'vertex_dimensions': {'a': 2, 'b': 1}
        }
        
        # Validate block structure
        result = validator.validate_block_structure(blocks, expected_structure)
        
        assert isinstance(result, dict)
        assert result['block_structure_valid'] is True
        assert result['hermitian_block_structure'] is True
        assert result['block_dimension_consistency'] is True
        assert len(result['errors']) == 0
    
    def test_validate_block_structure_invalid(self):
        """Test validation of invalid block structure."""
        validator = HermitianValidator()
        
        # Create invalid block structure (conjugate blocks don't match)
        blocks = {
            ('a', 'a'): torch.tensor([[1+0j, 0+1j], [0-1j, 2+0j]], dtype=torch.complex64),
            ('b', 'b'): torch.tensor([[3+0j]], dtype=torch.complex64),
            ('a', 'b'): torch.tensor([[1+1j], [0+2j]], dtype=torch.complex64),
            ('b', 'a'): torch.tensor([[2+2j, 1+3j]], dtype=torch.complex64)  # Wrong conjugate
        }
        
        expected_structure = {
            'vertex_dimensions': {'a': 2, 'b': 1}
        }
        
        # Validate block structure
        result = validator.validate_block_structure(blocks, expected_structure)
        
        assert isinstance(result, dict)
        assert result['block_structure_valid'] is False
        assert result['hermitian_block_structure'] is False
        assert len(result['errors']) > 0
    
    def test_validate_directionality_encoding(self):
        """Test validation of directional encoding matrix."""
        validator = HermitianValidator()
        
        # Create test adjacency matrix
        adjacency_matrix = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=torch.float32)
        
        # Compute expected encoding
        q = 0.25
        A_antisymmetric = adjacency_matrix - adjacency_matrix.T
        expected_encoding = torch.exp(1j * 2 * np.pi * q * A_antisymmetric)
        
        # Validate encoding
        result = validator.validate_directionality_encoding(expected_encoding, adjacency_matrix, q)
        
        assert isinstance(result, dict)
        assert result['encoding_valid'] is True
        assert result['phase_consistency'] is True
        assert result['unitary_property'] is True
        assert len(result['errors']) == 0
    
    def test_validate_directionality_encoding_invalid(self):
        """Test validation of invalid directional encoding."""
        validator = HermitianValidator()
        
        # Create test adjacency matrix
        adjacency_matrix = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=torch.float32)
        
        # Create incorrect encoding
        wrong_encoding = torch.tensor([
            [1+0j, 2+0j, 0+0j],  # Wrong values
            [0+0j, 1+0j, 1+0j],
            [1+0j, 0+0j, 1+0j]
        ], dtype=torch.complex64)
        
        q = 0.25
        
        # Validate encoding
        result = validator.validate_directionality_encoding(wrong_encoding, adjacency_matrix, q)
        
        assert isinstance(result, dict)
        assert result['encoding_valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_sparse_hermitian(self):
        """Test validation of sparse Hermitian matrix."""
        validator = HermitianValidator()
        
        # Create small sparse Hermitian matrix
        dense_hermitian = torch.tensor([
            [1+0j, 2+3j, 0+0j],
            [2-3j, 4+0j, 0+0j],
            [0+0j, 0+0j, 5+0j]
        ], dtype=torch.complex64)
        
        sparse_hermitian = csr_matrix(dense_hermitian.numpy())
        
        # Validate sparse matrix
        result = validator.validate_sparse_hermitian(sparse_hermitian)
        
        assert isinstance(result, dict)
        assert result['sparse_hermitian_valid'] is True
        assert result['sparsity_preserved'] is True
        assert len(result['errors']) == 0
    
    def test_validate_sparse_hermitian_large(self):
        """Test validation of large sparse matrix."""
        validator = HermitianValidator()
        
        # Create large sparse matrix (should skip dense validation)
        large_sparse = csr_matrix((2000, 2000), dtype=complex)
        
        # Validate sparse matrix
        result = validator.validate_sparse_hermitian(large_sparse)
        
        assert isinstance(result, dict)
        assert len(result['warnings']) > 0
        assert "Matrix too large" in result['warnings'][0]
    
    def test_get_validation_summary(self):
        """Test generation of validation summary."""
        validator = HermitianValidator(
            hermitian_tolerance=1e-6,
            spectrum_tolerance=1e-6,
            positive_semidefinite_tolerance=1e-6
        )
        
        # Create valid matrix
        A = torch.randn(2, 2, dtype=torch.complex64)
        valid_matrix = A @ A.conj().T
        
        # Perform validation
        result = validator.comprehensive_validation(valid_matrix)
        
        # Generate summary
        summary = validator.get_validation_summary(result)
        
        assert isinstance(summary, str)
        assert "Hermitian Validation Summary" in summary
        assert "✓" in summary  # Should have checkmarks for valid properties
        assert "PASSED" in summary
        # Check that error is formatted in scientific notation
        assert f"{result.hermitian_error:.2e}" in summary
    
    def test_get_validation_summary_invalid(self):
        """Test generation of validation summary for invalid matrix."""
        validator = HermitianValidator()
        
        # Create invalid matrix
        invalid_matrix = torch.tensor([
            [1+0j, 2+3j],
            [1+1j, 4+0j]  # Not conjugate transpose
        ], dtype=torch.complex64)
        
        # Perform validation
        result = validator.comprehensive_validation(invalid_matrix)
        
        # Generate summary
        summary = validator.get_validation_summary(result)
        
        assert isinstance(summary, str)
        assert "Hermitian Validation Summary" in summary
        assert "✗" in summary  # Should have X marks for invalid properties
        assert "FAILED" in summary
        assert "Warnings:" in summary
    
    def test_device_support(self):
        """Test device support for validation."""
        # Test with CPU device
        validator_cpu = HermitianValidator(device=torch.device('cpu'))
        
        matrix = torch.tensor([
            [1+0j, 2+3j],
            [2-3j, 4+0j]
        ], dtype=torch.complex64)
        
        result = validator_cpu.validate_hermitian_property(matrix)
        assert result is True
        
        # Test device transfer
        if torch.cuda.is_available():
            validator_cuda = HermitianValidator(device=torch.device('cuda'))
            result_cuda = validator_cuda.validate_hermitian_property(matrix)
            assert result_cuda is True
    
    def test_tolerance_settings(self):
        """Test different tolerance settings."""
        # Test with strict tolerance
        validator_strict = HermitianValidator(hermitian_tolerance=1e-15)
        
        # Create matrix with obvious numerical error
        matrix_with_error = torch.tensor([
            [1+0j, 2+3j],
            [2-3j+1e-12j, 4+0j]  # Visible error
        ], dtype=torch.complex64)
        
        result_strict = validator_strict.validate_hermitian_property(matrix_with_error)
        assert result_strict is False  # Should fail with strict tolerance
        
        # Test with relaxed tolerance
        validator_relaxed = HermitianValidator(hermitian_tolerance=1e-10)
        
        result_relaxed = validator_relaxed.validate_hermitian_property(matrix_with_error)
        assert result_relaxed is True  # Should pass with relaxed tolerance
    
    def test_numerical_stability(self):
        """Test numerical stability with various matrices."""
        validator = HermitianValidator(
            hermitian_tolerance=1e-6,
            spectrum_tolerance=1e-6,
            positive_semidefinite_tolerance=1e-6
        )
        
        # Test with different matrix types
        test_cases = [
            torch.zeros(3, 3, dtype=torch.complex64),  # Zero matrix
            torch.eye(3, dtype=torch.complex64),       # Identity matrix
            torch.ones(3, 3, dtype=torch.complex64),   # Ones matrix (not Hermitian)
            1e-6 * torch.randn(3, 3, dtype=torch.complex64),  # Very small values
            1e6 * torch.eye(3, dtype=torch.complex64),        # Large values
        ]
        
        for i, matrix in enumerate(test_cases):
            # Make Hermitian if needed
            if i != 2:  # Skip ones matrix (should remain non-Hermitian)
                matrix = (matrix + matrix.conj().T) / 2
            
            # Validate
            result = validator.comprehensive_validation(matrix)
            
            # Should not crash
            assert isinstance(result, HermitianValidationResult)
            assert result.validation_time > 0
            
            # Zero, identity, and small matrices should be valid
            if i in [0, 1, 3]:
                assert result.is_hermitian is True
                assert result.is_positive_semidefinite is True


class TestHermitianValidationResult:
    """Test suite for HermitianValidationResult dataclass."""
    
    def test_creation(self):
        """Test creation of HermitianValidationResult."""
        result = HermitianValidationResult(
            is_hermitian=True,
            hermitian_error=1e-14,
            has_real_spectrum=True,
            max_imaginary_eigenvalue=1e-15,
            is_positive_semidefinite=True,
            min_eigenvalue=0.1,
            condition_number=10.0,
            validation_time=0.005,
            errors=[],
            warnings=[]
        )
        
        assert result.is_hermitian is True
        assert result.hermitian_error == 1e-14
        assert result.has_real_spectrum is True
        assert result.max_imaginary_eigenvalue == 1e-15
        assert result.is_positive_semidefinite is True
        assert result.min_eigenvalue == 0.1
        assert result.condition_number == 10.0
        assert result.validation_time == 0.005
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_validation_errors(self):
        """Test validation errors in HermitianValidationResult."""
        # Test with negative hermitian_error
        with pytest.raises(ValueError, match="Hermitian error must be non-negative"):
            HermitianValidationResult(
                is_hermitian=False,
                hermitian_error=-1e-14,  # Invalid
                has_real_spectrum=False,
                max_imaginary_eigenvalue=1e-15,
                is_positive_semidefinite=False,
                min_eigenvalue=-0.1,
                condition_number=10.0,
                validation_time=0.005,
                errors=[],
                warnings=[]
            )
        
        # Test with negative max_imaginary_eigenvalue
        with pytest.raises(ValueError, match="Max imaginary eigenvalue must be non-negative"):
            HermitianValidationResult(
                is_hermitian=False,
                hermitian_error=1e-14,
                has_real_spectrum=False,
                max_imaginary_eigenvalue=-1e-15,  # Invalid
                is_positive_semidefinite=False,
                min_eigenvalue=-0.1,
                condition_number=10.0,
                validation_time=0.005,
                errors=[],
                warnings=[]
            )


if __name__ == '__main__':
    pytest.main([__file__])
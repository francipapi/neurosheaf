"""Unit tests for RealToComplexReconstruction class.

Tests the mathematical correctness of real-to-complex reconstruction:
- Matrix reconstruction: [[X, -Y], [Y, X]] → Z = X + iY
- Eigenvalue reconstruction from conjugate pairs
- Eigenvector reconstruction
- Round-trip conversion accuracy
- Spectral property preservation
"""

import pytest
import torch
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Any

from neurosheaf.directed_sheaf.conversion.complex_reconstruction import RealToComplexReconstruction
from neurosheaf.directed_sheaf.conversion.real_embedding import ComplexToRealEmbedding


class TestRealToComplexReconstruction:
    """Test suite for RealToComplexReconstruction class."""
    
    def test_initialization(self):
        """Test initialization of RealToComplexReconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        assert reconstructor.validate_properties is True
        assert reconstructor.tolerance == 1e-12
        assert reconstructor.device == torch.device('cpu')
        
        # Test with custom parameters
        reconstructor_custom = RealToComplexReconstruction(
            validate_properties=False,
            tolerance=1e-8,
            device=torch.device('cpu')
        )
        assert reconstructor_custom.validate_properties is False
        assert reconstructor_custom.tolerance == 1e-8
    
    def test_reconstruct_matrix_basic(self):
        """Test basic matrix reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Create real embedded matrix: [[X, -Y], [Y, X]]
        real_matrix = torch.tensor([
            [1, 3, -2, -4],  # [X, -Y]
            [5, 7, -6, -8],
            [2, 4,  1,  3],  # [Y, X]
            [6, 8,  5,  7]
        ], dtype=torch.float32)
        
        complex_reconstructed = reconstructor.reconstruct_matrix(real_matrix)
        
        # Check dimensions
        assert complex_reconstructed.shape == (2, 2)
        assert complex_reconstructed.is_complex()
        
        # Check reconstruction
        expected = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        assert torch.allclose(complex_reconstructed, expected)
    
    def test_reconstruct_matrix_validation(self):
        """Test matrix reconstruction with validation."""
        reconstructor = RealToComplexReconstruction(validate_properties=True)
        
        # Create valid real embedded matrix
        real_matrix = torch.tensor([
            [1, 3, -2, -4],  # [X, -Y]
            [5, 7, -6, -8],
            [2, 4,  1,  3],  # [Y, X]
            [6, 8,  5,  7]
        ], dtype=torch.float32)
        
        # Should pass validation
        complex_reconstructed = reconstructor.reconstruct_matrix(real_matrix)
        assert complex_reconstructed.shape == (2, 2)
        
        # Test with validation disabled
        reconstructor_no_val = RealToComplexReconstruction(validate_properties=False)
        complex_reconstructed_no_val = reconstructor_no_val.reconstruct_matrix(real_matrix)
        
        # Should get same result
        assert torch.allclose(complex_reconstructed, complex_reconstructed_no_val)
    
    def test_reconstruct_matrix_invalid_input(self):
        """Test error handling for invalid inputs."""
        reconstructor = RealToComplexReconstruction()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            reconstructor.reconstruct_matrix("not a tensor")
        
        # Test with complex tensor
        with pytest.raises(ValueError, match="Input tensor must be real"):
            complex_tensor = torch.tensor([[1+2j, 3+4j]], dtype=torch.complex64)
            reconstructor.reconstruct_matrix(complex_tensor)
        
        # Test with odd dimensions
        with pytest.raises(ValueError, match="Real matrix dimensions must be even"):
            odd_matrix = torch.randn(3, 4)
            reconstructor.reconstruct_matrix(odd_matrix)
    
    def test_reconstruct_vector_basic(self):
        """Test basic vector reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Test 1D vector
        real_vector = torch.tensor([1, 3, 2, 4], dtype=torch.float32)
        complex_reconstructed = reconstructor.reconstruct_vector(real_vector)
        
        assert complex_reconstructed.shape == (2,)
        assert complex_reconstructed.is_complex()
        
        expected = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
        assert torch.allclose(complex_reconstructed, expected)
        
        # Test 2D vector
        real_vector_2d = torch.tensor([[1], [3], [2], [4]], dtype=torch.float32)
        complex_reconstructed_2d = reconstructor.reconstruct_vector(real_vector_2d)
        
        assert complex_reconstructed_2d.shape == (2, 1)
        expected_2d = torch.tensor([[1+2j], [3+4j]], dtype=torch.complex64)
        assert torch.allclose(complex_reconstructed_2d, expected_2d)
    
    def test_reconstruct_vector_invalid_input(self):
        """Test error handling for vector reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            reconstructor.reconstruct_vector("not a tensor")
        
        # Test with complex tensor
        with pytest.raises(ValueError, match="Input tensor must be real"):
            complex_vector = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
            reconstructor.reconstruct_vector(complex_vector)
        
        # Test with odd dimensions
        with pytest.raises(ValueError, match="Real vector dimension must be even"):
            odd_vector = torch.randn(3)
            reconstructor.reconstruct_vector(odd_vector)
    
    def test_reconstruct_eigenvalues_basic(self):
        """Test basic eigenvalue reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # For Hermitian matrices, eigenvalues are real and come in conjugate pairs
        # Since they're real, the pairs are identical
        real_eigenvalues = torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        
        reconstructed_eigenvalues = reconstructor.reconstruct_eigenvalues(real_eigenvalues)
        
        assert reconstructed_eigenvalues.shape == (2,)
        assert not reconstructed_eigenvalues.is_complex()
        
        expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert torch.allclose(reconstructed_eigenvalues, expected)
    
    def test_reconstruct_eigenvalues_invalid_input(self):
        """Test error handling for eigenvalue reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            reconstructor.reconstruct_eigenvalues("not a tensor")
        
        # Test with complex tensor
        with pytest.raises(ValueError, match="Input eigenvalues must be real"):
            complex_eigenvalues = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
            reconstructor.reconstruct_eigenvalues(complex_eigenvalues)
        
        # Test with odd number of eigenvalues
        with pytest.raises(ValueError, match="Number of eigenvalues must be even"):
            odd_eigenvalues = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            reconstructor.reconstruct_eigenvalues(odd_eigenvalues)
    
    def test_reconstruct_eigenvectors_basic(self):
        """Test basic eigenvector reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Create real eigenvectors from embedding
        real_eigenvectors = torch.tensor([
            [1, 0],    # Real parts
            [0, 1],
            [2, 0],    # Imaginary parts
            [0, 2]
        ], dtype=torch.float32)
        
        complex_eigenvectors = reconstructor.reconstruct_eigenvectors(real_eigenvectors)
        
        assert complex_eigenvectors.shape == (2, 2)
        assert complex_eigenvectors.is_complex()
        
        expected = torch.tensor([
            [1+2j, 0+0j],
            [0+0j, 1+2j]
        ], dtype=torch.complex64)
        
        assert torch.allclose(complex_eigenvectors, expected)
    
    def test_reconstruct_eigenvectors_invalid_input(self):
        """Test error handling for eigenvector reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            reconstructor.reconstruct_eigenvectors("not a tensor")
        
        # Test with complex tensor
        with pytest.raises(ValueError, match="Input eigenvectors must be real"):
            complex_eigenvectors = torch.tensor([[1+2j], [3+4j]], dtype=torch.complex64)
            reconstructor.reconstruct_eigenvectors(complex_eigenvectors)
        
        # Test with odd dimensions
        with pytest.raises(ValueError, match="Number of eigenvector components must be even"):
            odd_eigenvectors = torch.randn(3, 2)
            reconstructor.reconstruct_eigenvectors(odd_eigenvectors)
    
    def test_reconstruct_spectrum_basic(self):
        """Test complete spectrum reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Create real eigenvalues and eigenvectors
        real_eigenvalues = torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        real_eigenvectors = torch.tensor([
            [1, 0],    # Real parts
            [0, 1],
            [2, 0],    # Imaginary parts
            [0, 2]
        ], dtype=torch.float32)
        
        complex_eigenvalues, complex_eigenvectors = reconstructor.reconstruct_spectrum(
            real_eigenvalues, real_eigenvectors
        )
        
        # Check eigenvalues
        assert complex_eigenvalues.shape == (2,)
        assert not complex_eigenvalues.is_complex()
        
        # Check eigenvectors
        assert complex_eigenvectors.shape == (2, 2)
        assert complex_eigenvectors.is_complex()
    
    def test_reconstruct_sparse_matrix_basic(self):
        """Test sparse matrix reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Create real sparse matrix in embedded format
        # For simplicity, create a small example
        real_data = np.array([1.0, 3.0, -2.0, -4.0, 2.0, 4.0, 1.0, 3.0])
        real_row = np.array([0, 1, 0, 1, 2, 3, 2, 3])
        real_col = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        real_sparse = csr_matrix((real_data, (real_row, real_col)), shape=(4, 4))
        
        complex_sparse = reconstructor.reconstruct_sparse_matrix(real_sparse)
        
        # Check properties
        assert complex_sparse.shape == (2, 2)
        assert np.iscomplexobj(complex_sparse.data)
        assert complex_sparse.nnz > 0
    
    def test_reconstruct_sparse_matrix_invalid_input(self):
        """Test error handling for sparse matrix reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Test with complex sparse matrix
        complex_data = np.array([1+2j, 3+4j])
        row = np.array([0, 1])
        col = np.array([0, 1])
        complex_sparse = csr_matrix((complex_data, (row, col)), shape=(2, 2))
        
        with pytest.raises(ValueError, match="Input sparse matrix must be real"):
            reconstructor.reconstruct_sparse_matrix(complex_sparse)
        
        # Test with odd dimensions
        real_data = np.array([1.0, 2.0])
        row = np.array([0, 1])
        col = np.array([0, 1])
        odd_sparse = csr_matrix((real_data, (row, col)), shape=(3, 3))
        
        with pytest.raises(ValueError, match="Real sparse matrix dimensions must be even"):
            reconstructor.reconstruct_sparse_matrix(odd_sparse)
    
    def test_reconstruct_laplacian_blocks(self):
        """Test reconstruction of Laplacian blocks."""
        reconstructor = RealToComplexReconstruction()
        
        # Create real Laplacian blocks in valid embedding format
        # For complex matrix [[1+0j, 0+1j], [0-1j, 1+0j]]
        real_blocks = {
            ('a', 'a'): torch.tensor([
                [1, 0, 0, -1],  # [X, -Y]
                [0, 1, 1, 0],
                [0, 1, 1, 0],   # [Y, X]
                [-1, 0, 0, 1]
            ], dtype=torch.float32),
            # For complex matrix [[0+1j]]
            ('a', 'b'): torch.tensor([
                [0, -1],  # [X, -Y] where X=0, Y=1
                [1, 0]    # [Y, X] where Y=1, X=0
            ], dtype=torch.float32)
        }
        
        complex_blocks = reconstructor.reconstruct_laplacian_blocks(real_blocks)
        
        # Check results
        assert len(complex_blocks) == 2
        assert ('a', 'a') in complex_blocks
        assert ('a', 'b') in complex_blocks
        
        # Check dimensions
        assert complex_blocks[('a', 'a')].shape == (2, 2)
        assert complex_blocks[('a', 'b')].shape == (1, 1)
        
        # Check that all results are complex
        for block in complex_blocks.values():
            assert block.is_complex()
    
    def test_reconstruct_laplacian_blocks_invalid_input(self):
        """Test error handling for Laplacian block reconstruction."""
        reconstructor = RealToComplexReconstruction()
        
        # Test with non-dict input
        with pytest.raises(ValueError, match="real_blocks must be a dictionary"):
            reconstructor.reconstruct_laplacian_blocks("not a dict")
    
    def test_validate_round_trip(self):
        """Test round-trip validation."""
        reconstructor = RealToComplexReconstruction()
        
        # Create original complex matrix
        original = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        # Create reconstructed matrix (should be identical)
        reconstructed = original.clone()
        
        validation_result = reconstructor.validate_round_trip(original, reconstructed)
        
        # Check validation results
        assert validation_result['reconstruction_error'] < 1e-12
        assert validation_result['passes_tolerance'] is True
        assert 'relative_error' in validation_result
        assert 'spectral_error' in validation_result
    
    def test_validate_round_trip_invalid_input(self):
        """Test error handling for round-trip validation."""
        reconstructor = RealToComplexReconstruction()
        
        # Test with mismatched shapes
        original = torch.tensor([[1+2j, 3+4j]], dtype=torch.complex64)
        reconstructed = torch.tensor([[1+2j]], dtype=torch.complex64)
        
        with pytest.raises(ValueError, match="Original and reconstructed matrices must have same shape"):
            reconstructor.validate_round_trip(original, reconstructed)
    
    def test_get_reconstruction_metadata(self):
        """Test reconstruction metadata generation."""
        reconstructor = RealToComplexReconstruction()
        
        # Create real matrix
        real_matrix = torch.tensor([
            [1, 3, -2, -4],
            [5, 7, -6, -8],
            [2, 4,  1,  3],
            [6, 8,  5,  7]
        ], dtype=torch.float32)
        
        metadata = reconstructor.get_reconstruction_metadata(real_matrix)
        
        # Check metadata content
        assert metadata['real_shape'] == (4, 4)
        assert metadata['reconstructed_shape'] == (2, 2)
        assert metadata['dimension_scaling'] == 0.25
        assert metadata['memory_scaling'] == 0.5
        assert metadata['dtype'] == torch.float32
        assert metadata['is_valid_embedding'] is True
    
    def test_device_support(self):
        """Test device support for computations."""
        # Test with CPU device
        reconstructor = RealToComplexReconstruction(device=torch.device('cpu'))
        
        real_matrix = torch.tensor([
            [1, 3, -2, -4],
            [5, 7, -6, -8],
            [2, 4,  1,  3],
            [6, 8,  5,  7]
        ], dtype=torch.float32)
        
        complex_reconstructed = reconstructor.reconstruct_matrix(real_matrix)
        
        assert complex_reconstructed.device == torch.device('cpu')
        assert complex_reconstructed.is_complex()
        
        # Test device transfer
        if torch.cuda.is_available():
            reconstructor_cuda = RealToComplexReconstruction(device=torch.device('cuda'))
            complex_reconstructed_cuda = reconstructor_cuda.reconstruct_matrix(real_matrix)
            assert complex_reconstructed_cuda.device.type == 'cuda'


class TestRealToComplexReconstructionRoundTrip:
    """Test round-trip conversion with ComplexToRealEmbedding."""
    
    def test_round_trip_matrix_conversion(self):
        """Test complete round-trip matrix conversion."""
        embedder = ComplexToRealEmbedding(validate_properties=False)
        reconstructor = RealToComplexReconstruction(validate_properties=False)
        
        # Create original complex matrix
        original = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        # Round-trip conversion
        real_embedded = embedder.embed_matrix(original)
        reconstructed = reconstructor.reconstruct_matrix(real_embedded)
        
        # Check accuracy
        assert torch.allclose(original, reconstructed, atol=1e-12)
    
    def test_round_trip_vector_conversion(self):
        """Test complete round-trip vector conversion."""
        embedder = ComplexToRealEmbedding(validate_properties=False)
        reconstructor = RealToComplexReconstruction(validate_properties=False)
        
        # Create original complex vector
        original = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64)
        
        # Round-trip conversion
        real_embedded = embedder.embed_vector(original)
        reconstructed = reconstructor.reconstruct_vector(real_embedded)
        
        # Check accuracy
        assert torch.allclose(original, reconstructed, atol=1e-12)
    
    def test_round_trip_hermitian_matrix(self):
        """Test round-trip conversion preserves Hermitian property."""
        embedder = ComplexToRealEmbedding(validate_properties=False)
        reconstructor = RealToComplexReconstruction(validate_properties=False)
        
        # Create Hermitian matrix
        original = torch.tensor([
            [1+0j, 2+3j],
            [2-3j, 4+0j]
        ], dtype=torch.complex64)
        
        # Verify it's Hermitian
        assert torch.allclose(original, original.conj().T)
        
        # Round-trip conversion
        real_embedded = embedder.embed_matrix(original)
        reconstructed = reconstructor.reconstruct_matrix(real_embedded)
        
        # Check accuracy
        assert torch.allclose(original, reconstructed, atol=1e-12)
        
        # Check that Hermitian property is preserved
        assert torch.allclose(reconstructed, reconstructed.conj().T, atol=1e-12)
    
    def test_round_trip_spectral_properties(self):
        """Test that spectral properties are preserved in round-trip."""
        embedder = ComplexToRealEmbedding(validate_properties=False)
        reconstructor = RealToComplexReconstruction(validate_properties=False)
        
        # Create Hermitian matrix
        original = torch.tensor([
            [2+0j, 1+1j],
            [1-1j, 3+0j]
        ], dtype=torch.complex64)
        
        # Compute original eigenvalues
        original_eigenvalues = torch.linalg.eigvals(original)
        
        # Round-trip conversion
        real_embedded = embedder.embed_matrix(original)
        reconstructed = reconstructor.reconstruct_matrix(real_embedded)
        
        # Compute reconstructed eigenvalues
        reconstructed_eigenvalues = torch.linalg.eigvals(reconstructed)
        
        # Check that eigenvalues are preserved
        orig_sorted = torch.sort(original_eigenvalues.real)[0]
        recon_sorted = torch.sort(reconstructed_eigenvalues.real)[0]
        
        assert torch.allclose(orig_sorted, recon_sorted, atol=1e-6)
    
    def test_round_trip_large_matrices(self):
        """Test round-trip conversion with larger matrices."""
        embedder = ComplexToRealEmbedding(validate_properties=False)
        reconstructor = RealToComplexReconstruction(validate_properties=False)
        
        # Create larger complex matrix
        n = 10
        original = torch.randn(n, n, dtype=torch.complex64)
        
        # Round-trip conversion
        real_embedded = embedder.embed_matrix(original)
        reconstructed = reconstructor.reconstruct_matrix(real_embedded)
        
        # Check accuracy
        assert torch.allclose(original, reconstructed, atol=1e-6)
        
        # Check dimensions
        assert original.shape == reconstructed.shape
        assert real_embedded.shape == (2*n, 2*n)
    
    def test_round_trip_numerical_stability(self):
        """Test numerical stability of round-trip conversion."""
        embedder = ComplexToRealEmbedding(validate_properties=False)
        reconstructor = RealToComplexReconstruction(validate_properties=False)
        
        # Test with various matrices
        test_cases = [
            torch.zeros(3, 3, dtype=torch.complex64),  # All zeros
            torch.eye(3, dtype=torch.complex64),       # Identity
            torch.ones(3, 3, dtype=torch.complex64),   # All ones
            1e-12 * torch.randn(3, 3, dtype=torch.complex64),  # Very small values
            1e12 * torch.randn(3, 3, dtype=torch.complex64),   # Very large values
        ]
        
        for original in test_cases:
            # Round-trip conversion
            real_embedded = embedder.embed_matrix(original)
            reconstructed = reconstructor.reconstruct_matrix(real_embedded)
            
            # Check relative accuracy
            max_error = torch.abs(original - reconstructed).max().item()
            relative_error = max_error / (torch.abs(original).max().item() + 1e-12)
            
            assert relative_error < 1e-6, f"High relative error: {relative_error}"


if __name__ == '__main__':
    pytest.main([__file__])
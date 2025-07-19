"""Unit tests for ComplexToRealEmbedding class.

Tests the mathematical correctness of complex-to-real embedding:
- Matrix embedding: Z = X + iY → [[X, -Y], [Y, X]]
- Spectral property preservation
- Hermitian-to-symmetric mapping
- Sparse matrix support
- Filtration masking
"""

import pytest
import torch
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from typing import Dict, Any

from neurosheaf.directed_sheaf.conversion.real_embedding import ComplexToRealEmbedding
from neurosheaf.directed_sheaf.data_structures import DirectedSheaf


class TestComplexToRealEmbedding:
    """Test suite for ComplexToRealEmbedding class."""
    
    def test_initialization(self):
        """Test initialization of ComplexToRealEmbedding."""
        embedder = ComplexToRealEmbedding()
        
        assert embedder.validate_properties is True
        assert embedder.tolerance == 1e-12
        assert embedder.device == torch.device('cpu')
        
        # Test with custom parameters
        embedder_custom = ComplexToRealEmbedding(
            validate_properties=False,
            tolerance=1e-8,
            device=torch.device('cpu')
        )
        assert embedder_custom.validate_properties is False
        assert embedder_custom.tolerance == 1e-8
    
    def test_embed_matrix_basic(self):
        """Test basic matrix embedding."""
        embedder = ComplexToRealEmbedding()
        
        # Create simple complex matrix
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        real_embedded = embedder.embed_matrix(complex_matrix)
        
        # Check dimensions
        assert real_embedded.shape == (4, 4)
        assert not real_embedded.is_complex()
        
        # Check block structure: [[X, -Y], [Y, X]]
        # X = [[1, 3], [5, 7]]
        # Y = [[2, 4], [6, 8]]
        expected = torch.tensor([
            [1, 3, -2, -4],  # [X, -Y]
            [5, 7, -6, -8],
            [2, 4,  1,  3],  # [Y, X]
            [6, 8,  5,  7]
        ], dtype=torch.float32)
        
        assert torch.allclose(real_embedded, expected)
    
    def test_embed_matrix_validation(self):
        """Test matrix embedding with validation."""
        embedder = ComplexToRealEmbedding(validate_properties=True)
        
        # Create complex matrix
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        # Should pass validation
        real_embedded = embedder.embed_matrix(complex_matrix)
        assert real_embedded.shape == (4, 4)
        
        # Test with validation disabled
        embedder_no_val = ComplexToRealEmbedding(validate_properties=False)
        real_embedded_no_val = embedder_no_val.embed_matrix(complex_matrix)
        
        # Should get same result
        assert torch.allclose(real_embedded, real_embedded_no_val)
    
    def test_embed_matrix_invalid_input(self):
        """Test error handling for invalid inputs."""
        embedder = ComplexToRealEmbedding()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            embedder.embed_matrix("not a tensor")
        
        # Test with real tensor
        with pytest.raises(ValueError, match="Input tensor must be complex"):
            embedder.embed_matrix(torch.randn(2, 2))
    
    def test_embed_vector_basic(self):
        """Test basic vector embedding."""
        embedder = ComplexToRealEmbedding()
        
        # Test 1D vector
        complex_vector = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
        real_embedded = embedder.embed_vector(complex_vector)
        
        assert real_embedded.shape == (4,)
        assert not real_embedded.is_complex()
        
        expected = torch.tensor([1, 3, 2, 4], dtype=torch.float32)
        assert torch.allclose(real_embedded, expected)
        
        # Test 2D vector
        complex_vector_2d = torch.tensor([[1+2j], [3+4j]], dtype=torch.complex64)
        real_embedded_2d = embedder.embed_vector(complex_vector_2d)
        
        assert real_embedded_2d.shape == (4, 1)
        expected_2d = torch.tensor([[1], [3], [2], [4]], dtype=torch.float32)
        assert torch.allclose(real_embedded_2d, expected_2d)
    
    def test_embed_vector_invalid_input(self):
        """Test error handling for vector embedding."""
        embedder = ComplexToRealEmbedding()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            embedder.embed_vector("not a tensor")
        
        # Test with real tensor
        with pytest.raises(ValueError, match="Input tensor must be complex"):
            embedder.embed_vector(torch.randn(2))
    
    def test_embed_restrictions(self):
        """Test embedding of restriction maps."""
        embedder = ComplexToRealEmbedding()
        
        # Create directed restrictions
        directed_restrictions = {
            ('a', 'b'): torch.tensor([[1+2j, 3+4j]], dtype=torch.complex64),
            ('b', 'c'): torch.tensor([[5+6j]], dtype=torch.complex64)
        }
        
        real_restrictions = embedder.embed_restrictions(directed_restrictions)
        
        # Check results
        assert len(real_restrictions) == 2
        assert ('a', 'b') in real_restrictions
        assert ('b', 'c') in real_restrictions
        
        # Check shapes
        assert real_restrictions[('a', 'b')].shape == (2, 4)
        assert real_restrictions[('b', 'c')].shape == (2, 2)
        
        # Check that all results are real
        for restriction in real_restrictions.values():
            assert not restriction.is_complex()
    
    def test_embed_restrictions_invalid_input(self):
        """Test error handling for restriction embedding."""
        embedder = ComplexToRealEmbedding()
        
        # Test with non-dict input
        with pytest.raises(ValueError, match="directed_restrictions must be a dictionary"):
            embedder.embed_restrictions("not a dict")
    
    def test_embed_sheaf_data(self):
        """Test embedding of complete sheaf data."""
        embedder = ComplexToRealEmbedding()
        
        # Create mock directed sheaf
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        complex_stalks = {
            'a': torch.tensor([[1+2j, 3+4j]], dtype=torch.complex64),
            'b': torch.tensor([[5+6j]], dtype=torch.complex64),
            'c': torch.tensor([[7+8j]], dtype=torch.complex64)
        }
        
        directed_restrictions = {
            ('a', 'b'): torch.tensor([[1+1j]], dtype=torch.complex64),
            ('b', 'c'): torch.tensor([[2+2j]], dtype=torch.complex64)
        }
        
        directed_sheaf = DirectedSheaf(
            poset=poset,
            complex_stalks=complex_stalks,
            directed_restrictions=directed_restrictions
        )
        
        real_stalks, real_restrictions = embedder.embed_sheaf_data(directed_sheaf)
        
        # Check results
        assert len(real_stalks) == 3
        assert len(real_restrictions) == 2
        
        # Check that all results are real
        for stalk in real_stalks.values():
            assert not stalk.is_complex()
        
        for restriction in real_restrictions.values():
            assert not restriction.is_complex()
    
    def test_embed_sheaf_data_invalid_input(self):
        """Test error handling for sheaf data embedding."""
        embedder = ComplexToRealEmbedding()
        
        # Test with non-DirectedSheaf input
        with pytest.raises(ValueError, match="Input must be a DirectedSheaf"):
            embedder.embed_sheaf_data("not a sheaf")
    
    def test_embed_laplacian_blocks(self):
        """Test embedding of Laplacian blocks."""
        embedder = ComplexToRealEmbedding()
        
        # Create mock Laplacian blocks
        laplacian_blocks = {
            ('a', 'a'): torch.tensor([[1+0j, 0+1j], [0-1j, 1+0j]], dtype=torch.complex64),
            ('a', 'b'): torch.tensor([[0-1j, 1+0j]], dtype=torch.complex64),
            ('b', 'a'): torch.tensor([[0+1j], [1+0j]], dtype=torch.complex64),
            ('b', 'b'): torch.tensor([[2+0j]], dtype=torch.complex64)
        }
        
        real_blocks = embedder.embed_laplacian_blocks(laplacian_blocks)
        
        # Check results
        assert len(real_blocks) == 4
        assert all(key in real_blocks for key in laplacian_blocks.keys())
        
        # Check dimensions
        assert real_blocks[('a', 'a')].shape == (4, 4)
        assert real_blocks[('a', 'b')].shape == (2, 4)
        assert real_blocks[('b', 'a')].shape == (4, 2)
        assert real_blocks[('b', 'b')].shape == (2, 2)
        
        # Check that all results are real
        for block in real_blocks.values():
            assert not block.is_complex()
    
    def test_embed_sparse_matrix(self):
        """Test embedding of sparse matrices."""
        embedder = ComplexToRealEmbedding()
        
        # Create complex sparse matrix
        data = np.array([1+2j, 3+4j, 5+6j])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        complex_sparse = csr_matrix((data, (row, col)), shape=(3, 3))
        
        real_embedded = embedder.embed_sparse_matrix(complex_sparse)
        
        # Check properties
        assert real_embedded.shape == (6, 6)
        assert not np.iscomplexobj(real_embedded.data)
        
        # Check sparsity preserved
        assert real_embedded.nnz > 0
    
    def test_embed_sparse_matrix_invalid_input(self):
        """Test error handling for sparse matrix embedding."""
        embedder = ComplexToRealEmbedding()
        
        # Test with real sparse matrix
        data = np.array([1.0, 2.0, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        real_sparse = csr_matrix((data, (row, col)), shape=(3, 3))
        
        with pytest.raises(ValueError, match="Input sparse matrix must be complex"):
            embedder.embed_sparse_matrix(real_sparse)
    
    def test_embed_with_filtration_mask(self):
        """Test embedding with filtration masking."""
        embedder = ComplexToRealEmbedding()
        
        # Create complex matrix
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        # Create mask
        mask = torch.tensor([
            [True, False],
            [False, True]
        ])
        
        real_embedded = embedder.embed_with_filtration_mask(complex_matrix, mask)
        
        # Check dimensions
        assert real_embedded.shape == (4, 4)
        
        # Check that masked entries are zero
        # The mask should be applied to all four blocks
        expected_zeros = [
            (0, 1), (0, 3),  # Top row, second column (both X and -Y blocks)
            (1, 0), (1, 2),  # Second row, first column (both X and -Y blocks)
            (2, 1), (2, 3),  # Third row, second column (both Y and X blocks)
            (3, 0), (3, 2)   # Fourth row, first column (both Y and X blocks)
        ]
        
        for i, j in expected_zeros:
            assert abs(real_embedded[i, j]) < 1e-12
    
    def test_hermitian_to_symmetric_mapping(self):
        """Test that Hermitian matrices map to symmetric matrices."""
        embedder = ComplexToRealEmbedding()
        
        # Create Hermitian matrix
        hermitian_matrix = torch.tensor([
            [1+0j, 2+3j],
            [2-3j, 4+0j]
        ], dtype=torch.complex64)
        
        # Verify it's Hermitian
        assert torch.allclose(hermitian_matrix, hermitian_matrix.conj().T)
        
        # Embed to real representation
        real_embedded = embedder.embed_matrix(hermitian_matrix)
        
        # Check that real embedding is symmetric
        assert torch.allclose(real_embedded, real_embedded.T, atol=1e-12)
    
    def test_get_embedding_metadata(self):
        """Test embedding metadata generation."""
        embedder = ComplexToRealEmbedding()
        
        # Create complex matrix
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        metadata = embedder.get_embedding_metadata(complex_matrix)
        
        # Check metadata content
        assert metadata['original_shape'] == (2, 2)
        assert metadata['embedded_shape'] == (4, 4)
        assert metadata['dimension_scaling'] == 4
        assert metadata['memory_scaling'] == 4
        assert metadata['dtype'] == torch.complex64
        assert 'is_hermitian' in metadata
        assert 'is_sparse' in metadata
    
    def test_estimate_memory_overhead(self):
        """Test memory overhead estimation."""
        embedder = ComplexToRealEmbedding()
        
        # Test with simple shape
        complex_shape = (10, 10)
        estimates = embedder.estimate_memory_overhead(complex_shape)
        
        # Check estimates
        assert estimates['complex_memory_bytes'] == 10 * 10 * 2 * 4  # 2 components × 4 bytes
        assert estimates['real_memory_bytes'] == 4 * 10 * 10 * 4     # 4× elements × 4 bytes
        assert estimates['memory_overhead_ratio'] == 2.0
        assert estimates['dimension_scaling'] == (20, 20)
    
    def test_device_support(self):
        """Test device support for computations."""
        # Test with CPU device
        embedder = ComplexToRealEmbedding(device=torch.device('cpu'))
        
        complex_matrix = torch.tensor([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], dtype=torch.complex64)
        
        real_embedded = embedder.embed_matrix(complex_matrix)
        
        assert real_embedded.device == torch.device('cpu')
        assert not real_embedded.is_complex()
        
        # Test device transfer
        if torch.cuda.is_available():
            embedder_cuda = ComplexToRealEmbedding(device=torch.device('cuda'))
            real_embedded_cuda = embedder_cuda.embed_matrix(complex_matrix)
            assert real_embedded_cuda.device.type == 'cuda'
    
    def test_numerical_stability(self):
        """Test numerical stability of embedding."""
        embedder = ComplexToRealEmbedding(tolerance=1e-15)
        
        # Test with various complex matrices
        test_cases = [
            torch.zeros(3, 3, dtype=torch.complex64),  # All zeros
            torch.eye(3, dtype=torch.complex64),       # Identity
            torch.ones(3, 3, dtype=torch.complex64),   # All ones
            torch.randn(3, 3, dtype=torch.complex64),  # Random
        ]
        
        for complex_matrix in test_cases:
            real_embedded = embedder.embed_matrix(complex_matrix)
            
            # Check basic properties
            assert real_embedded.shape == (6, 6)
            assert not real_embedded.is_complex()
            
            # Check block structure
            n = complex_matrix.shape[0]
            real_part = complex_matrix.real
            imag_part = complex_matrix.imag
            
            # Verify blocks
            assert torch.allclose(real_embedded[:n, :n], real_part, atol=1e-15)
            assert torch.allclose(real_embedded[:n, n:], -imag_part, atol=1e-15)
            assert torch.allclose(real_embedded[n:, :n], imag_part, atol=1e-15)
            assert torch.allclose(real_embedded[n:, n:], real_part, atol=1e-15)


class TestComplexToRealEmbeddingMathematicalProperties:
    """Test mathematical properties of complex-to-real embedding."""
    
    def test_spectral_property_preservation(self):
        """Test that spectral properties are preserved."""
        embedder = ComplexToRealEmbedding()
        
        # Create complex matrix
        complex_matrix = torch.tensor([
            [1+0j, 2+3j],
            [2-3j, 4+0j]
        ], dtype=torch.complex64)
        
        # Compute original eigenvalues
        original_eigenvalues = torch.linalg.eigvals(complex_matrix)
        
        # Embed to real representation
        real_embedded = embedder.embed_matrix(complex_matrix)
        
        # Compute real eigenvalues
        real_eigenvalues = torch.linalg.eigvals(real_embedded)
        
        # For Hermitian matrices, eigenvalues should appear as conjugate pairs
        # Since the matrix is Hermitian, eigenvalues are real
        assert real_eigenvalues.shape[0] == 2 * original_eigenvalues.shape[0]
        
        # Check that eigenvalues are properly paired
        sorted_original = torch.sort(original_eigenvalues.real)[0]
        sorted_real_first_half = torch.sort(real_eigenvalues.real[:2])[0]
        sorted_real_second_half = torch.sort(real_eigenvalues.real[2:])[0]
        
        assert torch.allclose(sorted_original, sorted_real_first_half, atol=1e-6)
        assert torch.allclose(sorted_original, sorted_real_second_half, atol=1e-6)
    
    def test_dimension_scaling(self):
        """Test correct dimension scaling."""
        embedder = ComplexToRealEmbedding()
        
        # Test various matrix sizes
        test_shapes = [(2, 2), (3, 5), (10, 7), (1, 1)]
        
        for n, m in test_shapes:
            complex_matrix = torch.randn(n, m, dtype=torch.complex64)
            real_embedded = embedder.embed_matrix(complex_matrix)
            
            # Check dimension scaling
            assert real_embedded.shape == (2*n, 2*m)
    
    def test_positive_definiteness_preservation(self):
        """Test preservation of positive definiteness."""
        embedder = ComplexToRealEmbedding()
        
        # Create positive definite Hermitian matrix
        A = torch.randn(3, 3, dtype=torch.complex64)
        hermitian_matrix = A @ A.conj().T  # Guaranteed to be positive definite
        
        # Ensure it's positive definite
        eigenvalues = torch.linalg.eigvals(hermitian_matrix)
        assert torch.all(eigenvalues.real > 0)
        
        # Embed to real representation
        real_embedded = embedder.embed_matrix(hermitian_matrix)
        
        # Check that real embedding is positive definite
        real_eigenvalues = torch.linalg.eigvals(real_embedded)
        assert torch.all(real_eigenvalues.real > -1e-12)  # Allow for numerical errors
    
    def test_embedding_linearity(self):
        """Test linearity of embedding operation."""
        embedder = ComplexToRealEmbedding(validate_properties=False)
        
        # Create complex matrices
        A = torch.randn(3, 3, dtype=torch.complex64)
        B = torch.randn(3, 3, dtype=torch.complex64)
        
        # Test embedding linearity: embed(αA + βB) = α*embed(A) + β*embed(B)
        alpha, beta = 2.0, 3.0
        
        linear_combination = alpha * A + beta * B
        embedded_combination = embedder.embed_matrix(linear_combination)
        
        embedded_A = embedder.embed_matrix(A)
        embedded_B = embedder.embed_matrix(B)
        combination_of_embedded = alpha * embedded_A + beta * embedded_B
        
        assert torch.allclose(embedded_combination, combination_of_embedded, atol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__])
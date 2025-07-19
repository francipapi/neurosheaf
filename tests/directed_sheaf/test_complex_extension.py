"""Unit tests for ComplexStalkExtender class.

Tests the mathematical correctness of complex extension operations:
- Real to complex stalk extension
- Structure preservation
- Whitened property maintenance
- Integration with existing sheaf infrastructure
"""

import pytest
import torch
import numpy as np
import networkx as nx
from typing import Dict, Any

from neurosheaf.directed_sheaf.core.complex_extension import ComplexStalkExtender
from neurosheaf.sheaf.data_structures import Sheaf


class TestComplexStalkExtender:
    """Test suite for ComplexStalkExtender class."""
    
    def test_initialization(self):
        """Test initialization of ComplexStalkExtender."""
        extender = ComplexStalkExtender()
        
        assert extender.validate_extension is True
        assert extender.tolerance == 1e-12
        
        # Test with custom parameters
        extender_custom = ComplexStalkExtender(validate_extension=False, tolerance=1e-8)
        assert extender_custom.validate_extension is False
        assert extender_custom.tolerance == 1e-8
    
    def test_extend_stalk_basic(self):
        """Test basic stalk extension functionality."""
        extender = ComplexStalkExtender()
        
        # Test with identity matrix (typical whitened stalk)
        real_stalk = torch.eye(3, dtype=torch.float32)
        complex_stalk = extender.extend_stalk(real_stalk)
        
        # Check properties
        assert complex_stalk.is_complex()
        assert complex_stalk.shape == real_stalk.shape
        assert complex_stalk.dtype == torch.complex64
        
        # Check that real part equals original
        assert torch.allclose(complex_stalk.real, real_stalk)
        
        # Check that imaginary part is zero
        assert torch.allclose(complex_stalk.imag, torch.zeros_like(real_stalk))
    
    def test_extend_stalk_validation(self):
        """Test validation during stalk extension."""
        extender = ComplexStalkExtender(validate_extension=True, tolerance=1e-10)
        
        # Test with random real matrix
        real_stalk = torch.randn(4, 4, dtype=torch.float32)
        complex_stalk = extender.extend_stalk(real_stalk)
        
        # Validation should pass
        assert complex_stalk.is_complex()
        assert torch.allclose(complex_stalk.real, real_stalk, atol=1e-10)
        assert torch.allclose(complex_stalk.imag, torch.zeros_like(real_stalk), atol=1e-10)
    
    def test_extend_stalk_already_complex(self):
        """Test extension of already complex stalk."""
        extender = ComplexStalkExtender()
        
        # Create complex stalk
        real_part = torch.randn(3, 3, dtype=torch.float32)
        imag_part = torch.randn(3, 3, dtype=torch.float32)
        complex_stalk = torch.complex(real_part, imag_part)
        
        # Extension should return as-is
        result = extender.extend_stalk(complex_stalk)
        
        assert torch.allclose(result, complex_stalk)
        assert result.is_complex()
    
    def test_extend_stalk_invalid_input(self):
        """Test error handling for invalid inputs."""
        extender = ComplexStalkExtender()
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            extender.extend_stalk("not a tensor")
        
        # Test with integer tensor
        int_tensor = torch.randint(0, 10, (3, 3), dtype=torch.int32)
        with pytest.raises(ValueError, match="Input must be real floating-point tensor"):
            extender.extend_stalk(int_tensor)
    
    def test_extend_sheaf_stalks_basic(self):
        """Test extension of complete sheaf stalks."""
        extender = ComplexStalkExtender()
        
        # Create mock sheaf
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        stalks = {
            'a': torch.eye(3, dtype=torch.float32),
            'b': torch.eye(4, dtype=torch.float32),
            'c': torch.eye(2, dtype=torch.float32)
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks)
        
        # Extend stalks
        complex_stalks = extender.extend_sheaf_stalks(sheaf)
        
        # Check results
        assert len(complex_stalks) == 3
        assert set(complex_stalks.keys()) == {'a', 'b', 'c'}
        
        for node, complex_stalk in complex_stalks.items():
            original_stalk = stalks[node]
            
            assert complex_stalk.is_complex()
            assert complex_stalk.shape == original_stalk.shape
            assert torch.allclose(complex_stalk.real, original_stalk)
            assert torch.allclose(complex_stalk.imag, torch.zeros_like(original_stalk))
    
    def test_extend_sheaf_stalks_empty(self):
        """Test extension of empty sheaf."""
        extender = ComplexStalkExtender()
        
        # Create empty sheaf
        sheaf = Sheaf(poset=nx.DiGraph(), stalks={})
        
        # Extension should return empty dict
        complex_stalks = extender.extend_sheaf_stalks(sheaf)
        
        assert len(complex_stalks) == 0
        assert isinstance(complex_stalks, dict)
    
    def test_extend_sheaf_stalks_invalid_input(self):
        """Test error handling for invalid sheaf input."""
        extender = ComplexStalkExtender()
        
        # Test with non-Sheaf input
        with pytest.raises(ValueError, match="Input must be a Sheaf instance"):
            extender.extend_sheaf_stalks("not a sheaf")
    
    def test_extend_with_metadata(self):
        """Test extension with metadata generation."""
        extender = ComplexStalkExtender()
        
        # Create mock sheaf
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b')])
        
        stalks = {
            'a': torch.eye(3, dtype=torch.float32),
            'b': torch.eye(2, dtype=torch.float32)
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks)
        
        # Extend with metadata
        complex_stalks, metadata = extender.extend_with_metadata(sheaf)
        
        # Check complex stalks
        assert len(complex_stalks) == 2
        assert all(stalk.is_complex() for stalk in complex_stalks.values())
        
        # Check metadata
        assert metadata['extension_method'] == 'complex_tensor_extension'
        assert metadata['num_stalks_extended'] == 2
        assert metadata['memory_overhead_factor'] == 2.0
        assert metadata['validation_passed'] is True
        assert 'original_dimensions' in metadata
        assert 'complex_dimensions' in metadata
        assert 'stalk_analysis' in metadata
        
        # Check stalk analysis
        for node in ['a', 'b']:
            analysis = metadata['stalk_analysis'][node]
            assert analysis['dimension_preserved'] is True
            assert analysis['structure_preserved']['shapes_match'] is True
            assert analysis['structure_preserved']['real_part_preserved'] is True
            assert analysis['structure_preserved']['imaginary_part_zero'] is True
    
    def test_create_complex_identity_stalk(self):
        """Test creation of complex identity stalks."""
        extender = ComplexStalkExtender()
        
        # Create complex identity stalk
        complex_identity = extender.create_complex_identity_stalk(4)
        
        assert complex_identity.is_complex()
        assert complex_identity.shape == (4, 4)
        assert complex_identity.dtype == torch.complex64
        
        # Check it's identity matrix
        expected_identity = torch.eye(4, dtype=torch.complex64)
        assert torch.allclose(complex_identity, expected_identity)
        
        # Test with different dtype
        complex_identity_128 = extender.create_complex_identity_stalk(3, torch.complex128)
        assert complex_identity_128.dtype == torch.complex128
    
    def test_validate_whitened_property(self):
        """Test validation of whitened properties."""
        extender = ComplexStalkExtender()
        
        # Test with complex identity (should be whitened)
        complex_identity = torch.eye(3, dtype=torch.complex64)
        validation = extender.validate_whitened_property(complex_identity)
        
        assert validation['is_whitened'] is True
        assert validation['max_deviation'] < 1e-12
        assert validation['dimension'] == 3
        assert validation['real_part_identity'] < 1e-12
        assert validation['imaginary_part_zero'] < 1e-12
        
        # Test with non-identity matrix (should not be whitened)
        non_identity = torch.randn(3, 3, dtype=torch.complex64)
        validation = extender.validate_whitened_property(non_identity)
        
        assert validation['is_whitened'] is False
        assert validation['max_deviation'] > 1e-12
        
        # Test with non-square matrix
        non_square = torch.randn(2, 3, dtype=torch.complex64)
        validation = extender.validate_whitened_property(non_square)
        
        assert validation['is_whitened'] is False
        assert 'error' in validation
        assert 'not square matrix' in validation['error']
    
    def test_validate_whitened_property_invalid_input(self):
        """Test error handling for whitened property validation."""
        extender = ComplexStalkExtender()
        
        # Test with real tensor
        real_tensor = torch.randn(3, 3, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="Input must be a complex tensor"):
            extender.validate_whitened_property(real_tensor)
    
    def test_structure_preservation_check(self):
        """Test internal structure preservation checking."""
        extender = ComplexStalkExtender()
        
        # Create real stalk
        real_stalk = torch.randn(3, 3, dtype=torch.float32)
        
        # Create proper complex extension
        complex_stalk = torch.complex(real_stalk, torch.zeros_like(real_stalk))
        
        # Check structure preservation
        analysis = extender._check_structure_preservation(real_stalk, complex_stalk)
        
        assert analysis['shapes_match'] is True
        assert analysis['real_part_preserved'] is True
        assert analysis['imaginary_part_zero'] is True
        assert analysis['max_real_diff'] < 1e-12
        assert analysis['max_imag_abs'] < 1e-12
        
        # Test with corrupted complex stalk
        corrupted_stalk = torch.complex(real_stalk + 0.1, torch.ones_like(real_stalk))
        analysis = extender._check_structure_preservation(real_stalk, corrupted_stalk)
        
        assert analysis['real_part_preserved'] is False
        assert analysis['imaginary_part_zero'] is False
        assert analysis['max_real_diff'] > 1e-12
        assert analysis['max_imag_abs'] > 1e-12
    
    def test_validation_disabled(self):
        """Test behavior when validation is disabled."""
        extender = ComplexStalkExtender(validate_extension=False)
        
        # Extension should work without validation
        real_stalk = torch.randn(3, 3, dtype=torch.float32)
        complex_stalk = extender.extend_stalk(real_stalk)
        
        assert complex_stalk.is_complex()
        assert torch.allclose(complex_stalk.real, real_stalk)
        assert torch.allclose(complex_stalk.imag, torch.zeros_like(real_stalk))
    
    def test_tolerance_sensitivity(self):
        """Test behavior with different tolerance values."""
        # Test with very strict tolerance
        strict_extender = ComplexStalkExtender(tolerance=1e-15)
        
        real_stalk = torch.eye(3, dtype=torch.float32)
        complex_stalk = strict_extender.extend_stalk(real_stalk)
        
        # Should still work for exact extension
        assert complex_stalk.is_complex()
        
        # Test with relaxed tolerance
        relaxed_extender = ComplexStalkExtender(tolerance=1e-6)
        
        complex_stalk = relaxed_extender.extend_stalk(real_stalk)
        assert complex_stalk.is_complex()
    
    def test_memory_overhead_calculation(self):
        """Test memory overhead calculation in metadata."""
        extender = ComplexStalkExtender()
        
        # Create sheaf with known dimensions
        poset = nx.DiGraph()
        poset.add_node('a')
        
        stalks = {
            'a': torch.eye(10, dtype=torch.float32)  # 10*10 = 100 real elements
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks)
        
        # Extend with metadata
        complex_stalks, metadata = extender.extend_with_metadata(sheaf)
        
        # Check memory calculations
        assert metadata['total_real_elements'] == 100
        assert metadata['total_complex_elements'] == 100  # Same number of elements
        assert metadata['memory_overhead_factor'] == 2.0  # But 2x memory usage
        
        # Verify actual memory usage
        real_memory = stalks['a'].numel() * stalks['a'].element_size()
        complex_memory = complex_stalks['a'].numel() * complex_stalks['a'].element_size()
        
        assert complex_memory == 2 * real_memory  # Complex uses 2x memory


class TestComplexStalkExtenderIntegration:
    """Integration tests for ComplexStalkExtender with existing infrastructure."""
    
    def test_integration_with_whitened_stalks(self):
        """Test integration with whitened stalks from existing infrastructure."""
        extender = ComplexStalkExtender()
        
        # Create whitened stalks (identity matrices)
        whitened_stalks = {
            'layer1': torch.eye(5, dtype=torch.float32),
            'layer2': torch.eye(3, dtype=torch.float32),
            'layer3': torch.eye(4, dtype=torch.float32)
        }
        
        # Create sheaf
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        
        sheaf = Sheaf(poset=poset, stalks=whitened_stalks)
        
        # Extend to complex
        complex_stalks = extender.extend_sheaf_stalks(sheaf)
        
        # Verify whitened properties are preserved
        for node, complex_stalk in complex_stalks.items():
            validation = extender.validate_whitened_property(complex_stalk)
            assert validation['is_whitened'] is True
            assert validation['real_part_identity'] < 1e-12
            assert validation['imaginary_part_zero'] < 1e-12
    
    def test_performance_characteristics(self):
        """Test performance characteristics of complex extension."""
        extender = ComplexStalkExtender()
        
        # Test with larger matrices
        large_stalk = torch.eye(100, dtype=torch.float32)
        
        # Extension should be fast
        import time
        start_time = time.time()
        complex_stalk = extender.extend_stalk(large_stalk)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 1.0  # Less than 1 second
        
        # Check memory usage
        real_memory = large_stalk.numel() * large_stalk.element_size()
        complex_memory = complex_stalk.numel() * complex_stalk.element_size()
        
        assert complex_memory == 2 * real_memory


if __name__ == '__main__':
    pytest.main([__file__])
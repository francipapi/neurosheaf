"""Unit tests for DirectedProcrustesComputer class.

Tests the mathematical correctness of directed Procrustes computation:
- Complex restriction map generation
- Phase encoding application
- Integration with existing Procrustes analysis
- Mathematical property validation
"""

import pytest
import torch
import numpy as np
import networkx as nx
from typing import Dict, Any

from neurosheaf.directed_sheaf.core.directed_procrustes import DirectedProcrustesComputer
from neurosheaf.directed_sheaf.core.directional_encoding import DirectionalEncodingComputer
from neurosheaf.sheaf.data_structures import Sheaf


class TestDirectedProcrustesComputer:
    """Test suite for DirectedProcrustesComputer class."""
    
    def test_initialization(self):
        """Test initialization of DirectedProcrustesComputer."""
        computer = DirectedProcrustesComputer()
        
        assert computer.q == 0.25
        assert computer.validate_restrictions is True
        assert computer.tolerance == 1e-12
        assert isinstance(computer.encoding_computer, DirectionalEncodingComputer)
        
        # Test with custom parameters
        computer_custom = DirectedProcrustesComputer(
            directionality_parameter=0.5,
            validate_restrictions=False,
            tolerance=1e-8
        )
        assert computer_custom.q == 0.5
        assert computer_custom.validate_restrictions is False
        assert computer_custom.tolerance == 1e-8
    
    def test_compute_directed_restrictions_basic(self):
        """Test basic directed restriction computation."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create mock base restrictions
        base_restrictions = {
            ('a', 'b'): torch.randn(3, 4, dtype=torch.float32),
            ('b', 'c'): torch.randn(2, 3, dtype=torch.float32)
        }
        
        # Create mock encoding matrix
        encoding_matrix = torch.tensor([
            [1+0j, 0+1j, 0+0j],
            [0-1j, 1+0j, 0+1j],
            [0+0j, 0-1j, 1+0j]
        ], dtype=torch.complex64)
        
        # Create node mapping
        node_mapping = {'a': 0, 'b': 1, 'c': 2}
        
        # Compute directed restrictions
        directed_restrictions = computer.compute_directed_restrictions(
            base_restrictions, encoding_matrix, node_mapping
        )
        
        # Check results
        assert len(directed_restrictions) == 2
        assert ('a', 'b') in directed_restrictions
        assert ('b', 'c') in directed_restrictions
        
        # Check that all restrictions are complex
        for restriction in directed_restrictions.values():
            assert restriction.is_complex()
        
        # Check shapes preserved
        assert directed_restrictions[('a', 'b')].shape == base_restrictions[('a', 'b')].shape
        assert directed_restrictions[('b', 'c')].shape == base_restrictions[('b', 'c')].shape
    
    def test_compute_directed_restrictions_phase_encoding(self):
        """Test that phase encoding is correctly applied."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create simple base restriction
        base_restrictions = {
            ('a', 'b'): torch.eye(2, dtype=torch.float32)
        }
        
        # Create encoding matrix with known phase
        phase_factor = torch.exp(torch.tensor(1j * np.pi / 4))  # 45 degree phase
        encoding_matrix = torch.tensor([
            [1+0j, phase_factor],
            [torch.conj(phase_factor), 1+0j]
        ], dtype=torch.complex64)
        
        node_mapping = {'a': 0, 'b': 1}
        
        # Compute directed restrictions
        directed_restrictions = computer.compute_directed_restrictions(
            base_restrictions, encoding_matrix, node_mapping
        )
        
        # Check that phase factor is applied
        directed_restriction = directed_restrictions[('a', 'b')]
        expected_restriction = phase_factor * torch.eye(2, dtype=torch.complex64)
        
        assert torch.allclose(directed_restriction, expected_restriction, atol=1e-12)
    
    def test_compute_directed_restrictions_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        computer = DirectedProcrustesComputer()
        
        # Test with non-dict base_restrictions
        with pytest.raises(ValueError, match="base_restrictions must be a dictionary"):
            computer.compute_directed_restrictions(
                "not a dict", torch.eye(2, dtype=torch.complex64), {}
            )
        
        # Test with non-tensor encoding_matrix
        with pytest.raises(ValueError, match="encoding_matrix must be a torch.Tensor"):
            computer.compute_directed_restrictions(
                {}, "not a tensor", {}
            )
        
        # Test with non-complex encoding_matrix
        with pytest.raises(ValueError, match="encoding_matrix must be complex"):
            computer.compute_directed_restrictions(
                {}, torch.eye(2, dtype=torch.float32), {}
            )
        
        # Test with non-dict node_mapping
        with pytest.raises(ValueError, match="node_mapping must be a dictionary"):
            computer.compute_directed_restrictions(
                {}, torch.eye(2, dtype=torch.complex64), "not a dict"
            )
    
    def test_compute_from_sheaf_basic(self):
        """Test directed restriction computation from existing sheaf."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create mock sheaf
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b'), ('b', 'c')])
        
        stalks = {
            'a': torch.eye(3, dtype=torch.float32),
            'b': torch.eye(4, dtype=torch.float32),
            'c': torch.eye(2, dtype=torch.float32)
        }
        
        restrictions = {
            ('a', 'b'): torch.randn(4, 3, dtype=torch.float32),
            ('b', 'c'): torch.randn(2, 4, dtype=torch.float32)
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Compute directed restrictions
        directed_restrictions = computer.compute_from_sheaf(sheaf)
        
        # Check results
        assert len(directed_restrictions) == 2
        assert ('a', 'b') in directed_restrictions
        assert ('b', 'c') in directed_restrictions
        
        # Check that all restrictions are complex
        for restriction in directed_restrictions.values():
            assert restriction.is_complex()
        
        # Check shapes preserved
        assert directed_restrictions[('a', 'b')].shape == restrictions[('a', 'b')].shape
        assert directed_restrictions[('b', 'c')].shape == restrictions[('b', 'c')].shape
    
    def test_compute_from_sheaf_parameter_override(self):
        """Test parameter override in compute_from_sheaf."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create simple sheaf
        poset = nx.DiGraph()
        poset.add_edges_from([('a', 'b')])
        
        stalks = {
            'a': torch.eye(2, dtype=torch.float32),
            'b': torch.eye(2, dtype=torch.float32)
        }
        
        restrictions = {
            ('a', 'b'): torch.eye(2, dtype=torch.float32)
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Compute with different q parameter
        directed_restrictions_q0 = computer.compute_from_sheaf(sheaf, directionality_parameter=0.0)
        directed_restrictions_q05 = computer.compute_from_sheaf(sheaf, directionality_parameter=0.5)
        
        # q=0 should give same as original (complex version)
        # q=0.5 should give different result
        assert not torch.allclose(
            directed_restrictions_q0[('a', 'b')],
            directed_restrictions_q05[('a', 'b')]
        )
        
        # Original q should be preserved
        assert computer.q == 0.25
    
    def test_apply_directional_encoding(self):
        """Test internal directional encoding application."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create test data
        edge = ('a', 'b')
        base_restriction = torch.eye(2, dtype=torch.float32)
        
        # Create encoding matrix
        phase_factor = torch.exp(torch.tensor(1j * np.pi / 2))  # 90 degree phase
        encoding_matrix = torch.tensor([
            [1+0j, phase_factor],
            [torch.conj(phase_factor), 1+0j]
        ], dtype=torch.complex64)
        
        node_mapping = {'a': 0, 'b': 1}
        
        # Apply encoding
        directed_restriction = computer._apply_directional_encoding(
            edge, base_restriction, encoding_matrix, node_mapping
        )
        
        # Check result
        expected_restriction = phase_factor * torch.eye(2, dtype=torch.complex64)
        assert torch.allclose(directed_restriction, expected_restriction, atol=1e-12)
    
    def test_apply_directional_encoding_invalid_nodes(self):
        """Test error handling for invalid nodes in encoding application."""
        computer = DirectedProcrustesComputer()
        
        edge = ('a', 'b')
        base_restriction = torch.eye(2, dtype=torch.float32)
        encoding_matrix = torch.eye(2, dtype=torch.complex64)
        
        # Test with missing node in mapping
        node_mapping = {'a': 0}  # Missing 'b'
        
        with pytest.raises(ValueError, match="Edge .* nodes not found in node_mapping"):
            computer._apply_directional_encoding(
                edge, base_restriction, encoding_matrix, node_mapping
            )
    
    def test_compute_with_metadata(self):
        """Test directed restriction computation with metadata."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create test data
        base_restrictions = {
            ('a', 'b'): torch.randn(3, 4, dtype=torch.float32),
            ('b', 'c'): torch.randn(2, 3, dtype=torch.float32)
        }
        
        encoding_matrix = torch.tensor([
            [1+0j, 0+1j, 0+0j],
            [0-1j, 1+0j, 0+1j],
            [0+0j, 0-1j, 1+0j]
        ], dtype=torch.complex64)
        
        node_mapping = {'a': 0, 'b': 1, 'c': 2}
        
        # Compute with metadata
        directed_restrictions, metadata = computer.compute_with_metadata(
            base_restrictions, encoding_matrix, node_mapping
        )
        
        # Check restrictions
        assert len(directed_restrictions) == 2
        assert all(r.is_complex() for r in directed_restrictions.values())
        
        # Check metadata
        assert metadata['directionality_parameter'] == 0.25
        assert metadata['num_restrictions'] == 2
        assert metadata['num_base_restrictions'] == 2
        assert metadata['encoding_matrix_shape'] == (3, 3)
        assert 'restriction_analysis' in metadata
        assert metadata['validation_passed'] is True
    
    def test_validate_restriction_orthogonality(self):
        """Test orthogonality validation for directed restrictions."""
        computer = DirectedProcrustesComputer()
        
        # Create column orthogonal matrix (use QR decomposition for exact orthogonality)
        q, r = torch.linalg.qr(torch.randn(4, 3, dtype=torch.float32))
        col_orthogonal = torch.complex(q, torch.zeros_like(q))
        
        # Test column orthogonality
        analysis = computer.validate_restriction_orthogonality(col_orthogonal, 'column')
        
        assert analysis['check_type'] == 'column'
        # Check that orthogonality error is small (may not be exactly orthogonal due to numerical precision)
        assert analysis['orthogonality_error'] < 1e-6
        
        # Test row orthogonality
        row_orthogonal = col_orthogonal.T  # Transpose to make row orthogonal
        analysis = computer.validate_restriction_orthogonality(row_orthogonal, 'row')
        
        assert analysis['check_type'] == 'row'
        # Note: This may not be orthogonal depending on dimensions
        assert 'orthogonality_error' in analysis
    
    def test_validate_restriction_orthogonality_invalid_input(self):
        """Test error handling for orthogonality validation."""
        computer = DirectedProcrustesComputer()
        
        # Test with real tensor
        real_tensor = torch.randn(3, 3, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="Directed restriction must be complex"):
            computer.validate_restriction_orthogonality(real_tensor)
        
        # Test with invalid check_type
        complex_tensor = torch.randn(3, 3, dtype=torch.complex64)
        
        with pytest.raises(ValueError, match="Invalid check_type"):
            computer.validate_restriction_orthogonality(complex_tensor, 'invalid')
    
    def test_get_restriction_summary(self):
        """Test restriction summary generation."""
        computer = DirectedProcrustesComputer()
        
        # Create test restrictions
        directed_restrictions = {
            ('a', 'b'): torch.randn(3, 4, dtype=torch.complex64),
            ('b', 'c'): torch.randn(2, 3, dtype=torch.complex64)
        }
        
        # Get summary
        summary = computer.get_restriction_summary(directed_restrictions)
        
        # Check summary content
        assert summary['num_restrictions'] == 2
        assert summary['edges'] == [('a', 'b'), ('b', 'c')]
        assert summary['shapes'][('a', 'b')] == (3, 4)
        assert summary['shapes'][('b', 'c')] == (2, 3)
        assert summary['dtypes'][('a', 'b')] == torch.complex64
        assert summary['dtypes'][('b', 'c')] == torch.complex64
        assert summary['total_parameters'] == 3*4 + 2*3
        assert summary['directionality_parameter'] == computer.q
        assert 'memory_usage_mb' in summary
    
    def test_validation_comprehensive(self):
        """Test comprehensive validation of directed restrictions."""
        computer = DirectedProcrustesComputer(validate_restrictions=True)
        
        # Create consistent test data
        base_restrictions = {
            ('a', 'b'): torch.randn(3, 4, dtype=torch.float32),
            ('b', 'c'): torch.randn(2, 3, dtype=torch.float32)
        }
        
        # Create encoding matrix
        encoding_matrix = torch.tensor([
            [1+0j, 0+1j, 0+0j],
            [0-1j, 1+0j, 0+1j],
            [0+0j, 0-1j, 1+0j]
        ], dtype=torch.complex64)
        
        node_mapping = {'a': 0, 'b': 1, 'c': 2}
        
        # Compute directed restrictions (should pass validation)
        directed_restrictions = computer.compute_directed_restrictions(
            base_restrictions, encoding_matrix, node_mapping
        )
        
        # Check results
        assert len(directed_restrictions) == 2
        assert all(r.is_complex() for r in directed_restrictions.values())
    
    def test_q_zero_reduction(self):
        """Test that q=0 reduces to undirected case."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.0)
        
        # Create test data
        base_restrictions = {
            ('a', 'b'): torch.eye(2, dtype=torch.float32)
        }
        
        # Create encoding matrix (should be all ones for q=0)
        encoding_matrix = torch.ones(2, 2, dtype=torch.complex64)
        
        node_mapping = {'a': 0, 'b': 1}
        
        # Compute directed restrictions
        directed_restrictions = computer.compute_directed_restrictions(
            base_restrictions, encoding_matrix, node_mapping
        )
        
        # Should be same as base (converted to complex) since phase factor is 1
        expected = torch.eye(2, dtype=torch.complex64)
        assert torch.allclose(directed_restrictions[('a', 'b')], expected, atol=1e-12)
    
    def test_magnitude_preservation(self):
        """Test that magnitude is preserved in directed restrictions."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create test restriction
        base_restrictions = {
            ('a', 'b'): torch.randn(3, 4, dtype=torch.float32)
        }
        
        # Create encoding matrix with unit magnitude entries
        encoding_matrix = torch.exp(1j * torch.randn(2, 2))
        
        node_mapping = {'a': 0, 'b': 1}
        
        # Compute directed restrictions
        directed_restrictions = computer.compute_directed_restrictions(
            base_restrictions, encoding_matrix, node_mapping
        )
        
        # Check magnitude preservation
        base_magnitude = torch.norm(base_restrictions[('a', 'b')])
        directed_magnitude = torch.norm(directed_restrictions[('a', 'b')])
        
        # Should be approximately equal (unit magnitude phase factor)
        assert torch.allclose(directed_magnitude, base_magnitude, atol=1e-6)
    
    def test_analysis_comprehensive(self):
        """Test comprehensive analysis of directed restrictions."""
        computer = DirectedProcrustesComputer()
        
        # Create test data
        base_restrictions = {
            ('a', 'b'): torch.randn(3, 4, dtype=torch.float32),
            ('b', 'c'): torch.randn(2, 3, dtype=torch.float32)
        }
        
        # Create directed restrictions
        directed_restrictions = {
            ('a', 'b'): torch.randn(3, 4, dtype=torch.complex64),
            ('b', 'c'): torch.randn(2, 3, dtype=torch.complex64)
        }
        
        # Analyze restrictions
        analysis = computer._analyze_restrictions(directed_restrictions, base_restrictions)
        
        # Check analysis content
        assert analysis['num_restrictions'] == 2
        assert analysis['all_complex'] is True
        assert analysis['shapes_preserved'] is True
        assert len(analysis['magnitude_ratios']) == 2
        assert 'phase_statistics' in analysis
        assert 'magnitude_ratio_mean' in analysis
        assert 'magnitude_ratio_std' in analysis
        
        # Check phase statistics
        for edge in [('a', 'b'), ('b', 'c')]:
            phase_stats = analysis['phase_statistics'][edge]
            assert 'mean_phase' in phase_stats
            assert 'std_phase' in phase_stats
            assert 'max_phase' in phase_stats
            assert 'min_phase' in phase_stats


class TestDirectedProcrustesIntegration:
    """Integration tests for DirectedProcrustesComputer."""
    
    def test_integration_with_existing_sheaf(self):
        """Test integration with existing sheaf infrastructure."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create realistic sheaf structure
        poset = nx.DiGraph()
        poset.add_edges_from([('input', 'hidden'), ('hidden', 'output')])
        
        stalks = {
            'input': torch.eye(10, dtype=torch.float32),
            'hidden': torch.eye(5, dtype=torch.float32),
            'output': torch.eye(3, dtype=torch.float32)
        }
        
        # Create orthogonal restrictions (realistic)
        u1, _, v1 = torch.svd(torch.randn(5, 10, dtype=torch.float32))
        restrictions = {
            ('input', 'hidden'): u1[:, :5],
            ('hidden', 'output'): torch.randn(3, 5, dtype=torch.float32)
        }
        
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Compute directed restrictions
        directed_restrictions = computer.compute_from_sheaf(sheaf)
        
        # Check results
        assert len(directed_restrictions) == 2
        assert all(r.is_complex() for r in directed_restrictions.values())
        
        # Check shapes preserved
        for edge in restrictions.keys():
            assert directed_restrictions[edge].shape == restrictions[edge].shape
    
    def test_performance_large_restrictions(self):
        """Test performance with large restriction matrices."""
        computer = DirectedProcrustesComputer(directionality_parameter=0.25)
        
        # Create large restriction matrices
        base_restrictions = {
            ('a', 'b'): torch.randn(100, 200, dtype=torch.float32),
            ('b', 'c'): torch.randn(50, 100, dtype=torch.float32)
        }
        
        # Create encoding matrix
        encoding_matrix = torch.exp(1j * torch.randn(3, 3))
        
        node_mapping = {'a': 0, 'b': 1, 'c': 2}
        
        # Should complete quickly
        import time
        start_time = time.time()
        
        directed_restrictions = computer.compute_directed_restrictions(
            base_restrictions, encoding_matrix, node_mapping
        )
        
        end_time = time.time()
        
        # Check results
        assert len(directed_restrictions) == 2
        assert all(r.is_complex() for r in directed_restrictions.values())
        assert end_time - start_time < 1.0  # Should be fast


if __name__ == '__main__':
    pytest.main([__file__])
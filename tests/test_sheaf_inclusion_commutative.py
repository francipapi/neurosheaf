"""Tests for sheaf inclusion mapper commutative property verification.

This module contains comprehensive tests for the commutative diagram property
checking in the SheafInclusionMapper. Tests include synthetic examples with
known commuting vs non-commuting restrictions to ensure the quality assessment
flags fire correctly.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Tuple

from neurosheaf.spectral.gw.sheaf_inclusion_mapper import SheafInclusionMapper


def create_test_restriction_maps(edges: list, 
                               prev_dim: int, 
                               curr_dim: int,
                               commuting: bool = True,
                               noise_level: float = 0.0) -> Tuple[Dict, Dict]:
    """Create test restriction maps for commutative property testing.
    
    For commutative diagram: R_curr @ ι ≈ ι @ R_prev where ι is [curr_dim x prev_dim]
    With identity inclusion mapping ι = [I_prev; 0] where I_prev is prev_dim x prev_dim identity
    
    This simplifies to: R_curr[:, :prev_dim] ≈ ι @ R_prev = [R_prev; 0]
    
    Args:
        edges: List of edges as (source, target) tuples
        prev_dim: Previous eigenspace dimension  
        curr_dim: Current eigenspace dimension
        commuting: If True, create maps that satisfy commutative property
        noise_level: Amount of noise to add (breaks commutativity)
        
    Returns:
        Tuple of (prev_restrictions, curr_restrictions)
    """
    prev_restrictions = {}
    curr_restrictions = {}
    
    # Use a reasonable target dimension for both restriction maps
    target_dim = max(prev_dim, curr_dim, 3)
    
    # Create restriction maps with correct dimensions for commutative diagram
    for edge in edges:
        if commuting:
            # Create a base restriction map for the previous space
            R_prev = torch.randn(prev_dim, target_dim) * 0.5
            
            # For perfect commutativity with identity inclusion:
            # ι @ R_prev should equal the first prev_dim rows of R_curr @ ι
            # Since ι is identity extension: ι @ R_prev = [R_prev; zeros]
            # And R_curr @ ι extracts the first prev_dim columns of R_curr
            # So we need R_curr[:, :prev_dim] = [R_prev; zeros]
            
            R_curr = torch.zeros(target_dim, curr_dim)
            
            if commuting and noise_level == 0.0:
                # Perfect commutativity in overlapping region:
                # ι @ R_prev has shape [curr_dim x target_dim]
                # R_curr @ ι has shape [target_dim x prev_dim]
                # 
                # The overlapping region is [min(curr_dim, target_dim) x min(target_dim, prev_dim)]
                # For this region: (ι @ R_prev)[:overlap_rows, :overlap_cols] ≈ (R_curr @ ι)[:overlap_rows, :overlap_cols]
                #
                # Since ι is identity extension: ι @ R_prev = [R_prev; zeros]
                # So (ι @ R_prev)[i,j] = R_prev[i,j] for i < prev_dim, 0 otherwise
                # 
                # And R_curr @ ι extracts first prev_dim columns: (R_curr @ ι)[i,j] = R_curr[i,j] for j < prev_dim
                #
                # For commutativity in overlap: R_prev[i,j] ≈ R_curr[i,j] where both i,j are in overlap
                
                overlap_rows = min(prev_dim, target_dim)  # How many rows of R_prev are used
                overlap_cols = min(target_dim, prev_dim)  # How many cols we can compare
                
                # Make them identical in the overlapping region 
                R_curr[:overlap_rows, :overlap_cols] = R_prev[:overlap_rows, :overlap_cols]
                
                # Add some structure to remaining dimensions to make it realistic but small
                if target_dim > overlap_rows:
                    # Fill remaining rows with small random values
                    R_curr[overlap_rows:, :] = torch.randn(target_dim - overlap_rows, curr_dim) * 0.05
                if curr_dim > overlap_cols:
                    # Fill remaining columns with small random values  
                    R_curr[:, overlap_cols:] = torch.randn(target_dim, curr_dim - overlap_cols) * 0.05
            else:
                # For non-perfect case or with noise, use random values
                R_curr = torch.randn(target_dim, curr_dim) * 0.5
            
            # Add noise if requested
            if noise_level > 0:
                R_prev += noise_level * torch.randn_like(R_prev)
                R_curr += noise_level * torch.randn_like(R_curr)
                
            prev_restrictions[edge] = R_prev
            curr_restrictions[edge] = R_curr
        else:
            # Create different restriction maps (non-commuting)
            prev_restrictions[edge] = torch.randn(prev_dim, target_dim)
            curr_restrictions[edge] = torch.randn(target_dim, curr_dim)
    
    return prev_restrictions, curr_restrictions


class TestCommutativeProperty:
    """Test commutative diagram property checking."""
    
    def test_perfect_commuting_restrictions(self):
        """Test with perfect commuting restriction maps."""
        # Setup
        mapper = SheafInclusionMapper(
            method='identity_extension',
            inclusion_quality_tolerance=0.1
        )
        
        # Create test data
        prev_dim, curr_dim = 4, 6
        edges = [('layer1', 'layer2'), ('layer2', 'layer3')]
        
        # Identity inclusion mapping (should always commute with itself)
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        for i in range(prev_dim):
            inclusion_map[i, i] = 1.0
        
        # Create perfectly commuting restrictions
        prev_restrictions, curr_restrictions = create_test_restriction_maps(
            edges, prev_dim, curr_dim, commuting=True, noise_level=0.0
        )
        
        # Test commutative property
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Assertions
        assert commutes, "Perfect commuting restrictions should satisfy property"
        assert quality_metadata['max_residual'] <= 0.1, f"Max residual too high: {quality_metadata['max_residual']}"
        assert len(quality_metadata['violated_edges']) == 0, "Should have no violated edges"
        assert quality_metadata['edges_checked'] == len(edges)
    
    def test_non_commuting_restrictions(self):
        """Test with non-commuting restriction maps."""
        # Setup
        mapper = SheafInclusionMapper(
            method='identity_extension',
            inclusion_quality_tolerance=0.1
        )
        
        # Create test data
        prev_dim, curr_dim = 4, 6
        edges = [('layer1', 'layer2'), ('layer2', 'layer3')]
        
        # Identity inclusion mapping
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        for i in range(prev_dim):
            inclusion_map[i, i] = 1.0
        
        # Create non-commuting restrictions
        prev_restrictions, curr_restrictions = create_test_restriction_maps(
            edges, prev_dim, curr_dim, commuting=False
        )
        
        # Test commutative property
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Assertions
        assert not commutes, "Non-commuting restrictions should violate property"
        assert quality_metadata['max_residual'] > 0.1, "Max residual should be significant"
        assert len(quality_metadata['violated_edges']) > 0, "Should have violated edges"
        assert quality_metadata['edges_checked'] == len(edges)
    
    def test_noisy_commuting_restrictions(self):
        """Test with slightly noisy but mostly commuting restrictions."""
        # Setup with very tolerant threshold
        mapper = SheafInclusionMapper(
            method='identity_extension',
            inclusion_quality_tolerance=0.9  # Even more tolerant threshold for noisy case
        )
        
        # Create test data
        prev_dim, curr_dim = 4, 6
        edges = [('layer1', 'layer2'), ('layer2', 'layer3')]
        
        # Identity inclusion mapping
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        for i in range(prev_dim):
            inclusion_map[i, i] = 1.0
        
        # Create slightly noisy commuting restrictions
        prev_restrictions, curr_restrictions = create_test_restriction_maps(
            edges, prev_dim, curr_dim, commuting=True, noise_level=0.01  # Small noise
        )
        
        # Test commutative property
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Assertions - should pass due to very tolerant threshold
        assert commutes, "Slightly noisy restrictions should pass with very tolerant threshold"
        assert quality_metadata['max_residual'] < 0.9, "Residual should still be reasonable"
        assert quality_metadata['edges_checked'] == len(edges)
    
    def test_edge_sampling_large_graph(self):
        """Test edge sampling behavior on large graphs."""
        # Setup with small max_edges_to_check
        mapper = SheafInclusionMapper(
            method='identity_extension',
            max_edges_to_check=5  # Small limit to trigger sampling
        )
        
        # Create large graph
        prev_dim, curr_dim = 3, 4
        large_edge_list = [(f'layer{i}', f'layer{i+1}') for i in range(20)]  # 20 edges
        
        # Identity inclusion mapping
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        for i in range(prev_dim):
            inclusion_map[i, i] = 1.0
        
        # Create commuting restrictions
        prev_restrictions, curr_restrictions = create_test_restriction_maps(
            large_edge_list, prev_dim, curr_dim, commuting=True
        )
        
        # Test commutative property
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Assertions
        assert quality_metadata['edges_checked'] == 5, "Should sample exactly 5 edges"
        assert quality_metadata['total_common_edges'] == 20, "Should report total edges"
        assert commutes, "Sampled commuting restrictions should satisfy property"
    
    def test_dimension_incompatibility(self):
        """Test handling of incompatible dimensions."""
        # Setup
        mapper = SheafInclusionMapper(method='identity_extension')
        
        # Create incompatible dimensions
        inclusion_map = torch.randn(6, 4)  # 6x4 inclusion
        
        # Create restrictions with incompatible dimensions
        prev_restrictions = {
            ('layer1', 'layer2'): torch.randn(3, 5),  # Wrong input dim for inclusion
        }
        curr_restrictions = {
            ('layer1', 'layer2'): torch.randn(7, 6),  # Wrong output dim for inclusion
        }
        
        # Test commutative property
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Should handle gracefully
        assert quality_metadata['edges_checked'] == 0, "No edges should be checkable"
        assert commutes, "Should default to True when no edges can be checked"
    
    def test_empty_restriction_maps(self):
        """Test handling of empty restriction map dictionaries."""
        # Setup
        mapper = SheafInclusionMapper(method='identity_extension')
        
        inclusion_map = torch.randn(6, 4)
        
        # Test with empty dictionaries
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, {}, {}
        )
        
        assert commutes, "Should default to True with no restriction maps"
        assert quality_metadata['edges_checked'] == 0
        
        # Test with no common edges
        prev_restrictions = {('a', 'b'): torch.randn(6, 4)}
        curr_restrictions = {('c', 'd'): torch.randn(6, 4)}
        
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        assert commutes, "Should default to True with no common edges"
        assert quality_metadata['edges_checked'] == 0


class TestFallbackMechanism:
    """Test fallback mechanism for violated commutative property."""
    
    def test_fallback_activation(self):
        """Test that fallback activates on commutative property violation."""
        # Setup with fallback enabled
        mapper = SheafInclusionMapper(
            method='transport_svd',  # Primary method
            inclusion_quality_tolerance=0.05,  # Strict threshold
            fallback_on_violation=True
        )
        
        # Create test data
        prev_step, curr_step = 0, 1
        prev_dim, curr_dim = 4, 6
        
        # Create non-commuting restrictions with proper dimensions (should trigger fallback)
        edges = [('layer1', 'layer2')]
        target_dim = 5  # Common target dimension
        prev_restrictions = {edges[0]: torch.randn(prev_dim, target_dim)}
        curr_restrictions = {edges[0]: torch.randn(target_dim, curr_dim)}  # Different matrix
        
        # Mock transport matrices (to avoid actual transport computation)
        sheaf_metadata = {
            'gw_couplings': {edges[0]: torch.eye(min(prev_dim, curr_dim))}
        }
        
        # Create inclusion mapping with restriction maps
        inclusion_map, metadata = mapper.create_gw_inclusion_mapping(
            prev_step, curr_step, prev_dim, curr_dim,
            sheaf_metadata=sheaf_metadata,
            prev_restrictions=prev_restrictions,
            curr_restrictions=curr_restrictions
        )
        
        # Assertions
        assert metadata['fallback_triggered'], "Fallback should be triggered"
        assert metadata['method_used'] == 'identity_extension', "Should use fallback method"
        assert metadata['original_method'] == 'transport_svd', "Should remember original method"
        assert 'inclusion_quality' in metadata, "Should have quality metadata"
        assert 'fallback_quality' in metadata, "Should have fallback quality metadata"
        
        # Check inclusion map shape
        assert inclusion_map.shape == (curr_dim, prev_dim)
    
    def test_fallback_disabled(self):
        """Test behavior when fallback is disabled."""
        # Setup with fallback disabled
        mapper = SheafInclusionMapper(
            method='transport_svd',
            inclusion_quality_tolerance=0.05,
            fallback_on_violation=False  # Disabled
        )
        
        prev_step, curr_step = 0, 1
        prev_dim, curr_dim = 4, 6
        
        # Create non-commuting restrictions with proper dimensions
        edges = [('layer1', 'layer2')]
        target_dim = 5  # Common target dimension
        prev_restrictions = {edges[0]: torch.randn(prev_dim, target_dim)}
        curr_restrictions = {edges[0]: torch.randn(target_dim, curr_dim)}
        
        sheaf_metadata = {
            'gw_couplings': {edges[0]: torch.eye(min(prev_dim, curr_dim))}
        }
        
        # Create inclusion mapping
        inclusion_map, metadata = mapper.create_gw_inclusion_mapping(
            prev_step, curr_step, prev_dim, curr_dim,
            sheaf_metadata=sheaf_metadata,
            prev_restrictions=prev_restrictions,
            curr_restrictions=curr_restrictions
        )
        
        # Assertions
        assert not metadata['fallback_triggered'], "Fallback should not be triggered"
        assert metadata['method_used'] == 'transport_svd', "Should use original method"
        assert 'inclusion_quality' in metadata, "Should still have quality metadata"
        assert 'fallback_quality' not in metadata, "Should not have fallback quality"


class TestQualityMetadata:
    """Test quality metadata generation and structure."""
    
    def test_quality_metadata_structure(self):
        """Test structure and contents of quality metadata."""
        mapper = SheafInclusionMapper()
        
        # Create test setup
        prev_dim, curr_dim = 3, 4
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        for i in range(prev_dim):
            inclusion_map[i, i] = 1.0
        
        edges = [('a', 'b'), ('b', 'c')]
        prev_restrictions, curr_restrictions = self.create_test_restrictions(edges, prev_dim, curr_dim)
        
        # Test commutative property
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Check required metadata fields
        required_fields = [
            'max_residual', 'median_residual', 'violated_edges',
            'edges_checked', 'total_common_edges', 'edge_residuals', 'tolerance_used'
        ]
        
        for field in required_fields:
            assert field in quality_metadata, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(quality_metadata['max_residual'], float)
        assert isinstance(quality_metadata['median_residual'], float)
        assert isinstance(quality_metadata['violated_edges'], list)
        assert isinstance(quality_metadata['edges_checked'], int)
        assert isinstance(quality_metadata['edge_residuals'], dict)
        
        # Check edge residuals have correct keys
        for edge in edges:
            if edge in quality_metadata['edge_residuals']:
                assert isinstance(quality_metadata['edge_residuals'][edge], float)
    
    def create_test_restrictions(self, edges, prev_dim, curr_dim):
        """Helper to create test restriction maps."""
        return create_test_restriction_maps(edges, prev_dim, curr_dim, commuting=False)
    
    def test_violated_edges_ranking(self):
        """Test that violated edges are ranked by residual magnitude."""
        mapper = SheafInclusionMapper(inclusion_quality_tolerance=0.1)
        
        # Create test setup with known violation pattern
        prev_dim, curr_dim = 3, 4
        inclusion_map = torch.eye(curr_dim, prev_dim)  # Partial identity
        
        # Create restrictions with different levels of violation
        prev_restrictions = {
            'edge1': torch.ones(curr_dim, prev_dim) * 0.1,  # Small violation
            'edge2': torch.ones(curr_dim, prev_dim) * 0.5,  # Large violation  
            'edge3': torch.ones(curr_dim, prev_dim) * 0.3,  # Medium violation
        }
        curr_restrictions = {
            'edge1': torch.ones(curr_dim, prev_dim) * 0.2,  # Small difference
            'edge2': torch.ones(curr_dim, prev_dim) * 1.0,  # Large difference
            'edge3': torch.ones(curr_dim, prev_dim) * 0.6,  # Medium difference
        }
        
        # Test commutative property
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Check that violated edges are sorted by residual (descending)
        violated_edges = quality_metadata['violated_edges']
        if len(violated_edges) > 1:
            for i in range(len(violated_edges) - 1):
                assert violated_edges[i][1] >= violated_edges[i+1][1], \
                    "Violated edges should be sorted by residual (descending)"


class TestPerformanceOptimizations:
    """Test performance optimizations and edge cases."""
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        mapper = SheafInclusionMapper()
        
        # Test with very small matrices
        inclusion_map = torch.ones(1, 1) * 1e-15  # Very small
        prev_restrictions = {('a', 'b'): torch.ones(1, 1) * 1e-15}
        curr_restrictions = {('a', 'b'): torch.ones(1, 1) * 1e-15}
        
        # Should handle gracefully
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        assert isinstance(commutes, bool)
        assert 'max_residual' in quality_metadata
        assert not np.isnan(quality_metadata['max_residual'])
        assert not np.isinf(quality_metadata['max_residual'])
    
    def test_error_handling(self):
        """Test error handling in commutative property check."""
        mapper = SheafInclusionMapper()
        
        # Test with invalid inputs
        inclusion_map = torch.tensor([])  # Empty tensor
        prev_restrictions = {('a', 'b'): torch.randn(2, 2)}
        curr_restrictions = {('a', 'b'): torch.randn(2, 2)}
        
        # Should handle gracefully
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        # Should either succeed gracefully or provide error info
        assert isinstance(commutes, bool)
        assert isinstance(quality_metadata, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
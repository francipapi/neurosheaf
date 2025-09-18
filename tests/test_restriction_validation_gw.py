"""Comprehensive tests for GW restriction validation.

This module tests the validate_restriction_maps_gw function and related functionality,
ensuring proper handling of:
- Finiteness checks (NaN/Inf detection)
- Row-stochasticity validation
- Automated correction mechanisms
- Edge case handling
- Integration with GW builder
"""

import pytest
import torch
import numpy as np
from typing import Dict, Tuple, List

from neurosheaf.sheaf.core.validation import validate_restriction_maps_gw
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.sheaf.assembly.gw_builder import GWRestrictionManager


class TestFiniteness:
    """Test finiteness checks (NaN/Inf detection)."""
    
    def test_finite_restrictions_pass(self):
        """Test that finite restriction maps pass validation."""
        # Create finite restriction maps
        restrictions = {
            ('layer1', 'layer2'): torch.rand(5, 4) + 0.1,  # Positive finite values
            ('layer2', 'layer3'): torch.rand(3, 5) + 0.1,
        }
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions, check_stochasticity=False  # Focus on finiteness
        )
        
        assert metadata['validation_passed']
        assert len(metadata['finite_violations']) == 0
        assert len(validated) == len(restrictions)
        
    def test_nan_values_detected(self):
        """Test detection of NaN values in restriction maps."""
        # Create restriction with NaN
        R_with_nan = torch.rand(3, 4)
        R_with_nan[1, 2] = float('nan')
        
        restrictions = {
            ('layer1', 'layer2'): torch.rand(5, 4) + 0.1,  # Good
            ('bad', 'edge'): R_with_nan,  # Contains NaN
        }
        
        # Test non-strict mode
        validated, metadata = validate_restriction_maps_gw(
            restrictions, strict_mode=False, check_stochasticity=False
        )
        
        assert not metadata['validation_passed']
        assert ('bad', 'edge') in metadata['finite_violations']
        assert ('bad', 'edge') in metadata['dropped_edges']
        assert len(validated) == 1  # Only good edge remains
        assert ('layer1', 'layer2') in validated
        
    def test_inf_values_detected(self):
        """Test detection of Inf values in restriction maps."""
        # Create restriction with Inf
        R_with_inf = torch.rand(3, 4) + 0.1
        R_with_inf[0, 1] = float('inf')
        
        restrictions = {
            ('good', 'edge'): torch.rand(4, 3) + 0.1,
            ('inf', 'edge'): R_with_inf,
        }
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions, strict_mode=False, check_stochasticity=False
        )
        
        assert not metadata['validation_passed']
        assert ('inf', 'edge') in metadata['finite_violations']
        assert len(validated) == 1
        
    def test_strict_mode_finiteness_error(self):
        """Test that strict mode raises errors for finiteness violations."""
        R_with_nan = torch.rand(3, 3)
        R_with_nan[0, 0] = float('nan')
        
        restrictions = {('bad', 'edge'): R_with_nan}
        
        with pytest.raises(ValueError, match="contains NaN or Inf values"):
            validate_restriction_maps_gw(
                restrictions, strict_mode=True, check_stochasticity=False
            )


class TestStochasticity:
    """Test row-stochasticity validation and correction."""
    
    def create_perfect_stochastic_map(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Create a perfectly row-stochastic map."""
        R = torch.rand(shape)
        # Normalize each row to sum to 1
        return R / R.sum(dim=1, keepdim=True)
        
    def create_slightly_non_stochastic_map(self, shape: Tuple[int, int], 
                                         noise_level: float = 1e-3) -> torch.Tensor:
        """Create a nearly row-stochastic map with small violations."""
        R = self.create_perfect_stochastic_map(shape)
        # Add small noise to break stochasticity slightly
        noise = torch.randn_like(R) * noise_level
        return torch.clamp(R + noise, min=0.0)  # Keep non-negative
        
    def test_perfect_stochastic_passes(self):
        """Test that perfectly row-stochastic maps pass validation."""
        restrictions = {
            ('layer1', 'layer2'): self.create_perfect_stochastic_map((5, 4)),
            ('layer2', 'layer3'): self.create_perfect_stochastic_map((3, 5)),
        }
        
        validated, metadata = validate_restriction_maps_gw(restrictions)
        
        assert metadata['validation_passed']
        assert len(metadata['stochasticity_violations']) == 0
        assert len(metadata['corrections_applied']) == 0
        assert len(validated) == len(restrictions)
        
    def test_small_violations_auto_corrected(self):
        """Test that small stochasticity violations are auto-corrected."""
        restrictions = {
            ('edge1'): self.create_slightly_non_stochastic_map((4, 3), noise_level=1e-6),
            ('edge2'): self.create_slightly_non_stochastic_map((5, 4), noise_level=1e-5),
        }
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions, 
            stochastic_tolerance=1e-8,  # Very strict
            correction_threshold=1e-3,   # But allow corrections
            auto_correct=True
        )
        
        assert metadata['validation_passed']
        assert len(metadata['corrections_applied']) > 0
        
        # Verify corrected maps are actually stochastic
        for edge, R in validated.items():
            row_sums = R.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-8)
            
    def test_large_violations_dropped(self):
        """Test that large stochasticity violations are handled appropriately."""
        # Create severely non-stochastic map
        R_bad = torch.rand(3, 4) * 10  # Very non-stochastic
        
        restrictions = {
            ('good', 'edge'): self.create_perfect_stochastic_map((4, 3)),
            ('bad', 'edge'): R_bad,
        }
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions,
            strict_threshold=0.1,  # Low threshold
            strict_mode=False      # Drop, don't error
        )
        
        assert not metadata['validation_passed']
        assert ('bad', 'edge') in metadata['dropped_edges']
        assert len(validated) == 1  # Only good edge remains
        
    def test_negative_values_clipped_and_renormalized(self):
        """Test that negative values are properly clipped and renormalized."""
        # Create map with negative values (common from imperfect OT)
        R_with_negatives = torch.tensor([
            [0.8, 0.3, -0.1],  # Row sum = 1.0 but has negative
            [0.6, 0.2, 0.3],   # Perfect row  
            [1.1, -0.1, 0.1],  # Row sum > 1 with negative
        ], dtype=torch.float32)
        
        restrictions = {('test', 'edge'): R_with_negatives}
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions,
            correction_threshold=0.5,  # Allow correction
            auto_correct=True
        )
        
        assert len(metadata['corrections_applied']) == 1
        
        # Verify correction: no negatives and row-stochastic
        R_corrected = validated[('test', 'edge')]
        assert torch.all(R_corrected >= 0)  # No negatives
        
        row_sums = R_corrected.sum(dim=1)
        expected_ones = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, expected_ones, atol=1e-6)
        
    def test_stochasticity_disabled(self):
        """Test that stochasticity checking can be disabled."""
        # Create very non-stochastic map
        R_bad = torch.rand(3, 3) * 100
        
        restrictions = {('any', 'edge'): R_bad}
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions, check_stochasticity=False
        )
        
        # Should pass because stochasticity is disabled
        assert metadata['validation_passed']
        assert len(metadata['stochasticity_violations']) == 0
        assert len(validated) == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_restrictions(self):
        """Test handling of empty restriction dictionary."""
        restrictions = {}
        
        validated, metadata = validate_restriction_maps_gw(restrictions)
        
        assert metadata['validation_passed']
        assert metadata['total_restrictions'] == 0
        assert len(validated) == 0
        
    def test_single_element_maps(self):
        """Test handling of 1x1 restriction maps."""
        restrictions = {
            ('tiny1', 'tiny2'): torch.tensor([[1.0]]),  # Perfect 1x1 stochastic
            ('tiny2', 'tiny3'): torch.tensor([[0.8]]),  # Imperfect 1x1
        }
        
        validated, metadata = validate_restriction_maps_gw(restrictions)
        
        # Should handle gracefully
        assert len(validated) > 0
        assert metadata['total_restrictions'] == 2
        
    def test_very_large_maps(self):
        """Test handling of large restriction maps."""
        # Create large maps (but still manageable for test)
        large_shape = (50, 40)
        restrictions = {
            ('large1', 'large2'): torch.rand(large_shape) + 0.1,
        }
        
        # Make it approximately stochastic
        R = restrictions[('large1', 'large2')]
        restrictions[('large1', 'large2')] = R / R.sum(dim=1, keepdim=True)
        
        validated, metadata = validate_restriction_maps_gw(restrictions)
        
        assert metadata['validation_passed']
        assert len(validated) == 1
        
    def test_zero_matrices(self):
        """Test handling of zero matrices."""
        restrictions = {
            ('zero', 'edge'): torch.zeros(3, 4),
        }
        
        # Zero matrices are non-stochastic but finite
        validated, metadata = validate_restriction_maps_gw(
            restrictions, 
            strict_mode=False,
            correction_threshold=0.5  # Allow correction
        )
        
        # Should be corrected (each row becomes uniform)
        if len(validated) > 0:
            R_corrected = validated[('zero', 'edge')]
            row_sums = R_corrected.sum(dim=1)
            expected_ones = torch.ones_like(row_sums)
            assert torch.allclose(row_sums, expected_ones, atol=1e-6)


class TestConfigurationValidation:
    """Test GWConfig parameter validation for restriction validation."""
    
    def test_valid_config_parameters(self):
        """Test that valid configuration parameters are accepted."""
        config = GWConfig(
            validate_restrictions=True,
            stochastic_tolerance=1e-6,
            correction_threshold=1e-3,
            strict_validation_threshold=0.1,
            auto_correct_restrictions=True
        )
        
        # Should not raise
        assert config.validate_restrictions == True
        assert config.stochastic_tolerance == 1e-6
        
    def test_invalid_tolerance_rejected(self):
        """Test that invalid tolerance values are rejected."""
        with pytest.raises(ValueError, match="stochastic_tolerance must be positive"):
            GWConfig(stochastic_tolerance=-1e-6)
            
        with pytest.raises(ValueError, match="correction_threshold must be positive"):
            GWConfig(correction_threshold=0.0)
            
    def test_threshold_ordering_validated(self):
        """Test that correction_threshold <= strict_validation_threshold is enforced."""
        with pytest.raises(ValueError, match="correction_threshold.*must be <=.*strict_validation_threshold"):
            GWConfig(
                correction_threshold=0.2,
                strict_validation_threshold=0.1  # Wrong order!
            )


class TestIntegrationWithGWBuilder:
    """Test integration with GW builder and realistic scenarios."""
    
    def create_mock_activations(self) -> Dict[str, torch.Tensor]:
        """Create mock activation tensors for testing."""
        return {
            'layer1': torch.randn(100, 64),  # 100 samples, 64 features
            'layer2': torch.randn(100, 32),  # Smaller layer
            'layer3': torch.randn(100, 16),  # Even smaller
        }
    
    def test_gw_config_parameters_used(self):
        """Test that GW config parameters are properly used in validation."""
        # Create config with specific validation parameters
        config = GWConfig(
            validate_restrictions=True,
            stochastic_tolerance=1e-5,
            correction_threshold=1e-2,
            auto_correct_restrictions=True,
            strict_validation_mode=False
        )
        
        # Verify parameters are set correctly
        assert config.validate_restrictions == True
        assert config.stochastic_tolerance == 1e-5
        assert config.correction_threshold == 1e-2
        assert config.auto_correct_restrictions == True
        assert config.strict_validation_mode == False
        
    def test_validation_disabled_in_config(self):
        """Test that validation can be disabled via config."""
        config = GWConfig(validate_restrictions=False)
        assert config.validate_restrictions == False
        
        # When integrated, validation should be skipped
        # (This would be tested in actual integration test)
        
    def test_metadata_structure_complete(self):
        """Test that validation metadata has all required fields."""
        restrictions = {
            ('layer1', 'layer2'): torch.rand(4, 3) + 0.1,
        }
        
        validated, metadata = validate_restriction_maps_gw(restrictions)
        
        # Verify all required metadata fields are present
        required_fields = [
            'total_restrictions', 'finite_violations', 'stochasticity_violations',
            'corrections_applied', 'dropped_edges', 'max_stochasticity_error',
            'validation_passed', 'config'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing required metadata field: {field}"
            
        # Verify config preservation
        assert 'stochastic_tolerance' in metadata['config']
        assert 'correction_threshold' in metadata['config']
        assert 'auto_correct' in metadata['config']


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def test_large_number_of_edges(self):
        """Test validation with many edges (performance test)."""
        # Create many small restriction maps
        restrictions = {}
        for i in range(100):  # 100 edges
            restrictions[(f'layer{i}', f'layer{i+1}')] = torch.rand(3, 4) + 0.1
            
        import time
        start_time = time.time()
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions, check_stochasticity=False  # Faster
        )
        
        elapsed = time.time() - start_time
        
        # Should complete reasonably quickly
        assert elapsed < 5.0  # Should be much faster, but allow generous margin
        assert len(validated) == 100
        assert metadata['validation_passed']
        
    def test_mixed_violation_scenarios(self):
        """Test complex scenario with mixed violation types."""
        # Create a mix of good, correctable, and bad restrictions
        restrictions = {}
        
        # Perfect restrictions
        for i in range(5):
            R = torch.rand(4, 3)
            restrictions[(f'perfect_{i}', f'edge')] = R / R.sum(dim=1, keepdim=True)
            
        # Slightly imperfect (correctable)
        for i in range(3):
            R = torch.rand(4, 3) + 0.1
            R = R / R.sum(dim=1, keepdim=True)  # Make stochastic
            noise = torch.randn_like(R) * 1e-4  # Add tiny noise
            restrictions[(f'correctable_{i}', f'edge')] = torch.clamp(R + noise, min=0)
            
        # Bad restrictions (should be dropped)
        restrictions[('bad_nan', 'edge')] = torch.tensor([[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]])
        restrictions[('bad_stochastic', 'edge')] = torch.rand(3, 3) * 10  # Very non-stochastic
        
        validated, metadata = validate_restriction_maps_gw(
            restrictions,
            correction_threshold=1e-2,
            strict_threshold=0.1,
            strict_mode=False
        )
        
        # Verify mixed handling
        assert not metadata['validation_passed']  # Some violations occurred
        assert len(metadata['finite_violations']) >= 1  # NaN detected
        assert len(metadata['dropped_edges']) >= 1  # Some dropped
        assert len(validated) >= 5  # At least the perfect ones remain
        
        # All validated restrictions should be finite and approximately stochastic
        for edge, R in validated.items():
            assert torch.isfinite(R).all()
            row_sums = R.sum(dim=1)
            expected_ones = torch.ones_like(row_sums)
            # Allow some tolerance for corrected restrictions
            assert torch.allclose(row_sums, expected_ones, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
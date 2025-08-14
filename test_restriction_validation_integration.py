#!/usr/bin/env python3
"""
Integration test for GW restriction validation.

This test demonstrates the complete pipeline from GW computation through 
restriction validation, showing how the validation integrates with the 
GW builder and protects against corrupted restrictions.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict

from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.sheaf.core.validation import validate_restriction_maps_gw
from neurosheaf.sheaf.assembly.gw_builder import GWRestrictionManager
from neurosheaf.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_test_activations() -> Dict[str, torch.Tensor]:
    """Create synthetic activation tensors for testing."""
    torch.manual_seed(42)  # Reproducible results
    
    return {
        'layer1': torch.randn(50, 32),  # 50 samples, 32 features
        'layer2': torch.randn(50, 24),  # Smaller layer
        'layer3': torch.randn(50, 16),  # Even smaller
    }


def create_test_poset() -> nx.DiGraph:
    """Create a simple test poset structure."""
    G = nx.DiGraph()
    G.add_edges_from([
        ('layer1', 'layer2'),
        ('layer2', 'layer3'),
        ('layer1', 'layer3')  # Direct connection
    ])
    return G


def test_validation_enabled():
    """Test GW builder with validation enabled."""
    logger.info("🧪 Testing GW restriction validation (enabled)")
    
    # Create config with validation enabled
    config = GWConfig(
        validate_restrictions=True,
        stochastic_tolerance=1e-5,
        correction_threshold=1e-2,
        auto_correct_restrictions=True,
        strict_validation_mode=False,
        epsilon=0.1,
        max_iter=100  # Faster for testing
    )
    
    # Create test data
    activations = create_test_activations()
    poset = create_test_poset()
    
    # Initialize GW builder
    gw_builder = GWRestrictionManager(config=config)
    
    # Compute restrictions with validation
    try:
        restrictions, gw_costs, metadata = gw_builder.compute_all_restrictions(
            activations=activations,
            poset=poset,
            parallel=False  # Simpler for testing
        )
        
        logger.info(f"✅ GW computation succeeded: {len(restrictions)} restriction maps")
        
        # Check that validation metadata is present
        assert 'restriction_validation' in metadata
        validation_metadata = metadata['restriction_validation']
        
        if validation_metadata:
            logger.info(f"📊 Validation results:")
            logger.info(f"  - Total restrictions: {validation_metadata['total_restrictions']}")
            logger.info(f"  - Validation passed: {validation_metadata['validation_passed']}")
            logger.info(f"  - Corrections applied: {len(validation_metadata['corrections_applied'])}")
            logger.info(f"  - Finite violations: {len(validation_metadata['finite_violations'])}")
            logger.info(f"  - Max stochasticity error: {validation_metadata['max_stochasticity_error']:.6f}")
            
            # All restrictions should be finite and approximately stochastic
            for edge, R in restrictions.items():
                assert torch.isfinite(R).all(), f"Non-finite values in {edge}"
                
                row_sums = R.sum(dim=1)
                stochasticity_error = torch.abs(row_sums - 1.0).max().item()
                logger.debug(f"  Edge {edge}: stochasticity error = {stochasticity_error:.6f}")
                
                # Should be reasonably stochastic (allowing for some correction tolerance)
                assert stochasticity_error < 0.1, f"Poor stochasticity in {edge}: {stochasticity_error}"
        
        logger.info("✅ Validation integration test passed")
        
    except Exception as e:
        logger.error(f"❌ GW computation failed: {e}")
        raise


def test_validation_disabled():
    """Test GW builder with validation disabled."""
    logger.info("🧪 Testing GW restriction validation (disabled)")
    
    # Create config with validation disabled
    config = GWConfig(
        validate_restrictions=False,
        epsilon=0.1,
        max_iter=100
    )
    
    activations = create_test_activations()
    poset = create_test_poset()
    
    gw_builder = GWRestrictionManager(config=config)
    
    try:
        restrictions, gw_costs, metadata = gw_builder.compute_all_restrictions(
            activations=activations,
            poset=poset,
            parallel=False
        )
        
        logger.info(f"✅ GW computation succeeded: {len(restrictions)} restriction maps")
        
        # Validation metadata should be None when disabled
        assert metadata.get('restriction_validation') is None
        logger.info("✅ Validation correctly disabled")
        
    except Exception as e:
        logger.error(f"❌ GW computation failed: {e}")
        raise


def test_injected_corrupted_restrictions():
    """Test validation with artificially corrupted restrictions."""
    logger.info("🧪 Testing validation with corrupted restrictions")
    
    # Create good restrictions
    restrictions = {
        ('layer1', 'layer2'): torch.rand(5, 4) + 0.1,
        ('layer2', 'layer3'): torch.rand(3, 5) + 0.1,
    }
    
    # Make them row-stochastic
    for edge, R in restrictions.items():
        restrictions[edge] = R / R.sum(dim=1, keepdim=True)
    
    # Inject corrupted restriction
    corrupted_R = torch.rand(4, 3)
    corrupted_R[1, 1] = float('nan')  # Inject NaN
    restrictions[('corrupted', 'edge')] = corrupted_R
    
    # Another with bad stochasticity
    bad_stochastic_R = torch.rand(3, 4) * 10  # Very non-stochastic
    restrictions[('bad_stochastic', 'edge')] = bad_stochastic_R
    
    logger.info(f"Created {len(restrictions)} restrictions (2 corrupted)")
    
    # Test validation
    validated, metadata = validate_restriction_maps_gw(
        restrictions,
        stochastic_tolerance=1e-5,
        correction_threshold=1e-2,
        strict_threshold=0.1,
        strict_mode=False,
        auto_correct=True
    )
    
    logger.info(f"📊 Validation results:")
    logger.info(f"  - Original restrictions: {len(restrictions)}")
    logger.info(f"  - Validated restrictions: {len(validated)}")
    logger.info(f"  - Finite violations: {len(metadata['finite_violations'])}")
    logger.info(f"  - Dropped edges: {len(metadata['dropped_edges'])}")
    logger.info(f"  - Corrections applied: {len(metadata['corrections_applied'])}")
    
    # Should have detected and handled corrupted restrictions
    assert not metadata['validation_passed']  # Some violations occurred
    assert ('corrupted', 'edge') in metadata['finite_violations']
    assert ('corrupted', 'edge') in metadata['dropped_edges']
    assert len(validated) < len(restrictions)  # Some restrictions dropped
    
    # All remaining restrictions should be valid
    for edge, R in validated.items():
        assert torch.isfinite(R).all()
        row_sums = R.sum(dim=1)
        stochasticity_error = torch.abs(row_sums - 1.0).max().item()
        assert stochasticity_error < 0.1  # Reasonable stochasticity
    
    logger.info("✅ Corrupted restriction handling test passed")


def test_strict_mode():
    """Test strict validation mode."""
    logger.info("🧪 Testing strict validation mode")
    
    # Create restriction with NaN
    restrictions = {
        ('good', 'edge'): torch.ones(3, 3) / 3.0,  # Perfect stochastic
        ('bad', 'edge'): torch.tensor([[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]])
    }
    
    # Test non-strict mode (should handle gracefully)
    validated, metadata = validate_restriction_maps_gw(
        restrictions, strict_mode=False
    )
    
    assert not metadata['validation_passed']
    assert len(validated) == 1  # Only good edge remains
    logger.info("✅ Non-strict mode handled corruption gracefully")
    
    # Test strict mode (should raise error)
    try:
        validate_restriction_maps_gw(restrictions, strict_mode=True)
        assert False, "Strict mode should have raised an error"
    except ValueError as e:
        logger.info(f"✅ Strict mode correctly raised error: {e}")


def main():
    """Run all integration tests."""
    logger.info("🚀 Starting GW restriction validation integration tests")
    
    try:
        test_validation_enabled()
        test_validation_disabled()
        test_injected_corrupted_restrictions()
        test_strict_mode()
        
        logger.info("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
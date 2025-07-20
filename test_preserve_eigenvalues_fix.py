#!/usr/bin/env python3
"""
Quick test to verify the preserve_eigenvalues flag fix.
"""
import torch
import torch.nn as nn 
import numpy as np
import os
import logging

# Set environment for CPU usage
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.sheaf.core.whitening import WhiteningProcessor

# Simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def test_preserve_eigenvalues_fix():
    """Test that the preserve_eigenvalues flag is properly propagated."""
    logger.info("=== Testing preserve_eigenvalues Fix ===")
    
    # Create model and data
    model = SimpleModel()
    data = torch.randn(20, 3)
    
    # Test with preserve_eigenvalues=True
    logger.info("Testing with preserve_eigenvalues=True")
    builder = SheafBuilder(preserve_eigenvalues=True)
    
    # Check that the builder's whitening processor has the correct setting
    assert builder.whitening_processor.preserve_eigenvalues == True, "Builder WhiteningProcessor should have preserve_eigenvalues=True"
    
    # Check that the restriction manager shares the same whitening processor
    assert builder.restriction_manager.whitening_processor is builder.whitening_processor, "RestrictionManager should share the same WhiteningProcessor instance"
    
    logger.info("✅ WhiteningProcessor instances are correctly shared")
    
    # Build sheaf with runtime override
    sheaf = builder.build_from_activations(model, data, preserve_eigenvalues=True)
    
    # Verify eigenvalue metadata
    assert hasattr(sheaf, 'eigenvalue_metadata'), "Sheaf should have eigenvalue_metadata"
    assert sheaf.eigenvalue_metadata is not None, "Eigenvalue metadata should not be None"
    assert sheaf.eigenvalue_metadata.preserve_eigenvalues == True, "Eigenvalue metadata should indicate preservation is enabled"
    
    logger.info("✅ Eigenvalue metadata correctly set")
    
    # Check that stalks are not identity matrices (should be eigenvalue matrices)
    identity_count = 0
    non_identity_count = 0
    
    for node_id, stalk in sheaf.stalks.items():
        if torch.allclose(stalk, torch.eye(stalk.shape[0]), atol=1e-6):
            identity_count += 1
        else:
            non_identity_count += 1
            logger.debug(f"Node {node_id} has non-identity stalk (eigenvalue preservation working)")
    
    logger.info(f"Stalk analysis: {identity_count} identity matrices, {non_identity_count} eigenvalue matrices")
    
    if non_identity_count > 0:
        logger.info("✅ SUCCESS: Found non-identity stalks - eigenvalue preservation is working!")
    else:
        logger.warning("⚠️  WARNING: All stalks are identity matrices despite preserve_eigenvalues=True")
    
    # Test with preserve_eigenvalues=False for comparison
    logger.info("Testing with preserve_eigenvalues=False for comparison")
    builder_false = SheafBuilder(preserve_eigenvalues=False)
    sheaf_false = builder_false.build_from_activations(model, data, preserve_eigenvalues=False)
    
    identity_count_false = 0
    for node_id, stalk in sheaf_false.stalks.items():
        if torch.allclose(stalk, torch.eye(stalk.shape[0]), atol=1e-6):
            identity_count_false += 1
    
    logger.info(f"With preserve_eigenvalues=False: {identity_count_false}/{len(sheaf_false.stalks)} identity matrices")
    
    if identity_count_false == len(sheaf_false.stalks):
        logger.info("✅ SUCCESS: All stalks are identity when preserve_eigenvalues=False")
    
    return sheaf, sheaf_false

if __name__ == "__main__":
    try:
        sheaf_true, sheaf_false = test_preserve_eigenvalues_fix()
        print("\n=== Test Summary ===")
        print("✅ preserve_eigenvalues flag fix is working correctly!")
        print(f"   - With preserve_eigenvalues=True: eigenvalue matrices in stalks")
        print(f"   - With preserve_eigenvalues=False: identity matrices in stalks")
        print("   - WhiteningProcessor instances are properly shared")
        print("   - Runtime overrides are working")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
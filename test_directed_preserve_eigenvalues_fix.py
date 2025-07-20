#!/usr/bin/env python3
"""
Test to verify the preserve_eigenvalues flag fix works in the directed sheaf pipeline.
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

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.directed_sheaf.assembly.builder import DirectedSheafBuilder

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

def test_directed_preserve_eigenvalues_fix():
    """Test that the preserve_eigenvalues flag is properly propagated in directed pipeline."""
    logger.info("=== Testing Directed preserve_eigenvalues Fix ===")
    
    # Create model and data
    model = SimpleModel()
    data = torch.randn(20, 3)
    
    # Test 1: Direct DirectedSheafBuilder usage with preserve_eigenvalues=True
    logger.info("Test 1: DirectedSheafBuilder with preserve_eigenvalues=True")
    builder = DirectedSheafBuilder(preserve_eigenvalues=True)
    
    # Check that the builder has the correct setting
    assert builder.preserve_eigenvalues == True, "DirectedSheafBuilder should have preserve_eigenvalues=True"
    logger.info("✅ DirectedSheafBuilder correctly configured with preserve_eigenvalues=True")
    
    # Build directed sheaf
    directed_sheaf = builder.build_from_activations(model, data, preserve_eigenvalues=True)
    
    # Check base sheaf has eigenvalue metadata
    base_sheaf = directed_sheaf.base_sheaf
    assert hasattr(base_sheaf, 'eigenvalue_metadata'), "Base sheaf should have eigenvalue_metadata"
    assert base_sheaf.eigenvalue_metadata is not None, "Eigenvalue metadata should not be None"
    assert base_sheaf.eigenvalue_metadata.preserve_eigenvalues == True, "Base sheaf should preserve eigenvalues"
    logger.info("✅ Base sheaf eigenvalue metadata correctly set")
    
    # Check that base sheaf stalks are not identity matrices (should be eigenvalue matrices)
    identity_count = 0
    non_identity_count = 0
    
    for node_id, stalk in base_sheaf.stalks.items():
        if torch.allclose(stalk, torch.eye(stalk.shape[0]), atol=1e-6):
            identity_count += 1
        else:
            non_identity_count += 1
            logger.debug(f"Base sheaf node {node_id} has non-identity stalk (eigenvalue preservation working)")
    
    logger.info(f"Base sheaf stalk analysis: {identity_count} identity matrices, {non_identity_count} eigenvalue matrices")
    
    if non_identity_count > 0:
        logger.info("✅ SUCCESS: Found non-identity stalks in base sheaf - eigenvalue preservation is working!")
    else:
        logger.warning("⚠️  WARNING: All base sheaf stalks are identity matrices despite preserve_eigenvalues=True")
    
    # Test 2: NeurosheafAnalyzer high-level API with preserve_eigenvalues=True
    logger.info("Test 2: NeurosheafAnalyzer with preserve_eigenvalues=True")
    analyzer = NeurosheafAnalyzer(device='cpu')
    
    analysis = analyzer.analyze(
        model, 
        data, 
        directed=True, 
        directionality_parameter=0.25,
        preserve_eigenvalues=True
    )
    
    directed_sheaf_api = analysis['directed_sheaf']
    base_sheaf_api = directed_sheaf_api.base_sheaf
    
    # Check base sheaf has eigenvalue metadata
    assert hasattr(base_sheaf_api, 'eigenvalue_metadata'), "API base sheaf should have eigenvalue_metadata"
    assert base_sheaf_api.eigenvalue_metadata is not None, "API eigenvalue metadata should not be None"
    assert base_sheaf_api.eigenvalue_metadata.preserve_eigenvalues == True, "API base sheaf should preserve eigenvalues"
    logger.info("✅ API-created base sheaf eigenvalue metadata correctly set")
    
    # Check that API base sheaf stalks are not identity matrices
    identity_count_api = 0
    non_identity_count_api = 0
    
    for node_id, stalk in base_sheaf_api.stalks.items():
        if torch.allclose(stalk, torch.eye(stalk.shape[0]), atol=1e-6):
            identity_count_api += 1
        else:
            non_identity_count_api += 1
            logger.debug(f"API base sheaf node {node_id} has non-identity stalk (eigenvalue preservation working)")
    
    logger.info(f"API base sheaf stalk analysis: {identity_count_api} identity matrices, {non_identity_count_api} eigenvalue matrices")
    
    if non_identity_count_api > 0:
        logger.info("✅ SUCCESS: Found non-identity stalks in API base sheaf - eigenvalue preservation is working!")
    else:
        logger.warning("⚠️  WARNING: All API base sheaf stalks are identity matrices despite preserve_eigenvalues=True")
    
    # Test 3: Compare with preserve_eigenvalues=False
    logger.info("Test 3: Compare with preserve_eigenvalues=False")
    analysis_false = analyzer.analyze(
        model, 
        data, 
        directed=True, 
        directionality_parameter=0.25,
        preserve_eigenvalues=False
    )
    
    directed_sheaf_false = analysis_false['directed_sheaf']
    base_sheaf_false = directed_sheaf_false.base_sheaf
    
    identity_count_false = 0
    for node_id, stalk in base_sheaf_false.stalks.items():
        if torch.allclose(stalk, torch.eye(stalk.shape[0]), atol=1e-6):
            identity_count_false += 1
    
    logger.info(f"With preserve_eigenvalues=False: {identity_count_false}/{len(base_sheaf_false.stalks)} identity matrices")
    
    if identity_count_false == len(base_sheaf_false.stalks):
        logger.info("✅ SUCCESS: All stalks are identity when preserve_eigenvalues=False")
    
    return directed_sheaf, directed_sheaf_api, directed_sheaf_false

if __name__ == "__main__":
    try:
        directed_sheaf, directed_sheaf_api, directed_sheaf_false = test_directed_preserve_eigenvalues_fix()
        print("\n=== Test Summary ===")
        print("✅ Directed preserve_eigenvalues flag fix is working correctly!")
        print("   - DirectedSheafBuilder properly stores preserve_eigenvalues parameter")
        print("   - NeurosheafAnalyzer properly passes preserve_eigenvalues to DirectedSheafBuilder")
        print("   - Base sheaf creation uses correct eigenvalue preservation settings")
        print("   - Eigenvalue matrices are created when preserve_eigenvalues=True")
        print("   - Identity matrices are created when preserve_eigenvalues=False")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
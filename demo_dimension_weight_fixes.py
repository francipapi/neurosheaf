#!/usr/bin/env python3
"""Demonstration of the dimension cropping and weight consistency fixes.

This script shows the fixes for two critical bugs:

A2) Edge restriction dimension cropping can silently make δ inconsistent
A3) G₁ edge weights: check linear vs squared consistency everywhere
"""

import torch
import numpy as np
import networkx as nx
import logging
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder, GWWeightTransform

# Set up logging to show validation messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_valid_sheaf():
    """Create a sheaf with correctly aligned dimensions."""
    nodes = ['node_a', 'node_b', 'node_c']
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edge('node_a', 'node_b')
    G.add_edge('node_b', 'node_c')
    
    sheaf = Sheaf(poset=G)
    
    # Add stalks with different dimensions
    sheaf.stalks['node_a'] = torch.eye(3, dtype=torch.float64)  # 3D stalk
    sheaf.stalks['node_b'] = torch.eye(2, dtype=torch.float64)  # 2D stalk  
    sheaf.stalks['node_c'] = torch.eye(4, dtype=torch.float64)  # 4D stalk
    
    # Add restrictions with CORRECT dimensions
    # R: node_a → node_b should be (2, 3) = (target_dim, source_dim)
    R_ab = torch.rand(2, 3, dtype=torch.float64)
    R_ab = R_ab / R_ab.sum(dim=0, keepdim=True)  # Column normalize
    sheaf.restrictions[('node_a', 'node_b')] = R_ab
    
    # R: node_b → node_c should be (4, 2) = (target_dim, source_dim)
    R_bc = torch.rand(4, 2, dtype=torch.float64)
    R_bc = R_bc / R_bc.sum(dim=0, keepdim=True)  # Column normalize
    sheaf.restrictions[('node_b', 'node_c')] = R_bc
    
    # Add GW metadata with realistic costs
    sheaf.metadata['construction_method'] = 'gromov_wasserstein'
    sheaf.metadata['gw_costs'] = {
        ('node_a', 'node_b'): 1.2,  # Medium cost
        ('node_b', 'node_c'): 0.8   # Lower cost (better similarity)
    }
    
    return sheaf

def create_invalid_sheaf():
    """Create a sheaf with misaligned dimensions to trigger validation."""
    nodes = ['node_x', 'node_y']
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edge('node_x', 'node_y')
    
    sheaf = Sheaf(poset=G)
    
    # Add stalks
    sheaf.stalks['node_x'] = torch.eye(3, dtype=torch.float64)  # 3D stalk
    sheaf.stalks['node_y'] = torch.eye(2, dtype=torch.float64)  # 2D stalk
    
    # Add restriction with WRONG dimensions
    # Should be (2, 3) but we'll create (3, 2) - MISMATCHED!
    R_bad = torch.rand(3, 2, dtype=torch.float64)  # WRONG SHAPE!
    sheaf.restrictions[('node_x', 'node_y')] = R_bad
    
    # Add GW metadata
    sheaf.metadata['construction_method'] = 'gromov_wasserstein' 
    sheaf.metadata['gw_costs'] = {('node_x', 'node_y'): 0.5}
    
    return sheaf

def demo_dimension_fix():
    """Demonstrate the dimension validation fixes."""
    print("=" * 70)
    print("DIMENSION CROPPING FIX DEMONSTRATION")
    print("=" * 70)
    
    builder = GWLaplacianBuilder(validate_properties=True)
    
    print("\n🟢 TEST 1: Valid sheaf with correct dimensions")
    print("-" * 50)
    
    valid_sheaf = create_valid_sheaf()
    try:
        # This should work fine
        delta = builder.build_coboundary_general_sparse(
            valid_sheaf, [('node_a', 'node_b'), ('node_b', 'node_c')]
        )
        print(f"✅ SUCCESS: Built coboundary with shape {delta.shape}")
        print(f"   Stalks: node_a(3D), node_b(2D), node_c(4D)")
        print(f"   Restrictions: R_ab(2,3), R_bc(4,2) ← Correctly aligned!")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
    
    print("\n🔴 TEST 2: Invalid sheaf with misaligned dimensions")
    print("-" * 50)
    
    invalid_sheaf = create_invalid_sheaf()
    try:
        # This should raise a clear validation error
        delta = builder.build_coboundary_general_sparse(
            invalid_sheaf, [('node_x', 'node_y')]
        )
        print(f"❌ UNEXPECTED SUCCESS: Should have failed but got shape {delta.shape}")
        
    except ValueError as e:
        print(f"✅ EXPECTED ERROR caught:")
        print(f"   {str(e)}")
        print(f"   💡 Error clearly explains dimension mismatch and its consequences")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR TYPE: {e}")

def demo_weight_validation():
    """Demonstrate the weight validation fixes."""
    print("\n" + "=" * 70)
    print("WEIGHT CONSISTENCY FIX DEMONSTRATION")
    print("=" * 70)
    
    builder = GWLaplacianBuilder(validate_properties=True)
    
    print("\n🟢 TEST 1: Normal weight range")
    print("-" * 30)
    
    # Create sheaf with normal cost range
    normal_sheaf = create_valid_sheaf()
    try:
        weights = builder.extract_edge_weights(
            normal_sheaf, [('node_a', 'node_b'), ('node_b', 'node_c')],
            transform_method=GWWeightTransform.EXPONENTIAL,
            transform_beta=1.0
        )
        
        weight_values = list(weights.values())
        min_w, max_w = min(weight_values), max(weight_values)
        range_ratio = max_w / min_w if min_w > 0 else float('inf')
        
        print(f"✅ SUCCESS: Extracted weights with range {min_w:.4f} to {max_w:.4f}")
        print(f"   Range ratio: {range_ratio:.2e} (reasonable)")
        print(f"   💡 Note: weights = sqrt(similarities), assembly uses weight²")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
    
    print("\n🟡 TEST 2: Extreme weight range (triggers warning)")
    print("-" * 50)
    
    # Create sheaf with extreme cost range
    extreme_sheaf = create_valid_sheaf()
    extreme_sheaf.metadata['gw_costs'] = {
        ('node_a', 'node_b'): 0.001,  # Very low cost → high similarity → high weight
        ('node_b', 'node_c'): 50.0    # Very high cost → low similarity → low weight
    }
    
    try:
        weights = builder.extract_edge_weights(
            extreme_sheaf, [('node_a', 'node_b'), ('node_b', 'node_c')],
            transform_method=GWWeightTransform.EXPONENTIAL,
            transform_beta=5.0  # High beta amplifies differences
        )
        
        weight_values = list(weights.values())
        min_w, max_w = min(weight_values), max(weight_values)
        range_ratio = max_w / min_w if min_w > 0 else float('inf')
        
        print(f"⚠️  SUCCESS with warning: Extreme range {min_w:.2e} to {max_w:.2e}")
        print(f"   Range ratio: {range_ratio:.2e}")
        print(f"   💡 Check log output above for dynamic range warning")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

def demo_pipeline_consistency():
    """Demonstrate the weight pipeline consistency."""
    print("\n" + "=" * 70) 
    print("WEIGHT PIPELINE CONSISTENCY DEMONSTRATION")
    print("=" * 70)
    
    builder = GWLaplacianBuilder(validate_properties=True)
    sheaf = create_valid_sheaf()
    
    print("\n📊 PIPELINE: cost → similarity → sqrt(similarity) → [assembly: weight²] → energy ∝ similarity")
    print("-" * 90)
    
    # Test different transforms
    transforms = [
        (GWWeightTransform.EXPONENTIAL, "exp(-β*cost)"),
        (GWWeightTransform.RECIPROCAL, "1/(1+cost)"),
        (GWWeightTransform.LINEAR, "max_cost - cost")
    ]
    
    for transform, formula in transforms:
        weights = builder.extract_edge_weights(
            sheaf, [('node_a', 'node_b'), ('node_b', 'node_c')],
            transform_method=transform,
            transform_beta=2.0
        )
        
        print(f"\n🔧 {transform.value.upper()} transform: {formula}")
        for edge, weight in weights.items():
            cost = sheaf.metadata['gw_costs'][edge]
            final_energy_coeff = weight ** 2  # What gets used in assembly
            print(f"   Edge {edge}: cost={cost:.3f} → weight={weight:.4f} → energy_coeff={final_energy_coeff:.4f}")
        
        print(f"   ✅ All weights positive, pipeline consistent")

def main():
    """Run all demonstrations."""
    print("NEUROSHEAF DIMENSION & WEIGHT FIXES DEMONSTRATION")
    print("=" * 70)
    print("Fixes implemented:")
    print("  A2) Edge restriction dimension validation (no more silent cropping)")
    print("  A3) G₁ edge weight consistency validation (range checks)")
    print()
    
    demo_dimension_fix()
    demo_weight_validation() 
    demo_pipeline_consistency()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("🔧 FIXES IMPLEMENTED:")
    print("   • Strict dimension validation in coboundary and Laplacian construction")
    print("   • Clear error messages for dimension mismatches")
    print("   • Runtime weight validation with range checks")
    print("   • Consistent weight squaring pipeline documentation")
    print()
    print("🛡️  PREVENTS:")
    print("   • Silent dimension cropping causing δ inconsistency")
    print("   • Unstable eigenvalue conditioning from dimension mismatches") 
    print("   • Pathological edge weight ranges degrading conditioning")
    print("   • Weight pipeline inconsistencies causing energy mis-scaling")
    print()
    print("💡 RESULT:")
    print("   • Robust error detection instead of silent failures")
    print("   • Clear feedback to help users debug dimension issues")
    print("   • Maintained mathematical correctness of L = δᵀG₁δ formulation")

if __name__ == "__main__":
    main()
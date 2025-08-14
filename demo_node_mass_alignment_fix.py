#!/usr/bin/env python3
"""Demo script showing the node-mass alignment fix in action.

Before the fix:
- Tensor/list masses were accepted without validation
- Wrong alignment could cause malformed eigenvalue problems
- No warnings about ordering assumptions

After the fix:
- Clear validation errors for count mismatches
- Helpful warnings about ordering assumptions
- Explicit documentation of expected node order
"""

import torch
import numpy as np
import networkx as nx
import logging
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder

# Set up logging to see warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_sheaf():
    """Create test sheaf with non-alphabetical node IDs."""
    # Create graph with non-alphabetical ordering
    node_ids = ['node_gamma', 'node_alpha', 'node_beta']
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from([
        ('node_gamma', 'node_alpha'),
        ('node_alpha', 'node_beta'), 
        ('node_beta', 'node_gamma')
    ])
    
    # Create sheaf
    sheaf = Sheaf(poset=G)
    
    # Add stalks with different dimensions
    stalk_dims = [2, 3, 4]  # Different for each node
    for i, node in enumerate(node_ids):
        dim = stalk_dims[i]
        sheaf.stalks[node] = torch.eye(dim, dtype=torch.float64)
        
    # Add restrictions
    for u, v in G.edges():
        u_dim = sheaf.stalks[u].shape[0] 
        v_dim = sheaf.stalks[v].shape[0]
        R = torch.rand(v_dim, u_dim, dtype=torch.float64)
        R = R / R.sum(dim=0, keepdim=True)  # Column normalize
        sheaf.restrictions[(u, v)] = R
    
    # Add GW metadata
    sheaf.metadata['construction_method'] = 'gromov_wasserstein'
    sheaf.metadata['gw_costs'] = {edge: np.random.rand() for edge in G.edges()}
    
    return sheaf

def demo_fix():
    """Demonstrate the node-mass alignment fix."""
    print("=" * 60)
    print("NODE-MASS ALIGNMENT FIX DEMONSTRATION")
    print("=" * 60)
    
    sheaf = create_demo_sheaf()
    builder = GWLaplacianBuilder()
    
    # Show the node ordering that will be used internally
    nodes_sorted = sorted(sheaf.poset.nodes())
    print(f"\n📋 Nodes in sheaf: {list(sheaf.poset.nodes())}")
    print(f"🔄 Internal sorted order: {nodes_sorted}")
    print(f"💡 All mass tensors/arrays must align with sorted order!")
    
    print("\n" + "─" * 50)
    print("✅ TEST 1: Correct tensor mass count")
    print("─" * 50)
    
    # Correct count and proper warning
    correct_masses = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)  # 3 masses for 3 nodes
    sheaf.metadata['node_masses'] = correct_masses
    
    try:
        extracted = builder._extract_node_masses(sheaf)
        print(f"✅ SUCCESS: Extracted masses shape: {extracted.shape}")
        print(f"   Values: {extracted.numpy()}")
        print(f"   ⚠️  Note the warning above about ordering assumptions!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n" + "─" * 50) 
    print("❌ TEST 2: Wrong tensor mass count (should fail)")
    print("─" * 50)
    
    # Wrong count - should raise clear error
    wrong_masses = torch.tensor([0.4, 0.6], dtype=torch.float64)  # Only 2 masses for 3 nodes
    sheaf.metadata['node_masses'] = wrong_masses
    
    try:
        extracted = builder._extract_node_masses(sheaf)
        print(f"❌ UNEXPECTED: Should have failed but got: {extracted.shape}")
    except ValueError as e:
        print(f"✅ EXPECTED ERROR caught:")
        print(f"   {str(e)}")
        print(f"   💡 Error clearly shows expected node order and suggests dict alternative")
    
    print("\n" + "─" * 50)
    print("✅ TEST 3: Dict masses (always work correctly)")
    print("─" * 50)
    
    # Dict format - order doesn't matter, handled automatically
    dict_masses = {
        'node_gamma': 0.5,  # This would be first in original order
        'node_alpha': 0.2,  # But should be first in sorted order
        'node_beta': 0.3    # Should be second in sorted order  
    }
    sheaf.metadata['node_masses'] = dict_masses
    
    try:
        extracted = builder._extract_node_masses(sheaf)
        print(f"✅ SUCCESS: Dict masses converted correctly")
        print(f"   Shape: {extracted.shape}")
        print(f"   Values in sorted order: {extracted.numpy()}")
        print(f"   💡 Expected: [0.2, 0.3, 0.5] for ['node_alpha', 'node_beta', 'node_gamma']")
        
        # Verify correct ordering
        expected = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        if torch.allclose(extracted, expected):
            print(f"   ✅ Ordering verified correct!")
        else:
            print(f"   ❌ Ordering mismatch!")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    print("\n" + "─" * 50)
    print("❌ TEST 4: List mass count validation")
    print("─" * 50)
    
    # Test list validation  
    wrong_list = [0.25, 0.25, 0.25, 0.25]  # 4 masses for 3 nodes
    sheaf.metadata['node_masses'] = wrong_list
    
    try:
        extracted = builder._extract_node_masses(sheaf)
        print(f"❌ UNEXPECTED: Should have failed but got: {extracted.shape}")
    except ValueError as e:
        print(f"✅ EXPECTED ERROR caught for list:")
        print(f"   {str(e)}")
        
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("🔧 FIX IMPLEMENTED:")
    print("   • Validates tensor/list mass count matches node count")
    print("   • Provides clear errors with expected node ordering")
    print("   • Issues warnings about ordering assumptions")
    print("   • Suggests dict format for automatic ordering")
    print("   • Maintains backward compatibility for dict masses")
    print("\n🛡️ PREVENTS:")
    print("   • Malformed generalized eigenvalue problems")
    print("   • Wrong stalk blocks being scaled by incorrect masses")
    print("   • Silent failures leading to incorrect spectral analysis")
    print("   • Blown-up residuals and bad eigenvalue results")
    
    print("\n💡 RECOMMENDATION:")
    print("   Use dict format for masses to avoid ordering issues:")
    print("   sheaf.metadata['node_masses'] = {node: mass for node, mass in zip(nodes, masses)}")

if __name__ == "__main__":
    demo_fix()
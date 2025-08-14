#!/usr/bin/env python3
"""Demonstration of barycentric normalization for GW restriction maps.

This script shows how barycentric normalization ensures row-stochasticity
and improves functoriality in GW sheaf construction.
"""

import torch
import torch.nn as nn
import numpy as np
from neurosheaf.sheaf.core import GWConfig, GromovWassersteinComputer
from neurosheaf.sheaf.assembly import SheafBuilder, GWRestrictionManager
import networkx as nx


class ThreeLayerNet(nn.Module):
    """Simple 3-layer network for demonstration."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def demonstrate_barycentric_normalization():
    """Demonstrate the effects of barycentric normalization."""
    
    print("=" * 70)
    print("BARYCENTRIC NORMALIZATION DEMONSTRATION")
    print("=" * 70)
    
    # Create model and data
    model = ThreeLayerNet()
    batch_size = 8
    input_tensor = torch.randn(batch_size, 10)
    
    # Build sheaf with GW method (includes barycentric normalization)
    print("\n1. Building GW sheaf with barycentric normalization...")
    config = GWConfig(align_units=True)
    builder = SheafBuilder(restriction_method='gromov_wasserstein')
    
    sheaf = builder.build_from_activations(
        model, input_tensor,
        validate=True,
        gw_config=config
    )
    
    print(f"   Sheaf built with {len(sheaf.stalks)} nodes and {len(sheaf.restrictions)} edges")
    
    # Check row-stochasticity of all restriction maps
    print("\n2. Checking row-stochasticity of restriction maps:")
    print("   (Each row should sum to 1.0)")
    print("-" * 50)
    
    all_stochastic = True
    for (source, target), restriction in sheaf.restrictions.items():
        row_sums = restriction.sum(dim=1)
        min_sum = row_sums.min().item()
        max_sum = row_sums.max().item()
        mean_sum = row_sums.mean().item()
        
        is_stochastic = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        all_stochastic = all_stochastic and is_stochastic
        
        status = "✓" if is_stochastic else "✗"
        print(f"   {source:4} → {target:4}: {status} "
              f"Row sums: min={min_sum:.6f}, mean={mean_sum:.6f}, max={max_sum:.6f}")
    
    if all_stochastic:
        print("\n   ✓ All restriction maps are row-stochastic!")
    else:
        print("\n   ✗ Some restriction maps are not perfectly row-stochastic")
    
    # Check functoriality
    print("\n3. Checking functoriality (composition property):")
    print("   For path i→j→k: R_{ik} should equal R_{ij} @ R_{jk}")
    print("-" * 50)
    
    # Find a 3-node path in the poset
    poset = sheaf.poset
    paths_checked = 0
    total_violation = 0.0
    
    for node_i in poset.nodes():
        for node_j in poset.successors(node_i):
            for node_k in poset.successors(node_j):
                if (node_i, node_k) in poset.edges():
                    # We have a path i→j→k with direct edge i→k
                    edge_ij = (node_i, node_j)
                    edge_jk = (node_j, node_k)
                    edge_ik = (node_i, node_k)
                    
                    if all(e in sheaf.restrictions for e in [edge_ij, edge_jk, edge_ik]):
                        R_ij = sheaf.restrictions[edge_ij]
                        R_jk = sheaf.restrictions[edge_jk]
                        R_ik = sheaf.restrictions[edge_ik]
                        
                        # Compute composition
                        composed = R_ij @ R_jk
                        
                        # Compute violation
                        violation = torch.norm(R_ik - composed, 'fro').item()
                        total_violation += violation
                        paths_checked += 1
                        
                        print(f"   Path {node_i}→{node_j}→{node_k}: "
                              f"||R_ik - R_ij@R_jk||_F = {violation:.6f}")
    
    if paths_checked > 0:
        avg_violation = total_violation / paths_checked
        print(f"\n   Average functoriality violation: {avg_violation:.6f}")
        
        if avg_violation < 0.1:
            print("   ✓ Good functoriality (low violation)")
        else:
            print("   ⚠ Moderate functoriality violation")
    else:
        print("   No 3-node paths found for testing")
    
    # Show effect of barycentric normalization
    print("\n4. Effect of Barycentric Normalization:")
    print("-" * 50)
    
    # Get a sample restriction and its coupling
    if len(sheaf.restrictions) > 0:
        edge, restriction = next(iter(sheaf.restrictions.items()))
        source, target = edge
        
        print(f"   Example edge: {source} → {target}")
        print(f"   Restriction shape: {restriction.shape}")
        
        # Compute row sums
        row_sums = restriction.sum(dim=1)
        print(f"   Row sums (should all be 1.0):")
        print(f"     First 5 rows: {row_sums[:5].tolist()}")
        
        # Show that columns don't sum to 1 (not column-stochastic)
        col_sums = restriction.sum(dim=0)
        print(f"   Column sums (generally not 1.0):")
        print(f"     First 5 cols: {col_sums[:5].tolist()}")
        
        print("\n   Key insight: Barycentric normalization ensures ROW stochasticity,")
        print("   which means each target unit receives a proper probability distribution")
        print("   over source units, ensuring conservation of 'mass' in the transport.")
    
    # Display validation report if available
    if 'validation' in sheaf.metadata and sheaf.metadata['validation']:
        validation = sheaf.metadata['validation']
        print("\n5. Quasi-Sheaf Validation Report:")
        print("-" * 50)
        print(f"   Maximum functoriality violation: {validation.get('max_violation', 'N/A'):.6f}")
        print(f"   Mean violation: {validation.get('mean_violation', 'N/A'):.6f}")
        print(f"   Paths checked: {validation.get('num_paths_checked', 'N/A')}")
        print(f"   Satisfies quasi-sheaf property: {validation.get('satisfies_quasi_sheaf', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("- Barycentric normalization ensures row-stochastic restriction maps")
    print("- This improves numerical stability and functoriality")
    print("- Restriction maps properly represent probabilistic mappings between layers")
    print("=" * 70)


if __name__ == "__main__":
    # Suppress some warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    demonstrate_barycentric_normalization()
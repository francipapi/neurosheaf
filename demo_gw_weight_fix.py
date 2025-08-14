#!/usr/bin/env python3
"""Demonstration of GW cost-to-similarity weight transformation.

This script shows how the weight transformation fixes the inverted semantics
where raw GW costs (higher = worse match) dominated the Laplacian energy.
"""

import torch
import torch.nn as nn
import numpy as np
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.assembly import SheafBuilder, GWLaplacianBuilder
from neurosheaf.sheaf.assembly.gw_laplacian import GWWeightTransform


class ThreeLayerNet(nn.Module):
    """Simple network for demonstration."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 8)
        self.fc3 = nn.Linear(8, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def demonstrate_weight_transformation():
    """Demonstrate the GW weight transformation fix."""
    
    print("=" * 70)
    print("GW COST-TO-SIMILARITY WEIGHT TRANSFORMATION DEMONSTRATION")
    print("=" * 70)
    
    # Create model and data
    model = ThreeLayerNet()
    batch_size = 6
    input_tensor = torch.randn(batch_size, 10)
    
    # Build GW sheaf
    print("\n1. Building GW sheaf...")
    config = GWConfig(align_units=True)
    builder = SheafBuilder(restriction_method='gromov_wasserstein')
    
    sheaf = builder.build_from_activations(
        model, input_tensor,
        validate=False,  # Skip for speed
        gw_config=config
    )
    
    print(f"   Sheaf built with {len(sheaf.stalks)} nodes and {len(sheaf.restrictions)} edges")
    
    # Extract raw GW costs
    raw_costs = sheaf.metadata.get('gw_costs', {})
    if raw_costs:
        cost_values = list(raw_costs.values())
        print(f"   Raw GW costs: min={min(cost_values):.4f}, max={max(cost_values):.4f}, mean={sum(cost_values)/len(cost_values):.4f}")
        print("   Note: Lower costs = better matches")
    
    # Compare different weight transformations
    print("\n2. Comparing Weight Transformations:")
    print("-" * 50)
    
    transforms = [
        (GWWeightTransform.NONE, "Raw costs (deprecated)", "Higher costs dominate energy"),
        (GWWeightTransform.EXPONENTIAL, "Exponential: exp(-cost)", "Better matches get higher weights"),
        (GWWeightTransform.RECIPROCAL, "Reciprocal: 1/(1+cost)", "Bounded, robust to outliers"),
        (GWWeightTransform.LINEAR, "Linear: max_cost - cost", "Preserves relative ordering")
    ]
    
    all_weights = {}
    
    for transform_method, description, interpretation in transforms:
        print(f"\n   {description}:")
        
        laplacian_builder = GWLaplacianBuilder(
            weight_transform=transform_method,
            transform_beta=1.0
        )
        
        # Extract weights with this transformation
        weights = laplacian_builder.extract_edge_weights(
            sheaf, 
            transform_method=transform_method,
            transform_beta=1.0
        )
        
        all_weights[transform_method.value] = weights
        
        if weights:
            weight_values = list(weights.values())
            min_w, max_w, mean_w = min(weight_values), max(weight_values), sum(weight_values)/len(weight_values)
            print(f"     Weights: min={min_w:.4f}, max={max_w:.4f}, mean={mean_w:.4f}")
            print(f"     Interpretation: {interpretation}")
    
    # Show how the transformation affects specific edges
    print("\n3. Edge-by-Edge Comparison:")
    print("-" * 50)
    
    if raw_costs:
        # Sort edges by raw cost (best to worst matches)
        sorted_edges = sorted(raw_costs.items(), key=lambda x: x[1])
        
        print("   Edge               Raw Cost    Exp Weight   Rec Weight   Lin Weight")
        print("   " + "-" * 65)
        
        for (edge, cost) in sorted_edges[:3]:  # Show top 3
            exp_weight = all_weights['exponential'].get(edge, 'N/A')
            rec_weight = all_weights['reciprocal'].get(edge, 'N/A')
            lin_weight = all_weights['linear'].get(edge, 'N/A')
            
            print(f"   {str(edge):18} {cost:8.4f}    {exp_weight:8.4f}    {rec_weight:8.4f}    {lin_weight:8.4f}")
    
    # Build Laplacians with different transforms
    print("\n4. Impact on Laplacian Construction:")
    print("-" * 50)
    
    for transform_method, description, _ in transforms[:3]:  # Skip NONE for Laplacian
        if transform_method == GWWeightTransform.NONE:
            continue
            
        laplacian_builder = GWLaplacianBuilder(
            weight_transform=transform_method,
            transform_beta=1.0
        )
        
        try:
            laplacian = laplacian_builder.build_laplacian(sheaf, sparse=False)
            
            # Compute basic properties
            eigenvals = np.linalg.eigvalsh(laplacian)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
            
            print(f"\n   {description}:")
            print(f"     Laplacian shape: {laplacian.shape}")
            print(f"     Non-zero eigenvalues: {len(eigenvals)}")
            if len(eigenvals) > 0:
                print(f"     Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                print(f"     Spectral gap: {eigenvals[0]:.4f}")
        
        except Exception as e:
            print(f"   {description}: Failed to build Laplacian - {e}")
    
    # Demonstrate the key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("1. Raw GW costs are dissimilarities (lower = better match)")
    print("2. Using costs directly as weights inverts semantics:")
    print("   - Higher costs dominate the Laplacian energy")
    print("   - Worse matches get more influence")
    print("3. Transformations fix this by converting costs to similarities:")
    print("   - Better matches (lower costs) → higher weights")
    print("   - Square root accounts for w² scaling in energy")
    print("4. Different transforms offer different characteristics:")
    print("   - Exponential: High sensitivity, configurable with β")
    print("   - Reciprocal: Bounded [0,1], robust to cost outliers")
    print("   - Linear: Simple, preserves relative cost ordering")
    print("=" * 70)


if __name__ == "__main__":
    # Suppress some warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    demonstrate_weight_transformation()
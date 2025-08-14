#!/usr/bin/env python3
"""Demonstration of unit-based vs sample-based GW alignment.

This script shows the key differences between aligning units (neurons/channels)
versus aligning samples (datapoints) in GW sheaf construction.
"""

import torch
import torch.nn as nn
import numpy as np
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.assembly import SheafBuilder


class SimpleNet(nn.Module):
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


def demonstrate_alignment_modes():
    """Demonstrate the difference between unit and sample alignment."""
    
    print("=" * 60)
    print("GW Sheaf Construction: Unit vs Sample Alignment")
    print("=" * 60)
    
    # Create model and test data
    model = SimpleNet()
    
    # Test with different batch sizes
    batch_sizes = [4, 8]
    
    for align_units in [True, False]:
        mode = "UNIT" if align_units else "SAMPLE"
        print(f"\n{mode}-BASED ALIGNMENT (align_units={align_units})")
        print("-" * 40)
        
        stalks_by_batch = {}
        
        for batch_size in batch_sizes:
            # Create input
            input_tensor = torch.randn(batch_size, 10)
            
            # Build sheaf
            config = GWConfig(align_units=align_units)
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            
            try:
                sheaf = builder.build_from_activations(
                    model, input_tensor, 
                    validate=False,  # Skip validation for speed
                    gw_config=config
                )
                
                # Record stalk dimensions
                stalks_by_batch[batch_size] = {}
                for node_name, stalk in sheaf.stalks.items():
                    if 'fc' in node_name:
                        stalks_by_batch[batch_size][node_name] = stalk.shape[0]
                
                print(f"\nBatch size: {batch_size}")
                for node_name, dim in stalks_by_batch[batch_size].items():
                    print(f"  {node_name}: stalk dimension = {dim}")
                    
            except Exception as e:
                print(f"Error with batch size {batch_size}: {e}")
        
        # Check consistency across batches
        if len(stalks_by_batch) == 2:
            print("\nConsistency check across batch sizes:")
            batch1, batch2 = batch_sizes
            
            for node_name in stalks_by_batch[batch1].keys():
                dim1 = stalks_by_batch[batch1].get(node_name)
                dim2 = stalks_by_batch[batch2].get(node_name)
                
                if dim1 and dim2:
                    if dim1 == dim2:
                        print(f"  {node_name}: ✓ Consistent ({dim1})")
                    else:
                        print(f"  {node_name}: ✗ Inconsistent ({dim1} vs {dim2})")
            
            if align_units:
                print("\n✓ Unit alignment: Dimensions are batch-independent")
                print("  Stalks represent unit similarity (neurons/channels)")
            else:
                print("\n✗ Sample alignment: Dimensions depend on batch size")
                print("  Stalks represent sample similarity (datapoints)")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("1. Unit alignment (align_units=True) produces batch-independent results")
    print("2. Stalk dimensions match layer widths (20, 15, 5) not batch sizes")
    print("3. This makes the sheaf portable and architecture-aware")
    print("4. Sample alignment (deprecated) produces batch-dependent results")
    print("=" * 60)


if __name__ == "__main__":
    # Suppress some warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    demonstrate_alignment_modes()
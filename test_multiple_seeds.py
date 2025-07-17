#!/usr/bin/env python3
"""Test script to verify randomization is working properly."""

import torch
import torch.nn as nn
import numpy as np
from neurosheaf.sheaf import SheafBuilder, build_sheaf_laplacian

def test_seed(seed):
    """Test with a specific seed."""
    print(f"\n=== Testing with seed {seed} ===")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create model and initialize weights randomly
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    
    # Explicitly initialize model weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Generate data
    data = torch.randn(500, 100)
    
    # Build sheaf
    builder = SheafBuilder()
    sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=False)
    
    # Build laplacian
    laplacian, metadata = build_sheaf_laplacian(sheaf, validate=False)
    
    # Extract key properties
    stalk_dims = [sheaf.stalks[node].shape[0] for node in sheaf.stalks]
    restriction_norms = [torch.norm(R, 'fro').item() for R in sheaf.restrictions.values()]
    
    print(f"Data stats: mean={data.mean():.4f}, std={data.std():.4f}")
    print(f"Stalk dims: {stalk_dims}")
    print(f"Restriction norms: {[f'{norm:.4f}' for norm in restriction_norms]}")
    print(f"Laplacian: {laplacian.shape}, {laplacian.nnz} non-zeros")
    
    return {
        'seed': seed,
        'data_mean': data.mean().item(),
        'data_std': data.std().item(),
        'stalk_dims': stalk_dims,
        'restriction_norms': restriction_norms,
        'laplacian_shape': laplacian.shape,
        'laplacian_nnz': laplacian.nnz
    }

# Test multiple seeds
seeds = [42, 123, 999, 2025, 7]
results = []

for seed in seeds:
    try:
        result = test_seed(seed)
        results.append(result)
    except Exception as e:
        print(f"Error with seed {seed}: {e}")

print("\n=== SUMMARY COMPARISON ===")
print("Seed\tData Mean\tData Std\tStalk Dims\tLaplacian NNZ")
for r in results:
    print(f"{r['seed']}\t{r['data_mean']:.4f}\t\t{r['data_std']:.4f}\t\t{r['stalk_dims']}\t{r['laplacian_nnz']}")

# Check if results are truly different
if len(results) > 1:
    print("\n=== VARIABILITY CHECK ===")
    first_result = results[0]
    all_same = True
    
    for i, r in enumerate(results[1:], 1):
        if (r['stalk_dims'] != first_result['stalk_dims'] or
            r['laplacian_nnz'] != first_result['laplacian_nnz']):
            all_same = False
            print(f"Seed {r['seed']} differs from seed {first_result['seed']}")
            break
    
    if all_same:
        print("⚠️  WARNING: All results are identical - randomization may not be working!")
    else:
        print("✅ Results vary properly across seeds - randomization is working!")
#!/usr/bin/env python3
"""Detailed debug script to trace dimensional flow through sheaf construction."""

import torch
import torch.nn as nn
from neurosheaf.sheaf import SheafBuilder
from neurosheaf.sheaf.core import scaled_procrustes_whitened, WhiteningProcessor, compute_gram_matrices_from_activations
from neurosheaf.sheaf.extraction import extract_activations_fx, FXPosetExtractor

# Create the same model as in test_all.py
model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

data = torch.randn(500, 100)

print("=== STEP-BY-STEP DIMENSIONAL ANALYSIS ===")

# Step 1: Extract activations
print("\n1. EXTRACTING ACTIVATIONS")
activations = extract_activations_fx(model, data)
print(f"Number of activations: {len(activations)}")
for name, act in activations.items():
    print(f"  {name}: {act.shape}")

# Step 2: Extract poset
print("\n2. EXTRACTING POSET")
poset_extractor = FXPosetExtractor()
available_activations = set(activations.keys())
poset = poset_extractor.extract_activation_filtered_poset(model, available_activations)
print(f"Poset nodes: {list(poset.nodes())}")
print(f"Poset edges: {list(poset.edges())}")

# Step 3: Filter activations and compute Gram matrices
print("\n3. COMPUTING GRAM MATRICES")
poset_nodes = set(poset.nodes())
filtered_activations = {k: v for k, v in activations.items() if k in poset_nodes}
print(f"Filtered activations: {list(filtered_activations.keys())}")

gram_matrices = compute_gram_matrices_from_activations(filtered_activations)
print(f"Gram matrices computed: {len(gram_matrices)}")
for name, K in gram_matrices.items():
    rank = torch.linalg.matrix_rank(K.float()).item()
    print(f"  {name}: shape={K.shape}, rank={rank}")

# Step 4: Test individual restriction computation
print("\n4. TESTING RESTRICTION COMPUTATION")
whitening_processor = WhiteningProcessor()

for source, target in poset.edges():
    if source in gram_matrices and target in gram_matrices:
        K_source = gram_matrices[source]
        K_target = gram_matrices[target]
        
        print(f"\nEdge {source} → {target}:")
        print(f"  Source Gram: {K_source.shape}")
        print(f"  Target Gram: {K_target.shape}")
        
        # Get ranks via whitening
        _, W_s, info_s = whitening_processor.whiten_gram_matrix(K_source)
        _, W_t, info_t = whitening_processor.whiten_gram_matrix(K_target)
        
        r_s = W_s.shape[0]
        r_t = W_t.shape[0]
        print(f"  Source rank: {r_s}")
        print(f"  Target rank: {r_t}")
        print(f"  Expected restriction size: ({r_t}, {r_s})")
        
        # Compute restriction
        try:
            R_w, scale, info = scaled_procrustes_whitened(K_source, K_target, validate=False)
            print(f"  Actual restriction size: {R_w.shape}")
            print(f"  Scale: {scale:.4f}")
            print(f"  ✓ Dimensions match: {R_w.shape == (r_t, r_s)}")
        except Exception as e:
            print(f"  ❌ Error computing restriction: {e}")

# Step 5: Full sheaf construction
print("\n5. FULL SHEAF CONSTRUCTION")
builder = SheafBuilder()
try:
    sheaf = builder.build_from_activations(model, data, use_gram_regularization=False, validate=False)
    
    print(f"Sheaf constructed successfully")
    print(f"Number of stalks: {len(sheaf.stalks)}")
    print(f"Number of restrictions: {len(sheaf.restrictions)}")
    
    print("\nStalk dimensions:")
    total_stalk_dim = 0
    for node, stalk in sheaf.stalks.items():
        print(f"  {node}: {stalk.shape}")
        total_stalk_dim += stalk.shape[0]
    print(f"Total stalk dimension: {total_stalk_dim}")
    
    print("\nRestriction dimensions:")
    for edge, restriction in sheaf.restrictions.items():
        print(f"  {edge}: {restriction.shape}")
    
    # Check if dimensions are consistent
    print("\n6. DIMENSIONAL CONSISTENCY CHECK")
    for edge, restriction in sheaf.restrictions.items():
        source, target = edge
        source_stalk_dim = sheaf.stalks[source].shape[0]
        target_stalk_dim = sheaf.stalks[target].shape[0]
        expected_shape = (target_stalk_dim, source_stalk_dim)
        actual_shape = restriction.shape
        
        print(f"Edge {edge}:")
        print(f"  Expected: {expected_shape}")
        print(f"  Actual: {actual_shape}")
        print(f"  ✓ Match: {expected_shape == actual_shape}")
        
except Exception as e:
    print(f"❌ Sheaf construction failed: {e}")
    import traceback
    traceback.print_exc()
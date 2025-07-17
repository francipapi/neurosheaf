#!/usr/bin/env python3
"""Debug script to understand sheaf dimension issues."""

import torch
import torch.nn as nn
from neurosheaf.sheaf import SheafBuilder, build_sheaf_laplacian

# Create the same model as in test_all.py
model = nn.Sequential(                                                                                                                                         
        nn.Linear(100, 64),                                                                                                                                       
        nn.ReLU(),                                                                                                                                                 
        nn.Linear(64, 32),                                                                                                                                         
        nn.ReLU(),                                                                                                                                                
        nn.Linear(32, 10)
)

data = torch.randn(500,100)

builder = SheafBuilder()
sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=False)

print("=== Sheaf Analysis ===")
print(f"Number of nodes: {len(sheaf.stalks)}")
print(f"Number of edges: {len(sheaf.restrictions)}")
print(f"Poset nodes: {list(sheaf.poset.nodes())}")
print(f"Poset edges: {list(sheaf.poset.edges())}")

print("\n=== Stalk Dimensions ===")
total_dim = 0
for node_id, stalk in sheaf.stalks.items():
    print(f"Node {node_id}: {stalk.shape}")
    total_dim += stalk.shape[0]
print(f"Total stalk dimension: {total_dim}")

print("\n=== Restriction Dimensions ===")
for edge, restriction in sheaf.restrictions.items():
    print(f"Edge {edge}: {restriction.shape}")

print("\n=== Build Laplacian ===")
laplacian, metadata = build_sheaf_laplacian(sheaf, validate=False)
print(f"Laplacian shape: {laplacian.shape}")
print(f"Laplacian nnz: {laplacian.nnz}")

print("\n=== Metadata Analysis ===")
if hasattr(metadata, 'stalk_offsets'):
    print(f"Stalk offsets: {metadata.stalk_offsets}")
if hasattr(metadata, 'stalk_dimensions'):
    print(f"Stalk dimensions: {metadata.stalk_dimensions}")

print("\n=== Check for dimension mismatches ===")
if hasattr(metadata, 'stalk_offsets'):
    max_offset = max(metadata.stalk_offsets.values())
    max_dim = max(metadata.stalk_dimensions.values())
    expected_max_index = max_offset + max_dim - 1
    print(f"Max stalk offset: {max_offset}")
    print(f"Max stalk dimension: {max_dim}")
    print(f"Expected max index: {expected_max_index}")
    print(f"Actual matrix size: {laplacian.shape[0]}")
    print(f"Index overflow: {expected_max_index >= laplacian.shape[0]}")
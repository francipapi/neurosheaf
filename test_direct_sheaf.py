#!/usr/bin/env python3
"""Test script using direct sheaf construction like the working validation test."""

import torch
import torch.nn as nn
import networkx as nx
from neurosheaf.sheaf import Sheaf, build_sheaf_laplacian
from neurosheaf.spectral import PersistentSpectralAnalyzer

# Create simple sheaf like the working validation test
print("=== Creating Direct Sheaf Construction ===")

# Create a simple graph structure
graph = nx.DiGraph()
graph.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3'), ('layer3', 'layer4')])

# Create stalks as identity matrices (like validation test)
stalk_dims = {'layer1': 32, 'layer2': 24, 'layer3': 16, 'layer4': 8}
stalks = {node: torch.eye(dim, dtype=torch.float32) for node, dim in stalk_dims.items()}

# Create simple restriction maps with proper dimensions
restrictions = {}
for source, target in graph.edges():
    source_dim = stalk_dims[source]
    target_dim = stalk_dims[target]
    
    # Create simple projection (target_dim x source_dim)
    R = torch.zeros(target_dim, source_dim, dtype=torch.float32)
    min_dim = min(source_dim, target_dim)
    R[:min_dim, :min_dim] = torch.eye(min_dim) * 0.5
    restrictions[(source, target)] = R

# Create sheaf directly
sheaf = Sheaf(
    poset=graph,
    stalks=stalks,
    restrictions=restrictions,
    metadata={
        'construction_method': 'direct_test',
        'nodes': len(graph.nodes()),
        'edges': len(graph.edges()),
        'whitened': False,
        'test_construction': True
    }
)

print(f"Sheaf nodes: {list(sheaf.poset.nodes())}")
print(f"Sheaf edges: {list(sheaf.poset.edges())}")
print(f"Stalk dimensions: {[(node, stalk.shape) for node, stalk in sheaf.stalks.items()]}")
print(f"Restriction dimensions: {[(edge, R.shape) for edge, R in sheaf.restrictions.items()]}")

# Build Laplacian
print("\n=== Building Laplacian ===")
laplacian, metadata = build_sheaf_laplacian(sheaf, validate=False)
print(f"Laplacian shape: {laplacian.shape}")
print(f"Laplacian nnz: {laplacian.nnz}")

# Test persistence
print("\n=== Testing Persistence ===")
analyzer = PersistentSpectralAnalyzer(
    default_n_steps=10,
    default_filtration_type='threshold'
)

try:
    results = analyzer.analyze(
        sheaf,
        filtration_type='threshold',
        n_steps=10,
        param_range=(0.0, 1.0)
    )
    
    print("SUCCESS: Persistence analysis completed!")
    print(f"Birth events: {results['features']['num_birth_events']}")
    print(f"Death events: {results['features']['num_death_events']}")
    print(f"Infinite bars: {results['diagrams']['statistics']['n_infinite_bars']}")
    print(f"Finite pairs: {results['diagrams']['statistics']['n_finite_pairs']}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
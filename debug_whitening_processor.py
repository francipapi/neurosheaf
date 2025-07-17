#!/usr/bin/env python3
"""Debug the WhiteningProcessor rank computation issue."""

import torch
import torch.nn as nn
from neurosheaf.sheaf.core import WhiteningProcessor, compute_gram_matrices_from_activations
from neurosheaf.sheaf.extraction import extract_activations_fx

# Create the same model as in test_all.py
model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

data = torch.randn(500, 100)

print("=== WHITENING PROCESSOR INVESTIGATION ===")

# Extract activations and compute Gram matrices
activations = extract_activations_fx(model, data)
gram_matrices = compute_gram_matrices_from_activations(activations)

print("\n1. GRAM MATRIX RANKS COMPARISON")
whitening_processor = WhiteningProcessor()

for name, K in gram_matrices.items():
    print(f"\nNode {name}:")
    print(f"  Gram matrix shape: {K.shape}")
    
    # Method 1: torch.linalg.matrix_rank (current stalk method)
    rank_torch = torch.linalg.matrix_rank(K.float()).item()
    print(f"  torch.linalg.matrix_rank: {rank_torch}")
    
    # Method 2: WhiteningProcessor.whiten_gram_matrix (current restriction method)
    K_whitened, W, info = whitening_processor.whiten_gram_matrix(K)
    rank_whitening = W.shape[0]
    print(f"  WhiteningProcessor rank: {rank_whitening}")
    
    print(f"  ❌ Mismatch: {rank_torch != rank_whitening}")
    
    # Method 3: Check eigenvalues manually
    eigenvals = torch.linalg.eigvals(K).real
    eigenvals_sorted = torch.sort(eigenvals, descending=True)[0]
    
    # Count eigenvalues above different thresholds
    eps_1e12 = 1e-12
    eps_1e6 = 1e-6
    eps_default = whitening_processor.min_eigenvalue
    
    rank_1e12 = torch.sum(eigenvals_sorted > eps_1e12 * eigenvals_sorted[0]).item()
    rank_1e6 = torch.sum(eigenvals_sorted > eps_1e6 * eigenvals_sorted[0]).item()
    rank_default = torch.sum(eigenvals_sorted > eps_default * eigenvals_sorted[0]).item()
    
    print(f"  Manual count (eps=1e-12): {rank_1e12}")
    print(f"  Manual count (eps=1e-6): {rank_1e6}")
    print(f"  Manual count (eps={eps_default}): {rank_default}")
    print(f"  WhiteningProcessor.min_eigenvalue: {eps_default}")
    
    # Check what the actual activation dimensions suggest
    if name in activations:
        act = activations[name]
        expected_rank = min(act.shape)  # min(batch_size, feature_dim)
        print(f"  Expected rank (min of activation shape {act.shape}): {expected_rank}")

print("\n2. WHITENING PROCESSOR SETTINGS")
print(f"WhiteningProcessor.min_eigenvalue: {whitening_processor.min_eigenvalue}")
print(f"WhiteningProcessor.use_double_precision: {whitening_processor.use_double_precision}")

print("\n3. RECOMMENDATION")
print("The issue is likely that WhiteningProcessor uses a much smaller eigenvalue threshold")
print("than torch.linalg.matrix_rank, causing it to include more dimensions as 'significant'.")
print("We should either:")
print("1. Use consistent thresholds in both methods")
print("2. Use the same method for both stalk and restriction rank computation")
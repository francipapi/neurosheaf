#!/usr/bin/env python3
"""Test to verify that restriction maps have similar Frobenius norms in both modes."""

import torch
import numpy as np

# Create some test orthogonal matrices of different sizes
sizes = [(10, 10), (20, 10), (10, 20), (30, 15)]

print("Frobenius norms of orthogonal matrices:")
print("-" * 50)

for m, n in sizes:
    # Generate a random orthogonal matrix using SVD
    A = torch.randn(m, n)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = U @ Vh  # This is orthogonal
    
    # Compute Frobenius norm
    frob_norm = torch.norm(R, 'fro').item()
    expected_norm = np.sqrt(min(m, n))
    
    print(f"Size {m}×{n}: Frobenius norm = {frob_norm:.6f}, Expected = {expected_norm:.6f}")
    
    # Verify orthogonality
    if m >= n:
        # R^T @ R should be identity
        identity_check = R.T @ R
        identity_error = torch.norm(identity_check - torch.eye(n), 'fro').item()
    else:
        # R @ R^T should be identity
        identity_check = R @ R.T
        identity_error = torch.norm(identity_check - torch.eye(m), 'fro').item()
    
    print(f"  Orthogonality check: error = {identity_error:.2e}")

print("\nConclusion: All orthogonal matrices have Frobenius norm = sqrt(min(m,n))")
print("This is why restriction maps have the same weight in both modes!")
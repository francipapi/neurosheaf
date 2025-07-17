#!/usr/bin/env python3
"""Debug script to test restriction computation in isolation."""

import torch
import torch.nn as nn
from neurosheaf.sheaf.core import scaled_procrustes_whitened

# Create simple test case
print("Testing restriction computation...")

# Create two simple Gram matrices
K1 = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
K2 = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float32)

print(f"K1 shape: {K1.shape}")
print(f"K2 shape: {K2.shape}")

try:
    # Test restriction computation
    R, scale, info = scaled_procrustes_whitened(K1, K2, validate=True)
    print("SUCCESS: Restriction computed successfully")
    print(f"R shape: {R.shape}")
    print(f"Scale: {scale}")
    print(f"Info keys: {list(info.keys())}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test with validation disabled
try:
    R, scale, info = scaled_procrustes_whitened(K1, K2, validate=False)
    print("SUCCESS: Restriction computed successfully (validate=False)")
    print(f"R shape: {R.shape}")
    print(f"Scale: {scale}")
    print(f"Info keys: {list(info.keys())}")
except Exception as e:
    print(f"ERROR with validate=False: {e}")
    import traceback
    traceback.print_exc()
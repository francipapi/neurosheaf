#!/usr/bin/env python3
"""Debug version of test_all.py to identify issues."""

import torch.nn as nn 
import torch
import numpy as np
import os

# Set environment for CPU usage
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=== DEBUG: Starting test_all.py debugging ===")

try:
    from neurosheaf.sheaf.core.gw_config import GWConfig
    print("✅ GWConfig import OK")
except Exception as e:
    print(f"❌ GWConfig import failed: {e}")

try:
    from neurosheaf.sheaf.assembly.builder import SheafBuilder
    print("✅ SheafBuilder import OK")
except Exception as e:
    print(f"❌ SheafBuilder import failed: {e}")

try:
    from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
    print("✅ PersistentSpectralAnalyzer import OK")
except Exception as e:
    print(f"❌ PersistentSpectralAnalyzer import failed: {e}")

try:
    from neurosheaf.utils import load_model
    print("✅ load_model import OK")
except Exception as e:
    print(f"❌ load_model import failed: {e}")

try:
    from neurosheaf.api import NeurosheafAnalyzer
    print("✅ NeurosheafAnalyzer import OK")
except Exception as e:
    print(f"❌ NeurosheafAnalyzer import failed: {e}")

try:
    from neurosheaf.spectral import UnifiedStaticLaplacian
    print("✅ UnifiedStaticLaplacian import OK")
except Exception as e:
    print(f"❌ UnifiedStaticLaplacian import failed: {e}")

print("\n=== DEBUG: Testing basic model creation ===")

class MLPModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

try:
    model = MLPModel()
    print("✅ Model creation OK")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Model creation failed: {e}")

print("\n=== DEBUG: Testing data generation ===")
try:
    batch_size = 5  # Very small for debugging
    data = torch.randn(batch_size, 3)
    print(f"✅ Data generation OK: {data.shape}")
except Exception as e:
    print(f"❌ Data generation failed: {e}")

print("\n=== DEBUG: Testing GW config ===")
try:
    gw_config = GWConfig(
        epsilon=0.05,
        max_iter=10,  # Reduced for debugging
        tolerance=1e-6,
    )
    print("✅ GW config creation OK")
except Exception as e:
    print(f"❌ GW config creation failed: {e}")

print("\n=== DEBUG: Testing analyzer initialization ===")
try:
    analyzer = NeurosheafAnalyzer(device='cpu')
    print("✅ NeurosheafAnalyzer initialization OK")
except Exception as e:
    print(f"❌ NeurosheafAnalyzer initialization failed: {e}")

print("\n=== DEBUG: Testing sheaf building (this might be where it hangs) ===")
try:
    print("Starting sheaf analysis...")
    analysis = analyzer.analyze(
        model, data, 
        method='gromov_wasserstein', 
        gw_config=gw_config,
        exclude_final_single_output=False
    )
    sheaf = analysis['sheaf']
    print(f"✅ Sheaf building OK: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
except Exception as e:
    print(f"❌ Sheaf building failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG: Testing UnifiedStaticLaplacian ===")
try:
    normalized_laplacian = UnifiedStaticLaplacian(
        eigenvalue_method='auto',
        max_eigenvalues=10,  # Reduced for debugging
        use_double_precision=False,  # Use single precision for faster debugging
        validate_properties=False,   # Disable validation for debugging
        use_generalized_normalization=True,
        use_matrix_free=False
    )
    print("✅ UnifiedStaticLaplacian creation OK")
    print(f"   • Generalized normalization: {normalized_laplacian.use_generalized_normalization}")
    print(f"   • GW builder available: {normalized_laplacian.gw_builder is not None}")
except Exception as e:
    print(f"❌ UnifiedStaticLaplacian creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG: Complete ===")
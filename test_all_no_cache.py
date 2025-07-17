import torch.nn as nn 
import torch
import matplotlib.pyplot as plt
import numpy as np
from neurosheaf.sheaf import SheafBuilder, build_sheaf_laplacian, Sheaf
from neurosheaf.spectral import PersistentSpectralAnalyzer

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

laplacian, metadata = build_sheaf_laplacian(sheaf, validate=False)

# Create analyzer with caching disabled
analyzer = PersistentSpectralAnalyzer(
            default_n_steps=15, 
            default_filtration_type='threshold'
        )

# Disable caching in the static laplacian
analyzer.static_laplacian.enable_caching = False

results = analyzer.analyze(
            sheaf,
            filtration_type='threshold',
            n_steps=15,
            param_range=(0.0, 1.0)
        )

# Print results summary
print("\n=== Spectral Persistence Analysis Results ===")
print(f"Total filtration steps: {len(results['filtration_params'])}")
print(f"Birth events: {results['features']['num_birth_events']}")
print(f"Death events: {results['features']['num_death_events']}")
print(f"Crossing events: {results['features']['num_crossings']}")
print(f"Persistent paths: {results['features']['num_persistent_paths']}")
print(f"Infinite bars: {results['diagrams']['statistics']['n_infinite_bars']}")
print(f"Finite pairs: {results['diagrams']['statistics']['n_finite_pairs']}")
print(f"Mean lifetime: {results['diagrams']['statistics'].get('mean_lifetime', 0):.6f}")

print("SUCCESS: Persistence analysis completed without caching!")
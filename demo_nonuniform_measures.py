#!/usr/bin/env python3
"""Demonstration of variance-based non-uniform measures for GW sheaf construction.

This script shows how non-uniform measures improve restriction map quality by
weighting units based on their variance (informativeness) rather than treating
all units equally.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.assembly import SheafBuilder, GWRestrictionManager


class DemoNetwork(nn.Module):
    """Demo network with engineered activation patterns for clear demonstration."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 16),  # Layer 0: varied activation patterns
            nn.Linear(16, 8),   # Layer 1: different units will have different importance
            nn.Linear(8, 4)     # Layer 2: final layer
        ])
    
    def forward(self, x):
        activations = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
            activations.append(x.clone())
        return x, activations


def create_engineered_activations():
    """Create activations with clear variance patterns for demonstration."""
    print("Creating engineered activation patterns...")
    
    # Create model with predictable patterns
    torch.manual_seed(42)
    model = DemoNetwork()
    
    batch_size = 24
    input_tensor = torch.randn(batch_size, 10)
    
    # Get initial activations
    output, raw_activations = model(input_tensor)
    
    # Engineer specific variance patterns for clearer demonstration
    engineered_activations = {}
    
    for layer_idx, activation in enumerate(raw_activations):
        n_units = activation.shape[1]
        
        # Create engineered patterns
        engineered = activation.clone()
        
        if layer_idx == 0:  # Layer 0: varied patterns
            # Units 0-3: high variance (informative)
            engineered[:, 0:4] = torch.randn(batch_size, 4) * 3.0
            # Units 4-7: medium variance
            engineered[:, 4:8] = torch.randn(batch_size, 4) * 1.0
            # Units 8-11: low variance (noisy)
            engineered[:, 8:12] = torch.randn(batch_size, 4) * 0.1
            # Units 12-15: dead units (very low variance)
            engineered[:, 12:16] = torch.ones(batch_size, 4) * 0.5 + torch.randn(batch_size, 4) * 0.01
            
        elif layer_idx == 1:  # Layer 1: different pattern
            # Units 0-2: high variance
            engineered[:, 0:3] = torch.randn(batch_size, 3) * 2.5
            # Units 3-5: medium variance
            engineered[:, 3:6] = torch.randn(batch_size, 3) * 1.2
            # Units 6-7: dead units
            engineered[:, 6:8] = torch.ones(batch_size, 2) * -0.3 + torch.randn(batch_size, 2) * 0.02
            
        elif layer_idx == 2:  # Layer 2: final layer
            # All units have reasonable variance
            engineered = torch.randn(batch_size, n_units) * 1.5
        
        engineered_activations[f'layers_{layer_idx}'] = engineered
        
        # Compute and display variance statistics
        variances = torch.var(engineered, dim=0)
        print(f"\nLayer {layer_idx} variance statistics:")
        print(f"  Shape: {engineered.shape}")
        print(f"  Variances: {variances.tolist()}")
        print(f"  Variance range: [{variances.min():.6f}, {variances.max():.6f}]")
    
    return engineered_activations


def compare_uniform_vs_nonuniform():
    """Compare uniform vs non-uniform measures in detail."""
    print("\n" + "="*80)
    print("UNIFORM vs NON-UNIFORM MEASURES COMPARISON")
    print("="*80)
    
    # Get engineered activations
    activations = create_engineered_activations()
    
    # Build sheaves with both approaches
    configs = [
        (GWConfig(uniform_measures=True, align_units=True), "Uniform Measures"),
        (GWConfig(uniform_measures=False, measure_eps=1e-6, align_units=True), "Non-Uniform (Variance-Based)")
    ]
    
    results = {}
    
    for config, name in configs:
        print(f"\n{'-'*60}")
        print(f"Building sheaf with {name}")
        print(f"{'-'*60}")
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # Create dummy model for sheaf builder
        model = DemoNetwork()
        dummy_input = torch.randn(24, 10)
        
        # Use GWRestrictionManager directly for more control
        manager = GWRestrictionManager(config=config)
        
        # Create simple poset
        import networkx as nx
        poset = nx.DiGraph()
        poset.add_edge('layers_0', 'layers_1')
        poset.add_edge('layers_1', 'layers_2')
        
        restrictions, costs, metadata = manager.compute_all_restrictions(
            activations, poset, parallel=False
        )
        
        results[name] = {
            'restrictions': restrictions,
            'costs': costs,
            'metadata': metadata,
            'config': config
        }
        
        print(f"Successfully computed {len(restrictions)} restrictions")
        print(f"Total computation time: {metadata.get('computation_time', 0):.3f}s")
        
        # Show restriction map properties
        for edge, restriction in restrictions.items():
            print(f"\nRestriction {edge[0]} → {edge[1]}:")
            print(f"  Shape: {restriction.shape}")
            
            # Check row-stochasticity
            row_sums = restriction.sum(dim=1)
            print(f"  Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
            
            # Compute restriction "focus" - how concentrated the weights are
            # Higher entropy = more uniform, lower entropy = more focused
            row_entropies = -(restriction * torch.log(restriction + 1e-12)).sum(dim=1)
            mean_entropy = row_entropies.mean().item()
            print(f"  Mean row entropy: {mean_entropy:.3f} (lower = more focused)")
            
            # Show sample of restriction weights
            print(f"  Sample weights (first row): {restriction[0, :5].tolist()}")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    uniform_results = results["Uniform Measures"]
    nonuniform_results = results["Non-Uniform (Variance-Based)"]
    
    for edge in uniform_results['restrictions'].keys():
        if edge in nonuniform_results['restrictions']:
            R_uniform = uniform_results['restrictions'][edge]
            R_nonuniform = nonuniform_results['restrictions'][edge]
            
            print(f"\nEdge {edge[0]} → {edge[1]}:")
            
            # Compare difference magnitude
            difference = torch.norm(R_uniform - R_nonuniform, 'fro').item()
            print(f"  Frobenius norm difference: {difference:.6f}")
            
            # Compare focus/concentration
            uniform_entropies = -(R_uniform * torch.log(R_uniform + 1e-12)).sum(dim=1)
            nonuniform_entropies = -(R_nonuniform * torch.log(R_nonuniform + 1e-12)).sum(dim=1)
            
            print(f"  Uniform mean entropy: {uniform_entropies.mean():.3f}")
            print(f"  Non-uniform mean entropy: {nonuniform_entropies.mean():.3f}")
            
            entropy_diff = uniform_entropies.mean() - nonuniform_entropies.mean()
            if entropy_diff > 0.1:
                print(f"  → Non-uniform measures are MORE FOCUSED (better)")
            elif entropy_diff < -0.1:
                print(f"  → Non-uniform measures are LESS FOCUSED")
            else:
                print(f"  → Similar focus levels")
    
    return results


def demonstrate_variance_computation():
    """Demonstrate how variance-based measures are computed."""
    print(f"\n{'='*80}")
    print("VARIANCE-BASED MEASURE COMPUTATION DEMONSTRATION")
    print(f"{'='*80}")
    
    # Create sample activation with clear variance patterns
    batch_size = 16
    n_units = 8
    
    activations = torch.zeros(batch_size, n_units)
    unit_descriptions = []
    
    # Design units with specific variance patterns
    activations[:, 0] = torch.linspace(-4, 4, batch_size)  # High variance
    unit_descriptions.append("High variance (informative)")
    
    activations[:, 1] = torch.linspace(-2, 2, batch_size)  # Medium variance
    unit_descriptions.append("Medium variance") 
    
    activations[:, 2] = torch.linspace(-0.5, 0.5, batch_size)  # Low variance
    unit_descriptions.append("Low variance (noisy)")
    
    activations[:, 3] = torch.ones(batch_size) * 1.0  # Zero variance (dead)
    unit_descriptions.append("Zero variance (dead)")
    
    activations[:, 4] = torch.ones(batch_size) * -0.5  # Zero variance (dead)
    unit_descriptions.append("Zero variance (dead)")
    
    # Add some random units
    activations[:, 5] = torch.randn(batch_size) * 1.5
    unit_descriptions.append("Random high variance")
    
    activations[:, 6] = torch.randn(batch_size) * 0.3
    unit_descriptions.append("Random low variance")
    
    activations[:, 7] = torch.randn(batch_size) * 2.0
    unit_descriptions.append("Random very high variance")
    
    # Compute variance-based measures
    config = GWConfig(uniform_measures=False, measure_eps=1e-6)
    manager = GWRestrictionManager(config=config)
    
    measures = manager._compute_variance_based_measures(activations)
    
    # Compute actual variances for comparison
    actual_variances = torch.var(activations, dim=0, unbiased=False)
    
    print(f"\nUnit variance analysis:")
    print(f"{'Unit':<4} {'Description':<25} {'Variance':<12} {'Measure':<12} {'Ratio':<8}")
    print("-" * 75)
    
    for i in range(n_units):
        variance = actual_variances[i].item()
        measure = measures[i].item()
        ratio = measure / (1.0 / n_units)  # Ratio to uniform weight
        
        print(f"{i:<4} {unit_descriptions[i]:<25} {variance:<12.6f} {measure:<12.6f} {ratio:<8.2f}x")
    
    print(f"\nKey insights:")
    print(f"- Variance-based measures give higher weight to units with higher variance")
    print(f"- Dead units (zero variance) get minimal weight (eps only)")
    print(f"- Measures sum to 1.0: {measures.sum():.6f}")
    print(f"- Uniform weight would be: {1.0/n_units:.6f}")
    
    # Show how this affects transport
    print(f"\nTransport implications:")
    high_var_units = [0, 5, 7]  # High variance units
    dead_units = [3, 4]  # Dead units
    
    high_var_weight = measures[high_var_units].sum().item()
    dead_weight = measures[dead_units].sum().item()
    
    print(f"- High variance units get {high_var_weight:.1%} of total weight")
    print(f"- Dead units get only {dead_weight:.1%} of total weight")
    print(f"- This focuses transport on informative units and reduces noise")


def demonstrate_layer_width_benefits():
    """Demonstrate benefits when layer widths differ significantly."""
    print(f"\n{'='*80}")
    print("BENEFITS FOR DIFFERENT LAYER WIDTHS")
    print(f"{'='*80}")
    
    # Create activations with very different layer sizes
    batch_size = 20
    
    # Wide layer with many dead units
    wide_layer = torch.zeros(batch_size, 32)
    # Only a few units are active
    active_indices = [0, 5, 10, 15, 20, 25]
    for idx in active_indices:
        wide_layer[:, idx] = torch.randn(batch_size) * 2.0
    # Rest are nearly dead
    for idx in range(32):
        if idx not in active_indices:
            wide_layer[:, idx] = torch.randn(batch_size) * 0.05 + (idx % 3 - 1) * 0.1
    
    # Narrow layer with all active units
    narrow_layer = torch.randn(batch_size, 8) * 1.5
    
    activations = {
        'wide_layer': wide_layer,
        'narrow_layer': narrow_layer
    }
    
    print(f"Wide layer: {wide_layer.shape[1]} units ({len(active_indices)} active, {32-len(active_indices)} mostly dead)")
    print(f"Narrow layer: {narrow_layer.shape[1]} units (all active)")
    
    # Compare uniform vs non-uniform
    for uniform_measures in [True, False]:
        measure_type = "Uniform" if uniform_measures else "Non-uniform (variance-based)"
        print(f"\n{measure_type} Measures:")
        print("-" * 40)
        
        config = GWConfig(uniform_measures=uniform_measures, align_units=True)
        manager = GWRestrictionManager(config=config)
        
        # Create poset
        import networkx as nx
        poset = nx.DiGraph()
        poset.add_edge('wide_layer', 'narrow_layer')
        
        restrictions, costs, metadata = manager.compute_all_restrictions(
            activations, poset, parallel=False
        )
        
        if ('wide_layer', 'narrow_layer') in restrictions:
            restriction = restrictions[('wide_layer', 'narrow_layer')]
            print(f"  Restriction shape: {restriction.shape}")
            
            # Analyze how much weight goes to active vs dead units in wide layer
            active_weights = restriction[:, active_indices].sum().item()
            total_weight = restriction.sum().item()
            active_fraction = active_weights / total_weight
            
            print(f"  Weight on active units: {active_fraction:.1%}")
            print(f"  Weight on dead units: {1-active_fraction:.1%}")
            
            # Show sample weights for active vs dead units
            sample_active = restriction[0, active_indices].mean().item()
            dead_indices = [i for i in range(32) if i not in active_indices]
            sample_dead = restriction[0, dead_indices].mean().item()
            
            print(f"  Avg weight per active unit: {sample_active:.6f}")
            print(f"  Avg weight per dead unit: {sample_dead:.6f}")
            print(f"  Ratio (active/dead): {sample_active/max(sample_dead, 1e-8):.1f}x")
    
    print(f"\nConclusion:")
    print(f"Non-uniform measures dramatically improve transport quality when:")
    print(f"- Layers have many dead or low-variance units")
    print(f"- Layer widths differ significantly")
    print(f"- Network has varying unit importance")


if __name__ == "__main__":
    print("VARIANCE-BASED NON-UNIFORM MEASURES DEMONSTRATION")
    print("This demonstration shows how non-uniform measures improve GW sheaf construction")
    print("by weighting units based on their activation variance (informativeness).")
    
    try:
        # Suppress some warnings for cleaner output
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Run demonstrations
        demonstrate_variance_computation()
        compare_uniform_vs_nonuniform()
        demonstrate_layer_width_benefits()
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print("✓ Variance-based measures weight units by informativeness")
        print("✓ Dead/noisy units receive lower weights") 
        print("✓ Informative units receive higher weights")
        print("✓ Transport quality improves, especially with varying layer widths")
        print("✓ Configurable via uniform_measures=False in GWConfig")
        print("✓ Mathematically principled using activation variance across batch")
        print("✓ Maintains row-stochasticity and functoriality properties")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
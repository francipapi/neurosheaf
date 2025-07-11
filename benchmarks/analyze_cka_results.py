#!/usr/bin/env python3
"""Analyze the CKA matrix results from ResNet validation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def analyze_cka_matrix():
    """Load and analyze the CKA matrix results."""
    # Load the matrix
    cka_matrix = np.load('/Users/francescopapini/GitRepo/neurosheaf/cka_resnet50.npy')
    
    print("=== CKA MATRIX ANALYSIS ===")
    print(f"Matrix shape: {cka_matrix.shape}")
    print(f"Matrix range: [{cka_matrix.min():.4f}, {cka_matrix.max():.4f}]")
    print(f"Matrix mean: {cka_matrix.mean():.4f}")
    print(f"Matrix std: {cka_matrix.std():.4f}")
    
    # Layer names from the script
    layer_names = [
        "conv1",
        "layer1.0.conv1", "layer1.2.conv3",
        "layer2.0.conv1", "layer2.3.conv3", 
        "layer3.0.conv1", "layer3.5.conv3",
        "layer4.0.conv1", "layer4.2.conv3",
        "avgpool", "fc"
    ]
    
    print("\n=== FULL CKA MATRIX ===")
    print("Layer names:")
    for i, name in enumerate(layer_names):
        print(f"{i:2d}: {name}")
    
    print("\nCKA Matrix:")
    print("    ", end="")
    for i in range(11):
        print(f"{i:6d}", end="")
    print()
    
    for i in range(11):
        print(f"{i:2d}: ", end="")
        for j in range(11):
            print(f"{cka_matrix[i,j]:6.3f}", end="")
        print(f"  {layer_names[i]}")
    
    print("\n=== DIAGONAL ANALYSIS ===")
    diag_values = np.diag(cka_matrix)
    print(f"Diagonal values: {diag_values}")
    print(f"All diagonal ≈ 1.0: {np.allclose(diag_values, 1.0, atol=1e-2)}")
    
    print("\n=== STAGE ANALYSIS ===")
    # Define stages according to ResNet architecture
    stages = {
        'conv1': [0],
        'layer1': [1, 2],
        'layer2': [3, 4], 
        'layer3': [5, 6],
        'layer4': [7, 8],
        'final': [9, 10]  # avgpool, fc
    }
    
    # Within-stage similarities
    within_stage_values = []
    print("Within-stage similarities:")
    for stage_name, indices in stages.items():
        stage_vals = []
        for i in indices:
            for j in indices:
                if i < j:  # Upper triangle only
                    val = cka_matrix[i, j]
                    stage_vals.append(val)
                    within_stage_values.append(val)
        if stage_vals:
            print(f"  {stage_name}: {np.mean(stage_vals):.4f} (values: {stage_vals})")
    
    within_stage_mean = np.mean(within_stage_values) if within_stage_values else 0
    print(f"Overall within-stage mean: {within_stage_mean:.4f}")
    
    # Cross-stage similarities (early vs late)
    early_indices = [0, 1, 2]  # conv1, layer1.*
    late_indices = [7, 8, 9, 10]  # layer4.*, avgpool, fc
    
    cross_stage_values = []
    print("\nCross-stage similarities (early vs late):")
    for i in early_indices:
        for j in late_indices:
            val = cka_matrix[i, j]
            cross_stage_values.append(val)
            print(f"  {layer_names[i]} vs {layer_names[j]}: {val:.4f}")
    
    cross_stage_mean = np.mean(cross_stage_values)
    print(f"Early-vs-late mean: {cross_stage_mean:.4f}")
    
    print("\n=== LITERATURE COMPARISON ===")
    # Create expected pattern (coarse template)
    def literature_template(n_layers):
        template = np.zeros((n_layers, n_layers))
        # Stage boundaries: conv1=0, layer1=[1,2], layer2=[3,4], layer3=[5,6], layer4=[7,8], final=[9,10]
        boundaries = [0, 1, 3, 5, 7, 9, 11]
        for s in range(len(boundaries) - 1):
            i, j = boundaries[s], boundaries[s + 1]
            template[i:j, i:j] = 1.0 * (s + 1)
        template /= template.max()
        return template
    
    template = literature_template(11)
    
    # Compare upper triangles
    iu = np.triu_indices(11, k=1)
    rho, p_value = spearmanr(cka_matrix[iu], template[iu])
    
    print(f"Spearman correlation with template: ρ = {rho:.4f} (p = {p_value:.4f})")
    
    print("\n=== VALIDATION RESULTS ===")
    print(f"Diagonal ≈ 1.0: {'PASS' if np.allclose(diag_values, 1.0, atol=1e-2) else 'FAIL'}")
    print(f"Within-stage mean > 0.6: {'PASS' if within_stage_mean > 0.6 else 'FAIL'} ({within_stage_mean:.4f})")
    print(f"Early-vs-late mean < 0.3: {'PASS' if cross_stage_mean < 0.3 else 'FAIL'} ({cross_stage_mean:.4f})")
    print(f"Spearman ρ > 0.7: {'PASS' if rho > 0.7 else 'FAIL'} (ρ={rho:.4f})")
    
    # Plot the matrix
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Actual CKA matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cka_matrix, 
                xticklabels=[f"{i}\n{name}" for i, name in enumerate(layer_names)],
                yticklabels=[f"{i}\n{name}" for i, name in enumerate(layer_names)],
                annot=True, fmt='.3f', cmap='viridis', 
                vmin=0, vmax=1, cbar_kws={'label': 'CKA'})
    plt.title('Actual CKA Matrix (500 images)')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    
    # Plot 2: Expected template
    plt.subplot(1, 2, 2)
    sns.heatmap(template,
                xticklabels=[f"{i}\n{name}" for i, name in enumerate(layer_names)],
                yticklabels=[f"{i}\n{name}" for i, name in enumerate(layer_names)],
                annot=True, fmt='.3f', cmap='viridis',
                vmin=0, vmax=1, cbar_kws={'label': 'Expected'})
    plt.title('Expected Block-Diagonal Pattern')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    
    plt.tight_layout()
    plt.savefig('/Users/francescopapini/GitRepo/neurosheaf/cka_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: /Users/francescopapini/GitRepo/neurosheaf/cka_analysis.png")
    
    return cka_matrix, template, rho

if __name__ == "__main__":
    analyze_cka_matrix()
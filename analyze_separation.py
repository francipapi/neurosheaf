#!/usr/bin/env python3
"""
Quick script to analyze the separation quality of the elastic distance results.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def classify_model_type(filename: str) -> str:
    """Classify model as trained or random based on filename."""
    filename_lower = filename.lower()
    if 'random' in filename_lower:
        return 'random'
    elif 'seed' in filename_lower:
        return 'trained'
    else:
        return 'unknown'

# Load the optimal results
distance_matrix = np.load('elastic_eigs_optimal_distance.npy')
with open('elastic_eigs_optimal_index.json', 'r') as f:
    file_index = json.load(f)

print(f"Loaded distance matrix: {distance_matrix.shape}")
print(f"Number of models: {len(file_index)}")

# Classify models
labels = [classify_model_type(f) for f in file_index]
trained_mask = np.array([l == 'trained' for l in labels])
random_mask = np.array([l == 'random' for l in labels])

print(f"Trained models: {np.sum(trained_mask)}")
print(f"Random models: {np.sum(random_mask)}")

# Extract distances
trained_indices = np.where(trained_mask)[0]
random_indices = np.where(random_mask)[0]

# Intra-class distances
if len(trained_indices) > 1:
    trained_distances = distance_matrix[np.ix_(trained_indices, trained_indices)]
    triu_mask = np.triu(np.ones_like(trained_distances, dtype=bool), k=1)
    intra_trained = trained_distances[triu_mask]
else:
    intra_trained = np.array([])

if len(random_indices) > 1:
    random_distances = distance_matrix[np.ix_(random_indices, random_indices)]
    triu_mask = np.triu(np.ones_like(random_distances, dtype=bool), k=1)
    intra_random = random_distances[triu_mask]
else:
    intra_random = np.array([])

# Inter-class distances
inter_distances = distance_matrix[np.ix_(trained_indices, random_indices)]
inter_class = inter_distances.flatten()

# Combine intra-class distances
intra_class = np.concatenate([intra_trained, intra_random])

print("\n" + "="*60)
print("SEPARATION ANALYSIS RESULTS")
print("="*60)

mean_intra = np.mean(intra_class)
mean_inter = np.mean(inter_class)
inter_intra_ratio = mean_inter / mean_intra

print(f"Mean intra-class distance: {mean_intra:.4f}")
print(f"Mean inter-class distance: {mean_inter:.4f}")
print(f"Inter/Intra ratio: {inter_intra_ratio:.4f}")

# Statistical test
t_stat, p_value = ttest_ind(inter_class, intra_class)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.2e}")

# Additional statistics
print(f"\nIntra-class distances: min={np.min(intra_class):.4f}, max={np.max(intra_class):.4f}, std={np.std(intra_class):.4f}")
print(f"Inter-class distances: min={np.min(inter_class):.4f}, max={np.max(inter_class):.4f}, std={np.std(inter_class):.4f}")

# Separation score
separation_score = inter_intra_ratio * abs(t_stat) / (1 + abs(p_value))
print(f"\nSeparation Score: {separation_score:.4f}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of distances
ax1.hist(intra_class, bins=20, alpha=0.7, label='Intra-class (same type)', color='blue', density=True)
ax1.hist(inter_class, bins=20, alpha=0.7, label='Inter-class (different types)', color='red', density=True)
ax1.set_xlabel('Elastic Distance')
ax1.set_ylabel('Density')
ax1.set_title('Distance Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot
data_for_box = [intra_class, inter_class]
ax2.boxplot(data_for_box, labels=['Intra-class', 'Inter-class'])
ax2.set_ylabel('Elastic Distance')
ax2.set_title('Distance Comparison')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('separation_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to separation_analysis.png")

# Print model breakdown
print("\n" + "="*60)
print("MODEL BREAKDOWN")
print("="*60)
for i, (filename, label) in enumerate(zip(file_index, labels)):
    print(f"{i:2d}: {filename} [{label}]")
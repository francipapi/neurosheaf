# Multivariate DTW Neural Network Similarity Analysis Report
Generated: 2025-07-18 18:55:59

## Analysis Overview

This report presents the results of a comprehensive multivariate Dynamic Time Warping (DTW)
analysis comparing functional similarity between neural networks based on their full
eigenvalue evolution during spectral analysis.

## Model Summary

### MLPModel
**3 model(s):**

- **torch_mlp_acc_0.9857_epoch_100**
  - File Size: 0.04 MB
  - Training Epoch: 100
  - Accuracy: 0.9857

- **torch_mlp_acc_1.0000_epoch_200**
  - File Size: 0.04 MB
  - Training Epoch: 200
  - Accuracy: 1.0000

- **random_mlp_net_000_default_seed_42**
  - File Size: 0.04 MB

### ActualCustomModel
**2 model(s):**

- **random_custom_net_000_default_seed_42**
  - File Size: 0.02 MB

- **torch_custom_acc_1.0000_epoch_200**
  - File Size: 0.02 MB
  - Training Epoch: 200
  - Accuracy: 1.0000

## Analysis Configuration

- **Method**: Multivariate DTW (Dynamic Time Warping)
- **Eigenvalue Analysis**: Full eigenvalue evolution (all eigenvalues)
- **Data Shape**: [150, 3]
- **Device**: cpu
- **Total Pairwise Comparisons**: 10

## Similarity Matrix

**Multivariate DTW Similarity Scores** (1.0 = identical functional behavior, 0.0 = completely different)

| Model | torch_mlp_ac... | random_custo... | torch_mlp_ac... | random_mlp_n... | torch_custom... |
|-------|-------|-------|-------|-------|-------|
| torch_mlp_ac... | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random_custo... | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| torch_mlp_ac... | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| random_mlp_n... | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| torch_custom... | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## Top Similarity Pairs

**Most functionally similar model pairs based on multivariate DTW:**

1. **torch_mlp_acc_0.9857_epoch_100 ↔ random_custom_net_000_default_seed_42**
   - Similarity: 0.0000
   - DTW Distance: 2.8340

2. **torch_mlp_acc_0.9857_epoch_100 ↔ torch_mlp_acc_1.0000_epoch_200**
   - Similarity: 0.0000
   - DTW Distance: 2.7211

3. **torch_mlp_acc_0.9857_epoch_100 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 0.0000
   - DTW Distance: 2.8340

4. **torch_mlp_acc_0.9857_epoch_100 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.0000
   - DTW Distance: 2.8974

5. **random_custom_net_000_default_seed_42 ↔ torch_mlp_acc_1.0000_epoch_200**
   - Similarity: 0.0000
   - DTW Distance: 2.6711

6. **random_custom_net_000_default_seed_42 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 0.0000
   - DTW Distance: 2.9000

7. **random_custom_net_000_default_seed_42 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.0000
   - DTW Distance: 2.8152

8. **torch_mlp_acc_1.0000_epoch_200 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 0.0000
   - DTW Distance: 2.6711

9. **torch_mlp_acc_1.0000_epoch_200 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.0000
   - DTW Distance: 2.7314

10. **random_mlp_net_000_default_seed_42 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.0000
   - DTW Distance: 2.8152

## Architecture-Based Analysis

### Same Architecture Comparisons
- **Average Similarity**: 0.0000
- **Count**: 4 pairs
- **Range**: [0.0000, 0.0000]

### Cross-Architecture Comparisons
- **Average Similarity**: 0.0000
- **Count**: 6 pairs
- **Range**: [0.0000, 0.0000]

**🔍 Insight**: Same-architecture models show higher functional similarity than cross-architecture models.
This suggests that architecture plays a significant role in functional behavior.

## Clustering Analysis

**Unsupervised clustering based on DTW similarity patterns:**

- **Number of Clusters**: 3
- **Silhouette Score**: 0.0068
  (Higher scores indicate better-defined clusters)

### Cluster Assignments

**Cluster_0**: torch_mlp_acc_0.9857_epoch_100
**Cluster_1**: random_custom_net_000_default_seed_42, torch_mlp_acc_1.0000_epoch_200, torch_custom_acc_1.0000_epoch_200
**Cluster_2**: random_mlp_net_000_default_seed_42

## Technical Details

### Multivariate DTW Analysis

- **Method**: Dynamic Time Warping applied to full eigenvalue evolution sequences
- **Multivariate**: All eigenvalues considered simultaneously (not just single eigenvalue)
- **Eigenvalue Evolution**: Spectral analysis across filtration parameters
- **Similarity Conversion**: Similarity = 1 - normalized_DTW_distance
- **Architecture Agnostic**: Pure functional similarity ignoring structural differences

### Interpretation Guidelines

- **High Similarity (>0.8)**: Models exhibit very similar functional behavior patterns
- **Moderate Similarity (0.4-0.8)**: Models share some functional characteristics
- **Low Similarity (<0.4)**: Models have distinctly different functional behaviors
- **Cross-Architecture**: DTW enables fair comparison across different architectures

## Summary

- **Overall Average Similarity**: 0.0000
- **Maximum Similarity**: 0.0000
- **Minimum Similarity**: 0.0000
- **Models Analyzed**: 5
- **Successful Comparisons**: 10

**Overall Assessment**: Models show diverse functional behaviors with limited similarity.
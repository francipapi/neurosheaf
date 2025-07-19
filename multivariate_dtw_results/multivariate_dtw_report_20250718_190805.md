# Multivariate DTW Neural Network Similarity Analysis Report
Generated: 2025-07-18 19:08:05

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
| torch_mlp_ac... | 1.000 | 2.834 | 2.721 | 2.834 | 2.897 |
| random_custo... | 2.834 | 1.000 | 2.671 | 2.900 | 2.815 |
| torch_mlp_ac... | 2.721 | 2.671 | 1.000 | 2.671 | 2.731 |
| random_mlp_n... | 2.834 | 2.900 | 2.671 | 1.000 | 2.815 |
| torch_custom... | 2.897 | 2.815 | 2.731 | 2.815 | 1.000 |

## Top Similarity Pairs

**Most functionally similar model pairs based on multivariate DTW:**

1. **random_custom_net_000_default_seed_42 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 2.9000
   - DTW Distance: 2.9000

2. **torch_mlp_acc_0.9857_epoch_100 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 2.8974
   - DTW Distance: 2.8974

3. **torch_mlp_acc_0.9857_epoch_100 ↔ random_custom_net_000_default_seed_42**
   - Similarity: 2.8340
   - DTW Distance: 2.8340

4. **torch_mlp_acc_0.9857_epoch_100 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 2.8340
   - DTW Distance: 2.8340

5. **random_custom_net_000_default_seed_42 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 2.8152
   - DTW Distance: 2.8152

6. **random_mlp_net_000_default_seed_42 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 2.8152
   - DTW Distance: 2.8152

7. **torch_mlp_acc_1.0000_epoch_200 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 2.7314
   - DTW Distance: 2.7314

8. **torch_mlp_acc_0.9857_epoch_100 ↔ torch_mlp_acc_1.0000_epoch_200**
   - Similarity: 2.7211
   - DTW Distance: 2.7211

9. **random_custom_net_000_default_seed_42 ↔ torch_mlp_acc_1.0000_epoch_200**
   - Similarity: 2.6711
   - DTW Distance: 2.6711

10. **torch_mlp_acc_1.0000_epoch_200 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 2.6711
   - DTW Distance: 2.6711

## Architecture-Based Analysis

### Same Architecture Comparisons
- **Average Similarity**: 2.7603
- **Count**: 4 pairs
- **Range**: [2.6711, 2.8340]

### Cross-Architecture Comparisons
- **Average Similarity**: 2.8082
- **Count**: 6 pairs
- **Range**: [2.6711, 2.9000]

**🔍 Insight**: Cross-architecture models show higher functional similarity than same-architecture models!
This suggests that functional behavior is more influenced by training than architecture.

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

- **Overall Average Similarity**: 2.7890
- **Maximum Similarity**: 2.9000
- **Minimum Similarity**: 2.6711
- **Models Analyzed**: 5
- **Successful Comparisons**: 10

**Overall Assessment**: Models show high functional similarity despite architectural differences.
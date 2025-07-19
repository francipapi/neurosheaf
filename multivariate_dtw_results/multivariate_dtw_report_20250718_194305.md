# Multivariate DTW Neural Network Similarity Analysis Report
Generated: 2025-07-18 19:43:06

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
| torch_mlp_ac... | 1.000 | 0.921 | 0.839 | 0.921 | 0.991 |
| random_custo... | 0.921 | 1.000 | 0.737 | 1.000 | 0.900 |
| torch_mlp_ac... | 0.839 | 0.737 | 1.000 | 0.737 | 0.857 |
| random_mlp_n... | 0.921 | 1.000 | 0.737 | 1.000 | 0.900 |
| torch_custom... | 0.991 | 0.900 | 0.857 | 0.900 | 1.000 |

## Top Similarity Pairs

**Most functionally similar model pairs based on multivariate DTW:**

1. **random_custom_net_000_default_seed_42 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 1.0000
   - DTW Distance: 1.0000

2. **torch_mlp_acc_0.9857_epoch_100 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.9910
   - DTW Distance: 0.9910

3. **torch_mlp_acc_0.9857_epoch_100 ↔ random_custom_net_000_default_seed_42**
   - Similarity: 0.9209
   - DTW Distance: 0.9209

4. **torch_mlp_acc_0.9857_epoch_100 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 0.9209
   - DTW Distance: 0.9209

5. **random_custom_net_000_default_seed_42 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.9004
   - DTW Distance: 0.9004

6. **random_mlp_net_000_default_seed_42 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.9004
   - DTW Distance: 0.9004

7. **torch_mlp_acc_1.0000_epoch_200 ↔ torch_custom_acc_1.0000_epoch_200**
   - Similarity: 0.8570
   - DTW Distance: 0.8570

8. **torch_mlp_acc_0.9857_epoch_100 ↔ torch_mlp_acc_1.0000_epoch_200**
   - Similarity: 0.8393
   - DTW Distance: 0.8393

9. **random_custom_net_000_default_seed_42 ↔ torch_mlp_acc_1.0000_epoch_200**
   - Similarity: 0.7368
   - DTW Distance: 0.7368

10. **torch_mlp_acc_1.0000_epoch_200 ↔ random_mlp_net_000_default_seed_42**
   - Similarity: 0.7368
   - DTW Distance: 0.7368

## Architecture-Based Analysis

### Same Architecture Comparisons
- **Average Similarity**: 0.8493
- **Count**: 4 pairs
- **Range**: [0.7368, 0.9209]

### Cross-Architecture Comparisons
- **Average Similarity**: 0.9010
- **Count**: 6 pairs
- **Range**: [0.7368, 1.0000]

**🔍 Insight**: Cross-architecture models show higher functional similarity than same-architecture models!
This suggests that functional behavior is more influenced by training than architecture.

## Clustering Analysis

**Unsupervised clustering based on DTW similarity patterns:**

- **Number of Clusters**: 3
- **Silhouette Score**: 0.0405
  (Higher scores indicate better-defined clusters)

### Cluster Assignments

**Cluster_2**: torch_mlp_acc_0.9857_epoch_100
**Cluster_0**: random_custom_net_000_default_seed_42, torch_mlp_acc_1.0000_epoch_200
**Cluster_1**: random_mlp_net_000_default_seed_42, torch_custom_acc_1.0000_epoch_200

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

- **Overall Average Similarity**: 0.8803
- **Maximum Similarity**: 1.0000
- **Minimum Similarity**: 0.7368
- **Models Analyzed**: 5
- **Successful Comparisons**: 10

**Overall Assessment**: Models show high functional similarity despite architectural differences.
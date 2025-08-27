# Elastic Distance Analysis Report

## Executive Summary

⚠️ **PARTIAL SUCCESS**: The elastic distance shows some separation between model types.

- **Margin**: -0.6466 (negative indicates overlap)
- **Separation ratio**: 4.748

## Dataset Overview

- Total models analyzed: **80**
- Trained models: **40**
- Random models: **40**
- Other models: **0**

## Statistical Analysis

### Distance Statistics

| Metric | Within Trained | Cross (T vs R) | Difference |
|--------|---------------|----------------|------------|
| Mean | 0.3212 | 1.5250 | +1.2038 |
| Std | 0.1209 | 0.3733 | +0.2524 |
| Median | 0.3014 | 1.6255 | +1.3241 |
| 95% CI | [0.3127, 0.3298] | [1.5067, 1.5430] | - |

### Effect Sizes

- **Cohen's d**: 4.339 (very large)
- **Hedge's g**: 4.337
- **CLES**: 0.999 (probability that a random cross distance > within distance)

### Statistical Significance

| Test | P-value | Significant (α=0.05) |
|------|---------|---------------------|
| Mann-Whitney U | 0.000e+00 | Yes |
| Welch's t-test | 0.000e+00 | Yes |
| Permutation test | 2.000e-04 | Yes |

## Clustering Analysis

- **Optimal number of clusters**: 2
- **Silhouette score**: 0.667
- **Davies-Bouldin index**: 0.426

### Cluster Composition

| Cluster | Trained | Random | Other |
|---------|---------|--------|-------|
| 2 | 0 | 40 | 0 |
| 1 | 40 | 0 | 0 |

## Classification Performance

- **AUC-ROC**: 0.000
- **Optimal threshold**: inf
- **Accuracy at optimal threshold**: 0.500
- **F1 score**: 0.667

## Conclusions

The elastic distance metric based on eigenvalue evolution curves shows **promising separation** between trained and random models, though complete separation was not achieved. Further refinement of the distance metric or feature extraction may improve performance.

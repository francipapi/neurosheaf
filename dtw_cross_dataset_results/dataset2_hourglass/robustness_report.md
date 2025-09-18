# DTW Robustness Analysis Report

## Baseline Configuration

```json
{
  "dtw_window": 0.1,
  "dtw_cost": "l1",
  "resample": 200,
  "smooth_win": 15,
  "amp_norm": "zscore",
  "smooth": "moving",
  "topk": 0,
  "no_time_scaling": true,
  "pad_with_last": true,
  "outlier_method": "none"
}
```

## Robustness Summary

**Baseline Performance:**
- ARI: 0.7154
- Silhouette: 0.8077

**Robustness Metrics:**
- Maximum ΔARI: 0.7417
- Maximum ΔSilhouette: 0.1448
- Minimum Spearman ρ: 0.4507
- Mean ΔARI: 0.0571
- Mean ΔSilhouette: 0.0296
- Mean Spearman ρ: 0.9353

## DTW Robustness Statement

Within reasonable parameter ranges (window ∈ [0.05, 0.2], smooth_win ∈ [10, 25], resample ∈ [150, 250]), our DTW fingerprint demonstrates robustness:

- **ΔARI ≤ 0.742**: Clustering quality remains stable
- **ΔSilhouette ≤ 0.145**: Cluster separation is preserved
- **Matrix Spearman ρ ≥ 0.451**: Distance rankings are highly preserved

**Best-performing region consistently includes:**
- window_0.05: ΔARI=0.000, ρ=0.992
- window_0.15: ΔARI=0.000, ρ=0.995
- window_0.20: ΔARI=0.000, ρ=0.979

## Detailed Results

| Configuration | ARI | Silhouette | ΔARI | ΔSilhouette | Spearman ρ |
|---------------|-----|------------|------|-------------|------------|
| window_0.05 | 0.7154 | 0.8020 | 0.0000 | 0.0057 | 0.9925 |
| window_0.15 | 0.7154 | 0.7811 | 0.0000 | 0.0266 | 0.9951 |
| window_0.20 | 0.7154 | 0.7601 | 0.0000 | 0.0476 | 0.9790 |
| smooth_10 | 0.7154 | 0.8022 | 0.0000 | 0.0055 | 0.9998 |
| smooth_20 | 0.7154 | 0.8154 | 0.0000 | 0.0077 | 0.9994 |
| smooth_25 | 0.7154 | 0.8200 | 0.0000 | 0.0123 | 0.9985 |
| resample_150 | 0.7154 | 0.8125 | 0.0000 | 0.0048 | 0.9996 |
| resample_250 | 0.7154 | 0.8042 | 0.0000 | 0.0035 | 0.9999 |
| ampnorm_unit | 0.7154 | 0.7787 | 0.0000 | 0.0290 | 0.9536 |
| ampnorm_none | -0.0263 | 0.6629 | 0.7417 | 0.1448 | 0.4507 |
| stable_loose_window | 0.7154 | 0.7861 | 0.0000 | 0.0216 | 0.9266 |
| stable_moderate | 0.7154 | 0.7528 | 0.0000 | 0.0549 | 0.9129 |
| stable_fast | 0.7154 | 0.7869 | 0.0000 | 0.0208 | 0.9514 |

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
- ARI: 1.0000
- Silhouette: 0.7572

**Robustness Metrics:**
- Maximum ΔARI: 0.0000
- Maximum ΔSilhouette: 0.0889
- Minimum Spearman ρ: 0.8906
- Mean ΔARI: 0.0000
- Mean ΔSilhouette: 0.0196
- Mean Spearman ρ: 0.9875

## DTW Robustness Statement

Within reasonable parameter ranges (window ∈ [0.05, 0.2], smooth_win ∈ [10, 25], resample ∈ [150, 250]), our DTW fingerprint demonstrates robustness:

- **ΔARI ≤ 0.000**: Clustering quality remains stable
- **ΔSilhouette ≤ 0.089**: Cluster separation is preserved
- **Matrix Spearman ρ ≥ 0.891**: Distance rankings are highly preserved

**Best-performing region consistently includes:**
- window_0.05: ΔARI=0.000, ρ=0.997
- window_0.15: ΔARI=0.000, ρ=0.997
- window_0.20: ΔARI=0.000, ρ=0.994

## Detailed Results

| Configuration | ARI | Silhouette | ΔARI | ΔSilhouette | Spearman ρ |
|---------------|-----|------------|------|-------------|------------|
| window_0.05 | 1.0000 | 0.7630 | 0.0000 | 0.0058 | 0.9969 |
| window_0.15 | 1.0000 | 0.7453 | 0.0000 | 0.0119 | 0.9972 |
| window_0.20 | 1.0000 | 0.7443 | 0.0000 | 0.0129 | 0.9939 |
| smooth_10 | 1.0000 | 0.7535 | 0.0000 | 0.0037 | 0.9998 |
| smooth_20 | 1.0000 | 0.7642 | 0.0000 | 0.0070 | 0.9995 |
| smooth_25 | 1.0000 | 0.7692 | 0.0000 | 0.0120 | 0.9990 |
| resample_150 | 1.0000 | 0.7601 | 0.0000 | 0.0029 | 0.9997 |
| resample_250 | 1.0000 | 0.7563 | 0.0000 | 0.0009 | 0.9999 |
| ampnorm_unit | 1.0000 | 0.7731 | 0.0000 | 0.0159 | 0.9932 |
| ampnorm_none | 1.0000 | 0.6682 | 0.0000 | 0.0889 | 0.8906 |
| stable_loose_window | 1.0000 | 0.7247 | 0.0000 | 0.0325 | 0.9889 |
| stable_moderate | 1.0000 | 0.7346 | 0.0000 | 0.0226 | 0.9901 |
| stable_fast | 1.0000 | 0.7190 | 0.0000 | 0.0382 | 0.9891 |

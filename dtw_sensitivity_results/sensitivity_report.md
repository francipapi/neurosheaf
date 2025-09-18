# DTW Sensitivity Analysis Report

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

### Baseline Results

- Mean distance: 3.098022
- Distance std: 2.666547
- Distance range: [0.107815, 10.930564]
- Computation time: 0.19s
- Matrix rank: 80
- Condition number: 1.39e+04

## Parameter Sensitivity Summary

### dtw_window

| Value | Mean Distance | Std | Min | Max |
|-------|---------------|-----|-----|-----|
| 0.05 | 4.740438 | 1.228553 | 3.841227 | 6.140258 |
| 0.1 | 4.624565 | 5.619764 | 0.645824 | 18.142898 |
| 0.2 | 2.286046 | nan | 2.286046 | 2.286046 |
| 0.3 | 2.205668 | nan | 2.205668 | 2.205668 |
| 0.5 | 1.684503 | 0.732325 | 1.166672 | 2.202335 |
| 1.0 | 2.202328 | nan | 2.202328 | 2.202328 |

### dtw_cost

| Value | Mean Distance | Std | Min | Max |
|-------|---------------|-----|-----|-----|
| l1 | 2.765772 | 1.343764 | 0.659731 | 6.140258 |
| l2 | 11.539942 | 9.505128 | 0.645824 | 18.142898 |

### resample

| Value | Mean Distance | Std | Min | Max |
|-------|---------------|-----|-----|-----|
| 50 | 1.246437 | 0.112806 | 1.166672 | 1.326203 |
| 100 | 2.099017 | nan | 2.099017 | 2.099017 |
| 200 | 4.323279 | 5.060113 | 0.645824 | 18.142898 |
| 400 | 5.319682 | 1.160469 | 4.499107 | 6.140258 |

### smooth_win

| Value | Mean Distance | Std | Min | Max |
|-------|---------------|-----|-----|-----|
| 1 | 18.142898 | nan | 18.142898 | 18.142898 |
| 5 | 3.291244 | nan | 3.291244 | 3.291244 |
| 10 | 3.170463 | nan | 3.170463 | 3.170463 |
| 15 | 3.338467 | 3.773090 | 0.645824 | 15.831103 |
| 20 | 3.010806 | nan | 3.010806 | 3.010806 |
| 30 | 3.355645 | 0.686716 | 2.870064 | 3.841227 |

### amp_norm

| Value | Mean Distance | Std | Min | Max |
|-------|---------------|-----|-----|-----|
| none | 0.652778 | 0.009834 | 0.645824 | 0.659731 |
| unit | 1.474867 | nan | 1.474867 | 1.474867 |
| zscore | 4.534627 | 4.699404 | 1.166672 | 18.142898 |

## Key Findings

### Most Sensitive Parameters (by coefficient of variation)

1. **smooth_win**: CV = 1.0647
1. **amp_norm**: CV = 0.9211
1. **dtw_cost**: CV = 0.8674
1. **resample**: CV = 0.5836
1. **dtw_window**: CV = 0.4578

### Computational Performance

- Fastest configuration: w0.500_l1_r50_s15_zscore (0.18s)
- Slowest configuration: w0.100_l2_r200_s15_zscore (1.75s)
- Average computation time: 0.49s

### Distance Matrix Properties

- Best conditioned matrix: w0.500_l1_r50_s15_zscore (cond = 4.82e+03)
- Highest rank matrices: 80/21 configurations
- Average sparsity: 0.0000

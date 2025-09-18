# Comprehensive Parameter List

This document provides a complete list of all parameters used explicitly and implicitly in `testrun_parallel.py` and `compute_comprehensive_clustering_metrics.py`.

## ⚠️ CRITICAL FINDING: Hidden Normalized Laplacian Usage

**Despite `use_normalized_laplacian=False` being set in testrun_parallel.py, the normalized Laplacian IS actually used through the generalized eigenvalue problem!**

### The Parameter Flow Deception

1. **testrun_parallel.py**: Sets `use_normalized_laplacian=False` (line 1083)
2. **GWConfig**: Gets `use_normalized_laplacian=False` (overriding default True)
3. **BUT**: `UnifiedStaticLaplacian` has `use_generalized_normalization=True` by default (line 116)
4. **Result**: Eigenvalues computed using **L x = λ M x** (generalized eigenvalue problem)
5. **Mathematical fact**: L x = λ M x is equivalent to normalized Laplacian without explicit matrix inversion!

### Why This Matters for Reproducibility

The actual computation uses the **numerically superior generalized eigenvalue formulation** of the normalized Laplacian:
- Standard normalized: L_norm = M^(-1/2) L M^(-1/2), solve L_norm y = λ y
- Generalized form: Solve L x = λ M x directly (avoids matrix inversion)
- Same eigenvalues, better numerical stability

This means results are computed using normalized Laplacian mathematics, not standard Laplacian!

## testrun_parallel.py Parameters

### Command Line Arguments
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--models-dir` | `Path('models')` | Directory containing model files |
| `--output-dir` | `Path('eigenvalueData')` | Directory to save eigenvalue data |
| `--workers` | `mp.cpu_count() // 2` | Number of parallel workers |
| `--batch-size` | `1000` | Batch size for model input data |
| `--n-steps` | `30` | Number of filtration steps |
| `--pattern` | `'*'` | Filename pattern to match (e.g., "mlp*", "custom*") |
| `--resume` | `False` | Skip models that already have results |
| `--dry-run` | `False` | Preview without processing |

### ModelProcessor Class Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | `1000` | Default batch size for processing |
| `n_steps` | `30` | Default number of filtration steps |
| `random_seed` | `30` | Random seed for reproducibility |

### Data Generation Parameters
| Data Type | Parameters | Description |
|-----------|------------|-------------|
| MNIST CNN | `transforms.ToTensor()` | 2D images [batch_size, 1, 28, 28] |
| MNIST MLP | `transforms.ToTensor()` + `Lambda(flatten)` | Flattened [batch_size, 784] |
| Digits CNN | `view(batch_size, 1, 8, 8)` | Reshaped from [batch_size, 64] |
| Digits MLP | Direct load | [batch_size, 64] |
| Digits normalization | `/ data.max() * 16.0` | Normalize to [0, 16] range |
| Adult models | `torch.randn(batch_size, 104)` | Random data for 104 features |
| Default random | `10 * torch.randn(batch_size, 3)` | Scaled random data |

### NeurosheafAnalyzer Default Parameters
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `device` | `'cpu'` | Computation device |
| `memory_limit_gb` | `8.0` | Memory limit in GB |
| `enable_profiling` | `True` | Performance profiling |
| `log_level` | `"INFO"` | Logging level |

### NeurosheafAnalyzer.analyze() Method Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `method` | `'gromov_wasserstein'` | Sheaf construction method |
| `use_normalized_laplacian` | `False` | **⚠️ MISLEADING**: Set to False but normalized Laplacian still used via generalized eigenvalue problem |
| `exclude_final_single_output` | `True` | Exclude final single-output layers |
| `gw_config` | `None` | Uses GWConfig defaults |
| `batch_size` | `None` | Auto-detected |
| `layers` | `None` | Analyze all layers |
| `directed` | `False` | Undirected sheaf analysis |
| `directionality_parameter` | `0.25` | Directional strength (unused when directed=False) |
| `preserve_eigenvalues` | `None` | Uses builder default |
| `use_gram_regularization` | `False` | No Tikhonov regularization |
| `regularization_config` | `None` | No regularization config |

### GWConfig Default Parameters (31 parameters)
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `epsilon` | `0.05` | Entropic regularization strength |
| `max_iter` | `1000` | Maximum GW iterations |
| `tolerance` | `1e-9` | Convergence tolerance |
| `quasi_sheaf_tolerance` | `0.1` | ε-sheaf validation threshold |
| `use_gpu` | `True` | GPU acceleration |
| `cache_cost_matrices` | `True` | Cache expensive cost matrices |
| `cache_hash_method` | `'sha1'` | Hash method ('sha1' or 'id') |
| `validate_couplings` | `True` | Validate marginal constraints |
| `validate_costs` | `True` | Validate cost matrix properties |
| `uniform_measures` | `True` | Use uniform vs importance sampling |
| `weighted_inner_product` | `True` | Use p_i-weighted L2 inner products |
| `cost_matrix_eps` | `1e-12` | Numerical threshold for cost matrices |
| `coupling_eps` | `1e-10` | Threshold for coupling validation |
| `measure_eps` | `1e-6` | Floor value for variance-based measures |
| `max_cache_size_gb` | `2.0` | Maximum cache size in GB |
| `adaptive_epsilon` | `False` | Enable adaptive scaling |
| `base_epsilon` | `0.1` | Base epsilon for reference size |
| `reference_n` | `50.0` | Reference sample size |
| `epsilon_scaling_method` | `'sqrt'` | Scaling method |
| `epsilon_min` | `0.01` | Minimum allowed epsilon |
| `epsilon_max` | `0.5` | Maximum allowed epsilon |
| `align_units` | `True` | Align units/neurons vs samples |
| `computation_dtype` | `'float64'` | Primary dtype ('float32' or 'float64') |
| `strict_quality_mode` | `False` | Fail fast when POT unavailable |
| `exclude_fallback_edges` | `True` | Exclude fallback edges from Laplacian |
| `min_coupling_quality` | `0.1` | Minimum quality score threshold [0,1] |
| `validate_restrictions` | `True` | Enable restriction map validation |
| `stochastic_tolerance` | `1e-6` | Tolerance for row-stochasticity check |
| `correction_threshold` | `1e-3` | Auto-correction threshold |
| `strict_validation_threshold` | `0.1` | Threshold for strict mode violations |
| `strict_validation_mode` | `False` | Raise errors vs warnings |
| `auto_correct_restrictions` | `True` | Automatically correct small violations |
| `use_normalized_laplacian` | `True` | **⚠️ OVERRIDDEN**: Default True, but set to False by testrun_parallel.py (doesn't affect actual computation) |

### PersistentSpectralAnalyzer Parameters
| Parameter | Default Value | Override Value | Description |
|-----------|---------------|----------------|-------------|
| `default_n_steps` | `50` | `30` | Number of filtration steps (overridden) |
| `default_filtration_type` | `'threshold'` | `'threshold'` | Filtration type |
| `static_laplacian` | `None` | `None` | Auto-created StaticLaplacianWithMasking (= UnifiedStaticLaplacian) |
| `subspace_tracker` | `None` | `None` | Auto-created SubspaceTracker |
| `dtw_comparator` | `None` | `None` | Optional FiltrationDTW comparator |

### UnifiedStaticLaplacian Parameters (THE ACTUAL EIGENVALUE CONTROLLER)
**This is where the real eigenvalue computation method is determined!**

| Parameter | Default Value | Actual Value | Description |
|-----------|---------------|--------------|-------------|
| `use_generalized_normalization` | `True` | `True` | **🔑 CRITICAL**: Controls generalized eigenvalue problem L x = λ M x |
| `use_double_precision` | `True` | `True` | Use float64 for eigenvalue computations |
| `force_dense_eigenvalues` | `False` | `False` | Force dense eigenvalue computation |
| `use_matrix_free` | `False` | `False` | Use matrix-free LinearOperator for large problems |
| `force_dense_gw_solver` | `False` | `False` | Force dense solver for GW generalized eigenvalue problems |
| `validate_properties` | `True` | `True` | Validate mathematical properties |
| `sparsity_threshold` | `1e-12` | `1e-12` | Threshold for matrix sparsity |
| `eigenvalue_method` | `'auto'` | `'auto'` | Method for eigenvalue computation (auto-selected) |
| `max_eigenvalues` | `None` | `None` | Maximum number of eigenvalues to compute |
| `enable_caching` | `True` | `True` | Enable Laplacian matrix caching |

### GWLaplacianBuilder Parameters (Auto-created by UnifiedStaticLaplacian)
**The actual solver that implements L x = λ M x**

| Parameter | Default Value | Actual Value | Description |
|-----------|---------------|--------------|-------------|
| `use_normalized_laplacian` | `False` | `False` | GWLaplacianBuilder's own setting (irrelevant due to generalized formulation) |
| `validate_properties` | `True` | `True` | Validate mathematical properties |
| `sparsity_threshold` | `1e-12` | `1e-12` | Threshold below which values are zero |
| `use_weighted_inner_products` | `False` | `False` | Use p_i-weighted L2 inner products |
| `enable_caching` | `True` | `True` | Cache Cholesky factorizations and matrices |
| `weight_transform` | `EXPONENTIAL` | `EXPONENTIAL` | Method to convert GW costs to similarities |
| `transform_beta` | `1.0` | `1.0` | Parameter for exponential transform |
| `computation_dtype` | `None` | `'float64'` | Inferred from sheaf metadata |
| `force_dense_solver` | `False` | `False` | Force use of dense eigenvalue solver |

### Eigenvalue Solver Routing Logic
**Based on matrix size, the following solvers are used for L x = λ M x:**

| Matrix Size | Solver Used | Method | Notes |
|-------------|-------------|--------|-------|
| < 1000 | Dense | `scipy.linalg.eigh` | Maximum accuracy for small problems |
| 1000-5000 | Sparse | `scipy.sparse.linalg.eigsh` | With shift-invert for stability |
| > 5000 (matrix-free) | LOBPCG | `scipy.sparse.linalg.lobpcg` | Memory-efficient for large problems |
| Any size (force_dense=True) | Dense fallback | `scipy.linalg.eigh` | Force maximum accuracy |

### spectral_analyzer.analyze() Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `filtration_type` | `'threshold'` | Type of filtration |
| `n_steps` | `30` | Number of steps (from command line) |

## compute_comprehensive_clustering_metrics.py Parameters

### Command Line Arguments
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--distance-file` | Required | Path to distance matrix (.npy file) |
| `--index-file` | Required | Path to file index (.json of filenames) |
| `--output-prefix` | `'comprehensive_clustering'` | Output file prefix |
| `--n-bootstrap` | `1000` | Bootstrap iterations |
| `--n-permutations` | `999` | Permutations for hypothesis tests |

### Global Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `N_BOOTSTRAP_DEFAULT` | `1000` | Default bootstrap iterations |
| `N_PERMUTATIONS_DEFAULT` | `999` | Default permutation test iterations |
| `CONFIDENCE_LEVEL` | `0.95` | Confidence level for intervals |
| `RANDOM_STATE` | `42` | Random seed for MDS reproducibility |

### Label Classification Parameters
| Label Type | Numeric Value | Keywords |
|------------|---------------|-----------|
| Random models | `0` | 'random' |
| Trained models | `1` | 'trained', 'seed', 'acc' |
| Unknown models | `-1` | None of the above |

### AgglomerativeClustering Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_clusters` | `2` | Number of clusters (random vs trained) |
| `metric` | `'precomputed'` | Use precomputed distance matrix |
| `linkage` | `'average'` | Linkage method |

### MDS (Multidimensional Scaling) Parameters
| Parameter | Value/Formula | Description |
|-----------|---------------|-------------|
| `n_components` | `min(10, dm.shape[0] - 1)` if `dm.shape[0] > 1` else `1` | Number of dimensions |
| `dissimilarity` | `'precomputed'` | Use precomputed distance matrix |
| `random_state` | `42` | Random seed for reproducibility |

### Bootstrap and Statistical Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Bootstrap sample size | `N` (same as original) | Resample with replacement |
| Bootstrap replace | `True` | Sampling with replacement |
| Confidence interval | `alpha/2` and `(1-alpha/2)` | Where `alpha = 1 - 0.95` |
| `rankdata method` | `'average'` | Method for handling ties in ANOSIM |

### Internal Metrics Bootstrap Iterations
| Metric | Bootstrap Iterations | Description |
|--------|---------------------|-------------|
| Silhouette Coefficient | `n_bootstrap` | Full bootstrap iterations |
| Davies-Bouldin Index | `min(100, n_bootstrap)` | Limited for computational efficiency |
| Calinski-Harabasz Index | `min(100, n_bootstrap)` | Limited for computational efficiency |
| Dunn Index | `n_bootstrap` | Full bootstrap iterations |
| Cophenetic Correlation | `n_bootstrap` | Full bootstrap iterations |
| Clustering Stability | `min(100, n_bootstrap)` | Limited for computational efficiency |

### Hypothesis Testing Parameters
| Test | Parameter | Value | Description |
|------|-----------|-------|-------------|
| PERMANOVA | `n_permutations` | `999` | Permutation test iterations |
| ANOSIM | `n_permutations` | `999` | Permutation test iterations |

### Distance Matrix Preprocessing
| Operation | Value | Description |
|-----------|-------|-------------|
| Symmetrization | `0.5 * (D + D.T)` | Ensure symmetry |
| Diagonal | `fill_diagonal(0.0)` | Set self-distances to zero |
| Data type | `float64` | Convert to float64 |

### Hungarian Algorithm (Linear Assignment)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `confusion_matrix labels` | `[0, 1]` | Binary classification labels |
| Cost matrix | `-cm` | Negative confusion matrix for maximization |

### Numerical Validation Thresholds
| Check | Threshold | Description |
|-------|-----------|-------------|
| Finite values | `np.isfinite()` | Check for NaN/Inf values |
| Minimum groups | `len(np.unique(y)) >= 2` | At least 2 groups required |
| Minimum samples | Various (e.g., `D.shape[0] >= 4`) | Minimum sample requirements |

## Critical Parameter Interactions Summary

### The Three "use_normalized_laplacian" Parameters (AVOID CONFUSION!)

| Location | Parameter | Value | Actual Effect |
|----------|-----------|-------|---------------|
| testrun_parallel.py | `use_normalized_laplacian` | `False` | Passed to NeurosheafAnalyzer, overrides GWConfig default |
| GWConfig | `use_normalized_laplacian` | `False` | Overridden from default True, stored in sheaf metadata |
| GWLaplacianBuilder | `use_normalized_laplacian` | `False` | **IRRELEVANT**: Generalized eigenvalue problem used instead |

**THE CONTROLLING PARAMETER**: `UnifiedStaticLaplacian.use_generalized_normalization = True` (default)

### Actual Computational Method Used

**Eigenvalue Problem Solved**: L x = λ M x (generalized eigenvalue problem)
**Mathematical Equivalent**: Normalized Laplacian without explicit matrix inversion
**Numerical Benefit**: Avoids inverting mass matrix M, better stability
**Solver Selection**: Based on matrix size (dense/sparse/LOBPCG)

### Critical Parameters for Reproducibility

| Parameter | Source | Value | Impact |
|-----------|--------|-------|--------|
| `random_seed` | ModelProcessor | `30` | Controls all random number generation |
| `use_generalized_normalization` | UnifiedStaticLaplacian | `True` | **Controls actual eigenvalue method** |
| `computation_dtype` | GWConfig → GWLaplacianBuilder | `'float64'` | Numerical precision |
| `n_steps` | Command line | `30` | Filtration resolution |
| `exclude_final_single_output` | testrun_parallel.py | `True` | Sheaf structure |
| `use_double_precision` | UnifiedStaticLaplacian | `True` | Eigenvalue computation precision |

## Summary Statistics

### Total Parameter Count
- **testrun_parallel.py**: ~50 parameters
- **UnifiedStaticLaplacian**: 10 critical parameters
- **GWLaplacianBuilder**: 9 solver parameters
- **compute_comprehensive_clustering_metrics.py**: ~25 parameters
- **Total unique parameters**: ~95 parameters

### Parameter Categories
1. **Command-line arguments**: 11 parameters
2. **Neural network configuration**: 31 GWConfig parameters
3. **Eigenvalue computation control**: 10 UnifiedStaticLaplacian parameters
4. **Generalized eigenvalue solver**: 9 GWLaplacianBuilder parameters
5. **Data processing**: 8 parameters
6. **Statistical analysis**: 15 parameters
7. **Machine learning algorithms**: 6 parameters
8. **Numerical thresholds**: 5+ parameters

### The Bottom Line for Reproducibility

Despite the confusing parameter names, the actual computation in testrun_parallel.py uses:
- **Generalized eigenvalue problem**: L x = λ M x
- **Mathematically equivalent**: Normalized Laplacian (without matrix inversion)
- **Numerically superior**: Avoids explicit computation of M^(-1/2)
- **Solver routing**: Automatic based on matrix size
- **Precision**: float64 throughout

This is the **numerically optimal** way to compute normalized Laplacian eigenvalues!
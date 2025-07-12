# Neurosheaf Package Documentation

Welcome to the Neurosheaf package documentation. Neurosheaf provides a mathematically principled framework for analyzing neural network similarity using persistent sheaf Laplacians with optimized sparse matrix operations.

## Table of Contents

*   [Overview & Architecture](#overview--architecture)
*   [Pipeline Guide](#pipeline-guide)
*   [Best Practices](#best-practices)
*   [Main API (`neurosheaf.api`)](#main-api-neurosheafapi)
*   [CKA Module (`neurosheaf.cka`)](#cka-module-neurosheafcka)
*   [Sheaf Module (`neurosheaf.sheaf`)](#sheaf-module-neurosheafsheaf)
*   [Spectral Module (`neurosheaf.spectral`)](#spectral-module-neurosheafspectral)
*   [Utilities (`neurosheaf.utils`)](#utilities-neurosheafutils)
*   [Visualization (`neurosheaf.visualization`)](#visualization-neurosheafvisualization)
*   [Performance & Optimization](#performance--optimization)

---

## Overview & Architecture

### What is Neurosheaf?

Neurosheaf is a Python framework that enables rigorous mathematical analysis of neural network similarity patterns using **cellular sheaves** and **persistent topology**. The framework transforms neural networks into sheaf data structures and analyzes their spectral properties through optimized Laplacian construction.

### Core Mathematical Concepts

1. **Centered Kernel Alignment (CKA)**: Measures similarity between neural representations using debiased HSIC estimators
2. **Cellular Sheaves**: Mathematical structures that encode local data (stalks) and global relationships (restriction maps)
3. **Whitened Coordinates**: Optimal coordinate system ensuring exact metric compatibility and orthogonality
4. **Sheaf Laplacians**: Sparse matrices encoding the global topology of neural similarity patterns
5. **Persistent Spectral Analysis**: Study of how spectral properties change across filtrations

### Pipeline Architecture

```
Neural Network → Activations → CKA/Gram Matrices → Sheaf Construction → Laplacian → Spectral Analysis
      ↓              ↓              ↓                   ↓              ↓           ↓
   ResNet-18    [128,64,224,224]  [128,128]         Whitened      [3625,3625]  Eigenvalues
                                                   Coordinates    99.9% sparse
```

---

## Pipeline Guide

### Quick Start Example

```python
import torch
import torchvision.models as models
from neurosheaf.sheaf import SheafBuilder, SheafLaplacianBuilder
from neurosheaf.sheaf.enhanced_extraction import EnhancedActivationExtractor

# 1. Setup model and data
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()
input_batch = torch.randn(128, 3, 224, 224)

# 2. Extract activations
extractor = EnhancedActivationExtractor(capture_functional=True)
with torch.no_grad():
    activations = extractor.extract_comprehensive_activations(model, input_batch)

# 3. Build sheaf with whitened coordinates
builder = SheafBuilder(
    handle_dynamic=True,
    use_whitening=True,          # Enable whitened coordinates (recommended)
    residual_threshold=0.05,     # 5% filtering threshold
)
sheaf = builder.build_from_activations(model, activations, validate=True)

# 4. Construct optimized Laplacian
laplacian_builder = SheafLaplacianBuilder(
    assembly_method='preallocated',  # Optimized method (default)
    validate_properties=False       # Disable for production speed
)
L, metadata = laplacian_builder.build_laplacian(sheaf)

# 5. Analyze results
print(f"Sheaf: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
print(f"Laplacian: {L.shape}, {L.nnz:,} non-zeros ({1-L.nnz/(L.shape[0]*L.shape[1]):.1%} sparse)")
print(f"Construction time: {metadata.construction_time:.3f}s")
```

### Detailed Pipeline Steps

#### Step 1: Activation Extraction

```python
# Standard module extraction
extractor = EnhancedActivationExtractor(capture_functional=True)
activations = extractor.extract_comprehensive_activations(model, input_tensor)

# Result: Dictionary mapping layer names to activation tensors
# Format: {'conv1': tensor([128, 64]), 'layer1.0.conv1': tensor([128, 64]), ...}
```

**Key Points**:
- Captures both module-based and functional operations (ReLU, pooling)
- Automatically handles 4D tensors by averaging spatial dimensions
- Raw activations (no pre-centering) are required for mathematical correctness

#### Step 2: Sheaf Construction

```python
# Configure sheaf builder
builder = SheafBuilder(
    handle_dynamic=True,           # Handle non-traceable models
    use_whitening=True,           # CRITICAL: Enables whitened coordinates
    residual_threshold=0.05,      # Filter low-quality restriction maps
    restriction_method='scaled_procrustes'
)

# Build sheaf
sheaf = builder.build_from_activations(
    model, 
    activations, 
    use_gram_matrices=True,       # Use Gram matrices as stalks
    validate=True                 # Validate mathematical properties
)
```

**Whitened Coordinates (Critical)**:
- Transforms Gram matrices K → Identity matrices
- Ensures exact orthogonality of restriction maps
- Provides optimal numerical conditioning
- Required for mathematically valid Laplacian construction

#### Step 3: Optimized Laplacian Assembly

```python
# Choose assembly method based on your needs
laplacian_builder = SheafLaplacianBuilder(
    assembly_method='preallocated',  # Options: 'preallocated', 'block_wise', 'current'
    validate_properties=False,      # Set True for development, False for production
    enable_gpu=False,               # CPU optimizations are sufficient
    memory_efficient=True
)

L, metadata = laplacian_builder.build_laplacian(sheaf)
```

**Assembly Methods**:
- `'preallocated'`: **64x faster** - Pre-allocated COO arrays with vectorized operations
- `'block_wise'`: **17x faster** - Uses scipy.sparse.bmat for block assembly  
- `'current'`: Original implementation for compatibility

#### Step 4: Spectral Analysis

```python
from scipy.sparse.linalg import eigsh

# Compute smallest eigenvalues (harmonic components)
eigenvalues = eigsh(L, k=10, which='SM', return_eigenvectors=False)
harmonic_dimension = np.sum(eigenvalues < 1e-6)

print(f"Harmonic dimension: {harmonic_dimension}")
print(f"Smallest eigenvalues: {eigenvalues[:5]}")
```

### Performance Characteristics

| Phase | Time | Memory | Bottleneck |
|-------|------|--------|------------|
| **Activation Extraction** | ~5s | 1.8GB | Model forward pass |
| **Sheaf Construction** | ~0.2s | 0.01GB | Restriction map computation |
| **Laplacian Assembly** | **0.015s** | **0.01GB** | **Eliminated** |
| **Eigenvalue Validation** | ~50s | 0.001GB | ARPACK convergence |

**Total Pipeline**: 5.2s (excluding validation)

---

## Best Practices

### Mathematical Correctness

#### ✅ DO: Use Raw Activations
```python
# CORRECT: Raw activations for CKA/Gram matrices
activations = extract_activations(model, input_data)  
K = activations @ activations.T  # Raw Gram matrix
```

#### ❌ DON'T: Pre-center Activations
```python
# WRONG: Pre-centering causes double-centering bias
centered = activations - activations.mean(dim=0)  # NEVER DO THIS
K = centered @ centered.T  # Mathematically incorrect
```

#### ✅ DO: Enable Whitened Coordinates
```python
# CORRECT: Always use whitening for production
builder = SheafBuilder(use_whitening=True)  # Mathematical optimality
```

#### ❌ DON'T: Disable Whitening
```python
# SUBOPTIMAL: Raw coordinates have poor conditioning
builder = SheafBuilder(use_whitening=False)  # Avoid unless testing
```

### Performance Optimization

#### ✅ DO: Use Optimized Assembly
```python
# CORRECT: Default optimized method
laplacian_builder = SheafLaplacianBuilder()  # Uses 'preallocated' by default

# Or explicit optimization
laplacian_builder = SheafLaplacianBuilder(assembly_method='preallocated')
```

#### ✅ DO: Disable Validation in Production
```python
# CORRECT: Fast production assembly
laplacian_builder = SheafLaplacianBuilder(validate_properties=False)
```

#### ❌ DON'T: Use Validation in Performance-Critical Code
```python
# SLOW: Eigenvalue computation adds 50+ seconds
laplacian_builder = SheafLaplacianBuilder(validate_properties=True)  # Development only
```

### Memory Management

#### ✅ DO: Monitor Batch Sizes
```python
# CORRECT: Reasonable batch sizes for analysis
input_batch = torch.randn(128, 3, 224, 224)  # Good for most GPUs
```

#### ✅ DO: Use CPU for Large Laplacians
```python
# CORRECT: CPU optimization is sufficient and memory-efficient
laplacian_builder = SheafLaplacianBuilder(enable_gpu=False)
```

### Error Handling

#### ✅ DO: Validate Sheaf Properties
```python
# CORRECT: Always validate during development
sheaf = builder.build_from_activations(model, activations, validate=True)

# Check validation results
if sheaf.metadata.get('validation_passed', False):
    print("✓ Sheaf validation passed")
else:
    print("⚠ Sheaf validation failed - check restriction maps")
```

#### ✅ DO: Handle FX Tracing Failures
```python
# CORRECT: Enable dynamic model handling
builder = SheafBuilder(handle_dynamic=True)  # Fallback for complex models
```

### Troubleshooting Common Issues

#### Issue: "Dimension mismatch in restriction maps"
**Solution**: Check activation extraction and ensure consistent tensor shapes

#### Issue: "ARPACK convergence failure"
**Solution**: Disable validation for production: `validate_properties=False`

#### Issue: "High memory usage"
**Solution**: Reduce batch size or use Nyström approximation for CKA

#### Issue: "FX tracing failed"
**Solution**: Ensure `handle_dynamic=True` and check model compatibility

---

## Main API (`neurosheaf.api`)

The main API provides the high-level interface for using the Neurosheaf package.

### `NeurosheafAnalyzer`

The `NeurosheafAnalyzer` class is the primary entry point for performing neural network similarity analysis using the Neurosheaf framework. It handles device management, memory considerations, and orchestrates the overall analysis pipeline.

**Constructor: `NeurosheafAnalyzer(device=None, memory_limit_gb=8.0, enable_profiling=True, log_level="INFO")`**

*   **Purpose:** Initializes the analyzer.
*   **How it works:**
    *   Sets up logging.
    *   Detects the optimal computation device (`cpu`, `cuda`, `mps`) if not specified. It prioritizes MPS on Apple Silicon Macs, then CUDA if available, otherwise CPU.
    *   Stores memory limits and profiling preferences.
*   **Parameters:**
    *   `device` (Optional\[str]): The device to use for computations (e.g., 'cpu', 'cuda', 'mps'). If `None`, it's auto-detected.
    *   `memory_limit_gb` (float): The memory limit in gigabytes for computations. Default: `8.0`.
    *   `enable_profiling` (bool): If `True`, enables performance profiling features. Default: `True`.
    *   `log_level` (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING'). Default: `"INFO"`.

**Method: `analyze(model, data, batch_size=None, layers=None)`**

*   **Purpose:** Performs the complete Neurosheaf analysis on a given model and data. (Note: Currently, this method is a placeholder in Phase 1 Week 2 and will be fully implemented later).
*   **How it works (intended):**
    *   Validates the input model and data.
    *   Moves the model and data to the selected device.
    *   Orchestrates the extraction of activations, CKA computation, sheaf construction, and spectral analysis.
*   **Parameters:**
    *   `model` (torch.nn.Module): The PyTorch neural network model to analyze.
    *   `data` (torch.Tensor): The input data tensor for the model.
    *   `batch_size` (Optional\[int]): Batch size for processing data. If `None`, it might be auto-detected or use a default.
    *   `layers` (Optional\[List\[str]]): A list of specific layer names to analyze. If `None`, all relevant layers are analyzed.
*   **Returns:** (dict) A dictionary containing the analysis results. The exact structure will be defined as implementation progresses, but it's expected to include CKA matrices, sheaf structures, spectral data, etc. Currently returns placeholder information.

**Method: `get_system_info()`**

*   **Purpose:** Retrieves comprehensive information about the system, device, and analyzer configuration.
*   **How it works:** Collects details about the detected device, platform, Python/PyTorch versions, CUDA/MPS availability, memory configuration, and analyzer settings.
*   **Returns:** (dict) A dictionary containing system and hardware information.

**Method: `profile_memory_usage(model, data)`**

*   **Purpose:** Profiles the memory usage of a given model with specific data.
*   **How it works:**
    *   Executes a forward pass of the model with the data.
    *   Uses the profiling utilities (see `neurosheaf.utils.profiling`) to measure memory consumption during the operation.
    *   This method requires `enable_profiling` to be `True` during analyzer initialization.
*   **Parameters:**
    *   `model` (torch.nn.Module): The PyTorch model to profile.
    *   `data` (torch.Tensor): The input data for profiling.
*   **Returns:** (dict) A dictionary containing memory profiling results, including memory usage before/after and potentially peak usage.

**Internal Helper Methods (Conceptual Overview):**

*   `_detect_device(device)`: Implements the logic for selecting the optimal computation device.
*   `_validate_inputs(model, data)`: Performs basic validation checks on the model and data.
*   `_get_device_info()`: Gathers detailed information about the current compute device and related software (CUDA, MPS).
*   `_get_memory_info()`: Retrieves current system and device-specific memory usage.

---

## CKA Module (`neurosheaf.cka`)

This module contains implementations for Centered Kernel Alignment (CKA), including debiased versions, memory-efficient approximations, and sampling strategies. It's crucial to use **raw (uncentered) activations** with these CKA methods, as the unbiased HSIC estimator handles centering internally to avoid double-centering bias.

### `DebiasedCKA`

Computes CKA without the double-centering bias. This is the recommended method for accurate similarity measurements.

**Constructor: `DebiasedCKA(device=None, use_unbiased=True, enable_profiling=True, numerical_stability=1e-8, regularization=0.0, auto_promote_precision=True, safe_tensor_ops=True, strict_numerics=False, adaptive_epsilon=True, enable_gradients=False)`**

*   **Purpose:** Initializes the debiased CKA calculator.
*   **How it works:** Sets up the CKA computation environment, including device selection, choice of HSIC estimator (biased/unbiased), and parameters for numerical stability (e.g., epsilon, regularization, precision promotion).
*   **Parameters:**
    *   `device` (Optional\[Union\[str, torch.device]]): Computation device. Auto-detected if `None`.
    *   `use_unbiased` (bool): If `True` (default), uses the unbiased HSIC estimator, which is mathematically preferred.
    *   `enable_profiling` (bool): Enables performance profiling. Default: `True`.
    *   `numerical_stability` (float): Epsilon value for numerical stability in divisions. Default: `1e-8`.
    *   `regularization` (float): Regularization parameter for ill-conditioned kernel matrices. Default: `0.0` (no regularization).
    *   `auto_promote_precision` (bool): If `True`, automatically promotes tensors to `float64` for potentially improved numerical stability, especially for small sample sizes or on MPS devices. Default: `True`.
    *   `safe_tensor_ops` (bool): If `True`, clones tensors to avoid in-place modification risks. Default: `True`.
    *   `strict_numerics` (bool): If `True`, raises an error on detecting NaN/Inf values or high condition numbers. If `False`, logs a warning. Default: `False`.
    *   `adaptive_epsilon` (bool): If `True`, adapts the epsilon for HSIC computation based on kernel matrix condition numbers. Default: `True`.
    *   `enable_gradients` (bool): If `True`, allows gradients to be computed through the CKA calculation (experimental). Default: `False`.

**Method: `compute_cka(X, Y, validate_properties=True)` (alias: `compute`)**

*   **Purpose:** Computes the debiased CKA similarity score between two activation matrices `X` and `Y`.
*   **How it works:**
    1.  Validates input activations (must be raw, not pre-centered).
    2.  Optionally promotes precision (e.g., to `float64`) for stability.
    3.  Computes Gram matrices `K = X @ X.T` and `L = Y @ Y.T` from the raw activations.
    4.  Applies regularization to `K` and `L` if specified.
    5.  Computes CKA using either the unbiased HSIC estimator (default, recommended) or the biased one. The unbiased estimator handles centering internally.
    6.  Clamps the result to \[0, 1].
*   **Parameters:**
    *   `X` (torch.Tensor): The first activation matrix (samples x features). **Must be raw, uncentered activations.**
    *   `Y` (torch.Tensor): The second activation matrix (samples x features). **Must be raw, uncentered activations.**
    *   `validate_properties` (bool): If `True`, validates mathematical properties like CKA(X,X) ≈ 1. Default: `True`.
*   **Returns:** (float) The CKA similarity value, ranging from 0 to 1.

**Method: `compute_cka_matrix(activations, symmetric=True, warn_preprocessing=True)`**

*   **Purpose:** Computes the pairwise CKA similarity matrix for a dictionary of activation sets.
*   **How it works:** Iterates through all pairs of layers provided in the `activations` dictionary, computing the CKA similarity between each pair using `compute_cka`.
*   **Parameters:**
    *   `activations` (Dict\[str, torch.Tensor]): A dictionary mapping layer names to their raw activation tensors.
    *   `symmetric` (bool): If `True` (default), assumes CKA(X,Y) = CKA(Y,X) and only computes the upper triangle of the matrix.
    *   `warn_preprocessing` (bool): If `True` (default), issues a warning if activations appear to be pre-centered.
*   **Returns:** (torch.Tensor) A 2D tensor representing the CKA similarity matrix.

---

### `BaselineCKA`

A memory-intensive CKA implementation designed for establishing performance baselines and identifying optimization opportunities. **Not recommended for general use.**

**Constructor: `BaselineCKA(device=None, store_intermediates=True, enable_detailed_profiling=True)`**

*   **Purpose:** Initializes a CKA calculator that is intentionally memory-heavy for benchmarking.
*   **Parameters:**
    *   `device` (Optional\[Union\[str, torch.device]]): Computation device.
    *   `store_intermediates` (bool): If `True`, stores all intermediate matrices. Default: `True`.
    *   `enable_detailed_profiling` (bool): Enables detailed profiling. Default: `True`.

**Method: `compute_baseline_cka_matrix(activations, target_memory_gb=20.0)`**

*   **Purpose:** Computes the CKA matrix using a memory-intensive approach.
*   **How it works:** Moves all activations to the device at once, pre-computes all Gram matrices, and stores intermediates, aiming to consume significant memory.
*   **Parameters:**
    *   `activations` (Dict\[str, torch.Tensor]): Dictionary of activation tensors.
    *   `target_memory_gb` (float): A target for memory usage, though not strictly enforced. Default: `20.0`.
*   **Returns:** (Tuple\[torch.Tensor, Dict\[str, any]]) The CKA matrix and profiling data.

**Method: `profile_resnet50_baseline(batch_size=1000, num_layers=50, feature_dims=None)`**

*   **Purpose:** Profiles a ResNet50-like model to establish a high memory usage baseline (e.g., aiming for 1.5TB if resources allowed, though practically limited).
*   **Returns:** (Dict\[str, any]) Profiling results.

**Method: `generate_baseline_report()`**

*   **Purpose:** Generates a human-readable report from the profiling data collected.
*   **Returns:** (str) The formatted baseline report.

---

### `NystromCKA`

Provides a memory-efficient CKA computation using Nyström approximation, reducing memory complexity from O(n²) for exact CKA to O(n\*m), where n is samples and m is landmarks.

**Constructor: `NystromCKA(n_landmarks=256, landmark_selection='uniform', device=None, numerical_stability=1e-6, ..., use_qr_approximation=False, enable_psd_projection=True, adaptive_landmarks=True)`**

*   **Purpose:** Initializes the Nyström CKA approximator.
*   **How it works:** Configures the Nyström approximation method, including the number of landmarks, selection strategy, and parameters for numerical stability and approximation quality.
*   **Parameters:**
    *   `n_landmarks` (int): Number of landmark points for the approximation. Default: `256`.
    *   `landmark_selection` (str): Strategy for selecting landmarks ('uniform', 'kmeans', or 'spectral' via `use_spectral_landmarks`). Default: `'uniform'`.
    *   `device` (Optional\[Union\[str, torch.device]]): Computation device.
    *   `numerical_stability` (float): Epsilon for numerical stability, e.g., in matrix inversions. Default: `1e-6`.
    *   `use_qr_approximation` (bool): If `True`, uses a more stable QR-based Nyström approximation. Default: `False` (uses standard Nyström with SVD-based stable inverse).
    *   `enable_psd_projection` (bool): If `True`, projects the approximated kernel matrix to the positive semidefinite cone to ensure validity. Default: `True`.
    *   `adaptive_landmarks` (bool): If `True`, adapts the number of landmarks based on the estimated effective rank of the Gram matrix, potentially reducing `n_landmarks` if the data has lower intrinsic dimensionality. Default: `True`.
    *   `use_spectral_landmarks` (bool): If `True`, uses leverage score-based landmark selection for potentially better approximation quality. Default: `False`.
    *   Other parameters control aspects like k-means iterations, precision promotion, and regularization.

**Method: `compute(X, Y, validate_properties=True, return_approximation_info=False)`**

*   **Purpose:** Computes CKA between two activation matrices `X` and `Y` using Nyström approximation.
*   **How it works:**
    1.  Selects landmark points from `X` and `Y` based on the chosen strategy and `n_landmarks` (possibly adapted).
    2.  Computes Nyström approximations of the Gram matrices `K_approx` (from `X`) and `L_approx` (from `Y`). This involves sub-sampling the Gram matrix calculation using landmarks.
    3.  The approximated kernels `K_approx` and `L_approx` are then used to compute the CKA value via the (unbiased) HSIC estimator.
    4.  Includes options for QR-based approximation and PSD projection for improved stability and correctness of the approximated kernels.
*   **Parameters:**
    *   `X` (torch.Tensor): First raw activation matrix.
    *   `Y` (torch.Tensor): Second raw activation matrix.
    *   `validate_properties` (bool): If `True`, validates CKA properties. Default: `True`.
    *   `return_approximation_info` (bool): If `True`, returns additional information about the approximation quality. Default: `False`.
*   **Returns:** (float or Tuple\[float, Dict]) The CKA value. If `return_approximation_info` is `True`, also returns a dictionary with approximation metrics.

**Method: `estimate_memory_usage(n_samples, n_features)`**

*   **Purpose:** Estimates memory usage for Nyström CKA versus exact CKA for a given problem size.
*   **Returns:** (Dict\[str, float]) A dictionary with memory estimates in MB and the reduction factor.

**Method: `recommend_landmarks(n_samples, target_error=0.01)`**

*   **Purpose:** Suggests a number of landmarks based on the number of samples and a target approximation error.
*   **Returns:** (int) The recommended number of landmarks.

---

### `PairwiseCKA`

Efficiently computes a full pairwise CKA matrix between multiple layers, with features like memory monitoring, checkpointing, and automatic selection between exact and Nyström CKA.

**Constructor: `PairwiseCKA(cka_computer=None, memory_limit_mb=1024, checkpoint_dir=None, checkpoint_frequency=10, use_nystrom=False, nystrom_landmarks=256)`**

*   **Purpose:** Initializes the pairwise CKA calculator.
*   **How it works:** Sets up the environment for computing a matrix of CKA scores. It can use an existing `DebiasedCKA` instance or create one. It also configures memory limits and checkpointing behavior.
*   **Parameters:**
    *   `cka_computer` (Optional\[DebiasedCKA]): An instance of `DebiasedCKA` to use for individual CKA computations. If `None`, a default one is created.
    *   `memory_limit_mb` (float): Memory limit in MB that influences decisions like using Nyström or sampling. Default: `1024`.
    *   `checkpoint_dir` (Optional\[str]): Directory to save checkpoints. If `None`, checkpointing is disabled.
    *   `checkpoint_frequency` (int): Save a checkpoint every N computed pairs. Default: `10`.
    *   `use_nystrom` (bool): If `True`, defaults to using Nyström approximation for CKA pairs. Default: `False`.
    *   `nystrom_landmarks` (int): Number of landmarks if Nyström is used. Default: `256`.

**Method: `compute_matrix(activations, layer_names=None, progress_callback=None, adaptive_sampling=False, sampler=None, use_nystrom=None)`**

*   **Purpose:** Computes the full CKA matrix between the specified layers.
*   **How it works:**
    1.  Iterates through pairs of layers.
    2.  For each pair, it checks memory availability.
    3.  Optionally applies adaptive sampling if `adaptive_sampling` is `True` and memory is constrained.
    4.  Decides whether to use exact CKA (via `cka_computer`) or Nyström CKA (if `use_nystrom` is true or auto-configured).
    5.  Computes the CKA value for the pair.
    6.  Saves checkpoints periodically if `checkpoint_dir` is set.
    7.  Provides progress updates via `progress_callback` and `tqdm`.
*   **Parameters:**
    *   `activations` (Dict\[str, torch.Tensor]): Dictionary of raw activation tensors.
    *   `layer_names` (Optional\[List\[str]]): Subset of layer names from `activations` to compute the matrix for. If `None`, uses all layers.
    *   `progress_callback` (Optional\[Callable]): A function called with `(current_pair, total_pairs)` for progress updates.
    *   `adaptive_sampling` (bool): If `True`, enables adaptive sampling for large activations. Default: `False`.
    *   `sampler` (Optional\[AdaptiveSampler]): An `AdaptiveSampler` instance. If `None` and `adaptive_sampling` is `True`, a default one is created.
    *   `use_nystrom` (Optional\[bool]): Overrides the instance's default Nyström setting for this specific computation.
*   **Returns:** (torch.Tensor) The computed pairwise CKA matrix.

**Method: `auto_configure(activations)`**

*   **Purpose:** Automatically sets the computation method (exact CKA or Nyström CKA) based on the size of the activation data and the configured `memory_limit_mb`.
*   **How it works:** Estimates memory requirements for exact and Nyström CKA. If exact CKA fits within 80% of `memory_limit_mb`, it's chosen. Otherwise, if Nyström CKA fits, it's chosen. If neither fits comfortably, it will still likely use Nyström, potentially in conjunction with adaptive sampling if enabled in `compute_matrix`.

**Method: `get_memory_usage_estimate(activations)`**

*   **Purpose:** Provides memory usage estimates for different CKA computation methods (exact, Nyström, sampling) given the activations.
*   **Returns:** (Dict\[str, float]) A dictionary with memory estimates in MB.

---

### `AdaptiveSampler`

Provides strategies to sample large activation tensors, aiming to reduce memory and computational load while preserving representative information for CKA.

**Constructor: `AdaptiveSampler(min_samples=512, max_samples=4096, target_variance=0.01, random_seed=None)`**

*   **Purpose:** Initializes the sampler with constraints and targets.
*   **Parameters:**
    *   `min_samples` (int): Minimum number of samples to select. Default: `512`. (Note: Must be at least 4 for unbiased HSIC).
    *   `max_samples` (int): Maximum number of samples to select. Default: `4096`.
    *   `target_variance` (float): Target variance for more advanced (future) adaptive sampling methods. Not heavily used by current methods. Default: `0.01`.
    *   `random_seed` (Optional\[int]): Seed for reproducibility of random sampling.

**Method: `determine_sample_size(n_total, n_features, available_memory_mb, dtype=torch.float32)`**

*   **Purpose:** Calculates an optimal sample size that respects memory constraints while trying to use as many samples as possible up to `max_samples`.
*   **How it works:**
    *   Estimates the memory required for computing two kernel matrices of size `n x n`.
    *   If using all `n_total` samples (up to `max_samples`) fits within `available_memory_mb` (with a safety margin), that size is chosen.
    *   Otherwise, it performs a binary search to find the largest sample size (`n` between `min_samples` and `max_samples`) whose kernel matrix computations would fit in memory.
*   **Parameters:**
    *   `n_total` (int): Total number of available samples.
    *   `n_features` (int): Number of features (used for more advanced future heuristics, not directly in current memory calculation).
    *   `available_memory_mb` (float): Available memory in megabytes.
    *   `dtype` (torch.dtype): Data type of tensors, used for memory estimation. Default: `torch.float32`.
*   **Returns:** (int) The determined sample size.

**Method: `stratified_sample(n_total, n_samples, labels=None, return_mask=False)`**

*   **Purpose:** Selects `n_samples` from `n_total`. If `labels` are provided, performs stratified sampling to maintain class proportions. Otherwise, performs uniform random sampling.
*   **Returns:** (torch.Tensor or Tuple) Sample indices. If `return_mask` is `True`, also returns a boolean mask.

**Method: `progressive_sampling(n_total, batch_sizes, overlap=0.1)`**

*   **Purpose:** Generates batches of sample indices, potentially with overlap, for scenarios where CKA needs to be computed in multiple passes over subsets of data (e.g., extremely large datasets).
*   **Returns:** (List\[torch.Tensor]) A list of tensors, where each tensor contains sample indices for a batch.

**Method: `estimate_required_samples(target_error=0.01, confidence=0.95)`**

*   **Purpose:** Provides a heuristic estimate of the number of samples needed to achieve a target CKA estimation error with a given confidence level.
*   **Returns:** (int) The estimated number of samples.

---

## Sheaf Module (`neurosheaf.sheaf`)

This module is responsible for constructing and manipulating sheaf data structures. This involves extracting the network's computational graph (poset), defining data (stalks) on the graph nodes (layers), computing relationships (restriction maps) between connected layers, and assembling these components into a sheaf. The sheaf can then be used to build a sheaf Laplacian for spectral analysis.

### `FXPosetExtractor`

Extracts a Partially Ordered Set (poset) representing the computational graph of a PyTorch model using `torch.fx` symbolic tracing.

**Constructor: `FXPosetExtractor(handle_dynamic=True)`**

*   **Purpose:** Initializes the poset extractor.
*   **Parameters:**
    *   `handle_dynamic` (bool): If `True` (default), attempts a fallback extraction method (module inspection) if `torch.fx` tracing fails (e.g., for models with dynamic control flow).

**Method: `extract_poset(model)`**

*   **Purpose:** Extracts the poset from the given PyTorch model.
*   **How it works:**
    1.  Attempts to symbolically trace the `model` using `torch.fx`.
    2.  If tracing is successful, it builds a `networkx.DiGraph` where nodes represent layers/operations and edges represent data flow.
    3.  If tracing fails and `handle_dynamic` is `True`, it falls back to a simpler extraction method based on inspecting `model.named_modules()`.
*   **Parameters:**
    *   `model` (torch.nn.Module): The PyTorch model.
*   **Returns:** (networkx.DiGraph) The extracted poset, where nodes have attributes like `name`, `op`, `target`, and `level` (topological depth).

**Method: `extract_activation_filtered_poset(model, available_activations)`**

*   **Purpose:** Extracts a poset that only includes nodes for which activations are available, intelligently bridging gaps where intermediate operations might not have recorded activations.
*   **How it works:**
    1.  Traces the model using `torch.fx`.
    2.  Identifies FX nodes corresponding to the keys in `available_activations`.
    3.  Constructs a poset containing only these "active" nodes.
    4.  If a path exists in the full FX graph between two active nodes via one or more "inactive" nodes, an edge is added directly between the active nodes in the filtered poset.
*   **Parameters:**
    *   `model` (torch.nn.Module): The PyTorch model.
    *   `available_activations` (Set\[str]): A set of activation keys (typically FX node names) that are present.
*   **Returns:** (networkx.DiGraph) The filtered poset.

---

### `WhiteningProcessor`

Implements whitening transformations for Gram matrices, which is crucial for achieving exact metric compatibility and orthogonality for restriction maps in a "whitened" coordinate space. This is referred to as Patch P1 in some internal documentation.

**Constructor: `WhiteningProcessor(min_eigenvalue=1e-8, regularization=1e-10)`**

*   **Purpose:** Initializes the whitening processor.
*   **Parameters:**
    *   `min_eigenvalue` (float): Smallest eigenvalue to consider significant during spectral factorization of Gram matrices. Helps in determining effective rank. Default: `1e-8`.
    *   `regularization` (float): Small value added to eigenvalues during the computation of `Σ^(-1/2)` to prevent division by zero for very small eigenvalues. Default: `1e-10`.

**Method: `compute_whitening_map(K)`**

*   **Purpose:** Computes the whitening map `W = Σ^(-1/2) U^T` from a Gram matrix `K = U Σ U^T`.
*   **How it works:**
    1.  Performs eigendecomposition of the Gram matrix `K`.
    2.  Filters eigenvalues below `min_eigenvalue` to determine the effective rank `r`.
    3.  Constructs the whitening map `W` of shape `(r, n)` using the significant eigenvalues and corresponding eigenvectors.
*   **Parameters:**
    *   `K` (torch.Tensor): The Gram matrix (n x n).
*   **Returns:** (Tuple\[torch.Tensor, Dict])
    *   `W`: The whitening map (r x n tensor).
    *   `info`: A dictionary containing metadata like `effective_rank`, `condition_number`, `eigenvalue_range`.

**Method: `whiten_gram_matrix(K)`**

*   **Purpose:** Transforms a Gram matrix `K` into whitened coordinates, resulting in an identity matrix.
*   **How it works:** Computes the whitening map `W` for `K`. The whitened Gram matrix is then conceptually `W K W^T`, which should be an identity matrix of size `r x r` (where `r` is effective rank).
*   **Parameters:**
    *   `K` (torch.Tensor): The original Gram matrix.
*   **Returns:** (Tuple\[torch.Tensor, torch.Tensor, Dict])
    *   `K_whitened`: The identity matrix in the whitened space (r x r).
    *   `W`: The whitening map.
    *   `info`: Whitening metadata, including `whitening_error` (Frobenius norm of `W K W^T - I`).

**Method: `compute_whitened_restriction(R, W_source, W_target)`**

*   **Purpose:** Transforms an original restriction map `R` into the whitened coordinate system: `R_whitened = W_target @ R @ W_source_pseudoinverse`.
*   **How it works:** Applies the source and target whitening maps to the original restriction map. The resulting `R_whitened` should be (approximately) orthogonal if the original `R` was computed correctly.
*   **Parameters:**
    *   `R` (torch.Tensor): The original restriction map (n_target x n_source).
    *   `W_source` (torch.Tensor): Whitening map for the source layer's Gram matrix (r_source x n_source).
    *   `W_target` (torch.Tensor): Whitening map for the target layer's Gram matrix (r_target x n_target).
*   **Returns:** (Tuple\[torch.Tensor, Dict])
    *   `R_whitened`: The whitened restriction map (r_target x r_source).
    *   `info`: Metadata including `orthogonality_error` (Frobenius norm of `R_whitened^T @ R_whitened - I`).

---

### `ProcrustesMaps`

Computes restriction maps between layer representations (stalks) using scaled Procrustes analysis. These maps define how information transforms between connected layers in the sheaf.

**Constructor: `ProcrustesMaps(epsilon=1e-8, max_scale=100.0, min_scale=1e-3, use_whitened_coordinates=False)`**

*   **Purpose:** Initializes the Procrustes map calculator.
*   **Parameters:**
    *   `epsilon` (float): Small value for numerical stability (e.g., added to denominators). Default: `1e-8`.
    *   `max_scale` (float): Maximum allowed scaling factor for the restriction map. Default: `100.0`.
    *   `min_scale` (float): Minimum allowed scaling factor. Default: `1e-3`.
    *   `use_whitened_coordinates` (bool): If `True` (default), computes restriction maps in a whitened space for exact metric compatibility and orthogonality. This is generally recommended.

**Method: `compute_restriction_map(K_source, K_target, method='scaled_procrustes', validate=True, use_whitening=None)`**

*   **Purpose:** Computes the restriction map from a source Gram matrix `K_source` to a target Gram matrix `K_target`.
*   **How it works:**
    *   If `use_whitening` (or the class default `use_whitened_coordinates`) is `True`:
        1.  It calls `scaled_procrustes_whitened`.
        2.  This method whitens `K_source` and `K_target` using `WhiteningProcessor`.
        3.  In the whitened space (where Gram matrices are identity), the optimal restriction map `R_whitened` is constructed (often a rectangular identity or projection).
        4.  `R_whitened` is then transformed back to the original coordinate space to get the final restriction map `R`. The scale factor is 1.0 in whitened space.
    *   If not using whitening (raw coordinates):
        1.  It typically calls `scaled_procrustes`.
        2.  This solves the orthogonal Procrustes problem to find an orthogonal matrix `Q` and a scale `s` that minimize `||s * K_source @ Q - K_target||_F`. The restriction map is `R = s * Q`.
        3.  Handles dimension mismatches by falling back to `orthogonal_projection` (SVD-based).
        4.  `least_squares` is another simpler baseline method.
*   **Parameters:**
    *   `K_source` (torch.Tensor): Gram matrix of the source layer.
    *   `K_target` (torch.Tensor): Gram matrix of the target layer.
    *   `method` (str): The method to use ('scaled_procrustes', 'orthogonal_projection', 'least_squares'). Default: `'scaled_procrustes'`.
    *   `validate` (bool): If `True`, performs validation checks on the computed map. Default: `True`.
    *   `use_whitening` (Optional\[bool]): Overrides the class default for using whitened coordinates. Applies primarily when `method='scaled_procrustes'`.
*   **Returns:** (Tuple\[torch.Tensor, float, Dict])
    *   `R`: The computed restriction map.
    *   `scale`: The optimal scaling factor.
    *   `info`: A dictionary with details about the computation, including errors, method used, and validation results. If whitening was used, this includes `whitened_validation` info.

---

### `Sheaf` (Dataclass)

Represents the cellular sheaf data structure.

*   **Attributes:**
    *   `poset` (networkx.DiGraph): The underlying graph structure (layers as nodes, data flow as edges).
    *   `stalks` (Dict\[str, torch.Tensor]): Data associated with each node (layer). In whitened coordinates, these are identity matrices of reduced rank.
    *   `restrictions` (Dict\[Tuple\[str, str], torch.Tensor]): Restriction maps associated with each directed edge `(u, v)` in the poset. In whitened coordinates, these are (approximately) orthogonal maps.
    *   `metadata` (Dict\[str, Any]): Additional information about the sheaf, such as construction method, validation status, whitening info.
    *   `whitening_maps` (Dict\[str, torch.Tensor]): Stores the whitening transformations for each stalk when `use_whitening=True`.

*   **Method: `validate(tolerance=1e-2)`**
    *   Checks mathematical properties of the sheaf, primarily the transitivity of restriction maps: `R_ac = R_bc @ R_ab`.
    *   Returns a dictionary with validation results and updates `self.metadata['validation_passed']`.
*   **Method: `get_laplacian_structure()`**
    *   Provides information about the expected structure of the sheaf Laplacian (total dimension, sparsity).
*   **Method: `summary()`**
    *   Returns a string summary of the sheaf.

---

### `SheafBuilder`

Orchestrates the construction of a `Sheaf` object.

**Constructor: `SheafBuilder(handle_dynamic=True, procrustes_epsilon=1e-8, restriction_method='scaled_procrustes', use_whitening=True, residual_threshold=0.05, enable_edge_filtering=True)`**

*   **Purpose:** Initializes the sheaf builder.
*   **Parameters:**
    *   `handle_dynamic` (bool): Passed to `FXPosetExtractor` for handling non-traceable models. Default: `True`.
    *   `procrustes_epsilon` (float): Epsilon for `ProcrustesMaps`. Default: `1e-8`.
    *   `restriction_method` (str): Default method for `ProcrustesMaps`. Default: `'scaled_procrustes'`.
    *   `use_whitening` (bool): **CRITICAL PARAMETER**. If `True`, enables whitening for restriction map computation via `ProcrustesMaps`. **Essential for ensuring exact metric compatibility and orthogonality needed for the Laplacian definition.** Default: `True` (recommended).
    *   `residual_threshold` (float): For edge filtering. If the relative reconstruction error of a restriction map exceeds this, the edge might be dropped. Default: `0.05` (5%).
    *   `enable_edge_filtering` (bool): If `True`, enables filtering of restriction maps based on `residual_threshold`. Default: `True`.

**Method: `build_from_activations(model, activations, use_gram_matrices=True, validate=True)`**

*   **Purpose:** Constructs a sheaf from a PyTorch model and a dictionary of its layer activations.
*   **How it works:**
    1.  Uses `FXToModuleNameMapper` to unify activation keys with FX node names.
    2.  Extracts the poset using `FXPosetExtractor` (potentially filtered by available activations).
    3.  Assigns stalks: If `use_gram_matrices` is `True`, computes Gram matrices (`X @ X.T`) from activations for each layer. If `use_whitening` is enabled, transforms these to identity matrices in whitened space.
    4.  Computes restriction maps for edges in the poset using `ProcrustesMaps` (with whitening for exact orthogonality).
    5.  Applies edge filtering based on `residual_threshold` if `enable_edge_filtering` is `True`.
    6.  Creates and returns a `Sheaf` object with all components.
    7.  Optionally validates the sheaf transitivity properties.
*   **Parameters:**
    *   `model` (torch.nn.Module): The PyTorch model.
    *   `activations` (Dict\[str, torch.Tensor]): Dictionary mapping layer names to **raw** activation tensors.
    *   `use_gram_matrices` (bool): If `True` (default), stalks will be Gram matrices (identity in whitened space).
    *   `validate` (bool): If `True`, validates the constructed sheaf. Default: `True`.
*   **Returns:** (Sheaf) The constructed sheaf object.

**Method: `build_from_cka_matrices(poset, cka_matrices, validate=True)`**

*   **Purpose:** Constructs a sheaf using a pre-computed poset and CKA similarity matrices as stalks.
*   **How it works:** Similar to `build_from_activations`, but uses the provided `cka_matrices` directly as stalks and computes restriction maps between them.
*   **Returns:** (Sheaf) The constructed sheaf object.

---

### `SheafLaplacianBuilder`

Constructs the sparse sheaf Laplacian matrix (Δ = δ<sup>T</sup>δ) from a `Sheaf` object using highly optimized assembly methods. This Laplacian's spectral properties reveal information about the network's structure and similarity.

#### Mathematical Formulation (Validated)

The sheaf Laplacian is defined as **Δ = δᵀδ** where δ is the coboundary operator. For a 0-cochain f = {fᵥ ∈ Stalk(v)}, the coboundary acts on edge e=(u,v) as:
**(δf)ₑ = fᵥ - Rₑfᵤ**

The resulting Laplacian has the following **mathematically verified block structure**:

**Diagonal Blocks (Δᵥᵥ)**: 
```
Δᵥᵥ = Σ_{e=(v,w)} (RₑᵀRₑ) + Σ_{e=(u,v)} I
```
- Sum of RᵀR for all outgoing edges e=(v,w)  
- Identity matrix for each incoming edge e=(u,v)

**Off-Diagonal Blocks**:
```
Δᵥw = -Rₑᵀ  (for edge e=(v,w))
Δwᵥ = -Rₑ   (for edge e=(v,w))
```

#### Validated Mathematical Properties

✅ **Universal Properties** (Verified by comprehensive test suite):
- **Symmetry**: Δ = Δᵀ (error < 1e-15)
- **Positive Semi-Definite**: All eigenvalues ≥ 0 (min eigenvalue ≥ -1e-15)
- **Numerical Stability**: Robust under 1e-8 perturbations

✅ **Topological Properties**:
- **Block-Diagonal Structure**: Disconnected components create zero cross-blocks
- **Standard Laplacian Reduction**: 1D identity sheaves → exact combinatorial Laplacian
- **Kernel Analysis**: Dimension equals number of connected components for identity restrictions

✅ **Sheaf Configuration Properties**:
- **Trivial Kernel**: Diamond patterns with incompatible restrictions → 0D kernel
- **Non-trivial Kernel**: Identity restrictions on connected graphs → multi-dimensional kernel
- **Edge Cases**: Zero-weight edges, single nodes, empty graphs handled correctly

> **Validation Reference**: All properties verified by `comprehensive_laplacian_validation.py` with 18/18 tests passing (100% mathematical correctness)

**Constructor: `SheafLaplacianBuilder(enable_gpu=True, memory_efficient=True, validate_properties=True, assembly_method='preallocated')`**

*   **Purpose:** Initializes the Laplacian builder.
*   **Parameters:**
    *   `enable_gpu` (bool): If `True` and a GPU is available, attempts to use GPU-compatible sparse tensor operations. Default: `True`.
    *   `memory_efficient` (bool): If `True` (default), uses memory-efficient assembly patterns.
    *   `validate_properties` (bool): If `True` (default), validates mathematical properties of the constructed Laplacian (symmetry, positive semi-definiteness). **Disable for production speed**.
    *   `assembly_method` (str): **NEW OPTIMIZATION PARAMETER**. Choose assembly method:
        - `'preallocated'`: **64x faster** - Pre-allocated COO arrays with vectorized operations (default)
        - `'block_wise'`: **17x faster** - Uses scipy.sparse.bmat for efficient block assembly  
        - `'current'`: Original implementation for compatibility
        - `'auto'`: Automatically select best method

**Method: `build_laplacian(sheaf, edge_weights=None)`**

*   **Purpose:** Constructs the sparse sheaf Laplacian using the selected optimization method.
*   **How it works:**
    1.  Routes to the appropriate assembly method based on `assembly_method` parameter.
    2.  **Preallocated Method** (default):
        - Estimates total non-zeros for exact pre-allocation
        - Uses vectorized `np.where()` operations for block insertion
        - Eliminates dynamic memory reallocation
        - Achieves 64x speedup over original method
    3.  **Block-wise Method**:
        - Uses `scipy.sparse.bmat()` for efficient block matrix assembly
        - Avoids intermediate dense representations
        - Achieves 17x speedup over original method
    4.  Constructs the Laplacian block by block:
        *   **Diagonal blocks (Δ<sub>vv</sub>):** For each node `v`, accumulates `R_e^T R_e` (outgoing edges) and `R_{e'} R_{e'}^T` (incoming edges)
        *   **Off-diagonal blocks (Δ<sub>vw</sub>):** For edge `e=(v,w)`, sets `-R_e` and `-R_e^T` 
    5.  Ensures symmetry and validates properties if requested.
*   **Parameters:**
    *   `sheaf` (Sheaf): The input sheaf with whitened stalks (identity matrices) and orthogonal restriction maps.
    *   `edge_weights` (Optional\[Dict\[Tuple\[str, str], float]]): Optional weights for edges. If `None`, weights derived from restriction map properties.
*   **Returns:** (Tuple\[scipy.sparse.csr_matrix, LaplacianMetadata])
    *   `sparse_laplacian`: The constructed sheaf Laplacian as an optimized sparse matrix.
    *   `metadata`: Information about the construction process and performance.

**Method: `to_torch_sparse(laplacian)`**

*   **Purpose:** Converts a SciPy sparse CSR matrix to a PyTorch sparse COO tensor, potentially moving it to GPU if enabled.
*   **Returns:** (torch.sparse.FloatTensor) The Laplacian as a PyTorch sparse tensor.

---

### `FXToModuleNameMapper`

A utility class to resolve inconsistencies between node names generated by `torch.fx` symbolic tracing (e.g., `_0`, `_1_conv1`) and module names obtained from `model.named_modules()` (e.g., `0.conv1`, `layer1.0.conv1`). This mapping is critical for associating externally captured activations (usually hooked via module names) with the correct nodes in an FX-derived poset.

**Constructor: `FXToModuleNameMapper()`**

**Method: `build_mapping(model, fx_graph)`**

*   **Purpose:** Creates a bidirectional mapping between FX node names and standard module names.
*   **How it works:** Iterates through the `fx_graph` nodes. For `call_module` FX nodes, it attempts to match the `node.target` (which is how FX refers to the module) with the actual module names from `model.named_modules()`. It tries exact matches and heuristics for variations.
*   **Parameters:**
    *   `model` (torch.nn.Module): The PyTorch model.
    *   `fx_graph` (torch.fx.Graph): The FX graph obtained from tracing the model.
*   **Returns:** (Dict\[str, str]) The mapping from FX node names to module names. The instance also stores `self.module_to_fx`.

**Method: `translate_activations(activations, direction="module_to_fx")`**

*   **Purpose:** Translates the keys of an activation dictionary from one naming scheme to another using the built mapping.
*   **Parameters:**
    *   `activations` (Dict\[str, torch.Tensor]): The activation dictionary.
    *   `direction` (str): Either `"module_to_fx"` (default) to convert module names to FX names, or `"fx_to_module"`.
*   **Returns:** (Dict\[str, torch.Tensor]) A new dictionary with translated keys.

---

### `EnhancedActivationExtractor`

Extracts activations from a model, capturing outputs from both `nn.Module` instances and common functional operations (like `F.relu`, `F.adaptive_avg_pool2d`, `torch.flatten`) that don't have explicit module names.

**Constructor: `EnhancedActivationExtractor(capture_functional=True)`**

*   **Parameters:**
    *   `capture_functional` (bool): If `True` (default), enables the capture of activations from functional calls.

**Method: `extract_comprehensive_activations(model, input_tensor)`**

*   **Purpose:** Performs a forward pass and collects activations from both standard modules (via hooks) and patched functional operations.
*   **How it works:**
    *   Registers forward hooks on standard `nn.Module` layers.
    *   Uses a context manager (`FunctionalOperationCapture`) that temporarily patches functions in `torch.nn.functional` (and some `torch` functions like `flatten`) with wrappers. These wrappers execute the original function and then store its output as an activation.
    *   Automatically handles tensor dimensionality (4D→2D via spatial averaging).
    *   Functional activation names are generated like `relu_0`, `adaptive_avg_pool2d_1`, etc.
*   **Returns:** (Dict\[str, torch.Tensor]) A dictionary mapping operation names to their **raw** activation tensors.

---

### Utility Functions

*   **`create_sheaf_from_cka_analysis(cka_results, layer_names, network_structure=None)`:** A helper to build a `Sheaf` directly from the output of a CKA analysis (e.g., from `neurosheaf.cka`). If `network_structure` (a poset) isn't provided, it assumes a simple sequential one.
*   **`validate_sheaf_properties(restrictions, poset, tolerance=1e-2)`:** Checks the transitivity property (`R_ac = R_bc @ R_ab`) for restriction maps in a given poset.
*   **`create_unified_activation_dict(model, activations)`:** Convenience function that uses `FXToModuleNameMapper` to trace a model, build the name mapping, and translate an activation dictionary (keyed by module names) to be keyed by FX node names. This makes the activations compatible with an FX-derived poset.
*   **`build_sheaf_laplacian(sheaf, enable_gpu=True, memory_efficient=True, assembly_method='preallocated')`:** A shortcut to create a `SheafLaplacianBuilder` and call its `build_laplacian` method with optimized defaults.

---

## Spectral Module (`neurosheaf.spectral`)

This module focuses on spectral analysis tools, particularly for handling filtrations of sheaf Laplacians. The core idea is to analyze how the spectral properties (eigenvalues, eigenvectors) of the sheaf Laplacian change as the underlying graph connectivity is varied, typically by thresholding edge weights.

### `StaticMaskedLaplacian`

Implements an efficient way to perform filtrations on a sheaf Laplacian. Instead of rebuilding the Laplacian matrix at each threshold, it builds the full Laplacian once and then applies Boolean masks to simulate the removal of edges based on their weights.

**Constructor: `StaticMaskedLaplacian(static_laplacian, metadata, masking_metadata, enable_gpu=True)`**

*   **Purpose:** Initializes the static masked Laplacian object. This is typically not called directly by users but through `create_static_masked_laplacian`.
*   **How it works:**
    *   Stores the pre-built full `static_laplacian` (a `scipy.sparse.csr_matrix`).
    *   Converts the static Laplacian to a `torch.sparse.FloatTensor` (`L_torch`) if `enable_gpu` is `True` and a GPU is available, for faster masking operations.
    *   Stores `metadata` (from the original Laplacian construction) and `masking_metadata` (containing edge weights).
    *   Validates an internal `edge_cache` (part of `metadata.edge_positions`) which maps sheaf edges to their corresponding (row, col) indices in the Laplacian matrix. This cache is crucial for efficient masking.
*   **Parameters:**
    *   `static_laplacian` (scipy.sparse.csr_matrix): The full sheaf Laplacian including all edges, typically built using `SheafLaplacianBuilder`.
    *   `metadata` (LaplacianMetadata): Metadata associated with the `static_laplacian`. Must include `edge_positions`.
    *   `masking_metadata` (MaskingMetadata): Contains edge weights and will store statistics about masking operations.
    *   `enable_gpu` (bool): If `True`, attempts to use GPU for sparse tensor operations. Default: `True`.

**Method: `apply_threshold_mask(threshold, return_torch=False)`**

*   **Purpose:** Creates a filtered Laplacian by keeping only edges with weights strictly greater than the given `threshold`.
*   **How it works:**
    1.  Identifies edges whose weights are less than or equal to `threshold`.
    2.  Using the `edge_positions` from `construction_metadata`, it determines the matrix entries corresponding to these "removed" edges.
    3.  Creates a new sparse matrix (either SciPy CSR or PyTorch sparse tensor) where these entries are effectively zeroed out. The diagonal entries are also implicitly adjusted because the contributions from removed edges are no longer present.
    *   If `return_torch` is `True` and GPU is enabled, operations are performed on `self.L_torch`. Otherwise, on `self.L_static`.
*   **Parameters:**
    *   `threshold` (float): The weight threshold. Edges with `weight <= threshold` are masked (removed).
    *   `return_torch` (bool): If `True` and GPU is enabled, returns a `torch.sparse.FloatTensor`. Otherwise, returns a `scipy.sparse.csr_matrix`. Default: `False`.
*   **Returns:** (Union\[scipy.sparse.csr_matrix, torch.sparse.FloatTensor]) The filtered Laplacian.

**Method: `compute_filtration_sequence(thresholds, return_torch=False)`**

*   **Purpose:** Efficiently computes a sequence of filtered Laplacians for a list of threshold values.
*   **How it works:** Calls `apply_threshold_mask` for each threshold in the `thresholds` list.
*   **Parameters:**
    *   `thresholds` (List\[float]): A list of threshold values, typically in ascending order.
    *   `return_torch` (bool): Whether the Laplacians in the sequence should be PyTorch sparse tensors.
*   **Returns:** (List\[Union\[scipy.sparse.csr_matrix, torch.sparse.FloatTensor]]) A list of filtered Laplacians.

**Method: `get_weight_distribution(num_bins=50)`**

*   **Purpose:** Provides a histogram of the edge weights, which can help in selecting appropriate thresholds for filtration.
*   **Returns:** (Tuple\[np.ndarray, np.ndarray]) Bin centers and counts for the edge weight histogram.

**Method: `suggest_thresholds(num_thresholds=50, strategy='uniform')`**

*   **Purpose:** Generates a list of suggested threshold values for a filtration.
*   **Parameters:**
    *   `num_thresholds` (int): The desired number of thresholds.
    *   `strategy` (str): Method for selecting thresholds:
        *   `'uniform'`: Evenly spaced thresholds between min and max edge weights.
        *   `'quantile'`: Thresholds based on quantiles of the edge weight distribution.
        *   `'adaptive'`: Thresholds spaced more densely in regions with more edges.
*   **Returns:** (List\[float]) A sorted list of unique threshold values.

**Method: `validate_masking_integrity(threshold)`**

*   **Purpose:** Checks if the masked Laplacian at a given threshold retains key mathematical properties (e.g., symmetry, positive semi-definiteness).
*   **Returns:** (Dict\[str, Any]) A dictionary with validation results.

**Method: `get_memory_usage()`**

*   **Purpose:** Reports the memory usage of the `StaticMaskedLaplacian` object itself (static matrix, torch tensor, edge cache).
*   **Returns:** (Dict\[str, float]) Memory usage breakdown in GB.

---

### `MaskingMetadata` (Dataclass)

Stores metadata related to edge masking operations performed by `StaticMaskedLaplacian`.

*   **Attributes:**
    *   `edge_weights` (Dict\[Tuple\[str, str], float]): A dictionary mapping sheaf edges (source_node, target_node) to their weights. These weights are used for thresholding.
    *   `weight_range` (Tuple\[float, float]): The minimum and maximum edge weights observed.
    *   `threshold_count` (int): Number of times a threshold mask has been applied.
    *   `masking_times` (List\[float]): List of durations for each masking operation.
    *   `active_edges` (Dict\[float, int]): Maps each applied threshold to the number of edges remaining active (weight > threshold).
    *   `laplacian_ranks` (Dict\[float, int]): (Potentially) maps thresholds to the rank of the resulting masked Laplacian if computed.

---

### Utility Functions

*   **`create_static_masked_laplacian(sheaf, enable_gpu=True)`:**
    *   **Purpose:** A convenience function to create a `StaticMaskedLaplacian` object directly from a `Sheaf` object.
    *   **How it works:**
        1.  Builds the full static sheaf Laplacian using optimized `SheafLaplacianBuilder` from the input `sheaf`.
        2.  Extracts edge weights. Currently, it uses the Frobenius norm of the restriction maps as default weights. (Note: The source of these weights might be refined or made more configurable in future versions).
        3.  Initializes and returns a `StaticMaskedLaplacian` instance.
    *   **Parameters:**
        *   `sheaf` (Sheaf): The input sheaf, expected to have stalks and restriction maps.
        *   `enable_gpu` (bool): Passed to the underlying builders.
    *   **Returns:** (StaticMaskedLaplacian) The initialized static masked Laplacian object.

---

## Utilities (`neurosheaf.utils`)

This module provides a collection of helper utilities that support the core functionality of the Neurosheaf package. While primarily for internal use, understanding these utilities can offer insight into the package's behavior regarding configuration, error handling, performance, and device management.

*   **`config.py`:**
    *   **Purpose:** Centralizes configuration constants used throughout the framework.
    *   **Key Components:** Defines dataclasses like `NumericalConstants`, `MemoryConstants`, `CKAConstants`, `NystromConstants`, `SheafConstants`, `PerformanceConstants`, and `ValidationConstants`. These group related parameters (e.g., default epsilon values, memory thresholds, CKA sample size requirements).
    *   **How it works:** Provides a global `Config` class that holds instances of these constant groups, allowing consistent access to default values and thresholds.

*   **`device.py`:**
    *   **Purpose:** Manages device detection (CPU, CUDA, MPS) and ensures consistent device handling.
    *   **Key Components:**
        *   `detect_optimal_device(device=None)`: Auto-detects the best available device (MPS on Mac, CUDA elsewhere, then CPU).
        *   `get_device_info()`: Returns a dictionary with comprehensive platform and device details.
        *   `should_use_cpu_fallback(device, operation)`: Determines if an operation should fall back to CPU due to known stability issues on devices like MPS (e.g., for SVD).
        *   `safe_to_device(tensor, device, operation)`: Moves a tensor to a device, considering potential fallbacks.
        *   `clear_device_cache(device)`: Clears memory caches for CUDA or MPS.

*   **`error_handling.py`:**
    *   **Purpose:** Provides robust error handling mechanisms, including decorators for retrying operations and adding context to exceptions.
    *   **Key Components:**
        *   `@safe_torch_operation(operation, ...)`: Decorator for PyTorch operations that might fail on certain devices (like MPS), providing automatic CPU fallback and retries.
        *   `@safe_file_operation(operation, ...)`: Decorator for file I/O with retry logic.
        *   `validate_tensor_properties(tensor, ...)`: Validates various properties of a tensor (NaN/Inf, dimensions, shape, positive semi-definiteness).
        *   `handle_numerical_instability(func, ...)`: A wrapper to apply regularization or other measures if a numerical operation (like matrix inversion) initially fails.

*   **`exceptions.py`:**
    *   **Purpose:** Defines a custom hierarchy of exceptions for more specific error reporting.
    *   **Key Components:** All exceptions inherit from `NeurosheafError`. Specific exceptions include:
        *   `ValidationError`: For invalid inputs or parameters.
        *   `ComputationError`: For failures during numerical computations (e.g., singular matrix).
        *   `MemoryError`: When memory limits are exceeded.
        *   `ArchitectureError`: For unsupported model architectures or FX tracing failures.
        *   `ConfigurationError`: For issues with setup or configuration values.
        *   `ConvergenceError`: When iterative algorithms fail to converge.
    *   These exceptions often carry additional context about the error.

*   **`logging.py`:**
    *   **Purpose:** Implements a unified logging system for the package.
    *   **Key Components:**
        *   `setup_logger(name, level, ...)`: Configures and returns a logger instance with console and optional file output.
        *   Supports different log levels and formatting.
        *   Includes a `PerformanceHandler` for logging performance-specific messages.

*   **`memory.py`:**
    *   **Purpose:** Provides tools for monitoring and estimating memory usage.
    *   **Key Components:**
        *   `MemoryMonitor` class: Tracks current and available memory (CPU, CUDA, MPS).
            *   `get_memory_info()`: Returns detailed memory statistics.
            *   `available_mb()`: Gets available memory in MB.
            *   `check_memory_available(required_mb)`: Checks if enough memory is free.
            *   `estimate_tensor_memory(shape, dtype)`: Estimates memory for a tensor.
            *   `optimal_chunk_size(...)`: Heuristic to determine chunk sizes for memory-efficient processing.

*   **`profiling.py`:**
    *   **Purpose:** Offers decorators and utilities for detailed performance profiling (execution time and memory usage).
    *   **Key Components:**
        *   `@profile_memory(...)`: Decorator to measure memory usage of a function. It uses a `MemoryMeasurementSystem` with multiple backends (tracemalloc, psutil, MPS, CUDA, system) for robust and precise measurement, especially on Apple Silicon Macs with unified memory. It attempts to differentiate CPU and GPU memory usage.
        *   `@profile_time(...)`: Decorator to measure execution time.
        *   `@profile_comprehensive(...)`: Combines both memory and time profiling.
        *   `ProfileManager`: A global manager to collect and report profiling results.
        *   `MemoryMeasurementSystem`: Handles complex memory measurements across different hardware.
        *   `benchmark_function(...)`: Utility to benchmark a function over multiple runs.
        *   `assess_memory_reduction(...)`: Compares memory usage against targets.
        *   `get_mac_memory_info()`: Specific memory reporting for macOS.

*   **`validation.py`:**
    *   **Purpose:** Contains specific validation functions, primarily for CKA-related inputs.
    *   **Key Components:**
        *   `validate_activations(X, Y, min_samples)`: Checks types, dimensions, sample counts, and NaN/Inf values for activation matrices.
        *   `validate_no_preprocessing(activations, ...)`: Warns if activations appear to be pre-centered, which is incorrect for debiased CKA.
        *   `validate_gram_matrix(K, name)`: Checks if a matrix is square, symmetric, and free of NaN/Inf.
        *   `validate_sample_indices(indices, ...)`: Validates indices used for subsampling.

---

## Visualization (`neurosheaf.visualization`)

This module is planned for implementing visualization capabilities for the Neurosheaf package. The goal is to provide tools for visually inspecting and understanding the various data structures and analysis results generated by the framework.

**Current Status: Placeholder**

As of the current version, this module is a placeholder and does not yet contain implemented visualization functionalities. Future development (targeted for Phase 5) aims to include:

*   **Stalk Visualizers:** To inspect the data on the stalks of the sheaf (e.g., activation patterns, Gram matrix structures).
*   **Poset Visualizers:** To render the extracted computational graphs (posets), potentially annotating nodes and edges with relevant metrics.
*   **CKA Matrix Visualization:** Heatmaps or other representations of CKA similarity matrices.
*   **Persistence Visualizers:** Tools to plot persistence diagrams and barcodes derived from the spectral analysis of sheaf Laplacians, possibly with log-scale support.
*   **Interactive Dashboards:** Potentially an integrated dashboard for exploring different aspects of a Neurosheaf analysis.
*   **Automatic Backend Switching:** For plotting libraries, to adapt to different user environments (e.g., matplotlib, plotly).

Users interested in visualizing results from Neurosheaf in the interim would need to use external libraries (e.g., `matplotlib`, `seaborn`, `networkx` plotting utilities) to plot the tensor and graph data produced by other modules.

---

## Performance & Optimization

### Laplacian Assembly Optimization

The framework features **highly optimized sparse matrix assembly** for sheaf Laplacian construction:

#### Performance Comparison

| Assembly Method | Time | Speedup | Use Case |
|----------------|------|---------|----------|
| **Preallocated COO** | **0.015s** | **64x faster** | **Production (default)** |
| **Block-wise** | 0.054s | 17x faster | Alternative method |
| **Current** | 0.944s | baseline | Compatibility |

#### Key Optimizations

1. **Pre-allocated Arrays**: Eliminates dynamic memory reallocation
2. **Vectorized Operations**: Uses `np.where()` instead of nested loops  
3. **Memory Estimation**: Accurate pre-sizing prevents over-allocation
4. **Block Assembly**: Efficient `scipy.sparse.bmat()` utilization

#### Configuration

```python
# Optimal production settings
laplacian_builder = SheafLaplacianBuilder(
    assembly_method='preallocated',  # 64x speedup
    validate_properties=False,      # Disable for speed
    enable_gpu=False,               # CPU optimization sufficient
    memory_efficient=True
)
```

### Memory Optimization

#### Whitened Coordinates Benefits

- **Reduced Dimensions**: Identity matrices instead of full Gram matrices
- **Sparse Structure**: 99.9% sparsity in final Laplacian
- **Better Conditioning**: Exact orthogonality and metric compatibility

#### Memory Usage Patterns

| Phase | Memory | Optimization |
|-------|--------|-------------|
| Activation Extraction | 1.8GB | Use smaller batch sizes |
| Sheaf Construction | 0.01GB | Whitening reduces storage |
| Laplacian Assembly | 0.01GB | Sparse operations only |
| **Total Peak** | **2.3GB** | **1.6x under target** |

### Profiling and Monitoring

```python
# Enable detailed profiling
from neurosheaf.utils.profiling import profile_comprehensive

@profile_comprehensive()
def analyze_network(model, data):
    # Your analysis code here
    pass

# Memory monitoring
from neurosheaf.utils.memory import MemoryMonitor

monitor = MemoryMonitor()
print(f"Available memory: {monitor.available_mb():.1f}MB")
```

### Platform-Specific Optimizations

#### Apple Silicon (MPS)
- Automatic CPU fallback for SVD operations
- Unified memory optimization
- Native sparse tensor support

#### CUDA/GPU
- GPU-compatible sparse operations
- Automatic memory management
- Batch processing optimization

#### CPU/Multicore
- Vectorized numpy operations
- Cache-friendly memory access patterns
- BLAS optimization utilization

---

This documentation provides a comprehensive guide to the Neurosheaf package, from basic usage to advanced optimization techniques. The framework is designed for production use with mathematical rigor, computational efficiency, and ease of use.
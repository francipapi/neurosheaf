# Clean Architecture Baseline Metrics

**Status**: ✅ LOCKED - Established on [current date]
**Purpose**: Regression protection for clean sheaf architecture
**Test File**: `resnet_test_strict.py`

## Overview

These metrics were established from a working ResNet-18 style model processed through the clean sheaf architecture. All future test runs **MUST** match these exact values (within specified tolerances) to ensure no regression in the clean implementation.

## Test Configuration

- **Model**: ResNet-18 style with 2 layers (812,776 parameters)
- **Input**: 32x3x112x112 (batch_size=32, img_size=112x112)
- **Random Seed**: 42 (for reproducibility)
- **Architecture**: Clean whitened coordinates only

## Strict Baseline Metrics

### Structure Metrics (EXACT match required)

| Metric | Value | Tolerance | Notes |
|--------|-------|-----------|-------|
| `nodes` | 13 | ±0 | Exact count required |
| `edges` | 76 | ±0 | Exact count required |
| `stalks` | 13 | ±0 | Exact count required |
| `restrictions` | 76 | ±0 | Exact count required |
| `total_stalk_dimensions` | 416 | ±0 | Exact sum required |

### Laplacian Metrics (EXACT match required)

| Metric | Value | Tolerance | Notes |
|--------|-------|-----------|-------|
| `laplacian_shape` | (416, 416) | ±0 | Exact dimensions required |
| `laplacian_nnz` | 167,968 | ±0 | Exact non-zero count required |

### Mathematical Properties (Tight tolerances)

| Metric | Value/Range | Tolerance | Notes |
|--------|-------------|-----------|-------|
| `laplacian_sparsity` | 2.94% | ±0.1% | Range: [2.9%, 3.1%] |
| `symmetry_error` | 0.00e+00 | ≤ 1e-12 | Must be essentially zero |
| `min_eigenvalue` | -1.42e-07 | ≥ -1e-6 | PSD with numerical tolerance |
| `harmonic_dimension` | 1 | ±0 | Exact count of near-zero eigenvalues |

### Performance Bounds (5% tolerance from baseline)

| Metric | Baseline | Max Allowed | Notes |
|--------|----------|-------------|-------|
| `extraction_time` | 0.126s | 0.133s | 5% tolerance |
| `construction_time` | 0.044s | 0.046s | 5% tolerance |
| `laplacian_time` | 0.210s | 0.221s | 5% tolerance |
| `total_time` | 0.379s | 0.398s | 5% tolerance |

### Required Properties

| Property | Required Value | Notes |
|----------|---------------|-------|
| `whitened_coordinates` | `True` | Must use whitened coordinates |
| `validation_passed` | N/A | Validation disabled for performance |

## Mathematical Validation

### Eigenvalue Analysis
- **Smallest eigenvalues**: `[4.11e-02, 3.81e-02, 3.51e-02, 2.74e-02, -1.42e-07]`
- **Harmonic dimension**: 1 (eigenvalues < 1e-6)
- **PSD validation**: Min eigenvalue ≥ -1e-6 (allows small numerical errors)

### Whitened Coordinate Properties
- All stalks are approximately identity matrices (K → I transformation)
- All restriction maps satisfy orthogonality in whitened space
- Mathematical optimality achieved (vs 44% acceptance with back-transformation)

## Test Execution

### Running the Test
```bash
# Single test
python tests/resnet_test_strict.py

# Via pytest
python -m pytest tests/resnet_test_strict.py::test_clean_architecture_strict -v
```

### Expected Output
```
🔒 STRICT BASELINE VALIDATION - CLEAN ARCHITECTURE
======================================================================
✅ Structure matches baseline exactly
✅ Mathematical properties validated  
✅ Performance within bounds
✅ Clean architecture working correctly

🔒 BASELINE METRICS CONFIRMED - NO REGRESSION DETECTED
```

## Regression Detection

### Failure Scenarios
Any deviation from these metrics indicates:

1. **Structure Changes**: Changes in nodes/edges/stalks suggest poset extraction issues
2. **Dimension Changes**: Changes in stalk dimensions suggest whitening issues  
3. **Laplacian Changes**: Changes in shape/nnz suggest assembly issues
4. **Performance Regression**: Exceeding time bounds suggests efficiency issues
5. **Mathematical Issues**: Symmetry/PSD violations suggest numerical issues

### Debugging Failed Tests
1. Check if model architecture changed
2. Verify random seed is set to 42
3. Ensure whitened coordinates are being used
4. Check for import changes in neurosheaf modules
5. Verify no modifications to core mathematical functions

## Implementation Notes

### Clean Architecture Benefits
- **100% Mathematical Correctness**: All operations in whitened space
- **Optimal Conditioning**: Identity stalks, orthogonal restrictions
- **Performance**: Fast construction and assembly
- **Simplicity**: No legacy fallbacks or complex parameters

### Key Differences from Original
- Uses only whitened methods (removed legacy approaches)  
- Simpler API (SheafBuilder(), SheafLaplacianBuilder())
- Better numerical stability (exact orthogonality)
- Cleaner module structure (core/, extraction/, assembly/)

## File Dependencies

### Test Files
- `resnet_test_strict.py` - Main strict validation test
- `simple_baseline.py` - Basic functionality verification
- `test_resnet18_progressive.py` - Scaling verification

### Source Dependencies
- `neurosheaf.sheaf.SheafBuilder` - Main construction class
- `neurosheaf.sheaf.SheafLaplacianBuilder` - Laplacian assembly
- `neurosheaf.sheaf.EnhancedActivationExtractor` - Activation extraction

## Maintenance

### Updating Metrics
⚠️ **WARNING**: These metrics should only be updated if:
1. Intentional improvements are made to the clean architecture
2. New baseline is thoroughly validated
3. Mathematical correctness is preserved
4. Performance improvements are verified

### Version Control
- All changes to baseline metrics must be documented
- Previous baselines should be archived
- Regression tests should be updated consistently

---

**Last Updated**: [Current Date]  
**Established By**: Clean Architecture Validation Process  
**Status**: 🔒 LOCKED for regression protection
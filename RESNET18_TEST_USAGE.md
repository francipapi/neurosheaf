# ResNet18 End-to-End Pipeline Test Usage Guide

## Overview

The `resnet18_end_to_end_test.py` script provides a comprehensive test of the entire neurosheaf pipeline using ResNet18 from torchvision. It validates all phases from model loading through spectral analysis, serving as a baseline reference implementation.

## Quick Start

### Basic Usage
```bash
# Run with auto-detected device
python resnet18_end_to_end_test.py

# Run with specific device
python resnet18_end_to_end_test.py --device cpu

# Run with verbose output
python resnet18_end_to_end_test.py --verbose
```

### Prerequisites

Ensure you have the conda environment activated:
```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device to use: `cpu`, `mps`, `cuda` | Auto-detected |
| `--verbose` | Enable detailed logging and output | False |
| `--help` | Show help message | - |

## Test Stages

The script runs through 5 main stages:

### Stage 1: Model Setup & Activation Extraction
- Downloads ResNet18 with ImageNet weights (~45MB)
- Generates deterministic test batch (256×3×224×224)
- Extracts activations from all layers using hooks
- **Expected**: ~52-68 layer activations extracted

### Stage 2: CKA Computation & Whitening  
- Computes CKA similarity matrix between layers
- Applies whitening transformation (99% variance retention)
- Validates stalk dimensions across different layer types
- **Expected**: Total dimensions ~6,000-9,300

### Stage 3: Sheaf Construction
- Extracts poset structure using FX symbolic tracing
- Computes restriction maps in pure whitened coordinates
- Assembles sheaf structure with validation
- **Expected**: ~68 nodes, ~75 edges, residuals <5%

### Stage 4: Laplacian Assembly & Spectral Analysis
- Builds sparse sheaf Laplacian matrix
- Computes eigenvalues using Lanczos iteration
- Validates spectral properties
- **Expected**: Sparse matrix ~6k×6k, harmonic dimension ~4

### Stage 5: Performance & Numerical Validation
- Validates all metrics against expected ranges
- Checks performance targets (time, memory)
- Reports comprehensive test results
- **Expected**: All validations pass within tolerance

## Expected Results

### Performance Targets
- **Runtime**: ≤5 minutes total on Mac 12-core
- **Memory**: ≤3.8GB peak usage
- **Success**: All 5 stages complete successfully

### Key Metrics
- **Nodes**: ~68 (close to expected 79)
- **Edges**: ~75 (close to expected 85) 
- **Total Dimensions**: ~6,000-9,300
- **Restriction Residuals**: <5%
- **Laplacian Properties**: Symmetric, positive semi-definite
- **Harmonic Dimension**: ~4

### Sample Output
```
===============================================================================
STARTING RESNET18 END-TO-END PIPELINE TEST
===============================================================================
Stage 1: Model Setup & Activation Extraction
✅ Stage 1 completed in 12.34s (Memory: 456.7MB)
   Extracted 52 layer activations

Stage 2: CKA Computation & Whitening
✅ Stage 2 completed in 45.67s (Memory: 1234.5MB)
   Total dimensions: 7890

Stage 3: Sheaf Construction
✅ Stage 3 completed in 23.45s (Memory: 1456.7MB)
   Poset: 68 nodes, 75 edges
   Max restriction residual: 0.032

Stage 4: Laplacian Assembly & Spectral Analysis
✅ Stage 4 completed in 34.56s (Memory: 1678.9MB)
   Laplacian: 7890×7890, 89012 non-zeros
   Harmonic dimension: 4

Stage 5: Performance & Numerical Validation
✅ Stage 5 completed in 1.23s - All validations passed!

===============================================================================
RESNET18 END-TO-END PIPELINE TEST SUMMARY
===============================================================================
Overall Result: ✅ PASSED
Total Runtime: 117.25s (target: ≤300s)
Peak Memory: 1.68GB (target: ≤3.8GB)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure proper environment activation
   source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv
   
   # Verify neurosheaf installation
   python -c "import neurosheaf; print('OK')"
   ```

2. **Memory Issues**
   ```bash
   # Run with CPU device for lower memory usage
   python resnet18_end_to_end_test.py --device cpu
   ```

3. **Torchvision Warnings**
   - The libjpeg warning can be safely ignored
   - It doesn't affect the pipeline functionality

4. **Device Detection Issues**
   ```bash
   # Force specific device if auto-detection fails
   python resnet18_end_to_end_test.py --device cpu
   ```

### Expected Warnings

The following warnings are normal and can be ignored:
- `torchvision.io` libjpeg loading warnings
- FX tracing warnings for complex model structures
- MPS numerical stability warnings (automatic CPU fallback)

## Validation Criteria

The test is considered successful when:

✅ All 5 stages complete without fatal errors  
✅ Numerical properties within tolerance (residuals, symmetry, eigenvalues)  
✅ Performance targets met (runtime ≤5min, memory ≤3.8GB)  
✅ Pipeline metrics match expected ranges (±10% tolerance)

## Integration with Development

This test serves as:
- **Baseline validation** for the complete pipeline
- **Reference implementation** for neurosheaf usage
- **Performance benchmark** for optimization efforts
- **Regression test** for code changes

Run this test regularly during development to ensure the pipeline remains functional and performant.

## Next Steps

After successful validation:
1. Use the script as a template for other model architectures
2. Extend with additional validation metrics as needed
3. Integrate into automated testing workflows
4. Create ground truth data files for reproducible validation
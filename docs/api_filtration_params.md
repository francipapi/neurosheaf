# Filtration Parameters in Network Comparison API

## Overview

The `compare_networks` and `compare_multiple_networks` methods now support configurable filtration parameters that control how the spectral analysis is performed during DTW comparison.

## New Parameters

### `filtration_type` (str, default: 'threshold')

Controls how edges are filtered during the persistence analysis:

- **'threshold'**: Standard edge weight threshold filtration
  - Edges are included based on their weight relative to the threshold
  - Good for general network analysis
  - Parameter range auto-detected from edge weights

- **'cka_based'**: CKA (Centered Kernel Alignment) based filtration
  - Uses CKA similarity values for filtration
  - Better for analyzing layer-wise similarity evolution
  - Parameter range typically [0, 1]

- **'custom'**: Custom filtration function
  - Allows user-defined edge filtering logic
  - Requires `custom_threshold_func` parameter

### `n_steps` (int, default: 50)

Number of filtration steps in the persistence analysis:

- Higher values provide finer resolution but increase computation time
- Lower values are faster but may miss subtle patterns
- Typical ranges:
  - Quick analysis: 10-20 steps
  - Standard analysis: 30-50 steps
  - Detailed analysis: 100+ steps

## Usage Examples

### Basic Usage

```python
from neurosheaf.api import NeurosheafAnalyzer

analyzer = NeurosheafAnalyzer()

# Compare with default parameters
result = analyzer.compare_networks(
    model1, model2, data,
    method='dtw'
)

# Compare with custom filtration parameters
result = analyzer.compare_networks(
    model1, model2, data,
    method='dtw',
    filtration_type='threshold',
    n_steps=100  # Higher resolution
)
```

### CKA-Based Filtration

```python
# Use CKA-based filtration for layer similarity analysis
result = analyzer.compare_networks(
    model1, model2, data,
    method='dtw',
    filtration_type='cka_based',
    n_steps=30,
    eigenvalue_index=0  # Compare largest eigenvalue
)
```

### Multiple Network Comparison

```python
# Compare multiple networks with custom parameters
models = [resnet18, resnet50, vgg16]
result = analyzer.compare_multiple_networks(
    models, data,
    method='dtw',
    filtration_type='threshold',
    n_steps=25,  # Balanced speed/accuracy
    multivariate=False
)
```

## Performance Considerations

### Computation Time

The computation time scales approximately as:
- O(n_steps) for eigenvalue computation
- O(n_steps²) for DTW alignment

Typical timings (on CPU):
- 10 steps: ~0.5-1s per network
- 25 steps: ~1-2s per network
- 50 steps: ~2-4s per network
- 100 steps: ~5-10s per network

### Memory Usage

Memory usage is primarily determined by:
- Network size (number of layers/parameters)
- Data batch size
- Number of filtration steps (minimal impact)

## Choosing Parameters

### For Quick Exploration
```python
filtration_type='threshold'
n_steps=15
```

### For Standard Analysis
```python
filtration_type='threshold'
n_steps=50
```

### For Detailed Investigation
```python
filtration_type='threshold'
n_steps=100
```

### For Layer Similarity Focus
```python
filtration_type='cka_based'
n_steps=30
```

## Understanding Results

The results dictionary includes the filtration parameters used:

```python
result['comparison_metadata']['filtration_type']  # 'threshold'
result['comparison_metadata']['n_steps']          # 50
```

These parameters affect:
- **Similarity scores**: Different filtrations may reveal different similarity patterns
- **Computation time**: More steps = longer computation
- **Resolution**: More steps = finer temporal resolution in eigenvalue evolution

## Advanced Usage

### Custom Filtration Function

```python
def custom_edge_threshold(edge, param):
    """Custom edge filtering logic."""
    source, target = edge
    # Custom logic here
    return weight > param * custom_factor

result = analyzer.compare_networks(
    model1, model2, data,
    method='dtw',
    filtration_type='custom',
    n_steps=40,
    custom_threshold_func=custom_edge_threshold
)
```

### Parameter Tuning

To find optimal parameters:

1. Start with defaults (threshold, 50 steps)
2. If computation is too slow, reduce n_steps
3. If results seem coarse, increase n_steps
4. Try cka_based for layer-focused analysis
5. Monitor computation time vs result quality

## FAQ

**Q: Which filtration_type should I use?**
A: Start with 'threshold' for general analysis. Use 'cka_based' when focusing on layer-wise similarity.

**Q: How many steps do I need?**
A: 50 steps is a good default. Use fewer (20-30) for quick exploration, more (100+) for publication-quality analysis.

**Q: Does filtration_type affect all comparison methods?**
A: No, only the 'dtw' method uses these parameters. 'euclidean' and 'cosine' methods ignore them.

**Q: Can I use different parameters for each model?**
A: No, the same filtration parameters are applied to all models for consistency in comparison.
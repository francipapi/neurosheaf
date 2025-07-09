# Neurosheaf Implementation Guidelines for AI Coding Agent

## Overview
These guidelines maximize implementation success for the Neurosheaf project across all 7 phases. Follow these principles to ensure production-quality code that meets all performance and correctness requirements.

## Core Implementation Principles

### 1. **Always Reference the Plan**
- **Before coding**: Read the specific phase implementation plan and testing suite
- **During coding**: Check plan details for exact specifications and code examples
- **After coding**: Validate against plan requirements and success criteria
- **Key files**: `plan/phase{X}_*/implementation_plan.md` and `testing_suite.md`

### 2. **Critical Technical Requirements (Non-Negotiable)**
- **NO double-centering in CKA**: Use raw activations, not pre-centered data
- **Mathematical correctness**: All algorithms must match reference implementations
- **Memory targets**: <3GB for ResNet50 analysis
- **Performance targets**: <5 minutes for complete analysis pipeline

### 3. **Test-Driven Development**
- **Write tests first**: Start with critical tests from testing suites
- **Test immediately**: Run tests after every significant code change
- **Validate continuously**: Use `pytest -v` frequently during development
- **Cover edge cases**: Implement edge case tests before they become issues

## Phase-Specific Critical Points

### Phase 1: Foundation
- **Setup validation**: Ensure all imports work before proceeding
- **Logging first**: Implement logging before any complex logic
- **Performance baseline**: Establish 1.5TB memory baseline before optimization

### Phase 2: CKA (MOST CRITICAL)
```python
# ALWAYS use this pattern - NO pre-centering
K = X @ X.T  # Raw activations only!
L = Y @ Y.T  # Raw activations only!
# NEVER do: X_centered = X - X.mean(dim=0)
```
- **Validate mathematically**: CKA(X,X) = 1, CKA(X,Y) = CKA(Y,X), 0 ≤ CKA ≤ 1
- **Test against references**: Compare with known correct implementations

### Phase 3: Sheaf Construction
- **FX tracing**: Always implement fallback for dynamic models
- **Validate sheaf axioms**: Test transitivity and consistency properties
- **Sparse operations**: Verify >90% memory savings vs dense matrices

### Phase 4: Spectral Analysis
- **Subspace tracking**: Use principal angles, not index-based tracking
- **Handle crossings**: Test eigenvalue crossing scenarios extensively
- **Numerical stability**: Add epsilon for near-zero eigenvalues

### Phase 5: Visualization
- **Log-scale detection**: Auto-detect when CKA values need log scaling
- **Backend switching**: Implement matplotlib→Plotly→WebGL progression
- **Memory efficiency**: Use downsampling for large visualizations

### Phase 6: Testing
- **Coverage target**: >95% for all modules
- **Real architectures**: Test ResNet, Transformer, CNN, RNN variants
- **Performance validation**: All benchmarks must pass

### Phase 7: Deployment
- **Clean environment**: Test pip install in fresh virtual environment
- **Documentation**: All APIs must have docstrings and examples

## Code Quality Standards

### Error Handling
```python
# Always use custom exceptions
from neurosheaf.utils.exceptions import ValidationError, ComputationError

# Validate inputs
if not isinstance(cka_matrix, torch.Tensor):
    raise ValidationError("CKA matrix must be torch.Tensor")

# Handle edge cases gracefully
if torch.any(torch.isnan(eigenvals)):
    logger.warning("NaN eigenvalues detected, applying regularization")
    eigenvals = torch.clamp(eigenvals, min=1e-8)
```

### Performance Monitoring
```python
# Always add performance monitoring
from neurosheaf.utils.profiling import profile_memory

@profile_memory
def compute_cka_matrix(activations):
    # Implementation here
    pass
```

### Type Hints and Documentation
```python
def compute_restriction(
    source: torch.Tensor,
    target: torch.Tensor,
    method: str = 'scaled_procrustes'
) -> torch.Tensor:
    """Compute restriction map between activation spaces.
    
    Args:
        source: Source activations [n_samples, d_source]
        target: Target activations [n_samples, d_target]
        method: Computation method
        
    Returns:
        Restriction matrix [d_source, d_target]
        
    Raises:
        ValidationError: If inputs have mismatched sample dimensions
    """
```

## Implementation Workflow

### Starting a New Module
1. **Read the plan**: Study implementation_plan.md for the phase
2. **Understand tests**: Review testing_suite.md requirements
3. **Create skeleton**: Implement class/function signatures first
4. **Write critical tests**: Implement tests from testing suite
5. **Implement core logic**: Follow plan code examples exactly
6. **Test continuously**: Run tests after each function
7. **Handle edge cases**: Implement edge case handling
8. **Performance validate**: Check memory/speed requirements
9. **Document thoroughly**: Add docstrings and examples

### Making Changes to Existing Code
1. **Read existing tests**: Understand what behavior is expected
2. **Add new tests first**: For new functionality or bug fixes
3. **Refactor carefully**: Ensure all existing tests still pass
4. **Performance check**: Verify no regression in speed/memory
5. **Update documentation**: Modify docstrings if API changes

### Debugging Strategy
1. **Check the plan**: Verify implementation matches specification
2. **Test isolation**: Run specific test to isolate issue
3. **Add logging**: Use logger to trace execution flow
4. **Validate inputs**: Check tensor shapes, types, ranges
5. **Mathematical check**: Verify algorithm against paper/reference
6. **Memory profiling**: Use profiling tools if memory issues

## Common Pitfalls to Avoid

### Mathematical Errors
- **❌ Never**: Pre-center data for CKA computation
- **❌ Never**: Use index-based eigenvalue tracking
- **❌ Never**: Ignore numerical stability (add epsilon values)
- **✅ Always**: Validate mathematical properties in tests

### Performance Issues
- **❌ Never**: Create dense matrices for large networks
- **❌ Never**: Load entire dataset into memory simultaneously
- **❌ Never**: Ignore GPU memory limits
- **✅ Always**: Use sparse operations and adaptive sampling

### Code Quality Issues
- **❌ Never**: Commit code without tests
- **❌ Never**: Use hardcoded paths or magic numbers
- **❌ Never**: Ignore error handling
- **✅ Always**: Follow type hints and documentation standards

## File Organization Principles

### Module Structure
```python
# Each module should have clear separation
neurosheaf/module/
├── __init__.py          # Public API exports
├── core.py              # Main implementation
├── utils.py             # Helper functions
├── validation.py        # Input validation
└── exceptions.py        # Custom exceptions
```

### Import Organization
```python
# Standard library
import os
import json
from typing import Dict, List, Optional

# Third-party
import torch
import numpy as np
import networkx as nx

# Local imports
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError
```

## Testing Best Practices

### Test Organization
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test module interactions
- **Performance tests**: Validate speed and memory requirements
- **Edge case tests**: Test boundary conditions and error cases

### Test Naming
```python
def test_cka_no_double_centering():
    """Test CKA uses raw activations without pre-centering."""
    
def test_subspace_tracking_eigenvalue_crossings():
    """Test subspace tracker handles eigenvalue crossings correctly."""
    
def test_fx_poset_extraction_resnet_skip_connections():
    """Test FX extractor detects ResNet skip connections."""
```

### Assertion Strategy
```python
# Be specific about what you're testing
assert torch.allclose(cka_value, 1.0, atol=1e-6), "CKA(X,X) should be 1.0"
assert memory_used < 3000, f"Memory exceeded 3GB: {memory_used}MB"
assert computation_time < 300, f"Analysis too slow: {computation_time}s"
```

## Documentation Requirements

### Docstring Format
```python
def analyze(self, model: nn.Module, data: torch.Tensor) -> Dict:
    """Perform complete neurosheaf analysis.
    
    This function implements the complete pipeline from neural network
    to persistence diagrams using sheaf Laplacians.
    
    Args:
        model: PyTorch neural network model
        data: Input data tensor [batch_size, ...]
        
    Returns:
        Dictionary containing:
            - 'cka_matrix': CKA similarity matrix
            - 'sheaf': Constructed sheaf object
            - 'persistence': Persistence analysis results
            - 'features': Extracted topological features
            
    Raises:
        ValidationError: If model or data format is invalid
        ComputationError: If analysis fails due to numerical issues
        
    Example:
        >>> analyzer = NeurosheafAnalyzer()
        >>> results = analyzer.analyze(model, data)
        >>> cka_matrix = results['cka_matrix']
    """
```

## Performance Optimization Guidelines

### Memory Management
```python
# Use del and garbage collection for large tensors
del large_tensor
torch.cuda.empty_cache()  # Clear GPU memory

# Use context managers for temporary computations
with torch.no_grad():
    # Computation that doesn't need gradients
    pass
```

### GPU Utilization
```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensors efficiently
tensor = tensor.to(device, non_blocking=True)

# Use appropriate data types
tensor = tensor.float()  # or .half() for memory savings
```

## Success Validation Checklist

### Before Committing Code
- [ ] All tests pass locally
- [ ] Code follows type hints and documentation standards
- [ ] Performance requirements met (memory/speed)
- [ ] Edge cases handled gracefully
- [ ] Mathematical properties validated

### Before Moving to Next Phase
- [ ] All phase deliverables completed
- [ ] Integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] Documentation updated
- [ ] Code coverage >95%

### Before Final Release
- [ ] All phases integrated successfully
- [ ] End-to-end tests pass on real neural networks
- [ ] Memory usage <3GB for ResNet50
- [ ] Complete analysis <5 minutes
- [ ] Package installs cleanly via pip

## Emergency Debugging Protocol

### If Tests Fail
1. **Isolate**: Run single failing test with `-v -s` flags
2. **Check plan**: Verify implementation matches specification
3. **Add debugging**: Insert print statements or logging
4. **Validate inputs**: Check tensor shapes and data types
5. **Mathematical verification**: Verify algorithm correctness

### If Performance Issues
1. **Profile**: Use memory profiler to identify bottlenecks
2. **Check data types**: Ensure appropriate precision (float32 vs float64)
3. **GPU utilization**: Verify efficient GPU memory usage
4. **Algorithmic complexity**: Check if implementation matches expected complexity

### If Integration Issues
1. **API compatibility**: Verify function signatures match plan
2. **Data flow**: Check tensor shapes between modules
3. **Error propagation**: Ensure errors are handled and logged
4. **State management**: Verify no unexpected side effects

## Final Notes

- **Trust the plan**: The implementation plans are comprehensive and tested
- **Test early and often**: Catch issues before they compound
- **Document as you go**: Future you will thank present you
- **Performance matters**: Always validate against memory/speed targets
- **Mathematical correctness**: When in doubt, verify against references
- **Ask for help**: If stuck, refer to plan details or testing specifications

Success depends on methodical execution of these guidelines combined with the detailed implementation plans. Follow this framework for each phase to ensure production-quality results.
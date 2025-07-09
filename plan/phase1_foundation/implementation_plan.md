# Phase 1: Foundation Implementation Plan (Weeks 1-2)

## Overview
Establish the development environment, project structure, and baseline performance metrics for the Neurosheaf framework.

## Week 1: Project Setup and Infrastructure

### Day 1-2: Repository and Package Structure
- [ ] Initialize git repository with proper .gitignore
- [ ] Create package structure as defined in docs/neurosheaf_plan.md
- [ ] Set up pyproject.toml with all dependencies
- [ ] Create __init__.py files with proper exports
- [ ] Set up logging infrastructure in utils/logging.py
- [ ] Create custom exception hierarchy in utils/exceptions.py

### Day 3-4: Development Environment
- [ ] Set up pre-commit hooks (black, isort, flake8, mypy)
- [ ] Configure pytest with coverage settings
- [ ] Create Makefile for common tasks
- [ ] Set up development requirements.txt
- [ ] Create CONTRIBUTING.md with coding standards
- [ ] Set up virtual environment documentation

### Day 5: CI/CD Pipeline
- [ ] Configure GitHub Actions workflow for testing
- [ ] Set up matrix testing (Python 3.8-3.11)
- [ ] Configure code coverage reporting (codecov)
- [ ] Set up automatic documentation building
- [ ] Create release workflow for PyPI

## Week 2: Baseline Implementation and Profiling

### Day 6-7: Baseline CKA Implementation
- [ ] Implement naive CKA without optimizations
- [ ] Create memory profiling scripts
- [ ] Document baseline performance metrics
- [ ] Create benchmark datasets (synthetic)
- [ ] Implement basic activation extraction

### Day 8-9: Performance Profiling Tools
- [ ] Set up memory profiler (memory_profiler)
- [ ] Configure GPU profiling (nvidia-ml-py)
- [ ] Create performance benchmark harness
- [ ] Implement timing decorators
- [ ] Set up results tracking system

### Day 10: Baseline Metrics Collection
- [ ] Profile ResNet50 memory usage (target: 1.5TB baseline)
- [ ] Profile computation time for full analysis
- [ ] Document bottlenecks and optimization opportunities
- [ ] Create performance regression tests
- [ ] Generate baseline performance report

## Implementation Details

### Package Structure Setup
```python
# neurosheaf/__init__.py
"""Neurosheaf: Persistent Sheaf Laplacians for Neural Network Similarity"""

__version__ = "0.1.0"

from .api import NeurosheafAnalyzer
from .cka import DebiasedCKA
from .sheaf import SheafBuilder

__all__ = ["NeurosheafAnalyzer", "DebiasedCKA", "SheafBuilder"]
```

### Logging Infrastructure
```python
# neurosheaf/utils/logging.py
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Configure logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        )
        logger.addHandler(file_handler)
    
    return logger
```

### Exception Hierarchy
```python
# neurosheaf/utils/exceptions.py
class NeurosheafError(Exception):
    """Base exception for all Neurosheaf errors."""
    pass

class ValidationError(NeurosheafError):
    """Raised when input validation fails."""
    pass

class ComputationError(NeurosheafError):
    """Raised when numerical computation fails."""
    pass

class MemoryError(NeurosheafError):
    """Raised when memory limits are exceeded."""
    pass

class ArchitectureError(NeurosheafError):
    """Raised when model architecture is unsupported."""
    pass
```

### Performance Profiling
```python
# neurosheaf/utils/profiling.py
import functools
import time
import tracemalloc
from typing import Callable, Tuple, Any
import torch

def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # CPU memory
        tracemalloc.start()
        
        # GPU memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu = torch.cuda.memory_allocated()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if torch.cuda.is_available():
            end_gpu = torch.cuda.memory_allocated()
            peak_gpu = torch.cuda.max_memory_allocated()
        else:
            end_gpu = peak_gpu = 0
        
        print(f"{func.__name__} execution:")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  CPU Memory: {peak / 1024 / 1024:.2f} MB")
        print(f"  GPU Memory: {peak_gpu / 1024 / 1024:.2f} MB")
        
        return result
    return wrapper
```

## Testing Suite

### Test Structure
```
tests/
├── unit/
│   ├── test_logging.py
│   ├── test_exceptions.py
│   └── test_profiling.py
├── integration/
│   ├── test_setup.py
│   └── test_imports.py
└── performance/
    ├── test_baseline_memory.py
    └── test_baseline_speed.py
```

### Unit Tests
```python
# tests/unit/test_logging.py
import pytest
from neurosheaf.utils.logging import setup_logger
import logging

def test_logger_creation():
    """Test logger is created with correct configuration."""
    logger = setup_logger("test", level="DEBUG")
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1

def test_logger_file_output(tmp_path):
    """Test logger writes to file correctly."""
    log_file = tmp_path / "test.log"
    logger = setup_logger("test", log_file=log_file)
    logger.info("Test message")
    
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content
```

### Performance Tests
```python
# tests/performance/test_baseline_memory.py
import pytest
import torch
import numpy as np
from neurosheaf.utils.profiling import profile_memory

class TestBaselineMemory:
    """Test baseline memory usage to ensure optimizations work."""
    
    @pytest.mark.benchmark
    def test_cka_memory_scaling(self):
        """Test memory usage scales linearly with data size."""
        sizes = [100, 500, 1000, 2000]
        memories = []
        
        for n in sizes:
            X = torch.randn(n, 512)
            Y = torch.randn(n, 512)
            
            @profile_memory
            def compute_naive_cka():
                K = X @ X.T
                L = Y @ Y.T
                return K, L
            
            compute_naive_cka()
            # Memory should scale as O(n^2)
        
        # Verify quadratic scaling
        # Implementation details...
```

## Validation Criteria

### Phase 1 Success Metrics
1. **Setup Completeness**
   - All package files created and importable
   - CI/CD pipeline passes all checks
   - Documentation builds successfully

2. **Baseline Performance**
   - Memory profiling shows 1.5TB usage for ResNet50
   - Identified top 3 memory bottlenecks
   - Performance regression tests established

3. **Code Quality**
   - 100% test coverage for utils module
   - All code passes linting checks
   - Type hints pass mypy validation

## Deliverables
1. Fully configured repository with CI/CD
2. Baseline performance report
3. Development environment documentation
4. Initial test suite with >90% coverage
5. Memory profiling tools and results

## Next Phase Dependencies
- Functional package structure for CKA implementation
- Performance baselines for optimization comparison
- Testing infrastructure for continuous validation
# Phase 6: Testing & Documentation Implementation Plan (Weeks 14-15)

## Overview
Comprehensive testing suite, integration testing, performance benchmarks, and complete documentation for production-ready release.

## Week 14: Comprehensive Testing Suite

### Day 1-2: Unit Test Completion
- [ ] Complete unit tests for all modules (target: >95% coverage)
- [ ] Add property-based testing for numerical algorithms
- [ ] Implement parametric tests for different architectures
- [ ] Create edge case tests for boundary conditions
- [ ] Add numerical stability tests

### Day 3-4: Integration Testing
- [ ] End-to-end testing with real neural networks
- [ ] Cross-module integration validation
- [ ] Performance regression testing
- [ ] Memory leak detection tests
- [ ] GPU/CPU compatibility tests

### Day 5: System Testing
- [ ] Large-scale system tests (ResNet50, Transformers)
- [ ] Stress testing with extreme inputs
- [ ] Concurrent usage testing
- [ ] Error recovery and resilience testing
- [ ] Cross-platform compatibility testing

## Week 15: Documentation and Benchmarks

### Day 6-7: API Documentation
- [ ] Complete docstring coverage for all public APIs
- [ ] Generate Sphinx documentation
- [ ] Create API reference guide
- [ ] Add code examples for all functions
- [ ] Document configuration options

### Day 8-9: User Documentation
- [ ] Write comprehensive user guide
- [ ] Create tutorial notebooks
- [ ] Add architecture-specific guides
- [ ] Document troubleshooting procedures
- [ ] Create FAQ section

### Day 10: Performance Benchmarks
- [ ] Benchmark against baseline implementations
- [ ] Create performance comparison charts
- [ ] Document optimization guidelines
- [ ] Add memory usage profiles
- [ ] Create performance regression test suite

## Implementation Details

### Comprehensive Test Suite Structure
```
tests/
├── unit/                           # Unit tests for individual modules
│   ├── test_cka/
│   │   ├── test_debiased_cka.py
│   │   ├── test_nystrom.py
│   │   └── test_validation.py
│   ├── test_sheaf/
│   │   ├── test_poset_extraction.py
│   │   ├── test_restriction_maps.py
│   │   └── test_laplacian_assembly.py
│   ├── test_spectral/
│   │   ├── test_subspace_tracking.py
│   │   ├── test_persistence.py
│   │   └── test_eigenvalue_computation.py
│   └── test_visualization/
│       ├── test_cka_plots.py
│       ├── test_poset_plots.py
│       └── test_dashboard.py
├── integration/                    # Integration tests
│   ├── test_full_pipeline.py
│   ├── test_architectures/
│   │   ├── test_resnet.py
│   │   ├── test_transformer.py
│   │   ├── test_vgg.py
│   │   └── test_custom_models.py
│   └── test_real_world_scenarios.py
├── performance/                    # Performance tests
│   ├── test_memory_usage.py
│   ├── test_computation_speed.py
│   ├── test_scalability.py
│   └── benchmarks/
│       ├── benchmark_cka.py
│       ├── benchmark_sheaf.py
│       └── benchmark_persistence.py
├── system/                         # System-level tests
│   ├── test_large_networks.py
│   ├── test_concurrent_usage.py
│   ├── test_gpu_compatibility.py
│   └── test_error_handling.py
└── fixtures/                       # Test fixtures and utilities
    ├── sample_models.py
    ├── synthetic_data.py
    └── test_utilities.py
```

### Property-Based Testing Framework
```python
# tests/unit/test_cka/test_property_based.py
import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st
from neurosheaf.cka.debiased import DebiasedCKA

class TestCKAProperties:
    """Property-based tests for CKA computation."""
    
    @given(
        n_samples=st.integers(min_value=10, max_value=1000),
        n_features=st.integers(min_value=5, max_value=100),
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_cka_symmetry(self, n_samples, n_features, seed):
        """Test CKA symmetry property: CKA(X, Y) = CKA(Y, X)."""
        torch.manual_seed(seed)
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        cka = DebiasedCKA(use_unbiased=True)
        
        cka_xy = cka.compute(X, Y)
        cka_yx = cka.compute(Y, X)
        
        assert abs(cka_xy - cka_yx) < 1e-6
    
    @given(
        n_samples=st.integers(min_value=10, max_value=500),
        n_features=st.integers(min_value=5, max_value=50),
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_cka_self_similarity(self, n_samples, n_features, seed):
        """Test CKA self-similarity: CKA(X, X) = 1."""
        torch.manual_seed(seed)
        
        X = torch.randn(n_samples, n_features)
        
        cka = DebiasedCKA(use_unbiased=True)
        cka_xx = cka.compute(X, X)
        
        assert abs(cka_xx - 1.0) < 1e-6
    
    @given(
        n_samples=st.integers(min_value=10, max_value=500),
        n_features=st.integers(min_value=5, max_value=50),
        scale=st.floats(min_value=0.1, max_value=10.0),
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_cka_scale_invariance(self, n_samples, n_features, scale, seed):
        """Test CKA scale invariance: CKA(X, Y) = CKA(cX, Y)."""
        torch.manual_seed(seed)
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        cka = DebiasedCKA(use_unbiased=True)
        
        cka_original = cka.compute(X, Y)
        cka_scaled = cka.compute(X * scale, Y)
        
        assert abs(cka_original - cka_scaled) < 1e-5
    
    @given(
        n_samples=st.integers(min_value=10, max_value=500),
        n_features=st.integers(min_value=5, max_value=50),
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_cka_range(self, n_samples, n_features, seed):
        """Test CKA output range: 0 ≤ CKA(X, Y) ≤ 1."""
        torch.manual_seed(seed)
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        cka = DebiasedCKA(use_unbiased=True)
        cka_value = cka.compute(X, Y)
        
        assert 0 <= cka_value <= 1
```

### Integration Test Suite
```python
# tests/integration/test_full_pipeline.py
import pytest
import torch
import torch.nn as nn
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.cka import DebiasedCKA
from neurosheaf.sheaf import SheafBuilder
from neurosheaf.spectral import PersistentSpectralAnalyzer
from neurosheaf.visualization import CKAMatrixVisualizer

class TestFullPipeline:
    """Test complete analysis pipeline."""
    
    def test_end_to_end_resnet_analysis(self):
        """Test full pipeline with ResNet architecture."""
        # Create ResNet-like model
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Residual blocks would go here
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
        
        # Generate data
        x = torch.randn(50, 3, 224, 224)
        
        # Full analysis
        analyzer = NeurosheafAnalyzer(
            cka_samples=1000,
            n_persistence_steps=20,
            use_gpu=False
        )
        
        result = analyzer.analyze(model, x)
        
        # Validate results
        assert 'cka_matrix' in result
        assert 'sheaf' in result
        assert 'persistence' in result
        assert 'features' in result
        
        # Check CKA matrix properties
        cka_matrix = result['cka_matrix']
        assert cka_matrix.shape[0] == cka_matrix.shape[1]
        assert torch.all(torch.diag(cka_matrix) > 0.99)  # Diagonal should be ~1
        assert torch.all(cka_matrix >= 0) and torch.all(cka_matrix <= 1)
        
        # Check sheaf properties
        sheaf = result['sheaf']
        assert sheaf.validate()
        assert len(sheaf.stalks) > 0
        assert len(sheaf.restrictions) > 0
        
        # Check persistence results
        persistence = result['persistence']
        assert 'eigenvalue_sequences' in persistence
        assert 'tracking_info' in persistence
        assert 'diagrams' in persistence
        
    def test_transformer_analysis(self):
        """Test analysis with Transformer architecture."""
        # Create simple Transformer-like model
        class SimpleTransformer(nn.Module):
            def __init__(self, d_model=512, nhead=8, num_layers=6):
                super().__init__()
                self.embedding = nn.Linear(100, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=nhead,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, 
                    num_layers=num_layers
                )
                
                self.classifier = nn.Linear(d_model, 10)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = x.mean(dim=1)  # Global average pooling
                return self.classifier(x)
        
        model = SimpleTransformer()
        x = torch.randn(30, 20, 100)  # [batch, seq_len, input_dim]
        
        analyzer = NeurosheafAnalyzer(
            cka_samples=500,
            n_persistence_steps=15
        )
        
        result = analyzer.analyze(model, x)
        
        # Should complete without errors
        assert 'cka_matrix' in result
        assert 'persistence' in result
        
    def test_custom_architecture_analysis(self):
        """Test analysis with custom architecture."""
        class CustomNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch1 = nn.Sequential(
                    nn.Linear(100, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32)
                )
                
                self.branch2 = nn.Sequential(
                    nn.Linear(100, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32)
                )
                
                self.merger = nn.Linear(64, 10)
                
            def forward(self, x):
                b1 = self.branch1(x)
                b2 = self.branch2(x)
                merged = torch.cat([b1, b2], dim=1)
                return self.merger(merged)
        
        model = CustomNet()
        x = torch.randn(100, 100)
        
        analyzer = NeurosheafAnalyzer()
        result = analyzer.analyze(model, x)
        
        # Should handle parallel branches correctly
        assert 'cka_matrix' in result
        assert result['sheaf'].validate()
        
    def test_memory_efficient_analysis(self):
        """Test analysis with memory constraints."""
        # Create larger model
        layers = []
        for i in range(20):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        
        model = nn.Sequential(*layers)
        x = torch.randn(1000, 256)
        
        # Configure for memory efficiency
        analyzer = NeurosheafAnalyzer(
            cka_samples=200,  # Reduced samples
            use_nystrom=True,
            nystrom_landmarks=100,
            memory_limit_gb=2.0
        )
        
        result = analyzer.analyze(model, x)
        
        # Should complete within memory constraints
        assert 'cka_matrix' in result
        assert 'persistence' in result
```

### Performance Benchmarking Suite
```python
# tests/performance/benchmarks/benchmark_full_system.py
import pytest
import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.cka import DebiasedCKA
from neurosheaf.sheaf import SheafBuilder
from neurosheaf.spectral import PersistentSpectralAnalyzer

class TestSystemBenchmarks:
    """System-level performance benchmarks."""
    
    @pytest.mark.benchmark
    def test_resnet50_full_analysis_benchmark(self):
        """Benchmark full analysis on ResNet50-scale model."""
        # Create ResNet50-like model
        model = self._create_resnet50_model()
        x = torch.randn(100, 3, 224, 224)
        
        analyzer = NeurosheafAnalyzer(
            cka_samples=2000,
            n_persistence_steps=30
        )
        
        # Measure performance
        process = psutil.Process()
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Time analysis
        start_time = time.time()
        result = analyzer.analyze(model, x)
        analysis_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_used = mem_after - mem_before
        
        # Performance assertions
        assert analysis_time < 300  # Less than 5 minutes
        assert memory_used < 3000   # Less than 3GB
        
        # Quality assertions
        assert 'cka_matrix' in result
        assert result['sheaf'].validate()
        
        print(f"ResNet50 Analysis - Time: {analysis_time:.1f}s, Memory: {memory_used:.1f}MB")
    
    def _create_resnet50_model(self):
        """Create ResNet50-like model for benchmarking."""
        class BasicBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU()
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm2d(out_channels)
                    )
                    
            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = self.relu(out)
                return out
        
        # Simplified ResNet50
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            
            # Layer 1
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            
            # Layer 2
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            
            # Layer 3
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            
            # Layer 4
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
        
        return model
    
    @pytest.mark.benchmark
    def test_scaling_analysis(self):
        """Test performance scaling with network size."""
        sizes = [5, 10, 20, 30]
        times = []
        memories = []
        
        for n_layers in sizes:
            # Create model with n_layers
            layers = []
            for i in range(n_layers):
                layers.append(nn.Linear(128, 128))
                layers.append(nn.ReLU())
            
            model = nn.Sequential(*layers)
            x = torch.randn(200, 128)
            
            analyzer = NeurosheafAnalyzer(
                cka_samples=500,
                n_persistence_steps=10
            )
            
            # Measure
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            result = analyzer.analyze(model, x)
            elapsed = time.time() - start_time
            
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            
            times.append(elapsed)
            memories.append(memory_used)
            
            print(f"Layers: {n_layers}, Time: {elapsed:.1f}s, Memory: {memory_used:.1f}MB")
        
        # Check scaling is reasonable (not exponential)
        for i in range(len(sizes) - 1):
            size_ratio = sizes[i+1] / sizes[i]
            time_ratio = times[i+1] / times[i]
            
            # Time should scale less than quadratically
            assert time_ratio < size_ratio ** 2.5
```

### Documentation Generation System
```python
# docs/generate_docs.py
"""
Automated documentation generation system.
"""
import os
import subprocess
import shutil
from pathlib import Path

def generate_api_docs():
    """Generate API documentation using Sphinx."""
    print("Generating API documentation...")
    
    # Ensure docs directory exists
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Generate API docs
    subprocess.run([
        "sphinx-apidoc", 
        "-f", "-o", "docs/api", 
        "neurosheaf/", 
        "neurosheaf/tests/"
    ])
    
    # Build HTML documentation
    subprocess.run(["sphinx-build", "-b", "html", "docs/", "docs/_build/html"])
    
    print("API documentation generated in docs/_build/html/")

def generate_examples():
    """Generate example notebooks."""
    print("Generating example notebooks...")
    
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Create example notebooks
    notebooks = [
        "01_basic_usage.ipynb",
        "02_cka_analysis.ipynb", 
        "03_sheaf_construction.ipynb",
        "04_persistence_analysis.ipynb",
        "05_visualization.ipynb",
        "06_advanced_features.ipynb"
    ]
    
    for notebook in notebooks:
        create_example_notebook(examples_dir / notebook)
    
    print(f"Generated {len(notebooks)} example notebooks")

def create_example_notebook(notebook_path):
    """Create a template example notebook."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {notebook_path.stem.replace('_', ' ').title()}\n",
                    "\n",
                    "This notebook demonstrates the usage of Neurosheaf for neural network analysis.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "from neurosheaf.api import NeurosheafAnalyzer\n",
                    "\n",
                    "# Your example code here\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)

def generate_performance_report():
    """Generate performance benchmark report."""
    print("Generating performance report...")
    
    # Run benchmarks and collect results
    subprocess.run([
        "pytest", 
        "tests/performance/benchmarks/", 
        "--benchmark-json=benchmark_results.json"
    ])
    
    # Generate report
    report_content = """
# Neurosheaf Performance Report

## System Specifications
- Python Version: 3.8+
- PyTorch Version: 2.0+
- Hardware: [To be filled during testing]

## Benchmark Results

### Memory Usage
- ResNet50 Analysis: <3GB
- Transformer Analysis: <2GB
- Large Network (50 layers): <4GB

### Computation Time
- ResNet50 Analysis: <5 minutes
- CKA Matrix (50x50): <30 seconds
- Persistence Analysis: <2 minutes

### Scalability
- Linear scaling with network depth
- Quadratic scaling with CKA matrix size
- Efficient sparse operations for large Laplacians

## Optimization Recommendations
1. Use adaptive sampling for large datasets
2. Enable GPU acceleration when available
3. Consider Nyström approximation for memory-constrained environments
"""
    
    with open("docs/performance_report.md", "w") as f:
        f.write(report_content)

if __name__ == "__main__":
    generate_api_docs()
    generate_examples()
    generate_performance_report()
    print("Documentation generation complete!")
```

## Testing Strategy

### Continuous Integration Pipeline
```yaml
# .github/workflows/comprehensive_testing.yml
name: Comprehensive Testing

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=neurosheaf --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v -m "not slow"
    
    - name: Run system tests
      run: |
        pytest tests/system/ -v --maxfail=5
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  performance:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run performance benchmarks
      run: |
        pytest tests/performance/benchmarks/ -v --benchmark-json=benchmark.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
```

### Test Execution Commands
```bash
# Complete test suite
make test-all

# Unit tests only
make test-unit

# Integration tests
make test-integration

# Performance benchmarks
make test-performance

# Coverage report
make test-coverage

# Documentation tests
make test-docs
```

## Success Criteria

1. **Test Coverage**: >95% code coverage across all modules
2. **Performance**: All benchmarks pass within specified limits
3. **Documentation**: Complete API docs and user guides
4. **Integration**: All real-world architectures work correctly
5. **Reliability**: All tests pass consistently across platforms

## Phase 6 Deliverables

1. **Comprehensive Test Suite**
   - Unit tests for all modules
   - Integration tests with real architectures
   - Performance benchmarks
   - System tests for edge cases

2. **Documentation System**
   - Complete API documentation
   - User guides and tutorials
   - Example notebooks
   - Performance reports

3. **Quality Assurance**
   - Continuous integration pipeline
   - Code coverage reports
   - Performance regression detection
   - Cross-platform compatibility

4. **Production Readiness**
   - Error handling and logging
   - Input validation
   - Resource management
   - Monitoring and diagnostics
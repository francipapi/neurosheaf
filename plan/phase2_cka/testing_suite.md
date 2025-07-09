# Phase 2: CKA Testing Suite

## Critical Tests Overview

The most critical aspect of Phase 2 is ensuring NO double-centering occurs in the Debiased CKA implementation. This testing suite emphasizes validation of this requirement.

## Test Categories

### 1. Correctness Tests (CRITICAL)
```python
# tests/phase2_cka/critical/test_no_double_centering.py
import pytest
import torch
import numpy as np
from neurosheaf.cka.debiased import DebiasedCKA
from unittest.mock import patch

class TestNoDoubleCenteringCritical:
    """CRITICAL: Exhaustive tests to ensure NO double-centering."""
    
    def test_kernel_computation_uses_raw_data(self):
        """Verify kernels are computed from raw, uncentered activations."""
        # Create data with obvious non-zero mean
        X = torch.ones(50, 30) * 10 + torch.randn(50, 30) * 0.1
        Y = torch.ones(50, 30) * 5 + torch.randn(50, 30) * 0.1
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Capture intermediate computations
        captured_data = {}
        
        # Patch the matrix multiplication to capture inputs
        original_mm = torch.matmul
        def capturing_mm(a, b):
            if a.shape[1] == 30 and b.shape[0] == 30:  # Our X @ X.T operation
                captured_data['X_used'] = a.clone()
                captured_data['X_T_used'] = b.clone()
            return original_mm(a, b)
        
        with patch('torch.matmul', capturing_mm):
            cka_value = cka.compute(X, Y)
        
        # Verify the data used was NOT centered
        assert torch.allclose(captured_data['X_used'], X)
        assert torch.mean(captured_data['X_used']) > 9.0  # Should be ~10
    
    def test_biased_vs_unbiased_difference(self):
        """Ensure biased and unbiased estimators give different results."""
        X = torch.randn(100, 50) + 3.0
        Y = torch.randn(100, 40) + 2.0
        
        cka_unbiased = DebiasedCKA(use_unbiased=True)
        cka_biased = DebiasedCKA(use_unbiased=False)
        
        value_unbiased = cka_unbiased.compute(X, Y)
        value_biased = cka_biased.compute(X, Y)
        
        # Should be different
        assert abs(value_unbiased - value_biased) > 0.001
    
    def test_mean_shift_invariance(self):
        """CKA should be invariant to mean shifts when computed correctly."""
        X = torch.randn(100, 50)
        Y = torch.randn(100, 40)
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Original
        cka1 = cka.compute(X, Y)
        
        # Shifted
        X_shifted = X + 100
        Y_shifted = Y + 200
        cka2 = cka.compute(X_shifted, Y_shifted)
        
        # Should be approximately equal (within numerical precision)
        assert abs(cka1 - cka2) < 1e-5
```

### 2. Edge Cases and Numerical Stability
```python
# tests/phase2_cka/edge_cases/test_numerical_stability.py
import pytest
import torch
import numpy as np
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.utils.exceptions import ValidationError, ComputationError

class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_minimum_samples_unbiased(self):
        """Unbiased HSIC requires at least 4 samples."""
        cka = DebiasedCKA(use_unbiased=True)
        
        # Should fail with < 4 samples
        X = torch.randn(3, 10)
        Y = torch.randn(3, 10)
        
        with pytest.raises(ValidationError, match="at least 4 samples"):
            cka.compute(X, Y)
        
        # Should work with exactly 4
        X = torch.randn(4, 10)
        Y = torch.randn(4, 10)
        result = cka.compute(X, Y)
        assert 0 <= result <= 1
    
    def test_identical_inputs(self):
        """CKA(X, X) should be 1."""
        X = torch.randn(100, 50)
        cka = DebiasedCKA()
        
        result = cka.compute(X, X)
        assert abs(result - 1.0) < 1e-6
    
    def test_orthogonal_inputs(self):
        """CKA for orthogonal features should be near 0."""
        n = 100
        # Create orthogonal data
        X = torch.randn(n, 50)
        Q, _ = torch.linalg.qr(torch.randn(50, 50))
        Y = X @ Q  # Rotate to orthogonal space
        
        cka = DebiasedCKA()
        result = cka.compute(X, Y)
        
        # Should be close to 0 but not exactly due to finite samples
        assert result < 0.1
    
    def test_constant_features(self):
        """Handle constant features gracefully."""
        X = torch.ones(100, 50)  # All constant
        Y = torch.randn(100, 40)
        
        cka = DebiasedCKA()
        
        # Should not crash, but result is degenerate
        result = cka.compute(X, Y)
        assert 0 <= result <= 1
    
    def test_near_singular_kernels(self):
        """Handle near-singular kernel matrices."""
        # Create low-rank data
        n_samples = 100
        rank = 5
        
        U = torch.randn(n_samples, rank)
        X = U @ torch.randn(rank, 50)
        Y = U @ torch.randn(rank, 40)
        
        cka = DebiasedCKA()
        result = cka.compute(X, Y)
        
        # Should handle gracefully
        assert 0 <= result <= 1
        assert not torch.isnan(torch.tensor(result))
```

### 3. Memory Efficiency Tests
```python
# tests/phase2_cka/memory/test_memory_limits.py
import pytest
import torch
import gc
import psutil
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.cka.sampling import AdaptiveSampler
from neurosheaf.cka.nystrom import NystromCKA

class TestMemoryLimits:
    """Test memory-efficient implementations under constraints."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Aggressive cleanup between tests."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @pytest.mark.memory_intensive
    def test_adaptive_sampling_under_memory_pressure(self):
        """Test adaptive sampling respects memory limits."""
        sampler = AdaptiveSampler(min_samples=512, max_samples=4096)
        
        # Test with different memory limits
        test_cases = [
            (10000, 512, 50),    # 50MB limit
            (10000, 512, 200),   # 200MB limit
            (10000, 512, 1000),  # 1GB limit
        ]
        
        for n_total, n_features, memory_mb in test_cases:
            selected = sampler.determine_sample_size(
                n_total=n_total,
                n_features=n_features,
                available_memory_mb=memory_mb
            )
            
            # Verify memory requirement
            kernel_size_mb = (selected ** 2 * 4) / 1024 / 1024
            assert kernel_size_mb <= memory_mb * 1.1  # 10% tolerance
    
    @pytest.mark.slow
    def test_large_scale_nystrom(self):
        """Test Nyström on large-scale data."""
        # This would OOM with regular CKA
        n_samples = 50000
        n_features = 1024
        n_landmarks = 512
        
        # Create data in chunks to avoid OOM during creation
        chunk_size = 5000
        chunks_x = []
        chunks_y = []
        
        for i in range(0, n_samples, chunk_size):
            size = min(chunk_size, n_samples - i)
            chunks_x.append(torch.randn(size, n_features))
            chunks_y.append(torch.randn(size, n_features))
        
        X = torch.cat(chunks_x, dim=0)
        Y = torch.cat(chunks_y, dim=0)
        
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Compute with Nyström
        nystrom = NystromCKA(n_landmarks=n_landmarks)
        result = nystrom.compute(X, Y)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_used = mem_after - mem_before
        
        # Should use much less than full kernel would require
        full_kernel_mb = (n_samples ** 2 * 4 * 2) / 1024 / 1024  # ~20GB
        assert memory_used < 1000  # Should use < 1GB
        assert 0 <= result <= 1
```

### 4. Sampling Strategy Tests
```python
# tests/phase2_cka/sampling/test_sampling_strategies.py
import pytest
import torch
from neurosheaf.cka.sampling import AdaptiveSampler

class TestSamplingStrategies:
    """Test various sampling strategies for CKA."""
    
    def test_stratified_sampling_balance(self):
        """Test stratified sampling maintains class balance."""
        n_total = 1000
        n_classes = 5
        n_samples = 200
        
        # Create imbalanced labels
        labels = torch.cat([
            torch.full((200,), 0),
            torch.full((300,), 1),
            torch.full((100,), 2),
            torch.full((250,), 3),
            torch.full((150,), 4),
        ])
        
        sampler = AdaptiveSampler()
        indices = sampler.stratified_sample(n_total, n_samples, labels)
        
        # Check balance
        sampled_labels = labels[indices]
        for class_id in range(n_classes):
            class_count = (sampled_labels == class_id).sum()
            expected = n_samples // n_classes
            assert abs(class_count - expected) <= 2
    
    def test_sampling_determinism(self):
        """Test sampling can be made deterministic."""
        sampler = AdaptiveSampler()
        
        torch.manual_seed(42)
        indices1 = sampler.stratified_sample(1000, 100)
        
        torch.manual_seed(42)
        indices2 = sampler.stratified_sample(1000, 100)
        
        assert torch.all(indices1 == indices2)
    
    def test_adaptive_size_selection(self):
        """Test adaptive sample size selection logic."""
        sampler = AdaptiveSampler(
            min_samples=100,
            max_samples=1000
        )
        
        # Small data - use all
        size = sampler.determine_sample_size(
            n_total=500,
            n_features=100,
            available_memory_mb=1000
        )
        assert size == 500
        
        # Large data - respect max_samples
        size = sampler.determine_sample_size(
            n_total=10000,
            n_features=100,
            available_memory_mb=1000
        )
        assert size == 1000
        
        # Memory constrained
        size = sampler.determine_sample_size(
            n_total=10000,
            n_features=100,
            available_memory_mb=10  # Very limited
        )
        assert size < 500
```

### 5. Integration and Pipeline Tests
```python
# tests/phase2_cka/integration/test_full_pipeline.py
import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path
from neurosheaf.cka import DebiasedCKA
from neurosheaf.cka.pairwise import PairwiseCKA

class TestFullPipeline:
    """Test complete CKA computation pipeline."""
    
    def test_pairwise_matrix_properties(self):
        """Test pairwise CKA matrix has correct properties."""
        # Create diverse activations
        n_samples = 500
        activations = {
            'conv1': torch.randn(n_samples, 64),
            'conv2': torch.randn(n_samples, 128),
            'fc1': torch.randn(n_samples, 256),
            'fc2': torch.randn(n_samples, 128),
            'output': torch.randn(n_samples, 10)
        }
        
        cka = DebiasedCKA(use_unbiased=True)
        pairwise = PairwiseCKA(cka)
        
        matrix = pairwise.compute_matrix(activations)
        
        # Test properties
        n_layers = len(activations)
        assert matrix.shape == (n_layers, n_layers)
        
        # Symmetry
        assert torch.allclose(matrix, matrix.T, atol=1e-6)
        
        # Diagonal should be 1
        assert torch.all(torch.abs(torch.diag(matrix) - 1.0) < 1e-6)
        
        # Range [0, 1]
        assert torch.all(matrix >= 0)
        assert torch.all(matrix <= 1)
        
        # Positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(matrix)
        assert torch.all(eigenvalues >= -1e-6)
    
    def test_incremental_computation(self):
        """Test computing CKA for new layers incrementally."""
        n_samples = 300
        
        # Initial layers
        activations = {
            'layer1': torch.randn(n_samples, 100),
            'layer2': torch.randn(n_samples, 100)
        }
        
        cka = DebiasedCKA()
        pairwise = PairwiseCKA(cka)
        
        # Compute initial matrix
        matrix1 = pairwise.compute_matrix(activations)
        
        # Add new layer
        activations['layer3'] = torch.randn(n_samples, 100)
        
        # Compute extended matrix
        matrix2 = pairwise.compute_matrix(activations)
        
        # Original values should be unchanged
        assert torch.allclose(matrix1, matrix2[:2, :2])
        
        # New entries should be computed
        assert matrix2.shape == (3, 3)
    
    @pytest.mark.slow
    def test_checkpoint_recovery(self, tmp_path):
        """Test recovery from interrupted computation."""
        n_layers = 20
        n_samples = 200
        
        activations = {
            f'layer_{i}': torch.randn(n_samples, 64)
            for i in range(n_layers)
        }
        
        cka = DebiasedCKA()
        pairwise = PairwiseCKA(
            cka,
            checkpoint_dir=str(tmp_path)
        )
        
        # Simulate crash after 50 computations
        computed_pairs = []
        
        def tracking_compute(X, Y, **kwargs):
            computed_pairs.append(len(computed_pairs))
            if len(computed_pairs) == 50:
                raise RuntimeError("Simulated crash")
            return cka.compute(X, Y, **kwargs)
        
        # Patch compute method
        pairwise.cka_computer.compute = tracking_compute
        
        with pytest.raises(RuntimeError):
            pairwise.compute_matrix(activations)
        
        # Restore original compute
        pairwise.cka_computer = DebiasedCKA()
        
        # Resume should skip already computed pairs
        computed_pairs.clear()
        matrix = pairwise.compute_matrix(activations)
        
        # Should have computed remaining pairs only
        total_pairs = n_layers * (n_layers + 1) // 2
        assert len(computed_pairs) == total_pairs - 50
```

### 6. Performance Benchmarks
```python
# tests/phase2_cka/benchmarks/test_performance.py
import pytest
import torch
import time
import numpy as np
from neurosheaf.cka import DebiasedCKA
from neurosheaf.cka.nystrom import NystromCKA

class TestPerformanceBenchmarks:
    """Benchmark CKA implementations."""
    
    @pytest.mark.benchmark
    def test_cka_speed_scaling(self, benchmark):
        """Test CKA computation speed scaling."""
        cka = DebiasedCKA(use_unbiased=True)
        
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for n in sizes:
            X = torch.randn(n, 256)
            Y = torch.randn(n, 256)
            
            start = time.time()
            cka.compute(X, Y)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Should scale roughly as O(n^2)
        for i in range(len(sizes) - 1):
            ratio = sizes[i+1] / sizes[i]
            time_ratio = times[i+1] / times[i]
            # Allow some deviation from perfect quadratic
            assert time_ratio < ratio ** 2.5
    
    @pytest.mark.benchmark
    def test_nystrom_vs_exact_speed(self):
        """Compare Nyström vs exact CKA speed."""
        n_samples = 5000
        n_features = 512
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        # Exact CKA
        cka_exact = DebiasedCKA()
        start = time.time()
        exact_result = cka_exact.compute(X, Y)
        exact_time = time.time() - start
        
        # Nyström CKA
        nystrom = NystromCKA(n_landmarks=256)
        start = time.time()
        nystrom_result = nystrom.compute(X, Y)
        nystrom_time = time.time() - start
        
        # Nyström should be much faster
        assert nystrom_time < exact_time * 0.1
        
        # But still accurate
        assert abs(exact_result - nystrom_result) < 0.05
```

## Test Execution Strategy

### Continuous Integration Tests
```yaml
# .github/workflows/phase2_tests.yml
name: Phase 2 CKA Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run critical tests
      run: |
        pytest tests/phase2_cka/critical -v --tb=short
    
    - name: Run unit tests
      run: |
        pytest tests/phase2_cka/unit -v
    
    - name: Run edge case tests
      run: |
        pytest tests/phase2_cka/edge_cases -v
    
    - name: Run integration tests (excluding slow)
      run: |
        pytest tests/phase2_cka/integration -v -m "not slow"
```

### Local Development Testing
```bash
# Quick tests during development
make test-phase2-quick

# Full test suite
make test-phase2-full

# Memory profiling tests
make test-phase2-memory

# Benchmarks
make test-phase2-benchmark
```

## Success Metrics

1. **Test Coverage**: >98% for all CKA modules
2. **Critical Tests**: 100% pass rate for no-double-centering tests
3. **Memory Tests**: Pass with <3GB memory for ResNet50 scale
4. **Performance**: Meet or exceed baseline targets
5. **Integration**: All pipeline tests pass

## Common Issues and Debugging

### Issue: Numerical differences in CKA values
```python
# Debug helper
def debug_cka_computation(X, Y):
    """Print intermediate values for debugging."""
    print(f"X shape: {X.shape}, mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"Y shape: {Y.shape}, mean: {Y.mean():.4f}, std: {Y.std():.4f}")
    
    K = X @ X.T
    L = Y @ Y.T
    
    print(f"K diagonal mean: {torch.diag(K).mean():.4f}")
    print(f"L diagonal mean: {torch.diag(L).mean():.4f}")
    print(f"K off-diagonal mean: {(K - torch.diag(torch.diag(K))).mean():.4f}")
    
    # Continue with computation...
```

### Issue: Memory leaks in tests
```python
# Memory leak detector
import tracemalloc

def detect_memory_leak(func, iterations=5):
    """Detect memory leaks in function."""
    tracemalloc.start()
    
    snapshots = []
    for i in range(iterations):
        func()
        snapshot = tracemalloc.take_snapshot()
        snapshots.append(snapshot)
    
    # Compare first and last
    diff = snapshots[-1].compare_to(snapshots[0], 'lineno')
    
    for stat in diff[:10]:
        print(stat)
```
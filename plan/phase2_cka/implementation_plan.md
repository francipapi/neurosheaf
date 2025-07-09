# Phase 2: CKA Implementation Plan (Weeks 3-4)

## Overview
Implement the Debiased CKA module with correct handling (no double-centering), adaptive sampling, and Nyström approximation for memory efficiency.

## Week 3: Core CKA Implementation

### Day 1-2: Debiased CKA Foundation
**Critical**: Reference docs/updated-debiased-cka-v3.md - "Fixed Double-Centering in Debiased CKA"
- [ ] Implement basic CKA computation WITHOUT centering
- [ ] Create DebiasedCKA class in neurosheaf/cka/debiased.py
- [ ] Implement unbiased HSIC estimator
- [ ] Add input validation for activation tensors
- [ ] Create unit tests for correctness

### Day 3-4: Adaptive Sampling Strategy
- [ ] Implement adaptive sampling (512-4096 samples)
- [ ] Create sample size determination logic
- [ ] Implement stratified sampling for balanced representation
- [ ] Add memory-aware batch processing
- [ ] Create performance benchmarks

### Day 5: CKA Matrix Computation
- [ ] Implement efficient pairwise CKA computation
- [ ] Add progress tracking for long computations
- [ ] Implement checkpoint/resume functionality
- [ ] Create parallel computation support
- [ ] Add memory usage monitoring

## Week 4: Memory Optimization and Nyström

### Day 6-7: Nyström Approximation
**Reference**: docs/updated-debiased-cka-v3.md - "Nyström approximation for large-scale"
- [ ] Implement Nyström method for kernel approximation
- [ ] Create landmark selection strategies
- [ ] Add approximation quality metrics
- [ ] Implement adaptive rank selection
- [ ] Create accuracy vs memory trade-off analysis

### Day 8-9: Memory-Efficient Operations
- [ ] Implement chunked matrix operations
- [ ] Add GPU memory pooling
- [ ] Create memory-mapped file support for large matrices
- [ ] Implement sparse representation where applicable
- [ ] Add automatic precision reduction (fp16 when safe)

### Day 10: Integration and Validation
- [ ] Integrate CKA module with main API
- [ ] Validate against reference implementations
- [ ] Create comprehensive benchmarks
- [ ] Document API and usage examples
- [ ] Prepare for sheaf construction integration

## Implementation Details

### Debiased CKA (CRITICAL - No Double Centering!)
```python
# neurosheaf/cka/debiased.py
import torch
import numpy as np
from typing import Optional, Tuple, Union
from ..utils.validation import validate_activations
from ..utils.exceptions import ValidationError, ComputationError

class DebiasedCKA:
    """Debiased CKA implementation WITHOUT double-centering.
    
    CRITICAL: The unbiased HSIC estimator already performs centering
    internally. Pre-centering the data leads to incorrect results.
    """
    
    def __init__(
        self,
        use_unbiased: bool = True,
        eps: float = 1e-8
    ):
        self.use_unbiased = use_unbiased
        self.eps = eps
    
    def compute(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None
    ) -> float:
        """Compute CKA between activation tensors.
        
        Args:
            X: Activation tensor [n_samples, n_features_x]
            Y: Activation tensor [n_samples, n_features_y]
            sample_indices: Optional subset of samples to use
            
        Returns:
            CKA similarity value in [0, 1]
            
        IMPORTANT: X and Y should be raw activations, NOT centered!
        """
        # Validate inputs
        X, Y = validate_activations(X, Y)
        
        if sample_indices is not None:
            X = X[sample_indices]
            Y = Y[sample_indices]
        
        # Compute kernels WITHOUT centering
        K = X @ X.T  # Raw gram matrix
        L = Y @ Y.T  # Raw gram matrix
        
        if self.use_unbiased:
            hsic_xy = self._unbiased_hsic(K, L)
            hsic_xx = self._unbiased_hsic(K, K)
            hsic_yy = self._unbiased_hsic(L, L)
        else:
            hsic_xy = self._biased_hsic(K, L)
            hsic_xx = self._biased_hsic(K, K)
            hsic_yy = self._biased_hsic(L, L)
        
        # Compute CKA
        denominator = torch.sqrt(hsic_xx * hsic_yy) + self.eps
        cka = hsic_xy / denominator
        
        return float(torch.clamp(cka, 0, 1))
    
    def _unbiased_hsic(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Unbiased HSIC estimator (handles centering internally)."""
        n = K.shape[0]
        
        if n < 4:
            raise ValidationError("Unbiased HSIC requires at least 4 samples")
        
        # Fill diagonal with zeros
        K_0 = K - torch.diag(torch.diag(K))
        L_0 = L - torch.diag(torch.diag(L))
        
        # Compute terms for unbiased estimator
        term1 = torch.sum(K_0 * L_0)
        term2 = torch.sum(K_0) * torch.sum(L_0) / (n - 1) / (n - 2)
        term3 = 2 * torch.sum(K_0, dim=0) @ torch.sum(L_0, dim=0) / (n - 2)
        
        hsic = (term1 + term2 - term3) / (n * (n - 3))
        
        return hsic
    
    def _biased_hsic(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Biased HSIC estimator for comparison."""
        n = K.shape[0]
        
        # Center the kernels
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        K_c = H @ K @ H
        L_c = H @ L @ H
        
        # Compute HSIC
        hsic = torch.sum(K_c * L_c) / (n - 1) ** 2
        
        return hsic
```

### Adaptive Sampling Implementation
```python
# neurosheaf/cka/sampling.py
import torch
import numpy as np
from typing import Tuple, Optional
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class AdaptiveSampler:
    """Adaptive sampling strategy for CKA computation."""
    
    def __init__(
        self,
        min_samples: int = 512,
        max_samples: int = 4096,
        target_variance: float = 0.01
    ):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.target_variance = target_variance
    
    def determine_sample_size(
        self,
        n_total: int,
        n_features: int,
        available_memory_mb: float
    ) -> int:
        """Determine optimal sample size based on constraints."""
        # Memory requirement: O(n^2) for kernel matrices
        bytes_per_element = 4  # float32
        kernel_memory_mb = (n_total ** 2) * bytes_per_element / 1024 / 1024
        
        if kernel_memory_mb <= available_memory_mb:
            # Can use all samples
            return n_total
        
        # Binary search for largest feasible sample size
        left, right = self.min_samples, min(self.max_samples, n_total)
        
        while left < right:
            mid = (left + right + 1) // 2
            required_mb = (mid ** 2) * bytes_per_element / 1024 / 1024
            
            if required_mb <= available_memory_mb:
                left = mid
            else:
                right = mid - 1
        
        logger.info(f"Selected sample size: {left} (from {n_total} total)")
        return left
    
    def stratified_sample(
        self,
        n_total: int,
        n_samples: int,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Create stratified sample indices."""
        if labels is None:
            # Random sampling
            perm = torch.randperm(n_total)
            return perm[:n_samples]
        
        # Stratified sampling
        unique_labels = torch.unique(labels)
        samples_per_class = n_samples // len(unique_labels)
        indices = []
        
        for label in unique_labels:
            class_indices = torch.where(labels == label)[0]
            n_class = min(samples_per_class, len(class_indices))
            sampled = class_indices[torch.randperm(len(class_indices))[:n_class]]
            indices.append(sampled)
        
        return torch.cat(indices)
```

### Nyström Approximation
```python
# neurosheaf/cka/nystrom.py
import torch
import numpy as np
from typing import Tuple, Optional
from ..utils.exceptions import ComputationError

class NystromCKA:
    """Memory-efficient CKA using Nyström approximation."""
    
    def __init__(
        self,
        n_landmarks: int = 256,
        landmark_selection: str = 'uniform'
    ):
        self.n_landmarks = n_landmarks
        self.landmark_selection = landmark_selection
    
    def compute(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> float:
        """Compute CKA using Nyström approximation.
        
        Memory complexity: O(n*m) instead of O(n^2)
        where m is number of landmarks.
        """
        n_samples = X.shape[0]
        
        # Select landmarks
        if self.landmark_selection == 'uniform':
            landmarks = torch.randperm(n_samples)[:self.n_landmarks]
        elif self.landmark_selection == 'kmeans':
            landmarks = self._kmeans_landmarks(X, self.n_landmarks)
        else:
            raise ValueError(f"Unknown selection: {self.landmark_selection}")
        
        # Compute landmark kernels
        X_land = X[landmarks]
        Y_land = Y[landmarks]
        
        # Compute blocks: K = [K_mm, K_mn; K_nm, K_nn]
        K_mm = X_land @ X_land.T
        K_mn = X_land @ X.T
        L_mm = Y_land @ Y_land.T
        L_mn = Y_land @ Y.T
        
        # Nyström approximation
        K_mm_inv = self._stable_inverse(K_mm)
        L_mm_inv = self._stable_inverse(L_mm)
        
        # Approximate full kernels
        K_approx = K_mn.T @ K_mm_inv @ K_mn
        L_approx = L_mn.T @ L_mm_inv @ L_mn
        
        # Compute CKA on approximated kernels
        hsic_xy = self._compute_hsic_approx(K_approx, L_approx, K_mn, L_mn)
        hsic_xx = self._compute_hsic_approx(K_approx, K_approx, K_mn, K_mn)
        hsic_yy = self._compute_hsic_approx(L_approx, L_approx, L_mn, L_mn)
        
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)
        
        return float(torch.clamp(cka, 0, 1))
    
    def _stable_inverse(self, M: torch.Tensor) -> torch.Tensor:
        """Compute stable inverse using SVD."""
        U, S, Vt = torch.linalg.svd(M)
        S_inv = torch.where(S > 1e-6, 1.0 / S, 0.0)
        return Vt.T @ torch.diag(S_inv) @ U.T
    
    def _kmeans_landmarks(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """Select landmarks using K-means++."""
        # Implementation of K-means++ initialization
        # ...
        pass
```

### Memory-Efficient Pairwise CKA
```python
# neurosheaf/cka/pairwise.py
import torch
import numpy as np
from typing import Dict, List, Optional, Callable
from tqdm import tqdm
from ..utils.memory import MemoryMonitor
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PairwiseCKA:
    """Compute pairwise CKA matrix efficiently."""
    
    def __init__(
        self,
        cka_computer: DebiasedCKA,
        memory_limit_mb: float = 1024,
        checkpoint_dir: Optional[str] = None
    ):
        self.cka_computer = cka_computer
        self.memory_limit_mb = memory_limit_mb
        self.checkpoint_dir = checkpoint_dir
        self.memory_monitor = MemoryMonitor()
    
    def compute_matrix(
        self,
        activations: Dict[str, torch.Tensor],
        layer_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """Compute full CKA matrix between layers.
        
        Args:
            activations: Dict mapping layer names to activations
            layer_names: Subset of layers to compute (default: all)
            progress_callback: Called with (current, total) progress
            
        Returns:
            CKA matrix of shape [n_layers, n_layers]
        """
        if layer_names is None:
            layer_names = list(activations.keys())
        
        n_layers = len(layer_names)
        cka_matrix = torch.zeros(n_layers, n_layers)
        
        # Check if we can resume from checkpoint
        start_i, start_j = self._load_checkpoint(layer_names)
        
        total_pairs = n_layers * (n_layers + 1) // 2
        current_pair = start_i * n_layers + start_j
        
        with tqdm(total=total_pairs, initial=current_pair) as pbar:
            for i in range(start_i, n_layers):
                for j in range(start_j if i == start_i else i, n_layers):
                    # Check memory before computation
                    if self.memory_monitor.available_mb() < 100:
                        logger.warning("Low memory, triggering garbage collection")
                        torch.cuda.empty_cache()
                    
                    # Compute CKA
                    act_i = activations[layer_names[i]]
                    act_j = activations[layer_names[j]]
                    
                    cka_value = self.cka_computer.compute(act_i, act_j)
                    cka_matrix[i, j] = cka_value
                    cka_matrix[j, i] = cka_value
                    
                    # Update progress
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(current_pair, total_pairs)
                    
                    # Checkpoint periodically
                    if (current_pair + 1) % 100 == 0:
                        self._save_checkpoint(cka_matrix, i, j, layer_names)
                    
                    current_pair += 1
        
        return cka_matrix
```

## Testing Suite

### Test Structure
```
tests/phase2_cka/
├── unit/
│   ├── test_debiased_cka.py
│   ├── test_adaptive_sampling.py
│   ├── test_nystrom.py
│   └── test_memory_efficiency.py
├── integration/
│   ├── test_cka_pipeline.py
│   └── test_checkpoint_resume.py
└── validation/
    ├── test_no_double_centering.py
    └── test_accuracy_benchmarks.py
```

### Critical Test: No Double Centering
```python
# tests/phase2_cka/validation/test_no_double_centering.py
import pytest
import torch
import numpy as np
from neurosheaf.cka.debiased import DebiasedCKA

class TestNoDoubleCentering:
    """CRITICAL: Verify NO double-centering occurs."""
    
    def test_raw_activations_used(self):
        """Ensure CKA uses raw, uncentered activations."""
        n_samples = 100
        n_features = 50
        
        # Create activations with non-zero mean
        X = torch.randn(n_samples, n_features) + 5.0
        Y = torch.randn(n_samples, n_features) + 3.0
        
        # Compute CKA
        cka = DebiasedCKA(use_unbiased=True)
        
        # Hook to capture kernel matrices
        captured_K = None
        captured_L = None
        
        def capture_kernels(K, L):
            nonlocal captured_K, captured_L
            captured_K = K.clone()
            captured_L = L.clone()
        
        # Monkey-patch to capture
        original_unbiased = cka._unbiased_hsic
        def patched_unbiased(K, L):
            capture_kernels(K, L)
            return original_unbiased(K, L)
        
        cka._unbiased_hsic = patched_unbiased
        
        # Compute
        cka_value = cka.compute(X, Y)
        
        # Verify kernels are computed from raw data
        expected_K = X @ X.T
        expected_L = Y @ Y.T
        
        torch.testing.assert_close(captured_K, expected_K)
        torch.testing.assert_close(captured_L, expected_L)
        
        # Verify kernels are NOT centered
        K_mean = captured_K.mean()
        L_mean = captured_L.mean()
        
        assert abs(K_mean) > 1.0, "K appears to be centered"
        assert abs(L_mean) > 1.0, "L appears to be centered"
    
    def test_comparison_with_incorrect_implementation(self):
        """Compare correct vs incorrect (double-centered) implementation."""
        X = torch.randn(50, 30) + 2.0
        Y = torch.randn(50, 30) + 1.5
        
        # Correct implementation
        cka_correct = DebiasedCKA(use_unbiased=True)
        value_correct = cka_correct.compute(X, Y)
        
        # Incorrect implementation (with double centering)
        def incorrect_cka(X, Y):
            # WRONG: Pre-center the data
            X_c = X - X.mean(dim=0, keepdim=True)
            Y_c = Y - Y.mean(dim=0, keepdim=True)
            
            K = X_c @ X_c.T
            L = Y_c @ Y_c.T
            
            # Then apply unbiased HSIC (which centers again!)
            return cka_correct._unbiased_hsic(K, L)
        
        value_incorrect = incorrect_cka(X, Y)
        
        # Values should be different
        assert abs(value_correct - value_incorrect) > 0.01
        
        # Correct value should typically be higher
        assert value_correct > value_incorrect
```

### Memory Efficiency Tests
```python
# tests/phase2_cka/unit/test_memory_efficiency.py
import pytest
import torch
import psutil
import gc
from neurosheaf.cka.nystrom import NystromCKA
from neurosheaf.cka.debiased import DebiasedCKA

class TestMemoryEfficiency:
    """Test memory-efficient implementations."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean memory before/after tests."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        yield
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def test_nystrom_memory_scaling(self):
        """Verify Nyström uses O(n*m) memory instead of O(n^2)."""
        n_samples = 10000
        n_features = 512
        n_landmarks = 256
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        # Measure memory for Nyström
        nystrom = NystromCKA(n_landmarks=n_landmarks)
        
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        cka_nystrom = nystrom.compute(X, Y)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        memory_used = mem_after - mem_before
        
        # Expected memory: O(n * m)
        expected_mb = (n_samples * n_landmarks * 4 * 2) / 1024 / 1024
        
        # Should use much less than O(n^2)
        full_kernel_mb = (n_samples ** 2 * 4) / 1024 / 1024
        
        assert memory_used < full_kernel_mb * 0.1  # Less than 10% of full
        assert memory_used < expected_mb * 2  # Within 2x of theoretical
    
    @pytest.mark.parametrize("batch_size,expected_memory_mb", [
        (100, 1),
        (500, 10),
        (1000, 40),
        (2000, 160)
    ])
    def test_adaptive_sampling_memory(self, batch_size, expected_memory_mb):
        """Test memory usage scales correctly with sample size."""
        from neurosheaf.cka.sampling import AdaptiveSampler
        
        sampler = AdaptiveSampler()
        
        # Determine sample size given memory constraint
        n_total = 10000
        n_features = 512
        
        selected_size = sampler.determine_sample_size(
            n_total=n_total,
            n_features=n_features,
            available_memory_mb=expected_memory_mb
        )
        
        # Verify selection respects memory limit
        kernel_memory = (selected_size ** 2 * 4) / 1024 / 1024
        assert kernel_memory <= expected_memory_mb
```

### Integration Tests
```python
# tests/phase2_cka/integration/test_cka_pipeline.py
import pytest
import torch
import tempfile
from pathlib import Path
from neurosheaf.cka import DebiasedCKA
from neurosheaf.cka.pairwise import PairwiseCKA

class TestCKAPipeline:
    """Test complete CKA computation pipeline."""
    
    def test_end_to_end_cka_computation(self):
        """Test full pipeline from activations to CKA matrix."""
        # Create mock activations
        n_samples = 1000
        layer_dims = [512, 256, 128, 64]
        
        activations = {}
        for i, dim in enumerate(layer_dims):
            activations[f'layer_{i}'] = torch.randn(n_samples, dim)
        
        # Compute CKA matrix
        cka = DebiasedCKA(use_unbiased=True)
        pairwise = PairwiseCKA(cka, memory_limit_mb=512)
        
        cka_matrix = pairwise.compute_matrix(activations)
        
        # Validate properties
        assert cka_matrix.shape == (len(layer_dims), len(layer_dims))
        assert torch.allclose(cka_matrix, cka_matrix.T)  # Symmetric
        assert torch.all(torch.diag(cka_matrix) >= 0.99)  # Diagonal ~1
        assert torch.all((cka_matrix >= 0) & (cka_matrix <= 1))  # Range [0,1]
    
    def test_checkpoint_resume(self, tmp_path):
        """Test computation can resume from checkpoint."""
        n_layers = 10
        n_samples = 500
        
        activations = {
            f'layer_{i}': torch.randn(n_samples, 128)
            for i in range(n_layers)
        }
        
        # Start computation with checkpoint
        cka = DebiasedCKA()
        pairwise = PairwiseCKA(
            cka,
            checkpoint_dir=str(tmp_path)
        )
        
        # Simulate interruption after 20 pairs
        pairs_computed = 0
        
        def interrupt_callback(current, total):
            nonlocal pairs_computed
            pairs_computed = current
            if current >= 20:
                raise KeyboardInterrupt("Simulated interruption")
        
        with pytest.raises(KeyboardInterrupt):
            pairwise.compute_matrix(
                activations,
                progress_callback=interrupt_callback
            )
        
        # Resume computation
        cka_matrix = pairwise.compute_matrix(activations)
        
        # Should have resumed from checkpoint
        assert pairs_computed >= 20
        assert cka_matrix.shape == (n_layers, n_layers)
```

### Performance Benchmarks
```python
# tests/phase2_cka/validation/test_accuracy_benchmarks.py
import pytest
import torch
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from neurosheaf.cka import DebiasedCKA
from neurosheaf.cka.nystrom import NystromCKA

class TestAccuracyBenchmarks:
    """Validate CKA accuracy against reference implementations."""
    
    def test_cka_accuracy_vs_reference(self):
        """Compare against trusted reference implementation."""
        n_samples = 200
        X = torch.randn(n_samples, 100)
        Y = torch.randn(n_samples, 100)
        
        # Our implementation
        cka = DebiasedCKA(use_unbiased=True)
        our_value = cka.compute(X, Y)
        
        # Reference implementation (simplified)
        def reference_cka(X, Y):
            X_np = X.numpy()
            Y_np = Y.numpy()
            
            K = linear_kernel(X_np)
            L = linear_kernel(Y_np)
            
            # Unbiased HSIC
            n = K.shape[0]
            K_0 = K - np.diag(np.diag(K))
            L_0 = L - np.diag(np.diag(L))
            
            term1 = np.sum(K_0 * L_0)
            term2 = np.sum(K_0) * np.sum(L_0) / (n-1) / (n-2)
            term3 = 2 * np.sum(K_0, axis=0) @ np.sum(L_0, axis=0) / (n-2)
            
            hsic_xy = (term1 + term2 - term3) / (n * (n-3))
            
            # Same for XX and YY
            hsic_xx = reference_hsic(K, K)
            hsic_yy = reference_hsic(L, L)
            
            return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
        
        ref_value = reference_cka(X, Y)
        
        assert abs(our_value - ref_value) < 1e-5
    
    def test_nystrom_approximation_quality(self):
        """Test Nyström approximation accuracy."""
        n_samples = 2000
        n_features = 256
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        # Exact CKA
        cka_exact = DebiasedCKA()
        exact_value = cka_exact.compute(X, Y)
        
        # Nyström approximations with different landmark counts
        landmark_counts = [64, 128, 256, 512]
        errors = []
        
        for n_landmarks in landmark_counts:
            nystrom = NystromCKA(n_landmarks=n_landmarks)
            approx_value = nystrom.compute(X, Y)
            error = abs(approx_value - exact_value)
            errors.append(error)
        
        # Error should decrease with more landmarks
        assert all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        
        # With 512 landmarks, error should be < 1%
        assert errors[-1] < 0.01
```

## Success Criteria

1. **Correctness**
   - NO double-centering in implementation
   - CKA values match reference implementations
   - Proper handling of edge cases

2. **Performance**
   - 10x memory reduction vs naive implementation
   - Nyström approximation error < 1% with 512 landmarks
   - Can handle 50k samples with 4GB memory

3. **Robustness**
   - Checkpoint/resume functionality works
   - Handles GPU OOM gracefully
   - Adaptive sampling maintains accuracy

4. **Integration**
   - Clean API for downstream modules
   - Progress tracking and callbacks
   - Comprehensive error messages

## Phase 2 Deliverables

1. **Core CKA Module**
   - Debiased CKA implementation (NO double-centering)
   - Adaptive sampling strategy
   - Memory-efficient operations

2. **Nyström Approximation**
   - Landmark selection strategies
   - Accuracy vs memory trade-offs
   - Automatic parameter selection

3. **Integration Components**
   - Pairwise CKA computation
   - Checkpoint/resume support
   - Progress tracking

4. **Documentation**
   - API documentation
   - Usage examples
   - Performance tuning guide

5. **Test Suite**
   - 100% coverage of critical paths
   - Validation against references
   - Performance benchmarks
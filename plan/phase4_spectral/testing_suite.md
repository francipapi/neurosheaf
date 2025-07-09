# Phase 4: Spectral Analysis Testing Suite

## Overview

Phase 4 focuses on persistent spectral analysis with subspace tracking, edge masking, and multi-parameter persistence. The testing suite emphasizes correctness of eigenvalue tracking through crossings and mathematical validity of persistence computations.

## Test Categories

### 1. Subspace Tracking Tests (CRITICAL)
```python
# tests/phase4_spectral/critical/test_subspace_tracking.py
import pytest
import torch
import numpy as np
from neurosheaf.spectral.tracker import SubspaceTracker
from scipy.linalg import subspace_angles

class TestSubspaceTrackingCritical:
    """Critical tests for subspace tracking correctness."""
    
    def test_eigenvalue_crossing_continuity(self):
        """Test eigenspaces remain continuous through crossings."""
        tracker = SubspaceTracker(gap_eps=1e-6, cos_tau=0.9)
        
        # Create two eigenvalues that cross
        n_steps = 21
        eigenval_seqs = []
        eigenvec_seqs = []
        
        for i in range(n_steps):
            t = i / (n_steps - 1)  # t goes from 0 to 1
            
            # Two eigenvalues that cross at t=0.5
            eig1 = 1.0 - t
            eig2 = t
            eig3 = 2.0  # Stays constant
            
            eigenvals = torch.tensor([eig1, eig2, eig3])
            
            # Create eigenvectors that evolve smoothly
            angle = t * np.pi / 4  # Smooth rotation
            c, s = np.cos(angle), np.sin(angle)
            
            eigenvecs = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=torch.float32).T
            
            eigenval_seqs.append(eigenvals)
            eigenvec_seqs.append(eigenvecs)
        
        # Track through crossing
        params = list(range(n_steps))
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Should have continuous paths
        paths = tracking_info['eigenvalue_paths']
        assert len(paths) >= 2
        
        # Check path continuity
        for path in paths:
            if len(path) > 1:
                similarities = [step['similarity'] for step in path]
                # All similarities should be reasonably high
                assert all(sim > 0.7 for sim in similarities)
    
    def test_degeneracy_handling(self):
        """Test handling of degenerate eigenvalues."""
        tracker = SubspaceTracker(gap_eps=1e-4)
        
        # Create degenerate eigenvalues
        eigenvals = torch.tensor([0.0, 0.00005, 0.00010, 1.0, 1.0001, 2.0])
        eigenvecs = torch.eye(6)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should group close eigenvalues
        assert len(groups) == 3  # [0, 0.00005, 0.0001], [1.0, 1.0001], [2.0]
        assert len(groups[0]['eigenvalues']) == 3
        assert len(groups[1]['eigenvalues']) == 2
        assert len(groups[2]['eigenvalues']) == 1
        
        # Check subspace dimensions
        assert groups[0]['subspace'].shape[1] == 3
        assert groups[1]['subspace'].shape[1] == 2
        assert groups[2]['subspace'].shape[1] == 1
    
    def test_subspace_similarity_accuracy(self):
        """Test accuracy of subspace similarity computation."""
        tracker = SubspaceTracker()
        
        # Create orthogonal subspaces
        Q1 = torch.eye(4)[:, :2]
        Q2 = torch.eye(4)[:, 2:4]
        
        similarity = tracker._compute_subspace_similarity(Q1, Q2)
        assert abs(similarity) < 1e-6  # Should be nearly zero
        
        # Test identical subspaces
        similarity = tracker._compute_subspace_similarity(Q1, Q1)
        assert abs(similarity - 1.0) < 1e-6  # Should be 1
        
        # Test rotated subspace
        angle = np.pi / 6
        R = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        Q1_rot = Q1 @ R
        
        similarity = tracker._compute_subspace_similarity(Q1, Q1_rot)
        assert abs(similarity - 1.0) < 1e-6  # Should still be 1
    
    def test_birth_death_detection(self):
        """Test detection of birth and death events."""
        tracker = SubspaceTracker(cos_tau=0.8)
        
        # Create sequence where eigenspace appears and disappears
        eigenval_seqs = [
            torch.tensor([0.0, 1.0]),      # 2 eigenvalues
            torch.tensor([0.0, 0.5, 1.0]), # 3 eigenvalues (birth)
            torch.tensor([0.0, 1.0])       # 2 eigenvalues (death)
        ]
        
        eigenvec_seqs = [
            torch.eye(2),
            torch.eye(3),
            torch.eye(2)
        ]
        
        params = [0, 1, 2]
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Should detect birth and death
        assert len(tracking_info['birth_events']) >= 1
        assert len(tracking_info['death_events']) >= 1
```

### 2. Edge Masking and Laplacian Tests
```python
# tests/phase4_spectral/unit/test_edge_masking.py
import pytest
import torch
import networkx as nx
from neurosheaf.spectral.static_laplacian import StaticLaplacianWithMasking
from neurosheaf.sheaf.construction import Sheaf

class TestEdgeMasking:
    """Test edge masking functionality."""
    
    def test_edge_mask_creation(self):
        """Test creation of edge masks."""
        # Create simple sheaf
        poset = nx.path_graph(3, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(3)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking()
        
        # Extract edge info
        full_laplacian = analyzer.laplacian_builder.build_laplacian(sheaf)
        edge_info = analyzer._extract_edge_info(sheaf, full_laplacian)
        
        # Test different thresholds
        def threshold_func(weight, param):
            return weight >= param
        
        # Low threshold - should include all edges
        mask_low = analyzer._create_edge_mask(edge_info, 0.1, threshold_func)
        assert all(mask_low.values())
        
        # High threshold - should exclude some edges
        mask_high = analyzer._create_edge_mask(edge_info, 0.9, threshold_func)
        assert not all(mask_high.values())
    
    def test_filtration_sequence_consistency(self):
        """Test consistency of filtration sequence."""
        # Create sheaf with varying edge weights
        poset = nx.complete_graph(4, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(4)}
        
        weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        restrictions = {}
        idx = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    restrictions[(str(i), str(j))] = torch.eye(2) * weights[idx % len(weights)]
                    idx += 1
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking()
        
        # Test monotonicity
        params = [0.3, 0.5, 0.7, 0.9]
        
        def threshold_func(weight, param):
            return weight >= param
        
        results = analyzer.compute_persistence(sheaf, params, threshold_func)
        
        # Number of zero eigenvalues should be monotonic
        zero_counts = []
        for eigenvals in results['eigenvalue_sequences']:
            zero_count = torch.sum(eigenvals < 1e-6).item()
            zero_counts.append(zero_count)
        
        # Should be non-decreasing (more zeros with higher threshold)
        for i in range(len(zero_counts) - 1):
            assert zero_counts[i] <= zero_counts[i + 1] + 1  # Allow small violations
    
    def test_sparse_eigenvalue_computation(self):
        """Test sparse eigenvalue computation accuracy."""
        # Create known Laplacian
        n = 20
        # Create path graph Laplacian
        L = torch.zeros(n, n)
        for i in range(n - 1):
            L[i, i] = 1
            L[i, i + 1] = -1
            L[i + 1, i] = -1
            L[i + 1, i + 1] = 1
        L[-1, -1] = 1  # Fix last diagonal
        
        # Convert to sparse
        L_sparse = L.to_sparse()
        
        analyzer = StaticLaplacianWithMasking(max_eigenvalues=10)
        eigenvals, eigenvecs = analyzer._compute_eigenvalues(L_sparse)
        
        # Check properties
        assert len(eigenvals) <= 10
        assert torch.all(eigenvals >= -1e-6)  # Non-negative
        assert torch.allclose(eigenvals[0], torch.tensor(0.0), atol=1e-6)  # First eigenvalue is 0
```

### 3. Persistence Computation Tests
```python
# tests/phase4_spectral/unit/test_persistence_computation.py
import pytest
import torch
import numpy as np
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.construction import Sheaf
import networkx as nx

class TestPersistenceComputation:
    """Test persistence computation correctness."""
    
    def test_persistence_diagram_generation(self):
        """Test generation of persistence diagrams."""
        # Create simple sheaf
        poset = nx.path_graph(4, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(3) for i in range(4)}
        restrictions = {
            (str(i), str(i+1)): torch.eye(3) * (0.9 - i * 0.1)
            for i in range(3)
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=10)
        
        # Check diagram structure
        diagrams = result['diagrams']
        assert 'birth_death_pairs' in diagrams
        assert 'infinite_bars' in diagrams
        
        # Check pair validity
        for pair in diagrams['birth_death_pairs']:
            assert pair['birth'] <= pair['death']
            assert pair['lifetime'] >= 0
    
    def test_feature_extraction(self):
        """Test extraction of persistence features."""
        # Create sheaf with known structure
        poset = nx.complete_graph(5, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(5)}
        restrictions = {
            (str(i), str(j)): torch.eye(2) * 0.8
            for i in range(5) for j in range(5) if i != j
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=5)
        
        # Check feature structure
        features = result['features']
        expected_features = [
            'eigenvalue_evolution',
            'spectral_gap_evolution',
            'effective_dimension',
            'num_birth_events',
            'num_death_events'
        ]
        
        for feature in expected_features:
            assert feature in features
        
        # Check feature validity
        assert len(features['eigenvalue_evolution']) == 5
        assert len(features['spectral_gap_evolution']) == 5
        assert len(features['effective_dimension']) == 5
        assert features['num_birth_events'] >= 0
        assert features['num_death_events'] >= 0
    
    def test_filtration_parameter_generation(self):
        """Test automatic filtration parameter generation."""
        # Create sheaf with known edge weights
        poset = nx.path_graph(3, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(3)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test auto-detection
        params = analyzer._generate_filtration_params(
            sheaf, 'threshold', 10, None
        )
        
        assert len(params) == 10
        assert params[0] < params[-1]  # Should be increasing
        
        # Should cover range of edge weights
        min_param, max_param = min(params), max(params)
        assert min_param < 0.6  # Should be below minimum edge weight
        assert max_param > 0.8  # Should be above maximum edge weight
    
    def test_different_filtration_types(self):
        """Test different filtration types produce valid results."""
        # Create simple sheaf
        poset = nx.cycle_graph(4, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(4)}
        restrictions = {
            (str(i), str((i + 1) % 4)): torch.eye(2) * 0.7
            for i in range(4)
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test different filtration types
        filtration_types = ['threshold', 'cka_based']
        
        for ftype in filtration_types:
            result = analyzer.analyze(sheaf, filtration_type=ftype, n_steps=5)
            
            # Should produce valid results
            assert len(result['persistence_result']['eigenvalue_sequences']) == 5
            assert 'features' in result
            assert 'diagrams' in result
```

### 4. Performance and Scaling Tests
```python
# tests/phase4_spectral/performance/test_spectral_performance.py
import pytest
import torch
import torch.nn as nn
import time
import psutil
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.construction import SheafBuilder

class TestSpectralPerformance:
    """Test performance of spectral analysis."""
    
    @pytest.mark.slow
    def test_large_network_scaling(self):
        """Test scaling to large networks."""
        # Create large model
        layers = []
        for i in range(30):  # 30 layers
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        
        model = nn.Sequential(*layers)
        
        # Generate activations
        x = torch.randn(200, 256)
        activations = {}
        
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations[f'relu_{i}'] = x.clone()
        
        # Build sheaf
        sheaf_builder = SheafBuilder()
        sheaf = sheaf_builder.build_sheaf(model, activations)
        
        # Time analysis
        analyzer = PersistentSpectralAnalyzer()
        start_time = time.time()
        
        result = analyzer.analyze(sheaf, n_steps=20)
        
        analysis_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert analysis_time < 300  # 5 minutes
        assert len(result['persistence_result']['eigenvalue_sequences']) == 20
    
    @pytest.mark.memory
    def test_memory_efficiency(self):
        """Test memory usage remains bounded."""
        process = psutil.Process()
        
        # Create moderately large sheaf
        poset = nx.complete_graph(20, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(10) for i in range(20)}
        restrictions = {
            (str(i), str(j)): torch.eye(10) * 0.8
            for i in range(20) for j in range(20) if i != j
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Measure memory
        mem_before = process.memory_info().rss / 1024 / 1024
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=30)
        
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_used = mem_after - mem_before
        
        # Should use reasonable memory
        assert memory_used < 2000  # Less than 2GB
        assert len(result['persistence_result']['eigenvalue_sequences']) == 30
    
    def test_eigenvalue_computation_speed(self):
        """Test speed of eigenvalue computation."""
        # Create large sparse matrix
        n = 1000
        # Random sparse matrix
        indices = torch.randint(0, n, (2, n * 5))
        values = torch.randn(n * 5)
        sparse_matrix = torch.sparse_coo_tensor(indices, values, (n, n))
        
        from neurosheaf.spectral.static_laplacian import StaticLaplacianWithMasking
        analyzer = StaticLaplacianWithMasking(max_eigenvalues=50)
        
        # Time eigenvalue computation
        start_time = time.time()
        eigenvals, eigenvecs = analyzer._compute_eigenvalues(sparse_matrix)
        computation_time = time.time() - start_time
        
        # Should be reasonably fast
        assert computation_time < 10  # 10 seconds
        assert len(eigenvals) <= 50
```

### 5. Integration Tests
```python
# tests/phase4_spectral/integration/test_spectral_integration.py
import pytest
import torch
import torch.nn as nn
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.construction import SheafBuilder
from neurosheaf.cka import DebiasedCKA

class TestSpectralIntegration:
    """Test integration with other components."""
    
    def test_cka_integration(self):
        """Test integration with CKA computation."""
        # Create model
        model = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 20)
        )
        
        # Generate activations
        x = torch.randn(150, 20)
        activations = {}
        
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations[f'relu_{i}'] = x.clone()
        
        # Compute CKA matrices
        cka_computer = DebiasedCKA()
        cka_matrices = {}
        
        layer_names = list(activations.keys())
        for name in layer_names:
            cka_row = []
            for other_name in layer_names:
                cka_value = cka_computer.compute(
                    activations[name],
                    activations[other_name]
                )
                cka_row.append(cka_value)
            cka_matrices[name] = torch.tensor(cka_row)
        
        # Build sheaf with CKA
        sheaf_builder = SheafBuilder()
        sheaf = sheaf_builder.build_sheaf(model, activations, cka_matrices)
        
        # Analyze
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=10)
        
        # Should complete successfully
        assert len(result['persistence_result']['eigenvalue_sequences']) == 10
        assert 'features' in result
    
    def test_visualization_data_preparation(self):
        """Test preparation of data for visualization."""
        # Create simple sheaf
        poset = nx.path_graph(5, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(3) for i in range(5)}
        restrictions = {
            (str(i), str(i+1)): torch.eye(3) * (0.9 - i * 0.1)
            for i in range(4)
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=15)
        
        # Check data is suitable for visualization
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        assert len(eigenval_seqs) == 15
        
        # All eigenvalues should be real and non-negative
        for eigenvals in eigenval_seqs:
            assert torch.all(torch.isreal(eigenvals))
            assert torch.all(eigenvals >= -1e-6)
        
        # Features should be well-formed
        features = result['features']
        assert all(len(features[key]) == 15 for key in ['eigenvalue_evolution', 'spectral_gap_evolution'])
        
        # Diagrams should be plottable
        diagrams = result['diagrams']
        for pair in diagrams['birth_death_pairs']:
            assert 'birth' in pair and 'death' in pair
            assert pair['birth'] <= pair['death']
```

### 6. Edge Cases and Robustness Tests
```python
# tests/phase4_spectral/edge_cases/test_spectral_edge_cases.py
import pytest
import torch
import numpy as np
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.spectral.tracker import SubspaceTracker
from neurosheaf.sheaf.construction import Sheaf
import networkx as nx

class TestSpectralEdgeCases:
    """Test edge cases and robustness."""
    
    def test_empty_sheaf(self):
        """Test handling of empty sheaf."""
        poset = nx.DiGraph()
        poset.add_node('single')
        
        sheaf = Sheaf(
            poset=poset,
            stalks={'single': torch.eye(2)},
            restrictions={}
        )
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Should handle gracefully
        result = analyzer.analyze(sheaf, n_steps=5)
        assert len(result['persistence_result']['eigenvalue_sequences']) == 5
    
    def test_disconnected_poset(self):
        """Test handling of disconnected poset."""
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('C', 'D')])  # Two disconnected components
        
        stalks = {node: torch.eye(2) for node in poset.nodes()}
        restrictions = {
            ('A', 'B'): torch.eye(2) * 0.8,
            ('C', 'D'): torch.eye(2) * 0.6
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=5)
        
        # Should handle disconnected components
        assert len(result['persistence_result']['eigenvalue_sequences']) == 5
    
    def test_numerical_precision_issues(self):
        """Test handling of numerical precision issues."""
        # Create eigenvalues very close to machine precision
        eigenvals = torch.tensor([1e-15, 1e-14, 1e-13, 1.0])
        eigenvecs = torch.eye(4)
        
        tracker = SubspaceTracker(gap_eps=1e-12)
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should handle tiny eigenvalues
        assert len(groups) >= 1
        assert all(len(group['eigenvalues']) > 0 for group in groups)
    
    def test_degenerate_laplacian(self):
        """Test handling of degenerate Laplacian."""
        # Create sheaf with identical restrictions
        poset = nx.complete_graph(3, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(3)}
        restrictions = {
            (str(i), str(j)): torch.eye(2)  # All identical
            for i in range(3) for j in range(3) if i != j
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Should handle degenerate case
        result = analyzer.analyze(sheaf, n_steps=5)
        assert len(result['persistence_result']['eigenvalue_sequences']) == 5
    
    def test_extreme_filtration_parameters(self):
        """Test extreme filtration parameters."""
        # Create simple sheaf
        poset = nx.path_graph(3, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(3)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.5,
            ('1', '2'): torch.eye(2) * 0.5
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test extreme ranges
        extreme_params = [1e-10, 1e10]
        
        def threshold_func(weight, param):
            return weight >= param
        
        # Should handle without crashing
        from neurosheaf.spectral.static_laplacian import StaticLaplacianWithMasking
        static_analyzer = StaticLaplacianWithMasking()
        
        result = static_analyzer.compute_persistence(
            sheaf, extreme_params, threshold_func
        )
        
        assert len(result['eigenvalue_sequences']) == 2
```

## Test Execution Strategy

### Continuous Integration
```yaml
# .github/workflows/phase4_spectral.yml
name: Phase 4 Spectral Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install torch scipy networkx
        pip install pytest pytest-cov
        pip install -e .
    
    - name: Run critical tests
      run: |
        pytest tests/phase4_spectral/critical -v --tb=short
    
    - name: Run unit tests
      run: |
        pytest tests/phase4_spectral/unit -v
    
    - name: Run integration tests
      run: |
        pytest tests/phase4_spectral/integration -v -m "not slow"
    
    - name: Run edge case tests
      run: |
        pytest tests/phase4_spectral/edge_cases -v
```

### Local Testing Commands
```bash
# Quick development tests
make test-phase4-quick

# Full test suite
make test-phase4-full

# Performance tests
make test-phase4-performance

# Memory tests
make test-phase4-memory
```

## Success Metrics

1. **Test Coverage**: >95% code coverage for spectral module
2. **Subspace Tracking**: 100% success rate on eigenvalue crossing scenarios
3. **Performance**: Analysis completes in <5 minutes for 30-layer networks
4. **Memory**: Uses <3GB for large network analysis
5. **Robustness**: Handles all edge cases gracefully

## Common Issues and Solutions

### Issue: Eigenvalue solver convergence
```python
# Solution: Increase tolerance and iterations
def robust_eigenvalue_solve(matrix, max_iter=1000, tol=1e-8):
    try:
        return standard_solver(matrix, tol=tol)
    except ConvergenceError:
        return fallback_solver(matrix, max_iter=max_iter)
```

### Issue: Memory overflow with large matrices
```python
# Solution: Chunked processing
def chunked_eigenvalue_computation(matrix, chunk_size=1000):
    if matrix.shape[0] < chunk_size:
        return standard_computation(matrix)
    else:
        return approximate_computation(matrix, chunk_size)
```

### Issue: Numerical instability with small eigenvalues
```python
# Solution: Adaptive thresholding
def adaptive_eigenvalue_threshold(eigenvals, relative_threshold=1e-12):
    max_eigenval = torch.max(eigenvals)
    threshold = max_eigenval * relative_threshold
    return torch.clamp(eigenvals, min=threshold)
```

This comprehensive testing suite ensures the spectral analysis module is robust, efficient, and mathematically correct across all scenarios.
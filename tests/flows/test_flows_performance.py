"""Performance and stress tests for α-flow and t-flow implementations.

This module tests performance characteristics, memory usage, and scalability
of flow implementations under various conditions.

Tests cover:
- SLQ budget scaling (probes×iters constant, runtime constant)
- Memory footprint (no dense materializations for large cases)
- λ̂max estimation caching and cost
- Scaling behavior with matrix size
- Stress testing with extreme parameters
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import time
import psutil
import os
from typing import Dict, List, Tuple
import warnings
import torch

from neurosheaf.spectral.flows.alpha_flow import (
    AlphaGroupingPolicy, AlphaFlowBuilder
)
from neurosheaf.spectral.flows.diffusion_flow import (
    DiffusionSpec, DiffusionFlowAnalyzer
)
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.spectral.utils_numerical import estimate_lambda_max

from .fixtures import (
    small_path_sheaf, small_star_sheaf, FakeGWLaplacianBuilder,
    fallback_path_sheaf, fallback_star_sheaf
)


# Performance test decorators
slow_test = pytest.mark.slow
memory_intensive = pytest.mark.skipif(
    psutil.virtual_memory().total < 4 * 1024**3,  # 4GB
    reason="Requires at least 4GB RAM"
)


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def time_function(func, *args, **kwargs):
    """Time a function call and return (result, elapsed_time)."""
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time


class TestSLQBudgetScaling:
    """Test SLQ performance scaling with probe/iteration budgets."""
    
    @slow_test
    def test_slq_constant_budget_scaling(self, small_path_sheaf):
        """Test that runtime is roughly constant when probes×iters is constant."""
        fake_builder = FakeGWLaplacianBuilder(default_size=64)  # Larger for meaningful timing
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Build Laplacian once
        L, D, _ = analyzer._build_laplacian_all_edges()
        
        # Test different probe/iteration combinations with constant budget
        budget = 1000  # probes × iters = constant
        configurations = [
            (50, 20),   # 50 probes × 20 iters = 1000
            (100, 10),  # 100 probes × 10 iters = 1000
            (25, 40),   # 25 probes × 40 iters = 1000
        ]
        
        timings = []
        accuracies = []
        
        for probes, iters in configurations:
            spec = DiffusionSpec(
                t_grid=[0.5],  # Single t for timing
                probes=probes,
                slq_iters=iters
            )
            
            # Time the analysis
            result, elapsed = time_function(analyzer.analyze, spec)
            timings.append(elapsed)
            accuracies.append(result.heat_trace[0])
        
        # Check timing consistency (allow 2x variation)
        mean_timing = np.mean(timings)
        max_deviation = max(abs(t - mean_timing) for t in timings)
        relative_deviation = max_deviation / mean_timing
        
        assert relative_deviation < 1.0, \
            f"Runtime not constant with budget: timings={timings}, rel_dev={relative_deviation:.2f}"
        
        # Check accuracy is similar (SLQ should converge similarly)
        accuracy_std = np.std(accuracies)
        accuracy_mean = np.mean(accuracies)
        accuracy_cv = accuracy_std / (abs(accuracy_mean) + 1e-15)
        
        assert accuracy_cv < 0.2, \
            f"Accuracy varies too much with budget allocation: CV={accuracy_cv:.3f}"
    
    @slow_test
    def test_slq_scaling_with_matrix_size(self):
        """Test SLQ scaling behavior with increasing matrix size."""
        sizes = [32, 64, 128]  # Different matrix sizes
        base_probes = 32
        base_iters = 15
        
        timings = []
        
        for size in sizes:
            # Create appropriately sized test sheaf using fallback approach
            from .fixtures import create_minimal_sheaf_fallback
            test_sheaf = create_minimal_sheaf_fallback(f'size_{size}')
            
            fake_builder = FakeGWLaplacianBuilder(default_size=size)
            analyzer = DiffusionFlowAnalyzer(test_sheaf, fake_builder, random_seed=42)
            
            spec = DiffusionSpec(
                t_grid=[0.5],
                probes=base_probes,
                slq_iters=base_iters
            )
            
            # Time the analysis
            _, elapsed = time_function(analyzer.analyze, spec)
            timings.append(elapsed)
        
        # SLQ should scale roughly linearly with matrix size (for matvec operations)
        # Check that timing doesn't scale worse than quadratically
        for i in range(1, len(timings)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = timings[i] / timings[i-1]
            
            # Allow up to quadratic scaling (conservative bound)
            max_expected_ratio = size_ratio ** 2
            
            assert time_ratio <= max_expected_ratio * 2, \
                f"SLQ scaling too bad: size {sizes[i-1]}→{sizes[i]} " \
                f"(ratio {size_ratio:.1f}), time ratio {time_ratio:.2f} > {max_expected_ratio:.2f}"


class TestMemoryFootprint:
    """Test memory usage characteristics."""
    
    def test_no_dense_materialization_alpha_flow(self, fallback_path_sheaf):
        """Test that α-flow doesn't create dense matrices for large-ish problems."""
        # Use larger fake builder to test memory behavior
        large_size = 1000  # Large enough to matter if dense
        fake_builder = FakeGWLaplacianBuilder(default_size=large_size)
        
        memory_before = get_memory_usage_mb()
        
        builder = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        # Build α-flow decomposition
        build = builder.build(grouping=grouping, as_linear_operator=True)
        
        memory_after_build = get_memory_usage_mb()
        memory_increase_build = memory_after_build - memory_before
        
        # Memory increase should be modest (not O(n²) for dense matrices)
        # Large dense matrix would be ~1000² × 8 bytes ≈ 8MB just for one matrix
        # We have L_base, L_resid, D, so dense would be ~24MB minimum
        expected_dense_memory = (large_size ** 2) * 8 * 3 / 1024**2  # 3 matrices, 8 bytes each
        
        assert memory_increase_build < expected_dense_memory * 0.1, \
            f"Memory usage too high: {memory_increase_build:.1f}MB, " \
            f"dense would be ~{expected_dense_memory:.1f}MB"
        
        # Test operator construction (should also be memory-efficient)
        alpha_values = [0.0, 0.5, 1.0, 2.0]
        
        for alpha in alpha_values:
            L_alpha = builder.as_operator(build, alpha)
            memory_after_op = get_memory_usage_mb()
            
            # LinearOperator creation should be lightweight
            memory_increase_op = memory_after_op - memory_after_build
            assert memory_increase_op < 10, \
                f"LinearOperator creation too expensive: {memory_increase_op:.1f}MB"
    
    def test_sparse_matrix_memory_efficiency(self, small_star_sheaf):
        """Test that sparse matrices are used efficiently in t-flow."""
        # Use moderately large size for meaningful test
        medium_size = 500
        fake_builder = FakeGWLaplacianBuilder(default_size=medium_size)
        
        memory_before = get_memory_usage_mb()
        
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # Build Laplacian (should be sparse)
        L, D, metadata = analyzer._build_laplacian_all_edges()
        
        memory_after = get_memory_usage_mb()
        memory_increase = memory_after - memory_before
        
        # Check sparsity information
        assert 'L_nnz' in metadata, "Should track Laplacian sparsity"
        assert 'D_nnz' in metadata, "Should track mass matrix sparsity"
        
        L_nnz = metadata['L_nnz']
        D_nnz = metadata['D_nnz']
        
        # Sparse storage should be much more efficient than dense
        dense_memory_estimate = (medium_size ** 2) * 8 * 2 / 1024**2  # L and D, 8 bytes each
        sparse_memory_estimate = (L_nnz + D_nnz) * 8 / 1024**2  # Just non-zeros
        
        assert memory_increase < dense_memory_estimate * 0.2, \
            f"Memory usage suggests dense storage: {memory_increase:.1f}MB, " \
            f"dense estimate: {dense_memory_estimate:.1f}MB"
        
        # Sparsity should be significant for fake builder's tridiagonal structure
        sparsity_L = L_nnz / (medium_size ** 2)
        sparsity_D = D_nnz / (medium_size ** 2)
        
        assert sparsity_L < 0.1, f"L not sparse enough: {sparsity_L:.3f}"
        assert sparsity_D < 0.1, f"D not sparse enough: {sparsity_D:.3f}"
    
    @memory_intensive
    @slow_test
    def test_large_scale_memory_bounds(self):
        """Test memory usage stays within bounds for larger problems."""
        # Test with larger matrices to verify scalability
        large_sizes = [1000, 2000]  # Progressively larger
        memory_limit_mb = 1000  # 1GB limit for test
        
        for size in large_sizes:
            # Create minimal sheaf for large test using fallback approach
            from .fixtures import create_minimal_sheaf_fallback
            test_sheaf = create_minimal_sheaf_fallback(f'large_{size}')
            
            memory_start = get_memory_usage_mb()
            
            fake_builder = FakeGWLaplacianBuilder(default_size=size)
            analyzer = DiffusionFlowAnalyzer(test_sheaf, fake_builder, random_seed=42)
            
            # Analyze with modest parameters
            spec = DiffusionSpec(
                t_grid=[0.1, 1.0],
                probes=32,
                slq_iters=20
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress any warnings for stress test
                result = analyzer.analyze(spec)
            
            memory_peak = get_memory_usage_mb()
            memory_used = memory_peak - memory_start
            
            assert memory_used < memory_limit_mb, \
                f"Memory usage too high for size {size}: {memory_used:.1f}MB > {memory_limit_mb}MB"
            
            # Check that analysis completed successfully
            assert len(result.heat_trace) == len(spec.t_grid), \
                f"Analysis failed for size {size}"


class TestLambdaMaxEstimationCost:
    """Test λ_max estimation performance and caching."""
    
    def test_lambda_max_estimation_time(self, small_path_sheaf):
        """Test that λ_max estimation is reasonably fast."""
        sizes = [64, 128, 256]
        
        for size in sizes:
            fake_builder = FakeGWLaplacianBuilder(default_size=size)
            analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
            
            # Build Laplacian
            L, D, _ = analyzer._build_laplacian_all_edges()
            
            # Time λ_max estimation
            start_time = time.time()
            lambda_max = estimate_lambda_max(L, iters=40, rng=np.random.default_rng(42))
            estimation_time = time.time() - start_time
            
            # Should be much faster than eigenvalue computation
            # Power method should converge quickly for well-conditioned matrices
            max_time_seconds = 1.0  # Conservative bound
            
            assert estimation_time < max_time_seconds, \
                f"λ_max estimation too slow for size {size}: {estimation_time:.3f}s > {max_time_seconds}s"
            
            assert lambda_max > 0, f"λ_max should be positive: {lambda_max}"
            assert np.isfinite(lambda_max), f"λ_max should be finite: {lambda_max}"
    
    def test_lambda_max_caching_efficiency(self, small_star_sheaf):
        """Test that λ_max caching avoids recomputation cost."""
        fake_builder = FakeGWLaplacianBuilder(default_size=128)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # First call with auto t-grid (should compute λ_max)
        spec_auto = DiffusionSpec(t_grid='auto', probes=16, slq_iters=10)
        
        start_time = time.time()
        result1 = analyzer.analyze(spec_auto)
        first_call_time = time.time() - start_time
        
        # Second call (should use cached λ_max)
        start_time = time.time()
        result2 = analyzer.analyze(spec_auto)
        second_call_time = time.time() - start_time
        
        # Second call should be faster (no λ_max recomputation)
        speedup_ratio = first_call_time / second_call_time
        
        assert speedup_ratio > 1.1, \
            f"Insufficient speedup from caching: {speedup_ratio:.2f}x, " \
            f"times: {first_call_time:.3f}s → {second_call_time:.3f}s"
        
        # Results should be identical (deterministic caching)
        assert np.allclose(result1.t_grid, result2.t_grid), \
            "Cached results should be identical"
        assert np.allclose(result1.heat_trace, result2.heat_trace), \
            "Cached heat traces should be identical"
    
    def test_lambda_max_reasonable_values(self, small_path_sheaf):
        """Test that λ_max estimates are in reasonable ranges."""
        sizes = [32, 64, 128]
        
        for size in sizes:
            fake_builder = FakeGWLaplacianBuilder(default_size=size)
            analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
            
            L, D, _ = analyzer._build_laplacian_all_edges()
            
            # Estimate λ_max
            lambda_max = estimate_lambda_max(L, iters=50, rng=np.random.default_rng(42))
            
            # Should be reasonable for tridiagonal-like matrices from fake builder
            # Tridiagonal Laplacian eigenvalues are roughly O(1) to O(n)
            expected_min = 0.1
            expected_max = size * 10  # Conservative upper bound
            
            assert expected_min <= lambda_max <= expected_max, \
                f"λ_max out of expected range for size {size}: " \
                f"{lambda_max:.2e} not in [{expected_min}, {expected_max}]"


class TestStressAndExtremalCases:
    """Test behavior under extreme parameters and stress conditions."""
    
    @slow_test
    def test_many_alpha_values_stress(self, fallback_path_sheaf):
        """Test α-flow with many α values (stress test)."""
        fake_builder = FakeGWLaplacianBuilder(default_size=64)
        builder = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        # Large α grid
        alpha_grid = np.logspace(-3, 2, 50)  # 50 α values from 0.001 to 100
        
        start_time = time.time()
        
        traces = []
        for alpha in alpha_grid:
            L_alpha = builder.as_operator(build, alpha)
            # Quick trace estimate
            from .fixtures import trace_hutchinson
            trace_est = trace_hutchinson(L_alpha, power=1, probes=16, seed=42)
            traces.append(trace_est)
        
        total_time = time.time() - start_time
        
        # Should complete in reasonable time
        max_time = 30.0  # 30 seconds for 50 evaluations
        assert total_time < max_time, \
            f"Stress test too slow: {total_time:.1f}s > {max_time}s"
        
        # Check monotonicity is mostly preserved
        monotonic_violations = sum(1 for i in range(1, len(traces)) 
                                 if traces[i] < traces[i-1] - 1e-6)
        violation_rate = monotonic_violations / (len(traces) - 1)
        
        assert violation_rate < 0.1, \
            f"Too many monotonicity violations in stress test: {violation_rate:.2%}"
    
    @slow_test  
    def test_extreme_t_values_robustness(self, small_star_sheaf):
        """Test t-flow robustness with extreme t values."""
        fake_builder = FakeGWLaplacianBuilder(default_size=32)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # Extreme t values (very small and very large)
        extreme_t_grid = [1e-8, 1e-4, 1e0, 1e4, 1e8]
        
        spec = DiffusionSpec(
            t_grid=extreme_t_grid,
            probes=32,
            slq_iters=20
        )
        
        # Should handle extreme values gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Expect some numerical warnings
            
            start_time = time.time()
            result = analyzer.analyze(spec)
            analysis_time = time.time() - start_time
        
        # Should complete without hanging
        max_time = 20.0
        assert analysis_time < max_time, \
            f"Extreme value analysis too slow: {analysis_time:.1f}s"
        
        # Should produce finite results where possible
        finite_count = np.sum(np.isfinite(result.heat_trace))
        assert finite_count >= len(extreme_t_grid) * 0.5, \
            f"Too many non-finite results: {finite_count}/{len(extreme_t_grid)}"
        
        # Check that fallback mechanisms were used appropriately
        assert 'n_fallback_points' in result.meta, "Should track fallback usage"
    
    def test_minimal_probe_budget_robustness(self, small_path_sheaf):
        """Test behavior with minimal computational budgets."""
        fake_builder = FakeGWLaplacianBuilder(default_size=32)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Minimal budgets
        minimal_specs = [
            DiffusionSpec(t_grid=[0.5], probes=1, slq_iters=1),   # Absolute minimum
            DiffusionSpec(t_grid=[0.5], probes=2, slq_iters=2),   # Barely functional
            DiffusionSpec(t_grid=[0.5], probes=4, slq_iters=5),   # Low but reasonable
        ]
        
        for spec in minimal_specs:
            start_time = time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Expect convergence warnings
                result = analyzer.analyze(spec)
            
            analysis_time = time.time() - start_time
            
            # Should complete quickly with minimal budgets
            assert analysis_time < 1.0, \
                f"Minimal budget analysis too slow: {analysis_time:.3f}s"
            
            # Should produce some result (even if inaccurate)
            assert len(result.heat_trace) == len(spec.t_grid), \
                "Should produce results with minimal budget"
            
            # Heat trace should be finite (even if inaccurate)
            assert np.all(np.isfinite(result.heat_trace)), \
                "Heat trace should be finite even with minimal budget"
    
    @slow_test
    def test_high_precision_parameter_sweep(self, small_star_sheaf):
        """Test high-precision parameter sweep for accuracy validation."""
        fake_builder = FakeGWLaplacianBuilder(default_size=64)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # High-precision spec for reference
        high_precision_spec = DiffusionSpec(
            t_grid=[0.1, 0.5, 1.0],
            probes=256,  # High probe count
            slq_iters=50  # High iteration count
        )
        
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference_result = analyzer.analyze(high_precision_spec)
        
        high_precision_time = time.time() - start_time
        
        # Should complete in reasonable time even with high precision
        max_time = 60.0  # 1 minute for high precision
        assert high_precision_time < max_time, \
            f"High precision analysis too slow: {high_precision_time:.1f}s"
        
        # Should have high quality results
        assert 'heat_trace_std' in reference_result.meta, \
            "Should have standard deviation estimates"
        
        heat_trace_std = np.array(reference_result.meta['heat_trace_std'])
        finite_std = heat_trace_std[np.isfinite(heat_trace_std)]
        
        if len(finite_std) > 0:
            # Standard deviations should be small for high precision
            rel_std = finite_std / (np.abs(reference_result.heat_trace[:len(finite_std)]) + 1e-15)
            max_rel_std = np.max(rel_std)
            
            assert max_rel_std < 0.05, \
                f"High precision results not precise enough: max rel_std={max_rel_std:.3f}"
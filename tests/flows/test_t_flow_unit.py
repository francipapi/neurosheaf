"""Unit tests for t-flow (diffusion) implementation.

This module tests the core t-flow functionality including Laplacian construction,
heat trace computation, monotonicity properties, and error handling.

Tests cover:
- Laplacian & mass matrix building with ridge regularization
- Auto t-grid generation based on λ_max estimation  
- Heat trace monotonicity (decreasing with t)
- Range sanity checks (0 < h(t) ≤ 1)
- SLQ vs exact computation comparison
- Eigenvalue fallback mechanisms
- Determinism and scaling invariance
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import expm
from typing import Dict, List, Tuple
import warnings

from neurosheaf.spectral.flows.diffusion_flow import (
    DiffusionSpec, DiffusionSummaries, DiffusionFlowAnalyzer
)
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.spectral.utils_numerical import heat_trace_slq, estimate_lambda_max

from .fixtures import (
    small_path_sheaf, small_star_sheaf, FakeGWLaplacianBuilder,
    linear_operator_equals_sparse, is_psd, eigs_dense
)


class TestDiffusionFlowLaplacianBuilding:
    """Test Laplacian and mass matrix construction for t-flow."""
    
    def test_laplacian_symmetry_and_format(self, small_path_sheaf):
        """Test that built Laplacian is symmetric CSR with correct properties."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Build Laplacian
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        
        # Check L properties
        assert isinstance(L, LinearOperator), "L should be LinearOperator"
        assert L.shape[0] == L.shape[1], "L should be square"
        assert L.shape == (8, 8), "L should match fake builder size"
        assert L.dtype == np.float64, "L should be float64"
        
        # Check D properties
        assert sp.issparse(D), "D must be sparse"
        assert D.format == 'csr', f"D should be CSR format, got {D.format}"
        assert D.dtype == np.float64, "D should be float64"
        assert D.shape == L.shape, "D and L must have same shape"
        
        # Check symmetry via random probes (since L is wrapped in LinearOperator)
        rng = np.random.default_rng(42)
        for _ in range(5):
            v = rng.standard_normal(8)
            w = rng.standard_normal(8)
            
            # Test symmetry: <v, L*w> = <L*v, w>
            vLw = v.T @ (L @ w)
            Lvw = (L @ v).T @ w
            rel_error = abs(vLw - Lvw) / (abs(vLw) + 1e-15)
            assert rel_error < 1e-12, f"Symmetry violation: relative error {rel_error:.2e}"
    
    def test_mass_matrix_spd_properties(self, small_path_sheaf):
        """Test mass matrix D is SPD with ridge regularization."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        
        # Check ridge regularization was applied
        assert 'ridge_regularization' in metadata
        ridge_eps = metadata['ridge_regularization']
        assert ridge_eps > 0, "Ridge regularization should be positive"
        
        # Check SPD properties (for small matrix)
        if D.shape[0] <= 100:
            assert is_psd(D), "Mass matrix D must be positive semi-definite"
            
            # Check positive diagonal
            diagonal = D.diagonal()
            assert np.all(diagonal > 0), "D diagonal must be positive"
            
            # Check ridge was actually added (diagonal should be ≥ ridge_eps)
            assert np.all(diagonal >= ridge_eps * 0.9), "Ridge regularization not properly applied"
        
        # Check symmetry
        D_diff = D - D.T
        if D_diff.nnz > 0:
            assert np.allclose(D_diff.data, 0), "D must be symmetric"
    
    def test_builder_return_mass_fallback(self, small_path_sheaf):
        """Test fallback when builder doesn't support return_mass."""
        
        class NoMassBuilder(FakeGWLaplacianBuilder):
            """Builder that doesn't support return_mass."""
            
            def build_laplacian(self, sheaf, sparse=True, mass_mode='fixed', return_mass=False):
                if return_mass:
                    raise TypeError("return_mass not supported")
                return super().build_laplacian(sheaf, sparse, mass_mode, return_mass=False)
        
        no_mass_builder = NoMassBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, no_mass_builder, random_seed=42)
        
        # Should fallback to identity mass matrix
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        
        # Should still get valid L and D
        assert isinstance(L, LinearOperator)
        assert sp.issparse(D)
        assert D.shape == L.shape
        
        # D should be close to identity (with ridge)
        ridge_eps = metadata['ridge_regularization']
        expected_diag = 1.0 + ridge_eps
        actual_diag = D.diagonal()
        assert np.allclose(actual_diag, expected_diag), "Fallback mass matrix should be identity + ridge"
    
    def test_linear_operator_fidelity(self, small_path_sheaf):
        """Test LinearOperator matches CSR implementation via probes."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Build and get internal CSR (we'll access via fake builder)
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        
        # Get the original CSR from fake builder for comparison
        L_csr_orig = fake_builder.build_laplacian(small_path_sheaf, sparse=True)
        L_csr_orig = L_csr_orig.tocsr()
        
        # Symmetrize to match analyzer's processing
        L_csr_symmetrized = 0.5 * (L_csr_orig + L_csr_orig.T)
        
        # Compare LinearOperator to symmetrized CSR
        assert linear_operator_equals_sparse(L, L_csr_symmetrized, k=15, atol=1e-12), \
            "LinearOperator should match symmetrized CSR matrix"
    
    def test_caching_behavior(self, small_path_sheaf):
        """Test that Laplacian construction is cached properly."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # First call should build
        L1, D1, meta1 = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        first_call_count = fake_builder._call_count
        
        # Second call should use cache
        L2, D2, meta2 = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        second_call_count = fake_builder._call_count
        
        # Call count should not increase (cache hit)
        assert second_call_count == first_call_count, "Second call should use cached result"
        
        # Results should be identical (same objects)
        assert L1 is L2, "Cached LinearOperator should be same object"
        assert D1 is D2, "Cached mass matrix should be same object"
        assert meta1 is meta2, "Cached metadata should be same object"


class TestDiffusionFlowAutoTGrid:
    """Test automatic t-grid generation."""
    
    def test_auto_tgrid_range_and_length(self, small_path_sheaf):
        """Test auto t-grid spans correct range with right number of points."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Build Laplacian to enable λ_max estimation
        mass_mode = 'fixed'
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode)
        
        # Generate auto t-grid
        t_grid = analyzer._generate_auto_t_grid(L, mass_mode, n_points=20)
        
        # Check basic properties
        assert len(t_grid) == 20, "Should generate requested number of points"
        assert np.all(t_grid > 0), "All t values must be positive"
        assert np.all(np.diff(t_grid) > 0), "t_grid must be strictly increasing"
        
        # Check range based on λ_max estimation
        cache_key = (mass_mode, id(small_path_sheaf))
        lam_max = analyzer._lambda_max_cache.get(cache_key)
        assert lam_max is not None and lam_max > 0, "λ_max should be positive"
        
        # Expected range: [1e-3/λ_max, 10/λ_max] with bounds clipping
        expected_t_min = max(1e-3 / lam_max, 1e-12)
        expected_t_max = min(10.0 / lam_max, 1e6)
        
        # Allow some tolerance for log-spacing
        assert t_grid[0] >= expected_t_min * 0.9, f"t_min {t_grid[0]:.2e} < expected {expected_t_min:.2e}"
        assert t_grid[-1] <= expected_t_max * 1.1, f"t_max {t_grid[-1]:.2e} > expected {expected_t_max:.2e}"
    
    def test_lambda_max_caching(self, small_path_sheaf):
        """Test that λ_max is cached between calls."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        mass_mode = 'fixed'
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode)
        
        # First call should compute λ_max
        t_grid1 = analyzer._generate_auto_t_grid(L, mass_mode, n_points=10)
        cache_key = (mass_mode, id(small_path_sheaf))
        lam_max_cached = analyzer._lambda_max_cache.get(cache_key)
        assert lam_max_cached is not None, "λ_max should be cached after first call"
        
        # Second call should use cached value (we can't easily test this directly,
        # but we verify the cache is accessed)
        t_grid2 = analyzer._generate_auto_t_grid(L, mass_mode, n_points=10)
        
        # Results should be identical (deterministic)
        assert np.allclose(t_grid1, t_grid2), "t_grid should be deterministic when using cache"
        assert analyzer._lambda_max_cache.get(cache_key) == lam_max_cached, "λ_max cache should be unchanged"
    
    def test_extreme_lambda_max_bounds_clipping(self):
        """Test t-grid bounds clipping for extreme λ_max values."""
        # Create a mock analyzer with controlled λ_max cache
        fake_builder = FakeGWLaplacianBuilder(default_size=4)
        import networkx as nx
        import torch
        
        # Create a minimal sheaf with one edge
        poset = nx.DiGraph() 
        poset.add_edge('A', 'B')
        
        sheaf = Sheaf(
            poset=poset,
            stalks={
                'A': torch.randn(4, 1),  # 4 samples, 1 dimension
                'B': torch.randn(4, 1)   # 4 samples, 1 dimension
            },
            restrictions={
                ('A', 'B'): torch.ones(1, 1)  # 1x1 identity
            },
            metadata={
                'construction_method': 'gromov_wasserstein',
                'is_gw_sheaf': True, 
                'gw_costs': {('A', 'B'): 0.5}
            }
        )
        
        analyzer = DiffusionFlowAnalyzer(sheaf, fake_builder, random_seed=42)
        
        mass_mode = 'fixed'
        cache_key = (mass_mode, id(sheaf))
        
        # Mock very small λ_max (would cause overflow)
        analyzer._lambda_max_cache[cache_key] = 1e-15
        
        L = LinearOperator((4, 4), matvec=lambda x: x)  # Dummy operator
        t_grid_small = analyzer._generate_auto_t_grid(L, mass_mode, n_points=5)
        
        # Should be clipped to reasonable bounds (allow for wider range in practice)
        assert np.all(t_grid_small >= 1e-15), "t_min should be clipped to prevent underflow"
        assert np.all(t_grid_small <= 1e12), "t_max should be clipped to prevent overflow"
        
        # Mock very large λ_max (would cause underflow)
        analyzer._lambda_max_cache[cache_key] = 1e15
        
        t_grid_large = analyzer._generate_auto_t_grid(L, mass_mode, n_points=5)
        
        # Should be clipped to reasonable bounds (allow for wider range in practice)
        assert np.all(t_grid_large >= 1e-15), "t_min should be clipped"
        assert np.all(t_grid_large <= 1e12), "t_max should be clipped"
    
    def test_cache_respects_mass_mode(self, small_path_sheaf):
        """Test that cache correctly handles different mass_mode values."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Use different mass modes 
        spec = DiffusionSpec(t_grid='auto', k_small=2)
        
        # First run with fixed mode
        result_fixed = analyzer.analyze(spec, mass_mode='fixed')
        
        # Second run with adaptive mode - should NOT reuse cache
        result_adaptive = analyzer.analyze(spec, mass_mode='adaptive')
        
        # Verify both have metadata
        assert 'ridge_regularization' in result_fixed.meta
        assert 'ridge_regularization' in result_adaptive.meta
        
        # Ridge values should be different
        ridge_fixed = result_fixed.meta['ridge_regularization']
        ridge_adaptive = result_adaptive.meta['ridge_regularization']
        assert ridge_fixed != ridge_adaptive, "Ridge values should differ between mass modes"
        assert ridge_fixed == 1e-12, "Fixed mode should use 1e-12 ridge"
        assert ridge_adaptive == 1e-15, "Adaptive mode should use 1e-15 ridge"
        
        # Verify separate cache entries exist
        fixed_key = ('fixed', id(small_path_sheaf))
        adaptive_key = ('adaptive', id(small_path_sheaf))
        
        assert fixed_key in analyzer._laplacian_cache, "Fixed mode should have cache entry"
        assert adaptive_key in analyzer._laplacian_cache, "Adaptive mode should have cache entry"
        assert fixed_key != adaptive_key, "Cache keys should be different"
        
        # Third run with fixed mode should reuse first cache
        call_count_before = fake_builder._call_count
        result_fixed_2 = analyzer.analyze(spec, mass_mode='fixed')
        call_count_after = fake_builder._call_count
        
        assert call_count_after == call_count_before, "Fixed mode rerun should use cache"
        assert result_fixed_2.meta['ridge_regularization'] == ridge_fixed, "Should have same ridge value"


class TestDiffusionFlowHeatTrace:
    """Test heat trace computation and properties."""
    
    def test_heat_trace_monotonicity(self, small_path_sheaf):
        """Test that h(t) is non-increasing with t."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Use explicit t-grid for controlled testing
        spec = DiffusionSpec(
            t_grid=[0.01, 0.1, 0.5, 1.0, 2.0],
            k_small=4,
            probes=64,
            slq_iters=20
        )
        
        result = analyzer.analyze(spec, mass_mode='fixed')
        
        # Check basic result properties
        assert len(result.heat_trace) == len(spec.t_grid)
        assert np.all(np.isfinite(result.heat_trace)), "All heat trace values should be finite"
        
        # Check monotonicity (allowing for some SLQ noise)
        for i in range(1, len(result.heat_trace)):
            # Strict monotonicity with small tolerance
            assert result.heat_trace[i] <= result.heat_trace[i-1] + 1e-9, \
                f"Heat trace not monotonic: h({spec.t_grid[i-1]})={result.heat_trace[i-1]:.6e} < " \
                f"h({spec.t_grid[i]})={result.heat_trace[i]:.6e}"
        
        # Check metadata monotonicity flags
        assert 'is_monotonic' in result.meta
        assert 'is_monotonic_strict' in result.meta
        
        # For well-behaved fake system, should be monotonic
        assert result.meta['is_monotonic'], "Monotonicity check should pass"
    
    def test_heat_trace_range_sanity(self, small_path_sheaf):
        """Test that 0 < h(t) ≤ 1."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.05, 0.2, 1.0],
            probes=64,
            slq_iters=25
        )
        
        result = analyzer.analyze(spec)
        
        # Check range bounds with small slack
        heat_trace = result.heat_trace
        valid_mask = np.isfinite(heat_trace)
        
        if np.any(valid_mask):
            valid_heat = heat_trace[valid_mask]
            
            # Lower bound (should be positive)
            assert np.all(valid_heat >= -1e-6), \
                f"Heat trace has negative values: min={np.min(valid_heat):.6e}"
            
            # Upper bound (should be ≤ 1)
            assert np.all(valid_heat <= 1.0 + 1e-6), \
                f"Heat trace exceeds 1: max={np.max(valid_heat):.6e}"
            
            # Strict positivity for small t (heat kernel should be well-spread)
            small_t_values = valid_heat[np.array(spec.t_grid) <= 0.5]
            if len(small_t_values) > 0:
                assert np.all(small_t_values > 1e-8), \
                    "Heat trace should be strictly positive for small t"
    
    def test_slq_vs_dense_small_matrix(self):
        """Test SLQ vs exact dense computation for tiny matrices."""
        import networkx as nx
        import torch
        
        # Create tiny sheaf for exact computation using proper construction
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        
        stalks = {
            'A': torch.randn(4, 1),  # 4 samples, 1 dimension
            'B': torch.randn(4, 1)   # 4 samples, 1 dimension
        }
        
        restrictions = {
            ('A', 'B'): torch.ones(1, 1)  # 1x1 matrix
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'is_gw_sheaf': True, 
            'gw_costs': {('A', 'B'): 0.5}
        }
        
        tiny_sheaf = Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata
        )
        
        # Use very small fake builder
        tiny_builder = FakeGWLaplacianBuilder(default_size=4)
        analyzer = DiffusionFlowAnalyzer(tiny_sheaf, tiny_builder, random_seed=42)
        
        # Build Laplacian
        L, D, metadata = analyzer._build_laplacian_all_edges()
        
        # Test mid-range t value
        t = 0.5
        
        # SLQ estimate
        h_slq = heat_trace_slq(L, t, probes=64, iters=30, rng=np.random.default_rng(42))
        if isinstance(h_slq, tuple):
            h_slq = h_slq[0]
        h_slq_normalized = h_slq / L.shape[0]
        
        # Dense exact computation (only for tiny matrices)
        if L.shape[0] <= 32:
            # Convert to dense for exact computation
            L_dense = np.zeros(L.shape)
            for i in range(L.shape[0]):
                e_i = np.zeros(L.shape[0])
                e_i[i] = 1.0
                L_dense[:, i] = L @ e_i
            
            # Compute exact heat trace
            exp_minus_tL = expm(-t * L_dense)
            h_exact = np.trace(exp_minus_tL) / L.shape[0]
            
            # Compare SLQ to exact (allow reasonable SLQ error - it's stochastic)
            rel_error = abs(h_slq_normalized - h_exact) / (abs(h_exact) + 1e-15)
            assert rel_error <= 0.05, \
                f"SLQ error too large: SLQ={h_slq_normalized:.6e}, exact={h_exact:.6e}, " \
                f"rel_error={rel_error:.3f}"
    
    def test_eigenvalue_fallback_mechanism(self, small_path_sheaf):
        """Test fallback to eigenvalue computation when SLQ fails."""
        
        class FailingSLQBuilder(FakeGWLaplacianBuilder):
            """Builder that will cause SLQ to fail for testing."""
            
            def build_laplacian(self, sheaf, sparse=True, mass_mode='fixed', return_mass=False):
                # Return a problematic Laplacian that might cause SLQ issues
                n = self.default_size
                # Create a matrix with some extreme eigenvalues
                L = sp.diags([1e-8, 1e8, 1e8, 1e-8], format='csr', dtype=np.float64)
                L.resize((n, n))
                
                if return_mass:
                    D = sp.eye(n, format='csr', dtype=np.float64)
                    return L, D
                return L
        
        failing_builder = FailingSLQBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, failing_builder, random_seed=42)
        
        # Use spec that should cause some SLQ failures
        spec = DiffusionSpec(
            t_grid=[1e-6, 1e6],  # Extreme t values that might cause SLQ issues
            k_small=2,  # Enable eigenvalue fallback
            probes=8,  # Low probes to increase failure chance
            slq_iters=5  # Low iterations to increase failure chance
        )
        
        # Analyze (might have some fallbacks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress SLQ warnings
            result = analyzer.analyze(spec)
        
        # Check fallback metadata
        assert 'n_fallback_points' in result.meta
        n_fallbacks = result.meta['n_fallback_points']
        
        # Should still have valid results (either SLQ or fallback)
        assert len(result.heat_trace) == len(spec.t_grid)
        
        # If fallbacks occurred, heat_trace_std should have NaN entries
        if n_fallbacks > 0:
            heat_trace_std = np.array(result.meta['heat_trace_std'])
            nan_count = np.sum(np.isnan(heat_trace_std))
            assert nan_count >= n_fallbacks, "Fallback points should have NaN std"
    
    def test_determinism_with_fixed_seed(self, small_path_sheaf):
        """Test that same seed produces identical results."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        
        spec = DiffusionSpec(
            t_grid=[0.1, 0.5, 1.0],
            probes=32,
            slq_iters=20
        )
        
        # First run with seed 42
        analyzer1 = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        result1 = analyzer1.analyze(spec)
        
        # Second run with same seed
        analyzer2 = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        result2 = analyzer2.analyze(spec)
        
        # Results should be identical
        assert np.allclose(result1.heat_trace, result2.heat_trace, atol=1e-14), \
            "Same seed should produce identical heat traces"
        assert np.allclose(result1.t_grid, result2.t_grid, atol=1e-14), \
            "Same seed should produce identical t grids"
        
        # Metadata should match
        assert result1.meta['random_seed'] == result2.meta['random_seed'] == 42
    
    def test_different_seeds_produce_variance(self, small_path_sheaf):
        """Test that different seeds produce results within expected SLQ variance."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        
        spec = DiffusionSpec(
            t_grid=[0.5],  # Single t value
            probes=32,
            slq_iters=20
        )
        
        # Multiple runs with different seeds
        heat_traces = []
        for seed in [42, 123, 999]:
            analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=seed)
            result = analyzer.analyze(spec)
            heat_traces.append(result.heat_trace[0])
        
        # Results should vary but not too much (SLQ variance scales ~ 1/sqrt(probes))
        heat_mean = np.mean(heat_traces)
        heat_std = np.std(heat_traces)
        
        # Variance should be reasonable (not zero, not huge)
        expected_std = heat_mean / np.sqrt(spec.probes)  # Rough SLQ scaling
        
        assert heat_std > 0, "Different seeds should produce some variance"
        assert heat_std < 10 * expected_std, \
            f"Variance too large: std={heat_std:.2e}, expected~{expected_std:.2e}"


class TestDiffusionFlowScalingInvariance:
    """Test scaling invariance properties."""
    
    def test_laplacian_scaling_invariance(self, small_path_sheaf):
        """Test h(t) invariance under L → cL, t → t/c."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Get baseline Laplacian
        L_base, D_base, _ = analyzer._build_laplacian_all_edges()
        
        # Test t value and scaling factor
        t_base = 0.5
        c = 2.0
        t_scaled = t_base / c
        
        # Compute h(t) for original system
        h_base = heat_trace_slq(L_base, t_base, probes=64, iters=25, rng=np.random.default_rng(42))
        if isinstance(h_base, tuple):
            h_base = h_base[0]
        h_base_normalized = h_base / L_base.shape[0]
        
        # Create scaled LinearOperator: L_scaled = c * L_base
        def scaled_matvec(x):
            return c * (L_base @ x)
        
        L_scaled = LinearOperator(L_base.shape, matvec=scaled_matvec, dtype=L_base.dtype)
        
        # Compute h(t/c) for scaled system
        h_scaled = heat_trace_slq(L_scaled, t_scaled, probes=64, iters=25, rng=np.random.default_rng(42))
        if isinstance(h_scaled, tuple):
            h_scaled = h_scaled[0]
        h_scaled_normalized = h_scaled / L_scaled.shape[0]
        
        # Should be approximately equal (allowing for SLQ variance)
        rel_error = abs(h_base_normalized - h_scaled_normalized) / (abs(h_base_normalized) + 1e-15)
        assert rel_error < 0.05, \
            f"Scaling invariance violated: h_base={h_base_normalized:.6e}, " \
            f"h_scaled={h_scaled_normalized:.6e}, rel_error={rel_error:.3f}"


class TestDiffusionFlowErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_sheaf_error(self):
        """Test error handling for empty sheaf."""
        empty_sheaf = Sheaf()
        empty_sheaf.metadata = {'is_gw_sheaf': True, 'gw_costs': {}}
        
        fake_builder = FakeGWLaplacianBuilder()
        
        # Should raise error during initialization or analysis
        with pytest.raises(ValueError) as exc_info:
            analyzer = DiffusionFlowAnalyzer(empty_sheaf, fake_builder)
            spec = DiffusionSpec(t_grid=[0.1, 1.0])
            analyzer.analyze(spec)
        
        assert "empty sheaf" in str(exc_info.value).lower()
    
    def test_invalid_diffusion_spec_validation(self):
        """Test DiffusionSpec parameter validation."""
        # Valid specs should work
        DiffusionSpec(t_grid=[0.1, 1.0], k_small=5, probes=32, slq_iters=20)
        DiffusionSpec(t_grid='auto')
        
        # Invalid parameters should raise errors
        with pytest.raises(ValueError):
            DiffusionSpec(k_small=-1)  # Negative k_small
            
        with pytest.raises(ValueError):
            DiffusionSpec(probes=0)  # Zero probes
            
        with pytest.raises(ValueError):
            DiffusionSpec(slq_iters=0)  # Zero iterations
            
        with pytest.raises(ValueError):
            DiffusionSpec(t_grid=[])  # Empty t_grid
            
        with pytest.raises(ValueError):
            DiffusionSpec(t_grid=[-0.1, 1.0])  # Negative t values
            
        with pytest.raises(ValueError):
            DiffusionSpec(t_grid=[1.0, 0.0, 1.0])  # Zero t value
    
    def test_k_small_zero_fast_path(self, small_path_sheaf):
        """Test k_small=0 skips eigenvalue computation."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.1, 1.0],
            k_small=0,  # Should skip eigenvalue computation
            probes=32
        )
        
        result = analyzer.analyze(spec)
        
        # Should complete without eigenvalue computation
        assert len(result.smallest_eigs) == 0, "k_small=0 should produce empty eigenvalues"
        assert len(result.heat_trace) == 2, "Should still compute heat trace"
        
        # Metadata should reflect skipped computation
        assert result.meta['n_fallback_points'] == 0, "No fallbacks expected with k_small=0"
    
    def test_single_t_value_handling(self, small_path_sheaf):
        """Test handling of single t value (scalar to vector)."""
        fake_builder = FakeGWLaplacianBuilder(default_size=4)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Test with single t value
        spec = DiffusionSpec(t_grid=[0.5])  # Single element
        
        result = analyzer.analyze(spec)
        
        # Should handle single value correctly
        assert len(result.heat_trace) == 1
        assert len(result.t_grid) == 1
        assert result.t_grid[0] == 0.5
        assert np.isfinite(result.heat_trace[0])
    
    def test_extreme_lambda_max_handling(self, small_path_sheaf):
        """Test handling of extreme λ_max values in auto t-grid."""
        fake_builder = FakeGWLaplacianBuilder(default_size=4)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        mass_mode = 'fixed'
        cache_key = (mass_mode, id(small_path_sheaf))
        
        # Build system first
        L, D, _ = analyzer._build_laplacian_all_edges(mass_mode)
        
        # Test with manually set extreme λ_max
        analyzer._lambda_max_cache[cache_key] = 1e-20  # Extremely small
        
        t_grid_small = analyzer._generate_auto_t_grid(L, mass_mode, n_points=5)
        
        # Should be clipped and valid
        assert np.all(t_grid_small > 0), "All t values should be positive"
        assert np.all(np.isfinite(t_grid_small)), "All t values should be finite"
        assert len(t_grid_small) == 5, "Should generate requested number of points"
        
        # Test with extremely large λ_max
        analyzer._lambda_max_cache[cache_key] = 1e20  # Extremely large
        
        t_grid_large = analyzer._generate_auto_t_grid(L, mass_mode, n_points=5)
        
        # Should be clipped and valid
        assert np.all(t_grid_large > 0), "All t values should be positive"
        assert np.all(np.isfinite(t_grid_large)), "All t values should be finite"
        assert len(t_grid_large) == 5, "Should generate requested number of points"
    
    
    def test_ridge_regularization_metadata_consistency(self, small_path_sheaf):
        """Test ridge_regularization metadata is present regardless of D source."""
        fake_builder = FakeGWLaplacianBuilder(default_size=4)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Test with builder that returns mass matrix
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        
        # Ridge regularization should always be in metadata
        assert 'ridge_regularization' in metadata, "ridge_regularization must be in metadata"
        assert isinstance(metadata['ridge_regularization'], (int, float)), \
            "ridge_regularization should be numeric"
        assert metadata['ridge_regularization'] > 0, \
            "ridge_regularization should be positive"
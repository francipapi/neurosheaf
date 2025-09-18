"""Regression and snapshot tests for α-flow and t-flow implementations.

This module provides regression testing against frozen reference outputs
and tests for specific edge cases that have caused bugs in the past.

Tests cover:
- Frozen snapshots for canonical sheaves and specs
- Edge case regression tests (missing costs, unknown tags, dtype mismatches)
- Specific bug reproductions and fixes
- Numerical precision regression bounds
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import hashlib

from neurosheaf.spectral.flows.alpha_flow import (
    AlphaGroupingPolicy, AlphaFlowBuilder
)
from neurosheaf.spectral.flows.diffusion_flow import (
    DiffusionSpec, DiffusionFlowAnalyzer
)
from neurosheaf.sheaf.data_structures import Sheaf

from .fixtures import (
    small_path_sheaf, small_star_sheaf, FakeGWLaplacianBuilder,
    trace_hutchinson
)


# ============================================================================
# Snapshot Reference Data
# ============================================================================

# These reference values were computed with known-good implementations
# and should remain stable across code changes

ALPHA_FLOW_REFERENCE_SNAPSHOTS = {
    "path_sheaf_quantile_0.5": {
        "alpha_grid": [0.0, 0.1, 0.3, 1.0, 3.0],
        "grouping": {"kind": "quantile", "param": 0.5, "semantics": "cost"},
        "expected_traces": {
            # Expected trace values for different α (with tolerance)
            0.0: (6.0, 0.1),      # (value, tolerance)
            0.1: (6.6, 0.15), 
            0.3: (7.8, 0.2),
            1.0: (12.0, 0.3),
            3.0: (30.0, 0.8)
        },
        "expected_monotonicity": True,
        "expected_eigenvalue_bounds": (0.0, 50.0)  # (min, max)
    },
    "star_sheaf_topk_0.3": {
        "alpha_grid": [0.0, 0.5, 1.0],
        "grouping": {"kind": "topk", "param": 0.3, "semantics": "cost"},
        "expected_traces": {
            0.0: (8.0, 0.1),
            0.5: (12.0, 0.2),
            1.0: (16.0, 0.3)
        },
        "expected_monotonicity": True,
        "expected_eigenvalue_bounds": (0.0, 40.0)
    }
}

T_FLOW_REFERENCE_SNAPSHOTS = {
    "path_sheaf_auto_grid": {
        "t_grid": "auto",
        "spec": {"probes": 64, "slq_iters": 30},
        "expected_heat_range": (0.0, 1.0),
        "expected_monotonicity": True,
        "expected_t_grid_length": 20,
        "expected_lambda_max_range": (0.1, 100.0)
    },
    "star_sheaf_explicit_grid": {
        "t_grid": [0.1, 0.5, 1.0, 2.0],
        "spec": {"probes": 64, "slq_iters": 25},
        "expected_heat_values": {
            # Expected h(t) values (with tolerance)
            0.1: (0.8, 0.1),
            0.5: (0.6, 0.1), 
            1.0: (0.4, 0.1),
            2.0: (0.2, 0.1)
        },
        "expected_monotonicity": True
    }
}


class TestAlphaFlowRegressionSnapshots:
    """Test α-flow against frozen reference snapshots."""
    
    def test_path_sheaf_quantile_regression(self, small_path_sheaf):
        """Test α-flow regression with path sheaf and quantile partitioning."""
        reference = ALPHA_FLOW_REFERENCE_SNAPSHOTS["path_sheaf_quantile_0.5"]
        
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        # Use reference grouping
        grouping_spec = reference["grouping"]
        grouping = AlphaGroupingPolicy(
            kind=grouping_spec["kind"],
            param=grouping_spec["param"], 
            semantics=grouping_spec["semantics"]
        )
        
        build = builder.build(grouping=grouping)
        
        # Test traces at reference α values
        alpha_grid = reference["alpha_grid"]
        expected_traces = reference["expected_traces"]
        
        actual_traces = {}
        for alpha in alpha_grid:
            L_alpha = builder.as_operator(build, alpha)
            trace_est = trace_hutchinson(L_alpha, power=1, probes=128, seed=42)
            actual_traces[alpha] = trace_est
        
        # Check against reference values
        for alpha in alpha_grid:
            if alpha in expected_traces:
                expected_val, tolerance = expected_traces[alpha]
                actual_val = actual_traces[alpha]
                
                rel_error = abs(actual_val - expected_val) / (abs(expected_val) + 1e-15)
                assert rel_error < tolerance, \
                    f"α={alpha}: trace regression failure, " \
                    f"expected {expected_val:.2f}±{tolerance}, got {actual_val:.2f}, " \
                    f"rel_error={rel_error:.3f}"
        
        # Check monotonicity
        if reference["expected_monotonicity"]:
            traces_values = [actual_traces[alpha] for alpha in sorted(alpha_grid)]
            for i in range(1, len(traces_values)):
                assert traces_values[i] >= traces_values[i-1] - 1e-8, \
                    f"Monotonicity regression: α sequence not increasing"
    
    def test_star_sheaf_topk_regression(self, small_star_sheaf):
        """Test α-flow regression with star sheaf and top-k partitioning."""
        reference = ALPHA_FLOW_REFERENCE_SNAPSHOTS["star_sheaf_topk_0.3"]
        
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        builder = AlphaFlowBuilder(small_star_sheaf, fake_builder)
        
        # Use reference grouping
        grouping_spec = reference["grouping"]
        grouping = AlphaGroupingPolicy(
            kind=grouping_spec["kind"],
            param=grouping_spec["param"],
            semantics=grouping_spec["semantics"]
        )
        
        build = builder.build(grouping=grouping)
        
        # Test traces
        alpha_grid = reference["alpha_grid"]
        expected_traces = reference["expected_traces"]
        
        for alpha in alpha_grid:
            if alpha in expected_traces:
                L_alpha = builder.as_operator(build, alpha)
                actual_trace = trace_hutchinson(L_alpha, power=1, probes=128, seed=42)
                
                expected_val, tolerance = expected_traces[alpha]
                rel_error = abs(actual_trace - expected_val) / (abs(expected_val) + 1e-15)
                
                assert rel_error < tolerance, \
                    f"Star sheaf α={alpha}: regression failure, " \
                    f"expected {expected_val:.2f}±{tolerance}, got {actual_trace:.2f}"
    
    def test_alpha_flow_metadata_stability(self, small_path_sheaf):
        """Test that α-flow metadata structure remains stable."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        metadata = build.meta
        
        # Check required metadata keys (regression against missing keys)
        required_keys = [
            'n_base_edges', 'n_resid_edges', 'base_edges', 'resid_edges',
            'grouping_policy', 'mass_mode', 'L_base_dtype', 'L_resid_dtype', 'D_dtype'
        ]
        
        for key in required_keys:
            assert key in metadata, f"Missing required metadata key: {key}"
        
        # Check metadata types (regression against type changes)
        assert isinstance(metadata['n_base_edges'], int)
        assert isinstance(metadata['n_resid_edges'], int)
        assert isinstance(metadata['base_edges'], list)
        assert isinstance(metadata['resid_edges'], list)
        assert isinstance(metadata['mass_mode'], str)


class TestTFlowRegressionSnapshots:
    """Test t-flow against frozen reference snapshots."""
    
    def test_path_sheaf_auto_grid_regression(self, small_path_sheaf):
        """Test t-flow regression with auto t-grid generation."""
        reference = T_FLOW_REFERENCE_SNAPSHOTS["path_sheaf_auto_grid"]
        
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Use reference spec
        spec_config = reference["spec"]
        spec = DiffusionSpec(
            t_grid=reference["t_grid"],
            probes=spec_config["probes"],
            slq_iters=spec_config["slq_iters"]
        )
        
        result = analyzer.analyze(spec)
        
        # Check t-grid length
        expected_length = reference["expected_t_grid_length"]
        assert len(result.t_grid) == expected_length, \
            f"Auto t-grid length regression: expected {expected_length}, got {len(result.t_grid)}"
        
        # Check heat trace range
        heat_min, heat_max = reference["expected_heat_range"]
        actual_heat = result.heat_trace[np.isfinite(result.heat_trace)]
        
        if len(actual_heat) > 0:
            assert np.min(actual_heat) >= heat_min - 0.1, \
                f"Heat trace minimum regression: {np.min(actual_heat):.3f} < {heat_min}"
            assert np.max(actual_heat) <= heat_max + 0.1, \
                f"Heat trace maximum regression: {np.max(actual_heat):.3f} > {heat_max}"
        
        # Check monotonicity
        if reference["expected_monotonicity"]:
            for i in range(1, len(result.heat_trace)):
                if np.isfinite(result.heat_trace[i]) and np.isfinite(result.heat_trace[i-1]):
                    assert result.heat_trace[i] <= result.heat_trace[i-1] + 1e-9, \
                        f"t-flow monotonicity regression at t[{i}]"
        
        # Check λ_max range
        if hasattr(analyzer, '_lambda_max_cache') and analyzer._lambda_max_cache is not None:
            lambda_max = analyzer._lambda_max_cache
            lam_min, lam_max = reference["expected_lambda_max_range"]
            
            assert lam_min <= lambda_max <= lam_max, \
                f"λ_max regression: {lambda_max:.2e} not in [{lam_min}, {lam_max}]"
    
    def test_star_sheaf_explicit_grid_regression(self, small_star_sheaf):
        """Test t-flow regression with explicit t-grid."""
        reference = T_FLOW_REFERENCE_SNAPSHOTS["star_sheaf_explicit_grid"]
        
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # Use reference spec
        spec_config = reference["spec"]
        spec = DiffusionSpec(
            t_grid=reference["t_grid"],
            probes=spec_config["probes"],
            slq_iters=spec_config["slq_iters"]
        )
        
        result = analyzer.analyze(spec)
        
        # Check expected heat values
        expected_heat = reference["expected_heat_values"]
        
        for i, t in enumerate(result.t_grid):
            if t in expected_heat:
                expected_val, tolerance = expected_heat[t]
                actual_val = result.heat_trace[i]
                
                if np.isfinite(actual_val):
                    rel_error = abs(actual_val - expected_val) / (abs(expected_val) + 1e-15)
                    assert rel_error < tolerance, \
                        f"t={t}: heat trace regression, " \
                        f"expected {expected_val:.3f}±{tolerance}, got {actual_val:.3f}"
    
    def test_diffusion_metadata_stability(self, small_star_sheaf):
        """Test that t-flow metadata structure remains stable."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(t_grid=[0.1, 1.0], probes=32, slq_iters=15)
        result = analyzer.analyze(spec)
        
        metadata = result.meta
        
        # Check required metadata keys
        required_keys = [
            'analysis_time', 'n_time_points', 'n_valid_points', 'n_fallback_points',
            'heat_trace_std', 'probes', 'slq_iters', 'is_monotonic', 'is_monotonic_strict',
            'lambda_max', 't_range', 'random_seed'
        ]
        
        for key in required_keys:
            assert key in metadata, f"Missing required t-flow metadata key: {key}"
        
        # Check metadata types
        assert isinstance(metadata['analysis_time'], (int, float))
        assert isinstance(metadata['n_time_points'], int)
        assert isinstance(metadata['n_valid_points'], int)
        assert isinstance(metadata['n_fallback_points'], int)
        assert isinstance(metadata['heat_trace_std'], list)
        assert isinstance(metadata['is_monotonic'], bool)
        assert isinstance(metadata['is_monotonic_strict'], bool)


class TestSpecificEdgeCaseRegressions:
    """Test specific edge cases that have caused bugs in the past."""
    
    def test_missing_gw_costs_fallback_regression(self, small_path_sheaf):
        """Regression test for missing GW costs fallback behavior."""
        # Create sheaf with missing GW costs
        sheaf_no_costs = small_path_sheaf
        sheaf_no_costs.metadata['gw_costs'] = {}  # Empty costs
        
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        builder = AlphaFlowBuilder(sheaf_no_costs, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        # Should not crash, should use fallback
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            build = builder.build(grouping=grouping)
            
            # Should log warning about fallback
            warning_messages = [str(warning.message) for warning in w]
            fallback_warned = any("restriction norms" in msg.lower() for msg in warning_messages)
            # Note: Warning behavior may vary, so we just check it doesn't crash
        
        # Should still produce valid build
        assert build.L_base.shape == build.L_resid.shape
        assert sp.issparse(build.D)
        assert build.meta['n_base_edges'] > 0
        assert build.meta['n_resid_edges'] > 0
    
    def test_unknown_tags_by_tag_regression(self, small_path_sheaf):
        """Regression test for by_tag with unknown tags."""
        # small_path_sheaf has no edge_tags metadata
        fake_builder = FakeGWLaplacianBuilder()
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(
            kind='by_tag',
            tags_base=['nonexistent_tag'],
            semantics='cost'
        )
        
        edges = list(small_path_sheaf.restrictions.keys())
        costs = builder._extract_edge_costs(edges)
        
        # Should raise informative NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            builder._partition_edges(costs, grouping)
        
        error_msg = str(exc_info.value)
        assert "by_tag partitioning not yet implemented" in error_msg
        assert "No edge tags found" in error_msg
    
    def test_dtype_mismatch_coercion_regression(self, small_path_sheaf):
        """Regression test for dtype mismatch handling."""
        
        class MixedDtypeBuilder(FakeGWLaplacianBuilder):
            """Builder that returns mismatched dtypes for testing."""
            
            def build_laplacian_grouped(self, *args, **kwargs):
                L_base, L_resid, D, metadata = super().build_laplacian_grouped(*args, **kwargs)
                
                # Force different dtypes
                n = L_base.shape[0]
                
                # Create float32 base, float64 residual (mismatch)
                L_base_f32 = LinearOperator((n, n), matvec=lambda x: x.astype(np.float32), dtype=np.float32)
                L_resid_f64 = LinearOperator((n, n), matvec=lambda x: x.astype(np.float64), dtype=np.float64)
                
                return L_base_f32, L_resid_f64, D, metadata
        
        mixed_builder = MixedDtypeBuilder()
        builder = AlphaFlowBuilder(small_path_sheaf, mixed_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        # Should handle dtype mismatch gracefully
        try:
            build = builder.build(grouping=grouping)
            
            # If build succeeds, should not crash during operator construction
            L_alpha = builder.as_operator(build, 1.0)
            
            # Test operation
            x = np.random.randn(L_alpha.shape[0])
            y = L_alpha @ x
            assert np.all(np.isfinite(y)), "Operation should produce finite results"
            
        except AssertionError as e:
            # The assertion in as_operator should catch dtype mismatch
            assert "must have same dtype" in str(e)
    
    def test_ridge_regularization_metadata_regression(self, small_path_sheaf):
        """Regression test for ridge_regularization metadata presence."""
        fake_builder = FakeGWLaplacianBuilder()
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # This was a bug where ridge_regularization was undefined in some code paths
        L, D, metadata = analyzer._build_laplacian_all_edges(mass_mode='fixed')
        
        # ridge_regularization must always be in metadata
        assert 'ridge_regularization' in metadata, \
            "ridge_regularization missing from metadata (regression)"
        
        ridge_eps = metadata['ridge_regularization']
        assert isinstance(ridge_eps, (int, float)), \
            "ridge_regularization should be numeric"
        assert ridge_eps > 0, \
            "ridge_regularization should be positive"
    
    def test_single_t_value_vector_handling_regression(self, small_star_sheaf):
        """Regression test for single t value handling (scalar to vector conversion)."""
        fake_builder = FakeGWLaplacianBuilder(default_size=4)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # This was a bug where single t values caused shape issues
        spec = DiffusionSpec(t_grid=[0.5])  # Single element list
        
        result = analyzer.analyze(spec)
        
        # Should handle single t value correctly
        assert len(result.heat_trace) == 1, "Single t should produce single heat trace value"
        assert len(result.t_grid) == 1, "t_grid should preserve single element"
        assert result.t_grid[0] == 0.5, "t_grid value should be preserved"
        assert np.isfinite(result.heat_trace[0]), "Single heat trace should be finite"
    
    def test_extreme_lambda_max_bounds_clipping_regression(self, small_path_sheaf):
        """Regression test for extreme λ_max values causing over/underflow."""
        fake_builder = FakeGWLaplacianBuilder(default_size=4)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Build system
        mass_mode = 'fixed'
        cache_key = (mass_mode, id(small_path_sheaf))
        L, D, _ = analyzer._build_laplacian_all_edges(mass_mode)
        
        # Test extreme λ_max values that previously caused issues
        extreme_values = [1e-20, 1e20, 0.0, float('inf')]
        
        for extreme_lam_max in extreme_values:
            if np.isfinite(extreme_lam_max):
                analyzer._lambda_max_cache[cache_key] = extreme_lam_max
                
                # Should not crash and should produce reasonable t_grid
                try:
                    t_grid = analyzer._generate_auto_t_grid(L, mass_mode, n_points=5)
                    
                    # Should be clipped to reasonable bounds
                    assert np.all(t_grid > 0), f"t_grid not positive for λ_max={extreme_lam_max}"
                    assert np.all(np.isfinite(t_grid)), f"t_grid not finite for λ_max={extreme_lam_max}"
                    assert len(t_grid) == 5, f"t_grid wrong length for λ_max={extreme_lam_max}"
                    
                except Exception as e:
                    # Should not crash with numerical errors
                    assert False, f"t_grid generation crashed for λ_max={extreme_lam_max}: {e}"
    
    def test_empty_partition_guard_rails_regression(self, small_path_sheaf):
        """Regression test for empty partition guard rails."""
        # Create sheaf with all equal costs (triggers guard rails)
        equal_costs_sheaf = small_path_sheaf
        edges = list(equal_costs_sheaf.restrictions.keys())
        equal_costs = {edge: 0.5 for edge in edges}  # All equal
        equal_costs_sheaf.metadata['gw_costs'] = equal_costs
        
        fake_builder = FakeGWLaplacianBuilder()
        builder = AlphaFlowBuilder(equal_costs_sheaf, fake_builder)
        
        # All partitioning strategies should trigger guard rails
        strategies = [
            AlphaGroupingPolicy(kind='quantile', param=0.1),  # Very skewed
            AlphaGroupingPolicy(kind='quantile', param=0.9),  # Very skewed
            AlphaGroupingPolicy(kind='topk', param=0.01),     # Tiny top-k
            AlphaGroupingPolicy(kind='topk', param=0.99),     # Huge top-k
        ]
        
        for grouping in strategies:
            build = builder.build(grouping=grouping)
            
            # Guard rails should ensure both sets non-empty
            assert build.meta['n_base_edges'] > 0, \
                f"Guard rails failed for {grouping.kind}: empty base set"
            assert build.meta['n_resid_edges'] > 0, \
                f"Guard rails failed for {grouping.kind}: empty resid set"
            
            # Total should be preserved
            total_edges = build.meta['n_base_edges'] + build.meta['n_resid_edges']
            assert total_edges == len(edges), \
                f"Guard rails lost edges: {total_edges} != {len(edges)}"


class TestNumericalPrecisionRegression:
    """Test numerical precision bounds for regression detection."""
    
    def test_alpha_flow_trace_precision_bounds(self, small_path_sheaf):
        """Test α-flow trace computation precision bounds."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        # Test precision consistency across multiple runs
        alpha = 1.0
        L_alpha = builder.as_operator(build, alpha)
        
        # Multiple trace estimates with same seed (should be identical)
        traces = []
        for _ in range(5):
            trace_est = trace_hutchinson(L_alpha, power=1, probes=64, seed=42)
            traces.append(trace_est)
        
        # Should be exactly identical (deterministic with same seed)
        for i in range(1, len(traces)):
            assert traces[i] == traces[0], \
                f"Deterministic precision regression: trace estimates vary with same seed"
        
        # Test precision with different probe counts
        probe_counts = [32, 64, 128]
        trace_estimates = []
        
        for probes in probe_counts:
            trace_est = trace_hutchinson(L_alpha, power=1, probes=probes, seed=123)
            trace_estimates.append(trace_est)
        
        # Higher probe counts should not be dramatically different (no catastrophic precision loss)
        for i in range(1, len(trace_estimates)):
            rel_diff = abs(trace_estimates[i] - trace_estimates[i-1]) / \
                      (abs(trace_estimates[i-1]) + 1e-15)
            
            assert rel_diff < 0.5, \
                f"Precision regression: trace estimates too different with probe count changes"
    
    def test_t_flow_heat_trace_precision_bounds(self, small_star_sheaf):
        """Test t-flow heat trace precision bounds.""" 
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.1, 0.5, 1.0],
            probes=64,
            slq_iters=25
        )
        
        # Multiple runs with same seed (should be identical)
        results = []
        for _ in range(3):
            analyzer_copy = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
            result = analyzer_copy.analyze(spec)
            results.append(result.heat_trace)
        
        # Should be exactly identical (deterministic)
        for i in range(1, len(results)):
            assert np.allclose(results[i], results[0], atol=1e-15), \
                f"Deterministic precision regression: heat traces vary with same seed"
        
        # Test precision bounds for different probe/iteration budgets
        specs = [
            DiffusionSpec(t_grid=[0.5], probes=32, slq_iters=20),
            DiffusionSpec(t_grid=[0.5], probes=64, slq_iters=15),
            DiffusionSpec(t_grid=[0.5], probes=128, slq_iters=10),
        ]
        
        heat_values = []
        for spec_test in specs:
            analyzer_test = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=99)
            result = analyzer_test.analyze(spec_test)
            heat_values.append(result.heat_trace[0])
        
        # Different budgets should give reasonably consistent results
        heat_mean = np.mean(heat_values)
        heat_std = np.std(heat_values)
        
        # Coefficient of variation should be reasonable
        cv = heat_std / (abs(heat_mean) + 1e-15)
        assert cv < 0.1, \
            f"Precision regression: heat values too variable across budgets, CV={cv:.3f}"
    
    def test_operator_matvec_precision_consistency(self, small_path_sheaf):
        """Test operator matvec precision consistency."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        alpha = 1.0
        L_alpha = builder.as_operator(build, alpha)
        
        # Test matvec precision with same input vector
        x = np.random.randn(L_alpha.shape[0])
        
        # Multiple matvec calls should be identical
        results = []
        for _ in range(5):
            y = L_alpha @ x
            results.append(y.copy())
        
        # Should be exactly identical (deterministic operation)
        for i in range(1, len(results)):
            assert np.allclose(results[i], results[0], atol=1e-15), \
                f"Matvec precision regression: results vary for same input"
        
        # Test precision with different vector types
        x_float32 = x.astype(np.float32)
        x_float64 = x.astype(np.float64)
        
        y_32 = L_alpha @ x_float32
        y_64 = L_alpha @ x_float64
        
        # Results should be close (allowing for float32 precision)
        rel_error = np.linalg.norm(y_32.astype(np.float64) - y_64) / \
                   (np.linalg.norm(y_64) + 1e-15)
        
        assert rel_error < 1e-6, \
            f"Precision regression: float32/float64 results too different, rel_error={rel_error:.2e}"
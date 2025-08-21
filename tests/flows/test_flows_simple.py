"""Simple tests using fallback fixtures to verify flow implementations work.

This module provides a basic smoke test of the flow implementations using
the fallback fixtures when the full NeurosheafAnalyzer is not available.
"""

import pytest
import numpy as np
import warnings

from neurosheaf.spectral.flows.alpha_flow import (
    AlphaGroupingPolicy, AlphaFlowBuilder
)
from neurosheaf.spectral.flows.diffusion_flow import (
    DiffusionSpec, DiffusionFlowAnalyzer
)

from .fixtures import (
    FakeGWLaplacianBuilder, fallback_path_sheaf, fallback_star_sheaf,
    trace_hutchinson, linear_operator_equals_sparse
)


class TestAlphaFlowBasicFunctionality:
    """Basic smoke tests for α-flow functionality."""
    
    def test_alpha_flow_build_and_evaluate(self, fallback_path_sheaf):
        """Test complete α-flow build and evaluation pipeline."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        builder = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        
        # Test different grouping strategies
        groupings = [
            AlphaGroupingPolicy(kind='quantile', param=0.5, semantics='cost'),
            AlphaGroupingPolicy(kind='topk', param=0.6, semantics='cost'),
        ]
        
        for grouping in groupings:
            build = builder.build(grouping=grouping)
            
            # Verify build structure
            assert build.L_base.shape == build.L_resid.shape
            assert build.L_base.shape[0] == build.L_base.shape[1]  # Square
            assert build.meta['n_base_edges'] > 0
            assert build.meta['n_resid_edges'] > 0
            
            # Test operator evaluation
            alpha_values = [0.0, 0.5, 1.0]
            traces = []
            
            for alpha in alpha_values:
                L_alpha = builder.as_operator(build, alpha)
                assert L_alpha.shape == build.L_base.shape
                
                # Test matvec operation
                x = np.random.randn(L_alpha.shape[0])
                y = L_alpha @ x
                assert np.all(np.isfinite(y))
                
                # Estimate trace
                trace_est = trace_hutchinson(L_alpha, power=1, probes=32, seed=42)
                traces.append(trace_est)
            
            # Check monotonicity
            for i in range(1, len(traces)):
                assert traces[i] >= traces[i-1] - 1e-6, \
                    f"Monotonicity violation: {traces[i-1]} -> {traces[i]}"
    
    def test_alpha_flow_edge_partitioning(self, fallback_star_sheaf):
        """Test edge partitioning with by_tag strategy."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        builder = AlphaFlowBuilder(fallback_star_sheaf, fake_builder)
        
        # Test by_tag partitioning (should work with fallback_star_sheaf)
        grouping = AlphaGroupingPolicy(
            kind='by_tag',
            tags_base=['primary'],
            semantics='cost'
        )
        
        build = builder.build(grouping=grouping)
        
        # Should successfully partition based on tags
        assert build.meta['n_base_edges'] > 0
        assert build.meta['n_resid_edges'] > 0
        
        # Check edge assignments
        base_edges = build.meta['base_edges']
        resid_edges = build.meta['resid_edges']
        
        # Verify no overlap
        assert not (set(base_edges) & set(resid_edges))
        
        # Total should match sheaf edges
        total_edges = len(base_edges) + len(resid_edges)
        sheaf_edges = len(fallback_star_sheaf.restrictions)
        assert total_edges == sheaf_edges
    
    def test_alpha_flow_error_handling(self):
        """Test error handling for invalid parameters."""
        from .fixtures import create_minimal_sheaf_fallback
        
        sheaf = create_minimal_sheaf_fallback('test')
        fake_builder = FakeGWLaplacianBuilder()
        builder = AlphaFlowBuilder(sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        # Test negative alpha error
        with pytest.raises(ValueError, match="Alpha must be non-negative"):
            builder.as_operator(build, -1.0)


class TestTFlowBasicFunctionality:
    """Basic smoke tests for t-flow functionality."""
    
    def test_t_flow_build_and_analyze(self, fallback_path_sheaf):
        """Test complete t-flow analysis pipeline."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(fallback_path_sheaf, fake_builder, random_seed=42)
        
        # Test explicit t-grid
        spec = DiffusionSpec(
            t_grid=[0.1, 0.5, 1.0],
            k_small=2,
            probes=32,
            slq_iters=15
        )
        
        result = analyzer.analyze(spec, mass_mode='fixed')
        
        # Verify result structure
        assert len(result.heat_trace) == len(spec.t_grid)
        assert len(result.t_grid) == len(spec.t_grid)
        assert np.allclose(result.t_grid, spec.t_grid)
        
        # Check heat trace properties
        assert np.all(np.isfinite(result.heat_trace))
        assert np.all(result.heat_trace >= -1e-6)  # Should be non-negative
        
        # Check metadata
        assert 'analysis_time' in result.meta
        assert 'n_time_points' in result.meta
        assert 'probes' in result.meta
        assert result.meta['probes'] == spec.probes
    
    def test_t_flow_auto_grid_generation(self, fallback_star_sheaf):
        """Test automatic t-grid generation."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(fallback_star_sheaf, fake_builder, random_seed=42)
        
        # Test auto t-grid
        spec = DiffusionSpec(
            t_grid='auto',
            probes=16,
            slq_iters=10
        )
        
        result = analyzer.analyze(spec)
        
        # Should generate reasonable t-grid
        assert len(result.t_grid) == 20  # Default auto length
        assert np.all(result.t_grid > 0)  # All positive
        assert np.all(np.diff(result.t_grid) > 0)  # Strictly increasing
        
        # Check λ_max was estimated
        assert analyzer._lambda_max_cache is not None
        assert analyzer._lambda_max_cache > 0
    
    def test_t_flow_caching_behavior(self, fallback_path_sheaf):
        """Test that caching works correctly."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(fallback_path_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(t_grid=[0.5], probes=16, slq_iters=10)
        
        # First call
        initial_call_count = fake_builder._call_count
        result1 = analyzer.analyze(spec)
        after_first_call = fake_builder._call_count
        
        # Second call should use cache
        result2 = analyzer.analyze(spec)
        after_second_call = fake_builder._call_count
        
        # Builder should only be called once
        assert after_first_call > initial_call_count, "First call should invoke builder"
        assert after_second_call == after_first_call, "Second call should use cache"
        
        # Results should be similar (SLQ is stochastic but with same seed should be close)
        rel_diff = np.abs(result1.heat_trace - result2.heat_trace) / (np.abs(result1.heat_trace) + 1e-15)
        assert np.max(rel_diff) < 0.1, f"Heat traces too different: {result1.heat_trace} vs {result2.heat_trace}"
    
    def test_t_flow_error_handling(self):
        """Test error handling for invalid specifications."""
        # Test invalid DiffusionSpec parameters
        with pytest.raises(ValueError):
            DiffusionSpec(t_grid=[-0.1, 1.0])  # Negative t
            
        with pytest.raises(ValueError):
            DiffusionSpec(probes=0)  # Zero probes
            
        with pytest.raises(ValueError):
            DiffusionSpec(k_small=-1)  # Negative k_small


class TestCrossFlowBasicProperties:
    """Test basic mathematical properties that should hold across flows."""
    
    def test_determinism_with_fixed_seed(self, fallback_path_sheaf):
        """Test that results are deterministic with fixed random seeds."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        
        # α-flow determinism
        builder1 = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        builder2 = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        build1 = builder1.build(grouping=grouping)
        build2 = builder2.build(grouping=grouping)
        
        # Operator should give same results with same input
        L1 = builder1.as_operator(build1, 1.0)
        L2 = builder2.as_operator(build2, 1.0)
        
        x = np.random.randn(L1.shape[0])
        y1 = L1 @ x
        y2 = L2 @ x
        
        assert np.allclose(y1, y2, atol=1e-14), "α-flow should be deterministic"
        
        # t-flow determinism
        analyzer1 = DiffusionFlowAnalyzer(fallback_path_sheaf, fake_builder, random_seed=42)
        analyzer2 = DiffusionFlowAnalyzer(fallback_path_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(t_grid=[0.5], probes=32, slq_iters=15)
        
        result1 = analyzer1.analyze(spec)
        result2 = analyzer2.analyze(spec)
        
        assert np.allclose(result1.heat_trace, result2.heat_trace, atol=1e-10), \
            "t-flow should be deterministic with same seed"
    
    def test_matrix_properties(self, fallback_star_sheaf):
        """Test basic matrix properties of constructed operators."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        
        # α-flow: test symmetry via random probes
        builder = AlphaFlowBuilder(fallback_star_sheaf, fake_builder)
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        L_alpha = builder.as_operator(build, 1.0)
        
        # Test symmetry: <x, L*y> = <L*x, y>
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.standard_normal(L_alpha.shape[0])
            y = rng.standard_normal(L_alpha.shape[0])
            
            xLy = x.T @ (L_alpha @ y)
            Lxy = (L_alpha @ x).T @ y
            
            rel_error = abs(xLy - Lxy) / (abs(xLy) + 1e-15)
            assert rel_error < 1e-12, f"Symmetry violation: rel_error={rel_error:.2e}"
        
        # t-flow: test Laplacian is symmetric
        analyzer = DiffusionFlowAnalyzer(fallback_star_sheaf, fake_builder, random_seed=42)
        L, D, metadata = analyzer._build_laplacian_all_edges()
        
        # Test symmetry of wrapped LinearOperator
        for _ in range(3):
            x = rng.standard_normal(L.shape[0])
            y = rng.standard_normal(L.shape[0])
            
            xLy = x.T @ (L @ y)
            Lxy = (L @ x).T @ y
            
            rel_error = abs(xLy - Lxy) / (abs(xLy) + 1e-15)
            assert rel_error < 1e-12, f"t-flow Laplacian symmetry violation: {rel_error:.2e}"
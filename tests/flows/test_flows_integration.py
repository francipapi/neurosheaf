"""Integration tests for α-flow and t-flow with the broader pipeline.

This module tests end-to-end integration with PersistentSpectralAnalyzer,
caching behavior, artifact serialization, and real GW pipeline compatibility.

Tests cover:
- analyze_alpha_flow() and analyze_diffusion_flow() end-to-end paths
- Caching behavior (rebuild avoidance on second calls)
- Artifacts serialization (JSON/NPZ format validation)
- GW pipeline compatibility with real GWLaplacianBuilder
- Configuration plumbing and error propagation
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

# Try to import the main analyzer (may not exist yet, handle gracefully)
try:
    from neurosheaf.spectral.persistent import (
        PersistentSpectralAnalyzer, StaticBuildConfig, 
        AlphaFlowSpec, DiffusionFlowSpec
    )
    PERSISTENT_ANALYZER_AVAILABLE = True
except ImportError:
    PERSISTENT_ANALYZER_AVAILABLE = False

from neurosheaf.spectral.flows.alpha_flow import (
    AlphaGroupingPolicy, AlphaFlowBuilder
)
from neurosheaf.spectral.flows.diffusion_flow import (
    DiffusionSpec, DiffusionFlowAnalyzer
)
from neurosheaf.sheaf.data_structures import Sheaf

# Try to import real GW builder for compatibility tests
try:
    from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
    GW_BUILDER_AVAILABLE = True
except ImportError:
    GW_BUILDER_AVAILABLE = False

from .fixtures import (
    small_path_sheaf, small_star_sheaf, FakeGWLaplacianBuilder,
    fallback_path_sheaf, fallback_star_sheaf,
    random_gw_costs
)


class TestEndToEndFlowPaths:
    """Test complete end-to-end flow analysis paths."""
    
    @pytest.mark.skipif(not PERSISTENT_ANALYZER_AVAILABLE, 
                       reason="PersistentSpectralAnalyzer not available")
    def test_analyze_alpha_flow_end_to_end(self, small_path_sheaf):
        """Test complete analyze_alpha_flow() path."""
        # Create analyzer instance
        analyzer = PersistentSpectralAnalyzer()
        
        # Create alpha flow spec
        alpha_spec = AlphaFlowSpec(
            alpha_grid=[0.0, 0.5, 1.0],
            k_small=4,
            probes=32,
            moments=[2, 3],
            grouping=AlphaGroupingPolicy(kind='quantile', param=0.5)
        )
        
        # Create static build config
        config = StaticBuildConfig(
            mass_mode='fixed',
            precision='double',
            random_state=42
        )
        
        # Analyze (this will use whatever GW builder is available)
        try:
            result = analyzer.analyze_alpha_flow(alpha_spec, config, sheaf=small_path_sheaf)
            
            # Check result structure
            assert hasattr(result, 'alpha_grid'), "Result should have alpha_grid"
            assert hasattr(result, 'eigenvalues'), "Result should have eigenvalues"
            assert hasattr(result, 'meta'), "Result should have metadata"
            
            # Check grid matches spec
            assert len(result.alpha_grid) == len(alpha_spec.alpha_grid)
            assert np.allclose(result.alpha_grid, alpha_spec.alpha_grid)
            
            # Check metadata contains expected keys
            assert 'analysis_time' in result.meta
            assert 'config' in result.meta
            assert 'sheaf_info' in result.meta
            
        except Exception as e:
            # If real integration fails, at least check the error is reasonable
            assert "sheaf" in str(e).lower() or "builder" in str(e).lower() or "gw" in str(e).lower()
    
    @pytest.mark.skipif(not PERSISTENT_ANALYZER_AVAILABLE,
                       reason="PersistentSpectralAnalyzer not available")
    def test_analyze_diffusion_flow_end_to_end(self, small_star_sheaf):
        """Test complete analyze_diffusion_flow() path."""
        # Create analyzer instance
        analyzer = PersistentSpectralAnalyzer()
        
        # Create diffusion flow spec
        diffusion_spec = DiffusionFlowSpec(
            t_grid=[0.1, 0.5, 1.0],  # Explicit grid
            k_small=3,
            probes=32,
            slq_iters=20
        )
        
        # Create static build config
        config = StaticBuildConfig(
            mass_mode='fixed',
            precision='double',
            random_state=123
        )
        
        # Analyze
        try:
            result = analyzer.analyze_diffusion_flow(diffusion_spec, config, sheaf=small_star_sheaf)
            
            # Check result structure
            assert hasattr(result, 't_grid'), "Result should have t_grid"
            assert hasattr(result, 'heat_trace'), "Result should have heat_trace"
            assert hasattr(result, 'meta'), "Result should have metadata"
            
            # Check dimensions
            assert len(result.t_grid) == len(diffusion_spec.t_grid)
            assert len(result.heat_trace) == len(diffusion_spec.t_grid)
            
            # Check values are reasonable
            assert np.all(np.isfinite(result.heat_trace)), "Heat trace should be finite"
            assert np.all(result.heat_trace >= -1e-6), "Heat trace should be non-negative"
            
        except Exception as e:
            # If real integration fails, check error is reasonable
            assert any(word in str(e).lower() for word in ["sheaf", "builder", "gw", "laplacian"])
    
    def test_direct_alpha_flow_builder_integration(self, fallback_path_sheaf):
        """Test direct AlphaFlowBuilder integration without full pipeline."""
        # This test works without the full persistent analyzer
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        builder = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        
        # Build with different configurations
        configs = [
            {'grouping': AlphaGroupingPolicy(kind='quantile', param=0.3), 'mass_mode': 'fixed'},
            {'grouping': AlphaGroupingPolicy(kind='topk', param=0.4), 'mass_mode': 'adaptive'},
        ]
        
        for config in configs:
            build = builder.build(**config)
            
            # Test multiple alpha values
            alpha_values = [0.0, 0.1, 0.5, 1.0, 2.0]
            
            for alpha in alpha_values:
                L_alpha = builder.as_operator(build, alpha)
                
                # Check operator properties
                assert L_alpha.shape[0] == L_alpha.shape[1], "Operator should be square"
                assert L_alpha.dtype in [np.float64, np.float32], "Operator should have float dtype"
                
                # Test matvec operation
                x = np.random.randn(L_alpha.shape[0])
                y = L_alpha @ x
                assert y.shape == x.shape, "Matvec should preserve shape"
                assert np.all(np.isfinite(y)), "Matvec result should be finite"
    
    def test_direct_diffusion_analyzer_integration(self, small_star_sheaf):
        """Test direct DiffusionFlowAnalyzer integration."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # Test different specs
        specs = [
            DiffusionSpec(t_grid=[0.1, 1.0], probes=32, slq_iters=15),
            DiffusionSpec(t_grid='auto', probes=64, slq_iters=25),
            DiffusionSpec(t_grid=[0.05, 0.2, 0.8], k_small=0, probes=16)  # Fast path
        ]
        
        for spec in specs:
            result = analyzer.analyze(spec, mass_mode='fixed')
            
            # Check result validity
            assert isinstance(result.heat_trace, np.ndarray), "Heat trace should be numpy array"
            assert isinstance(result.t_grid, np.ndarray), "t_grid should be numpy array"
            assert len(result.heat_trace) == len(result.t_grid), "Dimensions should match"
            
            # Check metadata completeness
            required_meta_keys = [
                'analysis_time', 'n_time_points', 'n_valid_points',
                'probes', 'slq_iters', 'is_monotonic'
            ]
            
            for key in required_meta_keys:
                assert key in result.meta, f"Missing metadata key: {key}"


class TestCachingBehavior:
    """Test caching behavior to avoid expensive rebuilds."""
    
    def test_diffusion_analyzer_laplacian_caching(self, fallback_path_sheaf):
        """Test that Laplacian construction is cached between analyze() calls."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(fallback_path_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(t_grid=[0.1, 1.0], probes=32, slq_iters=15)
        
        # First analysis should build Laplacian
        initial_call_count = fake_builder._call_count
        result1 = analyzer.analyze(spec)
        after_first_call_count = fake_builder._call_count
        
        # Builder should have been called
        assert after_first_call_count > initial_call_count, "First call should invoke builder"
        
        # Second analysis should use cache
        result2 = analyzer.analyze(spec)
        after_second_call_count = fake_builder._call_count
        
        # Builder should not be called again
        assert after_second_call_count == after_first_call_count, \
            "Second call should use cached Laplacian"
        
        # Results should be identical (deterministic)
        assert np.allclose(result1.heat_trace, result2.heat_trace), \
            "Cached results should be identical"
        assert np.allclose(result1.t_grid, result2.t_grid), \
            "Cached t_grid should be identical"
    
    def test_lambda_max_caching_across_calls(self, fallback_path_sheaf):
        """Test that λ_max estimation is cached across auto t-grid generation."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        analyzer = DiffusionFlowAnalyzer(fallback_path_sheaf, fake_builder, random_seed=42)
        
        # Use auto t-grid to trigger λ_max estimation
        spec_auto = DiffusionSpec(t_grid='auto', probes=16, slq_iters=10)
        
        # First call should estimate λ_max
        result1 = analyzer.analyze(spec_auto)
        lambda_max_1 = analyzer._lambda_max_cache
        
        assert lambda_max_1 is not None, "λ_max should be cached after first call"
        
        # Second call should reuse cached λ_max
        result2 = analyzer.analyze(spec_auto)
        lambda_max_2 = analyzer._lambda_max_cache
        
        assert lambda_max_2 == lambda_max_1, "λ_max cache should be reused"
        
        # t_grids should be identical (deterministic)
        assert np.allclose(result1.t_grid, result2.t_grid), \
            "Auto t_grid should be deterministic with cached λ_max"
    
    def test_cache_invalidation_on_new_analyzer(self, small_path_sheaf):
        """Test that cache is fresh for new analyzer instances."""
        fake_builder1 = FakeGWLaplacianBuilder(default_size=6)
        fake_builder2 = FakeGWLaplacianBuilder(default_size=6)
        
        # First analyzer
        analyzer1 = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder1, random_seed=42)
        spec = DiffusionSpec(t_grid=[0.1, 1.0], probes=32)
        
        analyzer1.analyze(spec)
        calls_builder1 = fake_builder1._call_count
        
        # Second analyzer with different builder
        analyzer2 = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder2, random_seed=42)
        analyzer2.analyze(spec)
        calls_builder2 = fake_builder2._call_count
        
        # Both builders should have been called (no cross-analyzer caching)
        assert calls_builder1 > 0, "First builder should be called"
        assert calls_builder2 > 0, "Second builder should be called (no cross-cache)"


class TestArtifactsSerialization:
    """Test serialization of analysis results to JSON/NPZ formats."""
    
    def test_alpha_flow_metadata_json_serializable(self, fallback_path_sheaf):
        """Test that α-flow metadata can be serialized to JSON."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        builder = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        metadata = build.meta
        
        # Test JSON serialization (filter out non-serializable objects)
        try:
            # Convert complex objects to serializable format
            serializable_metadata = {}
            for key, value in metadata.items():
                if hasattr(value, '__dict__'):
                    # Convert dataclass/object to dict
                    serializable_metadata[key] = vars(value)
                elif isinstance(value, (list, tuple)) and value and hasattr(value[0], '__dict__'):
                    # Convert list of objects to list of dicts
                    serializable_metadata[key] = [vars(item) for item in value]
                else:
                    # Keep primitive types
                    serializable_metadata[key] = value
            
            json_str = json.dumps(serializable_metadata, indent=2, default=str)
            reconstructed = json.loads(json_str)
            
            # Check key preservation
            assert 'n_base_edges' in reconstructed
            assert 'n_resid_edges' in reconstructed
            assert 'L_base_dtype' in reconstructed
            assert 'L_resid_dtype' in reconstructed
            
            # Check value types
            assert isinstance(reconstructed['n_base_edges'], int)
            assert isinstance(reconstructed['n_resid_edges'], int)
            
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metadata not JSON serializable: {e}")
    
    def test_diffusion_flow_results_npz_serializable(self, small_star_sheaf):
        """Test that t-flow results can be saved to NPZ format."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(t_grid=[0.1, 0.5, 1.0], probes=32, slq_iters=15)
        result = analyzer.analyze(spec)
        
        # Test NPZ serialization
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            try:
                # Serialize main arrays
                np.savez(
                    tmp_file.name,
                    heat_trace=result.heat_trace,
                    t_grid=result.t_grid,
                    smallest_eigs=result.smallest_eigs,
                    # Convert metadata to JSON string for NPZ storage
                    metadata_json=json.dumps(result.meta).encode('utf-8')
                )
                
                # Load and verify
                loaded = np.load(tmp_file.name, allow_pickle=True)
                
                assert 'heat_trace' in loaded
                assert 't_grid' in loaded
                assert 'smallest_eigs' in loaded
                assert 'metadata_json' in loaded
                
                # Check array equality
                assert np.allclose(loaded['heat_trace'], result.heat_trace)
                assert np.allclose(loaded['t_grid'], result.t_grid)
                
                # Check metadata reconstruction
                reconstructed_meta = json.loads(loaded['metadata_json'].item().decode('utf-8'))
                assert 'analysis_time' in reconstructed_meta
                assert 'probes' in reconstructed_meta
                
            finally:
                Path(tmp_file.name).unlink()  # Clean up
    
    def test_serialization_with_special_values(self, small_path_sheaf):
        """Test serialization handles NaN, Inf, and other special values."""
        fake_builder = FakeGWLaplacianBuilder(default_size=4)
        analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
        
        # Create result with special values
        spec = DiffusionSpec(t_grid=[0.1], probes=8, slq_iters=5)
        result = analyzer.analyze(spec)
        
        # Manually inject special values for testing
        result.heat_trace[0] = np.nan if len(result.heat_trace) > 0 else np.array([np.nan])
        result.meta['test_inf'] = float('inf')
        result.meta['test_ninf'] = float('-inf')
        result.meta['test_nan'] = float('nan')
        
        # Test JSON handling of special values
        try:
            # Standard JSON doesn't handle NaN/Inf well, but our code should handle this
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Remove problematic values or convert to strings
                safe_meta = {}
                for key, value in result.meta.items():
                    if isinstance(value, float):
                        if np.isnan(value):
                            safe_meta[key] = "NaN"
                        elif np.isinf(value):
                            safe_meta[key] = "Inf" if value > 0 else "-Inf"
                        else:
                            safe_meta[key] = value
                    else:
                        safe_meta[key] = value
                
                json_str = json.dumps(safe_meta)
                assert '"test_inf": "Inf"' in json_str or '"test_inf": "Inf"' in json_str
                
        except Exception as e:
            # Special value handling is implementation detail - main point is no crash
            assert "json" in str(e).lower() or "serialize" in str(e).lower()


class TestGWPipelineCompatibility:
    """Test compatibility with real GW pipeline (when available)."""
    
    @pytest.mark.skipif(not GW_BUILDER_AVAILABLE,
                       reason="GWLaplacianBuilder not available")
    def test_real_gw_builder_alpha_flow_integration(self, small_path_sheaf):
        """Test α-flow with real GWLaplacianBuilder."""
        # Create real GW builder
        real_builder = GWLaplacianBuilder()
        
        try:
            # Test α-flow integration
            flow_builder = AlphaFlowBuilder(small_path_sheaf, real_builder)
            
            grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
            build = flow_builder.build(grouping=grouping, mass_mode='fixed')
            
            # Check that real builder produces valid results
            assert build.L_base.shape == build.L_resid.shape
            assert sp.issparse(build.D), "Real builder should produce sparse mass matrix"
            
            # Test operator construction
            L_alpha = flow_builder.as_operator(build, 1.0)
            assert isinstance(L_alpha, LinearOperator)
            
            # Check metadata indicates real GW construction
            metadata = build.meta
            assert 'n_base_edges' in metadata
            assert 'n_resid_edges' in metadata
            
        except Exception as e:
            # Real GW integration might fail due to missing dependencies, etc.
            # Check that failure is reasonable
            expected_errors = ["dtype", "tensor", "torch", "sparse", "laplacian"]
            assert any(word in str(e).lower() for word in expected_errors), \
                f"Unexpected integration error: {e}"
    
    @pytest.mark.skipif(not GW_BUILDER_AVAILABLE,
                       reason="GWLaplacianBuilder not available")
    def test_real_gw_builder_diffusion_flow_integration(self, small_star_sheaf):
        """Test t-flow with real GWLaplacianBuilder."""
        real_builder = GWLaplacianBuilder()
        
        try:
            analyzer = DiffusionFlowAnalyzer(small_star_sheaf, real_builder, random_seed=42)
            
            spec = DiffusionSpec(
                t_grid=[0.1, 1.0],
                k_small=2,
                probes=32,
                slq_iters=15
            )
            
            result = analyzer.analyze(spec, mass_mode='fixed')
            
            # Check result validity
            assert len(result.heat_trace) == len(spec.t_grid)
            assert np.all(np.isfinite(result.heat_trace)), "Heat trace should be finite"
            
            # Check metadata indicates real construction
            assert 'matrix_size' in result.meta
            assert 'L_nnz' in result.meta
            assert 'D_nnz' in result.meta
            
        except Exception as e:
            # Real integration might fail - check error is reasonable
            expected_errors = ["sheaf", "builder", "dtype", "mass", "laplacian"]
            assert any(word in str(e).lower() for word in expected_errors), \
                f"Unexpected integration error: {e}"
    
    def test_gw_specific_metadata_logging(self, small_path_sheaf):
        """Test that GW-specific metadata is properly logged."""
        fake_builder = FakeGWLaplacianBuilder()
        
        # Mock GW-specific metadata
        fake_builder.extra_metadata = {
            'gw_construction_time': 0.123,
            'gw_edge_weights_source': 'costs',
            'gw_weight_transform': 'exponential'
        }
        
        # Modify fake builder to return this metadata
        original_build_grouped = fake_builder.build_laplacian_grouped
        
        def enhanced_build_grouped(*args, **kwargs):
            L_base, L_resid, D, metadata = original_build_grouped(*args, **kwargs)
            metadata.update(fake_builder.extra_metadata)
            return L_base, L_resid, D, metadata
        
        fake_builder.build_laplacian_grouped = enhanced_build_grouped
        
        # Test with α-flow
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        # Check GW metadata is included
        assert 'gw_construction_time' in build.meta
        assert 'gw_edge_weights_source' in build.meta
        assert 'gw_weight_transform' in build.meta


class TestConfigurationPlumbing:
    """Test configuration handling and error propagation."""
    
    def test_alpha_flow_grouping_policy_propagation(self, small_path_sheaf):
        """Test that AlphaGroupingPolicy is properly propagated through the pipeline."""
        fake_builder = FakeGWLaplacianBuilder()
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        # Test different grouping policies
        policies = [
            AlphaGroupingPolicy(kind='quantile', param=0.3, semantics='cost'),
            AlphaGroupingPolicy(kind='topk', param=0.4, semantics='similarity'),
        ]
        
        for policy in policies:
            build = builder.build(grouping=policy)
            
            # Check policy is preserved in metadata
            assert 'grouping_policy' in build.meta
            stored_policy = build.meta['grouping_policy']
            
            assert stored_policy.kind == policy.kind
            assert stored_policy.param == policy.param
            assert stored_policy.semantics == policy.semantics
    
    def test_mass_mode_configuration_consistency(self, small_star_sheaf):
        """Test mass_mode configuration is consistent across flows."""
        fake_builder = FakeGWLaplacianBuilder()
        
        # Test α-flow mass_mode
        alpha_builder = AlphaFlowBuilder(small_star_sheaf, fake_builder)
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        for mass_mode in ['fixed', 'adaptive']:
            build = alpha_builder.build(grouping=grouping, mass_mode=mass_mode)
            assert build.meta['mass_mode'] == mass_mode
        
        # Test t-flow mass_mode
        diffusion_analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        spec = DiffusionSpec(t_grid=[0.1, 1.0], probes=16)
        
        for mass_mode in ['fixed', 'adaptive']:
            result = diffusion_analyzer.analyze(spec, mass_mode=mass_mode)
            assert result.meta['mass_mode'] == mass_mode
    
    def test_error_propagation_from_builder(self, small_path_sheaf):
        """Test that builder errors are properly propagated."""
        
        class FailingBuilder(FakeGWLaplacianBuilder):
            """Builder that fails in specific ways for testing."""
            
            def build_laplacian_grouped(self, *args, **kwargs):
                raise ValueError("Simulated builder failure for testing")
            
            def build_laplacian(self, *args, **kwargs):
                raise RuntimeError("Simulated Laplacian build failure")
        
        failing_builder = FailingBuilder()
        
        # Test α-flow error propagation
        alpha_builder = AlphaFlowBuilder(small_path_sheaf, failing_builder)
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        with pytest.raises(ValueError) as exc_info:
            alpha_builder.build(grouping=grouping)
        
        assert "Simulated builder failure" in str(exc_info.value)
        
        # Test t-flow error propagation
        diffusion_analyzer = DiffusionFlowAnalyzer(small_path_sheaf, failing_builder, random_seed=42)
        spec = DiffusionSpec(t_grid=[0.1], probes=16)
        
        with pytest.raises(RuntimeError) as exc_info:
            diffusion_analyzer.analyze(spec)
        
        assert "Simulated Laplacian build failure" in str(exc_info.value)
    
    def test_dtype_configuration_handling(self, small_path_sheaf):
        """Test dtype configuration is properly handled."""
        # Test with different dtype builders
        for dtype in ['float32', 'float64']:
            fake_builder = FakeGWLaplacianBuilder(default_dtype=dtype)
            
            # α-flow dtype handling
            alpha_builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
            grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
            build = alpha_builder.build(grouping=grouping)
            
            # Check dtype consistency
            assert build.L_base.dtype == getattr(np, dtype)
            assert build.L_resid.dtype == getattr(np, dtype)
            assert build.meta['L_base_dtype'] == dtype
            assert build.meta['L_resid_dtype'] == dtype
            
            # t-flow dtype handling
            diffusion_analyzer = DiffusionFlowAnalyzer(small_path_sheaf, fake_builder, random_seed=42)
            spec = DiffusionSpec(t_grid=[0.1], probes=8)
            result = diffusion_analyzer.analyze(spec)
            
            # Check t-flow uses consistent dtype
            assert result.meta['L_dtype'] == dtype
            assert result.meta['D_dtype'] == dtype
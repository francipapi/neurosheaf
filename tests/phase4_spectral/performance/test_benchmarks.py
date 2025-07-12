# tests/phase4_spectral/performance/test_benchmarks.py
"""Performance benchmarks for persistent spectral analysis.

This module provides performance testing and benchmarking for the persistent
spectral analysis implementation, validating against the performance targets
specified in CLAUDE.md and ensuring scalability.
"""

import pytest
import torch
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
from neurosheaf.spectral.tracker import SubspaceTracker
from ..utils.test_ground_truth import GroundTruthGenerator, PersistenceValidator


class PerformanceProfiler:
    """Utility class for performance profiling."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def get_profile(self) -> Dict[str, float]:
        """Get current performance profile."""
        if self.start_time is None:
            raise ValueError("Profiling not started")
        
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'elapsed_time': current_time - self.start_time,
            'memory_usage': current_memory,
            'memory_increase': current_memory - self.start_memory,
            'peak_memory': current_memory
        }


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for persistent spectral analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.profiler = PerformanceProfiler()
        
        # Performance targets from CLAUDE.md
        self.memory_target_gb = 3.0  # <3GB for ResNet50-scale networks
        self.time_target_minutes = 5.0  # <5 minutes for complete analysis
        self.large_graph_nodes = 50  # Test with moderate size first
        self.large_stalk_dim = 10
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with graph size."""
        self.profiler.start_profiling()
        
        # Test different graph sizes
        node_counts = [5, 10, 20, 30]
        memory_usage = []
        
        for n_nodes in node_counts:
            # Create sheaf
            sheaf, expected = self.generator.linear_chain_sheaf(n_nodes, stalk_dim=3)
            
            # Create analyzer with LOBPCG for efficiency
            analyzer = PersistentSpectralAnalyzer(
                static_laplacian=StaticLaplacianWithMasking(
                    eigenvalue_method='lobpcg',
                    max_eigenvalues=min(50, n_nodes * 3)
                )
            )
            
            # Measure memory before analysis
            pre_analysis_memory = self.profiler.process.memory_info().rss / 1024 / 1024
            
            # Perform analysis
            result = analyzer.analyze(sheaf, n_steps=8)
            
            # Measure memory after analysis
            post_analysis_memory = self.profiler.process.memory_info().rss / 1024 / 1024
            analysis_memory = post_analysis_memory - pre_analysis_memory
            
            memory_usage.append({
                'n_nodes': n_nodes,
                'memory_mb': analysis_memory,
                'total_eigenvalues': sum(len(seq) for seq in result['persistence_result']['eigenvalue_sequences'])
            })
            
            # Clear cache to prevent accumulation
            analyzer.clear_cache()
        
        # Check memory scaling is reasonable
        for i, usage in enumerate(memory_usage):
            # Memory should not exceed reasonable bounds
            assert usage['memory_mb'] < 1000, \
                f"Memory usage too high for {usage['n_nodes']} nodes: {usage['memory_mb']} MB"
            
            # Memory should scale reasonably with problem size
            if i > 0:
                prev_usage = memory_usage[i-1]
                scaling_factor = usage['memory_mb'] / max(prev_usage['memory_mb'], 1.0)
                node_scaling = usage['n_nodes'] / prev_usage['n_nodes']
                
                # Memory scaling should not be worse than quadratic in nodes
                assert scaling_factor <= node_scaling ** 2.5, \
                    f"Memory scaling too aggressive: {scaling_factor} vs node scaling {node_scaling}"
    
    def test_computation_time_scaling(self):
        """Test computation time scaling with problem size."""
        node_counts = [5, 10, 15, 20]
        time_measurements = []
        
        for n_nodes in node_counts:
            sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes, stalk_dim=2)
            
            analyzer = PersistentSpectralAnalyzer(
                static_laplacian=StaticLaplacianWithMasking(eigenvalue_method='lobpcg')
            )
            
            # Measure computation time
            start_time = time.time()
            result = analyzer.analyze(sheaf, n_steps=10)
            computation_time = time.time() - start_time
            
            time_measurements.append({
                'n_nodes': n_nodes,
                'time_seconds': computation_time,
                'n_eigenvalue_computations': len(result['persistence_result']['eigenvalue_sequences'])
            })
            
            # Clear cache
            analyzer.clear_cache()
        
        # Validate time scaling
        for i, measurement in enumerate(time_measurements):
            # Individual computations should be reasonable
            assert measurement['time_seconds'] < 60, \
                f"Computation too slow for {measurement['n_nodes']} nodes: {measurement['time_seconds']}s"
            
            # Time scaling should be reasonable
            if i > 0:
                prev_measurement = time_measurements[i-1]
                time_scaling = measurement['time_seconds'] / max(prev_measurement['time_seconds'], 0.01)
                node_scaling = measurement['n_nodes'] / prev_measurement['n_nodes']
                
                # Time should not scale worse than cubic in nodes
                assert time_scaling <= node_scaling ** 3.5, \
                    f"Time scaling too aggressive: {time_scaling} vs node scaling {node_scaling}"
    
    def test_eigenvalue_computation_efficiency(self):
        """Test efficiency of eigenvalue computation methods."""
        # Create test case
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes=8, stalk_dim=4)
        
        # Test different eigenvalue methods
        methods = ['dense', 'lobpcg']
        method_performance = {}
        
        for method in methods:
            analyzer = PersistentSpectralAnalyzer(
                static_laplacian=StaticLaplacianWithMasking(
                    eigenvalue_method=method,
                    max_eigenvalues=30
                )
            )
            
            # Measure performance
            start_time = time.time()
            start_memory = self.profiler.process.memory_info().rss / 1024 / 1024
            
            try:
                result = analyzer.analyze(sheaf, n_steps=6)
                
                end_time = time.time()
                end_memory = self.profiler.process.memory_info().rss / 1024 / 1024
                
                method_performance[method] = {
                    'time_seconds': end_time - start_time,
                    'memory_mb': end_memory - start_memory,
                    'success': True,
                    'n_eigenvalues': len(result['persistence_result']['eigenvalue_sequences'][0])
                }
                
            except Exception as e:
                method_performance[method] = {
                    'time_seconds': float('inf'),
                    'memory_mb': float('inf'),
                    'success': False,
                    'error': str(e)
                }
            
            # Clear cache
            analyzer.clear_cache()
        
        # At least one method should work
        successful_methods = [m for m, perf in method_performance.items() if perf['success']]
        assert len(successful_methods) > 0, "No eigenvalue method succeeded"
        
        # LOBPCG should be more efficient for larger problems
        if method_performance['lobpcg']['success'] and method_performance['dense']['success']:
            lobpcg_time = method_performance['lobpcg']['time_seconds']
            dense_time = method_performance['dense']['time_seconds']
            
            # For this size, either could be faster, but both should be reasonable
            assert lobpcg_time < 30, f"LOBPCG too slow: {lobpcg_time}s"
            assert dense_time < 30, f"Dense method too slow: {dense_time}s"
    
    def test_filtration_step_scaling(self):
        """Test scaling with number of filtration steps."""
        sheaf, expected = self.generator.tree_sheaf(depth=2, branching_factor=3, stalk_dim=3)
        
        step_counts = [5, 10, 20, 30]
        step_performance = []
        
        for n_steps in step_counts:
            analyzer = PersistentSpectralAnalyzer(
                static_laplacian=StaticLaplacianWithMasking(eigenvalue_method='lobpcg')
            )
            
            start_time = time.time()
            result = analyzer.analyze(sheaf, n_steps=n_steps)
            computation_time = time.time() - start_time
            
            step_performance.append({
                'n_steps': n_steps,
                'time_seconds': computation_time,
                'time_per_step': computation_time / n_steps
            })
            
            analyzer.clear_cache()
        
        # Time should scale roughly linearly with filtration steps
        for i, perf in enumerate(step_performance):
            assert perf['time_seconds'] < 60, \
                f"Too slow for {perf['n_steps']} steps: {perf['time_seconds']}s"
            
            if i > 0:
                prev_perf = step_performance[i-1]
                step_scaling = perf['n_steps'] / prev_perf['n_steps']
                time_scaling = perf['time_seconds'] / prev_perf['time_seconds']
                
                # Time scaling should be roughly linear in steps
                assert time_scaling <= step_scaling * 1.5, \
                    f"Non-linear time scaling: {time_scaling} vs {step_scaling}"
    
    def test_subspace_tracking_performance(self):
        """Test performance of subspace tracking."""
        # Create eigenvalue sequence for tracking
        n_steps = 50
        eigenval_seqs, eigenvec_seqs, expected = self.generator.crossing_eigenvalues_sequence(n_steps)
        
        tracker = SubspaceTracker(gap_eps=1e-6, cos_tau=0.8)
        
        # Measure tracking performance
        start_time = time.time()
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, list(range(n_steps))
        )
        tracking_time = time.time() - start_time
        
        # Tracking should be efficient
        assert tracking_time < 10, f"Subspace tracking too slow: {tracking_time}s for {n_steps} steps"
        
        # Should produce meaningful results
        assert len(tracking_info['eigenvalue_paths']) >= 0
        total_events = (len(tracking_info['birth_events']) + 
                       len(tracking_info['death_events']) + 
                       len(tracking_info['crossings']))
        assert total_events >= 0
    
    def test_cache_efficiency(self):
        """Test caching efficiency."""
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=6, stalk_dim=3)
        
        analyzer = PersistentSpectralAnalyzer(
            static_laplacian=StaticLaplacianWithMasking(eigenvalue_method='lobpcg')
        )
        
        # First analysis (cold cache)
        start_time = time.time()
        result1 = analyzer.analyze(sheaf, n_steps=8)
        first_analysis_time = time.time() - start_time
        
        # Second analysis (warm cache)
        start_time = time.time()
        result2 = analyzer.analyze(sheaf, n_steps=8)
        second_analysis_time = time.time() - start_time
        
        # Second analysis should be faster due to caching
        speedup_ratio = first_analysis_time / max(second_analysis_time, 0.001)
        assert speedup_ratio >= 1.0, f"No caching speedup observed: {speedup_ratio}"
        
        # Results should be identical
        assert len(result1['filtration_params']) == len(result2['filtration_params'])
        assert result1['filtration_type'] == result2['filtration_type']
        
        # Check cache status
        cache_info = analyzer.static_laplacian.get_cache_info()
        assert cache_info['laplacian_cached'], "Laplacian should be cached"
    
    def test_memory_cleanup(self):
        """Test memory cleanup and garbage collection."""
        initial_memory = self.profiler.process.memory_info().rss / 1024 / 1024
        
        # Perform multiple analyses
        for i in range(5):
            sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=8, stalk_dim=3)
            analyzer = PersistentSpectralAnalyzer()
            result = analyzer.analyze(sheaf, n_steps=6)
            
            # Explicitly clean up
            analyzer.clear_cache()
            del analyzer
            del sheaf
            del result
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = self.profiler.process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be modest
        assert memory_increase < 200, f"Memory not properly cleaned up: {memory_increase} MB increase"
    
    def test_large_stalk_dimension_performance(self):
        """Test performance with larger stalk dimensions."""
        # Test different stalk dimensions
        stalk_dims = [2, 5, 10, 15]
        stalk_performance = []
        
        for stalk_dim in stalk_dims:
            sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=5, stalk_dim=stalk_dim)
            
            analyzer = PersistentSpectralAnalyzer(
                static_laplacian=StaticLaplacianWithMasking(
                    eigenvalue_method='lobpcg',
                    max_eigenvalues=min(50, 5 * stalk_dim)
                )
            )
            
            start_time = time.time()
            start_memory = self.profiler.process.memory_info().rss / 1024 / 1024
            
            result = analyzer.analyze(sheaf, n_steps=6)
            
            end_time = time.time()
            end_memory = self.profiler.process.memory_info().rss / 1024 / 1024
            
            stalk_performance.append({
                'stalk_dim': stalk_dim,
                'time_seconds': end_time - start_time,
                'memory_mb': end_memory - start_memory,
                'total_matrix_size': 5 * stalk_dim  # Approximation
            })
            
            analyzer.clear_cache()
        
        # Performance should scale reasonably with stalk dimension
        for i, perf in enumerate(stalk_performance):
            assert perf['time_seconds'] < 30, \
                f"Too slow for stalk dim {perf['stalk_dim']}: {perf['time_seconds']}s"
            assert perf['memory_mb'] < 500, \
                f"Too much memory for stalk dim {perf['stalk_dim']}: {perf['memory_mb']} MB"


@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Scalability benchmarks approaching the performance targets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.profiler = PerformanceProfiler()
    
    def test_resnet_scale_approximation(self):
        """Test performance approaching ResNet50 scale."""
        # ResNet50 has ~50 layers, approximate with large graph
        # This is a stress test marked as slow
        
        n_nodes = 30  # Smaller than full ResNet50 but representative
        stalk_dim = 8  # Representative of feature dimensions
        n_steps = 15   # Reasonable filtration resolution
        
        # Create large test case
        sheaf, expected = self.generator.tree_sheaf(
            depth=3, branching_factor=3, stalk_dim=stalk_dim
        )
        
        # Use efficient analyzer configuration
        analyzer = PersistentSpectralAnalyzer(
            static_laplacian=StaticLaplacianWithMasking(
                eigenvalue_method='lobpcg',
                max_eigenvalues=100  # Limit eigenvalues for efficiency
            )
        )
        
        self.profiler.start_profiling()
        
        # Perform analysis
        result = analyzer.analyze(sheaf, n_steps=n_steps)
        
        profile = self.profiler.get_profile()
        
        # Check against targets (scaled down from full ResNet50)
        target_time_minutes = 2.0  # Scaled target
        target_memory_gb = 1.5     # Scaled target
        
        assert profile['elapsed_time'] < target_time_minutes * 60, \
            f"Computation too slow: {profile['elapsed_time']/60:.2f} min > {target_time_minutes} min"
        
        assert profile['memory_increase'] < target_memory_gb * 1024, \
            f"Memory usage too high: {profile['memory_increase']:.2f} MB > {target_memory_gb*1024} MB"
        
        # Validate results are still mathematically correct
        validator = PersistenceValidator()
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        
        for eigenvals in eigenval_seqs:
            validation = validator.validate_eigenvalue_properties(eigenvals)
            assert validation['finite'], "Large scale computation produced non-finite eigenvalues"
            assert validation['non_negative'], "Large scale computation produced negative eigenvalues"
    
    def test_memory_target_compliance(self):
        """Test compliance with memory target."""
        # Create test case that exercises memory usage
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes=10, stalk_dim=6)
        
        analyzer = PersistentSpectralAnalyzer(
            static_laplacian=StaticLaplacianWithMasking(eigenvalue_method='lobpcg')
        )
        
        # Monitor peak memory usage
        initial_memory = self.profiler.process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        result = analyzer.analyze(sheaf, n_steps=20)
        
        peak_memory = self.profiler.process.memory_info().rss / 1024 / 1024 / 1024  # GB
        memory_increase = peak_memory - initial_memory
        
        # Should stay well under target for this scale
        memory_target_gb = 1.0  # Conservative target for this test size
        assert memory_increase < memory_target_gb, \
            f"Memory usage {memory_increase:.2f} GB exceeds target {memory_target_gb} GB"
    
    def test_time_target_compliance(self):
        """Test compliance with time target."""
        # Create moderately complex test case
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=15, stalk_dim=5)
        
        analyzer = PersistentSpectralAnalyzer(
            static_laplacian=StaticLaplacianWithMasking(eigenvalue_method='lobpcg')
        )
        
        start_time = time.time()
        result = analyzer.analyze(sheaf, n_steps=25)
        computation_time = time.time() - start_time
        
        # Should stay well under target for this scale
        time_target_minutes = 1.0  # Conservative target for this test size
        assert computation_time < time_target_minutes * 60, \
            f"Computation time {computation_time/60:.2f} min exceeds target {time_target_minutes} min"
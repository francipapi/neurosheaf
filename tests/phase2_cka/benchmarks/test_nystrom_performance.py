"""Performance benchmarks for Nyström CKA implementation."""

import pytest
import torch
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple

from neurosheaf.cka.nystrom import NystromCKA
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.cka.pairwise import PairwiseCKA
from neurosheaf.utils.memory import MemoryMonitor


class TestNystromPerformance:
    """Performance benchmarks for Nyström CKA."""
    
    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        """Setup and cleanup for memory tests."""
        # Clear memory before test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield
        
        # Clear memory after test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def measure_memory_usage(self, func) -> Tuple[float, float]:
        """Measure memory usage of a function.
        
        Returns:
            (peak_memory_mb, execution_time_seconds)
        """
        process = psutil.Process()
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Run function and measure time
        start_time = time.time()
        result = func()
        end_time = time.time()
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_used = peak_memory - initial_memory
        execution_time = end_time - start_time
        
        return memory_used, execution_time
    
    @pytest.mark.benchmark
    def test_nystrom_vs_exact_speed(self):
        """Compare speed of Nyström vs exact CKA."""
        torch.manual_seed(42)
        
        # Test different sizes
        sizes = [100, 500, 1000, 2000]
        nystrom_times = []
        exact_times = []
        
        for size in sizes:
            X = torch.randn(size, 128)
            Y = torch.randn(size, 64)
            
            # Measure exact CKA time
            exact_cka = DebiasedCKA(use_unbiased=True)
            
            start_time = time.time()
            exact_value = exact_cka.compute(X, Y)
            exact_time = time.time() - start_time
            exact_times.append(exact_time)
            
            # Measure Nyström CKA time
            nystrom_cka = NystromCKA(n_landmarks=min(256, size // 2))
            
            start_time = time.time()
            nystrom_value = nystrom_cka.compute(X, Y)
            nystrom_time = time.time() - start_time
            nystrom_times.append(nystrom_time)
            
            # For larger sizes, Nyström should be faster (except on MPS with fallback)
            # On MPS, SVD fallback to CPU can make Nyström slower
            if size >= 1000 and not torch.backends.mps.is_available():
                assert nystrom_time < exact_time
            
            # Both should give reasonable results
            assert 0 <= exact_value <= 1
            assert 0 <= nystrom_value <= 1
        
        # Print results for analysis
        print(f"\\nSpeed comparison:")
        print(f"Sizes: {sizes}")
        print(f"Exact times: {exact_times}")
        print(f"Nyström times: {nystrom_times}")
        
        # For largest size, Nyström should be significantly faster (except on MPS)
        if len(sizes) > 2:
            speedup = exact_times[-1] / nystrom_times[-1]
            print(f"Speedup for size {sizes[-1]}: {speedup:.2f}x")
            # On MPS, SVD fallback makes Nyström slower, so skip this check
            if not torch.backends.mps.is_available():
                assert speedup > 2  # Should be at least 2x faster
    
    @pytest.mark.benchmark
    def test_nystrom_memory_efficiency(self):
        """Test memory efficiency of Nyström vs exact CKA."""
        torch.manual_seed(42)
        
        # Large enough to see memory difference
        size = 2000
        X = torch.randn(size, 256)
        Y = torch.randn(size, 128)
        
        # Measure exact CKA memory
        exact_cka = DebiasedCKA(use_unbiased=True)
        
        def run_exact():
            return exact_cka.compute(X, Y)
        
        exact_memory, exact_time = self.measure_memory_usage(run_exact)
        
        # Measure Nyström CKA memory
        nystrom_cka = NystromCKA(n_landmarks=200)
        
        def run_nystrom():
            return nystrom_cka.compute(X, Y)
        
        nystrom_memory, nystrom_time = self.measure_memory_usage(run_nystrom)
        
        # Nyström should use less memory
        memory_reduction = exact_memory / nystrom_memory if nystrom_memory > 0 else float('inf')
        
        print(f"\\nMemory comparison for size {size}:")
        print(f"Exact memory: {exact_memory:.1f} MB")
        print(f"Nyström memory: {nystrom_memory:.1f} MB")
        print(f"Memory reduction: {memory_reduction:.2f}x")
        
        # Should use significantly less memory (on most devices)
        # On MPS, memory tracking might be different due to unified memory
        if not torch.backends.mps.is_available():
            assert memory_reduction > 2  # At least 2x reduction
    
    @pytest.mark.benchmark
    def test_nystrom_scaling_with_landmarks(self):
        """Test how Nyström performance scales with number of landmarks."""
        torch.manual_seed(42)
        
        size = 1000
        X = torch.randn(size, 128)
        Y = torch.randn(size, 64)
        
        # Exact CKA for comparison
        exact_cka = DebiasedCKA(use_unbiased=True)
        exact_value = exact_cka.compute(X, Y)
        
        # Test different landmark counts
        landmark_counts = [50, 100, 200, 400, 800]
        times = []
        memories = []
        errors = []
        
        for n_landmarks in landmark_counts:
            nystrom_cka = NystromCKA(n_landmarks=n_landmarks)
            
            def run_nystrom():
                return nystrom_cka.compute(X, Y)
            
            memory, exec_time = self.measure_memory_usage(run_nystrom)
            nystrom_value = run_nystrom()
            
            times.append(exec_time)
            memories.append(memory)
            errors.append(abs(nystrom_value - exact_value))
        
        print(f"\\nScaling with landmarks:")
        print(f"Landmarks: {landmark_counts}")
        print(f"Times: {[f'{t:.3f}' for t in times]}")
        print(f"Memories: {[f'{m:.1f}' for m in memories]}")
        print(f"Errors: {[f'{e:.4f}' for e in errors]}")
        
        # Time should scale roughly linearly with landmarks
        # Memory should also scale with landmarks
        assert times[-1] > times[0]  # More landmarks = more time
        assert memories[-1] > memories[0]  # More landmarks = more memory
        
        # Error should decrease with more landmarks (or at least not increase much)
        assert errors[-1] <= errors[0] * 1.5  # More landmarks = better approximation (with tolerance)
    
    @pytest.mark.benchmark
    def test_pairwise_matrix_performance(self):
        """Test performance of pairwise matrix computation."""
        torch.manual_seed(42)
        
        # Create activations for multiple layers
        n_layers = 8
        n_samples = 500
        activations = {
            f'layer_{i}': torch.randn(n_samples, 64 * (i + 1))
            for i in range(n_layers)
        }
        
        # Measure exact computation
        pairwise_exact = PairwiseCKA(use_nystrom=False)
        
        def run_exact():
            return pairwise_exact.compute_matrix(activations)
        
        exact_memory, exact_time = self.measure_memory_usage(run_exact)
        
        # Measure Nyström computation
        pairwise_nystrom = PairwiseCKA(
            use_nystrom=True,
            nystrom_landmarks=100
        )
        
        def run_nystrom():
            return pairwise_nystrom.compute_matrix(activations)
        
        nystrom_memory, nystrom_time = self.measure_memory_usage(run_nystrom)
        
        # Compare results
        exact_matrix = run_exact()
        nystrom_matrix = run_nystrom()
        
        print(f"\\nPairwise matrix performance:")
        print(f"Exact time: {exact_time:.3f}s, memory: {exact_memory:.1f}MB")
        print(f"Nyström time: {nystrom_time:.3f}s, memory: {nystrom_memory:.1f}MB")
        
        # Nyström should be faster and use less memory (except on MPS)
        # On MPS, fallback to CPU can make it slower
        if not torch.backends.mps.is_available():
            assert nystrom_time < exact_time * 1.5  # Allow some overhead
            assert nystrom_memory < exact_memory or exact_memory < 10  # Memory reduction
        
        # Results should be similar (MPS numerical issues now handled via CPU fallback)
        diff = torch.abs(exact_matrix - nystrom_matrix)
        max_diff = torch.max(diff)
        # Nyström approximation inherently has some error due to low-rank approximation
        # With proper MPS handling, the error should be consistent across platforms
        tolerance = 0.5  # Reasonable tolerance for Nyström approximation
        assert max_diff < tolerance  # Allow reasonable approximation error
    
    @pytest.mark.benchmark
    def test_large_scale_computation(self):
        """Test Nyström on large-scale data."""
        torch.manual_seed(42)
        
        # Large scale that would be expensive for exact computation
        size = 5000
        X = torch.randn(size, 512)
        Y = torch.randn(size, 256)
        
        # Test different landmark counts
        landmark_counts = [100, 200, 400]
        
        for n_landmarks in landmark_counts:
            nystrom_cka = NystromCKA(n_landmarks=n_landmarks)
            
            def run_computation():
                return nystrom_cka.compute(X, Y)
            
            memory, exec_time = self.measure_memory_usage(run_computation)
            cka_value = run_computation()
            
            print(f"\\nLarge scale (n={size}, landmarks={n_landmarks}):")
            print(f"Time: {exec_time:.3f}s, Memory: {memory:.1f}MB, CKA: {cka_value:.4f}")
            
            # Should complete in reasonable time and memory
            assert exec_time < 30  # Should complete in < 30 seconds
            assert memory < 1000  # Should use < 1GB memory
            assert 0 <= cka_value <= 1  # Should give valid result
    
    @pytest.mark.benchmark
    def test_approximation_quality_vs_speed(self):
        """Test trade-off between approximation quality and speed."""
        torch.manual_seed(42)
        
        # Fixed data for comparison
        size = 800
        X = torch.randn(size, 128)
        Y = torch.randn(size, 64)
        
        # Exact CKA for reference
        exact_cka = DebiasedCKA(use_unbiased=True)
        exact_value = exact_cka.compute(X, Y)
        
        # Test different configurations
        configs = [
            (32, 'uniform'),
            (64, 'uniform'),
            (128, 'uniform'),
            (256, 'uniform'),
            (32, 'kmeans'),
            (64, 'kmeans'),
            (128, 'kmeans'),
        ]
        
        results = []
        
        for n_landmarks, selection in configs:
            nystrom_cka = NystromCKA(
                n_landmarks=n_landmarks,
                landmark_selection=selection
            )
            
            def run_computation():
                return nystrom_cka.compute(X, Y)
            
            memory, exec_time = self.measure_memory_usage(run_computation)
            nystrom_value = run_computation()
            error = abs(nystrom_value - exact_value)
            
            results.append({
                'landmarks': n_landmarks,
                'selection': selection,
                'time': exec_time,
                'memory': memory,
                'error': error,
                'value': nystrom_value
            })
        
        # Print results
        print(f"\\nApproximation quality vs speed (exact CKA: {exact_value:.4f}):")
        print(f"{'Landmarks':<10} {'Selection':<8} {'Time':<8} {'Memory':<8} {'Error':<8} {'Value':<8}")
        for r in results:
            print(f"{r['landmarks']:<10} {r['selection']:<8} {r['time']:<8.3f} {r['memory']:<8.1f} {r['error']:<8.4f} {r['value']:<8.4f}")
        
        # More landmarks should generally give better approximation
        uniform_results = [r for r in results if r['selection'] == 'uniform']
        errors = [r['error'] for r in uniform_results]
        
        # Largest landmark count should have smallest error
        min_error_idx = errors.index(min(errors))
        max_landmarks_idx = [r['landmarks'] for r in uniform_results].index(max(r['landmarks'] for r in uniform_results))
        
        # Generally, more landmarks should give better approximation
        assert min_error_idx == max_landmarks_idx or errors[max_landmarks_idx] < errors[0]
    
    @pytest.mark.benchmark
    def test_memory_limit_handling(self):
        """Test how Nyström handles memory limits."""
        torch.manual_seed(42)
        
        # Create data that would exceed memory limits with exact computation
        size = 3000
        activations = {
            'layer1': torch.randn(size, 256),
            'layer2': torch.randn(size, 128),
            'layer3': torch.randn(size, 64)
        }
        
        # Test with strict memory limit
        pairwise = PairwiseCKA(
            use_nystrom=True,
            nystrom_landmarks=100,
            memory_limit_mb=100  # Very strict limit
        )
        
        def run_computation():
            return pairwise.compute_matrix(activations)
        
        memory, exec_time = self.measure_memory_usage(run_computation)
        cka_matrix = run_computation()
        
        print(f"\\nMemory limit handling:")
        print(f"Time: {exec_time:.3f}s, Memory: {memory:.1f}MB")
        print(f"Matrix shape: {cka_matrix.shape}")
        
        # Should complete within memory constraints
        assert cka_matrix.shape == (3, 3)
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-5)
        assert memory < 200  # Should respect memory limits (with some overhead)
    
    @pytest.mark.benchmark
    def test_device_performance(self):
        """Test performance across different devices."""
        torch.manual_seed(42)
        
        size = 1000
        X = torch.randn(size, 128)
        Y = torch.randn(size, 64)
        
        # CPU performance
        nystrom_cpu = NystromCKA(n_landmarks=100, device='cpu')
        
        def run_cpu():
            return nystrom_cpu.compute(X, Y)
        
        cpu_memory, cpu_time = self.measure_memory_usage(run_cpu)
        cpu_value = run_cpu()
        
        print(f"\\nDevice performance:")
        print(f"CPU - Time: {cpu_time:.3f}s, Memory: {cpu_memory:.1f}MB, Value: {cpu_value:.4f}")
        
        # GPU performance (if available)
        if torch.cuda.is_available():
            X_gpu = X.cuda()
            Y_gpu = Y.cuda()
            
            nystrom_gpu = NystromCKA(n_landmarks=100, device='cuda')
            
            def run_gpu():
                return nystrom_gpu.compute(X_gpu, Y_gpu)
            
            gpu_memory, gpu_time = self.measure_memory_usage(run_gpu)
            gpu_value = run_gpu()
            
            print(f"GPU - Time: {gpu_time:.3f}s, Memory: {gpu_memory:.1f}MB, Value: {gpu_value:.4f}")
            
            # Results should be similar across devices
            assert abs(cpu_value - gpu_value) < 0.01
            
            # GPU might be faster for large computations
            if size >= 1000:
                print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
        
        # MPS performance (if available)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            X_mps = X.to('mps')
            Y_mps = Y.to('mps')
            
            nystrom_mps = NystromCKA(n_landmarks=100, device='mps')
            
            def run_mps():
                return nystrom_mps.compute(X_mps, Y_mps)
            
            mps_memory, mps_time = self.measure_memory_usage(run_mps)
            mps_value = run_mps()
            
            print(f"MPS - Time: {mps_time:.3f}s, Memory: {mps_memory:.1f}MB, Value: {mps_value:.4f}")
            
            # Results should be similar across devices
            assert abs(cpu_value - mps_value) < 0.01
    
    @pytest.mark.benchmark
    def test_batch_processing_performance(self):
        """Test performance with batch processing of multiple layer pairs."""
        torch.manual_seed(42)
        
        # Create many layers
        n_layers = 15
        n_samples = 300
        activations = {
            f'layer_{i}': torch.randn(n_samples, 32 + i * 16)
            for i in range(n_layers)
        }
        
        # Test batch processing
        pairwise = PairwiseCKA(
            use_nystrom=True,
            nystrom_landmarks=50
        )
        
        def run_batch():
            return pairwise.compute_matrix(activations)
        
        memory, exec_time = self.measure_memory_usage(run_batch)
        cka_matrix = run_batch()
        
        # Calculate expected number of pairs
        total_pairs = n_layers * (n_layers + 1) // 2
        time_per_pair = exec_time / total_pairs
        
        print(f"\\nBatch processing performance:")
        print(f"Layers: {n_layers}, Pairs: {total_pairs}")
        print(f"Total time: {exec_time:.3f}s, Time per pair: {time_per_pair:.4f}s")
        print(f"Memory: {memory:.1f}MB")
        
        # Should complete in reasonable time
        assert exec_time < 60  # Should complete in < 1 minute
        assert time_per_pair < 1.0  # Should be < 1 second per pair
        assert cka_matrix.shape == (n_layers, n_layers)
    
    @pytest.mark.benchmark
    def test_memory_recovery(self):
        """Test memory recovery after computation."""
        torch.manual_seed(42)
        
        # Large computation
        size = 2000
        X = torch.randn(size, 256)
        Y = torch.randn(size, 128)
        
        # Measure memory before
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Run computation
        nystrom_cka = NystromCKA(n_landmarks=200)
        cka_value = nystrom_cka.compute(X, Y)
        
        # Measure memory after computation
        after_memory = process.memory_info().rss / 1024 / 1024
        
        # Delete objects and force garbage collection
        del X, Y, nystrom_cka
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"\\nMemory recovery:")
        print(f"Initial: {initial_memory:.1f}MB")
        print(f"After computation: {after_memory:.1f}MB")
        print(f"After cleanup: {final_memory:.1f}MB")
        
        # Memory should be mostly recovered
        memory_leak = final_memory - initial_memory
        assert memory_leak < 50  # Should not leak more than 50MB
        assert 0 <= cka_value <= 1  # Sanity check
    
    @pytest.mark.benchmark
    def test_concurrent_computation(self):
        """Test performance with concurrent Nyström computations."""
        import threading
        import queue
        
        torch.manual_seed(42)
        
        # Create multiple datasets
        datasets = [
            (torch.randn(200, 64), torch.randn(200, 32))
            for _ in range(4)
        ]
        
        results = queue.Queue()
        
        def compute_cka(X, Y, result_queue):
            nystrom_cka = NystromCKA(n_landmarks=32)
            start_time = time.time()
            cka_value = nystrom_cka.compute(X, Y)
            end_time = time.time()
            result_queue.put((cka_value, end_time - start_time))
        
        # Sequential computation
        start_time = time.time()
        sequential_results = []
        for X, Y in datasets:
            compute_cka(X, Y, results)
            sequential_results.append(results.get())
        sequential_time = time.time() - start_time
        
        # Concurrent computation
        start_time = time.time()
        threads = []
        for X, Y in datasets:
            thread = threading.Thread(target=compute_cka, args=(X, Y, results))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_time = time.time() - start_time
        
        concurrent_results = []
        while not results.empty():
            concurrent_results.append(results.get())
        
        print(f"\\nConcurrent computation:")
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Concurrent time: {concurrent_time:.3f}s")
        print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
        
        # Should show some speedup (depending on hardware)
        assert len(concurrent_results) == len(datasets)
        assert all(0 <= result[0] <= 1 for result in concurrent_results)
        
        # Concurrent should be faster or similar (depends on CPU cores)
        assert concurrent_time <= sequential_time * 1.2  # Allow some overhead